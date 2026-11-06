import asyncio
import json
import os
import tempfile
import time

import numpy as np
from PIL import Image

from . import core, snap, speech


def _input_dir():
    try:
        import folder_paths
        return folder_paths.get_input_directory()
    except Exception:
        return ""


def _resolve(name):
    """Graydient stages media fields as local filenames in the input dir; also accept a full path."""
    name = (name or "").strip()
    if not name:
        return ""
    for cand in (name, os.path.join(_input_dir(), name)):
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(f"media file not found: {name}")


def _image_tensor(path):
    import torch
    im = Image.open(path).convert("RGB")
    return torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0).unsqueeze(0)


def _save_refs(names, work):
    """-> (reference pngs, cue sheet or None). An image slot that holds a scene-audio cue sheet (pixel-data PNG) is the
    cue sheet, not a reference picture."""
    out, cues = [], None
    for n in names:
        path = _resolve(n)
        if not path:
            continue
        found = core.cues_from_png(path)
        if found:
            cues = found
            continue
        im = Image.open(path).convert("RGB")
        im.thumbnail((1024, 1024))
        p = os.path.join(work, f"given{len(out)}.png")
        im.save(p)
        out.append(p)
    return out, cues


class VoiceChainLoad:
    """Start or continue a voice chain.

    Step 1: image1..image8 (angles of the character; one is enough) + audio (+ prompt). Later steps: prev_video
    only (a voicechain mp4 carries the angles, the full audio, the prompt and the cursor). Chunk k uses angle
    (k-1) mod N, so two or three angles read as camera cuts over continuous audio. Images given on a later step
    replace the stored set. A non-empty prompt overrides the stored one.

    anchor=1 also pins the chosen picture as the chunk's first frame (the cut lands on that angle); with a single
    angle, continuing chunks are pinned to the previous video's last frame instead (a smooth join).
    Outputs this step's audio slice and frame count (17k+5 grid, cut near a quiet point) plus ctx for Finish.

    v5 (scene audio): pass the cue sheet PNG from scene-audio in any image slot (e.g. /image8:R<hash>) and the dry
    dialogue as guide_audio (audio2). Then chunks end inside the pauses between turns, the angle follows who is speaking
    (cue sheet speaker -> angle), the mp4 keeps two audio tracks (mix first, dialogue second) and audio_slice is the
    dialogue, so H3 is conditioned on the dry voice while the mix is what the final file plays.
    """

    @classmethod
    def INPUT_TYPES(cls):
        req = {"prev_video": ("STRING", {"default": ""})}
        for i in range(1, 9):
            req[f"image{i}"] = ("STRING", {"default": ""})
        req.update({
            "audio": ("STRING", {"default": ""}),
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "max_chunk_frames": ("INT", {"default": 124, "min": 22, "max": 124, "step": 1}),
            "anchor": ("INT", {"default": 1, "min": 0, "max": 1}),
        })
        return {"required": req, "optional": {"guide_audio": ("STRING", {"default": ""}),
                                              "auto_speech": ("INT", {"default": 1, "min": 0, "max": 1})}}

    RETURN_TYPES = ("IMAGE", "IMAGE", "AUDIO", "INT", "STRING", "STRING", "INT")
    RETURN_NAMES = ("ref_image", "first_frame", "audio_slice", "frames", "prompt", "ctx", "anchor_on")
    FUNCTION = "run"
    CATEGORY = "VoiceChain"

    async def run(self, prev_video, image1, image2, image3, image4, image5, image6, image7, image8,
                  audio, prompt, max_chunk_frames, anchor, guide_audio="", auto_speech=1):
        return await asyncio.to_thread(
            self._run, prev_video, [image1, image2, image3, image4, image5, image6, image7, image8],
            audio, prompt, max_chunk_frames, anchor, guide_audio, auto_speech)

    def _run(self, prev_video, images, audio, prompt, max_chunk_frames, anchor, guide_audio="", auto_speech=1):
        import torch
        work = tempfile.mkdtemp(prefix="voicechain_")
        prev = _resolve(prev_video) or None
        given, given_cues = _save_refs(images, work)
        guide_path, cues, st = "", given_cues, {}
        if prev:
            st, vsec = core.read_state(prev)
            if st.get("done"):
                raise RuntimeError("this voice chain is already complete (done=true): nothing left to render")
            cursor = float(st.get("cursor") or vsec)
            total = float(st["total"])
            if given:
                refs = given
            else:
                stored = os.path.join(work, "stored")
                os.makedirs(stored, exist_ok=True)
                refs = core.extract_covers(prev, stored)
            audio_src = prev
            cues = given_cues or st.get("cues")
            scene = prompt.strip() or st.get("prompt", "")
            words = st.get("words") or []
            step = int(st.get("step", 0)) + 1
        else:
            audio_path = _resolve(audio)
            if not given or not audio_path:
                raise RuntimeError("first step needs at least one image (image1) and an audio clip (audio1); "
                                   "to continue a chain reply with the previous video (video1)")
            refs = given
            audio_src = audio_path
            guide_path = _resolve(guide_audio)
            cursor, total, step = 0.0, core.audio_seconds(audio_path), 1
            scene = prompt.strip()
            words = []
        has_guide = bool(st.get("has_guide")) if prev else bool(guide_path)
        full = core.decode_audio(audio_src)
        total = min(total, full.shape[1] / core.SR) if full.shape[1] else total
        frames = core.pick_frames_cued(cues, cursor, total, max_chunk_frames) if cues else None
        if frames is None:
            frames = core.pick_frames(full, cursor, total, max_chunk_frames)
        if has_guide:   # H3 hears the dry dialogue; the mix (stream 0) is only delivered
            dry = core.decode_audio(prev, stream=1) if prev else core.decode_audio(guide_path)
            seg = core.slice_audio(dry, cursor, frames)
        else:
            seg = core.slice_audio(full, cursor, frames)
        if not prev and auto_speech:   # step 1: transcribe once; the words ride in the mp4 state
            words = speech.transcribe(guide_path or audio_src)
        if auto_speech or words:
            use_prompt = speech.chunk_prompt(scene, words, cursor, cursor + frames / core.FPS)
        else:
            use_prompt = scene   # auto_speech=0: the prompt is used exactly as typed
        chosen = refs[core.chunk_angle(cues, cursor, len(refs), step)]
        first = chosen
        if prev and len(refs) == 1:
            first = core.last_frame(prev, os.path.join(work, "last.png"))
        ctx = {"prev": prev, "audio_src": audio_src, "ref_pngs": refs, "work": work, "cursor": cursor,
               "frames": frames, "total": total, "step": step, "prompt": use_prompt, "scene": scene, "words": words, "n_refs": len(refs),
               "cues": cues, "has_guide": has_guide, "guide_src": guide_path}
        audio_out = {"waveform": torch.from_numpy(seg).unsqueeze(0), "sample_rate": core.SR}
        return (_image_tensor(chosen), _image_tensor(first), audio_out, int(frames), use_prompt, json.dumps(ctx),
                int(anchor))


class VoiceChainFinish:
    """Stitch the decoded clip after the previous video and save the self-contained voicechain mp4."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "ctx": ("STRING", {"forceInput": True}),
            "fps": ("INT", {"default": 24, "min": 8, "max": 60}),
            "filename_prefix": ("STRING", {"default": "video/VoiceChain"}),
        }}

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "VoiceChain"

    async def run(self, images, ctx, fps, filename_prefix):
        return await asyncio.to_thread(self._run, images, ctx, fps, filename_prefix)

    def _run(self, images, ctx, fps, filename_prefix):
        import folder_paths
        c = json.loads(ctx)
        frames = (images.cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)
        size = core.video_size(c["prev"]) if c["prev"] else (frames.shape[2], frames.shape[1])
        clip = core.write_clip(frames, fps, size, os.path.join(c["work"], "newclip.mp4"))
        new_cursor = c["cursor"] + c["frames"] / core.FPS
        done = new_cursor >= c["total"] - 0.05
        state = {"tag": core.TAG, "prompt": c.get("scene", c["prompt"]), "cursor": round(new_cursor, 4), "total": round(c["total"], 4),
                 "done": done, "step": c["step"], "n_refs": c["n_refs"]}
        if c.get("words"):
            state["words"] = c["words"]
        if c.get("cues"):
            state["cues"] = c["cues"]
        if c.get("has_guide"):
            state["has_guide"] = True
        out_dir = folder_paths.get_output_directory()
        sub, prefix = os.path.split(filename_prefix)
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
        name = f"{prefix}_{int(time.time())}_{c['step']:03d}.mp4"
        out_path = os.path.join(out_dir, sub, name)
        core.finish(c["work"], clip, c["prev"], c["ref_pngs"], c["audio_src"], state, out_path, c["total"],
                    c.get("guide_src") or None)
        return {"ui": {"images": [{"filename": name, "subfolder": sub, "type": "output"}], "animated": (True,)}}


class AudioSpeechPrompt:
    """Transcribe an AUDIO clip (Whisper) and append the spoken words to a scene prompt, so a video model is told
    exactly what is being said. auto=0 passes the scene through unchanged."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",), "scene": ("STRING", {"default": "", "multiline": True}),
                             "auto": ("INT", {"default": 1, "min": 0, "max": 1})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "run"
    CATEGORY = "VoiceChain"

    async def run(self, audio, scene, auto):
        return await asyncio.to_thread(self._run, audio, scene, auto)

    def _run(self, audio, scene, auto):
        if not auto:
            return (scene,)
        wav = speech.audio_to_wav(audio, os.path.join(tempfile.mkdtemp(prefix="speechprompt_"), "a.wav"))
        out = speech.speech_prompt(scene, speech.transcribe(wav))
        print(f"[VoiceChain] AudioSpeechPrompt -> {out[:300]!r}")
        return (out,)


class OptionalImageLoad:
    """Load an image by staged filename OR http(s) URL (Graydient fills either), or a 64x64 black placeholder when the slot
    is empty. has_image (0/1) lets a graph switch a guide's strength instead of failing on a missing optional image.
    required=1 raises a clear error instead of returning the placeholder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"filename": ("STRING", {"default": ""})},
                "optional": {"required": ("INT", {"default": 0, "min": 0, "max": 1})}}

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "has_image")
    FUNCTION = "run"
    CATEGORY = "VoiceChain"

    def run(self, filename, required=0):
        import torch
        name = (filename or "").strip()
        if name.lower().startswith(("http://", "https://")):
            import urllib.request
            tmp = os.path.join(tempfile.mkdtemp(prefix="optimg_"), "img")
            req = urllib.request.Request(name, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                f.write(r.read())
            return (_image_tensor(tmp), 1)
        path = _resolve(name)
        if not path:
            if required:
                raise RuntimeError("the first image is required: reply to your picture (init image) when you send this")
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), 0)
        return (_image_tensor(path), 1)


class PinSchedule:
    """Parse the prompt field into frame pins for up to 8 numbered pictures, plus the scene text.

    Syntax, separated by '::' (or new lines):  N.F  = picture N on frame F of the SPEECH (frame 0 = first speech frame, the
    0.5 s lead-in is added for you);  N.Ts = at T seconds (e.g. 2.4.5s);  N.end = last frame;  suffix ~s / ~c = snap to a
    speaking peak / a pause (reserved, applied by AudioPinSnap). Anything else is the scene description. Pictures that are
    present but not placed are spread evenly over the clip in slot order, the last one on the final frame.
    Outputs idx1..idx8 (absolute frame positions in the generated clip, -1 = last frame) and the scene text."""

    MAXPINS = 8

    @classmethod
    def INPUT_TYPES(cls):
        req = {"text": ("STRING", {"default": "", "multiline": True}),
               "total_frames": ("INT", {"default": 121, "min": 9, "max": 4097}),
               "pad_frames": ("INT", {"default": 12, "min": 0, "max": 96}),
               "default_scene": ("STRING", {"default": "", "multiline": True})}
        for i in range(1, cls.MAXPINS + 1):
            req[f"has{i}"] = ("INT", {"default": 0, "min": 0, "max": 1})
        return {"required": req}

    RETURN_TYPES = tuple(["INT"] * 8 + ["INT"] * 8 + ["STRING"])
    RETURN_NAMES = tuple([f"idx{i}" for i in range(1, 9)] + [f"mode{i}" for i in range(1, 9)] + ["scene"])
    FUNCTION = "run"
    CATEGORY = "VoiceChain"

    @staticmethod
    def parse(text, fps=24):
        import re
        pins, scene = {}, []
        for seg in re.split(r"::|\n", text or ""):
            seg = seg.strip()
            if not seg:
                continue
            m = re.match(r"^(\d+)\.(end|\d+(?:\.\d+)?)(s?)(?:~([sc]))?$", seg, re.I)
            if m and 1 <= int(m.group(1)) <= 8:
                k, when, secs, snap = int(m.group(1)), m.group(2).lower(), m.group(3), (m.group(4) or "").lower()
                if when == "end":
                    f = -1
                elif secs:
                    f = int(round(float(when) * fps))
                else:
                    f = int(float(when))
                pins[k] = (f, {"": 0, "s": 1, "c": 2}[snap])
            else:
                scene.append(seg)
        return pins, " ".join(scene)

    @staticmethod
    def place(pins, present, total, pad):
        last = total - 1
        m = len(present)
        idx = {}
        for r, k in enumerate(present, 1):
            if k in pins:
                f = pins[k][0]
                idx[k] = last if f < 0 else min(max(f + pad, pad + 1), last)
            else:
                idx[k] = last if r == m else min(pad + max(int(round(r * (last - pad) / m)), 1), last)
        # keep pins at least 8 frames apart (latent step), moving the later one forward
        order = sorted(present, key=lambda k: (idx[k], k))
        for a, b in zip(order, order[1:]):
            if idx[b] - idx[a] < 8:
                idx[b] = min(idx[a] + 8, last)
        return idx

    def run(self, text, total_frames, pad_frames, default_scene,
            has1, has2, has3, has4, has5, has6, has7, has8):
        pins, scene = self.parse(text)
        has = [has1, has2, has3, has4, has5, has6, has7, has8]
        present = [k for k in range(1, 9) if has[k - 1]]
        idx = self.place(pins, present, total_frames, pad_frames)
        last = total_frames - 1
        out_idx, out_mode = [], []
        for k in range(1, 9):
            v = idx.get(k, last)
            out_idx.append(-1 if v >= last else int(v))
            out_mode.append(pins.get(k, (0, 0))[1])
        print(f"[VoiceChain] PinSchedule present={present} pins={pins} idx={out_idx} scene={scene[:80]!r}")
        return tuple(out_idx + out_mode + [scene or default_scene])


class AudioPinSnap:
    """Snap pins marked ~s (speaking mouth shape) to the nearest speech peak and pins marked ~c (closed mouth) to the nearest
    pause of the SPEECH audio, so framed events land on the waveform. idx/mode come from PinSchedule; unmarked pins and the
    last-frame pin (-1) pass through unchanged."""

    @classmethod
    def INPUT_TYPES(cls):
        req = {"audio": ("AUDIO",), "total_frames": ("INT", {"default": 121, "min": 9, "max": 4097}),
               "pad_frames": ("INT", {"default": 12, "min": 0, "max": 96})}
        for i in range(1, 9):
            req[f"idx{i}"] = ("INT", {"default": -1, "min": -1, "max": 4097})
        for i in range(1, 9):
            req[f"mode{i}"] = ("INT", {"default": 0, "min": 0, "max": 2})
        return {"required": req}

    RETURN_TYPES = tuple(["INT"] * 8)
    RETURN_NAMES = tuple(f"idx{i}" for i in range(1, 9))
    FUNCTION = "run"
    CATEGORY = "VoiceChain"

    def run(self, audio, total_frames, pad_frames, idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8,
            mode1, mode2, mode3, mode4, mode5, mode6, mode7, mode8):
        idx = [idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8]
        modes = [mode1, mode2, mode3, mode4, mode5, mode6, mode7, mode8]
        if not any(modes):
            return tuple(idx)
        wave = audio["waveform"].detach().cpu().float().numpy()
        wave = wave[0] if wave.ndim == 3 else wave
        env = snap.envelope(wave, int(audio["sample_rate"]))
        out = snap.snap_indices(idx, modes, env, pad_frames, total_frames)
        print(f"[VoiceChain] AudioPinSnap {idx} -> {out} (modes {modes})")
        return tuple(out)


class PinColorMatch:
    """Deterministic white-balance / exposure match of an edited picture to a reference picture (numpy only, no model).

    i2i edits drift in white balance and brightness; used as guide frames those drifts make the video flicker. This shifts the
    picture's LAB mean (a, b = colour cast, L = exposure) onto the reference's, and scales its spread (contrast / saturation)
    toward the reference's with a clamped gain, so the correction can never blow a picture out. strength 0 = untouched, 1 = full.
    enable=0 (no picture in that slot) returns the input unchanged."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "reference": ("IMAGE",),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                             "enable": ("INT", {"default": 1, "min": 0, "max": 1})}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "VoiceChain"

    @staticmethod
    def _to_lab(rgb):
        lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        m = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
        xyz = (lin @ m.T) / np.array([0.95047, 1.0, 1.08883])
        f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
        return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)

    @staticmethod
    def _to_rgb(lab):
        fy = (lab[..., 0] + 16) / 116
        f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
        xyz = np.where(f ** 3 > 216 / 24389, f ** 3, (116 * f - 16) / (24389 / 27)) * np.array([0.95047, 1.0, 1.08883])
        m = np.linalg.inv(np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]]))
        lin = np.clip(xyz @ m.T, 0.0, 1.0)
        return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)

    def run(self, image, reference, strength=1.0, enable=1):
        import torch
        if not enable or strength <= 0:
            return (image,)
        ref = self._to_lab(np.clip(reference[0, ..., :3].cpu().numpy().astype(np.float64), 0, 1))
        r_mean, r_std = ref.reshape(-1, 3).mean(0), ref.reshape(-1, 3).std(0)
        out = []
        for i in range(image.shape[0]):
            src = np.clip(image[i, ..., :3].cpu().numpy().astype(np.float64), 0, 1)
            lab = self._to_lab(src)
            s_mean, s_std = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0)
            gain = np.clip(r_std / np.maximum(s_std, 1e-3), 0.8, 1.25)
            fixed = (lab - s_mean) * gain + r_mean
            fixed = lab + (fixed - lab) * float(strength)
            out.append(torch.from_numpy(self._to_rgb(fixed).astype(np.float32)))
        res = torch.stack(out, 0)
        print(f"[VoiceChain] PinColorMatch LAB mean {np.round(s_mean, 1).tolist()} -> {np.round(r_mean, 1).tolist()} strength {strength}")
        return (res.to(image.dtype),)


NODE_CLASS_MAPPINGS = {"VoiceChainLoad": VoiceChainLoad, "VoiceChainFinish": VoiceChainFinish,
                       "AudioSpeechPrompt": AudioSpeechPrompt, "OptionalImageLoad": OptionalImageLoad,
                       "PinSchedule": PinSchedule, "AudioPinSnap": AudioPinSnap, "PinColorMatch": PinColorMatch}
NODE_DISPLAY_NAME_MAPPINGS = {"VoiceChainLoad": "Voice Chain - Load / Continue",
                              "VoiceChainFinish": "Voice Chain - Stitch and Save",
                              "AudioSpeechPrompt": "Audio Speech Prompt (Whisper)",
                              "OptionalImageLoad": "Optional Image Load (placeholder if empty)",
                              "PinSchedule": "Pin Schedule (parse N.frame :: ...)",
                              "AudioPinSnap": "Audio Pin Snap (~s peak / ~c pause)",
                              "PinColorMatch": "Pin Colour Match (white balance to reference)"}
