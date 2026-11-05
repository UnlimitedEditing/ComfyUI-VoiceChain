import asyncio
import json
import os
import tempfile
import time

import numpy as np
from PIL import Image

from . import core, speech


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


NODE_CLASS_MAPPINGS = {"VoiceChainLoad": VoiceChainLoad, "VoiceChainFinish": VoiceChainFinish,
                       "AudioSpeechPrompt": AudioSpeechPrompt}
NODE_DISPLAY_NAME_MAPPINGS = {"VoiceChainLoad": "Voice Chain - Load / Continue",
                              "VoiceChainFinish": "Voice Chain - Stitch and Save",
                              "AudioSpeechPrompt": "Audio Speech Prompt (Whisper)"}
