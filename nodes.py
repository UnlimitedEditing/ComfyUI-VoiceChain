import asyncio
import json
import os
import tempfile
import time

import numpy as np
from PIL import Image

from . import core


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


class VoiceChainLoad:
    """Start or continue a voice chain.

    Step 1: ref_image + audio (+ prompt). Later steps: prev_video only (a voicechain mp4 carries the
    character, the full audio, the prompt and the cursor); a non-empty prompt overrides the stored one.
    Outputs this step's audio slice and frame count (17k+5 grid, cut near a quiet point) plus ctx for Finish.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "prev_video": ("STRING", {"default": ""}),
            "ref_image": ("STRING", {"default": ""}),
            "audio": ("STRING", {"default": ""}),
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "max_chunk_seconds": ("FLOAT", {"default": 5.17, "min": 0.92, "max": 5.17, "step": 0.01}),
        }}

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "STRING", "STRING")
    RETURN_NAMES = ("ref_image", "audio_slice", "frames", "prompt", "ctx")
    FUNCTION = "run"
    CATEGORY = "VoiceChain"

    async def run(self, prev_video, ref_image, audio, prompt, max_chunk_seconds):
        return await asyncio.to_thread(self._run, prev_video, ref_image, audio, prompt, max_chunk_seconds)

    def _run(self, prev_video, ref_image, audio, prompt, max_chunk_seconds):
        import torch
        work = tempfile.mkdtemp(prefix="voicechain_")
        prev = _resolve(prev_video) or None
        if prev:
            st, vsec = core.read_state(prev)
            if st.get("done"):
                raise RuntimeError("this voice chain is already complete (done=true): nothing left to render")
            cursor = float(st.get("cursor") or vsec)
            total = float(st["total"])
            ref_png = core.extract_cover(prev, os.path.join(work, "ref.png"))
            audio_src = prev
            use_prompt = prompt.strip() or st.get("prompt", "")
            step = int(st.get("step", 0)) + 1
        else:
            ref_path, audio_path = _resolve(ref_image), _resolve(audio)
            if not ref_path or not audio_path:
                raise RuntimeError("first step needs both an image (image1) and an audio clip (audio1); "
                                   "to continue a chain reply with the previous video (video1)")
            im = Image.open(ref_path).convert("RGB")
            im.thumbnail((1024, 1024))
            ref_png = os.path.join(work, "ref.png")
            im.save(ref_png)
            audio_src = audio_path
            cursor, total, step = 0.0, core.audio_seconds(audio_path), 1
            use_prompt = prompt.strip()
        full = core.decode_audio(audio_src)
        total = min(total, full.shape[1] / core.SR) if full.shape[1] else total
        max_f = max(core.MIN_FRAMES, int(max_chunk_seconds * core.FPS))
        frames = core.pick_frames(full, cursor, total)
        if frames > max_f:   # user capped the chunk: largest 17k+5 value that fits
            frames = max(core.MIN_FRAMES, ((max_f - 5) // 17) * 17 + 5)
        seg = core.slice_audio(full, cursor, frames)
        ctx = {"prev": prev, "audio_src": audio_src, "ref_png": ref_png, "work": work, "cursor": cursor,
               "frames": frames, "total": total, "step": step, "prompt": use_prompt}
        audio_out = {"waveform": torch.from_numpy(seg).unsqueeze(0), "sample_rate": core.SR}
        return (_image_tensor(ref_png), audio_out, int(frames), use_prompt, json.dumps(ctx))


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
        state = {"tag": core.TAG, "prompt": c["prompt"], "cursor": round(new_cursor, 4), "total": round(c["total"], 4),
                 "done": done, "step": c["step"]}
        out_dir = folder_paths.get_output_directory()
        sub, prefix = os.path.split(filename_prefix)
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
        name = f"{prefix}_{int(time.time())}_{c['step']:03d}.mp4"
        out_path = os.path.join(out_dir, sub, name)
        core.finish(c["work"], clip, c["prev"], c["ref_png"], c["audio_src"], state, out_path, c["total"])
        return {"ui": {"images": [{"filename": name, "subfolder": sub, "type": "output"}], "animated": (True,)}}


NODE_CLASS_MAPPINGS = {"VoiceChainLoad": VoiceChainLoad, "VoiceChainFinish": VoiceChainFinish}
NODE_DISPLAY_NAME_MAPPINGS = {"VoiceChainLoad": "Voice Chain - Load / Continue",
                              "VoiceChainFinish": "Voice Chain - Stitch and Save"}
