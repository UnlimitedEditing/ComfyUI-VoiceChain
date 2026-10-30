"""VoiceChain core: pure python + ffmpeg (no ComfyUI imports), so it can be tested on its own.

A chain is ONE mp4 that carries its own state, so the next step needs only that file:
  video stream 0   the rendered clips stitched so far (h264, stream-copy concat)
  audio stream     the FULL source audio (aac, longer than the video until the last step)
  video stream 1   the character reference image (attached_pic / cover art)
  metadata comment JSON: prompt, settings, cursor (seconds of audio consumed), total, done
The cursor is also the video duration, so a file with lost metadata can still continue.
"""
import json
import os
import subprocess
import tempfile

import numpy as np

FPS = 24
SR = 44100
MAX_FRAMES = 124          # 17k+5 grid; 124 = 5.17 s (measured to fit the 180 s budget at 1280x704)
MIN_FRAMES = 22
TAG = "voicechain/1"
CUE_TAG = "sceneaudio/1"


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, **kw)


def snap_up(frames):
    f = max(MIN_FRAMES, int(frames))
    while f % 17 != 5:
        f += 1
    return f


def probe(path):
    r = _run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path])
    if r.returncode:
        raise RuntimeError(f"ffprobe failed on {path}: {r.stderr[-300:]!r}")
    return json.loads(r.stdout)


def decode_audio(path, start=0.0, duration=None, stream=0):
    """-> float32 array [2, N] at SR. `stream` = which audio stream (a voicechain v5 mp4 holds the mix, then the guide)."""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.4f}", "-i", path]
    if duration is not None:
        cmd += ["-t", f"{duration:.4f}"]
    cmd += ["-map", f"0:a:{stream}", "-vn", "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"]
    r = _run(cmd)
    if r.returncode:
        raise RuntimeError(f"ffmpeg could not decode audio {path}: {r.stderr[-300:]!r}")
    a = np.frombuffer(r.stdout, dtype=np.float32)
    return a.reshape(-1, 2).T.copy() if a.size else np.zeros((2, 0), np.float32)


def audio_seconds(path):
    info = probe(path)
    for s in info["streams"]:
        if s["codec_type"] == "audio":
            if s.get("duration"):
                return float(s["duration"])
    return float(info["format"]["duration"])


def read_state(path):
    """-> (state dict, main video seconds). Raises if this is not a voicechain mp4."""
    info = probe(path)
    tags = {k.lower(): v for k, v in info.get("format", {}).get("tags", {}).items()}
    raw = tags.get("comment", "")
    st = json.loads(raw) if raw.startswith("{") else {}
    if st.get("tag") != TAG:
        raise RuntimeError(f"{os.path.basename(path)} is not a voicechain video (no state in its comment tag)")
    vsec = 0.0
    for s in info["streams"]:
        if s["codec_type"] == "video" and not s.get("disposition", {}).get("attached_pic"):
            vsec = float(s.get("duration") or info["format"]["duration"])
            break
    return st, vsec


def extract_covers(path, out_dir):
    """All attached reference pictures, in stream order -> list of png paths."""
    info = probe(path)
    out = []
    for s in info["streams"]:
        if s["codec_type"] == "video" and s.get("disposition", {}).get("attached_pic"):
            p = os.path.join(out_dir, f"ref{len(out)}.png")
            r = _run(["ffmpeg", "-y", "-v", "error", "-i", path, "-map", f"0:{s['index']}", "-frames:v", "1", p])
            if r.returncode:
                raise RuntimeError(f"cover extract failed: {r.stderr[-300:]!r}")
            out.append(p)
    if not out:
        raise RuntimeError("the video has no attached reference image")
    return out


def last_frame(path, out_png):
    """Last frame of the MAIN video stream. The container is longer than the video (it holds the full audio),
    so seek from the video stream's own duration, never from the end of the file."""
    vdur = None
    for st in probe(path)["streams"]:
        if st["codec_type"] == "video" and not st.get("disposition", {}).get("attached_pic"):
            vdur = float(st.get("duration") or 0)
            break
    if not vdur:
        raise RuntimeError("no main video stream")
    if os.path.isfile(out_png):
        os.remove(out_png)
    r = _run(["ffmpeg", "-y", "-v", "error", "-ss", f"{max(0.0, vdur - 0.25):.3f}", "-i", path, "-map", "0:v:0",
              "-update", "1", out_png])
    if r.returncode or not os.path.isfile(out_png):
        raise RuntimeError(f"last frame extract failed: {r.stderr[-300:]!r}")
    return out_png


def pick_frames(audio, cursor, total, max_frames=MAX_FRAMES):
    """Choose this step's frame count on the 17k+5 grid, ending at the quietest of the last few grid points.

    audio: [2, N] full source. cursor/total in seconds. max_frames = the per-chunk cap (the workflow's `length`).
    Returns frames (int).
    """
    max_frames = max(MIN_FRAMES, ((int(max_frames) - 5) // 17) * 17 + 5)
    remaining = total - cursor
    if remaining <= max_frames / FPS + 0.3:
        return snap_up(int(np.ceil(remaining * FPS)))
    mono = np.abs(audio[:, :]).mean(axis=0)
    win = int(0.04 * SR)
    best, best_e = max_frames, None
    for f in (max_frames, max_frames - 17, max_frames - 34):
        if f < MIN_FRAMES:
            continue
        t = cursor + f / FPS
        c = int(t * SR)
        seg = mono[max(0, c - win): c + win]
        e = float(seg.mean()) if seg.size else 0.0
        if best_e is None or e < best_e * 0.7:   # prefer the longer chunk unless a shorter one is clearly quieter
            best, best_e = f, e
    return best


# ---- scene-audio cue sheet (sceneaudio/1, made by ComfyUI-VoiceBank SceneAudioMix) -----------------------------------------
def cues_from_png(path):
    """Cue sheet dict if `path` is a pixel-data PNG holding one (4-byte big-endian length + UTF-8 JSON, 3 bytes/pixel,
    read in row order), else None. A character picture never decodes as one."""
    try:
        from PIL import Image
        raw = np.asarray(Image.open(path).convert("RGB"), np.uint8).reshape(-1).tobytes()
        n = int.from_bytes(raw[:4], "big")
        if not 2 <= n <= len(raw) - 4:
            return None
        cues = json.loads(raw[4:4 + n].decode("utf-8"))
    except Exception:
        return None
    return cues if isinstance(cues, dict) and cues.get("tag") == CUE_TAG else None


def chunk_speaker(cues, cursor):
    """Speaker of the turn at `cursor`; in a pause, the next turn's speaker. None if the sheet has no turns."""
    turns = cues.get("turns") or []
    for t in turns:
        if cursor < t["t1"]:
            return t["speaker"]
    return turns[-1]["speaker"] if turns else None


def chunk_angle(cues, cursor, n_refs, step):
    """0-based reference index for this chunk: the speaker's angle from the cue sheet, else the old rotation."""
    sp = chunk_speaker(cues, cursor) if cues else None
    info = (cues or {}).get("speakers", {}).get(str(sp)) if sp is not None else None
    if info:
        return (int(info["angle"]) - 1) % n_refs
    return (step - 1) % n_refs


def pick_frames_cued(cues, cursor, total, max_frames=MAX_FRAMES, tol=0.2):
    """Frame count on the 17k+5 grid whose end falls in a pause between turns (within `tol` s), the longest such <= cap.
    None when there is none (caller falls back to pick_frames' quietest-point rule). A final stretch is just rendered out."""
    max_frames = max(MIN_FRAMES, ((int(max_frames) - 5) // 17) * 17 + 5)
    remaining = total - cursor
    if remaining <= max_frames / FPS + 0.3:
        return snap_up(int(np.ceil(remaining * FPS)))
    turns = cues.get("turns") or []
    pauses = [(turns[i]["t1"] - tol, turns[i + 1]["t0"] + tol) for i in range(len(turns) - 1)]
    f = max_frames
    while f >= MIN_FRAMES:
        t = cursor + f / FPS
        if any(a <= t <= b for a, b in pauses):
            return f
        f -= 17
    return None


def slice_audio(audio, cursor, frames):
    n = int(round(frames / FPS * SR))
    a = int(round(cursor * SR))
    seg = audio[:, a:a + n]
    if seg.shape[1] < n:
        seg = np.pad(seg, ((0, 0), (0, n - seg.shape[1])))
    return seg


def write_clip(frames_rgb, fps, size, out_path):
    """frames_rgb: uint8 [F, H, W, 3] -> video-only h264 mp4 (fixed params so clips concat by stream copy)."""
    h, w = frames_rgb.shape[1:3]
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-"]
    if size and (w, h) != tuple(size):
        cmd += ["-vf", f"scale={size[0]}:{size[1]}:flags=lanczos"]
    cmd += ["-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p",
            "-r", str(fps), "-video_track_timescale", "24000", "-movflags", "+faststart", out_path]
    r = subprocess.run(cmd, input=np.ascontiguousarray(frames_rgb).tobytes(), capture_output=True)
    if r.returncode:
        raise RuntimeError(f"clip encode failed: {r.stderr[-400:]!r}")
    return out_path


def video_size(path):
    for s in probe(path)["streams"]:
        if s["codec_type"] == "video" and not s.get("disposition", {}).get("attached_pic"):
            return int(s["width"]), int(s["height"])
    raise RuntimeError("no video stream")


def finish(work, new_clip, prev_video, ref_pngs, audio_src, state, out_path, total, guide_src=None):
    """Stitch new_clip after prev_video's main stream and write the self-contained mp4.

    audio_src: a file whose FIRST audio stream is the full source (original upload on step 1,
    the previous voicechain mp4 afterwards: its aac is stream-copied, never re-encoded).
    """
    copy_audio = prev_video is not None
    parts = []
    if prev_video:
        prev_v = os.path.join(work, "prev_video.mp4")
        r = _run(["ffmpeg", "-y", "-v", "error", "-i", prev_video, "-map", "0:v:0", "-c", "copy", prev_v])
        if r.returncode:
            raise RuntimeError(f"prev video split failed: {r.stderr[-300:]!r}")
        parts.append(prev_v)
    parts.append(new_clip)
    lst = os.path.join(work, "concat.txt")
    with open(lst, "w", encoding="utf-8") as f:
        for p in parts:
            f.write("file '" + p.replace("\\", "/").replace("'", r"'\''") + "'\n")
    joined = os.path.join(work, "joined.mp4")
    r = _run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", joined])
    if r.returncode:
        raise RuntimeError(f"concat failed: {r.stderr[-300:]!r}")

    # v5: audio stream 0 = the mix (delivered), stream 1 = the dry dialogue (conditions H3), when state["has_guide"]
    guide = bool(state.get("has_guide"))
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", joined, "-i", audio_src]
    n_in = 2
    guide_map = None
    if guide:
        if prev_video:
            guide_map = "1:a:1"
        else:
            cmd += ["-i", guide_src]
            guide_map, n_in = f"{n_in}:a:0", n_in + 1
    for p in ref_pngs:
        cmd += ["-i", p]
    cmd += ["-map", "0:v:0", "-map", "1:a:0"]
    if guide:
        cmd += ["-map", guide_map]
    for i in range(len(ref_pngs)):
        cmd += ["-map", f"{n_in + i}:v:0"]
    cmd += ["-c:v:0", "copy", "-c:a", "copy" if copy_audio else "aac"]
    if not copy_audio:
        cmd += ["-b:a", "160k", "-ar", str(SR)]
    for i in range(len(ref_pngs)):
        cmd += [f"-c:v:{1 + i}", "png", f"-disposition:v:{1 + i}", "attached_pic"]
    cmd += ["-metadata", "comment=" + json.dumps(state, separators=(",", ":")), "-movflags", "+faststart"]
    if state.get("done"):
        cmd += ["-t", f"{total:.4f}"]
    cmd += [out_path]
    r = _run(cmd)
    if r.returncode:
        raise RuntimeError(f"final mux failed: {r.stderr[-400:]!r}")
    return out_path
