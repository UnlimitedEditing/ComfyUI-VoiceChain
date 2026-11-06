"""Speech transcript -> per-chunk prompt for voice chains.

faster-whisper runs once on step 1 (word timestamps); the words travel in the mp4's state so later steps never
transcribe again. Each chunk's prompt quotes the words spoken inside that chunk's time window.
"""
import os

LEAD = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
SPEAK = ('The person in the picture speaks directly to the camera, saying: "{words}" Their lips and jaw move in time '
         "with each word, with natural head movement and an expression that matches the tone of the speech. "
         "Medium close-up, steady camera, cinematic lighting.")
SILENT = ("The person in the picture pauses between sentences, lips closed, breathing quietly, listening, with subtle "
          "natural head movement. Medium close-up, steady camera, cinematic lighting.")


def _model_ref(size):
    try:
        import folder_paths
        local = os.path.join(folder_paths.models_dir, "whisper", size)
        if os.path.isfile(os.path.join(local, "model.bin")):
            return local
    except Exception:
        pass
    return size


def transcribe(path, size="small"):
    """-> [[word, start, end], ...] in seconds, [] on any failure (the prompt then just has no quoted words)."""
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        print(f"[VoiceChain] faster-whisper unavailable ({e}); no transcript")
        return []
    ref = _model_ref(size)
    try:
        import importlib.metadata as md
        vers = {p: md.version(p) for p in ("faster-whisper", "ctranslate2", "huggingface_hub", "tokenizers", "onnxruntime", "av")
                if _has(p)}
        listing = sorted(os.listdir(ref)) if os.path.isdir(ref) else "(not a local dir)"
        print(f"[VoiceChain] whisper ref={ref!r} files={listing} versions={vers}")
    except Exception as e:
        print(f"[VoiceChain] whisper diagnostics failed: {e}")
    for device, ctype in (("cuda", "float16"), ("cpu", "int8")):
        try:
            model = WhisperModel(ref, device=device, compute_type=ctype)
            segs, info = model.transcribe(path, word_timestamps=True, vad_filter=True, beam_size=1)
            words = []
            for s in segs:
                for w in (s.words or []):
                    t = w.word.strip()
                    if t:
                        words.append([t, round(float(w.start), 2), round(float(w.end), 2)])
            print(f"[VoiceChain] whisper {size} on {device}: {len(words)} words, language={info.language}")
            return words
        except Exception as e:
            import traceback
            print(f"[VoiceChain] whisper on {device} failed: {e}")
            print("[VoiceChain] whisper traceback:\n" + "".join(traceback.format_exc().splitlines(True)[-14:]))
    return []


def _has(pkg):
    try:
        import importlib.metadata as md
        md.version(pkg)
        return True
    except Exception:
        return False


def words_in(words, start, end):
    """Words whose midpoint falls inside [start, end)."""
    return " ".join(w[0] for w in words if start <= (w[1] + w[2]) / 2 < end)


def chunk_prompt(scene, words, start, end):
    """scene = the user's own text ('' = automatic). {words} in it is replaced by the chunk's spoken words."""
    spoken = words_in(words, start, end).replace('"', "'")
    scene = (scene or "").strip()
    if scene.startswith("For the target video"):
        return scene.replace("{words}", spoken) if "{words}" in scene else scene
    body = SPEAK.format(words=spoken) if spoken else SILENT
    return f"{LEAD} {scene} {body}".replace("  ", " ") if scene else f"{LEAD} {body}"


def audio_to_wav(audio, path):
    """ComfyUI AUDIO dict -> mono 16-bit wav at `path` (what faster-whisper reads)."""
    import wave
    import numpy as np
    w = audio["waveform"].detach().cpu().float().numpy()
    w = w[0] if w.ndim == 3 else w
    mono = w.mean(axis=0) if w.ndim == 2 else w
    pcm = (np.clip(mono, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(int(audio["sample_rate"]))
        f.writeframes(pcm.tobytes())
    return path


def speech_prompt(scene, words):
    scene = (scene or "").strip()
    if not words:
        return scene
    spoken = " ".join(w[0] for w in words).replace('"', "'")
    return f'{scene} The person speaks, saying: "{spoken}" Their lips and jaw move in time with each spoken word.'.strip()
