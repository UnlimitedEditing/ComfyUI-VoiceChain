"""VoiceChain v5: cue sheet from scene-audio drives chunk ends (inside pauses), the angle (by speaker) and the two audio tracks."""
import os, sys, types, tempfile, subprocess, asyncio, importlib.util, wave
import numpy as np
import torch
from PIL import Image
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = tempfile.mkdtemp(prefix="vc5_out_"); inp = tempfile.mkdtemp(prefix="vc5_in_")
fp = types.ModuleType("folder_paths"); fp.get_output_directory = lambda: out; fp.get_input_directory = lambda: inp
sys.modules["folder_paths"] = fp
spec = importlib.util.spec_from_file_location("voicechain", os.path.join(root, "__init__.py"), submodule_search_locations=[root])
m = importlib.util.module_from_spec(spec); sys.modules["voicechain"] = m; spec.loader.exec_module(m)
N, core = m.NODE_CLASS_MAPPINGS, m.core

# scene.py lives in the sibling ComfyUI-VoiceBank repo; only used to make the cue sheet the same way the workflow does
sp = importlib.util.spec_from_file_location("scene", os.path.join(os.path.dirname(root), "ComfyUI-VoiceBank", "scene.py"))
scene = importlib.util.module_from_spec(sp); sp.loader.exec_module(scene)

SR = 24000
plan = [(1, 4.2), (2, 3.1), (1, 6.0), (2, 5.3), (1, 2.2), (2, 4.4)]   # (speaker, seconds); 0.5 s pause after each
spans, t, pieces = [], 0.0, []
for spk, d in plan:
    n = int(d * SR); tt = np.arange(n) / SR
    pieces.append((np.sin(2 * np.pi * (140 if spk == 1 else 330) * tt) * 0.3).astype(np.float32))
    spans.append((round(t, 3), round(t + d, 3))); t += d
    pieces.append(np.zeros(int(0.5 * SR), np.float32)); t += 0.5
dialogue = np.concatenate(pieces); total = len(dialogue) / SR
bed = (np.sin(2 * np.pi * 60 * np.arange(len(dialogue)) / SR) * 0.1).astype(np.float32)
mix = dialogue + bed


def write_wav(name, a):
    with wave.open(os.path.join(inp, name), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((a * 32767).astype(np.int16).tobytes())


write_wav("mix.wav", mix); write_wav("dry.wav", dialogue)
turns = [(s, "x") for s, _ in plan]
cues = scene.build_cues(turns, spans, total, "kitchen", "hum")
Image.fromarray(scene.cues_to_pixels(cues)).save(os.path.join(inp, "cues.png"))
for i, col in enumerate(("blue", "red"), 1):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c={col}:s=200x120", "-frames:v", "1",
                    os.path.join(inp, f"r{i}.png")], check=True)

# cue detection: a cue PNG is a sheet, a picture is not
assert core.cues_from_png(os.path.join(inp, "cues.png"))["tag"] == "sceneaudio/1"
assert core.cues_from_png(os.path.join(inp, "r1.png")) is None
assert core.chunk_speaker(cues, 0.1) == 1 and core.chunk_speaker(cues, 4.4) == 2   # 4.4 s is in the pause: next speaker
assert core.chunk_angle(cues, 0.1, 2, 1) == 0 and core.chunk_angle(cues, 5.0, 2, 1) == 1
assert core.chunk_angle(None, 0.0, 3, 5) == (5 - 1) % 3                           # no sheet: old rotation


def name_of(t):
    r, g, b = (int(v * 255) for v in t[0, 5, 5])
    return "blue" if b > 150 else "red" if r > 150 else f"?{r},{g},{b}"


prev, cursor = "", 0.0
ends, picked = [], []
for step in range(1, 12):
    imgs = ["r1.png", "r2.png", "cues.png"] + [""] * 5 if step == 1 else [""] * 8
    ref, first, aud, frames, prompt, ctx, anc = asyncio.run(N["VoiceChainLoad"]().run(
        prev, *imgs, "mix.wav" if step == 1 else "", "hi" if step == 1 else "", 124, 1, "dry.wav" if step == 1 else ""))
    c = __import__("json").loads(ctx)
    assert c["n_refs"] == 2, "the cue PNG must not be used as a reference picture"
    # audio_slice is the DRY dialogue (silent in pauses), not the mix (which carries the bed)
    w = aud["waveform"][0].numpy()
    expect_dry = core.slice_audio(core.decode_audio(os.path.join(inp, "dry.wav")), cursor, frames)
    assert np.abs(w - expect_dry).max() < 0.03, ("slice must come from the dry dialogue (aac tolerance; the bed would be 0.1)", np.abs(w - expect_dry).max())
    picked.append((cursor, name_of(ref), core.chunk_speaker(cues, cursor)))
    r = asyncio.run(N["VoiceChainFinish"]().run(torch.rand(frames, 120, 200, 3), ctx, 24, "video/VC"))
    f = r["ui"]["images"][0]; prev = os.path.join(out, f["subfolder"], f["filename"])
    st, vsec = core.read_state(prev)
    assert st["has_guide"] and st["cues"]["tag"] == "sceneaudio/1"
    cursor = st["cursor"]; ends.append(cursor)
    print(f"step {step}: frames {frames} -> cursor {cursor:.2f} done {st['done']}")
    if st["done"]:
        break
else:
    raise SystemExit("did not finish")

# every non-final chunk ends in (or within 0.2 s of) a pause
# (a turn longer than one chunk cannot be cut at a pause: then the end may fall inside it, but only if no pause was reachable)
starts = [0.0] + ends[:-1]
for s0, e in zip(starts, ends[:-1]):
    in_pause = any(spans[i][1] - 0.2 <= e <= spans[i + 1][0] + 0.2 for i in range(len(spans) - 1))
    reachable = any(s0 + 22 / 24 <= spans[i][1] + 0.0 and spans[i + 1][0] - 0.2 <= s0 + 124 / 24 for i in range(len(spans) - 1))
    assert in_pause or not reachable, (s0, e, spans)
# angle follows the speaker: speaker 1 -> blue (image1), speaker 2 -> red (image2)
for cur, col, spk in picked:
    assert col == ("blue" if spk == 1 else "red"), (cur, col, spk)
# two audio tracks: stream 0 = mix (bed audible in a pause), stream 1 = dry (silent in a pause)
streams = [s for s in core.probe(prev)["streams"] if s["codec_type"] == "audio"]
assert len(streams) == 2, streams
pause = (spans[0][1] + 0.1, spans[0][1] + 0.4)
a0 = core.decode_audio(prev, pause[0], 0.3, stream=0); a1 = core.decode_audio(prev, pause[0], 0.3, stream=1)
assert np.abs(a0).max() > 0.05 and np.abs(a1).max() < 0.02, (np.abs(a0).max(), np.abs(a1).max())
print("chunk ends", [round(e, 2) for e in ends], "spans", spans)
print("V5 OK")
