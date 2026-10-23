import os, sys, json, tempfile, subprocess
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core

work = tempfile.mkdtemp(prefix="vc_test_")
src = os.path.join(work, "src.mp3")
# 13 s of tone bursts with gaps (speech-like)
subprocess.run(["ffmpeg","-y","-v","error","-f","lavfi","-i","sine=f=300:d=13","-af",
  "volume='if(lt(mod(t,2.3),1.7),1,0)':eval=frame","-c:a","libmp3lame",src],check=True)
png = os.path.join(work, "ref.png")
subprocess.run(["ffmpeg","-y","-v","error","-f","lavfi","-i","color=c=red:s=64x64","-frames:v","1",png],check=True)

total = core.audio_seconds(src)
audio = core.decode_audio(src)
print("total", total, audio.shape)
prev = None; cursor = 0.0; audio_src = src; step = 0
while True:
    step += 1
    frames = core.pick_frames(audio, cursor, total)
    seg = core.slice_audio(audio, cursor, frames)
    assert seg.shape[1] == round(frames/24*44100)
    clip = core.write_clip((np.random.rand(frames, 96, 160, 3)*255).astype(np.uint8), 24, (160, 96),
                           os.path.join(work, f"clip{step}.mp4"))
    new_cursor = cursor + frames/24
    done = new_cursor >= total - 0.05
    st = {"tag": core.TAG, "prompt": "p", "cursor": new_cursor, "total": total, "done": done, "step": step}
    out = os.path.join(work, f"out{step}.mp4")
    core.finish(work, clip, prev, [png], prev or src, st, out, total)
    rs, vsec = core.read_state(out)
    print(f"step {step}: frames {frames} cursor {cursor:.2f}->{new_cursor:.2f} video {vsec:.2f}s state {rs['cursor']:.2f} done {done}")
    assert abs(vsec - (total if done else new_cursor)) < 0.15, (vsec, new_cursor)
    prev = out; cursor = rs["cursor"]
    if done: break
    assert step < 10
info = core.probe(prev)
for s in info["streams"]: print(s["index"], s["codec_type"], s["codec_name"], s.get("duration"), s.get("disposition",{}).get("attached_pic"))
print("final format duration", info["format"]["duration"], "target", total)
print("covers", [os.path.getsize(p) for p in core.extract_covers(prev, work)]); lf = core.last_frame(prev, os.path.join(work, "last.png")); print("last frame ok", os.path.getsize(lf))
