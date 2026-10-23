import os, sys, types, tempfile, subprocess, asyncio, importlib.util
import numpy as np, torch
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = tempfile.mkdtemp(prefix="vc_out_"); inp = tempfile.mkdtemp(prefix="vc_in_")
fp = types.ModuleType("folder_paths"); fp.get_output_directory = lambda: out; fp.get_input_directory = lambda: inp
sys.modules["folder_paths"] = fp
spec = importlib.util.spec_from_file_location("voicechain", os.path.join(root, "__init__.py"), submodule_search_locations=[root])
m = importlib.util.module_from_spec(spec); sys.modules["voicechain"] = m; spec.loader.exec_module(m)
N = m.NODE_CLASS_MAPPINGS
subprocess.run(["ffmpeg","-y","-v","error","-f","lavfi","-i","sine=f=300:d=9","-c:a","libmp3lame",os.path.join(inp,"a.mp3")],check=True)
subprocess.run(["ffmpeg","-y","-v","error","-f","lavfi","-i","color=c=blue:s=200x120","-frames:v","1",os.path.join(inp,"r.png")],check=True)
prev = ""
for step in range(1, 5):
    ref, aud, frames, prompt, ctx = asyncio.run(N["VoiceChainLoad"]().run(prev, "r.png" if step==1 else "", "a.mp3" if step==1 else "", "hello" if step==1 else "", 5.17))
    print("step", step, "frames", frames, "audio", tuple(aud["waveform"].shape), "prompt", repr(prompt), "ref", tuple(ref.shape))
    imgs = torch.rand(frames, 120, 200, 3)
    r = asyncio.run(N["VoiceChainFinish"]().run(imgs, ctx, 24, "video/VC"))
    f = r["ui"]["images"][0]; prev = os.path.join(out, f["subfolder"], f["filename"])
    st, v = m.core.read_state(prev); print("  ->", f["filename"], st["cursor"], st["done"], round(v,2))
    if st["done"]: break
else:
    sys.exit("never finished")
try:
    asyncio.run(N["VoiceChainLoad"]().run(prev, "", "", "", 5.17)); sys.exit("should have refused")
except RuntimeError as e: print("refuses continuing a finished chain:", e)
