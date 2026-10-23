import os, sys, types, tempfile, subprocess, asyncio, importlib.util
import torch
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = tempfile.mkdtemp(prefix="vc_out_"); inp = tempfile.mkdtemp(prefix="vc_in_")
fp = types.ModuleType("folder_paths"); fp.get_output_directory = lambda: out; fp.get_input_directory = lambda: inp
sys.modules["folder_paths"] = fp
spec = importlib.util.spec_from_file_location("voicechain", os.path.join(root, "__init__.py"), submodule_search_locations=[root])
m = importlib.util.module_from_spec(spec); sys.modules["voicechain"] = m; spec.loader.exec_module(m)
N = m.NODE_CLASS_MAPPINGS
subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=f=300:d=22", "-c:a", "libmp3lame",
                os.path.join(inp, "a.mp3")], check=True)
for i, col in enumerate(("blue", "red", "green"), 1):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c={col}:s=200x120", "-frames:v", "1",
                    os.path.join(inp, f"r{i}.png")], check=True)


def name_of(t):
    r, g, b = (int(v * 255) for v in t[0, 5, 5])
    return "blue" if b > 150 else "red" if r > 150 else "green" if g > 60 else f"?{r},{g},{b}"


def run(prev, imgs, aud, prompt, maxf=124, anchor=1):
    imgs = list(imgs) + [""] * (8 - len(imgs))
    return asyncio.run(N["VoiceChainLoad"]().run(prev, *imgs, aud, prompt, maxf, anchor))


def chain(imgs, label):
    prev = ""; chosen = []; firsts = []
    for step in range(1, 9):
        ref, first, aud, frames, prompt, ctx, anc = run(prev, imgs if step == 1 else [], "a.mp3" if step == 1 else "",
                                                        "hello" if step == 1 else "")
        chosen.append(name_of(ref)); firsts.append(name_of(first))
        r = asyncio.run(N["VoiceChainFinish"]().run(torch.rand(frames, 120, 200, 3), ctx, 24, "video/VC"))
        f = r["ui"]["images"][0]; prev = os.path.join(out, f["subfolder"], f["filename"])
        st, v = m.core.read_state(prev)
        print(f"  [{label}] step {step}: frames {frames} ref {chosen[-1]} first-frame {firsts[-1]} n_refs {st['n_refs']} "
              f"cursor {st['cursor']} done {st['done']}")
        if st["done"]:
            break
    else:
        sys.exit("never finished")
    info = m.core.probe(prev)
    npics = sum(1 for s in info["streams"] if s.get("disposition", {}).get("attached_pic"))
    return chosen, firsts, npics, step


c3, f3, n3, s3 = chain(["r1.png", "r2.png", "r3.png"], "3 angles")
assert n3 == 3 and s3 >= 4
assert c3[:4] == ["blue", "red", "green", "blue"], c3
assert f3 == c3                                   # with several angles the cut lands on the chosen angle
c1, f1, n1, s1 = chain(["r1.png"], "1 angle")
assert n1 == 1 and f1[0] == "blue"
assert all(c == "blue" for c in c1)
try:
    run("", [], "", "")
    sys.exit("should have refused")
except RuntimeError as e:
    print("no images refused:", str(e)[:60])
print("OK")
