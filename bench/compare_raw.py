"""Raw engine throughput, same model, same machine, same threads.

    python bench/compare_raw.py --name qwen3.5-0.8b --eim m.eim --gguf m-Q4_K_M.gguf \
        --hf HF_DIR [--engines edgeinfer,llama.cpp,llama-cpp-python,ollama,transformers] [--threads 4]

Tests (llama-bench conventions):
  pp512        prompt processing, 512 tokens
  tg128        generation, 128 tokens after a short prompt
  tg64@d4096   generation, 64 tokens with 4096 tokens already in context
Each engine runs in its own subprocess so memory and threads never overlap;
peak RSS is recorded per engine. Results append to --out as JSON lines.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOME = Path.home()
LLAMA_BIN = Path(os.environ.get("LLAMA_BIN", HOME / "llm-lab" / "bin"))
PY_LLAMA = os.environ.get("PY_LLAMA", str(HOME / "Dev" / ".venv" / "bin" / "python"))  # has llama_cpp
PY_TORCH = os.environ.get("PY_TORCH", str(HOME / "vision-lab" / ".venv" / "bin" / "python"))  # has torch+transformers
OLLAMA = os.environ.get("OLLAMA", str(HOME / "edgeinfer-data" / "ollama" / "bin" / "ollama"))


PEAK = {"mb": 0}


def tree_rss_mb(root: int) -> float:
    """RSS of a process and all its descendants (Ollama runs the model in a child)."""
    kids, rss = {}, {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
            with open(f"/proc/{d}/statm") as f:
                rss[int(d)] = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
            kids.setdefault(ppid, []).append(int(d))
        except (OSError, ValueError, IndexError):
            continue
    total, stack = 0.0, [root]
    while stack:
        p = stack.pop()
        total += rss.get(p, 0)
        stack += kids.get(p, [])
    return total


class Sampler(threading.Thread):
    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid, self.stop, self.peak = pid, False, 0.0

    def run(self):
        while not self.stop:
            self.peak = max(self.peak, tree_rss_mb(self.pid))
            time.sleep(0.2)


def run(cmd, env=None, timeout=3600):
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    smp = Sampler(p.pid)
    smp.start()
    out, err = p.communicate(timeout=timeout)
    smp.stop = True
    PEAK["mb"] = max(PEAK["mb"], smp.peak)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {err[-800:]}")
    return out, time.time() - t0


# ------------------------------------------------------------------ engines
def bench_edgeinfer(a):
    exe = ROOT / "build" / "ei-bench"
    rows = []
    out, _ = run([str(exe), "-m", a.eim, "-t", str(a.threads), "-p", "512", "-n", "128", "-r", str(a.reps)])
    out2, _ = run([str(exe), "-m", a.eim, "-t", str(a.threads), "-p", "0", "-n", "64", "-d", "4096", "-r", "1"])
    pp, tg, tgd = [], [], []
    for line in (out + out2).splitlines():
        d = json.loads(line)
        if d["event"] == "pp":
            pp.append(d["tok_s"])
        elif d["event"] == "tg" and d["depth"] >= 4096:
            tgd.append(d["tok_s"])
        elif d["event"] == "tg":
            tg.append(d["tok_s"])
    rows += [("pp512", max(pp)), ("tg128", max(tg)), ("tg64@d4096", max(tgd))]
    return rows, Path(a.eim).stat().st_size


def bench_llamacpp(a):
    exe = LLAMA_BIN / "llama-bench"
    env = {**os.environ, "LD_LIBRARY_PATH": str(LLAMA_BIN)}
    out, _ = run([str(exe), "-m", a.gguf, "-t", str(a.threads), "-p", "512", "-n", "128", "-r", str(a.reps), "-o", "json"], env)
    out2, _ = run([str(exe), "-m", a.gguf, "-t", str(a.threads), "-p", "0", "-n", "64", "-d", "4096", "-r", "1",
                   "-o", "json"], env)
    rows = []
    for d in json.loads(out) + json.loads(out2):
        if d["n_prompt"] and not d["n_gen"]:
            rows.append(("pp512", d["avg_ts"]))
        elif d["n_gen"] and d.get("n_depth", 0) >= 4096:
            rows.append(("tg64@d4096", d["avg_ts"]))
        elif d["n_gen"]:
            rows.append(("tg128", d["avg_ts"]))
    return rows, Path(a.gguf).stat().st_size


PY_LLAMA_SCRIPT = textwrap.dedent('''
    import sys, time, json, random
    from llama_cpp import Llama
    path, threads = sys.argv[1], int(sys.argv[2])
    llm = Llama(model_path=path, n_ctx=4800, n_batch=512, n_threads=threads, n_threads_batch=threads, verbose=False)
    rnd = random.Random(1)
    toks = [rnd.randrange(1000, 140000) for _ in range(512)]
    res = {}
    llm.reset(); t = time.time(); llm.eval(toks); res["pp512"] = 512 / (time.time() - t)
    t = time.time()
    for _ in range(128):
        tok = llm.sample(temp=0.0)
        llm.eval([tok])
    res["tg128"] = 128 / (time.time() - t)
    llm.reset(); llm.eval([rnd.randrange(1000, 140000) for _ in range(4096)])
    t = time.time()
    for _ in range(64):
        tok = llm.sample(temp=0.0)
        llm.eval([tok])
    res["tg64@d4096"] = 64 / (time.time() - t)
    print(json.dumps(res))
''')


def bench_llama_cpp_python(a):
    out, _ = run([PY_LLAMA, "-c", PY_LLAMA_SCRIPT, a.gguf, str(a.threads)])
    d = json.loads(out.strip().splitlines()[-1])
    return list(d.items()), Path(a.gguf).stat().st_size


PY_HF_SCRIPT = textwrap.dedent('''
    import sys, time, json, torch
    from transformers import AutoModelForImageTextToText
    path, threads = sys.argv[1], int(sys.argv[2])
    torch.set_num_threads(threads)
    m = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.float32).eval()
    g = torch.Generator().manual_seed(1)
    x = torch.randint(1000, 140000, (1, 512), generator=g)
    res = {}
    with torch.no_grad():
        t = time.time(); out = m(input_ids=x, use_cache=True); res["pp512"] = 512 / (time.time() - t)
        pkv = out.past_key_values; tok = out.logits[:, -1:].argmax(-1)
        t = time.time()
        for _ in range(32):
            out = m(input_ids=tok, past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values; tok = out.logits[:, -1:].argmax(-1)
        res["tg128"] = 32 / (time.time() - t)
    print(json.dumps(res))
''')


def bench_transformers(a):
    out, _ = run([PY_TORCH, "-c", PY_HF_SCRIPT, a.hf, str(a.threads)], timeout=7200)
    d = json.loads(out.strip().splitlines()[-1])
    return list(d.items()), None


def bench_ollama(a):
    """Imports the same GGUF into a private Ollama (user-space binary, own model dir)."""
    import urllib.request
    home = HOME / "edgeinfer-data" / "ollama"
    env = {**os.environ, "OLLAMA_MODELS": str(home / "models"), "OLLAMA_HOST": "127.0.0.1:11435",
           "OLLAMA_NUM_PARALLEL": "1", "OLLAMA_KEEP_ALIVE": "10m"}
    srv = subprocess.Popen([OLLAMA, "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    smp = Sampler(srv.pid)
    smp.start()
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen("http://127.0.0.1:11435/api/version", timeout=1)
                break
            except Exception:
                time.sleep(1)
        tag = f"bench-{a.name}"
        mf = home / f"{tag}.Modelfile"
        mf.write_text(f"FROM {a.gguf}\n")
        try:
            run([OLLAMA, "create", tag, "-f", str(mf)], env)
        except RuntimeError:
            # Ollama could not import this GGUF: use its own library build of the model
            tag = "qwen3.5:" + a.name.split("-")[-1].lower()
            run([OLLAMA, "pull", tag], env, timeout=7200)

        def gen(prompt_tokens, n_predict, num_ctx=4800):
            body = {"model": tag, "prompt": " ".join(["hello"] * prompt_tokens), "raw": True, "stream": False,
                    "options": {"num_predict": n_predict, "num_thread": a.threads, "temperature": 0,
                                "num_ctx": num_ctx, "seed": 1}}
            req = urllib.request.Request("http://127.0.0.1:11435/api/generate", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=3600) as r:
                return json.loads(r.read())
        gen(8, 4)  # load
        d = gen(512, 128)
        rows = [("pp512", d["prompt_eval_count"] / (d["prompt_eval_duration"] / 1e9)),
                ("tg128", d["eval_count"] / (d["eval_duration"] / 1e9))]
        d = gen(4096, 64)
        rows.append(("tg64@d4096", d["eval_count"] / (d["eval_duration"] / 1e9)))
        return rows, Path(a.gguf).stat().st_size
    finally:
        smp.stop = True
        PEAK["mb"] = max(PEAK["mb"], smp.peak)
        srv.terminate()
        srv.wait(timeout=30)


ENGINES = {"edgeinfer": bench_edgeinfer, "llama.cpp": bench_llamacpp, "llama-cpp-python": bench_llama_cpp_python,
           "ollama": bench_ollama, "transformers": bench_transformers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--eim")
    ap.add_argument("--gguf")
    ap.add_argument("--hf")
    ap.add_argument("--engines", default="edgeinfer,llama.cpp,llama-cpp-python,ollama,transformers")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=str(HOME / "edgeinfer-data" / "results" / "raw.jsonl"))
    a = ap.parse_args()
    for eng in a.engines.split(","):
        PEAK["mb"] = 0
        t0 = time.time()
        try:
            rows, size = ENGINES[eng](a)
            err = None
        except Exception as e:
            rows, size, err = [], None, f"{type(e).__name__}: {e}"[:400]
        rss = round(PEAK["mb"]) or None
        for test, v in rows:
            r = {"engine": eng, "model": a.name, "test": test, "tok_s": round(float(v), 2), "threads": a.threads,
                 "file_mb": round(size / 2**20) if size else None, "peak_rss_mb": rss,
                 "at": time.strftime("%Y-%m-%d %H:%M")}
            print(json.dumps(r), flush=True)
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")
        if err:
            print(json.dumps({"engine": eng, "model": a.name, "error": err}), flush=True)
            with open(a.out, "a") as f:
                f.write(json.dumps({"engine": eng, "model": a.name, "error": err, "at": time.strftime("%Y-%m-%d %H:%M")}) + "\n")
        print(f"# {eng} done in {time.time() - t0:.0f}s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
