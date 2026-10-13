"""Quantization quality on wikitext-2: perplexity and KL divergence against the
bf16 model, on identical tokens, for edgeinfer files and (optionally) GGUF
files through llama-cpp-python.

    python bench/quality.py --base bf16.eim --hf HF_DIR --test q4.eim --test q8.eim \
        [--gguf model-Q4_K_M.gguf] [--chunks 16 --ctx 512] [--out results.jsonl]

Each chunk of `ctx` tokens is scored independently (fresh state), the second
half of every chunk is scored (the first half is context), like llama.cpp's
perplexity tool. KL is computed per position from full-vocabulary log-softmax.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgeinfer.engine import Context, Model  # noqa: E402

DATA = Path.home() / "edgeinfer-data" / "eval" / "wikitext2-test.parquet"


def load_tokens(hf_dir: str, n: int) -> np.ndarray:
    from tokenizers import Tokenizer
    txt = DATA.with_suffix(".txt")
    if not txt.exists():
        import pyarrow.parquet as pq
        txt.write_text("\n".join(pq.read_table(DATA).column("text").to_pylist()), encoding="utf-8")
    text = txt.read_text(encoding="utf-8")
    tok = Tokenizer.from_file(f"{hf_dir}/tokenizer.json")
    ids = tok.encode(text[: n * 8]).ids
    return np.asarray(ids[:n], dtype=np.int32)


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


class EiScorer:
    def __init__(self, path, threads):
        self.m = Model(path, threads=threads)
        self.c = Context(self.m, n_slots=1, n_ctx=4096, max_batch=64, max_draft=1)
        self.name = Path(path).name

    def logits(self, toks: np.ndarray, first: int) -> np.ndarray:
        """Log-softmax for positions first..len-2 (predicting tokens first+1..)."""
        self.c.reset(0)
        out = []
        i = 0
        while i < len(toks) - 1:
            n = min(64, len(toks) - 1 - i)
            want = max(0, min(n, i + n - first))
            self.c.forward([(0, toks[i:i + n], want, 0)])
            for r in range(want):
                out.append(log_softmax(self.c.logits(r)).astype(np.float32))
            i += n
        return np.stack(out)


class GgufScorer:
    def __init__(self, path, threads):
        from llama_cpp import Llama
        self.llm = Llama(model_path=str(path), n_ctx=4096, n_batch=512, logits_all=True, n_threads=threads,
                         verbose=False)
        self.name = Path(path).name

    def logits(self, toks: np.ndarray, first: int) -> np.ndarray:
        self.llm.reset()
        self.llm.eval(toks[:-1].tolist())
        sc = np.asarray(self.llm.scores[: len(toks) - 1])
        return np.stack([log_softmax(sc[p]).astype(np.float32) for p in range(first, len(toks) - 1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--test", action="append", default=[])
    ap.add_argument("--gguf", action="append", default=[])
    ap.add_argument("--chunks", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out")
    a = ap.parse_args()

    toks = load_tokens(a.hf, a.chunks * a.ctx)
    chunks = [toks[i * a.ctx:(i + 1) * a.ctx] for i in range(a.chunks)]
    first = a.ctx // 2
    # all scorers stay loaded; each chunk is scored by every model and reduced
    # immediately (full-vocab log-probs for a whole run would not fit in RAM)
    scorers = [EiScorer(a.base, a.threads)] + [EiScorer(p, a.threads) for p in a.test] +               [GgufScorer(p, a.threads) for p in a.gguf]
    names = [s.name for s in scorers]
    names[1 + len(a.test):] = [n + " (llama.cpp)" for n in names[1 + len(a.test):]]
    acc = {n: {"nll": [], "kl": [], "top1": [], "sec": 0.0} for n in names}
    t0 = time.time()
    for ci, ch in enumerate(chunks):
        tg = ch[first + 1:]
        idx = np.arange(len(tg))
        blp = None
        for n, sc in zip(names, scorers):
            t1 = time.time()
            lp = sc.logits(ch, first)
            acc[n]["sec"] += time.time() - t1
            if blp is None:
                blp = lp
                p = np.exp(blp.astype(np.float64))
            acc[n]["nll"].append(-lp[idx, tg])
            acc[n]["kl"].append((p * (blp - lp)).sum(-1))
            acc[n]["top1"].append(lp.argmax(-1) == blp.argmax(-1))
        print(f"chunk {ci + 1}/{len(chunks)} {time.time() - t0:.0f}s", flush=True)
    for n in names:
        nll, kl, top1 = (np.concatenate(acc[n][k]) for k in ("nll", "kl", "top1"))
        row = {"model": n, "tokens": int(len(kl)), "ppl": round(float(np.exp(nll.mean())), 4),
               "kld_mean": round(float(kl.mean()), 5), "kld_p99": round(float(np.percentile(kl, 99)), 4),
               "top1_agree": round(float(top1.mean()), 4), "score_s": round(acc[n]["sec"], 1)}
        print(json.dumps(row))
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
