"""Collect an importance matrix (mean x^2 per input channel of every weight)
by running a high-precision model over calibration text.

    python tools/imatrix.py MODEL.eim HF_DIR out.npz [--tokens 32768] [--text file.txt]

The default calibration text is the wikitext-2 *train* split (evaluation
uses the test split) mixed with chat-formatted turns, so the statistics
cover both prose and the chat template the server actually feeds.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgeinfer._lib import lib  # noqa: E402
from edgeinfer.engine import Context, Model  # noqa: E402

TRAIN = Path.home() / "edgeinfer-data" / "eval" / "wikitext2-train.parquet"

CHAT = [
    ("Summarise the main idea of this paragraph in one line.", "The paragraph argues that small models can run well on old laptops."),
    ("Which sector in Faridabad has the most property dealers?", "Based on the listings, Sector 21 has the most dealers, followed by Greater Faridabad."),
    ("Write a SQL query that counts creators by city.", "SELECT city, COUNT(*) AS n FROM creators GROUP BY city ORDER BY n DESC;"),
    ("kya aap mujhe Hindi mein jawab de sakte ho?", "Haan bilkul, main Hindi aur Hinglish dono mein jawab de sakta hoon."),
]


def calib_text(n_chars: int) -> str:
    import pyarrow.parquet as pq
    lines = [t for t in pq.read_table(TRAIN).column("text").to_pylist() if t.strip()]
    prose = "".join(lines)[:n_chars]
    chats = "".join(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{a}<|im_end|>\n"
                    for q, a in CHAT) * 8
    return chats + prose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("hf_dir")
    ap.add_argument("out")
    ap.add_argument("--tokens", type=int, default=32768)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--text")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{a.hf_dir}/tokenizer.json")
    text = Path(a.text).read_text() if a.text else calib_text(a.tokens * 6)
    ids = np.asarray(tok.encode(text).ids[: a.tokens], dtype=np.int32)
    m = Model(a.model, threads=a.threads)
    c = Context(m, n_slots=1, n_ctx=a.ctx + 8, max_batch=256, max_draft=1)
    lib.ei_calib(c.h, 1)
    t0 = time.time()
    for i in range(0, len(ids), a.ctx):
        chunk = ids[i:i + a.ctx]
        c.reset(0)
        for j in range(0, len(chunk), 256):
            part = chunk[j:j + 256]
            c.forward([(0, part, min(len(part), 64), 0)])  # logits for some rows: covers the LM head too
        print(f"\r{i + len(chunk)}/{len(ids)} tokens {time.time() - t0:.0f}s", end="", flush=True)
    print()
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        path = f.name
    n = lib.ei_calib_dump(c.h, path.encode())
    out = {}
    for line in open(path):
        parts = line.split()
        name, cols = parts[0], int(parts[1])
        out[name] = np.asarray(parts[3:3 + cols], dtype=np.float32)
    # fused tensors share their input: blk.N.ssm_qkvz etc. are keyed by the fused name already
    np.savez(a.out, **out)
    print(f"wrote {a.out}: {n} tensors from {len(ids)} tokens")


if __name__ == "__main__":
    main()
