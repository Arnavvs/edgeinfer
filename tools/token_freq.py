"""Frequency-ranked draft vocabulary for the MTP head (FR-Spec, Zhao et al. 2025).

    python tools/token_freq.py HF_DIR out.npy [--text extra.txt ...] [--size 32768]

Qwen3.5 has a 248K-token vocabulary; the LM head is the single largest matmul
of a decode step on small models. Drafting only needs the tokens that are
likely to be accepted, so the draft head is restricted to the most frequent
tokens of a representative corpus (verification still uses the full head,
so outputs are unchanged). Special/chat tokens are always included.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

TRAIN = Path.home() / "edgeinfer-data" / "eval" / "wikitext2-train.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_dir")
    ap.add_argument("out")
    ap.add_argument("--text", action="append", default=[])
    ap.add_argument("--size", type=int, default=32768)
    a = ap.parse_args()
    tok = Tokenizer.from_file(f"{a.hf_dir}/tokenizer.json")
    texts = []
    if TRAIN.exists():
        import pyarrow.parquet as pq
        texts.append("".join(pq.read_table(TRAIN).column("text").to_pylist()))
    for t in a.text:
        texts.append(Path(t).read_text(encoding="utf-8", errors="ignore"))
    cnt = Counter()
    for t in texts:
        for i in range(0, len(t), 200_000):
            cnt.update(tok.encode(t[i:i + 200_000]).ids)
    specials = [tok.token_to_id(s) for s in ["<|im_end|>", "<|im_start|>", "<|endoftext|>", "<think>", "</think>",
                                              "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]]
    ids = [i for i in specials if i is not None]
    for i, _ in cnt.most_common():
        if len(ids) >= a.size:
            break
        if i not in ids:
            ids.append(i)
    covered = sum(cnt[i] for i in ids) / max(1, sum(cnt.values()))
    np.save(a.out, np.asarray(ids, dtype=np.int32))
    print(f"wrote {a.out}: {len(ids)} ids covering {covered:.2%} of {sum(cnt.values())} corpus tokens "
          f"({len(cnt)} distinct)")


if __name__ == "__main__":
    main()
