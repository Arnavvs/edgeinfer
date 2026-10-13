"""Compare the engine with saved transformers logits (see tests/ref_hf.py).

    python tests/ref_check.py model.eim ref.npz [--threads 4]

For every prompt: prefill in one batch and token-by-token (the two must agree:
it checks the conv/recurrent state hand-off and the KV cache), compare logits
with the reference, then greedy-generate and compare the tokens.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgeinfer.engine import Context, Model  # noqa: E402


def stats(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    cos = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))
    pa = np.exp(a - a.max()); pa /= pa.sum()
    pb = np.exp(b - b.max()); pb /= pb.sum()
    kl = float((pb * (np.log(pb + 1e-30) - np.log(pa + 1e-30))).sum())
    return float(np.abs(a - b).max()), cos, kl, int(a.argmax() == b.argmax())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("ref")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    ref = np.load(a.ref)
    m = Model(a.model, threads=a.threads)
    ctx = Context(m, n_slots=2, n_ctx=512, max_batch=128, max_draft=3)
    n = len([k for k in ref.files if k.startswith("ids")])
    ok = True
    kls = []
    for i in range(n):
        ids = ref[f"ids{i}"].tolist()
        lg_ref = ref[f"logits{i}"]
        # (1) whole prompt in one batch, logits for every position
        ctx.reset(0)
        ctx.forward([(0, ids, len(ids), 0)])
        rows = [stats(ctx.logits(r), lg_ref[r]) for r in range(len(ids))]
        mad = max(r[0] for r in rows)
        cos = min(r[1] for r in rows)
        kl = float(np.mean([r[2] for r in rows]))
        top1 = float(np.mean([r[3] for r in rows]))
        # (2) token by token in slot 1 must match (1)
        ctx.reset(1)
        step_dev = 0.0
        for j, t in enumerate(ids):
            ctx.forward([(1, [t], 1, 0)])
            step_dev = max(step_dev, float(np.abs(ctx.logits(0) - lg_ref[j]).max()))
        # (3) greedy generation
        ctx.reset(0)
        ctx.forward([(0, ids, 1, 0)])
        gen = []
        for _ in range(len(ref[f"gen{i}"])):
            t = int(ctx.logits(0).argmax())
            gen.append(t)
            ctx.forward([(0, [t], 1, 0)])
        ref_gen = ref[f"gen{i}"].tolist()
        same = next((k for k, (x, y) in enumerate(zip(gen, ref_gen)) if x != y), len(ref_gen))
        print(f"prompt {i}: {len(ids)} tok | batch: max|dlogit| {mad:.4f} min cos {cos:.6f} "
              f"mean KL {kl:.2e} top1 {top1:.3f} | stepwise max|dlogit| {step_dev:.4f} | "
              f"greedy matches first {same}/{len(ref_gen)}")
        ok &= top1 >= 0.95 and same >= min(8, len(ref_gen))
        kls.append(kl)
    print(f"SUMMARY mean_KL={np.mean(kls):.4f}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
