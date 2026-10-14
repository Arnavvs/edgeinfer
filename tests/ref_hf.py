"""Reference logits from Hugging Face transformers (fp32) for the engine check.

    python tests/ref_hf.py HF_DIR out.npz [--gen 24]

Runs in an environment with torch + transformers; tests/ref_check.py then
compares the engine against the saved logits without needing torch.
"""
import argparse
import time

import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import AutoModelForImageTextToText

PROMPTS = [
    "<|im_start|>user\nExplain in two sentences why the sky looks blue.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "The capital of France is Paris. The capital of Germany is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Delhi mein sabse acchi chai kahan milti hai? Chandni Chowk ya Connaught Place?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_dir")
    ap.add_argument("out")
    ap.add_argument("--gen", type=int, default=24)
    a = ap.parse_args()
    tok = Tokenizer.from_file(f"{a.hf_dir}/tokenizer.json")
    torch.set_num_threads(4)
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(a.hf_dir, dtype=torch.float32)
    model.eval()
    print(f"loaded in {time.time() - t0:.1f}s")
    out = {}
    for i, p in enumerate(PROMPTS):
        ids = tok.encode(p).ids
        x = torch.tensor([ids])
        with torch.no_grad():
            lg = model(input_ids=x).logits[0].float().numpy()
            g = model.generate(input_ids=x, max_new_tokens=a.gen, do_sample=False)[0, len(ids):].tolist()
        out[f"ids{i}"] = np.array(ids, dtype=np.int32)
        out[f"logits{i}"] = lg.astype(np.float32)
        out[f"gen{i}"] = np.array(g, dtype=np.int32)
        print(i, len(ids), "->", repr(tok.decode(g)))
    np.savez(a.out, **out)


if __name__ == "__main__":
    main()
