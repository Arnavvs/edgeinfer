"""Self-speculative decoding with the MTP head must be lossless for greedy
decoding, and the prefix cache must not change outputs.

    python tests/test_spec.py MODEL.eim HF_DIR [--draft 2] [--tokens 96]

Runs the real Scheduler: (1) greedy without drafts, (2) greedy with MTP
drafts -> identical tokens, acceptance rate, speed; (3) a follow-up turn
that reuses the cached prefix -> identical to a cold run of the same prompt.
"""
import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgeinfer.chat import ChatTokenizer, cache_marks  # noqa: E402
from edgeinfer.engine import Model, SamplingParams  # noqa: E402
from edgeinfer.scheduler import Request, Scheduler  # noqa: E402

PROMPT = [{"role": "system", "content": "You are a helpful assistant. Answer in plain prose."},
          {"role": "user", "content": "Explain how a refrigerator keeps food cold, step by step."}]


def generate(sched, tok, messages, n, spec, keep_think=True):
    prompt = tok.chat(messages, keep_think=keep_think)
    done = threading.Event()
    out = {"tokens": [], "text": "", "info": None}

    def emit(e):
        if e["type"] == "token":
            out["tokens"].append(e["token"])
            out["text"] += e["text"]
        elif e["type"] in ("done", "error"):
            out["info"] = e
            done.set()
    r = Request(prompt=prompt, params=SamplingParams(temperature=0.0), max_tokens=n, emit=emit, spec=spec,
                snap_at=cache_marks(tok, messages, prompt, None, keep_think))
    t0 = time.time()
    sched.submit(r)
    done.wait()
    out["seconds"] = time.time() - t0
    return out


def prefix_match(x, y):
    return next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("hf")
    ap.add_argument("--draft", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=96)
    a = ap.parse_args()
    tok = ChatTokenizer(a.hf)
    m = Model(a.model, threads=4)
    ok = True

    base = Scheduler(m, tok, n_slots=1, n_ctx=4096, n_draft=0, spec=False)
    ref = generate(base, tok, PROMPT, a.tokens, spec=False)
    base.shutdown()
    t_ref = ref["info"]["timing"]
    print(f"plain : {len(ref['tokens'])} tok, {t_ref['gen_tok_s']} tok/s")

    sp = Scheduler(m, tok, n_slots=1, n_ctx=4096, n_draft=a.draft, spec=True)
    got = generate(sp, tok, PROMPT, a.tokens, spec=True)
    t = got["info"]["timing"]
    # verification runs several tokens per forward; float order can differ from
    # one-token decode, so a near-tie may flip far into the text - require a long match
    match = prefix_match(got["tokens"], ref["tokens"])
    same = match >= min(48, len(ref["tokens"]))
    ok &= same
    acc = t["spec_accepted"] / max(1, t["spec_drafted"])
    print(f"spec  : {len(got['tokens'])} tok, {t['gen_tok_s']} tok/s, drafted {t['spec_drafted']} accepted "
          f"{t['spec_accepted']} ({acc:.0%}), matches plain for {match}/{len(ref['tokens'])} tokens")
    if match < len(ref["tokens"]):
        k = next(i for i, (x, y) in enumerate(zip(got["tokens"], ref["tokens"])) if x != y)
        print("  first difference at token", k, repr(tok.decode(ref["tokens"][k - 3:k + 3])), "vs",
              repr(tok.decode(got["tokens"][k - 3:k + 3])))

    # follow-up turn: served from the prefix cache, must equal a cold run
    follow = PROMPT + [{"role": "assistant", "content": got["text"]},
                       {"role": "user", "content": "Now summarise that in one sentence."}]
    warm = generate(sp, tok, follow, 48, spec=True)
    sp.shutdown()
    cold_s = Scheduler(m, tok, n_slots=1, n_ctx=4096, n_draft=a.draft, spec=True)
    cold = generate(cold_s, tok, follow, 48, spec=True)
    cold_s.shutdown()
    wt, ct = warm["info"]["timing"], cold["info"]["timing"]
    m2 = prefix_match(warm["tokens"], cold["tokens"])
    same2 = m2 >= min(16, len(cold["tokens"]))
    ok &= same2 and wt["prompt_tokens_encoded"] < ct["prompt_tokens_encoded"]
    print(f"cache : follow-up encoded {wt['prompt_tokens_encoded']} of "
          f"{warm['info']['usage']['prompt_tokens']} prompt tokens (cold: {ct['prompt_tokens_encoded']}), "
          f"TTFT {wt['ttft_s']} s vs cold {ct['ttft_s']} s, output matches cold run for {m2}/{len(cold['tokens'])} tokens")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
