"""End-to-end serving benchmark through each server's OpenAI-compatible API.

    python bench/compare_serve.py --name qwen3.5-0.8b --eim m.eim --tokenizer HF_DIR --gguf m.gguf \
        [--servers edgeinfer,llama-server,ollama] [--turns 5] [--concurrency 1,2]

Scenario A, multi-turn chat: a ~1K-token system prompt (a data schema, like
the Data Desk app) then N user turns; each turn resends the whole history, as
chat clients do. Per turn: time to first token (TTFT), decode rate, prompt
tokens. A server that reuses its cached prefix keeps TTFT flat as the history
grows; one that re-encodes the history gets slower every turn.

Scenario B, concurrency: C clients each send one fresh request at once;
aggregate generated tokens per second and mean TTFT.

All requests are greedy (temperature 0) with max_tokens fixed, thinking off.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
HOME = Path.home()
LLAMA_BIN = Path(os.environ.get("LLAMA_BIN", HOME / "edgeinfer-data" / "baselines" / "llama-b11606"))
OLLAMA = os.environ.get("OLLAMA", str(HOME / "edgeinfer-data" / "ollama" / "bin" / "ollama"))
PY = str(ROOT / ".venv" / "bin" / "python")

SCHEMA = "\n".join(
    f"TABLE t{i} ({200 + i * 37} rows): " + ", ".join(f"col_{i}_{j} {'INTEGER' if j % 3 else 'TEXT'}" for j in range(9))
    for i in range(22))
SYSTEM = ("You are a careful data analyst. Answer questions about the user's database. Tables (SQLite):\n" + SCHEMA +
          "\nRules: be concise; when a list helps, use a short markdown table.")
QUESTIONS = ["Describe what kinds of questions this database could answer, in five sentences.",
             "Write a SQL query joining t1 and t2 on their first columns, and explain it in three sentences.",
             "Which tables look like they store text, and why? Answer in four sentences.",
             "Suggest three charts someone could build from these tables, one sentence each.",
             "Summarise our conversation so far in four sentences.",
             "List five possible data quality problems in such a database.",
             "Write a query that counts rows per value of col_3_0 in t3, ordered by count."]


def find_port_free(port):
    import socket
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


async def wait_up(url, timeout=600):
    t0 = time.time()
    async with httpx.AsyncClient() as c:
        while time.time() - t0 < timeout:
            try:
                r = await c.get(url, timeout=2)
                if r.status_code < 500:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1)
    raise RuntimeError(f"{url} did not come up")


def start(server, a):
    env = dict(os.environ)
    if server == "edgeinfer":
        cfg = {"default": a.name, "idle_unload_s": 99999, "models": [{
            "name": a.name, "path": a.eim, "tokenizer": a.tokenizer, "ctx": 8192, "threads": a.threads,
            "slots": 2, "draft": a.draft, "speculative": a.draft > 0}]}
        p = Path(f"/tmp/bench-{a.name}.json")
        p.write_text(json.dumps(cfg))
        env["EDGEINFER_LIB"] = str(ROOT / "build" / "libedgeinfer.so")
        proc = subprocess.Popen([PY, "-m", "edgeinfer.server", "--config", str(p), "--port", "8191"], cwd=ROOT,
                                env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return proc, "http://127.0.0.1:8191", "/health", a.name
    if server == "llama-server":
        env["LD_LIBRARY_PATH"] = str(LLAMA_BIN)
        exe = next(LLAMA_BIN.rglob("llama-server"))
        env["LD_LIBRARY_PATH"] = str(exe.parent)
        proc = subprocess.Popen([str(exe), "-m", a.gguf, "-t", str(a.threads), "-c", "16384", "-np", "2",
                                 "--port", "8192", "--host", "127.0.0.1", "--jinja", "--no-webui"],
                                env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return proc, "http://127.0.0.1:8192", "/health", "default"
    if server == "ollama":
        home = HOME / "edgeinfer-data" / "ollama"
        env.update({"OLLAMA_MODELS": str(home / "models"), "OLLAMA_HOST": "127.0.0.1:11435",
                    "OLLAMA_NUM_PARALLEL": "2", "OLLAMA_KEEP_ALIVE": "30m", "OLLAMA_CONTEXT_LENGTH": "8192"})
        proc = subprocess.Popen([OLLAMA, "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tag = f"bench-{a.name}"
        time.sleep(3)
        mf = home / f"{tag}.Modelfile"
        mf.write_text(f"FROM {a.gguf}\nPARAMETER num_thread {a.threads}\n")
        subprocess.run([OLLAMA, "create", tag, "-f", str(mf)], env=env, capture_output=True)
        return proc, "http://127.0.0.1:11435", "/api/version", tag
    raise ValueError(server)


async def chat(client, base, model, messages, max_tokens):
    body = {"model": model, "messages": messages, "stream": True, "temperature": 0, "max_tokens": max_tokens,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False},
            "think": False}
    t0 = time.time()
    first = last = None
    text, n_chunks, usage = "", 0, {}
    async with client.stream("POST", f"{base}/v1/chat/completions", json=body, timeout=1800) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece:
                    now = time.time()
                    first = first or now
                    last = now
                    text += piece
                    n_chunks += 1
    n_gen = usage.get("completion_tokens") or n_chunks
    return {"ttft_s": round((first or time.time()) - t0, 3), "total_s": round(time.time() - t0, 3),
            "gen_tokens": n_gen, "decode_tok_s": round((n_gen - 1) / (last - first), 2) if first and last and last > first else None,
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"), "text": text}


async def scenario_turns(base, model, turns, max_tokens):
    msgs = [{"role": "system", "content": SYSTEM}]
    out = []
    async with httpx.AsyncClient() as c:
        for i in range(turns):
            msgs.append({"role": "user", "content": QUESTIONS[i % len(QUESTIONS)]})
            r = await chat(c, base, model, msgs, max_tokens)
            msgs.append({"role": "assistant", "content": r.pop("text")})
            out.append({"turn": i + 1, **r})
    return out


async def scenario_concurrency(base, model, n, max_tokens):
    async with httpx.AsyncClient() as c:
        t0 = time.time()
        rs = await asyncio.gather(*[chat(c, base, model, [{"role": "user", "content":
                                                          f"Request {k}: " + QUESTIONS[k % len(QUESTIONS)]}], max_tokens)
                                    for k in range(n)])
        wall = time.time() - t0
    gen = sum(r["gen_tokens"] for r in rs)
    return {"clients": n, "wall_s": round(wall, 2), "agg_tok_s": round(gen / wall, 2),
            "mean_ttft_s": round(sum(r["ttft_s"] for r in rs) / n, 3)}


async def main_async(a):
    out = Path(a.out)
    for server in a.servers.split(","):
        proc, base, health, model = start(server, a)
        try:
            await wait_up(base + health)
            async with httpx.AsyncClient() as c:  # warm-up: load weights, first request
                await chat(c, base, model, [{"role": "user", "content": "hi"}], 4)
            rows = []
            for r in await scenario_turns(base, model, a.turns, a.max_tokens):
                rows.append({"scenario": "multiturn", **r})
            for n in [int(x) for x in a.concurrency.split(",")]:
                rows.append({"scenario": "concurrency", **(await scenario_concurrency(base, model, n, a.max_tokens))})
            for r in rows:
                rec = {"server": server, "model": a.name, "at": time.strftime("%Y-%m-%d %H:%M"), **r}
                print(json.dumps(rec), flush=True)
                with open(out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
        except Exception as e:
            rec = {"server": server, "model": a.name, "error": f"{type(e).__name__}: {e}"[:300]}
            print(json.dumps(rec), flush=True)
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            await asyncio.sleep(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--eim")
    ap.add_argument("--tokenizer")
    ap.add_argument("--gguf")
    ap.add_argument("--servers", default="edgeinfer,llama-server,ollama")
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--concurrency", default="1,2")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--draft", type=int, default=2)
    ap.add_argument("--out", default=str(HOME / "edgeinfer-data" / "results" / "serve.jsonl"))
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
