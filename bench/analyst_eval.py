"""Accuracy / latency / cost of the Data Desk agent per model.

    python bench/analyst_eval.py --config dataset.json --questions qs.json \
        --models qwen3.5-0.8b,qwen3.5-2b,qwen3.5-4b [--llm http://127.0.0.1:8091]

questions.json: [{"q": "...", "sql": "ground-truth SELECT", "match": "all|any"}]
The ground-truth query runs against the analytics DB; an answer is correct
when it states the expected values (numbers within 1%, strings
case-insensitively). Reported per model: accuracy, tool-call validity (no SQL
errors), mean model calls, mean latency, prompt/cached/generated tokens.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sqlite3
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apps.analyst.agent import Agent  # noqa: E402
from apps.analyst.retrieval import Embedder, Retriever  # noqa: E402

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def expected_values(db: str, sql: str) -> list:
    c = sqlite3.connect(f"file:{Path(db).expanduser()}?mode=ro", uri=True)
    try:
        rows = c.execute(sql).fetchall()
    finally:
        c.close()
    vals = []
    for r in rows[:5]:
        for v in r:
            if v is not None and str(v).strip():
                vals.append(v)
    return vals


def found(v, text: str) -> bool:
    if isinstance(v, (int, float)):
        for n in NUM.findall(text):
            try:
                x = float(n.replace(",", ""))
            except ValueError:
                continue
            if abs(x - float(v)) <= max(0.01 * abs(float(v)), 0.051):
                return True
        return False
    return str(v).strip().lower() in text.lower()


async def run_one(agent, q, model):
    events = []

    async def emit(e):
        events.append(e)
    t0 = time.time()
    new = await agent.run([], q, model, emit)
    answer = next((m["content"] for m in reversed(new) if m["role"] == "assistant" and not m.get("tool_calls")), "")
    tools = [e for e in events if e["type"] == "tool"]
    done = next((e for e in events if e["type"] == "done"), {"totals": {}})
    return {"answer": answer, "seconds": time.time() - t0, "tools": len(tools),
            "tool_errors": sum(1 for e in tools if (e.get("meta") or {}).get("error")), **done["totals"]}


async def main_async(a):
    cfg = json.loads(Path(a.config).expanduser().read_text())
    qs = json.loads(Path(a.questions).read_text())
    ret = Retriever(cfg["analytics_db"], cfg["vectors"], Embedder(cfg["embedder"], threads=2))
    agent = Agent(cfg, ret, a.llm)
    for model in a.models.split(","):
        rows = []
        for item in qs:
            exp = expected_values(cfg["analytics_db"], item["sql"])
            r = await run_one(agent, item["q"], model)
            hits = [found(v, r["answer"]) for v in exp]
            ok = (all(hits) if item.get("match", "any") == "all" else any(hits)) if exp else False
            rows.append({**r, "ok": ok, "q": item["q"], "expected": [str(v) for v in exp[:3]]})
            print(f"[{model}] {'OK ' if ok else 'BAD'} {r['seconds']:6.1f}s  {item['q'][:70]}", flush=True)
        n = len(rows)
        summ = {"model": model, "questions": n, "accuracy": round(sum(r["ok"] for r in rows) / n, 3),
                "tool_error_rate": round(sum(r["tool_errors"] for r in rows) / max(1, sum(r["tools"] for r in rows)), 3),
                "mean_model_calls": round(statistics.mean(r.get("steps", 0) for r in rows), 2),
                "mean_s": round(statistics.mean(r["seconds"] for r in rows), 1),
                "p50_s": round(statistics.median(r["seconds"] for r in rows), 1),
                "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in rows),
                "cached_tokens": sum(r.get("cached_tokens", 0) for r in rows),
                "completion_tokens": sum(r.get("completion_tokens", 0) for r in rows),
                "at": time.strftime("%Y-%m-%d %H:%M")}
        print(json.dumps(summ), flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(summ) + "\n")
                for r in rows:
                    f.write(json.dumps({"model": model, "detail": {k: r[k] for k in ("q", "ok", "seconds", "answer", "expected")}},
                                       ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--llm", default="http://127.0.0.1:8091")
    ap.add_argument("--out")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
