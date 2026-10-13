"""Tool-using analyst agent on top of the OpenAI-compatible edgeinfer server.

Two tools:
  sql(query)               read-only SQLite over the curated analytics tables
  search(query, source?)   hybrid (BM25 + embedding) search over text chunks

The system prompt (schema + rules + examples) is identical for every
conversation, so the server's prefix cache serves it after the first request;
earlier turns of a conversation are cached the same way.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from typing import Awaitable, Callable

import httpx

from .retrieval import Retriever

MAX_ROWS = 40
MAX_CELL = 160

TOOLS = [
    {"type": "function", "function": {
        "name": "sql",
        "description": "Run one read-only SQLite SELECT over the tables listed in the system prompt. "
                       "Returns at most 40 rows.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "SQLite SELECT"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "search",
        "description": "Search the text (bios, captions, transcripts, comments, listings) by meaning and keywords. "
                       "Returns the best matching snippets with their source and reference.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "source": {"type": "string", "description": "optional: restrict to one source"}},
            "required": ["query"]}}},
]

SYSTEM = """You are Data Desk, a careful data analyst. You answer questions about the user's collected data.

{schema}

Tools:
- sql: for counts, filters, rankings, averages, comparisons. Write one SQLite SELECT. Use LIKE with % for fuzzy text, LOWER() for case. Always add LIMIT (max 40).
- search: for meaning-based questions over text (what people say, who mentions X). Sources: {sources}.

Rules:
- Use a tool before answering any question about the data. Never invent numbers, names or handles; use only what tools return.
- If a query fails, read the error, fix the query once. If the data does not contain the answer, say so plainly.
- Keep answers short: the direct answer first, then a small markdown table if a list helps. Mention the table or source you used.

Examples:
{examples}"""


def _fmt_cell(v) -> str:
    s = "" if v is None else str(v)
    s = s.replace("\n", " ").replace("|", "/")
    return s if len(s) <= MAX_CELL else s[:MAX_CELL] + "…"


class SqlTool:
    def __init__(self, db_path: str, timeout_s: float = 4.0):
        self.path = str(Path(db_path).expanduser())
        self.timeout = timeout_s

    def _con(self) -> sqlite3.Connection:
        c = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
        c.execute("PRAGMA query_only = 1")
        allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
                   getattr(sqlite3, "SQLITE_RECURSIVE", 33)}

        def auth(action, *_):
            return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY
        c.set_authorizer(auth)
        return c

    def run(self, query: str) -> tuple[str, dict]:
        q = query.strip().rstrip(";")
        if not q.lower().lstrip("( \n").startswith(("select", "with")):
            return "error: only SELECT queries are allowed", {"rows": 0, "error": True}
        c = self._con()
        deadline = time.time() + self.timeout
        c.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 10000)
        t0 = time.time()
        try:
            cur = c.execute(q)
            cols = [d[0] for d in cur.description or []]
            rows = cur.fetchmany(MAX_ROWS + 1)
        except sqlite3.Error as e:
            msg = "query took too long" if "interrupt" in str(e).lower() else str(e)
            return f"error: {msg}", {"rows": 0, "error": True, "ms": round((time.time() - t0) * 1000)}
        finally:
            c.close()
        more = len(rows) > MAX_ROWS
        rows = rows[:MAX_ROWS]
        lines = [" | ".join(cols)] + [" | ".join(_fmt_cell(v) for v in r) for r in rows]
        text = "\n".join(lines) if rows else "(no rows)"
        if more:
            text += f"\n(more than {MAX_ROWS} rows; add filters or ORDER BY ... LIMIT)"
        return text, {"rows": len(rows), "columns": cols, "preview": [list(map(_fmt_cell, r)) for r in rows[:12]],
                      "ms": round((time.time() - t0) * 1000), "truncated": more}

    def schema(self, notes: dict, colnotes: dict) -> str:
        c = self._con()
        out = []
        try:
            for (t,) in c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'docs%' "
                                  "AND name NOT IN ('meta') AND name NOT LIKE 'sqlite_%' ORDER BY name"):
                cols = []
                for _, name, typ, *_ in c.execute(f'PRAGMA table_info("{t}")'):
                    note = colnotes.get(f"{t}.{name}")
                    cols.append(f"{name} {typ}".strip() + (f" -- {note}" if note else ""))
                n = c.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                head = f"TABLE {t} ({n} rows)" + (f": {notes[t]}" if notes.get(t) else "")
                out.append(head + "\n  " + "\n  ".join(cols))
        finally:
            c.close()
        return "\n".join(out)


class Agent:
    def __init__(self, cfg: dict, retriever: Retriever, llm_url: str, api_key: str | None = None):
        self.cfg = cfg
        self.sql = SqlTool(cfg["analytics_db"])
        self.ret = retriever
        self.url = llm_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.refresh()

    def refresh(self) -> None:
        schema = self.sql.schema(self.cfg.get("tables", {}), self.cfg.get("columns", {}))
        sources = ", ".join(s for s, _ in self.ret.sources()) or "none"
        ex = "\n".join(f"Q: {e['question']}\nsql: {e['sql']}" for e in self.cfg.get("examples", []))
        self.system = SYSTEM.format(schema="Tables (SQLite):\n" + schema, sources=sources, examples=ex or "(none)")

    async def _complete(self, client: httpx.AsyncClient, messages, model, tools, emit, step) -> dict:
        body = {"model": model, "messages": messages, "stream": True, "temperature": 0.2, "top_p": 0.9,
                "top_k": 20, "max_tokens": 900, "stream_options": {"include_usage": True}}
        if tools:
            body["tools"] = tools
        content, calls, timing, usage, finish = "", [], {}, {}, None
        async with client.stream("POST", f"{self.url}/v1/chat/completions", json=body, headers=self.headers,
                                 timeout=httpx.Timeout(600, connect=10)) as r:
            if r.status_code != 200:
                raise RuntimeError(f"inference server: {r.status_code} {(await r.aread())[:300]!r}")
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[6:])
                if "error" in ev:
                    raise RuntimeError(ev["error"].get("message"))
                ch = ev["choices"][0]
                d = ch.get("delta") or {}
                if d.get("content"):
                    content += d["content"]
                    await emit({"type": "token", "text": d["content"], "step": step})
                if d.get("tool_calls"):
                    calls.extend(d["tool_calls"])
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                    timing = ev.get("timing") or {}
                    usage = ev.get("usage") or {}
        return {"content": content, "tool_calls": calls, "timing": timing, "usage": usage, "finish": finish}

    async def run(self, history: list[dict], question: str, model: str,
                  emit: Callable[[dict], Awaitable[None]], max_steps: int = 4) -> list[dict]:
        msgs = [{"role": "system", "content": self.system}] + history + [{"role": "user", "content": question}]
        new: list[dict] = []
        totals = {"steps": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0, "llm_s": 0.0,
                  "tool_s": 0.0, "ttft_s": None}
        t_start = time.time()
        async with httpx.AsyncClient() as client:
            for step in range(max_steps):
                t0 = time.time()
                res = await self._complete(client, msgs, model, TOOLS if step < max_steps - 1 else None, emit, step)
                totals["steps"] += 1
                totals["llm_s"] += time.time() - t0
                u, tm = res["usage"], res["timing"]
                totals["prompt_tokens"] += u.get("prompt_tokens", 0)
                totals["cached_tokens"] += (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                totals["completion_tokens"] += u.get("completion_tokens", 0)
                if totals["ttft_s"] is None and tm.get("ttft_s") is not None:
                    totals["ttft_s"] = tm["ttft_s"]
                await emit({"type": "step", "step": step, "timing": tm, "usage": u})
                if not res["tool_calls"]:
                    m = {"role": "assistant", "content": res["content"]}
                    msgs.append(m)
                    new.append(m)
                    break
                calls = [{"id": c.get("id"), "type": "function",
                          "function": {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}}
                         for c in res["tool_calls"]]
                m = {"role": "assistant", "content": res["content"], "tool_calls": calls}
                msgs.append(m)
                new.append(m)
                await emit({"type": "tool_intent", "step": step, "text": res["content"]})
                for c in calls:
                    name = c["function"]["name"]
                    try:
                        args = json.loads(c["function"]["arguments"] or "{}")
                    except ValueError:
                        args = {}
                    t1 = time.time()
                    if name == "sql":
                        text, meta = await asyncio.to_thread(self.sql.run, str(args.get("query", "")))
                    elif name == "search":
                        hits = await asyncio.to_thread(self.ret.search, str(args.get("query", "")), 8,
                                                       args.get("source") or None)
                        text = "\n".join(f"[{h['source']}:{h['ref']}] {h['title']}: {h['text'][:400]}" for h in hits) \
                            or "(no matches)"
                        meta = {"hits": [{k: h[k] for k in ("source", "ref", "title", "score")} | {"text": h["text"][:240]}
                                         for h in hits], "ms": round((time.time() - t1) * 1000)}
                    else:
                        text, meta = f"error: unknown tool {name}", {"error": True}
                    totals["tool_s"] += time.time() - t1
                    await emit({"type": "tool", "step": step, "name": name, "args": args, "meta": meta})
                    tm_ = {"role": "tool", "content": text}
                    msgs.append(tm_)
                    new.append(tm_)
        totals["total_s"] = round(time.time() - t_start, 2)
        totals["llm_s"] = round(totals["llm_s"], 2)
        totals["tool_s"] = round(totals["tool_s"], 3)
        await emit({"type": "done", "totals": totals, "model": model})
        return new
