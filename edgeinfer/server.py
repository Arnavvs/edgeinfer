"""OpenAI-compatible HTTP server.

    python -m edgeinfer.server --config models.json [--host 127.0.0.1 --port 8090]

Endpoints
  POST /v1/chat/completions   messages, tools, stream, temperature, top_p, top_k,
                              min_p, presence/frequency/repetition penalties,
                              max_tokens, stop, seed; extensions: enable_thinking,
                              keep_think, speculative
  POST /v1/completions        raw prompt
  GET  /v1/models             configured models and which one is loaded
  GET  /health                liveness + loaded model
  GET  /metrics               Prometheus text format
  GET  /v1/stats              JSON for dashboards (latency percentiles, cache, slots)
  POST /admin/load, /admin/unload

One model is resident at a time (the box has 8 GB and also runs other jobs);
it loads on first use and unloads after `idle_unload_s` without requests.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import resource
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request as HTTPRequest
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from .chat import ChatTokenizer, cache_marks, parse_tool_calls, render, split_thinking
from .engine import Model, SamplingParams
from .scheduler import Request, Scheduler


# ------------------------------------------------------------------ models
class Manager:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.specs = {m["name"]: m for m in cfg["models"]}
        self.default = cfg.get("default") or cfg["models"][0]["name"]
        self.idle = float(cfg.get("idle_unload_s", 900))
        self.name: str | None = None
        self.sched: Scheduler | None = None
        self.model: Model | None = None
        self.tok: ChatTokenizer | None = None
        self.loaded_at = 0.0
        self.load_s = 0.0
        self.lock = threading.Lock()

    def get(self, name: str | None) -> tuple[Scheduler, ChatTokenizer, str]:
        name = name or self.default
        if name not in self.specs:
            # accept aliases like "qwen3.5" -> default
            name = self.default if name in ("default", "", None) else name
        if name not in self.specs:
            raise HTTPException(404, f"unknown model {name!r}; available: {sorted(self.specs)}")
        with self.lock:
            if self.name != name:
                self._unload()
                self._load(name)
            return self.sched, self.tok, name

    def _load(self, name: str) -> None:
        s = self.specs[name]
        t0 = time.time()
        ex = lambda k: str(Path(s[k]).expanduser())  # noqa: E731
        self.model = Model(ex("path"), threads=s.get("threads", 4))
        if s.get("draft_vocab"):
            ids = np.load(ex("draft_vocab"))[: s.get("draft_vocab_size", 32768)]
            self.model.set_draft_vocab(ids.astype(np.int32))
        self.tok = ChatTokenizer(ex("tokenizer"))
        self.sched = Scheduler(self.model, self.tok, n_slots=s.get("slots", 2), n_ctx=s.get("ctx", 8192),
                               max_batch=s.get("max_batch", 256), prefill_chunk=s.get("prefill_chunk", 256),
                               n_draft=s.get("draft", 2), spec=s.get("speculative", True),
                               snap_every=s.get("snap_every", 1024), snap_budget_mb=s.get("snap_budget_mb", 512),
                               cache_dir=str(Path(self.cfg.get("cache_dir", "~/.cache/edgeinfer")).expanduser() / name))
        self.name = name
        self.loaded_at = time.time()
        self.load_s = time.time() - t0

    def _unload(self) -> None:
        if self.sched:
            self.sched.shutdown()
        if self.model:
            self.model.close()
        self.sched = self.model = self.tok = None
        self.name = None

    def unload(self) -> None:
        with self.lock:
            self._unload()

    def maybe_idle_unload(self) -> None:
        with self.lock:
            if self.sched and not self.sched.busy() and time.time() - self.sched.last_activity > self.idle:
                self._unload()


MANAGER: Manager | None = None
API_KEY = os.environ.get("EDGEINFER_API_KEY")
STARTED = time.time()


@asynccontextmanager
async def lifespan(app):
    async def idle_task():
        while True:
            await asyncio.sleep(30)
            await asyncio.to_thread(MANAGER.maybe_idle_unload)
    t = asyncio.create_task(idle_task())
    if MANAGER.cfg.get("preload"):
        await asyncio.to_thread(MANAGER.get, None)
    yield
    t.cancel()
    MANAGER.unload()


app = FastAPI(title="edgeinfer", lifespan=lifespan)


def check_auth(req: HTTPRequest) -> None:
    if API_KEY and req.headers.get("authorization", "") != f"Bearer {API_KEY}":
        raise HTTPException(401, "invalid API key")


def sampling_from(body: dict) -> SamplingParams:
    g = body.get
    return SamplingParams(
        temperature=float(g("temperature", 0.7)), top_p=float(g("top_p", 0.8)), min_p=float(g("min_p", 0.0)),
        top_k=int(g("top_k", 20)), repeat_penalty=float(g("repetition_penalty", 1.0)),
        presence_penalty=float(g("presence_penalty", 0.0)), frequency_penalty=float(g("frequency_penalty", 0.0)),
        seed=int(g("seed") or 0))


async def run_request(sched: Scheduler, req: Request):
    """Submits and yields scheduler events as they arrive."""
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    req.emit = lambda e: loop.call_soon_threadsafe(q.put_nowait, e)
    sched.submit(req)
    try:
        while True:
            e = await q.get()
            yield e
            if e["type"] in ("done", "error"):
                return
    finally:
        if req.state != "done":
            sched.cancel(req)


def sse(obj) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(http: HTTPRequest):
    check_auth(http)
    body = await http.json()
    sched, tok, name = await asyncio.to_thread(MANAGER.get, body.get("model"))
    messages = body.get("messages") or []
    tools = body.get("tools") or None
    kw = body.get("chat_template_kwargs") or {}
    thinking = bool(kw.get("enable_thinking", body.get("enable_thinking", False)))
    keep_think = bool(body.get("keep_think", True))
    try:
        text = render(messages, tools, True, thinking, keep_think)
    except (ValueError, KeyError, TypeError) as e:
        raise HTTPException(400, f"bad messages: {e}")
    prompt = tok.encode(text)
    stop = body.get("stop") or []
    stop = [stop] if isinstance(stop, str) else list(stop)
    req = Request(prompt=prompt, params=sampling_from(body),
                  max_tokens=int(body.get("max_completion_tokens") or body.get("max_tokens") or 1024),
                  stop=stop, spec=bool(body.get("speculative", True)),
                  snap_at=cache_marks(tok, messages, prompt, tools, keep_think))
    rid = "chatcmpl-" + uuid.uuid4().hex[:16]
    created = int(time.time())
    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

    if body.get("stream"):
        async def gen():
            head = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": name}
            yield sse({**head, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                                            "finish_reason": None}]})
            full, in_think, tool_mode = "", thinking, False
            async for e in run_request(sched, req):
                if e["type"] == "token":
                    full += e["text"]
                    delta = e["text"]
                    if tools and "<tool_call>" in full:
                        tool_mode = True
                    if tool_mode:
                        continue
                    if in_think:
                        if "</think>" in full:
                            in_think = False
                            r_part, _, c_part = delta.partition("</think>")
                            if r_part:
                                yield sse({**head, "choices": [{"index": 0, "delta": {"reasoning_content": r_part},
                                                                "finish_reason": None}]})
                            c_part = c_part.lstrip("\n")
                            if c_part:
                                yield sse({**head, "choices": [{"index": 0, "delta": {"content": c_part},
                                                                "finish_reason": None}]})
                        else:
                            yield sse({**head, "choices": [{"index": 0, "delta": {"reasoning_content": delta},
                                                            "finish_reason": None}]})
                        continue
                    yield sse({**head, "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}]})
                elif e["type"] == "error":
                    yield sse({"error": {"message": e["message"]}})
                    break
                else:
                    finish = e["finish_reason"]
                    if tool_mode:
                        _, calls = parse_tool_calls(full.split("</think>")[-1])
                        if calls:
                            finish = "tool_calls"
                            yield sse({**head, "choices": [{"index": 0, "delta": {"tool_calls": [
                                {**c, "index": i} for i, c in enumerate(calls)]}, "finish_reason": None}]})
                    yield sse({**head, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                               "timing": e["timing"], **({"usage": e["usage"]} if include_usage else {})})
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    full, done = "", None
    async for e in run_request(sched, req):
        if e["type"] == "token":
            full += e["text"]
        elif e["type"] == "error":
            raise HTTPException(500, e["message"])
        else:
            done = e
    reasoning, answer = split_thinking(full) if thinking else ("", full)
    content, calls = parse_tool_calls(answer) if tools else (answer, [])
    msg = {"role": "assistant", "content": content if (content or not calls) else None}
    if reasoning:
        msg["reasoning_content"] = reasoning
    if calls:
        msg["tool_calls"] = calls
    return {"id": rid, "object": "chat.completion", "created": created, "model": name,
            "choices": [{"index": 0, "message": msg,
                         "finish_reason": "tool_calls" if calls else done["finish_reason"]}],
            "usage": done["usage"], "timing": done["timing"]}


@app.post("/v1/completions")
async def completions(http: HTTPRequest):
    check_auth(http)
    body = await http.json()
    sched, tok, name = await asyncio.to_thread(MANAGER.get, body.get("model"))
    prompt = body.get("prompt", "")
    ids = prompt if isinstance(prompt, list) else tok.encode(prompt)
    stop = body.get("stop") or []
    req = Request(prompt=ids, params=sampling_from(body), max_tokens=int(body.get("max_tokens") or 256),
                  stop=[stop] if isinstance(stop, str) else list(stop), spec=bool(body.get("speculative", True)))
    rid, created = "cmpl-" + uuid.uuid4().hex[:16], int(time.time())
    if body.get("stream"):
        async def gen():
            async for e in run_request(sched, req):
                if e["type"] == "token":
                    yield sse({"id": rid, "object": "text_completion", "created": created, "model": name,
                               "choices": [{"index": 0, "text": e["text"], "finish_reason": None}]})
                elif e["type"] == "done":
                    yield sse({"id": rid, "object": "text_completion", "created": created, "model": name,
                               "choices": [{"index": 0, "text": "", "finish_reason": e["finish_reason"]}],
                               "usage": e["usage"], "timing": e["timing"]})
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    text, done = "", None
    async for e in run_request(sched, req):
        if e["type"] == "token":
            text += e["text"]
        elif e["type"] == "error":
            raise HTTPException(500, e["message"])
        else:
            done = e
    return {"id": rid, "object": "text_completion", "created": created, "model": name,
            "choices": [{"index": 0, "text": text, "finish_reason": done["finish_reason"]}],
            "usage": done["usage"], "timing": done["timing"]}


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [
        {"id": n, "object": "model", "owned_by": "edgeinfer", "loaded": n == MANAGER.name,
         "description": s.get("description", ""), "file_bytes": Path(s["path"]).expanduser().stat().st_size
         if Path(s["path"]).expanduser().exists() else None} for n, s in MANAGER.specs.items()]}


@app.get("/health")
async def health():
    return {"status": "ok", "loaded": MANAGER.name, "uptime_s": round(time.time() - STARTED)}


def _pct(xs, p):
    return round(float(np.percentile(xs, p)), 4) if xs else None


def _rss_mb() -> int:
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


def stats_dict() -> dict:
    s = MANAGER.sched
    out = {"loaded": MANAGER.name, "load_s": round(MANAGER.load_s, 2), "uptime_s": round(time.time() - STARTED),
           "rss_mb": _rss_mb(), "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
           "default": MANAGER.default, "idle_unload_s": MANAGER.idle}
    if s:
        m = s.metrics
        out.update({
            "model_info": MANAGER.model.info, "threads": MANAGER.model.threads,
            "requests": m["requests"], "prompt_tokens": m["prompt_tokens"], "cached_tokens": m["cached_tokens"],
            "cache_hit_rate": round(m["cached_tokens"] / m["prompt_tokens"], 4) if m["prompt_tokens"] else None,
            "generated_tokens": m["gen_tokens"], "steps": m["steps"], "forward_s": round(m["forward_s"], 2),
            "spec_acceptance": round(m["accepted"] / m["drafted"], 4) if m["drafted"] else None,
            "spec_drafted": m["drafted"], "snapshots": m["snapshots"], "disk_restores": m["disk_restores"],
            "disk_prefixes": [len(d.tokens) for d in s.disk],
            "ttft_p50_s": _pct(m["ttft_s"], 50), "ttft_p95_s": _pct(m["ttft_s"], 95),
            "tpot_p50_ms": _pct([x * 1000 for x in m["tpot_s"]], 50),
            "decode_tok_s_p50": round(1 / np.percentile(m["tpot_s"], 50), 2) if m["tpot_s"] else None,
            "prefill_tok_s_p50": _pct(m["prefill_tok_s"], 50),
            "active": len(s.active), "queued": s.waiting.qsize(), "slots": s.snapshot_info(),
            "engine": s.ctx.stats(), "ctx_mb": round(s.ctx.nbytes() / 2**20)})
    return out


@app.get("/v1/stats")
async def stats():
    return stats_dict()


@app.get("/metrics")
async def metrics():
    d = stats_dict()
    lines = ["# HELP edgeinfer_up 1 if the server is running", "# TYPE edgeinfer_up gauge", "edgeinfer_up 1",
             f'edgeinfer_model_loaded{{model="{d["loaded"] or ""}"}} {1 if d["loaded"] else 0}',
             f"edgeinfer_rss_megabytes {d['rss_mb']}"]
    for k, typ in [("requests", "counter"), ("prompt_tokens", "counter"), ("cached_tokens", "counter"),
                   ("generated_tokens", "counter"), ("spec_drafted", "counter"), ("active", "gauge"),
                   ("queued", "gauge"), ("ttft_p50_s", "gauge"), ("ttft_p95_s", "gauge"),
                   ("tpot_p50_ms", "gauge"), ("spec_acceptance", "gauge"), ("cache_hit_rate", "gauge")]:
        if d.get(k) is not None:
            lines += [f"# TYPE edgeinfer_{k} {typ}", f"edgeinfer_{k} {d[k]}"]
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


@app.post("/admin/load")
async def admin_load(http: HTTPRequest):
    check_auth(http)
    body = await http.json()
    _, _, name = await asyncio.to_thread(MANAGER.get, body.get("model"))
    return {"loaded": name, "load_s": round(MANAGER.load_s, 2)}


@app.post("/admin/unload")
async def admin_unload(http: HTTPRequest):
    check_auth(http)
    await asyncio.to_thread(MANAGER.unload)
    return {"loaded": None}


@app.exception_handler(HTTPException)
async def http_error(_, exc: HTTPException):
    return JSONResponse({"error": {"message": exc.detail, "code": exc.status_code}}, status_code=exc.status_code)


def main() -> None:
    global MANAGER
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    a = ap.parse_args()
    MANAGER = Manager(json.loads(Path(a.config).read_text()))
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
