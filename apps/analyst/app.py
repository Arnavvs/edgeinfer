"""Data Desk: a private, login-protected chat over your collected data.

    python -m apps.analyst.app --config dataset.json [--host 100.x.y.z --port 8780]
           [--llm http://127.0.0.1:8091] [--state ~/.config/edgeinfer-analyst]

The answers come from the local edgeinfer server (OpenAI API); the app adds
auth, conversation storage, the SQL/search tools and the UI.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import time
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse

from .agent import Agent
from .auth import RateLimiter, Sessions, Users
from .retrieval import LazyEmbedder, Retriever

HERE = Path(__file__).resolve().parent
COOKIE = "dd_session"

app = FastAPI(title="Data Desk", docs_url=None, redoc_url=None, openapi_url=None)
S: dict = {}  # runtime state


# ------------------------------------------------------------------ storage
def db() -> sqlite3.Connection:
    c = sqlite3.connect(S["state"] / "conversations.db")
    c.row_factory = sqlite3.Row
    c.executescript("""
        CREATE TABLE IF NOT EXISTS conv(id TEXT PRIMARY KEY, user TEXT, title TEXT, model TEXT,
                                        created REAL, updated REAL);
        CREATE TABLE IF NOT EXISTS msg(conv TEXT, idx INTEGER, role TEXT, body TEXT, created REAL,
                                       PRIMARY KEY(conv, idx));
        CREATE TABLE IF NOT EXISTS feedback(conv TEXT, question TEXT, answer TEXT, rating INTEGER, note TEXT,
                                            model TEXT, user TEXT, created REAL);
    """)
    return c


def conv_messages(c, cid: str) -> list[dict]:
    return [json.loads(r["body"]) for r in c.execute("SELECT body FROM msg WHERE conv=? ORDER BY idx", (cid,))]


# ------------------------------------------------------------------ auth
def current_user(req: Request) -> str | None:
    return S["sessions"].verify(req.cookies.get(COOKIE))


def require_user(req: Request) -> str:
    u = current_user(req)
    if not u:
        raise HTTPException(401, "login required")
    if req.method == "POST" and req.headers.get("x-requested-with") != "datadesk":
        raise HTTPException(403, "missing request header")  # CSRF guard on top of SameSite=Strict
    return u


@app.middleware("http")
async def security_headers(req: Request, call_next):
    resp = await call_next(req)
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Content-Security-Policy"] = ("default-src 'self'; style-src 'self' 'unsafe-inline'; "
                                               "script-src 'self' 'unsafe-inline'; img-src 'self' data:")
    return resp


@app.get("/login", response_class=HTMLResponse)
async def login_page(req: Request):
    if current_user(req):
        return RedirectResponse("/", 303)
    return (HERE / "static" / "login.html").read_text()


@app.post("/login")
async def login(req: Request, username: str = Form(...), password: str = Form(...)):
    ip = req.client.host if req.client else "?"
    if not S["limiter"].allowed(ip):
        return HTMLResponse((HERE / "static" / "login.html").read_text().replace(
            "<!--msg-->", '<p class="err">Too many attempts. Try again in a few minutes.</p>'), 429)
    ok = await asyncio.to_thread(S["users"].check, username.strip(), password)
    if not ok:
        S["limiter"].failed(ip)
        await asyncio.sleep(0.6)
        return HTMLResponse((HERE / "static" / "login.html").read_text().replace(
            "<!--msg-->", '<p class="err">Wrong username or password.</p>'), 401)
    resp = RedirectResponse("/", 303)
    resp.set_cookie(COOKIE, S["sessions"].issue(username.strip()), httponly=True, samesite="strict",
                    secure=req.url.scheme == "https", max_age=S["sessions"].ttl)
    return resp


@app.post("/logout")
async def logout():
    resp = RedirectResponse("/login", 303)
    resp.delete_cookie(COOKIE)
    return resp


# ------------------------------------------------------------------ pages
@app.get("/", response_class=HTMLResponse)
async def index(req: Request):
    if not current_user(req):
        return RedirectResponse("/login", 303)
    return (HERE / "static" / "index.html").read_text()


@app.get("/api/me")
async def me(req: Request):
    return {"user": require_user(req), "dataset": S["cfg"].get("name", "data"),
            "default_model": S["cfg"].get("default_model"), "starters": S["cfg"].get("starter_questions", [])}


@app.get("/api/models")
async def models(req: Request):
    require_user(req)
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{S['llm']}/v1/models", headers=S["llm_headers"])
            return r.json()
    except httpx.HTTPError as e:
        return {"data": [], "error": f"inference server unreachable: {e}"}


@app.get("/api/data")
async def data_info(req: Request):
    require_user(req)
    c = sqlite3.connect(f"file:{Path(S['cfg']['analytics_db']).expanduser()}?mode=ro", uri=True)
    try:
        build = json.loads(c.execute("SELECT value FROM meta WHERE key='build'").fetchone()[0])
    finally:
        c.close()
    return {"build": build, "tables": S["cfg"].get("tables", {}), "sources": S["retriever"].sources()}


@app.get("/api/conversations")
async def list_convs(req: Request):
    u = require_user(req)
    with db() as c:
        rows = c.execute("SELECT id, title, model, updated FROM conv WHERE user=? ORDER BY updated DESC LIMIT 200", (u,))
        return [dict(r) for r in rows]


@app.get("/api/conversations/{cid}")
async def get_conv(cid: str, req: Request):
    u = require_user(req)
    with db() as c:
        row = c.execute("SELECT * FROM conv WHERE id=? AND user=?", (cid, u)).fetchone()
        if not row:
            raise HTTPException(404, "not found")
        return {**dict(row), "messages": conv_messages(c, cid)}


@app.post("/api/conversations/{cid}/delete")
async def del_conv(cid: str, req: Request):
    u = require_user(req)
    with db() as c:
        c.execute("DELETE FROM msg WHERE conv IN (SELECT id FROM conv WHERE id=? AND user=?)", (cid, u))
        c.execute("DELETE FROM conv WHERE id=? AND user=?", (cid, u))
    return {"ok": True}


@app.post("/api/chat")
async def chat(req: Request):
    u = require_user(req)
    body = await req.json()
    question = (body.get("message") or "").strip()
    if not question:
        raise HTTPException(400, "empty message")
    model = body.get("model") or S["cfg"].get("default_model")
    cid = body.get("conversation") or uuid.uuid4().hex[:16]
    with db() as c:
        row = c.execute("SELECT user FROM conv WHERE id=?", (cid,)).fetchone()
        if row and row["user"] != u:
            raise HTTPException(404, "not found")
        if not row:
            c.execute("INSERT INTO conv VALUES (?,?,?,?,?,?)", (cid, u, question[:80], model, time.time(), time.time()))
        history = conv_messages(c, cid)
    # keep the prompt inside the context budget: drop the oldest whole turns
    budget = int(S["cfg"].get("history_chars", 24000))
    while history and sum(len(json.dumps(m)) for m in history) > budget:
        nxt = next((i for i, m in enumerate(history[1:], 1) if m["role"] == "user"), len(history))
        history = history[nxt:]

    q: asyncio.Queue = asyncio.Queue()

    async def emit(e):
        await q.put(e)

    async def work():
        try:
            new = await S["agent"].run(history, question, model, emit)
            with db() as c:
                n = c.execute("SELECT COUNT(*) FROM msg WHERE conv=?", (cid,)).fetchone()[0]
                for i, m in enumerate([{"role": "user", "content": question}] + new):
                    c.execute("INSERT INTO msg VALUES (?,?,?,?,?)", (cid, n + i, m["role"], json.dumps(m), time.time()))
                c.execute("UPDATE conv SET updated=?, model=? WHERE id=?", (time.time(), model, cid))
        except Exception as e:  # surface to the UI
            await q.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            await q.put(None)

    task = asyncio.create_task(work())

    async def stream():
        yield f"data: {json.dumps({'type': 'conversation', 'id': cid})}\n\n"
        try:
            while True:
                e = await q.get()
                if e is None:
                    break
                yield f"data: {json.dumps(e, ensure_ascii=False)}\n\n"
        finally:
            if not task.done():
                task.cancel()
    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.post("/api/feedback")
async def feedback(req: Request):
    """Thumbs up/down on an answer; rated answers seed the eval set (bench/analyst_eval.py)."""
    u = require_user(req)
    b = await req.json()
    rating = 1 if b.get("rating", 0) > 0 else -1
    with db() as c:
        if not c.execute("SELECT 1 FROM conv WHERE id=? AND user=?", (b.get("conversation"), u)).fetchone():
            raise HTTPException(404, "not found")
        c.execute("INSERT INTO feedback VALUES (?,?,?,?,?,?,?,?)",
                  (b.get("conversation"), str(b.get("question", ""))[:2000], str(b.get("answer", ""))[:8000], rating,
                   str(b.get("note", ""))[:1000], b.get("model"), u, time.time()))
    return {"ok": True}


@app.get("/api/health")
async def health():
    return {"ok": True}


@app.exception_handler(HTTPException)
async def http_error(req: Request, exc: HTTPException):
    if exc.status_code == 401 and not req.url.path.startswith("/api/"):
        return RedirectResponse("/login", 303)
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--llm", default="http://127.0.0.1:8091")
    ap.add_argument("--state", default=str(Path.home() / ".config" / "edgeinfer-analyst"))
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).expanduser().read_text())
    state = Path(a.state).expanduser()
    state.mkdir(parents=True, exist_ok=True)
    os.chmod(state, 0o700)
    S.update(cfg=cfg, state=state, llm=a.llm, users=Users(state / "users.json"),
             sessions=Sessions(state / "session.key"), limiter=RateLimiter(),
             llm_headers={"Authorization": f"Bearer {os.environ['EDGEINFER_API_KEY']}"}
             if os.environ.get("EDGEINFER_API_KEY") else {})
    S["retriever"] = Retriever(cfg["analytics_db"], cfg["vectors"], LazyEmbedder(cfg["embedder"], threads=2))
    S["agent"] = Agent(cfg, S["retriever"], a.llm, os.environ.get("EDGEINFER_API_KEY"))
    if not (state / "users.json").exists():
        print(f"!! no users yet: python -m apps.analyst.auth set-password <user> --users {state / 'users.json'}")
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning", proxy_headers=False)


if __name__ == "__main__":
    main()
