"""Data Desk: login-protected chat over your data, plus the public project page.

    python -m apps.analyst.app --config site.json [--host 127.0.0.1 --port 8780]
           [--llm http://127.0.0.1:8091] [--state ~/.config/edgeinfer-analyst]

`--config` is either one dataset config (apps/analyst/demo/demo.json shape) or a
site config with several workspaces and per-account rules:

    {"title": "edgeinfer",
     "workspaces": {"mine": "path/to/dataset.json", "demo": "path/to/demo.json", "chat": "chat"},
     "accounts": {"me":    {"workspaces": ["mine", "demo", "chat"]},
                  "guest": {"workspaces": ["demo", "chat"], "per_hour": 16, "models": ["qwen3.5-4b"],
                            "max_tokens": 700, "concurrent": 1}},
     "public": {"github": "...", "portfolio": "...", "guest": {"user": "guest", "password": "..."}}}

Passwords live separately (hashed) in the state dir; manage them with
`python -m apps.analyst.auth set-password <user>`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import time
import uuid
from collections import defaultdict
from pathlib import Path

import httpx
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse

from .agent import Agent, ChatAgent
from .auth import RateLimiter, Sessions, Users
from .retrieval import LazyEmbedder, Retriever

HERE = Path(__file__).resolve().parent
COOKIE = "dd_session"

app = FastAPI(title="Data Desk", docs_url=None, redoc_url=None, openapi_url=None)
S: dict = {}  # runtime state
INFLIGHT: dict[str, int] = defaultdict(int)


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
        CREATE TABLE IF NOT EXISTS usage(user TEXT, at REAL);
        CREATE INDEX IF NOT EXISTS usage_user_at ON usage(user, at);
    """)
    cols = {r[1] for r in c.execute("PRAGMA table_info(conv)")}
    if "workspace" not in cols:
        c.execute("ALTER TABLE conv ADD COLUMN workspace TEXT")
    return c


def conv_messages(c, cid: str) -> list[dict]:
    return [json.loads(r["body"]) for r in c.execute("SELECT body FROM msg WHERE conv=? ORDER BY idx", (cid,))]


# ------------------------------------------------------------------ accounts
def account(user: str) -> dict:
    acc = S["accounts"].get(user)
    if acc is None:  # accounts not listed in the site config get every workspace, no cap
        acc = {"workspaces": list(S["workspaces"])}
    return acc


def used_last_hour(c, user: str) -> int:
    return c.execute("SELECT COUNT(*) FROM usage WHERE user=? AND at > ?", (user, time.time() - 3600)).fetchone()[0]


def quota(c, user: str) -> dict | None:
    cap = account(user).get("per_hour")
    if not cap:
        return None
    used = used_last_hour(c, user)
    oldest = c.execute("SELECT MIN(at) FROM usage WHERE user=? AND at > ?", (user, time.time() - 3600)).fetchone()[0]
    return {"per_hour": cap, "used": used, "left": max(0, cap - used),
            "resets_in_s": int(oldest + 3600 - time.time()) if oldest and used >= cap else 0}


# ------------------------------------------------------------------ auth
def client_ip(req: Request) -> str:
    """Behind Tailscale Serve/Funnel the TCP peer is this machine; the proxy
    passes the visitor's address in X-Forwarded-For."""
    peer = req.client.host if req.client else "?"
    xff = req.headers.get("x-forwarded-for")
    if xff and (peer.startswith("127.") or peer == S.get("host") or peer.startswith("100.") or peer == "::1"):
        return xff.split(",")[0].strip()
    return peer


def is_https(req: Request) -> bool:
    return req.url.scheme == "https" or req.headers.get("x-forwarded-proto", "") == "https"


def current_user(req: Request) -> str | None:
    u = S["sessions"].verify(req.cookies.get(COOKIE))
    return u if u and u in S["users"].load() else None  # a deleted account loses its sessions


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
    resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; script-src 'self' 'unsafe-inline'; img-src 'self' data:; "
        "frame-ancestors 'none'")
    if is_https(req):
        resp.headers["Strict-Transport-Security"] = "max-age=31536000"
    return resp


def login_html(msg: str = "") -> str:
    html = (HERE / "static" / "login.html").read_text()
    g = S["public"].get("guest")
    hint = (f'<p class="hint">Recruiter / guest login: <code>{g["user"]}</code> / <code>{g["password"]}</code> '
            f'({account(g["user"]).get("per_hour", "unlimited")} questions per hour)</p>') if g else ""
    return html.replace("<!--msg-->", msg).replace("<!--hint-->", hint)


@app.get("/login", response_class=HTMLResponse)
async def login_page(req: Request):
    if current_user(req):
        return RedirectResponse("/app", 303)
    return login_html()


@app.post("/login")
async def login(req: Request, username: str = Form(...), password: str = Form(...)):
    ip, user = client_ip(req), username.strip()
    # per address, and per account so a shared proxy address cannot be used to lock anyone out for long
    if not S["limiter"].allowed(ip) or not S["user_limiter"].allowed(user):
        return HTMLResponse(login_html('<p class="err">Too many attempts. Try again in a few minutes.</p>'), 429)
    ok = await asyncio.to_thread(S["users"].check, user, password)
    if not ok:
        S["limiter"].failed(ip)
        S["user_limiter"].failed(user)
        await asyncio.sleep(0.6)
        return HTMLResponse(login_html('<p class="err">Wrong username or password.</p>'), 401)
    resp = RedirectResponse("/app", 303)
    resp.set_cookie(COOKIE, S["sessions"].issue(user), httponly=True, samesite="strict",
                    secure=is_https(req), max_age=S["sessions"].ttl)
    return resp


@app.post("/logout")
async def logout():
    resp = RedirectResponse("/", 303)
    resp.delete_cookie(COOKIE)
    return resp


# ------------------------------------------------------------------ pages
@app.get("/", response_class=HTMLResponse)
async def landing(req: Request):
    page = (HERE / "static" / "landing.html").read_text()
    data = json.dumps({**S["public"], "title": S["title"], "signed_in": current_user(req) is not None})
    return page.replace("__PUBLIC__", data.replace("</", "<" + chr(92) + "/"))


@app.get("/app", response_class=HTMLResponse)
async def index(req: Request):
    if not current_user(req):
        return RedirectResponse("/login", 303)
    return (HERE / "static" / "index.html").read_text()


async def llm_stats() -> dict:
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            s = (await c.get(f"{S['llm']}/v1/stats", headers=S["llm_headers"])).json()
            m = (await c.get(f"{S['llm']}/v1/models", headers=S["llm_headers"])).json()
    except (httpx.HTTPError, ValueError):
        return {"up": False}
    return {"up": True, "loaded": s.get("loaded"), "default": s.get("default"), "requests": s.get("requests", 0),
            "generated_tokens": s.get("generated_tokens", 0), "decode_tok_s": s.get("decode_tok_s_p50"),
            "ttft_p50_s": s.get("ttft_p50_s"), "cache_hit_rate": s.get("cache_hit_rate"),
            "spec_acceptance": s.get("spec_acceptance"), "uptime_s": s.get("uptime_s"),
            "models": [{"id": x["id"], "description": x.get("description", "")} for x in m.get("data", [])]}


@app.get("/api/public/status")
async def public_status():
    """No auth: what the landing page shows. Nothing about users or data."""
    return await llm_stats()


@app.get("/api/me")
async def me(req: Request):
    u = require_user(req)
    acc = account(u)
    ws = [{"id": w, "title": S["workspaces"][w]["title"], "kind": S["workspaces"][w]["kind"],
           "starters": S["workspaces"][w]["starters"]} for w in acc["workspaces"] if w in S["workspaces"]]
    with db() as c:
        q = quota(c, u)
    return {"user": u, "workspaces": ws, "models": acc.get("models"), "default_model": S["default_model"],
            "quota": q, "guest": bool(acc.get("per_hour"))}


@app.get("/api/models")
async def models(req: Request):
    u = require_user(req)
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            data = (await c.get(f"{S['llm']}/v1/models", headers=S["llm_headers"])).json()
    except httpx.HTTPError as e:
        return {"data": [], "error": f"inference server unreachable: {e}"}
    allowed = account(u).get("models")
    if allowed:
        data["data"] = [m for m in data.get("data", []) if m["id"] in allowed]
    return data


@app.get("/api/conversations")
async def list_convs(req: Request):
    u = require_user(req)
    with db() as c:
        rows = c.execute("SELECT id, title, model, workspace, updated FROM conv WHERE user=? "
                         "ORDER BY updated DESC LIMIT 200", (u,))
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
    acc = account(u)
    body = await req.json()
    question = (body.get("message") or "").strip()[:4000]
    if not question:
        raise HTTPException(400, "empty message")
    ws_id = body.get("workspace") or acc["workspaces"][0]
    if ws_id not in acc["workspaces"] or ws_id not in S["workspaces"]:
        raise HTTPException(403, "this account cannot use that workspace")
    ws = S["workspaces"][ws_id]
    model = body.get("model") or S["default_model"]
    if acc.get("models") and model not in acc["models"]:
        model = acc["models"][0]
    if INFLIGHT[u] >= acc.get("concurrent", 4):
        raise HTTPException(429, "one question at a time on this account; wait for the current answer")
    cid = body.get("conversation") or uuid.uuid4().hex[:16]
    with db() as c:
        q = quota(c, u)
        if q and q["left"] <= 0:
            mins = max(1, round(q["resets_in_s"] / 60))
            raise HTTPException(429, f"this account is limited to {q['per_hour']} questions per hour; "
                                     f"the next one is available in about {mins} min")
        row = c.execute("SELECT user, workspace FROM conv WHERE id=?", (cid,)).fetchone()
        if row and row["user"] != u:
            raise HTTPException(404, "not found")
        if row and row["workspace"] and row["workspace"] != ws_id:
            raise HTTPException(400, "this conversation belongs to another workspace")
        if not row:
            c.execute("INSERT INTO conv(id, user, title, model, created, updated, workspace) VALUES (?,?,?,?,?,?,?)",
                      (cid, u, question[:80], model, time.time(), time.time(), ws_id))
        c.execute("INSERT INTO usage VALUES (?, ?)", (u, time.time()))
        history = conv_messages(c, cid)
    budget = int(ws["cfg"].get("history_chars", 24000))
    while history and sum(len(json.dumps(m)) for m in history) > budget:
        nxt = next((i for i, m in enumerate(history[1:], 1) if m["role"] == "user"), len(history))
        history = history[nxt:]

    qq: asyncio.Queue = asyncio.Queue()

    async def emit(e):
        await qq.put(e)

    async def work():
        INFLIGHT[u] += 1
        try:
            new = await ws["agent"].run(history, question, model, emit, max_tokens=acc.get("max_tokens", 900))
            with db() as c:
                n = c.execute("SELECT COUNT(*) FROM msg WHERE conv=?", (cid,)).fetchone()[0]
                for i, m in enumerate([{"role": "user", "content": question}] + new):
                    c.execute("INSERT INTO msg VALUES (?,?,?,?,?)", (cid, n + i, m["role"], json.dumps(m), time.time()))
                c.execute("UPDATE conv SET updated=?, model=? WHERE id=?", (time.time(), model, cid))
        except Exception as e:  # surface to the UI
            await qq.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            INFLIGHT[u] -= 1
            await qq.put(None)

    task = asyncio.create_task(work())

    async def stream():
        yield f"data: {json.dumps({'type': 'conversation', 'id': cid, 'workspace': ws_id})}\n\n"
        try:
            while True:
                e = await qq.get()
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


# ------------------------------------------------------------------ startup
def load_site(path: str) -> dict:
    p = Path(path).expanduser()
    cfg = json.loads(p.read_text())
    if "workspaces" not in cfg:  # a single dataset config
        cfg = {"title": cfg.get("name", "Data Desk"), "workspaces": {"data": str(p)},
               "default_model": cfg.get("default_model")}
    base = p.parent
    ws = {}
    for wid, src in cfg["workspaces"].items():
        if src == "chat":
            ws[wid] = {"kind": "chat", "title": "Chat with the model", "starters": [
                "Explain speculative decoding in three sentences.",
                "What is the difference between RAG and fine-tuning?",
                "Write a SQL query that finds the top 5 customers by revenue."], "cfg": {}}
            continue
        dp = Path(src).expanduser()
        dp = dp if dp.is_absolute() else base / dp
        d = json.loads(dp.read_text())
        ws[wid] = {"kind": "data", "title": d.get("name", wid), "starters": d.get("starter_questions", []), "cfg": d}
    return {"title": cfg.get("title", "edgeinfer"), "workspaces": ws, "accounts": cfg.get("accounts", {}),
            "public": cfg.get("public", {}), "default_model": cfg.get("default_model")
            or next((w["cfg"].get("default_model") for w in ws.values() if w["cfg"].get("default_model")), None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--llm", default="http://127.0.0.1:8091")
    ap.add_argument("--state", default=str(Path.home() / ".config" / "edgeinfer-analyst"))
    a = ap.parse_args()
    site = load_site(a.config)
    state = Path(a.state).expanduser()
    state.mkdir(parents=True, exist_ok=True)
    os.chmod(state, 0o700)
    key = os.environ.get("EDGEINFER_API_KEY")
    S.update(site, state=state, llm=a.llm, host=a.host, users=Users(state / "users.json"),
             sessions=Sessions(state / "session.key"), limiter=RateLimiter(),
             user_limiter=RateLimiter(limit=10, window_s=900),
             llm_headers={"Authorization": f"Bearer {key}"} if key else {})
    for w in S["workspaces"].values():
        if w["kind"] == "chat":
            w["agent"] = ChatAgent(a.llm, key)
        else:
            w["retriever"] = Retriever(w["cfg"]["analytics_db"], w["cfg"]["vectors"],
                                       LazyEmbedder(w["cfg"]["embedder"], threads=2))
            w["agent"] = Agent(w["cfg"], w["retriever"], a.llm, key)
    if not (state / "users.json").exists():
        print(f"!! no users yet: python -m apps.analyst.auth set-password <user> --users {state / 'users.json'}")
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning", proxy_headers=False)


if __name__ == "__main__":
    main()
