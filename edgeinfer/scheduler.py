"""Request scheduler: continuous batching, prefix cache, speculative decoding.

One thread owns the engine. Every step it builds a single batch from
  * one decode token for every generating request (or, when exactly one
    request is generating, a VERIFY segment [pending, draft1..draftK] whose
    drafts come from the model's own MTP head), and
  * a prefill chunk for requests still encoding their prompt,
runs one forward pass, samples, streams tokens, and repeats.

Prefix cache. A slot keeps its token history; requests go to the slot that
shares the longest prefix with them. Hybrid models carry a recurrent state
that cannot be rewound, so the cache stores *snapshots* of that state at
chosen lengths (chat-turn boundaries, every `snap_every` tokens, end of each
answer); a request resumes from the longest snapshot inside its common
prefix and only encodes the rest.
"""
from __future__ import annotations

import ctypes as C
import hashlib
import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .chat import ChatTokenizer, StreamDecoder
from .engine import SEG_HIDDEN, SEG_VERIFY, Context, Model, SamplingParams


@dataclass
class Request:
    prompt: list[int]
    params: SamplingParams
    max_tokens: int = 512
    stop: list[str] = field(default_factory=list)
    emit: Callable[[dict], None] = lambda e: None
    spec: bool = True
    snap_at: list[int] = field(default_factory=list)  # prompt offsets worth a snapshot
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created: float = field(default_factory=time.time)
    # runtime
    slot: int = -1
    state: str = "queued"
    n_cached: int = 0
    pos: int = 0  # prompt tokens already in the slot
    pending: int | None = None
    out: list[int] = field(default_factory=list)
    decoder: StreamDecoder | None = None
    t_start: float = 0.0
    t_first: float = 0.0
    t_done: float = 0.0
    drafts: list[int] = field(default_factory=list)
    n_drafted: int = 0
    n_accepted: int = 0
    mtp_ok: bool = False
    cancelled: bool = False
    finish_reason: str | None = None
    rng: C.c_uint64 = field(default_factory=lambda: C.c_uint64(0))


@dataclass
class Snap:
    id: int
    length: int
    last_hidden: np.ndarray | None
    nbytes: int
    used: float


@dataclass
class DiskSnap:
    """A shared prefix (typically a system prompt) with its KV rows, persisted
    to disk so a restart does not have to re-encode it."""
    id: int
    tokens: list[int]
    last_hidden: np.ndarray | None
    nbytes: int


@dataclass
class SlotState:
    tokens: list[int] = field(default_factory=list)
    last_hidden: np.ndarray | None = None
    snaps: list[Snap] = field(default_factory=list)
    req: Request | None = None
    used: float = 0.0


def lcp(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    if n == 0:
        return 0
    x = np.asarray(a[:n], dtype=np.int64) != np.asarray(b[:n], dtype=np.int64)
    i = int(np.argmax(x))
    return n if not x[i] else i


class Scheduler:
    def __init__(self, model: Model, tok: ChatTokenizer, *, n_slots: int = 2, n_ctx: int = 8192,
                 max_batch: int = 256, prefill_chunk: int = 256, n_draft: int = 2, spec: bool = True,
                 snap_every: int = 1024, snap_budget_mb: int = 512, max_snaps_per_slot: int = 8,
                 cache_dir: str | None = None, disk_min_tokens: int = 256, max_disk_snaps: int = 4):
        self.model, self.tok = model, tok
        self.n_slots, self.n_ctx = n_slots, n_ctx
        self.max_batch, self.prefill_chunk = max_batch, prefill_chunk
        self.n_draft = n_draft if (spec and model.has_mtp) else 0
        self.snap_every, self.snap_budget = snap_every, snap_budget_mb << 20
        self.max_snaps = max_snaps_per_slot
        self.ctx = Context(model, n_slots=n_slots, n_ctx=n_ctx, max_batch=max_batch, max_draft=max(self.n_draft, 1))
        self.slots = [SlotState() for _ in range(n_slots)]
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self.disk_min, self.max_disk = disk_min_tokens, max_disk_snaps
        self.disk: list[DiskSnap] = []
        self.waiting: queue.Queue[Request] = queue.Queue()
        self.active: list[Request] = []
        self.inflight: list[Request] = []  # submitted and not yet done; read by busy()
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.stop_flag = False
        self.last_activity = time.time()
        self.metrics = {
            "requests": 0, "prompt_tokens": 0, "cached_tokens": 0, "gen_tokens": 0,
            "drafted": 0, "accepted": 0, "forward_s": 0.0, "steps": 0, "snapshots": 0,
            "ttft_s": [], "tpot_s": [], "prefill_tok_s": [], "disk_restores": 0,
        }
        self._load_disk()
        self.thread = threading.Thread(target=self._loop, name="edgeinfer-scheduler", daemon=True)
        self.thread.start()

    # ---------------------------------------------------------------- public
    def submit(self, req: Request) -> Request:
        req.decoder = StreamDecoder(self.tok)
        req.rng = C.c_uint64(req.params.seed or (time.time_ns() & 0xFFFFFFFFFFFF))
        with self.lock:
            self.inflight.append(req)
        self.waiting.put(req)
        self.wake.set()
        return req

    def cancel(self, req: Request) -> None:
        req.cancelled = True
        self.wake.set()

    def shutdown(self) -> None:
        self.stop_flag = True
        self.wake.set()
        self.thread.join(timeout=10)
        self.ctx.close()

    def busy(self) -> bool:
        # Not `active or waiting`: a request is in neither while _admit places it.
        with self.lock:
            self.inflight = [r for r in self.inflight if r.state != "done"]
            return bool(self.inflight)

    def snapshot_info(self) -> list[dict]:
        out = []
        for i, s in enumerate(self.slots):
            out.append({"slot": i, "tokens": len(s.tokens), "busy": s.req is not None,
                        "snapshots": [{"len": sn.length, "mb": round(sn.nbytes / 2**20, 1)} for sn in s.snaps]})
        return out

    # ---------------------------------------------------------------- admission
    def _admit(self) -> None:
        while len(self.active) < self.n_slots:
            try:
                req = self.waiting.get_nowait()
            except queue.Empty:
                return
            if req.cancelled:
                self._finish(req, "cancelled")
                continue
            if len(req.prompt) + 1 >= self.n_ctx:
                req.emit({"type": "error", "message": f"prompt has {len(req.prompt)} tokens; context is {self.n_ctx}"})
                req.state = "done"
                continue
            self._place(req)
            self.active.append(req)
            self.metrics["requests"] += 1
            self.metrics["prompt_tokens"] += len(req.prompt)

    def _place(self, req: Request) -> None:
        P = req.prompt
        best = (-1, -1, None, 0.0)  # reuse, slot, snap, -lru
        for i, s in enumerate(self.slots):
            if s.req is not None:
                continue
            common = lcp(s.tokens, P)
            limit = min(common, len(P) - 1)
            reuse, snap = 0, None
            if common == len(s.tokens) and len(s.tokens) <= len(P) - 1:
                reuse = len(s.tokens)  # the slot already holds exactly this prefix
            for sn in s.snaps:
                if sn.length <= limit and sn.length > reuse:
                    reuse, snap = sn.length, sn
            key = (reuse, -s.used)
            if (best[1] < 0) or key > (best[0], best[3]):
                best = (reuse, i, snap, -s.used)
        reuse, si, snap, _ = best
        s = self.slots[si]
        for d in self.disk:  # a persisted shared prefix beats a shorter per-slot one
            n = len(d.tokens)
            if reuse < n <= len(P) - 1 and P[:n] == d.tokens:
                self.ctx.snap_restore(si, d.id)
                s.tokens, s.last_hidden, reuse, snap = list(d.tokens), d.last_hidden, n, None
                self.metrics["disk_restores"] += 1
                self._invalidate_snaps(si, 0)
                break
        else:
            d = None
        if d is not None:
            pass
        elif snap is not None:
            self.ctx.snap_restore(si, snap.id)
            snap.used = time.time()
            s.tokens = P[:reuse]
            s.last_hidden = None if snap.last_hidden is None else snap.last_hidden.copy()
        elif reuse == 0 or reuse != len(s.tokens):
            self.ctx.reset(si)
            s.tokens, s.last_hidden, reuse = [], None, 0
        self._invalidate_snaps(si, reuse)  # positions >= reuse are about to be overwritten
        s.req = req
        s.used = time.time()
        req.slot, req.pos, req.n_cached = si, reuse, reuse
        req.state = "prefill"
        req.t_start = time.time()
        # MTP drafting needs MTP entries for every position before the new tokens
        req.mtp_ok = self.n_draft > 0 and req.spec and (
            reuse == 0 or (s.last_hidden is not None and self.ctx.mtp_len(si) >= reuse - 1))
        if req.mtp_ok:
            self.ctx.mtp_truncate(si, max(0, reuse - 1))
        self.metrics["cached_tokens"] += reuse

    # ---------------------------------------------------------------- snapshots
    def _snapshot(self, si: int) -> None:
        s = self.slots[si]
        L = len(s.tokens)
        if L == 0 or any(sn.length == L for sn in s.snaps):
            return
        sid = self.ctx.snap_save(si, False)
        s.snaps.append(Snap(sid, L, None if s.last_hidden is None else s.last_hidden.copy(),
                            self.ctx.snap_bytes(sid), time.time()))
        self.metrics["snapshots"] += 1
        # per-slot cap, then a global byte budget (least recently used first)
        while len(s.snaps) > self.max_snaps:
            old = min(s.snaps, key=lambda x: x.used)
            self._drop(si, old)
        total = sum(sn.nbytes for sl in self.slots for sn in sl.snaps)
        while total > self.snap_budget:
            sl_i, old = min(((i, sn) for i, sl in enumerate(self.slots) for sn in sl.snaps), key=lambda x: x[1].used)
            self._drop(sl_i, old)
            total -= old.nbytes

    # ---------------------------------------------------------------- disk cache
    def _key(self, tokens: list[int]) -> str:
        return hashlib.sha1(np.asarray(tokens, dtype=np.int32).tobytes()).hexdigest()[:16]

    def _load_disk(self) -> None:
        if not self.cache_dir or not self.cache_dir.exists():
            return
        metas = sorted(self.cache_dir.glob("*.json"), key=lambda p: -p.stat().st_mtime)[: self.max_disk]
        for meta in metas:
            try:
                info = json.loads(meta.read_text())
                sid = self.ctx.snap_read(str(meta.with_suffix(".snap")))
                lh = np.load(meta.with_suffix(".npy")) if meta.with_suffix(".npy").exists() else None
                self.disk.append(DiskSnap(sid, info["tokens"], lh, self.ctx.snap_bytes(sid)))
            except Exception:
                continue  # stale or from another model build: ignore

    def _persist(self, si: int) -> None:
        if not self.cache_dir:
            return
        toks = list(self.slots[si].tokens)
        if any(d.tokens == toks for d in self.disk):
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key = self._key(toks)
        sid = self.ctx.snap_save(si, True)
        self.ctx.snap_write(sid, str(self.cache_dir / f"{key}.snap"))
        lh = self.slots[si].last_hidden
        if lh is not None:
            np.save(self.cache_dir / f"{key}.npy", lh)
        (self.cache_dir / f"{key}.json").write_text(json.dumps({"tokens": toks, "saved": time.time()}))
        self.disk.append(DiskSnap(sid, toks, None if lh is None else lh.copy(), self.ctx.snap_bytes(sid)))
        while len(self.disk) > self.max_disk:
            old = self.disk.pop(0)
            self.ctx.snap_free(old.id)

    def _drop(self, si: int, sn: Snap) -> None:
        self.ctx.snap_free(sn.id)
        self.slots[si].snaps.remove(sn)

    def _invalidate_snaps(self, si: int, keep_upto: int) -> None:
        """Snapshots beyond the point where the slot's history diverged are stale."""
        for sn in list(self.slots[si].snaps):
            if sn.length > keep_upto:
                self._drop(si, sn)

    # ---------------------------------------------------------------- main loop
    def _loop(self) -> None:
        while not self.stop_flag:
            self._admit()
            if not self.active:
                self.wake.wait(0.5)
                self.wake.clear()
                continue
            try:
                self._step()
            except Exception as e:  # never kill the thread; fail the batch
                for r in list(self.active):
                    r.emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
                    self._finish(r, "error")
            self.last_activity = time.time()

    def _step(self) -> None:
        for r in list(self.active):
            if r.cancelled:
                self._finish(r, "cancelled")
        if not self.active:
            return
        decoding = [r for r in self.active if r.state == "decode"]
        prefilling = [r for r in self.active if r.state == "prefill"]
        spec = len(decoding) == 1 and not prefilling and decoding[0].mtp_ok and decoding[0].drafts
        segs, plan = [], []
        budget = self.max_batch
        for r in decoding:
            si = r.slot
            if spec:
                room = self.n_ctx - len(self.slots[si].tokens) - 1
                drafts = r.drafts[:max(0, min(len(r.drafts), room - 1))]
                toks = [r.pending] + drafts
                segs.append((si, toks, len(toks), SEG_VERIFY))
                plan.append(("spec", r, drafts))
            else:
                segs.append((si, [r.pending], 1, 0))
                plan.append(("dec", r, None))
            budget -= len(segs[-1][1])
        for r in prefilling:
            if budget <= 0:
                break
            end = min(len(r.prompt), r.pos + min(self.prefill_chunk, budget))
            # stop chunks exactly at snapshot marks so the snapshot lands there
            marks = [m for m in r.snap_at if r.pos < m < end]
            nxt = (r.pos // self.snap_every + 1) * self.snap_every
            if r.pos < nxt < end:
                marks.append(nxt)
            if marks:
                end = min(marks)
            toks = r.prompt[r.pos:end]
            last = end == len(r.prompt)
            segs.append((r.slot, toks, 1 if last else 0, SEG_HIDDEN if r.mtp_ok else 0))
            plan.append(("pre", r, (r.pos, end, last)))
            budget -= len(toks)
        if not segs:
            return

        t0 = time.time()
        self.ctx.forward(segs)
        self.metrics["forward_s"] += time.time() - t0
        self.metrics["steps"] += 1

        row = 0
        tok_off = 0
        for (kind, r, extra), seg in zip(plan, segs):
            n_tok, n_log = len(seg[1]), seg[2]
            if kind == "dec":
                self._after_decode(r, row)
            elif kind == "spec":
                self._after_spec(r, row, extra)
            else:
                self._after_prefill(r, row, tok_off, extra)
            row += n_log
            tok_off += n_tok

    # ---------------------------------------------------------------- per-kind
    def _history(self, r: Request, extra: list[int] = ()) -> np.ndarray:
        return np.asarray(r.out + list(extra), dtype=np.int32)

    def _after_prefill(self, r: Request, row: int, tok_off: int, span) -> None:
        a, b, last = span
        s = self.slots[r.slot]
        si = r.slot
        if r.mtp_ok:
            H = self.ctx.hidden_tokens(tok_off + (b - a))[tok_off:tok_off + (b - a)]
            # MTP entries p in [mtp_len, b-2] (b-1 too when the prompt continues)
            p0 = self.ctx.mtp_len(si)
            p1 = b - 1 if not last else b - 2
            if p0 < a - 1:
                r.mtp_ok = False
            elif p1 >= p0:
                hid = []
                for p in range(p0, p1 + 1):
                    hid.append(s.last_hidden if p == a - 1 else H[p - a])
                if any(h is None for h in hid):
                    r.mtp_ok = False
                else:
                    # Resuming from a snapshot adds the entry before the chunk, so a full
                    # chunk is max_batch + 1 entries: more than one engine call may take.
                    toks, hid = r.prompt[p0 + 1:p1 + 2], np.stack(hid)
                    for i in range(0, len(toks), self.max_batch):
                        self.ctx.mtp_forward(si, p0 + i, toks[i:i + self.max_batch], hid[i:i + self.max_batch])
            s.last_hidden = H[-1].copy()
        else:
            s.last_hidden = None  # unknown here; snapshots taken now must not offer MTP resume
        s.tokens.extend(r.prompt[a:b])
        r.pos = b
        if b in r.snap_at or b % self.snap_every == 0:
            self._snapshot(si)
            if r.snap_at and b == r.snap_at[0] and b >= self.disk_min:
                self._persist(si)
        if not last:
            return
        # first token
        now = time.time()
        r.t_first = now
        self.metrics["ttft_s"].append(now - r.created)
        if b - r.n_cached > 0:
            self.metrics["prefill_tok_s"].append((b - r.n_cached) / max(1e-6, now - r.t_start))
        if not r.mtp_ok:
            s.last_hidden = self.ctx.hidden(row).copy() if self.n_draft else None
        t, _ = self.ctx.sample(row, r.params.c(), self._history(r), r.rng)
        r.state = "decode"
        _, reason = self._emit_tokens(r, [t])
        if reason:
            self._finish(r, reason)
            return
        r.pending = t
        if r.mtp_ok:
            self._draft(r, len(s.tokens) - 1, [t], s.last_hidden[None, :])

    def _after_decode(self, r: Request, row: int) -> None:
        s = self.slots[r.slot]
        s.tokens.append(r.pending)
        if self.n_draft:
            s.last_hidden = self.ctx.hidden(row).copy()
        if r.mtp_ok:  # a plain step leaves the MTP head behind; it cannot catch up cheaply
            r.mtp_ok = False
            r.drafts = []
        t, _ = self.ctx.sample(row, r.params.c(), self._history(r), r.rng)
        _, reason = self._emit_tokens(r, [t])
        if reason:
            self._finish(r, reason)
            return
        r.pending = t

    def _after_spec(self, r: Request, row: int, drafts: list[int]) -> None:
        si = r.slot
        s = self.slots[si]
        L = len(s.tokens)  # position of the pending token
        accepted, nxt = [], None
        for i, d in enumerate(drafts):
            t, ok = self.ctx.spec_accept(row + i, r.params.c(), self._history(r, accepted), d, r.rng)
            if ok:
                accepted.append(d)
            else:
                nxt = t
                break
        if nxt is None:
            nxt, _ = self.ctx.sample(row + len(drafts), r.params.c(), self._history(r, accepted), r.rng)
        r.n_drafted += len(drafts)
        r.n_accepted += len(accepted)
        self.metrics["drafted"] += len(drafts)
        self.metrics["accepted"] += len(accepted)
        # Stream first: if the answer ends inside the accepted drafts (stop token,
        # max_tokens, stop string), only the drafts that reached the user may stay
        # in the model state, or the next turn's prompt no longer matches the cache.
        n_used, reason = self._emit_tokens(r, accepted + [nxt])
        accepted = accepted[:min(len(accepted), n_used)]
        n_keep = 1 + len(accepted)
        self.ctx.commit(si, n_keep)
        H = np.stack([self.ctx.hidden(row + i).copy() for i in range(n_keep)])
        s.tokens.append(r.pending)
        s.tokens.extend(accepted)
        s.last_hidden = H[-1].copy()
        if reason:
            self._finish(r, reason)
            return
        r.pending = nxt
        self.ctx.mtp_truncate(si, L)
        self._draft(r, L, accepted + [nxt], H)

    def _draft(self, r: Request, pos0: int, tokens: list[int], H: np.ndarray) -> None:
        si = r.slot
        room = self.n_ctx - (pos0 + len(tokens)) - 2
        k = min(self.n_draft, room)
        if k <= 0:
            r.drafts = []
            return
        d = [self.ctx.mtp_forward(si, pos0, tokens, H)]
        pos = pos0 + len(tokens)
        while len(d) < k:
            d.append(self.ctx.mtp_forward(si, pos, [d[-1]], None))
            pos += 1
        r.drafts = d

    # ---------------------------------------------------------------- output
    def _emit_tokens(self, r: Request, toks: list[int]) -> tuple[int, str | None]:
        """Streams tokens until the request ends. Returns (tokens that are part of
        the output, finish reason or None). A stop token is not part of the output."""
        for i, t in enumerate(toks):
            if t in self.tok.stop_ids:
                return i, "stop"
            r.out.append(t)
            self.metrics["gen_tokens"] += 1
            delta = r.decoder.push(t)
            if r.stop and delta:
                text = r.decoder.text
                hit = min((text.find(x) for x in r.stop if x in text), default=-1)
                if hit >= 0:
                    keep = len(delta) - (len(text) - hit)
                    if keep > 0:
                        r.emit({"type": "token", "text": delta[:keep], "token": t})
                    return i + 1, "stop"
            if delta:
                r.emit({"type": "token", "text": delta, "token": t})
            if len(r.out) >= r.max_tokens:
                return i + 1, "length"
            if len(self.slots[r.slot].tokens) + len(toks) + 2 >= self.n_ctx:
                return i + 1, "length"
        return len(toks), None

    def _finish(self, r: Request, reason: str) -> None:
        if r.state == "done":
            return
        r.state = "done"
        r.finish_reason = reason
        r.t_done = time.time()
        if r in self.active:
            self.active.remove(r)
        if r.slot >= 0:
            s = self.slots[r.slot]
            s.req = None
            s.used = time.time()
            if reason in ("stop", "length"):
                self._snapshot(r.slot)
        n_gen = len(r.out)
        if n_gen > 1 and r.t_first:
            self.metrics["tpot_s"].append((r.t_done - r.t_first) / max(1, n_gen - 1))
        usage = {"prompt_tokens": len(r.prompt), "completion_tokens": n_gen, "total_tokens": len(r.prompt) + n_gen,
                 "prompt_tokens_details": {"cached_tokens": r.n_cached}}
        timing = {"ttft_s": round((r.t_first or r.t_done) - r.created, 4),
                  "total_s": round(r.t_done - r.created, 4),
                  "gen_tok_s": round((n_gen - 1) / (r.t_done - r.t_first), 2) if n_gen > 1 and r.t_first else None,
                  "prompt_tokens_encoded": len(r.prompt) - r.n_cached,
                  "spec_drafted": r.n_drafted, "spec_accepted": r.n_accepted}
        r.emit({"type": "done", "finish_reason": reason, "usage": usage, "timing": timing})
