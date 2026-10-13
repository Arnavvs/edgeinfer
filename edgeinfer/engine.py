"""Thin, allocation-free Python wrapper over the C engine."""
from __future__ import annotations

import ctypes as C
import json
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ._lib import FP, I32P, SEG_HIDDEN, SEG_VERIFY, Sampling, Seg, lib

__all__ = ["Model", "Context", "SamplingParams", "SEG_VERIFY", "SEG_HIDDEN"]


class Model:
    def __init__(self, path: str, threads: int = 4):
        err = C.create_string_buffer(1024)
        h = lib.ei_model_load(str(path).encode(), int(threads), err, 1024)
        if not h:
            raise RuntimeError(f"cannot load {path}: {err.value.decode()}")
        self.h = h
        self.path = str(path)
        self.info = json.loads(lib.ei_model_info(h).decode())
        self.n_vocab = lib.ei_n_vocab(h)
        self.n_embd = lib.ei_n_embd(h)
        self.has_mtp = bool(lib.ei_has_mtp(h))

    @property
    def threads(self) -> int:
        return lib.ei_get_threads(self.h)

    @threads.setter
    def threads(self, n: int) -> None:
        lib.ei_set_threads(self.h, int(n))

    def set_draft_vocab(self, ids: Sequence[int] | None) -> None:
        a = np.ascontiguousarray(ids if ids is not None else [], dtype=np.int32)
        rc = lib.ei_set_draft_vocab(self.h, a.ctypes.data_as(I32P), len(a))
        if rc != 0:
            raise RuntimeError(f"ei_set_draft_vocab failed ({rc})")

    def close(self) -> None:
        if self.h:
            lib.ei_model_free(self.h)
            self.h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


@dataclass
class SamplingParams:
    temperature: float = 0.7
    top_p: float = 0.8
    min_p: float = 0.0
    top_k: int = 20
    repeat_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int = 0

    def c(self) -> Sampling:
        return Sampling(self.temperature, self.top_p, self.min_p, self.top_k, self.repeat_penalty,
                        self.presence_penalty, self.frequency_penalty, self.seed)


class Context:
    """KV cache + recurrent state for n_slots independent sequences."""

    def __init__(self, model: Model, n_slots: int = 1, n_ctx: int = 4096, max_batch: int = 256, max_draft: int = 3):
        h = lib.ei_ctx_new(model.h, n_slots, n_ctx, max_batch, max_draft)
        if not h:
            raise MemoryError("ei_ctx_new failed")
        self.h = h
        self.model = model
        self.n_slots, self.n_ctx, self.max_batch, self.max_draft = n_slots, n_ctx, max_batch, max_draft
        self.V = model.n_vocab
        self.D = model.n_embd

    # ---- forward ------------------------------------------------------------
    def forward(self, segs: Iterable[tuple]) -> int:
        """segs: (slot, tokens, n_logits, flags). Returns logit rows produced."""
        segs = list(segs)
        arr = (Seg * len(segs))()
        keep = []
        for i, (slot, toks, nl, fl) in enumerate(segs):
            a = np.ascontiguousarray(toks, dtype=np.int32)
            keep.append(a)
            arr[i].slot = slot
            arr[i].n_tokens = len(a)
            arr[i].tokens = a.ctypes.data_as(I32P)
            arr[i].pos0 = lib.ei_slot_len(self.h, slot)
            arr[i].n_logits = nl
            arr[i].flags = fl
        r = lib.ei_forward(self.h, arr, len(segs))
        if r < 0:
            raise RuntimeError(f"ei_forward failed with code {r}")
        return r

    def logits_ptr(self, row: int):
        p = lib.ei_logits(self.h, row)
        if not p:
            raise IndexError(row)
        return p

    def logits(self, row: int) -> np.ndarray:
        """View of a logits row; valid until the next forward."""
        return np.ctypeslib.as_array(self.logits_ptr(row), shape=(self.V,))

    def hidden(self, row: int) -> np.ndarray:
        return np.ctypeslib.as_array(lib.ei_hidden(self.h, row), shape=(self.D,))

    # ---- slots --------------------------------------------------------------
    def slot_len(self, slot: int) -> int:
        return lib.ei_slot_len(self.h, slot)

    def reset(self, slot: int) -> None:
        lib.ei_slot_reset(self.h, slot)

    def commit(self, slot: int, n_keep: int) -> int:
        r = lib.ei_commit(self.h, slot, n_keep)
        if r < 0:
            raise RuntimeError(f"ei_commit failed ({r})")
        return r

    # ---- snapshots ----------------------------------------------------------
    def snap_save(self, slot: int, with_kv: bool = False) -> int:
        r = lib.ei_snap_save(self.h, slot, int(with_kv))
        if r < 0:
            raise RuntimeError(f"ei_snap_save failed ({r})")
        return r

    def snap_restore(self, slot: int, snap: int) -> int:
        r = lib.ei_snap_restore(self.h, slot, snap)
        if r < 0:
            raise RuntimeError(f"ei_snap_restore failed ({r})")
        return r

    def snap_free(self, snap: int) -> None:
        lib.ei_snap_free(self.h, snap)

    def snap_len(self, snap: int) -> int:
        return lib.ei_snap_len(self.h, snap)

    def snap_bytes(self, snap: int) -> int:
        return lib.ei_snap_bytes(self.h, snap)

    def snap_write(self, snap: int, path: str) -> None:
        if lib.ei_snap_write(self.h, snap, str(path).encode()) != 0:
            raise RuntimeError("ei_snap_write failed")

    def snap_read(self, path: str) -> int:
        r = lib.ei_snap_read(self.h, str(path).encode())
        if r < 0:
            raise RuntimeError(f"ei_snap_read failed ({r})")
        return r

    # ---- sampling -----------------------------------------------------------
    def sample(self, row: int, sp: Sampling, history: np.ndarray, rng: C.c_uint64) -> tuple[int, float]:
        prob = C.c_float(0)
        t = lib.ei_sample(self.h, self.logits_ptr(row), C.byref(sp), history.ctypes.data_as(I32P), len(history),
                          C.byref(rng), C.byref(prob))
        return int(t), float(prob.value)

    def spec_accept(self, row: int, sp: Sampling, history: np.ndarray, draft: int, rng: C.c_uint64) -> tuple[int, bool]:
        acc = C.c_int(0)
        t = lib.ei_spec_accept(self.h, self.logits_ptr(row), C.byref(sp), history.ctypes.data_as(I32P), len(history),
                               int(draft), C.byref(rng), C.byref(acc))
        return int(t), bool(acc.value)

    # ---- MTP ----------------------------------------------------------------
    def mtp_len(self, slot: int) -> int:
        return lib.ei_mtp_len(self.h, slot)

    def mtp_truncate(self, slot: int, length: int) -> None:
        lib.ei_mtp_truncate(self.h, slot, length)

    def mtp_forward(self, slot: int, pos0: int, tokens: Sequence[int], hidden: np.ndarray | None) -> int:
        """hidden: [len(tokens)][n_embd] float32, or None to chain on the MTP head's own output."""
        a = np.ascontiguousarray(tokens, dtype=np.int32)
        hp = None
        if hidden is not None:
            hidden = np.ascontiguousarray(hidden, dtype=np.float32).reshape(len(a), self.D)
            hp = hidden.ctypes.data_as(FP)
        r = lib.ei_mtp_forward(self.h, slot, pos0, len(a), a.ctypes.data_as(I32P), hp)
        if r < 0:
            raise RuntimeError(f"ei_mtp_forward failed ({r})")
        return int(r)

    def hidden_tokens(self, n: int) -> np.ndarray:
        """Final-norm hidden states of the first n tokens of the last batch (EI_SEG_HIDDEN segments)."""
        return np.ctypeslib.as_array(lib.ei_hidden_tok(self.h, 0), shape=(n, self.D))

    # ---- introspection ------------------------------------------------------
    def stats(self) -> dict:
        return json.loads(lib.ei_ctx_stats(self.h).decode())

    def reset_stats(self) -> None:
        lib.ei_ctx_reset_stats(self.h)

    def nbytes(self) -> int:
        return lib.ei_ctx_bytes(self.h)

    def close(self) -> None:
        if self.h:
            lib.ei_ctx_free(self.h)
            self.h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
