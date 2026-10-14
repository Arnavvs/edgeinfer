"""ctypes bindings for libedgeinfer.so (see engine/ei.h)."""
from __future__ import annotations

import ctypes as C
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent

F32, F16, BF16, Q8, Q4, Q4K, Q6 = 0, 1, 2, 3, 4, 5, 6
DTYPES = {"f32": F32, "f16": F16, "bf16": BF16, "q8": Q8, "q4": Q4, "q4k": Q4K, "q6": Q6}
SEG_VERIFY, SEG_HIDDEN = 1, 2


def _find() -> str:
    cands = [os.environ.get("EDGEINFER_LIB"), _HERE / "libedgeinfer.so",
             _HERE.parent / "build" / "libedgeinfer.so"]
    for p in cands:
        if p and Path(p).exists():
            return str(p)
    raise OSError("libedgeinfer.so not found - build it: cmake -B build && cmake --build build -j")


lib = C.CDLL(_find())


class Seg(C.Structure):
    _fields_ = [("slot", C.c_int32), ("n_tokens", C.c_int32), ("tokens", C.POINTER(C.c_int32)),
                ("pos0", C.c_int32), ("n_logits", C.c_int32), ("flags", C.c_int32)]


class Sampling(C.Structure):
    _fields_ = [("temperature", C.c_float), ("top_p", C.c_float), ("min_p", C.c_float),
                ("top_k", C.c_int32), ("repeat_penalty", C.c_float), ("presence_penalty", C.c_float),
                ("frequency_penalty", C.c_float), ("seed", C.c_uint64)]


def _sig(name, res, *args):
    f = getattr(lib, name)
    f.restype = res
    f.argtypes = list(args)
    return f


P = C.c_void_p
I32P = C.POINTER(C.c_int32)
FP = C.POINTER(C.c_float)

_sig("ei_tensor_bytes", C.c_int64, C.c_int, C.c_int64, C.c_int64)
_sig("ei_quantize", C.c_int, C.c_int, P, C.c_int64, C.c_int64, P, P, C.c_int)
_sig("ei_dequantize_row", C.c_int, C.c_int, P, C.c_int64, C.c_int64, C.c_int64, P)
_sig("ei_model_load", P, C.c_char_p, C.c_int, C.c_char_p, C.c_int)
_sig("ei_model_free", None, P)
_sig("ei_model_info", C.c_char_p, P)
_sig("ei_n_vocab", C.c_int, P)
_sig("ei_n_embd", C.c_int, P)
_sig("ei_has_mtp", C.c_int, P)
_sig("ei_get_threads", C.c_int, P)
_sig("ei_set_threads", None, P, C.c_int)
_sig("ei_set_draft_vocab", C.c_int, P, I32P, C.c_int)
_sig("ei_ctx_new", P, P, C.c_int, C.c_int, C.c_int, C.c_int)
_sig("ei_ctx_free", None, P)
_sig("ei_forward", C.c_int, P, C.POINTER(Seg), C.c_int)
_sig("ei_logits", FP, P, C.c_int)
_sig("ei_hidden", FP, P, C.c_int)
_sig("ei_hidden_tok", FP, P, C.c_int)
_sig("ei_slot_len", C.c_int, P, C.c_int)
_sig("ei_slot_reset", None, P, C.c_int)
_sig("ei_commit", C.c_int, P, C.c_int, C.c_int)
_sig("ei_snap_save", C.c_int, P, C.c_int, C.c_int)
_sig("ei_snap_restore", C.c_int, P, C.c_int, C.c_int)
_sig("ei_snap_len", C.c_int, P, C.c_int)
_sig("ei_snap_bytes", C.c_int64, P, C.c_int)
_sig("ei_snap_free", None, P, C.c_int)
_sig("ei_snap_write", C.c_int, P, C.c_int, C.c_char_p)
_sig("ei_snap_read", C.c_int, P, C.c_char_p)
_sig("ei_sample", C.c_int32, P, FP, C.POINTER(Sampling), I32P, C.c_int, C.POINTER(C.c_uint64), FP)
_sig("ei_spec_accept", C.c_int32, P, FP, C.POINTER(Sampling), I32P, C.c_int, C.c_int32,
     C.POINTER(C.c_uint64), C.POINTER(C.c_int))
_sig("ei_argmax", C.c_int32, FP, C.c_int)
_sig("ei_mtp_truncate", None, P, C.c_int, C.c_int)
_sig("ei_mtp_len", C.c_int, P, C.c_int)
_sig("ei_mtp_forward", C.c_int32, P, C.c_int, C.c_int, C.c_int, I32P, FP)
_sig("ei_ctx_stats", C.c_char_p, P)
_sig("ei_ctx_reset_stats", None, P)
_sig("ei_ctx_bytes", C.c_int64, P)
_sig("ei_calib", None, P, C.c_int)
_sig("ei_calib_dump", C.c_int, P, C.c_char_p)
_sig("ei_matmul", C.c_int, C.c_int, P, C.c_int64, C.c_int64, P, C.c_int, P, C.c_int)
