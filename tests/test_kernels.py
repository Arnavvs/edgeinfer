"""Kernel unit tests: quantize -> dequantize error, and the matmul kernels
against a float64 reference computed from the dequantized weights.

    python tests/test_kernels.py
"""
import ctypes as C
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgeinfer._lib import DTYPES, lib  # noqa: E402

lib.ei_matmul.restype = C.c_int
lib.ei_matmul.argtypes = [C.c_int, C.c_void_p, C.c_int64, C.c_int64, C.c_void_p, C.c_int, C.c_void_p, C.c_int]


def pack(dt, w):
    rows, cols = w.shape
    out = np.empty(lib.ei_tensor_bytes(dt, rows, cols), np.uint8)
    assert lib.ei_quantize(dt, w.ctypes.data, rows, cols, out.ctypes.data, None, 4) == 0
    return out


def unpack(dt, buf, rows, cols):
    o = np.empty((rows, cols), np.float32)
    for r in range(rows):
        lib.ei_dequantize_row(dt, buf.ctypes.data, rows, cols, r, o[r].ctypes.data)
    return o


def main():
    rng = np.random.default_rng(0)
    ok = True
    for name in ["q8", "q6", "q4", "q4k", "bf16", "f32"]:
        dt = DTYPES[name]
        for rows, cols, m in [(64, 1024, 1), (70, 512, 3), (256, 2560, 8)]:
            w = (rng.standard_normal((rows, cols)) * 0.02).astype(np.float32)
            w[:, 5] *= 20  # an outlier column, as real layers have
            x = rng.standard_normal((m, cols)).astype(np.float32)
            buf = pack(dt, w)
            wd = unpack(dt, buf, rows, cols)
            rel_w = np.linalg.norm(wd - w) / np.linalg.norm(w)
            y = np.zeros((m, rows), np.float32)
            assert lib.ei_matmul(dt, buf.ctypes.data, rows, cols, x.ctypes.data, m, y.ctypes.data, 4) == 0
            ref = x.astype(np.float64) @ wd.astype(np.float64).T
            rel_y = np.linalg.norm(y - ref) / np.linalg.norm(ref)
            lim_w = {"q8": 0.01, "q6": 0.04, "q4": 0.13, "q4k": 0.11, "bf16": 0.005, "f32": 1e-7}[name]
            lim_y = {"q8": 0.01, "q6": 0.01, "q4": 0.01, "q4k": 0.01, "bf16": 1e-5, "f32": 1e-5}[name]  # vs dequantized weights
            good = rel_w < lim_w and rel_y < lim_y
            ok &= good
            print(f"{name:4s} {rows:4d}x{cols:<5d} m={m}  weight rel err {rel_w:.4f}  matmul rel err {rel_y:.2e}"
                  f"  {'ok' if good else 'BAD'}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
