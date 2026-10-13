"""Convert a Hugging Face Qwen3.5 checkpoint (safetensors) to an edgeinfer .eim file.

    python -m edgeinfer.convert HF_DIR OUT.eim --qtype q4 --head q8 [--imatrix imp.npz]

No torch needed: safetensors are read with numpy, bf16 is widened bit-exactly,
and quantization runs in the engine's own C code so the packing lives in one
place. Fusions done here so the engine runs fewer, larger matmuls:
  * DeltaNet in_proj_qkv + in_proj_z  -> ssm_qkvz,   in_proj_b + in_proj_a -> ssm_ab
  * attention q_proj + k_proj + v_proj -> attn_qkv
  * mlp gate_proj + up_proj           -> ffn_gate_up
Qwen3.5's zero-centred RMSNorms (y = x_hat * (1 + w)) get the +1 folded in, so
the engine treats every norm weight as a plain multiplier.
"""
from __future__ import annotations

import argparse
import ctypes as C
import fnmatch
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

from ._lib import DTYPES, lib

ALIGN = 64


class SafeTensors:
    """Lazy reader over one or more .safetensors files."""

    def __init__(self, root: Path):
        idx = root / "model.safetensors.index.json"
        files = sorted({v for v in json.loads(idx.read_text())["weight_map"].values()}) if idx.exists() \
            else [p.name for p in root.glob("*.safetensors")]
        self.loc = {}
        for f in files:
            path = root / f
            with open(path, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            mm = np.memmap(path, dtype=np.uint8, mode="r")
            for k, v in hdr.items():
                if k == "__metadata__":
                    continue
                a, b = v["data_offsets"]
                self.loc[k] = (mm, 8 + n + a, 8 + n + b, v["dtype"], v["shape"])

    def has(self, k): return k in self.loc

    def get(self, k) -> np.ndarray:
        mm, a, b, dt, shape = self.loc[k]
        raw = mm[a:b]
        if dt == "BF16":
            x = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif dt == "F16":
            x = raw.view(np.float16).astype(np.float32)
        elif dt == "F32":
            x = raw.view(np.float32).astype(np.float32)
        else:
            raise ValueError(f"{k}: unsupported dtype {dt}")
        return x.reshape(shape)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hf_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--qtype", default="q4", choices=list(DTYPES), help="matmul weights")
    ap.add_argument("--head", default="q8", choices=list(DTYPES), help="embedding / LM head")
    ap.add_argument("--mtp-qtype", default=None, choices=list(DTYPES), help="MTP head weights (default: --qtype)")
    ap.add_argument("--no-mtp", action="store_true")
    ap.add_argument("--imatrix", type=Path, help="npz of per-input-channel importance, keyed by output tensor name")
    ap.add_argument("--profile", choices=["q4m", "q4s"],
                    help="q4m: q4k + Q8 for ffn_down in the sensitive layers (llama.cpp's use_more_bits rule), "
                         "ssm_out and the LM head; q4s: the same with a q4k LM head (faster decode)")
    ap.add_argument("--override", action="append", default=[], metavar="GLOB=DTYPE",
                    help="per-tensor dtype, e.g. 'blk.*.ffn_down=q8' (repeatable, last match wins)")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args(argv)
    if a.profile:
        n = json.loads((a.hf_dir / "config.json").read_text())
        n = n.get("text_config", n)["num_hidden_layers"]
        more = [i for i in range(n) if i < n // 8 or i >= 7 * n // 8 or (i - n // 8) % 3 == 2]
        a.qtype = "q4k"
        a.head = "q8" if a.profile == "q4m" else "q4k"
        a.override = [f"blk.{i}.ffn_down=q8" for i in more] + ["blk.*.ssm_out=q8"] + a.override

    cfg_all = json.loads((a.hf_dir / "config.json").read_text())
    cfg = cfg_all.get("text_config", cfg_all)
    st = SafeTensors(a.hf_dir)
    pre = "model.language_model." if st.has("model.language_model.embed_tokens.weight") else "model."
    imat = dict(np.load(a.imatrix)) if a.imatrix else {}
    rope = cfg.get("rope_parameters") or {}
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    partial = rope.get("partial_rotary_factor", cfg.get("partial_rotary_factor", 1.0))
    L = cfg["num_hidden_layers"]
    interval = cfg.get("full_attention_interval", 1)
    layer_types = cfg.get("layer_types") or ["full_attention" if (i + 1) % interval == 0 else "linear_attention"
                                             for i in range(L)]
    types = [1 if t == "full_attention" else 0 for t in layer_types]
    mtp_layers = 0 if a.no_mtp or not st.has("mtp.fc.weight") else cfg.get("mtp_num_hidden_layers", 0)
    qt, ht, mt = DTYPES[a.qtype], DTYPES[a.head], DTYPES[a.mtp_qtype or a.qtype]

    nums = {
        "hidden_size": cfg["hidden_size"], "num_hidden_layers": L, "intermediate_size": cfg["intermediate_size"],
        "num_attention_heads": cfg["num_attention_heads"], "num_key_value_heads": cfg["num_key_value_heads"],
        "head_dim": head_dim, "vocab_size": cfg["vocab_size"],
        "linear_num_key_heads": cfg.get("linear_num_key_heads", 0),
        "linear_num_value_heads": cfg.get("linear_num_value_heads", 0),
        "linear_key_head_dim": cfg.get("linear_key_head_dim", 0),
        "linear_value_head_dim": cfg.get("linear_value_head_dim", 0),
        "linear_conv_kernel_dim": cfg.get("linear_conv_kernel_dim", 4),
        "rope_theta": rope.get("rope_theta", cfg.get("rope_theta", 1e6)),
        "rope_dim": int(head_dim * partial), "rms_norm_eps": cfg["rms_norm_eps"],
        "attn_output_gate": int(cfg.get("attn_output_gate", False)),
        "mtp_num_hidden_layers": mtp_layers,
    }

    # ---- plan: (name, dtype, rows, cols, producer) ----
    plan = []

    def mat(name, dtype, keys, transform=None):
        def produce():
            parts = [st.get(k) for k in keys]
            x = np.concatenate(parts, 0) if len(parts) > 1 else parts[0]
            if transform:
                x = transform(x)
            return np.ascontiguousarray(x.reshape(x.shape[0], -1), dtype=np.float32)
        shapes = [st.loc[k][4] for k in keys]
        rows = sum(s[0] for s in shapes)
        cols = int(np.prod(shapes[0][1:])) if len(shapes[0]) > 1 else 1
        if transform is not None and len(shapes[0]) == 1:  # 1-D vectors stored as [1][n]
            rows, cols = 1, shapes[0][0]
        plan.append((name, dtype, rows, cols, produce))

    def vec(name, key, plus_one=False, fn=None):
        def t(x):
            x = x.astype(np.float32)
            if plus_one:
                x = x + 1.0
            if fn:
                x = fn(x)
            return x.reshape(1, -1)
        mat(name, DTYPES["f32"], [key], t)

    def layer(p_hf, p_out, typ, dt):
        vec(p_out + "attn_norm", p_hf + "input_layernorm.weight", plus_one=True)
        vec(p_out + "ffn_norm", p_hf + "post_attention_layernorm.weight", plus_one=True)
        mat(p_out + "ffn_gate_up", dt, [p_hf + "mlp.gate_proj.weight", p_hf + "mlp.up_proj.weight"])
        mat(p_out + "ffn_down", dt, [p_hf + "mlp.down_proj.weight"])
        if typ == 0:
            la = p_hf + "linear_attn."
            mat(p_out + "ssm_qkvz", dt, [la + "in_proj_qkv.weight", la + "in_proj_z.weight"])
            mat(p_out + "ssm_ab", DTYPES["f32"], [la + "in_proj_b.weight", la + "in_proj_a.weight"])
            mat(p_out + "ssm_out", dt, [la + "out_proj.weight"])
            mat(p_out + "ssm_conv", DTYPES["f32"], [la + "conv1d.weight"],
                lambda x: x.reshape(x.shape[0], -1).astype(np.float32))
            vec(p_out + "ssm_dt_bias", la + "dt_bias")
            vec(p_out + "ssm_a", la + "A_log", fn=lambda x: -np.exp(x))
            vec(p_out + "ssm_norm", la + "norm.weight")
        else:
            sa = p_hf + "self_attn."
            mat(p_out + "attn_qkv", dt, [sa + "q_proj.weight", sa + "k_proj.weight", sa + "v_proj.weight"])
            mat(p_out + "attn_o", dt, [sa + "o_proj.weight"])
            vec(p_out + "attn_q_norm", sa + "q_norm.weight", plus_one=True)
            vec(p_out + "attn_k_norm", sa + "k_norm.weight", plus_one=True)

    mat("tok_embd", ht, [pre + "embed_tokens.weight"])
    if not cfg_all.get("tie_word_embeddings", cfg.get("tie_word_embeddings", True)) and st.has("lm_head.weight"):
        mat("output", ht, ["lm_head.weight"])
    vec("output_norm", pre + "norm.weight", plus_one=True)
    for i in range(L):
        layer(f"{pre}layers.{i}.", f"blk.{i}.", types[i], qt)
    if mtp_layers:
        mat("mtp.fc", mt, ["mtp.fc.weight"])
        vec("mtp.norm_emb", "mtp.pre_fc_norm_embedding.weight", plus_one=True)
        vec("mtp.norm_hidden", "mtp.pre_fc_norm_hidden.weight", plus_one=True)
        vec("mtp.output_norm", "mtp.norm.weight", plus_one=True)
        layer("mtp.layers.0.", "mtp.blk.", 1, mt)

    # conv weights are [C][1][4] -> rows=C, cols=4
    overrides = [(g, DTYPES[d]) for g, d in (o.rsplit("=", 1) for o in a.override)]
    fixed = []
    for name, dt, rows, cols, prod in plan:
        if dt not in (DTYPES["f32"],):
            for glob, odt in overrides:
                if fnmatch.fnmatchcase(name, glob):
                    dt = odt
        if name.endswith("ssm_conv"):
            shp = st.loc[f"{pre}layers.{int(name.split('.')[1])}.linear_attn.conv1d.weight"][4]
            rows, cols = shp[0], shp[-1]
        fixed.append((name, dt, rows, cols, prod))
    plan = fixed

    # ---- header ----
    offs, off = [], 0
    for name, dt, rows, cols, _ in plan:
        nb = lib.ei_tensor_bytes(dt, rows, cols)
        if nb < 0:
            raise SystemExit(f"{name}: {rows}x{cols} not representable as {dt}")
        offs.append((off, nb))
        off = (off + nb + ALIGN - 1) // ALIGN * ALIGN
    names = {v: k for k, v in DTYPES.items()}
    lines = ["EIMODEL1", "arch qwen3_5"]
    lines += [f"cfg {k} {v}" for k, v in nums.items()]
    lines.append("cfg_arr layer_types %d %s" % (len(types), " ".join(map(str, types))))
    for (name, dt, rows, cols, _), (o, nb) in zip(plan, offs):
        lines.append(f"tensor {name} {names[dt]} {rows} {cols} {o} {nb}")
    lines.append("end")
    header = ("\n".join(lines) + "\n").encode()
    pad = (-len(header)) % 4096

    t0 = time.time()
    tmp = a.out.with_suffix(".part")
    with open(tmp, "wb") as f:
        f.write(header + b"\0" * pad)
        base = f.tell()
        for (name, dt, rows, cols, prod), (o, nb) in zip(plan, offs):
            x = prod()
            assert x.shape == (rows, cols), (name, x.shape, rows, cols)
            out = np.empty(nb, dtype=np.uint8)
            imp = imat.get(name)
            imp_p = None
            if imp is not None:
                imp = np.ascontiguousarray(imp, dtype=np.float32)
                assert imp.shape == (cols,), (name, imp.shape)
                imp_p = imp.ctypes.data_as(C.c_void_p)
            rc = lib.ei_quantize(dt, x.ctypes.data_as(C.c_void_p), rows, cols, out.ctypes.data_as(C.c_void_p),
                                 imp_p, a.threads)
            if rc != 0:
                raise SystemExit(f"quantize failed for {name}")
            f.seek(base + o)
            f.write(out.tobytes())
            if name.endswith("_norm") and "ssm" not in name:
                assert 0.0 < float(x.mean()) < 50.0, f"{name}: mean {x.mean():.3f} - norm offset looks wrong"
            print(f"  {name:28s} {names[dt]:4s} {rows:>7}x{cols:<6} {nb / 2**20:8.2f} MiB", flush=True)
        end = base + (offs[-1][0] + offs[-1][1])
        f.truncate(end)
    tmp.replace(a.out)
    print(f"wrote {a.out} ({a.out.stat().st_size / 2**20:.1f} MiB) in {time.time() - t0:.1f}s; "
          f"mtp={'yes' if mtp_layers else 'no'} qtype={a.qtype} head={a.head}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
