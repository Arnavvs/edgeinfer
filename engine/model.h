// Model weights, context state and the forward pass for Qwen3.5-style hybrid
// models: every `full_attention_interval`-th layer is gated softmax attention
// (GQA, QK-norm, partial RoPE, sigmoid output gate); the others are Gated
// DeltaNet linear attention (short causal conv, L2-normed q/k, a per-head
// decayed delta-rule state, gated RMSNorm). Plain dense transformers
// (all layers attention, no output gate) fit the same code path.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common.h"
#include "ei.h"
#include "quant.h"
#include "threadpool.h"

namespace ei {

struct Config {
    std::string arch;
    int D = 0, L = 0, F = 0, Hq = 0, Hkv = 0, Dh = 0;
    int Hk = 0, Hv = 0, dk = 0, dv = 0, convk = 4;
    int V = 0, rope_dim = 0, mtp_layers = 0, attn_gate = 1;
    float rope_theta = 1e7f, eps = 1e-6f;
    std::vector<int> layer_types;  // 0 = linear (DeltaNet), 1 = full attention
    int C() const { return 2 * Hk * dk + Hv * dv; }
    int qkv_rows() const { return Hq * Dh * (attn_gate ? 2 : 1) + 2 * Hkv * Dh; }
};

struct LinW {
    Tensor qkvz, ab, out;
    const float *conv = nullptr, *dt_bias = nullptr, *a = nullptr, *norm = nullptr;
};

struct AttW {
    Tensor qkv, o;
    const float *qn = nullptr, *kn = nullptr;
};

struct LayerW {
    int type = 0, idx = 0;  // idx: index among layers of this type
    const float *attn_norm = nullptr, *ffn_norm = nullptr;
    LinW lin;
    AttW att;
    Tensor gate_up, down;
};

struct MtpW {
    bool present = false;
    Tensor fc;
    const float *norm_emb = nullptr, *norm_hid = nullptr, *out_norm = nullptr;
    LayerW layer;
};

}  // namespace ei

struct ei_model {
    ei::Config cfg;
    std::vector<ei::LayerW> layers;
    ei::Tensor embed, lm_head;
    const float* out_norm = nullptr;
    ei::MtpW mtp;
    int n_lin = 0, n_att = 0;
    void* map = nullptr;
    size_t map_size = 0;
    std::unique_ptr<ei::ThreadPool> pool;
    std::string info;
    std::vector<float> owned_f32;  // kept for derived small tensors
    // reduced draft vocabulary for the MTP head (FR-Spec)
    ei::Tensor draft_head;
    ei::Buf<uint8_t> draft_data;
    std::vector<int32_t> draft_ids;
};

namespace ei {

constexpr int MAX_VERIFY = 16;

struct Slot {
    int len = 0;
    int cur = 0;             // state buffer set holding the current state
    int verify_n = 0;        // >0 while a VERIFY segment awaits ei_commit
    int verify_pos0 = 0;
    int verify_buf[MAX_VERIFY];
    int mtp_len = 0;
};

struct Snap {
    int slot = -1, len = 0, mtp_len = 0;
    bool with_kv = false;
    Buf<float> rec, conv;
    Buf<uint16_t> k, v, mk, mv;
    int64_t bytes() const {
        return int64_t(rec.bytes() + conv.bytes() + k.bytes() + v.bytes() + mk.bytes() + mv.bytes());
    }
};

struct Tok {
    int slot, pos, seg;
};

enum Op { OP_EMBED, OP_NORM, OP_QACT, OP_MATMUL, OP_CONV, OP_DELTA, OP_ATTN, OP_MISC, OP_LMHEAD, OP_MTP, OP_N };
inline const char* op_names[OP_N] = {"embed", "norm", "quant_act", "matmul", "conv", "deltanet", "attention", "misc", "lm_head", "mtp"};

}  // namespace ei

struct ei_ctx {
    ei_model* m = nullptr;
    int n_slots = 0, n_ctx = 0, max_batch = 0, max_draft = 0, nbuf = 0, max_logits = 0;
    std::vector<ei::Slot> slots;
    // recurrent state: index slot*nbuf+buf -> [n_lin][Hv][dk][dv]
    std::vector<ei::Buf<float>> rec;
    // conv windows: index slot*nbuf+buf -> [n_lin][convk-1][C]
    std::vector<ei::Buf<float>> conv;
    // attention KV: per slot [n_att][n_ctx][Hkv*Dh] fp16
    std::vector<ei::Buf<uint16_t>> kc, vc;
    // MTP attention KV: per slot [n_ctx][Hkv*Dh]
    std::vector<ei::Buf<uint16_t>> mk, mv;

    // work buffers
    ei::Buf<float> x, xn, proj, ab, gb, o, y, logits, hid, hid_all, rc, rs, att_part, mtp_hid, cat;
    ei::QAct qa;
    std::vector<ei::Tok> toks;
    std::vector<int> seg_t0, seg_t1, seg_slot, seg_flags;
    int n_rows = 0;
    std::vector<int> logit_tok;  // token index of each logit row

    std::map<int, std::unique_ptr<ei::Snap>> snaps;
    int next_snap = 0;

    double op_t[ei::OP_N] = {};
    int64_t op_n[ei::OP_N] = {};
    int64_t n_forward = 0, n_tokens = 0;
    std::string stats_json;
    ei::Buf<float> samp;  // sampling scratch [V]
    bool calib = false;  // accumulate imatrix statistics in mm()
    std::map<std::string, std::vector<double>> calib_sum;
    std::map<std::string, int64_t> calib_n;
    std::vector<std::pair<float, int>> cand;

    size_t rec_floats() const { const auto& c = m->cfg; return size_t(m->n_lin) * c.Hv * c.dk * c.dv; }
    size_t conv_floats() const { const auto& c = m->cfg; return size_t(m->n_lin) * (c.convk - 1) * c.C(); }
    size_t kvd() const { return size_t(m->cfg.Hkv) * m->cfg.Dh; }
    float* rec_ptr(int slot, int buf, int li) {
        const auto& c = m->cfg;
        return rec[size_t(slot) * nbuf + buf].p + size_t(li) * c.Hv * c.dk * c.dv;
    }
    float* conv_ptr(int slot, int buf, int li) {
        const auto& c = m->cfg;
        return conv[size_t(slot) * nbuf + buf].p + size_t(li) * (c.convk - 1) * c.C();
    }
    uint16_t* k_ptr(int slot, int ai) { return kc[slot].p + size_t(ai) * n_ctx * kvd(); }
    uint16_t* v_ptr(int slot, int ai) { return vc[slot].p + size_t(ai) * n_ctx * kvd(); }
};
