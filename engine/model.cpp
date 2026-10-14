// Loader, context and forward pass. See model.h for the architecture.
#include "model.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <fstream>
#include <sstream>

using namespace ei;

// ============================================================== file format
//
//   "EIMODEL1\n" then text lines:
//     arch <name>
//     cfg <key> <number>
//     cfg_arr <key> <n> <v0> <v1> ...
//     tensor <name> <dtype> <rows> <cols> <offset> <nbytes>
//     end
//   padding to a 4096-byte boundary, then the tensor data (offsets relative
//   to that boundary, each tensor 64-byte aligned).

namespace {

struct TRec {
    int dtype;
    int64_t rows, cols, offset, nbytes;
};

struct Loader {
    const uint8_t* base = nullptr;
    size_t size = 0, data_off = 0;
    std::map<std::string, TRec> t;
    std::map<std::string, double> cfg;
    std::map<std::string, std::vector<double>> arr;
    std::string arch;

    void parse() {
        const char* p = reinterpret_cast<const char*>(base);
        if (size < 16 || std::memcmp(p, "EIMODEL1\n", 9) != 0) throw std::runtime_error("not an edgeinfer model file");
        size_t pos = 9;
        for (;;) {
            size_t e = pos;
            while (e < size && p[e] != '\n') ++e;
            if (e >= size) throw std::runtime_error("truncated header");
            std::string line(p + pos, e - pos);
            pos = e + 1;
            std::istringstream ss(line);
            std::string kind;
            ss >> kind;
            if (kind == "end") break;
            if (kind == "arch") ss >> arch;
            else if (kind == "cfg") { std::string k; double v; ss >> k >> v; cfg[k] = v; }
            else if (kind == "cfg_arr") {
                std::string k; int n; ss >> k >> n;
                std::vector<double> v(n);
                for (auto& x : v) ss >> x;
                arr[k] = v;
            } else if (kind == "tensor") {
                std::string name, dt; TRec r;
                ss >> name >> dt >> r.rows >> r.cols >> r.offset >> r.nbytes;
                r.dtype = dtype_from_name(dt.c_str());
                if (r.dtype < 0) throw std::runtime_error("bad dtype " + dt);
                t[name] = r;
            }
        }
        data_off = (pos + 4095) / 4096 * 4096;
    }

    bool has(const std::string& n) const { return t.count(n) != 0; }

    Tensor get(const std::string& n) const {
        auto it = t.find(n);
        if (it == t.end()) throw std::runtime_error("missing tensor " + n);
        const TRec& r = it->second;
        Tensor x;
        x.name = n;
        x.dtype = r.dtype;
        x.rows = r.rows;
        x.cols = r.cols;
        x.data = base + data_off + r.offset;
        x.nbytes = r.nbytes;
        if (data_off + r.offset + r.nbytes > size) throw std::runtime_error("tensor out of file: " + n);
        if (tensor_bytes(r.dtype, r.rows, r.cols) != r.nbytes) throw std::runtime_error("size mismatch: " + n);
        return x;
    }

    const float* f32(const std::string& n, int64_t expect) const {
        Tensor x = get(n);
        if (x.dtype != F32) throw std::runtime_error(n + " must be f32");
        if (expect >= 0 && x.rows * x.cols != expect) throw std::runtime_error("bad size for " + n);
        return reinterpret_cast<const float*>(x.data);
    }

    int ci(const char* k, int def = -1) const {
        auto it = cfg.find(k);
        if (it == cfg.end()) {
            if (def == -1) throw std::runtime_error(std::string("missing cfg ") + k);
            return def;
        }
        return int(it->second);
    }
};

void check_mat(const Tensor& t, int64_t rows, int64_t cols, const std::string& n) {
    if (t.rows != rows || t.cols != cols)
        throw std::runtime_error(n + ": expected [" + std::to_string(rows) + "," + std::to_string(cols) + "] got [" +
                                 std::to_string(t.rows) + "," + std::to_string(t.cols) + "]");
}

void load_layer(const Loader& ld, const Config& c, const std::string& pre, LayerW& L) {
    L.attn_norm = ld.f32(pre + "attn_norm", c.D);
    L.ffn_norm = ld.f32(pre + "ffn_norm", c.D);
    L.gate_up = ld.get(pre + "ffn_gate_up");
    check_mat(L.gate_up, 2 * c.F, c.D, pre + "ffn_gate_up");
    L.down = ld.get(pre + "ffn_down");
    check_mat(L.down, c.D, c.F, pre + "ffn_down");
    if (L.type == 0) {
        L.lin.qkvz = ld.get(pre + "ssm_qkvz");
        check_mat(L.lin.qkvz, c.C() + c.Hv * c.dv, c.D, pre + "ssm_qkvz");
        L.lin.ab = ld.get(pre + "ssm_ab");
        check_mat(L.lin.ab, 2 * c.Hv, c.D, pre + "ssm_ab");
        L.lin.out = ld.get(pre + "ssm_out");
        check_mat(L.lin.out, c.D, c.Hv * c.dv, pre + "ssm_out");
        L.lin.conv = ld.f32(pre + "ssm_conv", int64_t(c.C()) * c.convk);
        L.lin.dt_bias = ld.f32(pre + "ssm_dt_bias", c.Hv);
        L.lin.a = ld.f32(pre + "ssm_a", c.Hv);
        L.lin.norm = ld.f32(pre + "ssm_norm", c.dv);
    } else {
        L.att.qkv = ld.get(pre + "attn_qkv");
        check_mat(L.att.qkv, c.qkv_rows(), c.D, pre + "attn_qkv");
        L.att.o = ld.get(pre + "attn_o");
        check_mat(L.att.o, c.D, int64_t(c.Hq) * c.Dh, pre + "attn_o");
        L.att.qn = ld.f32(pre + "attn_q_norm", c.Dh);
        L.att.kn = ld.f32(pre + "attn_k_norm", c.Dh);
    }
}

}  // namespace

// ============================================================== context


namespace {

struct Timer {
    ei_ctx* c;
    int op;
    double t0;
    Timer(ei_ctx* c, int op) : c(c), op(op), t0(now_s()) {}
    ~Timer() { c->op_t[op] += now_s() - t0; c->op_n[op]++; }
};

inline void rmsnorm(const float* x, const float* w, float* out, int n, float eps) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) { __m256 v = _mm256_loadu_ps(x + i); acc = _mm256_fmadd_ps(v, v, acc); }
    float ss = hsum8(acc);
    for (; i < n; ++i) ss += x[i] * x[i];
    const float r = 1.0f / std::sqrt(ss / n + eps);
    const __m256 rv = _mm256_set1_ps(r);
    i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + i), rv), _mm256_loadu_ps(w + i)));
    for (; i < n; ++i) out[i] = x[i] * r * w[i];
}

inline float dot(const float* a, const float* b, int n) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
    }
    float s = hsum8(_mm256_add_ps(acc0, acc1));
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

inline void to_fp16(const float* x, uint16_t* out, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + i), _mm256_cvtps_ph(_mm256_loadu_ps(x + i), _MM_FROUND_TO_NEAREST_INT));
    for (; i < n; ++i) out[i] = f32_to_fp16(x[i]);
}

// rows [0,m) of x (stride ld) normalised into out (stride ldo)
void norm_rows(ei_ctx* c, const float* x, int ld, const float* w, float* out, int ldo, int m, int n) {
    Timer tm(c, OP_NORM);
    const float eps = c->m->cfg.eps;
    if (m < 8) {
        for (int i = 0; i < m; ++i) rmsnorm(x + size_t(i) * ld, w, out + size_t(i) * ldo, n, eps);
        return;
    }
    parallel_for(*c->m->pool, m, 4, [&](int a, int b, int) {
        for (int i = a; i < b; ++i) rmsnorm(x + size_t(i) * ld, w, out + size_t(i) * ldo, n, eps);
    });
}

void qact(ei_ctx* c, const float* x, int ld, int m, int k) {
    Timer tm(c, OP_QACT);
    c->qa.m = m;
    c->qa.k = k;
    c->qa.nb = k / QK;
    c->qa.ns = k / QKK;
    c->qa.has256 = k % QKK == 0;
    if (m * k < 32768) { quantize_act_rows(x, ld, k, c->qa, 0, m); return; }
    parallel_for(*c->m->pool, m, std::max(1, 8192 / k), [&](int a, int b, int) { quantize_act_rows(x, ld, k, c->qa, a, b); });
}

// y[m][rows] = W x  (x already in c->qa when W is quantized; xf for dense W)
// imatrix: running sum of x^2 per input column, per weight tensor
void calib_acc(ei_ctx* c, const Tensor& w, const float* xf, int ldx, int m) {
    auto& acc = c->calib_sum[w.name];
    if (acc.empty()) acc.assign(w.cols, 0.0);
    for (int i = 0; i < m; ++i) {
        const float* xi = xf + size_t(i) * ldx;
        for (int64_t j = 0; j < w.cols; ++j) acc[j] += double(xi[j]) * xi[j];
    }
    c->calib_n[w.name] += m;
}

void mm(ei_ctx* c, const Tensor& w, const float* xf, int ldx, int m, float* y, int ldy) {
    if (c->calib) calib_acc(c, w, xf, ldx, m);
    Timer tm(c, OP_MATMUL);
    matmul(*c->m->pool, w, &c->qa, xf, ldx, m, y, ldy);
}

// ---------------------------------------------------------------- Gated DeltaNet

void deltanet_block(ei_ctx* c, const LayerW& L, int M) {
    ei_model* m = c->m;
    const Config& cf = m->cfg;
    const int D = cf.D, C = cf.C(), Hk = cf.Hk, Hv = cf.Hv, dk = cf.dk, dv = cf.dv, K1 = cf.convk - 1;
    const int P = C + Hv * dv;
    const int li = L.idx;
    float* proj = c->proj.p;

    norm_rows(c, c->x.p, D, L.attn_norm, c->xn.p, D, M, D);
    qact(c, c->xn.p, D, M, D);
    mm(c, L.lin.qkvz, c->xn.p, D, M, proj, P);
    mm(c, L.lin.ab, c->xn.p, D, M, c->ab.p, 2 * Hv);

    const int nseg = int(c->seg_t0.size());
    // causal depthwise conv + SiLU, in place over the q/k/v channels
    {
        Timer tm(c, OP_CONV);
        const int CH = 256;
        const int chunks = (C + CH - 1) / CH;
        struct J { ei_ctx* c; const LayerW* L; int C, K1, P, chunks, CH, li; float* proj; };
        J j{c, &L, C, K1, P, chunks, CH, li, proj};
        m->pool->run(nseg * chunks, [](void* pp, int task, int) {
            J& j = *static_cast<J*>(pp);
            ei_ctx* c = j.c;
            const int s = task / j.chunks, ch = task % j.chunks;
            const int c0 = ch * j.CH, c1 = std::min(j.C, c0 + j.CH);
            const int slot = c->seg_slot[s];
            Slot& S = c->slots[slot];
            const bool verify = c->seg_flags[s] & EI_SEG_VERIFY;
            const float* w = j.L->lin.conv;  // [C][4]
            float* st = c->conv_ptr(slot, S.cur, j.li);  // [K1][C]
            const int t0 = c->seg_t0[s], t1 = c->seg_t1[s];
            for (int cc = c0; cc < c1; cc += 8) {
                // weights for 8 channels, transposed to 4 vectors
                alignas(32) float wt[4][8];
                for (int q = 0; q < 8; ++q)
                    for (int k = 0; k < 4; ++k) wt[k][q] = w[(cc + q) * 4 + k];
                const __m256 w0 = _mm256_load_ps(wt[0]), w1 = _mm256_load_ps(wt[1]);
                const __m256 w2 = _mm256_load_ps(wt[2]), w3 = _mm256_load_ps(wt[3]);
                __m256 s0 = _mm256_loadu_ps(st + 0 * j.C + cc);
                __m256 s1 = _mm256_loadu_ps(st + 1 * j.C + cc);
                __m256 s2 = _mm256_loadu_ps(st + 2 * j.C + cc);
                for (int t = t0; t < t1; ++t) {
                    float* xp = j.proj + size_t(t) * j.P + cc;
                    const __m256 raw = _mm256_loadu_ps(xp);
                    __m256 acc = _mm256_mul_ps(w0, s0);
                    acc = _mm256_fmadd_ps(w1, s1, acc);
                    acc = _mm256_fmadd_ps(w2, s2, acc);
                    acc = _mm256_fmadd_ps(w3, raw, acc);
                    _mm256_storeu_ps(xp, silu256(acc));
                    s0 = s1; s1 = s2; s2 = raw;
                    if (verify) {
                        const int b = S.verify_buf[t - t0];
                        float* vs = c->conv_ptr(slot, b, j.li);
                        _mm256_storeu_ps(vs + 0 * j.C + cc, s0);
                        _mm256_storeu_ps(vs + 1 * j.C + cc, s1);
                        _mm256_storeu_ps(vs + 2 * j.C + cc, s2);
                    }
                }
                if (!verify) {
                    _mm256_storeu_ps(st + 0 * j.C + cc, s0);
                    _mm256_storeu_ps(st + 1 * j.C + cc, s1);
                    _mm256_storeu_ps(st + 2 * j.C + cc, s2);
                }
            }
        }, &j);
    }

    // per token: L2-normalise q,k (q also scaled by 1/sqrt(dk)); gates
    {
        Timer tm(c, OP_MISC);
        const float qscale = 1.0f / std::sqrt(float(dk));
        auto body = [&](int a, int b, int) {
            for (int t = a; t < b; ++t) {
                float* pr = proj + size_t(t) * P;
                for (int h = 0; h < 2 * Hk; ++h) {
                    float* v = pr + h * dk;
                    const float ss = dot(v, v, dk);
                    float r = 1.0f / std::sqrt(ss + 1e-6f);
                    if (h < Hk) r *= qscale;
                    for (int i = 0; i < dk; ++i) v[i] *= r;
                }
                const float* abr = c->ab.p + size_t(t) * 2 * Hv;  // b then a
                float* g = c->gb.p + size_t(t) * 2 * Hv;          // decay then beta
                for (int h = 0; h < Hv; ++h) {
                    g[h] = std::exp(L.lin.a[h] * softplus(abr[Hv + h] + L.lin.dt_bias[h]));
                    g[Hv + h] = sigmoid(abr[h]);
                }
            }
        };
        if (M < 4) body(0, M, 0);
        else parallel_for(*m->pool, M, 2, body);
    }

    // the delta rule, per (segment, v-head, 32-column block); the state block
    // (dk x 32 floats = 16 KB) stays in L1 for the whole token run
    {
        Timer tm(c, OP_DELTA);
        const int JB = dv / 32;
        const int rep = Hv / Hk;
        struct J { ei_ctx* c; int Hv, JB, rep, dk, dv, Hk, P, li; float* proj; };
        J j{c, Hv, JB, rep, dk, dv, Hk, P, li, proj};
        m->pool->run(nseg * Hv * JB, [](void* pp, int task, int) {
            J& j = *static_cast<J*>(pp);
            ei_ctx* c = j.c;
            const int s = task / (j.Hv * j.JB);
            const int h = (task / j.JB) % j.Hv;
            const int jb = task % j.JB;
            const int slot = c->seg_slot[s];
            Slot& S = c->slots[slot];
            const bool verify = c->seg_flags[s] & EI_SEG_VERIFY;
            const int kh = h / j.rep;
            const int dk = j.dk, dv = j.dv;
            float* Sg = c->rec_ptr(slot, S.cur, j.li) + size_t(h) * dk * dv + jb * 32;
            alignas(32) float Sl[256 * 32];  // dk <= 256
            for (int i = 0; i < dk; ++i)
                for (int q = 0; q < 32; q += 8) _mm256_store_ps(Sl + i * 32 + q, _mm256_loadu_ps(Sg + size_t(i) * dv + q));
            const int Kq = 0, Kk = j.Hk * dk, Kv = 2 * j.Hk * dk;
            const int Cz = Kv + j.Hv * dv;  // == C
            (void)Cz;
            for (int t = c->seg_t0[s]; t < c->seg_t1[s]; ++t) {
                const float* pr = j.proj + size_t(t) * j.P;
                const float* q = pr + Kq + kh * dk;
                const float* k = pr + Kk + kh * dk;
                const float* v = pr + Kv + h * dv + jb * 32;
                const float* gb = c->gb.p + size_t(t) * 2 * j.Hv;
                const float e = gb[h], beta = gb[j.Hv + h];
                __m256 sk0 = _mm256_setzero_ps(), sk1 = sk0, sk2 = sk0, sk3 = sk0;
                __m256 sq0 = sk0, sq1 = sk0, sq2 = sk0, sq3 = sk0;
                for (int i = 0; i < dk; ++i) {
                    const float* r = Sl + i * 32;
                    const __m256 ki = _mm256_set1_ps(k[i]), qi = _mm256_set1_ps(q[i]);
                    const __m256 r0 = _mm256_load_ps(r), r1 = _mm256_load_ps(r + 8);
                    const __m256 r2 = _mm256_load_ps(r + 16), r3 = _mm256_load_ps(r + 24);
                    sk0 = _mm256_fmadd_ps(r0, ki, sk0); sk1 = _mm256_fmadd_ps(r1, ki, sk1);
                    sk2 = _mm256_fmadd_ps(r2, ki, sk2); sk3 = _mm256_fmadd_ps(r3, ki, sk3);
                    sq0 = _mm256_fmadd_ps(r0, qi, sq0); sq1 = _mm256_fmadd_ps(r1, qi, sq1);
                    sq2 = _mm256_fmadd_ps(r2, qi, sq2); sq3 = _mm256_fmadd_ps(r3, qi, sq3);
                }
                const float kq = dot(k, q, dk);
                const __m256 ev = _mm256_set1_ps(e), bv = _mm256_set1_ps(beta), kqv = _mm256_set1_ps(kq);
                // delta = beta * (v - e * S^T k);   o = e * S^T q + delta * (k.q)
                __m256 d0 = _mm256_mul_ps(bv, _mm256_fnmadd_ps(ev, sk0, _mm256_loadu_ps(v)));
                __m256 d1 = _mm256_mul_ps(bv, _mm256_fnmadd_ps(ev, sk1, _mm256_loadu_ps(v + 8)));
                __m256 d2 = _mm256_mul_ps(bv, _mm256_fnmadd_ps(ev, sk2, _mm256_loadu_ps(v + 16)));
                __m256 d3 = _mm256_mul_ps(bv, _mm256_fnmadd_ps(ev, sk3, _mm256_loadu_ps(v + 24)));
                float* out = c->o.p + size_t(t) * (j.Hv * dv) + h * dv + jb * 32;
                _mm256_storeu_ps(out, _mm256_fmadd_ps(d0, kqv, _mm256_mul_ps(ev, sq0)));
                _mm256_storeu_ps(out + 8, _mm256_fmadd_ps(d1, kqv, _mm256_mul_ps(ev, sq1)));
                _mm256_storeu_ps(out + 16, _mm256_fmadd_ps(d2, kqv, _mm256_mul_ps(ev, sq2)));
                _mm256_storeu_ps(out + 24, _mm256_fmadd_ps(d3, kqv, _mm256_mul_ps(ev, sq3)));
                // S = e * S + k delta^T
                for (int i = 0; i < dk; ++i) {
                    float* r = Sl + i * 32;
                    const __m256 ki = _mm256_set1_ps(k[i]);
                    _mm256_store_ps(r, _mm256_fmadd_ps(ki, d0, _mm256_mul_ps(ev, _mm256_load_ps(r))));
                    _mm256_store_ps(r + 8, _mm256_fmadd_ps(ki, d1, _mm256_mul_ps(ev, _mm256_load_ps(r + 8))));
                    _mm256_store_ps(r + 16, _mm256_fmadd_ps(ki, d2, _mm256_mul_ps(ev, _mm256_load_ps(r + 16))));
                    _mm256_store_ps(r + 24, _mm256_fmadd_ps(ki, d3, _mm256_mul_ps(ev, _mm256_load_ps(r + 24))));
                }
                if (verify) {
                    float* dst = c->rec_ptr(slot, S.verify_buf[t - c->seg_t0[s]], j.li) + size_t(h) * dk * dv + jb * 32;
                    for (int i = 0; i < dk; ++i)
                        for (int qq = 0; qq < 32; qq += 8) _mm256_storeu_ps(dst + size_t(i) * dv + qq, _mm256_load_ps(Sl + i * 32 + qq));
                }
            }
            if (!verify)
                for (int i = 0; i < dk; ++i)
                    for (int qq = 0; qq < 32; qq += 8) _mm256_storeu_ps(Sg + size_t(i) * dv + qq, _mm256_load_ps(Sl + i * 32 + qq));
        }, &j);
    }

    // gated RMSNorm per head: norm(o) * w * silu(z)
    {
        Timer tm(c, OP_MISC);
        const int HVD = Hv * dv;
        auto body = [&](int a, int b, int) {
            for (int t = a; t < b; ++t) {
                float* ot = c->o.p + size_t(t) * HVD;
                const float* z = proj + size_t(t) * P + C;
                for (int h = 0; h < Hv; ++h) {
                    float* oh = ot + h * dv;
                    const float* zh = z + h * dv;
                    const float r = 1.0f / std::sqrt(dot(oh, oh, dv) / dv + cf.eps);
                    for (int i = 0; i < dv; i += 8) {
                        __m256 v = _mm256_mul_ps(_mm256_loadu_ps(oh + i), _mm256_set1_ps(r));
                        v = _mm256_mul_ps(v, _mm256_loadu_ps(L.lin.norm + i));
                        _mm256_storeu_ps(oh + i, _mm256_mul_ps(v, silu256(_mm256_loadu_ps(zh + i))));
                    }
                }
            }
        };
        if (M < 4) body(0, M, 0);
        else parallel_for(*m->pool, M, 2, body);
    }
    qact(c, c->o.p, Hv * dv, M, Hv * dv);
    mm(c, L.lin.out, c->o.p, Hv * dv, M, c->y.p, D);
    for (int t = 0; t < M; ++t) {
        float* xt = c->x.p + size_t(t) * D;
        const float* yt = c->y.p + size_t(t) * D;
        for (int i = 0; i < D; ++i) xt[i] += yt[i];
    }
}

// ---------------------------------------------------------------- attention

struct KVSel {
    // returns K/V base pointers ([n_ctx][Hkv*Dh]) for a slot
    uint16_t* (*k)(ei_ctx*, int slot, int ai);
    uint16_t* (*v)(ei_ctx*, int slot, int ai);
};

uint16_t* main_k(ei_ctx* c, int slot, int ai) { return c->k_ptr(slot, ai); }
uint16_t* main_v(ei_ctx* c, int slot, int ai) { return c->v_ptr(slot, ai); }
uint16_t* mtp_k(ei_ctx* c, int slot, int) { return c->mk[slot].p; }
uint16_t* mtp_v(ei_ctx* c, int slot, int) { return c->mv[slot].p; }

void rope_inplace(float* v, const float* cs, const float* sn, int half) {
    for (int i = 0; i < half; ++i) {
        const float a = v[i], b = v[i + half];
        v[i] = a * cs[i] - b * sn[i];
        v[i + half] = b * cs[i] + a * sn[i];
    }
}

void attention_block(ei_ctx* c, const LayerW& L, int M, const KVSel& kv) {
    ei_model* m = c->m;
    const Config& cf = m->cfg;
    const int D = cf.D, Hq = cf.Hq, Hkv = cf.Hkv, Dh = cf.Dh;
    const int qw = cf.attn_gate ? 2 * Dh : Dh;  // per-head width of the q part
    const int P = cf.qkv_rows();
    const int KO = Hq * qw, VO = KO + Hkv * Dh;
    const int KVD = Hkv * Dh;
    const int half = cf.rope_dim / 2;
    const int ai = L.idx;
    float* proj = c->proj.p;

    norm_rows(c, c->x.p, D, L.attn_norm, c->xn.p, D, M, D);
    qact(c, c->xn.p, D, M, D);
    mm(c, L.att.qkv, c->xn.p, D, M, proj, P);

    // q/k norm, RoPE, scale q; write K/V to the cache
    {
        Timer tm(c, OP_MISC);
        const float qscale = 1.0f / std::sqrt(float(Dh));
        auto body = [&](int a, int b, int) {
            std::vector<float> tmp(Dh);
            for (int t = a; t < b; ++t) {
                float* pr = proj + size_t(t) * P;
                const float* cs = c->rc.p + size_t(t) * half;
                const float* sn = c->rs.p + size_t(t) * half;
                for (int h = 0; h < Hq; ++h) {
                    float* q = pr + h * qw;
                    rmsnorm(q, L.att.qn, q, Dh, cf.eps);
                    rope_inplace(q, cs, sn, half);
                    for (int i = 0; i < Dh; ++i) q[i] *= qscale;
                }
                const Tok& tk = c->toks[t];
                uint16_t* kd = kv.k(c, tk.slot, ai) + size_t(tk.pos) * KVD;
                uint16_t* vd = kv.v(c, tk.slot, ai) + size_t(tk.pos) * KVD;
                for (int g = 0; g < Hkv; ++g) {
                    float* k = pr + KO + g * Dh;
                    rmsnorm(k, L.att.kn, tmp.data(), Dh, cf.eps);
                    rope_inplace(tmp.data(), cs, sn, half);
                    to_fp16(tmp.data(), kd + g * Dh, Dh);
                    to_fp16(pr + VO + g * Dh, vd + g * Dh, Dh);
                }
            }
        };
        if (M < 4) body(0, M, 0);
        else parallel_for(*m->pool, M, 2, body);
    }

    // causal attention over the cache. Few query tokens (decode): split each
    // token's keys into chunks across threads and merge (flash-decoding).
    {
        Timer tm(c, OP_ATTN);
        const int rep = Hq / Hkv;
        const int threads = m->pool->size();
        int chunk = 1 << 30;
        if (M <= 4) {
            int maxk = 0;
            for (int t = 0; t < M; ++t) maxk = std::max(maxk, c->toks[t].pos + 1);
            chunk = std::max(128, (maxk + threads * 2 - 1) / (threads * 2));
            chunk = (chunk + 31) / 32 * 32;
        }
        struct Task { int t, g, k0, k1, part; };
        std::vector<Task> tasks;
        std::vector<int> part0(M + 1, 0);
        int parts = 0;
        for (int t = 0; t < M; ++t) {
            const int nk = c->toks[t].pos + 1;
            const int nch = (nk + chunk - 1) / chunk;
            part0[t] = parts;
            for (int g = 0; g < Hkv; ++g)
                for (int ch = 0; ch < nch; ++ch)
                    tasks.push_back({t, g, ch * chunk, std::min(nk, (ch + 1) * chunk), parts + g * nch + ch});
            parts += Hkv * nch;
        }
        part0[M] = parts;
        const size_t pstride = size_t(rep) * (Dh + 2);
        if (c->att_part.n < size_t(parts) * pstride) c->att_part.resize(size_t(parts) * pstride);
        struct J { ei_ctx* c; const KVSel* kv; Task* tasks; int ai, rep, Dh, KVD, P, qw; size_t pstride; float* proj; };
        J j{c, &kv, tasks.data(), ai, rep, Dh, KVD, P, qw, pstride, proj};
        m->pool->run(int(tasks.size()), [](void* pp, int ti, int) {
            J& j = *static_cast<J*>(pp);
            ei_ctx* c = j.c;
            const Task& T = j.tasks[ti];
            const Tok& tk = c->toks[T.t];
            const int Dh = j.Dh, rep = j.rep;
            const uint16_t* K = j.kv->k(c, tk.slot, j.ai) + T.g * Dh;
            const uint16_t* Vv = j.kv->v(c, tk.slot, j.ai) + T.g * Dh;
            const float* qbase = j.proj + size_t(T.t) * j.P;
            const float* q[16];
            for (int r = 0; r < rep; ++r) q[r] = qbase + (T.g * rep + r) * j.qw;
            float* part = c->att_part.p + size_t(T.part) * j.pstride;
            float mx[16], l[16];
            for (int r = 0; r < rep; ++r) { mx[r] = -FLT_MAX; l[r] = 0; std::memset(part + r * (Dh + 2), 0, Dh * 4); }
            alignas(32) float s[16][32];
            for (int k0 = T.k0; k0 < T.k1; k0 += 32) {
                const int nk = std::min(32, T.k1 - k0);
                for (int jj = 0; jj < nk; ++jj) {
                    const uint16_t* kr = K + size_t(k0 + jj) * j.KVD;
                    __m256 acc[16];
                    for (int r = 0; r < rep; ++r) acc[r] = _mm256_setzero_ps();
                    for (int i = 0; i < Dh; i += 8) {
                        const __m256 kf = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(kr + i)));
                        for (int r = 0; r < rep; ++r) acc[r] = _mm256_fmadd_ps(kf, _mm256_loadu_ps(q[r] + i), acc[r]);
                    }
                    for (int r = 0; r < rep; ++r) s[r][jj] = hsum8(acc[r]);
                }
                for (int r = 0; r < rep; ++r) {
                    float bm = -FLT_MAX;
                    for (int jj = 0; jj < nk; ++jj) bm = std::max(bm, s[r][jj]);
                    const float mn = std::max(mx[r], bm);
                    const float corr = std::exp(mx[r] - mn);
                    float sum = 0;
                    for (int jj = 0; jj < nk; ++jj) { s[r][jj] = std::exp(s[r][jj] - mn); sum += s[r][jj]; }
                    l[r] = l[r] * corr + sum;
                    mx[r] = mn;
                    if (corr != 1.0f) {
                        float* acc = part + r * (Dh + 2);
                        const __m256 cv = _mm256_set1_ps(corr);
                        for (int i = 0; i < Dh; i += 8) _mm256_storeu_ps(acc + i, _mm256_mul_ps(_mm256_loadu_ps(acc + i), cv));
                    }
                }
                for (int jj = 0; jj < nk; ++jj) {
                    const uint16_t* vr = Vv + size_t(k0 + jj) * j.KVD;
                    for (int i = 0; i < Dh; i += 8) {
                        const __m256 vf = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(vr + i)));
                        for (int r = 0; r < rep; ++r) {
                            float* acc = part + r * (Dh + 2) + i;
                            _mm256_storeu_ps(acc, _mm256_fmadd_ps(_mm256_set1_ps(s[r][jj]), vf, _mm256_loadu_ps(acc)));
                        }
                    }
                }
            }
            for (int r = 0; r < rep; ++r) { part[r * (Dh + 2) + Dh] = mx[r]; part[r * (Dh + 2) + Dh + 1] = l[r]; }
        }, &j);

        // merge partials -> o[t][h*Dh], apply the sigmoid output gate
        const int HQD = Hq * Dh;
        for (int t = 0; t < M; ++t) {
            const int nparts_g = (part0[t + 1] - part0[t]) / Hkv;
            const float* pr = proj + size_t(t) * P;
            for (int g = 0; g < Hkv; ++g)
                for (int r = 0; r < rep; ++r) {
                    const int h = g * rep + r;
                    float* out = c->o.p + size_t(t) * HQD + h * Dh;
                    float M_ = -FLT_MAX;
                    for (int p = 0; p < nparts_g; ++p) {
                        const float* pp = c->att_part.p + size_t(part0[t] + g * nparts_g + p) * pstride + r * (Dh + 2);
                        M_ = std::max(M_, pp[Dh]);
                    }
                    std::fill(out, out + Dh, 0.0f);
                    float den = 0;
                    for (int p = 0; p < nparts_g; ++p) {
                        const float* pp = c->att_part.p + size_t(part0[t] + g * nparts_g + p) * pstride + r * (Dh + 2);
                        const float w = std::exp(pp[Dh] - M_);
                        den += w * pp[Dh + 1];
                        for (int i = 0; i < Dh; ++i) out[i] += w * pp[i];
                    }
                    const float inv = 1.0f / den;
                    if (cf.attn_gate) {
                        const float* gate = pr + h * qw + Dh;
                        for (int i = 0; i < Dh; ++i) out[i] *= inv * sigmoid(gate[i]);
                    } else {
                        for (int i = 0; i < Dh; ++i) out[i] *= inv;
                    }
                }
        }
    }
    qact(c, c->o.p, Hq * Dh, M, Hq * Dh);
    mm(c, L.att.o, c->o.p, Hq * Dh, M, c->y.p, D);
    for (int t = 0; t < M; ++t) {
        float* xt = c->x.p + size_t(t) * D;
        const float* yt = c->y.p + size_t(t) * D;
        for (int i = 0; i < D; ++i) xt[i] += yt[i];
    }
}

void ffn_block(ei_ctx* c, const LayerW& L, int M) {
    ei_model* m = c->m;
    const int D = m->cfg.D, F = m->cfg.F;
    norm_rows(c, c->x.p, D, L.ffn_norm, c->xn.p, D, M, D);
    qact(c, c->xn.p, D, M, D);
    mm(c, L.gate_up, c->xn.p, D, M, c->proj.p, 2 * F);
    {
        Timer tm(c, OP_MISC);
        auto body = [&](int a, int b, int) {
            for (int t = a; t < b; ++t) {
                const float* g = c->proj.p + size_t(t) * 2 * F;
                float* h = c->o.p + size_t(t) * F;
                for (int i = 0; i < F; i += 8)
                    _mm256_storeu_ps(h + i, _mm256_mul_ps(silu256(_mm256_loadu_ps(g + i)), _mm256_loadu_ps(g + F + i)));
            }
        };
        if (M < 4) body(0, M, 0);
        else parallel_for(*m->pool, M, 2, body);
    }
    qact(c, c->o.p, F, M, F);
    mm(c, L.down, c->o.p, F, M, c->y.p, D);
    for (int t = 0; t < M; ++t) {
        float* xt = c->x.p + size_t(t) * D;
        const float* yt = c->y.p + size_t(t) * D;
        for (int i = 0; i < D; ++i) xt[i] += yt[i];
    }
}

void rope_tables(ei_ctx* c, int M) {
    const Config& cf = c->m->cfg;
    const int half = cf.rope_dim / 2;
    for (int t = 0; t < M; ++t) {
        const double pos = c->toks[t].pos;
        for (int i = 0; i < half; ++i) {
            const double inv = std::pow(double(cf.rope_theta), -2.0 * i / cf.rope_dim);
            c->rc.p[size_t(t) * half + i] = float(std::cos(pos * inv));
            c->rs.p[size_t(t) * half + i] = float(std::sin(pos * inv));
        }
    }
}

void embed_tokens(ei_ctx* c, const int32_t* ids, int M, float* out) {
    Timer tm(c, OP_EMBED);
    ei_model* m = c->m;
    for (int t = 0; t < M; ++t)
        dequant_row(m->embed.dtype, m->embed.data, m->embed.rows, m->embed.cols, ids[t], out + size_t(t) * m->cfg.D);
}

}  // namespace

// ============================================================== API: quant

extern "C" int64_t ei_tensor_bytes(int dtype, int64_t rows, int64_t cols) { return tensor_bytes(dtype, rows, cols); }

extern "C" int ei_quantize(int dtype, const float* src, int64_t rows, int64_t cols, void* dst, const float* imp, int nt) {
    if (tensor_bytes(dtype, rows, cols) < 0) return -1;
    quantize(dtype, src, rows, cols, dst, imp, nt);
    return 0;
}

extern "C" int ei_dequantize_row(int dtype, const void* data, int64_t rows, int64_t cols, int64_t row, float* out) {
    if (row < 0 || row >= rows) return -1;
    dequant_row(dtype, data, rows, cols, row, out);
    return 0;
}

extern "C" int ei_matmul(int dtype, const void* w, int64_t rows, int64_t cols, const float* x, int m, float* y,
                         int n_threads) {
    if (tensor_bytes(dtype, rows, cols) < 0 || m <= 0) return -1;
    ThreadPool pool(std::max(1, n_threads));
    Tensor t;
    t.dtype = dtype;
    t.rows = rows;
    t.cols = cols;
    t.data = static_cast<const uint8_t*>(w);
    t.nbytes = tensor_bytes(dtype, rows, cols);
    QAct qa;
    qa.reserve(m, int(cols));
    qa.m = m;
    qa.k = int(cols);
    qa.nb = int(cols / QK);
    qa.ns = int(cols / QKK);
    qa.has256 = cols % QKK == 0;
    if (dtype_is_tiled(dtype)) quantize_act_rows(x, int(cols), int(cols), qa, 0, m);
    matmul(pool, t, &qa, x, int(cols), m, y, int(rows));
    return 0;
}

// ============================================================== API: model

extern "C" ei_model* ei_model_load(const char* path, int n_threads, char* err, int err_len) {
    auto fail = [&](const std::string& msg) -> ei_model* {
        if (err && err_len > 0) { std::snprintf(err, size_t(err_len), "%s", msg.c_str()); }
        return nullptr;
    };
    int fd = open(path, O_RDONLY);
    if (fd < 0) return fail(std::string("cannot open ") + path);
    struct stat st;
    fstat(fd, &st);
    void* map = mmap(nullptr, size_t(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) return fail("mmap failed");
    madvise(map, size_t(st.st_size), MADV_WILLNEED);
    auto* m = new ei_model();
    m->map = map;
    m->map_size = size_t(st.st_size);
    try {
        Loader ld;
        ld.base = static_cast<const uint8_t*>(map);
        ld.size = m->map_size;
        ld.parse();
        Config& c = m->cfg;
        c.arch = ld.arch;
        c.D = ld.ci("hidden_size");
        c.L = ld.ci("num_hidden_layers");
        c.F = ld.ci("intermediate_size");
        c.Hq = ld.ci("num_attention_heads");
        c.Hkv = ld.ci("num_key_value_heads");
        c.Dh = ld.ci("head_dim");
        c.Hk = ld.ci("linear_num_key_heads", 0);
        c.Hv = ld.ci("linear_num_value_heads", 0);
        c.dk = ld.ci("linear_key_head_dim", 0);
        c.dv = ld.ci("linear_value_head_dim", 0);
        c.convk = ld.ci("linear_conv_kernel_dim", 4);
        c.V = ld.ci("vocab_size");
        c.rope_dim = ld.ci("rope_dim");
        c.mtp_layers = ld.ci("mtp_num_hidden_layers", 0);
        c.attn_gate = ld.ci("attn_output_gate", 1);
        c.rope_theta = float(ld.cfg.at("rope_theta"));
        c.eps = float(ld.cfg.at("rms_norm_eps"));
        for (double v : ld.arr.at("layer_types")) c.layer_types.push_back(int(v));
        for (auto& kv : ld.t)
            if (kv.second.dtype == Q4K && ld.ci("q4k_layout", 1) < 2)
                throw std::runtime_error("q4k tensors use an old layout; re-convert the model");
        if (int(c.layer_types.size()) != c.L) throw std::runtime_error("layer_types length");
        if (c.dk > 256 || c.dv % 32 || c.convk != 4 || c.Dh % 8 || c.Hq / std::max(1, c.Hkv) > 16)
            throw std::runtime_error("unsupported dimensions");
        m->layers.resize(c.L);
        for (int i = 0; i < c.L; ++i) {
            LayerW& L = m->layers[i];
            L.type = c.layer_types[i];
            L.idx = L.type == 0 ? m->n_lin++ : m->n_att++;
            load_layer(ld, c, "blk." + std::to_string(i) + ".", L);
        }
        m->embed = ld.get("tok_embd");
        check_mat(m->embed, c.V, c.D, "tok_embd");
        m->lm_head = ld.has("output") ? ld.get("output") : m->embed;
        m->out_norm = ld.f32("output_norm", c.D);
        if (c.mtp_layers > 0 && ld.has("mtp.fc")) {
            MtpW& t = m->mtp;
            t.present = true;
            t.fc = ld.get("mtp.fc");
            check_mat(t.fc, c.D, 2 * c.D, "mtp.fc");
            t.norm_emb = ld.f32("mtp.norm_emb", c.D);
            t.norm_hid = ld.f32("mtp.norm_hidden", c.D);
            t.out_norm = ld.f32("mtp.output_norm", c.D);
            t.layer.type = 1;
            t.layer.idx = 0;
            load_layer(ld, c, "mtp.blk.", t.layer);
        }
        m->pool = std::make_unique<ThreadPool>(n_threads > 0 ? n_threads : 4);

        // touch every page once so the first request does not pay page faults
        const uint8_t* base = static_cast<const uint8_t*>(map);
        const size_t size = m->map_size;
        std::vector<uint64_t> sink(size_t(8) * (m->pool->size() + 1), 0);
        parallel_for(*m->pool, int((size + (1 << 20) - 1) >> 20), 4, [&](int a, int b, int th) {
            uint64_t s = 0;
            for (size_t mb = size_t(a); mb < size_t(b); ++mb)
                for (size_t off = mb << 20; off < std::min(size, (mb + 1) << 20); off += 4096) s += base[off];
            sink[th * 8] += s;
        });

        std::ostringstream js;
        int64_t bytes[16] = {};
        for (auto& kv : ld.t) bytes[kv.second.dtype] += kv.second.nbytes;
        js << "{\"arch\":\"" << c.arch << "\",\"hidden_size\":" << c.D << ",\"layers\":" << c.L
           << ",\"linear_layers\":" << m->n_lin << ",\"attention_layers\":" << m->n_att
           << ",\"vocab\":" << c.V << ",\"mtp\":" << (m->mtp.present ? "true" : "false")
           << ",\"file_bytes\":" << size << ",\"tensors\":" << ld.t.size()
           << ",\"lm_head_dtype\":\"" << dtype_name(m->lm_head.dtype) << "\""
           << ",\"bytes_by_dtype\":{";
        bool first = true;
        for (int d = 0; d <= Q6; ++d)
            if (bytes[d]) { js << (first ? "" : ",") << "\"" << dtype_name(d) << "\":" << bytes[d]; first = false; }
        js << "}}";
        m->info = js.str();
    } catch (std::exception& e) {
        munmap(map, m->map_size);
        delete m;
        return fail(e.what());
    }
    return m;
}

extern "C" void ei_model_free(ei_model* m) {
    if (!m) return;
    m->pool.reset();
    if (m->map) munmap(m->map, m->map_size);
    delete m;
}

extern "C" const char* ei_model_info(const ei_model* m) { return m->info.c_str(); }
extern "C" int ei_n_vocab(const ei_model* m) { return m->cfg.V; }
extern "C" int ei_n_embd(const ei_model* m) { return m->cfg.D; }
extern "C" int ei_has_mtp(const ei_model* m) { return m->mtp.present ? 1 : 0; }
extern "C" int ei_get_threads(const ei_model* m) { return m->pool->size(); }
extern "C" void ei_set_threads(ei_model* m, int n) { m->pool->resize(std::max(1, n)); }

extern "C" int ei_set_draft_vocab(ei_model* m, const int32_t* ids, int n) {
    m->draft_ids.clear();
    m->draft_head = Tensor();
    if (n <= 0) return 0;
    // Copy the selected rows of the (tiled) LM head into a new tiled tensor,
    // block by block, so the draft head is bit-identical to the full one.
    const Tensor& h = m->lm_head;
    if (!dtype_is_tiled(h.dtype)) return -1;
    const int64_t cols = h.cols;
    const int64_t rows = n;
    m->draft_data.resize(size_t(tensor_bytes(h.dtype, rows, cols)));
    m->draft_data.zero();
    for (int64_t r = 0; r < rows; ++r) {
        if (ids[r] < 0 || ids[r] >= h.rows) return -2;
        copy_tiled_row(h.dtype, cols, h.data, ids[r], m->draft_data.p, r);
    }
    m->draft_ids.assign(ids, ids + n);
    m->draft_head.dtype = h.dtype;
    m->draft_head.rows = rows;
    m->draft_head.cols = cols;
    m->draft_head.data = m->draft_data.p;
    m->draft_head.nbytes = tensor_bytes(h.dtype, rows, cols);
    return 0;
}

// ============================================================== API: context

extern "C" ei_ctx* ei_ctx_new(ei_model* m, int n_slots, int n_ctx, int max_batch, int max_draft) {
    try {
        auto* c = new ei_ctx();
        const Config& cf = m->cfg;
        c->m = m;
        c->n_slots = std::max(1, n_slots);
        c->n_ctx = std::max(16, n_ctx);
        c->max_batch = std::max(1, max_batch);
        c->max_draft = std::clamp(max_draft, 0, MAX_VERIFY - 2);
        c->nbuf = 1 + c->max_draft;
        c->max_logits = std::max(64, c->n_slots * (c->max_draft + 1));
        c->slots.resize(c->n_slots);
        for (int s = 0; s < c->n_slots; ++s) {
            for (int b = 0; b < c->nbuf; ++b) {
                c->rec.emplace_back(std::max<size_t>(1, c->rec_floats()));
                c->rec.back().zero();
                c->conv.emplace_back(std::max<size_t>(1, c->conv_floats()));
                c->conv.back().zero();
            }
            c->kc.emplace_back(std::max<size_t>(1, size_t(m->n_att) * c->n_ctx * c->kvd()));
            c->vc.emplace_back(std::max<size_t>(1, size_t(m->n_att) * c->n_ctx * c->kvd()));
            if (m->mtp.present) {
                c->mk.emplace_back(size_t(c->n_ctx) * c->kvd());
                c->mv.emplace_back(size_t(c->n_ctx) * c->kvd());
            }
        }
        const int MB = c->max_batch;
        int wide = std::max({cf.C() + cf.Hv * cf.dv, cf.qkv_rows(), 2 * cf.F, 2 * cf.D});
        int owide = std::max({cf.Hv * cf.dv, cf.Hq * cf.Dh, cf.F, 2 * cf.D});
        c->x.resize(size_t(MB) * cf.D);
        c->xn.resize(size_t(MB) * cf.D);
        c->y.resize(size_t(MB) * cf.D);
        c->proj.resize(size_t(MB) * wide);
        c->ab.resize(size_t(MB) * 2 * std::max(1, cf.Hv));
        c->gb.resize(size_t(MB) * 2 * std::max(1, cf.Hv));
        c->o.resize(size_t(MB) * owide);
        c->qa.reserve(MB, owide);
        c->logits.resize(size_t(c->max_logits) * cf.V);
        c->hid.resize(size_t(c->max_logits) * cf.D);
        c->hid_all.resize(size_t(MB) * cf.D);
        c->mtp_hid.resize(size_t(MB) * cf.D);
        c->cat.resize(size_t(MB) * 2 * cf.D);
        c->rc.resize(size_t(MB) * std::max(1, cf.rope_dim / 2));
        c->rs.resize(size_t(MB) * std::max(1, cf.rope_dim / 2));
        c->att_part.resize(size_t(64) * 16 * (cf.Dh + 2));
        c->samp.resize(cf.V);
        return c;
    } catch (...) {
        return nullptr;
    }
}

extern "C" void ei_ctx_free(ei_ctx* c) { delete c; }

extern "C" int64_t ei_ctx_bytes(ei_ctx* c) {
    int64_t b = 0;
    for (auto& x : c->rec) b += x.bytes();
    for (auto& x : c->conv) b += x.bytes();
    for (auto& x : c->kc) b += x.bytes();
    for (auto& x : c->vc) b += x.bytes();
    for (auto& x : c->mk) b += x.bytes();
    for (auto& x : c->mv) b += x.bytes();
    b += c->x.bytes() + c->xn.bytes() + c->y.bytes() + c->proj.bytes() + c->o.bytes() + c->logits.bytes() +
         c->hid.bytes() + c->hid_all.bytes() + c->qa.q.bytes();
    return b;
}

extern "C" int ei_slot_len(ei_ctx* c, int slot) { return (slot >= 0 && slot < c->n_slots) ? c->slots[slot].len : -1; }

extern "C" void ei_slot_reset(ei_ctx* c, int slot) {
    if (slot < 0 || slot >= c->n_slots) return;
    Slot& S = c->slots[slot];
    S.len = 0;
    S.verify_n = 0;
    S.mtp_len = 0;
    c->rec[size_t(slot) * c->nbuf + S.cur].zero();
    c->conv[size_t(slot) * c->nbuf + S.cur].zero();
}

extern "C" int ei_forward(ei_ctx* c, const ei_seg* segs, int n_segs) {
    ei_model* m = c->m;
    const Config& cf = m->cfg;
    int M = 0, R = 0;
    std::vector<int> used(c->n_slots, 0);
    for (int s = 0; s < n_segs; ++s) {
        const ei_seg& g = segs[s];
        if (g.slot < 0 || g.slot >= c->n_slots) return -1;
        if (used[g.slot]++) return -2;
        Slot& S = c->slots[g.slot];
        if (S.verify_n) return -3;  // previous VERIFY not committed
        if (g.pos0 != S.len) return -4;
        if (g.n_tokens <= 0 || g.pos0 + g.n_tokens > c->n_ctx) return -5;
        if (g.n_logits < 0 || g.n_logits > g.n_tokens) return -6;
        if ((g.flags & EI_SEG_VERIFY) && g.n_tokens > c->nbuf) return -7;
        M += g.n_tokens;
        R += g.n_logits;
    }
    if (M > c->max_batch) return -8;
    if (R > c->max_logits) return -9;

    c->toks.resize(M);
    c->seg_t0.assign(n_segs, 0);
    c->seg_t1.assign(n_segs, 0);
    c->seg_slot.assign(n_segs, 0);
    c->seg_flags.assign(n_segs, 0);
    c->logit_tok.clear();
    std::vector<int32_t> ids(M);
    int t = 0;
    for (int s = 0; s < n_segs; ++s) {
        const ei_seg& g = segs[s];
        Slot& S = c->slots[g.slot];
        c->seg_t0[s] = t;
        c->seg_slot[s] = g.slot;
        c->seg_flags[s] = g.flags;
        if (g.flags & EI_SEG_VERIFY) {
            // token 0 updates the current buffers in place (it is always kept);
            // tokens 1.. write to the other buffers, picked at commit time
            S.verify_buf[0] = S.cur;
            int b = 0;
            for (int i = 1; i < g.n_tokens; ++i) {
                if (b == S.cur) ++b;
                S.verify_buf[i] = b++;
            }
            S.verify_pos0 = g.pos0;
        }
        for (int i = 0; i < g.n_tokens; ++i, ++t) {
            ids[t] = g.tokens[i];
            if (ids[t] < 0 || ids[t] >= cf.V) return -10;
            c->toks[t] = {g.slot, g.pos0 + i, s};
        }
        c->seg_t1[s] = t;
        for (int i = g.n_tokens - g.n_logits; i < g.n_tokens; ++i) c->logit_tok.push_back(c->seg_t0[s] + i);
    }

    embed_tokens(c, ids.data(), M, c->x.p);
    rope_tables(c, M);
    const KVSel kv{main_k, main_v};
    for (const LayerW& L : m->layers) {
        if (L.type == 0) deltanet_block(c, L, M);
        else attention_block(c, L, M, kv);
        ffn_block(c, L, M);
    }

    // hidden states after the final norm
    for (int s = 0; s < n_segs; ++s)
        if (c->seg_flags[s] & EI_SEG_HIDDEN)
            norm_rows(c, c->x.p + size_t(c->seg_t0[s]) * cf.D, cf.D, m->out_norm,
                      c->hid_all.p + size_t(c->seg_t0[s]) * cf.D, cf.D, c->seg_t1[s] - c->seg_t0[s], cf.D);
    R = int(c->logit_tok.size());
    for (int r = 0; r < R; ++r)
        rmsnorm(c->x.p + size_t(c->logit_tok[r]) * cf.D, m->out_norm, c->hid.p + size_t(r) * cf.D, cf.D, cf.eps);
    if (R > 0) {
        if (c->calib) calib_acc(c, m->lm_head, c->hid.p, cf.D, R);
        Timer tm(c, OP_LMHEAD);
        qact(c, c->hid.p, cf.D, R, cf.D);
        matmul(*m->pool, m->lm_head, &c->qa, c->hid.p, cf.D, R, c->logits.p, cf.V);
    }

    for (int s = 0; s < n_segs; ++s) {
        const ei_seg& g = segs[s];
        Slot& S = c->slots[g.slot];
        S.len = g.pos0 + g.n_tokens;
        if (g.flags & EI_SEG_VERIFY) S.verify_n = g.n_tokens;
    }
    c->n_rows = R;
    c->n_forward++;
    c->n_tokens += M;
    return R;
}

extern "C" const float* ei_logits(ei_ctx* c, int row) {
    if (row < 0 || row >= c->n_rows) return nullptr;
    return c->logits.p + size_t(row) * c->m->cfg.V;
}

extern "C" const float* ei_hidden(ei_ctx* c, int row) {
    if (row < 0 || row >= c->n_rows) return nullptr;
    return c->hid.p + size_t(row) * c->m->cfg.D;
}

extern "C" const float* ei_hidden_tok(ei_ctx* c, int i) {
    if (i < 0 || i >= int(c->toks.size())) return nullptr;
    return c->hid_all.p + size_t(i) * c->m->cfg.D;
}

extern "C" int ei_commit(ei_ctx* c, int slot, int n_keep) {
    if (slot < 0 || slot >= c->n_slots) return -1;
    Slot& S = c->slots[slot];
    if (!S.verify_n) return -2;
    if (n_keep < 1 || n_keep > S.verify_n) return -3;
    S.cur = S.verify_buf[n_keep - 1];
    S.len = S.verify_pos0 + n_keep;
    S.verify_n = 0;
    return S.len;
}

// ============================================================== snapshots

extern "C" int ei_snap_save(ei_ctx* c, int slot, int with_kv) {
    if (slot < 0 || slot >= c->n_slots) return -1;
    Slot& S = c->slots[slot];
    if (S.verify_n) return -2;
    auto sn = std::make_unique<Snap>();
    sn->slot = slot;
    sn->len = S.len;
    sn->mtp_len = S.mtp_len;
    sn->with_kv = with_kv != 0;
    sn->rec.resize(c->rec_floats());
    std::memcpy(sn->rec.p, c->rec[size_t(slot) * c->nbuf + S.cur].p, sn->rec.bytes());
    sn->conv.resize(c->conv_floats());
    std::memcpy(sn->conv.p, c->conv[size_t(slot) * c->nbuf + S.cur].p, sn->conv.bytes());
    if (with_kv) {
        const size_t kvd = c->kvd(), na = size_t(c->m->n_att);
        sn->k.resize(std::max<size_t>(1, na * S.len * kvd));
        sn->v.resize(std::max<size_t>(1, na * S.len * kvd));
        for (size_t a = 0; a < na; ++a) {
            std::memcpy(sn->k.p + a * S.len * kvd, c->k_ptr(slot, int(a)), S.len * kvd * 2);
            std::memcpy(sn->v.p + a * S.len * kvd, c->v_ptr(slot, int(a)), S.len * kvd * 2);
        }
        if (c->m->mtp.present && S.mtp_len > 0) {
            sn->mk.resize(size_t(S.mtp_len) * kvd);
            sn->mv.resize(size_t(S.mtp_len) * kvd);
            std::memcpy(sn->mk.p, c->mk[slot].p, sn->mk.bytes());
            std::memcpy(sn->mv.p, c->mv[slot].p, sn->mv.bytes());
        }
    }
    const int id = c->next_snap++;
    c->snaps[id] = std::move(sn);
    return id;
}

extern "C" int ei_snap_restore(ei_ctx* c, int slot, int snap) {
    if (slot < 0 || slot >= c->n_slots) return -1;
    auto it = c->snaps.find(snap);
    if (it == c->snaps.end()) return -2;
    Snap& sn = *it->second;
    if (!sn.with_kv && sn.slot != slot) return -3;
    Slot& S = c->slots[slot];
    S.verify_n = 0;
    std::memcpy(c->rec[size_t(slot) * c->nbuf + S.cur].p, sn.rec.p, sn.rec.bytes());
    std::memcpy(c->conv[size_t(slot) * c->nbuf + S.cur].p, sn.conv.p, sn.conv.bytes());
    if (sn.with_kv) {
        const size_t kvd = c->kvd(), na = size_t(c->m->n_att);
        for (size_t a = 0; a < na; ++a) {
            std::memcpy(c->k_ptr(slot, int(a)), sn.k.p + a * sn.len * kvd, sn.len * kvd * 2);
            std::memcpy(c->v_ptr(slot, int(a)), sn.v.p + a * sn.len * kvd, sn.len * kvd * 2);
        }
        if (sn.mk.n && c->m->mtp.present) {
            std::memcpy(c->mk[slot].p, sn.mk.p, sn.mk.bytes());
            std::memcpy(c->mv[slot].p, sn.mv.p, sn.mv.bytes());
        }
    }
    S.len = sn.len;
    S.mtp_len = std::min(sn.len - 1, sn.with_kv ? int(sn.mk.n / std::max<size_t>(1, c->kvd())) : sn.mtp_len);
    S.mtp_len = std::max(0, std::min(S.mtp_len, sn.mtp_len));
    return S.len;
}

extern "C" int ei_snap_len(ei_ctx* c, int snap) {
    auto it = c->snaps.find(snap);
    return it == c->snaps.end() ? -1 : it->second->len;
}

extern "C" int64_t ei_snap_bytes(ei_ctx* c, int snap) {
    auto it = c->snaps.find(snap);
    return it == c->snaps.end() ? -1 : it->second->bytes();
}

extern "C" void ei_snap_free(ei_ctx* c, int snap) { c->snaps.erase(snap); }

extern "C" int ei_snap_write(ei_ctx* c, int snap, const char* path) {
    auto it = c->snaps.find(snap);
    if (it == c->snaps.end()) return -1;
    Snap& s = *it->second;
    if (!s.with_kv) return -2;
    FILE* f = std::fopen(path, "wb");
    if (!f) return -3;
    const int64_t hdr[8] = {0x31504e5349454445LL, s.len, s.mtp_len, int64_t(s.rec.n), int64_t(s.conv.n),
                            int64_t(s.k.n), int64_t(s.mk.n), c->m->cfg.D};
    std::fwrite(hdr, sizeof hdr, 1, f);
    std::fwrite(s.rec.p, 4, s.rec.n, f);
    std::fwrite(s.conv.p, 4, s.conv.n, f);
    std::fwrite(s.k.p, 2, s.k.n, f);
    std::fwrite(s.v.p, 2, s.v.n, f);
    if (s.mk.n) { std::fwrite(s.mk.p, 2, s.mk.n, f); std::fwrite(s.mv.p, 2, s.mv.n, f); }
    std::fclose(f);
    return 0;
}

extern "C" int ei_snap_read(ei_ctx* c, const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return -1;
    int64_t hdr[8];
    if (std::fread(hdr, sizeof hdr, 1, f) != 1 || hdr[0] != 0x31504e5349454445LL || hdr[7] != c->m->cfg.D ||
        size_t(hdr[3]) != c->rec_floats() || size_t(hdr[4]) != c->conv_floats()) {
        std::fclose(f);
        return -2;
    }
    auto s = std::make_unique<Snap>();
    s->with_kv = true;
    s->len = int(hdr[1]);
    s->mtp_len = int(hdr[2]);
    s->rec.resize(hdr[3]);
    s->conv.resize(hdr[4]);
    s->k.resize(std::max<int64_t>(1, hdr[5]));
    s->v.resize(std::max<int64_t>(1, hdr[5]));
    bool ok = std::fread(s->rec.p, 4, hdr[3], f) == size_t(hdr[3]) && std::fread(s->conv.p, 4, hdr[4], f) == size_t(hdr[4]) &&
              std::fread(s->k.p, 2, hdr[5], f) == size_t(hdr[5]) && std::fread(s->v.p, 2, hdr[5], f) == size_t(hdr[5]);
    if (ok && hdr[6] > 0) {
        s->mk.resize(hdr[6]);
        s->mv.resize(hdr[6]);
        ok = std::fread(s->mk.p, 2, hdr[6], f) == size_t(hdr[6]) && std::fread(s->mv.p, 2, hdr[6], f) == size_t(hdr[6]);
    }
    std::fclose(f);
    if (!ok || s->len > c->n_ctx) return -3;
    const int id = c->next_snap++;
    c->snaps[id] = std::move(s);
    return id;
}

// ============================================================== MTP drafting

extern "C" void ei_mtp_truncate(ei_ctx* c, int slot, int len) {
    if (slot < 0 || slot >= c->n_slots) return;
    c->slots[slot].mtp_len = std::max(0, std::min(len, c->slots[slot].mtp_len));
}

extern "C" int ei_mtp_len(ei_ctx* c, int slot) { return (slot >= 0 && slot < c->n_slots) ? c->slots[slot].mtp_len : -1; }

extern "C" int32_t ei_mtp_forward(ei_ctx* c, int slot, int pos0, int n, const int32_t* tokens,
                                  const float* hidden) {
    ei_model* m = c->m;
    if (!m->mtp.present) return -1;
    if (slot < 0 || slot >= c->n_slots || n <= 0 || n > c->max_batch) return -2;
    Slot& S = c->slots[slot];
    if (pos0 > S.mtp_len || pos0 + n > c->n_ctx) return -3;
    if (!hidden && n != 1) return -4;
    Timer tm(c, OP_MTP);
    const Config& cf = m->cfg;
    const int D = cf.D;
    const MtpW& T = m->mtp;

    // inputs: [norm(embed(tok)), norm(hidden)] -> fc -> x
    std::vector<float> hsrc;
    if (!hidden) hsrc.assign(c->mtp_hid.p, c->mtp_hid.p + D);
    for (int i = 0; i < n; ++i) {
        float* cat = c->cat.p + size_t(i) * 2 * D;
        dequant_row(m->embed.dtype, m->embed.data, m->embed.rows, m->embed.cols, tokens[i], c->y.p);
        rmsnorm(c->y.p, T.norm_emb, cat, D, cf.eps);
        const float* h = hidden ? hidden + size_t(i) * D : hsrc.data();
        rmsnorm(h, T.norm_hid, cat + D, D, cf.eps);
    }
    // the MTP layer reuses the main work buffers; toks describe its positions
    std::vector<Tok> saved_toks;
    saved_toks.swap(c->toks);
    std::vector<int> st0 = c->seg_t0, st1 = c->seg_t1, ssl = c->seg_slot, sfl = c->seg_flags;
    c->toks.resize(n);
    for (int i = 0; i < n; ++i) c->toks[i] = {slot, pos0 + i, 0};
    c->seg_t0 = {0};
    c->seg_t1 = {n};
    c->seg_slot = {slot};
    c->seg_flags = {0};
    qact(c, c->cat.p, 2 * D, n, 2 * D);
    mm(c, T.fc, c->cat.p, 2 * D, n, c->x.p, D);
    rope_tables(c, n);
    const KVSel kv{mtp_k, mtp_v};
    attention_block(c, T.layer, n, kv);
    ffn_block(c, T.layer, n);
    for (int i = 0; i < n; ++i) rmsnorm(c->x.p + size_t(i) * D, T.out_norm, c->mtp_hid.p + size_t(i) * D, D, cf.eps);
    // keep only the last row as "own output" for chained drafting
    if (n > 1) std::memmove(c->mtp_hid.p, c->mtp_hid.p + size_t(n - 1) * D, D * 4);
    S.mtp_len = pos0 + n;

    // draft logits for the last row only
    const Tensor& head = m->draft_head.valid() ? m->draft_head : m->lm_head;
    qact(c, c->mtp_hid.p, D, 1, D);
    float* lg = c->samp.p;  // scratch (V floats)
    matmul(*m->pool, head, &c->qa, c->mtp_hid.p, D, 1, lg, int(head.rows));
    const int32_t best = ei_argmax(lg, int(head.rows));
    c->toks.swap(saved_toks);
    c->seg_t0 = st0; c->seg_t1 = st1; c->seg_slot = ssl; c->seg_flags = sfl;
    return m->draft_head.valid() ? m->draft_ids[best] : best;
}

// ============================================================== stats

extern "C" const char* ei_ctx_stats(ei_ctx* c) {
    std::ostringstream js;
    js << "{\"forwards\":" << c->n_forward << ",\"tokens\":" << c->n_tokens << ",\"ops\":{";
    for (int i = 0; i < OP_N; ++i)
        js << (i ? "," : "") << "\"" << op_names[i] << "\":{\"s\":" << c->op_t[i] << ",\"n\":" << c->op_n[i] << "}";
    js << "}}";
    c->stats_json = js.str();
    return c->stats_json.c_str();
}

extern "C" void ei_calib(ei_ctx* c, int on) {
    c->calib = on != 0;
    if (on) { c->calib_sum.clear(); c->calib_n.clear(); }
}

extern "C" int ei_calib_dump(ei_ctx* c, const char* path) {
    FILE* f = std::fopen(path, "w");
    if (!f) return -1;
    for (auto& kv : c->calib_sum) {
        const int64_t n = c->calib_n[kv.first];
        std::fprintf(f, "%s %zu %lld", kv.first.c_str(), kv.second.size(), (long long)n);
        for (double v : kv.second) std::fprintf(f, " %.6g", v / std::max<int64_t>(1, n));
        std::fprintf(f, "\n");
    }
    std::fclose(f);
    return int(c->calib_sum.size());
}

extern "C" void ei_ctx_reset_stats(ei_ctx* c) {
    std::fill(c->op_t, c->op_t + OP_N, 0.0);
    std::fill(c->op_n, c->op_n + OP_N, 0);
    c->n_forward = c->n_tokens = 0;
}
