#include "quant.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

namespace ei {

const char* dtype_name(int d) {
    switch (d) {
        case F32: return "f32";
        case F16: return "f16";
        case BF16: return "bf16";
        case Q8: return "q8";
        case Q4: return "q4";
        case Q4K: return "q4k";
    }
    return "?";
}

int dtype_from_name(const char* s) {
    for (int d = F32; d <= Q4K; ++d)
        if (std::strcmp(s, dtype_name(d)) == 0) return d;
    return -1;
}

bool dtype_is_tiled(int d) { return d == Q8 || d == Q4 || d == Q4K; }

static inline int64_t q_bytes_per_block_row(int d) { return d == Q8 ? 32 : 16; }

int64_t tile_bytes(int d, int64_t cols) {
    int64_t nb = cols / QK;
    if (d == Q4) return nb * (TR * 16 + TR * 2);
    if (d == Q8) return nb * (TR * 32 + TR * 2);
    if (d == Q4K) return (nb * (TR * 16 + TR * 2) + (nb / 8) * TR * 4 + 63) / 64 * 64;
    return 0;
}

int64_t tensor_bytes(int d, int64_t rows, int64_t cols) {
    switch (d) {
        case F32: return rows * cols * 4;
        case F16:
        case BF16: return rows * cols * 2;
        case Q4K:
            if (cols % QKK) return -1;
            return (rows + TR - 1) / TR * tile_bytes(d, cols);
        case Q8:
        case Q4:
            if (cols % QK) return -1;
            return (rows + TR - 1) / TR * tile_bytes(d, cols);
    }
    return -1;
}

// ---------------------------------------------------------------- quantize

static void quant_block_q8(const float* x, int8_t* q, uint16_t* d) {
    float amax = 0;
    for (int i = 0; i < QK; ++i) amax = std::max(amax, std::fabs(x[i]));
    uint16_t h = f32_to_fp16(amax / 127.0f);
    float df = fp16_to_f32(h);
    float id = df > 0 ? 1.0f / df : 0.0f;
    for (int i = 0; i < QK; ++i) {
        int v = int(std::lrintf(x[i] * id));
        q[i] = int8_t(std::clamp(v, -127, 127));
    }
    *d = h;
}

// Symmetric 4-bit with a searched scale. The scale minimises the weighted
// squared error sum_i w_i (x_i - d*q_i)^2; w_i = importance_i * sqrt(var + x_i^2)
// (importance from calibration activations when available, AWQ/imatrix style).
static void quant_block_q4(const float* x, const float* imp, uint8_t* qs, uint16_t* d) {
    float w[QK];
    float sumx2 = 0;
    for (int i = 0; i < QK; ++i) sumx2 += x[i] * x[i];
    const float sigma2 = sumx2 / QK;
    float amax = 0, maxv = 0;
    for (int i = 0; i < QK; ++i) {
        w[i] = (imp ? imp[i] : 1.0f) * std::sqrt(sigma2 + x[i] * x[i]);
        if (std::fabs(x[i]) > amax) { amax = std::fabs(x[i]); maxv = x[i]; }
    }
    int8_t L[QK], Lbest[QK];
    if (amax < 1e-30f) {
        std::memset(qs, 0x88, 16);
        *d = 0;
        return;
    }
    float best = -1, best_scale = 0;
    for (int is = -9; is <= 9; ++is) {
        const float iscale = -(8.0f + 0.1f * is) / maxv;
        float sumlx = 0, suml2 = 0;
        for (int i = 0; i < QK; ++i) {
            int l = int(std::lrintf(iscale * x[i]));
            l = std::clamp(l, -8, 7);
            L[i] = int8_t(l);
            sumlx += w[i] * x[i] * l;
            suml2 += w[i] * l * l;
        }
        if (suml2 > 0 && sumlx * sumlx > best * suml2) {
            best = sumlx * sumlx / suml2;
            best_scale = sumlx / suml2;
            std::memcpy(Lbest, L, QK);
        }
    }
    // Requantize once against the fp16-rounded scale actually stored.
    uint16_t h = f32_to_fp16(best_scale);
    float df = fp16_to_f32(h);
    if (df != 0) {
        float id = 1.0f / df;
        for (int i = 0; i < QK; ++i) Lbest[i] = int8_t(std::clamp(int(std::lrintf(x[i] * id)), -8, 7));
    }
    for (int j = 0; j < 16; ++j)
        qs[j] = uint8_t((Lbest[j] + 8) | ((Lbest[j + 16] + 8) << 4));
    *d = h;
}

// Asymmetric 4-bit sub-block: x ~= scale * q - mn with q in 0..15, scale, mn >= 0.
// Weighted least squares over a grid of candidate scales (llama.cpp's
// make_qkx2 idea), with the min constrained so it can be stored unsigned.
static void q4k_fit(const float* x, const float* w, float& scale, float& mn) {
    float xmin = x[0], xmax = x[0], sw = 0, sx = 0;
    for (int i = 0; i < QK; ++i) {
        xmin = std::min(xmin, x[i]);
        xmax = std::max(xmax, x[i]);
        sw += w[i];
        sx += w[i] * x[i];
    }
    if (xmin > 0) xmin = 0;
    if (xmax <= xmin) { scale = 0; mn = -xmin; return; }
    float best_s = (xmax - xmin) / 15.0f, best_m = xmin, best_err = 0;
    for (int i = 0; i < QK; ++i) {
        int l = std::clamp(int(std::lrintf((x[i] - xmin) / best_s)), 0, 15);
        float d = best_s * l + xmin - x[i];
        best_err += w[i] * d * d;
    }
    for (int is = 0; is <= 20; ++is) {
        const float iscale = (-1.0f + 0.1f * is + 15.0f) / (xmax - xmin);
        float sl = 0, sl2 = 0, sxl = 0;
        uint8_t L[QK];
        for (int i = 0; i < QK; ++i) {
            int l = std::clamp(int(std::lrintf(iscale * (x[i] - xmin))), 0, 15);
            L[i] = uint8_t(l);
            sl += w[i] * l;
            sl2 += w[i] * l * l;
            sxl += w[i] * x[i] * l;
        }
        const float D = sw * sl2 - sl * sl;
        if (D <= 0) continue;
        float s = (sw * sxl - sx * sl) / D;
        float m = (sl2 * sx - sl * sxl) / D;
        if (m > 0) { m = 0; s = sl2 > 0 ? sxl / sl2 : 0; }
        if (s <= 0) continue;
        float err = 0;
        for (int i = 0; i < QK; ++i) { float d = s * L[i] + m - x[i]; err += w[i] * d * d; }
        if (err < best_err) { best_err = err; best_s = s; best_m = m; }
    }
    scale = best_s;
    mn = -best_m;
}

static void quantize_q4k_row(const float* x, const float* imp, int64_t cols, uint8_t* tile, int r) {
    const int64_t nb = cols / QK, ns = cols / QKK;
    uint8_t* qs = tile;
    uint8_t* sc = tile + nb * TR * 16;
    uint8_t* mn = sc + nb * TR;
    uint16_t* dd = reinterpret_cast<uint16_t*>(mn + nb * TR);
    uint16_t* dm = dd + ns * TR;
    for (int64_t s = 0; s < ns; ++s) {
        float scl[8], mns[8], w[QK];
        for (int j = 0; j < 8; ++j) {
            const float* xb = x + (s * 8 + j) * QK;
            const float* ib = imp ? imp + (s * 8 + j) * QK : nullptr;
            float s2 = 0;
            for (int i = 0; i < QK; ++i) s2 += xb[i] * xb[i];
            const float sig = s2 / QK;
            for (int i = 0; i < QK; ++i) w[i] = (ib ? ib[i] : 1.0f) * std::sqrt(sig + xb[i] * xb[i]);
            q4k_fit(xb, w, scl[j], mns[j]);
        }
        float smax = 0, mmax = 0;
        for (int j = 0; j < 8; ++j) { smax = std::max(smax, scl[j]); mmax = std::max(mmax, mns[j]); }
        const uint16_t hd = f32_to_fp16(smax / 255.0f), hm = f32_to_fp16(mmax / 255.0f);
        const float fd = fp16_to_f32(hd), fm = fp16_to_f32(hm);
        dd[s * TR + r] = hd;
        dm[s * TR + r] = hm;
        for (int j = 0; j < 8; ++j) {
            const int64_t b = s * 8 + j;
            const int isc = fd > 0 ? std::clamp(int(std::lrintf(scl[j] / fd)), 0, 255) : 0;
            const int imn = fm > 0 ? std::clamp(int(std::lrintf(mns[j] / fm)), 0, 255) : 0;
            sc[b * TR + r] = uint8_t(isc);
            mn[b * TR + r] = uint8_t(imn);
            const float S = fd * isc, M = fm * imn;
            const float* xb = x + b * QK;
            uint8_t L[QK];
            for (int i = 0; i < QK; ++i)
                L[i] = S > 0 ? uint8_t(std::clamp(int(std::lrintf((xb[i] + M) / S)), 0, 15)) : 0;
            uint8_t* qq = qs + (b * TR + r) * 16;
            for (int i = 0; i < 16; ++i) qq[i] = uint8_t(L[i] | (L[i + 16] << 4));
        }
    }
}

static void quantize_tiles(int dtype, const float* src, int64_t rows, int64_t cols, uint8_t* dst,
                           const float* imp, int64_t t0, int64_t t1) {
    const int64_t nb = cols / QK, tb = tile_bytes(dtype, cols), qb = q_bytes_per_block_row(dtype);
    std::vector<float> zero(cols, 0.0f);
    for (int64_t t = t0; t < t1; ++t) {
        uint8_t* tile = dst + t * tb;
        if (dtype == Q4K) {
            for (int r = 0; r < TR; ++r) {
                const int64_t row = t * TR + r;
                quantize_q4k_row(row < rows ? src + row * cols : zero.data(), imp, cols, tile, r);
            }
            continue;
        }
        uint8_t* qs = tile;
        uint16_t* dd = reinterpret_cast<uint16_t*>(tile + nb * TR * qb);
        for (int r = 0; r < TR; ++r) {
            int64_t row = t * TR + r;
            const float* x = row < rows ? src + row * cols : zero.data();
            for (int64_t b = 0; b < nb; ++b) {
                uint8_t* q = qs + (b * TR + r) * qb;
                uint16_t* d = dd + b * TR + r;
                if (dtype == Q8)
                    quant_block_q8(x + b * QK, reinterpret_cast<int8_t*>(q), d);
                else
                    quant_block_q4(x + b * QK, imp ? imp + b * QK : nullptr, q, d);
            }
        }
    }
}

void quantize(int dtype, const float* src, int64_t rows, int64_t cols, void* dst, const float* imp,
              int n_threads) {
    if (dtype == F32) { std::memcpy(dst, src, rows * cols * 4); return; }
    if (dtype == F16 || dtype == BF16) {
        uint16_t* o = static_cast<uint16_t*>(dst);
        for (int64_t i = 0; i < rows * cols; ++i) {
            if (dtype == F16) o[i] = f32_to_fp16(src[i]);
            else { uint32_t u; std::memcpy(&u, &src[i], 4); u += 0x7fff + ((u >> 16) & 1); o[i] = uint16_t(u >> 16); }
        }
        return;
    }
    const int64_t tiles = (rows + TR - 1) / TR;
    n_threads = std::max(1, n_threads);
    std::vector<std::thread> th;
    for (int i = 0; i < n_threads; ++i) {
        int64_t a = tiles * i / n_threads, b = tiles * (i + 1) / n_threads;
        th.emplace_back(quantize_tiles, dtype, src, rows, cols, static_cast<uint8_t*>(dst), imp, a, b);
    }
    for (auto& t : th) t.join();
}

void dequant_row(int dtype, const void* data, int64_t rows, int64_t cols, int64_t row, float* out) {
    (void)rows;
    const uint8_t* p = static_cast<const uint8_t*>(data);
    switch (dtype) {
        case F32: std::memcpy(out, p + row * cols * 4, cols * 4); return;
        case F16: {
            const uint16_t* h = reinterpret_cast<const uint16_t*>(p) + row * cols;
            for (int64_t i = 0; i < cols; ++i) out[i] = fp16_to_f32(h[i]);
            return;
        }
        case BF16: {
            const uint16_t* h = reinterpret_cast<const uint16_t*>(p) + row * cols;
            for (int64_t i = 0; i < cols; ++i) out[i] = bf16_to_f32(h[i]);
            return;
        }
    }
    const int64_t nb = cols / QK, qb = q_bytes_per_block_row(dtype);
    const uint8_t* tile = p + (row / TR) * tile_bytes(dtype, cols);
    const int r = int(row % TR);
    if (dtype == Q4K) {
        const uint8_t* sc = tile + nb * TR * 16;
        const uint8_t* mn = sc + nb * TR;
        const uint16_t* dd = reinterpret_cast<const uint16_t*>(mn + nb * TR);
        const uint16_t* dm = dd + (nb / 8) * TR;
        for (int64_t b = 0; b < nb; ++b) {
            const float S = fp16_to_f32(dd[(b / 8) * TR + r]) * sc[b * TR + r];
            const float M = fp16_to_f32(dm[(b / 8) * TR + r]) * mn[b * TR + r];
            const uint8_t* qq = tile + (b * TR + r) * 16;
            float* o = out + b * QK;
            for (int j = 0; j < 16; ++j) { o[j] = S * (qq[j] & 0x0F) - M; o[j + 16] = S * (qq[j] >> 4) - M; }
        }
        return;
    }
    const uint16_t* dd = reinterpret_cast<const uint16_t*>(tile + nb * TR * qb);
    for (int64_t b = 0; b < nb; ++b) {
        const float d = fp16_to_f32(dd[b * TR + r]);
        const uint8_t* q = tile + (b * TR + r) * qb;
        float* o = out + b * QK;
        if (dtype == Q8) {
            for (int j = 0; j < QK; ++j) o[j] = d * float(int8_t(q[j]));
        } else {
            for (int j = 0; j < 16; ++j) {
                o[j] = d * float(int(q[j] & 0x0F) - 8);
                o[j + 16] = d * float(int(q[j] >> 4) - 8);
            }
        }
    }
}

void copy_tiled_row(int dtype, int64_t cols, const uint8_t* src, int64_t sr, uint8_t* dst, int64_t dr) {
    const int64_t nb = cols / QK, tb = tile_bytes(dtype, cols), qb = q_bytes_per_block_row(dtype);
    const uint8_t* st = src + (sr / TR) * tb;
    uint8_t* dt = dst + (dr / TR) * tb;
    const int a = int(sr % TR), b = int(dr % TR);
    for (int64_t k = 0; k < nb; ++k) std::memcpy(dt + (k * TR + b) * qb, st + (k * TR + a) * qb, qb);
    if (dtype == Q4K) {
        const int64_t o1 = nb * TR * 16, o2 = o1 + nb * TR, o3 = o2 + nb * TR;
        for (int64_t k = 0; k < nb; ++k) {
            dt[o1 + k * TR + b] = st[o1 + k * TR + a];
            dt[o2 + k * TR + b] = st[o2 + k * TR + a];
        }
        const uint16_t* sh = reinterpret_cast<const uint16_t*>(st + o3);
        uint16_t* dh = reinterpret_cast<uint16_t*>(dt + o3);
        for (int64_t s = 0; s < 2 * (nb / 8); ++s) dh[s * TR + b] = sh[s * TR + a];
    } else {
        const uint16_t* sh = reinterpret_cast<const uint16_t*>(st + nb * TR * qb);
        uint16_t* dh = reinterpret_cast<uint16_t*>(dt + nb * TR * qb);
        for (int64_t k = 0; k < nb; ++k) dh[k * TR + b] = sh[k * TR + a];
    }
}

// ---------------------------------------------------------------- activations

void quantize_act_rows(const float* x, int ldx, int k, QAct& qa, int r0, int r1) {
    const int nb = k / QK;
    const __m256 sign = _mm256_set1_ps(-0.0f);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (int i = r0; i < r1; ++i) {
        const float* xi = x + size_t(i) * ldx;
        int8_t* q = qa.q.p + size_t(i) * k;
        float* d = qa.d.p + size_t(i) * nb;
        float* s = qa.s.p + size_t(i) * nb;
        for (int b = 0; b < nb; ++b) {
            __m256 v0 = _mm256_loadu_ps(xi + b * 32 + 0);
            __m256 v1 = _mm256_loadu_ps(xi + b * 32 + 8);
            __m256 v2 = _mm256_loadu_ps(xi + b * 32 + 16);
            __m256 v3 = _mm256_loadu_ps(xi + b * 32 + 24);
            __m256 m = _mm256_max_ps(_mm256_max_ps(_mm256_andnot_ps(sign, v0), _mm256_andnot_ps(sign, v1)),
                                     _mm256_max_ps(_mm256_andnot_ps(sign, v2), _mm256_andnot_ps(sign, v3)));
            __m128 m4 = _mm_max_ps(_mm256_extractf128_ps(m, 1), _mm256_castps256_ps128(m));
            m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
            m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
            const float amax = _mm_cvtss_f32(m4);
            const float dd = amax / 127.0f;
            const float id = amax > 0 ? 127.0f / amax : 0.0f;
            const __m256 mul = _mm256_set1_ps(id);
            __m256i i0 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v0, mul), _MM_FROUND_TO_NEAREST_INT));
            __m256i i1 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v1, mul), _MM_FROUND_TO_NEAREST_INT));
            __m256i i2 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v2, mul), _MM_FROUND_TO_NEAREST_INT));
            __m256i i3 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v3, mul), _MM_FROUND_TO_NEAREST_INT));
            __m256i sum = _mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3));
            __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
            s4 = _mm_hadd_epi32(s4, s4);
            s4 = _mm_hadd_epi32(s4, s4);
            const int isum = _mm_cvtsi128_si32(s4);
            i0 = _mm256_packs_epi32(i0, i1);
            i2 = _mm256_packs_epi32(i2, i3);
            i0 = _mm256_packs_epi16(i0, i2);
            i0 = _mm256_permutevar8x32_epi32(i0, perm);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(q + b * 32), i0);
            d[b] = dd;
            s[b] = dd * float(isum);
        }
    }
}

// ---------------------------------------------------------------- kernels

static inline __m256i expand_q4(const uint8_t* p, __m256i m4) {
    __m256i v = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
    __m256i hi = _mm256_srli_epi16(v, 4);
    return _mm256_and_si256(_mm256_blend_epi32(v, hi, 0xF0), m4);
}

// One 4-row Q4 / Q4K tile against NT tokens starting at m0.
// Q4:  w = d * (q - 8)          -> y = sum d*da*dot(q,a) - 8 * sum d*sa
// Q4K: w = d*sc * q - dm*mn     -> y = sum d*sc*da*dot(q,a) - sum dm*mn*sa
template <int NT, bool ASYM>
static inline void q4_tile(const uint8_t* tile, int nb, const QAct& qa, int m0, float* y, int ldy,
                           int64_t row0, int nvalid) {
    const uint16_t* dd = reinterpret_cast<const uint16_t*>(tile + size_t(nb) * 64);
    const uint8_t* ksc = tile + size_t(nb) * 64;
    const uint8_t* kmn = ksc + size_t(nb) * 4;
    const uint16_t* kd = reinterpret_cast<const uint16_t*>(kmn + size_t(nb) * 4);
    const uint16_t* kdm = kd + size_t(nb / 8) * 4;
    __m128 sd = _mm_setzero_ps(), sdm = _mm_setzero_ps();
    const int8_t* aq[NT];
    const float* ad[NT];
    const float* as[NT];
    for (int t = 0; t < NT; ++t) { aq[t] = qa.qrow(m0 + t); ad[t] = qa.drow(m0 + t); as[t] = qa.srow(m0 + t); }
    __m256 acc[TR][NT];
    __m128 corr[NT];
    for (int t = 0; t < NT; ++t) {
        corr[t] = _mm_setzero_ps();
        for (int r = 0; r < TR; ++r) acc[r][t] = _mm256_setzero_ps();
    }
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    alignas(32) float sc[NT][4];
    for (int b = 0; b < nb; ++b) {
        __m128 dw, mw;
        if (ASYM) {
            if ((b & 7) == 0) {
                sd = _mm_cvtph_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(kd + (b / 8) * 4)));
                sdm = _mm_cvtph_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(kdm + (b / 8) * 4)));
            }
            int s4, m4i;
            std::memcpy(&s4, ksc + b * 4, 4);
            std::memcpy(&m4i, kmn + b * 4, 4);
            dw = _mm_mul_ps(sd, _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(s4))));
            mw = _mm_mul_ps(sdm, _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(m4i))));
        } else {
            dw = _mm_cvtph_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(dd + b * 4)));
            mw = dw;
        }
        __m256i a[NT];
        for (int t = 0; t < NT; ++t) {
            _mm_store_ps(sc[t], _mm_mul_ps(dw, _mm_set1_ps(ad[t][b])));
            corr[t] = _mm_fmadd_ps(mw, _mm_set1_ps(as[t][b]), corr[t]);
            a[t] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(aq[t] + b * 32));
        }
        const uint8_t* qb = tile + size_t(b) * 64;
        for (int r = 0; r < TR; ++r) {
            const __m256i w = expand_q4(qb + r * 16, m4);
            for (int t = 0; t < NT; ++t) {
                __m256i p = _mm256_maddubs_epi16(w, a[t]);
                p = _mm256_madd_epi16(p, ones);
                acc[r][t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_broadcast_ss(&sc[t][r]), acc[r][t]);
            }
        }
    }
    for (int t = 0; t < NT; ++t) {
        alignas(16) float c[4];
        _mm_store_ps(c, corr[t]);
        float* yo = y + size_t(m0 + t) * ldy + row0;
        for (int r = 0; r < nvalid; ++r) yo[r] = hsum8(acc[r][t]) - (ASYM ? 1.0f : 8.0f) * c[r];
    }
}

template <int NT>
static inline void q8_tile(const uint8_t* tile, int nb, const QAct& qa, int m0, float* y, int ldy,
                           int64_t row0, int nvalid) {
    const uint16_t* dd = reinterpret_cast<const uint16_t*>(tile + size_t(nb) * 128);
    const int8_t* aq[NT];
    const float* ad[NT];
    for (int t = 0; t < NT; ++t) { aq[t] = qa.qrow(m0 + t); ad[t] = qa.drow(m0 + t); }
    __m256 acc[TR][NT];
    for (int r = 0; r < TR; ++r)
        for (int t = 0; t < NT; ++t) acc[r][t] = _mm256_setzero_ps();
    const __m256i ones = _mm256_set1_epi16(1);
    alignas(32) float sc[NT][4];
    for (int b = 0; b < nb; ++b) {
        const __m128 dw = _mm_cvtph_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(dd + b * 4)));
        __m256i a[NT];
        for (int t = 0; t < NT; ++t) {
            _mm_store_ps(sc[t], _mm_mul_ps(dw, _mm_set1_ps(ad[t][b])));
            a[t] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(aq[t] + b * 32));
        }
        const uint8_t* qb = tile + size_t(b) * 128;
        for (int r = 0; r < TR; ++r) {
            const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qb + r * 32));
            const __m256i aw = _mm256_sign_epi8(w, w);
            for (int t = 0; t < NT; ++t) {
                __m256i p = _mm256_maddubs_epi16(aw, _mm256_sign_epi8(a[t], w));
                p = _mm256_madd_epi16(p, ones);
                acc[r][t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_broadcast_ss(&sc[t][r]), acc[r][t]);
            }
        }
    }
    for (int t = 0; t < NT; ++t) {
        float* yo = y + size_t(m0 + t) * ldy + row0;
        for (int r = 0; r < nvalid; ++r) yo[r] = hsum8(acc[r][t]);
    }
}

static inline __m256 load8(int dtype, const void* p, int64_t i) {
    if (dtype == F32) return _mm256_loadu_ps(static_cast<const float*>(p) + i);
    const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const uint16_t*>(p) + i));
    if (dtype == F16) return _mm256_cvtph_ps(h);
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
}

static void dense_rows(const Tensor& w, const float* x, int ldx, int m, float* y, int ldy, int64_t r0, int64_t r1) {
    const int64_t k = w.cols;
    const int esz = w.dtype == F32 ? 4 : 2;
    for (int64_t r = r0; r < r1; ++r) {
        const uint8_t* wr = w.data + r * k * esz;
        for (int i = 0; i < m; ++i) {
            const float* xi = x + size_t(i) * ldx;
            __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
            int64_t j = 0;
            for (; j + 16 <= k; j += 16) {
                acc0 = _mm256_fmadd_ps(load8(w.dtype, wr, j), _mm256_loadu_ps(xi + j), acc0);
                acc1 = _mm256_fmadd_ps(load8(w.dtype, wr, j + 8), _mm256_loadu_ps(xi + j + 8), acc1);
            }
            for (; j + 8 <= k; j += 8) acc0 = _mm256_fmadd_ps(load8(w.dtype, wr, j), _mm256_loadu_ps(xi + j), acc0);
            float s = hsum8(_mm256_add_ps(acc0, acc1));
            for (; j < k; ++j) {
                float wv = w.dtype == F32 ? reinterpret_cast<const float*>(wr)[j]
                         : w.dtype == F16 ? fp16_to_f32(reinterpret_cast<const uint16_t*>(wr)[j])
                                          : bf16_to_f32(reinterpret_cast<const uint16_t*>(wr)[j]);
                s += wv * xi[j];
            }
            y[size_t(i) * ldy + r] = s;
        }
    }
}

// Rows [r0, r1) of a tiled tensor (r0 a multiple of TR). Token blocks keep the
// activations of a block hot in L1/L2 while the tiles of this range stream.
static void tiled_rows(const Tensor& w, const QAct& qa, int m, float* y, int ldy, int64_t r0, int64_t r1) {
    const int nb = int(w.cols / QK);
    const int64_t tb = tile_bytes(w.dtype, w.cols);
    const int mb = std::max(2, int((32 * 1024) / w.cols) & ~1);
    for (int ma = 0; ma < m; ma += mb) {
        const int me = std::min(m, ma + mb);
        for (int64_t row = r0; row < r1; row += TR) {
            const uint8_t* tile = w.data + (row / TR) * tb;
            const int nvalid = int(std::min<int64_t>(TR, w.rows - row));
            int i = ma;
            if (w.dtype == Q4) {
                for (; i + 2 <= me; i += 2) q4_tile<2, false>(tile, nb, qa, i, y, ldy, row, nvalid);
                if (i < me) q4_tile<1, false>(tile, nb, qa, i, y, ldy, row, nvalid);
            } else if (w.dtype == Q4K) {
                for (; i + 2 <= me; i += 2) q4_tile<2, true>(tile, nb, qa, i, y, ldy, row, nvalid);
                if (i < me) q4_tile<1, true>(tile, nb, qa, i, y, ldy, row, nvalid);
            } else {
                for (; i + 2 <= me; i += 2) q8_tile<2>(tile, nb, qa, i, y, ldy, row, nvalid);
                if (i < me) q8_tile<1>(tile, nb, qa, i, y, ldy, row, nvalid);
            }
        }
    }
}

void matmul_rows(const Tensor& w, const QAct* qa, const float* xf, int ldx, int m, float* y, int ldy,
                 int64_t r0, int64_t r1) {
    if (dtype_is_tiled(w.dtype)) tiled_rows(w, *qa, m, y, ldy, r0, r1);
    else dense_rows(w, xf, ldx, m, y, ldy, r0, r1);
}

struct MatmulJob {
    const Tensor* w;
    const QAct* qa;
    const float* xf;
    int ldx, m, ldy;
    float* y;
    int64_t rows_per_task;
};

void matmul(ThreadPool& pool, const Tensor& w, const QAct* qa, const float* xf, int ldx, int m, float* y,
            int ldy) {
    // Aim for ~8 tasks per thread for load balance (hyperthreads run unevenly),
    // but keep each task at least a few tiles.
    const int64_t rows = w.rows;
    const int threads = pool.size();
    int64_t per = (rows + threads * 8 - 1) / (threads * 8);
    per = std::max<int64_t>(per, 16);
    per = (per + TR - 1) / TR * TR;
    MatmulJob job{&w, qa, xf, ldx, m, ldy, y, per};
    const int tasks = int((rows + per - 1) / per);
    pool.run(tasks, [](void* p, int task, int) {
        auto* j = static_cast<MatmulJob*>(p);
        int64_t a = task * j->rows_per_task, b = std::min(j->w->rows, a + j->rows_per_task);
        matmul_rows(*j->w, j->qa, j->xf, j->ldx, j->m, j->y, j->ldy, a, b);
    }, &job);
}

}  // namespace ei
