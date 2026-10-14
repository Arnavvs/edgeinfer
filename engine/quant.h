// Weight formats and the matmul kernels.
//
// Quantized weights are stored in tiles of 4 output rows. Inside a tile the
// 32-value blocks of the 4 rows are interleaved, so one sequential stream
// feeds a 4-row microkernel and every 64-byte cache line is fully used:
//
//   Q4 tile: qs[nb][4][16]  (two 4-bit values per byte, stored as q+8 in 0..15;
//                            byte j holds element j (low) and j+16 (high))
//            d [nb][4]      (fp16 scale per row-block)          -> 4.5 bits/weight
//   Q8 tile: qs[nb][4][32]  (int8)
//            d [nb][4]      (fp16)                              -> 8.5 bits/weight
//   Q4K tile (cols % 256 == 0): asymmetric, w = d*sc*q - dm*mn
//            qs[nb][4][16] (unsigned nibbles), sc[nb][4] (uint8 per 32-value
//            sub-block), mn[ns][4][8] (uint8, per row), d[ns][4], dm[ns][4]
//            (fp16 per 256)                                       -> 4.63 bits/weight
//            Q4K runs against activations quantized per 256 (QAct q2/d2/bs2):
//            the integer sub-block scales go into vpmaddwd and the 8 sub-blocks
//            accumulate in int32, one int->float conversion per super-block.
//   Q6 tile: qs[nb][4][24] (q+32 in 0..63: 16 bytes of low nibbles as Q4, 8 bytes
//            of high 2-bit pairs), d[nb][4] (fp16)                -> 6.5 bits/weight
//
// Activations are quantized on the fly to int8 per 32-block (QAct), so the
// inner loop is integer multiply-adds (vpmaddubsw/vpmaddwd). For Q4 the
// weights stay unsigned (q+8), which avoids the sign trick on every pair; the
// constant offset is removed with the per-block activation sums.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "common.h"
#include "threadpool.h"

namespace ei {

enum DType : int { F32 = 0, F16 = 1, BF16 = 2, Q8 = 3, Q4 = 4, Q4K = 5, Q6 = 6 };
constexpr int QK = 32;    // values per block
constexpr int QKK = 256;  // values per Q4K super-block
constexpr int TR = 4;   // rows per tile

const char* dtype_name(int dtype);
int dtype_from_name(const char* s);
bool dtype_is_tiled(int dtype);
int64_t tile_bytes(int dtype, int64_t cols);
int64_t tensor_bytes(int dtype, int64_t rows, int64_t cols);

void quantize(int dtype, const float* src, int64_t rows, int64_t cols, void* dst,
              const float* importance, int n_threads);
void dequant_row(int dtype, const void* data, int64_t rows, int64_t cols, int64_t row, float* out);
// copy row sr of a tiled tensor into row dr of another tiled tensor of the same dtype/cols
void copy_tiled_row(int dtype, int64_t cols, const uint8_t* src, int64_t sr, uint8_t* dst, int64_t dr);

// Activations, quantized two ways from the same floats:
//   per 32-block (q, d, s = d*sum(q))           for Q8 / Q6 / Q4 weights
//   per 256-block (q2, d2, bs2 = per-32 sums)   for Q4K weights (when k % 256 == 0)
struct QAct {
    Buf<int8_t> q, q2;
    Buf<float> d, s, d2;
    Buf<int16_t> bs2;
    int m = 0, k = 0, nb = 0, ns = 0;
    bool has256 = false;
    void reserve(int max_m, int max_k) {
        q.resize(size_t(max_m) * max_k);
        d.resize(size_t(max_m) * (max_k / QK));
        s.resize(size_t(max_m) * (max_k / QK));
        q2.resize(size_t(max_m) * max_k);
        d2.resize(size_t(max_m) * (max_k / QKK + 1));
        bs2.resize(size_t(max_m) * (max_k / QK));
    }
    const int8_t* qrow(int i) const { return q.p + size_t(i) * k; }
    const float* drow(int i) const { return d.p + size_t(i) * nb; }
    const float* srow(int i) const { return s.p + size_t(i) * nb; }
    const int8_t* q2row(int i) const { return q2.p + size_t(i) * k; }
    const float* d2row(int i) const { return d2.p + size_t(i) * ns; }
    const int16_t* bsrow(int i) const { return bs2.p + size_t(i) * nb; }
};

// Quantize rows [r0, r1) of x (row stride ldx) into qa; also the per-256 form
// when qa.has256 (set by the caller when k % 256 == 0).
void quantize_act_rows(const float* x, int ldx, int k, QAct& qa, int r0, int r1);

struct Tensor {
    std::string name;
    int dtype = F32;
    int64_t rows = 0, cols = 0;
    const uint8_t* data = nullptr;
    int64_t nbytes = 0;
    bool valid() const { return data != nullptr; }
};

// y[i][r] (row stride ldy) = sum_k W[r][k] * x[i][k], for i < m and every row
// r of W. Quantized W reads qa; F32/F16/BF16 W reads xf (row stride ldx).
// Parallel over row tiles.
void matmul(ThreadPool& pool, const Tensor& w, const QAct* qa, const float* xf, int ldx, int m,
            float* y, int ldy);

// Same, for a subset of rows only (used by the reduced draft vocabulary).
void matmul_rows(const Tensor& w, const QAct* qa, const float* xf, int ldx, int m, float* y, int ldy,
                 int64_t r0, int64_t r1);

}  // namespace ei
