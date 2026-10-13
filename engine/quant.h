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
//            qs[nb][4][16] (unsigned nibbles), sc[nb][4], mn[nb][4] (uint8 per
//            32-value sub-block), d[ns][4], dm[ns][4] (fp16 per 256) -> 4.63 bits/weight
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

enum DType : int { F32 = 0, F16 = 1, BF16 = 2, Q8 = 3, Q4 = 4, Q4K = 5 };
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

// Activations quantized per 32-block: q int8, d scale, s = d * sum(q).
struct QAct {
    Buf<int8_t> q;
    Buf<float> d, s;
    int m = 0, k = 0, nb = 0;
    void reserve(int max_m, int max_k) {
        q.resize(size_t(max_m) * max_k);
        d.resize(size_t(max_m) * (max_k / QK));
        s.resize(size_t(max_m) * (max_k / QK));
    }
    const int8_t* qrow(int i) const { return q.p + size_t(i) * k; }
    const float* drow(int i) const { return d.p + size_t(i) * nb; }
    const float* srow(int i) const { return s.p + size_t(i) * nb; }
};

// Quantize m rows of x (row stride ldx) into qa (single-threaded part).
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
