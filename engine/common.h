// Small shared helpers: aligned buffers, fp16/bf16 conversion, timers.
#pragma once

#include <immintrin.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ei {

inline void* aalloc(size_t bytes, size_t align = 64) {
    if (bytes == 0) bytes = align;
    bytes = (bytes + align - 1) / align * align;
    void* p = std::aligned_alloc(align, bytes);
    if (!p) throw std::runtime_error("out of memory allocating " + std::to_string(bytes) + " bytes");
    return p;
}

template <class T>
struct Buf {
    T* p = nullptr;
    size_t n = 0;
    Buf() = default;
    explicit Buf(size_t count) { resize(count); }
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    Buf(Buf&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; o.n = 0; }
    Buf& operator=(Buf&& o) noexcept {
        if (this != &o) { std::free(p); p = o.p; n = o.n; o.p = nullptr; o.n = 0; }
        return *this;
    }
    ~Buf() { std::free(p); }
    void resize(size_t count) {
        std::free(p);
        p = static_cast<T*>(aalloc(count * sizeof(T)));
        n = count;
    }
    void zero() { if (p) std::memset(p, 0, n * sizeof(T)); }
    T& operator[](size_t i) { return p[i]; }
    const T& operator[](size_t i) const { return p[i]; }
    size_t bytes() const { return n * sizeof(T); }
};

inline float fp16_to_f32(uint16_t h) { return _cvtsh_ss(h); }
inline uint16_t f32_to_fp16(float f) { return _cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT); }
inline float bf16_to_f32(uint16_t b) {
    uint32_t u = uint32_t(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// horizontal sum of 8 floats
inline float hsum8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(lo);
    __m128 s = _mm_add_ps(lo, sh);
    sh = _mm_movehl_ps(sh, s);
    s = _mm_add_ss(s, sh);
    return _mm_cvtss_f32(s);
}

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float softplus(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

// Fast vectorised exp for x <= 0 range use in softmax/silu (Cephes-style).
inline __m256 exp256(__m256 x) {
    const __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
    const __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);
    const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
    const __m256 c1 = _mm256_set1_ps(0.693359375f);
    const __m256 c2 = _mm256_set1_ps(-2.12194440e-4f);
    x = _mm256_min_ps(x, exp_hi);
    x = _mm256_max_ps(x, exp_lo);
    __m256 fx = _mm256_fmadd_ps(x, log2e, _mm256_set1_ps(0.5f));
    fx = _mm256_floor_ps(fx);
    x = _mm256_sub_ps(x, _mm256_mul_ps(fx, c1));
    x = _mm256_sub_ps(x, _mm256_mul_ps(fx, c2));
    __m256 z = _mm256_mul_ps(x, x);
    __m256 y = _mm256_set1_ps(1.9875691500E-4f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507E-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073E-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894E-2f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459E-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201E-1f));
    y = _mm256_fmadd_ps(y, z, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.0f));
    __m256i e = _mm256_cvttps_epi32(fx);
    e = _mm256_add_epi32(e, _mm256_set1_epi32(0x7f));
    e = _mm256_slli_epi32(e, 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(e));
}

inline __m256 silu256(__m256 x) {
    __m256 e = exp256(_mm256_sub_ps(_mm256_setzero_ps(), x));
    return _mm256_div_ps(x, _mm256_add_ps(_mm256_set1_ps(1.0f), e));
}

}  // namespace ei
