// Sampling: penalties -> temperature -> top-k -> top-p -> min-p, and the
// acceptance rule for speculative decoding with a deterministic (greedy) draft:
// accept the draft d with probability p(d); otherwise sample from p with d
// removed. That keeps the output distribution exactly p.
#include <algorithm>
#include <cfloat>
#include <queue>
#include <unordered_map>

#include "model.h"

using namespace ei;

namespace {

uint64_t splitmix(uint64_t& s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

float urand(uint64_t& s) { return float(splitmix(s) >> 40) * (1.0f / 16777216.0f); }

bool has_penalties(const ei_sampling* p) {
    return (p->repeat_penalty != 1.0f && p->repeat_penalty > 0) || p->presence_penalty != 0 || p->frequency_penalty != 0;
}

// Copies logits into scratch and applies the penalties. Returns the scratch.
const float* penalised(ei_ctx* c, const float* logits, const ei_sampling* p, const int32_t* hist, int nh) {
    const int V = ei_n_vocab(c->m);
    if (!has_penalties(p) || nh <= 0) return logits;
    float* l = c->samp.p;
    std::memcpy(l, logits, size_t(V) * 4);
    std::unordered_map<int32_t, int> cnt;
    for (int i = 0; i < nh; ++i)
        if (hist[i] >= 0 && hist[i] < V) cnt[hist[i]]++;
    for (auto& kv : cnt) {
        float& x = l[kv.first];
        if (p->repeat_penalty > 0 && p->repeat_penalty != 1.0f) x = x > 0 ? x / p->repeat_penalty : x * p->repeat_penalty;
        x -= p->presence_penalty + p->frequency_penalty * float(kv.second);
    }
    return l;
}

// Filtered, normalised distribution as (prob, id), sorted by prob desc.
void distribution(ei_ctx* c, const float* l, const ei_sampling* p, std::vector<std::pair<float, int>>& out) {
    const int V = ei_n_vocab(c->m);
    const int K = p->top_k > 0 ? std::min(p->top_k, V) : std::min(1024, V);
    // top-K by logit with a min-heap
    using P = std::pair<float, int>;
    std::priority_queue<P, std::vector<P>, std::greater<P>> heap;
    for (int i = 0; i < V; ++i) {
        if (int(heap.size()) < K) heap.push({l[i], i});
        else if (l[i] > heap.top().first) { heap.pop(); heap.push({l[i], i}); }
    }
    out.clear();
    while (!heap.empty()) { out.push_back(heap.top()); heap.pop(); }
    std::reverse(out.begin(), out.end());
    const float T = p->temperature;
    const float mx = out[0].first;
    double Z = 0;
    if (p->top_k > 0) {
        for (auto& e : out) Z += std::exp((e.first - mx) / T);
    } else {
        // normaliser over the whole vocabulary, so top-p is exact
        const __m256 mv = _mm256_set1_ps(mx), it = _mm256_set1_ps(1.0f / T);
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= V; i += 8) acc = _mm256_add_ps(acc, exp256(_mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(l + i), mv), it)));
        Z = hsum8(acc);
        for (; i < V; ++i) Z += std::exp((l[i] - mx) / T);
    }
    for (auto& e : out) e.first = float(std::exp((e.first - mx) / T) / Z);
    size_t keep = out.size();
    if (p->top_p > 0 && p->top_p < 1.0f) {
        double cum = 0;
        for (size_t i = 0; i < out.size(); ++i) {
            cum += out[i].first;
            if (cum >= p->top_p) { keep = i + 1; break; }
        }
    }
    if (p->min_p > 0) {
        const float th = p->min_p * out[0].first;
        size_t k2 = 0;
        while (k2 < keep && out[k2].first >= th) ++k2;
        keep = std::max<size_t>(1, k2);
    }
    out.resize(keep);
    double s = 0;
    for (auto& e : out) s += e.first;
    for (auto& e : out) e.first = float(e.first / s);
}

int32_t draw(const std::vector<std::pair<float, int>>& d, uint64_t& rng, float* prob) {
    float u = urand(rng), cum = 0;
    for (auto& e : d) {
        cum += e.first;
        if (u < cum) { if (prob) *prob = e.first; return e.second; }
    }
    if (prob) *prob = d.back().first;
    return d.back().second;
}

}  // namespace

extern "C" int32_t ei_argmax(const float* x, int n) {
    __m256 best = _mm256_set1_ps(-FLT_MAX);
    int i = 0;
    for (; i + 8 <= n; i += 8) best = _mm256_max_ps(best, _mm256_loadu_ps(x + i));
    alignas(32) float b[8];
    _mm256_store_ps(b, best);
    float mv = b[0];
    for (int k = 1; k < 8; ++k) mv = std::max(mv, b[k]);
    for (; i < n; ++i) mv = std::max(mv, x[i]);
    for (int k = 0; k < n; ++k)
        if (x[k] == mv) return k;
    return 0;
}

extern "C" int32_t ei_sample(ei_ctx* c, const float* logits, const ei_sampling* p, const int32_t* hist, int nh,
                             uint64_t* rng, float* prob_out) {
    const float* l = penalised(c, logits, p, hist, nh);
    if (p->temperature <= 0) {
        if (prob_out) *prob_out = 1.0f;
        return ei_argmax(l, ei_n_vocab(c->m));
    }
    distribution(c, l, p, c->cand);
    return draw(c->cand, *rng, prob_out);
}

extern "C" int32_t ei_spec_accept(ei_ctx* c, const float* logits, const ei_sampling* p, const int32_t* hist, int nh,
                                  int32_t draft, uint64_t* rng, int* accepted) {
    const float* l = penalised(c, logits, p, hist, nh);
    if (p->temperature <= 0) {
        const int32_t t = ei_argmax(l, ei_n_vocab(c->m));
        *accepted = t == draft;
        return t;
    }
    distribution(c, l, p, c->cand);
    float pd = 0;
    for (auto& e : c->cand)
        if (e.second == draft) { pd = e.first; break; }
    if (pd > 0 && urand(*rng) < pd) { *accepted = 1; return draft; }
    *accepted = 0;
    // residual distribution: p with the draft removed
    std::vector<std::pair<float, int>> rest;
    double s = 0;
    for (auto& e : c->cand)
        if (e.second != draft) { rest.push_back(e); s += e.first; }
    if (rest.empty()) return draft;
    for (auto& e : rest) e.first = float(e.first / s);
    return draw(rest, *rng, nullptr);
}
