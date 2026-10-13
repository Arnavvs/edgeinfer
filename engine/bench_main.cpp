// ei-bench: llama-bench style throughput test, no Python in the loop.
//   ei-bench -m model.eim [-t threads] [-p prompt_tokens] [-n gen_tokens]
//            [-d depth] [-r repetitions] [-b batch] [--mtp]
// Prints one JSON line per measurement.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ei.h"

static double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    std::string path;
    int threads = 4, pp = 512, tg = 128, depth = 0, reps = 2, batch = 512, mtp = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&] { return i + 1 < argc ? argv[++i] : (char*)"0"; };
        if (a == "-m") path = next();
        else if (a == "-t") threads = std::atoi(next());
        else if (a == "-p") pp = std::atoi(next());
        else if (a == "-n") tg = std::atoi(next());
        else if (a == "-d") depth = std::atoi(next());
        else if (a == "-r") reps = std::atoi(next());
        else if (a == "-b") batch = std::atoi(next());
        else if (a == "--mtp") mtp = 1;
    }
    if (path.empty()) { std::fprintf(stderr, "usage: ei-bench -m model.eim [-t 4] [-p 512] [-n 128] [-d 0]\n"); return 1; }
    char err[512] = {0};
    double t0 = now();
    ei_model* m = ei_model_load(path.c_str(), threads, err, sizeof err);
    if (!m) { std::fprintf(stderr, "load failed: %s\n", err); return 1; }
    const double load_s = now() - t0;
    const int V = ei_n_vocab(m);
    const int n_ctx = depth + pp + tg + 64;
    ei_ctx* c = ei_ctx_new(m, 1, n_ctx, batch, 4);
    if (!c) { std::fprintf(stderr, "ctx failed\n"); return 1; }
    std::printf("{\"event\":\"load\",\"seconds\":%.3f,\"info\":%s}\n", load_s, ei_model_info(m));

    // deterministic pseudo-random prompt tokens from the ordinary vocabulary
    auto tokens = [&](int n, uint32_t seed) {
        std::vector<int32_t> v(n);
        for (int i = 0; i < n; ++i) { seed = seed * 1664525u + 1013904223u; v[i] = int32_t(1000 + seed % 140000); }
        return v;
    };
    auto prefill = [&](const std::vector<int32_t>& v) {
        int done = 0;
        while (done < int(v.size())) {
            int n = std::min(batch, int(v.size()) - done);
            ei_seg s{0, n, v.data() + done, ei_slot_len(c, 0), done + n == int(v.size()) ? 1 : 0, 0};
            if (ei_forward(c, &s, 1) < 0) { std::fprintf(stderr, "forward failed\n"); std::exit(1); }
            done += n;
        }
    };
    for (int r = 0; r < reps; ++r) {
        ei_slot_reset(c, 0);
        if (depth > 0) prefill(tokens(depth, 7));
        if (pp > 0) {
            ei_ctx_reset_stats(c);
            double a = now();
            prefill(tokens(pp, 11 + r));
            double s = now() - a;
            std::printf("{\"event\":\"pp\",\"threads\":%d,\"n\":%d,\"depth\":%d,\"seconds\":%.4f,\"tok_s\":%.2f,\"stats\":%s}\n",
                        threads, pp, depth, s, pp / s, ei_ctx_stats(c));
        }
        if (tg > 0) {
            int32_t tok = 1000;
            if (pp == 0 && depth == 0) { ei_seg s{0, 1, &tok, 0, 1, 0}; ei_forward(c, &s, 1); }
            ei_ctx_reset_stats(c);
            double a = now();
            for (int i = 0; i < tg; ++i) {
                tok = ei_argmax(ei_logits(c, 0), V);
                ei_seg s{0, 1, &tok, ei_slot_len(c, 0), 1, 0};
                if (ei_forward(c, &s, 1) < 0) { std::fprintf(stderr, "decode failed\n"); return 1; }
            }
            double s = now() - a;
            std::printf("{\"event\":\"tg\",\"threads\":%d,\"n\":%d,\"depth\":%d,\"seconds\":%.4f,\"tok_s\":%.2f,\"stats\":%s}\n",
                        threads, tg, depth + pp, s, tg / s, ei_ctx_stats(c));
        }
        std::fflush(stdout);
    }
    (void)mtp;
    ei_ctx_free(c);
    ei_model_free(m);
    return 0;
}
