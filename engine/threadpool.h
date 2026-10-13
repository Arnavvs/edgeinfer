// Persistent worker pool with dynamic task scheduling.
//
// Workers spin (with PAUSE) for a short while after each job, so the many
// small parallel ops of one forward pass dispatch in about a microsecond;
// after ~1 ms without work they sleep on a condition variable, so an idle
// server costs no CPU (this laptop also runs the phone rig).
#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "common.h"

namespace ei {

class ThreadPool {
public:
    using Fn = void (*)(void* ctx, int task, int thread);

    explicit ThreadPool(int n_threads) { start(n_threads); }
    ~ThreadPool() { stop(); }

    int size() const { return n_threads_; }

    void resize(int n_threads) {
        if (n_threads == n_threads_) return;
        stop();
        start(n_threads);
    }

    // Runs fn(ctx, task, thread) for every task in [0, n_tasks). The calling
    // thread works too (thread index 0). Returns when all tasks are done.
    void run(int n_tasks, Fn fn, void* ctx) {
        if (n_tasks <= 0) return;
        if (n_threads_ == 1 || n_tasks == 1) {
            for (int t = 0; t < n_tasks; ++t) fn(ctx, t, 0);
            return;
        }
        fn_ = fn;
        ctx_ = ctx;
        n_tasks_ = n_tasks;
        next_.store(0, std::memory_order_relaxed);
        pending_.store(int(workers_.size()), std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(mu_);
            gen_.fetch_add(1, std::memory_order_release);
        }
        if (sleepers_.load(std::memory_order_acquire) > 0) cv_.notify_all();
        work(0);
        while (pending_.load(std::memory_order_acquire) != 0) _mm_pause();
    }

private:
    void start(int n_threads) {
        n_threads_ = n_threads < 1 ? 1 : n_threads;
        quit_.store(false);
        // The generation is read here, not in the thread: a worker that is
        // scheduled late must still see the first job as new.
        const uint32_t g = gen_.load();
        for (int i = 1; i < n_threads_; ++i) workers_.emplace_back([this, i, g] { loop(i, g); });
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            quit_.store(true);
            gen_.fetch_add(1);
        }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
        workers_.clear();
    }

    void work(int thread) {
        for (;;) {
            int t = next_.fetch_add(1, std::memory_order_relaxed);
            if (t >= n_tasks_) break;
            fn_(ctx_, t, thread);
        }
    }

    void loop(int id, uint32_t seen) {
        for (;;) {
            double t0 = 0;
            int spins = 0;
            while (gen_.load(std::memory_order_acquire) == seen) {
                if (++spins < 64) { _mm_pause(); continue; }
                if (t0 == 0) t0 = now_s();
                if (now_s() - t0 < 1e-3) { spins = 0; _mm_pause(); continue; }
                std::unique_lock<std::mutex> lk(mu_);
                sleepers_.fetch_add(1);
                cv_.wait(lk, [&] { return gen_.load() != seen; });
                sleepers_.fetch_sub(1);
                break;
            }
            seen = gen_.load(std::memory_order_acquire);
            if (quit_.load()) return;
            work(id);
            pending_.fetch_sub(1, std::memory_order_acq_rel);
        }
    }

    int n_threads_ = 1;
    std::vector<std::thread> workers_;
    alignas(64) std::atomic<uint32_t> gen_{0};
    alignas(64) std::atomic<int> next_{0};
    alignas(64) std::atomic<int> pending_{0};
    std::atomic<int> sleepers_{0};
    std::atomic<bool> quit_{false};
    std::mutex mu_;
    std::condition_variable cv_;
    Fn fn_ = nullptr;
    void* ctx_ = nullptr;
    int n_tasks_ = 0;
};

// Convenience: parallel for over [0, n) in chunks, with a lambda.
template <class F>
struct ForCtx {
    F* f;
    int n, chunk;
};

template <class F>
inline void parallel_for(ThreadPool& pool, int n, int chunk, F&& f) {
    if (n <= 0) return;
    if (chunk < 1) chunk = 1;
    int tasks = (n + chunk - 1) / chunk;
    using FF = std::remove_reference_t<F>;
    ForCtx<FF> c{&f, n, chunk};
    pool.run(tasks, [](void* p, int task, int thread) {
        auto* c = static_cast<ForCtx<FF>*>(p);
        int a = task * c->chunk, b = a + c->chunk;
        if (b > c->n) b = c->n;
        (*c->f)(a, b, thread);
    }, &c);
}

}  // namespace ei
