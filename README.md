# edgeinfer

**A from-scratch C++ inference engine and OpenAI-compatible server for hybrid
Gated-DeltaNet language models (Qwen3.5), built for a 2017 laptop with no GPU.**
It also includes a login-protected "chat with your data" app (SQL + hybrid RAG) that runs on it.

Target machine: Intel Core i5-7300U, 2 cores / 4 threads, AVX2, turbo off (2.6 GHz),
8 GB single-channel DDR4, Debian 13. The same box runs other always-on jobs, so the server
runs at low CPU priority and unloads the model when idle.

<!--RESULTS-->

## What is in here

| part | what it does |
|---|---|
| [`engine/`](engine) | C++17, no dependencies: model loader (mmap), AVX2 int8 kernels for 3 weight formats, Gated DeltaNet + gated attention forward pass, batched multi-sequence forward, verify/commit state buffers for speculative decoding, state snapshots for prefix caching, MTP draft head, sampling, own thread pool |
| [`edgeinfer/`](edgeinfer) | Python: ctypes bindings, HF -> `.eim` converter (imatrix, mixed precision), chat template + tool-call parser, continuous-batching scheduler with prefix cache and self-speculative decoding, FastAPI server |
| [`apps/analyst/`](apps/analyst) | *Data Desk*: login, conversations, tool-using agent (read-only SQL + hybrid search), FAISS + FTS5 retrieval with e5 embeddings, streaming UI |
| [`bench/`](bench) | quality (KL / perplexity vs bf16), raw throughput vs llama.cpp / llama-cpp-python / Ollama / transformers, end-to-end serving (multi-turn TTFT, concurrency), agent accuracy |
| [`tests/`](tests) | logits vs transformers, kernel unit tests, chat template vs Jinja, lossless speculative decoding, prefix-cache and disk-cache invariants |
| [`docs/DESIGN.md`](docs/DESIGN.md) | how and why: formats, kernels, threading, MTP + FR-Spec, snapshots |

## Engineering highlights

* **Correctness first:** logits match Hugging Face transformers (fp32) to cosine >= 0.999999
  / KL ~1e-7 with bf16 weights; greedy continuations identical; batched prefill equals
  token-by-token decode.
* **Quantization:** tiled 4-row layouts, int8 activation x 4/8-bit weight kernels
  (`vpmaddubsw`/`vpmaddwd`), an asymmetric 4-bit format with 8-bit sub-scales, activation-aware
  importance (imatrix/AWQ-style calibration) and a mixed-precision recipe that **matches llama.cpp
  Q4_K_M on KL divergence**.
* **Self-speculative decoding** with the model's own multi-token-prediction head: zero-copy
  state checkpoints per drafted token, exact acceptance rule (greedy and sampled), and
  **FR-Spec** frequency-ranked draft vocabulary so the 248K-token LM head does not eat the gain.
* **Prefix caching for a recurrent model:** the DeltaNet state cannot be truncated, so the
  scheduler snapshots it at chat-turn boundaries and restores the longest usable snapshot;
  long system prompts are also **persisted to disk** and survive restarts.
* **Serving:** continuous batching with chunked prefill, OpenAI API with streaming and tool
  calls, Prometheus metrics, lazy load + idle unload, per-request timing (TTFT, tok/s,
  cached tokens, drafts accepted).

## Quick start

```bash
git clone https://github.com/Arnavvs/edgeinfer && cd edgeinfer
make build venv
hf download Qwen/Qwen3.5-0.8B --local-dir ~/edgeinfer-data/hf/Qwen3.5-0.8B
make imatrix convert            # -> ~/edgeinfer-data/models/qwen3.5-0.8b-q4m.eim
cat > ~/edgeinfer-data/models.json <<'EOF'
{"models": [{"name": "qwen3.5-0.8b", "path": "~/edgeinfer-data/models/qwen3.5-0.8b-q4m.eim",
             "tokenizer": "~/edgeinfer-data/hf/Qwen3.5-0.8B", "ctx": 16384, "threads": 4}]}
EOF
make serve                      # OpenAI API on 127.0.0.1:8091
curl -s localhost:8091/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"stream":false}'
```

Data Desk on the synthetic demo data: `make demo`, then
`python -m apps.analyst.auth set-password me` and
`python -m apps.analyst.app --config apps/analyst/demo/demo.json`.

## License

Apache-2.0. Qwen3.5 weights are Apache-2.0 (Alibaba Qwen team).
