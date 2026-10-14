# edgeinfer

**A from-scratch C++ inference engine and OpenAI-compatible server for hybrid
Gated-DeltaNet language models (Qwen3.5), built for a 2017 laptop with no GPU.**
It also includes a login-protected "chat with your data" app (SQL + hybrid RAG) that runs on it.

**Live demo:** https://orchestrator.taila27abc.ts.net:8443/ (served from the laptop itself through Tailscale
Funnel; sign in with the guest account shown on the page: 16 questions per hour, all three models, the
collected data, the demo data and plain chat).
Portfolio: https://arnavvs.github.io/

Target machine: Intel Core i5-7300U, 2 cores / 4 threads, AVX2, turbo off (2.6 GHz),
8 GB single-channel DDR4, Debian 13. The same box runs other always-on jobs, so the server
runs at low CPU priority and unloads the model when idle.

## Results

All numbers below were measured on the target laptop (4 threads, idle machine, single user unless stated).
Raw logs: [`docs/results/`](docs/results). Method and caveats: [`docs/DESIGN.md`](docs/DESIGN.md).

### Speed (tokens/s, higher is better)

edgeinfer runs its `q4h` file (4-bit body, 6-bit LM head); llama.cpp b11606, llama-cpp-python 0.3.35 and
Ollama 0.40.3 run the same Q4_K_M GGUF files. Both are ~4.6-4.9 bits/weight with equal quality (next table).

| model | test | **edgeinfer** | llama.cpp | Ollama | llama-cpp-python | transformers (fp32) |
|---|---|---|---|---|---|---|
| 0.8B | prompt, 512 tok | **98.5** | 88.8 | 85.6 | 64.6 | 50.2 |
| 0.8B | generate, 128 tok | **20.9** | 19.4 | 18.8 | 17.4 | 3.4 |
| 0.8B | generate at 4K context | **18.6** | 13.0 | 13.9 | 15.6 | – |
| 2B | prompt, 512 tok | **39.6** | 38.5 | 37.6 | 29.1 | out of RAM |
| 2B | generate, 128 tok | **9.9** | 9.6 | 9.5 | 9.2 | |
| 2B | generate at 4K context | **9.3** | 7.7 | 8.1 | 8.6 | |
| 4B | prompt, 512 tok | **15.3** | 15.2 | 15.0 | 11.7 | out of RAM |
| 4B | generate, 128 tok | **4.5** | 4.3 | 4.3 | 4.3 | |
| 4B | generate at 4K context | **4.2** | 3.2 | 3.4 | 3.9 | |
| | peak RAM 0.8B / 2B / 4B (MB) | **713 / 1517 / 3323** | 957 / 2027 / 4465 | 989 / 1751 / 3485 | 1436 / 2501 / 4920 | 5810 / – / – |

**Self-speculative decoding** (MTP head + 32K-token FR-Spec draft vocabulary, 1 draft per step, greedy,
output identical to plain decoding): 0.8B 22.0 -> **23.9** tok/s (+8%, 42% of drafts accepted),
2B 10.5 -> **14.8** (+41%, 72%), 4B 4.8 -> **6.9** (+45%, 70%). Without FR-Spec the full 248K-row draft head
made 0.8B *slower* (16.4 tok/s); more than one draft per step is slower on this CPU.

### Quality (KL divergence per token vs the higher-precision model, wikitext-2; lower is better)

| model | edgeinfer `q4m` (8-bit head) | edgeinfer `q4h` (6-bit head) | edgeinfer `q4s` (4-bit head) | llama.cpp Q4_K_M |
|---|---|---|---|---|
| 0.8B vs bf16 | 0.0351 · 585 MiB | 0.0358 · 524 MiB | 0.0457 · 468 MiB | **0.0325** · 508 MiB |
| 2B vs Q8 | **0.0219** · 1416 MiB | **0.0219** · 1295 MiB | 0.0307 · 1181 MiB | 0.0228 · 1222 MiB |
| 4B vs Q8 | 0.0220 · 2993 MiB | **0.0217** · 2842 MiB | – | 0.0229 · 2614 MiB |

Top-1 token agreement with the base model: 0.8B 90.0% (llama.cpp 89.7%), 2B 91.1% (91.1%), 4B 92.8% (92.4%).
The fast Q4K kernel quantizes activations per 256 values; on 0.8B that costs a little (KL 0.0335 -> 0.0351
compared with per-32 activations), on 2B/4B it does not show.

### Chat serving (OpenAI API, ~1.8K-token system prompt + 5 user turns, 64 new tokens per turn)

| | time to first token, turn 1 | turns 2-5 (mean) | total time per turn, turns 2-5 | 2 concurrent clients, total tok/s |
|---|---|---|---|---|
| **edgeinfer**, 0.8B | 19.2 s | **0.45 s** | **3.4 s** | **30.8** |
| llama-server, 0.8B | 21.1 s | 1.56 s | 5.3 s | 22.0 |
| Ollama, 0.8B | – | – | 4.3 s | 19.0 |
| **edgeinfer**, 2B | 44.6 s | **0.94 s** | **6.9 s** | **14.8** |
| llama-server, 2B | 45.6 s | 2.99 s | 9.6 s | 11.1 |
| Ollama, 2B | – | – | 8.1 s | 9.2 |

Ollama's OpenAI endpoint returned each answer as one chunk in this setup, so only its total time is
comparable. Speculative decoding was off for this table (it is a pure engine + cache comparison).
After a restart, edgeinfer restores a persisted system prompt from disk: 16 of 977 prompt tokens
encoded, first token in 0.23 s instead of 10.6 s.

### Data Desk agent (12 questions with SQL-verified answers over the author's private data)

| model | correct | median time | model calls / question | prompt tokens served from cache |
|---|---|---|---|---|
| Qwen3.5-0.8B | 4 / 12 | 6.8 s | 2.1 | 94% |
| Qwen3.5-2B | 9 / 12 | 12.7 s | 2.1 | 89% |
| **Qwen3.5-4B** | **12 / 12** | 23.0 s | 2.0 | 94% |

### Which model to run

* **Qwen3.5-4B** (`q4h`, 2.8 GB): the default for analysis. Every eval question right; ~7 tok/s with drafting.
* **Qwen3.5-2B** (`q4h`, 1.3 GB): ~15 tok/s with drafting; good for chat and most lookups (9/12).
* **Qwen3.5-0.8B** (`q4h`, 0.5 GB): ~24 tok/s; quick chat, drafting, classification. Too weak for multi-step SQL.

Larger models do not fit this box usefully: an 8-9B model at 4 bits needs ~5 GB and would decode at ~2 tok/s.


## What is in here

| part | what it does |
|---|---|
| [`engine/`](engine) | C++17, no dependencies: model loader (mmap), AVX2 int8 kernels for 4 weight formats (q4k / q4 / q6 / q8), Gated DeltaNet + gated attention forward pass, batched multi-sequence forward, verify/commit state buffers for speculative decoding, state snapshots for prefix caching, MTP draft head, sampling, own thread pool |
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

### Running the site

`apps/analyst/app.py` serves a public project page at `/` and the login-protected chat at `/app`. A site
config lists workspaces (a dataset config, or `"chat"` for plain chat with the model) and per-account rules:
which workspaces an account may use, a questions-per-hour cap, allowed models, a token cap and concurrency.
Caps are counted in SQLite, so they survive restarts. Behind a reverse proxy (Tailscale Serve/Funnel) the app
uses `X-Forwarded-For` for per-visitor login limits and marks cookies `Secure` from `X-Forwarded-Proto`.

Data Desk on the synthetic demo data: `make demo`, then
`python -m apps.analyst.auth set-password me` and
`python -m apps.analyst.app --config apps/analyst/demo/demo.json`.

## License

Apache-2.0. Qwen3.5 weights are Apache-2.0 (Alibaba Qwen team).
