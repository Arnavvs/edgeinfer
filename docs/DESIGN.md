# edgeinfer design notes

edgeinfer runs Qwen3.5 (0.8B / 2B / 4B) on a 2017 laptop CPU: Intel i5-7300U
(2 cores, 4 threads, AVX2 + FMA + F16C, no AVX-512, no VNNI), turbo off (2.6 GHz),
8 GB of single-channel DDR4. Nothing here assumes a GPU.

On this machine a decode step is **memory-bandwidth bound** (every weight is read once
per token; the bus delivers ~12-13 GB/s) and prefill is **compute bound** (int8
multiply-adds on two cores). Every design choice below follows from those two facts.

## 1. The model: Qwen3.5 hybrid attention

Qwen3.5 small models interleave two token mixers, 3:1:

* **Gated DeltaNet** (linear attention), 3 of every 4 layers: a depthwise causal
  conv (kernel 4) + SiLU over q/k/v, L2-normalised q and k, and a per-head
  matrix state `S` (128x128) updated with the gated delta rule
  `S <- exp(g) S + k (beta (v - exp(g) S^T k))^T`, output `o = S^T q`, then a
  gated RMSNorm (`norm(o) * w * silu(z)`).
* **Gated attention**, every 4th layer: GQA with QK-RMSNorm, RoPE on the first
  25% of each 256-dim head (theta 1e7), and a sigmoid output gate from a second
  half of `q_proj`.

Consequences for an inference engine:

* Only 1/4 of the layers keep a KV cache, so long contexts are cheap
  (0.8B: 12 KB per token in fp16).
* The DeltaNet state cannot be rewound or truncated. Anything that needs to
  "go back" (speculative-decoding rejections, prefix reuse) must use
  **snapshots** of the state, see sections 5 and 6.
* The model ships a **multi-token-prediction (MTP) head** (one extra
  attention+MLP block) trained to predict token t+2, which makes
  self-speculative decoding possible without a separate draft model.

The engine was checked against Hugging Face transformers (fp32) with the bf16
weights: cosine >= 0.999999, mean KL ~1e-7 per position, identical greedy
continuations on all test prompts, and identical results for batched prefill
vs token-by-token decode (`tests/ref_check.py`).

## 2. Weight formats and kernels (`engine/quant.*`)

Weights are stored in **tiles of 4 output rows**; inside a tile the 32-value
blocks of the 4 rows are interleaved, so a 4-row microkernel streams one
contiguous region and uses every byte of each cache line.

| format | layout | bits/weight |
|---|---|---|
| `q8` | int8, fp16 scale per 32 | 8.5 |
| `q4` | symmetric 4-bit stored as `q+8`, fp16 scale per 32 | 4.5 |
| `q4k` | asymmetric 4-bit `w = d*sc*q - dm*mn`: uint8 scale and min per 32-value sub-block, fp16 `d`, `dm` per 256 | 4.63 |
| `q6` | symmetric 6-bit (low nibbles + 2-bit high plane), fp16 scale per 32; used for the LM head | 6.5 |

Activations are quantized on the fly to int8 per 32-block, with the block sum
kept alongside. The inner loop is `vpmaddubsw` (u8 x s8 -> s16 pairs) +
`vpmaddwd` (-> s32) + convert + FMA with the combined scale. Two tricks:

* **Unsigned weights.** 4-bit weights stay unsigned (0..15), so `vpmaddubsw` takes them
  directly. The offset (`-8` for `q4`, `-dm*mn` for `q4k`) is removed once
  per block from the activation block sums, a single 128-bit FMA for all
  4 rows, instead of the per-pair sign trick.
* **Integer super-blocks for `q4k`.** Activations for `q4k` weights are quantized per 256 values, so the
  8-bit sub-block scales become the 16-bit multipliers of `vpmaddwd` (broadcast with one `vpshufb`, on the
  otherwise idle shuffle port) and the 8 sub-blocks of a super-block accumulate in int32: one int->float
  conversion and FMA per 256 values instead of per 32. Prompt processing on 0.8B went from 84.6 to 101 tok/s.
* **4 rows x 2 tokens per microkernel** in prefill, so each expanded weight
  block is reused for two tokens and each activation block for four rows.
  Token blocks are sized to keep the activations in L1/L2 while tiles stream.

Nibble expansion is one `vbroadcasti128` (a load-port op) + shift + blend + and.

## 3. Quantization quality

Measured as KL divergence against the bf16 model on wikitext-2 (`bench/quality.py`),
the same metric llama.cpp uses:

* Symmetric 4-bit with a searched scale is too lossy for a 0.8B model
  (KL 0.10). The error is spread over all layers (`--override` sensitivity scan).
* `q4k` (asymmetric, finer scales) cuts it to 0.070.
* **Activation-aware importance** (imatrix / AWQ idea): `tools/imatrix.py` runs
  the model over calibration text (wikitext *train*, plus chat turns) and
  records `E[x^2]` per input channel of every weight. The quantizer minimises
  the importance-weighted error. Result: 0.046.
* **Mixed precision** (`--profile q4m`): `ffn_down` in the sensitive layers
  (llama.cpp's `use_more_bits` rule: first and last eighth, then every third)
  and the DeltaNet `ssm_out` go to Q8, matching llama.cpp's Q4_K_M quality
  (KL 0.033) with a better 99th percentile.

## 4. Threading (`engine/threadpool.h`)

A persistent pool with dynamic task grabbing (`fetch_add`), because the 4
hyperthreads on 2 cores progress at different speeds. Workers spin with PAUSE
for ~1 ms after a job, so the ~200 small parallel ops of a token cost about a
microsecond each to dispatch, then sleep on a condition variable, so an idle
server uses no CPU (the laptop also runs other jobs). A startup race (a late
worker missing the first job) was found with a native stack dump and fixed
by handing each worker the generation counter from before it was started.

The DeltaNet recurrence is parallel over (sequence, head, 32-column block).
Columns of `S` evolve independently, so each task keeps a 128x32 slice of the
state (16 KB) in L1 for the whole token run. Attention for few query tokens
(decode) splits the keys into chunks across threads and merges the partial
softmaxes (flash-decoding).

## 5. Self-speculative decoding with MTP (+ FR-Spec)

Decode is bandwidth bound, so checking several tokens in one pass costs
about the same as one token. Each step:

1. **Draft:** the MTP head takes the main model's final hidden state and the
   embedding of the newest token and proposes the next token; chaining it on
   its own output gives more drafts.
2. **Verify:** one forward over `[pending, d1, d2]` in `VERIFY` mode. The
   DeltaNet layers write the state after every token into separate buffers
   (ping-pong, so this costs no extra memory traffic beyond the reads and writes
   the recurrence does anyway).
3. **Accept:** greedy, accept while the target argmax equals the draft. With
   sampling, accept draft `d` with probability `p(d)` and otherwise resample
   from `p` without `d`. This is the exact speculative-sampling rule for a
   deterministic proposal, so the output distribution is unchanged.
4. **Commit:** `ei_commit(slot, n_keep)` picks the state buffer of the last
   accepted token and truncates the KV cache. Nothing is copied.

On a 248K-token vocabulary the draft head's LM-head matmul is as large as a
big chunk of the model, which would eat the gain. **FR-Spec** restricts the
*draft* head to the 32K most frequent tokens (`tools/token_freq.py`), copied
bit-exactly out of the full head. Verification still uses the full head, so
results are unchanged.

## 6. Prefix caching for a model with recurrent state

Chat clients resend the whole conversation every turn. The scheduler keeps
each slot's token history and **snapshots** (DeltaNet state + conv window;
the KV rows stay in place) at:

* the end of the system prompt and the start of the last user turn
  (offsets computed by rendering the conversation prefix; `cache_marks`),
* every 1024 tokens of a long prefill,
* the end of every answer.

A new request goes to the slot with the longest usable snapshot inside its
common prefix, restores it (a memcpy of ~19-50 MB) and encodes only the rest.
Two details made it work in practice:

* Qwen3.5's template drops the empty `<think></think>` block from earlier
  assistant turns, so the re-rendered history no longer matches what the model
  generated. The renderer can keep it (`keep_think`, default on in the server).
  It is the model's own output, so this is harmless.
* With speculative decoding, drafts accepted in the final step must not be
  committed past the point where the answer ends (stop token, max tokens),
  or the cached sequence contains a token the next prompt does not.

## 7. Serving (`edgeinfer/scheduler.py`, `edgeinfer/server.py`)

* One scheduler thread owns the engine. Each step forms one batch: a decode
  token for every generating request plus prefill chunks (chunked prefill)
  within a token budget, so a long prompt does not freeze other streams.
  MTP drafting is used when a single stream is generating, where it pays.
* OpenAI-compatible API with SSE streaming, tool calls (Qwen's XML format
  parsed into `tool_calls`), reasoning/answer split, per-request `timing`
  (TTFT, decode rate, cached prompt tokens, drafts accepted), Prometheus
  `/metrics` and a JSON `/v1/stats`.
* One model resident at a time, lazy load, unload after idle (the 8 GB box
  also runs other jobs). The service runs at `Nice=10`.

## 8. Data Desk (`apps/analyst`)

A login-protected chat over the user's SQLite data:

* **Tools:** `sql` (read-only connection, SQLite authorizer allowing only
  SELECT/READ/FUNCTION, 4 s timeout, 40-row cap) and `search`.
* **Retrieval:** chunking (600 chars, sentence boundaries, overlap) ->
  multilingual-e5-small (ONNX, CPU) in FAISS + SQLite FTS5 BM25 ->
  reciprocal rank fusion. Embeddings are cached by content hash, so the
  nightly rebuild only embeds new text.
* **Auth:** scrypt password hashes, HMAC-signed HttpOnly SameSite=Strict
  cookies, a custom-header CSRF check, login rate limiting, a strict CSP.
* The static system prompt (tools + schema + examples) is served from the
  prefix cache after the first request, and so are earlier turns.
