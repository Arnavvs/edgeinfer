/* edgeinfer - a from-scratch CPU inference engine for hybrid (Gated DeltaNet +
 * gated attention) language models such as Qwen3.5.
 *
 * C API. Everything the Python server needs: load a model, run batched
 * forward passes over several sequences ("slots") at once, verify draft tokens
 * and keep only the accepted ones, snapshot a slot's state for prefix reuse,
 * draft tokens with the model's own multi-token-prediction (MTP) head, sample.
 *
 * Thread-safety: one ei_ctx must be driven by one thread at a time. The
 * engine parallelises each call internally with its own thread pool.
 */
#ifndef EDGEINFER_H
#define EDGEINFER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ei_model ei_model;
typedef struct ei_ctx ei_ctx;

/* tensor storage types (also used by the converter) */
enum { EI_F32 = 0, EI_F16 = 1, EI_BF16 = 2, EI_Q8 = 3, EI_Q4 = 4, EI_Q4K = 5 };

/* ---- quantization helpers for the converter --------------------------- */
int64_t ei_tensor_bytes(int dtype, int64_t rows, int64_t cols);
/* src is row-major float32 [rows][cols]; importance is optional [cols] */
int ei_quantize(int dtype, const float* src, int64_t rows, int64_t cols,
                void* dst, const float* importance, int n_threads);
int ei_dequantize_row(int dtype, const void* data, int64_t rows, int64_t cols,
                      int64_t row, float* out);
/* y[m][rows] = W x for a weight in engine layout (unit tests, benchmarks) */
int ei_matmul(int dtype, const void* w, int64_t rows, int64_t cols, const float* x, int m,
              float* y, int n_threads);

/* ---- model -------------------------------------------------------------- */
ei_model* ei_model_load(const char* path, int n_threads, char* err, int err_len);
void ei_model_free(ei_model* m);
const char* ei_model_info(const ei_model* m); /* JSON */
int ei_n_vocab(const ei_model* m);
int ei_n_embd(const ei_model* m);
int ei_has_mtp(const ei_model* m);
int ei_get_threads(const ei_model* m);
void ei_set_threads(ei_model* m, int n_threads);
/* Restrict the MTP draft head to these token ids (frequency-ranked vocab,
 * "FR-Spec"). n=0 restores the full vocabulary. */
int ei_set_draft_vocab(ei_model* m, const int32_t* ids, int n);

/* ---- context: slots, KV cache, recurrent state --------------------------- */
ei_ctx* ei_ctx_new(ei_model* m, int n_slots, int n_ctx, int max_batch, int max_draft);
void ei_ctx_free(ei_ctx* c);

enum {
    EI_SEG_VERIFY = 1, /* keep a state checkpoint after every token; finish with ei_commit */
    EI_SEG_HIDDEN = 2  /* keep the final-norm hidden state of every token (for MTP prefill) */
};

typedef struct {
    int32_t slot;
    int32_t n_tokens;
    const int32_t* tokens;
    int32_t pos0;     /* position of tokens[0]; must equal the slot's length */
    int32_t n_logits; /* produce logits for the last n_logits tokens */
    int32_t flags;
} ei_seg;

/* Runs the batch. Returns the number of logit rows produced (rows are in
 * segment order), or a negative error code. */
int ei_forward(ei_ctx* c, const ei_seg* segs, int n_segs);
const float* ei_logits(ei_ctx* c, int row);
/* hidden state (after the final norm) of logit row `row` */
const float* ei_hidden(ei_ctx* c, int row);
/* with EI_SEG_HIDDEN: hidden state of token i of the last batch */
const float* ei_hidden_tok(ei_ctx* c, int i);

int ei_slot_len(ei_ctx* c, int slot);
void ei_slot_reset(ei_ctx* c, int slot);
/* after a VERIFY segment of n tokens: keep the first n_keep (0..n) */
int ei_commit(ei_ctx* c, int slot, int n_keep);

/* ---- snapshots (prefix cache) -------------------------------------------- */
/* Saves the recurrent/conv state at the slot's current length. With with_kv
 * the attention KV rows [0,len) are copied too, so the snapshot can be
 * restored into a different slot. Returns an id >= 0. */
int ei_snap_save(ei_ctx* c, int slot, int with_kv);
int ei_snap_restore(ei_ctx* c, int slot, int snap);
int ei_snap_len(ei_ctx* c, int snap);
int64_t ei_snap_bytes(ei_ctx* c, int snap);
void ei_snap_free(ei_ctx* c, int snap);
int ei_snap_write(ei_ctx* c, int snap, const char* path);
int ei_snap_read(ei_ctx* c, const char* path);

/* ---- sampling ------------------------------------------------------------ */
typedef struct {
    float temperature;   /* <= 0: greedy */
    float top_p;         /* 1 = off */
    float min_p;         /* 0 = off */
    int32_t top_k;       /* 0 = off */
    float repeat_penalty;    /* 1 = off (multiplicative, llama.cpp style) */
    float presence_penalty;  /* 0 = off (OpenAI style) */
    float frequency_penalty; /* 0 = off */
    uint64_t seed;
} ei_sampling;

/* Samples from logit row `row`. `history` are previous token ids for the
 * penalties. If prob_out is not NULL it receives the token's probability
 * under the final (filtered, tempered) distribution. */
int32_t ei_sample(ei_ctx* c, const float* logits, const ei_sampling* p,
                  const int32_t* history, int n_history, uint64_t* rng_state, float* prob_out);
/* Speculative sampling with a deterministic draft: accept `draft` with
 * probability p(draft); on reject returns a token sampled from p with the
 * draft removed. Returns the token; *accepted is set to 1/0. */
int32_t ei_spec_accept(ei_ctx* c, const float* logits, const ei_sampling* p,
                       const int32_t* history, int n_history, int32_t draft,
                       uint64_t* rng_state, int* accepted);
int32_t ei_argmax(const float* logits, int n);

/* ---- multi-token prediction (self-speculative drafting) ------------------ */
/* Forget MTP entries at positions >= len. */
void ei_mtp_truncate(ei_ctx* c, int slot, int len);
int ei_mtp_len(ei_ctx* c, int slot);
/* Runs the MTP head over n positions pos0..pos0+n-1 (pos0 <= ei_mtp_len).
 * Entry i combines the main model's final-norm hidden state at position
 * pos0+i (hidden[i], [n][n_embd]) with the embedding of tokens[i], the token
 * at position pos0+i+1. hidden == NULL uses the MTP head's own last output
 * (chained drafting, n must be 1). Returns the greedy draft for the token
 * after tokens[n-1]. */
int32_t ei_mtp_forward(ei_ctx* c, int slot, int pos0, int n, const int32_t* tokens,
                       const float* hidden);

/* ---- calibration (activation-aware quantization) ------------------------- */
/* While on, every matmul adds sum(x^2) per input column for its weight. */
void ei_calib(ei_ctx* c, int on);
/* Text file, one line per weight: name cols count mean_x2[0..cols) */
int ei_calib_dump(ei_ctx* c, const char* path);

/* ---- introspection ------------------------------------------------------- */
const char* ei_ctx_stats(ei_ctx* c); /* JSON: op timings, bytes moved */
void ei_ctx_reset_stats(ei_ctx* c);
int64_t ei_ctx_bytes(ei_ctx* c);

#ifdef __cplusplus
}
#endif
#endif
