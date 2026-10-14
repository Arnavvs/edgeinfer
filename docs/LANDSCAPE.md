# CPU inference stacks for Qwen3.5, October 2026

What exists for running a small hybrid (Gated DeltaNet + attention) model on an
old x86 laptop CPU, and what was measured here. "Measured" rows have numbers in
the README; the rest were checked against their docs/release notes on 2026-10-14.

| stack | what it is | Qwen3.5 (hybrid) on x86 CPU | notes |
|---|---|---|---|
| **llama.cpp** (b11606) | C/C++ ggml runtime, GGUF, K-quants, `llama-server` | yes (`qwen35` arch) | **measured**: the strongest baseline; repacked Q4_0/K-quant AVX2 kernels, prompt cache, context checkpoints for recurrent layers |
| **llama-cpp-python** 0.3.35 | Python bindings to llama.cpp | yes (bundled llama.cpp) | **measured**: the usual "Python app" route; inherits llama.cpp speed minus per-call overhead |
| **Ollama** 0.40.3 | daemon + model registry around ggml | yes | **measured**: what most people run locally; one model runner process per model |
| **Hugging Face transformers** 5.18 | PyTorch reference implementation | yes (torch fallback for the delta rule and causal conv; FLA/causal-conv1d kernels are CUDA-only) | **measured**: fp32/bf16 only on CPU, no int4/int8 path for this architecture; the correctness reference for edgeinfer |
| OpenVINO GenAI | Intel's runtime, IR models | Intel publishes Qwen3.5 2B/4B/9B fp16/int8 IR (`VLMPipeline`, nightly wheels, OpenVINO >= 2026.2) | not measured: no int4 export published, nightly-only support for the arch at the time |
| ONNX Runtime GenAI | ORT + model builder | Qwen3.5 added with `LinearAttention`/`CausalConvWithState` ops for the **WebGPU** EP (ORT 1.25) | no CPU EP path found |
| vLLM (CPU backend) | PagedAttention server | Qwen3.5 recipes are GPU-only; GDN kernels are Triton (GPU) | no x86 CPU path found |
| ik_llama.cpp | llama.cpp fork with extra CPU quant kernels | has `qwen35` | not measured (build from source); community reports are mixed for Qwen3.5 |
| mllm, rlx-qwen35 | research / Rust engines | GDN on CPU reported | small projects, no OpenAI server |

Why a custom engine still makes sense on this box:

1. **Speculative decoding with the model's own MTP head** is not available in the
   CPU stacks above for Qwen3.5; it needs zero-copy rollback of the DeltaNet state,
   which edgeinfer gets from per-token state buffers in its verify pass.
2. **Prefix reuse for a recurrent model** needs state snapshots at the right
   places (system prompt, turn boundaries, answer end) and a chat renderer that
   keeps the re-rendered history byte-identical to what was generated.
3. **Small-model overheads**: on a 0.8B model the 248K-row LM head is ~40% of
   a decode step; a frequency-ranked draft head (FR-Spec) and per-tensor precision
   choices matter more than raw GEMM speed.

Sources: llama.cpp releases (github.com/ggml-org/llama.cpp/releases),
OpenVINO Qwen3.5 IR cards (huggingface.co/OpenVINO/Qwen3.5-2B-int8-ov),
ONNX Runtime 1.25.1 release notes, vLLM Qwen3.5 recipes (recipes.vllm.ai),
Ampere community Qwen3.5 CPU benchmarks.
