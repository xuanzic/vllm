# Experimental NVFP4-to-FP8 KV cache bridge

This branch contains a functional and accuracy-evaluation bridge for models
whose head dimension is not supported by FlashInfer's fused NVFP4 attention
kernels. It was developed for Gemma 4 full-attention layers with head
dimension 512.

The data path is:

```text
model K/V
  -> physical paged NVFP4 cache
  -> standalone NVFP4 dequantization directly to FP8 E4M3
  -> existing H512 FP8 FMHA
  -> BF16 attention output
```

Set `GEMMA4_NVFP4_PHYSICAL_DEQUANT_FP8_BRIDGE=1` and launch vLLM with
`--kv-cache-dtype nvfp4 --attention-backend FLASHINFER`. The bridge keeps the
physical vLLM cache in NVFP4, plans attention as FP8, and uses K/V scales of
1.0 for FMHA because the NVFP4 global scale has already been applied by the
standalone dequantization kernel.

The direct FP8 output requires the FlashInfer change from
[`xuanzic/flashinfer@ef50007a`](https://github.com/xuanzic/flashinfer/commit/ef50007a0c7f2f75a4b1b07d01aa2fc78bd9636b).
When using the vLLM 0.26.0 CUDA 12.9 image, use the corresponding FlashInfer
0.6.14 compatibility backport at
[`xuanzic/flashinfer@820d851`](https://github.com/xuanzic/flashinfer/commit/820d851)
so the Python attention wrappers continue to match the image's precompiled
cubins.

## Run from the development checkouts

This is a Python-only vLLM change. Follow vLLM's editable-install workflow and
select the CUDA variant that matches the image. For example, with
`vllm/vllm-openai:v0.26.0-cu129-ubuntu2404`:

```bash
git clone --branch gemma4-nvfp4-direct-fp8-v0.26 \
  https://github.com/xuanzic/vllm.git
git clone --branch nvfp4-kv-dequant-fp8-output-v0.6.14 \
  https://github.com/xuanzic/flashinfer.git flashinfer-v0.6.14-fp8

cd vllm
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_VARIANT=cu129 \
uv pip install --editable .
export PYTHONPATH="$PWD:$(dirname "$PWD")/flashinfer-v0.6.14-fp8"
```

Do not mix the default CUDA 13.0 v0.26.0 wheel with a CUDA 12.9 image. An
equivalent container-development setup can reuse the exact native extension
artifacts already installed in the matching image while loading all Python
modules from the two checkouts above.

Start the tested model route with:

```bash
export GEMMA4_NVFP4_PHYSICAL_DEQUANT_FP8_BRIDGE=1
export FLASHINFER_LOG_LEVEL=4

python -m vllm.entrypoints.cli.main serve \
  jtchen0528/gemma-4-31B-it-FP8 \
  --served-model-name gemma \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-batched-tokens 2048 \
  --enable-chunked-prefill \
  --max-num-seqs 1 \
  --kv-cache-dtype nvfp4 \
  --attention-backend FLASHINFER \
  --no-enable-flashinfer-autotune \
  --enforce-eager \
  --language-model-only \
  --hf-overrides '{"text_config":{"use_bidirectional_attention":null}}'
```

The short context limits and eager execution are smoke-test settings, not
production recommendations.

## Validation

The implementation was tested on one B200 using the CUDA 12.9 vLLM 0.26.0
image and `jtchen0528/gemma-4-31B-it-FP8`:

- FlashInfer dequantization tests: 17 passed, including direct FP8 E4M3 output
  and head dimension 512.
- End-to-end deterministic smoke tests: 3/3 passed (exact response,
  arithmetic, and long-prompt retrieval).
- Runtime logs reported physical NVFP4 cache allocation, the experimental
  dequantization bridge, and an H512 paged FMHA kernel with E4M3 K/V.

This path is not a performance implementation. It dequantizes the physical
cache to a temporary FP8 allocation for each attention call, does not support
cascade attention, and does not represent the latency or memory traffic of a
future fused H512 NVFP4 FMHA kernel. Its purpose is to measure functional and
model-accuracy impact while retaining real NVFP4 cache storage and
quantization error.
