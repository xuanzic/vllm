# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GELU-tanh-and-mul fused with dynamic per-token FP8 quantization.

Replaces `gelu_tanh_and_mul` + `scaled_fp8_quant(use_per_token_if_dynamic=True)`
between the two GEMMs of the Triton MoE experts when the second GEMM takes
per-token-quantized FP8 activations. The unfused path writes the BF16/FP16
activation to memory and reads it back to quantize it; this kernel reads the
first GEMM's output once and writes only the FP8 values and the scales.
"""

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import get_fp8_min_max
from vllm.platforms import current_platform
from vllm.triton_utils import tl, tldevice, triton


@triton.jit
def _gelu_tanh_mul(x, mask, D: tl.constexpr, dt: tl.constexpr):
    gate = tl.load(x, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x + D, mask=mask, other=0.0).to(tl.float32)
    # Same expression and roundings as gelu_tanh_and_mul (activation_kernels.cu):
    # the GELU is rounded to the input dtype, multiplied by `up` in fp32 and
    # rounded again, so the scales and FP8 values match the unfused path.
    inner = 0.7978845608028654 * (gate + 0.044715 * (gate * gate * gate))
    gelu = (0.5 * gate * (1.0 + tldevice.tanh(inner))).to(dt).to(tl.float32)
    return (gelu * up).to(dt).to(tl.float32)


@triton.jit
def _quant_store(q_ptr, act, scale, mask, fp8_max):
    # IEEE division like the CUDA quant kernel (Triton's default `/` is not).
    q = tl.div_rn(act, scale[:, None])
    q = tl.minimum(tl.maximum(q, -fp8_max), fp8_max)
    tl.store(q_ptr, q.to(q_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["rows"])
def _gelu_tanh_mul_token_quant_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    rows,
    stride_x,
    stride_q,
    fp8_max,
    min_scale,
    D: tl.constexpr,
    BLOCK_D0: tl.constexpr,
    BLOCK_D1: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    # A row is processed as two power-of-two column tiles, [0, BLOCK_D0) and
    # [BLOCK_D0, BLOCK_D0 + BLOCK_D1), so that a width like 352 = 256 + 96 pads
    # to 384 lanes instead of 512. The exact tanh and IEEE divisions make the
    # kernel instruction-bound, so padded lanes cost time.
    dt: tl.constexpr = x_ptr.dtype.element_ty
    row = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    row_mask = row < rows
    row = row.to(tl.int64)
    x = x_ptr + row[:, None] * stride_x
    q = q_ptr + row[:, None] * stride_q

    cols0 = tl.arange(0, BLOCK_D0)
    mask0 = row_mask[:, None] & (cols0[None, :] < D)
    act0 = _gelu_tanh_mul(x + cols0[None, :], mask0, D, dt)
    amax = tl.max(tl.abs(act0), axis=1)
    if BLOCK_D1 > 0:
        cols1 = BLOCK_D0 + tl.arange(0, BLOCK_D1)
        mask1 = row_mask[:, None] & (cols1[None, :] < D)
        act1 = _gelu_tanh_mul(x + cols1[None, :], mask1, D, dt)
        amax = tl.maximum(amax, tl.max(tl.abs(act1), axis=1))

    # Same scale as dynamic_per_token_scaled_fp8_quant: amax / fp8_max, floored.
    scale = tl.maximum(tl.div_rn(amax, fp8_max), min_scale)
    _quant_store(q + cols0[None, :], act0, scale, mask0, fp8_max)
    if BLOCK_D1 > 0:
        _quant_store(q + cols1[None, :], act1, scale, mask1, fp8_max)
    tl.store(scale_ptr + row, scale, mask=row_mask)


def gelu_tanh_mul_token_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`gelu_tanh(gate) * up` of `x` (`[rows, 2 * d]`, gate then up), quantized
    to FP8 per row.

    Returns the FP8 activation `[rows, d]` and float32 scales `[rows, 1]`
    (dequant = q * scale), as `gelu_tanh_and_mul` followed by
    `scaled_fp8_quant(use_per_token_if_dynamic=True)` would.
    """
    assert x.dim() == 2 and x.stride(-1) == 1 and x.size(1) % 2 == 0
    assert x.dtype in (torch.bfloat16, torch.float16)
    rows, d = x.size(0), x.size(1) // 2
    q = torch.empty((rows, d), dtype=current_platform.fp8_dtype(), device=x.device)
    scale = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
    if rows == 0:
        return q, scale
    _, fp8_max = get_fp8_min_max()
    block_d0 = triton.next_power_of_2(d)
    if block_d0 != d:
        block_d0 //= 2
    block_d1 = triton.next_power_of_2(d - block_d0) if d > block_d0 else 0
    # Launch shapes from a sweep on H100 (d = 176 to 4096): few, wide rows want
    # more threads per row; otherwise about 1K elements per program. Both
    # values must be powers of two.
    tile = triton.next_power_of_2(block_d0 + block_d1)
    if rows <= 1024 and tile >= 512:
        block_t, num_warps = 1, 8
    else:
        block_t = max(1, 1024 // tile)
        num_warps = min(8, max(2, block_t * tile // 512))
    _gelu_tanh_mul_token_quant_kernel[(triton.cdiv(rows, block_t),)](
        x,
        q,
        scale,
        rows,
        x.stride(0),
        q.stride(0),
        fp8_max,
        1.0 / (fp8_max * 512.0),
        D=d,
        BLOCK_D0=block_d0,
        BLOCK_D1=block_d1,
        BLOCK_T=block_t,
        num_warps=num_warps,
    )
    return q, scale
