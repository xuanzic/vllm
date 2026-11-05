# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused GELU-tanh-and-mul + per-token FP8 quant used by `TritonExperts` must
match `gelu_tanh_and_mul` followed by `scaled_fp8_quant(use_per_token_if_dynamic=True)`.
"""

from unittest.mock import patch

import pytest
import torch

from tests.kernels.moe.utils import (
    make_dummy_moe_config,
    make_test_quant_config,
    modular_triton_fused_moe,
)
from vllm import _custom_ops as ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts import triton_moe
from vllm.model_executor.layers.fused_moe.experts.gelu_tanh_mul_token_quant import (
    gelu_tanh_mul_token_quant,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and current_platform.supports_fp8()
        and not current_platform.is_fp8_fnuz()
    ),
    reason="needs a CUDA GPU with float8_e4m3fn support",
)

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def _reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    act = torch.empty((x.size(0), x.size(1) // 2), dtype=x.dtype, device=x.device)
    torch.ops._C.gelu_tanh_and_mul(act, x)
    return ops.scaled_fp8_quant(act, None, use_per_token_if_dynamic=True)


# rows = tokens * top_k. d = 352 is Gemma 4 26B-A4B at TP2 and d = 176 at TP4 (two
# column tiles, the latter not a power of two in total); d = 512 is one tile.
# 1 and 33 rows take the decode launch for d >= 352; 8191 rows take the
# multi-row-per-program launch with a partial last program.
@pytest.mark.parametrize("rows", [1, 33, 8191])
@pytest.mark.parametrize("d", [176, 352, 512])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gelu_tanh_mul_token_quant(rows, d, dtype):
    set_random_seed(0)
    x = torch.randn(rows, 2 * d, device="cuda", dtype=dtype) * 3
    if rows > 1:
        x[0] = 0.0  # all-zero row
        x[1:-1:4] *= 1e-3  # amax below FP8_MAX * min scale: hits the scale floor
        x[-1, : d // 2] = 100.0  # saturated tanh, both signs
        x[-1, d // 2 : d] = -100.0

    q, s = gelu_tanh_mul_token_quant(x)
    ref_q, ref_s = _reference(x)

    assert q.shape == ref_q.shape and q.dtype == ref_q.dtype
    assert s.shape == ref_s.shape == (rows, 1) and s.dtype == torch.float32
    torch.testing.assert_close(s, ref_s, rtol=0, atol=0)
    # Same FP8 values (compared as floats, so -0.0 == +0.0).
    torch.testing.assert_close(q.float(), ref_q.float(), rtol=0, atol=0)
    if rows > 1:
        min_scale = 1.0 / (FP8_MAX * 512.0)
        assert ref_s[0] == min_scale and (q[0].float() == 0).all()  # zero row
        assert (ref_s[1:-1:4] == min_scale).all()  # tiny rows


def test_gelu_tanh_mul_token_quant_no_rows():
    q, s = gelu_tanh_mul_token_quant(
        torch.empty(0, 704, device="cuda", dtype=torch.bfloat16)
    )
    assert q.shape == (0, 352) and s.shape == (0, 1)


def _triton_experts_runner(m: int, n: int, per_act_token_quant: bool):
    e, k, topk = 16, 256, 8
    w1, w2, quant_config = make_test_quant_config(
        e,
        n,
        k,
        torch.bfloat16,
        torch.float8_e4m3fn,
        per_act_token_quant=per_act_token_quant,
    )
    kernel = modular_triton_fused_moe(
        make_dummy_moe_config(
            num_experts=e,
            experts_per_token=topk,
            hidden_dim=k,
            intermediate_size=n,
            max_num_tokens=m,
            activation=MoEActivation.GELU_TANH,
        ),
        quant_config,
    )
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) / 10
    topk_weights = torch.rand(m, topk, device="cuda").softmax(-1)
    topk_ids = torch.rand(m, e, device="cuda").argsort(-1)[:, :topk].to(torch.int32)

    def run(activation: MoEActivation) -> torch.Tensor:
        return kernel.apply(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation=activation,
            global_num_experts=e,
            expert_map=None,
            apply_router_weight_on_input=False,
        )

    return run


@pytest.mark.parametrize("m", [1, 1024])
@pytest.mark.parametrize("n", [352, 512])
@torch.inference_mode()
def test_triton_experts_gelu_tanh_per_token_fp8(m, n, workspace_init):
    """`TritonExperts` takes the fused kernel for GELU-tanh with per-token FP8
    activations, and its output equals the unfused path's."""
    set_random_seed(0)
    run = _triton_experts_runner(m, n, per_act_token_quant=True)
    with set_current_vllm_config(VllmConfig()):
        with patch.object(
            triton_moe, "gelu_tanh_mul_token_quant", wraps=gelu_tanh_mul_token_quant
        ) as fused:
            out = run(MoEActivation.GELU_TANH)
            assert fused.call_count == 1
            run(MoEActivation.SILU)  # other activations keep the unfused path
            assert fused.call_count == 1
        with patch.object(
            triton_moe, "gelu_tanh_mul_token_quant", side_effect=_reference
        ):
            expected = run(MoEActivation.GELU_TANH)

    assert torch.equal(out, expected)


@torch.inference_mode()
def test_triton_experts_gelu_tanh_per_tensor_fp8_unfused(workspace_init):
    """Per-tensor dynamic FP8 activations keep the unfused path."""
    set_random_seed(0)
    run = _triton_experts_runner(33, 352, per_act_token_quant=False)
    with (
        set_current_vllm_config(VllmConfig()),
        patch.object(
            triton_moe, "gelu_tanh_mul_token_quant", wraps=gelu_tanh_mul_token_quant
        ) as fused,
    ):
        run(MoEActivation.GELU_TANH)
    assert fused.call_count == 0
