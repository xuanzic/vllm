# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import pytest
import torch


class _TestWorkspace:
    def get_output(self, pages, num_heads, page_size, head_dim, device):
        return torch.empty(
            (pages + 1, 2, num_heads, page_size, head_dim),
            dtype=torch.float32,
            device=device,
        )

    def get_block_table(self, rows, width, dtype, device):
        return torch.empty((rows, width), dtype=dtype, device=device)


def test_materialize_active_nvfp4_pages_bounds_and_remaps(monkeypatch):
    from flashinfer.quantization import kernels
    from vllm.utils.torch_utils import nvfp4_kv_cache_full_dim
    from vllm.v1.attention.backends import flashinfer as backend

    head_size = 64
    page_size = 4
    num_physical_pages = 16
    packed_dim = nvfp4_kv_cache_full_dim(head_size)
    packed_k = torch.empty(
        num_physical_pages, 1, page_size, packed_dim, dtype=torch.uint8
    )
    packed_v = torch.empty_like(packed_k)
    for page_id in range(num_physical_pages):
        packed_k[page_id].fill_(page_id)
        packed_v[page_id].fill_(page_id + 32)

    def fake_materialize(
        k_data,
        v_data,
        _k_scales,
        _v_scales,
        physical_page_ids,
        *,
        output,
    ):
        output.zero_()
        ids = physical_page_ids.long()
        k_values = k_data.index_select(0, ids)[..., :1].float()
        v_values = v_data.index_select(0, ids)[..., :1].float()
        output[1:, 0].copy_(k_values.expand_as(output[1:, 0]))
        output[1:, 1].copy_(v_values.expand_as(output[1:, 1]))
        return output

    monkeypatch.setattr(
        backend,
        "_get_nvfp4_context_workspace",
        lambda *_args, **_kwargs: _TestWorkspace(),
    )
    monkeypatch.setattr(
        kernels,
        "nvfp4_kv_materialize_active_pages_qmul4",
        fake_materialize,
    )

    block_tables = torch.tensor(
        [[3, 7, 0, 0, 0], [5, 9, 11, 0, 0]], dtype=torch.int32
    )
    (compact_k, compact_v), compact_tables = (
        backend._materialize_active_nvfp4_pages_to_fp8(
            packed_k,
            packed_v,
            block_tables,
            max_seq_len=12,
            page_counts=(2, 3),
        )
    )

    torch.testing.assert_close(
        compact_tables,
        torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.int32),
    )
    assert compact_k.shape == (6, 1, page_size, head_size)
    assert compact_v.shape == (6, 1, page_size, head_size)
    torch.testing.assert_close(
        compact_k[1:, 0, 0, 0], torch.tensor([3, 7, 5, 9, 11]).float()
    )
    torch.testing.assert_close(
        compact_v[1:, 0, 0, 0], torch.tensor([35, 39, 37, 41, 43]).float()
    )


@pytest.mark.parametrize(
    ("page_counts", "message"),
    [((1,), "one entry"), ((0, 1), "must be in"), ((4, 1), "must be in")],
)
def test_materialize_active_nvfp4_pages_rejects_invalid_page_counts(
    page_counts, message
):
    from vllm.v1.attention.backends import flashinfer as backend

    packed = torch.zeros((2, 1, 4, 36), dtype=torch.uint8)
    block_tables = torch.zeros((2, 3), dtype=torch.int32)
    with pytest.raises(ValueError, match=message):
        backend._materialize_active_nvfp4_pages_to_fp8(
            packed,
            packed,
            block_tables,
            max_seq_len=12,
            page_counts=page_counts,
        )


def test_prims_ts_max_kv_len_bucket():
    from vllm.v1.attention.backends.flashinfer import _prims_ts_max_kv_len_bucket

    assert _prims_ts_max_kv_len_bucket(1, 64) == 64
    assert _prims_ts_max_kv_len_bucket(65, 64) == 128
    assert _prims_ts_max_kv_len_bucket(262144, 64) == 262144


@pytest.mark.parametrize(
    ("qo_indptr", "num_decode_tokens", "allow_zero_padding", "expected"),
    [
        ([0, 1, 2], 2, False, 1),
        ([0, 4, 8], 8, False, 4),
        ([0, 4, 8, 8], 12, True, 4),
    ],
)
def test_get_uniform_prims_ts_decode_query_len(
    qo_indptr, num_decode_tokens, allow_zero_padding, expected
):
    from vllm.v1.attention.backends.flashinfer import (
        _get_uniform_prims_ts_decode_query_len,
    )

    assert (
        _get_uniform_prims_ts_decode_query_len(
            torch.tensor(qo_indptr, dtype=torch.int32),
            len(qo_indptr) - 1,
            num_decode_tokens,
            allow_zero_padding=allow_zero_padding,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("qo_indptr", "num_decode_tokens", "allow_zero_padding"),
    [
        ([0, 4, 8, 8], 12, False),
        ([0, 4, 6], 6, False),
        ([0, 4, 6], 6, True),
        ([0, 4, 8], 7, False),
    ],
)
def test_get_uniform_prims_ts_decode_query_len_rejects_unsupported_shapes(
    qo_indptr, num_decode_tokens, allow_zero_padding
):
    from vllm.v1.attention.backends.flashinfer import (
        _get_uniform_prims_ts_decode_query_len,
    )

    with pytest.raises(NotImplementedError):
        _get_uniform_prims_ts_decode_query_len(
            torch.tensor(qo_indptr, dtype=torch.int32),
            len(qo_indptr) - 1,
            num_decode_tokens,
            allow_zero_padding=allow_zero_padding,
        )


def test_prepare_prims_ts_decode_metadata_reuses_unpadded_storage():
    from vllm.v1.attention.backends.flashinfer import (
        _prepare_prims_ts_decode_metadata,
    )

    indptr = torch.tensor([0, 2, 5, 6], dtype=torch.int32)
    indices = torch.tensor([3, 7, 5, 9, 11, 13], dtype=torch.int32)
    last_page_len = torch.tensor([32, 17, 8], dtype=torch.int32)
    block_tables = torch.tensor(
        [[3, 7, 0], [5, 9, 11], [13, 0, 0]], dtype=torch.int32
    )

    decode_indptr, decode_indices, decode_last_page_len = (
        _prepare_prims_ts_decode_metadata(
            [2, 3],
            indptr,
            indices,
            last_page_len,
            last_page_len.numpy(),
            block_tables,
            reuse_existing=True,
        )
    )

    assert decode_indptr.data_ptr() == indptr.data_ptr()
    assert decode_indices.data_ptr() == indices.data_ptr()
    assert decode_last_page_len.data_ptr() == last_page_len.data_ptr()
    torch.testing.assert_close(
        decode_indptr, torch.tensor([0, 2, 5], dtype=torch.int32)
    )
    torch.testing.assert_close(
        decode_indices, torch.tensor([3, 7, 5, 9, 11], dtype=torch.int32)
    )
    torch.testing.assert_close(
        decode_last_page_len, torch.tensor([32, 17], dtype=torch.int32)
    )


def test_prepare_prims_ts_decode_metadata_pads_empty_rows():
    from vllm.v1.attention.backends.flashinfer import (
        _prepare_prims_ts_decode_metadata,
    )

    indptr = torch.tensor([0, 2, 2], dtype=torch.int32)
    indices = torch.tensor([3, 7], dtype=torch.int32)
    last_page_len = torch.tensor([32, 0], dtype=torch.int32)
    block_tables = torch.tensor([[3, 7], [0, 0]], dtype=torch.int32)

    decode_indptr, decode_indices, decode_last_page_len = (
        _prepare_prims_ts_decode_metadata(
            [2, 0],
            indptr,
            indices,
            last_page_len,
            np.array([32, 0], dtype=np.int32),
            block_tables,
            reuse_existing=True,
        )
    )

    torch.testing.assert_close(
        decode_indptr, torch.tensor([0, 2, 3], dtype=torch.int32)
    )
    torch.testing.assert_close(
        decode_indices, torch.tensor([3, 7, 3], dtype=torch.int32)
    )
    torch.testing.assert_close(
        decode_last_page_len, torch.tensor([32, 1], dtype=torch.int32)
    )


def test_prepare_prims_ts_decode_metadata_materializes_for_dynamic_plan():
    from vllm.v1.attention.backends.flashinfer import (
        _prepare_prims_ts_decode_metadata,
    )

    indptr = torch.tensor([0, 2], dtype=torch.int32)
    indices = torch.tensor([3, 7], dtype=torch.int32)
    last_page_len = torch.tensor([32], dtype=torch.int32)
    block_tables = torch.tensor([[3, 7]], dtype=torch.int32)

    decode_indptr, decode_indices, decode_last_page_len = (
        _prepare_prims_ts_decode_metadata(
            [2],
            indptr,
            indices,
            last_page_len,
            np.array([32], dtype=np.int32),
            block_tables,
            reuse_existing=False,
        )
    )

    assert decode_indptr.data_ptr() != indptr.data_ptr()
    assert decode_indices.data_ptr() != indices.data_ptr()
    assert decode_last_page_len.data_ptr() != last_page_len.data_ptr()
    torch.testing.assert_close(
        decode_indptr, torch.tensor([0, 2], dtype=torch.int32)
    )
    torch.testing.assert_close(
        decode_indices, torch.tensor([3, 7], dtype=torch.int32)
    )
    torch.testing.assert_close(
        decode_last_page_len, torch.tensor([32], dtype=torch.int32)
    )


def test_update_prims_ts_paged_kv_metadata_preserves_storage():
    from vllm.v1.attention.backends.flashinfer import (
        _update_prims_ts_paged_kv_metadata,
    )

    class Wrapper:
        _kv_prefix_mode = "dynamic"
        _kv_lengths_mode = "dynamic"
        _page_size = 64
        _paged_kv_indptr = torch.empty(3, dtype=torch.int32)
        _paged_kv_indices = torch.empty(8, dtype=torch.int32)
        _paged_kv_last_page_len = torch.empty(2, dtype=torch.int32)
        _seq_lens = torch.empty(2, dtype=torch.int32)

    wrapper = Wrapper()
    addresses = (
        wrapper._paged_kv_indptr.data_ptr(),
        wrapper._paged_kv_indices.data_ptr(),
        wrapper._paged_kv_last_page_len.data_ptr(),
        wrapper._seq_lens.data_ptr(),
    )
    _update_prims_ts_paged_kv_metadata(
        wrapper,
        torch.tensor([0, 2, 5], dtype=torch.int32),
        torch.tensor([7, 3, 9, 4, 1], dtype=torch.int32),
        torch.tensor([32, 17], dtype=torch.int32),
    )

    assert addresses == (
        wrapper._paged_kv_indptr.data_ptr(),
        wrapper._paged_kv_indices.data_ptr(),
        wrapper._paged_kv_last_page_len.data_ptr(),
        wrapper._seq_lens.data_ptr(),
    )
    torch.testing.assert_close(
        wrapper._paged_kv_indptr, torch.tensor([0, 2, 5], dtype=torch.int32)
    )
    torch.testing.assert_close(
        wrapper._paged_kv_indices[:5],
        torch.tensor([7, 3, 9, 4, 1], dtype=torch.int32),
    )
    torch.testing.assert_close(
        wrapper._seq_lens, torch.tensor([96, 145], dtype=torch.int32)
    )


@pytest.mark.parametrize(
    ("physical_nvfp4", "head_dim", "expected"),
    [
        (True, 256, True),
        (True, 512, True),
        (True, 128, False),
        (False, 256, False),
        (False, 512, False),
    ],
)
def test_prims_ts_nvfp4_decode_selection(
    physical_nvfp4: bool, head_dim: int, expected: bool
):
    from vllm.v1.attention.backends.flashinfer import (
        _use_prims_ts_nvfp4_decode,
    )

    assert _use_prims_ts_nvfp4_decode(physical_nvfp4, head_dim) is expected


def test_enable_prims_ts_nvfp4_aot_plan(monkeypatch):
    from vllm.v1.attention.backends.flashinfer import (
        _enable_prims_ts_nvfp4_aot_plan,
    )

    env_name = "FLASHINFER_PRIMS_TS_FORCE_DYNAMIC_PLAN"
    monkeypatch.delenv(env_name, raising=False)
    _enable_prims_ts_nvfp4_aot_plan()
    assert os.environ[env_name] == "1"
