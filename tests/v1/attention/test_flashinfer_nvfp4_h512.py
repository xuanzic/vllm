# SPDX-License-Identifier: Apache-2.0

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
