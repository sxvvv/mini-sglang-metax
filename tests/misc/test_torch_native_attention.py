from __future__ import annotations

import torch

from minisgl.attention.torch_native import torch_attention_for_request


def test_cached_prefix_mask_uses_prefix_and_prior_query_tokens() -> None:
    q = torch.zeros((2, 1, 1), dtype=torch.float32)
    k = torch.zeros((3, 1, 1), dtype=torch.float32)
    v = torch.tensor([[[1.0]], [[2.0]], [[4.0]]])

    output = torch_attention_for_request(q, k, v, cached_len=1, scale=1.0)

    expected = torch.tensor([[[1.5]], [[7.0 / 3.0]]])
    torch.testing.assert_close(output, expected)


def test_grouped_query_heads_repeat_kv_heads() -> None:
    q = torch.zeros((1, 4, 2), dtype=torch.float32)
    k = torch.zeros((1, 2, 2), dtype=torch.float32)
    v = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    output = torch_attention_for_request(q, k, v, cached_len=0, scale=1.0)

    assert output.shape == (1, 4, 2)
    torch.testing.assert_close(output[0, 0], v[0, 0])
    torch.testing.assert_close(output[0, 1], v[0, 0])
    torch.testing.assert_close(output[0, 2], v[0, 1])
    torch.testing.assert_close(output[0, 3], v[0, 1])


def test_invalid_prefix_shape_is_rejected() -> None:
    q = torch.zeros((2, 1, 4))
    k = torch.zeros((4, 1, 4))
    v = torch.zeros_like(k)

    try:
        torch_attention_for_request(q, k, v, cached_len=1, scale=0.5)
    except ValueError as exc:
        assert "cached prefix plus query length" in str(exc)
    else:
        raise AssertionError("invalid cached-prefix shape was accepted")
