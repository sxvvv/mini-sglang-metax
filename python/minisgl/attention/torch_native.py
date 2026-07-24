from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
import torch.nn.functional as F
from minisgl.core import Batch, get_global_ctx

from .base import BaseAttnBackend, BaseAttnMetadata

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass
class TorchNativeMetadata(BaseAttnMetadata):
    last_indices: torch.Tensor

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.last_indices[:bs]


def _repeat_kv_heads(x: torch.Tensor, num_qo_heads: int) -> torch.Tensor:
    num_kv_heads = x.shape[1]
    if num_qo_heads == num_kv_heads:
        return x
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError(
            f"query heads ({num_qo_heads}) must be divisible by KV heads ({num_kv_heads})"
        )
    return x.repeat_interleave(num_qo_heads // num_kv_heads, dim=1)


def torch_attention_for_request(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cached_len: int,
    scale: float,
) -> torch.Tensor:
    """Reference attention for one request with a cached-prefix-aware mask."""
    q_len, num_qo_heads, _ = q.shape
    kv_len = k.shape[0]
    if cached_len + q_len != kv_len:
        raise ValueError(
            "cached prefix plus query length must equal the visible KV length: "
            f"{cached_len} + {q_len} != {kv_len}"
        )

    k = _repeat_kv_heads(k, num_qo_heads)
    v = _repeat_kv_heads(v, num_qo_heads)
    q_4d = q.transpose(0, 1).unsqueeze(0)
    k_4d = k.transpose(0, 1).unsqueeze(0)
    v_4d = v.transpose(0, 1).unsqueeze(0)

    query_positions = cached_len + torch.arange(q_len, device=q.device)
    key_positions = torch.arange(kv_len, device=q.device)
    allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    output = F.scaled_dot_product_attention(
        q_4d,
        k_4d,
        v_4d,
        attn_mask=allowed,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )
    return output.squeeze(0).transpose(0, 1).contiguous()


class TorchNativeAttentionBackend(BaseAttnBackend):
    """Portable eager correctness backend for the MetaX bring-up gates."""

    def __init__(self, config: ModelConfig) -> None:
        self.kvcache = get_global_ctx().kv_cache
        self.scale = config.head_dim**-0.5

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
    ) -> torch.Tensor:
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        page_table = get_global_ctx().page_table
        k_flat = self.kvcache.k_cache(layer_id).view(-1, k.shape[1], k.shape[2])
        v_flat = self.kvcache.v_cache(layer_id).view(-1, v.shape[1], v.shape[2])

        outputs = []
        q_offset = 0
        for req in batch.padded_reqs:
            q_len = req.extend_len
            q_req = q[q_offset : q_offset + q_len]
            locations = page_table[req.table_idx, : req.device_len].long()
            k_req = k_flat.index_select(0, locations)
            v_req = v_flat.index_select(0, locations)
            outputs.append(
                torch_attention_for_request(
                    q_req,
                    k_req,
                    v_req,
                    cached_len=req.cached_len,
                    scale=self.scale,
                )
            )
            q_offset += q_len

        if q_offset != q.shape[0]:
            raise RuntimeError(
                f"attention consumed {q_offset} query tokens, received {q.shape[0]}"
            )
        return torch.cat(outputs, dim=0)

    def prepare_metadata(self, batch: Batch) -> None:
        lengths = torch.tensor(
            [req.extend_len for req in batch.padded_reqs],
            dtype=torch.int64,
            device=get_global_ctx().page_table.device,
        )
        batch.attn_metadata = TorchNativeMetadata(last_indices=lengths.cumsum(0) - 1)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        if bs_list:
            raise RuntimeError("torch_native attention supports eager execution only")

    def prepare_for_capture(self, batch: Batch) -> None:
        raise RuntimeError("torch_native attention does not support CUDA Graph capture")

    def prepare_for_replay(self, batch: Batch) -> None:
        raise RuntimeError("torch_native attention does not support CUDA Graph replay")


__all__ = [
    "TorchNativeAttentionBackend",
    "TorchNativeMetadata",
    "torch_attention_for_request",
]
