from __future__ import annotations

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even
from minisgl.utils.platform import is_metax_platform

from .base import BaseKVCachePool


class MHAKVCache(BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        # nhd layout — the natural CUDA / MetaX paged-KV shape:
        # (2, num_layers, num_pages, page_size, local_kv_heads, head_dim).
        self._kv_buffer = torch.empty(
            (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._page_size = page_size
        self._local_kv_heads = local_kv_heads
        self._head_dim = head_dim
        self._num_layers = num_layers
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._device = device
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[index]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        # device_type=cuda vs platform=metax split: NVIDIA CUDA uses the fused
        # ``store_cache`` kernel from the CUDA-only ``minisgl.kernel`` package;
        # MetaX has no such compiled artefact, so it scatters with pure PyTorch
        # ``index_copy_`` instead.
        if not is_metax_platform():
            from minisgl.kernel import store_cache

            store_cache(
                k_cache=self._k_buffer[layer_id].view(self._storage_shape),
                v_cache=self._v_buffer[layer_id].view(self._storage_shape),
                indices=out_loc,
                k=k,
                v=v,
            )
        else:
            k_cache = self._k_buffer[layer_id].view(self._storage_shape)
            v_cache = self._v_buffer[layer_id].view(self._storage_shape)
            locations = out_loc.long()
            values_shape = (-1, self._local_kv_heads, self._head_dim)
            k_cache.index_copy_(0, locations, k.reshape(values_shape))
            v_cache.index_copy_(0, locations, v.reshape(values_shape))

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
