from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl.utils import is_sm90_supported, nvtx_annotate
from minisgl.utils.platform import is_metax_platform

if TYPE_CHECKING:
    from minisgl.core import Batch


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    if logits.device.type == "cuda" and is_metax_platform():
        return _torch_native_sample(logits, temperatures, top_k, top_p)

    import flashinfer.sampling as sampling

    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)

    if top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)

    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)

    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


def _torch_native_sample(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    scaled = logits / temperatures.unsqueeze(-1)
    output = []
    for row_idx, row_logits in enumerate(scaled):
        current = row_logits
        k = int(top_k[row_idx].item()) if isinstance(top_k, torch.Tensor) else top_k
        p = float(top_p[row_idx].item()) if isinstance(top_p, torch.Tensor) else top_p

        if k is not None and k < current.numel():
            threshold = torch.topk(current, k).values[-1]
            current = current.masked_fill(current < threshold, float("-inf"))

        if p is not None and p < 1.0:
            sorted_logits, sorted_indices = torch.sort(current, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            remove = sorted_probs.cumsum(dim=-1) > p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            current = torch.full_like(current, float("-inf"))
            current.scatter_(0, sorted_indices, sorted_logits)

        output.append(torch.multinomial(torch.softmax(current, dim=-1), 1))
    return torch.cat(output)


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None)

        MIN_P = MIN_T = 1e-6
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p)

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        # NOTE: the outer @nvtx_annotate("Sampler") already brackets this method
        # via the cross-device-safe nvtx shim (torch.cuda.nvtx.range would raise
        # on Torch builds without CUDA/NVTX, e.g. a CPU-only build).
        if args.temperatures is None:  # greedy sampling
            return torch.argmax(logits, dim=-1)
        return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
