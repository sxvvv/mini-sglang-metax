# MetaX Port — Technical Notes

This directory contains technical documentation for the MetaX C500/C550 adaptation of mini-sglang.

## Design: platform / device_type decoupling

MetaX vendor PyTorch surfaces GPU devices through `torch.cuda`, making the
device API identical to NVIDIA. However, NVIDIA-compiled extensions (FlashInfer,
`sgl_kernel`, CUDA Graph, PyNCCL) do not run on MACA.

The port separates two things the upstream code treats as one:

```
device_type = "cuda"   # PyTorch API surface — torch.cuda.*, streams, events
platform    = "metax"  # Hardware vendor — controls kernel and backend routing
```

On NVIDIA both values agree. On MetaX they intentionally disagree:
`device_type` stays `"cuda"` so all PyTorch device plumbing works; `platform`
flips to `"metax"` so the code routes around compiled NVIDIA extensions.

```
┌─────────────────────────────────────────────────────────┐
│                  mini-sglang-metax                       │
│                                                          │
│  Python / PyTorch layer  (portable, device_type="cuda") │
│         ↓                                                │
│  platform == "metax" ?                                   │
│     ├── attention  →  torch_native  (eager matmul)       │
│     ├── MoE        →  MetaxMoe      (pure-PyTorch)       │
│     ├── TP comm    →  MCCL via torch.distributed         │
│     └── KV cache   →  index_copy_  (no fused kernel)     │
│                                                          │
│  platform == "nvidia" ?                                  │
│     ├── attention  →  FlashAttention / FlashInfer        │
│     ├── MoE        →  sgl_kernel fused_moe               │
│     ├── TP comm    →  NCCL / PyNCCL                      │
│     └── KV cache   →  store_cache CUDA kernel            │
└─────────────────────────────────────────────────────────┘
```

## MoE Backend: MetaxMoe (M1)

SGLang's default `fused_moe` backend calls into `sgl_kernel` (NVIDIA-compiled).
`MetaxMoe` is a pure-PyTorch drop-in replacement registered as `"metax"` in the
backend registry. The engine selects it automatically when `platform == "metax"`.

Key implementation choices:
- Router: `torch.softmax` + `torch.topk` (replaces `sgl_kernel.topk_softmax`)
- Expert loop: gather → gate-up proj → SiLU → down proj → scatter-add
- Supports `silu`/`gelu`, `renormalize`, `apply_router_weight_on_input`
- 20 CPU unit tests — no GPU required to verify correctness

See [`python/minisgl/moe/metax.py`](../../python/minisgl/moe/metax.py) and
[`tests/moe/test_metax_moe.py`](../../tests/moe/test_metax_moe.py).

## Optimization directions

See [`step35_optimization_directions.md`](step35_optimization_directions.md)
for the current performance analysis and next-step priorities on 30B MoE models.
