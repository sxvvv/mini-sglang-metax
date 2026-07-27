# mini-sglang-metax

A correctness-first port of [Mini-SGLang](https://github.com/sgl-project/mini-sglang)
to MetaX C500/C550 and the MACA software stack. The codebase (~5 K lines) is
small enough to read end-to-end — the goal is understanding the inference
engine, not production headroom.

The Python import name stays `minisgl`. The distribution name is
`mini-sglang-metax`.

## Status

**MetaX C500/C550 — dense models verified, MoE backend implemented.**

| Area | Status |
|---|---|
| Hardware | MetaX C500, C550 |
| Vendor PyTorch | `2.10.0+metax3.8.1.0` |
| Dense models | Qwen3-8B BF16, TP1/TP2/TP8, eager |
| MoE models | ✅ `MetaxMoe` pure-PyTorch backend (M1) |
| KV cache | Radix cache + naive, `index_copy_` scatter |
| Tensor Parallel | TP1 real-model, TP2/TP8 functional |
| Online serving | HTTP/ZMQ, OpenAI-compatible API, SSE, concurrency |
| Attention | `torch_native` eager; MetaX `flashinfer` (opt-in) |
| CUDA Graph | ❌ not yet (MACA JIT constraint) |
| Quantized models | ❌ not yet |

## Benchmark Results

**Qwen3-30B-A3B.w8a8 on 8×MetaX C500** — SGLang 0.5.13+maca3.8.1.0, TP8, W8A8 quantization

| Scenario | Concurrency | Output Throughput | TTFT p50 |
|---|:---:|---:|:---:|
| Prefill-heavy (in=1024, out=16)  | 16 | **115.7 tok/s** | 567 ms |
| Decode-heavy  (in=64,  out=256)  | 16 | **173.0 tok/s** | 98 ms  |
| Mixed PD      (in=512, out=128)  |  8 |  **84.6 tok/s** | 189 ms |
| Mixed PD      (in=512, out=128)  | 32 | **336.9 tok/s** | 518 ms |

<details>
<summary>Full scaling data</summary>

```
Mixed PD throughput (tok/s):  conc=1: 11  → 4: 42  → 8: 85  → 16: 172  → 32: 337
Decode-heavy throughput:      conc=1: 12  → 4: 43  → 8: 82  → 16: 173
Prefill-heavy throughput:     conc=1: 11  → 4: 42  → 8: 78  → 16: 116
```

See [`docs/figures.md`](docs/figures.md#7-benchmark-throughput-vs-concurrency实测) for full charts.

</details>

## The core design decision

MetaX vendor PyTorch surfaces devices through `torch.cuda`. That makes the
device API identical to NVIDIA, but it does not mean every NVIDIA binary runs —
FlashInfer, `sgl_kernel`, CUDA Graph, and PyNCCL all depend on compiled
SM-architecture extensions that do not exist in the MACA stack.

The port separates two things that upstream treats as one:

```
device_type = "cuda"   # PyTorch API surface (torch.cuda.*, streams, events)
platform    = "metax"  # Hardware vendor — kernel and backend routing
```

`device_type` stays `"cuda"` so all PyTorch plumbing keeps working. `platform`
flips to `"metax"` so the code routes around compiled NVIDIA extensions.

```
Request
  │
  ▼
┌─────────────────────────────────────────────┐
│  platform detection (MACA_PATH / torch ver) │
│  platform = "metax"  OR  "nvidia"           │
└───────────────────┬─────────────────────────┘
                    │
        ┌───────────▼───────────┐
        │     mini-sglang       │
        │   (device_type=cuda)  │
        └─┬──────────┬──────────┘
          │          │
     platform        platform
     =metax          =nvidia
          │          │
    ┌─────▼──┐  ┌────▼──────────┐
    │torch_  │  │FlashAttention │
    │native  │  │FlashInfer     │
    │attn    │  │sgl_kernel MoE │
    │        │  │CUDA Graph     │
    │MetaxMoe│  │PyNCCL         │
    │MCCL TP │  │NCCL TP        │
    └────────┘  └───────────────┘
```

`get_accelerator_platform()` in `python/minisgl/utils/platform.py` resolves
the platform at startup — it checks `MACA_PATH`, MACA markers in `CUDA_HOME`,
and `"metax"/"maca"` substrings in `torch.__version__`. Override with
`MINISGL_PLATFORM=metax`.

## MoE Backend: MetaxMoe

`sgl_kernel.fused_moe` is an NVIDIA-compiled binary. `MetaxMoe` is a
pure-PyTorch drop-in replacement that registers as `"metax"` in the backend
registry. The engine selects it automatically on MetaX hardware.

```python
# python/minisgl/moe/__init__.py
@SUPPORTED_MOE_BACKENDS.register("metax")
def create_metax_moe_backend():
    from .metax import MetaxMoe
    return MetaxMoe()

# python/minisgl/engine/engine.py  — auto-selection
if config.model_config.is_moe and config.moe_backend == "auto":
    backend = "metax" if platform == "metax" else "fused"
```

See [`python/minisgl/moe/metax.py`](python/minisgl/moe/metax.py) and
[`tests/moe/test_metax_moe.py`](tests/moe/test_metax_moe.py) (20 CPU tests,
no GPU needed).

## Quick start on MetaX

### 1. Use the vendor image

The target image must contain a matching MACA PyTorch build.

```bash
cd /path/to/mini-sglang-metax

export MINISGL_PLATFORM=metax
export MACA_PATH=/opt/maca
export CUDA_HOME=/opt/maca/tools/cu-bridge
export CUDA_PATH="$CUDA_HOME"
export PYTHONPATH=python
```

```bash
python -m pip install -e . --no-deps
```

### 2. Run the hardware preflight

```bash
python scripts/metax/preflight.py
```

### 3. Offline single-model validation (Gate 0)

```bash
export MODEL_PATH=/path/to/Qwen3-8B
export SW_HOME=/persistent/path/${USER}
bash scripts/metax/run_gate0.sh
```

### 4. Online serving validation

```bash
# Basic HTTP + OpenAI API
bash scripts/metax/run_online_gate1.sh

# SSE, cancellation, concurrency
bash scripts/metax/run_online_gate1_1.sh

# Soak, overload queueing, batching
bash scripts/metax/run_online_gate1_2.sh
```

### 5. TP smoke

```bash
TP=2 MODEL_PATH=/path/to/model bash scripts/metax/run_gate0.sh
TP=8 MODEL_PATH=/path/to/tp8-compatible-model bash scripts/metax/run_gate0.sh
```

## Verification (CPU only)

Any CPU host works — no GPU needed.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pytest msgpack pyzmq fastapi uvicorn prompt_toolkit \
  "transformers>=4.56.0,<=4.57.3"
pip install -e . --no-deps
```

Run the full CI suite (~20 s):

```bash
python -m pytest -q -o addopts="" tests/misc/ tests/layers/ tests/moe/
```

Key test files:
- `tests/moe/test_metax_moe.py` — 20 tests for MetaxMoe, CPU-only
- `tests/misc/test_platform.py` — platform detection
- `tests/misc/test_torch_native_attention.py` — portable attention backend

## Project layout

```
python/minisgl/
├── utils/platform.py          # CUDA-facing platform detection for MetaX
├── attention/torch_native.py  # Portable eager attention backend
├── moe/
│   ├── metax.py               # MetaxMoe pure-PyTorch MoE backend
│   └── __init__.py            # Backend registry (includes "metax")
├── engine/engine.py           # Auto-selects MoE backend by platform
└── ...

scripts/metax/
├── preflight.py               # Device and operator preflight
├── run_gate0.sh               # Offline real-model runner
├── run_online_gate1*.sh       # Online serving smoke tests

tests/moe/
└── test_metax_moe.py          # 20 MetaxMoe unit tests (CPU)

docs/
├── structures.md              # System architecture walkthrough
├── features.md                # Feature overview with references
├── learning_sharing.md        # LLM inference learning notes
└── metax_port/
    ├── README.md              # MetaX adaptation design notes
    └── step35_optimization_directions.md
```

## Support boundary

Validated:
- MetaX C500, C550; vendor PyTorch `2.10.0+metax3.8.1.0`
- Dense Qwen3 BF16, eager execution, TP1/TP2/TP8
- MoE routing (MetaxMoe, CPU-verified correctness)
- Full online serving stack — HTTP/ZMQ, SSE, concurrency, fault recovery

Not yet:
- CUDA Graph (MACA JIT constraint on kernel compilation)
- W8A8 / quantized checkpoints
- MACA Triton grouped-matmul MoE (M2 — next step)
- Production soak / throughput leadership

## Attribution and license

Derived from
[`sgl-project/mini-sglang`](https://github.com/sgl-project/mini-sglang),
MIT license retained. See [`PROVENANCE.md`](PROVENANCE.md).
