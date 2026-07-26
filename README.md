# mini-sglang-metax

A correctness-first port of Mini-SGLang to MetaX C500/C550 and the MACA
software stack. The codebase (~5 K lines) is small enough to read end-to-end,
which is the point — the goal is understanding the inference engine, not
production headroom.

The Python import name stays `minisgl`. The distribution name is
`mini-sglang-metax`.

See [`PROVENANCE.md`](PROVENANCE.md) for source lineage and attribution, and
[`docs/metax_port/open_source_readiness.md`](docs/metax_port/open_source_readiness.md)
for public-release scope and remaining blockers.

## Status

**MetaX technical preview — offline Gate 0 and bounded online Gate 1.2
passed on real hardware.**

| Area | Result |
| --- | --- |
| Hardware | MetaX C500, C550 via vendor `torch.cuda` |
| Vendor PyTorch | `2.10.0+metax3.8.1.0` |
| Real model | Qwen3-8B BF16, TP1, eager, `torch_native` attention |
| Repeated requests | Two deterministic greedy runs in one `LLM` instance |
| KV cache invariant | `512/512` available after each real-model request |
| Multi-card transport | TP2 MCCL all-reduce |
| Framework TP | Synthetic Qwen3 TP2/TP8 end-to-end |
| Online — Gate 1.2 | 24/24 bounded requests, overload queueing, fault recovery, scheduler batching on C550 |
| Vendor attention | MetaX `flashinfer` TP1 Gate 1 (signature-fixed; explicit opt-in only) |

Evidence: [`docs/metax_port/gate0_verdict.md`](docs/metax_port/gate0_verdict.md),
[`docs/metax_port/online_gate1_verdict.md`](docs/metax_port/online_gate1_verdict.md),
[`docs/metax_port/online_gate1_2_verdict.md`](docs/metax_port/online_gate1_2_verdict.md).

## The core design decision

MetaX vendor PyTorch surfaces devices through `torch.cuda`. That makes
the device API identical to NVIDIA, but it does not mean every NVIDIA binary
runs — FlashInfer, `sgl_kernel`, CUDA Graph, and PyNCCL all depend on compiled
SM-architecture extensions that do not exist in the MACA stack.

The project separates two things that the upstream code treats as one:

```
device_type = "cuda"   # the PyTorch API surface (torch.cuda.*, streams, events)
platform    = "metax"  # the hardware vendor, used for kernel and backend routing
```

`device_type` drives everything that goes through `torch.cuda.*` directly:
stream creation, event recording, memory queries, and distributed backend
selection. `platform` drives which implementation runs at each operator site.
On NVIDIA both agree. On MetaX they disagree on purpose: `device_type` stays
`"cuda"` so the whole PyTorch device plumbing keeps working; `platform` flips
to `"metax"` so the code routes around compiled NVIDIA extensions.

`get_accelerator_platform()` in `python/minisgl/utils/platform.py` resolves
the platform at startup — it checks `MACA_PATH`, MACA markers in `CUDA_HOME` /
`CUDA_PATH`, and `"metax"` / `"maca"` substrings in `torch.__version__`. Set
`MINISGL_PLATFORM=metax` to override.

**What this enables on MetaX today:**

- Eager execution with `torch_native` attention (no compiled attention kernel)
- KV cache scatter via `index_copy_` (no fused `store_cache` CUDA kernel)
- Pure-PyTorch NeoX RoPE with fp32 mid-calculation
- TP1 deployment with zero `torch.distributed` bootstrap (no gloo sidecar)
- TP2/TP8 via vendor `torch.distributed` / MCCL
- BF16 dense Qwen models (MoE not yet supported)

## Quick start on MetaX

### 1. Use the vendor image

The target image must already contain a matching MACA PyTorch build. Do not
replace the vendor `torch` wheel with a public PyPI wheel.

```bash
cd /path/to/mini-sglang-metax

export MINISGL_PLATFORM=metax
export MACA_PATH=/opt/maca
export CUDA_HOME=/opt/maca/tools/cu-bridge
export CUDA_PATH="$CUDA_HOME"
export CUCC_PATH="$CUDA_HOME"
export PYTHONPATH=python
```

For editable installation in a vendor image, preserve the existing PyTorch:

```bash
python -m pip install -e . --no-deps
```

### 2. Run the hardware preflight

```bash
python scripts/metax/preflight.py
```

The preflight checks platform detection, device visibility, BF16 GEMM, and the
portable attention micro-path.

### 3. Reproduce the validated real-model Gate 0 case

```bash
export MODEL_PATH=/path/to/Qwen3-8B
export SW_HOME=/persistent/path/${USER}

bash scripts/metax/run_gate0.sh
```

The default Gate 0 run is TP1, eager BF16, four greedy output tokens, two
requests in one process, and 512 KV-cache pages. Results are written under:

```text
${SW_HOME}/results/mini-sglang-metax/<date>/
```

### 4. Reproduce the bounded online Gate 1 case

The online path requires the declared `msgpack`, `pyzmq`, FastAPI, and Uvicorn
dependencies. Preserve the vendor PyTorch installation when adding them.

```bash
export MODEL_PATH=/path/to/Qwen3-8B
export SW_HOME=/persistent/path/${USER}

bash scripts/metax/run_online_gate1.sh
```

The script starts an isolated process group, waits for backend readiness,
checks `/v1/models`, sends two deterministic non-streaming OpenAI chat
requests, validates the JSON responses, and always tears down the service.

### 5. Reproduce streaming, cancellation, and concurrent Gate 1.1

```bash
export MODEL_PATH=/path/to/Qwen3-8B
export SW_HOME=/persistent/path/${USER}

bash scripts/metax/run_online_gate1_1.sh
```

Gate 1.1 additionally validates the complete SSE sequence, deliberately
disconnects a live stream and requires its AbortAck, sends four simultaneous
ragged requests, verifies a final recovery request, and checks clean shutdown.

### 6. Reproduce bounded soak, queueing, and batch observability Gate 1.2

```bash
export MODEL_PATH=/path/to/Qwen3-8B
export SW_HOME=/persistent/path/${USER}

bash scripts/metax/run_online_gate1_2.sh
```

The default profile runs 3 rounds of 8 simultaneous requests against a server
limited to 2 running requests, injects malformed input and a live disconnect,
then requires a recovery request and server-side multi-request batch evidence.

For the bounded experimental vendor-attention probe:

```bash
ATTENTION_BACKEND=fi RESULT_PREFIX=online_gate1_fi_probe \
  bash scripts/metax/run_online_gate1.sh
```

### 7. Run a bounded TP smoke

Use a model whose query and KV head counts are compatible with the requested
tensor-parallel size.

```bash
TP=2 MODEL_PATH=/path/to/model bash scripts/metax/run_gate0.sh
TP=8 MODEL_PATH=/path/to/tp8-compatible-model bash scripts/metax/run_gate0.sh
```

## Validated real-model result

The C500 host ran Qwen3-8B BF16 from a read-only mounted model store:

```text
/path/to/Qwen3-8B
```

Observed result:

- load completed successfully;
- post-rebrand model load: `14.7772 s`;
- request 1: `2.2605 s`, tokens `[25010, 10, 4999, 1725]`;
- request 2: `0.1823 s`, identical tokens;
- KV availability returned to `512/512` after both requests;
- clean process exit code `0`.

These timings were collected by `mini-sglang-metax/scripts/metax/run_gate0.sh`
after the model files were warm in the filesystem cache. They are functional
evidence, not a performance comparison.

## Project layout

```text
python/minisgl/                 Framework implementation (import name unchanged)
python/minisgl/utils/platform.py
                                CUDA-facing platform detection for MetaX
python/minisgl/attention/torch_native.py
                                Portable eager attention backend
scripts/metax/preflight.py      Device and operator preflight
scripts/metax/run_gate0.sh      Persistent, report-producing Gate 0 entry point
scripts/metax/run_tiny_e2e.py   Offline TP1/TPn repeated-request driver
scripts/metax/run_online_gate1.sh
                                Bounded TP1 HTTP/ZMQ and OpenAI API smoke
scripts/metax/run_online_gate1_1.sh
                                SSE, cancellation/Ack, concurrency, recovery
scripts/metax/run_online_gate1_2.sh
                                Bounded soak, overload queueing, faults, batching
scripts/metax/online_gate1_client.py
                                Standard-library extended online client
scripts/metax/batch_observability.py
                                Parses actual SchedulerBatch records
scripts/metax/vendor_inventory.py
                                Device and vendor attention package inventory
docs/metax_port/                Plans, evidence, verdicts, and reports
REPORT.md                       Management/Jira-ready project summary
PROVENANCE.md                   Source lineage and retained compatibility boundary
```

## Support boundary

Validated now:

- MetaX C500;
- MetaX C550;
- vendor PyTorch `2.10.0+metax3.8.1.0`;
- dense Qwen3 BF16;
- eager execution;
- TP1 real-model inference;
- TP2/TP8 functional transport and synthetic-model inference;
- deterministic greedy sampling and repeated-request cache recovery;
- TP1 HTTP/ZMQ service startup, model listing, repeated non-streaming OpenAI
  chat requests, and clean shutdown;
- SSE finish and `[DONE]` delivery, client disconnect plus AbortAck, four
  simultaneous ragged chat requests, and post-cancel recovery;
- bounded Gate 1.2 with 24/24 completions, offered concurrency 8 against
  running limit 2, scheduler backlog 6, and 99 actual multi-request batches;
- experimental MetaX `flashinfer` TP1 Gate 1 with real Qwen3-8B weights.

Not yet claimed:

- production readiness;
- multi-hour soak, sustained production load, HTTP load shedding, and TP2+
  online-service coverage;
- CUDA Graph;
- PyNCCL/MCCL wrapper parity beyond vendor `torch.distributed`;
- MoE, quantized checkpoints, automatic vendor-attention selection, or direct
  `mcoplib` paged-attention integration;
- throughput or latency leadership over SGLang, vLLM, or other frameworks.

## Verification

Any CPU host works for the portable tests — no GPU needed.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pytest msgpack pyzmq fastapi uvicorn prompt_toolkit \
  "transformers>=4.56.0,<=4.57.3"
pip install -e . --no-deps
```

CI suite (~20 s):

```bash
python -m pytest -q -o addopts="" \
  tests/misc/test_platform.py \
  tests/misc/test_torch_native_attention.py \
  tests/misc/test_pyproject_config.py \
  tests/misc/test_metax_vendor_inventory.py \
  tests/misc/test_metax_flashinfer_compat.py \
  tests/misc/test_metax_online_gate_client.py \
  tests/misc/test_metax_online_gate1_2_client.py \
  tests/misc/test_metax_batch_observability.py \
  tests/misc/test_scheduler_batch_observability.py \
  tests/misc/test_scheduler_abort_ack.py \
  tests/misc/test_exposed_path_abort_ack.py
```

Broader control-plane and layer suite:

```bash
python -m pytest -q -o addopts="" tests/misc/ tests/layers/
```

Real-hardware validation uses `scripts/metax/` — preserve the vendor PyTorch
installation.

## Contributing and security

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the portable test matrix and
hardware-evidence requirements. Report vulnerabilities according to
[`SECURITY.md`](SECURITY.md), and never attach private infrastructure logs or
model artifacts to a public issue.

## Attribution and license

Derived from
[`sgl-project/mini-sglang`](https://github.com/sgl-project/mini-sglang),
MIT license retained.
