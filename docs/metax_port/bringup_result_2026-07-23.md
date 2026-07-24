# MetaX C500 Bring-up Result - 2026-07-23

## Scope

This record contains both synthetic-checkpoint coverage and a real Qwen3-8B
BF16 TP1 result. It is a correctness result, not a model-quality result or a
performance benchmark.

## Target environment

- Host: Internal 8-device MetaX allocation (identifier omitted)
- Accelerator: 8 x MetaX C500
- PyTorch: `2.10.0+metax3.8.1.0`
- CUDA compatibility version reported by PyTorch: `11.6`
- Device API: `torch.cuda`
- `CUDA_PATH`: `/opt/maca/tools/cu-bridge`
- No Ascend CANN tree, `npu-smi`, `torch_npu`, or `/dev/davinci*` was present.
- Source and results were kept on a persistent writable mount.
- Model checkpoints were loaded from a separate read-only mount.
- Internal mount names and storage topology are omitted because they are not
  part of the framework contract.

## Code change retained

The first end-to-end failure was:

```text
ModuleNotFoundError: No module named 'msgpack'
```

`SchedulerIOMixin` imported ZMQ queue classes at module load time even when
`offline_mode=True`. Offline inference returns before constructing any queue,
so the queue import now occurs only after the offline early return. Online
serving still imports and uses the same queue classes.

Regression coverage:

```text
tests/misc/test_scheduler_sync_all_ranks.py::
test_scheduler_io_import_does_not_require_msgpack
```

## Evidence

### Runtime preflight

`scripts/metax/preflight.py` passed device detection, bf16 GEMM, and the
`torch_native` attention micro-smoke with all 8 devices visible.

### TP2 collective

Two ranks initialized the vendor PyTorch `nccl` backend and completed an
all-reduce:

```text
TP2_COLLECTIVE rank=0 value=3.0 device=MetaX C500
TP2_COLLECTIVE rank=1 value=3.0 device=MetaX C500
```

### Mini-SGLang TP1

- Status: PASS
- Attention: `torch_native`, eager
- Output token ids: `[45, 251, 33, 33]`
- KV tokens available after completion: `256`
- Load time: `10.1901 s`
- Generate time: `39.6749 s`

The TP1 timing includes first-use initialization and must not be compared with
the later warmed multi-rank runs.

### Mini-SGLang TP2

- Status: PASS on ranks 0 and 1
- Identical output token ids: `[45, 251, 33, 33]`
- KV tokens available after completion on both ranks: `256`
- Load time: `10.1846-10.1876 s`
- Generate time: `0.8902-0.9871 s`

### Mini-SGLang TP8

- Status: PASS on ranks 0 through 7
- Identical output token ids: `[199, 16, 39, 89]`
- KV tokens available after completion on every rank: `256`
- Load time: `10.3383-10.3957 s`
- Generate time: `2.5843-2.9859 s`

TP8 used a second synthetic Qwen3 checkpoint with 8 query heads and 8 KV
heads so every tensor-parallel shard was valid.

### Real Qwen3-8B BF16 TP1, two requests

Checkpoint:

```text
/path/to/Qwen3-8B
```

Checkpoint facts: `Qwen3ForCausalLM`, 36 layers, hidden size 4096, 32 query
heads, 8 KV heads, head dimension 128, BF16, 2 safetensors shards totaling
16,381,516,808 bytes.

- Status: PASS
- Process exit code: `0`
- TP: `1`
- Attention: `torch_native`, eager
- Model load time with a warm filesystem cache: `15.4430 s`
- Request 1: `0.9243 s`, token ids `[25010, 10, 4999, 1725]`
- Request 2: `0.1841 s`, token ids `[25010, 10, 4999, 1725]`
- KV tokens available after each request: `512/512`
- Both requests ran through the same live `LLM` instance before shutdown.

The fixed input is a short token-id sequence rather than a natural-language
prompt. This proves weight loading, prefill, decode, deterministic greedy
sampling, allocator recovery, and repeated-request liveness; it does not
claim useful answer quality.

## Reproduction entry point

```bash
export MINISGL_PLATFORM=metax
export MACA_PATH=/opt/maca
export CUDA_HOME=/opt/maca/tools/cu-bridge
export PYTHONPATH=python

python scripts/metax/preflight.py

python scripts/metax/run_tiny_e2e.py \
  --model-path \
  /path/to/Qwen3-8B \
  --num-pages 512 \
  --repeats 2

torchrun --standalone --nproc_per_node=2 \
  scripts/metax/run_tiny_e2e.py \
  --model-path /path/to/tiny-qwen3-random

torchrun --standalone --nproc_per_node=8 \
  scripts/metax/run_tiny_e2e.py \
  --model-path /path/to/tiny-qwen3-random-tp8
```

## Subsequent completion

The package inventory and online HTTP/ZMQ work listed in the original bring-up
record were completed on the later C550 allocation. See
`online_gate1_verdict.md`, `online_gate1_1_verdict.md`,
`online_gate1_2_verdict.md`, and `vendor_attention_verdict.md`.

Natural-language quality evaluation and performance comparison remain outside
the Gate 0 correctness contract.
