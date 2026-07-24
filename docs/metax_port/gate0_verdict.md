# MetaX Gate 0 Verdict

**Project:** `mini-sglang-metax`
**Date:** 2026-07-24
**Result:** PASS for the bounded offline correctness path
**Production readiness:** NOT CLAIMED

## Validated envelope

| Item | Value |
| --- | --- |
| Hardware | 8 x MetaX C500 host; TP1 used for the real-model case |
| Vendor PyTorch | `2.10.0+metax3.8.1.0` |
| Device API | `torch.cuda` |
| Platform route | `metax` |
| Model | Qwen3-8B BF16 |
| Model path | Read-only local Qwen3-8B checkpoint (path omitted) |
| Execution | eager, BF16, dense |
| Attention | `torch_native` |
| Sampling | greedy, four output tokens |
| Repeated requests | two requests in one live `LLM` instance |
| KV pool | 512 tokens, integrity checked after each request |

## Evidence

The process loaded two safetensors shards totaling 16,381,516,808 bytes and
completed both requests without restarting the model:

| Run | Generate time | Token ids | KV available after run |
| --- | ---: | --- | ---: |
| 1 | `0.9243 s` | `[25010, 10, 4999, 1725]` | `512/512` |
| 2 | `0.1841 s` | `[25010, 10, 4999, 1725]` | `512/512` |

The warmed-filesystem model load took `15.4430 s`. An earlier cold-path run
loaded successfully in `93.6483 s`. The final process exit code was `0`.

After the project was packaged and renamed to `mini-sglang-metax`, its new
`scripts/metax/run_gate0.sh` entry point reproduced PASS with exit code `0`:

| Stage | Post-rebrand result |
| --- | ---: |
| Model load | `14.7772 s` |
| Request 1 | `2.2605 s` |
| Request 2 | `0.1823 s` |
| Output ids | `[25010, 10, 4999, 1725]` on both runs |
| KV recovery | `512/512` after both runs |

Persistent evidence on the host:

```text
<persistent-root>/results/mini-sglang-metax/2026-07-23/
  environment.txt
  qwen3_8b_bf16_tp1.log
  qwen3_8b_bf16_tp1.rc
  qwen3_8b_bf16_tp1_two_requests.log
  qwen3_8b_bf16_tp1_two_requests.rc

<persistent-root>/results/mini-sglang-metax/2026-07-24/
  preflight.log
  gate0_tp1.log
  gate0_tp1.rc
  gate0_wrapper_console.log
  host_focused_tests.log
```

The validation container used a persistent writable mount for source and
results. Its internal mount topology is intentionally omitted from the public
record because it is not part of the framework contract.

## First failure and retained fix

The first real end-to-end failure was:

```text
ModuleNotFoundError: No module named 'msgpack'
```

Offline inference never constructs ZMQ queues, but the scheduler I/O module
imported the queue implementation at module load time. The retained fix moves
that import after the `offline_mode` early return. Online behavior is unchanged
and still requires `msgpack` and `pyzmq`.

## Additional multi-card evidence

- TP2 vendor `torch.distributed` all-reduce passed on two C500 devices.
- Synthetic Qwen3 TP2 end-to-end passed on both ranks with identical output.
- Synthetic Qwen3 TP8 end-to-end passed on ranks 0 through 7 with identical
  output and complete KV-pool recovery.

These synthetic cases prove framework sharding and collective control flow;
they are not model-quality or performance evidence.

## Verdict boundary

Gate 0 passes for real-model offline loading, prefill, decode, deterministic
sampling, repeated-request liveness, and allocator recovery on MetaX C500.

This verdict does not cover the online HTTP/ZMQ server, CUDA Graph, PyNCCL,
MoE, quantized models, fused MetaX attention kernels, soak testing, or any
performance comparison.
