# MetaX Online Gate 1.2 Verdict

**Project:** `mini-sglang-metax`
**Date:** 2026-07-24
**Result:** PASS for bounded TP1 concurrency, scheduler admission, fault recovery, and batch observability
**Production readiness:** NOT CLAIMED

## Validated envelope

| Item | Value |
| --- | --- |
| Host | Internal single-node MetaX allocation (identifier omitted) |
| Hardware | 1 x MetaX C550 |
| Vendor PyTorch | `2.10.0+metax3.8.1.0` |
| Model | Qwen3-8B BF16 |
| Execution | TP1, eager, `torch_native`, 512 KV pages |
| Scheduler limit | `max_running_requests=2` |
| Offered load | 3 rounds x 8 simultaneous requests |
| Per-request output bound | 8 to 11 tokens |

## Real-hardware result

| Check | Result |
| --- | --- |
| Bounded requests | `24/24` completed; `0` failed |
| Peak client in-flight | `8` |
| Bounded client elapsed time | `7.2059 s` |
| Admission behavior | offered concurrency 8 exceeded running limit 2; requests queued and completed |
| Malformed input | HTTP `422` |
| Live disconnect | first SSE event received, connection closed deliberately, AbortAck completed |
| Recovery | `/v1/models` and final chat request passed |
| Exit and cleanup | rc `0`, port 1919 closed, no `python -m minisgl` process |

The elapsed time only bounds this run. It is not a throughput result and is
not comparable to another framework.

## Scheduler batch evidence

`SchedulerBatch` records are emitted immediately after the scheduler selects
a batch and before `_prepare_batch` or model forward. They therefore describe
actual scheduler batches, not HTTP client concurrency.

| Observation | Value |
| --- | --- |
| Total scheduler batches | `144` |
| Prefill / decode | `25 / 119` |
| Multi-request batches | `99` |
| Maximum batch size | `2` |
| Maximum pending requests | `6` |
| Maximum running requests | `2` |
| Maximum batch token count | `44` |

This proves scheduler batching and capacity-limited queueing. The current API
accepts overload into the scheduler queue; it does not implement HTTP 429
load shedding or a bounded frontend queue.

## Test and evidence files

- Local new tests: `8 passed`; existing Gate 1.1 client tests: `8 passed`.
- Target new tests: `8 passed`; cancellation/AbortAck regression: `12 passed`.

Persistent evidence:

```text
<persistent-root>/results/mini-sglang-metax/2026-07-24/
  online_gate1_2.rc
  online_gate1_2_console.log
  online_gate1_2_server.log
  online_gate1_2_soak_summary.json
  online_gate1_2_batch_summary.json
  gate1_2_new_tests.log
  gate1_2_abort_regression_tests.log
```

## Verdict boundary

Gate 1.2 is a bounded functional soak. It does not claim multi-hour stability,
production admission control, performance leadership, TP2+ online serving, or
automatic remediation after device or worker failure.
