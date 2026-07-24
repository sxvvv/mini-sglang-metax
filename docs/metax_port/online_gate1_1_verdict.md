# MetaX Online Gate 1.1 Verdict

**Project:** `mini-sglang-metax`
**Date:** 2026-07-24
**Result:** PASS for bounded streaming, cancellation, and concurrent TP1 serving
**Production readiness:** NOT CLAIMED

## Validated envelope

| Item | Value |
| --- | --- |
| Hardware | 1 x MetaX C550 |
| Vendor PyTorch | `2.10.0+metax3.8.1.0` |
| Model | Qwen3-8B BF16 |
| Model path | Read-only local Qwen3-8B checkpoint (path omitted) |
| Execution | TP1, eager, BF16, dense, `torch_native` attention |
| Streaming | OpenAI chat SSE events, finish event, and `[DONE]` marker |
| Cancellation | Live SSE disconnect, `AbortMsg`, and frontend `AbortAckReply` cleanup |
| Concurrency | Four simultaneous prompts with `max_tokens` 4, 5, 6, and 7 |
| Recovery | Model metadata after cancellation and a final chat request |
| Shutdown | Port closed and no server, scheduler, tokenizer, or spawn worker remained |

## Real-hardware result

The strict rerun returned `0` and produced this bounded result:

| Check | Result |
| --- | --- |
| SSE data events | `4` |
| SSE finish event | PASS |
| SSE `[DONE]` marker | PASS |
| First cancellation event received | PASS |
| Client connection deliberately closed | PASS |
| Frontend cancel log | `Aborting request for user 3` |
| Frontend Ack log | `Abort acknowledged for user 3` |
| Post-cancel `/v1/models` | PASS; one model |
| Concurrent requests | `4/4` returned non-empty completions |
| Concurrent request durations | `0.3178`, `0.3792`, `0.4322`, `0.4732` seconds |
| Final recovery request | PASS |
| Exit and cleanup | rc `0`, port closed, no residual worker |

The duration values only prove that all bounded requests completed. They are
not throughput claims and must not be compared with another framework.

## Test evidence

- Eight hermetic client/parser tests passed locally and on the MetaX host.
- Twelve Linux/MetaX cancellation and AbortAck tests passed on the target host.
- The existing focused regression set remained at 24 passing tests.

Persistent evidence:

```text
<persistent-root>/results/mini-sglang-metax/2026-07-24/
  online_gate1_1.rc
  online_gate1_1_preflight.log
  online_gate1_1_server.log
  online_gate1_1_models.json
  online_gate1_1_request.json
  online_gate1_1_chat_1.json
  online_gate1_1_chat_2.json
  online_gate1_1_summary.json
  online_gate1_1_extended_summary.json
  gate1_1_abort_tests.log
```

## Retained implementation

- `scripts/metax/online_gate1_client.py` uses only the Python standard library
  for JSON, HTTP, SSE, cancellation, and concurrent request validation.
- `scripts/metax/run_online_gate1_1.sh` selects the extended profile while
  preserving the simpler Gate 1 entry point.
- `FrontendManager.listen` logs a real pending AbortAck after it completes the
  frontend cleanup. Duplicate or unknown acknowledgements remain idempotent.

## Verdict boundary

Gate 1.1 proves a bounded TP1 online correctness path. It does not prove
multi-hour stability, sustained high concurrency, TP2+ online serving,
production supervision, admission control under overload, API completeness,
security, or performance leadership.
