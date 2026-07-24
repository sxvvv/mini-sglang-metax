# MetaX Online Gate 1 Verdict

**Project:** `mini-sglang-metax`
**Date:** 2026-07-24
**Result:** PASS for the bounded TP1 HTTP/ZMQ correctness path
**Production readiness:** NOT CLAIMED

## Validated envelope

| Item | Value |
| --- | --- |
| Hardware | 1 x MetaX C550 |
| Vendor PyTorch | `2.10.0+metax3.8.1.0` |
| Model | Qwen3-8B BF16 |
| Model path | Read-only local Qwen3-8B checkpoint (path omitted) |
| Execution | TP1, eager, BF16, dense |
| Attention | `torch_native` |
| Transport | HTTP frontend plus local ZMQ process pipeline |
| API checks | `GET /v1/models`; two non-streaming `POST /v1/chat/completions` |
| Shutdown | Process group terminated; port closed; no worker process remained |

## Evidence

The service completed backend initialization, loaded five safetensors shards,
and reported scheduler and tokenizer readiness before Uvicorn started serving.

The bounded client then observed:

| Check | Result |
| --- | --- |
| `/v1/models` | HTTP 200; exactly one card with the requested model path |
| Chat request 1 | HTTP 200; non-empty `chat.completion` JSON |
| Chat request 2 | HTTP 200; non-empty `chat.completion` JSON |
| Repeated output | Identical for both greedy requests |
| Script exit | `0` |
| Port after exit | Closed |
| Residual workers | None |

Persistent evidence:

```text
<persistent-root>/results/mini-sglang-metax/2026-07-24/
  online_gate1.rc
  online_gate1_preflight.log
  online_gate1_server.log
  online_gate1_models.json
  online_gate1_request.json
  online_gate1_chat_1.json
  online_gate1_chat_2.json
  online_gate1_summary.json
```

## First failure and retained fix

The vendor image contained `pyzmq`, FastAPI, and Uvicorn but did not contain
`msgpack`. Gate 0 did not expose this because offline mode intentionally avoids
constructing ZMQ queues. The online path correctly failed its dependency check.

`msgpack 1.2.1` was staged under the persistent
`<persistent-root>/python-packages` directory. The Gate 1 runner adds this
directory to `PYTHONPATH` before checking dependencies or launching workers.
The staged pure-Python implementation passed a pack/unpack round trip on the
target Python 3.10 runtime.

## Verdict boundary

This Gate proves bounded TP1 online startup, frontend/backend ZMQ connectivity,
model metadata, repeated non-streaming chat completion, and deterministic
cleanup on MetaX C550.

It does not cover streaming responses, client cancellation, concurrent or
ragged requests, dynamic batching, soak behavior, TP2+ online serving,
authentication, production supervision, or performance comparison.
