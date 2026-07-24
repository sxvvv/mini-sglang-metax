# MetaX Vendor Attention Inventory and Verdict

**Date:** 2026-07-24
**Default correctness backend:** `torch_native`
**Experimental result:** MetaX `flashinfer` TP1 online Gate 1 PASS

## Installed inventory

| Probe | Result |
| --- | --- |
| `mcflashinfer` | module not found |
| `McFlashInfer` | module not found |
| `mc_flashinfer` | module not found |
| `flashinfer` | import PASS, `0.2.6+metax3.8.1.0torch2.10` |
| `mcoplib` | import PASS, distribution `0.4.8+maca3.8.0.24.torch2.10` |
| MACA check from `mcoplib` | runtime 3.8.1 satisfies minimum 3.7.0 |
| Optional FLA package | not installed; `gdn_prefill` unavailable |

The installed MetaX fork exports paged-KV prefill/decode wrappers. `mcoplib`
imports and reports its compatibility metadata, but this task did not identify
or validate a direct `mcoplib` paged-attention API for Mini-SGLang.

## Compatibility fixes driven by real failures

The first real-model `fi` launch loaded Qwen3-8B and failed because the MetaX
decode wrapper constructor does not accept `backend=`. The second launch
reached the first prefill batch and failed because the MetaX `plan()` does not
accept `seq_lens=`. Both are optional in the current Mini-SGLang call contract.

`python/minisgl/attention/fi.py` now passes these optional arguments only when
the installed callable signature exposes them. Required page tables, head
counts, dimensions, dtypes, and causal settings are unchanged. Three hermetic
compatibility tests pass locally and on the MetaX host.

## Real-model verdict

The final probe used Qwen3-8B BF16, TP1, eager execution, 512 KV pages, and
`ATTENTION_BACKEND=fi` on one C550.

| Check | Result |
| --- | --- |
| `/v1/models` | HTTP 200 |
| Chat requests | `2/2` HTTP 200 with identical non-empty output |
| Scheduler phases | prefill and decode both observed |
| Runner | rc `0` |
| Cleanup | port 1922 closed; no residual `python -m minisgl` process |

Persistent evidence:

```text
<persistent-root>/results/mini-sglang-metax/2026-07-24/
  vendor_inventory.json
  vendor_inventory.log
  flashinfer_api_signatures.log
  flashinfer_compat_tests.log
  online_gate1_fi_probe_server.log
  online_gate1_fi_fixed_server.log
  online_gate1_fi_fixed2.rc
  online_gate1_fi_fixed2_server.log
  online_gate1_fi_fixed2_summary.json
```

## Boundary

`fi` remains explicit opt-in. The default stays `torch_native` because the
vendor path has only bounded TP1 Gate 1 coverage, not Gate 1.1 cancellation,
Gate 1.2 soak, TP2, CUDA Graph, broad model shapes, or performance comparison.
The successful process emitted Python `resource_tracker` semaphore warnings
during shutdown, although the port and worker process checks were clean.
