# MetaX TP2 Online Resource Blocker

**Date:** 2026-07-24
**Status:** BLOCKED BY CURRENT ALLOCATION

The validated 2026-07-24 C550 allocation reports:

```text
torch=2.10.0+metax3.8.1.0
device_count=1
devices=['MetaX C550']
```

A real-model TP2 online matrix requires at least two visible accelerators in
the same allocation. Only one C550 is visible, so no TP2 process launch or
result is claimed in this task. Existing TP2/TP8 transport and synthetic-model
evidence from the earlier C500 allocation remains valid but is not a substitute
for a real-model TP2 online run.

Exact next action: obtain a 2+ C550/C500 allocation, re-run the device inventory,
select a dense model whose query and KV head counts divide or replicate safely
at TP2, then run Gate 1, Gate 1.1, and Gate 1.2 with `--tensor-parallel-size 2`.
