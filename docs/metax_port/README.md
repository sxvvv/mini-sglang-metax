# MetaX documentation index

## Scope boundary

This directory is the current MetaX product and evidence surface. MetaX uses
the MACA/MCCL vendor stack and a CUDA-facing vendor PyTorch API, but it is not
the NVIDIA CUDA backend and it does not use the Ascend `torch_npu`/CANN/HCCL
route. Files under `docs/ascend_port/` are retained only as provenance,
device-abstraction history, and compatibility references; they are not MetaX
validation evidence.

Current evidence:

- [`../../PROVENANCE.md`](../../PROVENANCE.md): project overview, architecture,
  borrowed foundations, attribution, and MetaX adaptation boundary.
- [`gate0_verdict.md`](gate0_verdict.md): formal real-hardware verdict.
- [`online_gate1_verdict.md`](online_gate1_verdict.md): bounded online API Gate 1.
- [`online_gate1_1_verdict.md`](online_gate1_1_verdict.md): SSE, cancellation,
  concurrency, and recovery Gate 1.1.
- [`online_gate1_2_verdict.md`](online_gate1_2_verdict.md): bounded soak,
  overload queueing, real scheduler batching, and fault recovery.
- [`vendor_attention_verdict.md`](vendor_attention_verdict.md): installed
  `flashinfer`/`mcoplib` inventory and experimental real-model `fi` verdict.
- [`tp2_resource_blocker.md`](tp2_resource_blocker.md): exact single-card
  blocker for the real-model TP2 online matrix.
- [`open_source_readiness.md`](open_source_readiness.md): public release
  hygiene, supported claims, and remaining hardware blockers.
- [`bringup_result_2026-07-23.md`](bringup_result_2026-07-23.md): detailed
  environment, synthetic TP2/TP8, and real Qwen3-8B evidence.
- [`../../REPORT.md`](../../REPORT.md): management and Jira-ready summary.

Execution contract:

- [`gate0.md`](gate0.md): original correctness-first Gate 0 contract.
- `scripts/metax/preflight.py`: device and operator preflight.
- `scripts/metax/run_gate0.sh`: persistent report-producing runner.
- `scripts/metax/run_tiny_e2e.py`: repeated-request offline driver.
- `scripts/metax/run_online_gate1.sh`: bounded non-streaming online runner.
- `scripts/metax/run_online_gate1_1.sh`: SSE/cancellation/concurrency runner.
- `scripts/metax/run_online_gate1_2.sh`: bounded soak and batching runner.
- `scripts/metax/vendor_inventory.py`: reproducible vendor package inventory.

Planning history:

- `project_plan.md`, `todo.md`, `daily_report_2026-07-23.md`,
  `weekly_report_2026-07-23.md`, and `jira_draft.md` were written before the
  real C500/C550 runs. Statements in those files saying that hardware evidence
  was unavailable are historical. The current verdicts above supersede them.
