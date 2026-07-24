# Changelog

## v0.1.0a1 - MetaX Gate 0 technical preview

- Establishes `mini-sglang-metax` as the MetaX-first project and distribution
  name while retaining the upstream-compatible `minisgl` Python import.
- Adds explicit MetaX platform detection on top of the vendor `torch.cuda`
  device API.
- Documents the distinct NVIDIA CUDA/NCCL, Ascend `torch_npu`/CANN/HCCL, and
  MetaX MACA/MCCL routing contracts, including why a CUDA-facing MetaX tensor
  must not select NVIDIA-only kernels.
- Expands the project overview and provenance with architecture, borrowed
  foundations, attribution, MetaX adaptation ownership, evidence policy, and
  third-party dependency boundaries.
- Adds the eager `torch_native` attention backend and portable operator and
  sampling fallbacks for correctness-first bring-up.
- Disables CUDA Graph and PyNCCL on the MetaX path until separately validated.
- Records real-hardware Qwen3-8B BF16 TP1 repeated-request PASS evidence on a
  MetaX C500 host.
- Records TP2 collective and TP2/TP8 synthetic-model functional evidence.
- Adds a bounded online Gate 1 runner covering TP1 HTTP/ZMQ startup,
  `/v1/models`, repeated non-streaming OpenAI chat requests, response
  validation, and process-group cleanup.
- Records Qwen3-8B BF16 online Gate 1 PASS evidence on a MetaX C550 host.
- Adds online Gate 1.1 coverage for complete chat SSE delivery, live-client
  disconnect, AbortAck cleanup, four concurrent ragged requests, recovery,
  and residual-process checks.
- Adds an explicit frontend log when a real pending cancellation receives its
  `AbortAckReply`; duplicate acknowledgements remain idempotent.
- Adds bounded online Gate 1.2 with 3 x 8 concurrent requests, scheduler limit
  2, malformed-input and disconnect recovery, and clean shutdown checks.
- Adds stable `SchedulerBatch` observations plus a parser that proves actual
  multi-request batching and backlog independently of client concurrency.
- Records Gate 1.2 PASS on one C550: 24/24 requests, 99 multi-request batches,
  maximum batch 2, maximum pending 6, and no residual worker.
- Records the current TP2 online blocker precisely: only one C550 is visible.
- Inventories MetaX `flashinfer` and `mcoplib`, adapts optional wrapper
  constructor/plan kwargs to the installed signatures, and records a bounded
  Qwen3-8B TP1 `fi` Gate 1 PASS while retaining `torch_native` as default.
- Detects a crashed scheduler worker during server readiness so failed backend
  startup enters process-group cleanup immediately instead of waiting only for
  the startup timeout.
- Adds host-independent GitHub Actions checks, public contribution and security
  guidance, cross-platform ZMQ test transport, and explicit pinned-memory test
  requirements.
- Removes user-specific paths, private host identifiers, and internal storage
  topology from the public documentation while preserving reproducible Gate
  parameters and verdicts.

This release is not production-ready and makes no performance-superiority
claim.

## Historical Ascend work

Earlier Ascend technical-preview records remain under `docs/ascend_port/` as
device-abstraction and validation references. They are not the current
`mini-sglang-metax` product surface.
