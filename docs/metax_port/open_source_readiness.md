# Open-source readiness

**Date:** 2026-07-24

## Completed release surface

- MetaX C500/C550 TP1 real-model correctness evidence is documented.
- Online Gate 1, Gate 1.1, and bounded Gate 1.2 are reproducible scripts.
- Server-side scheduler batch observations distinguish client concurrency from
  actual multi-request batches.
- The experimental MetaX FlashInfer route is explicit opt-in; `torch_native`
  remains the correctness default.
- Portable CI covers platform selection, attention fallback, online clients,
  batch parsing, cancellation, and package metadata.
- The repository includes MIT licensing, provenance, contribution, and
  security guidance.
- User-specific paths, private hostnames, internal URLs, and raw result logs
  are excluded from the public Git history.

## Release-candidate checks

| Check | Result |
| --- | --- |
| Windows CPU portable CI set | `49 passed`, `4 skipped` (no pinned allocator) |
| MetaX C550 portable CI set | `53 passed` |
| Python compile check | PASS locally and on the MetaX host |
| MetaX shell syntax check | PASS |
| Ruff on the MetaX/publication surface | PASS |
| Python sdist and wheel build | PASS |
| Private path/hostname/credential scan | PASS; only documented placeholders and loopback addresses remain |
| Residual `python -m minisgl` process | none |

Target-side logs are retained outside the repository under the persistent
results directory as `open_source_ci_tests.log` and
`open_source_target_audit.log`.

## Evidence-backed blockers

- Real-model TP2 online Gate 1/1.1/1.2 requires an allocation with at least two
  visible compatible accelerators. The validated 2026-07-24 allocation exposed
  one C550.
- Multi-hour soak, production load shedding, CUDA Graph, broad model coverage,
  direct `mcoplib` integration, and performance leadership are not claimed.

## Exact next hardware action

Obtain a 2+ C500/C550 allocation, record the device inventory, choose a dense
checkpoint compatible with TP2 head partitioning, and run the existing online
Gate 1, Gate 1.1, and Gate 1.2 scripts with tensor parallel size 2.
