# Project Overview and Provenance

## What this project is

`mini-sglang-metax` is an independent, correctness-first adaptation of
Mini-SGLang for MetaX C500/C550 accelerators and the MACA software stack. It is
intended to make the complete inference path small enough to study and debug:

```text
OpenAI API / Python LLM
  -> tokenize
  -> schedule and batch
  -> allocate/reuse KV cache
  -> run model and attention
  -> sample
  -> detokenize and return output
```

The project is a technical preview, not a production serving claim. Its value
is the combination of readable framework code, explicit accelerator routing,
reproducible gate scripts, and evidence from real MetaX hardware.

## Design boundary

MetaX vendor PyTorch presents accelerator tensors through a CUDA-facing
`torch.cuda` API. That API compatibility is useful, but it does not establish
binary compatibility with NVIDIA kernels or libraries. The project therefore
keeps two separate facts:

```text
device_type = cuda       # PyTorch API surface
platform    = metax      # accelerator vendor and kernel-routing decision
```

This prevents a MetaX tensor from selecting NVIDIA-only FlashInfer,
`sgl_kernel`, CUDA Graph, PyNCCL, or architecture probes merely because its
device type is named `cuda`. It also keeps MetaX separate from the Huawei
Ascend route, which uses `torch_npu`, CANN, HCCL, and the explicit `npu_fia`
attention backend.

The default MetaX path is eager execution with `torch_native` attention and
portable PyTorch operator fallbacks. The installed MetaX `flashinfer` package
is available only through explicit experimental opt-in. Vendor PyTorch, MACA,
MCCL, model weights, and drivers remain external prerequisites.

## Source lineage

`mini-sglang-metax` was created from the local `metax-port` working tree based
on:

```text
Ray-RP/mini-sglang-ascend
branch: metax-port
base commit: 85e3886
```

That branch was used because it already contained a device-abstraction and
evidence-driven gate structure derived from upstream Mini-SGLang. The current
MetaX project does not claim that Ascend-specific NPU/HCCL/FIA code applies to
MetaX; those paths are retained only as compatibility and historical reference.

The upstream project is:

```text
https://github.com/sgl-project/mini-sglang
license: MIT
```

The upstream-compatible Python package and import name remains `minisgl`. The
new distribution, repository, documentation, scripts, and evidence surface use
the name `mini-sglang-metax`.

## Ideas and implementation lineage

This repository does not present established serving ideas as new inventions.
The main inherited or referenced foundations are:

| Source | What is retained or referenced | Treatment in this project |
| --- | --- | --- |
| [Mini-SGLang](https://github.com/sgl-project/mini-sglang) | Compact process architecture, scheduler/engine/KV-cache/model organization, Python package name, and MIT-licensed implementation baseline | Preserved where portable; accelerator assumptions are routed explicitly |
| [SGLang](https://github.com/sgl-project/sglang) | Radix-cache design lineage and OpenAI-compatible serving context | Cited as the upstream design source; this repository remains a compact learning and porting surface |
| [Sarathi-Serve](https://arxiv.org/abs/2403.02310) | Chunked-prefill concept | Retained as an attributed scheduling feature, not claimed as a MetaX invention |
| [NanoFlow](https://arxiv.org/abs/2408.12757) | Overlap-scheduling concept | Retained as an attributed framework technique |
| `Ray-RP/mini-sglang-ascend` `metax-port` working tree | Device-abstraction patterns and evidence-driven gate structure | Used as the immediate engineering starting point; Ascend NPU/HCCL/FIA results are not treated as MetaX evidence |

The architecture and feature descriptions in `docs/structures.md` and
`docs/features.md` retain their upstream design context. The historical
`docs/ascend_port/` records are kept for provenance and compatibility study,
not as the current product surface.

## MetaX adaptation surface

The MetaX work in this repository includes:

1. Accelerator-platform detection independent of the PyTorch device namespace.
2. MetaX-safe routing for activation, normalization, rotary embedding,
   embedding, attention, sampling, graph selection, and distributed setup.
3. The eager `torch_native` attention backend and portable operator fallbacks.
4. Vendor `torch.distributed` use for validated MCCL/NCCL-compatible
   collectives while PyNCCL remains disabled on the default MetaX route.
5. Signature-aware compatibility for the optional MetaX `flashinfer` wrapper.
6. Online cancellation acknowledgement, scheduler batch observations, bounded
   concurrency/queueing/fault clients, and cleanup checks.
7. Reproducible C500/C550 gate scripts and public verdict documents.

Primary files and directories for that work are:

```text
python/minisgl/utils/platform.py
python/minisgl/utils/device.py
python/minisgl/utils/device_runtime.py
python/minisgl/attention/torch_native.py
python/minisgl/attention/fi.py
python/minisgl/engine/engine.py
python/minisgl/scheduler/scheduler.py
python/minisgl/server/api_server.py
scripts/metax/
docs/metax_port/
REPORT.md
```

## Validation model

Claims are tied to bounded gates rather than inferred from imports or synthetic
micro-tests alone:

- Gate 0 covers real Qwen3-8B BF16 loading, prefill, decode, deterministic
  repeated requests, and KV-cache recovery.
- Online Gate 1/1.1 covers model discovery, non-streaming and SSE chat,
  cancellation acknowledgement, ragged concurrency, recovery, and cleanup.
- Bounded Gate 1.2 covers sustained concurrent waves, queueing behavior,
  malformed input, disconnect recovery, and server-side proof of actual
  scheduler batching.
- The optional MetaX `flashinfer` route has a bounded TP1 Gate 1 result but is
  not the default and has not completed the wider Gate 1.1/1.2 matrix.

The exact supported claims and blockers are recorded under `docs/metax_port/`.
In particular, a real-model TP2 online matrix is not claimed because the final
allocation exposed only one C550.

## Repository map

```text
python/minisgl/          framework and runtime implementation
scripts/metax/          MetaX preflight and hardware gates
tests/                  hermetic and platform-routing tests
docs/metax_port/        current MetaX evidence and decisions
docs/ascend_port/       retained historical/reference material
README.md               public entry point and quick start
REPORT.md               Chinese engineering-stage report
```

## External dependencies

MetaX MACA, MCCL, vendor PyTorch, external model checkpoints, and related
drivers are not redistributed by this project. Their names are referenced only
to document the validated runtime contract.

All referenced trademarks and product names belong to their respective
owners. Reference to an external project or vendor does not imply endorsement.
The source code remains under the repository's MIT license; third-party
packages and model weights retain their own licenses and distribution terms.
