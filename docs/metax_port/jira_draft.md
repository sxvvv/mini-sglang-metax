# Jira 草稿：Mini-SGLang CUDA 路径在 MetaX/MACA 上的兼容性验证与最小适配

## Summary

验证 Mini-SGLang 原生 CUDA-facing 路径在 MetaX/MACA 环境中的可复用范围，定位 NVIDIA 专用依赖的首个真实不兼容点，并完成最小的框架侧修复。

## Background

完整 SGLang 已在现有 MetaX/MACA vendor 栈上完成过单卡 dense 模型服务、profiling 以及 TP8/MCCL 链路验证，但完整框架规模较大，调度、缓存、并行和算子问题容易混在一起。Mini-SGLang 代码更小，适合作为理解请求、调度、KV Cache、Attention 和采样全链路的白盒实验载体。

MetaX 是 GPGPU，MACA vendor PyTorch 通过 `torch.cuda` 提供设备、Stream、Event 和显存等接口。因此本项目不实现 MACA、MCCL 或新的 PyTorch device type，也不照搬 Ascend 的 `torch.npu`/HCCL/FIA 路径。项目的技术问题是：Mini-SGLang 原生 CUDA 路径中哪些接口已被 MACA/vendor wheel 兼容，哪些 NVIDIA 专用假设需要最小路由或 fallback。

## Goal

在一张 MetaX C500/C550 上，使用 dense Qwen 小模型、bf16、TP1、eager 和 greedy 生成，完成从环境盘点、原始 CUDA 基线、首错定位、最小修复到端到端生成的可复现记录

## In scope

- Launch -> Scheduler -> Engine -> Attention/KV Cache -> Sampler 源码主链梳理。
- MetaX vendor 环境与 Mini-SGLang 依赖/API 盘点。
- 未加 MetaX fallback 的 CUDA 路径基线运行。
- FlashInfer、vendor `sgl_kernel`、Mini-SGLang `.cu` JIT 和架构判定的兼容性验证。
- 仅针对实测失败点保留的最小框架侧修复。
- dense Qwen、TP1、bf16、eager、greedy 正确性验证。
- 环境、命令、日志、失败证据、代码 diff 和 verdict 文档。

## Out of scope for Gate 0

- 实现或修改 MACA、MCCL、vendor PyTorch 和 vendor kernel 内部代码。
- MoE、W8A8、MTP 和 TP8。
- CUDA Graph 适配与性能调优。
- PyNCCL 性能、通信优化和多机。
- 与 SGLang、vLLM 或 NVIDIA GPU 的性能优劣结论。

## Current progress - 2026-07-23

- [x] 核对当前上游基线：`sgl-project/mini-sglang main=9a91cfa`。
- [x] 核对 Ascend 分支：`Ray-RP/mini-sglang-ascend ascend-port=85e3886`，其 merge-base 为当前上游 `main`。
- [x] 完成 Ascend 硬件改动分类：NPU runtime、HCCL、FIA、NPU KV 布局和 NPU 融合算子不适用于 MetaX。
- [x] 完成 Launch、Scheduler 和 Engine 初始化主链的第一轮走读与注释。
- [x] 列出 MetaX P0 验证点：Attention/vendor wheel API、FlashInfer 形式基础算子、`.cu` JIT、SM 判定和安装约束。
- [x] 准备 MetaX 平台识别、`torch_native` Attention、基础算子 fallback、preflight 和 TP1 脚本的本地候选实现。
- [x] 候选实现已完成 Windows CPU 定向验证：平台/Attention/device 相关测试 99 项通过，Engine/device 静态回归 64 项通过；该结果不是 MetaX 真卡结论。
- [ ] 在 MetaX 真卡上完成原始 CUDA 路径基线。
- [ ] 保存并分类第一个有效失败。
- [ ] 根据实测证据删减候选 diff，保留最小修复。
- [ ] 完成 TP1 端到端生成和 Gate 0 verdict。

## Gate 0 acceptance criteria

1. 记录 GPU、驱动、MACA、Python、PyTorch、`sgl_kernel`、FlashInfer 和容器版本。
2. 保留一份未加 MetaX fallback 的原始 CUDA 路径结果，无论成功还是失败。
3. 若失败，提供第一个有效错误、最小复现命令和环境/配置/依赖/API/算子分类。
4. 若需要改代码，diff 中的每项修改都能对应一条真卡失败证据或明确的 capability check。
5. dense Qwen 小模型在 TP1、bf16、eager、greedy 下完成两次连续请求，输出非空，服务保持存活。
6. 输出完整的环境、命令、日志、commit/diff、通过项、失败项和未覆盖项。

## Next actions

1. 完成 `Engine.forward_batch()` -> Attention -> KV Cache -> Sampler 调用链笔记。
2. 建立真卡 vendor 包/API inventory，明确哪些 import 和 symbol 需要检查。
3. 保留干净的 upstream/Ascend 基线与当前 `metax-port` 候选 diff，不在真卡结果前把全部 fallback 当成必要适配提交。
4. 服务器到位后先运行原始 CUDA 路径：dense Qwen、TP1、bf16、eager、greedy、CUDA Graph 关闭。
5. 保留首错后一次只处理一个变量，重跑相同 case，逐步收敛最小 diff。

## Risks / blockers

- Blocker：尚未获得 MetaX C500/C550 真卡环境，无法给出 Mini-SGLang 端到端结论。
- Risk：vendor PyTorch 与仓库 `torch<2.10.0` 依赖约束存在冲突，安装时不能让 pip 替换 vendor torch。
- Risk：完整 SGLang 中的 vendor `sgl_kernel` 能力不能自动外推为 Mini-SGLang 所需 Python symbol 全部相容。
- Risk：Mini-SGLang PyNCCL 直接链接 NCCL 2.27 API，不等于 vendor `torch.distributed` 下的 MCCL 兼容路径。
