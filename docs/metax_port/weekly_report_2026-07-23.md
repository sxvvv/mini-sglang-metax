# Mini-SGLang MetaX 预研周报（2026-07-20 至 2026-07-24）

## 本周目标

完成 Mini-SGLang MetaX 方向的可行性调研，建立框架源码主线，并明确通用 CUDA 路径、vendor 软件栈与框架侧最小适配的边界。

## 本周完成

1. 核对了当前代码基线。上游 `sgl-project/mini-sglang main` 为 `9a91cfa`，`Ray-RP/mini-sglang-ascend ascend-port` 为 `85e3886`，Ascend 分支直接建立在当前上游主线上，本地学习基线未过期。

2. 完成了 Ascend 硬件改动分类。Ascend 的核心差异为 `torch.npu`、HCCL、FIA Attention、NPU 特定 KV 布局和 NPU 融合算子；请求生命周期、Scheduler 正确性和 API 处理中的多数后续改动属于通用框架修复，不能全部算作硬件适配。

3. 完成了 MetaX/MACA 官方路线调研。官方 `mcPytorch` 使用 `.cuda()`/`cuda:0`；`vLLM-metax` 将 MACA 定义为 cuda-like backend，平台信息使用 `device_type="cuda"`、`dispatch_key="CUDA"`、`dist_backend="nccl"`，同时单独提供 MACA Attention/Triton 路由与 MCCL wrapper。这说明通用 CUDA 控制面可复用，但 NVIDIA 专属算子与二进制接口仍需验证。

4. 完成了框架源码主链的第一轮走读。已梳理 `launch_server()` 的多进程结构、`Scheduler.__init__()` 中 Engine/Table/Cache/Prefill/Decode Manager 的初始化关系，并进入 `Engine.__init__()` 的设备、通信、模型、KV Cache、Attention 和 Graph Runner 链路。

5. 形成了 MetaX 兼容性 inventory。初步确认 Launch、API、Tokenizer、Scheduler、Req/Batch 和缓存策略属于通用层；待真卡验证的 P0 区域为 FlashInfer/vendor `sgl_kernel`、Norm/RoPE/Activation、Mini-SGLang `.cu` JIT、SM 架构判定、PyNCCL 和 CUDA Graph。

6. 准备了一版 correctness-first 本地候选实现，包括 MetaX 平台识别、纯 PyTorch Attention、基础算子 fallback、preflight 和 TP1 脚本。Windows CPU 上的平台/Attention/device 相关测试 99 项通过，Engine/device 静态回归 64 项通过。该候选实现尚未在 MetaX 真卡上验证，不作为最终适配结论。

## 阶段性结论与价值

`Mini-SGLang on MetaX` 可以继续作为一个小项目，但应定位为“Mini-SGLang CUDA-like 路径在 MACA 上的兼容性验证与最小框架集成”，而不是重写 MetaX 后端或实现 MACA/MCCL。

该项目的价值是：用更小的代码建立对推理框架完整链路的理解，同时把 MetaX 上的环境、配置、依赖、API 和算子问题拆成可复现、可验证的小问题，为后续完整 SGLang 的问题定位和性能分析提供小型实验载体。

## 当前边界

- 已有 MetaX 真卡证据来自完整 SGLang，不是 Mini-SGLang。
- 当前本地测试来自 Windows CPU，只证明候选代码结构和逻辑，不证明 MetaX 算子可用性。
- 不能声称 Mini-SGLang 已在 MetaX 跑通。
- 不能在真卡前断言所有 FlashInfer/自定义算子必须被纯 PyTorch fallback 替换；MetaX 官方已提供 McFlashInfer 和 mcoplib，应先核对版本与 API。

## 下周计划

1. 完成 `Engine.forward_batch()` -> Attention -> KV Cache -> Sampler 源码走读和笔记。
2. 整理 MetaX vendor 包/API inventory 与真卡 preflight 输出模板。
3. 准备干净 `ascend-port=85e3886` 保留的 CUDA 路径启动和日志保存流程，同时准备 `main=9a91cfa` 对照方法。
4. 服务器到位后，使用 dense Qwen 小模型、TP1、bf16、eager、greedy 并关闭 CUDA Graph，先运行干净 `85e3886` 的 CUDA 路径。
5. 保留第一个有效失败，建立最小复现，再决定当前候选 fallback 的保留或删除。
6. 形成 Gate 0 verdict，明确通过项、失败项、保留 diff 和未覆盖范围。

## 需要的支持

- 一张可访问的 MetaX C500 或 C550。
- 已匹配的驱动、MACA、vendor PyTorch、`sgl_kernel`/McFlashInfer 镜像。
- 一个本地可读的 dense Qwen 小模型 checkpoint。

## 公开资料

- [MetaX mcPytorch](https://github.com/MetaX-MACA/mcPytorch/blob/2.4/README.md#L1-L6)
- [vLLM MetaX Plugin](https://github.com/MetaX-MACA/vLLM-metax/blob/master/README.md#L41-L51)
- [vLLM MetaX platform](https://github.com/MetaX-MACA/vLLM-metax/blob/master/vllm_metax/platform.py#L160-L167)
- [vLLM MetaX MCCL/PyNCCL wrapper](https://github.com/MetaX-MACA/vLLM-metax/blob/master/vllm_metax/patch/plugin_enhancement/distributed/pynccl_wrapper.py#L5-L12)
- [MetaX McFlashInfer](https://github.com/MetaX-MACA/McFlashInfer/blob/master/README.md)
- [MetaX mcoplib SGLang support](https://github.com/MetaX-MACA/mcoplib/blob/main/README.md#L407-L414)
- [MetaX MCCL](https://github.com/MetaX-MACA/mccl-nccl/blob/2.20.5/README.md#L1-L19)
