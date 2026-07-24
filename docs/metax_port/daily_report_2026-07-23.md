# Mini-SGLang MetaX 日报（2026-07-23）

## 可直接发送版本

今日主要完成了 Mini-SGLang MetaX 方向的项目边界确认和源码主链梳理。首先核对了上游 Mini-SGLang 与 Ascend 分支：当前上游 `main` 为 `9a91cfa`，Ascend 分支 `85e3886` 直接建立在该基线上。对比后确认，Ascend 的主要适配点是 `torch.npu`、HCCL、FIA Attention 和 NPU 特定 KV 布局；MetaX/MACA 对外仍提供 `torch.cuda` 形式的接口，因此不应照搬 Ascend 的 NPU 分支，而应先验证 Mini-SGLang 原生 CUDA-facing 路径能够复用多少。

源码学习方面，已梳理 `launch_server()` 的 API、Tokenizer/Detokenizer 和每个 TP rank 的 Scheduler 进程启动关系；已进入 `Scheduler.__init__()` 和 `Engine.__init__()`，理清 Engine、TableManager、CacheManager、PrefillManager 和 DecodeManager 的初始化关系，以及 Engine 中设备绑定、通信、模型加载、KV Cache、Attention 和 Graph Runner 的先后顺序。

同时形成了 MetaX 兼容风险清单。Launch、API、Tokenizer、Scheduler 主流程、请求生命周期和缓存策略原则上属于通用层，不需要 MetaX 专用改造；真正需要在真卡上验证的是 FlashInfer/`sgl_kernel` Attention 接口、Norm/RoPE/Activation 融合算子、Mini-SGLang 自带 `.cu` JIT、SM 架构判定、CUDA Graph 以及 TP 多卡下 PyNCCL 与 vendor `torch.distributed`/MCCL 的边界。

目前本地已准备 MetaX 平台识别、纯 PyTorch Attention 和基础算子 fallback 等候选代码，并完成两组 Windows CPU 定向验证：平台/Attention/device 相关测试 99 项通过，Engine/device 静态回归 64 项通过。这些只能证明候选路径在静态和 CPU 测试中可用，尚不能证明这些改动在 MetaX 上全部必要，也不能声称 Mini-SGLang 已在 MetaX 真卡跑通。

下一步，服务器到位前继续完成 `Engine.forward_batch()` 到 Attention/KV Cache 的调用链笔记，并整理真卡环境/API 检查表。服务器到位后，先在 vendor 镜像中保留原有 PyTorch、MACA 和 `sgl_kernel` 包，使用未加 MetaX fallback 的干净 `ascend-port=85e3886` 走它保留的 CUDA 路径，运行 dense Qwen 小模型、TP1、bf16、eager、greedy，并关闭 CUDA Graph。保留第一个有效错误后，将其分类为环境、配置、依赖 API 或算子问题；如果需要判断错误是否由 Ascend 分支引入，再用上游 `main=9a91cfa` 做对照，最后只保留能解决真实问题的最小代码改动。

## 汇报时可用的一句话

我前期已经在 MetaX 上做过完整 SGLang 的服务、benchmark 和 profiling 验证，现在选择 Mini-SGLang 作为更小的实验载体，从黑盒使用转到白盒理解；当前已完成框架主链和 MetaX 兼容边界梳理，下一步就是在真卡上用原生 CUDA 路径找到第一个真实不兼容点，再做最小适配。

## 证据与边界

| 类别 | 当前证据 | 能说明什么 | 不能说明什么 |
| --- | --- | --- | --- |
| 上游基线 | `main=9a91cfa`，`ascend-port=85e3886` | 本地基线与当前上游对齐 | 不代表 MetaX 可运行 |
| Ascend 对比 | 确认 NPU/HCCL/FIA/KV 布局为硬件专属改动 | 可以用它学习分层和验收方法 | 不能直接复制为 MetaX 适配 |
| 既有 MetaX 经验 | 完整 SGLang 已跑过 C550 dense eager 和 C500 TP8/MCCL | MACA `torch.cuda` 与 vendor 算子链路真实存在 | 不代表 Mini-SGLang 的 FlashInfer/JIT/PyNCCL API 相容 |
| 本地候选代码 | 平台识别、`torch_native` Attention、fallback、preflight 和定向单测 | 已准备一条 correctness-first 兜底路径 | 尚未证明这些改动在真卡上必要 |
