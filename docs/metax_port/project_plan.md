# Mini-SGLang MetaX 兼容性验证与最小适配计划

> 历史计划：本文保留真卡到位前的验收思路。当前完成状态以仓库 `README.md`、
> `REPORT.md` 和本目录 Gate 0/Gate 1.x verdict 为准。

## 1. 项目定位

本项目不重写 MACA、MCCL 或 PyTorch 设备后端。项目以 Mini-SGLang 为小型白盒实验载体，学习“请求接入 -> 调度 -> KV Cache -> 模型 -> Attention -> 采样 -> 输出”全链路，同时验证 Mini-SGLang 原生 CUDA-facing 路径在 MetaX/MACA 上的可复用范围。

最终交付不是“写了多少 MetaX 分支”，而是一条可证明的链路：

```text
干净 CUDA 基线
-> 第一个真实不兼容点
-> 问题分类与最小复现
-> 最小框架侧修复
-> 端到端正确性证据
```

## 2. 为什么这个项目成立

1. MetaX 官方维护的 `vLLM-metax` 将 MACA 定义为 cuda-like backend，说明框架层可以复用大量 CUDA 控制面，再通过插件/路由处理硬件差异。
2. 工作区已有完整 SGLang 在 MetaX C550 的 dense eager 服务与 profiling 证据，也有 C500 TP8/MCCL 运行证据，说明 vendor 运行时、`torch.cuda` 和融合算子链已存在。
3. 上游 Mini-SGLang 仍使用 FlashInfer、`sgl_kernel`、SM 架构判定、`.cu` JIT、PyNCCL 和 CUDA Graph，并非只要 `torch.cuda.is_available()` 就必然可用。
4. 因此这个项目具有明确的研究问题、可运行的基础和可验收的输出。

## 3. 代码基线的用法

| 基线 | commit/分支 | 用途 |
| --- | --- | --- |
| 干净项目基线 | `origin/ascend-port` / `85e3886` | Gate 0 首跑；不加 MetaX fallback，使用该分支保留的 CUDA 路径，同时用于学习设备抽象、分层和验收方法 |
| 上游对照基线 | `origin/main` / `9a91cfa` | 当干净项目基线失败时，判断问题来自上游 NVIDIA 假设还是 Ascend 分支差异 |
| MetaX 实验分支 | `metax-port` | 保存候选代码，最终只留下真卡证据要求的最小 diff |

## 4. 技术原则

- 先运行未修改基线，再使用 fallback。
- 优先复用 vendor PyTorch、MACA、`sgl_kernel` 和 `torch.distributed` 能力。
- 优先使用 capability/API 检查，只有硬件语义确实不同时才增加 `platform == "metax"` 分支。
- 一次只改一个主要变量，每次修改都重跑同一个最小 case。
- 第一阶段固定 dense、bf16、TP1、eager、greedy，不混入 MoE、量化、MTP、TP8 和性能优化。
- 每个结论标记为源码事实、本地 CPU 测试、MetaX 真卡现象或待验证推断。

## 5. 分阶段计划

| 阶段 | 唯一目标 | 主要交付物 | 状态 |
| --- | --- | --- | --- |
| Phase 0 | 建立框架源码主链 | Launch -> Scheduler -> Engine -> Attention/KV Cache -> Sampler 笔记 | 进行中 |
| Gate 0A | 建立真卡环境事实 | vendor 版本、设备 API、包/symbol inventory | 等待服务器 |
| Gate 0B | 运行原始 CUDA 路径 | 完整命令、日志、首个有效错误或直接 PASS | 等待服务器 |
| Gate 0C | 解决一个已证实的不兼容点 | 最小复现、最小 diff、A/B 日志 | 未开始 |
| Gate 1 | 完成 TP1 正确性 | 两次 greedy 生成、输出与资源状态 | 未开始 |
| Gate 2 | 验证调度与缓存 | equal/ragged prefill、decode、prefix cache、取消请求 | 未开始 |
| Gate 3 | 验证 TP2 vendor 通信 | 优先 `torch.distributed`/MCCL，再决定是否需要 PyNCCL 处理 | 未开始 |
| Gate 4 | 评估性能路径 | CUDA Graph、融合 Attention、JIT 和代表性 profile | 未开始 |
| Gate 5 | 扩展模型能力 | MoE、量化、MTP、TP8，每次只引入一类变量 | 未开始 |

## 6. 通用层与待验证层

| 框架区域 | 初始判断 | 原因/动作 |
| --- | --- | --- |
| Launch、API、ZMQ、Tokenizer/Detokenizer | 直接复用 | CPU/进程与消息层，不应加 MetaX 分支 |
| Scheduler、Req/Batch、Prefill/Decode Manager | 直接复用 | 调度策略与硬件类型原则上解耦 |
| `torch.cuda` device/Stream/Event/memory | 先直接复用 | MACA vendor PyTorch 已使用该接口，逐项真卡检查 |
| Attention backend | P0 验证 | 检查 FlashInfer 与 vendor `sgl_kernel.flash_attn` API |
| RMSNorm、RoPE、Activation、sampling | P0 验证 | CUDA 分支多处直接 import FlashInfer |
| KV `store/index` JIT | P0 验证 | 验证 MACA compiler、tvm-ffi 和 `libcudart.so` 链接名称 |
| 架构判定 | P0 验证 | `get_device_capability()` 和 SM90/SM100 判定含 NVIDIA 语义 |
| CUDA Graph | 首轮关闭 | 已有 MetaX SGLang smoke 也先使用 eager，后续单独验证 |
| PyNCCL | TP1 不作为主题，TP2 优先禁用 | Mini-SGLang 直接链接 NCCL 2.27 API，不等于 vendor MCCL 通路 |
| MoE/Triton | 延后 | 变量太多，不影响 dense TP1 Gate 0 |

## 7. 服务器到位后的执行顺序

1. 只读记录 GPU、驱动、MACA、Python、PyTorch、`sgl_kernel`、FlashInfer、Triton 和容器版本。
2. 检查 `torch.cuda` 设备、bf16 GEMM、Stream/Event、显存查询和必需模块/symbol，不先改代码。
3. 保留 vendor PyTorch 和 vendor wheel，避免安装过程覆盖定制包。
4. 用干净 `85e3886` 保留的 CUDA 路径启动 dense Qwen、TP1、bf16、eager、greedy，关闭 CUDA Graph；需要分支归因时再跑 `9a91cfa` 对照。
5. 完整保存第一个有效错误，删去 worker 退出后的派生噪音，建立最小复现。
6. 按“环境 -> 配置 -> 依赖/symbol -> 框架 API -> 算子”的顺序定位。
7. 只引入一项最小修复，重跑相同 case，记录 A/B 结果。
8. 基础生成通过后，再做数值、调度、缓存、TP 和性能验证。

## 8. 当前状态

已完成上游/分支基线核对、Ascend 硬件改动分类、Launch/Scheduler/Engine 初步走读、MetaX 风险点 inventory，以及一版本地 correctness-first 候选代码与 CPU 测试。

当前候选代码尚未在 MetaX 真卡验证，不应将其全部视为必要改动，也不应在当前阶段声称 Mini-SGLang 已在 MetaX 跑通。
