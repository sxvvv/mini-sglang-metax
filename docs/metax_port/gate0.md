# MetaX Gate 0：原生 CUDA 路径基线与最小适配

## 目标

在一张 MetaX C500/C550 上，先使用未加 MetaX fallback 的 Mini-SGLang CUDA-facing 路径运行 dense Qwen 小模型，得到一个可复现的 PASS 或第一个有效失败。若失败，只为该失败增加最小框架侧修复，然后重跑相同 case。

Gate 0 的目标是确定适配边界和完成功能正确性，不是证明性能领先。

## 固定变量

| 项目 | Gate 0 固定值 |
| --- | --- |
| 硬件 | 1 x MetaX C500 或 C550 |
| 模型 | Qwen2.5-0.5B-Instruct 或 Qwen3-0.6B dense |
| dtype | bf16 |
| TP | 1 |
| 执行 | eager，CUDA Graph 关闭 |
| 采样 | greedy / `temperature=0` |
| 不覆盖 | MoE、量化、MTP、TP2+、性能对比 |

## Stage A：环境事实

必须记录：

- GPU 型号与数量。
- 驱动、MACA 和容器版本。
- Python、PyTorch、`sgl_kernel`、FlashInfer、Triton 版本与安装路径。
- `torch.cuda.is_available()`、`torch.cuda.device_count()`、设备名称与 capability 输出。
- MACA/CUDA compatibility 环境变量与 `libcudart.so` 解析结果。
- Mini-SGLang 的 commit、分支与工作区状态。

安装或加载项目时不得让 pip 替换 vendor PyTorch、`sgl_kernel` 或其他已匹配的定制包。

## Stage B：API/symbol inventory

首先检查而不修改代码：

1. `torch.cuda` 设备、Stream、Event、synchronize、mem_get_info 和 bf16 GEMM。
2. Mini-SGLang 选定 Attention backend 需要的 FlashInfer 或 vendor `sgl_kernel` symbol。
3. RMSNorm、RoPE、SwiGLU 和 greedy sampling 所需 API。
4. tvm-ffi 与 Mini-SGLang `store/index` `.cu` JIT 所需编译器、header 和链接库。
5. `torch.cuda.get_device_capability()` 的输出是否可被 Mini-SGLang 的 SM90/SM100 分支正确解释。

inventory 失败不等于 Gate 0 结束；它用于预测可能的首错，真正的框架结论仍由 Stage C 产生。

## Stage C：未修改 CUDA 基线

首先使用干净 `ascend-port=85e3886` 保留的 CUDA 路径，运行 dense Qwen、TP1、bf16、eager、greedy，并关闭 CUDA Graph。不先启用本地 `torch_native` Attention 或其他 MetaX fallback。若需要区分上游 NVIDIA 假设与 Ascend 分支差异，再使用 `main=9a91cfa` 运行相同对照 case。

必须保留：

- 完整启动命令。
- 当时的 commit 和 diff 状态。
- 从进程启动到失败/请求结束的完整日志。
- 第一个有效错误，而不是 worker 退出后的通信中断等派生错误。
- 进程退出码、服务存活状态和设备显存状态。

## Stage D：首错分类与最小修复

按以下顺序处理：

1. **环境**：容器、驱动、MACA、编译器、库路径或版本不匹配。
2. **配置**：后端选择、Graph、模型参数或启动参数错误。
3. **依赖/symbol**：vendor wheel 存在，但 Mini-SGLang 需要的模块或 symbol 不存在。
4. **框架 API**：输入、输出、dtype、layout 或异步语义不兼容。
5. **算子**：必需运算未实现或数值错误。

前一类能解决时，不进入后一类。每次只增加一项修复，重跑相同 Stage C case 并保存 A/B 日志。

## 候选 fallback 的使用规则

当前 `metax-port` 中的平台识别、`torch_native` Attention、基础算子 fallback 和 PyNCCL/Graph 限制属于 correctness-first 候选实现。

候选改动只在以下条件成立时保留：

- 已有原始 CUDA 基线的真卡失败证据。
- 已有能单独复现问题的最小测试。
- 修改后相同测试与端到端 case 通过。
- 无法通过 vendor 环境、现有配置或已有 API 解决。

## PASS 标准

1. Stage A 和 Stage B 记录完整。
2. Stage C 的原始基线日志已保存。
3. 每项保留的代码改动都对应真卡证据和最小复现。
4. 模型加载成功，两次连续 greedy 请求返回非空输出。
5. 第二次请求后服务保持存活，无明显资源泄漏。
6. verdict 明确列出通过项、失败项、未覆盖项和下一 Gate。
