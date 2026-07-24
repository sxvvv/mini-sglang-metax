# Mini-SGLang MetaX 阶段汇报

## 一句话结论

已将现有 Mini-SGLang 适配收口为独立项目 `mini-sglang-metax`，并在
8 张 MetaX C500 的真实环境中完成 Qwen3-8B BF16 单卡真权重加载、Prefill、
Decode、Greedy Sampling、连续两请求和 KV Cache 回收验证；同时在单张 MetaX
C550 上完成 HTTP/ZMQ、非流式与 SSE OpenAI API、取消确认和四路并发验证。
进一步完成 3 轮 x 8 并发的有界 Gate 1.2、调度器真实 batch 观测、过载排队与
故障恢复验证，并打通 MetaX `flashinfer` 实验性 TP1 真模型路径。限定范围内离线
Gate 0、在线 Gate 1.2 和 `fi` Gate 1 均为 PASS。

## 项目定位

`mini-sglang-metax` 不是重写一套 MACA/MCCL，也不是把 Ascend 的 NPU 分支直接
复制到 MetaX。项目目标是利用 Mini-SGLang 约 5K 行的紧凑实现，建立一条可读、
可改、可验证的 MetaX 推理框架白盒链路：

```text
Request -> Scheduler -> KV Cache -> Model -> Attention -> Sampler -> Output
```

MetaX vendor PyTorch 对外仍是 `torch.cuda` 接口，但 NVIDIA 专属的 FlashInfer、
`sgl_kernel`、CUDA Graph、PyNCCL 和架构探测不能默认兼容。因此项目新增独立
`metax` 平台标识，只在硬件语义确实不同的位置路由到 correctness-first fallback。

三条硬件路线在本项目中的边界如下：

| 路线 | PyTorch 设备接口 | 厂商栈与集合通信 | 本项目中的执行路径 |
| --- | --- | --- | --- |
| NVIDIA | `torch.cuda` | CUDA / NCCL | `platform=nvidia`，满足依赖时使用上游 NVIDIA fused backend |
| 华为 Ascend | `torch.npu` + `torch_npu` | CANN / HCCL | `platform=ascend`，走独立 NPU runtime dispatch 与显式 `npu_fia` 路径 |
| MetaX | vendor `torch.cuda` 兼容接口 | MACA / MCCL | `platform=metax`，默认 eager `torch_native`，MetaX `flashinfer` 仅显式实验启用 |

这里比较的是框架路由契约，不是二进制兼容性或性能排名。MetaX 暴露
`torch.cuda` 只表示 vendor PyTorch 保留了 CUDA-facing API，并不意味着
FlashInfer、`sgl_kernel`、CUDA Graph 或 PyNCCL 可以按 NVIDIA 环境直接加载；
同样，Ascend 的 `torch_npu`、HCCL 和 FIA 路径也不能复制后当作 MetaX 适配。

因此 MetaX 版本的核心增量是：把“设备 API 名称”和“真实加速器平台”解耦；
按 `platform=metax` 绕开 NVIDIA-only fused op；提供可读的 eager PyTorch 正确性
路径；使用 vendor `torch.distributed` 兼容层；按目标机已安装签名适配可选
MetaX `flashinfer`；最后以 C500/C550 上的真模型、KV Cache、在线取消、真实
batching、受限过载与恢复证据闭环，而不是只证明模块能够导入。

## 本阶段完成内容

1. 建立独立项目目录和发行名 `mini-sglang-metax`；Python import 保留 `minisgl`，
   避免破坏上游 API。
2. 完成 MetaX 平台识别，区分 `device_type=cuda` 与 `platform=metax`。
3. 增加 eager `torch_native` Attention，以及 Norm、RoPE、Activation、Embedding、
   Sampling 等 NVIDIA 专属路径的 PyTorch fallback。
4. MetaX 路径默认关闭 CUDA Graph 和 PyNCCL，TP 多卡使用 vendor
   `torch.distributed` NCCL/MCCL 兼容层。
5. 修复离线推理被无关 `msgpack/ZMQ` 顶层导入阻断的问题。
6. 建立 `preflight.py`、统一 Gate 0 脚本和结构化 PASS/FAIL 输出。
7. 将源码、环境清单和日志保存到持久 DTFS 目录。
8. 新增在线 Gate 1 脚本，验证服务启动、模型列表、两次非流式 OpenAI Chat 请求、
   JSON 响应和进程组清理。
9. 新增在线 Gate 1.1，验证完整 SSE、客户端主动断流、AbortAck、四路不同长度并发、
   取消后恢复请求和无残留退出。
10. 新增在线 Gate 1.2：服务端 `max_running_requests=2`，3 轮各 8 路同时请求，
    覆盖排队、HTTP 422 输入故障、实时断流、恢复请求和进程组清理。
11. 在调度器选出 batch 后、模型 forward 前输出稳定 `SchedulerBatch` 记录，区分
    “客户端并发”与“服务端实际 batching”。
12. 盘点 MetaX `flashinfer`/`mcoplib`，修复可选 `backend`/`seq_lens` API 差异，
    完成 Qwen3-8B TP1 `fi` 在线 Gate 1；默认仍为 `torch_native`。

## 真卡结果

| 项目 | 结果 |
| --- | --- |
| 硬件 | 8 x MetaX C500 |
| PyTorch | `2.10.0+metax3.8.1.0` |
| 正式模型 | Qwen3-8B BF16，16.38 GB |
| 真权重并行度 | TP1 |
| 执行方式 | BF16、eager、`torch_native` Attention |
| 独立项目加载 | `14.7772 s`（文件缓存已热） |
| 第一次请求 | `2.2605 s`，tokens `[25010, 10, 4999, 1725]` |
| 第二次请求 | `0.1823 s`，输出完全一致 |
| KV Cache | 两次请求后均恢复 `512/512` |
| 进程状态 | exit code `0` |

补充验证：TP2 all-reduce PASS；合成 Qwen3 检查点 TP2、TP8 端到端均 PASS，
各 rank 输出一致且缓存完整回收。

上海 C550 补充验证：从只读模型存储加载 Qwen3-8B BF16 真权重，在线 TP1 服务
成功启动；`/v1/models` 和两次 `/v1/chat/completions` 均返回 HTTP 200，输出一致，
脚本返回码为 `0`，退出后端口关闭且无残留 worker。

在线 Gate 1.1 严格重跑：SSE 收到 4 个数据事件、finish 事件和 `[DONE]`；主动断流
后日志依次出现 `Aborting request for user 3` 和
`Abort acknowledged for user 3`；四路并发请求全部完成，随后恢复请求成功；脚本
返回码 `0`，1919 端口关闭且无残留进程。

在线 Gate 1.2：24/24 有界请求完成、失败 0、客户端峰值并发 8；配置运行上限 2，
调度日志观测到 144 个真实 batch，其中 99 个为多请求 batch，最大 batch 2、最大
pending 6。输入故障返回 HTTP 422，实时断流完成 AbortAck，最终恢复请求成功；
脚本返回码 `0`、端口关闭且无残留进程。`7.2059 s` 只描述本次有界流程，不作为
性能比较。

厂商 Attention：环境中 `flashinfer 0.2.6+metax3.8.1.0torch2.10` 与
`mcoplib 0.4.8+maca3.8.0.24.torch2.10` 可导入；`mcflashinfer` 模块名不存在。
两次失败探测分别定位到 decode 构造器不接受 `backend=`、plan 不接受
`seq_lens=`，按安装签名过滤可选参数后，最终 Qwen3-8B TP1 `fi` Gate 1 返回码
`0`，两个 Chat 请求均为 HTTP 200，端口关闭且无残留 worker。

当前上海分配仅暴露 `device_count=1`、`MetaX C550`，因此未运行也未声称 TP2
真模型在线矩阵；该项明确标记为资源阻塞。

## 解决问题链

```text
环境确认
  -> 识别机器实际为 8 x C500，不是 Ascend NPU
  -> 发现无 CANN/torch_npu，存在 MACA cu-bridge
  -> MetaX 平台识别与 CUDA-facing API 解耦
  -> Gate 0 预检 PASS
  -> 首个端到端错误：离线模式误依赖 msgpack
  -> 延迟导入 ZMQ 队列
  -> 随机小模型 TP1/TP2/TP8 PASS
  -> 挂载 Qwen3-8B BF16 真权重
  -> 同进程连续双请求 PASS
```

## 可复现入口

```bash
cd /path/to/mini-sglang-metax
export MODEL_PATH=/path/to/Qwen3-8B
export SW_HOME=/persistent/path/${USER}
bash scripts/metax/run_gate0.sh

# 上海 C550 在线 Gate 1
export MODEL_PATH=/path/to/Qwen3-8B
bash scripts/metax/run_online_gate1.sh

# 上海 C550 在线 Gate 1.1
bash scripts/metax/run_online_gate1_1.sh

# 上海 C550 有界在线 Gate 1.2
bash scripts/metax/run_online_gate1_2.sh

# 实验性 MetaX flashinfer Gate 1（默认后端仍为 torch_native）
ATTENTION_BACKEND=fi RESULT_PREFIX=online_gate1_fi_probe \
  bash scripts/metax/run_online_gate1.sh
```

持久证据目录：

```text
${SW_HOME}/results/mini-sglang-metax/2026-07-23/
${SW_HOME}/results/mini-sglang-metax/2026-07-24/
```

## 结论边界

当前可以汇报：Mini-SGLang 已在 MetaX C500/C550 上完成真权重、完整离线推理
主链和连续请求正确性验证，完成 TP2/TP8 多卡控制面验证，并在 C550 上通过
限定范围的 TP1 HTTP/ZMQ、OpenAI API、SSE、取消确认、有界过载排队、真实
batching 和故障恢复验证；MetaX `flashinfer` 实验性 TP1 真模型路径也已通过
基础 Gate 1。

当前不能汇报：生产可用、性能领先、多小时长稳、生产级过载拒绝、TP2+ 真模型
在线服务、CUDA Graph/PyNCCL 已适配、MoE/量化模型已支持。`fi` 尚未覆盖
Gate 1.1/1.2、TP2 或广泛模型形状；`mcoplib` 仅完成版本/API 盘点，未直接集成。

开源发布候选补充检查：Windows CPU 公开 CI 集合 `49 passed, 4 skipped`
（跳过项仅因无 pinned allocator）；MetaX C550 同集合 `53 passed`；目标端
Python 编译、shell 语法与残留进程审计通过；sdist/wheel 构建和公开卫生扫描通过。
时间数据只用于确认流程结束，不用于跨框架性能比较。

## 下一阶段

1. 获得至少 2 张同机可见 C550/C500 后，运行 TP2 真模型 Gate 1/1.1/1.2 矩阵。
2. 为在线前端增加显式队列上限与 HTTP 429/503 策略，再验证拒绝和恢复语义。
3. 将 `fi` 扩展到 Gate 1.1/1.2、更多 batch/序列形状和数值对照后再讨论默认策略。
4. 运行真实多小时 soak，并单独发布内存、吞吐、尾延迟和故障注入结果。
5. 在证据完成后再发布性能数据和正式版本。
