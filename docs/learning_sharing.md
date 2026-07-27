# LLM 推理系统学习分享
## —— 从 SGLang 架构到 MetaX C500 适配实践

> **作者**：m01684 | **日期**：2026-07  
> **关键词**：LLM 推理、SGLang、KV Cache、MoE、MetaX C500、硬件适配

---

## 一、为什么要学推理系统？

训练框架（PyTorch、DeepSpeed）之外，**推理系统**是大模型落地的最后一公里。

一个好的推理系统需要同时优化：
- **延迟**（用户等待时间，TTFT / ITL）
- **吞吐**（单位时间服务多少用户，tokens/s）
- **显存利用率**（模型能不能跑起来）

近年来开源推理框架竞争激烈：vLLM、TensorRT-LLM、SGLang……其中 [SGLang](https://github.com/sgl-project/sglang) 由 LMSYS 开源，目前是 xAI（Grok）100K+ GPU 生产部署所用框架，在吞吐和延迟上均有领先优势。

**本次学习的出发点**：在 MetaX C500 上运行 30B MoE 模型（Step-3.5 W8A8），理解推理系统核心原理，并完成 MetaX 适配工作。

---

## 二、LLM 推理基础：Prefill 与 Decode

### 2.1 两阶段计算特性

每个推理请求都经历两个截然不同的阶段：

```
输入 prompt (N tokens)                   输出 (逐 token 生成)
┌──────────────────────────┐            ┌──┬──┬──┬──┬──┐
│  PREFILL（计算密集）       │   →  →  →  │t1│t2│t3│t4│..│
│  并行处理所有输入 token     │            └──┴──┴──┴──┴──┘
│  产生第一个输出 token       │            DECODE（内存带宽密集）
│  GPU 算力充分利用           │            每次只生成 1 token
└──────────────────────────┘            需要读取全部 KV Cache
```

| 阶段 | 计算特点 | 瓶颈 | 关键指标 |
|---|---|---|---|
| Prefill | 计算密集（矩阵乘法） | GPU 算力 | TTFT（首 token 延迟） |
| Decode | 内存带宽密集 | HBM 带宽 | ITL（token 间延迟）、TPOT |

### 2.2 KV Cache：推理系统的核心资源

Decode 阶段每步都要读取**历史所有 token 的 Key/Value**，这就是 KV Cache。

- KV Cache 大小 = `num_layers × 2 × seq_len × num_heads × head_dim × dtype_bytes`
- 一个 30B 模型，seq_len=32K，KV Cache 就需要数十 GB 显存
- **如何高效管理 KV Cache** 是推理系统最核心的问题

---

## 三、SGLang 系统架构

### 3.1 进程设计

SGLang 是多进程架构，各进程通过 **ZeroMQ** 交换控制消息，通过 **NCCL** 传输张量：

```
                    ┌─────────────┐
   用户请求  ──────→ │  API Server  │
                    └──────┬──────┘
                           │ ZMQ
                    ┌──────▼──────┐
                    │  Tokenizer   │
                    └──────┬──────┘
                           │
              ┌────────────▼────────────────────┐
              │   Scheduler (Rank 0)             │
              │   ↕ NCCL broadcast               │
              │   Scheduler (Rank 1..N-1)        │
              └────────────┬────────────────────┘
                           │
                    ┌──────▼──────┐
                    │ Detokenizer  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  API Server  │ → 流式返回用户
                    └─────────────┘
```

*图源：[LMSYS mini-sglang 设计文档](https://lmsys.org/blog/minisgl)*

### 3.2 模块分工

| 模块 | 职责 |
|---|---|
| `api_server` | OpenAI 兼容 API，FastAPI，接收/返回请求 |
| `tokenizer` | 文本 ↔ token 转换 |
| `scheduler` | 调度 batch，管理 KV Cache 分配，TP rank 0 是 master |
| `engine` | 单 GPU 上的模型前向计算、CUDA Graph 回放 |
| `kvcache` | KV Cache pool 管理（Radix / Naive） |

---

## 四、核心优化技术

### 4.1 RadixAttention — KV Cache 自动复用

**论文/博客**：[LMSYS Blog - RadixAttention（2024.01）](https://www.lmsys.org/blog/2024-01-17-sglang/)

**核心思想**：用 Radix Tree（前缀树）管理 KV Cache，相同前缀的请求**自动复用**缓存，无需重新计算。

```
传统方式：每个请求独立计算 KV Cache，相同前缀重复计算
              Req A: [System Prompt][User1] → 计算一次
              Req B: [System Prompt][User2] → System Prompt 重复计算！

RadixAttention：
              System Prompt → [node: cached ✓]
                              ├── User1 → [leaf]
                              └── User2 → [leaf]  ← 命中 System Prompt 缓存
```

![RadixAttention KV Cache 复用示意](https://lmsys.org/images/blog/sglang/radix_attn.jpg)
*RadixAttention 中 Radix Tree 的演化过程：绿色=新增，蓝色=缓存命中，红色=被驱逐*

**四种典型复用场景**：
1. **Few-shot learning**：所有请求共享相同的 few-shot 示例前缀
2. **多轮对话**：对话历史随轮次增长，前缀逐步复用
3. **Self-consistency**：同一 prompt 多次采样，前缀完全相同
4. **Tree-of-thought**：分支推理共享根节点

**性能收益**：比 vLLM 吞吐提升最高 **5x**（KV Cache 命中率高的场景）

**实现细节**：
- Radix Tree 在 CPU 上维护（低开销）
- KV Cache 本体以 **paged layout** 存储在 GPU HBM（每 token 一页）
- LRU 驱逐策略：递归驱逐叶节点

### 4.2 Chunked Prefill — 长 prompt 防 OOM

**来源**：[Sarathi-Serve（arXiv 2403.02310）](https://arxiv.org/abs/2403.02310)

**问题**：长 prompt（如 128K tokens）的 prefill 会占用极大内存并产生超长延迟，导致其他请求被阻塞。

**解法**：把 prefill 切成固定大小的 chunk，与 decode batch **交替执行**：

```
传统方式（一次性 prefill）：
  [===Prefill 128K tokens===] → [Decode] → [Decode] → ...
   ^很长，期间其他请求饥饿

Chunked Prefill（chunk_size=2K）：
  [Prefill 2K] → [Decode batch] → [Prefill 2K] → [Decode batch] → ...
   ^分散计算，decode 不被饥饿
```

**配置**：`--max-prefill-length N`（chunk 大小），太小会有 overhead，建议 ≥ 512。

### 4.3 Overlap Scheduling — CPU/GPU 流水线

**博客**：[SGLang v0.4（2024.12）](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)  
**原始思想**：[NanoFlow（arXiv 2408.12757）](https://arxiv.org/abs/2408.12757)

**问题**：传统方式中，CPU 调度（Radix Cache 查找、batch 组装）在 GPU 计算完成后才开始，GPU 有空闲。

```
传统调度：
  GPU: [compute batch N] [idle] [compute batch N+1] [idle]
  CPU:                   [schedule N+1]              [schedule N+2]

Overlap Scheduling：
  GPU: [compute batch N] [compute batch N+1] [compute batch N+2]
  CPU: [schedule N+1]    [schedule N+2]      [schedule N+3]
```

![Overlap Scheduling 示意](https://lmsys.org/images/blog/sglang_v0_4/scheduler.jpg)

**实现**：Scheduler 提前一个 batch 完成调度，准备好下一个 batch 的所有 metadata，GPU 不再等待 CPU。

**收益**：在小模型 + 大 TP 配置下提升最显著，整体约 **1.1x ~ 1.3x** 吞吐提升。

### 4.4 CUDA Graph — Decode 阶段 CPU 零开销

**问题**：Decode 每步计算量很小（单 token），但 Python → CUDA kernel 的 launch overhead 占比极高。

**解法**：预先"录制" GPU 操作序列（CUDA Graph），Decode 阶段**直接回放**而非逐 op 提交：

```python
# 第一次：录制
with torch.cuda.graph(cuda_graph):
    output = model.forward(input)

# 后续：回放（极低 CPU overhead）
cuda_graph.replay()
```

**限制**：只适合固定 shape（batch size 固定），因此 prefill 不能用 CUDA Graph。  
**配置**：`--cuda-graph-max-bs N`，N=0 关闭。

---

## 五、MoE 模型推理的特殊挑战

### 5.1 MoE 架构回顾

MoE（Mixture of Experts）将 FFN 层替换为多个"专家"，每个 token 只激活其中 Top-K 个：

```
Dense FFN:
  token → Linear(h→4h) → Activation → Linear(4h→h)
           ↑ 每个 token 都经过同一个 FFN

MoE FFN:
  token → Router → [Expert 0] ↘
                 → [Expert 1] ─ Weighted Sum → output
                 → [Expert 2] ↗
       (Top-K=2, 只激活 K 个专家)
```

**优势**：参数量大（记忆容量强），但激活参数少（计算量不变）。  
**代价**：推理时需要 **all-to-all 通信**（Expert Parallelism），负载天然不均衡。

### 5.2 Expert Parallelism（EP）

大规模部署时，Expert 分散在不同 GPU 上：

```
GPU0: Expert 0,1,2...
GPU1: Expert 8,9,10...
...

每个 token 需要路由到对应 GPU 上的专家 → all-to-all 通信
```

**负载不均衡**：少数热门 expert 处理大量 token，影响整体吞吐。  
**解法**（来自 [LMSYS 大规模 EP blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/)）：

- **EPLB**：设置冗余 expert（256 + 32 个），根据使用频率统计重新分配
- 效果：prefill **1.49x**，decode **2.54x** 吞吐提升（96 H100 规模）

### 5.3 我的实践：在 MetaX C500 上适配 MoE

**背景**：SGLang 的 MoE 计算依赖 `sgl_kernel`（NVIDIA 编译产物），在 MetaX C500 上无法使用。

**解法**：实现纯 PyTorch 的 `MetaxMoe` backend，绕过 NVIDIA kernel 依赖：

```python
# python/minisgl/moe/metax.py
class MetaxMoe(BaseMoeBackend):
    def forward(self, hidden_states, w1, w2, gating_output, topk, renormalize,
                activation="silu", apply_router_weight_on_input=False):
        T, K = hidden_states.shape
        E = w1.shape[0]
        
        # 1. Router：softmax + topk（替代 sgl_kernel.topk_softmax）
        topk_weights, topk_ids = _pt_topk_softmax(gating_output, topk, renormalize)
        
        output = torch.zeros(T, K, ...)
        for e in range(E):
            # 2. 找出路由到 expert e 的 token
            token_mask = (topk_ids == e).any(dim=-1)
            if not token_mask.any(): continue
            
            # 3. Gate-Up projection + SiLU + Down projection
            x = hidden_states[token_mask]
            gate_up = x @ w1[e].T
            gate, up = gate_up.chunk(2, dim=-1)
            x_mid = F.silu(gate) * up
            x_out = x_mid @ w2[e].T
            
            # 4. 按 routing weight 加权累加
            w_out = (slot_mask[token_mask].float() * topk_weights[token_mask]).sum(-1)
            output[token_mask] += x_out * w_out.unsqueeze(-1)
        
        return output
```

**注册到 backend registry**：
```python
@SUPPORTED_MOE_BACKENDS.register("metax")
def create_metax_moe_backend():
    from .metax import MetaxMoe
    return MetaxMoe()
```

**engine 自动选择**：
```python
if config.model_config.is_moe and config.moe_backend == "auto":
    backend = "metax" if platform == "metax" else "fused"
```

**验证**：20 个 CPU 单元测试全部通过，无需 GPU 即可验证正确性。

---

## 六、Speculative Decoding（投机解码）

### 6.1 原理

核心思想：用小模型"草拟"多步 token，大模型一次性"验证"：

```
普通 Decode：
  大模型 → t1 → 大模型 → t2 → 大模型 → t3  （3次大模型调用）

Speculative Decode：
  小模型 → [t1, t2, t3]（草稿）
  大模型 → 一次验证，接受 t1, t2，拒绝 t3，输出修正 token
  （1次大模型调用，但产出2个 token）
```

**收益**：在大模型是瓶颈（而非小模型）时，有效提升 token 吞吐。

### 6.2 MTP（Multi-Token Prediction）

Step-3.5 等模型内置了 MTP（多 token 预测）层，可直接作为草稿模型，不需要额外小模型。

SGLang 中通过 `--speculative-algorithm EAGLE --speculative-num-steps 3` 启用。

### 6.3 踩坑实录：speculative-eagle-topk 缺失

```
# Bug 现象：
TypeError: '>' not supported between instances of 'NoneType' and 'int'
# 位置：speculative_hook.py:378

# 原因：未传 --speculative-eagle-topk，内部读到 None
# 修复：加上 --speculative-eagle-topk 1
```

---

## 七、性能测试方法论

### 7.1 三种测试模式

正确的 benchmark 需要明确测什么、如何隔离变量：

| 模式 | 目标 | Server 配置 | Bench 配置 |
|---|---|---|---|
| **Prefill-only** | 测 prefill 吞吐 | 关闭 prefix cache，关闭 MTP | output_len=1 |
| **Decode-only** | 测 decode 吞吐 | 开启 prefix cache + MTP | input_len=0, random_prefix_len=输入长度（全命中 cache）|
| **PD 一体** | 综合性能 | MTP 按需，prefix cache 按需 | 正常 input/output |

**关键设计思路**：
- Decode-only 为何用 prefix cache 命中？→ 让 prefill 开销趋近于0，纯测 decode
- Prefill-only 为何 output=1？→ 最小化 decode 阶段影响

### 7.2 关键指标解释

```
TTFT  = Time To First Token    首 token 延迟，主要由 prefill 决定
TPOT  = Time Per Output Token  平均每个输出 token 的时间（含等待）
ITL   = Inter-Token Latency    token 间隔，主要由 decode 决定
E2EL  = End-to-End Latency     总延迟
```

### 7.3 并发数与吞吐的关系

```
吞吐（tokens/s）
    ↑
    │          ████████████  ← 饱和后维持平台（compute bound）
    │      ████
    │   ███
    │ ██
    │█
    └──────────────────────→ 并发数（batch size）
    1  8  16  32  64  128

关键问题：找到"膝盖点"（knee point）—— 延迟开始显著增长的并发数
```

### 7.4 实测结果：Qwen3-30B-A3B.w8a8 on 8×MetaX C500

> 环境：SGLang 0.5.13+maca3.8.1.0，W8A8 量化，8卡张量并行  
> 测试方式：流式 `/v1/completions`，多线程并发，各点跑 `max(conc×3, 8)` 个请求

**Prefill-heavy（in=1024, out=16）** — 测 prefill 吞吐，探首 token 延迟

| 并发数 | Output 吞吐 (tok/s) | TTFT p50 (ms) |
|:------:|--------------------:|:-------------:|
|   1    |              10.8   |    136        |
|   4    |              42.3   |    254        |
|   8    |              77.6   |    375        |
|  16    |             115.7   |    567        |

**Decode-heavy（in=64, out=256）** — 测 decode 吞吐，probe token 间延迟

| 并发数 | Output 吞吐 (tok/s) | TTFT p50 (ms) |
|:------:|--------------------:|:-------------:|
|   1    |              11.6   |    109        |
|   4    |              43.4   |     50        |
|   8    |              82.2   |    141        |
|  16    |             173.0   |     98        |

**Mixed PD（in=512, out=128）** — 综合场景，接近生产负载

| 并发数 | Output 吞吐 (tok/s) | TTFT p50 (ms) |
|:------:|--------------------:|:-------------:|
|   1    |              11.3   |    125        |
|   4    |              41.8   |    187        |
|   8    |              84.6   |    189        |
|  16    |             171.7   |    363        |
|  32    |             336.9   |    518        |

**吞吐 vs 并发数（Mixed PD）：**

```
tok/s
 350 │                                              △  337
     │
 250 │
     │
 170 │                              ○ △  172
     │
  85 │              ○ △   85
     │
  42 │  ○ △   42
     │
  11 │ △  11
   0 └──────────────────────────────────────────→ concurrency
       1           4             8             16            32

   △ Mixed PD (in=512,out=128)
   ○ Decode-heavy (in=64,out=256)   [最高并发只到16]
```

**关键观察：**
- Decode-heavy 在 conc=16 达 **173 tok/s**，比 prefill-heavy（116 tok/s）快 **50%** — decode 更易批处理
- Mixed PD 吞吐从 conc=1 到 conc=32 提升约 **30x**，接近线性扩展
- TTFT 在 conc=32 仍保持 **518ms**，适合交互场景

---

## 八、硬件适配方法论：如何把 NVIDIA-only 代码跑在 MetaX 上

### 8.1 核心挑战

SGLang 代码有三层依赖需要解耦：

```
Python API 层（可移植）
    ↓
PyTorch 算子层（device="cuda" 可能实际运行在 MACA 上）
    ↓
Kernel 层（sgl_kernel、FlashAttention 等，NVIDIA 编译，不可移植）
```

MetaX 已解决第二层（`torch` 在 MACA 上可以运行），主要工作在第三层。

### 8.2 解耦 platform 与 device_type

关键设计：分离"硬件平台"和"设备类型标识"：

```python
# 错误做法：device_type = platform  （绑定太死）
# 正确做法：
device_type = "cuda"   # 让PyTorch算子走 CUDA 路径（MACA 兼容）
platform = "metax"     # 控制业务逻辑（选择哪个 kernel）

if platform == "metax":
    moe_backend = "metax"   # 用纯 PyTorch 实现
else:
    moe_backend = "fused"   # 用 sgl_kernel CUDA kernel
```

### 8.3 调试经验汇总

| 问题 | 现象 | 根因 | 解法 |
|---|---|---|---|
| GPU 被占 | `mxkwCreateQueueBlock timeout` | 旧 server 进程未清理，占用 GPU 0 compute queue | 启动前 `kill -9 <pid>` |
| SSH heredoc | `exit 255` with single quotes | SSH 单引号转义问题 | 改用 heredoc `<< 'ENDSSH'` |
| MTP crash | `KeyError: weight_scale` | checkpoint 无 MTP 权重（仅配置 num_nextn=3 但无实际权重）| 确认 checkpoint 完整性 |
| OS page cache | 第二次 load 快 70x | 内核 page cache 命中 | 测新功能前 `echo 3 > /proc/sys/vm/drop_caches` |

---

## 九、总结

### 学到了什么

```
理论层面：                          实践层面：
┌─────────────────────────┐        ┌──────────────────────────────┐
│ Prefill/Decode 两阶段    │        │ MetaxMoe 纯 PyTorch 实现      │
│ KV Cache 生命周期管理     │        │ platform/device_type 解耦     │
│ RadixAttention 前缀复用  │        │ CUDA Graph 在非 NVIDIA 上限制 │
│ Chunked Prefill 流水线   │        │ MTP speculative decode 踩坑  │
│ Overlap Scheduling       │        │ benchmark 三模式设计思路      │
│ MoE EP + 负载均衡        │        │ SSH heredoc 调试技巧          │
└─────────────────────────┘        └──────────────────────────────┘
```

### 一个关键认知

**推理系统的优化本质是资源的时间/空间复用**：
- RadixAttention = KV Cache 的**空间复用**（相同前缀只算一次）
- Chunked Prefill = 计算的**时间复用**（prefill/decode 交替，避免饥饿）
- Overlap Scheduling = CPU/GPU 的**时间并行**（调度与计算重叠）
- CUDA Graph = kernel launch 的**时间压缩**（批量提交替代逐 op 提交）

### 下一步

- **M2**：用 MACA Triton grouped-matmul 替代 MetaxMoe 的 Python for-loop（深度优化 MoE 专家计算）
- **EP 适配**：deep_ep 在 MetaX 上的 all-to-all 通信适配（跨卡 MoE dispatch）
- ~~**Benchmark**：在 C500 上跑 Qwen3-30B W8A8 benchmark~~ ✅ 已完成（详见 §7.4）

---

## 参考资料

| 资料 | 链接 |
|---|---|
| SGLang 官方 Blog（RadixAttention） | [lmsys.org/blog/2024-01-17-sglang](https://www.lmsys.org/blog/2024-01-17-sglang/) |
| SGLang v0.4 Overlap Scheduling | [lmsys.org/blog/2024-12-04-sglang-v0-4](https://lmsys.org/blog/2024-12-04-sglang-v0-4/) |
| Large-Scale EP + PD Disaggregation | [lmsys.org/blog/2025-05-05-large-scale-ep](https://lmsys.org/blog/2025-05-05-large-scale-ep/) |
| NanoFlow (Overlap Scheduling 来源) | [arxiv.org/abs/2408.12757](https://arxiv.org/abs/2408.12757) |
| Sarathi-Serve (Chunked Prefill 来源) | [arxiv.org/abs/2403.02310](https://arxiv.org/abs/2403.02310) |
| mini-sglang-metax 项目 | `Documents/sglang框架知识学习理解/mini-sglang-metax/` |
