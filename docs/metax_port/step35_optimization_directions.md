# Step-3.5 W8A8 C500 TP8 优化方向

> 日期：2026-07-27  
> 模型：Step-3.5 W8A8（`vllm_quant_model_main`，30B+ MoE，45 层，TP8）  
> 设备：MetaX C500，8 卡  
> 框架：SGLang 0.5.13（MetaX 定制版）  
> 当前状态：MTP smoke 进行中（2026-07-27 10:26– CST），M1（MetaX PyTorch MoE backend）已完成

---

## 1. 当前基线摘要

### Prefill（4096×1，`random-ids`，`inf` arrival rate）

| 并发 | 输入吞吐 (tok/s) | 平均 TTFT (ms) |
|---:|---:|---:|
| 1  | 8 771 | 465 |
| 8  | 11 488 | 2 244 |
| **16** | **14 181** | 4 090 |
| 32 | 14 249 (+0.5%) | 7 766 |

**拐点：并发 16**，之后吞吐饱和，TTFT 线性增长。

### Decode（4096×256，`random-ids`，`inf` arrival rate）

#### C550 TP4 基线（对比参考）

| 并发 | 输出吞吐 (tok/s) | 平均 TPOT (ms) | 平均 TTFT (ms) |
|---:|---:|---:|---:|
| 1  | 8.37   | 118 | 436 |
| 8  | 60.11  | 126 | 2 022 |
| 16 | 114.52 | 129 | 2 991 |
| 32 | 201.39 | 139 | 5 289 |

#### C500 TP8 实测（2026-07-27，`fake_sharegpt`，`SGLang 0.5.13+maca3.8.1`）

| 并发 | req/s | 输出吞吐 (tok/s) | 平均 TTFT (ms) | 平均 TPOT (ms) | 平均 ITL (ms) |
|---:|---:|---:|---:|---:|---:|
| 32  | 1.06 | **271** | 2 019 | 226 | 187 |
| 48  | 1.35 | **346** | 2 585 | 223 | 189 |
| 64  | 1.63 | **417** | 3 233 | 274 | 202 |
| 96  | 2.06 | **527** | 4 510 | 264 | 236 |
| 128 | 2.51 | **643** | 5 724 | 302 | 269 |

**结论：**
- c=128 仍无吞吐拐点，throughput 线性增长（271→643 tok/s），未到 GPU 算力/显存上限
- ITL 随并发线性增长（187→269 ms），每增加 32 并发 ITL +约 30ms → `torch_native.py` per-request 串行 decode attention 的典型特征
- TPOT 240–302 ms 区间，比 C550 TP4 的 139 ms 高约 1.7×（8 卡 TP 通信开销 + MoE routing 串行）
- **下一步**：推至 c=192/256 找饱和点；优先实施 E3（批量 decode attention）以压制 ITL 增长

### 当前活跃 workaround（均为性能限制项）

| 开关 | 原因 | 待解决 |
|---|---|---|
| `SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=0` | JIT 编译在 MetaX 上失败（libcudart 路径缺失）；**已测试 JIT=1（2026-07-27）：patch 了 `moe_fused_gate.py` 中 `routed_scaling_factor=None` crash，但 MACA Triton JIT kernel 性能与 fallback 相当（TPOT 271ms vs 274ms，噪声级别）** | **维持 JIT=0，不再追踪** |
| `--disable-cuda-graph` | MetaX 尚未实测 Graph 可用性 | 探索性验证（低优先） |
| `--disable-radix-cache` | 正确性未在 MetaX 上验证 | 先做 smoke，再开启 |
| `--json-model-override-args '{"routed_scaling_factor":3.0}'` | checkpoint 字段命名不一致 | 已 workaround，结构性解决需上游修复 |

---

## 2. 优化方向

### 方向 A：JIT Fused TopK ~~修复~~（**已关闭**，2026-07-27）

**结论：MACA Triton JIT 无提升，维持 `SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=0`**

#### A/B 实测结果（c=64，Step-3.5 W8A8 C500 TP8）

| 指标 | JIT=0（基线） | JIT=1（实测） | Δ |
|---|---|---|---|
| req/s | 1.63 | 1.62 | −0.6%（噪声） |
| TTFT (ms) | 3 233 | 3 026 | −6%（波动范围内） |
| TPOT (ms) | 274 | 271.6 | −0.9%（噪声） |
| ITL (ms) | 202 | 202 | 0% |

**过程记录**

- libcudart symlink 修复已应用：`ln -s /opt/maca/tools/cu-bridge/lib/libcuda.so ~/cuda-runtime-lib/libcudart.so`
- JIT 编译成功，server 正常启动（09:21:52 CST）
- `moe_fused_gate.py` Bug 发现并已 patch：`float(routed_scaling_factor if routed_scaling_factor is not None else 1.0)`（Step-3.5 checkpoint 中 `routed_scaling_factor` 为 None，JIT 路径无 None guard）
- Benchmark（c=64, n=128）完整运行通过，所有指标与 JIT=0 在误差范围内持平

**结论分析**：MACA Triton JIT 编译的 `biased_topk` kernel 与 PyTorch fallback 路径在 C500 上性能相当。可能原因：
1. MACA Triton 后端生成的 ISA 与 C500 native 实现等效
2. Step-3.5 MoE routing 的 TopK 操作不是当前瓶颈（瓶颈在 Decode Attention 串行 dispatch，即方向 E3）

**行动**：`SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=0` 维持现状，方向 A 关闭。

---

### 方向 B：Decode 吞吐上限测量（高优先，零代码改动）

**背景**

当前 Decode 基线在并发 32 时输出吞吐 201 tok/s，TPOT 仍在增长但未饱和。缺少上限数据，无法量化其他优化的"提升空间"。

**实验**：扩展并发至 64、128（参见 `scripts/metax/bench_step35_decode_sweep.sh`）

**预期观察**
- 出现明显 TPOT 拐点（TPOT 急剧上升、吞吐增长放缓）→ 确认 GPU 算力边界
- 若一直未饱和 → 瓶颈在 KV Cache 容量或批调度，而非算力
- P99 TTFT 超出可接受范围的并发 → 确定生产可用并发上限

---

### 方向 C：MTP（Multi-Token Prediction）Smoke（**进行中**，2026-07-27）

**背景**

checkpoint：Step-3.5 W8A8 MTP variant（`num_nextn_predict_layers=3`）  
模型配置：`num_nextn_predict_layers=3`  
SGLang 0.5.13 识别了 Step3p5 EAGLE 路径（`Enable multi-layer EAGLE speculative decoding for Step3p5ForCausalLM model.`）。

**实验脚本**：`scripts/metax/mtp_smoke.sh`，STEP=1（load-only验证）

**已发现并修复的 bug**

| # | 错误 | 修复 |
|---|---|---|
| 1 | `--num-speculative-steps` → `unrecognized arguments` | 改为 `--speculative-num-steps` |
| 2 | `speculative_eagle_topk > 1` → `TypeError: '>' not supported … NoneType` | 加入 `--speculative-eagle-topk 1` |
| 3 | 旧 minisgl Qwen3-8B server（port 1919）占用 GPU 0 compute queue | 启动 MTP smoke 前先 kill 旧 server |

**当前状态（2026-07-27 10:35 CST）**

- 服务器正在加载模型：`Multi-thread loading shards: 2% | 1/44, ~4min`
- W8A8 量化正确识别：`CompressedTensorsW8A8Int8DynamicMoE`
- 结论待定（READY 后确认）

```bash
# 在启动日志中确认 MTP 层数被正确识别
# （从 scripts/metax/mtp_smoke.sh 启动，含所有必要 flag）
```

**预期影响**：MTP/EAGLE 在 decode-heavy 负载下可提升 2–3× 输出吞吐（取决于 draft acceptance rate）。  
**风险**：SGLang 0.5.13 的 Step3p5 MTP 路径在 MetaX 上的 speculative decode 内核兼容性未知；加载失败直接停止，不需要额外回滚。

---

### 方向 D：RadixCache 正确性 Smoke（**已测试**，2026-07-27）

**结论：RadixCache 在 C500 上无 cache 命中，`--disable-radix-cache` 暂维持**

#### Smoke 实测结果

| 指标 | req-1（cold） | req-2（warm） |
|---|---|---|
| TTFT | 9.15s | 9.22s |
| TTFT ratio | — | **1.007**（期望 < 0.50） |
| 输出 | `运，\n调度器设计...` | `</think>\n</think>\nSGLang...` |

- **无 TTFT 减少**：RadixCache prefix lookup 完全未命中
- **输出不一致**：Step-3.5 是 thinking model，temperature=0 下 int8 MoE 仍有非确定性（`</think>` token 结构），与 RC 无直接关系（因为 TTFT 未降低，说明第二次请求走了全量 prefill，而非 RC 复用）

**根本原因（待查）**：

可能原因一：RadixCache tree 的 token hash 匹配逻辑在 MetaX 环境下未正确触发（需对比 `kvcache/radix_cache.py` 的 `insert()` / `match_prefix()` 路径）

可能原因二：Step-3.5 的 tokenizer 对长重复文本产生不一致的 token sequence（需打印实际 token ids 验证）

可能原因三：MetaX 上 KV cache 的 attention backend（`torch_native`）未正确与 RadixCache 的 cache slots 对接

**行动**：
1. `--disable-radix-cache` 维持现状，方向 D 标记为"待诊断"
2. 后续调查：在 `radix_cache_smoke.sh` 中加打印 cache hit stats（`/get_server_info` endpoint）

**脚本**：已提交至 `scripts/metax/radix_cache_smoke.sh`

---

### 方向 E：mini-sglang-metax 工程方向（中长期）

以下是针对开源项目的代码级优化，需要在真卡上验证后推进：

#### E1：C500 专用 MoE Triton Block Size 配置

**文件**：`python/minisgl/moe/fused.py` → `get_default_config()`

当前代码对所有硬件使用同一套 block size（M=64/16，N=64/32，K=32/64）。  
C500 有不同的 SM 数量、L2 cache 大小和 shared memory 限制，与 A100 不同。

**方向**：
1. 在 `try_get_optimal_moe_config()` 中加入基于 `platform` 的分支
2. 对 Step-3.5 的实际 MoE shape（E=384，K=hidden_size/TP，N=ffn_intermediate/TP）做 block size 扫描
3. 记录 C500 上每个 shape 的最优配置，提交为 `DEVICE_CONFIG` 表

```python
# 示例：platform-aware 配置
def get_default_config(M, E, N, K, topk, platform="cuda"):
    if platform == "metax":
        # TODO: 根据 C500 tuning 结果填充
        return {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 4}
    ...
```

**预期影响**：MoE GEMM 在 Prefill 阶段占主导，block size 调优可带来 10–30% 的 kernel 级别提升。

#### E2：MetaX-native TopK 和 MoE Align 替代实现（**M1 已完成**，2026-07-27）

**背景**

`moe/fused.py` 的 `fused_topk()` 和 `moe_align_block_size()` 分别依赖 `sgl_kernel.topk_softmax` 和 `sgl_kernel.moe_align_block_size`，这两者均为 NVIDIA 编译产物，在 MetaX 上不可用，因此必须绕过（`SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=0`）。

**M1 实现状态：已完成**

新建文件 [`python/minisgl/moe/metax.py`](../../python/minisgl/moe/metax.py)，实现 `MetaxMoe(BaseMoeBackend)`：

- 路由：纯 PyTorch `torch.softmax + torch.topk`（替代 `sgl_kernel.topk_softmax`）
- Expert 计算：Python for-loop 逐 expert gather/scatter（无 `moe_align_block_size` 依赖）
- 激活：SwiGLU / GeGLU
- 支持 `renormalize`、`apply_router_weight_on_input`、topk≥1

注册到 `moe/__init__.py`（`SUPPORTED_MOE_BACKENDS["metax"]`），`engine.py` 在 `platform=="metax"` 时自动选择。

测试：[`tests/moe/test_metax_moe.py`](../../tests/moe/test_metax_moe.py)，20 测试全部通过（CPU）。

**下一步（M2）**：用 MACA Triton grouped-matmul 替代 Python 循环，消除 O(E) Python dispatch 开销。

#### E3：Decode 阶段批量 Paged Attention 内核

**背景**

`torch_native.py` 的 decode 路径对每个 request 分别调用 `index_select` + `F.scaled_dot_product_attention`，是顺序 Python dispatch。在并发 32 时，这是 32 次独立的 SDPA 调用。

**方向**：实现 Triton-based 批量 Paged Attention（类似 vLLM/SGLang 的 `PagedAttention`），一次 kernel launch 处理整个 batch 的 decode attention。

```text
工作量估计：这是 Gate 3+ 的主要工程目标，优先级低于 MoE 和量化支持。
```

#### E4：Step3p5ForCausalLM 模型注册（Gate 2 候选）

当前 mini-sglang-metax 支持 Qwen2/3、Llama、Mistral（均为 dense）。Step-3.5 的 MoE 架构（`Step3p5ForCausalLM`）是第一个需要支持的 MoE 模型。

**前置条件**：
- E1/E2 中的 MoE Triton 替代实现可用
- W8A8 量化权重加载支持（否则只能运行 BF16 版本，如果存在的话）

---

## 3. 优先级排序与执行顺序

```
┌─────────────────────────────────────────────────────────┐
│  立即可做（本周，SGLang 0.5.13 生产路径）                │
│                                                         │
│  ~~A（JIT TopK）~~ ← 已关闭，无提升                    │
│  ~~D（RadixCache smoke）~~ ← 已测试，无 cache hit       │
│                                                         │
│  C（MTP smoke）← 当前（进行中，2026-07-27）             │
│       ↓                                                 │
│  B 高并发（c=192/256 饱和点）                           │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│  mini-sglang-metax 中长期（下一个 Gate）                │
│                                                         │
│  ~~E2（MetaX native TopK/Align）~~ ← M1 已完成         │
│       ↓                                                 │
│  M2：MACA Triton grouped-matmul 替换 Python loop        │
│       ↓                                                 │
│  E1（C500 MoE Triton 调优）                             │
│       ↓                                                 │
│  E4（Step3p5 模型注册）+ W8A8 加载                      │
│       ↓                                                 │
│  E3（批量 Decode Attention）                            │
└─────────────────────────────────────────────────────────┘
```

---

## 4. 已排除的方向

| 方向 | 排除原因 |
|---|---|
| CUDA Graph 开启 | MetaX MACA 堆栈上尚未验证，crash 风险高，收益不确定 |
| Piecewise CUDA Graph | 同上 |
| 更换 Chunked Prefill chunk size | Decode 上限未知前，Chunked Prefill 调优意义不大 |
| 增大 mem-fraction-static | 当前 0.7 已足够，提升上限有限且 OOM 风险增加 |

---

## 5. 参考文件

- 基线（Prefill）：`SGLang-C500-TP8-4096x1-基线结果-2026-07-17.md`
- 基线（Decode）：`SGLang-C500-TP8-Decode基线与Profiling进展-2026-07-17.md`
- 环境记录：`SGLang-C500-Step3.7-TP8当前环境.md`
- MoE kernel 实现：`mini-sglang-metax/python/minisgl/moe/fused.py`
- Attention 实现：`mini-sglang-metax/python/minisgl/attention/torch_native.py`
