# Step-3.5 W8A8 C500 TP8 优化方向

> 日期：2026-07-27  
> 模型：Step-3.5 W8A8（`vllm_quant_model_main`，30B+ MoE，45 层，TP8）  
> 设备：MetaX C500，8 卡  
> 框架：SGLang 0.5.13（MetaX 定制版）  
> 当前状态：Decode sweep 已完成（C500 TP8，2026-07-27 08:45–08:56 CST）

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
| `SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=0` | JIT 编译在 MetaX 上失败（libcudart 路径缺失） | 应用 C550 的 libcudart symlink 修复 |
| `--disable-cuda-graph` | MetaX 尚未实测 Graph 可用性 | 探索性验证（低优先） |
| `--disable-radix-cache` | 正确性未在 MetaX 上验证 | 先做 smoke，再开启 |
| `--json-model-override-args '{"routed_scaling_factor":3.0}'` | checkpoint 字段命名不一致 | 已 workaround，结构性解决需上游修复 |

---

## 2. 优化方向

### 方向 A：JIT Fused TopK 修复（高优先，低风险）

**背景**

`SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=0` 强制退回慢路径。  
C550 上的修复已验证：SGLang JIT 按名称查找 `libcudart.so`，而 MetaX 的 CUDA Runtime shim 位于 `/opt/maca/tools/cu-bridge/lib/libcuda.so`，名称不匹配。

**修复步骤**

```bash
# 1. 确认 shim 导出 Runtime 符号
nm -D /opt/maca/tools/cu-bridge/lib/libcuda.so | grep cudaMalloc

# 2. 创建名称映射（仅影响 JIT 链接，不替换系统库）
mkdir -p ~/cuda-runtime-lib
ln -s /opt/maca/tools/cu-bridge/lib/libcuda.so ~/cuda-runtime-lib/libcudart.so

# 3. 注入到编译和运行链路
export LIBRARY_PATH=$HOME/cuda-runtime-lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/cuda-runtime-lib:$LD_LIBRARY_PATH

# 4. 最小验证
python -c "import ctypes; ctypes.CDLL('libcudart.so'); print('cudart binding OK')"

# 5. 去掉 workaround 重新启动服务，观察启动日志是否有 JIT 编译信息
# SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK 不再设为 0
```

**预期影响**：MoE Router 每层都有 TopK 调用；JIT fused 路径比回退路径快（具体提升取决于 MACA Triton JIT 生成的 kernel 质量，但方向确定为正向）。

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

### 方向 C：MTP（Multi-Token Prediction）Smoke（中优先，中风险）

**背景**

checkpoint 路径：`/mxstorage/pde_ai/models/model_quant_opt/Stepfun/step3_5_W8A8/vllm_quant_model_with_mtp`  
模型配置：`num_nextn_predict_layers=3`  
SGLang 0.5.13 对 Step3p5 MTP 的 MetaX 兼容性**尚未验证**。

**实验步骤（渐进式）**

```bash
# Step 1：只验证模型能否加载（不做推理）
# 在启动日志中确认 MTP 层数被正确识别
MODEL=/mxstorage/pde_ai/models/model_quant_opt/Stepfun/step3_5_W8A8/vllm_quant_model_with_mtp
env SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=0 \
  python -m sglang.launch_server \
  --model-path "$MODEL" --host 0.0.0.0 --port 30001 \
  --tp-size 8 --mem-fraction-static 0.7 \
  --disable-cuda-graph --disable-piecewise-cuda-graph \
  --disable-radix-cache \
  --json-model-override-args '{"routed_scaling_factor":3.0}' \
  --speculative-algorithm EAGLE \
  --num-speculative-steps 3 \
  2>&1 | head -100

# Step 2：如果加载成功，发单个请求验证输出正确性
# Step 3：如果正确性通过，做同口径 A/B throughput 对比
```

**预期影响**：MTP/EAGLE 在 decode-heavy 负载下可提升 2–3× 输出吞吐（取决于 draft acceptance rate）。  
**风险**：SGLang 0.5.13 的 Step3p5 MTP 路径在 MetaX 上的 speculative decode 内核兼容性未知；加载失败直接停止，不需要额外回滚。

---

### 方向 D：RadixCache 正确性 Smoke（中优先，低代码风险）

**背景**

`--disable-radix-cache` 是保守禁用，非功能性障碍。mini-sglang-metax 的 RadixCache 实现已存在（`kvcache/radix_cache.py`）且在通用测试中通过。

**实验步骤**

```bash
# 1. 去掉 --disable-radix-cache，同时缩小规模确保不影响正在跑的任务
# 2. 用相同 prompt 发两次请求，验证第二次 TTFT 明显低于第一次
# 3. 验证两次输出内容一致（greedy decoding）

# smoke 脚本示例：
python - <<'EOF'
import requests, json

PROMPT = "请介绍 SGLang 框架的调度器设计。" * 200  # ~4000 chars

def chat(prompt):
    resp = requests.post("http://localhost:30000/v1/chat/completions",
        json={"model": "step3.5", "messages":[{"role":"user","content":prompt}],
              "max_tokens": 32, "temperature": 0})
    return resp.json()

r1 = chat(PROMPT)
r2 = chat(PROMPT)
print("TTFT 1:", r1.get("usage"))
print("TTFT 2:", r2.get("usage"))
assert r1["choices"][0]["message"]["content"] == r2["choices"][0]["message"]["content"], "输出不一致！"
print("RadixCache smoke: PASSED")
EOF
```

**预期影响**：在真实对话流量（高前缀重用率）下可将 TTFT 降低 30–70%，random-ids 基线不受影响（无前缀复用）。  
**风险**：若 MetaX 上 KV cache 读写有未检测的精度问题，smoke 会发现输出不一致。

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

#### E2：MetaX-native TopK 和 MoE Align 替代实现

**背景**

`moe/fused.py` 的 `fused_topk()` 和 `moe_align_block_size()` 分别依赖 `sgl_kernel.topk_softmax` 和 `sgl_kernel.moe_align_block_size`，这两者均为 NVIDIA 编译产物，在 MetaX 上不可用，因此必须绕过（`SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=0`）。

**方向**：在 `minisgl.kernel` 或 `minisgl.moe` 下提供 platform=metax 时的纯 Triton / PyTorch 替代：

```python
# python/minisgl/kernel/moe_metax.py（新增）
def topk_softmax_metax(topk_weights, topk_ids, gating_output, renormalize):
    """纯 PyTorch 实现，无 sgl_kernel 依赖"""
    scores = torch.softmax(gating_output.float(), dim=-1)
    topk_weights_, topk_ids_ = torch.topk(scores, k=topk_weights.shape[1], dim=-1)
    topk_weights.copy_(topk_weights_)
    topk_ids.copy_(topk_ids_.to(torch.int32))

def moe_align_block_size_metax(topk_ids, block_size, num_experts):
    """纯 PyTorch 替代，避免 sgl_kernel 依赖"""
    ...
```

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
│  A → Decode sweep（B）                                  │
│       ↓                                                 │
│  C（MTP smoke，需要A先稳定）                            │
│       ↓                                                 │
│  D（RadixCache smoke，独立分支）                        │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│  mini-sglang-metax 中长期（下一个 Gate）                │
│                                                         │
│  E2（MetaX native TopK/Align）                         │
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
- C550 JIT 修复参考：`MC3-12293-C550-Profiling-Jira更新-2026-07-15.md`
- MoE kernel 实现：`mini-sglang-metax/python/minisgl/moe/fused.py`
- Attention 实现：`mini-sglang-metax/python/minisgl/attention/torch_native.py`
