#!/usr/bin/env bash
# mtp_smoke.sh
#
# Step-3.5 W8A8 MTP checkpoint — 分步兼容性 smoke 测试
#
# 目标：验证 SGLang 0.5.13 能否在 MetaX C500 上加载并运行 MTP 版本模型。
# 采用渐进式：Step 1 只验证加载；Step 2 验证单次推理；Step 3 做吞吐 A/B 对比。
#
# 用法（在 C500 机器上运行）：
#   bash scripts/metax/mtp_smoke.sh          # 仅 Step 1（默认）
#   STEP=2 bash scripts/metax/mtp_smoke.sh   # 到 Step 2
#   STEP=3 bash scripts/metax/mtp_smoke.sh   # 完整 A/B

set -euo pipefail

STEP=${STEP:-1}
MODEL_MAIN=${MODEL_MAIN:-/mxstorage/pde_ai/models/model_quant_opt/Stepfun/step3_5_W8A8/vllm_quant_model_main}
MODEL_MTP=${MODEL_MTP:-/mxstorage/pde_ai/models/model_quant_opt/Stepfun/step3_5_W8A8/vllm_quant_model_with_mtp}
PORT_MAIN=${PORT_MAIN:-30000}
PORT_MTP=${PORT_MTP:-30001}
LOG_MTP=/tmp/sglang-step35-mtp.log
OUT_DIR=${OUT_DIR:-/sw_home/m01684/benchmark_result/step35_c500_tp8/mtp_smoke_$(date +%Y%m%d)}
mkdir -p "$OUT_DIR"

echo "[mtp_smoke] $(date)"
echo "[mtp_smoke] MTP 模型路径: $MODEL_MTP"
echo "[mtp_smoke] Step: $STEP"

# ────────────────────────────────────────────────────────────────────────
# Step 1：尝试加载 MTP 模型，观察启动日志中的层数识别
# ────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Step 1: MTP 模型加载验证 ==="

# 检查模型路径存在
if [[ ! -d "$MODEL_MTP" ]]; then
    echo "[mtp_smoke] ERROR: MTP 模型路径不存在: $MODEL_MTP" >&2
    exit 1
fi
echo "[mtp_smoke] 模型路径存在 OK"

# 打印 config.json 中的 MTP 相关字段
python3 - "$MODEL_MTP" <<'PYEOF'
import json, os, sys
cfg_path = os.path.join(sys.argv[1], "config.json")
if not os.path.exists(cfg_path):
    print("[mtp_smoke] WARNING: config.json 不存在")
    exit(0)
cfg = json.load(open(cfg_path))
mtp_keys = [k for k in cfg if "mtp" in k.lower() or "nextn" in k.lower() or "speculative" in k.lower()]
print("[mtp_smoke] MTP 相关配置字段:")
for k in mtp_keys:
    print(f"  {k}: {cfg[k]}")
if not mtp_keys:
    print("[mtp_smoke] WARNING: 未找到 MTP/nextn/speculative 字段，可能不是 MTP checkpoint")
PYEOF

# 以超时方式启动服务，只等待初始化日志
echo "[mtp_smoke] 启动 MTP 服务（端口 $PORT_MTP），观察初始化..."
nohup env SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=0 \
    /opt/conda/bin/python -m sglang.launch_server \
    --model-path "$MODEL_MTP" \
    --host 0.0.0.0 --port "$PORT_MTP" \
    --tp-size 8 --mem-fraction-static 0.7 \
    --disable-cuda-graph --disable-piecewise-cuda-graph \
    --disable-radix-cache \
    --json-model-override-args '{"routed_scaling_factor":3.0}' \
    --speculative-algorithm EAGLE \
    --num-speculative-steps 3 \
    >"$LOG_MTP" 2>&1 &
MTP_PID=$!
echo "[mtp_smoke] MTP 服务 PID: $MTP_PID"

echo "[mtp_smoke] 等待启动（最多 300s）或首个错误..."
for i in $(seq 1 60); do
    sleep 5
    if ! kill -0 "$MTP_PID" 2>/dev/null; then
        echo "[mtp_smoke] Step 1 FAIL: 进程已退出"
        echo "=== 最后 30 行日志 ==="
        tail -30 "$LOG_MTP"
        exit 1
    fi
    if curl -sf "http://localhost:$PORT_MTP/health" > /dev/null 2>&1; then
        echo "[mtp_smoke] Step 1 PASS: MTP 服务就绪 ($((i*5))s)"
        break
    fi
    # 检查是否有明确的错误日志
    if grep -q "Error\|error\|Exception\|FAILED" "$LOG_MTP" 2>/dev/null; then
        echo "[mtp_smoke] Step 1 WARN: 日志中出现错误，仍在尝试..."
    fi
    printf "."
    if [[ $i -eq 60 ]]; then
        echo ""
        echo "[mtp_smoke] Step 1 TIMEOUT: 300s 后服务未就绪"
        echo "=== 最后 50 行日志 ==="
        tail -50 "$LOG_MTP"
        kill "$MTP_PID" 2>/dev/null || true
        exit 1
    fi
done

[[ "$STEP" -lt 2 ]] && { echo "[mtp_smoke] Step 1 完成，STEP<2 退出"; kill "$MTP_PID" 2>/dev/null || true; exit 0; }

# ────────────────────────────────────────────────────────────────────────
# Step 2：单次推理正确性验证
# ────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Step 2: 单次推理正确性验证 ==="

# 同一 prompt 发给 Main 服务和 MTP 服务，对比输出
PROMPT="请用一句话介绍 SGLang 框架。"

get_output() {
    local port=$1
    python3 - "$port" "$PROMPT" <<'PYEOF'
import requests, sys
resp = requests.post(f"http://localhost:{sys.argv[1]}/v1/chat/completions",
    json={"model": "step3.5",
          "messages": [{"role": "user", "content": sys.argv[2]}],
          "max_tokens": 64,
          "temperature": 0},
    timeout=120)
data = resp.json()
print(data["choices"][0]["message"]["content"])
PYEOF
}

echo "[mtp_smoke] 调用 Main 服务（端口 $PORT_MAIN）..."
OUTPUT_MAIN=$(get_output "$PORT_MAIN") && echo "[main] $OUTPUT_MAIN" || { echo "[mtp_smoke] Step 2 FAIL: Main 服务调用失败"; exit 1; }

echo "[mtp_smoke] 调用 MTP 服务（端口 $PORT_MTP）..."
OUTPUT_MTP=$(get_output "$PORT_MTP") && echo "[mtp]  $OUTPUT_MTP" || { echo "[mtp_smoke] Step 2 FAIL: MTP 服务调用失败"; exit 1; }

if [[ "$OUTPUT_MAIN" == "$OUTPUT_MTP" ]]; then
    echo "[mtp_smoke] Step 2 PASS: 输出一致"
else
    echo "[mtp_smoke] Step 2 WARN: 输出不一致（greedy decoding 下可能是 speculative decode 引入的差异，需人工核查）"
    echo "  Main: $OUTPUT_MAIN"
    echo "  MTP:  $OUTPUT_MTP"
fi

[[ "$STEP" -lt 3 ]] && { echo "[mtp_smoke] Step 2 完成，STEP<3 退出"; kill "$MTP_PID" 2>/dev/null || true; exit 0; }

# ────────────────────────────────────────────────────────────────────────
# Step 3：同口径 A/B 吞吐对比（并发 16，4096×256）
# ────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Step 3: A/B 吞吐对比（c=16，4096×256）==="

run_bench() {
    local tag=$1
    local port=$2
    local out="$OUT_DIR/ab_${tag}.json"
    python -m sglang.bench_serving \
        --backend sglang --host 127.0.0.1 --port "$port" \
        --dataset-name random \
        --random-input-len 4096 --random-output-len 256 \
        --num-prompts 32 --max-concurrency 16 \
        --output-file "$out" --seed 1 \
        2>&1 | tail -5
    python3 - "$out" "$tag" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"[{sys.argv[2]}] output_throughput={d.get('output_throughput','N/A'):.2f} tok/s  "
      f"mean_TPOT={d.get('mean_tpot_ms','N/A'):.1f} ms  "
      f"mean_TTFT={d.get('mean_ttft_ms','N/A'):.1f} ms")
PYEOF
}

echo "[mtp_smoke] 运行 Main 基线..."
run_bench "main" "$PORT_MAIN"
echo "[mtp_smoke] 运行 MTP..."
run_bench "mtp" "$PORT_MTP"

echo "[mtp_smoke] Step 3 完成，结果在: $OUT_DIR"
kill "$MTP_PID" 2>/dev/null || true
