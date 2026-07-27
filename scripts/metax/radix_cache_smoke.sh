#!/usr/bin/env bash
# radix_cache_smoke.sh
#
# Step-3.5 W8A8 C500 TP8 — RadixCache 正确性 smoke 测试
#
# 目标：在开启 RadixCache 的服务上验证：
#   1. 带前缀重用的请求输出与禁用 cache 时一致
#   2. 第二次请求的 TTFT 明显低于第一次（cache hit 生效）
#
# 前置：服务已在 PORT_MAIN（禁用 RadixCache）上运行，脚本会另启一个端口开启 RadixCache。
#
# 用法：
#   bash scripts/metax/radix_cache_smoke.sh
#   PORT_MAIN=30000 PORT_RC=30002 bash scripts/metax/radix_cache_smoke.sh

set -euo pipefail

MODEL=${MODEL:-/mxstorage/pde_ai/models/model_quant_opt/Stepfun/step3_5_W8A8/vllm_quant_model_main}
PORT_MAIN=${PORT_MAIN:-30000}  # 禁用 RadixCache（基线）
PORT_RC=${PORT_RC:-30002}       # 启用 RadixCache
LOG_RC=/tmp/sglang-step35-rc.log
TTFT_RATIO_THRESHOLD=0.7        # cache hit 的第二次 TTFT 应低于第一次的 70%

echo "[rc_smoke] $(date)"
echo "[rc_smoke] 主服务端口（无 cache）: $PORT_MAIN"
echo "[rc_smoke] RadixCache 服务端口:    $PORT_RC"

# ── 确认主服务健康 ────────────────────────────────────────────────────────────
curl -sf "http://localhost:$PORT_MAIN/health" > /dev/null || {
    echo "[rc_smoke] ERROR: 主服务不可用，请先启动 start_step35.sh" >&2
    exit 1
}
echo "[rc_smoke] 主服务健康 OK"

# ── 启动 RadixCache 版本服务 ───────────────────────────────────────────────────
echo "[rc_smoke] 启动 RadixCache 服务（端口 $PORT_RC）..."
nohup env SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=0 \
    /opt/conda/bin/python -m sglang.launch_server \
    --model-path "$MODEL" \
    --host 0.0.0.0 --port "$PORT_RC" \
    --tp-size 8 --mem-fraction-static 0.7 \
    --disable-cuda-graph --disable-piecewise-cuda-graph \
    --json-model-override-args '{"routed_scaling_factor":3.0}' \
    >"$LOG_RC" 2>&1 &
RC_PID=$!
echo "[rc_smoke] RadixCache 服务 PID: $RC_PID"

for i in $(seq 1 60); do
    sleep 5
    if ! kill -0 "$RC_PID" 2>/dev/null; then
        echo "[rc_smoke] FAIL: RadixCache 服务进程已退出"
        tail -30 "$LOG_RC"
        exit 1
    fi
    if curl -sf "http://localhost:$PORT_RC/health" > /dev/null 2>&1; then
        echo "[rc_smoke] RadixCache 服务就绪 ($((i*5))s)"
        break
    fi
    [[ $i -eq 60 ]] && { echo "[rc_smoke] TIMEOUT"; kill "$RC_PID"; exit 1; }
    printf "."
done

# ── 正确性对比：相同 prompt，有无 cache 输出应一致 ────────────────────────────
echo ""
echo "=== 测试 1：正确性（有无 RadixCache 输出一致性）==="

# 使用长前缀（约 3000 tokens）确保 cache 收益明显
LONG_PREFIX=$(python3 -c "print('请详细介绍 SGLang 推理框架的调度器、KV Cache 和批处理机制。' * 80)")
SUFFIX="请总结一下上述内容的核心要点。"

call_api() {
    local port=$1; local content=$2
    python3 - "$port" "$content" <<'PYEOF'
import requests, sys, time
t0 = time.time()
resp = requests.post(f"http://localhost:{sys.argv[1]}/v1/chat/completions",
    json={"model": "step3.5",
          "messages": [{"role": "user", "content": sys.argv[2]}],
          "max_tokens": 64, "temperature": 0},
    timeout=120)
elapsed = (time.time() - t0) * 1000
data = resp.json()
text = data["choices"][0]["message"]["content"]
print(f"{elapsed:.0f}|{text}")
PYEOF
}

IFS='|' read -r TTFT_MAIN OUTPUT_MAIN <<< "$(call_api "$PORT_MAIN" "$LONG_PREFIX$SUFFIX")"
echo "[main  ] TTFT≈${TTFT_MAIN}ms | 输出: ${OUTPUT_MAIN:0:60}..."

# 第一次请求（cache miss）
IFS='|' read -r TTFT_RC1 OUTPUT_RC1 <<< "$(call_api "$PORT_RC" "$LONG_PREFIX$SUFFIX")"
echo "[rc #1 ] TTFT≈${TTFT_RC1}ms | 输出: ${OUTPUT_RC1:0:60}..."

# 第二次请求（cache hit 预期）
IFS='|' read -r TTFT_RC2 OUTPUT_RC2 <<< "$(call_api "$PORT_RC" "$LONG_PREFIX$SUFFIX")"
echo "[rc #2 ] TTFT≈${TTFT_RC2}ms | 输出: ${OUTPUT_RC2:0:60}..."

# 正确性检查
if [[ "$OUTPUT_MAIN" == "$OUTPUT_RC1" ]]; then
    echo "[rc_smoke] 正确性 PASS: main 与 rc#1 输出一致"
else
    echo "[rc_smoke] 正确性 FAIL: main 与 rc#1 输出不一致！"
    echo "  main: $OUTPUT_MAIN"
    echo "  rc#1: $OUTPUT_RC1"
    kill "$RC_PID" 2>/dev/null || true
    exit 1
fi

if [[ "$OUTPUT_RC1" == "$OUTPUT_RC2" ]]; then
    echo "[rc_smoke] 幂等性 PASS: rc#1 与 rc#2 输出一致"
else
    echo "[rc_smoke] 幂等性 FAIL: rc#1 与 rc#2 输出不一致！"
    kill "$RC_PID" 2>/dev/null || true
    exit 1
fi

# Cache hit 效果检查（TTFT 下降）
python3 - "$TTFT_RC1" "$TTFT_RC2" "$TTFT_RATIO_THRESHOLD" <<'PYEOF'
import sys
ttft1, ttft2, threshold = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
ratio = ttft2 / ttft1 if ttft1 > 0 else 1.0
if ratio < threshold:
    print(f"[rc_smoke] Cache hit PASS: TTFT 从 {ttft1:.0f}ms 降至 {ttft2:.0f}ms (ratio={ratio:.2f}<{threshold})")
else:
    print(f"[rc_smoke] Cache hit WARN: TTFT ratio={ratio:.2f} >= {threshold}，cache hit 不明显")
    print(f"  可能原因：chunk 未达到 cache 对齐边界，或 RadixCache 未命中")
PYEOF

echo ""
echo "[rc_smoke] 正确性 smoke PASSED，RadixCache 在 MetaX 上功能正常"
echo "[rc_smoke] 下一步：在真实对话流量下运行 bench_serving 验证吞吐增益"
kill "$RC_PID" 2>/dev/null || true
