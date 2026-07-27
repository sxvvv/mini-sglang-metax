#!/usr/bin/env bash
# bench_step35_decode_sweep.sh
#
# Step-3.5 W8A8 C500 TP8 — Decode 吞吐上限扫描
#
# 用途：补全并发 32 以上的 Decode 基线，确认吞吐拐点。
# 前置：SGLang 服务已启动（见 start_step35.sh），/health 返回 200。
#
# 用法：
#   bash scripts/metax/bench_step35_decode_sweep.sh
#   # 自定义端口和输出目录：
#   PORT=30001 OUT_DIR=/sw_home/m01684/bench2 bash scripts/metax/bench_step35_decode_sweep.sh

set -euo pipefail

PORT=${PORT:-30000}
OUT_DIR=${OUT_DIR:-/sw_home/m01684/benchmark_result/step35_c500_tp8/decode_sweep_$(date +%Y%m%d)}
INPUT_LEN=${INPUT_LEN:-4096}
OUTPUT_LEN=${OUTPUT_LEN:-256}

# 并发序列：从已有基线 32 开始，扩展到 128
CONCURRENCIES=(32 48 64 96 128)

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/sweep.log"

echo "[sweep] $(date) 开始 Decode 吞吐扫描" | tee -a "$LOG"
echo "[sweep] 目标服务: http://localhost:$PORT" | tee -a "$LOG"
echo "[sweep] 输入/输出: ${INPUT_LEN}/${OUTPUT_LEN}" | tee -a "$LOG"
echo "[sweep] 并发序列: ${CONCURRENCIES[*]}" | tee -a "$LOG"
echo "---" | tee -a "$LOG"

# 健康检查
if ! curl -sf "http://localhost:$PORT/health" > /dev/null; then
    echo "[sweep] ERROR: 服务未响应，退出" | tee -a "$LOG"
    exit 1
fi
echo "[sweep] 服务健康检查 OK" | tee -a "$LOG"

for C in "${CONCURRENCIES[@]}"; do
    N=$((C * 2))  # 请求数 = 2× 并发，与已有基线口径一致
    JSON="$OUT_DIR/decode_c${C}.json"
    CLOG="$OUT_DIR/decode_c${C}.log"

    echo "[sweep] 开始并发=$C 请求数=$N ..." | tee -a "$LOG"

    python -m sglang.bench_serving \
        --backend sglang \
        --host "127.0.0.1" \
        --port "$PORT" \
        --dataset-name "random" \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --num-prompts "$N" \
        --max-concurrency "$C" \
        --output-file "$JSON" \
        --seed 1 \
        2>&1 | tee "$CLOG"

    # 提取关键指标并追加到 sweep log
    if [[ -f "$JSON" ]]; then
        python3 - "$JSON" "$C" | tee -a "$LOG" <<'PYEOF'
import json, sys
data = json.load(open(sys.argv[1]))
c = sys.argv[2]
ttft_mean = data.get("mean_ttft_ms", "N/A")
tpot_mean = data.get("mean_tpot_ms", "N/A")
out_thr   = data.get("output_throughput", "N/A")
print(f"[result] c={c:>3}  output_throughput={out_thr:.2f} tok/s  "
      f"mean_TTFT={ttft_mean:.1f} ms  mean_TPOT={tpot_mean:.1f} ms")
PYEOF
    fi

    echo "[sweep] 并发=$C 完成，等待 15s..." | tee -a "$LOG"
    sleep 15
done

echo "---" | tee -a "$LOG"
echo "[sweep] $(date) 全部完成，结果在: $OUT_DIR" | tee -a "$LOG"

# 打印汇总表
echo ""
echo "=== 汇总表 ==="
printf "%-8s %-22s %-16s %-16s\n" "并发" "输出吞吐(tok/s)" "TTFT均值(ms)" "TPOT均值(ms)"
for C in "${CONCURRENCIES[@]}"; do
    JSON="$OUT_DIR/decode_c${C}.json"
    if [[ -f "$JSON" ]]; then
        python3 - "$JSON" "$C" <<'PYEOF'
import json, sys
data = json.load(open(sys.argv[1]))
c = sys.argv[2]
print(f"{c:<8} {data.get('output_throughput', 'N/A'):<22.2f} "
      f"{data.get('mean_ttft_ms', 'N/A'):<16.1f} "
      f"{data.get('mean_tpot_ms', 'N/A'):<16.1f}")
PYEOF
    fi
done
