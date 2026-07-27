#!/usr/bin/env bash
# MTP (Multi-Token Prediction / EAGLE speculative decode) smoke test
# Usage: STEP=1 bash scripts/metax/mtp_smoke.sh
#
#   STEP=1  Load-only: verify model loads, check MTP layers recognized
#   STEP=2  Load + single inference: verify output is generated
#   STEP=3  Load + A/B throughput: compare MTP vs non-MTP at c=16
#
# Default STEP=1 (safest, doesn't require long GPU time)
set -euo pipefail

STEP=${STEP:-1}
MTP_MODEL=/mxstorage/pde_ai/models/model_quant_opt/Stepfun/step3_5_W8A8/vllm_quant_model_with_mtp
BASE_MODEL=/mxstorage/pde_ai/models/model_quant_opt/Stepfun/step3_5_W8A8/vllm_quant_model_main
PORT=30000
LOG=/sw_home/m01684/logs/step35_c500_mtp_smoke.log
RESULT_DIR=/sw_home/m01684/benchmark_result/step35_c500_tp8
mkdir -p "$RESULT_DIR" "$(dirname "$LOG")"

# ── environment ───────────────────────────────────────────────────────────────
export PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
CUDA_RT_DIR=$HOME/cuda-runtime-lib
mkdir -p "$CUDA_RT_DIR"
ln -sf /opt/maca/tools/cu-bridge/lib/libcuda.so "$CUDA_RT_DIR/libcudart.so"
export LIBRARY_PATH="$CUDA_RT_DIR:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_RT_DIR:${LD_LIBRARY_PATH:-}"
export MACA_PATH=/opt/maca
export CUDA_HOME=/opt/maca/tools/cu-bridge
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=0

echo "========================================"
echo " MTP Smoke Test  STEP=$STEP"
echo " $(date)"
echo "========================================"

# ── check model path ──────────────────────────────────────────────────────────
if [ ! -d "$MTP_MODEL" ]; then
    echo "[mtp] FATAL: MTP model not found at $MTP_MODEL"
    exit 1
fi
echo "[mtp] MTP model found: $MTP_MODEL"
echo "[mtp] num_nextn_predict_layers: $(grep -m1 'num_nextn' "$MTP_MODEL/config.json" 2>/dev/null || echo 'not in config')"

# ── kill existing server ──────────────────────────────────────────────────────
echo "[mtp] killing existing SGLang server..."
pkill -f 'sglang.launch_server' 2>/dev/null || true
sleep 5

# ── STEP 1: load-only (head -200 to exit early) ───────────────────────────────
echo "[mtp] STEP 1: starting MTP server (will exit after first 200 log lines or READY)..."
echo "" > "$LOG"

nohup /opt/conda/bin/python -m sglang.launch_server \
    --model-path "$MTP_MODEL" \
    --host 0.0.0.0 --port "$PORT" \
    --tp-size 8 --mem-fraction-static 0.7 \
    --disable-cuda-graph --disable-piecewise-cuda-graph \
    --disable-radix-cache \
    --json-model-override-args '{"routed_scaling_factor":3.0}' \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    >> "$LOG" 2>&1 &
MTP_PID=$!
echo "[mtp] server PID=$MTP_PID"

# Wait up to 17 min or fail-fast on crash
READY=0
for i in $(seq 1 100); do
    sleep 10
    if ! kill -0 "$MTP_PID" 2>/dev/null; then
        echo "[mtp] STEP1 FATAL: process died at ${i}×10s"
        echo "[mtp] last log lines:"
        tail -30 "$LOG"
        echo "RESULT STEP1: FAIL (process crash)"
        exit 1
    fi
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "[mtp] READY at $(date +%H:%M:%S) (waited ${i}×10s)"
        READY=1
        break
    fi
    # Show progress every 30s
    if (( i % 3 == 0 )); then
        LAST=$(tail -1 "$LOG")
        echo "[mtp] [${i}/72] $LAST"
    fi
    # Fail-fast on clear error patterns (skip first 30s of expected warnings)
    if (( i > 3 )); then
        if tail -5 "$LOG" | grep -qE '^(Traceback|RuntimeError|AssertionError|TypeError|CUDA error)'; then
            echo "[mtp] STEP1 FATAL: error detected"
            tail -30 "$LOG"
            echo "RESULT STEP1: FAIL (runtime error)"
            kill "$MTP_PID" 2>/dev/null || true
            exit 1
        fi
    fi
done

# Check MTP-specific log markers
echo ""
echo "[mtp] --- STEP1 load diagnostic ---"
grep -E 'speculative|EAGLE|nextn|MTP|draft|target|num_spec' "$LOG" | head -20 || echo "(no speculative/MTP lines found)"
echo ""

if [ $READY -eq 0 ]; then
    echo "RESULT STEP1: FAIL (timeout)"
    kill "$MTP_PID" 2>/dev/null || true
    exit 1
fi
echo "RESULT STEP1: PASS — MTP server loaded and READY"

# ── STEP 2: single inference ──────────────────────────────────────────────────
if [ "$STEP" -ge 2 ]; then
    echo ""
    echo "[mtp] STEP 2: single inference check..."
    RESULT2=$(/opt/conda/bin/python - << 'PYEOF' 2>&1
import requests, time, sys

r = requests.post("http://localhost:30000/v1/chat/completions", json={
    "model": "step3.5",
    "messages": [{"role": "user", "content": "用一句话介绍SGLang框架。"}],
    "max_tokens": 32, "temperature": 0
}, timeout=60)
if r.status_code != 200:
    print(f"HTTP {r.status_code}: {r.text[:200]}")
    sys.exit(1)
d = r.json()
out = d["choices"][0]["message"]["content"]
print(f"output: {out[:100]!r}")
print("STEP2: PASS")
PYEOF
    )
    echo "$RESULT2"
    if echo "$RESULT2" | grep -q "STEP2: PASS"; then
        echo "RESULT STEP2: PASS — MTP inference OK"
    else
        echo "RESULT STEP2: FAIL"
        kill "$MTP_PID" 2>/dev/null || true
        exit 1
    fi
fi

# ── STEP 3: throughput A/B (MTP vs non-MTP at c=16) ─────────────────────────
if [ "$STEP" -ge 3 ]; then
    echo ""
    echo "[mtp] STEP 3: throughput A/B at c=16 ..."
    RESULT_FILE="$RESULT_DIR/mtp_ab_$(date +%Y%m%d_%H%M%S).txt"

    echo "=== MTP (EAGLE, num_spec=3) ===" | tee "$RESULT_FILE"
    /opt/conda/bin/python -m sglang.bench_serving \
        --backend sglang --host 127.0.0.1 --port "$PORT" \
        --dataset-name random --dataset-path /tmp/fake_sharegpt.json \
        --random-input-len 4096 --random-output-len 256 \
        --num-prompts 32 --max-concurrency 16 --seed 1 2>&1 | tee -a "$RESULT_FILE"

    # Restart with non-MTP base model for comparison
    echo "" | tee -a "$RESULT_FILE"
    echo "[mtp] restarting with non-MTP model for baseline..." | tee -a "$RESULT_FILE"
    kill "$MTP_PID" 2>/dev/null || true; sleep 5
    nohup /opt/conda/bin/python -m sglang.launch_server \
        --model-path "$BASE_MODEL" --host 0.0.0.0 --port "$PORT" \
        --tp-size 8 --mem-fraction-static 0.7 \
        --disable-cuda-graph --disable-piecewise-cuda-graph \
        --disable-radix-cache \
        --json-model-override-args '{"routed_scaling_factor":3.0}' \
        >> /sw_home/m01684/logs/step35_c500_jit0.log 2>&1 &
    BASE_PID=$!
    for i in $(seq 1 72); do
        sleep 10
        if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then break; fi
    done

    echo "=== non-MTP baseline ===" | tee -a "$RESULT_FILE"
    /opt/conda/bin/python -m sglang.bench_serving \
        --backend sglang --host 127.0.0.1 --port "$PORT" \
        --dataset-name random --dataset-path /tmp/fake_sharegpt.json \
        --random-input-len 4096 --random-output-len 256 \
        --num-prompts 32 --max-concurrency 16 --seed 1 2>&1 | tee -a "$RESULT_FILE"

    echo ""
    echo "Results saved to: $RESULT_FILE"
    MTP_PID=$BASE_PID
fi

# ── cleanup: restore main server ─────────────────────────────────────────────
echo ""
if [ "$STEP" -lt 3 ]; then
    echo "[mtp] killing MTP server, restarting main JIT=0 server..."
    kill "$MTP_PID" 2>/dev/null || true
    sleep 5
    nohup bash /tmp/start_jit0.sh >> /sw_home/m01684/logs/step35_c500_jit0.log 2>&1 &
fi
echo "[mtp] done. log: $LOG"
