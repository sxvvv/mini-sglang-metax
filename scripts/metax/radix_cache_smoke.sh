#!/usr/bin/env bash
# RadixCache correctness + TTFT-reduction smoke test on MetaX C500
# Usage: bash scripts/metax/radix_cache_smoke.sh [port]
#
# What it does:
#   1. Kills any running SGLang server
#   2. Starts a new server WITH radix cache enabled on PORT_RC
#   3. Sends the same long prompt twice (greedy, max_tokens=64)
#   4. Verifies output1 == output2  (correctness)
#   5. Verifies TTFT2 < TTFT1 × 0.5 (cache hit cuts TTFT in half)
#   6. Reports PASS / FAIL and kills the test server
#
# If PASS → the main start script should drop --disable-radix-cache
set -euo pipefail

PORT_RC=${1:-30000}
MODEL=/mxstorage/pde_ai/models/model_quant_opt/Stepfun/step3_5_W8A8/vllm_quant_model_main
LOG=/sw_home/m01684/logs/step35_c500_rc_smoke.log
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

# ── kill any existing server ──────────────────────────────────────────────────
echo "[smoke] killing existing SGLang server..."
pkill -f 'sglang.launch_server' 2>/dev/null || true
sleep 5

# ── start server WITH radix cache ────────────────────────────────────────────
echo "[smoke] starting server with RadixCache enabled → $LOG"
nohup /opt/conda/bin/python -m sglang.launch_server \
    --model-path "$MODEL" \
    --host 0.0.0.0 --port "$PORT_RC" \
    --tp-size 8 --mem-fraction-static 0.7 \
    --disable-cuda-graph --disable-piecewise-cuda-graph \
    --json-model-override-args '{"routed_scaling_factor":3.0}' \
    > "$LOG" 2>&1 &
SERVER_PID=$!
echo "[smoke] server PID=$SERVER_PID"

# ── wait for READY ────────────────────────────────────────────────────────────
echo "[smoke] waiting for server READY..."
for i in $(seq 1 60); do
    sleep 10
    if curl -sf "http://localhost:${PORT_RC}/health" >/dev/null 2>&1; then
        echo "[smoke] READY at $(date +%H:%M:%S) (waited ${i}×10s)"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[smoke] FATAL: server process died"
        tail -20 "$LOG"
        exit 1
    fi
    echo "[smoke] waiting... ${i}/60"
done

if ! curl -sf "http://localhost:${PORT_RC}/health" >/dev/null 2>&1; then
    echo "[smoke] FATAL: server not ready after timeout"
    exit 1
fi

# ── run Python smoke test ─────────────────────────────────────────────────────
RESULT_FILE="$RESULT_DIR/radix_cache_smoke_$(date +%Y%m%d_%H%M%S).txt"
echo "[smoke] running correctness + TTFT test..."

/opt/conda/bin/python - << PYEOF 2>&1 | tee "$RESULT_FILE"
import time, requests, json, sys

PORT = ${PORT_RC}
BASE  = f"http://localhost:{PORT}"

# Long prompt with shared prefix — ~4000 chars (~3000 tokens)
SHARED = "请详细介绍 SGLang 框架的调度器设计、内存管理机制、以及与 vLLM 的主要区别。" * 80

def chat(prompt, label):
    payload = {
        "model": "step3.5",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
        "temperature": 0,
    }
    t0 = time.time()
    r = requests.post(f"{BASE}/v1/chat/completions", json=payload, timeout=120)
    elapsed = time.time() - t0
    if r.status_code != 200:
        print(f"[{label}] HTTP {r.status_code}: {r.text[:200]}")
        sys.exit(1)
    d = r.json()
    content = d["choices"][0]["message"]["content"]
    usage   = d.get("usage", {})
    print(f"[{label}] elapsed={elapsed:.2f}s | tokens={usage} | output[:80]={content[:80]!r}")
    return elapsed, content

print("=" * 60)
print("RadixCache Smoke Test")
print("=" * 60)

t1, out1 = chat(SHARED, "req-1 (cold)")
time.sleep(1)
t2, out2 = chat(SHARED, "req-2 (warm)")

print()
print(f"TTFT-1 (cold):  {t1:.2f}s")
print(f"TTFT-2 (warm):  {t2:.2f}s")
print(f"TTFT ratio:     {t2/t1:.3f}  (want < 0.50)")
print()

# Correctness check
if out1 == out2:
    print("✓ Correctness: outputs are IDENTICAL")
    correct = True
else:
    print("✗ Correctness: outputs DIFFER!")
    print(f"  out1: {out1[:120]!r}")
    print(f"  out2: {out2[:120]!r}")
    correct = False

# TTFT reduction check — radix cache should cut TTFT by >50%
if t2 < t1 * 0.50:
    print(f"✓ Cache hit: TTFT-2 ({t2:.2f}s) < 50% of TTFT-1 ({t1:.2f}s)")
    cache_hit = True
else:
    print(f"✗ No cache hit: TTFT-2 ({t2:.2f}s) is NOT < 50% of TTFT-1 ({t1:.2f}s)")
    cache_hit = False

print()
if correct and cache_hit:
    print("RESULT: PASS — RadixCache is correct and effective on MetaX C500")
    sys.exit(0)
elif correct and not cache_hit:
    print("RESULT: PARTIAL — outputs match but no TTFT speedup (cache miss?)")
    sys.exit(2)
else:
    print("RESULT: FAIL — output mismatch detected")
    sys.exit(1)
PYEOF

SMOKE_EXIT=$?

# ── cleanup ───────────────────────────────────────────────────────────────────
echo ""
echo "[smoke] stopping RC test server (PID=$SERVER_PID)"
kill "$SERVER_PID" 2>/dev/null || true
sleep 3

echo "[smoke] results saved to: $RESULT_FILE"
echo "[smoke] server log:        $LOG"
echo ""
if [ $SMOKE_EXIT -eq 0 ]; then
    echo "========================================"
    echo " RadixCache smoke: PASS"
    echo " → Safe to remove --disable-radix-cache"
    echo " → Update start_step35.sh accordingly"
    echo "========================================"
elif [ $SMOKE_EXIT -eq 2 ]; then
    echo "========================================"
    echo " RadixCache smoke: PARTIAL"
    echo " → Outputs correct, but no TTFT speedup"
    echo " → Check cache config / prefix length"
    echo "========================================"
else
    echo "========================================"
    echo " RadixCache smoke: FAIL"
    echo " → Keep --disable-radix-cache"
    echo "========================================"
fi

exit $SMOKE_EXIT
