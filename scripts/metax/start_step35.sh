#!/usr/bin/env bash
# start_step35.sh
#
# Step-3.5 W8A8 C500 TP8 — SGLang 0.5.13 标准启动脚本
#
# 包含所有已知 workaround，并提供 JIT TopK 修复的可选路径。
# 用法：
#   bash scripts/metax/start_step35.sh            # 使用 JIT TopK 回退（稳定）
#   JIT_TOPK=1 bash scripts/metax/start_step35.sh # 尝试启用 JIT TopK（需要 libcudart 修复）
#   PORT=30001 bash scripts/metax/start_step35.sh # 自定义端口

set -euo pipefail

# ── 配置 ─────────────────────────────────────────────────────────────────────
MODEL=${MODEL:-/mxstorage/pde_ai/models/model_quant_opt/Stepfun/step3_5_W8A8/vllm_quant_model_main}
PORT=${PORT:-30000}
LOG=${LOG:-/tmp/sglang-step35-c500-tp8.log}
MEM_FRAC=${MEM_FRAC:-0.7}
TP=${TP:-8}

# JIT TopK 开关（默认关闭，稳定优先）
# 设 JIT_TOPK=1 前请先执行 libcudart symlink 修复（见方向 A 说明）
JIT_TOPK=${JIT_TOPK:-0}

# ── libcudart symlink 修复（JIT TopK=1 时自动应用）────────────────────────────
if [[ "$JIT_TOPK" == "1" ]]; then
    CUDA_RT_DIR="$HOME/cuda-runtime-lib"
    SHIM="/opt/maca/tools/cu-bridge/lib/libcuda.so"

    if [[ ! -f "$SHIM" ]]; then
        echo "[start] ERROR: MetaX CUDA shim 不在 $SHIM，无法应用 libcudart 修复" >&2
        exit 1
    fi

    if [[ ! -L "$CUDA_RT_DIR/libcudart.so" ]]; then
        echo "[start] 创建 libcudart.so 符号链接..."
        mkdir -p "$CUDA_RT_DIR"
        ln -sf "$SHIM" "$CUDA_RT_DIR/libcudart.so"
        echo "[start] libcudart.so -> $SHIM"
    fi

    export LIBRARY_PATH="$CUDA_RT_DIR:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="$CUDA_RT_DIR:${LD_LIBRARY_PATH:-}"

    # 验证
    python3 -c "import ctypes; ctypes.CDLL('libcudart.so'); print('[start] cudart binding OK')" || {
        echo "[start] ERROR: libcudart 绑定失败，回退到 JIT_TOPK=0" >&2
        JIT_TOPK=0
    }
fi

# ── 启动 ─────────────────────────────────────────────────────────────────────
echo "[start] $(date)"
echo "[start] 模型: $MODEL"
echo "[start] 端口: $PORT"
echo "[start] TP:   $TP"
echo "[start] 日志: $LOG"
echo "[start] JIT TopK: $JIT_TOPK"
echo "[start] mem-fraction-static: $MEM_FRAC"

LAUNCH_CMD=(
    /opt/conda/bin/python -m sglang.launch_server
    --model-path "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --tp-size "$TP"
    --mem-fraction-static "$MEM_FRAC"
    --disable-cuda-graph
    --disable-piecewise-cuda-graph
    --disable-radix-cache
    --json-model-override-args '{"routed_scaling_factor":3.0}'
)

nohup env \
    SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK="$JIT_TOPK" \
    "${LAUNCH_CMD[@]}" \
    >"$LOG" 2>&1 &

SERVER_PID=$!
echo "[start] 后台 PID: $SERVER_PID"
echo "$SERVER_PID" > /tmp/sglang-step35.pid

# ── 等待服务就绪 ──────────────────────────────────────────────────────────────
echo "[start] 等待服务就绪（最多 300s）..."
for i in $(seq 1 60); do
    sleep 5
    if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "[start] 服务就绪 (${i}×5s = $((i*5))s)"
        echo "[start] 查看日志: tail -f $LOG"
        exit 0
    fi
    # 检查进程是否仍在运行
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[start] ERROR: 服务进程已退出，查看日志: tail -50 $LOG" >&2
        tail -50 "$LOG" >&2
        exit 1
    fi
    printf "."
done

echo ""
echo "[start] WARNING: 300s 后服务仍未就绪，请手动检查: tail -f $LOG" >&2
