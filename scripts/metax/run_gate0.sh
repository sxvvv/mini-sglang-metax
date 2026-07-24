#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a local dense Qwen checkpoint}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PERSIST_ROOT="${SW_HOME:-/sw_home/${USER:-user}}"
TP="${TP:-1}"
NUM_PAGES="${NUM_PAGES:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4}"
REPEATS="${REPEATS:-2}"
RESULT_DIR="${RESULT_DIR:-${PERSIST_ROOT}/results/mini-sglang-metax/$(date +%F)}"

if ! [[ "$TP" =~ ^[1-9][0-9]*$ ]]; then
  echo "TP must be a positive integer, got: $TP" >&2
  exit 2
fi

mkdir -p "$RESULT_DIR"

export MINISGL_PLATFORM="${MINISGL_PLATFORM:-metax}"
export MACA_PATH="${MACA_PATH:-/opt/maca}"
export CUDA_HOME="${CUDA_HOME:-${MACA_PATH}/tools/cu-bridge}"
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
export CUCC_PATH="${CUCC_PATH:-${CUDA_HOME}}"
export PYTHONPATH="${ROOT_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"

cd "$ROOT_DIR"

PRECHECK_LOG="${RESULT_DIR}/preflight.log"
RUN_LOG="${RESULT_DIR}/gate0_tp${TP}.log"
RC_FILE="${RESULT_DIR}/gate0_tp${TP}.rc"

echo "[mini-sglang-metax] model=${MODEL_PATH}"
echo "[mini-sglang-metax] tp=${TP} result_dir=${RESULT_DIR}"

python scripts/metax/preflight.py 2>&1 | tee "$PRECHECK_LOG"

args=(
  scripts/metax/run_tiny_e2e.py
  --model-path "$MODEL_PATH"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --num-pages "$NUM_PAGES"
  --repeats "$REPEATS"
)

set +e
if [[ "$TP" == "1" ]]; then
  python "${args[@]}" 2>&1 | tee "$RUN_LOG"
  rc=${PIPESTATUS[0]}
else
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
    torchrun --standalone --nproc_per_node="$TP" "${args[@]}" 2>&1 | tee "$RUN_LOG"
  rc=${PIPESTATUS[0]}
fi
set -e

printf '%s\n' "$rc" > "$RC_FILE"
echo "[mini-sglang-metax] exit_code=${rc} log=${RUN_LOG}"
exit "$rc"
