#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a local dense Qwen checkpoint}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PERSIST_ROOT="${SW_HOME:-/sw_home/${USER:-user}}"
PERSIST_SITE_PACKAGES="${PERSIST_SITE_PACKAGES:-${PERSIST_ROOT}/python-packages}"
RESULT_DIR="${RESULT_DIR:-${PERSIST_ROOT}/results/mini-sglang-metax/$(date +%F)}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-1919}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-180}"
NUM_PAGES="${NUM_PAGES:-512}"
MAX_TOKENS="${MAX_TOKENS:-4}"
RESULT_PREFIX="${RESULT_PREFIX:-online_gate1}"
ONLINE_EXTENDED="${ONLINE_EXTENDED:-0}"
ONLINE_GATE1_2="${ONLINE_GATE1_2:-0}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-torch_native}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-256}"
SOAK_CONCURRENCY="${SOAK_CONCURRENCY:-8}"
SOAK_ROUNDS="${SOAK_ROUNDS:-3}"
SOAK_BASE_MAX_TOKENS="${SOAK_BASE_MAX_TOKENS:-8}"

SERVER_LOG="${RESULT_DIR}/${RESULT_PREFIX}_server.log"
PRECHECK_LOG="${RESULT_DIR}/${RESULT_PREFIX}_preflight.log"
RC_FILE="${RESULT_DIR}/${RESULT_PREFIX}.rc"
MODELS_RESPONSE="${RESULT_DIR}/${RESULT_PREFIX}_models.json"
REQUEST_BODY="${RESULT_DIR}/${RESULT_PREFIX}_request.json"
CHAT_RESPONSE_1="${RESULT_DIR}/${RESULT_PREFIX}_chat_1.json"
CHAT_RESPONSE_2="${RESULT_DIR}/${RESULT_PREFIX}_chat_2.json"
SUMMARY_FILE="${RESULT_DIR}/${RESULT_PREFIX}_summary.json"
EXTENDED_SUMMARY_FILE="${RESULT_DIR}/${RESULT_PREFIX}_extended_summary.json"
SOAK_SUMMARY_FILE="${RESULT_DIR}/${RESULT_PREFIX}_soak_summary.json"
BATCH_SUMMARY_FILE="${RESULT_DIR}/${RESULT_PREFIX}_batch_summary.json"

SERVER_PID=""

cleanup() {
  local rc=$?
  trap - EXIT INT TERM

  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || kill -TERM "${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 50); do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
      kill -KILL -- "-${SERVER_PID}" 2>/dev/null || kill -KILL "${SERVER_PID}" 2>/dev/null || true
    fi
    wait "${SERVER_PID}" 2>/dev/null || true
  fi

  printf '%s\n' "${rc}" > "${RC_FILE}"
  exit "${rc}"
}
trap cleanup EXIT INT TERM

mkdir -p "${RESULT_DIR}"

export MINISGL_PLATFORM="${MINISGL_PLATFORM:-metax}"
export MACA_PATH="${MACA_PATH:-/opt/maca}"
export CUDA_HOME="${CUDA_HOME:-${MACA_PATH}/tools/cu-bridge}"
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
export CUCC_PATH="${CUCC_PATH:-${CUDA_HOME}}"
export PYTHONPATH="${PERSIST_SITE_PACKAGES}:${ROOT_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"

for command_name in curl setsid; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Missing required command: ${command_name}" >&2
    exit 2
  fi
done

python - <<'PY'
import fastapi
import msgpack
import uvicorn
import zmq
PY

cd "${ROOT_DIR}"
python scripts/metax/preflight.py 2>&1 | tee "${PRECHECK_LOG}"

python - "${MODEL_PATH}" "${MAX_TOKENS}" "${REQUEST_BODY}" <<'PY'
import json
import sys

model_path, max_tokens, output_path = sys.argv[1:]
payload = {
    "model": model_path,
    "messages": [{"role": "user", "content": "Count from one to three."}],
    "max_tokens": int(max_tokens),
    "temperature": 0.0,
    "top_k": 1,
    "top_p": 1.0,
    "stream": False,
    "ignore_eos": True,
}
with open(output_path, "w", encoding="utf-8") as file:
    json.dump(payload, file)
PY

echo "[mini-sglang-metax] starting online Gate 1 on ${HOST}:${PORT}"
setsid python -m minisgl \
  --model-path "${MODEL_PATH}" \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --disable-pynccl \
  --cuda-graph-max-bs 0 \
  --attention-backend "${ATTENTION_BACKEND}" \
  --num-pages "${NUM_PAGES}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --host "${HOST}" \
  --port "${PORT}" \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + STARTUP_TIMEOUT))
while true; do
  if curl --silent --show-error --fail \
    "http://${HOST}:${PORT}/v1/models" \
    --output "${MODELS_RESPONSE}"; then
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Server exited before readiness" >&2
    tail -n 80 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  if grep -q '^Process minisgl-TP.*scheduler:' "${SERVER_LOG}"; then
    echo "Scheduler worker failed before readiness" >&2
    tail -n 80 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  if ((SECONDS >= deadline)); then
    echo "Server did not become ready within ${STARTUP_TIMEOUT}s" >&2
    tail -n 80 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  sleep 1
done

for response_file in "${CHAT_RESPONSE_1}" "${CHAT_RESPONSE_2}"; do
  http_code="$(curl --silent --show-error \
    --request POST \
    --header 'Content-Type: application/json' \
    --data-binary "@${REQUEST_BODY}" \
    --output "${response_file}" \
    --write-out '%{http_code}' \
    "http://${HOST}:${PORT}/v1/chat/completions")"
  if [[ "${http_code}" != "200" ]]; then
    echo "Chat completion returned HTTP ${http_code}" >&2
    cat "${response_file}" >&2 || true
    exit 1
  fi
done

python - \
  "${MODEL_PATH}" \
  "${MODELS_RESPONSE}" \
  "${CHAT_RESPONSE_1}" \
  "${CHAT_RESPONSE_2}" \
  "${SUMMARY_FILE}" <<'PY'
import json
import sys

model_path, models_path, chat_1_path, chat_2_path, summary_path = sys.argv[1:]

with open(models_path, encoding="utf-8") as file:
    models = json.load(file)
with open(chat_1_path, encoding="utf-8") as file:
    chat_1 = json.load(file)
with open(chat_2_path, encoding="utf-8") as file:
    chat_2 = json.load(file)

cards = models.get("data", [])
assert len(cards) == 1, models
assert cards[0]["id"] == model_path, models
assert cards[0]["root"] == model_path, models

def content(response: dict) -> str:
    assert response.get("object") == "chat.completion", response
    assert response.get("model") == model_path, response
    choices = response.get("choices", [])
    assert len(choices) == 1, response
    value = choices[0]["message"]["content"]
    assert isinstance(value, str) and value, response
    return value

content_1 = content(chat_1)
content_2 = content(chat_2)
assert content_1 == content_2, (content_1, content_2)

summary = {
    "status": "PASS",
    "model_path": model_path,
    "models_count": len(cards),
    "chat_requests": 2,
    "deterministic_output": True,
    "output": content_1,
}
with open(summary_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False))
PY

if [[ "${ONLINE_EXTENDED}" == "1" ]]; then
  python scripts/metax/online_gate1_client.py \
    --base-url "http://${HOST}:${PORT}" \
    --model-path "${MODEL_PATH}" \
    --summary-file "${EXTENDED_SUMMARY_FILE}"

  # The extended client closes a live SSE connection deliberately. Require
  # proof that the disconnect reached the frontend and completed AbortAck.
  for _ in $(seq 1 30); do
    if grep -q "Aborting request for user" "${SERVER_LOG}" && \
      grep -q "Abort acknowledged for user" "${SERVER_LOG}"; then
      break
    fi
    sleep 0.1
  done
  if ! grep -q "Aborting request for user" "${SERVER_LOG}"; then
    echo "Cancellation did not reach the frontend abort path" >&2
    exit 1
  fi
  if ! grep -q "Abort acknowledged for user" "${SERVER_LOG}"; then
    echo "Cancellation did not complete the AbortAck path" >&2
    exit 1
  fi
fi

if [[ "${ONLINE_GATE1_2}" == "1" ]]; then
  python scripts/metax/online_gate1_2_client.py \
    --base-url "http://${HOST}:${PORT}" \
    --model-path "${MODEL_PATH}" \
    --summary-file "${SOAK_SUMMARY_FILE}" \
    --concurrency "${SOAK_CONCURRENCY}" \
    --rounds "${SOAK_ROUNDS}" \
    --base-max-tokens "${SOAK_BASE_MAX_TOKENS}" \
    --admission-limit "${MAX_RUNNING_REQUESTS}"

  for _ in $(seq 1 30); do
    if grep -q "Aborting request for user" "${SERVER_LOG}" && \
      grep -q "Abort acknowledged for user" "${SERVER_LOG}"; then
      break
    fi
    sleep 0.1
  done
  if ! grep -q "Aborting request for user" "${SERVER_LOG}"; then
    echo "Gate 1.2 cancellation did not reach the frontend abort path" >&2
    exit 1
  fi
  if ! grep -q "Abort acknowledged for user" "${SERVER_LOG}"; then
    echo "Gate 1.2 cancellation did not complete the AbortAck path" >&2
    exit 1
  fi

  python scripts/metax/batch_observability.py \
    --server-log "${SERVER_LOG}" \
    --summary-file "${BATCH_SUMMARY_FILE}" \
    --min-batch-size 2 \
    --min-pending-requests 1
fi

echo "[mini-sglang-metax] online Gate 1 PASS; summary=${SUMMARY_FILE}"
if [[ "${ONLINE_EXTENDED}" == "1" ]]; then
  echo "[mini-sglang-metax] online Gate 1.1 PASS; summary=${EXTENDED_SUMMARY_FILE}"
fi
if [[ "${ONLINE_GATE1_2}" == "1" ]]; then
  echo "[mini-sglang-metax] online Gate 1.2 PASS; summary=${SOAK_SUMMARY_FILE}"
  echo "[mini-sglang-metax] batch observations PASS; summary=${BATCH_SUMMARY_FILE}"
fi
