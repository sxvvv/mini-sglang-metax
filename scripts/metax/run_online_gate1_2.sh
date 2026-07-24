#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ONLINE_GATE1_2=1
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-2}"
export RESULT_PREFIX="${RESULT_PREFIX:-online_gate1_2}"
exec bash "${SCRIPT_DIR}/run_online_gate1.sh"
