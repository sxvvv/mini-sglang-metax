#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ONLINE_EXTENDED=1
export RESULT_PREFIX="${RESULT_PREFIX:-online_gate1_1}"
exec bash "${SCRIPT_DIR}/run_online_gate1.sh"
