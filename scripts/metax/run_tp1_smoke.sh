#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TP=1
exec bash "${SCRIPT_DIR}/run_gate0.sh"
