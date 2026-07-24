from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "metax" / "batch_observability.py"
)
SPEC = importlib.util.spec_from_file_location("batch_observability", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_batch_observation_ignores_other_log_lines() -> None:
    assert MODULE.parse_batch_observation("Scheduler is idle") is None


def test_summarize_proves_multi_request_batching_and_backlog() -> None:
    lines = [
        "prefix SchedulerBatch phase=prefill batch_size=2 token_count=19 "
        "pending_requests=6 running_requests=0 uids=[4,5] suffix",
        "SchedulerBatch phase=decode batch_size=2 token_count=2 "
        "pending_requests=6 running_requests=2 uids=[4,5]",
        "SchedulerBatch phase=decode batch_size=1 token_count=1 "
        "pending_requests=0 running_requests=1 uids=[9]",
    ]

    summary = MODULE.summarize_batch_observations(lines)

    assert summary == {
        "status": "PASS",
        "observation_count": 3,
        "phase_counts": {"prefill": 1, "decode": 2},
        "max_batch_size": 2,
        "multi_request_batch_count": 2,
        "max_pending_requests": 6,
        "max_running_requests": 2,
        "max_batch_token_count": 19,
    }
