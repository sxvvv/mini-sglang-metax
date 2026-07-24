from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "metax"
    / "online_gate1_2_client.py"
)
SPEC = importlib.util.spec_from_file_location("online_gate1_2_client", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_chat_payload_is_greedy_and_bounded() -> None:
    payload = MODULE.chat_payload("/model", "hello", max_tokens=9)

    assert payload["model"] == "/model"
    assert payload["max_tokens"] == 9
    assert payload["temperature"] == 0.0
    assert payload["top_k"] == 1
    assert payload["top_p"] == 1.0
    assert payload["ignore_eos"] is True


def test_validate_chat_requires_non_empty_content() -> None:
    with pytest.raises(AssertionError, match="non-empty"):
        MODULE.validate_chat(
            {
                "object": "chat.completion",
                "model": "/model",
                "choices": [{"message": {"content": ""}}],
            },
            "/model",
        )


def test_run_rejects_load_that_does_not_exceed_admission_limit() -> None:
    with pytest.raises(ValueError, match="exceed admission_limit"):
        MODULE.run(
            "http://127.0.0.1:1919",
            "/model",
            concurrency=2,
            rounds=1,
            base_max_tokens=4,
            admission_limit=2,
            timeout=1,
        )


def test_runner_detects_dead_scheduler_before_startup_timeout() -> None:
    runner = (SCRIPT_PATH.parent / "run_online_gate1.sh").read_text(encoding="utf-8")

    worker_check = runner.index("Scheduler worker failed before readiness")
    timeout_check = runner.index("Server did not become ready")
    assert worker_check < timeout_check
