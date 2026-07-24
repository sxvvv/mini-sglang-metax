from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "metax" / "online_gate1_client.py"
)
SPEC = importlib.util.spec_from_file_location("online_gate1_client", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_chat_payload_is_greedy_and_bounded() -> None:
    payload = MODULE._chat_payload("/model", "hello", max_tokens=7, stream=True)

    assert payload["model"] == "/model"
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["max_tokens"] == 7
    assert payload["temperature"] == 0.0
    assert payload["top_k"] == 1
    assert payload["stream"] is True


def test_validate_models_accepts_exact_card() -> None:
    response = {"data": [{"id": "/model", "root": "/model"}]}

    assert MODULE._validate_models(response, "/model") == 1


def test_validate_models_rejects_wrong_card() -> None:
    with pytest.raises(AssertionError, match="Unexpected model card"):
        MODULE._validate_models({"data": [{"id": "/other", "root": "/other"}]}, "/model")


def test_chat_content_accepts_non_empty_completion() -> None:
    response = {
        "object": "chat.completion",
        "model": "/model",
        "choices": [{"message": {"content": "ready"}}],
    }

    assert MODULE._chat_content(response, "/model") == "ready"


@pytest.mark.parametrize(
    ("line", "expected_kind"),
    [
        (b"\n", "ignore"),
        (b'data: {"choices": []}\n', "event"),
        (b"data: [DONE]\n", "done"),
    ],
)
def test_decode_sse_line(line: bytes, expected_kind: str) -> None:
    kind, _ = MODULE._decode_sse_line(line)

    assert kind == expected_kind


def test_decode_sse_line_rejects_unknown_wire_format() -> None:
    with pytest.raises(AssertionError, match="Unexpected SSE line"):
        MODULE._decode_sse_line(b"event: token\n")
