from __future__ import annotations

import argparse
import http.client
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


def _chat_payload(
    model_path: str,
    prompt: str,
    *,
    max_tokens: int,
    stream: bool,
) -> dict[str, Any]:
    return {
        "model": model_path,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "stream": stream,
        "ignore_eos": True,
    }


class HttpClient:
    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError(f"Expected an http URL, got: {base_url}")
        self.host = parsed.hostname
        self.port = parsed.port or 80
        self.timeout = timeout

    def connect(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    def json_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode()
        headers = {} if body is None else {"Content-Type": "application/json"}
        connection = self.connect()
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            raw = response.read()
            if response.status != 200:
                raise AssertionError(
                    f"{method} {path} returned HTTP {response.status}: "
                    f"{raw.decode(errors='replace')}"
                )
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise AssertionError(f"Expected JSON object from {path}, got: {value!r}")
            return value
        finally:
            connection.close()


def _validate_models(response: dict[str, Any], model_path: str) -> int:
    cards = response.get("data", [])
    if not isinstance(cards, list) or len(cards) != 1:
        raise AssertionError(f"Expected one model card, got: {response!r}")
    if cards[0].get("id") != model_path or cards[0].get("root") != model_path:
        raise AssertionError(f"Unexpected model card: {cards[0]!r}")
    return len(cards)


def _chat_content(response: dict[str, Any], model_path: str) -> str:
    if response.get("object") != "chat.completion" or response.get("model") != model_path:
        raise AssertionError(f"Unexpected chat response metadata: {response!r}")
    choices = response.get("choices", [])
    if not isinstance(choices, list) or len(choices) != 1:
        raise AssertionError(f"Expected one chat choice, got: {response!r}")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str) or not content:
        raise AssertionError(f"Expected non-empty chat content, got: {response!r}")
    return content


def _decode_sse_line(raw_line: bytes) -> tuple[str, dict[str, Any] | None]:
    line = raw_line.decode("utf-8").strip()
    if not line:
        return "ignore", None
    if not line.startswith("data: "):
        raise AssertionError(f"Unexpected SSE line: {line!r}")
    data = line[6:]
    if data == "[DONE]":
        return "done", None
    value = json.loads(data)
    if not isinstance(value, dict):
        raise AssertionError(f"Expected SSE JSON object, got: {value!r}")
    return "event", value


def _stream_chat(client: HttpClient, model_path: str) -> dict[str, Any]:
    payload = _chat_payload(
        model_path,
        "Reply with four short words.",
        max_tokens=4,
        stream=True,
    )
    connection = client.connect()
    events = 0
    content_parts: list[str] = []
    saw_finish = False
    saw_done = False
    try:
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        if response.status != 200:
            raise AssertionError(f"Streaming request returned HTTP {response.status}")
        while raw_line := response.readline():
            kind, event = _decode_sse_line(raw_line)
            if kind == "ignore":
                continue
            if kind == "done":
                saw_done = True
                break
            assert event is not None
            events += 1
            choices = event.get("choices", [])
            if len(choices) != 1:
                raise AssertionError(f"Unexpected streaming event: {event!r}")
            choice = choices[0]
            delta = choice.get("delta", {})
            if isinstance(delta.get("content"), str):
                content_parts.append(delta["content"])
            if choice.get("finish_reason") == "stop":
                saw_finish = True
    finally:
        connection.close()
    if events == 0 or not saw_finish or not saw_done:
        raise AssertionError(
            f"Incomplete SSE sequence: events={events}, finish={saw_finish}, done={saw_done}"
        )
    return {
        "events": events,
        "finish_event": saw_finish,
        "done_marker": saw_done,
        "content": "".join(content_parts),
    }


def _cancel_stream(client: HttpClient, model_path: str) -> dict[str, Any]:
    payload = _chat_payload(
        model_path,
        "Write a long numbered explanation with many details.",
        max_tokens=128,
        stream=True,
    )
    connection = client.connect()
    first_event = False
    try:
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        if response.status != 200:
            raise AssertionError(f"Cancellation request returned HTTP {response.status}")
        while raw_line := response.readline():
            kind, _ = _decode_sse_line(raw_line)
            if kind == "event":
                first_event = True
                break
    finally:
        # Deliberately close a live streaming response. The server must route
        # this disconnect through AbortMsg/AbortAck and remain usable.
        connection.close()
    if not first_event:
        raise AssertionError("Cancellation stream ended before its first SSE event")
    return {"first_event_received": True, "connection_closed": True}


def _concurrent_chat(
    client: HttpClient,
    model_path: str,
    concurrency: int,
) -> list[dict[str, Any]]:
    prompts = [
        "Reply with one word.",
        "Reply with exactly two short words.",
        "Name three colors.",
        "Give a four-word greeting.",
    ]
    barrier = threading.Barrier(concurrency)

    def worker(index: int) -> dict[str, Any]:
        barrier.wait(timeout=10)
        started = time.monotonic()
        response = client.json_request(
            "POST",
            "/v1/chat/completions",
            _chat_payload(
                model_path,
                prompts[index % len(prompts)],
                max_tokens=4 + index,
                stream=False,
            ),
        )
        content = _chat_content(response, model_path)
        return {
            "index": index,
            "max_tokens": 4 + index,
            "output_chars": len(content),
            "elapsed_seconds": round(time.monotonic() - started, 4),
        }

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        return list(executor.map(worker, range(concurrency)))


def run(base_url: str, model_path: str, concurrency: int) -> dict[str, Any]:
    client = HttpClient(base_url)
    model_count = _validate_models(client.json_request("GET", "/v1/models"), model_path)
    streaming = _stream_chat(client, model_path)
    cancellation = _cancel_stream(client, model_path)

    time.sleep(0.5)
    post_cancel_models = _validate_models(
        client.json_request("GET", "/v1/models"), model_path
    )
    concurrent = _concurrent_chat(client, model_path, concurrency)
    final_content = _chat_content(
        client.json_request(
            "POST",
            "/v1/chat/completions",
            _chat_payload(
                model_path,
                "Reply with the word ready.",
                max_tokens=4,
                stream=False,
            ),
        ),
        model_path,
    )
    return {
        "status": "PASS",
        "model_path": model_path,
        "models_count": model_count,
        "streaming": streaming,
        "cancellation": cancellation,
        "post_cancel_models_count": post_cancel_models,
        "concurrent_requests": concurrent,
        "final_request_output_chars": len(final_content),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise the extended MetaX online gate")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--summary-file", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")

    summary = run(args.base_url, args.model_path, args.concurrency)
    args.summary_file.parent.mkdir(parents=True, exist_ok=True)
    args.summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
