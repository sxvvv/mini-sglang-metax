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


def chat_payload(
    model_path: str,
    prompt: str,
    *,
    max_tokens: int,
    stream: bool = False,
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
    def __init__(self, base_url: str, timeout: float) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError(f"Expected an http URL, got: {base_url}")
        self.host = parsed.hostname
        self.port = parsed.port or 80
        self.timeout = timeout

    def connect(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> tuple[int, bytes]:
        body = None if payload is None else json.dumps(payload).encode()
        headers = {} if body is None else {"Content-Type": "application/json"}
        connection = self.connect()
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def json_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        status, raw = self.request(method, path, payload)
        if status != 200:
            raise AssertionError(
                f"{method} {path} returned HTTP {status}: {raw.decode(errors='replace')}"
            )
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise AssertionError(f"Expected JSON object from {path}, got: {value!r}")
        return value


def validate_models(response: dict[str, Any], model_path: str) -> None:
    cards = response.get("data", [])
    if not isinstance(cards, list) or len(cards) != 1:
        raise AssertionError(f"Expected one model card, got: {response!r}")
    if cards[0].get("id") != model_path or cards[0].get("root") != model_path:
        raise AssertionError(f"Unexpected model card: {cards[0]!r}")


def validate_chat(response: dict[str, Any], model_path: str) -> int:
    if response.get("object") != "chat.completion" or response.get("model") != model_path:
        raise AssertionError(f"Unexpected chat metadata: {response!r}")
    choices = response.get("choices", [])
    if not isinstance(choices, list) or len(choices) != 1:
        raise AssertionError(f"Expected one choice, got: {response!r}")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str) or not content:
        raise AssertionError(f"Expected non-empty content, got: {response!r}")
    return len(content)


def cancel_live_stream(client: HttpClient, model_path: str) -> dict[str, bool]:
    connection = client.connect()
    received_event = False
    try:
        payload = chat_payload(
            model_path,
            "Write a long numbered explanation with many details.",
            max_tokens=128,
            stream=True,
        )
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
            line = raw_line.decode("utf-8").strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                received_event = True
                break
    finally:
        connection.close()
    if not received_event:
        raise AssertionError("Cancellation stream ended before its first event")
    return {"first_event_received": True, "connection_closed": True}


def run_soak_round(
    client: HttpClient,
    model_path: str,
    *,
    round_index: int,
    concurrency: int,
    base_max_tokens: int,
) -> tuple[list[dict[str, object]], int]:
    barrier = threading.Barrier(concurrency)
    lock = threading.Lock()
    active = 0
    peak_active = 0

    def worker(index: int) -> dict[str, object]:
        nonlocal active, peak_active
        barrier.wait(timeout=15)
        with lock:
            active += 1
            peak_active = max(peak_active, active)
        started = time.monotonic()
        try:
            prompt = (
                f"Round {round_index}, request {index}: reply with a short sentence "
                f"containing the number {index}."
            )
            max_tokens = base_max_tokens + (index % 4)
            response = client.json_request(
                "POST",
                "/v1/chat/completions",
                chat_payload(model_path, prompt, max_tokens=max_tokens),
            )
            output_chars = validate_chat(response, model_path)
            return {
                "round": round_index,
                "index": index,
                "max_tokens": max_tokens,
                "output_chars": output_chars,
                "elapsed_seconds": round(time.monotonic() - started, 4),
            }
        finally:
            with lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(worker, range(concurrency)))
    return results, peak_active


def run(
    base_url: str,
    model_path: str,
    *,
    concurrency: int,
    rounds: int,
    base_max_tokens: int,
    admission_limit: int,
    timeout: float,
) -> dict[str, object]:
    if concurrency <= admission_limit:
        raise ValueError("concurrency must exceed admission_limit for overload coverage")
    client = HttpClient(base_url, timeout)
    validate_models(client.json_request("GET", "/v1/models"), model_path)

    invalid_status, _ = client.request(
        "POST",
        "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "missing model"}]},
    )
    if invalid_status != 422:
        raise AssertionError(f"Expected malformed request HTTP 422, got {invalid_status}")

    started = time.monotonic()
    requests: list[dict[str, object]] = []
    peak_client_inflight = 0
    for round_index in range(rounds):
        round_results, round_peak = run_soak_round(
            client,
            model_path,
            round_index=round_index,
            concurrency=concurrency,
            base_max_tokens=base_max_tokens,
        )
        requests.extend(round_results)
        peak_client_inflight = max(peak_client_inflight, round_peak)

    cancellation = cancel_live_stream(client, model_path)
    time.sleep(0.5)
    validate_models(client.json_request("GET", "/v1/models"), model_path)
    final_output_chars = validate_chat(
        client.json_request(
            "POST",
            "/v1/chat/completions",
            chat_payload(model_path, "Reply with the word recovered.", max_tokens=4),
        ),
        model_path,
    )
    expected_requests = rounds * concurrency
    if len(requests) != expected_requests:
        raise AssertionError(f"Completed {len(requests)}/{expected_requests} soak requests")
    return {
        "status": "PASS",
        "scope": "bounded",
        "model_path": model_path,
        "rounds": rounds,
        "concurrency": concurrency,
        "configured_admission_limit": admission_limit,
        "offered_concurrency_exceeded_limit": concurrency > admission_limit,
        "completed_requests": len(requests),
        "failed_requests": 0,
        "peak_client_inflight": peak_client_inflight,
        "elapsed_seconds": round(time.monotonic() - started, 4),
        "malformed_request_status": invalid_status,
        "cancellation": cancellation,
        "post_fault_models_ok": True,
        "final_recovery_output_chars": final_output_chars,
        "requests": requests,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded MetaX online Gate 1.2")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--summary-file", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--base-max-tokens", type=int, default=8)
    parser.add_argument("--admission-limit", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    if min(args.concurrency, args.rounds, args.base_max_tokens, args.admission_limit) < 1:
        parser.error("concurrency, rounds, token count, and admission limit must be positive")

    summary = run(
        args.base_url,
        args.model_path,
        concurrency=args.concurrency,
        rounds=args.rounds,
        base_max_tokens=args.base_max_tokens,
        admission_limit=args.admission_limit,
        timeout=args.timeout,
    )
    args.summary_file.parent.mkdir(parents=True, exist_ok=True)
    args.summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
