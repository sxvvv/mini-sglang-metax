from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

BATCH_PATTERN = re.compile(
    r"SchedulerBatch phase=(?P<phase>prefill|decode) "
    r"batch_size=(?P<batch_size>\d+) token_count=(?P<token_count>\d+) "
    r"pending_requests=(?P<pending_requests>\d+) "
    r"running_requests=(?P<running_requests>\d+) "
    r"uids=\[(?P<uids>[0-9,]*)\]"
)


def parse_batch_observation(line: str) -> dict[str, object] | None:
    match = BATCH_PATTERN.search(line)
    if match is None:
        return None
    raw_uids = match.group("uids")
    return {
        "phase": match.group("phase"),
        "batch_size": int(match.group("batch_size")),
        "token_count": int(match.group("token_count")),
        "pending_requests": int(match.group("pending_requests")),
        "running_requests": int(match.group("running_requests")),
        "uids": [] if not raw_uids else [int(value) for value in raw_uids.split(",")],
    }


def summarize_batch_observations(lines: Iterable[str]) -> dict[str, object]:
    observations = [
        observation
        for line in lines
        if (observation := parse_batch_observation(line)) is not None
    ]
    phase_counts = {
        phase: sum(observation["phase"] == phase for observation in observations)
        for phase in ("prefill", "decode")
    }
    return {
        "status": "PASS" if observations else "FAIL",
        "observation_count": len(observations),
        "phase_counts": phase_counts,
        "max_batch_size": max(
            (int(observation["batch_size"]) for observation in observations),
            default=0,
        ),
        "multi_request_batch_count": sum(
            int(observation["batch_size"]) > 1 for observation in observations
        ),
        "max_pending_requests": max(
            (int(observation["pending_requests"]) for observation in observations),
            default=0,
        ),
        "max_running_requests": max(
            (int(observation["running_requests"]) for observation in observations),
            default=0,
        ),
        "max_batch_token_count": max(
            (int(observation["token_count"]) for observation in observations),
            default=0,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SchedulerBatch log records")
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--summary-file", type=Path, required=True)
    parser.add_argument("--min-batch-size", type=int, default=1)
    parser.add_argument("--min-pending-requests", type=int, default=0)
    args = parser.parse_args()

    summary = summarize_batch_observations(
        args.server_log.read_text(encoding="utf-8", errors="replace").splitlines()
    )
    if int(summary["max_batch_size"]) < args.min_batch_size:
        raise AssertionError(
            f"Expected batch_size >= {args.min_batch_size}, got {summary!r}"
        )
    if int(summary["max_pending_requests"]) < args.min_pending_requests:
        raise AssertionError(
            "Expected scheduler backlog "
            f">= {args.min_pending_requests}, got {summary!r}"
        )

    args.summary_file.parent.mkdir(parents=True, exist_ok=True)
    args.summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
