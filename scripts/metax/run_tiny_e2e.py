from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded Mini-SGLang end-to-end smoke on MetaX",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--num-pages", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    os.environ.setdefault("MINISGL_PLATFORM", "metax")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        os.environ.setdefault("MINISGL_DISTRIBUTED_ADDR", "env://")

    report: dict[str, object] = {
        "model_path": args.model_path,
        "rank": rank,
        "device": f"cuda:{local_rank}",
        "tp": world_size,
        "attention_backend": "torch_native",
        "max_new_tokens": args.max_new_tokens,
        "num_pages": args.num_pages,
        "repeats": args.repeats,
        "status": "FAIL",
    }
    llm = None

    try:
        import torch
        from minisgl.core import SamplingParams
        from minisgl.distributed import DistributedInfo
        from minisgl.llm import LLM

        started_at = time.perf_counter()
        llm = LLM(
            model_path=args.model_path,
            dtype=torch.bfloat16,
            tp_info=DistributedInfo(rank=rank, size=world_size),
            attention_backend="torch_native",
            max_running_req=4,
            max_extend_tokens=128,
            max_seq_len_override=128,
            num_page_override=args.num_pages,
            page_size=1,
            cuda_graph_bs=[],
            cuda_graph_max_bs=0,
            use_pynccl=False,
        )
        report["load_seconds"] = round(time.perf_counter() - started_at, 4)

        sampling_params = SamplingParams(
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            max_tokens=args.max_new_tokens,
            ignore_eos=True,
        )
        runs = []
        for run_index in range(args.repeats):
            started_at = time.perf_counter()
            outputs = llm.generate([[1, 4, 5, 6]], sampling_params)
            torch.cuda.synchronize()
            output_ids = outputs[0]["token_ids"]
            if len(output_ids) != args.max_new_tokens:
                raise RuntimeError(
                    f"run {run_index}: expected {args.max_new_tokens} output tokens, "
                    f"got {len(output_ids)}"
                )
            llm.cache_manager.check_integrity()
            runs.append(
                {
                    "run": run_index + 1,
                    "generate_seconds": round(time.perf_counter() - started_at, 4),
                    "output": outputs[0],
                    "available_tokens_after": llm.cache_manager.available_size,
                }
            )

        report["runs"] = runs
        llm.shutdown()
        report["status"] = "PASS"
        print("METAX_E2E_RESULT=" + json.dumps(report, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        if llm is not None:
            try:
                llm.shutdown()
            except BaseException as shutdown_exc:  # noqa: BLE001
                report["shutdown_error"] = repr(shutdown_exc)
        print("METAX_E2E_RESULT=" + json.dumps(report, ensure_ascii=False), flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
