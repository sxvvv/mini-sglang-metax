from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))


def main() -> int:
    import torch
    from minisgl.attention.torch_native import torch_attention_for_request
    from minisgl.utils.device import get_device_type
    from minisgl.utils.platform import get_accelerator_platform

    device_type = get_device_type()
    platform = get_accelerator_platform(device_type)
    report = {
        "torch": torch.__version__,
        "torch_cuda_version": str(getattr(torch.version, "cuda", None)),
        "device_type": device_type,
        "platform": platform,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "maca_path": os.environ.get("MACA_PATH"),
        "cuda_home": os.environ.get("CUDA_HOME"),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if device_type != "cuda" or not torch.cuda.is_available():
        print("FAIL: the MetaX torch build must expose at least one torch.cuda device")
        return 2
    if platform != "metax":
        print("FAIL: MetaX was not detected; export MINISGL_PLATFORM=metax and retry")
        return 3

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    x = torch.randn((32, 32), device=device, dtype=torch.bfloat16)
    y = x @ x
    if not torch.isfinite(y).all().item():
        print("FAIL: bf16 matmul produced non-finite output")
        return 4

    q = torch.zeros((2, 2, 8), device=device, dtype=torch.bfloat16)
    k = torch.zeros((3, 1, 8), device=device, dtype=torch.bfloat16)
    v = torch.randn((3, 1, 8), device=device, dtype=torch.bfloat16)
    attention = torch_attention_for_request(q, k, v, cached_len=1, scale=8**-0.5)
    torch.cuda.synchronize()
    if attention.shape != q.shape or not torch.isfinite(attention).all().item():
        print("FAIL: torch_native attention smoke test failed")
        return 5

    print("PASS: MetaX Gate 0 runtime prerequisites are available")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
