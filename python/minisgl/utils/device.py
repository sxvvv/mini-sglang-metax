from __future__ import annotations

import functools
from typing import Literal

DeviceType = Literal["cuda", "cpu"]

__all__ = [
    "DeviceType",
    "get_device_type",
    "is_cuda_available",
]


def _probe_cuda() -> bool:
    """Return True if a CUDA-enabled ``torch`` build reports at least one device.

    All failure modes (torch missing, driver missing, transient runtime errors)
    are treated as "CUDA not available" — never raised to callers.

    On MetaX the vendor ``torch`` keeps the CUDA-facing API, so this returns
    True for MetaX devices too; the accelerator vendor is resolved separately in
    :mod:`minisgl.utils.platform`.
    """
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


@functools.cache
def is_cuda_available() -> bool:
    """Cached ``True`` iff a usable CUDA device is present. Never raises."""
    return _probe_cuda()


@functools.cache
def get_device_type() -> DeviceType:
    """Return the preferred accelerator, preferring CUDA over CPU.

    Contract: no side effects, never raises, never sets or selects a device.
    Cached — call ``get_device_type.cache_clear()`` from tests when swapping
    the underlying probes via monkeypatch.
    """
    if is_cuda_available():
        return "cuda"
    return "cpu"
