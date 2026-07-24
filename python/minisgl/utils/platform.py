from __future__ import annotations

import functools
import os
from typing import Literal

from .device import DeviceType, get_device_type

AcceleratorPlatform = Literal["nvidia", "metax", "ascend", "cpu"]

_PLATFORM_ENV = "MINISGL_PLATFORM"
_SUPPORTED_PLATFORMS = {"nvidia", "metax", "ascend", "cpu"}


def _probe_metax() -> bool:
    """Detect the MACA compatibility stack without touching device state."""
    if os.environ.get("MACA_PATH"):
        return True

    path_markers = " ".join(
        os.environ.get(name, "")
        for name in ("CUDA_HOME", "CUDA_PATH", "CUCC_PATH", "MACA_CLANG_PATH")
    ).lower()
    if "maca" in path_markers or "metax" in path_markers:
        return True

    try:
        import torch
    except Exception:
        return False

    version_markers = " ".join(
        str(value)
        for value in (
            getattr(torch, "__version__", ""),
            getattr(getattr(torch, "version", None), "cuda", ""),
        )
    ).lower()
    return "metax" in version_markers or "maca" in version_markers


@functools.cache
def get_accelerator_platform(
    device_type: DeviceType | None = None,
) -> AcceleratorPlatform:
    """Resolve hardware vendor separately from PyTorch's device API.

    MetaX reports CUDA-compatible devices through ``torch.cuda``. It must keep
    ``device_type == 'cuda'`` while selecting different kernels from NVIDIA.
    """
    override = os.environ.get(_PLATFORM_ENV, "").strip().lower()
    if override:
        if override not in _SUPPORTED_PLATFORMS:
            supported = ", ".join(sorted(_SUPPORTED_PLATFORMS))
            raise ValueError(
                f"unsupported {_PLATFORM_ENV}={override!r}; expected one of: {supported}"
            )
        return override  # type: ignore[return-value]

    resolved = device_type if device_type is not None else get_device_type()
    if resolved == "npu":
        return "ascend"
    if resolved == "cpu":
        return "cpu"
    return "metax" if _probe_metax() else "nvidia"


def is_metax_platform() -> bool:
    return get_accelerator_platform() == "metax"


__all__ = ["AcceleratorPlatform", "get_accelerator_platform", "is_metax_platform"]
