"""Unit tests for :mod:`minisgl.utils.device`.

The device-probe layer must be safe to import on hosts without CUDA (e.g.
developer macOS boxes). These tests use ``monkeypatch`` plus isolated module
loading — no real hardware, no reliance on the wider ``minisgl`` runtime deps.

To keep the tests hermetic the ``device`` module is loaded directly from its
file location, bypassing ``minisgl.utils.__init__`` which pulls in ``torch``,
``transformers`` and other heavyweight optional dependencies that are not part
of the device-probe contract under test.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEVICE_PATH = _REPO_ROOT / "python" / "minisgl" / "utils" / "device.py"
_ARCH_PATH = _REPO_ROOT / "python" / "minisgl" / "utils" / "arch.py"


def _load_isolated(module_name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None, f"cannot build spec for {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Import device.py in isolation and expose it under its canonical dotted name
# so that ``from .device import ...`` inside arch.py resolves to the same
# object we monkeypatch below.
device = _load_isolated("minisgl.utils.device", _DEVICE_PATH)


@pytest.fixture(autouse=True)
def _reset_device_cache():
    """Clear the ``@functools.cache`` on each probe before and after every test."""
    device.is_cuda_available.cache_clear()
    device.get_device_type.cache_clear()
    yield
    device.is_cuda_available.cache_clear()
    device.get_device_type.cache_clear()


def test_module_imports_cleanly() -> None:
    """The module must load on a host that has no CUDA build at all."""
    # Public API surface exists and is callable without side effects.
    assert callable(device.get_device_type)
    assert callable(device.is_cuda_available)


def test_cuda_available_returns_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """When CUDA reports available, the device type is 'cuda'."""
    monkeypatch.setattr(device, "_probe_cuda", lambda: True)

    assert device.is_cuda_available() is True
    assert device.get_device_type() == "cuda"


def test_cuda_absent_returns_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """No CUDA must fall through to 'cpu' cleanly."""
    monkeypatch.setattr(device, "_probe_cuda", lambda: False)

    assert device.is_cuda_available() is False
    assert device.get_device_type() == "cpu"


def test_get_device_type_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """``get_device_type`` memoises — repeated probe swaps do not leak through."""
    monkeypatch.setattr(device, "_probe_cuda", lambda: True)
    assert device.get_device_type() == "cuda"

    # Change the underlying probe: cached value should stick until cleared.
    monkeypatch.setattr(device, "_probe_cuda", lambda: False)
    assert device.get_device_type() == "cuda"

    device.get_device_type.cache_clear()
    device.is_cuda_available.cache_clear()
    assert device.get_device_type() == "cpu"


def test_arch_module_uses_device_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    """``arch.is_arch_supported`` must delegate CUDA presence to the device layer."""
    # Load arch.py in isolation; its ``from .device import is_cuda_available``
    # will resolve to the already-registered ``minisgl.utils.device`` above.
    # Also register a minimal ``minisgl.utils`` package so the relative import
    # can bind ``.device`` without executing the real ``__init__.py``.
    if "minisgl" not in sys.modules:
        pkg_minisgl = types.ModuleType("minisgl")
        pkg_minisgl.__path__ = [str(_REPO_ROOT / "python" / "minisgl")]
        sys.modules["minisgl"] = pkg_minisgl
    if "minisgl.utils" not in sys.modules:
        pkg_utils = types.ModuleType("minisgl.utils")
        pkg_utils.__path__ = [str(_REPO_ROOT / "python" / "minisgl" / "utils")]
        sys.modules["minisgl.utils"] = pkg_utils

    arch = _load_isolated("minisgl.utils.arch", _ARCH_PATH)
    arch._get_torch_cuda_version.cache_clear()

    monkeypatch.setattr(device, "_probe_cuda", lambda: False)

    assert arch.is_arch_supported(9, 0) is False
    assert arch.is_sm90_supported() is False
    assert arch.is_sm100_supported() is False

    arch._get_torch_cuda_version.cache_clear()
