from __future__ import annotations

import sys
import types

import pytest

from minisgl.utils import platform


@pytest.fixture(autouse=True)
def clear_platform_cache(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MINISGL_PLATFORM", raising=False)
    monkeypatch.delenv("MACA_PATH", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.delenv("CUCC_PATH", raising=False)
    monkeypatch.delenv("MACA_CLANG_PATH", raising=False)
    platform.get_accelerator_platform.cache_clear()
    yield
    platform.get_accelerator_platform.cache_clear()


def test_explicit_metax_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINISGL_PLATFORM", "metax")
    assert platform.get_accelerator_platform("cuda") == "metax"


def test_maca_path_detects_metax(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MACA_PATH", "/opt/maca")
    assert platform.get_accelerator_platform("cuda") == "metax"


def test_metax_torch_version_detects_metax(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "2.10.0+metax20260709.998"
    fake_torch.version = types.SimpleNamespace(cuda="MACA 3.8")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert platform.get_accelerator_platform("cuda") == "metax"


def test_plain_cuda_defaults_to_nvidia(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "_probe_metax", lambda: False)
    assert platform.get_accelerator_platform("cuda") == "nvidia"


def test_invalid_override_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINISGL_PLATFORM", "unknown")
    with pytest.raises(ValueError, match="unsupported MINISGL_PLATFORM"):
        platform.get_accelerator_platform("cuda")
