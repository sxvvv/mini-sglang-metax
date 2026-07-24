from __future__ import annotations

from minisgl.attention.fi import _optional_backend_kwarg, _supported_optional_kwargs


def test_optional_backend_kwarg_keeps_upstream_flashinfer_selection() -> None:
    class Wrapper:
        def __init__(self, workspace, *, backend="auto") -> None:
            pass

    assert _optional_backend_kwarg(Wrapper, "fa2") == {"backend": "fa2"}


def test_optional_backend_kwarg_supports_metax_decode_signature() -> None:
    class Wrapper:
        def __init__(self, workspace, *, use_tensor_cores=False) -> None:
            pass

    assert _optional_backend_kwarg(Wrapper, "fa2") == {}


def test_optional_plan_kwargs_follow_installed_signature() -> None:
    def upstream_plan(*, seq_lens=None, non_blocking=False) -> None:
        pass

    def metax_plan(*, non_blocking=False) -> None:
        pass

    optional = {"seq_lens": object(), "non_blocking": True}
    assert _supported_optional_kwargs(upstream_plan, **optional) == optional
    assert _supported_optional_kwargs(metax_plan, **optional) == {"non_blocking": True}
