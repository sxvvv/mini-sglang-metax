"""Device-runtime stream primitives for CUDA and CPU.

This module is a **pure dispatch layer** over ``torch.cuda`` for the stream
entrypoints the engine touches — creation, binding as the current stream,
reading the current stream, device-wide synchronization, events, and memory
queries. Every call routes through here so that porting the engine off the raw
``torch.cuda.*`` API only touches this file.

On MetaX the vendor ``torch.cuda`` API is used directly, since the MACA stack
preserves the CUDA-facing surface.

Design notes:

* CPU is a real, fully-supported device type: ``create_stream("cpu")`` returns
  ``None`` and the other stream calls are no-ops. This matches how
  :mod:`torch` itself treats CPU — there is no stream object.
* Unknown ``device_type`` values surface as ``ValueError`` — the same error
  shape :func:`minisgl.distributed.runtime.bind_local_device` and
  :func:`minisgl.distributed.backend.get_distributed_backend` use.
"""
from __future__ import annotations

import contextlib
from typing import Any, ContextManager, Optional

from .device import DeviceType

__all__ = [
    "create_stream",
    "set_stream",
    "current_stream",
    "stream_context",
    "synchronize_device",
    "create_event",
    "record_event",
    "empty_device_cache",
    "reset_peak_memory_stats",
    "get_free_memory_bytes",
]


def _unsupported(device_type: Any) -> ValueError:
    return ValueError(
        f"unsupported device_type for device_runtime: {device_type!r}; "
        f"expected one of: cpu, cuda"
    )


def create_stream(device_type: DeviceType) -> Optional[Any]:
    """Create a fresh stream on the given device type.

    * ``cuda`` → ``torch.cuda.Stream()``
    * ``cpu``  → ``None`` (CPU has no stream concept)
    """
    if device_type == "cuda":
        import torch  # lazy: only touched on CUDA hosts

        return torch.cuda.Stream()

    if device_type == "cpu":
        return None

    raise _unsupported(device_type)


def set_stream(device_type: DeviceType, stream: Optional[Any]) -> None:
    """Bind ``stream`` as the current stream on the given device type.

    * ``cuda`` → ``torch.cuda.set_stream(stream)``
    * ``cpu``  → no-op regardless of ``stream``

    CPU accepts ``None`` (or any value) silently — matches ``create_stream``'s
    return contract.
    """
    if device_type == "cuda":
        import torch

        torch.cuda.set_stream(stream)
        return

    if device_type == "cpu":
        return

    raise _unsupported(device_type)


def current_stream(device_type: DeviceType) -> Optional[Any]:
    """Return the current stream on the given device type.

    * ``cuda`` → ``torch.cuda.current_stream()``
    * ``cpu``  → ``None``
    """
    if device_type == "cuda":
        import torch

        return torch.cuda.current_stream()

    if device_type == "cpu":
        return None

    raise _unsupported(device_type)


def synchronize_device(device_type: DeviceType) -> None:
    """Block until all previously-queued work on the given device completes.

    * ``cuda`` → ``torch.cuda.synchronize()``
    * ``cpu``  → no-op
    """
    if device_type == "cuda":
        import torch

        torch.cuda.synchronize()
        return

    if device_type == "cpu":
        return

    raise _unsupported(device_type)


def stream_context(
    device_type: DeviceType, stream: Optional[Any]
) -> ContextManager[None]:
    """Return a context manager that binds ``stream`` as the current stream
    for the duration of the ``with`` block.

    * ``cuda`` → ``torch.cuda.stream(stream)``
    * ``cpu``  → :func:`contextlib.nullcontext` (no stream concept on CPU)

    The returned object is the backend-native ``StreamContext`` — reusable via
    repeated ``with`` blocks, and cheap to construct once and cache on the caller.
    """
    if device_type == "cuda":
        import torch

        return torch.cuda.stream(stream)

    if device_type == "cpu":
        return contextlib.nullcontext()

    raise _unsupported(device_type)


def create_event(device_type: DeviceType) -> Optional[Any]:
    """Create a fresh event on the given device type.

    * ``cuda`` → ``torch.cuda.Event()``
    * ``cpu``  → ``None`` (CPU has no event concept)

    Only the default no-argument constructor is exposed; timing / blocking-sync
    / IPC flavours are not needed by the engine.
    """
    if device_type == "cuda":
        import torch

        return torch.cuda.Event()

    if device_type == "cpu":
        return None

    raise _unsupported(device_type)


def record_event(
    device_type: DeviceType,
    event: Optional[Any],
    stream: Optional[Any] = None,
) -> None:
    """Record ``event`` on ``stream`` (or the current stream if ``None``).

    * ``cuda`` → ``event.record(stream)``
    * ``cpu``  → no-op (``event`` and ``stream`` are ignored)

    ``event`` and ``stream`` are the objects previously returned by
    :func:`create_event` / :func:`create_stream` / :func:`current_stream` on
    the *same* device type. Cross-device usage is not supported and is not
    validated here — that's PyTorch's job.
    """
    if device_type == "cuda":
        event.record(stream)
        return

    if device_type == "cpu":
        return

    raise _unsupported(device_type)


def empty_device_cache(device_type: DeviceType) -> None:
    """Release cached device memory back to the allocator's free pool.

    * ``cuda`` → ``torch.cuda.empty_cache()``
    * ``cpu``  → no-op

    Only the plain no-argument variant used by
    :meth:`Engine._sync_get_memory` is exposed.
    """
    if device_type == "cuda":
        import torch

        torch.cuda.empty_cache()
        return

    if device_type == "cpu":
        return

    raise _unsupported(device_type)


def reset_peak_memory_stats(device_type: DeviceType) -> None:
    """Reset the "peak memory" counter tracked by the device allocator.

    * ``cuda`` → ``torch.cuda.reset_peak_memory_stats()``
    * ``cpu``  → no-op (CPU has no peak-memory counter)

    Only the no-argument, current-device form is exposed. Today
    :meth:`Engine._sync_get_memory` already bound the process to a single
    device via ``bind_local_device`` before touching this counter, so an
    explicit ``device=`` override would just re-encode that binding.
    """
    if device_type == "cuda":
        import torch

        torch.cuda.reset_peak_memory_stats()
        return

    if device_type == "cpu":
        return

    raise _unsupported(device_type)


def get_free_memory_bytes(device_type: DeviceType, device: Any) -> int:
    """Return the currently-free device memory in bytes.

    * ``cuda`` → ``torch.cuda.mem_get_info(device)[0]`` → ``int``
    * ``cpu``  → raises :class:`NotImplementedError` — CPU has no dedicated
      device memory pool distinct from host RAM.

    ``device`` is forwarded verbatim to the vendor ``mem_get_info`` call. It
    accepts whatever that call accepts today (``int`` index, ``torch.device``,
    ``str``) — this dispatch layer does not narrow the contract.

    Only the free half of the tuple is returned; ``total_bytes`` is not exposed
    on purpose (Engine's memory bookkeeping never needs it).
    """
    if device_type == "cuda":
        import torch

        free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
        return int(free_bytes)

    if device_type == "cpu":
        raise NotImplementedError(
            "free device memory query is not implemented for CPU"
        )

    raise _unsupported(device_type)
