from __future__ import annotations

from types import SimpleNamespace

from minisgl.core import Batch
from minisgl.scheduler import scheduler as scheduler_module
from minisgl.scheduler.scheduler import Scheduler, _format_batch_observation


def _batch() -> Batch:
    return Batch(
        reqs=[
            SimpleNamespace(uid=7, extend_len=3),
            SimpleNamespace(uid=9, extend_len=5),
        ],
        phase="prefill",
    )


def test_format_batch_observation_is_stable_and_complete() -> None:
    line = _format_batch_observation(
        _batch(),
        pending_requests=4,
        running_requests=1,
    )

    assert line == (
        "SchedulerBatch phase=prefill batch_size=2 token_count=8 "
        "pending_requests=4 running_requests=1 uids=[7,9]"
    )


def test_schedule_logs_the_batch_selected_for_prepare(monkeypatch) -> None:
    batch = _batch()
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.prefill_budget = 32
    scheduler.prefill_manager = SimpleNamespace(
        pending_list=[object(), object()],
        schedule_next_batch=lambda _budget: batch,
    )
    scheduler.decode_manager = SimpleNamespace(
        running_reqs={"running"},
        schedule_next_batch=lambda: None,
    )
    prepared = object()
    scheduler._prepare_batch = lambda selected: prepared if selected is batch else None
    observed: list[str] = []
    monkeypatch.setattr(scheduler_module.logger, "info_rank0", observed.append)

    assert scheduler._schedule_next_batch() is prepared
    assert observed == [
        "SchedulerBatch phase=prefill batch_size=2 token_count=8 "
        "pending_requests=2 running_requests=1 uids=[7,9]"
    ]
