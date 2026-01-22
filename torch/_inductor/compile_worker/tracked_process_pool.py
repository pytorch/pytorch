import atexit
import concurrent
import dataclasses
import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from time import time
from typing import Any, Optional, TypeVar
from typing_extensions import ParamSpec

# _thread_safe_fork is needed because the subprocesses in the pool can read
# justknobs, e.g., in the Triton compiler. For internal, the import installs
# functionality to destroy singletons before forking and re-enable them after.
import torch._thread_safe_fork  # noqa: F401


_P = ParamSpec("_P")
_R = TypeVar("_R")


log = logging.getLogger(__name__)


@dataclass
class _QueueStats:
    # Mapping from id(future) -> start time
    pending: dict[int, float] = dataclasses.field(default_factory=dict)
    timing: list[float] = dataclasses.field(default_factory=list)
    enqueue_count: int = 0
    dequeue_count: int = 0
    max_queue_depth: int = 0
    pool_count: int = 0


# The queue statistics tracked by TrackedProcessPoolExecutor. Always grab
# _queue_stats_lock before touching.
_queue_stats = _QueueStats()
_queue_stats_lock = threading.Lock()


class TrackedProcessPoolExecutor(ProcessPoolExecutor):
    def __init__(
        self,
        max_workers: Optional[int] = None,
        mp_context: Optional[BaseContext] = None,
        initializer: Optional[Callable[[], object]] = None,
    ) -> None:
        with _queue_stats_lock:
            _queue_stats.pool_count += 1
        super().__init__(max_workers, mp_context, initializer)

    def _record_dequeue(self, f: Future[Any]) -> None:
        now = time()
        with _queue_stats_lock:
            stats = _queue_stats
            if (start_time := stats.pending.pop(id(f), None)) is None:
                return
            stats.dequeue_count += 1
            duration = now - start_time
            stats.timing.append(duration)

    def _record_enqueue(self, f: Future[Any]) -> None:
        # Monkeypatch the set_running_or_notify_cancel so we can track when the Future moves out of PENDING.
        saved_running_or_notify_cancel = f.set_running_or_notify_cancel

        def set_running_or_notify_cancel() -> Any:
            self._record_dequeue(f)
            return saved_running_or_notify_cancel()

        now = time()
        with _queue_stats_lock:
            stats = _queue_stats
            stats.pending[id(f)] = now
            stats.enqueue_count += 1
            stats.max_queue_depth = max(stats.max_queue_depth, len(stats.pending))
            f.set_running_or_notify_cancel = set_running_or_notify_cancel  # type: ignore[method-assign]

        if f._state != concurrent.futures._base.PENDING:
            self._record_dequeue(f)

    def submit(
        self, fn: Callable[_P, _R], /, *args: _P.args, **kwargs: _P.kwargs
    ) -> Future[_R]:
        # pyrefly: ignore [bad-argument-type]
        f = super().submit(fn, *args, **kwargs)
        self._record_enqueue(f)
        return f


@atexit.register
def _queue_stats_report() -> None:
    stats = _queue_stats
    if stats.pool_count == 0:
        return

    timing = stats.timing
    timing.sort()

    log.info("AsyncCompile Metrics:")
    log.info("  Pools %s", stats.pool_count)
    log.info(
        "  Items %d enqueued / %d dequeued", stats.enqueue_count, stats.dequeue_count
    )
    log.info("  Max Queue Depth: %d", stats.max_queue_depth)
    n = len(timing)
    if n > 0:
        log.info("  Longest queue time: %0.2fs", timing[-1])
        log.info("  P50: %0.2fs", timing[n // 2])
        if n >= 20:
            log.info("  P95: %0.2fs", timing[n * 95 // 100])
