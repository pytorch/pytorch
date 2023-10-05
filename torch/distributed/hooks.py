import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

from torch._C._distributed_c10d import (
    _dequeue_c10d_event,
    _enable_event_collection,
    EventKind,
)

__all__ = [
    "CollectiveStatus",
    "COLLECTIVE_HOOK_TYPE",
    "PG_HOOK_TYPE",
    "register_collective_start_hook",
    "register_collective_end_hook",
    "register_process_group_hook",
]


@dataclass
class CollectiveStatus:
    pg_name: str = "unknown"  # This name matches the one informed in the pgreg cb
    backend: str = "unknown"  # Name of the backend used
    sequence_number: int = -1  # This name matches the one informed in the pgreg cb
    operation: str = "unknown"  # collective name
    timestamp: int = 0  # timestamp to the earliest time we noticed this event
    duration: Optional[float] = None  # value in milliseconds it took executing
    drop_count: int = 0  # number of events dropped following this one


COLLECTIVE_HOOK_TYPE = Callable[[CollectiveStatus], None]
PG_HOOK_TYPE = Callable[[dist.ProcessGroup, str], None]

# This controls the number of internal failures we'll tolerate before giving up
_MAX_INTERNAL_FAILURES = 10

logger = logging.getLogger(__name__)
_cb_thread: Optional[threading.Thread] = None
_start_callbacks: List[COLLECTIVE_HOOK_TYPE] = []
_end_callbacks: List[COLLECTIVE_HOOK_TYPE] = []
_pp_r = -1
_pp_w = -1


def _c10d_pg_hooks_loops():
    internal_failures = 0
    while True:
        # we don't care about the result, this is how we implement notification
        _ = os.read(_pp_r, 1)
        evt: Dict[str, object] = _dequeue_c10d_event()
        try:
            event_kind = evt.pop("event_kind", None)
            if event_kind is None:
                logger.warning(
                    "c10d returned event dictionary %s without 'event_kind' key, cannot dispatch",
                    evt,
                )
                internal_failures += 1
                if internal_failures >= _MAX_INTERNAL_FAILURES:
                    logger.warning(
                        "too many internal c10d failures processing callback loop. stopping"
                    )
                    return
                time.sleep(1)
                continue

            if event_kind == int(EventKind.START):  # type: ignore[call-overload]
                cb_list = _start_callbacks
            elif event_kind == int(EventKind.END):  # type: ignore[call-overload]
                cb_list = _end_callbacks
            else:
                logger.warning(
                    "c10d event %s with invalid 'event_kind' with value %d",
                    evt,
                    event_kind,
                )
                internal_failures += 1
                if internal_failures >= _MAX_INTERNAL_FAILURES:
                    logger.warning(
                        "too many internal c10d failures processing callback loop. stopping"
                    )
                    return
                time.sleep(1)
                continue

            status = CollectiveStatus(**evt)  # type: ignore[arg-type]
            for cb in cb_list:
                try:
                    cb(status)
                except Exception as e:
                    logger.info(
                        "c10d event callback %s with event %s threw exception %s",
                        cb,
                        status,
                        e,
                    )
        except Exception as e:
            # We have to keep processing otherwise the queue will grown infinitely large
            logger.warning(
                "c10d callback thread when processing event %s raised exception %s.",
                evt,
                e,
            )
            internal_failures += 1
            if internal_failures >= _MAX_INTERNAL_FAILURES:
                logger.warning(
                    "too many internal c10d failures processing callback loop. stopping"
                )
                return

            # Sleep for a second to avoid hogging the GIL in case of a persistent failure
            time.sleep(1)


def _lazy_init():
    global _cb_thread
    if _cb_thread is not None:
        return
    global _pp_r
    global _pp_w
    _pp_r, _pp_w = os.pipe()
    _enable_event_collection(_pp_w)
    c10d._enable_collectives_timing()
    _cb_thread = threading.Thread(target=_c10d_pg_hooks_loops, daemon=True)
    _cb_thread.start()
    logger.info("c10d::hooks thread enabled")


def register_collective_start_hook(hook: COLLECTIVE_HOOK_TYPE) -> None:
    """
    Register a hook that is called every time a collective starts.

    The hook is invoked on a background thread.
    Exceptions raised by the callback are ignored and non-fatal.
    """
    _start_callbacks.append(hook)
    _lazy_init()


def register_collective_end_hook(hook: COLLECTIVE_HOOK_TYPE) -> None:
    """
    Register a hook that is called every time a collective finishes.


    The hook is invoked on a background thread.
    Exceptions raised by the callback are ignored and non-fatal.
    """
    _end_callbacks.append(hook)
    _lazy_init()


def register_process_group_hook(hook: PG_HOOK_TYPE) -> None:
    """
    Register a hook that is called every time a process group is created on this rank.

    This hook is only invoked if the current rank is part of the PG being created.
    The pg_name is unique to the whole cluster and should be treated as an opaque identified subject to change.

    The hook is invoked on a background thread.
    Exceptions raised by the callback are ignored and non-fatal.
    """
    c10d._register_creation_hook(hook)
