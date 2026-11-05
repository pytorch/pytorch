# Cache compiled kernels and shape-invariant host decisions by operand signature. None is a
# cached decline, avoiding repeated plan derivation on eager calls.

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable


def cached_plan(cache: dict, key, build: Callable, *, op: str | None = None):
    """Memoize build() by key, including None declines. When op is set, instrument
    the miss-only build so every real compile emits TLParse data.
    """
    plan = cache.get(key, _MISS)
    if plan is _MISS:
        if op is not None:
            from torch._native.instrumentation import instrument_cutedsl_compile

            # build is a miss-only closure without cache_info, so mark compilation explicitly.
            plan = instrument_cutedsl_compile(
                op, key_fn=lambda: str(key), compiled=True
            )(build)()
        else:
            plan = build()
        cache[key] = plan
    return plan


_MISS = object()  # sentinel so a cached None (declined) is distinct from a miss


class LRUCache(OrderedDict):
    """A dict that keeps its `maxsize` most recently used entries. Only get and item
    assignment hold its lock, so only they are safe to call concurrently (free-threaded
    builds included); other dict methods are not."""

    def __init__(self, maxsize: int) -> None:
        super().__init__()
        self.maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, key: Any, default: Any = None) -> Any:
        with self._lock:
            if key not in self:
                return default
            self.move_to_end(key)
            return super().__getitem__(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        with self._lock:
            super().__setitem__(key, value)
            self.move_to_end(key)
            if len(self) > self.maxsize:
                self.popitem(last=False)
