# Cache compiled kernels and shape-invariant host decisions by operand signature. None is a
# cached decline, avoiding repeated plan derivation on eager calls.

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


def cached_plan(cache: dict | LRUCache, key, build: Callable, *, op: str | None = None):
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


class LRUCache:
    """A mapping that keeps its `maxsize` most recently used entries. Every method holds
    its lock, so concurrent use is safe (free-threaded builds included). Only `get` and
    item assignment update recency; iteration and `items()` return snapshots.
    """

    def __init__(self, maxsize: int) -> None:
        self.maxsize = maxsize
        self._d: OrderedDict[Any, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Any, default: Any = None) -> Any:
        with self._lock:
            try:
                value = self._d[key]
            except KeyError:
                return default
            self._d.move_to_end(key)
            return value

    def __setitem__(self, key: Any, value: Any) -> None:
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            if len(self._d) > self.maxsize:
                self._d.popitem(last=False)

    def __contains__(self, key: Any) -> bool:
        with self._lock:
            return key in self._d

    def __len__(self) -> int:
        with self._lock:
            return len(self._d)

    def __iter__(self) -> Iterator[Any]:
        with self._lock:
            return iter(list(self._d))

    def items(self) -> list[tuple[Any, Any]]:
        with self._lock:
            return list(self._d.items())

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
