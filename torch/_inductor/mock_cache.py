# mypy: ignore-errors

from __future__ import annotations

import contextlib
import dataclasses
import sys
import threading
from typing import Any, Callable, Dict, Optional, Type, TYPE_CHECKING
from typing_extensions import override, Self
from unittest.mock import patch

from torch._inductor import config
from torch._inductor.remote_cache import RemoteCacheBackend


if TYPE_CHECKING:
    from types import TracebackType


@dataclasses.dataclass
class Stats:
    num_put: int = 0
    num_get_hit: int = 0
    num_get_miss: int = 0

    def __iadd__(self, other: Stats) -> Self:
        self.num_put += other.num_put
        self.num_get_hit += other.num_get_hit
        self.num_get_miss += other.num_get_miss
        return self

    def reset(self) -> None:
        self.num_put = 0
        self.num_get_hit = 0
        self.num_get_miss = 0

    def __str__(self) -> str:
        return "".join(
            (
                f"puts: {self.num_put}, ",
                f"misses: {self.num_get_miss}, ",
                f"hits: {self.num_get_hit}, ",
            )
        )

    def __eq__(self, other: object) -> bool:
        # Dataclass's default __eq__ checks that the types are the same so can't
        # be used with _GlobalItemStats.
        return (
            isinstance(other, (Stats, _GlobalItemStats))
            and self.num_put == other.num_put
            and self.num_get_hit == other.num_get_hit
            and self.num_get_miss == other.num_get_miss
        )


class _GlobalItemStats(Stats):
    cache: Dict[str, object]

    def __init__(self) -> None:
        super().__init__()
        self.cache = {}

    def reset(self) -> None:
        super().reset()
        self.cache = {}


# The cache states are thread-local so if we're running multiple tests at once
# they won't cross contaminate. However - it needs to be "global" because we
# allow code to create new cache clients which refer to the same cache (because
# it's a remote cache).


class _GlobalStats(threading.local):
    def __init__(self) -> None:
        self.autotune_local = _GlobalItemStats()
        self.autotune_remote = _GlobalItemStats()
        self.bundled_autotune = _GlobalItemStats()
        self.fx_graph = _GlobalItemStats()
        self.triton = _GlobalItemStats()
        self.aot_autograd = _GlobalItemStats()
        self.dynamo_pgo = _GlobalItemStats()

    def reset(self) -> None:
        self.autotune_local.reset()
        self.autotune_remote.reset()
        self.bundled_autotune.reset()
        self.fx_graph.reset()
        self.triton.reset()
        self.aot_autograd.reset()
        self.dynamo_pgo.reset()

    def get_stat(self, name: str) -> _GlobalItemStats:
        return getattr(self, name)

    def report(self):
        subs = (
            ("autotune_local", self.autotune_local),
            ("autotune_remote", self.autotune_remote),
            ("bundled_autotune", self.bundled_autotune),
            ("fx_graph", self.fx_graph),
            ("triton", self.triton),
            ("aot_autograd", self.aot_autograd),
            ("dynamo_pgo", self.dynamo_pgo),
        )

        print("Cache Stats:", file=sys.stderr)
        for name, sub in subs:
            print(f"  {name}: {sub}", file=sys.stderr)

        print("Cache Entries:", file=sys.stderr)
        for name, sub in subs:
            if sub.cache:
                print(f"  {name}:", file=sys.stderr)
                for k, v in sorted(sub.cache.items()):
                    v = repr(v)
                    if len(v) > 100:
                        v = v[:100] + "..."
                    print(f"    {k!r}: {v}", file=sys.stderr)


global_stats = _GlobalStats()


class MockBackend(RemoteCacheBackend[Any]):
    def __init__(self, name: str) -> None:
        self._name = name

    @staticmethod
    def with_name(name: str) -> Callable[[], MockBackend]:
        def wrapper() -> MockBackend:
            return MockBackend(name)

        return wrapper

    @override
    def _get(self, key: str) -> Optional[Any]:
        stat = global_stats.get_stat(self._name)
        if key in stat.cache:
            stat += Stats(num_get_hit=1)
            return stat.cache.get(key)
        else:
            stat += Stats(num_get_miss=1)
            return None

    @override
    def _put(self, key: str, data: Any) -> None:
        stat = global_stats.get_stat(self._name)
        stat += Stats(num_put=1)
        stat.cache[key] = data


# List of configs for each cache
_CACHE_CONFIG_EN = (
    "fx_graph_cache",
    "fx_graph_remote_cache",
    "autotune_local_cache",
    "autotune_remote_cache",
    "bundled_autotune_remote_cache",
)


class PatchCaches(contextlib.AbstractContextManager):
    @classmethod
    def setUp(cls):
        # If this test is using PatchCaches then disable all the caches by
        # default, letting the tests turn them on explicitly. This is because
        # tests using PatchCaches will often want to check stats explicitly.
        cls._savedCacheState = {}
        for name in _CACHE_CONFIG_EN:
            if hasattr(config, name):
                cls._savedCacheState[name] = getattr(config, name)
            setattr(config, name, False)

    @classmethod
    def tearDown(cls):
        # Restore cache defaults
        for name in _CACHE_CONFIG_EN:
            delattr(config, name)
            if name in cls._savedCacheState:
                setattr(config, name, cls._savedCacheState[name])

    def __init__(self) -> None:
        self._stack = contextlib.ExitStack()

    def __enter__(self) -> Self:
        global_stats.reset()
        self._stack.__enter__()

        ctx = patch(
            "torch._inductor.runtime.autotune_cache.LocalAutotuneCache.backend_override_cls",
            MockBackend.with_name("autotune_local"),
        )
        self._stack.enter_context(ctx)

        ctx = patch(
            "torch._inductor.remote_cache.RemoteAutotuneCache.backend_override_cls",
            MockBackend.with_name("autotune_remote"),
        )
        self._stack.enter_context(ctx)

        ctx = patch(
            "torch._inductor.remote_cache.RemoteBundledAutotuneCache.backend_override_cls",
            MockBackend.with_name("bundled_autotune"),
        )
        self._stack.enter_context(ctx)

        ctx = patch(
            "torch._inductor.remote_cache.RemoteFxGraphCache.backend_override_cls",
            MockBackend.with_name("fx_graph"),
        )
        self._stack.enter_context(ctx)

        ctx = patch(
            "torch._inductor.remote_cache.RemoteAOTAutogradCache.backend_override_cls",
            MockBackend.with_name("aot_autograd"),
        )
        self._stack.enter_context(ctx)

        ctx = patch(
            "torch._inductor.remote_cache.RemoteDynamoPGOCache.backend_override_cls",
            MockBackend.with_name("dynamo_pgo"),
        )
        self._stack.enter_context(ctx)

        if config.is_fbcode():
            ctx = patch(
                "torch._inductor.fb.remote_cache.FbRemoteAutotuneCache.backend_override_cls",
                MockBackend.with_name("autotune_remote"),
            )
            self._stack.enter_context(ctx)

            ctx = patch(
                "torch._inductor.fb.remote_cache.FbRemoteBundledAutotuneCache.backend_override_cls",
                MockBackend.with_name("bundled_autotune"),
            )
            self._stack.enter_context(ctx)

            ctx = patch(
                "torch._inductor.fb.remote_cache.FbRemoteFxGraphCache.backend_override_cls",
                MockBackend.with_name("fx_graph"),
            )
            self._stack.enter_context(ctx)

            ctx = patch(
                "triton.fb.fb_memcache.FbMemcacheRemoteKernelCache.backend_override_cls",
                MockBackend.with_name("triton"),
            )
            self._stack.enter_context(ctx)

            ctx = patch(
                "torch._inductor.fb.remote_cache.FbRemoteAOTAutogradCache.backend_override_cls",
                MockBackend.with_name("aot_autograd"),
            )
            self._stack.enter_context(ctx)

            ctx = patch(
                "torch._inductor.fb.remote_cache.FbRemoteDynamoPGOCache.backend_override_cls",
                MockBackend.with_name("dynamo_pgo"),
            )
            self._stack.enter_context(ctx)

        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self._stack.__exit__(exc_type, exc_value, traceback)
