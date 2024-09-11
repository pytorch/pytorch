# Owner(s): ["module: inductor"]
from __future__ import annotations

import contextlib
import dataclasses
import sys
import threading
from typing import Any, Callable, Dict, Generator, Optional, Type, TYPE_CHECKING
from typing_extensions import override, Self
from unittest.mock import patch

import torch
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


# The cache states are thread-local so if we're running multiple tests at once
# they won't cross contaminate. However - it needs to be "global" because we
# allow code to create new cache clients which refer to the same cache (because
# it's a remote cache).


class _GlobalStats(Stats, threading.local):
    def __init__(self) -> None:
        self.autotune = Stats()
        self.fx_graph = Stats()
        self.triton = Stats()

    def reset(self) -> None:
        self.autotune.reset()
        self.fx_graph.reset()
        self.triton.reset()

    def update(self, name: str, delta: Stats) -> None:
        stat = getattr(self, name)
        stat += delta

    def report(self):
        print("Cache Stats:", file=sys.stderr)
        print(f"  autotune: {self.autotune}", file=sys.stderr)
        print(f"  fx_graph: {self.fx_graph}", file=sys.stderr)
        print(f"  triton:   {self.triton}", file=sys.stderr)


global_stats = _GlobalStats()


class MockBackend(RemoteCacheBackend[Any]):
    def __init__(self, name: str, cache: Dict[str, object]) -> None:
        self._cache = cache
        self._name = name

    @staticmethod
    def with_name(name: str) -> Callable[[], MockBackend]:
        cache = {}

        def wrapper() -> MockBackend:
            return MockBackend(name, cache)

        return wrapper

    @override
    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            global_stats.update(self._name, Stats(num_get_hit=1))
            return self._cache.get(key)
        else:
            global_stats.update(self._name, Stats(num_get_miss=1))
            return None

    @override
    def put(self, key: str, data: Any) -> None:
        global_stats.update(self._name, Stats(num_put=1))
        self._cache[key] = data


# List of configs for each cache
_CACHE_CONFIG_EN = (
    "fx_graph_cache",
    "fx_graph_remote_cache",
    "autotune_local_cache",
    "autotune_remote_cache",
    # "bundled_autotune_cache",
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
            "torch._inductor.remote_cache.RemoteAutotuneCache.backend_override_cls",
            MockBackend.with_name("autotune"),
        )
        self._stack.enter_context(ctx)

        ctx = patch(
            "torch._inductor.remote_cache.RemoteFxGraphCache.backend_override_cls",
            MockBackend.with_name("fx_graph"),
        )
        self._stack.enter_context(ctx)

        if config.is_fbcode():
            ctx = patch(
                "torch._inductor.fb.remote_cache.FbRemoteAutotuneCache.backend_override_cls",
                MockBackend.with_name("autotune"),
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

        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self._stack.__exit__(exc_type, exc_value, traceback)


@contextlib.contextmanager
def patch_fbcode(state: bool) -> Generator[None, None, None]:
    if hasattr(torch.version, "git_version"):
        # Currently non-fbcode
        if state:
            old = torch.version.git_version
            delattr(torch.version, "git_version")
            try:
                yield
            finally:
                torch.version.git_version = old
        else:
            yield
    else:
        # Currently fbcode
        if state:
            yield
        else:
            torch.version.git_version = "12345+"
            try:
                yield
            finally:
                delattr(torch.version, "git_version")
