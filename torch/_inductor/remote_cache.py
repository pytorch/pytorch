from __future__ import annotations

import atexit
import collections
import dataclasses
import functools
import json
import logging
import os
import sys
import typing
from abc import abstractmethod
from typing import Any, Callable, Generic, Optional, TypeVar, Union
from typing_extensions import override, TypeAlias

from torch._dynamo.utils import dynamo_timed
from torch._inductor import config
from torch.monitor import _WaitCounter


try:
    import redis
except ImportError:
    redis = None  # type: ignore[assignment]


log = logging.getLogger(__name__)


if config.is_fbcode():
    from rfe.scubadata.scubadata_py3 import (  # type: ignore[import-not-found]
        Sample as Sample_,
    )

    Sample: TypeAlias = Sample_
else:
    Sample: TypeAlias = type[object]  # type: ignore[misc,no-redef]


_T = TypeVar("_T")
_U = TypeVar("_U")


remote_fx_cache_get_timed = functools.partial(
    dynamo_timed,
    "FbRemoteFxGraphCache.get",
    phase_name="remote_fx_graph_cache_get",
    log_pt2_compile_event=False,
    dynamo_compile_column_us="remote_fx_graph_cache_get_time_us",
    log_waitcounter=True,
)
remote_fx_cache_put_timed = functools.partial(
    dynamo_timed,
    "FbRemoteFxGraphCache.put",
    phase_name="remote_fx_graph_cache_put",
    log_pt2_compile_event=False,
    dynamo_compile_column_us="remote_fx_graph_cache_put_time_us",
    log_waitcounter=True,
)


class RemoteCacheBackend(Generic[_T]):
    """
    A backend implementation for accessing a remote/distributed cache.  Only
    works with bytes in/out.  For structured data use a RemoteCache.
    """

    def __init__(self) -> None:
        self._name = f"backend:{type(self).__name__}"

    @abstractmethod
    def _get(self, key: str) -> Optional[_T]:
        pass

    @abstractmethod
    def _put(self, key: str, data: _T) -> None:
        pass

    def get(self, key: str) -> Optional[_T]:
        try:
            value = self._get(key)
            cache_stats.get(self._name, value)
        except Exception:
            cache_stats.exception(self._name)
            raise
        return value

    def put(self, key: str, data: _T) -> None:
        try:
            self._put(key, data)
            cache_stats.put(self._name)
        except Exception:
            cache_stats.exception(self._name)
            raise


# Serde that encodes from _T to _U and decodes from _U to _T.
class RemoteCacheSerde(Generic[_T, _U]):
    @abstractmethod
    def encode(self, data: _T) -> _U:
        pass

    @abstractmethod
    def decode(self, data: _U) -> _T:
        pass


JsonDataTy = Optional[
    Union[int, float, str, bool, dict[str, "JsonDataTy"], list["JsonDataTy"]]
]


class RemoteCacheJsonSerde(RemoteCacheSerde[JsonDataTy, bytes]):
    def encode(self, data: JsonDataTy) -> bytes:
        return bytes(json.dumps(data), "ascii")

    def decode(self, data: bytes) -> JsonDataTy:
        return json.loads(data)


class RemoteCachePassthroughSerde(RemoteCacheSerde[_T, _T]):
    def encode(self, data: _T) -> _T:
        return data

    def decode(self, data: _T) -> _T:
        return data


# This class is the top of a RemoteCache. A RemoteCache is fundamentally made of
# three parts:
#
# 1. The controller (this class).
# 2. A serializer/deserializer (instance of RemoteCacheSerde).
# 3. A backend (instance of RemoteCacheBackend).
#
# To write (`put`), the RemoteCache takes data, uses the RemoteCacheSerde to
# convert it for the backend and passes it to the backend.
#
# Conversely when reading (`get`), the RemoteCache takes data from the backend,
# uses the RemoteCacheSerde to convert it and returns it.
#
# The RemoteCacheBackend is generic on _U - which is the type of data the
# backend can directly cache (usually `bytes`).
#
# The RemoteCacheSerde is responsible for converting between _T (the type of
# data the RemoteCache accepts in `put` and returns in `get`) and _U.
#
# When instantiating a RemoteCache you should override, not directly create a
# RemoteCache. The reason is that when logging cache use (`TORCH_LOGS=cache`) we
# use the concrete type of the RemoteCache as the reported cache. See
# RemoteFxGraphCache below as an example.
class RemoteCache(Generic[_T]):
    backend_override_cls: Optional[Callable[[], RemoteCacheBackend[Any]]] = None

    def __init__(
        self, backend: RemoteCacheBackend[_U], serde: RemoteCacheSerde[_T, _U]
    ) -> None:
        # Support for testing to mock out the backend on a class-by-class basis.
        if (override_cls := self.__class__.backend_override_cls) is not None:
            self.backend = override_cls()
        else:
            self.backend = backend
        self.serde = serde

    # See if the cache contains `key`. Returns `None` if the value is not
    # present in the cache.
    def get(self, key: str) -> Optional[_T]:
        with _WaitCounter("pytorch.remote_cache.get").guard():
            sample = self._create_sample()
            try:
                result = self._get(key, sample)
                cache_stats.get(type(self).__name__, result)
            except Exception:
                cache_stats.exception(type(self).__name__)
                raise
            self._log_sample(sample)
            return result

    # Add `value` to the cache with the key `key`. Note that `None` is not a
    # valid value even if _T supports it (because you can't tell the difference
    # between `None` and a missing cache entry).
    def put(self, key: str, value: _T) -> None:
        with _WaitCounter("pytorch.remote_cache.put").guard():
            assert value is not None
            sample = self._create_sample()
            try:
                self._put(key, value, sample)
                cache_stats.put(type(self).__name__)
            except Exception:
                cache_stats.exception(type(self).__name__)
                raise
            self._log_sample(sample)

    # Used to convert data from the cache into structured data.
    def _decode(self, data: _U, sample: Optional[Sample]) -> _T:  # type: ignore[override]
        return self.serde.decode(data)  # type: ignore[arg-type]

    # Used to convert structured data into data for the cache.
    def _encode(self, value: _T, sample: Optional[Sample]) -> object:  # returns _U
        return self.serde.encode(value)

    # Get structured data from the cache.
    # Separate from `get` so that it can be overridden.
    def _get(self, key: str, sample: Optional[Sample]) -> Optional[_T]:
        if data := self._backend_get(key):
            return self._decode(data, sample)
        return None

    # Get unstructured data from the cache.
    # Separate from `get` so that it can be overridden.
    # Returns _U - but we aren't actually generic on _U
    def _backend_get(self, key: str) -> object:
        return self.backend.get(key)

    # Put structured data into the cache.
    # Separate from `put` so that it can be overridden.
    def _put(self, key: str, value: _T, sample: Optional[Sample]) -> None:
        data = self._encode(value, sample)
        self._backend_put(key, data)

    # Put unstructured data into the cache.
    # Separate from `put` so that it can be overridden.
    # Takes data: _U - but we aren't actually generic on _U
    def _backend_put(self, key: str, data: object) -> None:
        self.backend.put(key, data)

    # Create a logging Sample - used with internal loggers to monitor cache
    # effectiveness.
    def _create_sample(self) -> Optional[Sample]:
        return None

    # Write the logging Sample to the logger.
    def _log_sample(self, sample: Optional[Sample]) -> None:
        pass


class RedisRemoteCacheBackend(RemoteCacheBackend[bytes]):
    """
    A Redis implementation of a remote/distributed cache.
    """

    _redis: Optional[redis.Redis] = None

    def __init__(self, cache_id: str) -> None:
        super().__init__()
        if not redis:
            raise RuntimeError("redis not available but required for remote cache")

        if "TORCHINDUCTOR_REDIS_URL" in os.environ:
            self._redis = redis.Redis.from_url(os.environ["TORCHINDUCTOR_REDIS_URL"])
        else:
            self._redis = redis.Redis(
                host=os.environ.get("TORCHINDUCTOR_REDIS_HOST", "localhost"),
                port=int(os.environ.get("TORCHINDUCTOR_REDIS_PORT", 6379)),
            )

    @override
    def _get(self, key: str) -> Optional[bytes]:
        if not self._redis:
            # Either redis wasn't found or we already had some trouble...
            return None

        try:
            value = self._redis.get(key)
        except redis.exceptions.ConnectionError:
            # Redis is lazy and doesn't actually attempt to connect until the
            # first use. Mark is as unavailable now.
            self._redis = None
            return None

        # In theory redis.get() can return an Awaitable as well...
        assert value is None or isinstance(value, bytes)
        return value

    @override
    def _put(self, key: str, data: bytes) -> None:
        if not self._redis:
            # Either redis wasn't found or we already had some trouble...
            return

        try:
            self._redis.set(key, data)
        except redis.exceptions.ConnectionError:
            # Redis is lazy and doesn't actually attempt to connect until the
            # first use. Mark is as unavailable now.
            self._redis = None


class RedisRemoteCache(RemoteCache[JsonDataTy]):
    def __init__(self, cache_id: str) -> None:
        # Special test handling: If we're just going to override the backend
        # anyway don't require redis
        if self.__class__.backend_override_cls:
            # This is totally bogus but it works for now...
            backend = typing.cast(RemoteCacheBackend[bytes], None)
        else:
            backend = RedisRemoteCacheBackend(cache_id)
        serde = RemoteCacheJsonSerde()
        super().__init__(backend, serde)
        version = 1  # consistency between various types of keys
        self._key_fmt = f"pt2:{cache_id}::{{key}}:c{version}"

    def _get_key(self, key: str) -> str:
        return self._key_fmt.format(key=key)

    @override
    def _get(self, key: str, sample: Optional[Sample]) -> Optional[JsonDataTy]:
        key = self._get_key(key)
        return super()._get(key, sample)

    @override
    def _put(self, key: str, value: JsonDataTy, sample: Optional[Sample]) -> None:
        key = self._get_key(key)
        super()._put(key, value, sample)


class RemoteAutotuneCache(RedisRemoteCache):
    pass


class RemoteBundledAutotuneCache(RedisRemoteCache):
    pass


class RemoteFxGraphCache(RedisRemoteCache):
    pass


class RemoteAOTAutogradCache(RedisRemoteCache):
    pass


class RemoteDynamoPGOCache(RedisRemoteCache):
    pass


def create_cache(
    key: str,
    is_fbcode: bool,
    fb_cache_cls: str,
    oss_cache_cls: str,
) -> Optional[RemoteCache[JsonDataTy]]:
    try:
        if is_fbcode:
            import torch._inductor.fb.remote_cache

            cache_cls = getattr(torch._inductor.fb.remote_cache, fb_cache_cls)
            return cache_cls(key)
        else:
            this_module = sys.modules[__name__]

            cache_cls = getattr(this_module, oss_cache_cls)
            return cache_cls(key)

    except Exception:
        log.warning("Unable to create a remote cache", exc_info=True)
        return None


# Some simple stat capture
@dataclasses.dataclass
class _CacheStat:
    miss: int = 0
    hit: int = 0
    put: int = 0
    exception: int = 0

    def __str__(self) -> str:
        return f"{{hit: {self.hit}, miss: {self.miss}, put: {self.put}, exception: {self.exception}}}"


class _CacheStats:
    _stats: dict[str, _CacheStat]

    def __init__(self) -> None:
        self._stats = collections.defaultdict(_CacheStat)

    def miss(self, name: str, count: int = 1) -> None:
        self._stats[name].miss += count

    def hit(self, name: str, count: int = 1) -> None:
        self._stats[name].hit += count

    def get(self, name: str, value: Optional[object]) -> None:
        if value is None:
            self.miss(name)
        else:
            self.hit(name)

    def put(self, name: str, count: int = 1) -> None:
        self._stats[name].put += count

    def exception(self, name: str, count: int = 1) -> None:
        self._stats[name].exception += count


cache_stats = _CacheStats()


@atexit.register
def dump_cache_stats() -> None:
    if not log.isEnabledFor(logging.INFO):
        return

    import io

    out = io.StringIO()

    if not cache_stats._stats:
        print(" None", file=out)
    else:
        print(file=out)
        for k, v in sorted(cache_stats._stats.items()):
            print(f"  {k}: {v}", file=out)

    log.info("Cache Metrics:%s", out.getvalue())
