from __future__ import annotations

import json
import os
import typing
from abc import abstractmethod
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar, Union
from typing_extensions import override, TypeAlias

from torch._inductor import config


try:
    import redis
except ImportError:
    redis = None  # type: ignore[assignment]


if config.is_fbcode():
    from rfe.scubadata.scubadata_py3 import (  # type: ignore[import-not-found]
        Sample as Sample_,
    )

    Sample: TypeAlias = Sample_
else:
    Sample: TypeAlias = Type[object]  # type: ignore[misc,no-redef]


_T = TypeVar("_T")
_U = TypeVar("_U")


class RemoteCacheBackend(Generic[_T]):
    """
    A backend implementation for accessing a remote/distributed cache.  Only
    works with bytes in/out.  For structured data use a RemoteCache.
    """

    @abstractmethod
    def get(self, key: str) -> Optional[_T]:
        pass

    @abstractmethod
    def put(self, key: str, data: _T) -> None:
        pass


# Serde that encodes from _T to _U and decodes from _U to _T.
class RemoteCacheSerde(Generic[_T, _U]):
    @abstractmethod
    def encode(self, data: _T) -> _U:
        pass

    @abstractmethod
    def decode(self, data: _U) -> _T:
        pass


JsonDataTy = Optional[
    Union[int, float, str, bool, Dict[str, "JsonDataTy"], List["JsonDataTy"]]
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


class RemoteCache(Generic[_T]):
    backend_override_cls: Optional[Callable[[], RemoteCacheBackend[Any]]] = None

    def __init__(
        self, backend: RemoteCacheBackend[_U], serde: RemoteCacheSerde[_T, _U]
    ) -> None:
        # Support for testing.
        if (override_cls := self.__class__.backend_override_cls) is not None:
            self.backend = override_cls()
        else:
            self.backend = backend
        self.serde = serde

    def get(self, key: str) -> Optional[_T]:
        sample = self._create_sample()
        result = self._get(key, sample)
        self._log_sample(sample)
        return result

    def put(self, key: str, value: _T) -> None:
        sample = self._create_sample()
        self._put(key, value, sample)
        self._log_sample(sample)

    def _decode(self, data: _U, sample: Optional[Sample]) -> _T:
        return self.serde.decode(data)

    def _encode(self, value: _T, sample: Optional[Sample]) -> Any:  # returns _U
        return self.serde.encode(value)

    def _get(self, key: str, sample: Optional[Sample]) -> Optional[_T]:
        if data := self.backend.get(key):
            return self._decode(data, sample)
        return None

    def _put(self, key: str, value: _T, sample: Optional[Sample]) -> None:
        data = self._encode(value, sample)
        self.backend.put(key, data)

    def _create_sample(self) -> Optional[Sample]:
        return None

    def _log_sample(self, sample: Optional[Sample]) -> None:
        pass


class RedisRemoteCacheBackend(RemoteCacheBackend[bytes]):
    """
    A Redis implementation of a remote/distributed cache.
    """

    _key_fmt: str
    _redis: Optional[redis.Redis] = None

    def __init__(self, cache_id: str) -> None:
        if not redis:
            # We had trouble importing redis - just skip init.
            return

        self._key_fmt = f"pt2:{cache_id}:{{key}}"
        self._redis = redis.Redis(
            host=os.environ.get("TORCHINDUCTOR_REDIS_HOST", "localhost"),
            port=int(os.environ.get("TORCHINDUCTOR_REDIS_PORT", 6379)),
        )

    def __get_key(self, key: str) -> str:
        return self._key_fmt.format(key=key)

    @override
    def get(self, key: str) -> Optional[bytes]:
        if not self._redis:
            # Either redis wasn't found or we already had some trouble...
            return None

        try:
            value = self._redis.get(self.__get_key(key))
        except redis.exceptions.ConnectionError:
            # Redis is lazy and doesn't actually attempt to connect until the
            # first use. Mark is as unavailable now.
            self._redis = None
            return None

        # In theory redis.get() can return an Awaitable as well...
        assert value is None or isinstance(value, bytes)
        return value

    @override
    def put(self, key: str, data: bytes) -> None:
        if not self._redis:
            # Either redis wasn't found or we already had some trouble...
            return

        try:
            self._redis.set(self.__get_key(key), data)
        except redis.exceptions.ConnectionError:
            # Redis is lazy and doesn't actually attempt to connect until the
            # first use. Mark is as unavailable now.
            self._redis = None


class RedisRemoteCache(RemoteCache[JsonDataTy]):
    def __init__(self, key: str) -> None:
        # Special test handling: If we're just going to override the backend
        # anyway don't require redis
        if self.__class__.backend_override_cls:
            # This is totally bogus but it works for now...
            backend = typing.cast(RemoteCacheBackend[bytes], None)
        else:
            backend = RedisRemoteCacheBackend(key)
        serde = RemoteCacheJsonSerde()
        super().__init__(backend, serde)


class RemoteAutotuneCache(RedisRemoteCache):
    pass


class RemoteFxGraphCache(RedisRemoteCache):
    pass
