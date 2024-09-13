import os
from abc import abstractmethod
from typing import Optional


class RemoteCacheBackend:
    """
    A backend implementation for accessing a remote/distributed cache.
    """

    def __init__(self, cache_id: str) -> None:
        pass

    @abstractmethod
    def get(self, key: str) -> Optional[object]:
        pass

    @abstractmethod
    def put(self, key: str, data: bytes) -> None:
        pass


class RedisRemoteCacheBackend(RemoteCacheBackend):
    """
    A Redis implementation of a remote/distributed cache.
    """

    def __init__(self, cache_id: str) -> None:
        import redis

        self._key_fmt = f"pt2:{cache_id}:{{key}}"
        self._redis = redis.Redis(
            host=os.environ.get("TORCHINDUCTOR_REDIS_HOST", "localhost"),
            port=int(os.environ.get("TORCHINDUCTOR_REDIS_PORT", 6379)),
        )

    def _get_key(self, key: str) -> str:
        return self._key_fmt.format(key=key)

    def get(self, key: str) -> Optional[bytes]:
        value = self._redis.get(self._get_key(key))
        # In theory redis.get() can return an Awaitable as well...
        assert value is None or isinstance(value, bytes)
        return value

    def put(self, key: str, data: bytes) -> None:
        self._redis.set(self._get_key(key), data)
