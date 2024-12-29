# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from collections.abc import Iterable
from contextlib import contextmanager
from datetime import timedelta

from redis import Redis

from hypothesis.database import ExampleDatabase
from hypothesis.internal.validation import check_type


class RedisExampleDatabase(ExampleDatabase):
    """Store Hypothesis examples as sets in the given :class:`~redis.Redis` datastore.

    This is particularly useful for shared databases, as per the recipe
    for a :class:`~hypothesis.database.MultiplexedDatabase`.

    .. note::

        If a test has not been run for ``expire_after``, those examples will be allowed
        to expire.  The default time-to-live persists examples between weekly runs.
    """

    def __init__(
        self,
        redis: Redis,
        *,
        expire_after: timedelta = timedelta(days=8),
        key_prefix: bytes = b"hypothesis-example:",
    ):
        check_type(Redis, redis, "redis")
        check_type(timedelta, expire_after, "expire_after")
        check_type(bytes, key_prefix, "key_prefix")
        self.redis = redis
        self._expire_after = expire_after
        self._prefix = key_prefix

    def __repr__(self) -> str:
        return (
            f"RedisExampleDatabase({self.redis!r}, expire_after={self._expire_after!r})"
        )

    @contextmanager
    def _pipeline(self, *reset_expire_keys, transaction=False, auto_execute=True):
        # Context manager to batch updates and expiry reset, reducing TCP roundtrips
        pipe = self.redis.pipeline(transaction=transaction)
        yield pipe
        for key in reset_expire_keys:
            pipe.expire(self._prefix + key, self._expire_after)
        if auto_execute:
            pipe.execute()

    def fetch(self, key: bytes) -> Iterable[bytes]:
        with self._pipeline(key, auto_execute=False) as pipe:
            pipe.smembers(self._prefix + key)
        yield from pipe.execute()[0]

    def save(self, key: bytes, value: bytes) -> None:
        with self._pipeline(key) as pipe:
            pipe.sadd(self._prefix + key, value)

    def delete(self, key: bytes, value: bytes) -> None:
        with self._pipeline(key) as pipe:
            pipe.srem(self._prefix + key, value)

    def move(self, src: bytes, dest: bytes, value: bytes) -> None:
        with self._pipeline(src, dest) as pipe:
            pipe.srem(self._prefix + src, value)
            pipe.sadd(self._prefix + dest, value)
