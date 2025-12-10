# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import base64
import json
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import timedelta
from typing import Any

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
        listener_channel: str = "hypothesis-changes",
    ):
        super().__init__()
        check_type(Redis, redis, "redis")
        check_type(timedelta, expire_after, "expire_after")
        check_type(bytes, key_prefix, "key_prefix")
        check_type(str, listener_channel, "listener_channel")
        self.redis = redis
        self._expire_after = expire_after
        self._prefix = key_prefix
        self.listener_channel = listener_channel
        self._pubsub: Any = None

    def __repr__(self) -> str:
        return (
            f"RedisExampleDatabase({self.redis!r}, expire_after={self._expire_after!r})"
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, RedisExampleDatabase)
            and self.redis == other.redis
            and self._prefix == other._prefix
            and self.listener_channel == other.listener_channel
        )

    @contextmanager
    def _pipeline(
        self,
        *reset_expire_keys,
        execute_and_publish=True,
        event_type=None,
        to_publish=None,
    ):
        # Context manager to batch updates and expiry reset, reducing TCP roundtrips
        pipe = self.redis.pipeline()
        yield pipe
        for key in reset_expire_keys:
            pipe.expire(self._prefix + key, self._expire_after)
        if execute_and_publish:
            changed = pipe.execute()
            # pipe.execute returns the rows modified for each operation, which includes
            # the operations performed during the yield, followed by the n operations
            # from pipe.exire. Look at just the operations from during the yield.
            changed = changed[: -len(reset_expire_keys)]
            if any(count > 0 for count in changed):
                assert to_publish is not None
                assert event_type is not None
                self._publish((event_type, to_publish))

    def _publish(self, event):
        event = (event[0], tuple(self._encode(v) for v in event[1]))
        self.redis.publish(self.listener_channel, json.dumps(event))

    def _encode(self, value: bytes) -> str:
        return base64.b64encode(value).decode("ascii")

    def _decode(self, value: str) -> bytes:
        return base64.b64decode(value)

    def fetch(self, key: bytes) -> Iterable[bytes]:
        with self._pipeline(key, execute_and_publish=False) as pipe:
            pipe.smembers(self._prefix + key)
        yield from pipe.execute()[0]

    def save(self, key: bytes, value: bytes) -> None:
        with self._pipeline(key, event_type="save", to_publish=(key, value)) as pipe:
            pipe.sadd(self._prefix + key, value)

    def delete(self, key: bytes, value: bytes) -> None:
        with self._pipeline(key, event_type="delete", to_publish=(key, value)) as pipe:
            pipe.srem(self._prefix + key, value)

    def move(self, src: bytes, dest: bytes, value: bytes) -> None:
        if src == dest:
            self.save(dest, value)
            return

        with self._pipeline(src, dest, execute_and_publish=False) as pipe:
            pipe.srem(self._prefix + src, value)
            pipe.sadd(self._prefix + dest, value)

        changed = pipe.execute()
        if changed[0] > 0:
            # did the value set of the first key change?
            self._publish(("delete", (src, value)))
        if changed[1] > 0:
            # did the value set of the second key change?
            self._publish(("save", (dest, value)))

    def _handle_message(self, message: dict) -> None:
        # other message types include "subscribe" and "unsubscribe". these are
        # sent to the client, but not to the pubsub channel.
        assert message["type"] == "message"
        data = json.loads(message["data"])
        event_type = data[0]
        self._broadcast_change(
            (event_type, tuple(self._decode(v) for v in data[1]))  # type: ignore
        )

    def _start_listening(self) -> None:
        self._pubsub = self.redis.pubsub()
        self._pubsub.subscribe(**{self.listener_channel: self._handle_message})

    def _stop_listening(self) -> None:
        self._pubsub.unsubscribe()
        self._pubsub.close()
        self._pubsub = None
