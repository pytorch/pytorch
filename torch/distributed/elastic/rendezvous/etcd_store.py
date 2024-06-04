# mypy: allow-untyped-defs
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import random
import time
from base64 import b64decode, b64encode
from typing import Optional

import etcd  # type: ignore[import]

# pyre-ignore[21]: Could not find name `Store` in `torch.distributed`.
from torch.distributed import Store


# Delay (sleep) for a small random amount to reduce CAS failures.
# This does not affect correctness, but will reduce requests to etcd server.
def cas_delay():
    time.sleep(random.uniform(0, 0.1))


# pyre-fixme[11]: Annotation `Store` is not defined as a type.
class EtcdStore(Store):
    """
    Implement a c10 Store interface by piggybacking on the rendezvous etcd instance.

    This is the store object returned by ``EtcdRendezvous``.
    """

    def __init__(
        self,
        etcd_client,
        etcd_store_prefix,
        # Default timeout same as in c10d/Store.hpp
        timeout: Optional[datetime.timedelta] = None,
    ):
        super().__init__()  # required for pybind trampoline.

        self.client = etcd_client
        self.prefix = etcd_store_prefix

        if timeout is not None:
            self.set_timeout(timeout)

        if not self.prefix.endswith("/"):
            self.prefix += "/"

    def set(self, key, value):
        """
        Write a key/value pair into ``EtcdStore``.

        Both key and value may be either Python ``str`` or ``bytes``.
        """
        self.client.set(key=self.prefix + self._encode(key), value=self._encode(value))

    def get(self, key) -> bytes:
        """
        Get a value by key, possibly doing a blocking wait.

        If key is not immediately present, will do a blocking wait
        for at most ``timeout`` duration or until the key is published.


        Returns:
            value ``(bytes)``

        Raises:
            LookupError - If key still not published after timeout
        """
        b64_key = self.prefix + self._encode(key)
        kvs = self._try_wait_get([b64_key])

        if kvs is None:
            raise LookupError(f"Key {key} not found in EtcdStore")

        return self._decode(kvs[b64_key])

    def add(self, key, num: int) -> int:
        """
        Atomically increment a value by an integer amount.

        The integer is represented as a string using base 10. If key is not present,
        a default value of ``0`` will be assumed.

        Returns:
             the new (incremented) value


        """
        b64_key = self._encode(key)
        # c10d Store assumes value is an integer represented as a decimal string
        try:
            # Assume default value "0", if this key didn't yet:
            node = self.client.write(
                key=self.prefix + b64_key,
                value=self._encode(str(num)),  # i.e. 0 + num
                prevExist=False,
            )
            return int(self._decode(node.value))
        except etcd.EtcdAlreadyExist:
            pass

        while True:
            # Note: c10d Store does not have a method to delete keys, so we
            # can be sure it's still there.
            node = self.client.get(key=self.prefix + b64_key)
            new_value = self._encode(str(int(self._decode(node.value)) + num))
            try:
                node = self.client.test_and_set(
                    key=node.key, value=new_value, prev_value=node.value
                )
                return int(self._decode(node.value))
            except etcd.EtcdCompareFailed:
                cas_delay()

    def wait(self, keys, override_timeout: Optional[datetime.timedelta] = None):
        """
        Wait until all of the keys are published, or until timeout.

        Raises:
            LookupError - if timeout occurs
        """
        b64_keys = [self.prefix + self._encode(key) for key in keys]
        kvs = self._try_wait_get(b64_keys, override_timeout)
        if kvs is None:
            raise LookupError("Timeout while waiting for keys in EtcdStore")
        # No return value on success

    def check(self, keys) -> bool:
        """Check if all of the keys are immediately present (without waiting)."""
        b64_keys = [self.prefix + self._encode(key) for key in keys]
        kvs = self._try_wait_get(
            b64_keys,
            override_timeout=datetime.timedelta(microseconds=1),  # as if no wait
        )
        return kvs is not None

    #
    # Encode key/value data in base64, so we can store arbitrary binary data
    # in EtcdStore. Input can be `str` or `bytes`.
    # In case of `str`, utf-8 encoding is assumed.
    #
    def _encode(self, value) -> str:
        if type(value) == bytes:
            return b64encode(value).decode()
        elif type(value) == str:
            return b64encode(value.encode()).decode()
        raise ValueError("Value must be of type str or bytes")

    #
    # Decode a base64 string (of type `str` or `bytes`).
    # Return type is `bytes`, which is more convenient with the Store interface.
    #
    def _decode(self, value) -> bytes:
        if type(value) == bytes:
            return b64decode(value)
        elif type(value) == str:
            return b64decode(value.encode())
        raise ValueError("Value must be of type str or bytes")

    #
    # Get all of the (base64-encoded) etcd keys at once, or wait until all the keys
    # are published or timeout occurs.
    # This is a helper method for the public interface methods.
    #
    # On success, a dictionary of {etcd key -> etcd value} is returned.
    # On timeout, None is returned.
    #
    def _try_wait_get(self, b64_keys, override_timeout=None):
        timeout = self.timeout if override_timeout is None else override_timeout  # type: ignore[attr-defined]
        deadline = time.time() + timeout.total_seconds()

        while True:
            # Read whole directory (of keys), filter only the ones waited for
            all_nodes = self.client.get(key=self.prefix)
            req_nodes = {
                node.key: node.value for node in all_nodes.children if node.key in b64_keys
            }

            if len(req_nodes) == len(b64_keys):
                # All keys are available
                return req_nodes

            watch_timeout = deadline - time.time()
            if watch_timeout <= 0:
                return None

            try:
                self.client.watch(
                    key=self.prefix,
                    recursive=True,
                    timeout=watch_timeout,
                    index=all_nodes.etcd_index + 1,
                )
            except etcd.EtcdWatchTimedOut:
                if time.time() >= deadline:
                    return None
                else:
                    continue
            except etcd.EtcdEventIndexCleared:
                continue
