#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import datetime
from multiprocessing.pool import ThreadPool
from typing import List

import torch.distributed as dist
import torch.distributed.elastic.utils.store as store_util
from torch.distributed.elastic.utils.logging import get_logger
from torch.testing._internal.common_utils import run_tests, TestCase


class MockStore:
    def __init__(self) -> None:
        self.ops = []

    def set_timeout(self, timeout: float) -> None:
        self.ops.append(("set_timeout", timeout))

    @property
    def timeout(self) -> datetime.timedelta:
        self.ops.append(("timeout",))

        return datetime.timedelta(seconds=1234)

    def set(self, key: str, value: str) -> None:
        self.ops.append(("set", key, value))

    def get(self, key: str) -> str:
        self.ops.append(("get", key))
        return "value"

    def multi_get(self, keys: List[str]) -> List[str]:
        self.ops.append(("multi_get", keys))
        return ["value"] * len(keys)

    def add(self, key: str, val: int) -> int:
        self.ops.append(("add", key, val))
        return 3

    def wait(self, keys: List[str]) -> None:
        self.ops.append(("wait", keys))


class StoreUtilTest(TestCase):
    def test_get_all_rank_0(self):
        world_size = 3

        store = MockStore()

        store_util.get_all(store, 0, "test/store", world_size)

        self.assertListEqual(
            store.ops,
            [
                ("multi_get", ["test/store0", "test/store1", "test/store2"]),
                ("add", "test/store/finished/num_members", 1),
                ("set", "test/store/finished/last_member", "<val_ignored>"),
                ("wait", ["test/store/finished/last_member"]),
            ],
        )

    def test_get_all_rank_n(self):
        store = MockStore()
        world_size = 3
        store_util.get_all(store, 1, "test/store", world_size)

        self.assertListEqual(
            store.ops,
            [
                ("multi_get", ["test/store0", "test/store1", "test/store2"]),
                ("add", "test/store/finished/num_members", 1),
                ("set", "test/store/finished/last_member", "<val_ignored>"),
            ],
        )

    def test_synchronize(self):
        store = MockStore()

        data = b"data0"
        store_util.synchronize(store, data, 0, 3, key_prefix="test/store")

        self.assertListEqual(
            store.ops,
            [
                ("timeout",),
                ("set_timeout", datetime.timedelta(seconds=300)),
                ("set", "test/store0", data),
                ("multi_get", ["test/store0", "test/store1", "test/store2"]),
                ("add", "test/store/finished/num_members", 1),
                ("set", "test/store/finished/last_member", "<val_ignored>"),
                ("wait", ["test/store/finished/last_member"]),
                ("set_timeout", datetime.timedelta(seconds=1234)),
            ],
        )

    def test_synchronize_hash_store(self) -> None:
        N = 4

        store = dist.HashStore()

        def f(i: int):
            return store_util.synchronize(
                store, f"data{i}", i, N, key_prefix="test/store"
            )

        with ThreadPool(N) as pool:
            out = pool.map(f, range(N))

        self.assertListEqual(out, [[f"data{i}".encode() for i in range(N)]] * N)

    def test_barrier(self):
        store = MockStore()

        store_util.barrier(store, 3, key_prefix="test/store")

        self.assertListEqual(
            store.ops,
            [
                ("timeout",),
                ("set_timeout", datetime.timedelta(seconds=300)),
                ("add", "test/store/num_members", 1),
                ("set", "test/store/last_member", "<val_ignored>"),
                ("wait", ["test/store/last_member"]),
                ("set_timeout", datetime.timedelta(seconds=1234)),
            ],
        )

    def test_barrier_hash_store(self) -> None:
        N = 4

        store = dist.HashStore()

        def f(i: int):
            store_util.barrier(store, N, key_prefix="test/store")

        with ThreadPool(N) as pool:
            out = pool.map(f, range(N))

        self.assertEqual(out, [None] * N)


class UtilTest(TestCase):
    def test_get_logger_different(self):
        logger1 = get_logger("name1")
        logger2 = get_logger("name2")
        self.assertNotEqual(logger1.name, logger2.name)

    def test_get_logger(self):
        logger1 = get_logger()
        self.assertEqual(__name__, logger1.name)

    def test_get_logger_none(self):
        logger1 = get_logger(None)
        self.assertEqual(__name__, logger1.name)

    def test_get_logger_custom_name(self):
        logger1 = get_logger("test.module")
        self.assertEqual("test.module", logger1.name)


if __name__ == "__main__":
    run_tests()
