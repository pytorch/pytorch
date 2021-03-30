#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch.distributed.elastic.utils.store as store_util
from torch.distributed.elastic.utils.logging import get_logger


class TestStore:
    def get(self, key: str):
        return f"retrieved:{key}"


class StoreUtilTest(unittest.TestCase):
    def test_get_data(self):
        store = TestStore()
        data = store_util.get_all(store, "test/store", 10)
        for idx in range(0, 10):
            self.assertEqual(f"retrieved:test/store{idx}", data[idx])

    def test_synchronize(self):
        class DummyStore:
            def __init__(self):
                self._data = {
                    "torchelastic/test0": "data0".encode(encoding="UTF-8"),
                    "torchelastic/test1": "data1".encode(encoding="UTF-8"),
                    "torchelastic/test2": "data2".encode(encoding="UTF-8"),
                }

            def set(self, key, value):
                self._data[key] = value

            def get(self, key):
                return self._data[key]

            def set_timeout(self, timeout):
                pass

        data = "data0".encode(encoding="UTF-8")
        store = DummyStore()
        res = store_util.synchronize(store, data, 0, 3, key_prefix="torchelastic/test")
        self.assertEqual(3, len(res))
        for idx, res_data in enumerate(res):
            actual_str = res_data.decode(encoding="UTF-8")
            self.assertEqual(f"data{idx}", actual_str)


class UtilTest(unittest.TestCase):
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
