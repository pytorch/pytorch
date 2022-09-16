#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import torch.distributed.elastic.utils.store as store_util
from torch.distributed.elastic.utils.logging import get_logger
from torch.testing._internal.common_utils import run_tests, TestCase


class StoreUtilTest(TestCase):
    def test_get_all_rank_0(self):
        store = mock.MagicMock()
        world_size = 3
        store_util.get_all(store, 0, "test/store", world_size)
        # omit empty kwargs, get only key
        actual_set_call_args = [
            call_args[0][0] for call_args in store.set.call_args_list
        ]
        self.assertListEqual(["test/store0.FIN"], actual_set_call_args)

        actual_get_call_args = [call_args[0] for call_args in store.get.call_args_list]
        expected_get_call_args = [
            ("test/store0",),
            ("test/store1",),
            ("test/store2",),
            ("test/store0.FIN",),
            ("test/store1.FIN",),
            ("test/store2.FIN",),
        ]
        self.assertListEqual(expected_get_call_args, actual_get_call_args)

    def test_get_all_rank_n(self):
        store = mock.MagicMock()
        world_size = 3
        store_util.get_all(store, 1, "test/store", world_size)
        # omit empty kwargs, get only key
        actual_set_call_args = [
            call_args[0][0] for call_args in store.set.call_args_list
        ]
        self.assertListEqual(["test/store1.FIN"], actual_set_call_args)

        actual_get_call_args = [call_args[0] for call_args in store.get.call_args_list]
        expected_get_call_args = [
            ("test/store0",),
            ("test/store1",),
            ("test/store2",),
        ]
        self.assertListEqual(expected_get_call_args, actual_get_call_args)

    def test_synchronize(self):
        store_mock = mock.MagicMock()
        data = "data0".encode(encoding="UTF-8")
        store_util.synchronize(store_mock, data, 0, 3, key_prefix="torchelastic/test")
        actual_set_call_args = store_mock.set.call_args_list
        # omit empty kwargs
        actual_set_call_args = [call_args[0] for call_args in actual_set_call_args]
        expected_set_call_args = [
            ("torchelastic/test0", b"data0"),
            ("torchelastic/test0.FIN", b"FIN"),
        ]
        self.assertListEqual(expected_set_call_args, actual_set_call_args)

        expected_get_call_args = [
            ("torchelastic/test0",),
            ("torchelastic/test1",),
            ("torchelastic/test2",),
            ("torchelastic/test0.FIN",),
            ("torchelastic/test1.FIN",),
            ("torchelastic/test2.FIN",),
        ]
        actual_get_call_args = store_mock.get.call_args_list
        actual_get_call_args = [call_args[0] for call_args in actual_get_call_args]
        self.assertListEqual(expected_get_call_args, actual_get_call_args)


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
