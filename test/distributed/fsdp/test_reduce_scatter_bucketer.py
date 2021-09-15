# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import itertools
import sys
import unittest
from unittest import mock

import torch
from torch.distributed._fsdp.reduce_scatter_bucketer import ReduceScatterBucketer
from torch.testing._internal.common_fsdp import spawn_and_init, teardown


CONFIG_OPTIONS = [
    dict(zip(["bucket_cap_mb", "shard_size"], config)) for config in itertools.product([0, 0.25], [1, 262144])
]

class TestReduceScatterBucketer(unittest.TestCase):
    # TODO(sshleifer): check if possible to reuse `DistributedTest, spawn_and_init`.
    def setUp(self):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available, skipping test")
        if sys.platform == "win32":
            raise unittest.SkipTest("NCCL doesn't support Windows, skipping test")
        if torch.cuda.device_count() < 2:
            raise unittest.SkipTest("distributed tests require 2+ GPUs, skipping")

    def tearDown(self):
        teardown()

    def test_reduce_scatter(self):
        for config in CONFIG_OPTIONS:
            spawn_and_init(functools.partial(self._test_reduce_scatter, **config))

    @staticmethod
    def _test_reduce_scatter(rank, group, bucket_cap_mb=None, shard_size=None):
        bucketer = ReduceScatterBucketer(bucket_cap_mb=bucket_cap_mb)
        world_size = group.size()

        tensors = [torch.ones(shard_size).cuda() for _ in range(world_size)]
        tensors[rank].fill_(0)

        input_bytes = shard_size * world_size * 4
        bucket_bytes = bucket_cap_mb * 1024 * 1024

        callback = mock.MagicMock()
        bucketer.reduce_scatter_async(tensors, group, callback_fn=callback)

        if bucket_cap_mb > 0 and input_bytes < bucket_bytes:
            assert callback.call_count == 0
            bucketer.flush()
        assert callback.call_count == 1

        result = callback.call_args[0][0]  # get first positional arg
        assert torch.is_tensor(result), result
        assert torch.all(result == (world_size - 1))

    def test_out_of_order_reduction(self):
        spawn_and_init(self._test_out_of_order_reduction)

    @staticmethod
    def _test_out_of_order_reduction(rank, group):
        bucketer = ReduceScatterBucketer(bucket_cap_mb=0.25)
        world_size = group.size()

        small_tensors = [torch.ones(1).cuda() for _ in range(world_size)]
        big_tensors = [torch.ones(262144).cuda() for _ in range(world_size)]
        more_small_tensors = [torch.ones(2).cuda() for _ in range(world_size)]

        callback1 = mock.MagicMock()
        callback2 = mock.MagicMock()
        callback3 = mock.MagicMock()

        bucketer.reduce_scatter_async(small_tensors, group, callback_fn=callback1)
        assert callback1.call_count == 0
        bucketer.reduce_scatter_async(big_tensors, group, callback_fn=callback2)
        assert callback1.call_count == 0
        assert callback2.call_count == 1
        bucketer.reduce_scatter_async(more_small_tensors, group, callback_fn=callback3)
        assert callback1.call_count == 0
        assert callback2.call_count == 1
        assert callback3.call_count == 0

        bucketer.flush()
        assert callback1.call_count == 1
        assert callback2.call_count == 1
        assert callback3.call_count == 1


if __name__ == "__main__":
    unittest.main()
