# Owner(s): ["oncall: distributed"]

import sys
import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_distributed import (
    spawn_threads_and_init_comms,
    MultiThreadedTestCase

)
from torch.testing._internal.common_utils import TestCase, run_tests

DEFAULT_WORLD_SIZE = 4

class TestCollectivesWithWrapper(TestCase):
    @spawn_threads_and_init_comms(world_size=4)
    def test_broadcast_object_list(self):
        val = 99 if dist.get_rank() == 0 else None
        object_list = [val] * dist.get_world_size()

        dist.broadcast_object_list(object_list=object_list)
        self.assertEqual(99, object_list[0])

class TestCollectivesWithBaseClass(MultiThreadedTestCase):
    @property
    def world_size(self):
        return 4

    def test_allgather(self):
        input_tensor = torch.ones(3, 3) * dist.get_rank()
        output_tensors = [torch.empty_like(input_tensor) for _ in range(self.world_size)]
        dist.all_gather(output_tensors, input_tensor)
        for rank, out_tensor in enumerate(output_tensors):
            self.assertEqual(out_tensor, torch.ones(3, 3) * rank)

    def test_broadcast(self):
        input_tensor = torch.ones(3, 3) * dist.get_rank()
        for rank in range(self.world_size):
            cloned_input = input_tensor.clone()
            dist.broadcast(cloned_input, src=rank)
            self.assertEqual(cloned_input, torch.ones(3, 3) * rank)

    def test_scatter(self):
        if dist.get_rank() == 0:
            scatter_list = [torch.ones(3, 3) * rank for rank in range(self.world_size)]
        else:
            scatter_list = None
        output_tensor = torch.empty(3, 3)

        dist.scatter(output_tensor, scatter_list)
        self.assertEqual(output_tensor, torch.ones(3, 3) * dist.get_rank())

    def test_reduce_scatter(self):
        to_reduce_scatter = [torch.ones(3, 3) * rank for rank in range(self.world_size)]
        output_tensor = torch.empty(3, 3)

        dist.reduce_scatter(output_tensor, to_reduce_scatter)
        expected_tensor = torch.ones(3, 3) * dist.get_rank() * self.world_size
        self.assertEqual(output_tensor, expected_tensor)

    def test_broadcast_object_list(self):
        val = 99 if dist.get_rank() == 0 else None
        object_list = [val] * dist.get_world_size()
        print(f"{dist.get_rank()} -> {dist.get_world_size()}")

        dist.broadcast_object_list(object_list=object_list)
        self.assertEqual(99, object_list[0])

    def test_all_reduce(self):
        output = torch.ones(3, 3) * dist.get_rank()
        dist.all_reduce(output)
        res_num = ((0 + self.world_size - 1) * self.world_size) / 2
        self.assertEqual(output, torch.ones(3, 3) * res_num)

        # Test unimplemented error
        with self.assertRaisesRegex(NotImplementedError, "only supports SUM on threaded pg for now"):
            dist.all_reduce(output, op=ReduceOp.MAX)


if __name__ == "__main__":
    run_tests()
