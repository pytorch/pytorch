# Owner(s): ["oncall: distributed"]

import os
import sys
import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp
from unittest import skip, SkipTest
import operator
from functools import reduce
import threading
import torch.autograd

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_distributed import (
    spawn_threads_and_init_comms,
    MultiThreadedTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    IS_SANDCASTLE,
)


DEFAULT_WORLD_SIZE = 4

class TestCollectivesWithWrapper(TestCase):
    @spawn_threads_and_init_comms(world_size=4)
    def test_broadcast_object_list(self):
        val = 99 if dist.get_rank() == 0 else None
        object_list = [val] * dist.get_world_size()

        dist.broadcast_object_list(object_list=object_list)
        self.assertEqual(99, object_list[0])

    def test_collective_error_on_rank_zero(self):
        @spawn_threads_and_init_comms(world_size=4)
        def _test_method(self):
            input_tensor = torch.ones(3, 3) * dist.get_rank()  # perform 1st all gather
            output_tensors = [torch.empty_like(input_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(output_tensors, input_tensor)

            if dist.get_rank() == 0:
                raise AssertionError("Mimic real test failure.")  # fail on rank 0

            dist.all_gather(output_tensors, input_tensor)  # perform 2nd all gather

        with self.assertRaises(RuntimeError):
            _test_method(self)

    def test_collective_error_on_rank_non_zero(self):
        @spawn_threads_and_init_comms(world_size=4)
        def _test_method(self):
            input_tensor = torch.ones(3, 3) * dist.get_rank()  # perform 1st all gather
            output_tensors = [torch.empty_like(input_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(output_tensors, input_tensor)

            if dist.get_rank() == 1:
                raise AssertionError("Mimic real test failure.")  # fail on rank 1

            dist.all_gather(output_tensors, input_tensor)  # perform 2nd all gather

        with self.assertRaises(RuntimeError):
            _test_method(self)

    def test_collective_error_on_rank_non_zero_all(self):
        @spawn_threads_and_init_comms(world_size=4)
        def _test_method(self):
            input_tensor = torch.ones(3, 3) * dist.get_rank()  # perform 1st all gather
            output_tensors = [torch.empty_like(input_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(output_tensors, input_tensor)

            if dist.get_rank() > 0:
                raise AssertionError("Mimic real test failure.")  # fail on all non-zero rank

            dist.all_gather(output_tensors, input_tensor)  # perform 2nd all gather

        with self.assertRaises(RuntimeError):
            _test_method(self)

    def test_skip(self):
        @spawn_threads_and_init_comms(world_size=4)
        @skip("check if skip exception can be captured correctly.")
        def _test_method(self):
            pass

        if not IS_SANDCASTLE:
            with self.assertRaises(SkipTest):
                _test_method(self)

class TestCollectivesWithBaseClass(MultiThreadedTestCase):
    @property
    def world_size(self):
        return 4

    def setUp(self):
        os.environ["TORCH_DIST_INIT_BARRIER"] = "1"
        super().setUp()
        self._spawn_threads()

    def tearDown(self):
        super().tearDown()
        os.environ["TORCH_DIST_INIT_BARRIER"] = "0"

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

    def test_all_to_all(self):
        rank = self.rank
        world_size = self.world_size
        input_tensor_list = [
            torch.ones(3, 3) * x
            for x in range(rank * world_size, (rank + 1) * world_size)
        ]
        output_tensor_list = [torch.empty_like(tensor) for tensor in input_tensor_list]
        dist.all_to_all(output_tensor_list, input_tensor_list)
        expected_tensor_list = [
            torch.ones(3, 3) * x
            for x in range(rank, world_size * world_size, world_size)
        ]
        self.assertEqual(expected_tensor_list, output_tensor_list)

    def test_all_reduce_ops(self):
        tensor = torch.tensor([dist.get_rank() + 1])
        dist.all_reduce(tensor, op=ReduceOp.PRODUCT)
        expected = reduce(operator.mul, range(1, self.world_size + 1))
        self.assertEqual(expected, tensor.item())

        tensor = torch.tensor([dist.get_rank() + 1])
        dist.all_reduce(tensor, op=ReduceOp.MIN)
        self.assertEqual(1, tensor.item())

        tensor = torch.tensor([dist.get_rank() + 1])
        dist.all_reduce(tensor, op=ReduceOp.MAX)
        self.assertEqual(self.world_size, tensor.item())

        tensor = torch.tensor([dist.get_rank() + 1])
        dist.all_reduce(tensor, op=ReduceOp.BAND)
        expected = reduce(operator.and_, range(1, self.world_size + 1))
        self.assertEqual(expected, tensor.item())

        tensor = torch.tensor([dist.get_rank() + 1])
        dist.all_reduce(tensor, op=ReduceOp.BOR)
        expected = reduce(operator.or_, range(1, self.world_size + 1))
        self.assertEqual(expected, tensor.item())

        tensor = torch.tensor([dist.get_rank() + 1])
        dist.all_reduce(tensor, op=ReduceOp.BXOR)
        expected = reduce(operator.xor, range(1, self.world_size + 1))
        self.assertEqual(expected, tensor.item())

    def test_assert_equal_on_rank(self):
        # RNG is shared across threads. So instead of asserting on all threads
        # we only assert on rank 0
        self_tensor = torch.rand(3, 3)
        rank_0_tensor = self_tensor.clone()
        dist.broadcast(rank_0_tensor, src=0)
        self.assertEqualOnRank(rank_0_tensor, self_tensor, rank=0)
        self.assertNotEqualOnRank(rank_0_tensor, self_tensor, rank=1)

    def test_subpg(self):
        subpg0 = dist.new_group([0, 1])
        subpg1 = dist.new_group([2, 3])
        current_rank = dist.get_rank()
        output = torch.ones(3, 3) * current_rank

        # call all_reduce on subpg0 and subpg1 concurrently
        if current_rank in [0, 1]:
            dist.all_reduce(output, group=subpg0)
        else:
            dist.all_reduce(output, group=subpg1)

        if current_rank in [0, 1]:
            self.assertEqual(output, torch.ones(3, 3) * 1)
        else:
            self.assertEqual(output, torch.ones(3, 3) * 5)

    def test_using_pg_from_another_thread(self):
        def stuff_in_other_thread(pg):
            x = torch.rand(4, requires_grad=True)
            dist.all_reduce(x, group=pg)

        t = threading.Thread(target=stuff_in_other_thread, args=(dist.group.WORLD,))
        t.start()
        t.join()

    def test_gather(self):
        if dist.get_rank() == 0:
            gather_list = [torch.empty(3, 3) for _ in range(self.world_size)]
        else:
            gather_list = None
        input_tensor = torch.ones(3, 3) * dist.get_rank()

        dist.gather(input_tensor, gather_list)
        if dist.get_rank() == 0:
            for i in range(self.world_size):
                self.assertEqual(gather_list[i], torch.ones(3, 3) * i)

    def test_all_reduce_coalesced(self):
        t0 = torch.ones(3, 3) * dist.get_rank()
        t1 = torch.ones(3, 3) * dist.get_rank() * 2
        dist.all_reduce_coalesced([t0, t1])
        res_num = ((0 + self.world_size - 1) * self.world_size) / 2
        self.assertEqual(t0, torch.ones(3, 3) * res_num)
        self.assertEqual(t1, torch.ones(3, 3) * (res_num * 2))

    @skip_if_lt_x_gpu(1)
    def test_bwd_sees_fwd_pg(self):
        fwd_tid = threading.current_thread().ident

        class MyFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, rank):
                result = rank * 2

                ctx.save_for_backward(result, rank)
                assert int(rank.item()) == dist.get_rank()
                return result

            @staticmethod
            def backward(ctx, grad_output):
                result, rank = ctx.saved_tensors
                bwd_tid = threading.current_thread().ident

                self.assertEqual(fwd_tid, bwd_tid, f"bwd not running in the same thread a fwd for rank {rank.item()}")
                self.assertTrue(dist.is_initialized())
                self.assertEqual(int(rank.item()), dist.get_rank())
                dist.all_reduce(result)
                self.assertEqual(int(result.item()), 12)  # (0 + 1 + 2 + 3) * 2

                return grad_output * result

        x = torch.tensor([dist.get_rank()], dtype=torch.float, device="cuda", requires_grad=True)
        x = MyFunc.apply(x)
        x.sum().backward()

if __name__ == "__main__":
    run_tests()
