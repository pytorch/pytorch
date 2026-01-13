# Owner(s): ["oncall: distributed"]
import unittest

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import FakeWork, ProcessGroup
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._functional_collectives import (
    all_gather_into_tensor_coalesced,
    all_gather_tensor,
    all_gather_tensor_autograd,
    all_reduce,
    all_reduce_coalesced,
    all_to_all_single,
    all_to_all_single_autograd,
    broadcast,
    reduce_scatter_tensor,
    reduce_scatter_tensor_autograd,
    reduce_scatter_tensor_coalesced,
    wait_tensor,
)
from torch.distributed._tools.fake_collectives import (
    collective_ops,
    CollectiveOp,
    non_functional_collectives,
)
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.utils._python_dispatch import TorchDispatchMode


aten = torch.ops.aten
c10d = torch.ops.c10d
_c10d_functional = torch.ops._c10d_functional
_c10d_functional_autograd = torch.ops._c10d_functional_autograd


class TestFakeCollectives(TestCase):
    def _setup_distributed(self):
        world_size = 4
        store = FakeStore()
        dist.init_process_group("fake", rank=0, world_size=world_size, store=store)
        torch.cuda.set_device(torch.cuda.current_device())

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_collectives(self):
        try:
            self._setup_distributed()
            with FakeTensorMode(), CollectiveTest(test=self):
                test_tensor_list = [torch.randn(100, device="cuda") for _ in range(4)]
                test_tensor_list_2 = [torch.randn(400, device="cuda") for _ in range(4)]
                test_tensor = torch.randn(100, device="cuda")
                # Used as gather output or scatter input
                test_tensor2 = torch.randn(400, device="cuda")

                # Testing non-functional collective operations
                dist.broadcast(test_tensor, src=0)
                dist.all_reduce(test_tensor)
                dist.reduce(test_tensor, dst=0)
                dist.send(test_tensor, dst=1)
                dist.recv(test_tensor, src=1)
                dist.all_gather(test_tensor_list, test_tensor)
                dist.reduce_scatter(test_tensor, test_tensor_list)
                dist.reduce_scatter_tensor(test_tensor, test_tensor2)
                dist.scatter(test_tensor, scatter_list=test_tensor_list, src=0)
                dist.gather(test_tensor, gather_list=test_tensor_list, dst=0)
                dist.all_gather_into_tensor(test_tensor2, test_tensor)
                dist.all_to_all(test_tensor_list, test_tensor_list)
                dist.all_to_all_single(test_tensor2, test_tensor2)
                dist.barrier()

                # Testing functional collectives
                wait_tensor(test_tensor)
                broadcast(test_tensor, src=0, group=dist.group.WORLD)
                all_reduce(test_tensor, reduceOp="avg", group=dist.group.WORLD)
                all_gather_tensor(test_tensor, gather_dim=0, group=dist.group.WORLD)
                all_gather_tensor_autograd(
                    test_tensor, gather_dim=0, group=dist.group.WORLD
                )
                reduce_scatter_tensor(
                    test_tensor2, scatter_dim=0, reduceOp="sum", group=dist.group.WORLD
                )
                reduce_scatter_tensor_autograd(
                    test_tensor2, scatter_dim=0, reduceOp="sum", group=dist.group.WORLD
                )
                all_to_all_single(
                    test_tensor,
                    output_split_sizes=[0],
                    input_split_sizes=[1],
                    group=dist.group.WORLD,
                )
                all_reduce_coalesced(
                    test_tensor_list, reduceOp="avg", group=dist.group.WORLD
                )
                all_gather_into_tensor_coalesced(
                    test_tensor_list, group=dist.group.WORLD
                )
                reduce_scatter_tensor_coalesced(
                    test_tensor_list_2,
                    scatter_dim=[0] * 4,
                    reduceOp="sum",
                    group=dist.group.WORLD,
                )
                all_to_all_single_autograd(
                    test_tensor,
                    output_split_sizes=[0],
                    input_split_sizes=[1],
                    group=dist.group.WORLD,
                )
        finally:
            if dist.group.WORLD is not None:
                dist.destroy_process_group()


class CollectiveTest(TorchDispatchMode):
    collective_size_exclude = {
        c10d.barrier.default,
        c10d.monitored_barrier_.default,
        _c10d_functional.wait_tensor.default,
    }

    def __init__(self, test: TestFakeCollectives, _dispatch_key=None):
        super().__init__(_dispatch_key)
        self.test = test

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        res = func(*args, **(kwargs or {}))

        if func in collective_ops:
            if func != _c10d_functional.wait_tensor.default:
                pg = CollectiveOp.get_process_group(func, args)
                self.test.assertIsInstance(
                    pg, ProcessGroup, "Error: pg is not an instance of ProcessGroup"
                )
                self.test.assertEqual(
                    pg, dist.group.WORLD, "Error: pg is not equal to dist.group.WORLD"
                )
                self.test.assertEqual(
                    pg.size(),
                    4,
                    f"Error: Expected pg.size() to be 4, but got {pg.size()}",
                )
                self.test.assertNotEqual(
                    pg.name(), "", "Error: pg.name() should not be an empty string"
                )

            if func not in CollectiveTest.collective_size_exclude:
                # Compute expected communication tensor size
                computed_size = CollectiveOp.get_comm_tensor_size(
                    func, res, args, kwargs
                )
                expected_size = self.get_expected_size(func, res, args, kwargs)

                self.test.assertEqual(
                    computed_size,
                    expected_size,
                    msg=f"Size mismatch for {func.__name__}: expected {expected_size}, got {computed_size}",
                )

        if (
            func in non_functional_collectives
            and func != c10d.monitored_barrier_.default
        ):
            work = res[-1] if isinstance(res, (tuple, list)) else res
            self.test.assertIsInstance(FakeWork.unbox(work), FakeWork)

        return res

    @staticmethod
    def get_expected_size(func, res, args, kwargs):
        """Return expected tensor size for collectives explicitly used in run_test()."""
        WORLD_SIZE, TENSOR_100, TENSOR_400 = 4, 100 * 4, 400 * 4
        TENSOR_LIST_100, TENSOR_LIST_400 = (
            WORLD_SIZE * TENSOR_100,
            WORLD_SIZE * TENSOR_400,
        )

        size_map = {
            # Non-functional collectives
            c10d.broadcast_.default: TENSOR_100,
            c10d.allreduce_.default: TENSOR_100,
            c10d.reduce_.default: TENSOR_100,
            c10d.send.default: TENSOR_100,
            c10d.recv_.default: TENSOR_100,
            c10d.allgather_.default: TENSOR_LIST_100,
            c10d.reduce_scatter_.default: TENSOR_LIST_100,
            c10d._reduce_scatter_base_.default: TENSOR_400,
            c10d.scatter_.default: TENSOR_LIST_100,
            c10d.gather_.default: TENSOR_LIST_100,
            c10d._allgather_base_.default: TENSOR_400,
            c10d.alltoall_.default: TENSOR_LIST_100,
            c10d.alltoall_base_.default: TENSOR_400,
            # Functional collectives
            _c10d_functional.broadcast.default: TENSOR_100,
            _c10d_functional.all_reduce.default: TENSOR_100,
            _c10d_functional.all_gather_into_tensor.default: TENSOR_LIST_100,
            _c10d_functional_autograd.all_gather_into_tensor.default: TENSOR_LIST_100,
            _c10d_functional.reduce_scatter_tensor.default: TENSOR_400,
            _c10d_functional_autograd.reduce_scatter_tensor.default: TENSOR_400,
            _c10d_functional.all_to_all_single.default: TENSOR_100,
            _c10d_functional_autograd.all_to_all_single.default: TENSOR_100,
            _c10d_functional.all_reduce_coalesced.default: TENSOR_LIST_100,
            _c10d_functional.all_gather_into_tensor_coalesced.default: TENSOR_LIST_400,
            _c10d_functional.reduce_scatter_tensor_coalesced.default: TENSOR_LIST_100,
        }

        if func in size_map:
            return size_map[func]

        raise ValueError(f"Unhandled function: {func}")


if __name__ == "__main__":
    run_tests()
