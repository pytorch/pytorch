# Owner(s): ["oncall: distributed"]

import time
import unittest
import weakref

import test_c10d_common

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._C._distributed_c10d import _create_work_from_future
from torch.futures import Future
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_distributed import MultiThreadedTestCase
from torch.testing._internal.common_utils import run_tests, TestCase


def create_work(result):
    future = Future()
    future.set_result(result)
    return _create_work_from_future(future)


class MyWork(dist._Work):
    def __init__(self, result, pg):
        super().__init__()
        self.result_ = result
        self.future_ = torch.futures.Future()
        self.future_.set_result(result)
        self.pg_ = weakref.ref(pg)

    def wait(self, timeout):
        self.pg_().wait_count += 1
        return True

    def get_future(self):
        self.pg_().get_future_count += 1
        return self.future_


class LonelyRankProcessGroup(dist.ProcessGroup):
    """
    This PG only supports world_size of 1
    """

    def __init__(self, rank, world, use_wrapper):
        super().__init__(rank, world)
        if rank != 0:
            raise AssertionError(f"Expected rank == 0, got {rank}")
        if world != 1:
            raise AssertionError(f"Expected world == 1, got {world}")

        self._rank = rank
        self._world = world
        self.wait_count = 0
        self.get_future_count = 0
        self.use_wrapper = use_wrapper
        self._work = []

    def broadcast(self, tensor_list, opts):
        if self.use_wrapper:
            return create_work(tensor_list)
        res = MyWork(tensor_list, self)
        self._work.append(res)
        return res

    def allgather(self, output_tensors, input_tensor, opts):
        for o, i in zip(output_tensors[0], input_tensor):
            o.copy_(i)
        if self.use_wrapper:
            return create_work(output_tensors)

        res = MyWork(output_tensors, self)
        self._work.append(res)

        return res

    def allreduce(self, tensors, opts):
        if self.use_wrapper:
            return create_work(tensors)
        res = MyWork(tensors, self)
        self._work.append(res)
        return res

    def getSize(self):
        return self._world

    def getBackendName(self):
        return "lonely-pg"

    def __repr__(self):
        return f"PLG w:{self._world} r:{self._rank}"


class DummyAttrProcessGroup(dist.ProcessGroup):
    def getRank(self):
        return 123

    def getSize(self):
        return 456

    def getBackendName(self):
        return "dummy-attr"

    def setGroupName(self, name) -> None:
        self._group_name = "py:" + name

    def getGroupName(self) -> str:
        return self._group_name

    def setGroupDesc(self, group_desc) -> None:
        self._group_desc = "py:" + group_desc

    def getGroupDesc(self) -> str:
        return self._group_desc


# We cannot use parametrize as some tests are defined on the base class and use _get_process_group
class AbstractDDPSingleRank(test_c10d_common.CommonDistributedDataParallelTest):
    def setUp(self):
        super().setUp()
        self._spawn_threads()

    @property
    def world_size(self):
        return 1

    def _get_process_group(self):
        return LonelyRankProcessGroup(self.rank, self.world_size, self.use_wrapper)

    def test_ddp_invoke_work_object(self):
        pg = self._get_process_group()

        torch.manual_seed(123)
        model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
        wrapped_model = model
        input_tensor = torch.rand(2)
        model = DDP(model, process_group=pg)
        model(input_tensor).sum().backward()

        ddp_grad = wrapped_model[0].bias.grad.clone()

        wrapped_model.zero_grad()
        wrapped_model(input_tensor).sum().backward()
        self.assertEqual(wrapped_model[0].bias.grad, ddp_grad)
        if not self.use_wrapper:
            self.assertTrue(pg.wait_count > 0)
            self.assertTrue(pg.get_future_count > 0)

    def test_ddp_with_pypg(self):
        pg = self._get_process_group()

        self._test_ddp_with_process_group(pg, [torch.device("cpu")], device_ids=None)

    def test_ddp_with_pypg_with_grad_views(self):
        pg = self._get_process_group()

        self._test_ddp_with_process_group(
            pg, [torch.device("cpu")], device_ids=None, gradient_as_bucket_view=True
        )

    def test_ddp_no_init_sync(self):
        pg = self._get_process_group()

        model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
        model = DDP(model, process_group=pg, init_sync=False)

        self.assertEqual(pg.wait_count, 0)
        self.assertEqual(pg.get_future_count, 0)


class TestDDPWithWorkSubclass(AbstractDDPSingleRank, MultiThreadedTestCase):
    @property
    def use_wrapper(self):
        return False


class TestDDPWithWorkWrapper(AbstractDDPSingleRank, MultiThreadedTestCase):
    @property
    def use_wrapper(self):
        return True


class BlockWork(dist._Work):
    """
    Dummy work that is used to test blocking the current stream.
    """

    def __init__(self):
        super().__init__()
        self.future_ = torch.futures.Future()

    def get_future(self):
        return self.future_


class TestPyProcessGroup(TestCase):
    def test_attr_overrides(self):
        pg = DummyAttrProcessGroup(0, 1)
        self.assertEqual(pg.name(), "dummy-attr")
        self.assertEqual(pg.rank(), 123)
        self.assertEqual(pg.size(), 456)

        pg._set_group_name("name")
        self.assertEqual(pg.group_name, "py:name")

        pg._set_group_desc("desc")
        self.assertEqual(pg.group_desc, "py:desc")

    def test_abort_shutdown(self) -> None:
        # verify this are noops
        pg = DummyAttrProcessGroup(0, 1)
        pg.abort()
        pg.shutdown()

    @unittest.skipIf(not TEST_CUDA, "no cuda/xpu")
    def test_block_current_stream(self) -> None:
        torch.cuda.synchronize()

        stream = torch.cuda.Stream()
        with stream:
            # nothing in queue so instantly resolves
            event1 = torch.cuda.Event()
            event1.record()
            time.sleep(0.1)
            self.assertTrue(event1.query())

            work = BlockWork()
            work.block_current_stream()

            # stream is blocked so doesn't resolve
            event = torch.cuda.Event()
            event.record()
            time.sleep(0.1)
            self.assertFalse(event.query())

            # resolve the work
            work.get_future().set_result(None)

            stream.synchronize()
            self.assertTrue(event.query())

    @unittest.skipIf(not TEST_CUDA, "no cuda/xpu")
    def test_block_current_stream_use_after_free(self) -> None:
        """
        This tests that the CPU control tensor is not freed before the CUDA kernel executes.
        """
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        with stream:
            a = BlockWork()
            a.block_current_stream()

            b = BlockWork()
            b.block_current_stream()

            # unblock b first though a is still blocking
            b.get_future().set_result(None)
            # delete b
            del b

            # a is still blocking so this doesn't resolve
            event = torch.cuda.Event()
            event.record()
            time.sleep(0.1)
            self.assertFalse(event.query())

            # unblock a
            a.get_future().set_result(None)

            stream.synchronize()
            self.assertTrue(event.query())


if __name__ == "__main__":
    run_tests()
