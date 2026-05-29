#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.


import torch
import torch.comms
from torch._C._comms import TorchCommBackend
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class DummyWork:
    def __init__(self) -> None:
        self.waited = False

    def wait(self) -> None:
        self.waited = True


class DummyPyBackend(TorchCommBackend):
    def __init__(self) -> None:
        super().__init__()
        self._rank = 0
        self._size = 1
        self._name = ""
        self._device = torch.device("cpu")
        self._initialized = False

    def init(self, device, name, options) -> None:
        self._device = device
        self._name = name
        self._initialized = True

    def finalize(self) -> None:
        self._initialized = False

    def get_rank(self) -> int:
        return self._rank

    def get_size(self) -> int:
        return self._size

    def get_backend_name(self) -> str:
        return "dummy_py"

    def get_comm_name(self) -> str:
        return self._name

    def send(self, tensor, dst, async_op):
        return None

    def recv(self, tensor, src, async_op):
        return None

    def broadcast(self, tensor, root, async_op):
        return None

    def all_reduce(self, tensor, op, async_op):
        return None

    def reduce(self, tensor, root, op, async_op):
        return None

    def all_gather(self, tensor_list, tensor, async_op):
        return None

    def all_gather_v(self, tensor_list, tensor, async_op):
        return None

    def all_gather_single(self, output, input, async_op):
        return None

    def reduce_scatter(self, output, input_list, op, async_op):
        return None

    def reduce_scatter_v(self, output, input_list, op, async_op):
        return None

    def reduce_scatter_single(self, output, input, op, async_op):
        return None

    def all_to_all_single(self, output, input, async_op):
        return None

    def all_to_all_v_single(self, output, input, output_splits, input_splits, async_op):
        return None

    def all_to_all(self, output_list, input_list, async_op):
        return None

    def barrier(self, async_op):
        return None

    def scatter(self, output, input_list, root, async_op):
        return None

    def gather(self, output_list, input, root, async_op):
        return None

    def split(self, ranks, name, options):
        backend = DummyPyBackend()
        backend._rank = 0
        backend._size = len(ranks)
        backend._name = name
        backend._initialized = True
        return backend


class AllReduceSumBackend(DummyPyBackend):
    def all_reduce(self, tensor, op, async_op):
        if op.type == torch.comms.RedOpType.SUM:
            tensor.mul_(self._size)
        return None


class AsyncAllReduceBackend(DummyPyBackend):
    def all_reduce(self, tensor, op, async_op):
        work = DummyWork()
        return work


@skipIfTorchDynamo("python backend collectives rely on eager dispatch, not traced")
class TestPythonBackend(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.comms.register_backend("dummy_py", DummyPyBackend)
        torch.comms.register_backend("sum_py", AllReduceSumBackend)
        torch.comms.register_backend("async_py", AsyncAllReduceBackend)

    def test_register_and_create(self) -> None:
        comm = torch.comms.new_comm("dummy_py", torch.device("cpu"), name="test_basic")
        self.assertEqual(comm.get_rank(), 0)
        self.assertEqual(comm.get_size(), 1)
        self.assertEqual(comm.get_backend(), "dummy_py")
        comm.finalize()

    def test_all_reduce_sync(self) -> None:
        comm = torch.comms.new_comm("dummy_py", torch.device("cpu"), name="test_ar")
        tensor = torch.ones(4)
        work = comm.all_reduce(tensor, torch.comms.ReduceOp.SUM, async_op=False)
        work.wait()
        self.assertTrue(work.is_completed())
        comm.finalize()

    def test_broadcast(self) -> None:
        comm = torch.comms.new_comm("dummy_py", torch.device("cpu"), name="test_bc")
        tensor = torch.ones(4)
        work = comm.broadcast(tensor, root=0, async_op=False)
        work.wait()
        self.assertTrue(work.is_completed())
        comm.finalize()

    def test_barrier(self) -> None:
        comm = torch.comms.new_comm("dummy_py", torch.device("cpu"), name="test_bar")
        work = comm.barrier(async_op=False)
        work.wait()
        self.assertTrue(work.is_completed())
        comm.finalize()

    def test_send_recv(self) -> None:
        comm = torch.comms.new_comm("dummy_py", torch.device("cpu"), name="test_sr")
        tensor = torch.ones(4)
        work = comm.send(tensor, dst=0, async_op=False)
        work.wait()
        self.assertTrue(work.is_completed())
        work = comm.recv(tensor, src=0, async_op=False)
        work.wait()
        self.assertTrue(work.is_completed())
        comm.finalize()

    def test_all_gather(self) -> None:
        comm = torch.comms.new_comm("dummy_py", torch.device("cpu"), name="test_ag")
        tensor = torch.ones(4)
        output = [torch.zeros(4)]
        work = comm.all_gather(output, tensor, async_op=False)
        work.wait()
        self.assertTrue(work.is_completed())
        comm.finalize()

    def test_reduce_scatter_single(self) -> None:
        comm = torch.comms.new_comm("dummy_py", torch.device("cpu"), name="test_rss")
        input_tensor = torch.ones(4)
        output_tensor = torch.zeros(4)
        work = comm.reduce_scatter_single(
            output_tensor, input_tensor, torch.comms.ReduceOp.SUM, async_op=False
        )
        work.wait()
        self.assertTrue(work.is_completed())
        comm.finalize()

    def test_scatter_gather(self) -> None:
        comm = torch.comms.new_comm("dummy_py", torch.device("cpu"), name="test_sg")
        output = torch.zeros(4)
        input_list = [torch.ones(4)]
        work = comm.scatter(output, input_list, root=0, async_op=False)
        work.wait()

        output_list = [torch.zeros(4)]
        input_t = torch.ones(4)
        work = comm.gather(output_list, input_t, root=0, async_op=False)
        work.wait()
        self.assertTrue(work.is_completed())
        comm.finalize()

    def test_all_to_all(self) -> None:
        comm = torch.comms.new_comm("dummy_py", torch.device("cpu"), name="test_a2a")
        output = [torch.zeros(4)]
        input_list = [torch.ones(4)]
        work = comm.all_to_all(output, input_list, async_op=False)
        work.wait()
        self.assertTrue(work.is_completed())
        comm.finalize()

    def test_custom_all_reduce_logic(self) -> None:
        comm = torch.comms.new_comm("sum_py", torch.device("cpu"), name="test_sum")
        tensor = torch.ones(4)
        comm.all_reduce(tensor, torch.comms.ReduceOp.SUM, async_op=False)
        expected = torch.ones(4)
        self.assertTrue(torch.equal(tensor, expected))
        comm.finalize()

    def test_async_work(self) -> None:
        comm = torch.comms.new_comm("async_py", torch.device("cpu"), name="test_async")
        tensor = torch.ones(4)
        work = comm.all_reduce(tensor, torch.comms.ReduceOp.SUM, async_op=True)
        self.assertFalse(work.is_completed())
        work.wait()
        self.assertTrue(work.is_completed())
        comm.finalize()

    def test_hooks_with_python_backend(self) -> None:
        comm = torch.comms.new_comm("dummy_py", torch.device("cpu"), name="test_hooks")
        pre_hook_calls = []

        def pre_hook(name, op_id, args) -> None:
            pre_hook_calls.append(name)

        handle = comm.register_pre_hook(pre_hook)
        tensor = torch.ones(4)
        comm.all_reduce(tensor, torch.comms.ReduceOp.SUM, async_op=False)
        self.assertEqual(len(pre_hook_calls), 1)
        handle.remove()
        comm.finalize()

    def test_multiple_comms(self) -> None:
        comm1 = torch.comms.new_comm("dummy_py", torch.device("cpu"), name="comm1")
        comm2 = torch.comms.new_comm("dummy_py", torch.device("cpu"), name="comm2")
        self.assertEqual(comm1.get_rank(), 0)
        self.assertEqual(comm2.get_rank(), 0)
        comm1.finalize()
        comm2.finalize()

    def test_split(self) -> None:
        comm = torch.comms.new_comm("dummy_py", torch.device("cpu"), name="test_split")
        sub = comm.split(ranks=[0], name="sub_comm")
        self.assertEqual(sub.get_rank(), 0)
        self.assertEqual(sub.get_size(), 1)
        sub.finalize()
        comm.finalize()


if __name__ == "__main__":
    run_tests()
