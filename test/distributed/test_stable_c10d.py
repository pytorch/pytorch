# Owner(s): ["oncall: distributed"]

import gc
import os
import sys
import sysconfig
import tempfile
import threading
import unittest
import weakref
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    install_cpp_extension,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def setUpModule():
    if not dist.is_available():
        raise unittest.SkipTest("requires distributed support")
    if sysconfig.get_config_var("Py_GIL_DISABLED") == 1:
        raise unittest.SkipTest("requires CPython limited API")
    try:
        from libtorch_agn_2_16 import _c10d  # noqa: F401
    except ImportError:
        install_cpp_extension(
            Path(__file__).resolve().parents[1]
            / "cpp_extensions"
            / "libtorch_agn_2_16_extension"
        )


def run_rank(rank, rendezvous):
    from libtorch_agn_2_16 import _c10d as api

    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    case = TestCase()
    pg = dist.new_group([0, 1], backend="gloo", timeout=timedelta(seconds=45))
    group = api.process_group(pg)

    def wait(work):
        try:
            case.assertTrue(api.wait(work, 30000))
            case.assertTrue(api.is_completed(work))
        finally:
            api.close_work(work)

    try:
        case.assertEqual(api.group_info(group), (rank, 2, "gloo"))
        with case.assertRaisesRegex(RuntimeError, "expected ProcessGroup"):
            api.process_group(None)
        single_group = dist.new_group(
            [0], backend="gloo", timeout=timedelta(seconds=45)
        )
        if rank == 0:
            single = api.process_group(single_group)
            try:
                case.assertEqual(api.group_info(single), (0, 1, "gloo"))
                only = torch.full((2,), 11.0)
                wait(api.allreduce(single, [only]))
                case.assertEqual(only, torch.full((2,), 11.0))
            finally:
                api.close_group(single)
                dist.destroy_process_group(single_group)
        value = torch.full((4,), float(rank + 1))
        wait(api.allreduce(group, [value]))
        case.assertEqual(value, torch.full((4,), 3.0))
        tensors = [torch.full((2,), float(rank)), torch.full((3,), float(3 - rank))]
        wait(api.allreduce_coalesced(group, tensors, 4))
        case.assertEqual(tensors, [torch.ones(2), torch.full((3,), 3.0)])
        value.fill_(rank + 7)
        wait(api.broadcast(group, [value], 1))
        case.assertEqual(value, torch.full((4,), 8.0))
        with case.assertRaisesRegex(RuntimeError, "invalid root rank"):
            api.broadcast(group, [value], 2)
        with case.assertRaisesRegex(RuntimeError, "invalid stable reduction"):
            api.allreduce(group, [value], 99)
        with case.assertRaisesRegex(RuntimeError, "nonempty"):
            api.allreduce(group, [])
        outputs = [torch.empty(2), torch.empty(2)]
        inp = torch.full((2,), float(rank))
        wait(api.allgather(group, inp, outputs))
        case.assertEqual(outputs, [torch.zeros(2), torch.ones(2)])
        wait(api.barrier(group))
        value.fill_(rank + 1)
        work = api.allreduce(group, [value])
        # Work keeps native group/tensor references after releasing stable wrappers.
        api.close_group(group)
        del pg
        gc.collect()
        wait(work)
        case.assertEqual(value, torch.full((4,), 3.0))
    finally:
        api.close_group(group)
        dist.destroy_process_group()


@instantiate_parametrized_tests
@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "requires Gloo")
class TestStableC10d(TestCase):
    @parametrize("last_owner", ["group", "work"])
    @parametrize("gil_held", [False, True])
    def test_python_group_lifetime(self, last_owner, gil_held):
        from libtorch_agn_2_16 import _c10d as api

        class PythonWork(dist.Work):
            def __init__(self, group):
                super().__init__()
                self.group = weakref.ref(group)

            def wait(self, timeout):
                return self.group() is not None

        class PythonGroup(dist.ProcessGroup):
            def __init__(self):
                super().__init__(0, 1)

            def getRank(self):
                return 7

            def allreduce(self, tensors, options):
                return PythonWork(self)

        group = PythonGroup()
        group_ref = weakref.ref(group)
        original = api.process_group(group)
        handle = api.copy_group(original)
        api.close_group(original, gil_held)
        work = None
        try:
            del group
            gc.collect()
            self.assertIsNotNone(group_ref())
            self.assertEqual(api.group_info(handle)[:2], (7, 1))
            if last_owner == "work":
                original_work = api.allreduce(handle, [torch.ones(1)])
                work = api.copy_work(original_work)
                api.close_work(original_work, gil_held)
                api.close_group(handle, gil_held)
                gc.collect()
                self.assertIsNotNone(group_ref())
                self.assertTrue(api.wait(work, 0, gil_held))
                api.close_work(work, gil_held)
            else:
                api.close_group(handle, gil_held)
            gc.collect()
            self.assertIsNone(group_ref())
        finally:
            if work is not None:
                api.close_work(work)
            api.close_group(handle)

    def test_two_rank_collectives(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                run_rank, args=(os.path.join(directory, "store"),), nprocs=2, join=True
            )


def run_final_owner_delete(device, last_owner):
    from libtorch_agn_2_16 import _c10d as api

    torch.cuda.set_device(device)
    store = dist.HashStore()
    backend = dist.ProcessGroupNCCL(store, 0, 1)
    parent = dist.ProcessGroup(store, 0, 1)
    parent._register_backend(
        torch.device(device), dist.ProcessGroup.BackendType.NCCL, backend
    )
    parent._set_default_backend(dist.ProcessGroup.BackendType.NCCL)
    value = torch.ones(1, device=device)
    parent.allreduce([value]).wait()
    torch.cuda.synchronize(device)
    group = parent.split_group([0])
    group._enable_collectives_timing()
    entered, finished, blocker = threading.Event(), threading.Event(), threading.Event()

    def hook(info):
        entered.set()
        blocker.wait()
        finished.set()

    group._register_on_completion_hook(hook)
    handle = api.process_group(group)
    work = api.allreduce(handle, [value])
    if not api.wait(work):
        raise AssertionError("collective was aborted")
    torch.cuda.synchronize(device)
    if not entered.wait(timeout=10):
        raise AssertionError("completion hook was not entered")
    del group
    gc.collect()
    if last_owner == "group":
        api.close_work(work)
        close, last = api.close_group, handle
    else:
        api.close_group(handle)
        close, last = api.close_work, work
    interval = sys.getswitchinterval()
    try:
        # The hook must acquire the GIL during destruction, not before the call.
        sys.setswitchinterval(120)
        blocker.set()
        close(last, True)
    finally:
        sys.setswitchinterval(interval)
    if not finished.is_set():
        raise AssertionError("group destruction did not finish the hook")
    parent.shutdown()


@unittest.skipUnless(dist.is_available() and dist.is_nccl_available(), "requires NCCL")
class TestStableC10dNCCL(TestCase):
    @parametrize("last_owner", ["group", "work"])
    def test_final_owner_with_gil(self, device, last_owner):
        process = mp.get_context("spawn").Process(
            target=run_final_owner_delete, args=(device, last_owner)
        )
        process.start()
        try:
            process.join(timeout=60)
            self.assertFalse(process.is_alive(), "group destruction deadlocked")
            self.assertEqual(process.exitcode, 0)
        finally:
            if process.is_alive():
                process.kill()
                process.join()

    @parametrize(
        "backend_name",
        ["ProcessGroupNCCL", "ProcessGroupNCCL2", "ProcessGroupNCCLLazy"],
    )
    def test_collective_cuda_graph(self, device, backend_name):
        from libtorch_agn_2_16 import _c10d as api

        store = dist.HashStore()
        backend = getattr(dist, backend_name)(store, 0, 1)
        group = dist.ProcessGroup(store, 0, 1)
        group._register_backend(
            torch.device(device), dist.ProcessGroup.BackendType.NCCL, backend
        )
        handle = api.process_group(group)
        graph = None
        work = None
        try:
            value = torch.zeros(4, device=device)
            group.allreduce([value]).wait()
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                value.add_(1)
                work = api.allreduce(handle, [value])
                self.assertTrue(api.wait(work))
            # Only the stable ProcessGroup and Work retain ownership during replay.
            del group, backend
            gc.collect()
            for expected in (1, 2, 3):
                graph.replay()
                torch.cuda.synchronize(device)
                self.assertEqual(value, torch.full_like(value, expected))
        finally:
            torch.cuda.synchronize(device)
            if graph is not None:
                graph.reset()
            if work is not None:
                api.close_work(work)
            api.close_group(handle)


instantiate_device_type_tests(TestStableC10dNCCL, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
