# Owner(s): ["module: c10d"]
import gc
import re
import threading
import unittest
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch._C import FileCheck
from torch._inductor.utils import fresh_cache, run_and_get_code, run_and_get_triton_code
from torch.distributed._functional_collectives import (
    all_gather_into_tensor_coalesced,
    all_gather_tensor,
    all_reduce,
    all_reduce_coalesced,
    all_to_all_single,
    AsyncCollectiveTensor,
    reduce_scatter_tensor,
    reduce_scatter_tensor_coalesced,
)
from torch.testing._internal.common_cuda import SM90OrLater
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (  # type: ignore[attr-defined]
    run_tests,
    skipIfRocm,
    TestCase,
)
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.testing._internal.inductor_utils import HAS_GPU


def load_test_module(name):
    import sys
    from importlib.machinery import SourceFileLoader
    from pathlib import Path
    from unittest import mock

    testdir = Path(__file__).absolute().parent.parent
    with mock.patch("sys.path", [*sys.path, str(testdir)]):
        return SourceFileLoader(
            name, str(testdir / f"{name.replace('.', '/')}.py")
        ).load_module()


AOTIRunnerUtil = load_test_module("inductor.test_aot_inductor_utils").AOTIRunnerUtil

import sys


if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)


@requires_nccl()
class TestWithNCCL(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        return 2

    @property
    def ranks(self) -> list[int]:
        return list(range(self.world_size))

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{self.rank}")

    def _init_process_group(self) -> None:
        torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)

    @skip_if_lt_x_gpu(2)
    def test_all_reduce_single(self) -> None:
        self._init_process_group()

        input = torch.full((10, 10), float(self.rank), device=self.device)
        output = torch.ops._c10d_functional.all_reduce(
            input,
            "avg",
            "default",
        )
        output = torch.ops._c10d_functional.wait_tensor(output)
        assert id(output) != id(input)
        expect = sum(self.ranks) / self.world_size
        assert output.eq(expect).all()

        # Test Python API and AsyncCollectiveTensor
        output = all_reduce(
            input,
            "avg",
            "default",
        )
        assert isinstance(output, AsyncCollectiveTensor)
        assert not output.completed
        assert output.eq(expect).all()
        assert output.completed

    @skip_if_lt_x_gpu(2)
    def test_all_reduce_single_(self) -> None:
        self._init_process_group()

        input = torch.full((10, 10), float(self.rank), device=self.device)
        output = torch.ops._c10d_functional.all_reduce_(
            input,
            "avg",
            "default",
        )
        output = torch.ops._c10d_functional.wait_tensor(output)
        assert id(output) == id(input)
        expect = sum(self.ranks) / self.world_size
        assert output.eq(expect).all()

    @skip_if_lt_x_gpu(2)
    def test_all_reduce_coalesced(self) -> None:
        self._init_process_group()

        inputs = [
            torch.full((i, i), float(self.rank * i), device=self.device)
            for i in range(10)
        ]
        outputs = torch.ops._c10d_functional.all_reduce_coalesced(
            inputs,
            "avg",
            "default",
        )
        for i, (output, input) in enumerate(zip(outputs, inputs)):
            output = torch.ops._c10d_functional.wait_tensor(output)
            assert id(output) != id(input)
            assert output.eq(sum(self.ranks) / self.world_size * i).all()

        # Test Python API and AsyncCollectiveTensor
        outputs = all_reduce_coalesced(
            inputs,
            "avg",
            "default",
        )
        for i, (output, input) in enumerate(zip(outputs, inputs)):
            assert not output.completed
            assert output.eq(sum(self.ranks) / self.world_size * i).all()
            assert output.completed

    @skip_if_lt_x_gpu(2)
    def test_all_reduce_coalesced_(self) -> None:
        self._init_process_group()

        inputs = [
            torch.full((i, i), float(self.rank * i), device=self.device)
            for i in range(10)
        ]
        outputs = torch.ops._c10d_functional.all_reduce_coalesced_(
            inputs,
            "avg",
            "default",
        )
        for i, (output, input) in enumerate(zip(outputs, inputs)):
            output = torch.ops._c10d_functional.wait_tensor(output)
            assert id(output) == id(input)
            assert output.eq(sum(self.ranks) / self.world_size * i).all()

    @skip_if_lt_x_gpu(2)
    def test_all_gather_into_tensor_single(self) -> None:
        self._init_process_group()

        input = torch.full((10, 10), float(self.rank), device=self.device)
        output = torch.ops._c10d_functional.all_gather_into_tensor(
            input,
            self.world_size,
            "default",
        )
        output = torch.ops._c10d_functional.wait_tensor(output)
        expect = torch.cat(
            [
                torch.full((10, 10), float(rank), device=self.device)
                for rank in self.ranks
            ]
        )
        assert torch.allclose(output, expect)
        assert output.eq(expect).all()

        # Test out-variant of all_gather_into_tensor
        output = torch.empty(expect.shape, device=self.device)
        output = torch.ops._c10d_functional.all_gather_into_tensor_out(
            input,
            self.world_size,
            "default",
            out=output,
        )
        output = torch.ops._c10d_functional.wait_tensor(output)
        assert torch.allclose(output, expect)
        assert output.eq(expect).all()

        # Test Python API and AsyncCollectiveTensor
        output = all_gather_tensor(
            input,
            0,
            "default",
        )
        assert isinstance(output, AsyncCollectiveTensor)
        assert not output.completed
        assert output.eq(expect).all()
        assert output.completed

    # https://github.com/pytorch/pytorch/issues/133421
    @skip_if_lt_x_gpu(2)
    def test_functional_collectives_inference_mode(self) -> None:
        self._init_process_group()

        with torch.inference_mode():
            input = torch.full((2, 2), float(self.rank), device=self.device)
            out1 = funcol.all_gather_tensor(
                input, gather_dim=0, group=torch.distributed.group.WORLD
            )
            out2 = out1.to(dtype=torch.bfloat16)
            # this tests that the call to .to() properly triggered a wait() on the AsyncCollectiveTensor
            self.assertTrue(type(out2) is torch.Tensor)
            self.assertEqual(
                out2,
                torch.tensor(
                    [[0, 0], [0, 0], [1, 1], [1, 1]],
                    device=self.device,
                    dtype=torch.bfloat16,
                ),
            )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    # https://github.com/pytorch/pytorch/issues/126338
    def test_inductor_dtypeview_memory_leak(self):
        self._init_process_group()

        def func(arg: torch.Tensor) -> torch.Tensor:
            ag0 = torch.ops._c10d_functional.all_gather_into_tensor.default(
                arg,
                self.world_size,
                "default",
            )
            ag0_view = torch.ops.aten.view.dtype(ag0, torch.int32)
            return funcol.wait_tensor(ag0_view)

        arg = torch.full(
            (10, 10),
            float(self.rank),
            device=self.device,
            dtype=torch.float32,
        )
        compiled = torch.compile(func)
        mem_usage = {}
        # check if the aten.view.dtype is compiled to aten.view.dtype
        code = run_and_get_triton_code(compiled, arg)
        (
            FileCheck()
            .check("torch.ops._c10d_functional.wait_tensor.default(aten.view.dtype")
            .run(code)
        )
        # check memory leak
        for i in range(1, 10):
            mem_usage[i] = torch.cuda.max_memory_allocated()
            compiled(arg)

        assert mem_usage[9] == mem_usage[8]

    @skip_if_lt_x_gpu(2)
    def test_all_gather_into_tensor_coalesced(self) -> None:
        self._init_process_group()

        inputs = [
            torch.full((10, 10), float(self.rank * i), device=self.device)
            for i in range(10)
        ]
        outputs = torch.ops._c10d_functional.all_gather_into_tensor_coalesced(
            inputs,
            self.world_size,
            "default",
        )
        expect = [
            torch.cat(
                [
                    torch.full((10, 10), float(rank) * i, device=self.device)
                    for rank in self.ranks
                ]
            )
            for i in range(10)
        ]
        for i, output in enumerate(outputs):
            output = torch.ops._c10d_functional.wait_tensor(output)
            assert output.eq(expect[i]).all()

        # Test Python API and AsyncCollectiveTensor
        outputs = all_gather_into_tensor_coalesced(
            inputs,
            "default",
        )
        for i, output in enumerate(outputs):
            assert not output.completed
            assert output.eq(expect[i]).all()
            assert output.completed

    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter_tensor_single(self) -> None:
        self._init_process_group()

        input = torch.tensor(self.ranks, device=self.device)
        output = torch.ops._c10d_functional.reduce_scatter_tensor(
            input,
            "avg",
            self.world_size,
            "default",
        )
        output = torch.ops._c10d_functional.wait_tensor(output)
        assert output.eq(self.rank).all()

        # Test Python API and AsyncCollectiveTensor
        output = reduce_scatter_tensor(
            input,
            "avg",
            0,
            "default",
        )
        assert isinstance(output, AsyncCollectiveTensor)
        assert not output.completed
        assert output.eq(self.rank).all()
        assert output.completed

    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter_tensor_coalesced(self) -> None:
        self._init_process_group()

        inputs = [torch.tensor(self.ranks, device=self.device) * i for i in range(10)]
        outputs = torch.ops._c10d_functional.reduce_scatter_tensor_coalesced(
            inputs,
            "avg",
            self.world_size,
            "default",
        )
        for i, output in enumerate(outputs):
            output = torch.ops._c10d_functional.wait_tensor(output)
            assert output.eq(self.rank * i).all()

        # Test Python API and AsyncCollectiveTensor
        outputs = reduce_scatter_tensor_coalesced(
            inputs,
            "avg",
            [0] * 10,
            "default",
        )
        for i, output in enumerate(outputs):
            assert not output.completed
            assert output.eq(self.rank * i).all()
            assert output.completed

    @skip_if_lt_x_gpu(2)
    def test_all_to_all_single(self) -> None:
        self._init_process_group()
        torch.cuda.set_device(self.device)

        torch.manual_seed(42)
        send_sz_matrix = torch.randint(0, 20, (self.world_size, self.world_size))

        input_split_sizes = send_sz_matrix[self.rank].tolist()
        output_split_sizes = send_sz_matrix[:, self.rank].tolist()
        input = torch.full((sum(input_split_sizes),), float(self.rank)).cuda()

        output = torch.ops._c10d_functional.all_to_all_single(
            input,
            output_split_sizes,
            input_split_sizes,
            "default",
        )
        output = torch.ops._c10d_functional.wait_tensor(output)
        expect = torch.cat(
            [
                torch.full((sz,), float(rank)).cuda()
                for rank, sz in enumerate(output_split_sizes)
            ]
        )
        assert output.eq(expect).all()

        # Test Python API and AsyncCollectiveTensor
        output = all_to_all_single(
            input, output_split_sizes, input_split_sizes, "default"
        )
        assert not output.completed
        assert output.eq(expect).all()
        assert output.completed

    @skip_if_lt_x_gpu(2)
    def test_broadcast(self) -> None:
        self._init_process_group()

        input = torch.full((10, 10), float(self.rank), device=self.device)
        output = torch.ops._c10d_functional.broadcast(
            input,
            1,
            "default",
        )
        output = torch.ops._c10d_functional.wait_tensor(output)
        assert id(output) != id(input)
        expect = 1
        assert output.eq(expect).all()

        # Test Python API and AsyncCollectiveTensor
        output = funcol.broadcast(
            input,
            1,
            "default",
        )
        assert isinstance(output, AsyncCollectiveTensor)
        assert not output.completed
        assert output.eq(expect).all()
        assert output.completed

    @skip_if_lt_x_gpu(2)
    def test_wait_tensor(self) -> None:
        self._init_process_group()

        input = torch.full((10, 10), float(self.rank), device=self.device)
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
        output = torch.ops._c10d_functional.all_reduce(
            input,
            "avg",
            "default",
        )
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 1)
        torch.ops._c10d_functional.wait_tensor(output)
        # `wait_tensor(output)` will pop the work from the work registry immediately
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)

    @skip_if_lt_x_gpu(2)
    def test_unwaited(self) -> None:
        # Verify that the process can terminate gracefully
        # even with unwaited tensors
        self._init_process_group()

        input = torch.full((10, 10), float(self.rank), device=self.device)
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
        torch.ops._c10d_functional.all_reduce(
            input,
            "avg",
            "default",
        )
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 1)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    @fresh_cache()
    def test_threading(self):
        self._init_process_group()
        device = torch.device(f"cuda:{self.rank}")

        def func(arg: torch.Tensor) -> torch.Tensor:
            buf0 = arg + 42
            ar0 = funcol.all_reduce(buf0, "avg", "0")
            ar0 = funcol.wait_tensor(ar0)
            return ar0 + 1

        arg = torch.rand(4, 4, device=device)
        func(arg)

        compiled = torch.compile(func, fullgraph=True)
        code = run_and_get_triton_code(compiled, arg)
        FileCheck().check("all_reduce_.default(buf0, 'avg', '0')").run(code)

        # Unless explicitly specified (e.g. in a custom runtime), the process
        # group registry is shared among all threads in a process. Here we
        # verify that a process group registered in main thread can be resolved
        # in a different thread.
        class TestThread(threading.Thread):
            def run(self):
                self.exc = None
                try:
                    func(arg)
                    compiled(arg)
                except BaseException as exc:  # noqa: B036
                    self.exc = exc

            def join(self):
                threading.Thread.join(self)
                if self.exc:
                    raise self.exc

        t = TestThread()
        t.start()
        t.join()

    @skipIfRocm
    @unittest.skipIf(
        not SM90OrLater,
        "_scaled_mm currently only supports sm>=90",
    )
    @skip_if_lt_x_gpu(2)
    @fresh_cache()
    def test_fixed_striding(self):
        self._init_process_group()

        def scale(t):
            scale = (
                torch.finfo(torch.float8_e4m3fn).max
                / t.abs().amax(dim=-1, keepdim=True).float()
            )
            t = t.mul(scale).to(torch.float8_e4m3fn)
            return t, scale

        def fp8_rowwise_backward(in_, w, out_grad):
            out_grad_fp8, scale_out_grad = scale(out_grad)
            w_fp8, scale_w = scale(w.t().contiguous())
            out_grad_fp8 = funcol.all_gather_tensor(
                out_grad_fp8, gather_dim=0, group=torch.distributed.group.WORLD
            )
            scale_out_grad = funcol.all_gather_tensor(
                scale_out_grad, gather_dim=0, group=torch.distributed.group.WORLD
            )
            in_grad = torch._scaled_mm(
                out_grad_fp8,
                w_fp8.t(),
                scale_a=scale_out_grad,
                scale_b=scale_w.t(),
                out_dtype=torch.bfloat16,
            )

            out_grad = funcol.all_gather_tensor(
                out_grad.t().contiguous(),
                gather_dim=0,
                group=torch.distributed.group.WORLD,
            )
            w_grad = out_grad @ in_

            return in_grad, w_grad

        m, n, k = 128, 256, 64
        in_ = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
        out_grad = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)

        eager_in_grad, eager_w_grad = fp8_rowwise_backward(in_, w, out_grad)
        compile_in_grad, compile_w_grad = torch.compile(fp8_rowwise_backward)(
            in_, w, out_grad
        )

        self.assertTrue(torch.allclose(compile_w_grad, eager_w_grad))


def dummy_init_pg() -> None:
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo", rank=0, world_size=1, store=dist.HashStore()
        )


class _DummyWork(dist.Work):
    def __init__(self, pg: "ProcessGroupDummy") -> None:
        super().__init__()
        self.pg = pg

    def wait(self, timeout: Optional[timedelta] = None) -> bool:
        self.pg.waits += 1
        return True

    def __del__(self):
        self.pg.dels += 1


class ProcessGroupDummy(dist.ProcessGroup):
    """
    This process group discards all data passed to it and returns success. This
    is intended for rare cases where we want to discard certain operations
    without modifying the underlying library.

    This PG only supports world_size of 1.
    """

    def __init__(self) -> None:
        super().__init__(0, 1)

        self._group_name = "dummy:dummy"

        self.waits = 0
        self.dels = 0

    def broadcast(self, tensor_list: list[torch.Tensor], opts: object) -> dist.Work:
        return _DummyWork(self)

    def allgather_into_tensor_coalesced(
        self,
        output_lists: list[torch.Tensor],
        input_list: list[torch.Tensor],
        opts: object,
    ) -> dist.Work:
        return _DummyWork(self)

    def allreduce(self, tensors: list[torch.Tensor], opts: object) -> dist.Work:
        return _DummyWork(self)

    def reduce_scatter_tensor_coalesced(
        self,
        outputTensors: list[torch.Tensor],
        inputTensors: list[torch.Tensor],
        opts: object,
    ) -> dist.Work:
        return _DummyWork(self)

    @property
    def group_name(self) -> str:
        if self._group_name is None:
            raise ValueError("ProcessGroup name not set")
        return self._group_name

    def _set_group_name(self, name: str) -> None:
        self._group_name = name

    def register(self) -> dist.ProcessGroup:
        def create_pg(
            prefix_store: dist.PrefixStore, rank: int, world_size: int, timeout: float
        ) -> dist.ProcessGroup:
            return self

        dist.Backend.register_backend(self.group_name, create_pg, devices=["cpu"])

        return dist.new_group(
            ranks=[0],
            backend=self.group_name,
            group_desc=self.group_name,
            timeout=timedelta(seconds=60.0),  # this timeout isn't used
        )


class PyWorkTest(TestCase):
    """
    Native functional collectives have some interesting interactions with
    PyProcessGroup due to Python reference counting and pybind trampoline
    classes with C++ types. This validates that PyProcessGroup and PyWork
    aren't getting prematurely freed.
    """

    def test_wait_tensor(self) -> None:
        wait_called = False

        class MyWork(dist.Work):
            def wait(self, _):
                nonlocal wait_called
                wait_called = True

        # check registration and implicit unregistration

        tensor = torch.rand(2, 2)
        work = MyWork()
        torch._C._distributed_c10d._register_work(tensor, work)

        # Force GC collection of the MyWork object, if we're not doing correct
        # reference counting we'll deadlock in wait_tensor.
        del work
        gc.collect()

        torch.ops._c10d_functional.wait_tensor(tensor)
        self.assertTrue(wait_called)

    def test_collectives(self) -> None:
        dummy_init_pg()

        pg = ProcessGroupDummy().register()

        x = torch.rand(2, 2)
        x = funcol.all_reduce(x, "sum", group=pg)
        gc.collect()
        self.assertEqual(pg.dels, 0)
        x.wait()
        self.assertEqual(pg.waits, 1)
        self.assertEqual(pg.dels, 1)

        x = torch.rand(2, 2)
        x = funcol.broadcast(x, 0, group=pg)
        gc.collect()
        self.assertEqual(pg.dels, 1)
        x.wait()
        self.assertEqual(pg.waits, 2)
        self.assertEqual(pg.dels, 2)

        x = torch.rand(2, 2)
        x = funcol.all_gather_tensor(x, 0, group=pg)
        gc.collect()
        self.assertEqual(pg.dels, 2)
        x.wait()
        self.assertEqual(pg.waits, 3)
        self.assertEqual(pg.dels, 3)

        x = torch.rand(2, 2)
        x = funcol.reduce_scatter_tensor(x, "sum", 0, group=pg)
        gc.collect()
        self.assertEqual(pg.dels, 3)
        x.wait()
        self.assertEqual(pg.waits, 4)
        self.assertEqual(pg.dels, 4)


def find_buffer_assignments(code):
    pattern = r"buf(\d+) = empty_strided_"
    matches = re.finditer(pattern, code)
    return tuple(f"buf{match.group(1)}" for match in matches)


class CompileTestCPU(TestCase):
    def setUp(self):
        super().setUp()

        if not dist.is_initialized():
            self.rank = 0
            self.world_size = 2

            store = FakeStore()
            dist.init_process_group(
                backend="fake",
                world_size=self.world_size,
                rank=self.rank,
                store=store,
            )

    def tearDown(self):
        dist.destroy_process_group()

    @fresh_cache()
    def _test_inductor_all_reduce_cpu(self, cpp_wrapper=False):
        def func(arg: torch.Tensor) -> torch.Tensor:
            buf0 = arg + 42
            ar0 = funcol.all_reduce(buf0, "avg", "0")
            ar0 = funcol.wait_tensor(ar0)
            return ar0

        arg = torch.rand(4, 4, device="cpu")
        with torch._inductor.config.patch({"cpp_wrapper": cpp_wrapper}):
            compiled = torch.compile(func)

            _, (code,) = run_and_get_code(compiled, arg)
            include_ops = (
                [
                    "aoti_torch_cpu__c10d_functional_all_reduce_",
                    "aoti_torch_cpu__c10d_functional_wait_tensor",
                ]
                if cpp_wrapper
                else [
                    "torch.ops._c10d_functional.all_reduce_.default",
                    "torch.ops._c10d_functional.wait_tensor.default",
                ]
            )
            for op in include_ops:
                self.assertIn(op, code)

        # Test aoti
        AOTIRunnerUtil.run(func, (arg,))
        torch.cpu.synchronize()

    def test_inductor_all_reduce_cpu(self):
        self._test_inductor_all_reduce_cpu(cpp_wrapper=False)
        self._test_inductor_all_reduce_cpu(cpp_wrapper=True)


class CompileTest(TestCase):
    def setUp(self):
        super().setUp()

        self.rank = 0
        self.world_size = 2
        torch.cuda.set_device("cuda:0")

        store = FakeStore()
        dist.init_process_group(
            backend="fake",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )

    def tearDown(self):
        dist.destroy_process_group()

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_cache()
    def test_inductor_all_reduce_single(self):
        def func(arg: torch.Tensor) -> torch.Tensor:
            buf0 = arg + 42
            # Expect in-place with inductor allocated buf
            ar0 = funcol.all_reduce(buf0, "avg", "0")
            ar0 = funcol.wait_tensor(ar0)
            # Expect no in-place with graph input
            ar1 = funcol.all_reduce(arg, "avg", "0")
            ar1 = funcol.wait_tensor(ar1)
            return ar0, ar1

        arg = torch.rand(4, 4, device="cuda")
        compiled = torch.compile(func)

        code = run_and_get_triton_code(compiled, arg)
        buf0, buf1 = find_buffer_assignments(code)
        (
            FileCheck()
            .check(f"{buf0} = empty")
            .check(f"{buf1} = empty")
            # Expect in-place with inductor allocated buf
            .check(f"torch.ops._c10d_functional.all_reduce_.default({buf0}")
            .check(f"torch.ops._c10d_functional.wait_tensor.default({buf0}")
            # Expect no in-place with graph input
            .check(f"torch.ops._c10d_functional.all_reduce_.default({buf1}")
            .check(f"torch.ops._c10d_functional.wait_tensor.default({buf1}")
            # Expect no extra copy on return
            .check(f"return ({buf0}, {buf1}, )")
            .run(code)
        )
        assert "= torch.ops._c10d_functional.wait_tensor.default" not in code

        # Test aoti
        AOTIRunnerUtil.run(func, (arg,))
        torch.cuda.synchronize()

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_cache()
    def test_inductor_all_reduce_coalesced(self):
        def func(args: list[torch.Tensor]) -> torch.Tensor:
            bufs = [arg + 42 for arg in args]
            # Expect in-place with inductor allocated buf
            ar0 = funcol.all_reduce_coalesced(bufs, "avg", "0")
            ar0 = [funcol.wait_tensor(out) for out in ar0]
            # Expect no in-place with graph input
            ar1 = funcol.all_reduce_coalesced(args, "avg", "0")
            ar1 = [funcol.wait_tensor(out) for out in ar1]
            return ar0, ar1

        args = [torch.rand(4, 4, device="cuda") for _ in range(2)]
        compiled = torch.compile(func)
        code = run_and_get_triton_code(compiled, args)
        buf0, buf1, buf2, buf3 = find_buffer_assignments(code)
        (
            FileCheck()
            .check(f"{buf0} = empty")
            .check(f"{buf1} = empty")
            .check(f"{buf2} = empty")
            .check(f"{buf3} = empty")
            # Expect in-place with inductor allocated buf
            .check(
                f"torch.ops._c10d_functional.all_reduce_coalesced_.default([{buf0}, {buf2}]"
            )
            # Expect no in-place with graph input ({buf1}, {buf3} are clones)
            .check(
                f"torch.ops._c10d_functional.all_reduce_coalesced_.default([{buf1}, {buf3}]"
            )
            .check(f"torch.ops._c10d_functional.wait_tensor.default({buf0}")
            .check(f"torch.ops._c10d_functional.wait_tensor.default({buf2}")
            .check(f"torch.ops._c10d_functional.wait_tensor.default({buf1}")
            .check(f"torch.ops._c10d_functional.wait_tensor.default({buf3}")
            # Expect no extra copy on return
            .check(f"return ({buf0}, {buf2}, {buf1}, {buf3}, )")
            .run(code)
        )
        assert "= torch.ops._c10d_functional.wait_tensor.default" not in code

        # Test aoti
        out = AOTIRunnerUtil.run(func, (args,))  # noqa: F841
        torch.cuda.synchronize()

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_cache()
    def test_inductor_inplace_op_on_view(self):
        def func(arg: torch.Tensor) -> torch.Tensor:
            buf0 = (arg + 10)[:2]
            ar0 = funcol.all_reduce(buf0, "avg", "0")
            ar0 = funcol.wait_tensor(ar0)
            return ar0

        arg = torch.rand(4, 4, device="cuda")
        compiled = torch.compile(func)

        code = run_and_get_triton_code(compiled, arg)
        (buf0,) = find_buffer_assignments(code)
        (
            FileCheck()
            .check(f"{buf0} = empty")
            # We always call .contiguous() on the input to all_reduce_,
            # so input will not be a view anymore.
            .check(f"torch.ops._c10d_functional.all_reduce_.default({buf0}")
            .check(f"torch.ops._c10d_functional.wait_tensor.default({buf0}")
            .check(f"return ({buf0}")
            .run(code)
        )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_cache()
    def test_inductor_all_reduce_non_contig_input(self):
        def func(arg: torch.Tensor) -> torch.Tensor:
            ar0 = funcol.all_reduce(arg, "avg", "0")
            ar0 = funcol.wait_tensor(ar0)
            # Expect allocation
            return ar0

        arg = torch.rand(4, 4, device="cuda").T
        compiled = torch.compile(func)

        code = run_and_get_triton_code(compiled, arg)
        # clone induced by non contig input
        assert "torch.ops._c10d_functional.wait_tensor.default" in code

        def func2(arg: torch.Tensor) -> torch.Tensor:
            torch.ops._c10d_functional.all_reduce_(arg, "avg", "0")
            return arg

        compiled = torch.compile(func)

        code = run_and_get_triton_code(compiled, arg)
        # clone induced by non contig input
        assert "torch.ops._c10d_functional.wait_tensor.default" in code

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_cache()
    def test_inductor_reuse_buffer_after_inplace_collective(self):
        def func(arg: torch.Tensor) -> torch.Tensor:
            # Expect allocation
            buf0 = arg + 42
            ar0 = funcol.all_reduce(buf0, "avg", "0")
            ar0 = funcol.wait_tensor(ar0)
            # Expect allocation
            buf1 = torch.mm(arg, ar0)
            # Expect buf0 to be reused
            buf2 = torch.mm(arg, buf1)
            return buf1, buf2

        arg = torch.rand(4, 4, device="cuda")
        compiled = torch.compile(func)
        code = run_and_get_triton_code(compiled, arg)
        buf0, buf1 = find_buffer_assignments(code)
        (
            FileCheck()
            # Expect allocation
            .check(f"{buf0} = empty")
            .check(f"torch.ops._c10d_functional.all_reduce_.default({buf0}")
            .check(f"torch.ops._c10d_functional.wait_tensor.default({buf0}")
            # Expect allocation
            .check(f"{buf1} = empty")
            .check(f"extern_kernels.mm(arg0_1, {buf0}, out={buf1}")
            # Expect {buf0} to be reused
            .check(f"buf8 = {buf0}; del {buf0}  # reuse")
            .check(f"extern_kernels.mm(arg0_1, {buf1}, out=buf8")
            # Expect no extra copy on return
            .check(f"return ({buf1}, buf8, )")
            .run(code)
        )
        assert "= torch.ops._c10d_functional.wait_tensor.default" not in code

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_cache()
    def test_inductor_all_gather_into_tensor_single(self):
        def func(arg: torch.Tensor) -> torch.Tensor:
            ag0 = funcol.all_gather_tensor(arg, 0, "0")
            ag0 = funcol.wait_tensor(ag0)
            return ag0

        arg = torch.rand(4, 4, device="cuda")
        compiled = torch.compile(func)
        code = run_and_get_triton_code(compiled, arg)
        (
            FileCheck()
            .check(
                "buf0 = torch.ops._c10d_functional.all_gather_into_tensor.default(arg0_1"
            )
            .check("torch.ops._c10d_functional.wait_tensor.default(buf0")
            # Expect no extra copy on return
            .check("return (buf0, )")
            .run(code)
        )
        assert "= torch.ops._c10d_functional.wait_tensor.default" not in code

        # Test aoti
        AOTIRunnerUtil.run(func, (arg,))
        torch.cuda.synchronize()

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_cache()
    def test_inductor_all_gather_into_tensor_coalesced(self):
        def func(args: list[torch.Tensor]) -> torch.Tensor:
            ag0 = funcol.all_gather_into_tensor_coalesced(args, "0")
            ag0 = [funcol.wait_tensor(out) for out in ag0]
            return ag0

        args = [torch.rand(4, 4, device="cuda") for _ in range(4)]
        compiled = torch.compile(func)
        code = run_and_get_triton_code(compiled, args)
        (
            FileCheck()
            .check(
                "buf0 = torch.ops._c10d_functional.all_gather_into_tensor_coalesced"
                ".default([arg3_1, arg2_1, arg1_1, arg0_1]"
            )
            .check("buf1 = buf0[0]")
            .check("buf2 = buf0[1]")
            .check("buf3 = buf0[2]")
            .check("buf4 = buf0[3]")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf1")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf2")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf3")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf4")
            # Expect no extra copy on return
            .check("return (buf1, buf2, buf3, buf4, )")
            .run(code)
        )

        # Test aoti
        out = AOTIRunnerUtil.run(func, (args,))  # noqa: F841
        torch.cuda.synchronize()

    @unittest.skipIf(not HAS_GPU, "This is a GPU test!")
    @fresh_cache()
    def test_wait_tensor(self):
        def func(arg: torch.Tensor) -> torch.Tensor:
            t = torch.ops._c10d_functional.all_reduce(arg, "avg", "0")
            return funcol.wait_tensor(t)

        # Test aoti
        arg = torch.rand(4, 4, device="cuda")
        compiled = torch.compile(func)
        code = run_and_get_triton_code(compiled, arg)
        (
            FileCheck()
            .check("torch.ops._c10d_functional.wait_tensor.default(buf0")
            .check("return (buf0, )")
            .run(code)
        )

        # Test aoti
        AOTIRunnerUtil.run(func, (arg,))
        torch.cuda.synchronize()

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_cache()
    def test_inductor_reduce_scatter_tensor_single(self):
        def func(arg: torch.Tensor) -> torch.Tensor:
            rs0 = funcol.reduce_scatter_tensor(arg, "avg", 0, "0")
            rs0 = funcol.wait_tensor(rs0)
            return rs0

        arg = torch.rand(4, 4, device="cuda")
        compiled = torch.compile(func)
        code = run_and_get_triton_code(compiled, arg)
        (
            FileCheck()
            .check(
                "buf0 = torch.ops._c10d_functional.reduce_scatter_tensor.default(arg0_1"
            )
            .check("torch.ops._c10d_functional.wait_tensor.default(buf0")
            # Expect no extra copy on return
            .check("return (buf0, )")
            .run(code)
        )

        # Test aoti
        AOTIRunnerUtil.run(func, (arg,))
        torch.cuda.synchronize()

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_cache()
    def test_inductor_reduce_scatter_tensor_coalesced(self):
        def func(args: list[torch.Tensor]) -> torch.Tensor:
            rs0 = funcol.reduce_scatter_tensor_coalesced(
                args, "avg", [0] * len(args), "0"
            )
            rs0 = [funcol.wait_tensor(out) for out in rs0]
            return rs0

        args = [torch.rand(4, 4, device="cuda") for _ in range(4)]
        compiled = torch.compile(func)
        code = run_and_get_triton_code(compiled, args)
        (
            FileCheck()
            .check(
                "buf0 = torch.ops._c10d_functional.reduce_scatter_tensor_coalesced"
                ".default([arg0_1, arg1_1, arg2_1, arg3_1]"
            )
            .check("buf1 = buf0[0]")
            .check("buf2 = buf0[1]")
            .check("buf3 = buf0[2]")
            .check("buf4 = buf0[3]")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf1")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf2")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf3")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf4")
            # Expect no extra copy on return
            .check("return (buf1, buf2, buf3, buf4, )")
            .run(code)
        )

        # Test aoti
        AOTIRunnerUtil.run(func, (args,))
        torch.cuda.synchronize()

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_cache()
    def test_inductor_all_to_all_single(self):
        def _tolist_with_constrain_as_size(tensor):
            lst = tensor.tolist()
            for elem in lst:
                torch._check_is_size(elem)
            return lst

        def func(
            input: torch.Tensor,
            output_split_sizes: torch.Tensor,
            input_split_sizes: torch.Tensor,
        ) -> torch.Tensor:
            output = funcol.all_to_all_single(
                input,
                _tolist_with_constrain_as_size(output_split_sizes),
                _tolist_with_constrain_as_size(input_split_sizes),
                "0",
            )
            return funcol.wait_tensor(output)

        torch.manual_seed(42)
        send_sz_matrix = torch.randint(0, 20, (self.world_size, self.world_size))

        input_split_sizes = send_sz_matrix[self.rank]
        output_split_sizes = send_sz_matrix[:, self.rank].contiguous()
        input = torch.full((input_split_sizes.sum().item(),), float(self.rank)).cuda()

        with torch._dynamo.config.patch(
            dynamic_shapes=True,
            capture_dynamic_output_shape_ops=True,
            capture_scalar_outputs=True,
        ):
            compiled = torch.compile(func, dynamic=True)
            code = run_and_get_triton_code(
                compiled, input, output_split_sizes, input_split_sizes
            )
        (
            FileCheck()
            .check_regex(
                "torch.ops._c10d_functional.all_to_all_single.default\\("
                "arg\\d+_\\d+, \\[u\\d+, u\\d+\\], \\[u\\d+, u\\d+\\]"
            )
            .check("torch.ops._c10d_functional.wait_tensor.default(")
            .run(code)
        )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_cache()
    def test_inductor_broadcast(self):
        def func(arg: torch.Tensor) -> torch.Tensor:
            buf0 = arg + 42
            # Expect in-place with inductor allocated buf
            br0 = funcol.broadcast(buf0, 1, "0")
            br0 = funcol.wait_tensor(br0)
            # Expect no in-place with graph input
            br1 = funcol.broadcast(arg, 0, "0")
            br1 = funcol.wait_tensor(br1)
            return br0, br1

        arg = torch.rand(4, 4, device="cuda")
        compiled = torch.compile(func)

        code = run_and_get_triton_code(compiled, arg)
        buf0, buf1 = find_buffer_assignments(code)
        (
            FileCheck()
            .check(f"{buf0} = empty")
            .check(f"buf1 = {buf0}")
            .check(f"{buf1} = empty")
            # Expect in-place with inductor allocated buf
            .check("torch.ops._c10d_functional.broadcast_.default(buf1")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf1")
            # Expect no in-place with graph input (buf5 is a clone)
            .check(f"torch.ops._c10d_functional.broadcast_.default({buf1}")
            .check(f"torch.ops._c10d_functional.wait_tensor.default({buf1}")
            # Expect no extra copy on return
            .check(f"return (buf1, {buf1}, )")
            .run(code)
        )

        # Test aoti
        AOTIRunnerUtil.run(func, (arg,))
        torch.cuda.synchronize()

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_cache()
    def test_ranks_and_tag(self):
        def func(arg: torch.Tensor) -> torch.Tensor:
            buf0 = arg + 42
            # Expect in-place with inductor allocated buf
            ar0 = funcol.all_reduce(buf0, "avg", [0, 1], "")
            ar0 = funcol.wait_tensor(ar0)
            # Expect no in-place with graph input
            ar1 = funcol.all_reduce(arg, "avg", [0, 1], "")
            ar1 = funcol.wait_tensor(ar1)
            return ar0, ar1

        arg = torch.rand(4, 4, device="cuda")
        compiled = torch.compile(func, fullgraph=True)

        code = run_and_get_triton_code(compiled, arg)
        (FileCheck().check("all_reduce_.default(buf0, 'avg', '0')").run(code))


if __name__ == "__main__":
    run_tests()
