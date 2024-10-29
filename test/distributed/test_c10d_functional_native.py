# Owner(s): ["module: c10d"]
import threading
import unittest
from typing import List

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch._C import FileCheck
from torch._inductor.utils import fresh_inductor_cache, run_and_get_triton_code
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
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (  # type: ignore[attr-defined]
    run_tests,
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
    def ranks(self) -> List[int]:
        return list(range(self.world_size))

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{self.rank}")

    def _init_process_group(self) -> None:
        # Allow testing aoti after torch.compile
        torch._inductor.config.triton.store_cubin = True
        torch._inductor.config.debug = True

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
        output = torch.ops._c10d_functional.all_reduce(
            input,
            "avg",
            "default",
        )
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 1)

    @skip_if_lt_x_gpu(2)
    def test_py_work(self) -> None:
        self._init_process_group()

        wait_called = False

        class MyWork(dist.Work):
            def wait(self, _):
                nonlocal wait_called
                wait_called = True

        tensor = torch.rand(2, 2)
        torch._C._distributed_c10d._register_work(tensor, MyWork())
        torch.ops._c10d_functional.wait_tensor(tensor)
        self.assertTrue(wait_called)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    @fresh_inductor_cache()
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
                except BaseException as exc:
                    self.exc = exc

            def join(self):
                threading.Thread.join(self)
                if self.exc:
                    raise self.exc

        t = TestThread()
        t.start()
        t.join()


class CompileTest(TestCase):
    def setUp(self):
        # Allow testing aoti after torch.compile
        torch._inductor.config.triton.store_cubin = True
        torch._inductor.config.debug = True

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
    @fresh_inductor_cache()
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
        (
            FileCheck()
            .check("buf0 = empty")
            .check("buf7 = empty")
            # Expect in-place with inductor allocated buf
            .check("torch.ops._c10d_functional.all_reduce_.default(buf0")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf0")
            # Expect no in-place with graph input (buf5 is a clone)
            .check("torch.ops._c10d_functional.all_reduce_.default(buf7")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf7")
            # Expect no extra copy on return
            .check("return (buf0, buf7, )")
            .run(code)
        )
        assert "= torch.ops._c10d_functional.wait_tensor.default" not in code

        # Test aoti
        out = AOTIRunnerUtil.run("cuda", func, (arg,))
        torch.cuda.synchronize()

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_inductor_cache()
    def test_inductor_all_reduce_coalesced(self):
        def func(args: List[torch.Tensor]) -> torch.Tensor:
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
        (
            FileCheck()
            .check("buf0 = empty")
            .check("buf5 = empty")
            .check("buf1 = empty")
            .check("buf6 = empty")
            # Expect in-place with inductor allocated buf
            .check(
                "torch.ops._c10d_functional.all_reduce_coalesced_"
                ".default([buf0, buf1]"
            )
            # Expect no in-place with graph input (buf5, buf6 are clones)
            .check(
                "torch.ops._c10d_functional.all_reduce_coalesced_"
                ".default([buf5, buf6]"
            )
            .check("torch.ops._c10d_functional.wait_tensor.default(buf0")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf1")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf5")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf6")
            # Expect no extra copy on return
            .check("return (buf0, buf1, buf5, buf6, )")
            .run(code)
        )
        assert "= torch.ops._c10d_functional.wait_tensor.default" not in code

        # Test aoti
        out = AOTIRunnerUtil.run("cuda", func, (args,))
        torch.cuda.synchronize()

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_inductor_cache()
    def test_inductor_inplace_op_on_view(self):
        def func(arg: torch.Tensor) -> torch.Tensor:
            buf0 = (arg + 10)[:2]
            ar0 = funcol.all_reduce(buf0, "avg", "0")
            ar0 = funcol.wait_tensor(ar0)
            return ar0

        arg = torch.rand(4, 4, device="cuda")
        compiled = torch.compile(func)

        code = run_and_get_triton_code(compiled, arg)
        (
            FileCheck()
            .check("buf0 = empty")
            # We always call .contiguous() on the input to all_reduce_,
            # so input will not be a view anymore.
            .check("torch.ops._c10d_functional.all_reduce_.default(buf0")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf0")
            .check("return (buf0")
            .run(code)
        )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_inductor_cache()
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
    @fresh_inductor_cache()
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
        (
            FileCheck()
            # Expect allocation
            .check("buf0 = empty")
            .check("torch.ops._c10d_functional.all_reduce_.default(buf0")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf0")
            # Expect allocation
            .check("buf7 = empty")
            .check("extern_kernels.mm(arg0_1, buf0, out=buf7")
            # Expect buf0 to be reused
            .check("buf8 = buf0; del buf0  # reuse")
            .check("extern_kernels.mm(arg0_1, buf7, out=buf8")
            # Expect no extra copy on return
            .check("return (buf7, buf8, )")
            .run(code)
        )
        assert "= torch.ops._c10d_functional.wait_tensor.default" not in code

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_inductor_cache()
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
        out = AOTIRunnerUtil.run("cuda", func, (arg,))
        torch.cuda.synchronize()

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_inductor_cache()
    def test_inductor_all_gather_into_tensor_coalesced(self):
        def func(args: List[torch.Tensor]) -> torch.Tensor:
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
        out = AOTIRunnerUtil.run("cuda", func, (args,))
        torch.cuda.synchronize()

    @unittest.skipIf(not HAS_GPU, "This is a GPU test!")
    @fresh_inductor_cache()
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
        out = AOTIRunnerUtil.run("cuda", func, (arg,))
        torch.cuda.synchronize()

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_inductor_cache()
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
        out = AOTIRunnerUtil.run("cuda", func, (arg,))
        torch.cuda.synchronize()

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_inductor_cache()
    def test_inductor_reduce_scatter_tensor_coalesced(self):
        def func(args: List[torch.Tensor]) -> torch.Tensor:
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
        AOTIRunnerUtil.run("cuda", func, (args,))
        torch.cuda.synchronize()

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_inductor_cache()
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
    @fresh_inductor_cache()
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
        (
            FileCheck()
            .check("buf0 = empty")
            .check("buf7 = empty")
            # Expect in-place with inductor allocated buf
            .check("torch.ops._c10d_functional.broadcast_.default(buf0")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf0")
            # Expect no in-place with graph input (buf5 is a clone)
            .check("torch.ops._c10d_functional.broadcast_.default(buf7")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf7")
            # Expect no extra copy on return
            .check("return (buf0, buf7, )")
            .run(code)
        )

        # Test aoti
        out = AOTIRunnerUtil.run("cuda", func, (arg,))
        torch.cuda.synchronize()

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_inductor_cache()
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
