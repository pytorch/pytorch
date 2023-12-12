# Owner(s): ["module: inductor"]
import io
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from typing import Callable, List, Optional, Tuple

import torch
from torch import multiprocessing as mp
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import reset_rng_state
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.autotune_process import (
    BenchmarkRequest,
    CUDA_VISIBLE_DEVICES,
    TuningProcessPool,
)
from torch._inductor.codegen.cuda.cutlass_utils import (
    cuda_standalone_runner_compile_command,
    CUDACompileSourceCapturingContext,
)
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import Buffer, FixedLayout
from torch._inductor.kernel.mm_plus_mm import aten_mm_plus_mm
from torch._inductor.select_algorithm import (
    AlgorithmSelectorCache,
    ChoiceCaller,
    TritonTemplateCaller,
)
from torch._inductor.util_autotuning_log_parser import AutotuningLogParser
from torch._inductor.utils import cache_dir, run_and_get_code
from torch._inductor.virtualized import V
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import SM75OrLater, SM90OrLater
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    skipIfRocm,
)

from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA

torch.set_float32_matmul_precision("high")
if HAS_CUDA:
    torch.cuda.memory._set_allocator_settings("expandable_segments:False")

_CUTLASS_DIR = os.path.join(os.path.dirname(__file__), "../../third_party/cutlass/")


def _get_path_without_sccache() -> str:
    """
    Get the PATH environment variable without sccache.
    """
    path_envs = os.environ.get("PATH", "").split(":")
    path_envs = [env for env in path_envs if "/opt/cache/bin" not in env]
    return ":".join(path_envs)


def benchmark_choice(choice, args, out, expected_out, timings):
    result = choice.benchmark(*args, out=out)
    if expected_out is not None:
        torch.testing.assert_close(out, expected_out)

    timings.copy_(torch.tensor(result))


class FailChoiceCaller(ChoiceCaller):
    def benchmark(self, *args, out):
        raise RuntimeError("This choice caller will always throw")


@instantiate_parametrized_tests
class TestMaxAutotune(TestCase):
    def _create_buffer(self, name, shape):
        return Buffer(name, FixedLayout(torch.device("cuda:0"), torch.float32, shape))

    def cuda_test_compile_standalone_runner(
        self, src, name=None, do_compile=True, do_run=True, log=sys.stderr
    ):
        if name is None:
            name = "test_cuda_kernel"
            src_name = name + ".cu"

        target_dir = Path(cache_dir()) / self.id()
        target_dir.mkdir(parents=True, exist_ok=True)
        src_path = target_dir / src_name
        exe_path = target_dir / name
        print(f"Wrote CUDA Kernel source to {src_path}", file=log)
        src_path.write_text(src)
        compile_command = cuda_standalone_runner_compile_command(src_path, exe_path)
        print(f"Compilation command would be {compile_command}", file=log)
        print(compile_command, file=log)
        if do_compile:
            print(f"Compiling {src_path} to {exe_path}", file=log)
            print(
                subprocess.check_output(
                    compile_command.split(" "),
                    stderr=subprocess.STDOUT,
                    env=os.environ,
                    encoding="utf-8",
                ),
                file=log,
            )
            print(
                f"Wrote standalone CUDA Kernel executable to {exe_path}, source to {src_path}",
                file=log,
            )
            if do_run:
                print(f"Running {exe_path}", file=log)
                print(
                    subprocess.check_output(
                        [str(exe_path)],
                        stderr=subprocess.STDOUT,
                        env=os.environ,
                        encoding="utf-8",
                    ),
                    file=log,
                )
        return compile_command, src_path, exe_path

    def test_benchmark_choice_in_subproc(self):
        gm = make_fx(
            lambda: torch.zeros(2, 3)
        )()  # a dummy graph to construct the GraphLowering
        graph = GraphLowering(gm)

        # the graph handler is neede to create benchmark example value below
        with V.set_graph_handler(graph):
            buf1 = self._create_buffer("mat1", (2, 3))
            buf2 = self._create_buffer("mat2", (3, 2))
            buf3 = self._create_buffer("mat3", (2, 3))
            buf4 = self._create_buffer("mat4", (3, 2))

            layout = FixedLayout(torch.device("cuda:0"), torch.float32, (2, 2))

            mat1 = AlgorithmSelectorCache.benchmark_example_value(buf1)
            mat2 = AlgorithmSelectorCache.benchmark_example_value(buf2)
            mat3 = AlgorithmSelectorCache.benchmark_example_value(buf3)
            mat4 = AlgorithmSelectorCache.benchmark_example_value(buf4)

            out = AlgorithmSelectorCache.benchmark_example_value(layout)
            # expected_out = (mat1 @ mat2) + (mat3 @ mat4)
            expected_out = None

            choice = aten_mm_plus_mm.bind((buf1, buf2, buf3, buf4), layout)
            # use a tensor since the mutation to a python list in a sub process
            # is not synced back to the parent process
            timings = torch.zeros(3, dtype=torch.float32)
            ctx = mp.get_context("spawn")
            child = ctx.Process(
                target=benchmark_choice,
                args=(choice, (mat1, mat2, mat3, mat4), out, expected_out, timings),
            )
            child.start()
            child.join()
            self.assertEqual(0, child.exitcode)
            print(f"timings is {timings}, out {out}, expected_out {expected_out}")

    def test_benchmark_choice_fail_in_subproc(self):
        gm = make_fx(
            lambda: torch.zeros(2, 3)
        )()  # a dummy graph to construct the GraphLowering
        graph = GraphLowering(gm)

        # the graph handler is neede to create benchmark example value below
        with V.set_graph_handler(graph):
            buf1 = self._create_buffer("mat1", (2, 3))
            buf2 = self._create_buffer("mat2", (3, 2))
            buf3 = self._create_buffer("mat3", (2, 3))
            buf4 = self._create_buffer("mat4", (3, 2))

            layout = FixedLayout(torch.device("cuda:0"), torch.float32, (2, 2))

            mat1 = AlgorithmSelectorCache.benchmark_example_value(buf1)
            mat2 = AlgorithmSelectorCache.benchmark_example_value(buf2)
            mat3 = AlgorithmSelectorCache.benchmark_example_value(buf3)
            mat4 = AlgorithmSelectorCache.benchmark_example_value(buf4)

            out = AlgorithmSelectorCache.benchmark_example_value(layout)
            expected_out = (mat1 @ mat2) + (mat3 @ mat4)

            choice = FailChoiceCaller("fail_choice_caller", [], None)

            # use a tensor since python list is not synced back
            timings = torch.zeros(3, dtype=torch.float32)
            ctx = mp.get_context("spawn")
            child = ctx.Process(
                target=benchmark_choice,
                args=(choice, (mat1, mat2, mat3, mat4), out, expected_out, timings),
            )
            child.start()
            child.join()
            self.assertNotEqual(0, child.exitcode)

    @parametrize("autotune_in_subproc", (True, False))
    @parametrize("autotune_multi_device", (True, False))
    def test_max_autotune_mm_plus_mm(self, autotune_in_subproc, autotune_multi_device):
        """
        This crash previously due to a triton issue: https://github.com/openai/triton/issues/1298 .
        With autotuning in subprocess, we don't crash anymore.
        """
        m, n, k = 2048, 1536, 64

        def mm_plus_mm(a, b, c, d):
            return a @ b + c @ d

        a = torch.randn(m, k).cuda()
        b = torch.randn(k, n).cuda()
        c = torch.randn(m, k).cuda()
        d = torch.randn(k, n).cuda()

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": autotune_in_subproc,
                "autotune_multi_device": autotune_multi_device,
            }
        ):
            torch.compile(mm_plus_mm)(a, b, c, d)

    @parametrize("dynamic", (False, True))
    def test_max_autotune_mm_plus_mm_zero_size_input(self, dynamic):
        """
        Make sure autotuning mm_plus_mm with zero-size input works without crashes.
        """
        m, n, k = 0, 1536, 64

        def mm_plus_mm(a, b, c, d):
            return a @ b + c @ d

        a = torch.randn(m, k).cuda()
        b = torch.randn(k, n).cuda()
        c = torch.randn(m, k).cuda()
        d = torch.randn(k, n).cuda()

        with config.patch({"max_autotune": True}):
            torch.compile(mm_plus_mm, dynamic=dynamic)(a, b, c, d)

    @parametrize("dynamic", (False, True))
    def test_max_autotune_regular_mm(self, dynamic: bool):
        """
        Make sure autotuning mm in sub processes work without crashes.
        """

        def mm(a, b):
            a = torch.sin(a)
            return a @ b

        a = torch.randn(100, 10).cuda()
        b = torch.randn(10, 100).cuda()

        with config.patch({"max_autotune": True, "autotune_in_subproc": True}):
            torch.compile(mm, dynamic=dynamic)(a, b)

    @parametrize("dynamic", (False, True))
    def test_max_autotune_regular_mm_zero_size_input(self, dynamic: bool):
        """
        Make sure autotuning mm with zero-size input works without crashes.
        """

        def mm(a, b):
            a = torch.sin(a)
            return a @ b

        a = torch.randn(0, 10).cuda()
        b = torch.randn(10, 100).cuda()

        with config.patch({"max_autotune": True}):
            torch.compile(mm, dynamic=dynamic)(a, b)

    # TODO: Enable dynamic test cases when dynamic support is added.
    @unittest.skipIf(not SM75OrLater, "need sm_75")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    @parametrize("dynamic", (False,))
    @parametrize("max_autotune_gemm_backends", ("CUTLASS", "ATen,Triton,CUTLASS"))
    @unittest.mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_max_autotune_cutlass_backend_regular_mm(
        self, dynamic: bool, max_autotune_gemm_backends: str
    ):
        """
        Make sure autotuning mm in sub processes work without crashes.
        """

        if max_autotune_gemm_backends == "CUTLASS" and torch.version.hip:
            return

        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

        def mm(a, b):
            return a @ b

        a = torch.randn(100, 10).cuda().half()
        b = torch.randn(10, 100).cuda().half()

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": False,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cuda.cutlass_dir": _CUTLASS_DIR,
                "cuda.cutlass_max_profiling_configs": 2,
            }
        ):
            mm_compiled = torch.compile(mm, dynamic=dynamic)
            Y_compiled = mm_compiled(a, b)
            Y = mm(a, b)
            torch.testing.assert_close(Y_compiled, Y)

    def _test_max_autotune_cutlass_backend_epilogue_fusion(
        self,
        dynamic: bool = False,
        max_autotune_gemm_backends: str = "CUTLASS",
        mixed_precision=False,
        fp16=True,
        expected_fuse_count=1,
        mm: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        with_bias=False,
        bias_broadcast=[False, False],
        with_aux=False,
        m=1024,
        n=1024,
        k=1024,
        max_profiling_configs=4,
        batch_size=None,
        evt_only=True,
        aux_shape: Optional[Tuple[int]] = None,
        config_override={},
    ):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            mixed_precision
        )

        # Note: The ops that are available
        # also depend on the alignment of the shapes
        # so if these shapes don't all align to at least 8 elements
        # it can happen that no Cutlass 3.x op is available
        # that allows fusions
        if batch_size is None:
            a = torch.randn(m, k).mul(1.0 / 32).cuda()
            b = torch.randn(k, n).mul(1.0 / 32).cuda()
            if with_bias:
                bias = torch.randn(m, n).mul(1.0 / 32).cuda()
        else:
            a = torch.randn(batch_size, m, k).mul(1.0 / 32).cuda()
            b = torch.randn(batch_size, k, n).mul(1.0 / 32).cuda()
            if with_bias:
                bias_m = m
                bias_n = n
                if bias_broadcast[0]:
                    bias_m = 1
                if bias_broadcast[1]:
                    bias_n = 1
                bias = torch.randn(batch_size, bias_m, bias_n).mul(1.0 / 32).cuda()
            if with_aux:
                if aux_shape is None:
                    aux_shape = (batch_size, m, n)
                aux = torch.randn(*aux_shape).mul(1.0 / 32).cuda()
        if fp16:
            a = a.half()
            b = b.half()
            if with_bias:
                bias = bias.half()
            if with_aux:
                aux = aux.half()
        args = [a, b]
        if with_bias:
            args.append(bias)
        if with_aux:
            args.append(aux)
        conf_patch = {
            "max_autotune": True,
            "autotune_in_subproc": False,
            "benchmark_fusion": False,
            "max_autotune_gemm_backends": max_autotune_gemm_backends,
            "cuda.cutlass_dir": _CUTLASS_DIR,
            "cuda.cutlass_max_profiling_configs": max_profiling_configs,
            "cuda.cutlass_prefer_evt_capable_ops": evt_only,
            "cuda.version": "12.1",  # required to enable the Kernels we need
        }
        conf_patch.update(config_override)
        with config.patch(conf_patch):
            counters["inductor"]["cuda_epilogue_fusion_counter"] = 0
            Y = mm(*args)
            mm_jit = torch.compile(mm, dynamic=dynamic)
            Y_compiled = mm_jit(*args)
            actual_count = counters["inductor"]["cuda_epilogue_fusion_counter"]
            torch.testing.assert_close(Y_compiled, Y, atol=1e-2, rtol=1e-2)
            if expected_fuse_count is not None:
                assert (
                    actual_count == expected_fuse_count
                ), f"Expected fuse count of {expected_fuse_count} but got {actual_count}"

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_simple_fusion_fp16(self):
        def mm(a, b):
            return (a @ b) * 3.0

        #  The pointwise ops seem to be pre-fused into a single Pointwise
        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=False, fp16=True, expected_fuse_count=1, mm=mm
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_simple_mm_fp16_standalone_runner_large(self):
        def mm(a, b):
            return a @ b

        source_capture = CUDACompileSourceCapturingContext()
        with source_capture:
            #  The pointwise ops seem to be pre-fused into a single Pointwise
            self._test_max_autotune_cutlass_backend_epilogue_fusion(
                mixed_precision=False,
                fp16=True,
                expected_fuse_count=0,
                mm=mm,
                m=1024 * 10,
                n=1024 * 10,
                k=2048,
                batch_size=10,
                max_profiling_configs=4,
                config_override={"cuda.generate_test_runner": True},
            )

        self.cuda_test_compile_standalone_runner(
            source_capture.sources[-1], do_run=True
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_other_backends_simple_mm_fp16_standalone_runner_large(self):
        def mm(a, b):
            return (a @ b) * 3.0

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=False,
            fp16=True,
            expected_fuse_count=None,
            mm=mm,
            m=1024 * 10,
            n=1024 * 10,
            k=2048,
            batch_size=10,
            max_profiling_configs=4,
            max_autotune_gemm_backends="ATen,Triton,CUTLASS",
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_simple_fusion_fp16_unaligned_aten_fallback(
        self,
    ):
        def mm(a, b):
            return (a @ b) * 3.0

        #  For this, we have no Cutlass Kernel because of alignment constraints.
        # We expect the ATen fallback to be used, but this will not register a fusion,
        # therefore expected_fuse_count=0
        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=False,
            fp16=True,
            expected_fuse_count=0,
            mm=mm,
            m=1024,
            n=160,
            k=257,
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_double_matmul(
        self,
    ):
        def mm(a, b):
            return ((a @ b).T @ a) - 4.5

        #  For this, we have no Cutlass Kernel because of alignment constraints.
        # We expect the ATen fallback to be used, but this will not register a fusion,
        # therefore expected_fuse_count=0
        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=False,
            fp16=True,
            expected_fuse_count=1,
            mm=mm,
            m=128,
            n=128,
            k=128,
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_simple_fusion_fp32(self):
        def mm(a, b):
            return (a @ b) * 3.0

        #  The pointwise ops seem to be pre-fused into a single Pointwise
        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=False,
            fp16=False,
            expected_fuse_count=0,
            mm=mm,
            m=1024,
            n=512,
            k=72,
            batch_size=6,
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_simple_fusion_fp16_fp32acc(self):
        def mm(a, b):
            return (a @ b) * 3.0

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=True, fp16=True, expected_fuse_count=1, mm=mm
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_chained_fusion_fp16(self):
        def mm(a, b):
            return (a @ b) * 3.3 - 1.234

        #  The pointwise ops seem to be pre-fused into a single Pointwise
        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=False, fp16=True, expected_fuse_count=1, mm=mm
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_one_additional_input_simple(self):
        def mm(a, b, c):
            return (a @ b) - 3.3 * c

        with config.patch(
            {
                "trace.enabled": True,
                "trace.output_code": True,
                "trace.debug_dir": os.path.abspath(
                    os.path.dirname(__file__) + "/../../tmp/test_code/"
                ),
            }
        ):
            #  The pointwise ops seem to be pre-fused into a single Pointwise
            self._test_max_autotune_cutlass_backend_epilogue_fusion(
                mixed_precision=False,
                fp16=True,
                expected_fuse_count=1,
                mm=mm,
                with_bias=True,
                m=2048,
                n=512,
                k=4096,
            )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_one_additional_input_random_mask(self):
        def mm(a, b, c):
            return (a @ b) * 1.5 + c

        source_capture = CUDACompileSourceCapturingContext()
        with source_capture:
            try:
                self._test_max_autotune_cutlass_backend_epilogue_fusion(
                    mixed_precision=False,
                    fp16=True,
                    expected_fuse_count=1,
                    mm=mm,
                    with_bias=True,
                    m=64,
                    n=128,
                    k=128,
                    batch_size=1,
                    max_profiling_configs=1,
                )
            finally:
                self.cuda_test_compile_standalone_runner(
                    source_capture.sources[-1], do_run=True
                )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_two_additional_inputs_random_mask(self):
        def mm(a, b, c, aux):
            return ((a @ b) * torch.relu(c) * 1.0) + aux

        source_capture = CUDACompileSourceCapturingContext()
        with source_capture:
            try:
                self._test_max_autotune_cutlass_backend_epilogue_fusion(
                    mixed_precision=False,
                    fp16=True,
                    expected_fuse_count=1,
                    mm=mm,
                    with_bias=True,
                    with_aux=True,
                    m=256,
                    n=128,
                    k=128,
                    batch_size=1,
                    max_profiling_configs=1,
                )
            finally:
                self.cuda_test_compile_standalone_runner(
                    source_capture.sources[-1], do_run=True
                )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_two_additional_inputs_random_mask_broadcasted(
        self,
    ):
        def mm(a, b, c, aux):
            # aux = torch.reshape(aux, (1, 1, aux.shape[0]))
            return ((a @ b) * torch.relu(c) * 1.0) + aux

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=False,
            fp16=True,
            expected_fuse_count=1,
            mm=mm,
            with_bias=True,
            with_aux=True,
            m=128,
            n=128,
            k=64,
            batch_size=2,
            max_profiling_configs=3,
            aux_shape=(128,),
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_one_additional_input_random_mask_batched(
        self,
    ):
        def mm(a, b, c):
            return (a @ b) * torch.relu(c - 0.02)

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=False,
            fp16=True,
            expected_fuse_count=1,
            mm=mm,
            with_bias=True,
            m=2048,
            n=512,
            k=4096,
            batch_size=100,
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_chained_fusion_fp16_fp32acc(self):
        def mm(a, b):
            return (a @ b) * 3.3 - 1.234

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=True, fp16=True, expected_fuse_count=1, mm=mm
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_relu_fusion_fp16(self):
        def mm(a, b):
            return torch.nn.functional.relu((a @ b) * 3.3 - 1.234)

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=False, fp16=True, expected_fuse_count=1, mm=mm
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_simple_addmm(self):
        def mm(a, b, bias):
            return torch.addmm(bias, a, b)

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=False,
            fp16=True,
            expected_fuse_count=0,
            mm=mm,
            with_bias=True,
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_simple_addmm_broadcasted_row(self):
        def mm(a, b, bias):
            bias_slice = bias[0:1, :]
            return torch.addmm(bias_slice, a, b)

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=False,
            fp16=True,
            expected_fuse_count=0,
            mm=mm,
            with_bias=True,
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_simple_addmm_broadcasted_col(self):
        def mm(a, b, bias):
            bias_slice = bias[:, 0:1]
            return torch.addmm(bias_slice, a, b)

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=False,
            fp16=True,
            expected_fuse_count=0,
            mm=mm,
            with_bias=True,
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_simple_addmm_broadcasted_row2(self):
        def mm(a, b, bias):
            bias_slice = bias[0:1, :]
            return (a @ b) + bias_slice

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=False,
            fp16=True,
            expected_fuse_count=0,
            mm=mm,
            with_bias=True,
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_simple_addmm_broadcasted_col2(self):
        def mm(a, b, bias):
            bias_slice = bias[:, 0:1]
            return (a @ b) + bias_slice

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=False,
            fp16=True,
            expected_fuse_count=0,
            mm=mm,
            with_bias=True,
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_simple_aux_broadcasted_row(self):
        def mm(a, b, bias):
            bias_slice = bias[0:1, :]
            return (a @ b) - torch.relu(bias_slice)

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=False,
            fp16=True,
            expected_fuse_count=1,
            mm=mm,
            with_bias=True,
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_simple_aux_broadcasted_col(self):
        def mm(a, b, bias):
            bias_slice = bias[:, 0:1]
            return (a @ b) - torch.relu(bias_slice * 1.2)

        # This is not fused, because the aux argument requires to be contiguous in the non-broadcasted dim
        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=False,
            fp16=True,
            expected_fuse_count=0,
            mm=mm,
            with_bias=True,
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_simple_aux_broadcasted_col2(self):
        def mm(a, b, bias):
            return (a @ b) - torch.relu(bias * 1.2)

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=False,
            fp16=True,
            expected_fuse_count=1,
            mm=mm,
            with_bias=True,
            bias_broadcast=(True, False),
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_simple_aux_broadcasted_row2(self):
        def mm(a, b, bias):
            return (a @ b) - torch.relu(bias * 1.2)

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=False,
            fp16=True,
            expected_fuse_count=1,
            mm=mm,
            with_bias=True,
            bias_broadcast=(False, True),
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_relu_fusion_fp16_fp32acc(self):
        def mm(a, b):
            return torch.nn.functional.relu((a @ b) * 3.3 - 1.234)

        #  The pointwise ops seem to be pre-fused into a single Pointwise
        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=True, fp16=True, expected_fuse_count=1, mm=mm
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_relu6_fusion_fp16_fp32acc(self):
        def mm(a, b):
            return torch.clamp(torch.nn.functional.relu(a @ b), max=6.0)

        #  The pointwise ops seem to be pre-fused into a single Pointwise
        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=True, fp16=True, expected_fuse_count=1, mm=mm
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_simple_bmm(self):
        def mm(a, b):
            return torch.bmm(a, b)

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=False,
            fp16=True,
            expected_fuse_count=0,
            mm=mm,
            with_bias=False,
            batch_size=10,
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_simple_baddbmm(self):
        def mm(a, b, bias):
            return torch.baddbmm(bias, a, b)

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=True,
            fp16=True,
            expected_fuse_count=0,
            mm=mm,
            with_bias=True,
            batch_size=31,
            evt_only=False,
        )

    # TODO: Enable support for typecasts in fused epilogues
    # @unittest.skipIf(not SM90OrLater, "need sm_90")
    # @unittest.skipIf(torch.version.hip, "HIP not supported")
    # @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    # def test_max_autotune_cutlass_backend_no_fusion_dtype_mismatch(self):
    #     def mm(a, b):
    #         # this should not be fused, since the output dtype is different from the matmul dtype
    #         return (a @ b).to(torch.float32) * 0.00001
    #
    #     self._test_max_autotune_cutlass_backend_epilogue_fusion(
    #         mixed_precision=True, fp16=True, expected_fuse_count=0, mm=mm
    #     )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_shape_dependent_normalization_fusion(self):
        def mm(a, b):
            return (a @ b) / b.size(1)

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=True, fp16=True, expected_fuse_count=1, mm=mm
        )

    # TODO: Enable dynamic test cases when dynamic support is added.
    @unittest.skipIf(not SM75OrLater, "need sm_75")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    @parametrize("dynamic", (False,))
    @parametrize("max_autotune_gemm_backends", ("CUTLASS", "ATen,Triton,CUTLASS"))
    @parametrize("only_evt_capable", (True, False))
    @unittest.mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_max_autotune_cutlass_backend_mm_bias(
        self,
        dynamic: bool = False,
        only_evt_capable: bool = False,
        max_autotune_gemm_backends: str = "CUTLASS",
    ):
        """
        Make sure autotuning mm in sub processes work without crashes.
        """

        if max_autotune_gemm_backends == "CUTLASS" and torch.version.hip:
            return

        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

        def mm(a, b, bias):
            return torch.nn.functional.linear(a, b, bias)

        a = torch.randn(2048, 4096).cuda().half()
        bias = torch.randn(2048).cuda().half()

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": False,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cuda.cutlass_dir": _CUTLASS_DIR,
                "cuda.cutlass_max_profiling_configs": 2,
                "cuda.cutlass_prefer_evt_capable_ops": only_evt_capable,
            }
        ):
            Y = mm(a, a, bias)
            Y_compiled = torch.compile(mm, dynamic=dynamic)(a, a, bias)
            torch.testing.assert_close(Y_compiled, Y, atol=1e-1, rtol=1e-1)

    @parametrize("dynamic", (False, True))
    def test_max_autotune_addmm(self, dynamic=False):
        """
        Make sure autotuning addmm in sub processes work without crashes.
        """

        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

        def addmm(x, a, b):
            return torch.addmm(x, a, b)

        x = torch.randn(100).cuda()
        a = torch.randn(100, 10).cuda()
        b = torch.randn(10, 100).cuda()
        with config.patch({"max_autotune": True, "autotune_in_subproc": True}):
            Y_compiled = torch.compile(addmm, dynamic=dynamic)(x, a, b)
            Y = addmm(x, a, b)
            torch.testing.assert_close(Y_compiled, Y, atol=1e-2, rtol=1e-2)

    @parametrize("dynamic", (False, True))
    def test_max_autotune_addmm_zero_size_input(self, dynamic):
        """
        Make sure autotuning addmm with zero-size input works without crashes.
        """

        def addmm(x, a, b):
            return torch.addmm(x, a, b)

        x = torch.randn(100).cuda()
        a = torch.randn(0, 10).cuda()
        b = torch.randn(10, 100).cuda()
        with config.patch({"max_autotune": True}):
            torch.compile(addmm, dynamic=dynamic)(x, a, b)

    # TODO: Enable dynamic test cases when dynamic support is added.
    @unittest.skipIf(not SM75OrLater, "need sm_75")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    @parametrize("dynamic", (False,))
    @parametrize("max_autotune_gemm_backends", ("CUTLASS", "ATen,Triton,CUTLASS"))
    @parametrize("cutlass_prefer_evt_capable_ops", (True, False))
    @unittest.mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_max_autotune_cutlass_backend_addmm(
        self,
        dynamic=False,
        max_autotune_gemm_backends="CUTLASS",
        cutlass_prefer_evt_capable_ops=True,
    ):
        """
        Make sure autotuning addmm in sub processes work without crashes.
        """

        if max_autotune_gemm_backends == "CUTLASS" and torch.version.hip:
            return

        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

        def addmm(x, a, b, alpha, beta):
            return torch.addmm(x, a, b, alpha=alpha, beta=beta)

        def compare_results(
            m: int, k: int, n: int, alpha: float, beta: float, x_shape: List[int]
        ) -> None:
            x = torch.randn(x_shape).cuda().half()
            a = torch.randn(m, k).cuda().half()
            b = torch.randn(k, n).cuda().half()
            y_expected = addmm(x, a, b, alpha, beta)

            compiled_fn = torch.compile(addmm, dynamic=dynamic)
            y = compiled_fn(x, a, b, alpha, beta)
            torch.testing.assert_close(y, y_expected)

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": False,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cuda.cutlass_dir": _CUTLASS_DIR,
                "cuda.cutlass_max_profiling_configs": 2,
                "cuda.cutlass_prefer_evt_capable_ops": cutlass_prefer_evt_capable_ops,
            }
        ):
            # No broadcast
            compare_results(4096, 25728, 2048, 2.0, 0.4, [4096, 2048])
            # Broadcast first dim.
            compare_results(4096, 25728, 2048, 2.0, 0.4, [2048])
            # Broadcast last dim.
            if not SM90OrLater and max_autotune_gemm_backends == "CUTLASS":
                with self.assertRaisesRegex(RuntimeError, "No choices to select"):
                    # CUTLASS2 doesn't support Bias last-dim broadcast.
                    compare_results(4096, 25728, 2048, 2.0, 0.4, [4096, 1])
            else:
                compare_results(4096, 25728, 2048, 2.0, 0.4, [4096, 1])

    @skipIfRocm
    def test_autotune_conv1x1(self):
        # Assuming input has 3 channels and we want to produce 16 channels as output
        conv1x1 = (
            torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
            .to(memory_format=torch.channels_last)
            .cuda()
        )

        # Example input tensor: batch size = 4, channels = 3, height = 32, width = 32
        # The memory format is set to `channels_last`
        input_tensor = (
            torch.randn(4, 3, 32, 32)
            .contiguous(memory_format=torch.channels_last)
            .cuda()
        )

        with config.patch(
            {"max_autotune": True, "max_autotune_gemm_backends": "TRITON"}
        ):

            @torch.compile()
            def foo(mod, x):
                return mod(x)

            with torch.no_grad():
                out, code = run_and_get_code(foo, conv1x1, input_tensor)

            FileCheck().check_not("extern_kernels.convolution").run(code[0])
            self.assertEqual(conv1x1(input_tensor), out, atol=1e-2, rtol=0)

    def test_cat_addmm(self):
        def fn(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
            return torch.cat(
                [
                    torch.addmm(a, b, c),
                    torch.addmm(b, c, a),
                ],
                1,
            )

        args = [
            torch.randn(4, 4, device="cuda"),
            torch.randn(4, 4, device="cuda"),
            torch.randn(4, 4, device="cuda"),
        ]
        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "Triton",
            }
        ):
            expected = fn(*args)
            actual = torch.compile(fn)(*args)
            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    def test_autotuning_log_parser(self):
        example_log_lines = """{"backend": "extern", "kernel_call_name": "extern_kernels.bmm", "cuda_device_name": "NVIDIA H100", "cuda_device_count": 8, "input_nodes": [{"name": "bmm", "type": "StorageBox", "layout": "FlexibleLayout('cuda', torch.float32, size=[6, 1024, 72], stride=[73728, 72, 1])", "dtype": "torch.float32", "device": "cuda:0", "stride": "[73728, 72, 1]", "numel": "442368", "data": {"name": "bmm", "type": "ComputedBuffer", "layout": "FlexibleLayout('cuda', torch.float32, size=[6, 1024, 72], stride=[73728, 72, 1])", "dtype": "torch.float32", "device": "cuda:0", "stride": "[73728, 72, 1]", "numel": "442368", "data": {"name": "bmm", "type": "Pointwise", "dtype": "torch.float32", "device": "cuda:0", "numel": "442368"}}}, {"name": "bmm", "type": "StorageBox", "layout": "FlexibleLayout('cuda', torch.float32, size=[6, 72, 512], stride=[36864, 512, 1])", "dtype": "torch.float32", "device": "cuda:0", "stride": "[36864, 512, 1]", "numel": "221184", "data": {"name": "bmm", "type": "ComputedBuffer", "layout": "FlexibleLayout('cuda', torch.float32, size=[6, 72, 512], stride=[36864, 512, 1])", "dtype": "torch.float32", "device": "cuda:0", "stride": "[36864, 512, 1]", "numel": "221184", "data": {"name": "bmm", "type": "Pointwise", "dtype": "torch.float32", "device": "cuda:0", "numel": "221184"}}}], "autotuning_time": 3.0995004177093506, "benchmark_result": 0.0147141041931385}
{"tile_shape": "(64, 32, 64)", "num_stages": 2, "num_warps": 4, "allow_tf32": "True", "acc_type": "tl.float32", "backend": "Triton", "grid": "(128, 6, 1)", "cuda_device_name": "NVIDIA H100", "cuda_device_count": 8, "input_nodes": [{"name": "bmm", "type": "StorageBox", "layout": "FlexibleLayout('cuda', torch.float32, size=[6, 1024, 72], stride=[73728, 72, 1])", "dtype": "torch.float32", "device": "cuda:0", "stride": "[73728, 72, 1]", "numel": "442368", "data": {"name": "bmm", "type": "ComputedBuffer", "layout": "FlexibleLayout('cuda', torch.float32, size=[6, 1024, 72], stride=[73728, 72, 1])", "dtype": "torch.float32", "device": "cuda:0", "stride": "[73728, 72, 1]", "numel": "442368", "data": {"name": "bmm", "type": "Pointwise", "dtype": "torch.float32", "device": "cuda:0", "numel": "442368"}}}, {"name": "bmm", "type": "StorageBox", "layout": "FlexibleLayout('cuda', torch.float32, size=[6, 72, 512], stride=[36864, 512, 1])", "dtype": "torch.float32", "device": "cuda:0", "stride": "[36864, 512, 1]", "numel": "221184", "data": {"name": "bmm", "type": "ComputedBuffer", "layout": "FlexibleLayout('cuda', torch.float32, size=[6, 72, 512], stride=[36864, 512, 1])", "dtype": "torch.float32", "device": "cuda:0", "stride": "[36864, 512, 1]", "numel": "221184", "data": {"name": "bmm", "type": "Pointwise", "dtype": "torch.float32", "device": "cuda:0", "numel": "221184"}}}], "autotuning_time": 3.0995004177093506, "benchmark_result": 0.01217997465145754}
{"tile_shape": "(64, 32, 128)", "num_stages": 3, "num_warps": 4, "allow_tf32": "True", "acc_type": "tl.float32", "backend": "Triton", "grid": "(64, 6, 1)", "cuda_device_name": "NVIDIA H100", "cuda_device_count": 8, "input_nodes": [{"name": "bmm", "type": "StorageBox", "layout": "FlexibleLayout('cuda', torch.float32, size=[6, 1024, 72], stride=[73728, 72, 1])", "dtype": "torch.float32", "device": "cuda:0", "stride": "[73728, 72, 1]", "numel": "442368", "data": {"name": "bmm", "type": "ComputedBuffer", "layout": "FlexibleLayout('cuda', torch.float32, size=[6, 1024, 72], stride=[73728, 72, 1])", "dtype": "torch.float32", "device": "cuda:0", "stride": "[73728, 72, 1]", "numel": "442368", "data": {"name": "bmm", "type": "Pointwise", "dtype": "torch.float32", "device": "cuda:0", "numel": "442368"}}}, {"name": "bmm", "type": "StorageBox", "layout": "FlexibleLayout('cuda', torch.float32, size=[6, 72, 512], stride=[36864, 512, 1])", "dtype": "torch.float32", "device": "cuda:0", "stride": "[36864, 512, 1]", "numel": "221184", "data": {"name": "bmm", "type": "ComputedBuffer", "layout": "FlexibleLayout('cuda', torch.float32, size=[6, 72, 512], stride=[36864, 512, 1])", "dtype": "torch.float32", "device": "cuda:0", "stride": "[36864, 512, 1]", "numel": "221184", "data": {"name": "bmm", "type": "Pointwise", "dtype": "torch.float32", "device": "cuda:0", "numel": "221184"}}}], "autotuning_time": 3.0995004177093506, "benchmark_result": 0.01012531017369727}
"""  # noqa: B950
        example_input = io.StringIO(example_log_lines)
        try:
            parser = AutotuningLogParser(example_input)
            records = list(parser.get_records())
            assert len(records) == 3
            expected_json_records = '[{"backend": "extern", "name": "ATen", "problem_hash": "ef279ba8a6739a088efd1fdca60f0c31", "kernel_schedule": "", "tile_shape": "[]", "benchmark_result": 0.0147141041931385, "device": "unknown", "cuda_device_name": "NVIDIA H100", "problem_shape_MNK": [1024, 512, 72], "A_size": [6, 1024, 72], "A_stride": [73728, 72, 1], "A_type": "row_major", "A_dtype": "float32", "A_shape": ["6", "1024", "!72"], "B_size": [6, 72, 512], "B_stride": [36864, 512, 1], "B_type": "row_major", "B_dtype": "float32", "B_shape": ["6", "72", "!512"], "M": 1024, "N": 512, "K": 72}, {"backend": "Triton", "name": "ATen", "problem_hash": "ef279ba8a6739a088efd1fdca60f0c31", "kernel_schedule": "", "tile_shape": "(64, 32, 64)", "benchmark_result": 0.01217997465145754, "device": "unknown", "cuda_device_name": "NVIDIA H100", "problem_shape_MNK": [1024, 512, 72], "A_size": [6, 1024, 72], "A_stride": [73728, 72, 1], "A_type": "row_major", "A_dtype": "float32", "A_shape": ["6", "1024", "!72"], "B_size": [6, 72, 512], "B_stride": [36864, 512, 1], "B_type": "row_major", "B_dtype": "float32", "B_shape": ["6", "72", "!512"], "M": 1024, "N": 512, "K": 72}, {"backend": "Triton", "name": "ATen", "problem_hash": "ef279ba8a6739a088efd1fdca60f0c31", "kernel_schedule": "", "tile_shape": "(64, 32, 128)", "benchmark_result": 0.01012531017369727, "device": "unknown", "cuda_device_name": "NVIDIA H100", "problem_shape_MNK": [1024, 512, 72], "A_size": [6, 1024, 72], "A_stride": [73728, 72, 1], "A_type": "row_major", "A_dtype": "float32", "A_shape": ["6", "1024", "!72"], "B_size": [6, 72, 512], "B_stride": [36864, 512, 1], "B_type": "row_major", "B_dtype": "float32", "B_shape": ["6", "72", "!512"], "M": 1024, "N": 512, "K": 72}]'  # noqa: B950
            assert json.dumps(records) == expected_json_records, "Record parser failed"
            pd = None
            # The rest of this test requires pandas, which might not be installed.
            try:
                import pandas

                pd = pandas
            except ImportError:
                pass
            if pd is not None:
                df = parser.get_dataframe()
                assert len(df) == 3
                assert set(df.columns) == {
                    "backend",
                    "name",
                    "problem_hash",
                    "kernel_schedule",
                    "tile_shape",
                    "benchmark_result",
                    "device",
                    "cuda_device_name",
                    "problem_shape_MNK",
                    "A_size",
                    "A_stride",
                    "A_type",
                    "A_dtype",
                    "A_shape",
                    "B_size",
                    "B_stride",
                    "B_type",
                    "B_dtype",
                    "B_shape",
                    "Bias_shape",
                    "M",
                    "N",
                    "K",
                }
                analysis = parser.get_analysis()
                assert set(analysis.columns) == {
                    "problem_hash",
                    "M",
                    "N",
                    "K",
                    "A_shape",
                    "B_shape",
                    "Bias_shape",
                    "tile_shape",
                    "backend",
                    "kernel_schedule",
                    "benchmark_result",
                }
        finally:
            if example_input:
                example_input.close()

    def test_triton_template_with_epilogues_and_dynamic_shape(self):
        def fn(
            x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, mul: torch.Tensor
        ) -> torch.Tensor:
            return (
                torch.nn.functional.relu(
                    torch.matmul(torch.transpose(x, 0, 1), torch.transpose(w, 0, 1))
                    + bias
                )
                * mul
            )

        M0 = 5
        M1 = 8
        K = 4
        N = 3
        w = torch.rand(N, K).cuda().half()
        b = torch.rand(N).cuda().half()

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": "Triton",
            }
        ):
            compiled_fn = torch.compile(
                fn, fullgraph=True, dynamic=True, mode="max-autotune-no-cudagraphs"
            )

            x0 = torch.rand(K, M0).cuda().half()
            mul0 = torch.rand(M0, N).cuda().half()
            y0 = compiled_fn(x0, w, b, mul0)
            y0_expected = fn(x0, w, b, mul0)
            torch.testing.assert_close(y0, y0_expected)

            x1 = torch.rand(K, M1).cuda().half()
            mul1 = torch.rand(M1, N).cuda().half()
            y1 = compiled_fn(x1, w, b, mul1)
            y1_expected = fn(x1, w, b, mul1)
            torch.testing.assert_close(y1, y1_expected)

    @config.patch(
        benchmark_kernel=True,
        fallback_random=True,
        max_autotune_gemm=True,
    )
    @parametrize("device", ("cpu", "cuda"))
    def test_matmul_dropout(self, device):
        def fwd(a, b):
            x = a @ b
            x = torch.nn.functional.dropout(x, 0.1)
            return x

        def fn(a, b):
            x = fwd(a, b).sum()
            x.backward()
            return a.grad

        N = 128
        a = torch.randn(N, N, device=device, requires_grad=True)
        b = torch.randn(N, N, device=device)

        opt_fn = torch.compile(fn)
        reset_rng_state()
        ref = fn(a, b)
        reset_rng_state()
        act = opt_fn(a, b)

        if N <= 8:
            print(f"ref\n{ref}\nact\n{act}")
        torch.testing.assert_close(ref, act, atol=1e-1, rtol=1e-1)


class TestBenchmarkRequest(BenchmarkRequest):
    def __init__(
        self, value: float, multi_device: bool, parent_visible_devices: Optional[str]
    ) -> None:
        self.value = value
        self.multi_device = multi_device
        self.parent_visible_devices = parent_visible_devices

    def benchmark(
        self, *input_tensors: torch.Tensor, output_tensor: Optional[torch.Tensor] = None
    ) -> float:
        # Verify that the visible devices env var is set correctly. If multi-device
        # auto-tuning is disabled, the visible devices should be unmanipulated from
        # the parent process. If multi-device auto-tuning is enabled, the visible
        # devices should be a _single_ valid device number. Note that we can't perform
        # this validation directly from the test body because benchmarks execute in a
        # separate process. If the check fails, however, the test will detect the
        # failure by virtue of not receiving the expected result back.
        visible_devices = os.environ.get(CUDA_VISIBLE_DEVICES)
        if not self.multi_device:
            assert visible_devices == self.parent_visible_devices
        else:
            valid_devices = self.parent_visible_devices.split(",")
            assert visible_devices in valid_devices

        return self.value


class TestTritonTemplateCaller(TritonTemplateCaller):
    def __init__(self, bmreq: TestBenchmarkRequest):
        self.bmreq = bmreq

    def __str__(self) -> str:
        return "test"


class TestTuningProcess(TestCase):
    def test_tuning_pool_crash(self):
        # Use only one device/subprocess so we test the process restarts
        # and is usable after a "crash".
        with config.patch({"autotune_multi_device": False}):
            tuning_pool = TuningProcessPool()
            tuning_pool.initialize()

            # First force the tuning process to "crash" by setting a bogus
            # string for the expected visible devices.
            bmreq = TestBenchmarkRequest(3.14, False, "invalid")
            choice = TestTritonTemplateCaller(bmreq)

            timings = tuning_pool.benchmark([choice])
            self.assertTrue(choice in timings)
            self.assertEqual(timings[choice], float("inf"))

            # Then send another request and make sure the sub-process
            # has restarted and is operational. 'valid_devices' expected
            # to be None because autotune_multi_device is off.
            choice.bmreq.parent_visible_devices = os.environ.get(CUDA_VISIBLE_DEVICES)

            timings = tuning_pool.benchmark([choice])
            self.assertTrue(choice in timings)
            self.assertEqual(timings[choice], bmreq.value)

            tuning_pool.terminate()

    def test_tuning_pool_multiple_devices(self):
        with config.patch({"autotune_multi_device": True}):
            # Adapt the test to the available devices (and whether CUDA_VISIBLE_DEVICES
            # is already set in the environment); use a subset of the available devices
            # to ensure only the subset are visible to the sub-processes.
            if CUDA_VISIBLE_DEVICES in os.environ:
                visible_devices = os.environ[CUDA_VISIBLE_DEVICES].split(",")
            else:
                visible_devices = [str(d) for d in range(torch.cuda.device_count())]

            parent_visible_devices = ",".join(visible_devices[-2:])
            os.environ[CUDA_VISIBLE_DEVICES] = parent_visible_devices

            tuning_pool = TuningProcessPool()
            tuning_pool.initialize()

            choice1 = TestTritonTemplateCaller(
                TestBenchmarkRequest(3.14, True, parent_visible_devices),
            )
            choice2 = TestTritonTemplateCaller(
                TestBenchmarkRequest(2.718, True, parent_visible_devices),
            )

            timings = tuning_pool.benchmark([choice1, choice2])
            self.assertEqual(timings[choice1], choice1.bmreq.value)
            self.assertEqual(timings[choice2], choice2.bmreq.value)

            tuning_pool.terminate()


if __name__ == "__main__":
    from torch._inductor.utils import is_big_gpu

    # Set env to make it work in CI.
    if HAS_CUDA and HAS_CPU and is_big_gpu(0):
        run_tests()
