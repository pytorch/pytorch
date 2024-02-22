# Owner(s): ["module: inductor"]
import logging
import os
import subprocess
import sys
import unittest
from pathlib import Path

from typing import Callable, List, Optional, Tuple

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.codegen.cuda.cutlass_utils import _DISABLE_CUTLASS_BACKEND
from torch._inductor.ir import Buffer, FixedLayout
from torch._inductor.utils import cache_dir, fresh_inductor_cache
from torch.testing._internal.common_cuda import SM75OrLater, SM90OrLater
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA

torch.set_float32_matmul_precision("high")
if HAS_CUDA:
    torch.cuda.memory._set_allocator_settings("expandable_segments:False")

_CUTLASS_DIR = os.path.join(os.path.dirname(__file__), "../../third_party/cutlass/")

log = logging.getLogger(__name__)


def _get_path_without_sccache() -> str:
    """
    Get the PATH environment variable without sccache.
    """
    path_envs = os.environ.get("PATH", "").split(":")
    path_envs = [env for env in path_envs if "/opt/cache/bin" not in env]
    return ":".join(path_envs)


@instantiate_parametrized_tests
class TestCutlassBackend(TestCase):
    def setUp(self):
        super().setUp()
        torch.random.manual_seed(1234)

    def _create_buffer(self, name, shape):
        return Buffer(name, FixedLayout(torch.device("cuda:0"), torch.float32, shape))

    def cuda_test_compile_standalone_runner(
        self, src, name=None, do_compile=True, do_run=True, log=sys.stderr
    ):
        from torch._inductor.codegen.cuda.cutlass_utils import (
            cuda_standalone_runner_compile_command,
        )

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
            cmd_parts = compile_command.split(" ")
            remove_idx = cmd_parts.index("-lcudart")
            if remove_idx >= 0:
                del cmd_parts[remove_idx]
            print(
                subprocess.check_output(
                    cmd_parts,
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

    @unittest.skipIf(not SM75OrLater, "need sm_75")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    @unittest.mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
    def test_max_autotune_precompile(self):
        """
        Make sure autotuning mm in sub processes work without crashes.
        """

        if torch.version.hip:
            return

        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

        def mm(a, b):
            return a @ b

        a = torch.randn(100, 10).cuda().half()
        b = torch.randn(10, 100).cuda().half()

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": "CUTLASS,Triton,ATen",
                "compile_threads": 4,
                "cuda.cutlass_dir": _CUTLASS_DIR,
                "cuda.cutlass_max_profiling_configs": 2,
            }
        ):
            Y_compiled = torch.compile(mm, dynamic=False)(a, b)
            Y = mm(a, b)
            torch.testing.assert_close(Y_compiled, Y)

    # TODO: Enable dynamic test cases when dynamic support is added.
    @unittest.skipIf(not SM75OrLater, "need sm_75")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
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
        expected_fuse_count=0,
        mm: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        with_bias=False,
        bias_broadcast=(False, False),
        with_aux=False,
        with_more_inputs=(),
        m=1024,
        n=1024,
        k=1024,
        max_profiling_configs=4,
        batch_size=None,
        evt_only=True,
        aux_shape: Optional[Tuple[int]] = None,
        config_override=None,
        use_autotuning_cache=True,
    ):
        if config_override is None:
            config_override = {}
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            mixed_precision
        )
        with torch.no_grad():
            # Note: The ops that are available
            # also depend on the alignment of the shapes
            # so if these shapes don't all align to at least 8 elements
            # it can happen that no Cutlass 3.x op is available
            # that allows fusions

            if batch_size is None:
                a = torch.randn(m, k).mul(1.0 / 32).cuda()
                b = torch.randn(k, n).mul(1.0 / 32).cuda()

                if with_bias:
                    bias_m = m
                    bias_n = n
                    if bias_broadcast[0]:
                        bias_m = 1
                    if bias_broadcast[1]:
                        bias_n = 1
                    bias = torch.randn(bias_m, bias_n).mul(1.0 / 32).cuda()
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
            more_inputs = [
                torch.randn(*inp_shape).mul(1.0 / 32).cuda()
                for inp_shape in with_more_inputs
            ]

            if fp16:
                a = a.half()
                b = b.half()
                if with_bias:
                    bias = bias.half()
                if with_aux:
                    aux = aux.half()
                more_inputs = [inp.half() for inp in more_inputs]
            args = [a, b]
            if with_bias:
                args.append(bias)
            if with_aux:
                args.append(aux)
            args.extend(more_inputs)
            conf_patch = {
                "max_autotune": True,
                "autotune_in_subproc": False,
                "benchmark_fusion": False,
                "cuda.cutlass_backend_min_gemm_size": 1,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cuda.cutlass_dir": _CUTLASS_DIR,
                "cuda.cutlass_max_profiling_configs": max_profiling_configs,
                "cuda.version": "12.1",  # required to enable the Kernels we need
            }
            conf_patch.update(config_override)
            with config.patch(conf_patch):
                counters["inductor"]["cuda_epilogue_fusion_counter"] = 0
                Y = mm(*args)
                if use_autotuning_cache:
                    mm_jit = torch.compile(mm, dynamic=dynamic)
                    Y_compiled = mm_jit(*args)
                else:
                    with fresh_inductor_cache():
                        mm_jit = torch.compile(mm, dynamic=dynamic)
                        Y_compiled = mm_jit(*args)
                actual_count = counters["inductor"]["cuda_epilogue_fusion_counter"]
                torch.testing.assert_close(Y_compiled, Y, atol=1e-2, rtol=1e-2)
                if expected_fuse_count is not None:
                    assert (
                        actual_count == expected_fuse_count
                    ), f"Expected fuse count of {expected_fuse_count} but got {actual_count}"

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_simple_fusion_fp16(self):
        def mm(a, b):
            return (a @ b) * 3.0

        #  The pointwise ops seem to be pre-fused into a single Pointwise
        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=False, fp16=True, expected_fuse_count=0, mm=mm
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_simple_fusion_fp16_layout_opt(self):
        def mm(a, b, bias):
            return torch.addmm(bias, a, b) * 3.0

        #  The pointwise ops seem to be pre-fused into a single Pointwise
        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=False,
            fp16=True,
            expected_fuse_count=0,
            mm=mm,
            m=256,
            n=512,
            k=255,
            with_bias=True,
            config_override={
                "layout_optimization": True,
                "shape_padding": True,
                "cuda.cutlass_backend_min_gemm_size": 1,
            },
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_simple_fusion_fp16_layout_opt2(self):
        def mm(a, b):
            return torch.bmm(a, b) * 3.4

        #  The pointwise ops seem to be pre-fused into a single Pointwise
        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=False,
            fp16=True,
            expected_fuse_count=0,
            mm=mm,
            m=256,
            n=512,
            k=255,
            batch_size=10,
            with_bias=False,
            config_override={
                "layout_optimization": True,
                "shape_padding": True,
                "cuda.cutlass_backend_min_gemm_size": 1,
            },
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_simple_mm_fp16_standalone_runner_large(self):
        from torch._inductor.codegen.cuda.cutlass_utils import (
            CUDACompileSourceCapturingContext,
        )

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
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
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
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_simple_fusion_fp16_unaligned(
        self,
    ):
        def mm(a, b):
            return (a @ b) * 3.0

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
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_double_matmul(
        self,
    ):
        def mm(a, b):
            return ((a @ b).T @ a) - 4.5

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=False,
            fp16=True,
            expected_fuse_count=0,
            mm=mm,
            m=128,
            n=128,
            k=128,
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_simple_fusion_fp32(self):
        def mm(a, b):
            return (a @ b) * 3.0

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
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_simple_fusion_fp16_fp32acc(self):
        def mm(a, b):
            return (a @ b) * 3.0

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=True, fp16=True, expected_fuse_count=0, mm=mm
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_chained_fusion_fp16(self):
        def mm(a, b):
            return (a @ b) * 3.3 - 1.234

        #  The pointwise ops seem to be pre-fused into a single Pointwise
        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=False, fp16=True, expected_fuse_count=0, mm=mm
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
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
                expected_fuse_count=0,
                mm=mm,
                with_bias=True,
                m=2048,
                n=512,
                k=4096,
            )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_one_additional_input_random_mask(self):
        from torch._inductor.codegen.cuda.cutlass_utils import (
            CUDACompileSourceCapturingContext,
        )

        def mm(a, b, c):
            return (a @ b) * 1.5 + c

        source_capture = CUDACompileSourceCapturingContext()
        with source_capture:
            try:
                self._test_max_autotune_cutlass_backend_epilogue_fusion(
                    mixed_precision=False,
                    fp16=True,
                    expected_fuse_count=0,
                    mm=mm,
                    with_bias=True,
                    m=64,
                    n=128,
                    k=128,
                    batch_size=1,
                    max_profiling_configs=1,
                    use_autotuning_cache=False,
                )
            finally:
                self.cuda_test_compile_standalone_runner(
                    source_capture.sources[-1], do_run=True
                )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_two_additional_inputs_random_mask(self):
        from torch._inductor.codegen.cuda.cutlass_utils import (
            CUDACompileSourceCapturingContext,
        )

        def mm(a, b, c, aux):
            return ((a @ b) * torch.relu(c) * 1.0) + aux

        source_capture = CUDACompileSourceCapturingContext()
        with source_capture:
            try:
                self._test_max_autotune_cutlass_backend_epilogue_fusion(
                    mixed_precision=False,
                    fp16=True,
                    expected_fuse_count=0,
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
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_more_additional_inputs_random_mask(self):
        def mm(a, b, c, aux1, aux2, aux3):
            return ((a @ b) * torch.relu(c)) + aux3 - aux1 + aux2

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=False,
            fp16=True,
            expected_fuse_count=0,
            mm=mm,
            with_bias=True,
            with_aux=True,
            m=1024,
            n=512,
            k=2048,
            with_more_inputs=((1024, 512), (1024, 512)),
            batch_size=1,
            max_profiling_configs=3,
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
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
            expected_fuse_count=0,
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
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
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
            expected_fuse_count=0,
            mm=mm,
            with_bias=True,
            m=2048,
            n=512,
            k=4096,
            batch_size=100,
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_chained_fusion_fp16_fp32acc(self):
        def mm(a, b):
            return (a @ b) * 3.3 - 1.234

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=True, fp16=True, expected_fuse_count=0, mm=mm
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_relu_fusion_fp16(self):
        def mm(a, b):
            return torch.nn.functional.relu((a @ b) * 3.3 - 1.234)

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=False, fp16=True, expected_fuse_count=0, mm=mm
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
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
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
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
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_simple_addmm_large(self):
        def mm(a, b, bias):
            return torch.addmm(bias, a, b)

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=False,
            fp16=True,
            expected_fuse_count=0,
            mm=mm,
            with_bias=True,
            # bias_broadcast=(False, True),
            m=1024,
            k=256,
            n=109760,
        )

    @unittest.skipIf(
        True, "Flaky test due to accumulation of minimal numerical differences"
    )
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_simple_addmm_large_broadcasted_1(self):
        def mm(a, b, bias):
            return torch.addmm(bias, a, b)

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=False,
            fp16=True,
            expected_fuse_count=0,
            mm=mm,
            with_bias=True,
            bias_broadcast=(False, True),
            m=1024,
            k=256,
            n=109760,
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_simple_addmm_large_broadcasted_2(self):
        def mm(a, b, bias):
            return torch.addmm(bias, a, b)

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=False,
            fp16=True,
            expected_fuse_count=0,
            mm=mm,
            with_bias=True,
            bias_broadcast=(True, False),
            m=1024,
            k=256,
            n=4096 + 256,
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
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
            m=1024,
            k=256,
            n=4096 + 256,
            use_autotuning_cache=True,
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
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
            m=1024,
            k=256,
            n=4096 + 256,
            use_autotuning_cache=True,
        )

    @unittest.skipIf(
        True, "Flaky test, depending on chosen Cutlass op / autotuning result"
    )
    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
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
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_simple_aux_broadcasted_row(self):
        def mm(a, b, bias):
            bias_slice = bias[0:1, :]
            return (a @ b) - torch.relu(bias_slice)

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=False,
            fp16=True,
            expected_fuse_count=0,
            mm=mm,
            with_bias=True,
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
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
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_simple_aux_broadcasted_col2(self):
        def mm(a, b, bias):
            return (a @ b) - torch.relu(bias * 1.2)

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=False,
            fp16=True,
            expected_fuse_count=0,
            mm=mm,
            with_bias=True,
            bias_broadcast=(True, False),
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_simple_aux_broadcasted_row2(self):
        def mm(a, b, bias):
            return (a @ b) - torch.relu(bias * 1.2)

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=False,
            fp16=True,
            expected_fuse_count=0,
            mm=mm,
            with_bias=True,
            bias_broadcast=(False, True),
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_relu_fusion_fp16_fp32acc(self):
        def mm(a, b):
            return torch.nn.functional.relu((a @ b) * 3.3 - 1.234)

        #  The pointwise ops seem to be pre-fused into a single Pointwise
        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=True, fp16=True, expected_fuse_count=0, mm=mm
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_relu6_fusion_fp16_fp32acc(self):
        def mm(a, b):
            return torch.clamp(torch.nn.functional.relu(a @ b), max=6.0)

        #  The pointwise ops seem to be pre-fused into a single Pointwise
        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=True, fp16=True, expected_fuse_count=0, mm=mm
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
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
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_simple_baddbmm(self):
        def mm(a, b, bias):
            # disabled usage of torch.baddbmm because it implicitly casts all inputs to float32
            # during lowering in inductor for unknown reasons
            # return torch.baddbmm(bias, a, b)
            return (a @ b) + bias  # this is equivalent and doesn't cast to float32

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=True,
            fp16=True,
            expected_fuse_count=0,
            mm=mm,
            with_bias=True,
            batch_size=31,
            evt_only=True,
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_shape_dependent_normalization_fusion(self):
        def mm(a, b):
            return (a @ b) / b.size(1)

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=True, fp16=True, expected_fuse_count=0, mm=mm
        )

    # TODO: Enable dynamic test cases when dynamic support is added.
    @unittest.skipIf(not SM75OrLater, "need sm_75")
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
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
            }
        ):
            Y = mm(a, a, bias)
            Y_compiled = torch.compile(mm, dynamic=dynamic)(a, a, bias)
            torch.testing.assert_close(Y_compiled, Y, atol=1e-1, rtol=1e-1)

    @unittest.skipIf(not SM75OrLater, "need sm_75")
    @unittest.skipIf(_DISABLE_CUTLASS_BACKEND, "Cutlass backend disabled")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    @parametrize("dynamic", (False,))
    @parametrize("max_autotune_gemm_backends", ("CUTLASS", "ATen,Triton,CUTLASS"))
    @unittest.mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_max_autotune_cutlass_backend_addmm(
        self,
        dynamic=False,
        max_autotune_gemm_backends="CUTLASS",
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
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cuda.cutlass_dir": _CUTLASS_DIR,
                "cuda.cutlass_max_profiling_configs": 2,
                "cuda.cutlass_op_whitelist_regex": "warpspecialized_cooperative_epi_tma",
                "cuda.cutlass_op_blacklist_regex": "pingpong",  # Pingpong Kernels can lead to numerical issues
            }
        ):
            # No broadcast
            compare_results(4096, 25728, 2048, 2.0, 0.4, [4096, 2048])
            # Broadcast first dim.
            compare_results(4096, 25728, 2048, 2.0, 0.4, [2048])
            # Broadcast last dim.
            if not SM90OrLater and max_autotune_gemm_backends == "CUTLASS":
                compare_results(4096, 25728, 2048, 2.0, 0.4, [4096, 1])
            else:
                compare_results(4096, 25728, 2048, 2.0, 0.4, [4096, 1])


if __name__ == "__main__":
    from torch._inductor.utils import is_big_gpu

    # Set env to make it work in CI.
    if HAS_CUDA and HAS_CPU and is_big_gpu(0) and not _DISABLE_CUTLASS_BACKEND:
        run_tests()
