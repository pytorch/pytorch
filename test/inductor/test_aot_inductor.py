# Owner(s): ["module: inductor"]
import itertools
import logging
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest import skip
from unittest.mock import patch

import torch
import torch._export
import torch._inductor
import torch._inductor.config
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
import torch.nn as nn
from torch._dynamo import config as dynamo_config
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.testing import rand_strided, same
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.codecache import WritableTempFile
from torch._inductor.cpp_builder import normalize_path_separator
from torch._inductor.package import package_aoti
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.test_case import TestCase
from torch._inductor.utils import (
    is_big_gpu,
    maybe_aoti_standalone_config,
    run_and_get_cpp_code,
)
from torch._library import capture_triton
from torch._utils_internal import full_aoti_runtime_assert
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torch.export import Dim, export
from torch.export.pt2_archive._package import load_pt2
from torch.testing import FileCheck
from torch.testing._internal import common_utils
from torch.testing._internal.common_cuda import (
    _get_torch_cuda_version,
    CDNA2OrLater,
    IS_SM90,
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    PLATFORM_SUPPORTS_FP8,
    PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
    SM80OrLater,
    tf32_on_and_off,
)
from torch.testing._internal.common_device_type import (
    _has_sufficient_memory,
    e4m3_type,
    skipCUDAIf,
)
from torch.testing._internal.common_quantization import (
    _group_quantize_tensor,
    skip_if_no_torchvision,
    skipIfNoFBGEMM,
)
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    IS_CI,
    IS_FBCODE,
    IS_MACOS,
    IS_WINDOWS,
    MACOS_VERSION,
    MI300_ARCH,
    parametrize,
    runOnRocm,
    skipIfMPS,
    skipIfRocm,
    skipIfRocmArch,
    skipIfWindows,
    skipIfWindowsXPU,
    skipIfXpu,
    TEST_MPS,
    TEST_WITH_ROCM,
)
from torch.testing._internal.custom_tensor import CustomTensorPlainOut
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_GPU,
    HAS_XPU_AND_TRITON,
    IS_BIG_GPU,
)
from torch.testing._internal.logging_utils import LoggingTestCase, make_logging_test
from torch.testing._internal.triton_utils import requires_gpu
from torch.utils import _pytree as pytree
from torch.utils._triton import (
    has_triton_experimental_host_tma,
    has_triton_tensor_descriptor_host_tma,
)


if HAS_GPU:
    import triton  # @manual
    from triton import language as tl

    from torch.testing._internal.triton_utils import (
        add_kernel,
        add_kernel_2d_autotuned,
        add_kernel_autotuned,
        add_kernel_autotuned_weird_param_order,
        add_kernel_on_device_tma_new_api,
        add_kernel_on_device_tma_old_api,
        add_kernel_with_boolean_param,
        add_kernel_with_none_param_and_equal_to_1_arg,
        add_kernel_with_optional_param,
        add_kernel_with_scaling,
        add_kernel_with_tma_1d_new_api,
        add_kernel_with_tma_1d_old_api,
        add_kernel_with_tma_2d_new_api,
        add_kernel_with_tma_2d_old_api,
        create_tensor_descriptor_shim,
        mul2_inplace_kernel,
        strange_config_matmul_kernel,
        sub_kernel_autotuned,
    )

if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

try:
    try:
        from .test_aot_inductor_utils import (
            AOTIRunnerUtil,
            check_model,
            check_model_with_multiple_inputs,
            code_check_count,
        )
        from .test_control_flow import (
            CondModels,
            prepend_counters,
            prepend_predicates,
            WhileLoopModels,
        )
        from .test_torchinductor import copy_tests, requires_multigpu, TestFailure
    except ImportError:
        from test_aot_inductor_utils import (  # @manual=fbcode//caffe2/test/inductor:aot_inductor_utils-library
            AOTIRunnerUtil,
            check_model,
            check_model_with_multiple_inputs,
            code_check_count,
        )
        from test_control_flow import (  # @manual=fbcode//caffe2/test/inductor:control_flow-library
            CondModels,
            prepend_counters,
            prepend_predicates,
            WhileLoopModels,
        )
        from test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
            copy_tests,
            requires_multigpu,
            TestFailure,
        )
except (unittest.SkipTest, ImportError):
    if __name__ == "__main__":
        sys.exit(0)
    raise


def get_module_ext_type():
    if IS_WINDOWS:
        return "pyd"
    else:
        return "so"


class AOTInductorTestsTemplate:
    # Temporarily skipping test as pytorch/cpuinfo not able to retrieve cache size for
    # AMD EPYC 9575F 64-Core Processor CPU in gfx942 VM Runners
    @common_utils.parametrize("embed_kernel_binary", [False, True])
    @common_utils.parametrize("max_autotune", [False, True])
    @skipIfRocmArch(MI300_ARCH)
    def test_simple(self, embed_kernel_binary, max_autotune):
        if self.device == "cpu" and IS_MACOS and max_autotune:
            raise unittest.SkipTest("max_autotune not supported on macos")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x, y):
                return x + self.linear(y)

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        model = Model()
        with config.patch(
            {
                "aot_inductor.embed_kernel_binary": embed_kernel_binary,
                "max_autotune": max_autotune,
            }
        ):
            self.check_model(model, example_inputs)

            _, code = run_and_get_cpp_code(
                AOTIRunnerUtil.compile, model, example_inputs
            )
            if self.device == "mps":
                FileCheck().check("aoti_torch_mps_get_kernel_function(").run(code)
            elif self.device == GPU_TYPE:
                FileCheck().check("launchKernel(").run(code)
                if config.aot_inductor.embed_kernel_binary:
                    # Not expect to see launchKernel("CUBIN_FILE_NAME"
                    FileCheck().check_not('launchKernel("').run(code)

        if self.use_minimal_arrayref_interface:
            self.code_check_count(
                model, example_inputs, "AOTInductorModelRunMinimalArrayrefInterface(", 1
            )

    def test_triton_kernel_bool_param(self):
        if self.device != GPU_TYPE or self.device == "mps":
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            def forward(self, x):
                out = torch.zeros_like(x)
                add_kernel_with_boolean_param[1,](
                    in_ptr0=x,
                    in_ptr1=x,
                    out_ptr=out,
                    n_elements=x.numel(),
                    add_xy=True,
                    BLOCK_SIZE=1,
                )
                return out

        inputs = (torch.randn(4, device=self.device),)
        self.check_model(Model(), inputs)

    @unittest.skipIf(
        IS_FBCODE,
        "toolchain doesn't support ptx to fatbin",
    )
    @skipIfMPS
    # Skip embed_kernel_binary == True for now as it shows random
    # failure on CI
    @common_utils.parametrize("embed_kernel_binary", [False])
    @unittest.skipIf(
        torch.version.hip is None and _get_torch_cuda_version() < (12, 6),
        "Test is only supported on CUDA 12.6+",
    )
    def test_simple_multi_arch(self, embed_kernel_binary):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU_TYPE")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 16)

            def forward(self, x, y):
                return x + self.linear(y)

        example_inputs = (
            torch.randn(10, 16, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        model = Model()
        with config.patch(
            {
                "aot_inductor.embed_kernel_binary": embed_kernel_binary,
                "aot_inductor.emit_multi_arch_kernel": True,
            }
        ):
            self.check_model(model, example_inputs)
            if not embed_kernel_binary:
                _, code = run_and_get_cpp_code(
                    AOTIRunnerUtil.compile, model, example_inputs
                )
                file_extension = (
                    ".spv"
                    if self.device == "xpu"
                    else (".hsaco" if torch.version.hip else ".fatbin")
                )
                FileCheck().check(file_extension).run(code)

    def test_small_constant(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x)

        example_inputs = (torch.randn(4, 4, device=self.device),)
        with config.patch({"always_keep_tensor_constants": True}):
            self.check_model(Model().to(self.device), example_inputs)

    def test_output_path_1(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x, y):
                return x + self.linear(y)

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        with config.patch("aot_inductor.output_path", "tmp_output_"):
            self.check_model(Model(), example_inputs)

    def test_output_path_2(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x, y):
                return x + self.linear(y)

        model = Model().to(device=self.device)
        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        expected_path = normalize_path_separator(
            os.path.join(
                tempfile.mkdtemp(dir=cache_dir()), f"model.{get_module_ext_type()}"
            )
        )
        actual_path = AOTIRunnerUtil.legacy_compile(
            model, example_inputs, options={"aot_inductor.output_path": expected_path}
        )
        self.assertTrue(actual_path == expected_path)

    @unittest.skipIf(
        config.triton.native_matmul,
        "different # of input/output/constants in native matmul",
    )
    def test_empty_constant_folding(self):
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.w = torch.randn(4, 4, device=device)
                self.b = torch.randn(4, device=device)

            def forward(self, x):
                return torch.matmul(x, self.w) + self.b

        model = Model(self.device)
        example_inputs = (torch.randn(4, 4, device=self.device),)
        with config.patch({"aot_inductor.use_runtime_constant_folding": True}):
            so_path, code = run_and_get_cpp_code(
                AOTIRunnerUtil.legacy_compile, model, example_inputs
            )
            # We should have 1 input, 1 output, 2 constants for the model.
            FileCheck().check_count("AOTInductorModelBase(1,", 1).check_next(
                "1,"
            ).check_next("2,").run(code)

    def test_constant_folding(self):
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.w_pre = torch.randn(4, 4, device=device)
                self.b = torch.randn(4, device=device)

            def forward(self, x):
                w_transpose = torch.transpose(self.w_pre, 0, 1)
                w_relu = torch.nn.functional.relu(w_transpose)
                w = w_relu + self.b
                return torch.matmul(x, w)

        example_inputs = (torch.randn(4, 4, device=self.device),)
        with config.patch({"aot_inductor.use_runtime_constant_folding": True}):
            self.check_model(Model(self.device), example_inputs)

    def test_constant_folding_with_update(self):
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.w_pre = torch.randn(4, 4, device=device)
                self.b = torch.randn(4, device=device)

            def forward(self, x):
                w_transpose = torch.transpose(self.w_pre, 0, 1)
                w_relu = torch.nn.functional.relu(w_transpose)
                w = w_relu + self.b
                return torch.matmul(x, w)

        example_inputs = (torch.randn(4, 4, device=self.device),)
        with (
            torch.no_grad(),
            config.patch(
                {
                    "always_keep_tensor_constants": True,
                    "aot_inductor.use_runtime_constant_folding": True,
                }
            ),
        ):
            model = Model(self.device)
            so_path = AOTIRunnerUtil.legacy_compile(
                model=model,
                example_inputs=example_inputs,
            )

        runner = AOTIRunnerUtil.legacy_load_runner(self.device, so_path)

        def runner_call(*args, **kwargs):
            import torch.fx._pytree as fx_pytree

            call_spec = runner.get_call_spec()
            in_spec = pytree.treespec_loads(call_spec[0])
            out_spec = pytree.treespec_loads(call_spec[1])
            flat_inputs = fx_pytree.tree_flatten_spec((args, kwargs), in_spec)
            flat_inputs = [x for x in flat_inputs if isinstance(x, torch.Tensor)]
            flat_outputs = runner.run(flat_inputs)
            return pytree.tree_unflatten(flat_outputs, out_spec)

        test_inputs = torch.randn(4, 4, device=self.device)
        expected = model(test_inputs)
        output = runner_call(test_inputs)
        self.assertEqual(expected, output)

        # Update with new weights on active buffer
        new_weights = {
            "L__self___b": torch.randn(4, device=self.device),
            "L__self___w_pre": torch.randn(4, 4, device=self.device),
        }
        model.w_pre = new_weights["L__self___w_pre"]
        model.b = new_weights["L__self___b"]
        expected = model(test_inputs)
        runner.update_constant_buffer(new_weights, False, False)
        output = runner_call(test_inputs)
        self.assertEqual(expected, output)

        # Update with new weights on inactive buffer
        new_weights = {
            "L__self___b": torch.randn(4, device=self.device),
            "L__self___w_pre": torch.randn(4, 4, device=self.device),
        }
        model.w_pre = new_weights["L__self___w_pre"]
        model.b = new_weights["L__self___b"]
        expected = model(test_inputs)
        runner.update_constant_buffer(new_weights, True, False)
        new_output = runner_call(test_inputs)
        # We have not yet swapped the buffer, new_output should be the same as the old one.
        self.assertEqual(output, new_output)
        # Swap the buffer, should get the correct result now.
        runner.swap_constant_buffer()
        new_output = runner_call(test_inputs)
        self.assertEqual(expected, new_output)

    @requires_gpu
    def test_duplicate_constant_folding(self):
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.w1 = torch.randn(4, 4, device=device)
                self.w2 = torch.randn(4, 4, device=device)
                self.w3 = torch.randn(4, 4, device=device)
                self.w4 = torch.randn(4, 4, device=device)

            def forward(self, x):
                w_concat = torch.cat((self.w1, self.w2, self.w3, self.w4))
                return torch.cat((x, w_concat))

        example_inputs = (torch.randn(4, 4, device=self.device),)
        with config.patch({"aot_inductor.use_runtime_constant_folding": True}):
            self.check_model(Model(self.device), example_inputs)

    def test_autotune_with_constant_folding(self):
        class Model(torch.nn.Module):
            def __init__(self, device) -> None:
                super().__init__()
                self.x = torch.randn(2048, 2048, dtype=torch.float16, device=device)

            def _quantize(self, input):
                return torch.abs(input)

            def forward(self, y):
                abs_weight = self._quantize(self.x)
                abs_y = self._quantize(y)

                return abs_weight, abs_y

        input1 = (torch.rand(2048, 2048, dtype=torch.float16, device=self.device),)
        model = Model(self.device).to(self.device)

        _ = model(*input1)

        ep = torch.export.export(model, input1, dynamic_shapes=None, strict=False)
        torch._inductor.aoti_compile_and_package(
            ep, inductor_configs={"aot_inductor.use_runtime_constant_folding": True}
        )

    @unittest.skipIf(
        TEST_MPS and MACOS_VERSION < 14.0,
        "Compilation error",
    )
    def test_aot_inductor_consts_cpp_build(self):
        class Model(torch.nn.Module):
            def __init__(self, device) -> None:
                super().__init__()
                self.x = torch.randn(2048, 2048, dtype=torch.float16, device=device)

            def _quantize(self, input):
                return torch.abs(input)

            def forward(self, y):
                abs_weight = self._quantize(self.x)
                abs_y = self._quantize(y)

                return abs_weight, abs_y

        input1 = (torch.rand(2048, 2048, dtype=torch.float16, device=self.device),)
        model = Model(self.device).to(self.device)

        _ = model(*input1)

        ep = torch.export.export(model, input1, dynamic_shapes=None, strict=False)
        torch._inductor.aoti_compile_and_package(
            ep,
            inductor_configs={
                "aot_inductor.use_runtime_constant_folding": True,
                "aot_inductor.use_consts_asm_build": False,
            },
        )

    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("tma_version", ["new", "old"])
    def test_triton_kernel_on_device_tma(self, dynamic, tma_version):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")
        if tma_version == "new" and not has_triton_tensor_descriptor_host_tma():
            self.skipTest("requires triton.tools.tensor_descriptor TMA support")
        if tma_version == "old" and not has_triton_experimental_host_tma():
            self.skipTest("requires triton.tools.experimental_descriptor TMA support")

        kernel = (
            add_kernel_on_device_tma_new_api
            if tma_version == "new"
            else add_kernel_on_device_tma_old_api
        )

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, a, b):
                BLOCK_SIZE = 32
                out = torch.zeros_like(a)
                m, n = out.size()

                # Allocate workspace for on-device TMA descriptors
                # Need 128 bytes per descriptor, 3 descriptors total
                if tma_version == "old":
                    workspace = torch.zeros(3 * 128, dtype=torch.uint8, device=a.device)
                else:
                    workspace = None

                grid = (triton.cdiv(m, BLOCK_SIZE), triton.cdiv(n, BLOCK_SIZE))

                kernel[grid](
                    a,
                    b,
                    out,
                    m,
                    n,
                    workspace,
                    BLOCK_SIZE=BLOCK_SIZE,
                )

                return out

        a = torch.randn((32 * 4, 32 * 8), device=self.device)
        b = torch.randn((32 * 4, 32 * 8), device=self.device)
        example_inputs = (a, b)

        triton.set_allocator(
            lambda size, align, stream: torch.empty(
                size, dtype=torch.int8, device=GPU_TYPE
            )
        )

        dynamic_shapes = None
        if dynamic:
            dim0 = Dim("s0", min=2, max=1024)
            dim1 = Dim("s1", min=2, max=1024)
            dynamic_shapes = {
                "a": {0: dim0, 1: None},
                "b": {0: dim1, 1: None},
            }

        self.check_model(
            Model(),
            example_inputs=example_inputs,
            dynamic_shapes=dynamic_shapes,
        )

    @requires_gpu
    def test_multi_device(self):
        if self.device == "cpu" and GPU_TYPE == "xpu":
            raise unittest.SkipTest(
                "In this scenario, the test case will run XPU code in "
                "AOTIModelContainerRunnerCpu, which is not reasonable,"
                "See issue #140805"
            )

        class Model(torch.nn.Module):
            def forward(self, x):
                x = x + 1
                x = x.cpu()
                x = x + 2
                x = x.to(GPU_TYPE)
                return x

        example_inputs = (torch.randn(32, 64, device=self.device),)
        self.check_model(Model(), example_inputs)

    @unittest.skip(
        "install_free_tensors leads to OOM - https://github.com/pytorch/pytorch/issues/164062"
    )
    def test_large_weight(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(2048, 262144)

            def forward(self, x, y):
                return x + self.linear(y)

        example_inputs = (
            torch.randn(1, 262144, device=self.device),
            torch.randn(1, 2048, device=self.device),
        )

        # We only test compilation since we often get OOM running in CI.
        model = Model()
        model = model.to(self.device)
        AOTIRunnerUtil.compile(model, example_inputs)

    def test_constant_type_propagation(self):
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.w_pre = torch.randn(4, 4, device=device)
                self.b = torch.randn(4, device=device)

            def forward(self, x):
                w_transpose = torch.transpose(self.w_pre, 0, 1)
                w_relu = torch.nn.functional.relu(w_transpose)
                w = w_relu + self.b
                return torch.matmul(x, w)

        model = Model(self.device)
        example_inputs = (torch.randn(4, 4, device=self.device),)
        with config.patch({"aot_inductor.use_runtime_constant_folding": True}):
            so_path, code = run_and_get_cpp_code(
                AOTIRunnerUtil.legacy_compile, model, example_inputs
            )
            FileCheck().check_not("torch::aot_inductor::ConstantType::Unknown").run(
                code
            )

    def test_subclasses(self):
        device_to_init = self.device

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p1 = torch.nn.Parameter(torch.ones(3, 4, device=device_to_init))
                self.p2 = torch.nn.Parameter(
                    CustomTensorPlainOut(
                        torch.ones(3, 4, device=device_to_init),
                        torch.ones(3, 4, device=device_to_init),
                    )
                )

            def forward(self, x):
                a = (2 * self.p1 + self.p2).sum()
                return x + a

        m = Foo()
        ref_x = torch.randn(3, 4, device=device_to_init)

        with torch.no_grad():
            result = AOTIRunnerUtil.run(
                m,
                (ref_x,),
            )
        actual = m(ref_x)
        self.assertTrue(same(result, actual))

    def test_large_mmaped_weights(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(512, 250112)

            def forward(self, x, y):
                return x + self.linear(y)

        example_inputs = (
            torch.randn(1, 250112, device=self.device),
            torch.randn(1, 512, device=self.device),
        )
        with config.patch({"aot_inductor.force_mmap_weights": True}):
            self.check_model(Model(), example_inputs)

    def test_large_mmaped_weights_on_disk(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(512, 250112)

            def forward(self, x, y):
                return x + self.linear(y)

        example_inputs = (
            torch.randn(1, 250112, device=self.device),
            torch.randn(1, 512, device=self.device),
        )
        with config.patch(
            {"aot_inductor.package_constants_on_disk_format": "binary_blob"}
        ):
            self.check_model(Model(), example_inputs)

    def test_with_offset(self):
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.orig_tensor = torch.randn(2, 15, 10, device=device)[0]
                self.tensor = self.orig_tensor[5:, :]

            def forward(self, x, y):
                return (
                    x
                    + torch.nn.functional.linear(y, self.orig_tensor[:10, :])
                    + self.tensor
                )

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        self.check_model(Model(self.device), example_inputs)

    @unittest.skipIf(
        IS_FBCODE,
        "Not yet runnable in fbcode when the model.so is newly generated while older PyTorch is used",
    )
    def test_freezing(self):
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.weight = torch.randn(9, 10, device=device)
                self.padding = torch.randn(1, 10, device=device)

            def forward(self, x, y):
                padded_weight = torch.cat((self.weight, self.padding), dim=0)
                return x + torch.nn.functional.linear(y, padded_weight)

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )

        with config.patch({"freezing": True}):
            self.check_model(Model(self.device), example_inputs)

    @unittest.skipIf(
        IS_FBCODE,
        "Not yet runnable in fbcode when the model.so is newly generated while older PyTorch is used",
    )
    def test_conv_freezing(self):
        dtypes = [torch.bfloat16, torch.float] if SM80OrLater else [torch.float]
        for dtype, groups in itertools.product(dtypes, [1, 2]):
            iC = 2
            oC = 3

            class Model(torch.nn.Module):
                def __init__(self, device):
                    super().__init__()
                    self.weight = torch.randn(oC * groups, iC, 3, 3, device=device).to(
                        dtype
                    )

                def forward(self, y):
                    return torch.nn.functional.conv2d(y, self.weight, groups=groups)

            example_inputs = (
                torch.randn(2, iC * groups, 10, 10, device=self.device).to(dtype),
            )

            with config.patch({"freezing": True}):
                self.check_model(Model(self.device), example_inputs)

    @unittest.skipIf(
        IS_FBCODE,
        "Not yet runnable in fbcode when the model.so is newly generated while older PyTorch is used",
    )
    @tf32_on_and_off(0.005)
    def test_deconv_freezing(self):
        dtypes = [torch.float]
        if torch._C._has_mkldnn and torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        for dtype, groups in itertools.product(dtypes, [2, 1]):
            iC = 4
            oC = 2

            class Model(torch.nn.Module):
                def __init__(self, device):
                    super().__init__()
                    self.weight = torch.randn(iC, oC * groups, 2, 2, device=device).to(
                        dtype
                    )

                def forward(self, y):
                    return torch.nn.functional.conv_transpose2d(
                        y, self.weight, groups=groups
                    )

            example_inputs = (torch.randn(1, iC, 3, 3, device=self.device).to(dtype),)
            with config.patch({"freezing": True}):
                self.check_model(Model(self.device), example_inputs)

    @unittest.skipIf(
        IS_FBCODE,
        "Not yet runnable in fbcode when the model.so is newly generated while older PyTorch is used",
    )
    def test_linear_freezing(self):
        dtypes = [torch.bfloat16, torch.float] if SM80OrLater else [torch.float]
        for dtype in dtypes:

            class LinearModel(torch.nn.Module):
                def __init__(self, device):
                    super().__init__()
                    self.weight = torch.randn(10, 10, device=device).to(dtype)
                    self.bias = torch.randn(10, device=device).to(dtype)

                def forward(self, y):
                    return torch.nn.functional.linear(y, self.weight, self.bias)

            example_inputs = (torch.randn(10, 10, device=self.device).to(dtype),)

            with config.patch({"freezing": True}):
                model = LinearModel(device=self.device)
                self.check_model(model, example_inputs)

    def test_same_backing(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo2",
                "(Tensor a, Tensor b) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo2", "CompositeExplicitAutograd", lib=lib)
            def foo_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                return a + b

            class M(torch.nn.Module):
                def forward(self, a, b):
                    x = a.shape[0]
                    y = b.shape[0]
                    a = torch.cat([a, a])
                    a = torch.ops.mylib.foo2(a, a)
                    a = a * x
                    b = torch.cat([b, b])
                    b = torch.ops.mylib.foo2(b, b)
                    b = b * y
                    return a, b

            inp = (torch.ones(3, device=self.device), torch.ones(3, device=self.device))
            self.check_model(M(), inp)

    @unittest.skipIf(
        TEST_MPS and MACOS_VERSION < 14.0,
        "MPS BFloat16 is only supported on MacOS 14+",
    )
    def test_empty_cat_dtype_promotion(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                z = torch.cat([x, y], dim=1)
                z = z.to(dtype=torch.bfloat16)
                return z * 2

        model = Foo()
        inps = (torch.randn(4, 10, dtype=torch.bfloat16), torch.randn(4, 0))
        self.check_model(model, inps)

    @unittest.skipIf(
        not IS_BIG_GPU, "Skipping triton backend only since not big GPU (not enough SM)"
    )
    def test_linear_dynamic_maxautotune(self):
        if self.device == "cpu":
            raise unittest.SkipTest("using triton backend only is not supported on CPU")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)

            def forward(self, x):
                return self.linear(x)

        model = Model().to(device=self.device)
        compile_inputs = (torch.randn(2048, 1, device=self.device),)
        dim0_x = Dim("dim0_x", min=2, max=2048)
        dynamic_shapes = {"x": {0: dim0_x}}
        ep = torch.export.export(
            model, compile_inputs, dynamic_shapes=dynamic_shapes, strict=True
        )
        optimized = torch._inductor.aoti_load_package(
            torch._inductor.aoti_compile_and_package(
                ep,
                inductor_configs={
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON",
                },
            )
        )
        runtime_input = torch.randn(10, 1, device=self.device)
        self.assertTrue(same(optimized(runtime_input), model(runtime_input)))
        runtime_input = torch.randn(16, 1, device=self.device)
        self.assertTrue(same(optimized(runtime_input), model(runtime_input)))
        runtime_input = torch.randn(100, 1, device=self.device)
        self.assertTrue(same(optimized(runtime_input), model(runtime_input)))

    @torch._inductor.config.patch(
        pre_grad_fusion_options={
            "normalization_pass": {},
            "remove_split_with_size_one_pass": {},
            "merge_getitem_cat_pass": {},
            "merge_stack_tahn_unbind_pass": {},
            "merge_splits_pass": {},
            "mutate_cat_pass": {},
            "split_cat_pass": {},
            "unbind_stack_pass": {},
        },
        post_grad_fusion_options={},
    )
    def test_simple_split(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return torch.cat(tensors=torch.split(x, 4, dim=1), dim=-2)

        example_inputs = (torch.randn(2, 8, device=self.device),)
        counters.clear()
        model = Model().to(device=self.device)
        actual = AOTIRunnerUtil.legacy_run(self.device, model, example_inputs)
        self.assertTrue(same(model(*example_inputs), actual))
        self.assertEqual(counters["inductor"]["scmerge_split_removed"], 1)
        self.assertEqual(counters["inductor"]["scmerge_cat_removed"], 1)
        self.assertEqual(counters["inductor"]["scmerge_split_sections_removed"], 1)

    def test_amp_fallback_random(self):
        def fn(x, w):
            return torch.functional.F.linear(x, w)

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        with config.patch({"fallback_random": True}):
            with torch.amp.autocast(device_type=self.device):
                self.check_model(fn, example_inputs)

    def test_missing_output(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                a = torch.sin(x)
                b = torch.mm(a, y)
                c = torch.cos(b)
                return c

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        self.check_model(Model(), example_inputs)

    def test_output_misaligned(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                x_unsqueeze = torch.unsqueeze(x, dim=0)
                y_unsqueeze = torch.unsqueeze(y, dim=0)
                cat = torch.cat([x_unsqueeze, y_unsqueeze], dim=0)
                x_getitem = cat[0]
                y_getitem = cat[1]
                x_sigmoid = torch.sigmoid(x_getitem)
                return x_sigmoid, y_getitem

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        with config.patch({"aot_inductor.use_runtime_constant_folding": True}):
            self.check_model(Model(), example_inputs)

    @unittest.skipIf(
        not IS_BIG_GPU, "Skipping triton backend only since not big GPU (not enough SM)"
    )
    @skip("Test was marked as expected failure, but does not fail always anymore.")
    def test_dynamic_smem_above_default_limit(self):
        if self.device == "cpu":
            raise unittest.SkipTest("using triton backend only is not supported on CPU")

        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x @ y

        model = Model().to(self.device)
        # on A100, the generated Triton kernel for this MM
        # requires 55296 bytes of dynamic SMEM which is above
        # the A100's default dynamic SMEM limit of 49152 bytes.
        example_inputs = (
            torch.randn(10285, 96, device=self.device),
            torch.randn(96, 1, device=self.device),
        )
        self.check_model(
            model,
            example_inputs,
            options={
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
            },
        )

    @unittest.skipIf(IS_FBCODE, "Not yet runnable in fbcode")
    def test_seq(self):
        layernorm = torch.nn.LayerNorm(10)
        net = torch.nn.Sequential(
            layernorm,
            torch.nn.ReLU(),
            layernorm,
            torch.nn.ReLU(),
        )

        example_inputs = (torch.randn(10, device=self.device),)
        self.check_model(net.eval(), example_inputs)

    def test_addmm(self):
        class Model(torch.nn.Module):
            def __init__(self, n, k, device):
                super().__init__()
                self.weight = torch.randn(n, k, device=device)
                self.bias = torch.randn(n, device=device)

            def forward(self, a):
                return torch.nn.functional.linear(a, self.weight, self.bias)

        M = 8
        N = 6
        K = 16
        model = Model(N, K, self.device)
        batch = 2
        a = torch.randn(batch, M, K, device=self.device)
        # We should be able to call self.check_model here, but torch.export.export
        # constants (non-parameter, non-buffer) doesn't work today.
        example_inputs = (a,)
        self.check_model(model, example_inputs)

    def test_aliased_buffer_reuse(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                x = 2 * x
                y = 2 * y
                c = torch.cat([x, y], dim=-1)
                d = 1 + c
                m = torch.mm(d, d)
                return m[:, :2] + x

        example_inputs = (
            torch.randn(4, 2, device=self.device),
            torch.randn(4, 2, device=self.device),
        )
        self.check_model(Model(), example_inputs)

    def test_buffer_reuse(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                a = torch.sin(x)
                b = torch.cos(y)
                c = torch.mm(a, b)
                d = torch.relu(c)
                e = torch.sigmoid(d)
                f = torch.mm(x, y)
                g = e + f
                return g

        example_inputs = (
            torch.randn(4, 4, device=self.device),
            torch.randn(4, 4, device=self.device),
        )
        self.check_model(Model(), example_inputs)

    def test_duplicated_params(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.p = torch.nn.Parameter(torch.rand(6))
                self.q = self.p

            def forward(self, x):
                return self.p * x + self.q

        example_inputs = (torch.rand(6, device=self.device),)
        self.check_model(Model(), example_inputs)

    @unittest.skip("Skip this test, only for local test. SIGABRT is produced.")
    def test_inf(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x, y):
                return x + self.linear(y)

        x = torch.randn(10, 10, device=self.device)
        x[0][0] = float("Inf")
        example_inputs = (
            x,
            torch.randn(10, 10, device=self.device),
        )
        self.check_model(
            Model().to(self.device),
            example_inputs,
            options={"debug_check_inf_and_nan": True},
        )

    @unittest.skip("Skip this test, only for local test. SIGABRT is produced.")
    def test_nan(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x, y):
                return x + self.linear(y)

        x = torch.randn(10, 10, device=self.device)
        x[0][0] = float("nan")
        example_inputs = (
            x,
            torch.randn(10, 10, device=self.device),
        )
        self.check_model(
            Model().to(self.device),
            example_inputs,
            options={"debug_check_inf_and_nan": True},
        )

    @skipIfWindowsXPU(msg="crash on Windows XPU.")
    def test_assert_async(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU_TYPE")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                u0 = x.item()
                torch._check(u0 > 3)
                return torch.ones(u0)[0]

        x = torch.tensor(23, device=self.device)
        example_inputs = (x,)
        self.check_model(Model(), example_inputs)

    def test_simple_dynamic(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                add_0 = x + y
                return torch.nn.functional.relu(input=add_0, inplace=False)

        x = torch.randn(128, 2048, device=self.device)
        y = torch.randn(128, 2048, device=self.device)
        dim0_x = Dim("dim0_x", min=1, max=2048)
        dynamic_shapes = {"x": {0: dim0_x}, "y": {0: dim0_x}}
        example_inputs = (x, y)
        self.check_model(Model(), example_inputs, dynamic_shapes=dynamic_shapes)

    @skipIfWindows(msg="TODO: (xuhancn) confirm, Crash: access violation")
    def test_large_dynamic_dim(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                add_0 = x + y
                return torch.nn.functional.relu(input=add_0, inplace=False)

        x = torch.randn(128, 2048, device=self.device)
        y = torch.randn(128, 2048, device=self.device)
        # Use a dimension that exceeds the maximum value of a C long long (2^63 - 1)
        dim0_x = Dim("dim0_x", min=1, max=1171368248680556527362)
        dynamic_shapes = {"x": {0: dim0_x}, "y": {0: dim0_x}}
        example_inputs = (x, y)
        self.check_model(Model(), example_inputs, dynamic_shapes=dynamic_shapes)

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FP8,
        "FP8 is only supported on H100+, SM 8.9 and MI300+ devices",
    )
    @skipIfXpu
    def test_fp8(self):
        # cuda only
        if self.device != "cuda":
            return

        class Model(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.out_dtype = dtype

            def forward(self, x, weight, bias, scale_a, scale_b):
                weight = weight.to(e4m3_type)
                output = torch._scaled_mm(
                    x,
                    weight,
                    bias=input_bias,
                    out_dtype=self.out_dtype,
                    scale_a=scale_a,
                    scale_b=scale_b,
                )
                return output

        dtype = torch.float16

        a_scale = torch.Tensor([1.0]).to(device=GPU_TYPE)
        b_scale = torch.Tensor([1.0]).to(device=GPU_TYPE)
        input_bias = torch.rand(32, device=GPU_TYPE, dtype=dtype)
        weight_shape = (32, 16)
        weight = torch.rand(*weight_shape, device=GPU_TYPE, dtype=dtype).T
        a_inverse_scale = 1 / a_scale
        b_inverse_scale = 1 / b_scale

        x_shape = (16, 16)
        x = torch.rand(*x_shape, device=GPU_TYPE, dtype=dtype).to(e4m3_type)
        dim0_x = Dim("dim0_x", min=1, max=2048)
        dynamic_shapes = ({0: dim0_x}, None, None, None, None)
        self.check_model(
            Model(dtype),
            (x, weight, input_bias, a_inverse_scale, b_inverse_scale),
            dynamic_shapes=dynamic_shapes,
        )

    @unittest.skipIf(
        TEST_WITH_ROCM or not IS_SM90,
        "scaled_grouped_mm is only supported on SM90",
    )
    @skipIfXpu
    def test_scaled_grouped_mm(self):
        # Test torch._scaled_grouped_mm AOTI lowering
        # cuda only
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, weight, scale_a, scale_b, offsets):
                # x: [num_groups, batch, in_features] - FP8 inputs
                # weight: [total_out_features, in_features] - FP8 weights (transposed)
                # scale_a: [num_groups] - input scales
                # scale_b: [num_groups] - weight scales
                # offsets: [num_groups] - cumulative output sizes
                output = torch._scaled_grouped_mm(
                    x,
                    weight.t(),
                    scale_a=scale_a,
                    scale_b=scale_b,
                    offs=offsets,
                    use_fast_accum=True,
                )
                return output.half()

        dtype = torch.float16
        num_groups = 3
        batch_size = 64
        in_features = 128
        out_features_list = [64, 128, 256]  # Different output sizes for each group

        device = GPU_TYPE

        # Calculate offsets (cumulative output sizes)
        offsets = torch.cumsum(torch.tensor(out_features_list), dim=0).to(
            device, dtype=torch.int32
        )
        total_out_features = sum(out_features_list)

        # Create FP8 input tensors - stacked for all groups
        x_fp16 = torch.randn(
            num_groups, batch_size, in_features, dtype=dtype, device=device
        )
        x_fp8 = x_fp16.to(torch.float8_e4m3fn)

        # Create FP8 weight tensor - concatenated and transposed
        weight_fp16 = torch.randn(
            total_out_features, in_features, dtype=dtype, device=device
        )
        weight_fp8 = weight_fp16.to(torch.float8_e4m3fn)

        # Create scales
        scale_a = torch.ones(num_groups, batch_size, device=device, dtype=torch.float32)
        scale_b = torch.ones(total_out_features, device=device, dtype=torch.float32)

        self.check_model(
            Model(),
            (x_fp8, weight_fp8, scale_a, scale_b, offsets),
        )

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FP8,
        "FP8 is only supported on H100+, SM 8.9 and MI300+ devices",
    )
    @skipIfXpu
    def test_fp8_view_of_param(self):
        # cuda only
        if self.device != GPU_TYPE:
            return

        class Model(torch.nn.Module):
            def __init__(self, dtype, weight):
                super().__init__()
                self.out_dtype = dtype
                self.weight = weight

            def forward(self, x, bias, scale_a, scale_b):
                # test: do the view inside of the graph,
                # AOTI needs to materialize this view before passing
                # it into the scaled_mm extern kernel
                weight = self.weight.T
                output = torch._scaled_mm(
                    x,
                    weight,
                    bias=input_bias,
                    out_dtype=self.out_dtype,
                    scale_a=scale_a,
                    scale_b=scale_b,
                )
                return output

        dtype = torch.float16

        a_scale = torch.Tensor([1.0]).to(device=self.device)
        b_scale = torch.Tensor([1.0]).to(device=self.device)
        input_bias = torch.rand(32, device=self.device, dtype=dtype)
        weight_shape = (32, 16)
        weight = torch.rand(*weight_shape, device=self.device, dtype=dtype).to(
            e4m3_type
        )
        a_inverse_scale = 1 / a_scale
        b_inverse_scale = 1 / b_scale

        x_shape = (16, 16)
        x = torch.rand(*x_shape, device=self.device, dtype=dtype).to(e4m3_type)
        dim0_x = Dim("dim0_x", min=1, max=2048)
        dynamic_shapes = ({0: dim0_x}, None, None, None)
        self.check_model(
            Model(dtype, weight),
            (x, input_bias, a_inverse_scale, b_inverse_scale),
            dynamic_shapes=dynamic_shapes,
        )

    def test_poi_multiple_dynamic(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                add_0 = x + y
                return torch.nn.functional.relu(input=add_0, inplace=False)

        x = torch.randn(128, 2048, device=self.device)
        y = torch.randn(128, 2048, device=self.device)
        dim0_x = Dim("dim0_x", min=1, max=2048)
        dynamic_shapes = {"x": {0: dim0_x}, "y": {0: dim0_x}}
        list_example_inputs = [(x, y)]
        list_example_inputs.append(
            (
                torch.randn(64, 2048, device=self.device),
                torch.randn(64, 2048, device=self.device),
            ),
        )
        list_example_inputs.append(
            (
                torch.randn(211, 2048, device=self.device),
                torch.randn(211, 2048, device=self.device),
            ),
        )
        self.check_model_with_multiple_inputs(
            Model(), list_example_inputs, dynamic_shapes=dynamic_shapes
        )

    @unittest.skipIf(
        not IS_BIG_GPU, "Skipping triton backend only since not big GPU (not enough SM)"
    )
    def test_addmm_multiple_dynamic(self):
        if self.device == "cpu":
            raise unittest.SkipTest("using triton backend only is not supported on CPU")

        class Model(torch.nn.Module):
            def __init__(self, n, k, device):
                super().__init__()
                self.weight = torch.randn(n, k, device=device)
                self.bias = torch.randn(n, device=device)

            def forward(self, a):
                return torch.nn.functional.linear(a, self.weight, self.bias)

        M = 8
        N = 6
        K = 16
        model = Model(N, K, self.device)
        batch = 2
        a = torch.randn(batch, M, K, device=self.device)
        dim0_a = Dim("dim0_a", min=1, max=2048)
        dynamic_shapes = {"a": {0: dim0_a}}
        list_example_inputs = [(a,)]
        batch = 2048
        list_example_inputs.append(
            (torch.randn(batch, M, K, device=self.device),),
        )
        batch = 128
        list_example_inputs.append(
            (torch.randn(batch, M, K, device=self.device),),
        )
        self.check_model_with_multiple_inputs(
            model,
            list_example_inputs,
            dynamic_shapes=dynamic_shapes,
            options={
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
            },
        )

    @unittest.skipIf(
        not IS_BIG_GPU, "Skipping triton backend only since not big GPU (not enough SM)"
    )
    def test_bmm_multiple_dynamic(self):
        if self.device == "cpu":
            raise unittest.SkipTest("using triton backend only is not supported on CPU")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, a, b):
                return torch.bmm(a, b)

        M = 8
        N = 6
        K = 16
        model = Model()
        batch = 1024
        a = torch.randn(batch, M, K, device=self.device)
        b = torch.randn(batch, K, N, device=self.device)
        dim0_a = Dim("dim0_a", min=1, max=2048)
        dynamic_shapes = {"a": {0: dim0_a}, "b": {0: dim0_a}}
        list_example_inputs = [(a, b)]
        batch = 2048
        list_example_inputs.append(
            (
                torch.randn(batch, M, K, device=self.device),
                torch.randn(batch, K, N, device=self.device),
            ),
        )
        batch = 128
        list_example_inputs.append(
            (
                torch.randn(batch, M, K, device=self.device),
                torch.randn(batch, K, N, device=self.device),
            ),
        )
        self.check_model_with_multiple_inputs(
            model,
            list_example_inputs,
            options={
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
            },
            dynamic_shapes=dynamic_shapes,
        )

    @skipIfWindows(msg="TODO: (xuhancn) confirm, Crash: access violation")
    def test_foreach_multiple_dynamic(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                x_unsqueeze = torch.unsqueeze(x, dim=0)
                y_unsqueeze = torch.unsqueeze(y, dim=0)
                cat = torch.cat([x_unsqueeze, y_unsqueeze], dim=0)
                return cat

        model = Model()
        x = torch.randn(128, 2048, device=self.device)
        y = torch.randn(128, 2048, device=self.device)
        dim0_x = Dim("dim0_x", min=1, max=2048)
        dynamic_shapes = {"x": {0: dim0_x}, "y": {0: dim0_x}}
        list_example_inputs = [(x, y)]
        list_example_inputs.append(
            (
                torch.randn(64, 2048, device=self.device),
                torch.randn(64, 2048, device=self.device),
            ),
        )
        list_example_inputs.append(
            (
                torch.randn(211, 2048, device=self.device),
                torch.randn(211, 2048, device=self.device),
            ),
        )
        self.check_model_with_multiple_inputs(
            model,
            list_example_inputs,
            dynamic_shapes=dynamic_shapes,
        )

    # scaled_dot_product_flash_attention
    @unittest.skipIf(
        not HAS_XPU_AND_TRITON and not SM80OrLater, "bfloat16 only supported in sm80+"
    )
    def test_sdpa(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(q, k, v)[0]

        example_inputs = (
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device=self.device),
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device=self.device),
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device=self.device),
        )
        self.check_model(Model(), example_inputs)

    @unittest.skipIf(not SM80OrLater, "bfloat16 only supported in sm80+")
    @unittest.skipIf(
        # for archs where this isn't lowered to flash attention, the math
        # backend will be used and it doesn't work for bfloat16
        not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "Some archs don't support SDPA with bfloat16",
    )
    def test_sdpa_2(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, q, k, v, x):
                t = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=True
                )[0]
                return x + t

        example_inputs = (
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device=self.device),
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device=self.device),
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device=self.device),
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device=self.device),
        )
        self.check_model(Model(), example_inputs)

    @skipIfNoFBGEMM
    def test_quantized_linear(self):
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.weight = torch.randn(10, 10, device=device)
                self.bias = torch.randn(10, device=device)

            def forward(self, x):
                return torch.ops.quantized.linear_dynamic_fp16_unpacked_weight(
                    x, self.weight, self.bias
                )

        example_inputs = (torch.randn(10, 10, device=self.device),)
        with config.patch({"aot_inductor.use_runtime_constant_folding": True}):
            self.check_model(Model(self.device), example_inputs)

    @skipIfNoFBGEMM
    def test_quantized_linear_bias_none(self):
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.weight = torch.randn(10, 10, device=device)

            def forward(self, x):
                return torch.ops.quantized.linear_dynamic_fp16_unpacked_weight(
                    x, self.weight, None
                )

        example_inputs = (torch.randn(10, 10, device=self.device),)
        with config.patch({"aot_inductor.use_runtime_constant_folding": True}):
            self.check_model(Model(self.device), example_inputs)

    @skipIfNoFBGEMM
    def test_quanatized_int8_linear(self):
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.weight = torch.randn(10, 10, device=device)
                self.bias = torch.randn(10, device=device)
                self.input_scale = torch.tensor(0.1)
                self.input_zero_point = torch.tensor(0)
                self.weight_scale = torch.tensor(0.1)
                self.weight_zero_point = torch.tensor(0)
                self.output_scale = torch.tensor(0.1)
                self.output_zero_point = torch.tensor(0)
                self.out_channel = 10

            def forward(self, x):
                return torch.ops._quantized.wrapped_quantized_linear(
                    x,
                    self.input_scale,
                    self.input_zero_point,
                    self.weight,
                    self.weight_scale,
                    self.weight_zero_point,
                    self.bias,
                    self.output_scale,
                    self.output_zero_point,
                    self.out_channel,
                )

        example_inputs = (torch.randn(10, 10, device=self.device),)
        with config.patch({"aot_inductor.use_runtime_constant_folding": True}):
            self.check_model(Model(self.device), example_inputs)

    def test_zero_grid_with_unbacked_symbols(self):
        class Repro(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                nz = torch.nonzero(x)
                b = torch.ones_like(nz, dtype=torch.float16)
                c = torch.zeros_like(nz, dtype=torch.float16)
                d = (b + c) @ y
                return d.sum()

        example_inputs = (
            torch.tensor([1, 1, 1], device=self.device),
            torch.randn((1, 32), dtype=torch.float16, device=self.device),
        )
        self.check_model(Repro(), example_inputs)

    @skipIfMPS
    @config.patch({"unbacked_symint_fallback": 12})
    @parametrize("shift_k", [0, 1, 2, 3])
    @parametrize("use_static_size", [True, False])
    def test_unbacked_expr_replacements(self, shift_k, use_static_size):
        """
        Test parameters
        - shift_k: Validates that torch._check assertion order doesn't affect
        results by shifting the order of torch._checks
        - use_static_size: Tests torch._check compatibility between unbacked
        symbolic expressions and static shapes
        """

        if self.device != GPU_TYPE:
            raise unittest.SkipTest("Need triton for user-defined triton kernel")

        def realize_out_tensor_with_size(size):
            STATIC_DIM = 256  # large enough to hit IMA w/o compute-sanitizer
            tensor = torch.ones((size, STATIC_DIM), device=self.device)
            # Realize the tensor as an intermediate buffer
            nrows, ncols = tensor.shape
            numel = tensor.numel()
            add_kernel[nrows,](
                in_ptr0=tensor,
                in_ptr1=tensor,
                out_ptr=tensor,
                n_elements=numel,
                BLOCK_SIZE=ncols,
            )
            return tensor

        class Repro(torch.nn.Module):
            def forward(self, x, y, lst):
                STATIC_SIZE = 300
                s0, s1 = x.shape
                s2, s3 = y.shape
                u0, u1, u2, u3, u100 = lst.tolist()

                expr1 = s0 + u0
                expr2 = s1 + u1
                expr3 = (s2 * s3) + (u2 // u3)  # make this one a lil complicated
                expr4 = STATIC_SIZE if use_static_size else u100

                t1 = realize_out_tensor_with_size(expr1)
                t2 = realize_out_tensor_with_size(expr2)
                t3 = realize_out_tensor_with_size(expr3)
                t4 = realize_out_tensor_with_size(expr4)

                # shift tensors to change up the torch._check order
                tensors = [t1, t2, t3, t4]
                shifted_tensors = tensors[shift_k:] + tensors[:shift_k]

                # torch.cat implicitly runs torch._check(lhs == rhs)
                cat = torch.cat(shifted_tensors, dim=1)

                return cat * cat

        # Disable cuda caching allocator to check for IMA
        torch.cuda.caching_allocator_enable(False)
        model = Repro()
        example_inputs = (
            # s0, s1
            torch.randn((100, 200), device=self.device),
            # s2, s3
            torch.randn((100, 3), device=self.device),
            # u0, u1, u2, u3, u100
            torch.tensor([200, 100, 0, 1, 300], device=self.device, dtype=torch.int),
        )
        spec = {
            "x": (Dim.DYNAMIC, Dim.DYNAMIC),
            "y": (Dim.DYNAMIC, Dim.DYNAMIC),
            "lst": (Dim.STATIC,),
        }
        self.check_model(model, example_inputs, dynamic_shapes=spec)
        torch.cuda.caching_allocator_enable(True)

    @skipIfMPS
    @config.patch({"unbacked_symint_fallback": 12})
    @config.patch({"triton.autotune_at_compile_time": None})
    def test_replace_unbacked_symbol_with_backed_expr(self):
        # This will test how autotune_at_compile_time generates sample inputs
        # when the user torch._checks(s0 + s1 == u0).
        # We may fail with IMA if the generated input sizes aren't correct.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires triton")

        def force_realize(tensor):
            # Realize the tensor as an intermediate buffer
            nrows, ncols = tensor.shape
            numel = tensor.numel()
            add_kernel[nrows,](
                in_ptr0=tensor,
                in_ptr1=tensor,
                out_ptr=tensor,
                n_elements=numel,
                BLOCK_SIZE=ncols,
            )

        INNER_DIM = 256

        class Repro(torch.nn.Module):
            def forward(self, x, y, lengths):
                # Realize an intermediate buffer with backed shape: s0 + s1
                relevant_embeddings = torch.cat([x, y], dim=0)
                force_realize(relevant_embeddings)

                # Realize an intermediate buffer with unbacked shape: u0
                num_relevant_embeddings = lengths.nonzero().size(0)
                ones = torch.ones((num_relevant_embeddings, INNER_DIM), device=x.device)
                force_realize(ones)

                # Add deferred runtime assertion: s0 + s1 == u0
                torch._check(relevant_embeddings.size(0) == ones.size(0))
                relevant_embeddings += ones
                return relevant_embeddings * relevant_embeddings

        torch.cuda.caching_allocator_enable(False)
        model = Repro()
        example_inputs = (
            torch.randn((1000, INNER_DIM), device=self.device),
            torch.randn((2000, INNER_DIM), device=self.device),
            torch.ones(3000),
        )
        spec = {
            "x": (Dim.DYNAMIC, Dim.STATIC),
            "y": (Dim.DYNAMIC, Dim.STATIC),
            "lengths": (Dim.DYNAMIC,),
        }
        self.check_model(model, example_inputs, dynamic_shapes=spec)
        torch.cuda.caching_allocator_enable(True)

    @config.patch({"triton.autotune_at_compile_time": None})
    def test_stride_with_unbacked_expr(self):
        class Repro(torch.nn.Module):
            def forward(self, x, y):
                u0 = x.item()
                torch._check(u0 >= 1)
                s0 = y.size(0)
                expr = u0 * s0
                sevens = torch.empty_strided(
                    size=(10, expr, 32), stride=(expr * 32, 32, 1), device=x.device
                ).fill_(7)
                return sevens * 3

        example_inputs = (
            torch.scalar_tensor(2, dtype=torch.int, device=self.device),
            torch.ones(8, device=self.device),
        )
        self.check_model(Repro(), example_inputs)

    @unittest.skipIf(
        TEST_MPS and MACOS_VERSION < 14.0,
        "bfloat16 is only supported on MacOS 14+",
    )
    def test_size_with_unbacked_add_expr(self):
        # Tests AOTI autotuning to make sure the correct input tensor sizes
        # are generated for sizes that include an expr such as s0 + u0.

        class Repro(torch.nn.Module):
            def forward(self, values, repeats, mask, embeddings, x, z, scalar):
                repeat_interleave = torch.repeat_interleave(values, repeats)
                index = torch.clamp(repeat_interleave, min=0, max=400).int()
                index_select = torch.index_select(embeddings, 0, index)

                backed = z.size(0)
                unbacked = scalar.item()

                unbacked_add_expr = backed + unbacked
                repeated = x.repeat(unbacked_add_expr, 1)
                return torch.cat([repeated, index_select], dim=1)

        example_inputs = (
            torch.ones(64, dtype=torch.int64, device=self.device),
            torch.ones(64, dtype=torch.int64, device=self.device) * 12,
            torch.ones((768,), dtype=torch.int64, device=self.device).bool(),
            torch.randn((401, 8), dtype=torch.bfloat16, device=self.device),
            torch.randn((1, 256), dtype=torch.bfloat16, device=self.device),
            torch.ones(758, 127, dtype=torch.int64, device=self.device),
            torch.scalar_tensor(10, dtype=torch.int32, device=self.device),
        )
        spec = {
            "values": (Dim.DYNAMIC,),
            "repeats": (Dim.DYNAMIC,),
            "mask": (Dim.DYNAMIC,),
            "embeddings": (Dim.DYNAMIC, Dim.STATIC),
            "x": (Dim.STATIC, Dim.STATIC),
            "z": (Dim.DYNAMIC, Dim.STATIC),
            "scalar": (),
        }
        self.check_model(Repro(), example_inputs, dynamic_shapes=spec)

    @skipIfWindowsXPU(msg="crash on Windows XPU.")
    def test_size_with_unbacked_add_expr_transitive(self):
        # Edge case with torch._check(expr1, expr2) + torch._check(expr2, unbacked).
        # When generating example input sizes for autotuning, it should coalesce
        # expr1, expr2, unbacked into a single size.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Repro(torch.nn.Module):
            def forward(self, values, repeats, mask, embeddings, x, y, z, lst):
                index = torch.repeat_interleave(values, repeats)
                index_select = torch.index_select(embeddings, 0, index)

                u0, u1 = lst.tolist()
                backed0, backed1 = z.size(0), z.size(1)

                repeated0 = y.repeat(backed0 + u0, 1)
                repeated1 = x.repeat(backed1 + u1, 1)
                out1 = torch.empty_like(repeated1)
                add_kernel[(out1.numel(),)](
                    repeated1, repeated1, out1, out1.numel(), BLOCK_SIZE=2
                )

                # Implicitly add torch._check(expr2, unbacked)
                cat = torch.cat([out1, index_select], dim=1)
                add = repeated0 + repeated1

                # Explicitly add torch._check(expr1, expr2)
                torch._check(repeated0.size(0) == out1.size(0))
                return cat, add

        example_inputs = (
            torch.ones(64, dtype=torch.int64, device=self.device),
            torch.ones(64, dtype=torch.int64, device=self.device) * 24,
            torch.ones((768,), dtype=torch.int64, device=self.device).bool(),
            torch.randn((401, 8), dtype=torch.bfloat16, device=self.device),
            torch.randn((2, 256), dtype=torch.bfloat16, device=self.device),
            torch.randn((2, 256), dtype=torch.bfloat16, device=self.device),
            torch.ones(758, 758, dtype=torch.int64, device=self.device),
            torch.tensor([10, 10], dtype=torch.int32, device=self.device),
        )
        spec = {
            "values": (Dim.DYNAMIC,),
            "repeats": (Dim.DYNAMIC,),
            "mask": (Dim.DYNAMIC,),
            "embeddings": (Dim.DYNAMIC, Dim.STATIC),
            "x": (Dim.DYNAMIC, Dim.STATIC),
            "y": (Dim.DYNAMIC, Dim.STATIC),
            "z": (Dim.DYNAMIC, Dim.DYNAMIC),
            "lst": (Dim.STATIC,),
        }
        self.check_model(Repro(), example_inputs, dynamic_shapes=spec)

    @config.patch({"unbacked_symint_fallback": 128})
    def test_size_with_unbacked_add_and_mul_expr(self):
        # Edge case with torch._check(add_expr, mul_expr). When generating example
        # input sizes for autotuning, make sure they coalesce into a single size.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Repro(torch.nn.Module):
            def forward(self, values, repeats, mask, embeddings, x, y, z, lst):
                u0, u1, u2 = lst.tolist()
                backed = z.size(0)
                backed1 = z.size(1)

                unbacked_add_expr = backed + u0
                unbacked_mul_expr = backed1 + (u1 * u2)
                repeated0 = x.repeat(unbacked_add_expr, 1)
                repeated1 = y.repeat(unbacked_mul_expr, 1)
                out0 = torch.empty_like(repeated0)
                out1 = torch.empty_like(repeated1)
                add_kernel[(out0.numel(),)](
                    repeated0, repeated0, out0, out0.numel(), BLOCK_SIZE=2
                )
                add_kernel[(out1.numel(),)](
                    repeated1, repeated1, out1, out1.numel(), BLOCK_SIZE=2
                )

                return torch.cat([out1, out0], dim=1)

        example_inputs = (
            torch.ones(64, dtype=torch.int64, device=self.device),
            torch.ones(64, dtype=torch.int64, device=self.device) * 24,
            torch.ones((768,), dtype=torch.int64, device=self.device).bool(),
            torch.randn((401, 8), dtype=torch.bfloat16, device=self.device),
            torch.randn((2, 256), dtype=torch.bfloat16, device=self.device),
            torch.randn((2, 256), dtype=torch.bfloat16, device=self.device),
            torch.ones(758, 758, dtype=torch.int64, device=self.device),
            torch.tensor([10, 5, 2], dtype=torch.int32, device=self.device),
        )
        spec = {
            "values": (Dim.DYNAMIC,),
            "repeats": (Dim.DYNAMIC,),
            "mask": (Dim.DYNAMIC,),
            "embeddings": (Dim.DYNAMIC, Dim.STATIC),
            "x": (Dim.DYNAMIC, Dim.STATIC),
            "y": (Dim.DYNAMIC, Dim.STATIC),
            "z": (Dim.DYNAMIC, Dim.DYNAMIC),
            "lst": (Dim.STATIC,),
        }
        self.check_model(Repro(), example_inputs, dynamic_shapes=spec)

    @skipIfXpu(msg="_scaled_dot_product_flash_attention is not supported on XPU yet")
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Some archs don't support flash SDPA"
    )
    def test_fallback_kernel_with_symexpr_output(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Module(torch.nn.Module):
            def forward(self, q, k, v):
                q = q.reshape(
                    q.shape[0],
                    2,
                    q.shape[2] * q.shape[3],
                    q.shape[1] // 2,
                )
                k = k.reshape(
                    k.shape[0],
                    2,
                    k.shape[2] * k.shape[3],
                    k.shape[1] // 2,
                )
                v = v.reshape(
                    v.shape[0],
                    2,
                    v.shape[2] * v.shape[3],
                    v.shape[1] // 2,
                )

                res = torch.ops.aten._scaled_dot_product_flash_attention.default(
                    q,
                    k,
                    v,
                )
                return res[0]

        m = Module().to(device=self.device)
        tensor_shape = (4, 32, 4, 4)
        inputs = (
            torch.randn(tensor_shape, dtype=torch.float16, device=self.device),
            torch.randn(tensor_shape, dtype=torch.float16, device=self.device),
            torch.randn(tensor_shape, dtype=torch.float16, device=self.device),
        )

        dynamic_shapes = {
            "q": {2: Dim.DYNAMIC, 3: Dim.DYNAMIC},
            "k": {2: Dim.DYNAMIC, 3: Dim.DYNAMIC},
            "v": {2: Dim.DYNAMIC, 3: Dim.DYNAMIC},
        }
        ep = torch.export.export(m, inputs, dynamic_shapes=dynamic_shapes, strict=False)
        path = torch._inductor.aot_compile(ep.module(), inputs)
        aot_model = torch._export.aot_load(path, device=self.device)
        torch.testing.assert_close(m(*inputs), aot_model(*inputs))

    def test_aoti_constant_tensor(self):
        class Foo(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.a = torch.ones(4, 4, device=device)
                self.b = torch.ones(4, 4, device=device)

            def forward(self, x):
                return torch.ops.aten.linear.default(x, self.a, self.b)

        example_inputs = (torch.ones(4, 4, device=self.device),)
        self.check_model(Foo(self.device), example_inputs)

    def test_aoti_constant_tensor_name_collision(self):
        class SubModule(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.register_buffer(
                    "_tensor_constant1",
                    torch.ones(1, device=device, dtype=torch.float32),
                    persistent=True,
                )

            def forward(self, x):
                return self.linear(x)

        class Foo(torch.nn.Module):
            def __init__(self, user_float_feature_idx, device):
                super().__init__()
                self.user_float_feature_idx = user_float_feature_idx
                self.register_buffer(
                    "_tensor_constant0",
                    torch.ones(5, device=device, dtype=torch.float32),
                    persistent=True,
                )
                self.register_buffer(
                    "_tensor_constant1",
                    torch.ones(1, device=device, dtype=torch.float32),
                    persistent=True,
                )
                self.sub_mod = SubModule(device)

            def forward(self, x):
                self._tensor_constant0[1:2] = 1
                return (
                    torch.index_select(
                        x, 1, torch.tensor(self.user_float_feature_idx, device=x.device)
                    ),
                    self._tensor_constant0,
                    self._tensor_constant1,
                    self.sub_mod._tensor_constant1,
                )

        example_inputs = (torch.ones(4, 4, device=self.device),)
        user_float_feature_idx = [1]
        # we have to have run_decomposition first to trigger the name collision
        ep = torch.export.export(
            Foo(user_float_feature_idx, self.device), example_inputs, strict=False
        ).run_decompositions()
        gm = ep.module()
        self.check_model(gm.to(self.device), example_inputs)

    def test_large_grid(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, primals_5):
                view = torch.ops.aten.reshape.default(primals_5, [-1, 2, 4])
                primals_5 = None
                permute = torch.ops.aten.permute.default(view, [0, 2, 1])
                clone = torch.ops.aten.clone.default(
                    permute, memory_format=torch.contiguous_format
                )
                return clone

        # let y_grid = 65537
        s0 = 16777472
        s1 = 8
        example_inputs = (torch.rand(s0, s1, device=self.device),)
        self.check_model(Model(), example_inputs)

    def test_cond_simple(self):
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        dim0_ab = Dim("s0", min=2, max=1024)
        dynamic_shapes = {
            "p": {},
            "a": {0: dim0_ab, 1: None},
            "b": {0: dim0_ab, 1: None},
        }
        self.check_model_with_multiple_inputs(
            CondModels.Simple(),
            prepend_predicates(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    def test_cond_nested(self):
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        dim0_abc = Dim("s0", min=2, max=1024)
        dynamic_shapes = {
            "p0": {},
            "p1": {},
            "p2": {},
            "a": {0: dim0_abc, 1: None},
            "b": {0: dim0_abc, 1: None},
            "c": {0: dim0_abc, 1: None},
        }
        self.check_model_with_multiple_inputs(
            CondModels.Nested(),
            prepend_predicates(inputs, num_predicates=3),
            dynamic_shapes=dynamic_shapes,
        )

    def test_cond_with_parameters(self):
        inputs = (torch.randn((10, 20), device=self.device),)
        dim0_abc = Dim("s0", min=2, max=1024)
        dynamic_shapes = {
            "p": {},
            "a": {0: dim0_abc, 1: None},
        }
        self.check_model_with_multiple_inputs(
            CondModels.Parameters(self.device),
            prepend_predicates(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    def test_cond_with_reinterpret_view_inputs_outputs(self):
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        # TODO: the min value need to be 5 because in the body_fn, we're slicing over z1[2:],
        # since the output size is [dim0_ab-3], when we extract tensor metadata out of the output
        # we call guard_size_oblivious, which assumes the dim0_ab-3 != 0 or 1. So we have to set
        # the minimum to 5 for now. We need to relax this restriction either by writing a less
        # constrained shape checking in fake impl of cond.
        dim0_ab = Dim("s0", min=5, max=1024)
        dynamic_shapes = {
            "p": {},
            "a": {0: dim0_ab, 1: None},
            "b": {0: dim0_ab, 1: None},
        }
        self.check_model_with_multiple_inputs(
            CondModels.ReinterpretView(),
            prepend_predicates(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    def test_cond_with_multiple_outputs(self):
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
            torch.randn((30, 40), device=self.device),
        )
        dim0_ab = Dim("s0", min=2, max=1024)
        dim0_c = Dim("s1", min=2, max=1024)
        dynamic_shapes = {
            "p": {},
            "a": {0: dim0_ab, 1: None},
            "b": {0: dim0_ab, 1: None},
            "c": {0: dim0_c, 1: None},
        }
        self.check_model_with_multiple_inputs(
            CondModels.MultipleOutputs(),
            prepend_predicates(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    def test_cond_with_outer_code_before_after(self):
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        dim0_ab = Dim("s0", min=2, max=1024)
        dynamic_shapes = {
            "p": {},
            "a": {0: dim0_ab, 1: None},
            "b": {0: dim0_ab, 1: None},
        }
        self.check_model_with_multiple_inputs(
            CondModels.OuterCode(),
            prepend_predicates(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    def test_cond_use_buffers_from_outer_scope(self):
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        dim0_abc = Dim("s0", min=2, max=1024)
        dynamic_shapes = {
            "p": {},
            "a": {0: dim0_abc, 1: None},
            "b": {0: dim0_abc, 1: None},
            "c": {0: dim0_abc, 1: None},
        }
        self.check_model_with_multiple_inputs(
            CondModels.OuterBuffers(),
            prepend_predicates(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    @common_utils.parametrize("dynamic", [False, True])
    def test_cond_non_tensor_predicates(self, dynamic):
        inputs1 = (
            torch.randn((10, 20), device=self.device),
            torch.randn((15, 20), device=self.device),
        )
        inputs2 = (
            torch.randn((10, 20), device=self.device),
            torch.randn((5, 20), device=self.device),
        )
        inputs = (inputs1,)
        dynamic_shapes = None
        if dynamic:
            inputs = (inputs1, inputs2)
            dim0_a = Dim("s0", min=2, max=1024)
            dim0_b = Dim("s1", min=2, max=1024)
            dynamic_shapes = {
                "a": {0: dim0_a, 1: None},
                "b": {0: dim0_b, 1: None},
            }
        self.check_model_with_multiple_inputs(
            CondModels.WithNonTensorPredicate(),
            inputs,
            dynamic_shapes=dynamic_shapes,
        )

    @common_utils.parametrize("dynamic", [False, True])
    def test_cond_unbacked_symint_closure(self, dynamic):
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((15, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        dynamic_shapes = None
        if dynamic:
            dim0_a = Dim("s0", min=2, max=1024)
            dim0_b = Dim("s1", min=2, max=1024)
            dynamic_shapes = {
                "p": {},
                "x": {0: dim0_a, 1: None},
                "y": {0: dim0_b, 1: None},
                "z": {0: dim0_a, 1: None},
            }
        self.check_model_with_multiple_inputs(
            CondModels.UnbackedSymIntClosure(),
            prepend_predicates(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    @skipIfWindows(msg="TODO: (xuhancn) confirm, Crash: access violation")
    @common_utils.parametrize("dynamic", [False, True])
    def test_cond_mismatched_branch_output(self, dynamic):
        inputs = (
            torch.randn(10, 20, device=self.device),
            torch.randn(10, 20, device=self.device),
            torch.randn(10, 20, device=self.device),
        )
        dynamic_shapes = None
        if dynamic:
            # Note the minimum has to be 4 because the model
            # is slicing over the first dim with [2:], if first
            # dim is 2 or 3, the slicing will be 0/1 specialized,
            # causing a constraint violation error.
            dim0_a = Dim("s0", min=4, max=1024)
            dim0_b = Dim("s1", min=4, max=1024)
            dynamic_shapes = {
                "p": {},
                "x": {0: dim0_a, 1: None},
                "y": {0: dim0_b, 1: None},
                "z": {0: dim0_a, 1: None},
            }
        self.check_model_with_multiple_inputs(
            CondModels.MismatchedOutputSize(),
            prepend_predicates(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    def test_cond_symint_input(self):
        class M(torch.nn.Module):
            def forward(self, x, y, z):
                a = y.shape[0]
                b = z.shape[0]

                def true_fn(x):
                    return x + a

                def false_fn(x):
                    return x + b * z

                return torch.cond(x.shape[0] > 5, true_fn, false_fn, (x,))

        input1 = (
            torch.ones(3, 3, device=self.device),
            torch.ones(5, device=self.device),
            torch.ones(3, 3, device=self.device),
        )
        input2 = (
            torch.ones(10, 3, device=self.device),
            torch.ones(6, device=self.device),
            torch.ones(10, 3, device=self.device),
        )
        inputs = (input1, input2)
        dynamic_shapes = {"x": {0: Dim("d")}, "y": {0: Dim("d1")}, "z": {0: Dim("d")}}
        self.check_model_with_multiple_inputs(
            M(),
            inputs,
            dynamic_shapes=dynamic_shapes,
        )

    def test_cond_symint_input_disable_one_pass(self):
        class M(torch.nn.Module):
            def forward(self, x, y, z):
                a = y.shape[0]
                b = z.shape[0]

                def true_fn(x):
                    return x + a

                def false_fn(x):
                    return x + b * z

                return torch.cond(x.shape[0] > 5, true_fn, false_fn, (x,))

        input1 = (
            torch.ones(3, 3, device=self.device),
            torch.ones(5, device=self.device),
            torch.ones(3, 3, device=self.device),
        )
        input2 = (
            torch.ones(10, 3, device=self.device),
            torch.ones(6, device=self.device),
            torch.ones(10, 3, device=self.device),
        )
        inputs = (input1, input2)
        dynamic_shapes = {"x": {0: Dim("d")}, "y": {0: Dim("d1")}, "z": {0: Dim("d")}}
        with torch._inductor.config.patch({"triton.autotune_at_compile_time": False}):
            self.check_model_with_multiple_inputs(
                M(),
                inputs,
                dynamic_shapes=dynamic_shapes,
            )

    def test_while_loop_simple(self):
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        dim0_ab = Dim("s0", min=2, max=1024)
        dynamic_shapes = {
            "ci": {},
            "a": {0: dim0_ab, 1: None},
            "b": {0: dim0_ab, 1: None},
        }
        self.check_model_with_multiple_inputs(
            WhileLoopModels.Simple(),
            prepend_counters(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    def test_while_loop_nested(self):
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        dim0_ab = Dim("s0", min=2, max=1024)
        dynamic_shapes = {
            "ci": {},
            "cj": {},
            "a": {0: dim0_ab, 1: None},
            "b": {0: dim0_ab, 1: None},
        }
        self.check_model_with_multiple_inputs(
            WhileLoopModels.Nested(),
            prepend_counters(inputs, num_counters=2),
            dynamic_shapes=dynamic_shapes,
        )

    def test_while_loop_with_outer_code(self):
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        dim0_ab = Dim("s0", min=2, max=1024)
        dynamic_shapes = {
            "c": {},
            "a": {0: dim0_ab, 1: None},
            "b": {0: dim0_ab, 1: None},
        }
        self.check_model_with_multiple_inputs(
            WhileLoopModels.OuterCode(),
            prepend_counters(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    # mps doesn't support float64
    @skipIfMPS
    @unittest.skipIf(
        config.triton.native_matmul,
        "FIXME: cannot do get_size on FakeTensor during lowering.",
    )
    def test_while_loop_with_parameters(self):
        inputs = (
            torch.randn(
                (
                    10,
                    20,
                ),
                dtype=torch.float64,
                device=self.device,
            ),
        )
        dim0_a = Dim("s0", min=2, max=1024)
        dynamic_shapes = {
            "c": {},
            "a": {0: dim0_a, 1: None},
        }
        self.check_model_with_multiple_inputs(
            WhileLoopModels.Parameters(self.device),
            prepend_counters(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    def test_while_loop_with_outer_buffers(self):
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        # dynamic shapes don't work now due to
        # https://github.com/pytorch/pytorch/issues/123596
        # dim0_ab = Dim("s0", min=2, max=1024)
        # dynamic_shapes = {
        #     "c": {},
        #     "a": {0: dim0_ab, 1: None},
        #     "b": {0: dim0_ab, 1: None},
        # }
        dynamic_shapes = None
        self.check_model_with_multiple_inputs(
            WhileLoopModels.OuterBuffers(),
            prepend_counters(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    def test_while_loop_with_pytree_inputs(self):
        inputs = (
            torch.tensor(0, device=self.device),
            (
                [torch.randn(10, 20, device=self.device)],
                {
                    "x": torch.randn(10, 20, device=self.device),
                    "y": torch.randn(10, 20, device=self.device),
                },
            ),
        )
        self.check_model_with_multiple_inputs(
            WhileLoopModels.PytreeCarry(),
            [inputs],
            dynamic_shapes=None,
        )

    @common_utils.parametrize("dynamic", [False, True])
    def test_while_loop_with_unbacked_symint_closure(self, dynamic):
        inputs = (
            torch.randn(10, 20, device=self.device),
            torch.randn(10, 20, device=self.device),
        )
        dim0_ab = Dim("s0", min=2, max=1024)
        dynamic_shapes = None
        if dynamic:
            dynamic_shapes = {
                "c": {},
                "a": {0: dim0_ab, 1: None},
                "b": {0: dim0_ab, 1: None},
            }
        self.check_model_with_multiple_inputs(
            WhileLoopModels.UnbackedSymIntClosure(),
            prepend_counters(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    @common_utils.parametrize("dynamic", [False, True])
    def test_while_loop_with_mixed_device(self, dynamic):
        inputs = (
            torch.randn(10, 20, device=self.device),
            torch.randn(10, 20, device=self.device),
        )
        dim0_ab = Dim("s0", min=2, max=1024)
        dynamic_shapes = None
        if dynamic:
            dynamic_shapes = {
                "c": {},
                "a": {0: dim0_ab, 1: None},
                "b": {0: dim0_ab, 1: None},
            }
        self.check_model_with_multiple_inputs(
            WhileLoopModels.MixedDevice(),
            prepend_counters(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    @common_utils.parametrize("dynamic", [False, True])
    def test_while_loop_with_sym_expr_cond(self, dynamic):
        inputs = (
            torch.randn(10, 20, device=self.device),
            torch.randn(10, 20, device=self.device),
        )
        dim0_ab = Dim("s0", min=2, max=1024)
        dynamic_shapes = None
        if dynamic:
            dynamic_shapes = {
                "c": {},
                "a": {0: dim0_ab, 1: None},
                "b": {0: dim0_ab, 1: None},
            }
        self.check_model_with_multiple_inputs(
            WhileLoopModels.SymExprCond(),
            prepend_counters(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    @common_utils.parametrize("dynamic", [False, True])
    def test_while_loop_with_conv(self, dynamic):
        inputs = (torch.randn(2, 4, 4, 4, device=self.device, dtype=torch.float64),)
        dim0_ab = Dim("s0", min=2, max=1024)
        dynamic_shapes = None
        if dynamic:
            dynamic_shapes = {
                "c": {},
                "x": {0: dim0_ab, 1: None},
            }
        self.check_model_with_multiple_inputs(
            WhileLoopModels.Conv(self.device),
            prepend_counters(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    @config.patch({"is_predispatch": True})
    def test_constant(self):
        class M(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.device = device

            def forward(self, x):
                t = torch.tensor(x.size(-1), device=self.device, dtype=torch.float)
                t = torch.sqrt(t * 3)
                return x * t

        self.check_model(M(self.device), (torch.randn(5, 5, device=self.device),))

    @unittest.skipIf(IS_MACOS, "no CUDA on Mac")
    def test_zero_grid_with_backed_symbols(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Repro(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, b):
                return x + b

        example_inputs = (
            torch.randn((3, 2), device=self.device),
            torch.randn((1, 2), device=self.device),
        )
        dynamic_shapes = {
            "x": {0: Dim("dx"), 1: Dim.STATIC},
            "b": None,
        }

        # Compile & run model where dynamic dim size > 0.
        package_path: str = AOTIRunnerUtil.compile(
            Repro(),
            example_inputs,
            dynamic_shapes=dynamic_shapes,
        )
        aot_inductor_module = torch._inductor.aoti_load_package(package_path)
        aot_inductor_module(*example_inputs)

        # Re-run where dynamic dim size is 0.
        example_inputs = (
            torch.randn((0, 2), device=self.device),
            torch.randn((1, 2), device=self.device),
        )
        actual = aot_inductor_module(*example_inputs)
        expected = Repro()(*example_inputs)
        torch.testing.assert_close(actual, expected)

    def test_repeat_interleave(self):
        class Repro(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return torch.ops.aten.repeat_interleave.Tensor(x, output_size=12)

        example_inputs = (torch.ones((1,), dtype=torch.int32, device=self.device) * 12,)
        self.check_model(Repro(), example_inputs)

    def test_dynamic_cat(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, a, b):
                return torch.cat([a, b], dim=0)

        a = torch.randn(2, 4, device=self.device)
        b = torch.randn(3, 4, device=self.device)
        dim0_a = Dim("dim0_a", min=1, max=10)
        dim0_b = Dim("dim0_b", min=1, max=20)
        dynamic_shapes = {"a": {0: dim0_a}, "b": {0: dim0_b}}
        example_inputs = (a, b)
        self.check_model(Model(), example_inputs, dynamic_shapes=dynamic_shapes)

    def test_buffer_mutation_1(self):
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.foo = torch.nn.Buffer(torch.randn(4, 4, device=device))

            def forward(self, x):
                self.foo.add_(1)
                return self.foo + x

        example_inputs = (torch.rand(4, 4, device=self.device),)
        self.check_model(Model(self.device), example_inputs)

    def test_non_tensor_input(self):
        class Model(torch.nn.Module):
            def forward(self, a, b, alpha=1.0):
                return torch.add(a, b, alpha=alpha)

        a = torch.randn(10, device=self.device)
        b = torch.randn(10, device=self.device)

        for simdlen in [0, None]:
            with torch._inductor.config.patch({"cpp.simdlen": simdlen}):
                so_path = torch._export.aot_compile(
                    torch.ops.aten.add,
                    args=(a, b),
                    kwargs={"alpha": 2.0},
                )
                kernel_runner = AOTIRunnerUtil.legacy_load_runner(self.device, so_path)
                res = kernel_runner.run([a, b])
                self.assertTrue(isinstance(res, list))
                self.assertTrue(len(res) == 1)
                self.assertEqual(Model()(a, b, alpha=2.0), res[0])

    def test_buffer_mutation_2(self):
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.foo = torch.nn.Buffer(torch.arange(10, device=device))
                self.bar = torch.nn.Buffer(torch.arange(10, device=device))

            def forward(self, x):
                self.bar.mul_(2)
                self.foo[5] = self.bar[0]
                return x + self.bar, x * self.foo

        example_inputs = (torch.randn(10, device=self.device),)
        self.check_model(Model(self.device), example_inputs)

    @skipIfWindows(
        msg="OpenMP crashed application on windows"
    )  # TODO: (xuhancn) need to root cause and fix.
    def test_buffer_mutation_3(self):
        class KVCache(torch.nn.Module):
            def __init__(
                self,
                max_batch_size,
                max_seq_length,
                n_heads,
                head_dim,
                dtype=torch.float,
            ):
                super().__init__()
                cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
                self.k_cache = torch.nn.Buffer(torch.zeros(cache_shape, dtype=dtype))
                self.v_cache = torch.nn.Buffer(torch.zeros(cache_shape, dtype=dtype))

            def update(self, input_pos, k_val, v_val):
                # input_pos: [S], k_val: [B, H, S, D]
                k_out = self.k_cache
                v_out = self.v_cache
                k_out[:, :, input_pos] = k_val
                v_out[:, :, input_pos] = v_val

                return k_out, v_out

        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.kv_cache = KVCache(1, 256, 6, 48)

            def forward(self, inp_pos, k, v):
                self.kv_cache.update(inp_pos, k, v)
                return self.kv_cache.k_cache + 1, self.kv_cache.v_cache / 2

        example_inputs = (
            torch.tensor([0], device=self.device),
            torch.randn(1, 6, 1, 48, device=self.device),
            torch.randn(1, 6, 1, 48, device=self.device),
        )
        model = Model(self.device)
        self.check_model(model, example_inputs)
        self.code_check_count(model, example_inputs, "empty_strided", 2)

    def test_buffer_mutation_4(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer(
                    "_tensor_constant0",
                    torch.randint(1, size=[38], dtype=torch.int64, device="cpu"),
                )

            def forward(self, x):
                return x + self._tensor_constant0.to(
                    torch.device(type=GPU_TYPE, index=0)
                )

        example_inputs = (
            torch.randint(1, size=[38], dtype=torch.int64, device=GPU_TYPE),
        )
        torch._export.aot_compile(Model(), example_inputs)

    @skipCUDAIf(True, "Test for x86 backend")
    @unittest.skipIf(IS_FBCODE, "Need newer ideep")
    def test_buffer_mutation_and_force_mmap_weights(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(16, 15)
                self.linear2 = torch.nn.Linear(15, 14)

            def forward(self, x):
                x = self.linear1(x)
                out = self.linear2(x)
                return out

        example_inputs = (torch.randn(32, 16),)
        model = Model().eval()
        with (
            config.patch({"freezing": True, "aot_inductor.force_mmap_weights": True}),
            torch.no_grad(),
        ):
            exported_model = export(model, example_inputs, strict=True).module()
            quantizer = X86InductorQuantizer()
            quantizer.set_global(
                xiq.get_default_x86_inductor_quantization_config(reduce_range=True)
            )
            prepared_model = prepare_pt2e(exported_model, quantizer)
            prepared_model(*example_inputs)
            converted_model = convert_pt2e(prepared_model)
            torch.ao.quantization.move_exported_model_to_eval(converted_model)

            self.check_model(converted_model, example_inputs)

    @skipIfMPS
    def test_fallback_mem_leak_fix(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y, idx):
                tmp = x + y
                w = torch.ops.aten.as_strided(tmp, x.shape, x.stride())
                out = torch.ops.aten.index.Tensor(w, [idx])
                return w, out

        example_inputs = (
            torch.randn(4, 1, 4, device=GPU_TYPE),
            torch.randn(4, 1, 4, device=GPU_TYPE),
            torch.randn(4, device=GPU_TYPE) > 0,
        )

        dim0 = Dim("dim0", min=1, max=2048)
        dynamic_shapes = {
            "x": {0: dim0},
            "y": {0: dim0},
            "idx": {0: dim0},
        }
        package_path: str = AOTIRunnerUtil.compile(
            Model(),
            example_inputs,
            dynamic_shapes=dynamic_shapes,
        )
        aot_inductor_module = torch._inductor.aoti_load_package(package_path)
        device_interface = get_interface_for_device(GPU_TYPE)
        device: int = device_interface.current_device()
        mem_before = device_interface.memory_allocated(device)
        aot_inductor_module(*example_inputs)
        mem_after = device_interface.memory_allocated(device)
        self.assertEqual(mem_before, mem_after)

        actual = aot_inductor_module(*example_inputs)
        expected = Model()(*example_inputs)
        torch.testing.assert_close(actual, expected)

    @requires_multigpu()
    @skipIfMPS
    def test_replicate_on_devices(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            def __init__(self, w1, w2):
                super().__init__()
                self.w1 = w1
                self.w2 = w2

            def forward(self, x, y):
                a = x * self.w1
                b = y * self.w2
                return a + b

        w1 = torch.randn(10, 10)
        w2 = torch.randn(10, 10)
        inputs = (torch.randn(10, 10), torch.randn(10, 10))
        result_cpu = Model(w1, w2)(*inputs)

        # Compile model with AOTInductor
        device_interface = get_interface_for_device(GPU_TYPE)
        with device_interface.device(0):
            package_path = AOTIRunnerUtil.compile(
                model=Model(
                    w1.to(torch.device(GPU_TYPE, 0)), w2.to(torch.device(GPU_TYPE, 0))
                ),
                example_inputs=tuple(t.to(torch.device(GPU_TYPE, 0)) for t in inputs),
            )

        # Run model on gpu:N
        for i in range(device_interface.device_count()):
            with device_interface.device(i):
                example_inputs = tuple(t.to(torch.device(GPU_TYPE, i)) for t in inputs)
                optimized = torch._inductor.aoti_load_package(package_path)
                result_gpu = optimized(*example_inputs)
            self.assertTrue(same(result_cpu, result_gpu.cpu()))

    @requires_multigpu()
    @skipIfMPS
    def test_on_gpu_device1(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        device_interface = get_interface_for_device(GPU_TYPE)
        try:
            device_interface.get_device_properties(1)
        except AssertionError:
            raise unittest.SkipTest("GPU device 1 is not available") from None

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(10, 16)
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(16, 1)
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                x = self.sigmoid(x)
                return x

        device = f"{GPU_TYPE}:1"
        model = Model().to(device)
        example_inputs = (torch.randn(8, 10, device=device),)
        expected = model(*example_inputs)

        so_path = AOTIRunnerUtil.legacy_compile(model, example_inputs)
        optimized = AOTIRunnerUtil.legacy_load(device, so_path)
        actual = optimized(*example_inputs)
        torch.testing.assert_close(actual, expected)

    def test_pytree_inputs(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: dict[str, torch.Tensor]):
                device = next(iter(x.values())).device
                add_ = torch.zeros(5, device=device)
                mul_ = torch.ones(5, device=device)
                for v in x.values():
                    add_ += v
                    mul_ *= v

                return [add_, mul_]

        self.check_model(
            M(),
            (
                {
                    "x": torch.ones(5, device=self.device),
                    "y": torch.ones(5, device=self.device),
                },
            ),
        )

    @requires_multigpu()
    def test_non_default_gpu_device(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight

            def forward(self, x, y):
                return x + torch.nn.functional.linear(y, self.weight)

        weight = torch.randn(10, 10)
        inputs = (torch.randn(10, 10), torch.randn(10, 10))
        result_cpu = Model(weight)(*inputs)

        device_interface = get_interface_for_device(GPU_TYPE)
        with device_interface.device(0), torch.no_grad():
            result_gpu_0 = AOTIRunnerUtil.run(
                Model(weight.to(torch.device(GPU_TYPE, 0))),
                tuple(t.to(torch.device(GPU_TYPE, 0)) for t in inputs),
            )

        with device_interface.device(1), torch.no_grad():
            result_gpu_1 = AOTIRunnerUtil.run(
                Model(weight.to(torch.device(GPU_TYPE, 1))),
                tuple(t.to(torch.device(GPU_TYPE, 1)) for t in inputs),
            )

        self.assertTrue(same(result_cpu, result_gpu_0.cpu()))
        self.assertTrue(same(result_cpu, result_gpu_1.cpu()))

    @requires_multigpu()
    def test_load_package_multiple_gpus(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight

            def forward(self, x, y):
                return x + torch.nn.functional.linear(y, self.weight)

        weight = torch.randn(10, 10, device=self.device)
        inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        model = Model(weight).to(device=self.device)
        result_ref = model(*inputs)

        package_path = AOTIRunnerUtil.compile(model, inputs)

        # Load AOT package on gpu:N
        device_interface = get_interface_for_device(GPU_TYPE)
        for i in range(device_interface.device_count()):
            device = torch.device(GPU_TYPE, i)
            with device_interface.device(i), torch.no_grad():
                model_package = torch._inductor.aoti_load_package(
                    package_path, device_index=i
                )
                inputs_on_device = [input.to(device=device) for input in inputs]
                result_package = model_package(*inputs_on_device)
            self.assertTrue(same(result_ref.cpu(), result_package.cpu()))

    @unittest.skipIf(
        config.triton.native_matmul, "sin and mm are fused in native matmul"
    )
    def test_reuse_kernel(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                a = torch.sin(x)
                b = torch.mm(a, y)
                c = torch.sin(b)
                d = torch.mm(b, c)
                return d

        example_inputs = (
            torch.randn(87, 87, device=self.device),
            torch.randn(87, 87, device=self.device),
        )
        model = Model()

        # 1e-4 is the tol value used in pytorch/torch/_dynamo/utils.py
        self.check_model(model, example_inputs, atol=1e-4, rtol=1e-4)

        if self.device == "mps":
            self.code_check_count(
                model, example_inputs, "aoti_torch_mps_get_kernel_function(", 1
            )
        elif self.device == GPU_TYPE:
            self.code_check_count(
                model, example_inputs, "triton_poi_fused_sin_0 = loadKernel(", 1
            )

    def test_reuse_kernel_dynamic(self):
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.cst = torch.randn(48, device=device, dtype=torch.float)
                self.weights = torch.randn(6, 48, 48, device=device, dtype=torch.float)
                self.cst_1 = torch.randn(48, device=device, dtype=torch.float)
                self.weights_1 = torch.randn(
                    6, 48, 48, device=device, dtype=torch.float
                )

            def forward(self, x, y, z):
                dim0 = x.size(1)
                add_0 = z + z
                expand_2 = add_0.expand(-1, -1, 48)
                # [s0, 6, 48]
                mul_3 = add_0 * expand_2
                # [6, s0, 48]
                permute_4 = torch.permute(mul_3, (1, 0, 2))
                # [6, s0, 48]
                bmm_5 = torch.bmm(permute_4, self.weights)
                add_6 = bmm_5 + self.cst
                reshape_7 = torch.reshape(add_6, [6, dim0 * 6, 8])
                # [6*s0, 6, 8]
                permute_8 = torch.permute(reshape_7, (1, 0, 2))
                mul_9 = permute_8 * 0.123
                reshape_10 = torch.reshape(y, [8, dim0 * 6, 4])
                # [6*s0, 8, 4]
                permute_11 = torch.permute(reshape_10, (1, 0, 2))
                bmm_12 = torch.bmm(mul_9, permute_11)

                add_0_1 = z + z
                expand_2_1 = add_0_1.expand(-1, -1, 48)
                # [s0, 6, 48]
                mul_3_1 = add_0_1 * expand_2_1
                # [6, s0, 48]
                permute_4_1 = torch.permute(mul_3_1, (1, 0, 2))
                # [6, s0, 48]
                bmm_5_1 = torch.bmm(permute_4_1, self.weights_1)
                add_6_1 = bmm_5_1 + self.cst_1
                reshape_7_1 = torch.reshape(add_6_1, [6, dim0 * 6, 8])
                # [6*s0, 6, 8]
                permute_8_1 = torch.permute(reshape_7_1, (1, 0, 2))
                mul_9_1 = permute_8_1 * 0.123
                reshape_10_1 = torch.reshape(y, [8, dim0 * 6, 4])
                # [6*s0, 8, 4]
                permute_11_1 = torch.permute(reshape_10_1, (1, 0, 2))
                bmm_12_1 = torch.bmm(mul_9_1, permute_11_1)
                return bmm_12 + bmm_12_1

        x = torch.randn(6, 2, 48, device=self.device, dtype=torch.float)
        y = torch.randn(48, 2, 4, device=self.device, dtype=torch.float)
        z = torch.randn(2, 6, 1, device=self.device, dtype=torch.float)
        dim0 = Dim("dim0", min=1, max=2048)
        dynamic_shapes = {
            "x": {1: dim0},
            "y": {1: dim0},
            "z": {0: dim0},
        }

        example_inputs = (x, y, z)
        model = Model(self.device).to(dtype=torch.float)
        self.check_model(
            model,
            example_inputs,
            dynamic_shapes=dynamic_shapes,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_fake_tensor_device_validation(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                return x + y

        example_inputs = (torch.randn(10, 10), torch.randn(10, 10))

        # Export on CPU
        exported_program = export(Model(), example_inputs, strict=True)

        # Compile exported model on GPU
        gm = exported_program.graph_module.to(self.device)
        with self.assertRaisesRegex(ValueError, "Device mismatch between fake input"):
            torch._inductor.aot_compile(
                gm, tuple(i.to(self.device) for i in example_inputs)
            )

    def test_fx_gm_return_tuple_validation(self):
        from torch.fx.experimental.proxy_tensor import make_fx

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                return x + y

        example_inputs = (torch.randn(10, 10), torch.randn(10, 10))

        gm = make_fx(Model(), tracing_mode="symbolic")(*example_inputs)
        with self.assertRaisesRegex(
            AssertionError,
            r"Graph output must be a tuple\(\). This is so that we can avoid "
            "pytree processing of the outputs.",
        ):
            torch._inductor.aot_compile(gm, example_inputs)

    def test_consecutive_compiles(self):
        """Test that compilation behaves correctly with cache hits"""

        class TestModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return x + 1

        mod = TestModule()
        inp = torch.rand(1)
        mod(inp)
        mod2 = torch.fx.symbolic_trace(mod, concrete_args=[inp])
        so = torch._export.aot_compile(mod2, (inp,))
        assert so is not None
        # compile the 2nd time with cache hit
        so = torch._export.aot_compile(mod2, (inp,))
        assert so is not None

    def test_normal_functional(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return torch.ops.aten.normal_functional.default(x)

        self.check_model(Model(), (torch.empty(4, 1, 4, 4, device=self.device),))

    def test_empty_graph(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return x

        example_inputs = (torch.randn(8, 4, 4, device=self.device),)
        self.check_model(Model(), example_inputs)

    @patch("torch._dynamo.utils.CompileEventLogger.log_instant_event")
    def test_backward_no_op_logging(self, mock_log_instant_event):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return x

        model = Model()
        dummy_input = torch.randn(1, 5)

        from torch._dynamo.utils import CompileEventLogLevel
        from torch._inductor import compile_fx

        graph_module = torch.fx.symbolic_trace(model)
        compile_fx._compile_fx_inner(graph_module, (dummy_input,))
        mock_log_instant_event.assert_called_once_with(
            "backward no-op",
            metadata={"compile_id": None},
            log_level=CompileEventLogLevel.PT2_COMPILE,
        )

    @unittest.skipIf(IS_FBCODE, "Not runnable in fbcode")
    def test_dup_unbacked_sym_decl(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                abs_1 = torch.ops.aten.abs.default(x)
                lt = torch.ops.aten.lt.Scalar(abs_1, 0.001)
                eq = torch.ops.aten.eq.Scalar(lt, 0)
                index_1 = torch.ops.aten.index.Tensor(x, [eq])
                sin = torch.ops.aten.sin.default(index_1)
                index_2 = torch.ops.aten.index.Tensor(x, [eq])
                div_3 = torch.ops.aten.div.Tensor(sin, index_2)
                return div_3

        example_inputs = (torch.randn(4, 4, 4, 4).to(self.device),)
        self.check_model(Model(), example_inputs)

    # This exercises _eliminate_unbacked path in ShapeEnv
    @unittest.skipIf(IS_FBCODE, "Not runnable in fbcode")
    def test_dup_unbacked_sym_decl_with_refinement(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                abs_1 = torch.ops.aten.abs.default(x)
                lt = torch.ops.aten.lt.Scalar(abs_1, 0.001)
                eq = torch.ops.aten.eq.Scalar(lt, 0)
                index_1 = torch.ops.aten.index.Tensor(x, [eq])
                torch._check(index_1.size(0) == 4**4)
                sin = torch.ops.aten.sin.default(index_1)
                index_2 = torch.ops.aten.index.Tensor(x, [eq])
                div_3 = torch.ops.aten.div.Tensor(sin, index_2)
                return div_3

        example_inputs = (torch.ones(4, 4, 4, 4).to(self.device),)
        self.check_model(Model(), example_inputs)

    def test_run_with_grad_enabled(self):
        class Model(torch.nn.Module):
            def forward(self, x, weight, bias):
                return torch.ops.aten.addmm(bias, weight, x)

        m = Model().to(device=self.device)
        x = torch.rand(8, 8, device=self.device, requires_grad=True)
        weight = torch.rand(8, 8, device=self.device, requires_grad=True)
        bias = torch.rand(8, device=self.device, requires_grad=True)
        example_inputs = (x, weight, bias)

        expected = m(*example_inputs)
        expected = pytree.tree_leaves(expected)

        # compiler under no_grad
        with torch.no_grad():
            package_path = AOTIRunnerUtil.compile(m, example_inputs)

        # run under grad enabled
        self.assertTrue(torch.is_grad_enabled())

        optimized = torch._inductor.aoti_load_package(package_path)
        actual = optimized(*example_inputs)
        actual = pytree.tree_leaves(actual)

        self.assertTrue(same(actual, expected))

    def test_return_constant(self):
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.cst = torch.randn(5, 5, device=device)

            def forward(self, x):
                a = self.cst.clone()
                return (x, a)

        x = torch.randn(5, device=self.device)
        self.check_model(Model(self.device), (x,))

    def test_return_view_constant(self):
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.cst = torch.randn(5, 5, device=device)

            def forward(self, x):
                a = torch.transpose(self.cst, 0, 1)
                return (x, a)

        x = torch.randn(5, device=self.device)
        self.check_model(Model(self.device), (x,))

    def test_profile_benchmark_harness(self):
        batch_size = 32
        seq_length = 50
        hidden_size = 768

        def create_test_fn():
            def test_fn():
                inp = torch.randn(
                    batch_size, seq_length, hidden_size, device=self.device
                )
                weight = torch.randn(hidden_size, hidden_size, device=self.device)
                matmul_output = inp @ weight
                torch.nn.LayerNorm(hidden_size, device=self.device)(matmul_output)
                return True

            return test_fn

        fn = torch.compile(
            options={"profile_bandwidth_output": "foo", "benchmark_harness": False}
        )(create_test_fn())
        fn()

    def test_with_profiler(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x, y):
                return x + self.linear(y)

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        with config.patch({"profile_bandwidth": "1", "profile_bandwidth_regex": ""}):
            self.check_model(Model(), example_inputs)

    def test_with_no_triton_profiler(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return torch.permute(x, (1, 0))

        example_inputs = (torch.randn(10, 10, device=self.device),)
        with config.patch({"profile_bandwidth": "1", "profile_bandwidth_regex": ""}):
            self.check_model(Model(), example_inputs)

    def test_repeat_output(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                y = torch.sin(x)
                return y, y

        example_inputs = (torch.randn(3, 10, device=self.device),)
        self.check_model(Model(), example_inputs)

    def test_repeated_calling(self):
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return torch.sin(x)

        example_inputs = (torch.randn(10, 10, device=self.device),)
        optimized = torch._inductor.aoti_load_package(
            torch._inductor.aoti_compile_and_package(
                torch.export.export(Model(), example_inputs, strict=True)
            )
        )
        try:
            torch.cuda.memory.empty_cache()
            torch.cuda.memory._record_memory_history(context=None)
            for _ in range(10):
                optimized(*example_inputs)
        finally:
            torch.cuda.memory._record_memory_history(False)
        segments = torch.cuda.memory._snapshot()["segments"]
        self.assertEqual(segments[0]["requested_size"], 400)

    def test_view_outputs(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                y = torch.sin(x)
                y_same_size = y.view(*y.shape)
                y_diff_size = y.view(1, *y.shape)
                return y, y_same_size, y_diff_size

        example_inputs = (torch.randn(3, 10, device=self.device),)
        self.check_model(Model(), example_inputs)

    @skip_if_no_torchvision
    def test_missing_cubin(self):
        from torchvision.models.resnet import Bottleneck, ResNet

        class Model(ResNet):
            def __init__(self) -> None:
                super().__init__(
                    block=Bottleneck,
                    layers=[3, 4, 6, 3],
                    replace_stride_with_dilation=[False, False, True],
                    norm_layer=None,
                )

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                f1 = x
                x = self.maxpool(x)
                x = self.layer1(x)
                f2 = x
                x = self.layer2(x)
                f3 = x
                x = self.layer3(x)
                x = self.layer4(x)
                f4 = x
                return [f1, f2, f3, f4]

        # Call eval() here so that batch_norm won't update the running stats
        # Use float64 to avoid numeric difference failure
        dtype = torch.float32 if self.device == "mps" else torch.float64
        model = Model().to(device=self.device, dtype=dtype).eval()
        example_inputs = (torch.randn(4, 3, 64, 64, device=self.device, dtype=dtype),)
        self.check_model(model, example_inputs)

    def test_triton_next_power_of_2(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            def forward(self, a, b, lengths):
                n_elements = a.numel()
                out = torch.empty_like(a)
                max_len = int(lengths.max())
                scaling_factor = triton.next_power_of_2(max_len)
                add_kernel_with_scaling[(n_elements,)](
                    a,
                    b,
                    out,
                    n_elements,
                    scaling_factor,
                    BLOCK_SIZE=16,
                )
                return out

        example_inputs = (
            torch.randn(2, device=self.device),
            torch.randn(2, device=self.device),
            torch.arange(end=4, device=self.device),
        )
        self.check_model(Model(), example_inputs)

    @common_utils.parametrize("minmax", [min, max])
    @skipIfWindowsXPU(msg="crash on Windows XPU.")
    def test_sympy_cpp_printer_min_max(self, minmax):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            def forward(self, a, b, ranks):
                n_elements = a.numel()
                out = torch.empty_like(a)
                backed = a.size(0)
                unbacked = int(ranks.max())
                scaling_factor = minmax(backed, unbacked, 100)
                add_kernel_with_scaling[(n_elements,)](
                    a,
                    b,
                    out,
                    n_elements,
                    scaling_factor,
                    BLOCK_SIZE=16,
                )
                return out

        example_inputs = (
            torch.randn(16, device=self.device),
            torch.randn(16, device=self.device),
            torch.arange(end=4, device=self.device, dtype=torch.int16),
        )
        torch._dynamo.mark_dynamic(example_inputs[0], 0)
        torch._dynamo.mark_dynamic(example_inputs[1], 0)
        self.check_model(Model(), example_inputs)

    @skipIfMPS
    @common_utils.parametrize("grid_type", [1, 2, 3])
    @common_utils.parametrize("num_dims", [1, 2])
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("autotune", [False, True])
    def test_triton_kernel(self, grid_type, num_dims, dynamic, autotune):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                output = torch.zeros_like(x)
                if autotune and num_dims == 2:
                    x_elements = output.size()[0]
                    y_elements = output.size()[1]
                else:
                    n_elements = output.numel()

                # Select grid
                if autotune and num_dims == 2:
                    if grid_type == 1:
                        grid = (x_elements, y_elements)
                    elif grid_type == 2:
                        grid = lambda meta: (  # noqa: E731
                            triton.cdiv(x_elements, meta["BLOCK_SIZE_X"]),
                            triton.cdiv(y_elements, meta["BLOCK_SIZE_Y"]),
                        )
                    else:

                        def grid_fn(meta):
                            return (
                                triton.cdiv(x_elements, meta["BLOCK_SIZE_X"]),
                                triton.cdiv(y_elements, meta["BLOCK_SIZE_Y"]),
                            )

                        grid = grid_fn
                else:
                    if grid_type == 1:
                        grid = (n_elements,)
                    elif grid_type == 2:
                        grid = lambda meta: (  # noqa: E731
                            triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
                        )
                    else:

                        def grid_fn(meta):
                            return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

                        grid = grid_fn

                # Select kernel
                if autotune:
                    if num_dims == 1:
                        add_kernel_autotuned[grid](x, y, output, n_elements)
                    else:
                        add_kernel_2d_autotuned[grid](
                            x, y, output, x_elements, y_elements
                        )
                else:
                    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
                return output

        dims = [10] * num_dims
        x = torch.randn(*dims, device=self.device)
        y = torch.randn(*dims, device=self.device)
        dynamic_shapes = []
        if dynamic:
            dim0_x = Dim("dim0_x", min=1, max=10)
            dim0_y = Dim("dim0_y", min=1, max=10)
            dynamic_shapes = {"x": {0: dim0_x}, "y": {0: dim0_y}}
        self.check_model(Model(), (x, y), dynamic_shapes=dynamic_shapes)

    def test_triton_kernel_dynamic_shape_with_div(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        @triton.jit
        def pass_kernel(x, num):
            pass

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                num = x.numel() // 4

                grid = lambda meta: (triton.cdiv(num, 16),)  # noqa: E731
                pass_kernel[grid](x, num)
                return x

        x = torch.randn(10, device=self.device)
        dim0_x = Dim("dim0_x", min=1, max=10)
        dynamic_shapes = {"x": {0: dim0_x}}
        self.check_model(Model(), (x,), dynamic_shapes=dynamic_shapes)

    def test_triton_kernel_reinterpret_view(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        @triton.jit
        def pass_kernel(x, y):
            pass

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                out = torch.zeros_like(x[:, 4:])
                # the slicing below creates two ReinterpretView
                # instances: with offset=3 and offset=4
                add_kernel[(10,)](
                    in_ptr0=x[:, 3:-1],
                    in_ptr1=x[:, 4:],
                    out_ptr=out,
                    n_elements=160,
                    BLOCK_SIZE=16,
                )
                return out

        example_inputs = (torch.randn(10, 20, device=self.device),)
        self.check_model(Model(), example_inputs)

    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("tma_version", ["new", "old"])
    def test_triton_kernel_tma_descriptor_1d(self, dynamic, tma_version):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")
        if tma_version == "new" and not has_triton_tensor_descriptor_host_tma():
            self.skipTest("requires triton.tools.tensor_descriptor TMA support")
        if tma_version == "old" and not has_triton_experimental_host_tma():
            self.skipTest("requires triton.tools.experimental_descriptor TMA support")

        kernel = (
            add_kernel_with_tma_1d_new_api
            if tma_version == "new"
            else add_kernel_with_tma_1d_old_api
        )

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, a, b):
                BLOCK_SIZE = 256
                out = torch.zeros_like(a)
                n_elements = out.numel()

                desc_a, desc_b, desc_out = (
                    create_tensor_descriptor_shim(
                        t, [BLOCK_SIZE], new_api=(tma_version == "new")
                    )
                    for t in (a, b, out)
                )

                grid = lambda meta: (  # noqa: E731
                    triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
                )
                kernel[grid](
                    desc_a,
                    desc_b,
                    desc_out,
                    BLOCK_SIZE=BLOCK_SIZE,
                )

                return out

        a = torch.randn(301, device=self.device)
        b = torch.randn(301, device=self.device)
        example_inputs = (a, b)

        dynamic_shapes = None
        if dynamic:
            dim0_ab = Dim("s0", min=2, max=1024)
            dynamic_shapes = {
                "a": {0: dim0_ab, 1: None},
                "b": {0: dim0_ab, 1: None},
            }

        self.check_model(
            Model(),
            example_inputs=example_inputs,
            dynamic_shapes=dynamic_shapes,
        )

    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("tma_version", ["new", "old"])
    def test_triton_kernel_tma_descriptor_2d(self, dynamic, tma_version):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")
        if tma_version == "new" and not has_triton_tensor_descriptor_host_tma():
            self.skipTest("requires triton.tools.tensor_descriptor TMA support")
        if tma_version == "old" and not has_triton_experimental_host_tma():
            self.skipTest("requires triton.tools.experimental_descriptor TMA support")

        kernel = (
            add_kernel_with_tma_2d_new_api
            if tma_version == "new"
            else add_kernel_with_tma_2d_old_api
        )

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, a, b):
                BLOCK_SIZE_X = 16
                BLOCK_SIZE_Y = 32
                out = torch.zeros_like(a)
                x_size, y_size = out.size()

                desc_a, desc_b, desc_out = (
                    create_tensor_descriptor_shim(
                        t,
                        [BLOCK_SIZE_X, BLOCK_SIZE_Y],
                        new_api=(tma_version == "new"),
                    )
                    for t in (a, b, out)
                )

                grid = lambda meta: (  # noqa: E731
                    triton.cdiv(x_size, meta["BLOCK_SIZE_X"]),
                    triton.cdiv(y_size, meta["BLOCK_SIZE_Y"]),
                )
                kernel[grid](
                    desc_a,
                    desc_b,
                    desc_out,
                    BLOCK_SIZE_X=BLOCK_SIZE_X,
                    BLOCK_SIZE_Y=BLOCK_SIZE_Y,
                )

                return out

        a = torch.randn((25, 16), device=self.device)
        b = torch.randn((25, 16), device=self.device)
        example_inputs = (a, b)

        dynamic_shapes = None
        if dynamic:
            dim0_ab = Dim("s0", min=2, max=1024)
            dynamic_shapes = {
                "a": {0: dim0_ab, 1: None},
                "b": {0: dim0_ab, 1: None},
            }

        self.check_model(
            Model(),
            example_inputs=example_inputs,
            dynamic_shapes=dynamic_shapes,
        )

    def test_triton_kernel_sympy_expr_arg(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            def forward(self, x, e):
                sympy_expr = max(1, e.item())
                out = torch.zeros_like(x)
                add_kernel[(1,)](
                    in_ptr0=x,
                    in_ptr1=x,
                    out_ptr=out,
                    n_elements=sympy_expr,
                    BLOCK_SIZE=1,
                )
                return out

        NUMEL = 64
        inputs = (
            torch.randn(NUMEL, device=self.device),
            torch.tensor(NUMEL, device=self.device),
        )
        self.check_model(Model(), inputs)

    def test_triton_kernel_sympy_fn_like_arg(self):
        # This test should hit sympy.expand("sqrt") which crashes with
        # AttributeError: 'function' object has no attribute 'expand'.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            def forward(self, x):
                out = torch.zeros_like(x)
                add_kernel_with_optional_param[1,](
                    in_ptr0=x,
                    in_ptr1=x,
                    out_ptr=out,
                    n_elements=x.numel(),
                    BLOCK_SIZE=1,
                    ARGS_PASSED="sqrt",  # sqrt is a valid sympy fn
                )
                return out

        inputs = (torch.randn(4, device=self.device),)
        self.check_model(Model(), inputs)

    def test_triton_kernel_with_none_input(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                n_elements = x.size()[0]
                BLOCK_SIZE = 1024

                output_wo_y = torch.empty_like(x)
                output_with_y = torch.empty_like(x)

                add_kernel_with_optional_param[(1,)](
                    x,
                    None,
                    output_wo_y,
                    n_elements,
                    ARGS_PASSED="one",
                    BLOCK_SIZE=BLOCK_SIZE,
                )
                add_kernel_with_optional_param[(1,)](
                    x,
                    y,
                    output_with_y,
                    n_elements,
                    ARGS_PASSED="two",
                    BLOCK_SIZE=BLOCK_SIZE,
                )

                return 2.71 * output_wo_y + 3.14 * output_with_y

        example_inputs = (
            torch.randn(1023, device=self.device),
            torch.randn(1023, device=self.device),
        )

        self.check_model(Model(), example_inputs)

    def test_triton_kernel_equal_to_1_arg(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            def forward(self, x, y):
                out = torch.empty_like(x)
                n_elements = x.numel()
                add_kernel[(n_elements,)](x, y, out, n_elements, BLOCK_SIZE=16)
                return out

        example_inputs = (
            torch.randn(1, device=self.device),
            torch.randn(1, device=self.device),
        )

        self.check_model(Model(), example_inputs)

    def test_triton_kernel_with_none_inputs_and_equal_to_1_arg(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                n_elements = x.size()[0]
                BLOCK_SIZE = 1024
                out1 = torch.empty_like(x)
                out2 = torch.empty_like(x)
                # Run the same kernel multiple times to test the optimization
                # of removing None arguments and then update the indices of
                # equal_to_1 arguments. The None arguments need to be before
                # the equal_to_1 arguments
                add_kernel_with_none_param_and_equal_to_1_arg[(1,)](
                    x,
                    None,
                    out1,
                    n_elements,
                    x.stride(0),  # equal to 1
                    ARGS_PASSED="one",
                    BLOCK_SIZE=BLOCK_SIZE,
                )
                add_kernel_with_none_param_and_equal_to_1_arg[(1,)](
                    2.71 * out1,
                    None,
                    out2,
                    n_elements,
                    x.stride(0),  # equal to 1
                    ARGS_PASSED="one",
                    BLOCK_SIZE=BLOCK_SIZE,
                )
                return out2

        example_inputs = (torch.randn(1023, device=self.device),)
        self.check_model(Model(), example_inputs)

    @common_utils.parametrize("dynamic", [False, True])
    def test_triton_kernel_equal_to_1_float_arg(self, dynamic):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            def forward(self, x, y):
                out = torch.empty_like(x)
                n_elements = x.numel()
                scaling_factor = (n_elements**0) / 1.0
                add_kernel_with_scaling[(n_elements,)](
                    x,
                    y,
                    out,
                    n_elements,
                    scaling_factor,
                    BLOCK_SIZE=16,
                )
                return out

        dynamic_shapes = None
        if dynamic:
            dim0_xy = Dim("s0", min=2, max=1024)
            dynamic_shapes = {
                "x": {0: dim0_xy},
                "y": {0: dim0_xy},
            }
        example_inputs = (
            torch.randn(2, device=self.device),
            torch.randn(2, device=self.device),
        )
        self.check_model(
            Model(),
            example_inputs,
            dynamic_shapes=dynamic_shapes,
        )

    def test_triton_kernel_weird_param_order(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                out = torch.empty_like(x)
                add_kernel_autotuned_weird_param_order[16,](
                    in_ptr0=x,
                    in_ptr1=x,
                    n_elements=x.numel(),
                    out_ptr=out,
                )
                return out

        x = torch.randn(16, 16, device=self.device)
        self.check_model(Model(), (x,))

    @skipIfWindowsXPU(msg="crash on Windows XPU.")
    def test_triton_kernel_dynamic_grid(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        import math

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y, n_elements_tensor):
                output = torch.zeros_like(x)
                n_elements_symint = n_elements_tensor.item()
                n_elements = x.numel()

                def grid(meta):
                    n_elements_complicated = n_elements_symint // 1.0
                    return (math.trunc(n_elements_complicated / meta["BLOCK_SIZE"]),)

                add_kernel_autotuned[grid](
                    x,
                    y,
                    output,
                    n_elements,
                )

                return output

        x = torch.randn(128, device=self.device)
        y = torch.randn(128, device=self.device)
        n_elem = torch.tensor(128)
        dim0_x = Dim("dim0_x", min=8, max=256)
        dim0_y = Dim("dim0_y", min=8, max=256)
        dynamic_shapes = {"x": {0: dim0_x}, "y": {0: dim0_y}, "n_elements_tensor": {}}
        self.check_model(Model(), (x, y, n_elem), dynamic_shapes=dynamic_shapes)

    def test_shifted_constraint_ranges(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(
                self,
                x: torch.Tensor,
                y: torch.Tensor,
            ):
                torch._check(y.size(0) == x.size(0) + 1)
                return x.sum(0) + y.sum(0)

        a = torch.randn((4, 5), device=self.device)
        b = torch.randn((5, 5), device=self.device)
        dim0_x = Dim("dim0_x", min=2, max=1024)
        dim0_y = dim0_x + 1
        dynamic_shapes = {"x": {0: dim0_x}, "y": {0: dim0_y}}
        self.check_model(
            Model(),
            (a, b),
            dynamic_shapes=dynamic_shapes,
        )

    def test_scatter_fallback(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(
                self,
                inp: torch.Tensor,
                index: torch.Tensor,
                src: torch.Tensor,
            ):
                return torch.scatter(inp, 1, index, src)

        inputs = (
            torch.ones((3, 5), device=self.device, dtype=torch.int64),
            torch.tensor([[0, 1, 2, 0]], device=self.device, dtype=torch.int64),
            torch.zeros((2, 5), device=self.device, dtype=torch.int64),
        )

        self.check_model(Model(), inputs)

    def test_scatter_reduce_fallback(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(
                self,
                inp: torch.Tensor,
                index: torch.Tensor,
                src: torch.Tensor,
            ):
                return torch.scatter_reduce(inp, 0, index, src, reduce="sum")

        inputs = (
            torch.tensor([1, 10, 100, 1000], device=self.device, dtype=torch.int64),
            torch.tensor([0, 1, 0, 1, 2, 1], device=self.device, dtype=torch.int64),
            torch.tensor([1, 2, 3, 4, 5, 6], device=self.device, dtype=torch.int64),
        )

        self.check_model(Model(), inputs)

    def test_index_put_fallback(self):
        # index_put falls back in the deterministic mode
        with DeterministicGuard(True):

            class Model(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()

                def forward(
                    self,
                    self_tensor: torch.Tensor,
                    indices: tuple[torch.Tensor],
                    values: torch.Tensor,
                ):
                    return torch.index_put(
                        self_tensor, indices, values, accumulate=True
                    )

            inputs = (
                torch.ones(4, device=self.device, dtype=torch.int64),
                (torch.tensor([1, 1, 2, 2], device=self.device, dtype=torch.bool),),
                torch.ones(4, device=self.device, dtype=torch.int64),
            )

            self.check_model(Model(), inputs)

    def test_narrow_fallback(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, inp: torch.Tensor, dim: int, start: int, length: int):
                return torch.ops.aten.narrow(inp, dim, start, length)

        inputs = (torch.rand((3, 4), device=self.device), 0, 0, 2)

        self.check_model(Model(), inputs)

    def test_pad_fallback(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(
                self,
                inp: torch.Tensor,
                pad: tuple[int, ...],
            ):
                return torch.ops.aten.pad(inp, pad)

        inputs = (torch.rand((3, 3, 4, 2), device=self.device), (0, 1, 2, 1, 3, 3))

        self.check_model(Model(), inputs)

    def test_fill__fallback(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, inp: torch.Tensor, scalar: float):
                torch.ops.aten.fill_(inp, scalar)
                return inp

        inputs = (torch.rand((3, 3, 4, 2), device=self.device), 0.5)
        self.check_model(Model(), inputs)

    @common_utils.parametrize("embed_kernel_binary", [False, True])
    def test_repeated_user_defined_triton_kernel(self, embed_kernel_binary):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                for _ in range(3):
                    mul2_inplace_kernel[4,](x, n_elements=4, BLOCK_SIZE=16)
                return x

        inputs = (torch.randn(4, 4, device=self.device),)
        with config.patch({"aot_inductor.embed_kernel_binary": embed_kernel_binary}):
            model = Model()
            self.check_model(model, inputs)
            _, code = run_and_get_cpp_code(AOTIRunnerUtil.compile, model, inputs)
            FileCheck().check("launchKernel(").run(code)
            if config.aot_inductor.embed_kernel_binary:
                # Not expect to see launchKernel("CUBIN_FILE_NAME"
                FileCheck().check_not('launchKernel("').run(code)

    @unittest.skipIf(
        not IS_BIG_GPU, "Skipping triton backend only since not big GPU (not enough SM)"
    )
    def test_convolution(self):
        if self.device == "cpu":
            raise unittest.SkipTest("using triton backend only is not supported on CPU")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, w, b):
                return torch.ops.aten.convolution(x, w, b, [4], [0], [1], True, [0], 1)

        example_inputs = (
            torch.randn([2, 32, 90], device=self.device),
            torch.randn([32, 16, 8], device=self.device),
            torch.randn([16], device=self.device),
        )
        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "Triton",
            }
        ):
            self.check_model(Model(), example_inputs)

    def test_zero_size_weight(self):
        class Model(torch.nn.Module):
            def __init__(self, channel, r=8):
                super().__init__()
                self.pool = torch.nn.AdaptiveAvgPool2d(1)
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(channel, channel // r, bias=False),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(channel // r, channel, bias=False),
                    torch.nn.Sigmoid(),
                )

            def forward(self, inp):
                b, c, _, _ = inp.shape
                x = self.pool(inp).view(b, c)
                x = self.net(x).view(b, c, 1, 1)
                x = inp * x
                return x

        inputs = (torch.rand(4, 4, 4, 4, device=self.device),)
        self.check_model(Model(4), inputs)

    def test_zero_size_buffer(self):
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.foo = torch.nn.Buffer(torch.zeros((0, 0), device=device))

            def forward(self, x):
                return x + 1, self.foo

        example_inputs = (torch.rand(4, 4, device=self.device),)
        self.check_model(Model(self.device), example_inputs)

    def test_no_args(self):
        class Model(torch.nn.Module):
            def __init__(self, m, n):
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.randn(m, n),
                )
                self.alpha = torch.nn.Parameter(torch.randn(m, n))

            def forward(self):
                return self.weight * self.alpha

        self.check_model(Model(6, 4), ())

    def test_dynamic_scalar(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.criterion_ce = torch.nn.CrossEntropyLoss(reduction="none")

            def forward(self, inputs, targets, split_index=None):
                statistics = {}
                total_loss = self.criterion_ce(inputs, targets).sum()
                statistics["dl"] = total_loss.item()
                return total_loss, statistics

        inputs = (
            torch.rand(4, 4, 4, 4, device=self.device),
            torch.rand(4, 4, 4, 4, device=self.device),
        )
        self.check_model(Model(), inputs)

    def test_symint_item(self):
        class Model(torch.nn.Module):
            def forward(self, tensor):
                return tensor.item()

        inputs = (torch.tensor([1], dtype=torch.int, device=self.device),)
        self.check_model(Model(), inputs)

    def test_symbool_item(self):
        class Model(torch.nn.Module):
            def forward(self, tensor):
                return tensor.item()

        inputs = (torch.tensor([0], dtype=torch.bool, device=self.device),)
        self.check_model(Model(), inputs)

    def test_symfloat_item(self):
        class Model(torch.nn.Module):
            def forward(self, tensor):
                return tensor.item()

        inputs = (torch.tensor([3.14], dtype=torch.float, device=self.device),)
        self.check_model(Model(), inputs)

    def test_constant_original_fqn_and_dtype(self):
        class FooBarModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_parameter("0", torch.nn.Parameter(torch.randn(3, 4)))
                self.test_buf = torch.nn.Buffer(torch.randn(3, 4))
                self.register_parameter(
                    "test_param", torch.nn.Parameter(torch.randn(3, 4))
                )

            def forward(self, x):
                return ((x + self.test_buf) * getattr(self, "0")) / self.test_param

        class TestModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo_bar = FooBarModule()
                self.register_parameter(
                    "test_param", torch.nn.Parameter(torch.randn(3, 4))
                )
                self.test_buf = torch.nn.Buffer(torch.randn(3, 4))

            def forward(self, x):
                return (self.foo_bar(x) + self.test_param) * self.test_buf

        with torch.no_grad():
            so_path = AOTIRunnerUtil.legacy_compile(
                model=TestModule().to(device=self.device),
                example_inputs=(torch.rand(3, 4, device=self.device),),
            )
        runner = AOTIRunnerUtil.legacy_load_runner(self.device, so_path)

        expected_original_fqns = {
            "L__self___test_param": "test_param",
            "L__self___test_buf": "test_buf",
            "L__self___foo_bar_0": "foo_bar.0",
            "L__self___foo_bar_test_param": "foo_bar.test_param",
            "L__self___foo_bar_test_buf": "foo_bar.test_buf",
        }
        self.assertEqual(
            expected_original_fqns, runner.get_constant_names_to_original_fqns()
        )

        expected_dtypes = {
            "L__self___test_param": 6,
            "L__self___test_buf": 6,
            "L__self___foo_bar_0": 6,
            "L__self___foo_bar_test_param": 6,
            "L__self___foo_bar_test_buf": 6,
        }
        self.assertEqual(expected_dtypes, runner.get_constant_names_to_dtypes())

    def test_masked_select_dynamic(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                mask = x.ge(0.5)
                return torch.masked_select(x, mask)

        example_args = (torch.randn(3, 4, 5, device=self.device),)
        dim0_x_max, dim1_x_max = 100, 7
        dynamic_shapes = {
            "x": {
                0: Dim("dim0_x", max=dim0_x_max),
                1: Dim("dim1_x_max", max=dim1_x_max),
            }
        }
        m = M()
        self.check_model(m, example_args, dynamic_shapes=dynamic_shapes)

    def test_proxy_executor_permute(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return torch.ops.aten.permute.default(x, [0, 2, 1])

        example_args = (torch.randn((1, 3001, 201), dtype=torch.complex64),)
        m = M()
        self.check_model(m, example_args)

    def test_proxy_executor_abs(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return torch.ops.aten.abs.default(x)

        example_args = (torch.randn((1, 3001, 201), dtype=torch.complex64),)
        m = M()
        self.check_model(m, example_args)

    def test_proxy_executor_squeeze(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return torch.ops.aten.squeeze.dim(x, 0)

        example_args = (torch.randn((1, 300, 201), dtype=torch.complex64),)
        m = M()
        self.check_model(m, example_args)

    def test_proxy_executor_hann(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self):
                return torch.ops.aten.hann_window.default(400)

        example_args = ()
        m = M()
        self.check_model(m, example_args)

    def test_fqn(self):
        class NestedChild(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.nestedchild3buffer = torch.nn.Buffer(torch.ones(2, 3) * 3)

            def forward(self, x):
                return x / self.nestedchild3buffer

        class Child1(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.nested = NestedChild()
                self.register_parameter(
                    "child1param", torch.nn.Parameter(torch.ones(2, 3))
                )

            def forward(self, x):
                x = self.nested(x)
                return x + self.child1param

        class Child2(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.child2buffer = torch.nn.Buffer(torch.ones(2, 3) * 2)

            def forward(self, x):
                return x - self.child2buffer

        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = Child1()
                self.bar = Child2()
                self.register_parameter(
                    "rootparam", torch.nn.Parameter(torch.ones(2, 3) * 4)
                )

            def forward(self, x):
                x = x * self.rootparam
                x = self.foo(x)
                x = self.bar(x)
                return x

        self.check_model(MyModule(), (torch.randn(2, 3, device=self.device),))

    def test_model_modified_weights(self):
        class Model(torch.nn.Module):
            def __init__(self, n, k, device):
                super().__init__()
                self.weight = torch.randn(n, k, device=device)
                self.bias = torch.randn(n, device=device)

            def forward(self, a):
                return torch.nn.functional.linear(a, self.weight, self.bias)

        M = 16
        N = 10
        K = 128
        example_inputs = (torch.randn(2, M, K, device=self.device),)
        model = Model(N, K, self.device)
        self.check_model(model, example_inputs)

        # Update model weights, after this AOTInductor should re-generate model.so
        # if weights are stored in the model.so
        model.weight += 1
        self.check_model(model, example_inputs)

    @skipIfWindowsXPU(msg="crash on Windows XPU.")
    def test_triton_kernel_extern_kernel_arg(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            def forward(self, x, y):
                out = torch.zeros_like(x)
                # torch.mm is ExternKernelOut
                add_kernel[(4,)](x, torch.mm(x, y), out, 4, 16)
                return out

        example_inputs = (
            torch.randn(4, 4, device=GPU_TYPE),
            torch.randn(4, 4, device=GPU_TYPE),
        )

        self.check_model(Model(), example_inputs)

    def test_triton_kernel_multi_output_arg(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            def forward(self, x, y):
                out = torch.zeros_like(x)
                # torch.sort creates fallback kernel and hence MultiOutput
                add_kernel[(4,)](x, torch.sort(y).values, out, 4, 16)
                return out

        example_inputs = (
            torch.randn(4, 4, device=GPU_TYPE),
            torch.randn(4, 4, device=GPU_TYPE),
        )

        self.check_model(Model(), example_inputs)

    def test_triton_kernel_reinterpret_view_mem_leak(self):
        # Check for memory leak when using user-defined Triton Kernel + AOTI.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                out = torch.zeros_like(x)
                yy = y * y
                # reshape creates a ReinterpretView
                add_kernel[(4,)](x, yy.reshape_as(x), out, 4, 16)
                return out

        example_inputs = (
            torch.randn(4, 4, device=GPU_TYPE),
            torch.randn(1, 16, device=GPU_TYPE),
        )

        package_path: str = AOTIRunnerUtil.compile(
            Model(),
            example_inputs,
        )
        aot_inductor_module = torch._inductor.aoti_load_package(package_path)
        # Don't assign outputs to a variable b/c it will allocate GPU memory.
        device_interface = get_interface_for_device(GPU_TYPE)
        device: int = device_interface.current_device()
        mem_before = device_interface.memory_allocated(device)
        aot_inductor_module(*example_inputs)
        aot_inductor_module(*example_inputs)
        mem_after = device_interface.memory_allocated(device)
        self.assertEqual(mem_before, mem_after)

        actual = aot_inductor_module(*example_inputs)
        expected = Model()(*example_inputs)
        torch.testing.assert_close(actual, expected)

    @skipIfMPS
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("autotuning", [False, True])
    def test_triton_kernel_unbacked_symint_in_grid(self, dynamic, autotuning):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            def forward(self, x, y, n_elements_tensor):
                output = torch.zeros_like(x)
                n_elements_symint = n_elements_tensor.item()
                n_elements = x.numel()

                def grid(meta):
                    return (triton.cdiv(n_elements_symint, meta["BLOCK_SIZE"]),)

                if autotuning:
                    add_kernel_autotuned[grid](
                        x,
                        y,
                        output,
                        n_elements,
                    )
                else:
                    add_kernel[grid](
                        x,
                        y,
                        output,
                        n_elements,
                        BLOCK_SIZE=16,
                    )

                return output

        example_inputs = (
            torch.randn(123, device=GPU_TYPE),
            torch.randn(123, device=GPU_TYPE),
            torch.tensor(123),
        )

        dynamic_shapes = None
        if dynamic:
            dim0 = Dim("s0", min=2, max=1024)
            dynamic_shapes = {
                "x": {0: dim0},
                "y": {0: dim0},
                "n_elements_tensor": {},
            }

        self.check_model(
            Model(),
            example_inputs,
            dynamic_shapes=dynamic_shapes,
        )

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Some archs don't support mem eff SDPA"
    )
    def test_scaled_dot_product_efficient_attention(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            def forward(self, q, k, v, attn_bias):
                return torch.ops.aten._scaled_dot_product_efficient_attention(
                    q, k, v, attn_bias, False
                )[0]

        example_inputs = (
            torch.randn(4, 4, 36, 36, device=GPU_TYPE),
            torch.randn(4, 4, 36, 36, device=GPU_TYPE),
            torch.randn(4, 4, 36, 36, device=GPU_TYPE),
            torch.randn(4, 4, 36, 36, device=GPU_TYPE),
        )
        self.check_model(Model(), example_inputs)

    def test_aoti_runtime_asserts(self):
        from torch.export._draft_export import draft_export, FailureType

        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor a, Tensor b) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            def foo(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                return a[: b.item()]

            @torch.library.register_fake("mylib::foo", lib=lib)
            def foo_fake_impl(a, b):
                ctx = torch.library.get_ctx()
                u = ctx.new_dynamic_size()
                return torch.empty(u)

            class M(torch.nn.Module):
                def forward(self, a, b):
                    res = torch.ops.mylib.foo(a, b)
                    s = res.shape[0]
                    torch._check(s > 3)
                    torch._check(s < a.shape[0])
                    return a[s - 3]

            example_inputs = (torch.randn(100), torch.tensor(10))
            ep = draft_export(M(), example_inputs)
            report = ep._report
            need_config_patch = any(
                not f.xfail and f.failure_type == FailureType.MISMATCHED_FAKE_KERNEL
                for f in report.failures
            )
            m = ep.module()

            # This should no longer be needed after #150093
            from torch._functorch import config as functorch_config

            with functorch_config.patch(
                {"generate_fake_kernels_from_real_mismatches": need_config_patch}
            ):
                pt2_file = torch._inductor.aoti_compile_and_package(ep)
            optimized = torch._inductor.aoti_load_package(pt2_file)

            self.assertTrue(same(optimized(*example_inputs), m(*example_inputs)))

            with self.assertRaisesRegex(Exception, "run_func_(.*) API call failed "):
                optimized(torch.randn(100), torch.tensor(2))

    @patch.dict(os.environ, {"TORCHINDUCTOR_SCALAR_ASSERTS_FULL": "1"})
    def test_aoti_runtime_asserts_backed_symint(self):
        if not full_aoti_runtime_assert():
            raise unittest.SkipTest("full runtime assert not turned on")

        class Model(torch.nn.Module):
            def forward(self, x):
                y = x.reshape(100, -1).clone()
                y = y + 1
                return y

        model = Model().to(self.device)
        input1 = (torch.rand(100, device=self.device),)
        input2 = (torch.rand(2099, device=self.device),)
        dynamic_shapes = {
            "x": {0: torch.export.Dim.DYNAMIC},
        }
        package_path = AOTIRunnerUtil.compile(
            model,
            input1,
            dynamic_shapes=dynamic_shapes,
        )
        optimized = torch._inductor.aoti_load_package(package_path)
        self.assertEqual(model(*input1), optimized(*input1))
        with self.assertRaisesRegex(Exception, "run_func_(.*) API call failed "):
            optimized(*input2)

    @skipIfWindows(msg="TODO: (xuhancn) confirm, Crash: access violation")
    def test_index_put_with_none_index(self):
        # index_put falls back in the deterministic mode
        with DeterministicGuard(True):

            class Model(torch.nn.Module):
                def forward(self, x, i1, i2, y):
                    return torch.ops.aten.index_put(
                        x,
                        (None, None, i1, i2.transpose(0, 1)),
                        y,
                        accumulate=True,
                    )

            example_inputs = (
                torch.rand(8, 192, 30, 30, device=self.device),
                torch.zeros(3, 14, 1, 1, dtype=torch.int64, device=self.device),
                torch.ones(14, 3, dtype=torch.int64, device=self.device),
                torch.randn(8, 192, 3, 14, 3, 14, device=self.device),
            )
            self.check_model(Model(), example_inputs)

    @patch.dict(os.environ, {"AOTI_RUNTIME_CHECK_INPUTS": "1"})
    def test_runtime_checks(self):
        class Model(torch.nn.Module):
            def forward(self, inputs):
                return list(inputs.values())

        inputs = {}
        dtypes = [
            torch.float16,
            torch.float32,
            torch.bool,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ]

        if not TEST_MPS:
            dtypes.append(torch.float64)
        if SM80OrLater:
            dtypes.append(torch.bfloat16)

        for dtype in dtypes:
            inputs[f"x_{str(dtype)}"] = torch.ones(
                4, 8, 10, dtype=dtype, device=self.device
            )

        dim0 = Dim("s0", min=2, max=1024)
        dim1 = Dim("s1", min=2, max=512)
        dim2 = Dim("s2", min=2, max=128)
        dynamic_shapes = {
            "x_torch.float16": {0: dim0},
            "x_torch.float32": {0: dim0},
            "x_torch.bool": {1: dim1},
            "x_torch.int8": {1: dim1},
            "x_torch.int16": {},
            "x_torch.int32": {2: dim2},
            "x_torch.int64": {2: dim2},
            "x_torch.uint8": {2: dim2},
        }
        if not TEST_MPS:
            dynamic_shapes["x_torch.float64"] = {0: dim0}
        if SM80OrLater:
            dynamic_shapes["x_torch.bfloat16"] = {1: dim1}

        m = Model()
        inputs = (inputs,)
        dynamic_shapes = (dynamic_shapes,)
        with torch.no_grad():
            so_path = AOTIRunnerUtil.legacy_compile(
                m, inputs, dynamic_shapes=dynamic_shapes
            )

        # Expected results for the following checks:
        # ("unmatched dtype", "unmatched dim value at", "dim value is too", "unmatched stride value at")
        if SM80OrLater:
            # 10 dynamic dims
            expected_results = (10, 21, 18, 21)
        elif TEST_MPS:
            # 8 dynamic dims
            expected_results = (8, 17, 14, 16)
        else:
            # 9 dynamic dims
            expected_results = (9, 19, 16, 19)

        with open(os.path.splitext(so_path)[0] + ".cpp") as cpp:
            src_code = cpp.read()
            FileCheck().check_count(
                "unmatched dtype",
                expected_results[0],
                exactly=True,
            ).run(src_code)
            FileCheck().check_count(
                "unmatched dim value at",
                expected_results[1],
                exactly=True,
            ).run(src_code)
            FileCheck().check_count(
                "dim value is too",
                expected_results[2],
                exactly=True,
            ).run(src_code)
            FileCheck().check_count(
                "unmatched stride value at",
                expected_results[3],
                exactly=True,
            ).run(src_code)

        self.check_model(m, inputs)

    @unittest.skipIf(TEST_WITH_ROCM, "FP8 is not supported on ROCM")
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FP8,
        "FP8 is only supported on H100+, SM 8.9 and MI300+ devices",
    )
    @patch.dict(os.environ, {"AOTI_RUNTIME_CHECK_INPUTS": "1"})
    def test_runtime_checks_fp8(self):
        # cuda only
        if self.device != "cuda":
            return

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x0, x1):
                t = x0.to(torch.float) + x1.to(torch.float)
                return t

        inputs = []
        for dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            # FP8 funz are for AMD
            # see https://github.com/pytorch/pytorch/issues/126734
            # torch.float8_e4m3fnuz,
            # torch.float8_e5m2fnuz,
        ):
            inputs.append(torch.ones(8, 8, 8, dtype=dtype, device=self.device))
        dim0 = Dim("s0", min=2, max=1024)
        dynamic_shapes = {
            "x0": {0: dim0},
            "x1": {0: dim0},
        }
        with torch.no_grad():
            self.check_model(
                Model(),
                tuple(inputs),
                dynamic_shapes=dynamic_shapes,
            )

    @skipIfXpu(msg="Total size of kernel arguments exceeds driver limit on XPU")
    def test_runtime_checks_large(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, *inputs):
                result = inputs[0]
                for i in range(1, len(inputs)):
                    result = result + inputs[i]
                return result

        inputs = []
        for _ in range(1000):
            inputs.append(torch.ones(8, 8, 8, dtype=torch.float16, device=self.device))
        inputs = tuple(inputs)
        model = Model()
        with torch.no_grad():
            AOTIRunnerUtil.compile(
                model,
                inputs,
            )

    def test_runtime_checks_complex(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x0, x1, x2):
                return (x0, x1, x2)

        inputs = []
        x0 = torch.tensor([1, -1], dtype=torch.complex32, device=self.device)
        x1 = torch.tensor(
            [1 + 1j, -1 + 1j, -2 + 2j, 3 - 3j, 0, 1j, 1, -1],
            dtype=torch.complex64,
            device=self.device,
        )
        x2 = torch.tensor(128, dtype=torch.complex128, device=self.device)
        inputs.append(x0)
        inputs.append(x1)
        inputs.append(x2)
        dim0 = Dim("s0", min=2, max=1024)
        dynamic_shapes = {
            "x0": {0: dim0},
            "x1": {},
            "x2": {},
        }
        with torch.no_grad():
            self.check_model(
                Model(),
                tuple(inputs),
                dynamic_shapes=dynamic_shapes,
            )

    @unittest.skipIf(IS_FBCODE, "Not yet runnable in fbcode")
    @patch.dict(os.environ, {"AOTI_RUNTIME_CHECK_INPUTS": "1"})
    def test_runtime_checks_dtype_failed(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                y = x.type(torch.float)
                return y

        x = torch.randn(1, 4, dtype=torch.float16, device=self.device)
        model = Model()
        with torch.no_grad():
            package_path: str = AOTIRunnerUtil.compile(
                model,
                (x,),
            )
        aot_inductor_module = torch._inductor.aoti_load_package(package_path)
        x_casted = x.float()
        with self.assertRaisesRegex(Exception, ""):
            aot_inductor_module(x_casted)

    @patch.dict(os.environ, {"AOTI_RUNTIME_CHECK_INPUTS": "1"})
    def test_runtime_checks_device_type_failed(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return x + 1

        x = torch.randn(1, 4, dtype=torch.float16, device="cpu")
        model = Model()
        with torch.no_grad():
            package_path: str = AOTIRunnerUtil.compile(
                model,
                (x,),
            )

        aot_inductor_module = torch._inductor.aoti_load_package(package_path)
        aot_inductor_module(x)
        x_casted = x.to(GPU_TYPE)
        with self.assertRaisesRegex(Exception, ""):
            aot_inductor_module(x_casted)

    def test_non_contiguous_output_alias(self):
        # Test return x, x.contiguous() where x is non-contiguous.
        class Model(torch.nn.Module):
            def forward(self, x):
                squared = x * x
                transposed = squared.t()  # non-contiguous
                contig = transposed.contiguous()
                return transposed, contig

        x = torch.randn(3, 4, dtype=torch.float16, device=self.device)
        model = Model()
        with torch.no_grad():
            result = AOTIRunnerUtil.run(
                model,
                (x,),
            )
        actual = model(x)
        self.assertTrue(same(result, actual))

        # contiguous() should create a new tensor
        self.assertTrue(result[0].data_ptr() != result[1].data_ptr())

    def test_multiple_output_alias(self):
        # Test when multiple outputs alias the same tensor
        class Model(torch.nn.Module):
            def forward(self, x):
                squared = x * x
                contig = squared.contiguous()  # alias
                reshaped = squared.reshape(squared.shape)  # alias
                cubed = squared * x
                return squared, contig, reshaped, cubed

        x = torch.randn(3, 4, dtype=torch.float32, device=self.device)
        model = Model()

        with torch.no_grad():
            result = AOTIRunnerUtil.run(
                model,
                (x,),
            )
        actual = model(x)
        self.assertTrue(same(result, actual))

        # squared, contig and reshaped alias the same tensor.
        self.assertTrue(result[0].data_ptr() == result[1].data_ptr())
        self.assertTrue(result[0].data_ptr() == result[2].data_ptr())
        # cubed shouldn't be an alias.
        self.assertTrue(result[0].data_ptr() != result[3].data_ptr())

    @patch.dict(os.environ, {"AOTI_RUNTIME_CHECK_INPUTS": "1"})
    def test_runtime_checks_shape_failed(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return x

        x = torch.randn(4, 4, 4, dtype=torch.float16, device=self.device)
        y0 = torch.randn(8, 4, 4, dtype=torch.float16, device=self.device)
        y1 = torch.randn(4, 8, 4, dtype=torch.float16, device=self.device)
        y2 = rand_strided(
            (4, 4, 4), (16, 1, 4), dtype=torch.float16, device=self.device
        )
        # batch size is outside of the range
        y3 = torch.randn(2048, 3, 4, dtype=torch.float16, device=self.device)
        y4 = torch.randn(2048, 4, 4, dtype=torch.float16, device=self.device)
        dim0 = Dim("s0", min=4, max=1024)
        dynamic_shapes = {
            "x": {0: dim0},
        }
        model = Model()
        with torch.no_grad():
            package_path: str = AOTIRunnerUtil.compile(
                model, (x,), dynamic_shapes=dynamic_shapes
            )
        aot_inductor_module = torch._inductor.aoti_load_package(package_path)
        # dynamic dim works fine
        _ = aot_inductor_module(y0)
        with self.assertRaisesRegex(Exception, ""):
            aot_inductor_module(y1)
        with self.assertRaisesRegex(Exception, ""):
            aot_inductor_module(y2)
        with self.assertRaisesRegex(Exception, ""):
            aot_inductor_module(y3)
        with self.assertRaisesRegex(Exception, ""):
            aot_inductor_module(y4)

    def test_add_complex(self):
        class Model(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        x = torch.tensor(
            [1 + 1j, -1 + 1j, -2 + 2j, 3 - 3j, 0, 1j, 1, -1], device=self.device
        )
        y = torch.tensor(
            [1 + 1j, -1 + 1j, -2 + 2j, 3 - 3j, 0, 1j, 1, -1], device=self.device
        )
        self.check_model(Model(), (x, y))

    def test_embedding_bag(self):
        class Model(torch.nn.Module):
            def forward(self, w, i, o):
                return torch.ops.aten._embedding_bag(w, i, o, False, 0, False, None)

        example_inputs = (
            torch.randn([10, 4], device=self.device),
            torch.randint(10, [8], device=self.device),
            torch.tensor([0, 2, 6], device=self.device),
        )
        self.check_model(Model(), example_inputs)

    @unittest.skipIf(
        TEST_MPS and MACOS_VERSION < 14.0,
        "FFT operations are only supported on MacOS 14+",
    )
    def test_fft_c2c(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.fft.fftn(x), torch.fft.fftn(x).real

        example_inputs = (torch.randn(16, 16, 16, device=self.device),)
        self.check_model(Model(), example_inputs)

    def test_bool_input(self):
        # Specialize on whichever branch the example input for b is
        class Model(torch.nn.Module):
            def forward(self, x, b):
                if b:
                    return x * x
                else:
                    return x + x

        example_inputs = (torch.randn(3, 3, device=self.device), True)
        self.check_model(Model(), example_inputs)

    def test_int_list_input(self):
        class Model(torch.nn.Module):
            def forward(self, x, i):
                return x * i[0] * i[1]

        example_inputs = (torch.randn(3, 3, device=self.device), [3, 4])
        self.check_model(Model(), example_inputs)

    def test_nested_tensor_from_jagged(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mlp = nn.Sequential(
                    nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32), nn.Sigmoid()
                )

            def forward(self, values, offsets):
                nt = torch.nested.nested_tensor_from_jagged(values, offsets)
                res = self.mlp(nt)
                return res.values()

        model = Model().to(device=self.device)

        example_inputs_1 = (
            torch.randn((15, 128), device=self.device),
            torch.tensor([0, 3, 4, 10, 15], device=self.device),
        )

        # same "NT batch size", different actual amount of data
        example_inputs_2 = (
            torch.randn((31, 128), device=self.device),
            torch.tensor([0, 1, 20, 25, 31], device=self.device),
        )

        # same actual amount of data, different "NT batch size"
        example_inputs_3 = (
            torch.randn((15, 128), device=self.device),
            torch.tensor([0, 3, 10, 15], device=self.device),
        )

        # different "NT batch size"
        example_inputs_4 = (
            torch.randn((37, 128), device=self.device),
            torch.tensor([0, 5, 16, 25, 29, 37], device=self.device),
        )

        dim0_values = Dim("dim0_values", min=1, max=128)
        dim0_offsets = Dim("dim0_offsets", min=1, max=9)
        dynamic_shapes = {"values": {0: dim0_values}, "offsets": {0: dim0_offsets}}
        example_inputs_list = [
            example_inputs_1,
            example_inputs_2,
            example_inputs_3,
            example_inputs_4,
        ]
        for example_input in example_inputs_list:
            actual = AOTIRunnerUtil.legacy_run(
                self.device,
                model,
                example_input,
                dynamic_shapes=dynamic_shapes,
            )
            self.assertTrue(same(model(*example_input), actual))

    # Temporarily skipping test as pytorch/cpuinfo not able to retrieve cache size for
    # AMD EPYC 9575F 64-Core Processor CPU in gfx942 VM Runners
    @common_utils.parametrize("max_autotune", [True, False])
    @skipIfRocmArch(MI300_ARCH)
    def test_misc_1(self, max_autotune):
        if self.device == "cpu" and IS_MACOS and max_autotune:
            raise unittest.SkipTest("max_autotune not supported on macos")

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mlp = nn.Sequential(
                    nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32), nn.Sigmoid()
                )
                self.emb = nn.EmbeddingBag(num_embeddings=128, embedding_dim=32)
                self.over_arch = nn.Sequential(
                    nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 32), nn.Sigmoid()
                )

            def forward(self, x, y):
                mlp_output = self.mlp(x)
                emb_output = self.emb(y)
                return self.over_arch(torch.concat([mlp_output, emb_output], dim=1))

        example_inputs = (
            torch.randn(16, 128, device=self.device),
            torch.randint(0, 128, (16, 10), device=self.device),
        )
        self.check_model(
            Model(), example_inputs, options=dict(max_autotune=max_autotune)
        )

    @skip_if_no_torchvision
    def test_torchvision_transforms_functional_tensor_resize(self):
        import torchvision

        # https://fb.workplace.com/groups/1075192433118967/permalink/1501860707118802/
        class A(torch.nn.Module):
            def forward(self, image: torch.Tensor, target_size: torch.Tensor):
                target_h, target_w = target_size.tolist()
                torch._check(target_h > 0)
                torch._check(target_w > 0)
                torch._check(target_h <= 4000)
                torch._check(target_w <= 4000)

                return torchvision.transforms._functional_tensor.resize(
                    image,
                    size=[target_h, target_w],
                    interpolation="bilinear",
                    antialias=False,
                )

        model = A()
        example_inputs = (
            torch.ones([3, 800, 600], device=self.device),
            torch.tensor([448, 336], device=self.device),
        )
        dynamic_shapes = {
            "image": {
                1: torch.export.Dim("height", min=1, max=4000),
                2: torch.export.Dim("width", min=1, max=4000),
            },
            "target_size": None,
        }
        self.check_model(model, example_inputs, dynamic_shapes=dynamic_shapes)

    @unittest.skipIf(config.triton.native_matmul, "matmul is generated")
    def test_aoti_debug_printer_codegen(self):
        # basic addmm model to test codegen for aoti intermediate debug printer
        class Model(torch.nn.Module):
            def __init__(self, n, k, device):
                super().__init__()
                self.weight = torch.randn(n, k, device=device)
                self.bias = torch.randn(n, device=device)

            def forward(self, a):
                return torch.nn.functional.linear(a, self.weight, self.bias)

        M = 8
        N = 6
        K = 16
        model = Model(N, K, self.device)
        batch = 2
        a = torch.randn(batch, M, K, device=self.device)
        example_inputs = (a,)

        if self.device == "mps":
            kernel_calls = [("aoti_torch_mps_addmm_out", 2)]
        elif self.device == GPU_TYPE:
            kernel_calls = [
                ("triton_poi_fused_0", 1),
                (f"aoti_torch_{GPU_TYPE}_addmm_out", 2),
            ]
        else:
            kernel_calls = [("aoti_torch_cpu_addmm_out", 2)]

        # test default debug printing all tensor values codegen
        with config.patch({"aot_inductor.debug_intermediate_value_printer": "2"}):
            result, code = run_and_get_cpp_code(
                AOTIRunnerUtil.legacy_compile, model, example_inputs
            )

            # check the c shim print_tensor_handle call is triggered by the config and injected the cpp output code as expected
            self.assertEqual("aoti_torch_print_tensor_handle" in code, True)

            # check the codegen for debug printing around the actual kernel call is expected

            for kernel_call, count in kernel_calls:
                FileCheck().check_count(
                    f"before_launch - {kernel_call}",
                    count,
                ).run(code)
                FileCheck().check_count(
                    f"after_launch - {kernel_call}",
                    count,
                ).run(code)

        # test printing selected kernel's tensor values codegen
        filtered_kernel_name = f"aoti_torch_{self.device}_addmm_out"
        with config.patch(
            {
                "aot_inductor.debug_intermediate_value_printer": "2",
                "aot_inductor.filtered_kernel_names": filtered_kernel_name,
            }
        ):
            result, code = run_and_get_cpp_code(
                AOTIRunnerUtil.legacy_compile, model, example_inputs
            )
            filtered_kernel_calls = [
                (filtered_kernel_name, 2),
            ]
            for kernel_call, count in filtered_kernel_calls:
                FileCheck().check_count(
                    f"before_launch - {kernel_call}",
                    count,
                ).run(code)
                FileCheck().check_count(
                    f"after_launch - {kernel_call}",
                    count,
                ).run(code)

            kernel_calls_not_to_print = [
                kernel_call
                for kernel_call in kernel_calls
                if kernel_call[0] != filtered_kernel_name
            ]
            for kernel_name, _ in kernel_calls_not_to_print:
                FileCheck().check_not(f"before_launch - {kernel_name}").run(code)
                FileCheck().check_not(f"after_launch - {kernel_name}").run(code)

    @unittest.skipIf(
        config.triton.native_matmul, "different kernel name when native matmul"
    )
    @common_utils.parametrize("enable_kernel_profile", (True, False))
    def test_aoti_profiler(self, enable_kernel_profile):
        # basic addmm model
        class Model(torch.nn.Module):
            def __init__(self, n, k, device):
                super().__init__()
                self.weight = torch.randn(n, k, device=device)
                self.bias = torch.randn(n, device=device)

            def forward(self, a):
                return torch.nn.functional.linear(a, self.weight, self.bias)

        if sys.platform not in ["linux", "win32"]:
            raise unittest.SkipTest(
                "enable_kernel_profile only supported on linux and win32"
            )

        M = 8
        N = 6
        K = 16
        model = Model(N, K, self.device)
        batch = 2
        a = torch.randn(batch, M, K, device=self.device)
        example_inputs = (a,)
        kernel_calls = (
            f"aoti_torch_{GPU_TYPE}_addmm_out"
            if self.device == GPU_TYPE
            else "aoti_torch_cpu_addmm_out"
        )
        with config.patch({"cpp.enable_kernel_profile": enable_kernel_profile}):
            _, code = run_and_get_cpp_code(
                AOTIRunnerUtil.compile, model, example_inputs
            )
            shim_fn_codes = f'RAIIAtenRecordFunctionHandle .*\\("{kernel_calls}"'
            if enable_kernel_profile:
                FileCheck().check_regex(shim_fn_codes).run(code)
            else:
                FileCheck().check_not("RAIIAtenRecordFunctionHandle").run(code)

            self.check_model(Model(N, K, self.device), example_inputs)

    def test_aoti_user_defined_triton_kernel_profiling(self):
        if self.device != GPU_TYPE or self.device == "mps":
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                out = torch.zeros_like(x)
                add_kernel[(4,)](x, y, out, n_elements=4, BLOCK_SIZE=16)
                return out

        example_inputs = (
            torch.randn(4, 4, device=self.device),
            torch.randn(4, 4, device=self.device),
        )

        with (
            config.patch({"cpp.enable_kernel_profile": True}),
            torch.profiler.profile(
                record_shapes=True,
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    getattr(torch.profiler.ProfilerActivity, GPU_TYPE.upper()),
                ],
            ) as prof,
        ):
            self.check_model(Model(), example_inputs)
        with common_utils.TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                import json

                j = json.load(f)
                op_events = [
                    e
                    for e in j["traceEvents"]
                    if e.get("name", "") == "kernels_.add_kernel_0"
                ]
                self.assertEqual(len(op_events), 1)
                self.assertEqual(
                    op_events[0]["args"].get("Input Args", ""),
                    ["in_ptr0", "in_ptr1", "out_ptr", "n_elements"],
                )

    def test_aoti_debug_printer_user_defined_triton_kernel(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                out = torch.zeros_like(x)
                add_kernel[(4,)](x, y, out, n_elements=4, BLOCK_SIZE=16)
                return out

        example_inputs = (
            torch.randn(4, 4, device=self.device),
            torch.randn(4, 4, device=self.device),
        )

        kernel_calls = [
            ("add_kernel_0", 3),
        ]

        with config.patch({"aot_inductor.debug_intermediate_value_printer": "2"}):
            result, code = run_and_get_cpp_code(
                AOTIRunnerUtil.compile, Model(), example_inputs
            )
            # check the c shim print_tensor_handle call is triggered by the config and injected the cpp output code as expected
            self.assertEqual("aoti_torch_print_tensor_handle" in code, True)
            # check the codegen for debug printing around the actual kernel call is expected
            for kernel_call, count in kernel_calls:
                FileCheck().check_count(
                    f"before_launch - {kernel_call}",
                    count,
                ).run(code)
                FileCheck().check_count(
                    f"after_launch - {kernel_call}",
                    count,
                ).run(code)

    def test_aoti_debug_printer_cpp_kernel(self):
        if self.device != "cpu":
            raise unittest.SkipTest("cpu test case only")

        # a simple cpp kernel test case for testing the debug printer codegen
        # on cpp kernel cpu device.
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                t = torch.tensor(x.size(-1), device="cpu", dtype=torch.float)
                t = torch.sqrt(t * 3)
                return x * t

        example_inputs = (torch.randn(4, 4, device="cpu"),)

        kernel_calls = [
            ("cpp_fused_mul_sqrt_0", 2),
        ]

        with config.patch({"aot_inductor.debug_intermediate_value_printer": "2"}):
            result, code = run_and_get_cpp_code(
                AOTIRunnerUtil.compile, Model(), example_inputs
            )
            # check the c shim print_tensor_handle call is triggered by the config and injected the cpp output code as expected
            self.assertEqual("aoti_torch_print_tensor_handle" in code, True)
            # check the codegen for debug printing around the actual kernel call is expected
            for kernel_call, count in kernel_calls:
                FileCheck().check_count(
                    f"before_launch - {kernel_call}",
                    count,
                ).run(code)
                FileCheck().check_count(
                    f"after_launch - {kernel_call}",
                    count,
                ).run(code)

    def test_aoti_debug_printer_sym_inputs(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        from torch.testing._internal.triton_utils import add_kernel

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                maxlen = max(x.item(), 512)
                a = torch.ones(maxlen, device=GPU_TYPE)
                b = torch.ones(maxlen, device=GPU_TYPE)
                out = torch.zeros_like(a)
                # unbacked symint in grid
                add_kernel[(1, 1, maxlen)](a, b, out, maxlen, 32)
                return out

        example_inputs = (torch.randint(high=1024, size=(1,), device=self.device),)

        expected_scalar_args = [
            "triton_poi_fused_zeros_like_0_xnumel",
            "triton_poi_fused_ones_1_xnumel",
            "std::max(static_cast<int64_t>(512L), static_cast<int64_t>(u0))",
        ]

        with config.patch({"aot_inductor.debug_intermediate_value_printer": "2"}):
            result, code = run_and_get_cpp_code(
                AOTIRunnerUtil.compile, Model(), example_inputs
            )
            self.assertEqual("aoti_torch_print_tensor_handle" in code, True)
            for scalar in expected_scalar_args:
                FileCheck().check_count(
                    f"{scalar}",
                    2,
                ).run(code)

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FP8,
        "FP8 is only supported on H100+, SM 8.9 and MI300+ devices",
    )
    @skipIfXpu
    def test_aoti_debug_printer_fp8_dtype(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.out_dtype = dtype

            def forward(self, x, weight, bias, scale_a, scale_b):
                weight = weight.to(e4m3_type)
                output = torch._scaled_mm(
                    x,
                    weight,
                    bias=input_bias,
                    out_dtype=self.out_dtype,
                    scale_a=scale_a,
                    scale_b=scale_b,
                )
                return output

        dtype = torch.float16

        a_scale = torch.Tensor([1.0]).to(device=GPU_TYPE)
        b_scale = torch.Tensor([1.0]).to(device=GPU_TYPE)
        input_bias = torch.rand(32, device=GPU_TYPE, dtype=dtype)
        weight_shape = (32, 16)
        weight = torch.rand(*weight_shape, device=GPU_TYPE, dtype=dtype).T
        a_inverse_scale = 1 / a_scale
        b_inverse_scale = 1 / b_scale

        x_shape = (16, 16)
        x = torch.rand(*x_shape, device=GPU_TYPE, dtype=dtype).to(e4m3_type)

        kernel_calls = [
            (f"aoti_torch_{GPU_TYPE}__scaled_mm_out", 5),
        ]

        # test default debug printing all tensor values codegen
        with config.patch({"aot_inductor.debug_intermediate_value_printer": "2"}):
            result, code = run_and_get_cpp_code(
                AOTIRunnerUtil.legacy_compile,
                Model(dtype),
                (x, weight, input_bias, a_inverse_scale, b_inverse_scale),
            )

            # check the c shim print_tensor_handle call is triggered by the config and injected the cpp output code as expected
            self.assertEqual("aoti_torch_print_tensor_handle" in code, True)

            # check the codegen for debug printing around the actual kernel call is expected and float8 dtype is printed as expected
            for kernel_call, count in kernel_calls:
                FileCheck().check_count(
                    f"before_launch - {kernel_call}",
                    count,
                ).run(code)
                FileCheck().check_count(
                    f"after_launch - {kernel_call}",
                    count,
                ).run(code)

    def test_aoti_debug_printing_model_inputs_codegen(self):
        if self.device not in ["cuda", "xpu"]:
            raise unittest.SkipTest("requires CUDA/XPU")

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b, c):
                x = a * 3.14
                y = torch.addmm(c, x, b)
                z = torch.nn.functional.gelu(y)
                return z

        example_inputs = (
            torch.randn(10, 20, device=GPU_TYPE),
            torch.randn(20, 30, device=GPU_TYPE),
            torch.randn(10, 30, device=GPU_TYPE),
        )
        model = Model()
        kernel_calls = [
            ("aoti_model_inputs", 3),
        ]

        with config.patch({"aot_inductor.debug_intermediate_value_printer": "2"}):
            result, code = run_and_get_cpp_code(
                AOTIRunnerUtil.compile, model, example_inputs
            )
            self.assertEqual("aoti_torch_print_tensor_handle" in code, True)

            # check if the triton kernel is printed as comment
            self.assertEqual("def triton_" in code, True)

            # check the codegen for debug printing around aoti model inputs is expected
            for kernel_call, count in kernel_calls:
                FileCheck().check_count(
                    f"{kernel_call}",
                    count,
                ).run(code)

    def test_size_from_multi_output(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                _x, _i = torch.unique(x, sorted=True, return_inverse=True)
                _x = _x.detach().clone()
                return self.relu(_x), _i

        example_inputs = (torch.randn(8, device=self.device),)
        self.check_model(Model(), example_inputs)

    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_sym_i64_input_codegen(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        from torch.testing._internal.triton_utils import add_kernel

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                x_symint = x.item()
                a = torch.ones(x_symint, device=GPU_TYPE)
                b = torch.ones(x_symint, device=GPU_TYPE)
                out = torch.zeros_like(a)
                # unbacked symint in grid
                add_kernel[(1, 1, x_symint)](a, b, out, x_symint, 32)
                return out

        example_inputs = (
            torch.randint(high=1024, size=(1,), device=self.device, dtype=torch.int32),
        )
        # This simple unit test case model generates two triton kernels:
        # 1. triton_poi_fused_ones_1:
        # triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i64'}
        # 2. add_kernel:
        # triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr': '*fp32', 'n_elements': 'i64'}
        # input u0 was defined as int32_t initially, verify for every kernel var args downstream,
        # it gets explicitly declared using its data types in the cpp wrapper codegen code.
        expected_scalar_args = [
            "buf3, u0",
            "buf4, u0",
            "buf4, buf5, buf3, u0",
        ]
        if full_aoti_runtime_assert():
            # we'll have one more assertion
            expected_scalar_args = [
                "buf4, u0",
                "buf5, u0",
                "buf5, buf6, buf4, u0",
            ]
        # check the new behavior of codegen is expected
        result, code = run_and_get_cpp_code(
            AOTIRunnerUtil.compile, Model(), example_inputs
        )
        for scalar_line in expected_scalar_args:
            FileCheck().check_count(
                scalar_line,
                1,
            ).run(code)
        self.check_model(Model(), example_inputs)

    def test_input_codegen_with_sympy_expr(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class MyModel(torch.nn.Module):
            def forward(self, getitem_54, getitem_52, getitem_19, values_2, offsets):
                bitwise_or = torch.bitwise_or(getitem_54, getitem_52)
                combined = torch.cat([getitem_19, values_2], dim=0)
                add = combined + bitwise_or

                sliced = values_2[:-1] + offsets
                return add, sliced

        inps = (
            torch.randint(0, 1, (240,), device=GPU_TYPE, dtype=torch.uint8),
            torch.randint(0, 1, (240,), device=GPU_TYPE, dtype=torch.uint8),
            torch.randn((192,), device=GPU_TYPE),
            torch.randn((48,), device=GPU_TYPE),
            torch.randint(0, 100, (47,), device=GPU_TYPE, dtype=torch.uint8),
        )

        dim = torch.export.Dim("dimensionality")
        derived_dim = 2 * dim
        spec = {
            "getitem_54": (Dim.AUTO,),  # [s33 + 2*s40 + 1]
            "getitem_52": (Dim.AUTO,),  # [s33 + 2*s40 + 1]
            "getitem_19": (derived_dim,),  # [2*s40]
            "values_2": (Dim.AUTO,),  # [s33 + 1]
            "offsets": (Dim.AUTO,),  # [s33]
        }

        self.check_model(MyModel(), inps, dynamic_shapes=spec)

    @common_utils.parametrize("mark_unbacked", (True, False))
    def test_unbacked_equals_input_size_runtime_assertion(self, mark_unbacked: bool):
        # This test checks the unbacked symint runtime assertions, for the following cases:
        # (A) an unbacked symint equals an unbacked symint (mark_unbacked=True)
        # (B) an unbacked symint equals a backed symint    (mark_unbacked=False)
        class Model(torch.nn.Module):
            def forward(self, a, b, c):
                nz = torch.nonzero(a)
                ones = a.new_ones([nz.size(0), b.size(0)])
                torch._check(ones.size(0) >= 1)
                equals = torch.add(ones, c)
                return equals

        model = Model()
        example_inputs = (
            torch.ones(64, device=self.device),
            b := torch.randn((32,), device=self.device),
            c := torch.randn((64, 32), device=self.device),
        )
        if mark_unbacked:
            torch._dynamo.decorators.mark_unbacked(c, 0)
        else:
            torch._dynamo.mark_dynamic(c, 0)

        # Check the runtime assertion is codegen'ed.
        so_path, code = run_and_get_cpp_code(
            AOTIRunnerUtil.legacy_compile, model, example_inputs
        )
        lowerbound_check = "u1 >= 1" if mark_unbacked else "u0 >= 2"
        FileCheck().check_count(lowerbound_check, 1).run(code)

        compiled = AOTIRunnerUtil.legacy_load(self.device, so_path)
        compiled(*example_inputs)

        # Check the runtime assertion.
        with self.assertRaisesRegex(Exception, ""):
            unexpected_inputs = (torch.ones(0, device=self.device), b, c)
            compiled(*unexpected_inputs)

        # Try it again without runtime assertions.
        with config.patch({"scalar_asserts": False}):
            AOTIRunnerUtil.run_multiple(model, [example_inputs, unexpected_inputs])

    def test_none_args_aot_codegen(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        @triton.autotune(
            configs=[
                triton.Config({"BLOCK_SIZE": 32}, num_stages=5, num_warps=2),
                triton.Config({"BLOCK_SIZE": 64}, num_stages=4, num_warps=4),
            ],
            key=["n_elements"],
        )
        @triton.jit
        def sin_kernel(
            in_ptr0,
            out_ptr,
            # We want to include an arg known to be 1 at compile time
            # This is because we remove None args from the arg list; changing the eq_1/constexpr arg indices.
            # We want to make sure we recompute these correctly
            EQ_1_ARG,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            if in_ptr0 is not None:
                x = tl.load(in_ptr0 + offsets, mask=mask)
            else:
                x = 0.0
            output = tl.sin(x) + EQ_1_ARG
            tl.store(out_ptr + offsets, output, mask=mask)

        def sin_triton(x, out):
            n_elements = out.numel()
            sin_kernel[(n_elements,)](x, out, 1, n_elements)
            return out

        x = torch.randn(65, device=self.device)
        out = torch.empty_like(x)

        not_none_inputs = (x, out)
        none_inputs = (None, out)

        # AOTI compilation specializes on either None or non-None inputs
        # So we have to check twice here

        self.check_model(sin_triton, none_inputs)
        self.check_model(sin_triton, not_none_inputs)

    @skipIfRocm  # RoCM does not support the config block size in test suite.
    def test_autotune_int64_user_defined_triton_kernel(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        @triton.jit
        def add_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0).to(tl.int64)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        @torch.library.triton_op("mylib::add", mutates_args=())
        def custom_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            output = torch.empty_like(x)
            n_elements = output.numel()

            def grid(meta):
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            capture_triton(add_kernel)[grid](x, y, output, n_elements, 16)
            return output

        class Model(torch.nn.Module):
            def forward(self, x):
                x = custom_add(x, x)
                split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(
                    x, [512, 512, 512, 512], 1
                )
                getitem_29 = split_with_sizes_1[0]
                return getitem_29 * 3

        n = 1379584

        try:
            buf196 = torch.randint(
                0, 100, (n, 2048), dtype=torch.int8, device=self.device
            )
            example_inputs = (buf196,)

            self.check_model(
                Model(),
                example_inputs,
                dynamic_shapes={
                    "x": (Dim("x", max=1379584), Dim.STATIC),
                },
                options={"max_autotune": True},
            )
        except torch.OutOfMemoryError:
            # CI can OOM because this test uses too much memory
            raise unittest.SkipTest("OOM. Test is too large") from None

    @skipIfWindows(
        msg="OpenMP crashed application on windows"
    )  # TODO: (xuhancn) need to root cause and fix.
    def test_issue_140766(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(128, 512),
                    torch.nn.ReLU(),
                    torch.nn.Linear(512, 128),
                )
                self.norm = torch.nn.LayerNorm(128)
                self.attn = torch.nn.functional.scaled_dot_product_attention

            def forward(self, x):
                # [2, 128, 4096]
                x = x.transpose(1, 2)
                # [2, 4096, 128]
                for _ in range(2):
                    x = self.forward_block(x)
                return x

            def forward_block(self, x):
                # x: B, H*W, C
                B = x.shape[0]
                H, W, C = 64, 64, 128
                shortcut = x
                x = self.norm(x)
                x = x.reshape(B, H, W, C)
                # B, H, W, C
                x = self.attn(x, x, x)
                x = x.reshape(B, H // 8, W // 8, 8, 8, -1)
                x = x.transpose(2, 3).reshape(B, H * W, -1)

                x = shortcut + x
                x = x + self.mlp(self.norm(x))
                return x

        bs = torch.export.Dim("bs", max=12)
        example_inputs = (torch.randn(2, 128, 4096, device=self.device),)
        self.check_model(Model(), example_inputs, dynamic_shapes={"x": {0: bs}})

    @requires_gpu
    def test_d2h_copy(self):
        # device to copy host should always have the same stride
        if "cuda" not in self.device:
            raise unittest.SkipTest("This test is only for CUDA")

        class ToCpuModel(nn.Module):
            def forward(self, x):
                predictions = x.permute(1, 0)
                predictions = torch.nan_to_num(
                    predictions, nan=0.0, posinf=0.0, neginf=0.0
                )
                predictions = predictions.to("cpu", non_blocking=True)
                p = predictions[0]
                ones = p.new_ones(1)
                return p, ones

        model = ToCpuModel().to(GPU_TYPE)
        input_tensor = torch.randn(5, 10, device=GPU_TYPE).to(dtype=torch.float16)
        ep = torch.export.export(model, (input_tensor,))
        package_path = torch._inductor.aoti_compile_and_package(ep)

        aoti_model = torch._inductor.aoti_load_package(package_path)

        expect_res = model(input_tensor)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        ) as prof:
            true_res = aoti_model(input_tensor)

        self.assertEqual(expect_res, true_res)
        all_ops = [event.key for event in prof.key_averages()]
        self.assertTrue(not any("aten::contiguous" in op for op in all_ops))

    def test_so_without_weight(self):
        class Model(torch.nn.Module):
            def __init__(self, n, k, device):
                super().__init__()
                self.weight = torch.randn(n, k, device=device)
                self.bias = torch.randn(n, device=device)

            def forward(self, a):
                return torch.nn.functional.linear(a, self.weight, self.bias)

        M, N, K = 128, 2048, 4096
        model = Model(N, K, self.device)
        a = torch.randn(M, K, device=self.device)
        example_inputs = (a,)
        with (
            torch.no_grad(),
            config.patch(
                {
                    "always_keep_tensor_constants": True,
                    "aot_inductor.package_constants_in_so": True,
                }
            ),
        ):
            so_path = AOTIRunnerUtil.legacy_compile(
                model=model,
                example_inputs=example_inputs,
            )

        with (
            torch.no_grad(),
            config.patch(
                {
                    "always_keep_tensor_constants": True,
                    "aot_inductor.package_constants_in_so": False,
                }
            ),
        ):
            so_path_weightless = AOTIRunnerUtil.legacy_compile(
                model=model,
                example_inputs=example_inputs,
            )
        self.assertTrue(os.path.getsize(so_path) > 10_000_000)
        self.assertTrue(os.path.getsize(so_path_weightless) < 10_000_000)

        runner = AOTIRunnerUtil.legacy_load_runner(self.device, so_path_weightless)

        # Let's check whether the model has correct constant name mapping.
        expected_original_fqns = {
            "L__self___weight": "L__self___weight",
            "L__self___bias": "L__self___bias",
        }
        self.assertEqual(
            expected_original_fqns, runner.get_constant_names_to_original_fqns()
        )

        def runner_call(*args, **kwargs):
            import torch.fx._pytree as fx_pytree

            call_spec = runner.get_call_spec()
            in_spec = pytree.treespec_loads(call_spec[0])
            out_spec = pytree.treespec_loads(call_spec[1])
            flat_inputs = fx_pytree.tree_flatten_spec((args, kwargs), in_spec)
            flat_inputs = [x for x in flat_inputs if isinstance(x, torch.Tensor)]
            flat_outputs = runner.run(flat_inputs)
            return pytree.tree_unflatten(flat_outputs, out_spec)

        test_inputs = torch.randn(M, K, device=self.device)
        attach_weights = {
            "L__self___weight": model.weight,
            "L__self___bias": model.bias,
        }
        runner.update_constant_buffer(attach_weights, False, False)
        expected = model(test_inputs)
        output = runner_call(test_inputs)

        atol, rtol = 3e-4, 3e-4
        self.assertEqual(expected, output, atol=atol, rtol=rtol)

    def test_weight_on_disk_legacy(self):
        class Model(torch.nn.Module):
            def __init__(self, n, k, device):
                super().__init__()
                self.weight = torch.randn(n, k, device=device)
                self.bias = torch.randn(n, device=device)

            def forward(self, a):
                return torch.nn.functional.linear(a, self.weight, self.bias)

        M, N, K = 128, 2048, 4096
        model = Model(N, K, self.device)
        a = torch.randn(M, K, device=self.device)
        example_inputs = (a,)

        with (
            torch.no_grad(),
            config.patch(
                {
                    "always_keep_tensor_constants": True,
                    "aot_inductor.package_constants_in_so": False,
                    "aot_inductor.package_constants_on_disk_format": "pickle_weights",
                    "aot_inductor.package": True,
                }
            ),
        ):
            aoti_files = AOTIRunnerUtil.legacy_compile(
                model=model,
                example_inputs=example_inputs,
            )

        with WritableTempFile(suffix=".pt2") as f:
            package_path = package_aoti(
                f.name,
                {"model": aoti_files},
            )
            pt2_contents = load_pt2(package_path, load_weights_from_disk=True)
            loaded1 = pt2_contents.aoti_runners["model"]

        atol, rtol = 3e-4, 3e-4
        self.assertEqual(loaded1(a), model(a), atol=atol, rtol=rtol)

    def test_extract_constants_map(self):
        class Model(torch.nn.Module):
            def __init__(self, n, k, device):
                super().__init__()
                self.weight = torch.randn(n, k, device=device)
                self.bias = torch.randn(n, device=device)

            def forward(self, a):
                return torch.nn.functional.linear(a, self.weight, self.bias)

        M, N, K = 8, 6, 16
        model = Model(N, K, self.device)
        a = torch.randn(M, K, device=self.device)
        example_inputs = (a,)
        with torch.no_grad(), config.patch({"always_keep_tensor_constants": True}):
            so_path = AOTIRunnerUtil.legacy_compile(
                model=model,
                example_inputs=example_inputs,
            )

        runner = AOTIRunnerUtil.legacy_load_runner(self.device, so_path)

        def runner_call(*args, **kwargs):
            import torch.fx._pytree as fx_pytree

            call_spec = runner.get_call_spec()
            in_spec = pytree.treespec_loads(call_spec[0])
            out_spec = pytree.treespec_loads(call_spec[1])
            flat_inputs = fx_pytree.tree_flatten_spec((args, kwargs), in_spec)
            flat_inputs = [x for x in flat_inputs if isinstance(x, torch.Tensor)]
            flat_outputs = runner.run(flat_inputs)
            return pytree.tree_unflatten(flat_outputs, out_spec)

        test_inputs = torch.randn(M, K, device=self.device)
        expected = model(test_inputs)
        output = runner_call(test_inputs)
        self.assertEqual(expected, output)

        original_weights = {
            "L__self___weight": model.weight,
            "L__self___bias": model.bias,
        }
        new_weights = {
            "L__self___weight": torch.randn(N, K, device=self.device),
            "L__self___bias": torch.randn(N, device=self.device),
        }

        # Extract weights with use_inactive = False, this should be the current weight.
        extracted_original_weights = runner.extract_constants_map(False)
        self.assertEqual(original_weights, extracted_original_weights)

        # update the inactive weights with new_weights, extract inactive weights.
        runner.update_constant_buffer(new_weights, True, False)
        extracted_new_weights = runner.extract_constants_map(True)
        self.assertEqual(new_weights, extracted_new_weights)

        # Swap constant buffer, this should give us the opposite weights.
        runner.swap_constant_buffer()

        extracted_inactive_weights = runner.extract_constants_map(True)
        extracted_active_weights = runner.extract_constants_map(False)
        self.assertEqual(original_weights, extracted_inactive_weights)
        self.assertEqual(new_weights, extracted_active_weights)

    def test_update_constant_buffer(self):
        class Model(torch.nn.Module):
            def __init__(self, n, k, device):
                super().__init__()
                self.weight = torch.randn(n, k, device=device)
                self.bias = torch.randn(n, device=device)

            def forward(self, a):
                return torch.nn.functional.linear(a, self.weight, self.bias)

        M, N, K = 8, 6, 16
        model = Model(N, K, self.device)
        a = torch.randn(M, K, device=self.device)
        example_inputs = (a,)
        # Attribute naming has changed in the new export API, so still use the legacy API here.
        with torch.no_grad(), config.patch({"always_keep_tensor_constants": True}):
            so_path = AOTIRunnerUtil.legacy_compile(
                model=model,
                example_inputs=example_inputs,
            )

        runner = AOTIRunnerUtil.legacy_load_runner(self.device, so_path)

        # Let's check whether the model has correct constant name mapping.
        expected_original_fqns = {
            "L__self___weight": "L__self___weight",
            "L__self___bias": "L__self___bias",
        }
        self.assertEqual(
            expected_original_fqns, runner.get_constant_names_to_original_fqns()
        )

        def runner_call(*args, **kwargs):
            import torch.fx._pytree as fx_pytree

            call_spec = runner.get_call_spec()
            in_spec = pytree.treespec_loads(call_spec[0])
            out_spec = pytree.treespec_loads(call_spec[1])
            flat_inputs = fx_pytree.tree_flatten_spec((args, kwargs), in_spec)
            flat_inputs = [x for x in flat_inputs if isinstance(x, torch.Tensor)]
            flat_outputs = runner.run(flat_inputs)
            return pytree.tree_unflatten(flat_outputs, out_spec)

        test_inputs = torch.randn(M, K, device=self.device)
        expected = model(test_inputs)
        output = runner_call(test_inputs)
        self.assertEqual(expected, output)

        new_weights = {
            "L__self___weight": torch.randn(N, K, device=self.device),
            "L__self___bias": torch.randn(N, device=self.device),
        }
        runner.update_constant_buffer(new_weights, False, False)
        new_output = runner_call(test_inputs)
        new_expected = torch.nn.functional.linear(
            test_inputs, new_weights["L__self___weight"], new_weights["L__self___bias"]
        )
        self.assertEqual(new_expected, new_output)

    def test_update_constant_buffer_simple(self):
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.weight = torch.randn((3, 3), device=device)

            def forward(self, a):
                return a + self.weight

        model = Model(self.device)
        a = torch.randn((3, 3), device=self.device)
        example_inputs = (a,)

        with torch.no_grad(), config.patch({"always_keep_tensor_constants": True}):
            so_path = AOTIRunnerUtil.legacy_compile(
                model=model,
                example_inputs=example_inputs,
            )

        runner = AOTIRunnerUtil.legacy_load_runner(self.device, so_path)

        # Let's check whether the model has correct constant name mapping.
        expected_original_fqns = {
            "L__self___weight": "L__self___weight",
        }
        self.assertEqual(
            expected_original_fqns, runner.get_constant_names_to_original_fqns()
        )

        test_inputs = torch.randn((3, 3), device=self.device)
        new_weight = torch.randn((3, 3), device=self.device)
        model.weight = new_weight
        attach_weights = {"L__self___weight": new_weight}
        runner.update_constant_buffer(attach_weights, False, False, False)
        expected = model(test_inputs)

        def runner_call(*args, **kwargs):
            call_spec = runner.get_call_spec()  # type: ignore[attr-defined]
            out_spec = pytree.treespec_loads(call_spec[1])
            flat_inputs = pytree.tree_flatten((args, kwargs))[0]
            flat_inputs = [x for x in flat_inputs if isinstance(x, torch.Tensor)]
            flat_outputs = runner.run(flat_inputs)  # type: ignore[attr-defined]
            return pytree.tree_unflatten(flat_outputs, out_spec)

        output = runner_call(test_inputs)
        self.assertEqual(expected, output)

    def test_update_inactive_constant_buffer(self):
        class Model(torch.nn.Module):
            def __init__(self, n, k, device):
                super().__init__()
                self.weight = torch.randn(n, k, device=device)
                self.bias = torch.randn(n, device=device)

            def forward(self, a):
                return torch.nn.functional.linear(a, self.weight, self.bias)

        M, N, K = 8, 6, 16
        model = Model(N, K, self.device)
        a = torch.randn(M, K, device=self.device)
        example_inputs = (a,)
        with torch.no_grad(), config.patch({"always_keep_tensor_constants": True}):
            so_path = AOTIRunnerUtil.legacy_compile(
                model=model,
                example_inputs=example_inputs,
            )

        runner = AOTIRunnerUtil.legacy_load_runner(self.device, so_path)

        def runner_call(*args, **kwargs):
            import torch.fx._pytree as fx_pytree

            call_spec = runner.get_call_spec()
            in_spec = pytree.treespec_loads(call_spec[0])
            out_spec = pytree.treespec_loads(call_spec[1])
            flat_inputs = fx_pytree.tree_flatten_spec((args, kwargs), in_spec)
            flat_inputs = [x for x in flat_inputs if isinstance(x, torch.Tensor)]
            flat_outputs = runner.run(flat_inputs)
            return pytree.tree_unflatten(flat_outputs, out_spec)

        test_inputs = torch.randn(M, K, device=self.device)
        expected = model(test_inputs)
        output = runner_call(test_inputs)
        self.assertEqual(expected, output, atol=1e-3, rtol=1e-3)

        new_weights = {
            "L__self___weight": torch.randn(N, K, device=self.device),
            "L__self___bias": torch.randn(N, device=self.device),
        }
        new_expected = torch.nn.functional.linear(
            test_inputs, new_weights["L__self___weight"], new_weights["L__self___bias"]
        )

        runner.update_constant_buffer(new_weights, True, False)
        output_before_swap = runner_call(test_inputs)
        runner.swap_constant_buffer()
        output_after_swap = runner_call(test_inputs)

        self.assertEqual(expected, output_before_swap)
        self.assertEqual(new_expected, output_after_swap)

    def test_free_inactive_buffer(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            def __init__(self, n, k, device):
                super().__init__()
                self.weight = torch.randn(n, k, device=device)
                self.bias = torch.randn(n, device=device)

            def forward(self, a):
                return torch.nn.functional.linear(a, self.weight, self.bias)

        M, N, K = 8, 6, 16
        model = Model(N, K, self.device)
        a = torch.randn(M, K, device=self.device)
        example_inputs = (a,)
        with torch.no_grad(), config.patch({"always_keep_tensor_constants": True}):
            so_path = AOTIRunnerUtil.legacy_compile(
                model=model,
                example_inputs=example_inputs,
            )

        runner = AOTIRunnerUtil.legacy_load_runner(self.device, so_path)

        def runner_call(*args, **kwargs):
            import torch.fx._pytree as fx_pytree

            call_spec = runner.get_call_spec()
            in_spec = pytree.treespec_loads(call_spec[0])
            out_spec = pytree.treespec_loads(call_spec[1])
            flat_inputs = fx_pytree.tree_flatten_spec((args, kwargs), in_spec)
            flat_inputs = [x for x in flat_inputs if isinstance(x, torch.Tensor)]
            flat_outputs = runner.run(flat_inputs)
            return pytree.tree_unflatten(flat_outputs, out_spec)

        test_inputs = torch.randn(M, K, device=self.device)
        expected = model(test_inputs)
        output = runner_call(test_inputs)
        # Check the outputs, make sure the model is correct here.
        self.assertEqual(expected, output)

        new_weights = {
            "L__self___weight": torch.randn(N, K, device=self.device),
            "L__self___bias": torch.randn(N, device=self.device),
        }
        new_expected = torch.nn.functional.linear(
            test_inputs, new_weights["L__self___weight"], new_weights["L__self___bias"]
        )
        runner.update_constant_buffer(new_weights, True, False)

        # Make sure we have swapped buffer
        runner.swap_constant_buffer()
        output_after_swap = runner_call(test_inputs)
        self.assertEqual(new_expected, output_after_swap)

        # Free the secondary buffer
        runner.free_inactive_constant_buffer()

        # Create a new set of weights to refill into the already freed buffer.
        new_weights_1 = {
            "L__self___weight": torch.randn(N, K, device=self.device),
            "L__self___bias": torch.randn(N, device=self.device),
        }
        new_expected_1 = torch.nn.functional.linear(
            test_inputs, new_weights["L__self___weight"], new_weights["L__self___bias"]
        )
        runner.update_constant_buffer(new_weights_1, True, False)

        output_after_swap_1 = runner_call(test_inputs)
        self.assertEqual(new_expected_1, output_after_swap_1)

        runner.free_inactive_constant_buffer()

    def test_update_user_managed_buffer(self):
        if self.device not in ["cuda", "xpu"]:
            raise unittest.SkipTest("requires CUDA/XPU")

        class Model(torch.nn.Module):
            def __init__(self, n, k, device):
                super().__init__()
                self.weight = torch.randn(n, k, device=device)
                self.bias = torch.randn(n, device=device)

            def forward(self, a):
                return torch.nn.functional.linear(a, self.weight, self.bias)

        M, N, K = 1024, 4096, 4096
        model = Model(N, K, self.device)
        a = torch.randn(M, K, device=self.device)
        example_inputs = (a,)
        # Attribute naming has changed in the new export API, so still use the legacy API here.
        with torch.no_grad(), config.patch({"always_keep_tensor_constants": True}):
            so_path = AOTIRunnerUtil.legacy_compile(
                model=model,
                example_inputs=example_inputs,
            )

        runner = AOTIRunnerUtil.legacy_load_runner(self.device, so_path)

        def runner_call(*args, **kwargs):
            import torch.fx._pytree as fx_pytree

            call_spec = runner.get_call_spec()
            in_spec = pytree.treespec_loads(call_spec[0])
            out_spec = pytree.treespec_loads(call_spec[1])
            flat_inputs = fx_pytree.tree_flatten_spec((args, kwargs), in_spec)
            flat_inputs = [x for x in flat_inputs if isinstance(x, torch.Tensor)]
            flat_outputs = runner.run(flat_inputs)
            return pytree.tree_unflatten(flat_outputs, out_spec)

        test_inputs = torch.randn(M, K, device=self.device)
        expected = model(test_inputs)
        output = runner_call(test_inputs)
        self.assertEqual(expected, output, atol=1e-3, rtol=1e-3)

        new_weights = {
            "L__self___weight": torch.randn(N, K, device=self.device),
            "L__self___bias": torch.randn(N, device=self.device),
        }
        mem_before, _ = getattr(torch, GPU_TYPE).mem_get_info(self.device)
        # Do not use user managed_buffer, should have less free memory.
        runner.update_constant_buffer(new_weights, True, False, False)
        mem_after, _ = getattr(torch, GPU_TYPE).mem_get_info(self.device)
        self.assertGreater(mem_before, mem_after)

        runner.swap_constant_buffer()
        new_output = runner_call(test_inputs)
        new_expected = torch.nn.functional.linear(
            test_inputs, new_weights["L__self___weight"], new_weights["L__self___bias"]
        )
        self.assertEqual(new_expected, new_output, atol=1e-3, rtol=1e-3)

        # Inplace substitube tensor, without user managed buffer, result should be different.
        new_weights["L__self___weight"].add_(1)
        new_weights["L__self___bias"].add_(1)

        new_output = runner_call(test_inputs)
        # Same as the previous result
        self.assertEqual(new_expected, new_output, atol=1e-3, rtol=1e-3)
        new_expected = torch.nn.functional.linear(
            test_inputs, new_weights["L__self___weight"], new_weights["L__self___bias"]
        )
        # Differ from latest result
        self.assertNotEqual(new_expected, new_output)

        # Clear out all buffers
        runner.free_inactive_constant_buffer()
        runner.swap_constant_buffer()
        runner.free_inactive_constant_buffer()

        new_weights = {
            "L__self___weight": torch.randn(N, K, device=self.device),
            "L__self___bias": torch.randn(N, device=self.device),
        }
        mem_before, _ = getattr(torch, GPU_TYPE).mem_get_info(self.device)
        # Try user managed_buffer, should have same free memory.
        runner.update_constant_buffer(new_weights, True, False, True)
        mem_after, _ = getattr(torch, GPU_TYPE).mem_get_info(self.device)
        self.assertEqual(mem_before, mem_after, atol=1e-3, rtol=1e-3)

        runner.swap_constant_buffer()
        new_output = runner_call(test_inputs)
        new_expected = torch.nn.functional.linear(
            test_inputs, new_weights["L__self___weight"], new_weights["L__self___bias"]
        )
        self.assertEqual(new_expected, new_output, atol=1e-3, rtol=1e-3)

        # Inplace substitube tensor, with user managed buffer, result should be the same.
        new_weights["L__self___weight"].add_(1)
        new_weights["L__self___bias"].add_(1)

        new_output = runner_call(test_inputs)
        new_expected = torch.nn.functional.linear(
            test_inputs, new_weights["L__self___weight"], new_weights["L__self___bias"]
        )
        self.assertEqual(new_expected, new_output, atol=1e-3, rtol=1e-3)

        new_weights = {
            "L__self___weight": torch.randn(N, K, device=self.device),
            "L__self___bias": torch.randn(N, device=self.device),
        }

        runner.update_constant_buffer(new_weights, True, False, True)
        runner.swap_constant_buffer()

        model.weight = torch.nn.Parameter(new_weights["L__self___weight"])
        model.bias = torch.nn.Parameter(new_weights["L__self___bias"])

        updated_state_dict = {
            "weight": torch.ones_like(model.weight),
            "bias": torch.zeros_like(model.bias),
        }

        model.load_state_dict(updated_state_dict)

        new_output = runner_call(test_inputs)
        expected_output = model(test_inputs)
        torch.testing.assert_close(new_output, expected_output, atol=1e-3, rtol=1e-3)

        with self.assertRaises(AssertionError):
            torch.testing.assert_close(new_expected, new_output, atol=1e-3, rtol=1e-3)

    def test_cond_share_predicte(self):
        class Model(torch.nn.Module):
            def forward(self, predicate, x):
                y = torch.cond(
                    predicate,
                    lambda: x + 1,
                    lambda: x + 2,
                )

                z = torch.cond(
                    predicate,
                    lambda: y + 1,
                    lambda: y + 2,
                )
                return (z,)

        example_inputs = (
            torch.tensor([True]).to(self.device),
            torch.tensor([1, 2, 3]).to(self.device),
        )
        self.check_model(Model(), example_inputs)

    @unittest.skipIf(
        IS_FBCODE,
        "To enable after the C shim FC window ends",
    )
    def test_misaligned_input_1(self):
        if self.device not in ["cuda", "xpu"]:
            raise unittest.SkipTest("CUDA/XPU test only")

        class Model(torch.nn.Module):
            def forward(self, x):
                return x.sin() + x.cos()

        N = 64 * 64 * 64 + 64
        arg = torch.randn(N, device=self.device)
        example_inputs = (arg,)
        model = Model()
        expected = model(*example_inputs)
        package_path = AOTIRunnerUtil.compile(model, example_inputs)
        optimized = torch._inductor.aoti_load_package(package_path)
        # If the model is compiled with aligned inputs, the generated
        # code will check inputs alignment at runtime
        self.code_check_count(
            model, example_inputs, "aoti_torch_clone_preserve_strides", 1
        )

        misaligned_arg = torch.zeros(N + 1, device=self.device)
        misaligned_arg = misaligned_arg[1:]
        misaligned_arg.copy_(arg)
        actual = optimized(misaligned_arg)
        torch.testing.assert_close(actual, expected)

    def test_misaligned_input_2(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("GPU test only")

        class Model(torch.nn.Module):
            def forward(self, x):
                return x.sin() + x.cos()

        N = 64 * 64 * 64 + 64
        arg = torch.randn(N, device=self.device)
        misaligned_arg = torch.zeros(N + 1, device=self.device)
        misaligned_arg = misaligned_arg[1:]
        misaligned_arg.copy_(arg)
        example_inputs = (misaligned_arg,)

        model = Model()
        self.check_model(model, example_inputs)
        # If the model is already compiled with a misaligned input, the
        # generated code should NOT contain an alignment check for that input.
        self.code_check_count(
            model, example_inputs, "aoti_torch_clone_preserve_strides", 0
        )

    def test_autotuning_args_reuse(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            def forward(self, x, y):
                x_out = torch.empty_strided(
                    (x.size()[0], x.size()[1]), (x.size()[1], 1), device=GPU_TYPE
                )
                x_out = torch.permute(x_out, [0, 1])
                add_kernel_autotuned[(4,)](x, x, x_out, 16)

                y_out = torch.empty_strided(
                    (y.size()[0], y.size()[1]), (y.size()[1], 1), device=GPU_TYPE
                )
                y_out = torch.permute(y_out, [0, 1])
                add_kernel_autotuned[(64,)](y, y, y_out, 64)

                sub_kernel_autotuned[(4,)](x, x, x_out, 16)

                return x_out, y_out

        example_inputs = (
            torch.randn(4, 4, device=GPU_TYPE),
            torch.randn(8, 8, device=GPU_TYPE),
        )
        dim0_x = Dim("dim0_x", min=1, max=2048)
        dim0_y = Dim("dim0_y", min=1, max=2048)
        dynamic_shapes = {"x": {0: dim0_x}, "y": {0: dim0_y}}
        self.check_model(
            Model(),
            example_inputs,
            dynamic_shapes=dynamic_shapes,
            options={"max_autotune": True},
        )

    @unittest.skipIf(IS_FBCODE, "Not runnable in fbcode")
    @unittest.skipIf(
        TEST_MPS and MACOS_VERSION < 14.0,
        "FFT operations are only supported on MacOS 14+",
    )
    def test_stft(self):
        N_FFT = 400
        HOP_LENGTH = 160

        class Model(torch.nn.Module):
            def forward(self, x):
                window = torch.hann_window(N_FFT, device=x.device)
                stft = torch.stft(
                    x, N_FFT, HOP_LENGTH, window=window, return_complex=True
                )
                magnitudes = stft[..., :-1].abs() ** 2
                return magnitudes

        model = Model()
        example_inputs = (torch.randn(500, device=self.device),)
        self.check_model(model, example_inputs)

    @skipIfXpu
    def test_conv3d(self):
        if self.device != GPU_TYPE or not is_big_gpu():
            raise unittest.SkipTest("requires modern GPU to run max-autotune")

        if not _has_sufficient_memory(self.device, 2**35):
            raise unittest.SkipTest("insufficient memory")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(
                self,
                convert_element_type_1271,
                convert_element_type_1272,
                convert_element_type_1273,
            ):
                return torch.ops.aten.convolution.default(
                    convert_element_type_1271,
                    convert_element_type_1272,
                    convert_element_type_1273,
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    False,
                    [0, 0],
                    1,
                )

        example_inputs = (
            torch.randn(1, 64, 5160, 5160, device=self.device),
            torch.randn(3, 64, 3, 3, device=self.device),
            torch.randn(3, device=self.device),
        )
        dynamic_shapes = {
            "convert_element_type_1271": {
                3: torch.export.Dim.DYNAMIC,
                4: torch.export.Dim.DYNAMIC,
            },
            "convert_element_type_1272": None,
            "convert_element_type_1273": None,
        }
        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_conv_backends": "TRITON",
            }
        ):
            self.check_model(
                Model(),
                example_inputs,
                atol=0.1,
                rtol=1e-3,
                dynamic_shapes=dynamic_shapes,
            )

    def test__int_mm(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                return torch._int_mm(x, y)

        example_inputs = (
            torch.randint(-10, 10, (64, 32), device=self.device, dtype=torch.int8),
            torch.randint(-10, 10, (32, 64), device=self.device, dtype=torch.int8),
        )
        self.check_model(Model(), example_inputs)

    @skipIfMPS
    @skipIfXpu(
        msg="aten::convert_weight_to_int4pack is not currently implemented for XPU"
    )
    @parametrize("m", [32])
    @parametrize("n", [64])
    @parametrize("q_group", [32, 64])
    @parametrize("num_groups", [1, 2])
    def test__weight_int4pack_mm(self, m, n, q_group, num_groups):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        if TEST_WITH_ROCM:
            if not CDNA2OrLater():
                self.skipTest("_int4_mm is supported only for CDNA2 or later")

        class Model(torch.nn.Module):
            def __init__(self, weight, scale_and_zeros) -> None:
                super().__init__()
                self.weight = weight
                self.scale_and_zeros = scale_and_zeros

            def forward(self, a):
                return torch._weight_int4pack_mm(
                    a, self.weight, q_group, self.scale_and_zeros
                )

        def convert_weight_to_int4pack(b):
            b_int32, b_scales_and_zeros = _group_quantize_tensor(
                b, n_bit=4, q_group_size=q_group
            )
            b_int4pack = torch._convert_weight_to_int4pack(b_int32, innerKTiles=2)
            return b_int4pack, b_scales_and_zeros

        k = q_group * num_groups
        a = torch.rand((m, k), device=self.device, dtype=torch.bfloat16)
        b = torch.rand((k, n), device=self.device, dtype=torch.bfloat16)
        b_int4pack, b_scales_and_zeros_f32 = convert_weight_to_int4pack(b)
        model = Model(b_int4pack, b_scales_and_zeros_f32)
        self.check_model(model, (a,))

    @parametrize("m", [32])
    @parametrize("n", [64])
    @parametrize("q_group", [32, 64])
    @parametrize("num_groups", [1, 2])
    def test__weight_int4pack_mm_with_scales_and_zeros(self, m, n, q_group, num_groups):
        if "xpu" not in self.device:
            raise unittest.SkipTest("requires Intel GPU")

        if TEST_WITH_ROCM:
            if not CDNA2OrLater():
                self.skipTest("_int4_mm is supported only for CDNA2 or later")

        class Model(torch.nn.Module):
            def __init__(self, weight, scale, zeros) -> None:
                super().__init__()
                self.weight = weight
                self.scale = scale
                self.zeros = zeros

            def forward(self, a):
                return torch._weight_int4pack_mm_with_scales_and_zeros(
                    a, self.weight, q_group, self.scale, self.zeros
                )

        def _group_quantize_tensor_xpu(w, n_bit=4, q_group_size=16):
            # w [k, n] = [32, 48]
            assert w.dim() == 2
            # w [n, k] = [48, 32]
            w = w.transpose(0, 1).contiguous()
            assert q_group_size > 1
            assert w.shape[-1] % q_group_size == 0

            # to_quant: [n * k / group_size, group_size]
            to_quant = w.reshape(-1, q_group_size)
            assert torch.isnan(to_quant).sum() == 0

            max_val = to_quant.amax(dim=1, keepdim=True)
            min_val = to_quant.amin(dim=1, keepdim=True)
            max_int = 2**n_bit - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-6) / max_int
            assert torch.isnan(scales).sum() == 0

            zeros = min_int - min_val.div(scales).round()
            zeros = torch.clamp(zeros, min_int, max_int)
            zeros = zeros.to(torch.int8)
            assert torch.isnan(zeros).sum() == 0

            out = to_quant.div(scales).add(zeros).round().clamp_(min_int, max_int)
            assert torch.isnan(out).sum() == 0

            # [n, k]
            out = out.to(dtype=torch.int32).reshape(w.shape)
            if out.device != torch.device("cpu"):
                out = (out[::, 1::2] << 4 | out[::, 0::2]).to(torch.uint8)

            # Scales and zeros for the same q-group should be contiguous, so we can
            # load as a 32-bit word
            scales = scales.view(w.shape[0], -1).transpose(0, 1).contiguous()
            zeros = zeros.view(w.shape[0], -1).transpose(0, 1).contiguous()

            return out, scales, zeros

        def convert_weight_to_int4pack(b):
            # b_uint8 [n, k //2]
            b_uint8, scales, zeros = _group_quantize_tensor_xpu(
                b, n_bit=4, q_group_size=q_group
            )
            # b_int4pack [k//8, n]
            b_int4pack = torch._convert_weight_to_int4pack(b_uint8, innerKTiles=2)

            return b_int4pack, scales, zeros

        k = q_group * num_groups
        a = torch.rand((m, k), device=self.device, dtype=torch.bfloat16)
        b = torch.rand((k, n), device=self.device, dtype=torch.bfloat16)
        b_int4pack, b_scales, zeros_int8 = convert_weight_to_int4pack(b)
        model = Model(b_int4pack, b_scales, zeros_int8)
        self.check_model(model, (a,))

    def test_assert_tensor_meta(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                torch.ops.aten._assert_tensor_metadata.default(
                    x,
                    dtype=torch.int32,
                )
                return (x + 1,)

        example_inputs = (torch.tensor(1, dtype=torch.int32),)
        with config.patch(
            {
                "implicit_fallbacks": False,
            }
        ):
            self.check_model(
                Module(),
                example_inputs,
                atol=0.1,
                rtol=1e-3,
            )

    @runOnRocm
    def test_rocm_triton_autotuning(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            def forward(self, x, y, m):
                _M, K = x.shape
                K, N = y.shape
                M = torch.abs(m)
                out = torch.empty((_M, N), device=x.device, dtype=torch.float32)
                grid = lambda META: (  # noqa: E731
                    triton.cdiv(
                        4096 * 2046, META["BLOCK_SIZE_M"] * META["BLOCK_SIZE_N"]
                    ),
                )
                strange_config_matmul_kernel[grid](
                    x,
                    y,
                    out,
                    M,
                    N,
                    K,
                )
                return out

        x = torch.randn(4096, 1024, device=self.device)
        y = torch.randn(1024, 2048, device=self.device)
        m = torch.tensor([4096], dtype=torch.int32, device=self.device)

        with (
            torch.no_grad(),
            config.patch(
                {
                    "triton.autotune_with_sample_inputs": True,
                    "aot_inductor.allow_stack_allocation": self.allow_stack_allocation,
                    "aot_inductor.use_minimal_arrayref_interface": self.use_minimal_arrayref_interface,
                }
            ),
        ):
            torch._export.aot_compile(Model(), (x, y, m))

    @skipIfRocm  # RoCM does not support the config block size in test suite.
    def test_triton_autotuning(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            def forward(self, x, y, m):
                _M, K = x.shape
                K, N = y.shape
                M = torch.abs(m)
                out = torch.empty((_M, N), device=x.device, dtype=torch.float32)
                grid = lambda META: (  # noqa: E731
                    triton.cdiv(
                        4096 * 2046, META["BLOCK_SIZE_M"] * META["BLOCK_SIZE_N"]
                    ),
                )
                strange_config_matmul_kernel[grid](
                    x,
                    y,
                    out,
                    M,
                    N,
                    K,
                )
                return out

        x = torch.randn(4096, 1024, device=self.device)
        y = torch.randn(1024, 2048, device=self.device)
        m = torch.tensor([4096], dtype=torch.int32, device=self.device)

        with config.patch("triton.autotune_with_sample_inputs", True):
            # The tuned best config on XPU is different with CUDA.
            grid_0 = 32736 if GPU_TYPE == "xpu" else 1023
            self.code_check_count(
                Model(), (x, y, m), f"uint32_t grid_0 = {grid_0}L;", 1
            )

    @skipIfRocm  # RoCM does not support the config block size in test suite.
    def test_triton_mutated_autotuning(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        @triton.jit
        def add_one_kernel(X, Y, N):
            pid = tl.program_id(axis=0)
            block_start = pid
            offsets = block_start + tl.arange(0, 1)

            x = tl.load(X + offsets, mask=offsets < N)
            y = x + 1
            tl.store(Y + offsets, y, mask=offsets < N)

        class Model(torch.nn.Module):
            def forward(self, x, y, m):
                _M, K = x.shape
                K, N = y.shape
                M = torch.empty((1), device=x.device, dtype=torch.int32)
                add_one_kernel[(1,)](m, M, 1)
                out = torch.empty((_M, N), device=x.device, dtype=torch.float32)
                grid = lambda META: (  # noqa: E731
                    triton.cdiv(
                        4096 * 2046, META["BLOCK_SIZE_M"] * META["BLOCK_SIZE_N"]
                    ),
                )
                strange_config_matmul_kernel[grid](
                    x,
                    y,
                    out,
                    M,
                    N,
                    K,
                )
                return out

        x = torch.randn(4096, 1024, device=self.device)
        y = torch.randn(1024, 2048, device=self.device)
        m = torch.tensor([4095], dtype=torch.int32, device=self.device)

        with config.patch("triton.autotune_with_sample_inputs", True):
            # The tuned best config on XPU is different with CUDA.
            grid_0 = 32736 if GPU_TYPE == "xpu" else 1023
            self.code_check_count(
                Model(), (x, y, m), f"uint32_t grid_0 = {grid_0}L;", 1
            )

    @skipIfRocm
    @patch.dict(os.environ, {"TRITON_DEBUG": "1"})
    def test_triton_dynamic_launcher_grid(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        @triton.autotune(
            configs=[
                triton.Config({"BLOCK_SIZE": 32}, num_stages=5, num_warps=2),
                triton.Config({"BLOCK_SIZE": 64}, num_stages=4, num_warps=4),
            ],
            key=["numel"],
        )
        @triton.jit
        def add_one_kernel(X, Y, numel, BLOCK_SIZE: "tl.constexpr"):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            tl.device_assert(block_start < numel)
            offsets = block_start + tl.arange(0, BLOCK_SIZE)

            x = tl.load(X + offsets)
            y = x + 1
            tl.store(Y + offsets, y)

        class Model(torch.nn.Module):
            def forward(self, x, value):
                numel = value.item()
                out = torch.zeros_like(x, dtype=torch.float16)

                grid = lambda META: (  # noqa: E731
                    triton.cdiv(numel, META["BLOCK_SIZE"]),
                )
                add_one_kernel[grid](x, out, numel)

                return out

        example_inputs = (
            torch.randn(1024, device=self.device),
            torch.tensor([1024], dtype=torch.int32, device=self.device),
        )

        with config.patch("triton.autotune_with_sample_inputs", True):
            dim0_x = Dim("dim0_x", min=2, max=8192)
            dynamic_shapes = {"x": {0: dim0_x}, "value": {0: Dim.AUTO}}
            self.check_model(Model(), example_inputs, dynamic_shapes=dynamic_shapes)

    @skipIfRocm
    @patch.dict(os.environ, {"TRITON_DEBUG": "1"})
    def test_triton_dynamic_launcher_grid_infer_from_tensor(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        @triton.autotune(
            configs=[
                triton.Config({"BLOCK_SIZE": 32}, num_stages=5, num_warps=2),
                triton.Config({"BLOCK_SIZE": 64}, num_stages=4, num_warps=4),
            ],
            key=["numel"],
        )
        @triton.jit
        def add_one_kernel(X, Y, numel, BLOCK_SIZE: "tl.constexpr"):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            tl.device_assert(block_start < numel)

            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            x = tl.load(X + offsets)
            y = x + 1
            tl.store(Y + offsets, y)

        class Model(torch.nn.Module):
            def forward(self, x, dim_D):
                numel = x.shape[1] * dim_D.item()
                x = x.repeat(dim_D, 1)
                out = torch.zeros_like(x, dtype=torch.float16)

                grid = lambda META: (  # noqa: E731
                    triton.cdiv(numel, META["BLOCK_SIZE"]),
                )
                add_one_kernel[grid](x, out, numel)

                return out

        example_inputs = (
            torch.randn(1, 1024, device=self.device),
            torch.tensor([2], dtype=torch.int32, device=self.device),
        )

        with config.patch("triton.autotune_with_sample_inputs", True):
            dim1_x = Dim("dim1_x", min=2, max=8192)
            dynamic_shapes = {"x": {0: Dim.AUTO, 1: dim1_x}, "dim_D": {0: Dim.AUTO}}
            self.check_model(Model(), example_inputs, dynamic_shapes=dynamic_shapes)

    def test_composed_dynamic_size(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x + 1

        example_inputs = (torch.randn(10, device=self.device),)
        dim = torch.export.Dim("dim_0")
        dim_even = 2 * dim
        dynamic_shapes = {
            "x": {0: dim_even},
        }
        self.check_model(Model(), example_inputs, dynamic_shapes=dynamic_shapes)

    def test_boolean_indexing(self):
        class Model(torch.nn.Module):
            def forward(self, x, y, z, x1, z1):
                a = x[y]
                a1 = x1[y]
                b = torch.cat([a, z], dim=1)
                b1 = torch.cat([a1, z1], dim=1)
                return b, b1

        x = torch.randn(3, 5, device=self.device)
        y = torch.tensor([0, 1, 1], dtype=torch.bool, device=self.device)
        z = torch.randn(2, 4, device=self.device)
        x1 = torch.randn(3, 5, device=self.device)
        z1 = torch.randn(2, 4, device=self.device)

        example_inputs = (x, y, z, x1, z1)
        s0 = Dim("s0", min=0, max=10240)
        s1 = Dim("s1", min=0, max=10240)
        s2 = Dim("s2", min=0, max=10240)
        s3 = Dim("s3", min=0, max=10240)
        dynamic_shapes = {
            "x": {0: s0, 1: s1},
            "y": {0: s0},
            "z": {0: s2, 1: s3},
            "x1": {0: s0, 1: s1},
            "z1": {0: s2, 1: s3},
        }
        self.check_model(Model(), example_inputs, dynamic_shapes=dynamic_shapes)

    def test_sym_expr_indexing(self):
        if self.device not in ["cuda", "xpu"]:
            raise unittest.SkipTest("requires CUDA/XPU")

        class Repro(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(
                self,
                episode_builder_position_encoding_observations_weight,
                add_15,
                add_16,
                add_17,
                add_18,
                add_13,
            ):
                arange_1 = torch.ops.aten.arange.start(
                    180,
                    181,
                    device=torch.device(type=GPU_TYPE, index=0),
                    pin_memory=False,
                )
                add_14 = torch.ops.aten.add.Tensor(arange_1, 198)
                arange_1 = None
                stack_1 = torch.ops.aten.stack.default(
                    [add_13, add_14, add_15, add_16, add_17, add_18]
                )
                add_13 = add_14 = add_15 = add_16 = add_17 = add_18 = None
                select_13 = torch.ops.aten.select.int(stack_1, 0, 0)
                stack_1 = None
                embedding_11 = torch.ops.aten.embedding.default(
                    episode_builder_position_encoding_observations_weight, select_13
                )
                episode_builder_position_encoding_observations_weight = select_13 = None
                return (embedding_11,)

        # Embedding weight: vocab_size x emb_dim
        episode_builder_position_encoding_observations_weight = torch.randn(
            100, 16, device=self.device
        )

        # These six must all be 1-D (shape [1]) and same dtype; use Long for embedding indices
        add_13 = torch.tensor(
            [7], dtype=torch.long, device=self.device
        )  # this one is used as the index
        add_15 = torch.tensor([5], dtype=torch.long, device=self.device)
        add_16 = torch.tensor([6], dtype=torch.long, device=self.device)
        add_17 = torch.tensor([7], dtype=torch.long, device=self.device)
        add_18 = torch.tensor([8], dtype=torch.long, device=self.device)

        # Instantiate and run
        m = Repro().to(self.device)

        example_inputs = (
            episode_builder_position_encoding_observations_weight,
            add_15,
            add_16,
            add_17,
            add_18,
            add_13,
        )
        self.check_model(m, example_inputs)

    def test_with_cudagraphs(self):
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        # define CUDAGraph handling wrapper (only works with kwargs for simplicity)
        def cudagraph(f):
            _graphs = {}

            def f_(**kwargs):
                key = hash(
                    tuple(
                        tuple(kwargs[a].shape)
                        for a in sorted(kwargs.keys())
                        if isinstance(kwargs[a], torch.Tensor)
                    )
                )
                if key in _graphs:
                    wrapped, *_ = _graphs[key]
                    return wrapped(**kwargs)
                g = torch.cuda.CUDAGraph()
                in_tensors = {
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in kwargs.items()
                }
                f(**in_tensors)  # stream warmup
                with torch.cuda.graph(g):
                    out_tensors = f(**in_tensors)

                def wrapped(**kwargs):
                    for key in kwargs:
                        in_tensors[key].copy_(kwargs[key])
                    g.replay()
                    if isinstance(out_tensors, torch.Tensor):
                        return out_tensors.clone()
                    elif isinstance(out_tensors, (list, tuple)):
                        return type(out_tensors)(o.clone() for o in out_tensors)
                    raise ValueError("unsupported output type encountered")

                _graphs[key] = (wrapped, g, in_tensors, out_tensors)
                return wrapped(**kwargs)

            return f_

        # define a simple model
        model = torch.nn.Linear(10, 20).to(device=self.device)

        # export + AOTI
        model_kwargs = {
            "input": torch.randn(3, 10, device=self.device),
        }
        ep = torch.export.export(model, args=(), kwargs=model_kwargs, strict=True)

        optimized = torch._inductor.aoti_load_package(
            torch._inductor.aoti_compile_and_package(
                ep,
                inductor_configs={"max_autotune": True},
            ),
            # NB: this flag avoids a CUDAGraph + AOTI runtime multi-threading conflict
            # "Error: operation not permitted when stream is capturing"
            run_single_threaded=True,
        )

        # enable CUDAGraphs
        optimized = cudagraph(optimized)

        # warmup -> run with CUDAGraphs
        for _ in range(3):
            optimized(**model_kwargs)

        # compare against eager
        self.assertEqual(optimized(**model_kwargs), model(**model_kwargs))

    def test_custom_op_in_subgraph(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo_add1",
                "(Tensor a) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo_add1", "CompositeExplicitAutograd", lib=lib)
            @torch.library.register_fake("mylib::foo_add1", lib=lib)
            def foo_add1_impl(a: torch.Tensor) -> torch.Tensor:
                return a + 1

            torch.library.define(
                "mylib::foo_add2",
                "(Tensor a) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo_add2", "CompositeExplicitAutograd", lib=lib)
            @torch.library.register_fake("mylib::foo_add2", lib=lib)
            def foo_add2_impl(a: torch.Tensor) -> torch.Tensor:
                return a + 2

            class M(torch.nn.Module):
                def forward(self, x):
                    return torch.cond(
                        x.shape[0] < 5,
                        torch.ops.mylib.foo_add1,
                        torch.ops.mylib.foo_add2,
                        (x,),
                    )

            list_example_inputs = [
                (torch.ones(6, device=self.device),),
                (torch.ones(3, device=self.device),),
            ]
            self.check_model_with_multiple_inputs(
                M(), list_example_inputs, dynamic_shapes=({0: Dim.DYNAMIC},)
            )

    def test_clamp_decomposition(self):
        class Model1(torch.nn.Module):
            def forward(self, x):
                return x.clamp(min=1.5)

        class Model2(torch.nn.Module):
            def forward(self, x):
                return x.clamp(min=2)

        x = torch.randint(4, (4,))

        # the output should have float32 type, not int
        self.check_model(Model1(), (x,))
        # the output should have int type
        self.check_model(Model2(), (x,))

    def test_upper_bound_i64(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        inp = (
            torch.randint(0, 100, (2**18,), device=self.device, dtype=torch.int8),
            torch.tensor([4], device=self.device, dtype=torch.int8),
        )
        ep = torch.export.export(
            Model(),
            inp,
            dynamic_shapes=({0: Dim("d", min=0, max=2**33)}, {0: Dim.STATIC}),
        )
        so_path = torch._inductor.aot_compile(ep.module(), inp)
        m = torch._export.aot_load(so_path, self.device)

        self.assertEqual(Model()(*inp), m(*inp))
        del inp

        inp = (
            torch.randint(0, 100, (3 * 2**30,), device=self.device, dtype=torch.int8),
            torch.tensor([4], device=self.device, dtype=torch.int8),
        )
        # don't check the accuracy of the result to reduce memory usage
        # this test is mostly checking to ensure there's no IMA.
        m(*inp)

    def test_using_model_name_for_files(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x, y):
                return x + self.linear(y)

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        model = Model().to(self.device)
        with torch.no_grad():
            package_path: str = AOTIRunnerUtil.compile(
                model,
                example_inputs,
                inductor_configs={
                    "aot_inductor.model_name_for_generated_files": "test_model"
                },
            )

        with zipfile.ZipFile(package_path, "r") as zip_ref:
            all_files = zip_ref.namelist()
            base_dir = "test_model.wrapper/data/aotinductor/model/test_model"
            ext_type = get_module_ext_type()
            self.assertTrue(f"{base_dir}.wrapper.cpp" in all_files)
            self.assertTrue(f"{base_dir}.kernel.cpp" in all_files)
            self.assertTrue(f"{base_dir}.wrapper.{ext_type}" in all_files)

        aot_inductor_module = torch._inductor.aoti_load_package(package_path)
        self.assertEqual(aot_inductor_module(*example_inputs), model(*example_inputs))

    def test_copy_non_blocking_is_pinned(self):
        if self.device == "cpu" or self.device == "mps":
            raise unittest.SkipTest("only matters for device-to-cpu copy")

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b):
                a_cpu = a.to(device="cpu", non_blocking=True)
                b_cpu = b.to(device="cpu", non_blocking=True)
                a_to_cpu_event = torch.Event()
                a_to_cpu_event.record()
                a_to_cpu_event.synchronize()
                return torch.cat([a_cpu, b_cpu])

        model = Model()
        a = torch.randn(2, 2, device=self.device)
        b = torch.randn(2, 2, device=self.device)
        example_inputs = (a, b)
        outputs = model(*example_inputs)
        package_path, code = run_and_get_cpp_code(
            AOTIRunnerUtil.compile, model, example_inputs
        )
        FileCheck().check("pinned").run(code)
        model_aoti = torch._inductor.aoti_load_package(package_path)
        outputs_aoti = model_aoti(*example_inputs)

        self.assertEqual(outputs, outputs_aoti)

    @unittest.skipIf(config.triton.native_matmul, "different code generated")
    def test_pad_non_zero_memory_leak(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("test is only for GPU_TYPE")

        class Model(torch.nn.Module):
            def forward(self, x):
                x = x + 1
                x = torch.ops.aten.constant_pad_nd(x, (0, 1, 0, 0), 12345.0)

                return x @ x

        model = Model()
        example_inputs = (torch.randn(2048, 2047, device=self.device),)
        package_path, code = run_and_get_cpp_code(
            AOTIRunnerUtil.compile, model, example_inputs
        )
        outputs = model(*example_inputs)
        model_aoti = torch._inductor.aoti_load_package(package_path)
        outputs_aoti = model_aoti(*example_inputs)

        self.assertEqual(outputs, outputs_aoti, atol=1e-2, rtol=1e-2)

        FileCheck().check_regex(
            r"aoti_torch_as_strided\(buf0_handle, .*, &buf0_handle_restrided\)"
        ).check("wrap_with_raii_handle_if_needed(buf0_handle);").check(
            "RAIIAtenTensorHandle buf0(buf0_handle_restrided);"
        ).run(code)

    def test_codegen_int_array_var_fix_memory_leak(self):
        """
        Fix https://github.com/pytorch/pytorch/issues/167630
        """
        if self.device != "cuda":
            raise unittest.SkipTest("test is only for cuda")

        def make_mlp(in_dim=128, hidden=256, out_dim=64, depth=3):
            layers = []
            d = in_dim
            for _ in range(depth):
                layers += [nn.Linear(d, hidden), nn.ReLU()]
                d = hidden
            layers += [nn.Linear(d, out_dim)]
            return nn.Sequential(*layers)

        batch = 32
        in_dim = 2048
        hidden = 512
        out_dim = 10
        depth = 6

        import gc

        allocated_memory = []
        for _ in range(3):
            torch.cuda.reset_peak_memory_stats()

            model = make_mlp(in_dim, hidden, out_dim, depth).to(self.device)
            example_inputs = (torch.randn(batch, in_dim, device=self.device),)
            ep = torch.export.export(
                model,
                example_inputs,
            )
            torch._inductor.aoti_compile_and_package(ep)

            del model, example_inputs, ep
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            allocated_memory.append(torch.cuda.memory_allocated())

        self.assertTrue(allocated_memory[1] == allocated_memory[2])

    @unittest.skipIf(IS_MACOS, "might have no readelf on Mac")
    def test_libtorch_free_so(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x, y):
                return x + self.linear(y)

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )

        model = Model().to(self.device)
        ep = torch.export.export(model, example_inputs)

        package_path = torch._inductor.aoti_compile_and_package(
            ep,
            inductor_configs={
                "aot_inductor.link_libtorch": False,
            },
        )

        torch_libs = {
            "libtorch.so",
            "libc10.so",
            "libtorch_cuda.so",
            "libc10_cuda.so",
            "libtorch_cpu.so",
            "libtorch_xpu.so",
            "libc10_xpu.so",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # Unpack
            with zipfile.ZipFile(package_path, "r") as zf:
                zf.extractall(tmpdir)

            so_files = list(pathlib.Path(tmpdir).rglob("*.so"))
            self.assertTrue(len(so_files) > 0)

            for so_file in so_files:
                so_copy = pathlib.Path(tmpdir) / f"{so_file.name}.checkcopy"
                so_copy.write_bytes(so_file.read_bytes())

                result = subprocess.run(
                    ["readelf", "-d", str(so_copy)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                for line in result.stdout.splitlines():
                    if "NEEDED" in line:
                        for lib in torch_libs:
                            self.assertTrue(lib not in line)

    def test_unbounded_expr_substitutions(self):
        class Model(torch.nn.Module):
            def forward(self, x, y, a, b):
                u0, s0 = a.item(), b.item()
                u_max = max(u0, 15)
                # construct the equality rule Max(15, u0) == s0 * Max(15, u0)
                torch._check(u_max == s0 * u_max)
                # size x - [Max(u0, 15), 64]
                x = x.expand(u_max, *x.shape).clone()
                return x @ y

        model = Model()

        example_inputs = (
            torch.randn((64,), dtype=torch.bfloat16, device=self.device),
            torch.randn((64, 16), dtype=torch.bfloat16, device=self.device),
            torch.tensor(19, device=self.device),
            torch.tensor(1, device=self.device),
        )
        torch._dynamo.mark_dynamic(example_inputs[-1], 0)

        so_path, code = run_and_get_cpp_code(
            AOTIRunnerUtil.legacy_compile, model, example_inputs
        )

        compiled = AOTIRunnerUtil.legacy_load(self.device, so_path)
        compiled_outputs = compiled(*example_inputs)

        eager_outputs = model(*example_inputs)
        torch.testing.assert_close(eager_outputs, compiled_outputs)

    @requires_gpu
    def test_mixed_device_1(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("Mixed-device test requires GPU")

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Buffers are on CPU
                self.register_buffer(
                    "index", torch.tensor([1, 4, 1, 7], device="cpu", dtype=torch.int64)
                )
                self.register_buffer(
                    "src", torch.ones(4, device="cpu", dtype=torch.int64)
                )

            def forward(self, matrix, vector):
                # Inputs are on CUDA
                # 1. Operation on CPU tensors
                z = torch.zeros((vector.shape[0],), device="cpu", dtype=torch.int64)
                scatter_result = z.scatter_add(0, self.index, self.src)

                # 2. Move result to CUDA and continue on CUDA
                v = vector + scatter_result.to(vector.dtype).to(GPU_TYPE)
                return torch.matmul(matrix, v)

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, device=self.device),
        )
        self.check_model(Model(), example_inputs, move_model_to_device=False)


class AOTInductorLoggingTest(LoggingTestCase):
    @make_logging_test(dynamic=logging.DEBUG)
    def test_shape_env_reuse(self, records):
        # make sure ShapeEnv is only created once and reused afterwards
        class Foo(torch.nn.Module):
            def forward(self, x):
                return x + 2

        inputs = (torch.randn(4, 4),)
        dynamic_shapes = {
            "x": {0: Dim.AUTO, 1: Dim.AUTO},
        }
        ep = export(Foo(), inputs, dynamic_shapes=dynamic_shapes, strict=False)
        with torch.no_grad():
            torch._inductor.aot_compile(ep.module(), inputs)
        self.assertEqual([r.msg == "create_env" for r in records].count(True), 1)

    @make_logging_test(dynamic=logging.DEBUG)
    def test_shape_env_reuse_zero_consts_use_consts_asm_false(self, records):
        # make sure ShapeEnv is only created once and reused afterwards
        class Foo(torch.nn.Module):
            def forward(self, x):
                return x + 2

        inputs = (torch.randn(4, 4),)
        dynamic_shapes = {
            "x": {0: Dim.AUTO, 1: Dim.AUTO},
        }
        ep = export(Foo(), inputs, dynamic_shapes=dynamic_shapes, strict=False)
        with (
            torch.no_grad(),
            config.patch({"aot_inductor.use_consts_asm_build": False}),
        ):
            torch._inductor.aot_compile(ep.module(), inputs)
        self.assertEqual([r.msg == "create_env" for r in records].count(True), 1)


class TestAOTInductorConfig(TestCase):
    def test_no_compile_standalone(self):
        with config.patch({"aot_inductor_mode.compile_standalone": False}):
            result = maybe_aoti_standalone_config({})
            self.assertEqual(result, {})

    def test_compile_standalone_sets_package_cpp(self):
        result = maybe_aoti_standalone_config(
            {"aot_inductor_mode.compile_standalone": True}
        )
        self.assertEqual(result["aot_inductor.package_cpp_only"], True)
        self.assertEqual(result["aot_inductor_mode.compile_standalone"], True)
        self.assertEqual(result["aot_inductor.embed_kernel_binary"], True)
        self.assertEqual(
            result["aot_inductor.emit_multi_arch_kernel"], not torch.version.hip
        )
        self.assertEqual(
            result["aot_inductor.model_name_for_generated_files"], "aoti_model"
        )
        self.assertEqual(result["aot_inductor.dynamic_linkage"], False)

    def test_compile_standalone_explicit_set(self):
        patches = {
            "aot_inductor_mode.compile_standalone": True,
            "aot_inductor.package_cpp_only": True,
            "aot_inductor.embed_kernel_binary": True,
            "aot_inductor.dynamic_linkage": False,
            "aot_inductor.link_libtorch": False,
            "aot_inductor.emit_multi_arch_kernel": not torch.version.hip,
            "aot_inductor.model_name_for_generated_files": "aoti_model",
        }
        result = maybe_aoti_standalone_config(patches)
        self.assertEqual(result, patches)

    def test_compile_standalone_package_cpp_false_raises(self):
        patches = {
            "aot_inductor_mode.compile_standalone": True,
            "aot_inductor.package_cpp_only": False,
        }
        with self.assertRaises(RuntimeError):
            maybe_aoti_standalone_config(patches)

        with config.patch({"aot_inductor.package_cpp_only": False}):
            patches = {
                "aot_inductor_mode.compile_standalone": True,
            }
            with self.assertRaises(RuntimeError):
                maybe_aoti_standalone_config(patches)

    def test_compile_standalone_cross_compile_windows_package_format(self):
        patches = {
            "aot_inductor.cross_target_platform": "windows",
            "aot_inductor.package_constants_in_so": True,
        }
        with self.assertRaises(RuntimeError):
            maybe_aoti_standalone_config(patches)


common_utils.instantiate_parametrized_tests(AOTInductorTestsTemplate)


def fail_cpu(is_skip=False):
    return TestFailure(
        ("cpu",),
        is_skip=is_skip,
    )


def fail_mps(is_skip=False):
    return TestFailure(
        ("mps",),
        is_skip=is_skip,
    )


def fail_gpu(suffixes: tuple[str, ...], is_skip=False):
    return TestFailure(
        suffixes,
        is_skip=is_skip,
    )


# test_failures, xfail by default, set is_skip=True to skip
CPU_TEST_FAILURES = {
    # TODO: failed internally
    "test_multiple_output_alias": fail_cpu(is_skip=True),
}

# test_failures, xfail by default, set is_skip=True to skip
GPU_TEST_FAILURES = {
    # quantized unsupported for GPU
    "test_quantized_linear": fail_gpu(("cuda", "xpu")),
    "test_quanatized_int8_linear": fail_gpu(("cuda", "xpu")),
    "test_quantized_linear_bias_none": fail_gpu(("cuda", "xpu")),
    # No scaled_dot_product_efficient_attention implementation for XPU yet.
    "test_scaled_dot_product_efficient_attention": fail_gpu(("xpu",)),
}

MPS_TEST_FAILURES = {
    # aten::_scaled_dot_product_efficient_attention is not currently implemented for the MPS device.
    "test_scaled_dot_product_efficient_attention": fail_mps(),
    # aten::_int_mm is not implemented for MPS backend
    "test__int_mm": fail_mps(),
    # MPS doesn't support float64
    "test_while_loop_with_conv_dynamic_True": fail_mps(),
    "test_while_loop_with_conv_dynamic_False": fail_mps(),
    # MPS doesn't support float8
    "test_fp8": fail_mps(),
    "test_fp8_view_of_param": fail_mps(),
    # cannot initialize a parameter of type 'double' with an rvalue of type 'std::nullptr_t'
    "test_fallback_kernel_with_symexpr_output": fail_mps(),
    # correctness issue
    "test_index_put_with_none_index": fail_mps(),
    # Error device may not be nil
    "test_zero_size_weight": fail_mps(is_skip=True),
    # MPSGraph does not support tensor dims > INT_MAX
    "test_upper_bound_i64": fail_mps(is_skip=True),
    # MPS doesn't support triton
    "test_autotuning_args_reuse": fail_mps(),
    "test_triton_autotuning": fail_mps(),
    "test_triton_dynamic_launcher_grid": fail_mps(),
    "test_triton_dynamic_launcher_grid_infer_from_tensor": fail_mps(),
    "test_triton_kernel_on_device_tma_dynamic_False_tma_version_new": fail_mps(),
    "test_triton_kernel_on_device_tma_dynamic_False_tma_version_old": fail_mps(),
    "test_triton_kernel_on_device_tma_dynamic_True_tma_version_new": fail_mps(),
    "test_triton_kernel_on_device_tma_dynamic_True_tma_version_old": fail_mps(),
    "test_size_with_unbacked_add_expr_transitive": fail_mps(),
    "test_size_with_unbacked_add_and_mul_expr": fail_mps(),
    "test_triton_next_power_of_2": fail_mps(),
    "test_sympy_cpp_printer_min_max_minmax0": fail_mps(),
    "test_sympy_cpp_printer_min_max_minmax1": fail_mps(),
    "test_triton_kernel_dynamic_shape_with_div": fail_mps(),
    "test_triton_kernel_reinterpret_view": fail_mps(),
    "test_triton_kernel_tma_descriptor_1d_dynamic_False_tma_version_new_mps": fail_mps(),
    "test_triton_kernel_tma_descriptor_1d_dynamic_False_tma_version_old_mps": fail_mps(),
    "test_triton_kernel_tma_descriptor_1d_dynamic_True_tma_version_new_mps": fail_mps(),
    "test_triton_kernel_tma_descriptor_1d_dynamic_True_tma_version_old_mps": fail_mps(),
    "test_triton_kernel_tma_descriptor_2d_dynamic_False_tma_version_new_mps": fail_mps(),
    "test_triton_kernel_tma_descriptor_2d_dynamic_False_tma_version_old_mps": fail_mps(),
    "test_triton_kernel_tma_descriptor_2d_dynamic_True_tma_version_new_mps": fail_mps(),
    "test_triton_kernel_tma_descriptor_2d_dynamic_True_tma_version_old_mps": fail_mps(),
    "test_triton_kernel_sympy_expr_arg": fail_mps(),
    "test_triton_kernel_sympy_fn_like_arg": fail_mps(),
    "test_triton_kernel_with_none_input": fail_mps(),
    "test_triton_kernel_equal_to_1_arg": fail_mps(),
    "test_triton_kernel_with_none_inputs_and_equal_to_1_arg": fail_mps(),
    "test_triton_kernel_equal_to_1_float_arg_dynamic_True": fail_mps(),
    "test_triton_kernel_equal_to_1_float_arg_dynamic_False": fail_mps(),
    "test_triton_kernel_weird_param_order": fail_mps(),
    "test_triton_kernel_dynamic_grid": fail_mps(),
    "test_repeated_user_defined_triton_kernel_embed_kernel_binary_False": fail_mps(),
    "test_repeated_user_defined_triton_kernel_embed_kernel_binary_True": fail_mps(),
    "test_triton_kernel_extern_kernel_arg": fail_mps(),
    "test_triton_kernel_multi_output_arg": fail_mps(),
    "test_triton_kernel_reinterpret_view_mem_leak": fail_mps(),
    "test_triton_mutated_autotuning": fail_mps(),
    "test_sym_i64_input_codegen": fail_mps(),
    "test_none_args_aot_codegen": fail_mps(),
    "test_aoti_debug_printer_sym_inputs": fail_mps(),
    "test_aoti_debug_printer_user_defined_triton_kernel": fail_mps(),
    "test_autotune_int64_user_defined_triton_kernel": fail_mps(),
}


class AOTInductorTestABICompatibleCpu(TestCase):
    device = "cpu"
    device_type = "cpu"
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs
    code_check_count = code_check_count
    allow_stack_allocation = False
    use_minimal_arrayref_interface = False


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestABICompatibleCpu,
    "cpu",
    CPU_TEST_FAILURES,
)


@unittest.skipIf(sys.platform == "darwin", "No CUDA on MacOS")
class AOTInductorTestABICompatibleGpu(TestCase):
    device = GPU_TYPE
    device_type = GPU_TYPE
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs
    code_check_count = code_check_count
    allow_stack_allocation = False
    use_minimal_arrayref_interface = False


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestABICompatibleGpu,
    GPU_TYPE,
    GPU_TEST_FAILURES,
)


@unittest.skipIf(not torch.backends.mps.is_available(), "No MPS backend available")
class AOTInductorTestABICompatibleMps(TestCase):
    device = "mps"
    device_type = "mps"
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs
    code_check_count = code_check_count
    allow_stack_allocation = False
    use_minimal_arrayref_interface = False


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestABICompatibleMps,
    "mps",
    MPS_TEST_FAILURES,
)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    # cpp_extension N/A in fbcode
    if HAS_GPU or sys.platform == "darwin":
        run_tests(needs="filelock")
