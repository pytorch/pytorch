# Owner(s): ["module: inductor"]
# ruff: noqa: F841
import itertools
import logging
import os
import sys
import tempfile
import unittest
from typing import Dict, Tuple
from unittest import skip

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
from torch._inductor.exc import CppWrapperCodegenError
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_cpp_code
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torch.export import Dim, export, export_for_training
from torch.testing import FileCheck
from torch.testing._internal import common_utils
from torch.testing._internal.common_cuda import SM80OrLater, SM90OrLater
from torch.testing._internal.common_device_type import (
    _has_sufficient_memory,
    skipCUDAIf,
)
from torch.testing._internal.common_quantization import (
    skip_if_no_torchvision,
    skipIfNoFBGEMM,
)
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    IS_CI,
    IS_FBCODE,
    IS_MACOS,
    IS_WINDOWS,
    skipIfRocm,
    skipIfXpu,
    TEST_WITH_ROCM,
)
from torch.testing._internal.inductor_utils import GPU_TYPE
from torch.testing._internal.logging_utils import LoggingTestCase, make_logging_test
from torch.testing._internal.triton_utils import HAS_GPU, requires_gpu
from torch.utils import _pytree as pytree
from torch.utils._triton import has_triton_tma


if HAS_GPU:
    import triton  # @manual
    from triton import language as tl

    from torch.testing._internal.triton_utils import (
        add_kernel,
        add_kernel_2d_autotuned,
        add_kernel_autotuned,
        add_kernel_autotuned_weird_param_order,
        add_kernel_with_optional_param,
        add_kernel_with_scaling,
        add_kernel_with_tma_1d,
        add_kernel_with_tma_2d,
        mul2_inplace_kernel,
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


class AOTInductorTestsTemplate:
    def test_simple(self):
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
        self.check_model(model, example_inputs)
        if self.use_minimal_arrayref_interface:
            self.code_check_count(
                model, example_inputs, "AOTInductorModelRunMinimalArrayrefInterface(", 1
            )

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
        expected_path = os.path.join(tempfile.mkdtemp(dir=cache_dir()), "model.so")
        actual_path = AOTIRunnerUtil.compile(
            model, example_inputs, options={"aot_inductor.output_path": expected_path}
        )
        self.assertTrue(actual_path == expected_path)

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
                self.check_model(LinearModel(self.device), example_inputs)

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
        self.check_model(Model(), example_inputs)
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
        self.check_model(Model(), example_inputs)

    @skip("Test was marked as expected failure, but does not fail always anymore.")
    def test_dynamic_smem_above_default_limit(self):
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

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
        "FP8 is only supported on H100+",
    )
    @skipIfRocm  # _scaled_mm_out_cuda  is not compiled for ROCm platform
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
                weight = weight.to(torch.float8_e4m3fn)
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
        x = torch.rand(*x_shape, device=GPU_TYPE, dtype=dtype).to(torch.float8_e4m3fn)
        dim0_x = Dim("dim0_x", min=1, max=2048)
        dynamic_shapes = ({0: dim0_x}, None, None, None, None)
        self.check_model(
            Model(dtype),
            (x, weight, input_bias, a_inverse_scale, b_inverse_scale),
            dynamic_shapes=dynamic_shapes,
        )

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
        "FP8 is only supported on H100+",
    )
    @skipIfRocm  # _scaled_mm_out_cuda  is not compiled for ROCm platform
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
            torch.float8_e4m3fn
        )
        a_inverse_scale = 1 / a_scale
        b_inverse_scale = 1 / b_scale

        x_shape = (16, 16)
        x = torch.rand(*x_shape, device=self.device, dtype=dtype).to(
            torch.float8_e4m3fn
        )
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

    def test_addmm_multiple_dynamic(self):
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

    def test_bmm_multiple_dynamic(self):
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
    @unittest.skipIf(IS_FBCODE, "Not yet runnable in fbcode")
    @unittest.skipIf(not SM80OrLater, "bfloat16 only supported in sm80+")
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

    @unittest.skipIf(IS_FBCODE, "Not yet runnable in fbcode")
    @unittest.skipIf(not SM80OrLater, "bfloat16 only supported in sm80+")
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

    @skipIfXpu(msg="_scaled_dot_product_flash_attention is not supported on XPU yet")
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
        dim0_ab = Dim("s0", min=3, max=1024)
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

    def test_while_loop_with_parameters(self):
        inputs = (torch.randn((10, 20), device=self.device),)
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
        so_path: str = AOTIRunnerUtil.compile(
            Repro(),
            example_inputs,
            dynamic_shapes=dynamic_shapes,
        )
        aot_inductor_module = AOTIRunnerUtil.load(self.device, so_path)
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
                kernel_runner = AOTIRunnerUtil.load_runner(self.device, so_path)
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
    @skipIfXpu
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
        with config.patch(
            {"freezing": True, "aot_inductor.force_mmap_weights": True}
        ), torch.no_grad():
            exported_model = export_for_training(model, example_inputs).module()
            quantizer = X86InductorQuantizer()
            quantizer.set_global(
                xiq.get_default_x86_inductor_quantization_config(reduce_range=True)
            )
            prepared_model = prepare_pt2e(exported_model, quantizer)
            prepared_model(*example_inputs)
            converted_model = convert_pt2e(prepared_model)
            torch.ao.quantization.move_exported_model_to_eval(converted_model)

            self.check_model(converted_model, example_inputs)

    @requires_multigpu()
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
            so_path = AOTIRunnerUtil.compile(
                model=Model(
                    w1.to(torch.device(GPU_TYPE, 0)), w2.to(torch.device(GPU_TYPE, 0))
                ),
                example_inputs=tuple(t.to(torch.device(GPU_TYPE, 0)) for t in inputs),
            )

        # Run model on gpu:N
        for i in range(device_interface.device_count()):
            with device_interface.device(i):
                example_inputs = tuple(t.to(torch.device(GPU_TYPE, i)) for t in inputs)
                optimized = AOTIRunnerUtil.load(GPU_TYPE, so_path)
                result_gpu = optimized(*example_inputs)
            self.assertTrue(same(result_cpu, result_gpu.cpu()))

    @requires_multigpu()
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

        so_path = AOTIRunnerUtil.compile(model, example_inputs)
        optimized = AOTIRunnerUtil.load(device, so_path)
        actual = optimized(*example_inputs)
        torch.testing.assert_close(actual, expected)

    def test_pytree_inputs(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: Dict[str, torch.Tensor]):
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
                GPU_TYPE,
                Model(weight.to(torch.device(GPU_TYPE, 0))),
                tuple(t.to(torch.device(GPU_TYPE, 0)) for t in inputs),
            )

        with device_interface.device(1), torch.no_grad():
            result_gpu_1 = AOTIRunnerUtil.run(
                GPU_TYPE,
                Model(weight.to(torch.device(GPU_TYPE, 1))),
                tuple(t.to(torch.device(GPU_TYPE, 1)) for t in inputs),
            )

        self.assertTrue(same(result_cpu, result_gpu_0.cpu()))
        self.assertTrue(same(result_cpu, result_gpu_1.cpu()))

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
        self.check_model(
            model, example_inputs, atol=1e-4, rtol=1e-4
        )  # 1e-4 is the tol value used in pytorch/torch/_dynamo/utils.py

        if self.device == GPU_TYPE:
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
        m = Model(self.device).to(dtype=torch.float)
        self.check_model(m, example_inputs, dynamic_shapes=dynamic_shapes)

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
        exported_program = export(Model(), example_inputs)

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

    @unittest.mock.patch("torch._inductor.graph.supported_dtype_of_cpp_wrapper")
    def test_unsupported_input_dtype(self, supported_dtype_of_cpp_wrapper_mock):
        supported_dtype_of_cpp_wrapper_mock.return_value = False

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                return x + y

        example_inputs = (
            torch.randn(10, 10).to(self.device),
            torch.randn(10, 10).to(self.device),
        )
        with self.assertRaisesRegex(
            CppWrapperCodegenError, "Unsupported input dtype torch.float32"
        ):
            torch._export.aot_compile(Model(), example_inputs)

        supported_dtype_of_cpp_wrapper_mock.assert_called_once_with(
            torch.float32, self.device
        )

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
            so_path = AOTIRunnerUtil.compile(m, example_inputs)

        # run under grad enabled
        self.assertTrue(torch.is_grad_enabled())

        optimized = AOTIRunnerUtil.load(self.device, so_path)
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
        model = Model().to(device=self.device, dtype=torch.float64).eval()
        example_inputs = (
            torch.randn(4, 3, 64, 64, device=self.device, dtype=torch.float64),
        )
        self.check_model(model, example_inputs)

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
    def test_triton_kernel_tma_descriptor_1d(self, dynamic):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")
        if not has_triton_tma():
            raise unittest.SkipTest("requires Triton TMA")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, a, b):
                BLOCK_SIZE = 256
                out = torch.zeros_like(a)
                n_elements = out.numel()

                desc_a, desc_b, desc_out = (
                    triton.tools.experimental_descriptor.create_1d_tma_descriptor(
                        t.data_ptr(),
                        n_elements,
                        BLOCK_SIZE,
                        t.element_size(),
                    )
                    for t in (a, b, out)
                )

                grid = lambda meta: (  # noqa: E731
                    triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
                )
                add_kernel_with_tma_1d[grid](
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
    def test_triton_kernel_tma_descriptor_2d(self, dynamic):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")
        if not has_triton_tma():
            raise unittest.SkipTest("requires Triton TMA")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, a, b):
                BLOCK_SIZE_X = 16
                BLOCK_SIZE_Y = 32
                out = torch.zeros_like(a)
                x_size, y_size = out.size()

                desc_a, desc_b, desc_out = (
                    triton.tools.experimental_descriptor.create_2d_tma_descriptor(
                        t.data_ptr(),
                        x_size,
                        y_size,
                        BLOCK_SIZE_X,
                        BLOCK_SIZE_Y,
                        t.element_size(),
                    )
                    for t in (a, b, out)
                )

                grid = lambda meta: (  # noqa: E731
                    triton.cdiv(x_size, meta["BLOCK_SIZE_X"]),
                    triton.cdiv(y_size, meta["BLOCK_SIZE_Y"]),
                )
                add_kernel_with_tma_2d[grid](
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
                "x": {0: dim0_xy, 1: None},
                "y": {0: dim0_xy, 1: None},
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
                    indices: Tuple[torch.Tensor],
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

    def test_repeated_user_defined_triton_kernel(self):
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
        self.check_model(Model(), inputs)

    def test_convolution(self):
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
            so_path = AOTIRunnerUtil.compile(
                model=TestModule().to(device=self.device),
                example_inputs=(torch.rand(3, 4, device=self.device),),
            )

        runner = AOTIRunnerUtil.load_runner(self.device, so_path)

        expected_original_fqns = {
            "L__self___test_param": "test_param",
            "L__self___test_buf": "test_buf",
            "getattr_L__self___foo_bar___0__": "foo_bar.0",
            "L__self___foo_bar_test_param": "foo_bar.test_param",
            "L__self___foo_bar_test_buf": "foo_bar.test_buf",
        }
        self.assertEqual(
            expected_original_fqns, runner.get_constant_names_to_original_fqns()
        )

        expected_dtypes = {
            "L__self___test_param": 6,
            "L__self___test_buf": 6,
            "getattr_L__self___foo_bar___0__": 6,
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

    # @skipIfXpu(msg="torch.xpu.memory_allocated not supported yet")
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

        so_path: str = AOTIRunnerUtil.compile(
            Model(),
            example_inputs,
        )
        aot_inductor_module = AOTIRunnerUtil.load(GPU_TYPE, so_path)

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

    def test_runtime_checks(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            if SM80OrLater:

                def forward(self, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9):
                    return (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9)

            else:

                def forward(self, x0, x1, x2, x4, x5, x6, x7, x8, x9):
                    return (x0, x1, x2, x4, x5, x6, x7, x8, x9)

        inputs = []
        dtypes = [
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bool,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ]
        if SM80OrLater:
            dtypes.append(torch.bfloat16)
        for dtype in dtypes:
            inputs.append(torch.ones(4, 8, 10, dtype=dtype, device=self.device))

        dim0 = Dim("s0", min=2, max=1024)
        dim1 = Dim("s1", min=2, max=512)
        dim2 = Dim("s2", min=2, max=128)
        dynamic_shapes = {
            "x0": {0: dim0},
            "x1": {0: dim0},
            "x2": {0: dim0},
            "x4": {1: dim1},
            "x5": {1: dim1},
            "x6": {},
            "x7": {2: dim2},
            "x8": {2: dim2},
            "x9": {2: dim2},
        }
        if SM80OrLater:
            dynamic_shapes["x3"] = {1: dim1}

        m = Model()
        inputs = tuple(inputs)
        with torch.no_grad(), config.patch(
            {
                "aot_inductor.debug_compile": True,
            }
        ):
            so_path = AOTIRunnerUtil.compile(m, inputs, dynamic_shapes=dynamic_shapes)
        with open(os.path.splitext(so_path)[0] + ".cpp") as cpp:
            src_code = cpp.read()
            FileCheck().check_count(
                "unmatched dtype",
                10 if SM80OrLater else 9,
                exactly=True,
            ).run(src_code)
            FileCheck().check_count(
                "unmatched dim value at",
                21
                if SM80OrLater
                else 19,  # we have 9 dynamic dims for which we generate different checks
                exactly=True,
            ).run(src_code)
            FileCheck().check_count(
                "dim value is too",
                18
                if SM80OrLater
                else 16,  # we have 9 dynamic dims for which we generate two checks
                exactly=True,
            ).run(src_code)
            FileCheck().check_count(
                "unmatched stride value at",
                21
                if SM80OrLater
                else 19,  # we have 9 symbolic strides for which we don't generate checks
                exactly=True,
            ).run(src_code)
        optimized = AOTIRunnerUtil.load(self.device, so_path)
        actual = optimized(*inputs)
        expected = m(*inputs)
        torch.testing.assert_close(actual, expected)

    @unittest.skipIf(TEST_WITH_ROCM, "FP8 is not supported on ROCM")
    @unittest.skipIf(not SM90OrLater, "FP8 is only supported on H100+")
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
        with torch.no_grad(), config.patch(
            {
                "aot_inductor.debug_compile": True,
            }
        ):
            self.check_model(
                Model(),
                tuple(inputs),
                dynamic_shapes=dynamic_shapes,
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
        with torch.no_grad(), config.patch(
            {
                "aot_inductor.debug_compile": True,
            }
        ):
            self.check_model(
                Model(),
                tuple(inputs),
                dynamic_shapes=dynamic_shapes,
            )

    @unittest.skipIf(IS_FBCODE, "Not yet runnable in fbcode")
    def test_runtime_checks_dtype_failed(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                y = x.type(torch.float)
                return y

        x = torch.randn(1, 4, dtype=torch.float16, device=self.device)
        model = Model()
        with torch.no_grad(), config.patch(
            {
                "aot_inductor.debug_compile": True,
            }
        ):
            so_path: str = AOTIRunnerUtil.compile(
                model,
                (x,),
            )
        aot_inductor_module = AOTIRunnerUtil.load(self.device, so_path)
        x_casted = x.float()
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
                self.device,
                model,
                (x,),
            )
        actual = model(x)
        self.assertTrue(same(result, actual))

        # contiguous() should create a new tensor
        self.assertTrue(result[0].data_ptr() != result[1].data_ptr())

    def test_multiple_output_alias(self):
        # Test when mutliple outputs alias the same tensor
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
                self.device,
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
        with torch.no_grad(), config.patch(
            {
                "aot_inductor.debug_compile": True,
            }
        ):
            so_path: str = AOTIRunnerUtil.compile(
                model, (x,), dynamic_shapes=dynamic_shapes
            )
        aot_inductor_module = AOTIRunnerUtil.load(self.device, so_path)
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

        self.check_model_with_multiple_inputs(
            model, example_inputs_list, dynamic_shapes=dynamic_shapes
        )

    @common_utils.parametrize("max_autotune", [True, False])
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

        kernel_calls = (
            [
                ("triton_poi_fused_0", 1),
                (f"aoti_torch_{GPU_TYPE}_addmm_out", 2),
            ]
            if self.device == GPU_TYPE
            else [
                ("aoti_torch_cpu_addmm_out", 2),
            ]
        )

        # test default debug printing all tensor values codegen
        with config.patch({"aot_inductor.debug_intermediate_value_printer": "2"}):
            result, code = run_and_get_cpp_code(
                AOTIRunnerUtil.compile, model, example_inputs
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
                AOTIRunnerUtil.compile, model, example_inputs
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
            "triton_poi_fused_1_xnumel",
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

    def test_aoti_debug_printing_model_inputs_codegen(self):
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b, c):
                x = a * 3.14
                y = torch.addmm(c, x, b)
                z = torch.nn.functional.gelu(y)
                return z

        example_inputs = (
            torch.randn(10, 20, device="cuda"),
            torch.randn(20, 30, device="cuda"),
            torch.randn(10, 30, device="cuda"),
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
            "int64_t var_1 = u0;",
            "int64_t var_3 = u0;",
            "int64_t var_5 = u0;",
            "int64_t var_9 = u0;",
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
        with torch.no_grad(), config.patch(
            {
                "always_keep_tensor_constants": True,
                "aot_inductor.package_constants_in_so": True,
            }
        ):
            so_path = AOTIRunnerUtil.compile(
                model=model,
                example_inputs=example_inputs,
            )

        with torch.no_grad(), config.patch(
            {
                "always_keep_tensor_constants": True,
                "aot_inductor.package_constants_in_so": False,
            }
        ):
            so_path_weightless = AOTIRunnerUtil.compile(
                model=model,
                example_inputs=example_inputs,
            )
        self.assertTrue(os.path.getsize(so_path) > 10_000_000)
        self.assertTrue(os.path.getsize(so_path_weightless) < 10_000_000)

        runner = AOTIRunnerUtil.load_runner(self.device, so_path_weightless)

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
        self.assertEqual(expected, output)

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
        with torch.no_grad(), config.patch({"always_keep_tensor_constants": True}):
            so_path = AOTIRunnerUtil.compile(
                model=model,
                example_inputs=example_inputs,
            )

        runner = AOTIRunnerUtil.load_runner(self.device, so_path)

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

    def test_misaligned_input_1(self):
        if self.device != "cuda":
            raise unittest.SkipTest("CUDA test only")

        class Model(torch.nn.Module):
            def forward(self, x):
                return x.sin() + x.cos()

        N = 64 * 64 * 64 + 64
        arg = torch.randn(N, device=self.device)
        example_inputs = (arg,)
        model = Model()
        expected = model(*example_inputs)
        so_path = AOTIRunnerUtil.compile(model, example_inputs)
        optimized = AOTIRunnerUtil.load(self.device, so_path)

        misaligned_arg = torch.zeros(N + 1, device=self.device)
        misaligned_arg = misaligned_arg[1:]
        misaligned_arg.copy_(arg)
        # If the model is compiled with aligned inputs, the generated
        # code will check inputs alignment at runtime, and throws an
        # error if any alignment assumption is violated.
        msg = ".* API call failed at .*"
        with self.assertRaisesRegex(RuntimeError, msg):
            actual = optimized(misaligned_arg)
            torch.testing.assert_close(actual, expected)

    def test_misaligned_input_2(self):
        if self.device != "cuda":
            raise unittest.SkipTest("CUDA test only")

        class Model(torch.nn.Module):
            def forward(self, x):
                return x.sin() + x.cos()

        N = 64 * 64 * 64 + 64
        arg = torch.randn(N, device=self.device)
        misaligned_arg = torch.zeros(N + 1, device=self.device)
        misaligned_arg = misaligned_arg[1:]
        misaligned_arg.copy_(arg)
        example_inputs = (misaligned_arg,)
        # If the model is already compiled with a misaligned input, the
        # generated code should NOT contain an alignment check for that input.
        self.check_model(Model(), example_inputs)

    def test_conv3d(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

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


common_utils.instantiate_parametrized_tests(AOTInductorTestsTemplate)


def fail_cpu(is_skip=False):
    return TestFailure(
        ("cpu",),
        is_skip=is_skip,
    )


def fail_gpu(suffixes: Tuple[str, ...], is_skip=False):
    return TestFailure(
        suffixes,
        is_skip=is_skip,
    )


# test_failures, xfail by default, set is_skip=True to skip
CPU_TEST_FAILURES = {
    # TODO: failed internally
    "test_multiple_output_alias": fail_cpu(is_skip=True),
    "test_update_constant_buffer": fail_cpu(is_skip=True),
    "test_so_without_weight": fail_cpu(is_skip=True),
}

# test_failures, xfail by default, set is_skip=True to skip
GPU_TEST_FAILURES = {
    # quantized unsupported for GPU
    "test_quantized_linear": fail_gpu(("cuda", "xpu")),
    "test_quanatized_int8_linear": fail_gpu(("cuda", "xpu")),
    # No fft implementation for XPU yet.
    "test_fft_c2c": fail_gpu(("xpu",)),
    # No scaled_dot_product_efficient_attention implementation for XPU yet.
    "test_scaled_dot_product_efficient_attention": fail_gpu(("xpu",)),
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

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    # cpp_extension N/A in fbcode
    if HAS_GPU or sys.platform == "darwin":
        run_tests(needs="filelock")
