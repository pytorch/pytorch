# Owner(s): ["module: inductor"]
import copy
import itertools
import os
import sys
import tempfile
import types
import unittest
from typing import Dict, Tuple
from unittest import skip

import torch
import torch._export
import torch._inductor
import torch._inductor.config
import torch.nn as nn
from torch._dynamo.testing import rand_strided, same
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.exc import CppWrapperCodeGenError
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.test_case import TestCase
from torch.export import Dim, export
from torch.testing import FileCheck
from torch.testing._internal import common_utils
from torch.testing._internal.common_cuda import SM80OrLater, SM90OrLater
from torch.testing._internal.common_quantization import (
    skip_if_no_torchvision,
    skipIfNoFBGEMM,
)
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    find_library_location,
    IS_CI,
    IS_FBCODE,
    IS_MACOS,
    IS_SANDCASTLE,
    IS_WINDOWS,
    skipIfRocm,
    TEST_WITH_ROCM,
)
from torch.testing._internal.triton_utils import HAS_CUDA, requires_cuda
from torch.utils import _pytree as pytree


if HAS_CUDA:
    import triton

    from torch.testing._internal.triton_utils import (
        add_kernel,
        add_kernel_2d_autotuned,
        add_kernel_autotuned,
        add_kernel_with_optional_param,
        add_kernel_with_scaling,
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
        from .test_aot_inductor_utils import AOTIRunnerUtil
        from .test_control_flow import (
            CondModels,
            prepend_counters,
            prepend_predicates,
            WhileLoopModels,
        )
        from .test_torchinductor import copy_tests, requires_multigpu, TestFailure
    except ImportError:
        from test_aot_inductor_utils import AOTIRunnerUtil
        from test_control_flow import (
            CondModels,
            prepend_counters,
            prepend_predicates,
            WhileLoopModels,
        )
        from test_torchinductor import copy_tests, requires_multigpu, TestFailure
except (unittest.SkipTest, ImportError) as e:
    if __name__ == "__main__":
        sys.exit(0)
    raise


def check_model(
    self: TestCase,
    model,
    example_inputs,
    options=None,
    dynamic_shapes=None,
    disable_constraint_solver=False,
    atol=None,
    rtol=None,
):
    with torch.no_grad(), config.patch(
        {
            "abi_compatible": self.abi_compatible,
            "allow_stack_allocation": self.allow_stack_allocation,
            "use_minimal_arrayref_interface": self.use_minimal_arrayref_interface,
        }
    ):
        torch.manual_seed(0)
        if not isinstance(model, types.FunctionType):
            model = model.to(self.device)
        ref_model = copy.deepcopy(model)
        ref_inputs = copy.deepcopy(example_inputs)
        expected = ref_model(*ref_inputs)

        torch.manual_seed(0)
        actual = AOTIRunnerUtil.run(
            self.device,
            model,
            example_inputs,
            options,
            dynamic_shapes,
            disable_constraint_solver,
        )

    self.assertEqual(actual, expected, atol=atol, rtol=rtol)


def check_model_with_multiple_inputs(
    self: TestCase,
    model,
    list_example_inputs,
    options=None,
    dynamic_shapes=None,
):
    with torch.no_grad(), config.patch(
        {
            "abi_compatible": self.abi_compatible,
            "allow_stack_allocation": self.allow_stack_allocation,
        }
    ):
        torch.manual_seed(0)
        model = model.to(self.device)
        ref_model = copy.deepcopy(model)
        ref_inputs = copy.deepcopy(list_example_inputs)
        list_expected = [ref_model(*inputs) for inputs in ref_inputs]

        torch.manual_seed(0)
        list_actual = AOTIRunnerUtil.run_multiple(
            self.device, model, list_example_inputs, options, dynamic_shapes
        )

    self.assertTrue(same(list_actual, list_expected))


def code_check_count(
    self: TestCase,
    model,
    example_inputs,
    target_str: str,
    target_count: int,
):
    so_path = torch._export.aot_compile(model, example_inputs)
    with open(os.path.splitext(so_path)[0] + ".cpp") as cpp:
        src_code = cpp.read()
        FileCheck().check_count(
            target_str,
            target_count,
            exactly=True,
        ).run(src_code)


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
        self.check_model(Model(), example_inputs)

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

    @requires_cuda
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

    @requires_cuda
    def test_multi_device(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                x = x + 1
                x = x.cpu()
                x = x + 2
                x = x.cuda()
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
        for dtype, groups in itertools.product([torch.bfloat16, torch.float], [1, 2]):
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
        if torch._C._has_mkldnn and torch.ops.onednn._is_onednn_bf16_supported():
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
        for dtype in [torch.float32, torch.bfloat16]:

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
        if self.device == "cuda":
            ctx = torch.cuda.amp.autocast
        elif self.device == "cpu":
            ctx = torch.cpu.amp.autocast
        else:
            raise AssertionError("Unsupported device")

        with config.patch({"fallback_random": True}):
            with ctx():
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
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

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
    def test_fp8(self):
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

        a_scale = torch.Tensor([1.0]).to(device="cuda")
        b_scale = torch.Tensor([1.0]).to(device="cuda")
        input_bias = torch.rand(32, device="cuda", dtype=dtype)
        weight_shape = (32, 16)
        weight = torch.rand(*weight_shape, device="cuda", dtype=dtype).T
        a_inverse_scale = 1 / a_scale
        b_inverse_scale = 1 / b_scale

        x_shape = (16, 16)
        x = torch.rand(*x_shape, device="cuda", dtype=dtype).to(torch.float8_e4m3fn)
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
    def test_fp8_view_of_param(self):
        # cuda only
        if self.device != "cuda":
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

    def test_large_grid(self):
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

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

    def test_zero_grid_with_backed_symbols(self):
        class Repro(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, b):
                return x + b

        example_inputs = (
            x := torch.randn((3, 2), device=self.device),
            torch.randn((1, 2), device=self.device),
        )
        torch._dynamo.mark_dynamic(x, index=0)  # Create dynamic symbol

        # Compile & run model where dynamic dim size > 0.
        so_path: str = AOTIRunnerUtil.compile(
            Repro(),
            example_inputs,
        )
        aot_inductor_module = AOTIRunnerUtil.load("cuda", so_path)
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
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer(
                    "_tensor_constant0",
                    torch.randint(1, size=[38], dtype=torch.int64, device="cpu"),
                )

            def forward(self, x):
                return x + self._tensor_constant0.to(torch.device(type="cuda", index=0))

        example_inputs = (
            torch.randint(1, size=[38], dtype=torch.int64, device="cuda"),
        )
        torch._export.aot_compile(Model(), example_inputs)

    @requires_multigpu()
    def test_replicate_on_devices(self):
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

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
        with torch.cuda.device(0), config.patch("abi_compatible", self.abi_compatible):
            so_path = AOTIRunnerUtil.compile(
                model=Model(w1.cuda(0), w2.cuda(0)),
                example_inputs=tuple(t.cuda(0) for t in inputs),
            )

        # Run model on cuda:N
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                example_inputs = tuple(t.cuda(i) for t in inputs)
                optimized = AOTIRunnerUtil.load("cuda", so_path)
                result_cuda = optimized(*example_inputs)
            self.assertTrue(same(result_cpu, result_cuda.cpu()))

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
    def test_non_default_cuda_device(self):
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        class Model(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight

            def forward(self, x, y):
                return x + torch.nn.functional.linear(y, self.weight)

        weight = torch.randn(10, 10)
        inputs = (torch.randn(10, 10), torch.randn(10, 10))
        result_cpu = Model(weight)(*inputs)

        with torch.cuda.device(0), torch.no_grad(), config.patch(
            "abi_compatible", self.abi_compatible
        ):
            result_cuda_0 = AOTIRunnerUtil.run(
                "cuda", Model(weight.cuda(0)), tuple(t.cuda(0) for t in inputs)
            )

        with torch.cuda.device(1), torch.no_grad(), config.patch(
            "abi_compatible", self.abi_compatible
        ):
            result_cuda_1 = AOTIRunnerUtil.run(
                "cuda", Model(weight.cuda(1)), tuple(t.cuda(1) for t in inputs)
            )

        self.assertTrue(same(result_cpu, result_cuda_0.cpu()))
        self.assertTrue(same(result_cpu, result_cuda_1.cpu()))

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

        if self.device == "cuda":
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
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                return x + y

        example_inputs = (torch.randn(10, 10), torch.randn(10, 10))

        # Export on CPU
        exported_program = export(Model(), example_inputs)

        # Compile exported model on CUDA
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
            CppWrapperCodeGenError, "Unsupported input dtype torch.float32"
        ):
            torch._export.aot_compile(Model(), example_inputs)

        supported_dtype_of_cpp_wrapper_mock.assert_called_once_with(
            torch.float32, self.device == "cuda"
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

        self.check_model(Model(), (torch.empty(4, 1, 4, 4),))

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
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

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
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

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
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

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

    def test_triton_kernel_sympy_expr_arg(self):
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

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
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

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
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                n_elements = x.size()[0]
                BLOCK_SIZE = 1024

                output_wo_y = torch.empty_like(x)
                output_with_y = torch.empty_like(x)

                wo_kernel = add_kernel_with_optional_param[(1,)](
                    x,
                    None,
                    output_wo_y,
                    n_elements,
                    ARGS_PASSED="one",
                    BLOCK_SIZE=BLOCK_SIZE,
                )
                with_kernel = add_kernel_with_optional_param[(1,)](
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
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

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
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

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
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

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

        orig_eager = MyModule()

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
        batch = 8
        example_inputs = (torch.randn(2, M, K, device=self.device),)
        model = Model(N, K, self.device)
        self.check_model(model, example_inputs)
        # Update model weights, after this AOTInductor should re-generate model.so
        # if weights are stored in the model.so
        model.weight += 1
        self.check_model(model, example_inputs)

    def test_custom_op_add(self) -> None:
        class M(torch.nn.Module):
            def forward(self, x, y):
                return torch.ops.aoti_custom_ops.custom_add(x, y)

        m = M().to(device=self.device)
        args = (
            torch.randn(3, 3, device=self.device),
            torch.randn(3, 3, device=self.device),
        )
        self.check_model(m, args)

    def test_triton_kernel_extern_kernel_arg(self):
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        class Model(torch.nn.Module):
            def forward(self, x, y):
                out = torch.zeros_like(x)
                # torch.mm is ExternKernelOut
                add_kernel[(4,)](x, torch.mm(x, y), out, 4, 16)
                return out

        example_inputs = (
            torch.randn(4, 4, device="cuda"),
            torch.randn(4, 4, device="cuda"),
        )

        self.check_model(Model(), example_inputs)

    def test_triton_kernel_multi_output_arg(self):
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        class Model(torch.nn.Module):
            def forward(self, x, y):
                out = torch.zeros_like(x)
                # torch.sort creates fallback kernel and hence MultiOutput
                add_kernel[(4,)](x, torch.sort(y).values, out, 4, 16)
                return out

        example_inputs = (
            torch.randn(4, 4, device="cuda"),
            torch.randn(4, 4, device="cuda"),
        )

        self.check_model(Model(), example_inputs)

    @config.patch({"abi_compatible": True})
    def test_triton_kernel_reinterpret_view_mem_leak(self):
        # Check for memory leak when using user-defined Triton Kernel + AOTI.
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

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
            torch.randn(4, 4, device="cuda"),
            torch.randn(1, 16, device="cuda"),
        )

        so_path: str = AOTIRunnerUtil.compile(
            Model(),
            example_inputs,
        )
        aot_inductor_module = AOTIRunnerUtil.load("cuda", so_path)

        # Don't assign outputs to a variable b/c it will allocate GPU memory.
        device: int = torch.cuda.current_device()
        mem_before = torch.cuda.memory_allocated(device)
        aot_inductor_module(*example_inputs)
        aot_inductor_module(*example_inputs)
        mem_after = torch.cuda.memory_allocated(device)
        self.assertEqual(mem_before, mem_after)

        actual = aot_inductor_module(*example_inputs)
        expected = Model()(*example_inputs)
        torch.testing.assert_close(actual, expected)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("autotuning", [False, True])
    def test_triton_kernel_unbacked_symint_in_grid(self, dynamic, autotuning):
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

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
            torch.randn(123, device="cuda"),
            torch.randn(123, device="cuda"),
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

    @skipIfRocm  # USE_MEM_EFF_ATTENTION was not enabled for build.
    def test_scaled_dot_product_efficient_attention(self):
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        class Model(torch.nn.Module):
            def forward(self, q, k, v, attn_bias):
                return torch.ops.aten._scaled_dot_product_efficient_attention(
                    q, k, v, attn_bias, False
                )[0]

        example_inputs = (
            torch.randn(4, 4, 36, 36, device="cuda"),
            torch.randn(4, 4, 36, 36, device="cuda"),
            torch.randn(4, 4, 36, 36, device="cuda"),
            torch.randn(4, 4, 36, 36, device="cuda"),
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

            def forward(self, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9):
                return (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9)

        inputs = []
        for dtype in (
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
            torch.bool,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            inputs.append(torch.ones(4, 8, 10, dtype=dtype, device=self.device))
        dim0 = Dim("s0", min=2, max=1024)
        dim1 = Dim("s1", min=2, max=512)
        dim2 = Dim("s2", min=2, max=128)
        dynamic_shapes = {
            "x0": {0: dim0},
            "x1": {0: dim0},
            "x2": {0: dim0},
            "x3": {1: dim1},
            "x4": {1: dim1},
            "x5": {1: dim1},
            "x6": {},
            "x7": {2: dim2},
            "x8": {2: dim2},
            "x9": {2: dim2},
        }
        m = Model()
        inputs = tuple(inputs)
        with torch.no_grad(), config.patch(
            {
                "abi_compatible": self.abi_compatible,
                "aot_inductor.debug_compile": True,
            }
        ):
            so_path = AOTIRunnerUtil.compile(m, inputs, dynamic_shapes=dynamic_shapes)
        with open(os.path.splitext(so_path)[0] + ".cpp") as cpp:
            src_code = cpp.read()
            FileCheck().check_count(
                "unmatched dtype",
                10,
                exactly=True,
            ).run(src_code)
            FileCheck().check_count(
                "unmatched dim value at",
                21,  # we have 9 dynamic dims for which we generate different checks
                exactly=True,
            ).run(src_code)
            FileCheck().check_count(
                "dim value is too",
                18,  # we have 9 dynamic dims for which we generate two checks
                exactly=True,
            ).run(src_code)
            FileCheck().check_count(
                "unmatched stride value at",
                21,  # we have 9 symbolic strides for which we don't generate checks
                exactly=True,
            ).run(src_code)
        optimized = AOTIRunnerUtil.load(self.device, so_path)
        actual = optimized(*inputs)
        expected = m(*inputs)
        torch.testing.assert_close(actual, expected)

    @unittest.skipIf(TEST_WITH_ROCM, "FP8 is not supported on ROCM")
    @unittest.skipIf(not SM90OrLater, "FP8 is only supported on H100+")
    def test_runtime_checks_fp8(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x0, x1, x2, x3):
                t = (
                    x0.to(torch.float)
                    + x1.to(torch.float)
                    + x2.to(torch.float)
                    + x3.to(torch.float)
                )
                return t

        inputs = []
        for dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2fnuz,
        ):
            inputs.append(torch.ones(8, 8, 8, dtype=dtype, device=self.device))
        dim0 = Dim("s0", min=2, max=1024)
        dynamic_shapes = {
            "x0": {0: dim0},
            "x1": {0: dim0},
            "x2": {0: dim0},
            "x3": {0: dim0},
        }
        with torch.no_grad(), config.patch(
            {
                "abi_compatible": self.abi_compatible,
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
                "abi_compatible": self.abi_compatible,
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
                "abi_compatible": self.abi_compatible,
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
        with torch.no_grad(), config.patch(
            {
                "abi_compatible": self.abi_compatible,
            }
        ):
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

        with torch.no_grad(), config.patch(
            {
                "abi_compatible": self.abi_compatible,
            }
        ):
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
                "abi_compatible": self.abi_compatible,
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

    @common_utils.parametrize("max_autotune", [False, True])
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


common_utils.instantiate_parametrized_tests(AOTInductorTestsTemplate)


class AOTITestCase(TestCase):
    def setUp(self):
        if IS_SANDCASTLE or IS_FBCODE:
            torch.ops.load_library("//caffe2/test/inductor:custom_ops")
        elif IS_MACOS:
            raise unittest.SkipTest("non-portable load_library call used in test")
        else:
            lib_file_path = find_library_location("libaoti_custom_ops.so")
            if IS_WINDOWS:
                lib_file_path = find_library_location("aoti_custom_ops.dll")
            torch.ops.load_library(str(lib_file_path))
        super().setUp()


class AOTInductorTestABICompatibleCpu(AOTITestCase):
    device = "cpu"
    abi_compatible = True
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs
    code_check_count = code_check_count
    allow_stack_allocation = False
    use_minimal_arrayref_interface = False


def fail_with_and_without_stack_allocation(is_skip=False):
    return TestFailure(
        (
            "abi_compatible_cpu",
            "abi_compatible_cpu_with_stack_allocation",
            "abi_compatible_cpu_with_stack_allocation_and_minimal_arrayref_interface",
        ),
        is_skip=is_skip,
    )


def fail_stack_allocation(is_skip=False):
    return TestFailure(
        (
            "abi_compatible_cpu_with_stack_allocation",
            "abi_compatible_cpu_with_stack_allocation_and_minimal_arrayref_interface",
        ),
        is_skip=is_skip,
    )


def fail_minimal_arrayref_interface(is_skip=False):
    return TestFailure(
        ("abi_compatible_cpu_with_stack_allocation_and_minimal_arrayref_interface",),
        is_skip=is_skip,
    )


def fail_cuda(is_skip=False):
    return TestFailure(
        ("abi_compatible_cuda", "non_abi_compatible_cuda"),
        is_skip=is_skip,
    )


def fail_abi_compatible_cuda(is_skip=False):
    return TestFailure(
        ("abi_compatible_cuda",),
        is_skip=is_skip,
    )


def fail_non_abi_compatible_cuda(is_skip=False):
    return TestFailure(
        ("non_abi_compatible_cuda",),
        is_skip=is_skip,
    )


# test_failures, xfail by default, set is_skip=True to skip
CPU_TEST_FAILURES = {
    # TODO: error: ‘complex64’ was not declared in this scope
    "test_add_complex": fail_minimal_arrayref_interface(is_skip=True),
    # TODO: test_conv_freezing_abi_compatible_cpu fails,
    #   AssertionError: None, i.e. optional output is not supported
    "test_conv_freezing": fail_with_and_without_stack_allocation(is_skip=True),
    # TODO: test_deconv_freezing_abi_compatible_cpu fails,
    #   AssertionError: None, i.e. optional output is not supported
    "test_deconv_freezing": fail_with_and_without_stack_allocation(is_skip=True),
    # FIXME: failed with Segfault while exiting the Python runtime
    "test_duplicate_constant_folding": fail_with_and_without_stack_allocation(
        is_skip=True
    ),
    # TODO: use of deleted function RAIIAtenTensorHandle
    "test_dup_unbacked_sym_decl": fail_minimal_arrayref_interface(is_skip=True),
    # TODO: use of deleted function RAIIAtenTensorHandle
    "test_dup_unbacked_sym_decl_with_refinement": fail_minimal_arrayref_interface(
        is_skip=True
    ),
    # TODO:  error: cannot convert ArrayRefTensor<float> to AtenTensorHandle
    "test_dynamic_cat": fail_minimal_arrayref_interface(),
    # https://github.com/pytorch/pytorch/issues/129550
    # https://github.com/pytorch/pytorch/issues/123691
    "test_dynamic_scalar": fail_minimal_arrayref_interface(is_skip=True),
    # https://github.com/pytorch/pytorch/issues/122980
    "test_fft_c2c": fail_stack_allocation(is_skip=True),
    # TODO: test_freezing_abi_compatible_cpu fails,
    #   AssertionError: None, i.e. optional output is not supported
    "test_freezing": fail_with_and_without_stack_allocation(is_skip=True),
    # TODO: test_linear_freezing_abi_compatible_cpu fails,
    #   AssertionError: None, i.e. optional output is not supported
    "test_linear_freezing": fail_with_and_without_stack_allocation(is_skip=True),
    # FIXME: failed with Segfault while exiting the Python runtime
    "test_missing_cubin": fail_with_and_without_stack_allocation(is_skip=True),
    # minimal arrayref interface only works with CPU; test crashes.
    # https://github.com/pytorch/pytorch/issues/122983
    "test_multi_device": fail_minimal_arrayref_interface(is_skip=True),
    # TODO: AssertionError: unsupported Optional type in convert_arg_type: Generator
    "test_normal_functional": fail_with_and_without_stack_allocation(is_skip=True),
    # TODO: The same issue as https://github.com/pytorch/pytorch/issues/122978
    # error: cannot convert ArrayRefTensor<float> to AtenTensorHandle
    "test_reuse_kernel_dynamic": fail_minimal_arrayref_interface(is_skip=True),
    # the test segfaults
    "test_repeat_output": fail_stack_allocation(is_skip=True),
    # TODO: failed internally
    "test_multiple_output_alias": fail_with_and_without_stack_allocation(is_skip=True),
    # segfault
    "test_buffer_mutation_1": fail_stack_allocation(is_skip=True),
    # segfault
    "test_buffer_mutation_2": fail_stack_allocation(is_skip=True),
    # segfault
    "test_bool_input": fail_stack_allocation(is_skip=True),
    # segfault
    "test_int_list_input": fail_stack_allocation(is_skip=True),
    # segfault
    # 'AOTInductorTestABICompatibleCpuWithStackAllocation' object has no attribute 'code_check_count'
    "test_buffer_mutation_3": fail_stack_allocation(is_skip=True),
    # FIXME: failed with Segfault while exiting the Python runtime
    "test_scatter_fallback": fail_stack_allocation(is_skip=True),
    # Looks like the same issue as https://github.com/pytorch/pytorch/issues/122978
    "test_scatter_reduce_fallback": fail_minimal_arrayref_interface(is_skip=True),
    # Looks like the same issue as https://github.com/pytorch/pytorch/issues/122978
    "test_index_put_fallback": fail_minimal_arrayref_interface(is_skip=True),
    # https://github.com/pytorch/pytorch/issues/122984
    "test_index_put_with_none_index": fail_minimal_arrayref_interface(is_skip=True),
    # FIXME: failed with Segfault while exiting the Python runtime
    "test_constant": fail_stack_allocation(is_skip=True),
    # Looks like the same issue as https://github.com/pytorch/pytorch/issues/122978
    "test_shifted_constraint_ranges": fail_with_and_without_stack_allocation(
        is_skip=True
    ),
    # https://github.com/pytorch/pytorch/issues/123691
    "test_amp_fallback_random": fail_minimal_arrayref_interface(is_skip=True),
    "test_simple_dynamic": fail_minimal_arrayref_interface(),
    # https://github.com/pytorch/pytorch/issues/123691
    "test_zero_grid_with_unbacked_symbols": fail_minimal_arrayref_interface(
        is_skip=True
    ),
    # failed on MacOS
    "test_zero_grid_with_backed_symbols": fail_with_and_without_stack_allocation(
        is_skip=True
    ),
    # https://github.com/pytorch/pytorch/issues/122990
    "test_cond_non_tensor_predicates_dynamic_False": fail_stack_allocation(
        is_skip=True
    ),
    # same issue as https://github.com/pytorch/pytorch/issues/122990
    "test_cond_non_tensor_predicates_dynamic_True": fail_stack_allocation(is_skip=True),
    # https://github.com/pytorch/pytorch/issues/122991
    "test_runtime_checks_complex": fail_with_and_without_stack_allocation(is_skip=True),
    "test_runtime_checks_fp8": fail_with_and_without_stack_allocation(is_skip=True),
    "test_while_loop_simple": fail_stack_allocation(is_skip=True),
    "test_while_loop_nested": fail_stack_allocation(is_skip=True),
    "test_while_loop_with_outer_code": fail_stack_allocation(is_skip=True),
    # TODO: error: cannot convert ArrayRefTensor<float> to AtenTensorHandle
    "test_while_loop_with_outer_buffers": fail_stack_allocation(is_skip=True),
    # TODO: use of undeclared identifier 'float8_e4m3fn' and 'half'
    "test_fp8": fail_minimal_arrayref_interface(is_skip=True),
    "test_custom_op_add": fail_minimal_arrayref_interface(is_skip=True),
}

# test_failures, xfail by default, set is_skip=True to skip
CUDA_TEST_FAILURES = {
    # TODO: AssertionError: unsupported Optional type in convert_arg_type: Generator
    "test_normal_functional": fail_abi_compatible_cuda(is_skip=True),
    # no runtime checks for non_abi_compatible mode
    "test_runtime_checks": fail_non_abi_compatible_cuda(is_skip=True),
    "test_runtime_checks_complex": fail_non_abi_compatible_cuda(is_skip=True),
    "test_runtime_checks_fp8": fail_non_abi_compatible_cuda(is_skip=True),
    "test_runtime_checks_dtype_failed": fail_non_abi_compatible_cuda(is_skip=True),
    "test_runtime_checks_shape_failed": fail_non_abi_compatible_cuda(is_skip=True),
    # quantized unsupported for GPU
    "test_quantized_linear": fail_cuda(is_skip=True),
    "test_custom_op_add": fail_non_abi_compatible_cuda(is_skip=True),
    # fp8 to be re-enabled for AOTI
    "test_fp8": fail_cuda(is_skip=True),
}


if not IS_FBCODE:
    # The following tests look like they pass in both pytest and unittest (xml
    # and terminal output say pass), but the process will segfault.  This only
    # happens in OSS CI and is fine internally.
    CPU_TEST_FAILURES.update(
        {
            "test_duplicated_params": fail_stack_allocation(is_skip=True),
            "test_embedding_bag": fail_stack_allocation(is_skip=True),
            "test_fqn": fail_stack_allocation(is_skip=True),
            "test_no_args": fail_stack_allocation(is_skip=True),
            "test_output_misaligned": fail_stack_allocation(is_skip=True),
            "test_pytree_inputs": fail_stack_allocation(is_skip=True),
            "test_seq": fail_stack_allocation(is_skip=True),
            "test_simple_split": fail_stack_allocation(is_skip=True),
            "test_addmm": fail_minimal_arrayref_interface(is_skip=True),
            "test_aliased_buffer_reuse": fail_minimal_arrayref_interface(is_skip=True),
            "test_buffer_reuse": fail_minimal_arrayref_interface(is_skip=True),
            "test_constant_folding": fail_minimal_arrayref_interface(is_skip=True),
            "test_convolution": fail_minimal_arrayref_interface(is_skip=True),
            "test_empty_graph": fail_minimal_arrayref_interface(is_skip=True),
            "test_large_weight": fail_minimal_arrayref_interface(is_skip=True),
            "test_large_mmaped_weights": fail_minimal_arrayref_interface(is_skip=True),
            "test_normal_functional": fail_minimal_arrayref_interface(is_skip=True),
            "test_misc_1": fail_minimal_arrayref_interface(is_skip=True),
            "test_missing_output": fail_minimal_arrayref_interface(is_skip=True),
            "test_model_modified_weights": fail_minimal_arrayref_interface(
                is_skip=True
            ),
            "test_output_path_1": fail_minimal_arrayref_interface(is_skip=True),
            "test_quantized_linear": fail_minimal_arrayref_interface(is_skip=True),
            "test_repeat_interleave": fail_minimal_arrayref_interface(is_skip=True),
            "test_return_constant": fail_minimal_arrayref_interface(is_skip=True),
            "test_reuse_kernel": fail_minimal_arrayref_interface(is_skip=True),
            "test_simple": fail_minimal_arrayref_interface(is_skip=True),
            "test_small_constant": fail_minimal_arrayref_interface(is_skip=True),
            "test_with_no_triton_profiler": fail_minimal_arrayref_interface(
                is_skip=True
            ),
            "test_with_offset": fail_minimal_arrayref_interface(is_skip=True),
            "test_with_profiler": fail_minimal_arrayref_interface(is_skip=True),
            "test_zero_size_weight": fail_minimal_arrayref_interface(is_skip=True),
        }
    )

copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestABICompatibleCpu,
    "abi_compatible_cpu",
    CPU_TEST_FAILURES,
)


class AOTInductorTestABICompatibleCpuWithStackAllocation(AOTITestCase):
    device = "cpu"
    abi_compatible = True
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs
    code_check_count = code_check_count
    allow_stack_allocation = True
    use_minimal_arrayref_interface = False


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestABICompatibleCpuWithStackAllocation,
    "abi_compatible_cpu_with_stack_allocation",
    CPU_TEST_FAILURES,
)


class AOTInductorTestABICompatibleCpuWithStackAllocationAndMinimalArrayRefInterface(
    TestCase
):
    device = "cpu"
    abi_compatible = True
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs
    allow_stack_allocation = True
    use_minimal_arrayref_interface = True


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestABICompatibleCpuWithStackAllocationAndMinimalArrayRefInterface,
    "abi_compatible_cpu_with_stack_allocation_and_minimal_arrayref_interface",
    CPU_TEST_FAILURES,
)


@unittest.skipIf(sys.platform == "darwin", "No CUDA on MacOS")
class AOTInductorTestABICompatibleCuda(AOTITestCase):
    device = "cuda"
    abi_compatible = True
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs
    code_check_count = code_check_count
    allow_stack_allocation = False
    use_minimal_arrayref_interface = False


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestABICompatibleCuda,
    "abi_compatible_cuda",
    CUDA_TEST_FAILURES,
)


@unittest.skipIf(
    IS_FBCODE or sys.platform == "darwin",
    "NonABI mode should not be used in fbcode nor on MacOS",
)
class AOTInductorTestNonABICompatibleCpu(AOTITestCase):
    device = "cpu"
    abi_compatible = False
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs
    code_check_count = code_check_count
    allow_stack_allocation = False
    use_minimal_arrayref_interface = False


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestNonABICompatibleCpu,
    "non_abi_compatible_cpu",
    # test_failures, xfail by default, set is_skip=True to skip
    {
        "test_duplicate_constant_folding": TestFailure(
            ("non_abi_compatible_cpu",), is_skip=True
        ),
        # no runtime checks for non_abi_compatible mode
        "test_runtime_checks": TestFailure(("non_abi_compatible_cpu",), is_skip=True),
        "test_runtime_checks_dtype_failed": TestFailure(
            ("non_abi_compatible_cpu",), is_skip=True
        ),
        "test_runtime_checks_shape_failed": TestFailure(
            ("non_abi_compatible_cpu",), is_skip=True
        ),
        "test_custom_op_add": TestFailure(("non_abi_compatible_cpu",), is_skip=True),
    },
)


@unittest.skipIf(
    IS_FBCODE or sys.platform == "darwin",
    "NonABI mode should not be used in fbcode nor on MacOS",
)
class AOTInductorTestNonABICompatibleCuda(AOTITestCase):
    device = "cuda"
    abi_compatible = False
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs
    code_check_count = code_check_count
    allow_stack_allocation = False
    use_minimal_arrayref_interface = False


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestNonABICompatibleCuda,
    "non_abi_compatible_cuda",
    CUDA_TEST_FAILURES,
)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    # cpp_extension N/A in fbcode
    if HAS_CUDA or sys.platform == "darwin":
        run_tests(needs="filelock")
