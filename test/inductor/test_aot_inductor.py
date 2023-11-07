# Owner(s): ["module: inductor"]
import copy
import os
import sys
import tempfile
import unittest
from typing import Dict

import torch
import torch._export
import torch._inductor
import torch.fx._pytree as fx_pytree
from torch._dynamo.testing import same
from torch._inductor import config
from torch._inductor.exc import CppWrapperCodeGenError
from torch._inductor.utils import aot_inductor_launcher, cache_dir

from torch.testing import FileCheck

from torch.testing._internal.common_utils import (
    IS_CI,
    IS_FBCODE,
    IS_WINDOWS,
    TEST_WITH_ROCM,
    TestCase,
)
from torch.testing._internal.inductor_utils import HAS_CUDA
from torch.utils import _pytree as pytree

if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

try:
    try:
        from .test_torchinductor import (
            copy_tests,
            requires_cuda,
            requires_multigpu,
            TestFailure,
        )
    except ImportError:
        from test_torchinductor import (
            copy_tests,
            requires_cuda,
            requires_multigpu,
            TestFailure,
        )
except (unittest.SkipTest, ImportError) as e:
    if __name__ == "__main__":
        sys.exit(0)
    raise


class AOTInductorModelRunner:
    @classmethod
    def compile(cls, model, example_inputs, options=None, constraints=None):
        # The exact API is subject to change
        so_path = torch._export.aot_compile(
            model,
            example_inputs,
            options=options,
            constraints=constraints,
            remove_runtime_assertions=True,
        )
        return so_path

    @classmethod
    def load(cls, device, so_path, example_inputs):
        if IS_FBCODE:
            from .fb import test_aot_inductor_model_runner_pybind

            module = test_aot_inductor_model_runner_pybind.Runner(
                so_path, device == "cpu"
            )

            call_spec = module.get_call_spec()
            in_spec = pytree.treespec_loads(call_spec[0])
            out_spec = pytree.treespec_loads(call_spec[1])

            def optimized(*args):
                flat_inputs = fx_pytree.tree_flatten_spec((*args, {}), in_spec)
                flat_outputs = module.run(flat_inputs)
                return pytree.tree_unflatten(flat_outputs, out_spec)

        else:
            module = torch.utils.cpp_extension.load_inline(
                name="aot_inductor",
                cpp_sources=[aot_inductor_launcher(so_path, device)],
                # use a unique build directory to avoid test interference
                build_directory=tempfile.mkdtemp(dir=cache_dir()),
                functions=["run", "get_call_spec"],
                with_cuda=(device == "cuda"),
            )

            call_spec = module.get_call_spec()
            in_spec = pytree.treespec_loads(call_spec[0])
            out_spec = pytree.treespec_loads(call_spec[1])

            def optimized(*args):
                flat_inputs = fx_pytree.tree_flatten_spec((*args, {}), in_spec)
                flat_outputs = module.run(flat_inputs)
                return pytree.tree_unflatten(flat_outputs, out_spec)

        return optimized

    @classmethod
    def run(cls, device, model, example_inputs, options=None, constraints=None):
        so_path = AOTInductorModelRunner.compile(
            model, example_inputs, options=options, constraints=constraints
        )
        optimized = AOTInductorModelRunner.load(device, so_path, example_inputs)
        return optimized(example_inputs)

    @classmethod
    def run_multiple(
        cls,
        device,
        model,
        list_example_inputs,
        options=None,
        constraints=None,
    ):
        so_path = AOTInductorModelRunner.compile(
            model,
            list_example_inputs[0],
            options=options,
            constraints=constraints,
        )
        optimized = AOTInductorModelRunner.load(device, so_path, list_example_inputs[0])
        list_output_tensors = []
        for example_inputs in list_example_inputs:
            list_output_tensors.append(optimized(example_inputs))
        return list_output_tensors


def check_model(
    self: TestCase,
    model,
    example_inputs,
    options=None,
    constraints=None,
):
    with torch.no_grad(), config.patch(
        {
            "aot_inductor.abi_compatible": self.abi_compatible,
            "allow_stack_allocation": self.allow_stack_allocation,
        }
    ):
        torch.manual_seed(0)
        model = model.to(self.device)
        ref_model = copy.deepcopy(model)
        ref_inputs = copy.deepcopy(example_inputs)
        expected = ref_model(*ref_inputs)

        torch.manual_seed(0)
        actual = AOTInductorModelRunner.run(
            self.device, model, example_inputs, options, constraints
        )

    self.assertTrue(same(actual, expected))


def check_model_with_multiple_inputs(
    self: TestCase,
    model,
    list_example_inputs,
    options=None,
    constraints=None,
):
    with torch.no_grad(), config.patch(
        {
            "aot_inductor.abi_compatible": self.abi_compatible,
            "allow_stack_allocation": self.allow_stack_allocation,
        }
    ):
        torch.manual_seed(0)
        model = model.to(self.device)
        ref_model = copy.deepcopy(model)
        ref_inputs = copy.deepcopy(list_example_inputs)
        list_expected = [ref_model(*inputs) for inputs in ref_inputs]

        torch.manual_seed(0)
        list_actual = AOTInductorModelRunner.run_multiple(
            self.device, model, list_example_inputs, options, constraints
        )

    self.assertTrue(same(list_actual, list_expected))


class AOTInductorTestsTemplate:
    def test_simple(self):
        class Model(torch.nn.Module):
            def __init__(self):
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
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x)

        example_inputs = (torch.randn(4, 4, device=self.device),)
        with config.patch({"always_keep_tensor_constants": True}):
            self.check_model(Model().to(self.device), example_inputs)

    def test_output_path_1(self):
        class Model(torch.nn.Module):
            def __init__(self):
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
            def __init__(self):
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
        actual_path = AOTInductorModelRunner.compile(
            model, example_inputs, options={"aot_inductor.output_path": expected_path}
        )
        self.assertTrue(actual_path == expected_path)

    @requires_cuda()
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

    def test_large(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(512, 250112)

            def forward(self, x, y):
                return x + self.linear(y)

        example_inputs = (
            torch.randn(1, 250112, device=self.device),
            torch.randn(1, 512, device=self.device),
        )
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

    def test_missing_output(self):
        class Model(torch.nn.Module):
            def __init__(self):
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
            def __init__(self):
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
            def __init__(self):
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
            def __init__(self):
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
            def __init__(self):
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
            def __init__(self):
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
            def __init__(self):
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

    def test_simple_dynamic(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                add_0 = x + y
                return torch.nn.functional.relu(input=add_0, inplace=False)

        a = torch.randn(128, 2048, device=self.device)
        b = torch.randn(128, 2048, device=self.device)
        constraints = [
            torch._export.dynamic_dim(a, 0) >= 1,
            torch._export.dynamic_dim(a, 0) <= 2048,
            torch._export.dynamic_dim(a, 0) == torch._export.dynamic_dim(b, 0),
        ]
        example_inputs = (a, b)
        self.check_model(Model(), example_inputs, constraints=constraints)

    def test_poi_multiple_dynamic(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                add_0 = x + y
                return torch.nn.functional.relu(input=add_0, inplace=False)

        a = torch.randn(128, 2048, device=self.device)
        b = torch.randn(128, 2048, device=self.device)
        constraints = [
            torch._export.dynamic_dim(a, 0) >= 1,
            torch._export.dynamic_dim(a, 0) <= 2048,
            torch._export.dynamic_dim(a, 0) == torch._export.dynamic_dim(b, 0),
        ]
        list_example_inputs = [(a, b)]
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
            Model(), list_example_inputs, constraints=constraints
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
        constraints = [
            torch._export.dynamic_dim(a, 0) >= 1,
            torch._export.dynamic_dim(a, 0) <= 2048,
        ]
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
            constraints=constraints,
            options={
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
            },
        )

    def test_bmm_multiple_dynamic(self):
        class Model(torch.nn.Module):
            def __init__(self):
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
        constraints = [
            torch._export.dynamic_dim(a, 0) >= 1,
            torch._export.dynamic_dim(a, 0) <= 2048,
            torch._export.dynamic_dim(a, 0) == torch._export.dynamic_dim(b, 0),
        ]
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
            constraints=constraints,
        )

    def test_foreach_multiple_dynamic(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x_unsqueeze = torch.unsqueeze(x, dim=0)
                y_unsqueeze = torch.unsqueeze(y, dim=0)
                cat = torch.cat([x_unsqueeze, y_unsqueeze], dim=0)
                return cat

        model = Model()
        a = torch.randn(128, 2048, device=self.device)
        b = torch.randn(128, 2048, device=self.device)
        constraints = [
            torch._export.dynamic_dim(a, 0) >= 1,
            torch._export.dynamic_dim(a, 0) <= 2048,
            torch._export.dynamic_dim(a, 0) == torch._export.dynamic_dim(b, 0),
        ]
        list_example_inputs = [(a, b)]
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
            constraints=constraints,
        )

    # scaled_dot_product_flash_attention
    @unittest.skipIf(IS_FBCODE, "Not yet runnable in fbcode")
    def test_sdpa(self):
        class Model(torch.nn.Module):
            def __init__(self):
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
    def test_sdpa_2(self):
        class Model(torch.nn.Module):
            def __init__(self):
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

    def test_zero_grid_with_unbacked_symbols(self):
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                nz = torch.nonzero(x)
                b = torch.ones_like(nz, dtype=torch.float16)
                c = torch.zeros_like(nz, dtype=torch.float16)
                d = (b + c) @ y
                return d.sum()

        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")
        example_inputs = (
            torch.tensor([1, 1, 1], device=self.device),
            torch.randn((1, 32), dtype=torch.float16, device=self.device),
        )
        self.check_model(Repro(), example_inputs)

    def test_repeat_interleave(self):
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.ops.aten.repeat_interleave.Tensor(x, output_size=12)

        example_inputs = (torch.ones((1,), dtype=torch.int32, device=self.device) * 12,)
        self.check_model(Repro(), example_inputs)

    def test_dynamic_cat(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x1, x2):
                return torch.cat([x1, x2], dim=0)

        a = torch.randn(2, 4, device=self.device)
        b = torch.randn(3, 4, device=self.device)
        constraints = [
            torch._export.dynamic_dim(a, 0) >= 1,
            torch._export.dynamic_dim(a, 0) <= 10,
            torch._export.dynamic_dim(b, 0) >= 1,
            torch._export.dynamic_dim(b, 0) <= 20,
        ]
        example_inputs = (a, b)
        self.check_model(Model(), example_inputs, constraints=constraints)

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
        with torch.cuda.device(0), config.patch(
            "aot_inductor.abi_compatible", self.abi_compatible
        ):
            so_path = AOTInductorModelRunner.compile(
                model=Model(w1.cuda(0), w2.cuda(0)),
                example_inputs=tuple(t.cuda(0) for t in inputs),
            )

        # Run model on cuda:N
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                example_inputs = tuple(t.cuda(i) for t in inputs)
                optimized = AOTInductorModelRunner.load("cuda", so_path, example_inputs)
                result_cuda = optimized(example_inputs)
            self.assertTrue(same(result_cpu, result_cuda.cpu()))

    def test_pytree_inputs(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x: Dict[str, torch.Tensor]):
                add_ = torch.zeros(5)
                mul_ = torch.ones(5)
                for v in x.values():
                    add_ += v
                    mul_ *= v

                return [add_, mul_]

        self.check_model(M(), ({"x": torch.ones(5), "y": torch.ones(5)},))

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
            "aot_inductor.abi_compatible", self.abi_compatible
        ):
            result_cuda_0 = AOTInductorModelRunner.run(
                "cuda", Model(weight.cuda(0)), tuple(t.cuda(0) for t in inputs)
            )

        with torch.cuda.device(1), torch.no_grad(), config.patch(
            "aot_inductor.abi_compatible", self.abi_compatible
        ):
            result_cuda_1 = AOTInductorModelRunner.run(
                "cuda", Model(weight.cuda(1)), tuple(t.cuda(1) for t in inputs)
            )

        self.assertTrue(same(result_cpu, result_cuda_0.cpu()))
        self.assertTrue(same(result_cpu, result_cuda_1.cpu()))

    def test_reuse_kernel(self):
        class Model(torch.nn.Module):
            def __init__(self):
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
        self.check_model(Model(), example_inputs)

        if self.device == "cuda":
            so_path = torch._export.aot_compile(Model(), example_inputs)
            with open(os.path.splitext(so_path)[0] + ".cpp") as cpp:
                src_code = cpp.read()
                FileCheck().check_count(
                    "triton_poi_fused_sin_0 = loadKernel(",
                    1,
                    exactly=True,
                ).run(src_code)

    def test_fake_tensor_device_validation(self):
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x + y

        example_inputs = (torch.randn(10, 10), torch.randn(10, 10))

        # Export on CPU
        exported_program = torch._export.export(
            Model(),
            example_inputs,
            constraints=[],
        )

        # Compile exported model on CUDA
        gm = exported_program.graph_module.to(self.device)
        with self.assertRaisesRegex(ValueError, "Device mismatch between fake input"):
            torch._inductor.aot_compile(
                gm, tuple(i.to(self.device) for i in example_inputs)
            )

    @unittest.mock.patch("torch._inductor.graph.supported_dtype_of_cpp_wrapper")
    def test_unsupported_input_dtype(self, supported_dtype_of_cpp_wrapper_mock):
        supported_dtype_of_cpp_wrapper_mock.return_value = False

        class Model(torch.nn.Module):
            def __init__(self):
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

    def test_normal_functional(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.ops.aten.normal_functional.default(x)

        self.check_model(Model(), (torch.empty(4, 1, 4, 4),))

    def test_empty_graph(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x

        example_inputs = (torch.randn(8, 4, 4, device=self.device),)
        self.check_model(Model(), example_inputs)

    def test_dup_unbacked_sym_decl(self):
        class Model(torch.nn.Module):
            def __init__(self):
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
            so_path = AOTInductorModelRunner.compile(m, example_inputs)

        # run under grad enabled
        self.assertTrue(torch.is_grad_enabled())

        optimized = AOTInductorModelRunner.load(self.device, so_path, example_inputs)
        actual = optimized(example_inputs)
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

    def test_repeat_output(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                y = torch.sin(x)
                return y, y

        example_inputs = (torch.randn(3, 10, device=self.device),)
        self.check_model(Model(), example_inputs)


class AOTInductorTestABICompatibleCpu(TestCase):
    device = "cpu"
    abi_compatible = True
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs
    allow_stack_allocation = False


def fail_with_and_without_stack_allocation(is_skip=False):
    return TestFailure(
        ("abi_compatible_cpu", "abi_compatible_cpu_with_stack_allocation"),
        is_skip=is_skip,
    )


# test_failures, xfail by default, set is_skip=True to skip
CPU_TEST_FAILURES = {
    "test_addmm_multiple_dynamic": fail_with_and_without_stack_allocation(),
    "test_bmm_multiple_dynamic": fail_with_and_without_stack_allocation(),
    "test_dup_unbacked_sym_decl": fail_with_and_without_stack_allocation(),
    "test_dynamic_cat": fail_with_and_without_stack_allocation(),
    "test_dynamic_smem_above_default_limit": fail_with_and_without_stack_allocation(),
    "test_foreach_multiple_dynamic": fail_with_and_without_stack_allocation(),
    # TODO: test_freezing_abi_compatible_cpu somehow fails on CI but not locally,
    #   NotImplementedError: Cannot access storage of OpaqueTensorImpl
    "test_freezing": fail_with_and_without_stack_allocation(is_skip=True),
    "test_normal_functional": fail_with_and_without_stack_allocation(),
    "test_poi_multiple_dynamic": fail_with_and_without_stack_allocation(),
    # There is a double-free issue which will be fixed in another PR
    "test_repeat_output": fail_with_and_without_stack_allocation(is_skip=True),
    "test_sdpa": fail_with_and_without_stack_allocation(),
    "test_sdpa_2": fail_with_and_without_stack_allocation(),
    "test_simple_dynamic": fail_with_and_without_stack_allocation(),
}

copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestABICompatibleCpu,
    "abi_compatible_cpu",
    CPU_TEST_FAILURES,
)


class AOTInductorTestABICompatibleCpuWithStackAllocation(TestCase):
    device = "cpu"
    abi_compatible = True
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs
    allow_stack_allocation = True


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestABICompatibleCpuWithStackAllocation,
    "abi_compatible_cpu_with_stack_allocation",
    CPU_TEST_FAILURES,
)


class AOTInductorTestABICompatibleCuda(TestCase):
    device = "cuda"
    abi_compatible = True
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs
    allow_stack_allocation = False


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestABICompatibleCuda,
    "abi_compatible_cuda",
    # test_failures, xfail by default, set is_skip=True to skip
    {
        "test_dup_unbacked_sym_decl": TestFailure(("abi_compatible_cuda",)),
        "test_normal_functional": TestFailure(("abi_compatible_cuda",)),
        # There is a double-free issue which will be fixed in another PR
        "test_repeat_output": TestFailure(("abi_compatible_cuda",), is_skip=True),
    },
)


@unittest.skipIf(IS_FBCODE, "NonABI mode should not be used in fbcode")
class AOTInductorTestNonABICompatibleCpu(TestCase):
    device = "cpu"
    abi_compatible = False
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs
    allow_stack_allocation = False


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestNonABICompatibleCpu,
    "non_abi_compatible_cpu",
    # test_failures, xfail by default, set is_skip=True to skip
    {
        "test_addmm_multiple_dynamic": TestFailure(("non_abi_compatible_cpu",)),
        "test_bmm_multiple_dynamic": TestFailure(("non_abi_compatible_cpu",)),
        "test_dynamic_smem_above_default_limit": TestFailure(
            ("non_abi_compatible_cpu",)
        ),
        # TODO: test_freezing_non_abi_compatible_cpu somehow fails on CI but not locally,
        #   NotImplementedError: Cannot access storage of OpaqueTensorImpl
        "test_freezing": TestFailure(("non_abi_compatible_cpu",), is_skip=True),
    },
)


@unittest.skipIf(IS_FBCODE, "NonABI mode should not be used in fbcode")
class AOTInductorTestNonABICompatibleCuda(TestCase):
    device = "cuda"
    abi_compatible = False
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs
    allow_stack_allocation = False


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestNonABICompatibleCuda,
    "non_abi_compatible_cuda",
)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    # cpp_extension N/A in fbcode
    if HAS_CUDA and not TEST_WITH_ROCM:
        run_tests(needs="filelock")
