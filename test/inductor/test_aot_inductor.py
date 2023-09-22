# Owner(s): ["module: inductor"]
import copy
import sys
import unittest

import torch
import torch._export
import torch._inductor

import torch.fx._pytree as fx_pytree
from torch._dynamo.testing import same
from torch._inductor import config
from torch._inductor.utils import aot_inductor_launcher

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
        from .test_torchinductor import copy_tests
    except ImportError:
        from test_torchinductor import copy_tests
except (unittest.SkipTest, ImportError) as e:
    if __name__ == "__main__":
        sys.exit(0)
    raise


class AOTInductorModelRunner:
    @classmethod
    def load(
        cls, model, example_inputs, example_outputs, options=None, constraints=None
    ):
        # AOTInductorModel relies on the caller to pass in output_tensors,
        # so we need to explicitly allocate output tensors here.
        if constraints is None:
            constraints = []
        output_tensors = []
        example_outputs, output_spec = pytree.tree_flatten(example_outputs)
        for output in example_outputs:
            output_tensors.append(torch.empty_like(output))

        # The exact API is subject to change
        so_path, exported = torch._export.aot_compile(
            model,
            example_inputs,
            options=options,
            constraints=constraints,
        )

        launcher = aot_inductor_launcher
        is_cpu = any(x.device.type == "cpu" for x in example_inputs)
        if is_cpu:
            launcher = launcher.replace("false /*is_cpu*/", "true /*is_cpu*/")

        optimized = torch.utils.cpp_extension.load_inline(
            name="aot_inductor",
            cpp_sources=[launcher],
            functions=["run"],
            extra_ldflags=[so_path],
            with_cuda=True,
        ).run

        return optimized, exported, output_tensors, output_spec

    @classmethod
    def run_compiled(
        cls, optimized, exported, example_inputs, output_tensors, output_spec
    ):
        flat_example_inputs = fx_pytree.tree_flatten_spec(
            (example_inputs, {}), exported.call_spec.in_spec
        )
        optimized(flat_example_inputs, output_tensors)
        return pytree.tree_unflatten(output_tensors, output_spec)

    @classmethod
    def run(
        cls, model, example_inputs, example_outputs, options=None, constraints=None
    ):
        if constraints is None:
            constraints = []
        example_outputs = copy.deepcopy(example_outputs)
        optimized, exported, output_tensors, output_spec = AOTInductorModelRunner.load(
            model, example_inputs, example_outputs, options, constraints=constraints
        )
        return AOTInductorModelRunner.run_compiled(
            optimized, exported, example_inputs, output_tensors, output_spec
        )

    @classmethod
    def run_multiple(
        cls,
        model,
        list_example_inputs,
        list_example_outputs,
        options=None,
        constraints=None,
    ):
        optimized, exported, _, output_spec = AOTInductorModelRunner.load(
            model,
            list_example_inputs[0],
            list_example_outputs[0],
            options=options,
            constraints=constraints,
        )
        list_output_tensors = []
        for example_inputs, example_outputs in zip(
            list_example_inputs, list_example_outputs
        ):
            output_tensors = [torch.empty_like(output) for output in example_outputs]
            list_output_tensors.append(
                AOTInductorModelRunner.run_compiled(
                    optimized, exported, example_inputs, output_tensors, output_spec
                )
            )
        return list_output_tensors


def check_model(
    self: TestCase,
    model,
    example_inputs,
    options=None,
    constraints=None,
):
    expected = model(*example_inputs)
    with config.patch("aot_inductor.abi_compatible", self.abi_compatible):
        actual = AOTInductorModelRunner.run(
            model, example_inputs, expected, options, constraints
        )
    self.assertTrue(same(actual, expected))


def check_model_with_multiple_inputs(
    self: TestCase,
    model,
    list_example_inputs,
    options=None,
    constraints=None,
):
    list_expected = [
        (model(*example_inputs),) for example_inputs in list_example_inputs
    ]
    with config.patch("aot_inductor.abi_compatible", self.abi_compatible):
        list_actual = AOTInductorModelRunner.run_multiple(
            model, list_example_inputs, list_expected, options, constraints
        )
    self.assertTrue(same(list_actual, list_expected))


@unittest.skipIf(IS_FBCODE, "cpp extension doesn't work in fbcode CI")
class AOTInductorTestsTemplate:
    def test_simple(self):
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.randn(10, 10, device="cuda")

            def forward(self, x, y):
                return x + torch.nn.functional.linear(y, self.weight)

        example_inputs = (
            torch.randn(10, 10, device="cuda"),
            torch.randn(10, 10, device="cuda"),
        )
        self.check_model(Repro(), example_inputs)

    def test_simple_cpu(self):
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.randn(10, 10, device="cpu")

            def forward(self, x, y):
                return x + torch.nn.functional.linear(y, self.weight)

        example_inputs = (
            torch.randn(10, 10, device="cpu"),
            torch.randn(10, 10, device="cpu"),
        )
        self.check_model(Repro(), example_inputs)

    def test_large(self):
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.randn(250112, 512, device="cuda")

            def forward(self, x, y):
                return x + torch.nn.functional.linear(y, self.weight)

        example_inputs = (
            torch.randn(1, 250112, device="cuda"),
            torch.randn(1, 512, device="cuda"),
        )
        self.check_model(Repro(), example_inputs)

    def test_with_offset(self):
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.orig_tensor = torch.randn(2, 15, 10, device="cuda")[0]
                self.tensor = self.orig_tensor[5:, :]

            def forward(self, x, y):
                return (
                    x
                    + torch.nn.functional.linear(y, self.orig_tensor[:10, :])
                    + self.tensor
                )

        example_inputs = (
            torch.randn(10, 10, device="cuda"),
            torch.randn(10, 10, device="cuda"),
        )
        self.check_model(Repro(), example_inputs)

    def test_missing_output(self):
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                a = torch.sin(x)
                b = torch.mm(a, y)
                c = torch.cos(b)
                return c

        example_inputs = (
            torch.randn(10, 10, device="cuda"),
            torch.randn(10, 10, device="cuda"),
        )
        self.check_model(Repro(), example_inputs)

    def test_output_misaligned(self):
        class Repro(torch.nn.Module):
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
            torch.randn(10, 10, device="cuda"),
            torch.randn(10, 10, device="cuda"),
        )
        self.check_model(Repro(), example_inputs)

    def test_dynamic_smem_above_default_limit(self):
        class Repro(torch.nn.Module):
            def forward(self, x, y):
                return x @ y

        model = Repro()
        # on A100, the generated Triton kernel for this MM
        # requires 55296 bytes of dynamic SMEM which is above
        # the A100's default dynamic SMEM limit of 49152 bytes.
        example_inputs = (
            torch.randn(10285, 96, device="cuda"),
            torch.randn(96, 1, device="cuda"),
        )
        self.check_model(
            model,
            example_inputs,
            options={
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
            },
        )

    def test_addmm(self):
        class Model(torch.nn.Module):
            def __init__(self, n, k):
                super().__init__()
                self.weight = torch.randn(n, k, device="cuda")
                self.bias = torch.randn(n, device="cuda")

            def forward(self, a):
                return torch.nn.functional.linear(a, self.weight, self.bias)

        M = 8
        N = 6
        K = 16
        model = Model(N, K)
        batch = 2
        a = torch.randn(batch, M, K, device="cuda")
        example_inputs = (a,)
        self.check_model(model, example_inputs)

    def test_aliased_buffer_reuse(self):
        class Repro(torch.nn.Module):
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
            torch.randn(4, 2, device="cuda"),
            torch.randn(4, 2, device="cuda"),
        )
        self.check_model(Repro(), example_inputs)

    def test_buffer_reuse(self):
        class Repro(torch.nn.Module):
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
            torch.randn(4, 4, device="cuda"),
            torch.randn(4, 4, device="cuda"),
        )
        self.check_model(Repro(), example_inputs)

    def test_duplicated_params(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.rand(6))
                self.q = self.p

            def forward(self, x):
                return self.p * x + self.q

        model = Model()
        example_inputs = (torch.rand(6),)
        expected = model(*example_inputs)
        actual = torch._export.export(model, example_inputs)(*example_inputs)
        self.assertTrue(same(actual, expected))

    def test_simple_dynamic(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                add_0 = x + y
                return torch.nn.functional.relu(input=add_0, inplace=False)

        a = torch.randn(128, 2048, device="cuda")
        b = torch.randn(128, 2048, device="cuda")
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

        a = torch.randn(128, 2048, device="cuda")
        b = torch.randn(128, 2048, device="cuda")
        constraints = [
            torch._export.dynamic_dim(a, 0) >= 1,
            torch._export.dynamic_dim(a, 0) <= 2048,
            torch._export.dynamic_dim(a, 0) == torch._export.dynamic_dim(b, 0),
        ]
        list_example_inputs = [(a, b)]
        list_example_inputs.append(
            (
                torch.randn(64, 2048, device="cuda"),
                torch.randn(64, 2048, device="cuda"),
            ),
        )
        list_example_inputs.append(
            (
                torch.randn(211, 2048, device="cuda"),
                torch.randn(211, 2048, device="cuda"),
            ),
        )
        self.check_model_with_multiple_inputs(
            Model(), list_example_inputs, constraints=constraints
        )

    def test_addmm_multiple_dynamic(self):
        class Model(torch.nn.Module):
            def __init__(self, n, k):
                super().__init__()
                self.weight = torch.randn(n, k, device="cuda")
                self.bias = torch.randn(n, device="cuda")

            def forward(self, a):
                return torch.nn.functional.linear(a, self.weight, self.bias)

        M = 8
        N = 6
        K = 16
        model = Model(N, K)
        batch = 2
        a = torch.randn(batch, M, K, device="cuda")
        constraints = [
            torch._export.dynamic_dim(a, 0) >= 1,
            torch._export.dynamic_dim(a, 0) <= 2048,
        ]
        list_example_inputs = [(a,)]
        batch = 2048
        list_example_inputs.append(
            (torch.randn(batch, M, K, device="cuda"),),
        )
        batch = 128
        list_example_inputs.append(
            (torch.randn(batch, M, K, device="cuda"),),
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
        a = torch.randn(batch, M, K, device="cuda")
        b = torch.randn(batch, K, N, device="cuda")
        constraints = [
            torch._export.dynamic_dim(a, 0) >= 1,
            torch._export.dynamic_dim(a, 0) <= 2048,
            torch._export.dynamic_dim(a, 0) == torch._export.dynamic_dim(b, 0),
        ]
        list_example_inputs = [(a, b)]
        batch = 2048
        list_example_inputs.append(
            (
                torch.randn(batch, M, K, device="cuda"),
                torch.randn(batch, K, N, device="cuda"),
            ),
        )
        batch = 128
        list_example_inputs.append(
            (
                torch.randn(batch, M, K, device="cuda"),
                torch.randn(batch, K, N, device="cuda"),
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
        a = torch.randn(128, 2048, device="cuda")
        b = torch.randn(128, 2048, device="cuda")
        constraints = [
            torch._export.dynamic_dim(a, 0) >= 1,
            torch._export.dynamic_dim(a, 0) <= 2048,
            torch._export.dynamic_dim(a, 0) == torch._export.dynamic_dim(b, 0),
        ]
        list_example_inputs = [(a, b)]
        list_example_inputs.append(
            (
                torch.randn(64, 2048, device="cuda"),
                torch.randn(64, 2048, device="cuda"),
            ),
        )
        list_example_inputs.append(
            (
                torch.randn(211, 2048, device="cuda"),
                torch.randn(211, 2048, device="cuda"),
            ),
        )
        self.check_model_with_multiple_inputs(
            model,
            list_example_inputs,
            constraints=constraints,
        )


class AOTInductorTestABICompatibile(TestCase):
    abi_compatible = True
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs


copy_tests(AOTInductorTestsTemplate, AOTInductorTestABICompatibile, "abi_compatible")


class AOTInductorTestNonABICompatible(TestCase):
    abi_compatible = False
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs


copy_tests(
    AOTInductorTestsTemplate, AOTInductorTestNonABICompatible, "non_abi_compatible"
)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    # cpp_extension N/A in fbcode
    if HAS_CUDA and not TEST_WITH_ROCM:
        run_tests(needs="filelock")
