# Owner(s): ["module: inductor"]
import copy
import functools
import unittest

import torch
import torch._export
import torch._inductor

import torch.fx._pytree as fx_pytree
from torch._dynamo.testing import same
from torch._inductor.utils import aot_inductor_launcher
from torch.testing import FileCheck

from torch.testing._internal.common_utils import IS_FBCODE, TEST_WITH_ROCM, TestCase
from torch.testing._internal.inductor_utils import HAS_CUDA
from torch.utils import _pytree as pytree

aten = torch.ops.aten
requires_cuda = functools.partial(unittest.skipIf, not HAS_CUDA, "requires cuda")


class AOTInductorModelRunner:
    @classmethod
    def load(cls, model, example_inputs, example_outputs, options=None):
        # AOTInductorModel relies on the caller to pass in output_tensors,
        # so we need to explicitly allocate output tensors here.
        output_tensors = []
        example_outputs, output_spec = pytree.tree_flatten(example_outputs)
        for output in example_outputs:
            output_tensors.append(torch.empty_like(output))

        # The exact API is subject to change
        so_path, exported = torch._export.aot_compile(
            model,
            example_inputs,
            options=options,
        )
        compiled_cpp = so_path.replace(".so", ".cpp")

        optimized = torch.utils.cpp_extension.load_inline(
            name="aot_inductor",
            cpp_sources=[aot_inductor_launcher],
            functions=["run"],
            extra_ldflags=[so_path],
            with_cuda=True,
        ).run

        return optimized, exported, output_tensors, output_spec, compiled_cpp

    @classmethod
    def run(cls, model, example_inputs, example_outputs, options=None):
        example_outputs = copy.deepcopy(example_outputs)
        (
            optimized,
            exported,
            output_tensors,
            output_spec,
            compiled_cpp,
        ) = AOTInductorModelRunner.load(model, example_inputs, example_outputs, options)
        flat_example_inputs = fx_pytree.tree_flatten_spec(
            example_inputs, exported.call_spec.in_spec
        )
        optimized(flat_example_inputs, output_tensors)
        return pytree.tree_unflatten(output_tensors, output_spec), compiled_cpp


class AotInductorTests(TestCase):
    def test_simple(self):
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.randn(10, 10, device="cuda")

            def forward(self, x, y):
                return x + torch.nn.functional.linear(y, self.weight)

        model = Repro()
        example_inputs = (
            torch.randn(10, 10, device="cuda"),
            torch.randn(10, 10, device="cuda"),
        )
        expected = model(*example_inputs)
        actual, compiled_cpp = AOTInductorModelRunner.run(
            model, example_inputs, expected
        )
        self.assertTrue(same(actual, expected))

    def test_large(self):
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.randn(250112, 512, device="cuda")

            def forward(self, x, y):
                return x + torch.nn.functional.linear(y, self.weight)

        model = Repro()
        example_inputs = (
            torch.randn(1, 250112, device="cuda"),
            torch.randn(1, 512, device="cuda"),
        )
        expected = model(*example_inputs)
        actual, compiled_cpp = AOTInductorModelRunner.run(
            model, example_inputs, expected
        )
        self.assertTrue(same(actual, expected))

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

        model = Repro()
        example_inputs = (
            torch.randn(10, 10, device="cuda"),
            torch.randn(10, 10, device="cuda"),
        )
        expected = model(*example_inputs)
        actual, compiled_cpp = AOTInductorModelRunner.run(
            model, example_inputs, expected
        )
        self.assertTrue(same(actual, expected))

    def test_missing_output(self):
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                a = torch.sin(x)
                b = torch.mm(a, y)
                c = torch.cos(b)
                return c

        model = Repro()
        example_inputs = (
            torch.randn(10, 10, device="cuda"),
            torch.randn(10, 10, device="cuda"),
        )
        expected = model(*example_inputs)
        actual, compiled_cpp = AOTInductorModelRunner.run(
            model, example_inputs, expected
        )
        self.assertTrue(same(actual, expected))

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

        model = Repro()
        example_inputs = (
            torch.randn(10, 10, device="cuda"),
            torch.randn(10, 10, device="cuda"),
        )
        expected = model(*example_inputs)
        actual, compiled_cpp = AOTInductorModelRunner.run(
            model, example_inputs, expected
        )
        self.assertTrue(same(actual, expected))

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
        expected = model(*example_inputs)
        actual, compiled_cpp = AOTInductorModelRunner.run(
            model,
            example_inputs,
            expected,
            options={
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
            },
        )
        self.assertTrue(same(actual, expected))

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
        expected = model(*example_inputs)
        actual, compiled_cpp = AOTInductorModelRunner.run(
            model, example_inputs, expected
        )
        self.assertTrue(same(actual, expected))

    def test_convolution(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.randn((32, 16, 8), device="cuda")
                self.bias = torch.randn((16), device="cuda")

            def forward(self, x):
                return (
                    aten.convolution(
                        x, self.weight, self.bias, [4], [0], [1], True, [0], 1
                    ),
                )

        model = Model()
        example_inputs = (torch.randn((2, 32, 90), device="cuda"),)
        expected = model(*example_inputs)
        actual, compiled_cpp = AOTInductorModelRunner.run(
            model, example_inputs, expected
        )
        self.assertTrue(same(actual, expected))

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

        model = Repro()
        example_inputs = (
            torch.randn(4, 2, device="cuda"),
            torch.randn(4, 2, device="cuda"),
        )
        expected = model(*example_inputs)
        actual, compiled_cpp = AOTInductorModelRunner.run(
            model, example_inputs, expected
        )
        with open(compiled_cpp) as f:
            src_code = f.read()
            FileCheck().check_count(
                "aoti_torch_empty_strided(",
                3,
                exactly=True,
            ).run(src_code)
            FileCheck().check_count(
                "aoti_torch_free_tensor_storage(",
                3,
                exactly=True,
            ).run(src_code)
        self.assertTrue(same(actual, expected))

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

        model = Repro()
        example_inputs = (
            torch.randn(4, 4, device="cuda"),
            torch.randn(4, 4, device="cuda"),
        )
        expected = model(*example_inputs)
        actual, compiled_cpp = AOTInductorModelRunner.run(
            model, example_inputs, expected
        )
        with open(compiled_cpp) as f:
            src_code = f.read()
            FileCheck().check_count(
                "aoti_torch_empty_strided(",
                3,
                exactly=True,
            ).run(src_code)
            FileCheck().check_count(
                "aoti_torch_free_tensor_storage(",
                3,
                exactly=True,
            ).run(src_code)
        self.assertTrue(same(actual, expected))

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


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    # cpp_extension N/A in fbcode
    if HAS_CUDA and not TEST_WITH_ROCM and not IS_FBCODE:
        run_tests(needs="filelock")
