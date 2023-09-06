# Owner(s): ["module: inductor"]
import copy
import functools
import unittest

import torch
import torch._export
import torch._inductor

import torch.fx._pytree as fx_pytree
from torch._dynamo.testing import same

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

        # Use a utility function for easier testing
        source = """
        #include <torch/csrc/inductor/aot_inductor_model.h>

        torch::aot_inductor::AOTInductorModel model;

        void run(
                const std::vector<at::Tensor>& input_tensors,
                std::vector<at::Tensor>& output_tensors) {
            model.run(input_tensors, output_tensors, at::cuda::getCurrentCUDAStream());
        }
        """
        optimized = torch.utils.cpp_extension.load_inline(
            name="aot_inductor",
            cpp_sources=[source],
            functions=["run"],
            extra_ldflags=[so_path],
            with_cuda=True,
        ).run

        return optimized, exported, output_tensors, output_spec

    @classmethod
    def run(cls, model, example_inputs, example_outputs, options=None):
        example_outputs = copy.deepcopy(example_outputs)
        optimized, exported, output_tensors, output_spec = AOTInductorModelRunner.load(
            model, example_inputs, example_outputs, options
        )
        param_buffer_values = list(exported.state_dict.values())
        flat_example_inputs = fx_pytree.tree_flatten_spec(
            example_inputs, exported.call_spec.in_spec
        )
        all_args = (*param_buffer_values, *flat_example_inputs)
        optimized(all_args, output_tensors)
        return pytree.tree_unflatten(output_tensors, output_spec)


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
        actual = AOTInductorModelRunner.run(model, example_inputs, expected)
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
        actual = AOTInductorModelRunner.run(model, example_inputs, expected)
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
        actual = AOTInductorModelRunner.run(model, example_inputs, expected)
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
        actual = AOTInductorModelRunner.run(
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
        actual = AOTInductorModelRunner.run(model, example_inputs, expected)
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
