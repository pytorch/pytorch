# Owner(s): ["module: inductor"]
import copy
import functools
import re
import unittest
from unittest.mock import patch

import torch
import torch._export
import torch._inductor

import torch.fx._pytree as fx_pytree
from torch._dynamo.testing import same

from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import IS_FBCODE, TEST_WITH_ROCM, TestCase
from torch.testing._internal.inductor_utils import HAS_CUDA
from torch.utils import _pytree as pytree

aten = torch.ops.aten
requires_cuda = functools.partial(unittest.skipIf, not HAS_CUDA, "requires cuda")
requires_cpp_extension = functools.partial(
    unittest.skipIf, IS_FBCODE, "cpp_extension N/A in fbcode"
)


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
        #include <torch/csrc/inductor/aot_inductor_model_container.h>

        torch::aot_inductor::AOTInductorModelContainer model(1);

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
        flat_example_inputs = fx_pytree.tree_flatten_spec(
            example_inputs, exported.call_spec.in_spec
        )
        optimized(flat_example_inputs, output_tensors)
        return pytree.tree_unflatten(output_tensors, output_spec)


class AotInductorTests(TestCase):
    @requires_cpp_extension()
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

    @requires_cpp_extension()
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

    @requires_cpp_extension()
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

    @requires_cuda()
    @patch("torch._inductor.config.comment_origin", True)
    def test_inductor_sequence_nr(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(
                    in_channels=16,
                    out_channels=16,
                    kernel_size=(1, 1),
                    stride=1,
                    padding="same",
                    bias=True,
                )
                self.bn1 = torch.nn.BatchNorm2d(num_features=16)
                self.relu1 = torch.nn.ReLU()
                self.loss_fn = torch.nn.L1Loss()

            def forward(self, x, target):
                y = x
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu1(x)
                x = x + y
                x = torch.flatten(x)
                output = self.loss_fn(x, target)
                return (output,)

        def get_triton_codegen(optimized_module, args):
            def run_with_backward():
                result = optimized_module(*args)
                result[0].backward()
                return result

            res, (fwd_code, bwd_code) = run_and_get_code(run_with_backward)
            return fwd_code, bwd_code

        x = torch.rand(100, 16, 32, 32, requires_grad=True, device="cuda")
        target = torch.rand(1, device="cuda")
        args = [x, target]
        model = Model().cuda()
        opt_model = torch.compile(model)
        fwd_code, bwd_code = get_triton_codegen(opt_model, args)

        bwd_seq_nr_set = set()
        fwd_seq_nr_set = set()
        for idx, code in enumerate([fwd_code, bwd_code]):
            seq_nr_set = bwd_seq_nr_set if idx > 0 else fwd_seq_nr_set
            prefix = "BWD" if idx > 0 else "FWD"
            for line in code.split("\n"):
                if "seq_nr" in line:
                    res = re.search(r"seq_nr:(\d+)", line)
                    if res:
                        seq_nr_set.add(int(res.group(1)))

        self.assertTrue(bwd_seq_nr_set.issubset(fwd_seq_nr_set))

    @requires_cpp_extension()
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


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if HAS_CUDA and not TEST_WITH_ROCM:
        run_tests(needs="filelock")
