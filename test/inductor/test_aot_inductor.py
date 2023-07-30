# Owner(s): ["module: inductor"]


import copy

import torch
import torch._export
import torch._inductor
import torch.fx._pytree as fx_pytree
from torch._dynamo.testing import same
from torch._inductor import config

from torch.testing._internal.common_utils import TEST_WITH_ROCM, TestCase
from torch.testing._internal.inductor_utils import HAS_CUDA
from torch.utils import _pytree as pytree

try:
    try:
        from . import test_torchinductor
    except ImportError:
        import test_torchinductor
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise


aten = torch.ops.aten


class AOTInductorModelRunner:
    @classmethod
    def load(cls, model, example_inputs, example_outputs):
        # AOTInductorModel relies on the caller to pass in output_tensors,
        # so we need to explicitly allocate output tensors here.
        output_tensors = []
        example_outputs, output_spec = pytree.tree_flatten(example_outputs)
        for output in example_outputs:
            output_tensors.append(torch.empty_like(output))

        # The exact API is subject to change
        exported = torch._export.export(model, example_inputs)
        param_buffer_values = list(exported.state_dict.values())
        flat_example_inputs = fx_pytree.tree_flatten_spec(
            example_inputs, exported.call_spec.in_spec
        )
        all_args = (*param_buffer_values, *flat_example_inputs)
        # AOT compile into a .so
        so_path = torch._inductor.aot_compile(exported.graph_module, all_args)

        # Use a utility function for easier testing
        if config.aot_inductor.abi_compatible:
            source = """
            #include <torch/csrc/aot_inductor/model.h>
            #include <c10/cuda/CUDAStream.h>

            torch::aot_inductor::AOTInductorModel model;

            void run(
                    std::vector<at::Tensor>& input_tensors,
                    std::vector<at::Tensor>& output_tensors) {
                aot_inductor_initialize();

                std::vector<AOTInductorTensorHandle> inputs;
                for (auto& input : input_tensors) {
                    inputs.push_back(convert_input_output_to_aot_tensor(&input));
                }
                std::vector<AOTInductorTensorHandle> outputs;
                for (auto& output : output_tensors) {
                    outputs.push_back(convert_input_output_to_aot_tensor(&output));
                }
                model.run(inputs, outputs, at::cuda::getCurrentCUDAStream());
                aot_inductor_destroy();
            }
            """
        else:
            source = """
            #include <torch/csrc/aot_inductor/model.h>
            #include <c10/cuda/CUDAStream.h>

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
            extra_cflags=[
                "-D AOT_INDUCTOR_ABI_COMPATIBLE"
                if config.aot_inductor.abi_compatible
                else ""
            ],
            extra_ldflags=[so_path],
            with_cuda=True,
        ).run

        return optimized, exported, output_tensors, output_spec

    @classmethod
    def run(cls, model, example_inputs, example_outputs):
        example_outputs = copy.deepcopy(example_outputs)
        optimized, exported, output_tensors, output_spec = AOTInductorModelRunner.load(
            model, example_inputs, example_outputs
        )
        param_buffer_values = list(exported.state_dict.values())
        flat_example_inputs = fx_pytree.tree_flatten_spec(
            example_inputs, exported.call_spec.in_spec
        )
        all_args = (*param_buffer_values, *flat_example_inputs)
        optimized(all_args, output_tensors)
        return pytree.tree_unflatten(output_tensors, output_spec)


def check_model(
    self: TestCase,
    model,
    example_inputs,
):
    expected = model(*example_inputs)
    with config.patch("aot_inductor.abi_compatible", self.abi_compatible):
        actual = AOTInductorModelRunner.run(model, example_inputs, expected)
    self.assertTrue(same(actual, expected))


class AOTInductorTestsTemplate:
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
        example_inputs = [
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        ]
        self.check_model(model, example_inputs)

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
        example_inputs = [
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        ]
        self.check_model(model, example_inputs)

    def test_addmm_activation(self):
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, mat1, mat2):
                a = aten._addmm_activation(x, mat1, mat2)
                return a

        model = Repro()
        example_inputs = [
            torch.randn(32, 32, device=self.device),
            torch.randn(32, 10, device=self.device),
            torch.randn(10, 32, device=self.device),
        ]
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

        model = Repro()
        example_inputs = [
            torch.randn(4, 2, device=self.device),
            torch.randn(4, 2, device=self.device),
        ]
        self.check_model(model, example_inputs)

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
        example_inputs = [
            torch.randn(4, 4, device=self.device),
            torch.randn(4, 4, device=self.device),
        ]
        self.check_model(model, example_inputs)


class AOTInductorAbiCompatibile(TestCase):
    abi_compatible = True
    device = "cuda"
    check_model = check_model


test_torchinductor.copy_tests(
    AOTInductorTestsTemplate, AOTInductorAbiCompatibile, "abi_compatible"
)


class AOTInductorNotAbiCompatibile(TestCase):
    abi_compatible = False
    device = "cuda"
    check_model = check_model


test_torchinductor.copy_tests(
    AOTInductorTestsTemplate, AOTInductorNotAbiCompatibile, "not_abi_compatible"
)


if __name__ == "__main__":
    if HAS_CUDA and not TEST_WITH_ROCM:
        from torch._dynamo.test_case import run_tests

        run_tests(needs="filelock")
