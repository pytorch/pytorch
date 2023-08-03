# Owner(s): ["module: inductor"]


import torch

import torch._export
import torch._inductor

import torch.fx._pytree as fx_pytree

from torch.testing._internal.common_utils import TEST_WITH_ROCM, TestCase

from torch.testing._internal.inductor_utils import HAS_CUDA
from torch.utils import _pytree as pytree

aten = torch.ops.aten


class AOTInductorModelCache:
    cache = dict()

    @classmethod
    def load(cls, model, example_inputs, example_outputs):
        key = id(model)
        if key not in cls.cache:
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
            source = """
            #include <torch/csrc/inductor/aot_inductor_model.h>

            torch::aot_inductor::AOTInductorModel model;

            void run(
                    const std::vector<at::Tensor>& input_tensors,
                    std::vector<at::Tensor>& output_tensors) {
                model.run(input_tensors, output_tensors, at::cuda::getCurrentCUDAStream());
            }
            """
            module = torch.utils.cpp_extension.load_inline(
                name="aot_inductor",
                cpp_sources=[source],
                functions=["run"],
                extra_ldflags=[so_path],
                with_cuda=True,
            ).run

            value = {
                "module": module,
                "exported": exported,
                "output_tensors": output_tensors,
                "output_spec": output_spec,
            }
            cls.cache[key] = value

        return (
            cls.cache[key]["module"],
            cls.cache[key]["exported"],
            cls.cache[key]["output_tensors"],
            cls.cache[key]["output_spec"],
        )


class AotInductorTests(TestCase):
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
            torch.randn(10, 10, device="cuda"),
            torch.randn(10, 10, device="cuda"),
        ]
        expected = model(*example_inputs)

        optimized, exported, output_tensors, output_spec = AOTInductorModelCache.load(
            model, example_inputs, expected
        )
        param_buffer_values = list(exported.state_dict.values())
        flat_example_inputs = fx_pytree.tree_flatten_spec(
            example_inputs, exported.call_spec.in_spec
        )
        all_args = (*param_buffer_values, *flat_example_inputs)
        optimized(all_args, output_tensors)
        actual = pytree.tree_unflatten(output_tensors, output_spec)

        self.assertTrue(torch.allclose(actual, expected))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if HAS_CUDA and not TEST_WITH_ROCM:
        run_tests(needs="filelock")
