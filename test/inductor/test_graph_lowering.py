# Owner(s): ["module: inductor"]

import operator
from unittest.mock import patch

import torch
from torch._higher_order_ops import triton_kernel_wrap
from torch._inductor.graph import GraphLowering
from torch.fx.graph import _BoxedCodeGen, _PyTreeCodeGen, _PyTreeInfo
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._pytree import tree_structure


@instantiate_parametrized_tests
class TestGraphLowering(TestCase):
    @parametrize("codegen", ("plain", "boxed", "pytree"))
    def test_extract_autotune_inputs(self, codegen):
        graph = torch.fx.Graph()
        tensor = graph.placeholder("tensor")
        blocks = graph.placeholder("blocks")
        kernel_input = graph.call_function(torch.square, (tensor,))
        grid_size = graph.call_function(operator.mul, (blocks, 2))
        kernel = triton_kernel_wrap.triton_kernel_wrapper_mutation
        launch = graph.call_function(
            kernel,
            kwargs={
                "kernel_idx": 0,
                "constant_args_idx": 0,
                "grid": [(grid_size, 1, 1)],
                "kwargs": {"data": kernel_input},
                "tma_descriptor_metadata": {},
            },
        )
        graph.output((kernel_input,))

        example = torch.arange(3, dtype=torch.float32)
        inputs = [example, 3]
        kernel_input.meta["val"] = example.square()
        if codegen == "boxed":
            graph.set_codegen(_BoxedCodeGen())
        elif codegen == "pytree":
            graph.set_codegen(
                _PyTreeCodeGen(
                    _PyTreeInfo(
                        ["sample"],
                        tree_structure(((inputs,), {})),
                        tree_structure({"result": example}),
                    )
                )
            )

        lowering = GraphLowering(torch.fx.GraphModule({}, graph))
        call_function = torch.fx.Interpreter.call_function

        def call_without_triton(interpreter, target, args, kwargs):
            if target is kernel:
                return None
            return call_function(interpreter, target, args, kwargs)

        with (
            patch.object(triton_kernel_wrap, "get_mutated_tensors", return_value=set()),
            patch.object(torch.fx.Interpreter, "call_function", call_without_triton),
        ):
            lowering.extract_autotune_inputs(inputs)

        self.assertEqual(lowering.autotuning_inputs, (example.square(),))
        self.assertEqual(lowering.autotuning_mapping, {launch.name: {"data": 0}})
        self.assertEqual(lowering.autotuning_grids, {launch.name: [(6, 1, 1)]})


if __name__ == "__main__":
    run_tests()
