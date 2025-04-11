# Owner(s): ["module: dynamo"]
from typing import Callable

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import EagerAndRecordGraphs
from torch.fx.graph_module import GraphModule


def compile_and_extract_graph(
    fn, *args, **kwargs
) -> tuple[Callable, list[torch.fx.GraphModule]]:
    backend = EagerAndRecordGraphs()
    result_fn = torch.compile(backend=backend, fullgraph=True)(fn)
    # Run fn to capture graph
    _ = result_fn(*args, **kwargs)
    return result_fn, backend.graphs


def get_num_input_nodes(graph: GraphModule) -> int:
    """Returns the number of input nodes in the input GraphModule
    by counting the number of placeholder tensors
    """
    placeholder_cnt = 0
    for node in graph.graph.nodes:
        if node.op == "placeholder" and isinstance(
            node.meta["example_value"], torch.Tensor
        ):
            placeholder_cnt += 1
    return placeholder_cnt


class SimpleLinearModule(torch.nn.Module):
    """
    Simple linear model with 1 parameter and 1 buffer
    for basic testing purposes
    """

    def __init__(self) -> None:
        super().__init__()
        self.fwd = torch.nn.Linear(5, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fwd(x)


class InstallParamsAsGraphAttrTests(torch._dynamo.test_case.TestCase):
    @torch._dynamo.config.patch(inline_inbuilt_nn_modules=True)
    @torch._dynamo.config.patch(install_params_as_graph_attr=False)
    def check_num_inputs_and_equality(
        self, original_fn, example_input, expected_num_inputs: int
    ) -> None:
        """Compiles the original fn, then:
        * Checks that the number of inputs in the graph is expected_num_inputs
        * Checks that the compiled fn and original fn are equal
        """
        opt_fn, graphs = compile_and_extract_graph(original_fn, example_input)
        self.assertEqual(len(graphs), 1, msg="Expected 1 graph (no breaks)")
        actual_num_inputs = get_num_input_nodes(graphs[0])
        self.assertEqual(actual_num_inputs, expected_num_inputs)
        self.assertEqual(opt_fn(example_input), original_fn(example_input))

    # ==================== Test Params and Buffer from NN Module ====================
    def test_optimizing_linear(self) -> None:
        net = SimpleLinearModule()
        input1 = torch.randn((1, 5))
        # Expected: 1 + 1 * 2 = 3
        self.check_num_inputs_and_equality(net, input1, 3)

    def test_breadth_linear(self) -> None:
        class BreadthModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fwd = torch.nn.Linear(1, 1)
                self.fwd2 = torch.nn.Linear(1, 1)
                self.fwd3 = torch.nn.Linear(1, 1)
                self.fwd4 = torch.nn.Linear(1, 1)
                self.fwd5 = torch.nn.Linear(1, 1)

            def forward(self, x) -> torch.Tensor:
                return (
                    self.fwd(x)
                    + self.fwd2(x)
                    + self.fwd3(x)
                    + self.fwd4(x)
                    + self.fwd5(x)
                )

        net = BreadthModel()
        input1 = torch.randn((1, 1))
        # Expected: 1 + 5 * 2 = 11
        self.check_num_inputs_and_equality(net, input1, 11)

    def test_nested_linear(self) -> None:
        class NestedModel(torch.nn.Module):
            def __init__(self, inner_module) -> None:
                super().__init__()
                self.fwd = torch.nn.Linear(1, 1)
                self.inner_module = inner_module

            def forward(self, x) -> torch.Tensor:
                return self.fwd(self.inner_module(x))

        # Nest 5x
        kDepth = 4
        net = SimpleLinearModule()
        for _ in range(kDepth):
            net = NestedModel(net)
        input1 = torch.randn((1, 5))
        self.check_num_inputs_and_equality(net, input1, 1 + 2 * (kDepth + 1))

    # TODO[@lucaskabela]: Test nontrivial such as resnet, ffn, or transformer

    # ==================== Test Parameters and Buffers as input ====================
    def test_optimizing_params_in_input(self) -> None:
        param = torch.nn.Parameter(torch.randn(1, 5))
        net = SimpleLinearModule()

        def test_fn(x):
            return net(x)

        self.check_num_inputs_and_equality(test_fn, param, 3)

    def test_optimizing_buffer_in_input(self) -> None:
        buf = torch.nn.Buffer(data=torch.ones((1, 5)))
        net = SimpleLinearModule()

        def test_fn(x) -> torch.Tensor:
            return net(x)

        self.check_num_inputs_and_equality(test_fn, buf, 3)

    def test_optimizing_buffer_and_param_in_input(self) -> None:
        param = torch.nn.Parameter(torch.randn(5, 1))
        buf = torch.nn.Buffer(data=torch.ones((1, 1)))
        x = torch.randn(1, 5)

        def test_linear(x: torch.Tensor) -> torch.Tensor:
            return param * x + buf

        self.check_num_inputs_and_equality(test_linear, x, 3)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
