# Owner(s): ["module: dynamo"]
import unittest

import torch
from functorch.experimental import control_flow
from torch import Tensor
from torch.testing._internal.common_utils import run_tests, TestCase
from torch._dynamo.eval_frame import is_dynamo_supported

from torch._export.verifier import (
    SpecViolationError,
    Verifier,
    ATenDialectVerifier,

)
from torch._export import export

@unittest.skipIf(not is_dynamo_supported(), "dynamo isn't supported")
class TestVerifier(TestCase):
    def test_verifier_basic(self) -> None:
        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

        ep = export(f, (torch.randn(100), torch.randn(100)))

        verifier = Verifier()
        verifier(ep.graph_module)

    def test_verifier_call_module(self) -> None:
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x: Tensor) -> Tensor:
                return self.linear(x)

        gm = torch.fx.symbolic_trace(M())

        verifier = Verifier()
        with self.assertRaises(SpecViolationError):
            verifier(gm)

    def test_verifier_no_functional(self) -> None:
        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

        ep = export(f, (torch.randn(100), torch.randn(100)))
        for node in ep.graph.nodes:
            if node.target == torch.ops.aten.add.Tensor:
                node.target = torch.ops.aten.add_.Tensor

        verifier = Verifier()
        with self.assertRaises(SpecViolationError):
            verifier(ep.graph_module)

    def test_verifier_higher_order(self) -> None:
        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            def true_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

            def false_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x - y

            return control_flow.cond(
                x.shape[0] > 2, true_fn, false_fn, [x, y]
            )

        ep = export(f, (torch.randn(3, 3), torch.randn(3, 3)))

        verifier = Verifier()
        verifier(ep.graph_module)

    def test_verifier_nested_invalid_module(self) -> None:
        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            def true_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

            def false_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x - y

            return control_flow.cond(
                x.shape[0] > 2, true_fn, false_fn, [x, y]
            )

        ep = export(f, (torch.randn(3, 3), torch.randn(3, 3)))
        for node in ep.graph_module.true_graph_0.graph.nodes:
            if node.target == torch.ops.aten.add.Tensor:
                node.target = torch.ops.aten.add_.Tensor

        verifier = Verifier()
        with self.assertRaises(SpecViolationError):
            verifier(ep.graph_module)

    def test_aten_verifier_wrong_op(self) -> None:
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.ops.aten._add_relu(x, x)

        m = TestModel()
        egm = torch.fx.symbolic_trace(m)
        verifier = ATenDialectVerifier()
        with self.assertRaises(SpecViolationError):
            verifier(egm)
        self.assertFalse(verifier.is_valid(egm))

    def test_ep_verifier_basic(self) -> None:
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x: Tensor) -> Tensor:
                return self.linear(x)

        ep = export(M(), (torch.randn(10, 10),))
        ep._validate()

    def test_ep_verifier_invalid_param(self) -> None:
        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

        ep = export(f, (torch.randn(100), torch.randn(100)))

        # Parameter doesn't exist in the state dict
        ep.graph_signature.parameters.append("bad_param")
        with self.assertRaisesRegex(SpecViolationError, "not in the state dict"):
            ep._validate()

        # Add non-torch.nn.Parameter parameter to the state dict
        ep.state_dict["bad_param"] = torch.randn(100)
        with self.assertRaisesRegex(
            SpecViolationError, "not an instance of torch.nn.Parameter"
        ):
            ep._validate()

        # Add torch.nn.Parameter to state dict, but this should still error
        # because there are an incorrect number of placeholder nodes
        ep.state_dict["bad_param"] = torch.nn.Parameter(torch.randn(100))
        with self.assertRaisesRegex(
            SpecViolationError, "not found in the exported program's parameter list"
        ):
            ep._validate()

    def test_ep_verifier_invalid_buffer(self) -> None:
        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

        ep = export(f, (torch.randn(100), torch.randn(100)))

        # Buffer doesn't exist in the state dict
        ep.graph_signature.buffers.append("bad_buffer")
        with self.assertRaisesRegex(SpecViolationError, "not in the state dict"):
            ep._validate()

        # Incorrect number of placeholder nodes
        ep.state_dict["bad_buffer"] = torch.randn(100)
        with self.assertRaisesRegex(
            SpecViolationError, "not found in the exported program's buffer list"
        ):
            ep._validate()

    def test_ep_verifier_buffer_mutate(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.my_parameter = torch.nn.Parameter(torch.tensor(2.0))

                self.register_buffer("my_buffer1", torch.tensor(3.0))
                self.register_buffer("my_buffer2", torch.tensor(4.0))

            def forward(self, x1, x2):
                # Use the parameter, buffers, and both inputs in the forward method
                output = (
                    x1 + self.my_parameter
                ) * self.my_buffer1 + x2 * self.my_buffer2

                # Mutate one of the buffers (e.g., increment it by 1)
                self.my_buffer2.add_(1.0)
                return output

        ep = export(M(), (torch.tensor(5.0), torch.tensor(6.0)))
        ep._validate()

    def test_ep_verifier_invalid_output(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.my_parameter = torch.nn.Parameter(torch.tensor(2.0))

                self.register_buffer("my_buffer1", torch.tensor(3.0))
                self.register_buffer("my_buffer2", torch.tensor(4.0))

            def forward(self, x1, x2):
                # Use the parameter, buffers, and both inputs in the forward method
                output = (
                    x1 + self.my_parameter
                ) * self.my_buffer1 + x2 * self.my_buffer2

                # Mutate one of the buffers (e.g., increment it by 1)
                self.my_buffer2.add_(1.0)
                return output

        ep = export(M(), (torch.tensor(5.0), torch.tensor(6.0)))

        output_node = list(ep.graph.nodes)[-1]
        with ep.graph.inserting_before(output_node):
            additional_output_node = ep.graph.call_function(
                torch.add, args=(output_node.args[0][0], output_node.args[0][0])
            )
            output_node.args = (
                (
                    output_node.args[0][0],
                    additional_output_node,
                    output_node.args[0][1],
                ),
            )

        with self.assertRaisesRegex(SpecViolationError, "Number of output nodes"):
            ep._validate()


if __name__ == '__main__':
    run_tests()
