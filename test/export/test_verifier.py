# Owner(s): ["oncall: export"]
import unittest

import torch
from functorch.experimental import control_flow
from torch import Tensor
from torch._dynamo.eval_frame import is_dynamo_supported
from torch._export.verifier import SpecViolationError, Verifier
from torch.export import export
from torch.export.exported_program import InputKind, InputSpec, TensorArgument
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests, TestCase


@unittest.skipIf(not is_dynamo_supported(), "dynamo isn't supported")
class TestVerifier(TestCase):
    def test_verifier_basic(self) -> None:
        class Foo(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

        f = Foo()

        ep = export(f, (torch.randn(100), torch.randn(100)))

        verifier = Verifier()
        verifier.check(ep)

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
            verifier._check_graph_module(gm)

    def test_verifier_no_functional(self) -> None:
        class Foo(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

        f = Foo()

        ep = export(f, (torch.randn(100), torch.randn(100)))
        for node in ep.graph.nodes:
            if node.target == torch.ops.aten.add.Tensor:
                node.target = torch.ops.aten.add_.Tensor

        verifier = Verifier()
        with self.assertRaises(SpecViolationError):
            verifier.check(ep)

    @unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
    def test_verifier_higher_order(self) -> None:
        class Foo(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                def true_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                    return x + y

                def false_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                    return x - y

                return control_flow.cond(x.sum() > 2, true_fn, false_fn, [x, y])

        f = Foo()

        ep = export(f, (torch.randn(3, 3), torch.randn(3, 3)))

        verifier = Verifier()
        verifier.check(ep)

    @unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
    def test_verifier_nested_invalid_module(self) -> None:
        class Foo(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                def true_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                    return x + y

                def false_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                    return x - y

                return control_flow.cond(x.sum() > 2, true_fn, false_fn, [x, y])

        f = Foo()

        ep = export(f, (torch.randn(3, 3), torch.randn(3, 3)))
        for node in ep.graph_module.true_graph_0.graph.nodes:
            if node.target == torch.ops.aten.add.Tensor:
                node.target = torch.ops.aten.add_.Tensor

        verifier = Verifier()
        with self.assertRaises(SpecViolationError):
            verifier.check(ep)

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
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_parameter(
                    name="a", param=torch.nn.Parameter(torch.randn(100))
                )

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y + self.a

        ep = export(M(), (torch.randn(100), torch.randn(100)))

        # Parameter doesn't exist in the state dict
        ep.graph_signature.input_specs[0] = InputSpec(
            kind=InputKind.PARAMETER, arg=TensorArgument(name="p_a"), target="bad_param"
        )
        with self.assertRaisesRegex(SpecViolationError, "not in the state dict"):
            ep._validate()

        # Add non-torch.nn.Parameter parameter to the state dict
        ep.state_dict["bad_param"] = torch.randn(100)
        with self.assertRaisesRegex(
            SpecViolationError, "not an instance of torch.nn.Parameter"
        ):
            ep._validate()

    def test_ep_verifier_invalid_buffer(self) -> None:
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.tensor(3.0)

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y + self.a

        ep = export(M(), (torch.randn(100), torch.randn(100)))

        # Buffer doesn't exist in the state dict
        ep.graph_signature.input_specs[0] = InputSpec(
            kind=InputKind.BUFFER,
            arg=TensorArgument(name="c_a"),
            target="bad_buffer",
            persistent=True,
        )
        with self.assertRaisesRegex(SpecViolationError, "not in the state dict"):
            ep._validate()

    def test_ep_verifier_buffer_mutate(self) -> None:
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

                self.my_parameter = torch.nn.Parameter(torch.tensor(2.0))

                self.my_buffer1 = torch.nn.Buffer(torch.tensor(3.0))
                self.my_buffer2 = torch.nn.Buffer(torch.tensor(4.0))

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
            def __init__(self) -> None:
                super().__init__()

                self.my_parameter = torch.nn.Parameter(torch.tensor(2.0))

                self.my_buffer1 = torch.nn.Buffer(torch.tensor(3.0))
                self.my_buffer2 = torch.nn.Buffer(torch.tensor(4.0))

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
        output_node.args = (
            (
                output_node.args[0][0],
                next(iter(ep.graph.nodes)),
                output_node.args[0][1],
            ),
        )

        with self.assertRaisesRegex(SpecViolationError, "Number of output nodes"):
            ep._validate()


if __name__ == "__main__":
    run_tests()
