# Owner(s): ["module: dynamo"]
import copy
from typing import Tuple
import unittest

import torch  # noqa: F401
import torch.nn as nn
import torch._dynamo as torchdynamo
from functorch import make_fx
from functorch.experimental import functionalize, control_flow
from torch import Tensor
from torch.testing._internal.common_utils import run_tests, TestCase
from torch._dynamo.eval_frame import is_dynamo_supported

from torch._export.verifier import (
    SpecViolationError,
    Verifier,
    ATenDialectVerifier,
)


@torch.no_grad()
def capture(f, args):
    torchdynamo.config.allow_rnn = True
    torchdynamo.reset()
    graphmodule, _ = torchdynamo.export(
        f,
        *copy.deepcopy(args),
        aten_graph=True,
    )

    def graph_with_interpreter(*args):
        with torch.fx.traceback.preserve_node_meta():
            return torch.fx.Interpreter(graphmodule).run(*args)

    functionalized_callable = functionalize(
        graph_with_interpreter,
        remove='mutations_and_views',
    )
    gm = make_fx(functionalized_callable, tracing_mode='fake', _allow_non_fake_inputs=True)(*args)
    return gm


class Transpose(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, dim0: int, dim1: int) -> Tensor:
        return x.transpose(dim0, dim1)


class Mul(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor, other: Tensor) -> Tensor:
        # or return torch.mul(input, other)
        return input * other

    def get_random_inputs(self) -> Tuple[Tensor, Tensor]:
        return (torch.randn(3, 2), torch.randn(3, 2))


class ElementwiseAdd(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return x + y

    def get_random_inputs(self) -> Tuple[Tensor, Tensor]:
        return (torch.randn(1, 3), torch.randn(1, 3))


class Cat(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    # def forward(self, tensors, dim=0):
    def forward(self, *args: Tensor, dim: int) -> Tensor:
        tensors = args[:-1]
        return torch.cat(tensors, dim)


class FeedForwardBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.layer_norm = nn.LayerNorm(input_dim)

        self.relu = nn.ReLU()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout()

        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.dropout2 = nn.Dropout()

    def forward(self, x: Tensor) -> Tensor:
        # LayerNorm -> Linear -> Dropout -> ReLU -> Linear -> Dropout
        y = self.layer_norm(x)
        y = self.linear1(y)
        y = self.dropout1(y)
        y = self.relu(y)
        y = self.linear2(y)
        y = self.dropout2(y)
        return y

class ControlFlow(nn.Module):

    def forward(self, pred: Tensor, x: Tensor) -> Tensor:
        return control_flow.cond(pred, lambda x: x.sin(), lambda x: x.cos(), (x,))


class VerifierTest(TestCase):

    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    def test_verifier(self) -> None:
        m = ElementwiseAdd()
        egm = capture(m, (torch.randn(100), torch.randn(100)))
        # assert not throw
        verifier = Verifier()
        verifier(egm)
        self.assertTrue(verifier.is_valid(egm))

    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    def test_verifier_call_module(self) -> None:
        m = FeedForwardBlock(10, 10)
        gm = torch.fx.symbolic_trace(m)
        # this would have modules that are not delegates
        verifier = Verifier()
        with self.assertRaises(SpecViolationError):
            verifier(gm)
        self.assertFalse(verifier.is_valid(gm))

    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    def test_verifier_no_functional(self) -> None:
        m = ElementwiseAdd()
        egm = capture(m, (torch.randn(100), torch.randn(100)))
        for node in egm.graph.nodes:
            if node.target == torch.ops.aten.add.Tensor:
                node.target = torch.ops.aten.add.out
        verifier = Verifier()
        with self.assertRaises(SpecViolationError):
            verifier(egm)
        self.assertFalse(verifier.is_valid(egm))

    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    def test_aten_dialect(self) -> None:
        m = ElementwiseAdd()
        egm = capture(m, (torch.randn(100), torch.randn(100)))
        verifier = ATenDialectVerifier()
        verifier(egm)
        self.assertTrue(verifier.is_valid(egm))

    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    def test_aten_wrong_mem_format(self) -> None:
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.parameter.Parameter(
                    torch.randn(1, 3, 100, 100).to(memory_format=torch.channels_last)
                )

            def forward(self, x):
                return self.a + x

        m = TestModel()
        egm = capture(m, (torch.randn(1, 3, 100, 100),))
        egm._apply(lambda t: t.to(memory_format=torch.channels_last))
        verifier = ATenDialectVerifier()
        with self.assertRaises(SpecViolationError):
            verifier(egm)
        self.assertFalse(verifier.is_valid(egm))

    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    def test_aten_wrong_mem_format_buffer(self) -> None:
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "a",
                    torch.randn(1, 3, 100, 100).to(memory_format=torch.channels_last),
                )

            def forward(self, x):
                return self.a + x

        m = TestModel()
        egm = capture(m, (torch.randn(1, 3, 100, 100),))
        egm._apply(lambda t: t.to(memory_format=torch.channels_last))
        verifier = ATenDialectVerifier()
        with self.assertRaises(SpecViolationError):
            verifier(egm)
        self.assertFalse(verifier.is_valid(egm))

    def test_aten_wrong_op(self) -> None:
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

    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    def test_verifier_control_flow_success(self) -> None:
        m = ControlFlow()
        gm = torch._export.export(m, (torch.tensor(True), torch.randn(3, 4))).graph_module
        # No error should be raised
        verifier = ATenDialectVerifier()
        verifier(gm)


if __name__ == '__main__':
    run_tests()
