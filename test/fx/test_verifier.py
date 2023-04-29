# Owner(s): ["module: fx"]
import os
import sys
import unittest
from torch.fx.verifier import (
    SpecViolationError,
    check_valid_aten_dialect,
    check_valid,
    is_valid_aten_dialect,
    is_valid,
)


from typing import Tuple


from torch.testing._internal.common_utils import TestCase
import torch  # noqa: F401
import torch.nn as nn
from torch import Tensor
import torch._dynamo as torchdynamo
import copy
from functorch import make_fx
from functorch.experimental import functionalize


@torch.no_grad()
def capture(f, args):
    torchdynamo.config.capture_scalar_outputs = True
    torchdynamo.config.guard_nn_modules = True
    torchdynamo.config.dynamic_shapes = True
    torchdynamo.config.allow_rnn = True
    torchdynamo.config.verbose = True
    torchdynamo.reset()
    graphmodule, _ = torchdynamo.export(
        f,
        *copy.deepcopy(args),
        aten_graph=True,
        tracing_mode='fake',
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


def skip_condition():
    return sys.version_info >= (3, 11) or os.name == 'nt'

class VerifierTest(TestCase):

    @unittest.skipIf(skip_condition(), "dynamo doesnt support 3.11")
    def test_verifier(self) -> None:
        m = ElementwiseAdd()
        egm = capture(m, (torch.randn(100), torch.randn(100)))
        # assert not throw
        check_valid(egm)
        self.assertTrue(is_valid(egm))

    @unittest.skipIf(skip_condition(), "dynamo doesnt support 3.11")
    def testr_verifier_call_module(self) -> None:
        m = FeedForwardBlock(10, 10)
        gm = torch.fx.symbolic_trace(m)
        # this would have modules that are not delegates
        with self.assertRaises(SpecViolationError):
            check_valid(gm)
        self.assertFalse(is_valid(gm))

    @unittest.skipIf(skip_condition(), "dynamo doesnt support 3.11")
    def test_verifier_no_functional(self) -> None:
        m = ElementwiseAdd()
        egm = capture(m, (torch.randn(100), torch.randn(100)))
        for node in egm.graph.nodes:
            if node.target == torch.ops.aten.add.Tensor:
                node.target = torch.ops.aten.add.out
        with self.assertRaises(SpecViolationError):
            check_valid(egm)
        self.assertFalse(is_valid(egm))

    @unittest.skipIf(skip_condition(), "dynamo doesnt support 3.11")
    def test_aten_dialect(self) -> None:
        m = ElementwiseAdd()
        egm = capture(m, (torch.randn(100), torch.randn(100)))
        check_valid_aten_dialect(egm)
        self.assertTrue(is_valid_aten_dialect(egm))

    @unittest.skipIf(skip_condition(), "dynamo doesnt support 3.11")
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
        with self.assertRaises(SpecViolationError):
            check_valid_aten_dialect(egm)
        self.assertFalse(is_valid_aten_dialect(egm))

    @unittest.skipIf(skip_condition(), "dynamo doesnt support 3.11")
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
        with self.assertRaises(SpecViolationError):
            check_valid_aten_dialect(egm)
        self.assertFalse(is_valid_aten_dialect(egm))


if __name__ == '__main__':
    unittest.main()
