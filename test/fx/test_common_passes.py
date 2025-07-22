# Owner(s): ["oncall: fx"]

import itertools

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.graph_module import GraphModule
from torch.fx.passes.dialect.common.cse_pass import CSEPass
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    raise_on_run_directly,
    TestCase,
)


def FactoryFunctionCall(x, device):
    y = torch.full(x.shape, 3, device=device)
    z = torch.add(y, x)
    return z


def TorchTensorCall(x):
    y = torch.tensor(3)
    return x + y


def TakeList(x):
    z = torch.cat([x, x])
    return z


def ReturnList(x):
    a = torch.arange(10).reshape(5, 2)
    z = torch.split(a, [1, 4])
    return z


def Mutation(x):
    y = x + 2
    y.add_(1)
    return x + y


def MutationInput(x):
    x.add_(1)
    y = x + 2
    return x + y


def MutationFactory(x, device):
    y = torch.full(x.shape, 3, device=device)
    y.add_(1)
    return x + y


def MutationTorchTensorCall(x):
    y = torch.tensor(3)
    y.add_(1)
    return x + y


def MutationMetadata(x):
    x.resize_(2)
    return x


Passes = [CSEPass]
Test_Cases = [
    TakeList,
    ReturnList,
    Mutation,
    MutationInput,
    MutationMetadata,
    MutationTorchTensorCall,
]
Factory_Test_Cases = [FactoryFunctionCall, MutationFactory]
Devices = ["cpu"]
if torch.cuda.is_available():
    Devices.append("cuda")


def name_fn(common_pass, f, device):
    """Names parameterized test cases."""
    return f"{type(common_pass()).__name__}_{f.__name__}_{device}"


@instantiate_parametrized_tests
class TestCommonPass(TestCase):
    @parametrize(
        "common_pass,f,device", itertools.product(Passes, Test_Cases, Devices), name_fn
    )
    def test_correctness(self, common_pass, f, device):
        inp = torch.randn(10, device=device)

        traced_m = make_fx(f)(inp)
        P = common_pass()

        res = P(traced_m)
        modified_m = res.graph_module
        assert isinstance(modified_m, GraphModule)

        inp_copy = inp.clone()
        expected = f(inp)
        result = modified_m(inp_copy)

        self.assertEqual(result, expected)

    @parametrize(
        "common_pass,f,device",
        itertools.product(Passes, Factory_Test_Cases, Devices),
        name_fn,
    )
    def test_correctness_factory(self, common_pass, f, device):
        inp = torch.randn(10, device=device)
        traced_m = make_fx(f)(inp, device)
        P = common_pass()

        res = P(traced_m)
        modified_m = res.graph_module
        assert isinstance(modified_m, GraphModule)

        inp_copy = inp.clone()
        expected = f(inp, device)
        result = modified_m(inp_copy, device)

        self.assertEqual(result, expected)


if __name__ == "__main__":
    raise_on_run_directly("test/test_fx.py")
