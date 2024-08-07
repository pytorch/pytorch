# Owner(s): ["module: inductor"]
import torch
from torch import _dynamo as dynamo, _inductor as inductor
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import gen_gm_and_inputs
from torch.fx import symbolic_trace
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.inductor_utils import HAS_CPU


class MyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Linear(10, 10)
        self.b = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.a(x))
        x = torch.sigmoid(self.b(x))
        return x


class MyModule2(MyModule):
    def forward(self, x):  # takes a dict of list
        a, b = x["key"]
        return {"result": super().forward(a) + b}


class MyModule3(MyModule):
    def forward(self, x):
        return (super().forward(x),)


class TestStandaloneInductor(TestCase):
    """
    These test check that you can call TorchInductor directly without
    going through TorchDynamo.
    """

    def test_inductor_via_fx(self):
        mod = MyModule3().eval()
        inp = torch.randn(10)
        correct = mod(inp)
        mod_opt = inductor.compile(symbolic_trace(mod), [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_fx_tensor_return(self):
        mod = MyModule().eval()
        inp = torch.randn(10)
        correct = mod(inp)
        mod_opt = inductor.compile(symbolic_trace(mod), [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_fx_dict_input(self):
        mod = MyModule2().eval()
        inp = {"key": [torch.randn(10), torch.randn(10)]}
        correct = mod(inp)
        mod_opt = inductor.compile(symbolic_trace(mod), [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_make_fx(self):
        mod = MyModule().eval()
        inp = torch.randn(10)
        correct = mod(inp)
        mod_opt = inductor.compile(make_fx(mod)(inp), [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_bare_module(self):
        mod = MyModule3().eval()
        inp = torch.randn(10)
        correct = mod(inp)
        # no FX graph at all (mod must return list/tuple in this case)
        mod_opt = inductor.compile(mod, [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_export1(self):
        mod = MyModule3().eval()
        inp = torch.randn(10)
        correct = mod(inp)
        gm, guards = dynamo.export(mod, inp, aten_graph=True)
        mod_opt = inductor.compile(gm, [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_export2(self):
        mod = MyModule2().eval()
        inp = {"key": [torch.randn(10), torch.randn(10)]}
        correct = mod(inp)
        gm, guards = dynamo.export(mod, inp)
        mod_opt = inductor.compile(gm, [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_op_with_multiple_outputs(self):
        x1 = torch.randn((2, 512, 128))
        x2 = [128]
        x3 = torch.randn(128)
        x4 = torch.randn((128,))
        x5 = 1e-6
        mod, inp = gen_gm_and_inputs(
            torch.ops.aten.native_layer_norm.default, (x1, x2, x3, x4, x5), {}
        )
        mod_opt = inductor.compile(mod, inp)
        self.assertEqual(mod(*inp), mod_opt(*inp))


if __name__ == "__main__":
    if HAS_CPU:
        run_tests()
