# Owner(s): ["module: functorch"]
import torch
import torch._dynamo
import torch._functorch
import torch._inductor
import torch._inductor.decomposition
from torch._higher_order_ops.torchbind import enable_torchbind_tracing
from torch._inductor import ir
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.torchbind_impls import (
    _empty_tensor_queue,
    init_torchbind_implementations,
)


class TestTorchbind(TestCase):
    def setUp(self):
        super().setUp()
        init_torchbind_implementations()

    def get_exported_model(self):
        """
        Returns the ExportedProgram, example inputs, and result from calling the
        eager model with those inputs
        """

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)
                self.b = torch.randn(2, 3)

            def forward(self, x):
                x = x + self.b
                a = torch.ops._TorchScriptTesting.takes_foo_tuple_return(self.attr, x)
                y = a[0] + a[1]
                b = torch.ops._TorchScriptTesting.takes_foo(self.attr, y)
                c = self.attr.add_tensor(x)
                return x + b + c

        m = M()
        inputs = (torch.ones(2, 3),)
        orig_res = m(*inputs)

        # We can't directly torch.compile because dynamo doesn't trace ScriptObjects yet
        with enable_torchbind_tracing():
            ep = torch.export.export(m, inputs, strict=False)

        return ep, inputs, orig_res, m

    def test_torchbind_inductor(self):
        ep, inputs, orig_res, _ = self.get_exported_model()
        compiled = torch._inductor.compile(ep.module(), inputs)

        new_res = compiled(*inputs)
        self.assertTrue(torch.allclose(orig_res, new_res))

    def test_torchbind_compile(self):
        _, inputs, orig_res, mod = self.get_exported_model()
        new_res = torch.compile(mod, backend="inductor")(*inputs)
        self.assertTrue(torch.allclose(orig_res, new_res))

    def test_torchbind_get_buf_bytes(self):
        a = torch.classes._TorchScriptTesting._Foo(10, 20)
        buffer = ir.TorchBindObject(name="a", value=a)
        size = buffer.get_buf_bytes()
        self.assertEqual(size, 0)

        t = torch.randn(2, 3)
        b = torch.classes._TorchScriptTesting._ContainsTensor(t)
        buffer = ir.TorchBindObject(name="b", value=b)
        size = buffer.get_buf_bytes()
        self.assertEqual(size, 2 * 3 * 4)

        q = _empty_tensor_queue()
        buffer = ir.TorchBindObject(name="q", value=q)
        size = buffer.get_buf_bytes()
        self.assertEqual(size, 0)

        q.push(torch.ones(2, 3))
        size = buffer.get_buf_bytes()
        self.assertEqual(size, 2 * 3 * 4)


if __name__ == "__main__":
    run_tests()
