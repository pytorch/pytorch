# Owner(s): ["module: functorch"]
import torch
import torch._dynamo
import torch._export
from torch._higher_order_ops.out_dtype import out_dtype
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing import FileCheck


class TestOutDtypeOp(TestCase):
    def test_out_dtype_make_fx(self):
        class M(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight

            def forward(self, x):
                return out_dtype(
                    torch.ops.aten.mm.default, torch.int32, x, self.weight
                )

        weight = torch.randint(-128, 127, (5, 5), dtype=torch.int8)
        m = M(weight)
        x = torch.randint(-128, 127, (5, 5), dtype=torch.int8)

        gm = make_fx(m)(x)
        self.assertTrue(torch.allclose(m(x), gm(x)))

        gm = make_fx(torch.func.functionalize(M(weight)))(x)
        self.assertTrue(torch.allclose(m(x), gm(x)))

        FileCheck().check("torch.ops.higher_order.out_dtype").check("aten.mm.default").run(gm.code)
        self.assertTrue(torch.allclose(m(x), gm(x)))
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target is out_dtype:
                # Result of this node should be int32
                self.assertTrue(node.meta["val"].dtype, torch.int32)
                # Argument of this node should be int8
                self.assertTrue(node.args[2].meta["val"].dtype, torch.int8)

        x_casted = x.to(torch.int32)
        weight_casted = weight.to(torch.int32)
        numerical_res = torch.ops.aten.mm.default(x_casted, weight_casted)
        self.assertTrue(torch.allclose(numerical_res, gm(x)))

    def test_out_dtype_dynamo(self):
        def f(x, y):
            return out_dtype(
                torch.ops.aten.mul.Scalar, torch.int32, x, y
            )

        inp = (torch.randint(-128, 127, (5, 5), dtype=torch.int8), 3.0)

        compiled = torch.compile(f, backend="eager")
        self.assertTrue(torch.allclose(f(*inp), compiled(*inp)))

        numerical_res = torch.ops.aten.mul.Scalar(inp[0].to(dtype=torch.int32), 3)
        self.assertTrue(torch.allclose(numerical_res, compiled(*inp)))

    def test_out_dtype_op_functional(self):
        class M(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight

            def forward(self, x):
                return out_dtype(
                    torch.ops.aten.mm.default, torch.int32, x, self.weight
                )

        weight = torch.randint(-128, 127, (5, 5), dtype=torch.int8)
        m = M(weight)
        x = torch.randint(-128, 127, (5, 5), dtype=torch.int8)
        ep = torch._export.export(
            m,
            (x,),
            _add_runtime_assertions=False,
        )
        FileCheck().check("torch.ops.higher_order.out_dtype").check("aten.mm.default").run(ep.graph_module.code)
        self.assertTrue(torch.allclose(m(x), ep(x)))
        for node in ep.graph.nodes:
            if node.op == "call_function" and node.target is out_dtype:
                # Result of this node should be int32
                self.assertTrue(node.meta["val"].dtype, torch.int32)
                # Argument of this node should be int8
                self.assertTrue(node.args[2].meta["val"].dtype, torch.int8)

    def test_out_dtype_non_functional(self):
        def f(x, y):
            return out_dtype(
                torch.ops.aten.add_.Tensor, torch.int32, x, y
            )

        with self.assertRaisesRegex(ValueError, "out_dtype's first argument needs to be a functional operator"):
            _ = torch._export.export(
                f, (torch.randint(-128, 127, (5, 5), dtype=torch.int8), torch.randint(-128, 127, (5, 5), dtype=torch.int8)),
            )

    def test_out_dtype_non_op_overload(self):
        def f(x, y):
            return out_dtype(
                torch.add, torch.int32, x, y
            )

        with self.assertRaisesRegex(ValueError, "out_dtype's first argument must be an OpOverload"):
            f(torch.randint(-128, 127, (5, 5), dtype=torch.int8), torch.randint(-128, 127, (5, 5), dtype=torch.int8))

    def test_out_dtype_no_autograd(self):
        def f(x, y):
            return out_dtype(
                torch.ops.aten.mm.default, torch.int32, x, y
            )

        inp = (torch.randn(5, 5, requires_grad=True), torch.randn(5, 5, requires_grad=True))
        with self.assertRaisesRegex(AssertionError, "Autograd is not supported for out_dtype"):
            f(*inp)


if __name__ == '__main__':
    run_tests()
