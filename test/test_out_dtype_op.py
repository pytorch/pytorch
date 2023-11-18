# Owner(s): ["module: functorch"]
import unittest

import torch
import torch._dynamo
import torch._inductor
import torch._inductor.decomposition
import torch._export
from torch._higher_order_ops.out_dtype import out_dtype
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import (
    run_tests, TestCase, IS_WINDOWS, TEST_WITH_ROCM, IS_FBCODE, IS_REMOTE_GPU, TEST_CUDA
)
from torch.testing._internal.common_quantization import skipIfNoDynamoSupport
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import SM80OrLater, _get_torch_cuda_version


@unittest.skipIf(not torch._dynamo.is_dynamo_supported(), "dynamo isn't support")
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
        )
        FileCheck().check("torch.ops.higher_order.out_dtype").check("aten.mm.default").run(ep.graph_module.code)
        self.assertTrue(torch.allclose(m(x), ep(x)))
        for node in ep.graph.nodes:
            if node.op == "call_function" and node.target is out_dtype:
                # Result of this node should be int32
                self.assertTrue(node.meta["val"].dtype, torch.int32)
                # Argument of this node should be int8
                self.assertTrue(node.args[2].meta["val"].dtype, torch.int8)

    def test_out_dtype_mm_numerical(self):
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

        compiled = torch.compile(f, backend="eager", fullgraph=True)
        self.assertTrue(torch.allclose(f(*inp), compiled(*inp)))

    def test_out_dtype_mul_scalar_numerical(self):
        def f(x, y):
            return out_dtype(
                torch.ops.aten.mul.Scalar, torch.int32, x, y
            )

        inp = (torch.randint(-128, 127, (5, 5), dtype=torch.int8), 3.0)

        gm = make_fx(f)(*inp)
        numerical_res = torch.ops.aten.mul.Scalar(inp[0].to(dtype=torch.int32), 3)
        self.assertTrue(torch.allclose(numerical_res, gm(*inp)))

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
        # error is delayed
        f(*inp)

        with torch.no_grad():
            f(*inp)

        with self.assertRaisesRegex(RuntimeError, "does not require grad and does not have a grad_fn"):
            out = f(*inp)
            loss = out - torch.ones(out.shape)
            loss.backward()

    @unittest.skipIf(IS_WINDOWS, "_int_mm unavailable")
    @unittest.skipIf(TEST_WITH_ROCM, "_int_mm unavailable")
    @unittest.skipIf(not SM80OrLater, "_int_mm unavailable")
    @unittest.skipIf(IS_FBCODE and IS_REMOTE_GPU, "cublas runtime error")
    @unittest.skipIf(_get_torch_cuda_version() >= (11, 7), "_int_mm unavailable")
    @unittest.skipIf(not TEST_CUDA, "_int_mm unavailable")
    @skipIfNoDynamoSupport
    def test_out_dtype_inductor_decomp(self) -> None:
        def func(x, w):
            return out_dtype(torch.ops.aten.mm.default, torch.int32, x, w)

        w = torch.randint(-128, 127, (32, 32), dtype=torch.int8, device="cuda")
        x = torch.randint(-128, 127, (32, 32), dtype=torch.int8, device="cuda")

        ref = torch._int_mm(x, w)
        test_out = func(x, w)
        func_comp = torch.compile(func, fullgraph=True, mode="max-autotune")
        test_out_c = func_comp(x, w)
        self.assertTrue(torch.allclose(ref, test_out))
        self.assertTrue(torch.allclose(ref, test_out_c))

    @unittest.skipIf(not TEST_CUDA, "cuda only")
    def test_out_dtype_inductor_decomp_trace(self) -> None:
        def func(x, w):
            return out_dtype(torch.ops.aten.mm.default, torch.int32, x, w)

        w = torch.randint(-128, 127, (32, 32), dtype=torch.int8, device="cuda")
        x = torch.randint(-128, 127, (32, 32), dtype=torch.int8, device="cuda")

        # Check that make_fx with inductor decomps produces _int_mm
        decomp_table = torch._inductor.decomposition.select_decomp_table()
        gm = make_fx(func, decomp_table, tracing_mode="symbolic")(x, w)
        self.assertExpectedInline(gm.code.strip(), """\
def forward(self, x_1, w_1):
    _int_mm = torch.ops.aten._int_mm.default(x_1, w_1);  x_1 = w_1 = None
    return _int_mm""")

    @unittest.skipIf(not TEST_CUDA, "cuda only")
    def test_out_dtype_int_mm_default_trace(self) -> None:
        def func(x, w):
            return out_dtype(torch.ops.aten.mm.default, torch.int32, x, w)

        w = torch.randint(-128, 127, (32, 32), dtype=torch.int8, device="cuda")
        x = torch.randint(-128, 127, (32, 32), dtype=torch.int8, device="cuda")

        # By default, out_dtype is preserved in the trace
        gm = make_fx(func, tracing_mode="symbolic")(x, w)
        self.assertExpectedInline(gm.code.strip(), """\
def forward(self, x_1, w_1):
    out_dtype = torch.ops.higher_order.out_dtype(torch.ops.aten.mm.default, torch.int32, x_1, w_1);  x_1 = w_1 = None
    return out_dtype""")

    def test_out_dtype_wrong_output(self) -> None:
        def multiple_out(x):
            return out_dtype(
                torch.ops.aten.topk.default, torch.int32, x, 5
            )

        inp = (torch.randn(10),)

        with self.assertRaisesRegex(ValueError, "out_dtype's can only apply to ops that return a single tensor"):
            multiple_out(*inp)

        def singleton_list_out(x):
            return out_dtype(
                torch.ops.aten.split_copy.Tensor, torch.int32, x, 10
            )

        with self.assertRaisesRegex(ValueError, "out_dtype's can only apply to ops that return a single tensor"):
            singleton_list_out(*inp)

if __name__ == '__main__':
    run_tests()
