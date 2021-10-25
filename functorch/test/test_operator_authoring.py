import torch
import unittest

import numpy as np

import torch
from torch import fx
from functorch.compile import pointwise_operator
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.jit_utils import JitTestCase
import unittest

LLVM_ENABLED = torch._C._llvm_enabled()
HAS_CUDA = torch.cuda.is_available()
HAS_SYMPY = False
try:
    import sympy

    HAS_SYMPY = True
except ImportError:
    pass


def pointwise_fn(a, b):
    return (a + b) * 42


nnc_pointwise_fn = pointwise_operator(pointwise_fn)


@pointwise_operator
def custom1(a):
    return a + 1.0


@pointwise_operator
def custom2(a):
    return a + 2.0


class TorchFunctionExample(object):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        assert func in (nnc_pointwise_fn, torch.Tensor.add)
        assert all(issubclass(t, (torch.Tensor, TorchFunctionExample)) for t in types)
        return torch.zeros_like(args[0])


class TestOperatorAuthoring(JitTestCase):
    device = None

    def rand(self, *args, dtype=torch.float32, **kwargs):
        return torch.randint(0, 100, args, dtype=dtype, device=self.device, **kwargs)

    def check(self, *args):
        result_aten = pointwise_fn(*args)
        result_nnc = nnc_pointwise_fn(*args)
        self.assertEqual(result_nnc.dtype, result_aten.dtype)
        self.assertEqual(result_nnc.size(), result_aten.size())
        self.assertEqual(result_nnc.stride(), result_aten.stride())
        self.assertEqual(result_nnc.requires_grad, result_aten.requires_grad)
        torch.testing.assert_allclose(result_aten, result_nnc)

    def test_broadcast1(self):
        self.check(self.rand(8, 16), self.rand(1))

    def test_broadcast2(self):
        self.check(self.rand(8, 1), self.rand(1, 8))

    def test_transposed1(self):
        self.check(self.rand(7, 3), self.rand(3, 7).transpose(0, 1))

    def test_transposed2(self):
        self.check(self.rand(8, 16).transpose(0, 1), self.rand(8, 16).transpose(0, 1))

    def test_slice1(self):
        self.check(self.rand(20, 20, 2)[:8, :16, 0], self.rand(8, 16))

    def test_slice2(self):
        self.check(self.rand(8, 16, 2)[:, :, 0], self.rand(8, 16, 2)[:, :, 0])

    def test_issue57611(self):
        self.check(self.rand(1, 32, 32, 2), self.rand(2, 1, 1, 2))

    def test_float_double(self):
        self.check(self.rand(8, 16), self.rand(8, 16, dtype=torch.float64))

    def test_int_long(self):
        self.check(
            self.rand(8, 16, dtype=torch.int32), self.rand(1, 1, dtype=torch.int64)
        )

    def test_float_int(self):
        self.check(
            self.rand(8, 16, dtype=torch.float32), self.rand(8, 16, dtype=torch.int32)
        )

    @unittest.skipIf(not HAS_SYMPY, "currently requires sympy")
    def test_requires_grad(self):
        self.check(self.rand(4, 2), self.rand(4, 2, requires_grad=True))

    @unittest.skipIf(not HAS_SYMPY, "currently requires sympy")
    def test_backwards(self):
        def grads(fn):
            a = self.rand(4, 2, requires_grad=True)
            b = self.rand(4, 2, requires_grad=True)
            c = self.rand(4, 2)
            d = self.rand(4, 2)
            fn(fn(a, fn(b, c)), d).sum().backward()
            return a.grad, b.grad

        a1, b1 = grads(pointwise_fn)
        a2, b2 = grads(nnc_pointwise_fn)
        torch.testing.assert_allclose(a1, a2)
        torch.testing.assert_allclose(b1, b2)

    def test_torch_function(self):
        self.check(self.rand(10), TorchFunctionExample())

    def test_fx_trace(self):
        def example(x):
            return custom1(custom2(x))

        graph = fx.symbolic_trace(example)
        self.assertIn("custom1", graph.code)
        self.assertIn("custom2", graph.code)
        x = torch.randn(8, device=self.device)
        torch.testing.assert_allclose(x + 3, graph(x))

    def test_unary_ops(self):
        unary_operators = [
            torch.sin,
            torch.cos,
            torch.tan,
            torch.asin,
            torch.acos,
            torch.atan,
            torch.sinh,
            torch.cosh,
            torch.tanh,
            torch.sigmoid,
            torch.exp,
            torch.expm1,
            torch.abs,
            torch.log,
            torch.log2,
            torch.log10,
            torch.log1p,
            torch.erf,
            torch.erfc,
            torch.sqrt,
            torch.rsqrt,
            torch.ceil,
            torch.floor,
            torch.round,
            torch.trunc,
            torch.lgamma,
            torch.ops.aten.sin,
            torch.ops.aten.cos,
            torch.ops.aten.tan,
            torch.ops.aten.asin,
            torch.ops.aten.acos,
            torch.ops.aten.atan,
            torch.ops.aten.sinh,
            torch.ops.aten.cosh,
            torch.ops.aten.tanh,
            torch.ops.aten.sigmoid,
            torch.ops.aten.exp,
            torch.ops.aten.expm1,
            torch.ops.aten.abs,
            torch.ops.aten.log,
            torch.ops.aten.log2,
            torch.ops.aten.log10,
            torch.ops.aten.log1p,
            torch.ops.aten.erf,
            torch.ops.aten.erfc,
            torch.ops.aten.sqrt,
            torch.ops.aten.rsqrt,
            torch.ops.aten.ceil,
            torch.ops.aten.floor,
            torch.ops.aten.round,
            torch.ops.aten.trunc,
            torch.ops.aten.lgamma,
            # TODO - Failure in generating loop nests here for the following ops
            # torch.frac,
            # torch.isnan,
        ]

        for unary_op in unary_operators:
            fn = lambda x: unary_op(x)
            pointwise_fn = pointwise_operator(fn)
            a = torch.rand(2, 3)
            ref = fn(a)
            res = pointwise_fn(a)
            assert torch.allclose(ref, res, atol=1e-3, rtol=1e-3)

    def test_binary_ops(self):
        binary_operators = [
            torch.add,
            torch.sub,
            torch.subtract,
            torch.mul,
            torch.multiply,
            torch.divide,
            torch.div,
            torch.fmod,
            torch.pow,
            torch.atan2,
            # torch.remainder, #TODO - Fails allclose check
            torch.ops.aten.add,
            torch.ops.aten.sub,
            torch.ops.aten.subtract,
            torch.ops.aten.mul,
            torch.ops.aten.multiply,
            torch.ops.aten.divide,
            torch.ops.aten.div,
            torch.ops.aten.fmod,
            torch.ops.aten.pow,
            torch.ops.aten.atan2,
        ]
        for binary_op in binary_operators:
            fn = lambda x, y: binary_op(x, y)
            pointwise_fn = pointwise_operator(fn)
            a = torch.rand(2, 3)
            b = torch.rand(2, 3)
            ref = fn(a, b)
            res = pointwise_fn(a, b)
            assert torch.allclose(ref, res, atol=1e-3, rtol=1e-3)

    def test_bias_gelu(self):
        def bias_gelu(bias, y):
            x = bias + y
            return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

        bias = torch.rand(1, 768)
        y = torch.rand(64, 768)
        ref = bias_gelu(bias, y)

        pointwise_fn = pointwise_operator(bias_gelu)
        res = pointwise_fn(bias, y)
        assert torch.allclose(ref, res, atol=1e-3, rtol=1e-3)


@unittest.skipIf(not HAS_CUDA, "GPU tests require CUDA")
class TestOperatorAuthoringGPU(TestOperatorAuthoring):
    device = "cuda"


@unittest.skipIf(not LLVM_ENABLED, "CPU tests require LLVM")
class TestOperatorAuthoringCPU(TestOperatorAuthoring):
    device = "cpu"


if __name__ == "__main__":
    run_tests()
