import torch

from torch.testing._internal.common_utils import TestCase
from functorch.experimental.cond import cond
from torch.fx.experimental.proxy_tensor import make_fx


class TestControlFlow(TestCase):
    def test_cond(self):
        def true_fn(x):
            return x.sin()

        def false_fn(x):
            return x.cos()

        x = torch.randn(4)
        result = cond(False, true_fn, false_fn, x)
        self.assertTrue(torch.allclose(result, torch.cos(x)))


class TestControlFlowTraced(TestCase):
    def test_cond_traced(self):
        def true_fn(x):
            return x.sin()

        def false_fn(x):
            return x.cos()


        def f(x, y):
            return cond(y, true_fn, false_fn, x)

        x = torch.randn(4)
        graph = make_fx(f)(x, torch.tensor(False))
        result_true = graph.forward(x, torch.tensor(True))
        result_false = graph.forward(x, torch.tensor(False))
        self.assertFalse(torch.allclose(result_true, result_false))
        self.assertTrue(torch.allclose(result_true, torch.sin(x)))
        self.assertTrue(torch.allclose(result_false, torch.cos(x)))

    def test_cond_nested_traced(self):
        def true_nested(y):
            return y * y

        def false_nested(y):
            return y + y

        def true_fn(x, pred2):
            return cond(pred2, true_nested, false_nested, x)

        def false_fn(x, _):
            return x.cos()

        def f(x, pred, pred2):
            return cond(pred, true_fn, false_fn, (x, pred2))

        x = torch.randn(4)
        graph = make_fx(f)(x, torch.tensor(False), torch.tensor(False))

        result_true_true = graph.forward(x, torch.tensor(True), torch.tensor(True))  # True + True -> x * x
        result_true_false = graph.forward(x, torch.tensor(True), torch.tensor(False))  # True + True -> x + x
        result_false_true = graph.forward(x, torch.tensor(False), torch.tensor(True))  #  False + either -> cos
        result_false_false = graph.forward(x, torch.tensor(False), torch.tensor(False))  #  False + either -> cos

        self.assertNotEqual(result_true_true, result_true_false)
        self.assertFalse(torch.allclose(result_false_true, result_true_true))

        self.assertTrue(torch.allclose(result_false_true, result_false_false))

        self.assertTrue(torch.allclose(result_true_true, x * x))
        self.assertTrue(torch.allclose(result_true_false, x + x))

        self.assertTrue(torch.allclose(result_false_true, torch.cos(x)))
