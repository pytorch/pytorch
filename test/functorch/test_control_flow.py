# Owner(s): ["module: functorch"]
import torch

from torch.testing._internal.common_utils import TestCase, run_tests
from functorch.experimental.cond import cond
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._python_dispatch import TorchPreDispatchMode


class TestControlFlow(TestCase):
    def test_cond_no_trace(self):
        def true_fn(x):
            return x.sin()

        def false_fn(x):
            return x.cos()

        x = torch.randn(4)
        result = cond(False, true_fn, false_fn, [x])
        self.assertEqual(result, torch.cos(x))


class TestControlFlowTraced(TestCase):
    def test_cond_traced_not_nested(self):
        def true_fn(x):
            return x.sin()

        def false_fn(x):
            return x.cos()

        def f(x, y):
            return cond(y, true_fn, false_fn, [x])

        x = torch.randn(4)
        graph = make_fx(f)(x, torch.tensor(False))
        result_true = graph.forward(x, torch.tensor(True))
        result_false = graph.forward(x, torch.tensor(False))
        self.assertFalse(torch.allclose(result_true, result_false))
        self.assertEqual(result_true, torch.sin(x))
        self.assertEqual(result_false, torch.cos(x))

    def test_cond_nested_traced(self):
        def true_nested(y):
            return y * y

        def false_nested(y):
            return y + y

        def true_fn(x, pred2):
            z = cond(pred2, true_nested, false_nested, [x])
            return x + z

        def false_fn(x, _):
            return x.cos()

        def f(x, pred, pred2):
            return cond(pred, true_fn, false_fn, [x, pred2])

        x = torch.randn(4)
        graph = make_fx(f)(x, torch.tensor(False), torch.tensor(False))

        result_true_true = graph.forward(x, torch.tensor(True), torch.tensor(True))  # True + True -> x * x
        result_true_false = graph.forward(x, torch.tensor(True), torch.tensor(False))  # True + True -> x + x
        result_false_true = graph.forward(x, torch.tensor(False), torch.tensor(True))  # False + either -> cos
        result_false_false = graph.forward(x, torch.tensor(False), torch.tensor(False))  # False + either -> cos

        self.assertNotEqual(result_true_true, result_true_false)
        self.assertFalse(torch.allclose(result_false_true, result_true_true))

        self.assertEqual(result_false_true, result_false_false)

        self.assertEqual(result_true_true, (x * x) + x)
        self.assertEqual(result_true_false, x + x + x)

        self.assertEqual(result_false_true, torch.cos(x))

    def test_cond_nested_traced_other_inputs(self):
        def true_nested(y):
            return y * y

        def false_nested(y):
            return y + y

        def true_fn(k, pred2):
            z = cond(pred2, true_nested, false_nested, [k])
            return torch.add(torch.tensor([.25, .25]), z)

        def false_fn(k, _):
            return k.cos()

        def f(k, pred, pred2):
            return cond(pred, true_fn, false_fn, [k, pred2])

        x = torch.tensor([0.5, 0.5])
        graph = make_fx(f)(x, torch.tensor(False), torch.tensor(False))

        a = torch.tensor([1.0, 1.0])
        result_true_true = graph.forward(a, torch.tensor(True), torch.tensor(True))
        self.assertEqual(result_true_true, (a * a) + torch.tensor([0.25, 0.25]))

        b = torch.tensor([2.0, 2.0])
        result_true_true = graph.forward(b, torch.tensor(True), torch.tensor(True))
        self.assertEqual(result_true_true, (b * b) + torch.tensor([0.25, 0.25]))

    def test_cond_nested_traced_multi(self):
        def true_a(y):
            return y * y

        def false_a(y):
            return y + y

        def true_b(y, z):
            return y + z

        def false_b(y, z):
            return y * z

        def f(x, pred, pred2):
            a_out = cond(pred, true_a, false_a, [x])
            b_out = cond(pred2, true_b, false_b, [x, x])
            return a_out + b_out

        x = torch.randn(4)
        graph = make_fx(f)(x, torch.tensor(False), torch.tensor(False))

        # Brittle, yet, delicious
        out = """
        def forward(self, x_1, pred_1, pred2_1):
            true_graph_0 = self.true_graph_0
            false_graph_0 = self.false_graph_0
            conditional = torch.ops.cond(pred_1, true_graph_0, false_graph_0, [[x_1]]);
            pred_1 = true_graph_0 = false_graph_0 = None
            true_graph_1 = self.true_graph_1
            false_graph_1 = self.false_graph_1
            conditional_1 = torch.ops.cond(pred2_1, true_graph_1, false_graph_1, [[x_1, x_1]]);
            pred2_1 = true_graph_1 = false_graph_1 = x_1 = None
            add = torch.ops.aten.add.Tensor(conditional, conditional_1);  conditional = conditional_1 = None
            return add
        """
        code = graph.code
        # Normalization hack, cause .code makes some weird whitespace
        code = "".join(code.split())
        out = "".join(out.split())
        self.assertEqual(code, out)

        code = graph.true_graph_0.code
        out = """
        def forward(self, flat_args):
            flat_args_1, = fx_pytree.tree_flatten_spec([flat_args], self._in_spec)
            mul = torch.ops.aten.mul.Tensor(flat_args_1, flat_args_1);  flat_args_1 = None
            return pytree.tree_unflatten([mul], self._out_spec)
        """
        # Normalization hack, cause .code makes some weird whitespace
        code = "".join(code.split())
        out = "".join(out.split())
        self.assertEqual(code, out)

    def test_assert_on_mismatch_type_size(self):
        def true_fn(x):
            return x.sin()

        def false_fn(x):
            return (x, x)

        def f(x, y):
            return cond(y, true_fn, false_fn, [x])

        x = torch.randn(4)
        with self.assertRaises(AssertionError):
            make_fx(f)(x, torch.tensor(False))


    def test_assert_on_mismatch_tensor_size(self):
        def true_fn(x):
            return x.sin()

        def false_fn(x):
            return torch.zeros([10, 10])

        def f(x, y):
            return cond(y, true_fn, false_fn, [x])

        x = torch.randn(4)
        with self.assertRaises(AssertionError):
            make_fx(f)(x, torch.tensor(False))

    def test_cond_traced_not_nested_fake_tensor(self):
        def true_fn(x):
            return x.sin()

        def false_fn(x):
            return x.cos()

        def f(x, y):
            return cond(y, true_fn, false_fn, [x])

        x = torch.randn(4)
        graph = make_fx(f, tracing_mode="fake")(x, torch.tensor(False))
        result_true = graph.forward(x, torch.tensor(True))
        result_false = graph.forward(x, torch.tensor(False))
        self.assertFalse(torch.allclose(result_true, result_false))
        self.assertEqual(result_true, torch.sin(x))
        self.assertEqual(result_false, torch.cos(x))

    def test_cond_nested_traced_fake_tensor(self):
        def true_nested(y):
            return y * y

        def false_nested(y):
            return y + y

        def true_fn(x, pred2):
            z = cond(pred2, true_nested, false_nested, [x])
            return x + z

        def false_fn(x, _):
            return x.cos()

        def f(x, pred, pred2):
            return cond(pred, true_fn, false_fn, [x, pred2])

        x = torch.randn(4)
        graph = make_fx(f, tracing_mode="fake")(x, torch.tensor(False), torch.tensor(False))

        result_true_true = graph.forward(x, torch.tensor(True), torch.tensor(True))  # True + True -> x * x
        result_true_false = graph.forward(x, torch.tensor(True), torch.tensor(False))  # True + True -> x + x
        result_false_true = graph.forward(x, torch.tensor(False), torch.tensor(True))  # False + either -> cos
        result_false_false = graph.forward(x, torch.tensor(False), torch.tensor(False))  # False + either -> cos

        self.assertNotEqual(result_true_true, result_true_false)
        self.assertFalse(torch.allclose(result_false_true, result_true_true))

        self.assertEqual(result_false_true, result_false_false)

        self.assertEqual(result_true_true, (x * x) + x)
        self.assertEqual(result_true_false, x + x + x)

        self.assertEqual(result_false_true, torch.cos(x))

    def test_cond_nested_traced_other_inputs_fake_tensor(self):
        def true_nested(y):
            return y * y

        def false_nested(y):
            return y + y

        def true_fn(k, pred2):
            z = cond(pred2, true_nested, false_nested, [k])
            return torch.add(torch.tensor([.25, .25]), z)

        def false_fn(k, _):
            return k.cos()

        def f(k, pred, pred2):
            return cond(pred, true_fn, false_fn, [k, pred2])

        x = torch.tensor([0.5, 0.5])
        graph = make_fx(f, tracing_mode="fake")(x, torch.tensor(False), torch.tensor(False))

        a = torch.tensor([1.0, 1.0])
        result_true_true = graph.forward(a, torch.tensor(True), torch.tensor(True))
        self.assertEqual(result_true_true, (a * a) + torch.tensor([0.25, 0.25]))

        b = torch.tensor([2.0, 2.0])
        result_true_true = graph.forward(b, torch.tensor(True), torch.tensor(True))
        self.assertEqual(result_true_true, (b * b) + torch.tensor([0.25, 0.25]))

    def test_cond_nested_traced_multi_fake_tensor(self):
        def true_a(y):
            return y * y

        def false_a(y):
            return y + y

        def true_b(y, z):
            return y + z

        def false_b(y, z):
            return y * z

        def f(x, pred, pred2):
            a_out = cond(pred, true_a, false_a, [x])
            b_out = cond(pred2, true_b, false_b, [x, x])
            return a_out + b_out

        x = torch.randn(4)
        graph = make_fx(f, tracing_mode="fake")(x, torch.tensor(False), torch.tensor(False))

        # Brittle, yet, delicious
        out = """
        def forward(self, x_1, pred_1, pred2_1):
            true_graph_0 = self.true_graph_0
            false_graph_0 = self.false_graph_0
            conditional = torch.ops.cond(pred_1, true_graph_0, false_graph_0, [[x_1]]);
            pred_1 = true_graph_0 = false_graph_0 = None
            true_graph_1 = self.true_graph_1
            false_graph_1 = self.false_graph_1
            conditional_1 = torch.ops.cond(pred2_1, true_graph_1, false_graph_1, [[x_1, x_1]]);
            pred2_1 = true_graph_1 = false_graph_1 = x_1 = None
            add = torch.ops.aten.add.Tensor(conditional, conditional_1);  conditional = conditional_1 = None
            return add
        """
        code = graph.code
        # Normalization hack, cause .code makes some weird whitespace
        code = "".join(code.split())
        out = "".join(out.split())
        self.assertEqual(code, out)

        code = graph.true_graph_0.code
        out = """
        def forward(self, flat_args):
            flat_args_1, = fx_pytree.tree_flatten_spec([flat_args], self._in_spec)
            mul = torch.ops.aten.mul.Tensor(flat_args_1, flat_args_1);  flat_args_1 = None
            return pytree.tree_unflatten([mul], self._out_spec)
        """
        # Normalization hack, cause .code makes some weird whitespace
        code = "".join(code.split())
        out = "".join(out.split())
        self.assertEqual(code, out)

    def test_assert_on_mismatch_type_size_fake_tensor(self):
        def true_fn(x):
            return x.sin()

        def false_fn(x):
            return (x, x)

        def f(x, y):
            return cond(y, true_fn, false_fn, [x])

        x = torch.randn(4)
        with self.assertRaises(AssertionError):
            make_fx(f, tracing_mode="fake")(x, torch.tensor(False))


    def test_assert_on_mismatch_tensor_size_fake_tensor(self):
        def true_fn(x):
            return x.sin()

        def false_fn(x):
            return torch.zeros([10, 10])

        def f(x, y):
            return cond(y, true_fn, false_fn, [x])

        x = torch.randn(4)
        with self.assertRaises(AssertionError):
            make_fx(f, tracing_mode="fake")(x, torch.tensor(False))

    def test_with_pre_dispatch_mode(self):
        class A(TorchPreDispatchMode):


if __name__ == '__main__':
    run_tests()
