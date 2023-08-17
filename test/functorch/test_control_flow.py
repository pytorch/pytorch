# Owner(s): ["module: functorch"]
import functools
import unittest

import torch
import torch.utils._pytree as pytree
from torch._functorch.aot_autograd import from_fun, to_fun
from functorch.experimental import control_flow
from functorch.experimental.control_flow import cond
from functorch.experimental.control_flow import UnsupportedAliasMutationException
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import run_tests, TestCase
from torch._dynamo.exc import CondOpArgsMismatchError

def _fake_map(f, x, *args):
    from functorch.experimental._map import _stack_pytree, _unstack_pytree
    x_pytrees = _unstack_pytree(x)
    zs = []
    for xp in x_pytrees:
        zs.append(f(xp, *args))
    return _stack_pytree(zs)


class TestControlFlow(TestCase):
    def test_cond_no_trace(self):
        def true_fn(x):
            return x.sin()

        def false_fn(x):
            return x.cos()

        x = torch.randn(4)
        result = cond(False, true_fn, false_fn, [x])
        self.assertEqual(result, torch.cos(x))

    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA.")
    def test_cond_gpu(self):
        def true_fn(x):
            return x.sin()

        def false_fn(x):
            return x.cos()

        x = torch.randn(4, device="cuda")
        pred = torch.tensor(False, device="cuda")
        result = cond(pred, true_fn, false_fn, [x])
        self.assertEqual(result, torch.cos(x))

    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA.")
    def test_map_gpu(self):
        def f(x, y):
            return x + y

        xs = torch.ones(3, 2, 2, device="cuda")
        y = torch.ones(2, device="cuda")
        res = control_flow.map(f, xs, y)
        expected = _fake_map(f, xs, y)
        self.assertEqual(expected, res)

    def test_map_illegal_inputs(self):
        def f(x, y):
            return x[0] + x[1] + y

        with self.assertRaisesRegex(RuntimeError,
                                    r"Mapped xs can only consist of tensors\. Got xs \[3, tensor\(\[1\., 1\.\]\)\]\."):
            _ = control_flow.map(f, (3, torch.ones(2)), torch.ones(2))

        with self.assertRaisesRegex(RuntimeError,
                                    r"Leading dimensions of mapped xs cannot be 0\."):
            _ = control_flow.map(f, (torch.ones(0, 1, 2), torch.ones(0, 1, 2)), torch.ones(2))

        with self.assertRaisesRegex(RuntimeError,
                                    r"Leading dimensions of mapped xs must be consistent\. "
                                    r"Got shapes \[torch\.Size\(\[3, 4, 5\]\), torch\.Size\(\[4, 4, 5\]\)\]\."):
            _ = control_flow.map(f, (torch.ones(3, 4, 5), torch.ones(4, 4, 5)), torch.ones(5))

    def test_map_illegal_outputs(self):
        def f(x, y):
            return x.item()

        def f1(x, y):
            return y.size()

        def f2(x, y):
            return None

        x = torch.ones([3])
        y = torch.ones([1, 2, 3])
        with self.assertRaisesRegex(RuntimeError, r"Expect outputs of map only contains tensors or None\."):
            _ = control_flow.map(f, x, y)

        with self.assertRaisesRegex(RuntimeError, r"Expect outputs of map only contains tensors or None\."):
            out = control_flow.map(f1, x, y)

        # return None is OK
        _ = control_flow.map(f2, x, y)


    def test_map_list_in_out(self):
        def f(x, y):
            return [[x[0][0] + y]]

        xs = [[torch.ones(3, 2, 2)]]
        y = torch.ones(2)
        res = control_flow.map(f, xs, y)
        expected = _fake_map(f, xs, y)
        self.assertEqual(len(res), 1)
        self.assertEqual(len(res[0]), 1)
        self.assertEqual(expected, res)

    def test_map_dict_in_out(self):
        def f(x, y):
            return {"c": x["a"]["b"] + y}

        xs = {"a": {"b": torch.ones(3, 2, 2)}}
        y = torch.ones(2)
        res = control_flow.map(f, xs, y)
        expected = _fake_map(f, xs, y)
        self.assertEqual(len(res), 1)
        self.assertTrue("c" in res)
        self.assertEqual(expected, res)

    def test_map_autograd_simple(self):
        def f(x, y):
            return x.sin().cos() * y.cos().sin()

        xs = torch.ones(3, 2, 2, requires_grad=True)
        y = torch.ones(2, requires_grad=True)
        res = control_flow.map(f, xs, y)
        expected_res = _fake_map(f, xs, y)
        grad_out = torch.ones_like(res)
        grads = torch.autograd.grad(res, (xs, y), grad_out)
        expected_grads = torch.autograd.grad(expected_res, (xs, y), grad_out)
        self.assertEqual(expected_res, res)
        self.assertEqual(expected_grads, grads)

    def test_map_autograd_simple_partial_grad(self):
        def f(x, y):
            return x.sin().cos() * y.cos().sin()

        xs = torch.ones(3, 2, 2, requires_grad=True)
        # Disable the gradient computation for y
        y = torch.ones(2, requires_grad=False)
        res = control_flow.map(f, xs, y)
        expected_res = _fake_map(f, xs, y)
        grad_out = torch.ones_like(res)
        grads = torch.autograd.grad(res, (xs,), grad_out)
        expected_grads = torch.autograd.grad(expected_res, (xs,), grad_out)
        self.assertEqual(expected_res, res)
        self.assertEqual(expected_grads, grads)

    def test_map_autograd_no_grad_output(self):
        def f(x, y):
            return x[0].sin().cos() + y, y.cos().sin()

        xs = [torch.ones(3, 2, 2, requires_grad=True), torch.ones(3, 3)]
        # Disable the gradient computation for y
        y = torch.ones(2, requires_grad=False)
        res = control_flow.map(f, xs, y)
        expected_res = _fake_map(f, xs, y)
        grad_out = torch.ones_like(res[0])
        grads = torch.autograd.grad(res[0], (xs[0],), grad_out)
        expected_grads = torch.autograd.grad(expected_res[0], (xs[0],), grad_out)
        self.assertEqual(expected_res, res)
        self.assertEqual(expected_grads, grads)


    def test_map_autograd_nested_list(self):
        import torch.utils._pytree as pytree

        def f(x, y):
            a, b = x
            c, d = a
            return [[b.sin() * c.cos()], d.sin() * y.cos()]

        def fwbw(map_op, f, x, y):
            z = map_op(f, x, y)
            flat_x, _ = pytree.tree_flatten(x)
            flat_z, _ = pytree.tree_flatten(z)
            grads = torch.autograd.grad(flat_z, flat_x, [torch.ones_like(z) for z in flat_z])
            return z, grads

        x = [[torch.randn(3, 2, 2, requires_grad=True), torch.randn(3, 2, 1, requires_grad=True)],
             torch.ones(3, 1, 2, requires_grad=True)]
        y = torch.ones(1, requires_grad=True)
        true_outs = fwbw(control_flow.map, f, x, y)
        fake_outs = fwbw(_fake_map, f, x, y)
        self.assertEqual(true_outs, fake_outs)


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

        graph = make_fx(f, tracing_mode="symbolic")(x, torch.tensor(False))
        self.assertEqual(graph(x, torch.tensor(True)), f(x, torch.tensor(True)))

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

        graph = make_fx(f, tracing_mode="symbolic")(x, torch.tensor(False), torch.tensor(False))
        self.assertEqual(graph(x, torch.tensor(True), torch.tensor(True)), f(x, torch.tensor(True), torch.tensor(True)))

    def test_cond_functionalized(self):
        def true_fn(x):
            y = x.sin()
            y.add_(4)
            return x.sin().max() + y.sum()

        def false_fn(x):
            return x.cos().min()

        def f(x):
            pred = x.shape[0] == 1
            return cond(pred, true_fn, false_fn, [x])

        example_inputs = (torch.ones(4, 5),)
        functional_f = torch.func.functionalize(f)
        self.assertEqual(functional_f(*example_inputs), f(*example_inputs))

        graph_module = make_fx(torch.func.functionalize(f))(*example_inputs)
        self.assertEqual(graph_module(*example_inputs), f(*example_inputs))

        all_ops_in_true_branch = []
        for node in graph_module.true_graph_0.graph.nodes:
            if node.op == "call_function":
                all_ops_in_true_branch.append(node.target)

        self.assertFalse(any(op._schema.is_mutable for op in all_ops_in_true_branch))

        graph_module = make_fx(torch.func.functionalize(f), tracing_mode="symbolic")(*example_inputs)
        self.assertEqual(graph_module(*example_inputs), f(*example_inputs))

    def test_cond_retrace_functionalized(self):
        def true_fn(x):
            return x.sin()

        def false_fn(x):
            return x.cos()

        def f(x):
            return cond(x.all(), true_fn, false_fn, (x,))

        inp = torch.ones(1, 2)
        gm_non_functional = make_fx(f, tracing_mode="real")(inp)
        gm_functional = make_fx(torch.func.functionalize(gm_non_functional), tracing_mode="real")(inp)
        self.assertEqual(gm_functional(torch.zeros(1, 2)), f(torch.zeros(1, 2)))

    def test_cond_functionalized_nested(self):
        def true_true_fn(x):
            y = x.cos()
            y.add_(4)
            return x.sin().max() + y.sin().max()

        def true_false_fn(x):
            return x.cos().min()

        def true_fn(x):
            pred = x.shape[0] == 1
            return cond(pred, true_true_fn, true_false_fn, [x])

        def false_fn(x):
            return x.sum()

        def f(x):
            pred = x.shape[0] == 1
            return cond(pred, true_fn, false_fn, [x])

        example_inputs = (torch.ones(4, 5),)
        functional_f = torch.func.functionalize(f)
        self.assertEqual(functional_f(*example_inputs), f(*example_inputs))

        graph_module = make_fx(torch.func.functionalize(f))(*example_inputs)
        self.assertEqual(graph_module(*example_inputs), f(*example_inputs))

        gm_true_true_branch = graph_module.true_graph_0.true_graph_0

        graph_module1 = make_fx(torch.func.functionalize(f), tracing_mode="symbolic")(*example_inputs)
        self.assertEqual(graph_module1(*example_inputs), f(*example_inputs))

        all_ops = []
        for node in gm_true_true_branch.graph.nodes:
            if node.op == "call_function":
                all_ops.append(node.target)

        self.assertFalse(any(op._schema.is_mutable for op in all_ops))

    def test_cond_functionalized_data_dependent_pred(self):
        def true_fn(x):
            return x.sin().sum()

        def false_fn(x):
            return x.cos().sum()

        def f(x):
            pred = x.nonzero().shape[0] == 1
            return cond(pred, true_fn, false_fn, [x])

        example_inputs = (torch.ones(4, 5),)
        functional_f = torch.func.functionalize(f)
        self.assertEqual(functional_f(*example_inputs), f(*example_inputs))

        graph_module = make_fx(torch.func.functionalize(f))(*example_inputs)
        self.assertEqual(graph_module(*example_inputs), f(*example_inputs))

    def test_cond_functionalized_input_mutation_on_true_branch(self):
        def true_fn(x):
            view_x = x.view(x.shape)
            view_x.add_(1)
            return view_x.sin().sum()

        def false_fn(x):
            return x.cos().sum()

        def f(x):
            pred = x.shape[0] == 4
            return cond(pred, true_fn, false_fn, [x])

        example_inputs = (torch.ones(4, 5),)
        functional_f = torch.func.functionalize(f)
        with self.assertRaisesRegex(UnsupportedAliasMutationException, "One of torch.cond branch"):
            functional_f(*example_inputs)

        with self.assertRaisesRegex(UnsupportedAliasMutationException, "One of torch.cond branch"):
            make_fx(torch.func.functionalize(f))(*example_inputs)

    def test_cond_functionalized_input_mutation_on_false_branch(self):
        def true_fn(x):
            return x.sin().sum()

        def false_fn(x):
            view_x = x.view(x.shape)
            view_x.add_(1)
            return view_x.cos().sum()

        def f(x):
            pred = x.shape[0] == 4
            return cond(pred, true_fn, false_fn, [x])

        example_inputs = (torch.ones(5, 5),)
        functional_f = torch.func.functionalize(f)
        with self.assertRaisesRegex(UnsupportedAliasMutationException, "One of torch.cond branch"):
            functional_f(*example_inputs)

        with self.assertRaisesRegex(UnsupportedAliasMutationException, "One of torch.cond branch"):
            make_fx(torch.func.functionalize(f))(*example_inputs)

    def test_cond_functionalized_output_alias_input(self):
        def true_fn(x):
            return x

        def false_fn(x):
            view_x = x.view(x.shape)
            return view_x

        def f(x):
            pred = x.shape[0] == 4
            return cond(pred, true_fn, false_fn, [x])

        example_inputs = (torch.ones(5, 5),)
        functional_f = torch.func.functionalize(f)

        with self.assertRaisesRegex(UnsupportedAliasMutationException, "One of torch.cond branch might be aliasing"):
            functional_f(*example_inputs)

        with self.assertRaisesRegex(UnsupportedAliasMutationException, "One of torch.cond branch might be aliasing"):
            make_fx(torch.func.functionalize(f))(*example_inputs)

    def test_cond_functionalized_nested_input_mutation(self):
        def true_true_fn(x):
            x.add_(4)
            return x.sin().max()

        def true_false_fn(x):
            return x.cos().min()

        def true_fn(x):
            pred = x.shape[0] == 1
            return cond(pred, true_true_fn, true_false_fn, [x])

        def false_fn(x):
            return x.sum()

        def f(x):
            pred = x.shape[0] == 1
            return cond(pred, true_fn, false_fn, [x])

        example_inputs = (torch.ones(4, 5),)
        functional_f = torch.func.functionalize(f)
        with self.assertRaisesRegex(UnsupportedAliasMutationException, "One of torch.cond branch"):
            functional_f(*example_inputs)

        with self.assertRaisesRegex(UnsupportedAliasMutationException, "One of torch.cond branch"):
            make_fx(torch.func.functionalize(f))(*example_inputs)

    def test_cond_functionalized_nested_input_mutation_with_aot_func(self):
        def true_true_fn(x):
            x.add_(4)
            return x.sin().max()

        def true_false_fn(x):
            return x.cos().min()

        def true_fn(x):
            pred = x.shape[0] == 1
            return cond(pred, true_true_fn, true_false_fn, [x])

        def false_fn(x):
            return x.sum()

        def f(x):
            pred = x.shape[0] == 1
            return cond(pred, true_fn, false_fn, [x])

        example_input = torch.ones(4, 5)
        example_input_func = to_fun(example_input)
        torch._enable_functionalization(reapply_views=False)
        try:
            with self.assertRaisesRegex(UnsupportedAliasMutationException, "One of torch.cond branch"):
                f(example_input_func)

            with self.assertRaisesRegex(UnsupportedAliasMutationException, "One of torch.cond branch"):
                make_fx(f)(example_input_func)
        finally:
            torch._disable_functionalization()

        def f_wrapper(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                torch._enable_functionalization(reapply_views=False)
                try:
                    return func(*args, **kwargs)
                finally:
                    torch._disable_functionalization()
            return wrapper

        with self.assertRaisesRegex(UnsupportedAliasMutationException, "One of torch.cond branch"):
            make_fx(f_wrapper(f))(example_input_func)

    def test_cond_functionalized_input_aliasing_with_aot_func(self):
        def true_fn(x):
            return x

        def false_fn(x):
            view_x = x.view(x.shape)
            return view_x

        def f(x):
            pred = x.shape[0] == 4
            return cond(pred, true_fn, false_fn, [x])

        example_input = torch.ones(5, 5)
        try:
            example_input_func = to_fun(example_input)
            torch._enable_functionalization(reapply_views=False)
            with self.assertRaisesRegex(UnsupportedAliasMutationException, "One of torch.cond branch might be aliasing"):
                f(example_input_func)
        finally:
            torch._disable_functionalization()

        def f_wrapper(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                torch._enable_functionalization(reapply_views=False)
                try:
                    func_args = pytree.tree_map(to_fun, args)
                    func_kwargs = pytree.tree_map(to_fun, kwargs)
                    return func(*func_args, **func_kwargs)
                finally:
                    torch._disable_functionalization()
            return wrapper

        with self.assertRaisesRegex(UnsupportedAliasMutationException, "One of torch.cond branch might be aliasing"):
            make_fx(f_wrapper(f))(example_input)

    def test_cond_functionalized_aot_func_check_functional(self):
        def true_fn(x):
            return x.cos()

        def false_fn(x):
            y = x.sin()
            y.add_(5)
            return y

        def f(x):
            pred = x.shape[0] == 4
            return cond(pred, true_fn, false_fn, [x])

        example_input = torch.ones(5, 5)
        example_input_func = to_fun(example_input)

        def f_wrapper(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                torch._enable_functionalization(reapply_views=False)
                try:
                    func_args = pytree.tree_map(to_fun, args)
                    func_kwargs = pytree.tree_map(to_fun, kwargs)
                    return pytree.tree_map(from_fun, func(*func_args, **func_kwargs))
                finally:
                    torch._disable_functionalization()
            return wrapper

        result_gm = make_fx(f_wrapper(f))(example_input)
        for node in result_gm.true_graph_0.graph.nodes:
            if node.op == "call_function":
                self.assertTrue(not node.target._schema.is_mutable)

        for node in result_gm.false_graph_0.graph.nodes:
            if node.op == "call_function":
                self.assertTrue(not node.target._schema.is_mutable)

        self.assertEqual(result_gm(torch.ones(5, 5)), f(torch.ones(5, 5)))

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
            conditional = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, [x_1]);
            pred_1 = true_graph_0 = false_graph_0 = None
            true_graph_1 = self.true_graph_1
            false_graph_1 = self.false_graph_1
            conditional_1 = torch.ops.higher_order.cond(pred2_1, true_graph_1, false_graph_1, [x_1, x_1]);
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
        def forward(self, y_1):
            mul = torch.ops.aten.mul.Tensor(y_1, y_1);  y_1 = None
            return mul
        """
        # Normalization hack, cause .code makes some weird whitespace
        code = "".join(code.split())
        out = "".join(out.split())
        self.assertEqual(code, out)

    def test_raise_error_on_mismatch_type_size(self):
        def true_fn(x):
            return x.sin()

        def false_fn(x):
            return (x, x)

        def f(x, y):
            return cond(y, true_fn, false_fn, [x])

        x = torch.randn(4)
        with self.assertRaisesRegex(
            CondOpArgsMismatchError,
            "Expected to return same number of outputs but got",
        ):
            make_fx(f)(x, torch.tensor(False))

    def test_raise_error_on_mismatch_tensor_size(self):
        def true_fn(x):
            return x.sin()

        def false_fn(x):
            return torch.zeros([10, 10])

        def f(x, y):
            return cond(y, true_fn, false_fn, [x])

        x = torch.randn(4)
        with self.assertRaisesRegex(
            CondOpArgsMismatchError,
            "Expected each tensor to have same metadata but got",
        ):
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
            conditional = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, [x_1]);
            pred_1 = true_graph_0 = false_graph_0 = None
            true_graph_1 = self.true_graph_1
            false_graph_1 = self.false_graph_1
            conditional_1 = torch.ops.higher_order.cond(pred2_1, true_graph_1, false_graph_1, [x_1, x_1]);
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
        def forward(self, y_1):
            mul = torch.ops.aten.mul.Tensor(y_1, y_1);  y_1 = None
            return mul
        """
        # Normalization hack, cause .code makes some weird whitespace
        code = "".join(code.split())
        out = "".join(out.split())
        self.assertEqual(code, out)

    def test_raise_error_on_mismatch_type_size_fake_tensor(self):
        def true_fn(x):
            return x.sin()

        def false_fn(x):
            return (x, x)

        def f(x, y):
            return cond(y, true_fn, false_fn, [x])

        x = torch.randn(4)
        with self.assertRaisesRegex(
            CondOpArgsMismatchError,
            "Expected to return same number of outputs but got",
        ):
            make_fx(f, tracing_mode="fake")(x, torch.tensor(False))


    def test_raise_error_on_mismatch_tensor_size_fake_tensor(self):
        def true_fn(x):
            return x.sin()

        def false_fn(x):
            return torch.zeros([10, 10])

        def f(x, y):
            return cond(y, true_fn, false_fn, [x])

        x = torch.randn(4)
        with self.assertRaisesRegex(
            CondOpArgsMismatchError,
            "Expected each tensor to have same metadata but got",
        ):
            make_fx(f, tracing_mode="fake")(x, torch.tensor(False))

    def check_map_count(self, gm, op_count):
        i = 0
        for m in gm.modules():
            for node in m.graph.nodes:
                if node.op == "call_function" and node.target == torch.ops.map_impl:
                    i += 1
        self.assertEqual(i, op_count)

    def test_tracing_map_real(self):
        def f(x, y):
            return x + y

        def g(xs, y):
            return control_flow.map(f, xs, y)

        gm = make_fx(g, tracing_mode="real")(torch.ones(3, 2, 2), torch.ones(2))
        x = torch.randn(3, 2, 2)
        y = torch.randn(2)
        res = gm(x, y)
        self.assertEqual(res, g(x, y))
        self.check_map_count(gm, 1)

    def test_tracing_map_symbolic_simple(self):
        def f(x, y):
            return x + y

        def g(xs, y):
            return control_flow.map(f, xs, y)

        gm = make_fx(g, tracing_mode="symbolic")(torch.ones(3, 2, 4), torch.ones(4))
        x = torch.randn(3, 2, 2)
        y = torch.randn(2)
        res = gm(x, y)
        self.assertEqual(res, g(x, y))
        self.check_map_count(gm, 1)

    def test_tracing_map_symbolic_list(self):
        def f(x, y):
            return [x[0][0] + y, x[1] * y]

        def g(xs, y, z):
            out = control_flow.map(f, xs, y)
            return out[0] + z, out[1] * z

        example_x = [[torch.ones(3, 4, 5)], torch.ones(3, 4, 5)]
        gm = make_fx(g, tracing_mode="symbolic")(example_x, torch.ones(5), torch.ones(5))
        x = [[torch.randn(4, 5, 6)], torch.ones(4, 5, 6)]
        y = torch.randn(6)
        z = torch.ones(6)
        res = gm(x, y, z)
        self.assertEqual(res, g(x, y, z))
        self.check_map_count(gm, 1)

    def test_tracing_map_symbolic_dict(self):
        def f(x, y):
            return {"d": x["b"]["a"] + y, "e": x["c"] * y}

        def g(xs, y, z):
            out = control_flow.map(f, xs, y)
            return {"f": out["d"] + z, "g": out["e"] * z}

        example_x = {"b": {"a": torch.ones(3, 4, 5)}, "c": torch.ones(3, 4, 5)}
        gm = make_fx(g, tracing_mode="symbolic")(example_x, torch.ones(5), torch.ones(5))
        x = {"b": {"a": torch.randn(4, 5, 6)}, "c": torch.ones(4, 5, 6)}
        y = torch.randn(6)
        z = torch.ones(6)
        res = gm(x, y, z)
        self.assertEqual(res, g(x, y, z))
        self.check_map_count(gm, 1)

    def test_tracing_map_autograd_symbolic_simple(self):
        def f(x, y):
            return x + y

        def g(xs, y):
            out = control_flow.map(f, xs, y)
            return torch.autograd.grad(out, (xs, y), torch.ones_like(out))

        gm = make_fx(g, tracing_mode="symbolic")(torch.ones(3, 4, 5, requires_grad=True), torch.ones(5, requires_grad=True))
        x = torch.randn(4, 5, 6, requires_grad=True)
        y = torch.randn(6, requires_grad=True)
        res = gm(x, y)
        self.assertEqual(res, g(x, y))
        self.check_map_count(gm, 2)


    def test_tracing_map_autograd_symbolic_list(self):
        import torch.utils._pytree as pytree

        def f(x, y):
            return [x[0].cos() + y.sin(), x[1].sin() * y.cos()]

        def g(xs, y):
            out = control_flow.map(f, xs, y)
            flat_out, _ = pytree.tree_flatten(out)
            flat_inp, _ = pytree.tree_flatten((xs, y))
            requires_grad_inp = [inp for inp in flat_inp if inp.requires_grad]
            return torch.autograd.grad(flat_out, requires_grad_inp, [torch.ones_like(out) for out in flat_out])

        gm = make_fx(g, tracing_mode="symbolic")(
            [torch.ones(3, 4, 5), torch.ones(3, 4, 5, requires_grad=True)],
            torch.ones(5, requires_grad=True))
        x = [torch.randn(4, 5, 6), torch.ones(4, 5, 6, requires_grad=True)]
        y = torch.randn(6, requires_grad=True)
        res = gm(x, y)
        self.assertEqual(res, g(x, y))
        self.check_map_count(gm, 2)

    def test_tracing_map_autograd_symbolic_dict(self):
        def f(x, y):
            return [x["a"] + y, x["b"] * y]

        def g(xs, y):
            out = control_flow.map(f, xs, y)
            flat_out, _ = pytree.tree_flatten(out)
            flat_inp, _ = pytree.tree_flatten((xs, y))
            requires_grad_inp = [inp for inp in flat_inp if inp.requires_grad]
            return torch.autograd.grad(flat_out, requires_grad_inp, [torch.ones_like(out) for out in flat_out])

        traced_x = {"a": torch.ones(3, 4, 5, requires_grad=True), "b": torch.ones(3, 4, 5, requires_grad=True)}
        gm = make_fx(g, tracing_mode="symbolic")(traced_x, torch.ones(5, requires_grad=True))
        x = {"a": torch.randn(4, 5, 6, requires_grad=True), "b": torch.ones(4, 5, 6, requires_grad=True)}
        y = torch.randn(6, requires_grad=True)
        res = gm(x, y)
        self.assertEqual(res, g(x, y))
        self.check_map_count(gm, 2)

    def test_tracing_map_autograd_aot_functionalized(self):
        def inner(x, y):
            z = x - 1
            z.add_(1)
            return z * y

        def f(xs, y):
            res = control_flow.map(inner, xs, y)
            grads = torch.autograd.grad(res, (xs, y), torch.ones_like(res))
            return grads

        def f_wrapper(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                torch._enable_functionalization(reapply_views=False)
                try:
                    return pytree.tree_map(from_fun, func(*args, **kwargs))
                finally:
                    torch._disable_functionalization()
            return wrapper

        example_inputs = (torch.ones(3, 2, 4, requires_grad=True), torch.ones(2, 4, requires_grad=True))
        gm = make_fx(f, tracing_mode="symbolic")(*example_inputs)
        fgm = make_fx(f_wrapper(f), tracing_mode="symbolic")(*example_inputs)
        xs = torch.ones(3, 4, 5, requires_grad=True)
        y = torch.ones(4, 5, requires_grad=True)

        self.assertEqual(gm(xs, y), f(xs, y))

        def count_mutable(gm):
            c = 0
            for node in gm.graph.nodes:
                if node.op == "call_function":
                    if node.target == torch.ops.map_impl:
                        c += count_mutable(getattr(gm, str(node.args[0])))
                    elif schema := getattr(node.target, "_schema", None):
                        c += int(schema.is_mutable)
            return c
        self.assertEqual(count_mutable(fgm), 0)
        # One for forward, one for recompuation logic in backward
        self.assertEqual(count_mutable(gm), 2)

    def test_map_functionalized(self):
        def map_fn(x, y):
            z = x + y
            z.add_(4)
            return z

        def f(xs, y):
            return control_flow.map(map_fn, xs, y)

        example_inputs = (torch.ones(3, 2, 4), torch.ones(4))
        functional_f = torch.func.functionalize(f)
        self.assertEqual(functional_f(*example_inputs), f(*example_inputs))

        gm = make_fx(torch.func.functionalize(f))(*example_inputs)
        self.assertEqual(gm(*example_inputs), f(*example_inputs))

        gm = make_fx(torch.func.functionalize(f), tracing_mode="symbolic")(*example_inputs)
        self.assertEqual(gm(*example_inputs), f(*example_inputs))

        for node in gm.body_graph_0.graph.nodes:
            if node.op == "call_function":
                self.assertTrue(not node.target._schema.is_mutable)
        self.check_map_count(gm, 1)

    def test_map_functionalized_aot_func(self):
        def map_fn(x, y):
            z = x + y
            z.add_(4)
            return z

        def f(xs, y):
            return control_flow.map(map_fn, xs, y)

        def f_wrapper(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                torch._enable_functionalization(reapply_views=False)
                try:
                    return pytree.tree_map(from_fun, func(*args, **kwargs))
                finally:
                    torch._disable_functionalization()
            return wrapper

        example_inputs = (torch.ones(3, 2, 4), torch.ones(4))

        gm = make_fx(f_wrapper(f))(*example_inputs)

        for node in gm.body_graph_0.graph.nodes:
            if node.op == "call_function":
                self.assertTrue(not node.target._schema.is_mutable)

        self.assertEqual(gm(*example_inputs), f(*example_inputs))

    def test_map_functionalized_arg_mutation(self):
        def map_fn(x, y):
            y.add_(4)
            return x + y

        def f(xs, y):
            return control_flow.map(map_fn, xs, y)

        example_inputs = (torch.ones(3, 2, 4), torch.ones(4))
        functional_f = torch.func.functionalize(f)
        with self.assertRaisesRegex(UnsupportedAliasMutationException, "torch.map is mutating the input!"):
            functional_f(*example_inputs)

    def test_map_functionalized_elem_mutation(self):
        def map_fn(x, y):
            x.add_(4)
            return x + y

        def f(xs, y):
            return control_flow.map(map_fn, xs, y)

        example_inputs = (torch.ones(3, 2, 4), torch.ones(4))
        functional_f = torch.func.functionalize(f)
        with self.assertRaisesRegex(UnsupportedAliasMutationException, "torch.map is mutating the input!"):
            functional_f(*example_inputs)

    def test_cond_autograd_fail(self):
        def true_fn(x):
            return x.cos()

        def false_fn(x):
            return x.sin()

        def f(x, y):
            return control_flow.cond(x.shape[0] > 4, true_fn, false_fn, [y])

        example_inputs = (torch.ones(3, 2, 4, requires_grad=True), torch.ones(4, requires_grad=True))
        with self.assertRaisesRegex(RuntimeError, "Autograd not implemented for cond"):
            f(*example_inputs).sum().backward()

        # Ensure no error is thrown when not running backward
        f(*example_inputs)

    def test_map_functionalized_elem_alias(self):
        def map_fn(x):
            x.view(x.shape)
            return x

        def f(xs):
            return control_flow.map(map_fn, xs)

        example_inputs = (torch.ones(3, 2, 4),)
        functional_f = torch.func.functionalize(f)
        with self.assertRaisesRegex(UnsupportedAliasMutationException, "torch.map is aliasing the input!"):
            functional_f(*example_inputs)

    def test_nested_map_cond_real(self):
        def true_fn(x, y):
            return x * y

        def false_fn(x, y):
            return x + y

        def f(x, pred, y):
            return cond(pred, true_fn, false_fn, [x, y])

        def g(pred, xs, y):
            return control_flow.map(f, xs, pred, y)

        gm = make_fx(g, tracing_mode="real")(
            torch.tensor(True), torch.ones(3, 2, 4), torch.ones(4)
        )
        pred = torch.tensor(False)
        x = torch.randn(3, 2, 4)
        y = torch.randn(4)
        res = gm(pred, x, y)
        self.assertEqual(res, g(pred, x, y))
        self.check_map_count(gm, 1)

    def test_nested_map_cond_symbolic(self):
        def true_fn(x, y):
            return x * y

        def false_fn(x, y):
            return x + y

        def f(x, pred, y):
            return cond(pred, true_fn, false_fn, [x, y])

        def g(pred, xs, y):
            return control_flow.map(f, xs, pred, y)

        gm = make_fx(g, tracing_mode="symbolic")(
            torch.tensor(True), torch.ones(3, 2, 4), torch.ones(4)
        )
        pred = torch.tensor(False)
        x = torch.randn(3, 2, 2)
        y = torch.randn(2)
        res = gm(pred, x, y)
        self.assertEqual(res, g(pred, x, y))
        self.check_map_count(gm, 1)

    def test_nested_cond_map_cond_symbolic(self):

        def true_fn(x, y):
            return x * y

        def false_fn(x, y):
            return x + y

        def f(x, pred, y):
            return cond(pred, true_fn, false_fn, [x, y])

        def g(pred, xs, y):
            return control_flow.map(f, xs, pred, y)

        def main_true_fn(pred, xs, y):
            return g(pred, xs, y) * 2

        def main_false_fn(pred, xs, y):
            return g(pred, xs, y) + 1

        def main(p, pred, xs, y):
            return cond(p, main_true_fn, main_false_fn, [pred, xs, y])

        gm = make_fx(main, tracing_mode="symbolic")(
            torch.tensor(True), torch.tensor(True), torch.ones(3, 2, 4), torch.ones(4)
        )
        p = torch.tensor(False)
        pred = torch.tensor(False)
        xs = torch.randn(3, 2, 2)
        y = torch.randn(2)
        res = gm(p, pred, xs, y)
        self.assertEqual(res, main(p, pred, xs, y))
        self.check_map_count(gm, 2)

    def test_cond_with_sym_pred(self):
        def true_fn(x):
            return x + x

        def false_fn(x):
            return x * x

        def foo(x):
            return cond(x.shape[0] == 4, true_fn, false_fn, [x])

        gm = make_fx(foo, tracing_mode="symbolic")(torch.ones(3, 2, 1))
        x = torch.ones(4, 3, 2)
        self.assertEqual(foo(x), gm(x))

if __name__ == '__main__':
    run_tests()
