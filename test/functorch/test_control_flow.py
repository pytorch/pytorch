# Owner(s): ["module: functorch"]
import contextlib
import functools
import unittest

import torch
import torch.utils._pytree as pytree
from functorch.experimental import control_flow
from functorch.experimental.control_flow import cond, UnsupportedAliasMutationException
from torch._higher_order_ops.while_loop import while_loop
from torch._subclasses.functional_tensor import (
    CppFunctionalizeAPI,
    FunctionalTensor,
    FunctionalTensorMode,
    PythonFunctionalizeAPI,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_quantization import skipIfNoDynamoSupport

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_WINDOWS,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
)


# TODO: pull these helpers from AOTAutograd later
def to_fun(t):
    if isinstance(t, torch.Tensor):
        return FunctionalTensor.to_functional(t)
    return t


def from_fun(t):
    if not isinstance(t, FunctionalTensor):
        # quick sanity assert
        if isinstance(t, torch.Tensor):
            assert not torch._is_functional_tensor(t)
        return t
    torch._sync(t)
    return torch._from_functional_tensor(t.elem)


def to_fun_old(t):
    if isinstance(t, torch.Tensor) and not torch._is_functional_tensor(t):
        out = torch._to_functional_tensor(t)
        torch._mirror_autograd_meta_to(t, out)
        return out
    return t


def from_fun_old(t):
    # quick sanity assert
    if isinstance(t, torch.Tensor):
        assert torch._is_functional_tensor(t)
        torch._sync(t)
        return torch._from_functional_tensor(t)
    return t


def _fake_map(f, x, *args):
    from functorch.experimental.control_flow import _stack_pytree, _unstack_pytree

    x_pytrees = _unstack_pytree(x)
    zs = []
    for xp in x_pytrees:
        zs.append(f(xp, *args))
    return _stack_pytree(zs)


def _fake_while_loop(cond_fn, body_fn, operands):
    while cond_fn(*operands):
        operands = body_fn(*operands)
    return operands


def _while_loop_tests():
    def simple(x):
        def cond_fn(x):
            return x.sum() < 10

        def body_fn(x):
            return (x + 1,)

        return while_loop(cond_fn, body_fn, (x,))

    def simple_with_mutation(x):
        def cond_fn(x):
            y = x.clone().add_(1).add_(-1)
            return y.sum() < 10

        def body_fn(x):
            y = x.clone().add_(1).add_(-1)
            return (y + 1,)

        return while_loop(cond_fn, body_fn, (x,))

    def nested(out_iter, it, y):
        def cond_fn(out_iter, it, y):
            return it.sum() < 10

        def body_fn(out_iter, it, y):
            return (out_iter.clone(), it + y, y + 1)

        def outer_cond_fn(out_iter, it, y):
            return out_iter.sum() < 2

        def outer_body_fn(out_iter, it, y):
            out_iter, it, y = while_loop(cond_fn, body_fn, (out_iter, it, y))
            return (out_iter + 1, it, y)

        return while_loop(outer_cond_fn, outer_body_fn, (out_iter, it, y))

    class Nested(torch.nn.Module):
        def forward(self, ci, cj, a, b):
            def cond_fn(i1, j1, x1, y1):
                return i1 > 0

            def body_fn(i1, j1, x1, y1):
                def cond_fn_nested(i2, j2, x2, y2):
                    return j2 > 0

                def body_fn_nested(i2, j2, x2, y2):
                    return i2.clone(), j2 - 1, x2 + 3.14, y2 - 2.71

                i1, j1, x1, y1 = while_loop(
                    cond_fn_nested, body_fn_nested, [i1, j1, x1, y1]
                )
                return i1 - 1, j1.clone(), x1 * 2, y1 / 2

            return while_loop(cond_fn, body_fn, (ci, cj, a, b))

    class SimpleWithLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
            self.register_buffer("dec", torch.tensor(1))

        def forward(self, iter, x):
            def cond_fn(it, x):
                return it - self.dec > 0

            def body_fn(it, x):
                return it - 1, self.linear(x)

            return while_loop(cond_fn, body_fn, (iter, x))

    class NestedWithLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mod = SimpleWithLinear()
            self.outer_linear = torch.nn.Linear(2, 2)
            self.register_buffer("dec", torch.tensor(1))

        def forward(self, iter, x):
            def cond_fn(it, x):
                return it - self.dec > 0

            def body_fn(it, x):
                return it - 1, self.outer_linear(self.mod(it, x)[1])

            return while_loop(cond_fn, body_fn, (iter, x))

    nested2 = Nested()
    simple_with_linear = SimpleWithLinear()
    nested_with_linear = NestedWithLinear()

    x = torch.zeros(1)
    y = torch.zeros(1)
    z = torch.zeros(1)
    return {
        "simple": (simple, (x,)),
        "nested": (nested, (x, y, z)),
        "nested2": (
            nested2,
            (torch.tensor(2), torch.tensor(2), torch.ones(2, 2), torch.ones(2, 2)),
        ),
        "simple_with_mutation": (simple_with_mutation, (x,)),
        "simple_with_linear": (
            simple_with_linear,
            (torch.tensor(3), torch.randn(2, 2)),
        ),
        "nested_with_linear": (
            nested_with_linear,
            (torch.tensor(3), torch.randn(2, 2)),
        ),
    }


WHILE_LOOP_TESTS = _while_loop_tests()


def collect_meta_for_filtered_nodes(
    gm: torch.fx.GraphModule, node_names, meta_field_name
):
    ret = []
    for mod in gm.modules():
        for node in mod.graph.nodes:
            if node.name in node_names:
                for field_name in meta_field_name:
                    ret.append(node.meta.get(field_name))
    return ret


def reduce_func(*operands):
    acc = 0
    for operand in operands:
        acc += operand
    return acc


class ReduceObj:
    def __call__(self, *operands):
        return reduce_func(*operands)


class ReduceMod(torch.nn.Module):
    def _reduce(self, *operands):
        return reduce_func(*operands)

    def forward(self, *operands):
        return self._reduce(*operands)


@unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
@skipIfNoDynamoSupport
class TestControlFlow(TestCase):
    def setUp(self):
        torch._dynamo.reset()
        super().setUp()

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

    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA.")
    def test_while_loop_gpu(self):
        def cond_fn(x):
            return x.sum() < 10

        def body_fn(x):
            return (x + 1,)

        x = torch.zeros(1, device="cuda")
        res = while_loop(cond_fn, body_fn, (x,))
        expected = _fake_while_loop(cond_fn, body_fn, (x,))
        self.assertEqual(expected, res)

    def test_map_illegal_inputs(self):
        def f(x, y):
            return x[0] + x[1] + y

        with self.assertRaisesRegex(
            RuntimeError,
            r"Mapped xs can only consist of tensors\. Got xs \[3, tensor\(\[1\., 1\.\]\)\]\.",
        ):
            _ = control_flow.map(f, (3, torch.ones(2)), torch.ones(2))

        with self.assertRaisesRegex(
            RuntimeError, r"Leading dimensions of mapped xs cannot be 0\."
        ):
            _ = control_flow.map(
                f, (torch.ones(0, 1, 2), torch.ones(0, 1, 2)), torch.ones(2)
            )

        with self.assertRaisesRegex(
            RuntimeError,
            r"Leading dimensions of mapped xs must be consistent\. "
            r"Got shapes \[torch\.Size\(\[3, 4, 5\]\), torch\.Size\(\[4, 4, 5\]\)\]\.",
        ):
            _ = control_flow.map(
                f, (torch.ones(3, 4, 5), torch.ones(4, 4, 5)), torch.ones(5)
            )

    def test_map_illegal_outputs(self):
        def f(x, y):
            return x.item()

        def f1(x, y):
            return y.size()

        def f2(x, y):
            return None

        x = torch.ones([3])
        y = torch.ones([1, 2, 3])
        with self.assertRaisesRegex(
            RuntimeError, r"Expect outputs of map only contains tensors or None\."
        ):
            _ = control_flow.map(f, x, y)

        with self.assertRaisesRegex(
            RuntimeError, r"Expect outputs of map only contains tensors or None\."
        ):
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
            flat_x = pytree.tree_leaves(x)
            flat_z = pytree.tree_leaves(z)
            grads = torch.autograd.grad(
                flat_z, flat_x, [torch.ones_like(z) for z in flat_z]
            )
            return z, grads

        x = [
            [
                torch.randn(3, 2, 2, requires_grad=True),
                torch.randn(3, 2, 1, requires_grad=True),
            ],
            torch.ones(3, 1, 2, requires_grad=True),
        ]
        y = torch.ones(1, requires_grad=True)
        true_outs = fwbw(control_flow.map, f, x, y)
        fake_outs = fwbw(_fake_map, f, x, y)
        self.assertEqual(true_outs, fake_outs)


@unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
@skipIfNoDynamoSupport
class TestControlFlowTraced(TestCase):
    def setUp(self):
        torch._dynamo.reset()
        super().setUp()

    def _check_tracing(self, fn, args, allow_non_fake_inputs=False):
        graphs = {}
        eager_res = fn(*args)
        for tracing_mode in ["symbolic", "real", "fake"]:
            graph = make_fx(
                fn,
                tracing_mode=tracing_mode,
                _allow_non_fake_inputs=allow_non_fake_inputs,
            )(*args)
            graphs[tracing_mode] = graph
            self.assertEqual(graph(*args), eager_res)
        return graphs

    def _check_compile(self, fn, args, *, backend="eager"):
        eager_res = fn(*args)
        compiled_fn = torch.compile(fn, backend=backend)
        self.assertEqual(compiled_fn(*args), eager_res)

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

    def test_while_loop_nested_traced(self):
        fn, inp = WHILE_LOOP_TESTS["nested"]
        graphs = self._check_tracing(fn, inp)
        self.assertExpectedInline(
            graphs["symbolic"].code.strip("\n"),
            """\
def forward(self, out_iter_1, it_1, y_1):
    while_loop_cond_graph_0 = self.while_loop_cond_graph_0
    while_loop_body_graph_0 = self.while_loop_body_graph_0
    while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_0, while_loop_body_graph_0, (out_iter_1, it_1, y_1), ());  while_loop_cond_graph_0 = while_loop_body_graph_0 = out_iter_1 = it_1 = y_1 = None
    getitem = while_loop[0]
    getitem_1 = while_loop[1]
    getitem_2 = while_loop[2];  while_loop = None
    return (getitem, getitem_1, getitem_2)
    """,  # noqa: B950
        )
        self.assertExpectedInline(
            graphs["symbolic"].while_loop_cond_graph_0.code.strip("\n"),
            """\
def forward(self, arg0_1, arg1_1, arg2_1):
    sum_1 = torch.ops.aten.sum.default(arg0_1);  arg0_1 = None
    lt = torch.ops.aten.lt.Scalar(sum_1, 2);  sum_1 = None
    return lt
    """,
        )
        self.assertExpectedInline(
            graphs["symbolic"].while_loop_body_graph_0.code.strip("\n"),
            """\
def forward(self, arg0_1, arg1_1, arg2_1):
    while_loop_cond_graph_0 = self.while_loop_cond_graph_0
    while_loop_body_graph_0 = self.while_loop_body_graph_0
    while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_0, while_loop_body_graph_0, (arg0_1, arg1_1, arg2_1), ());  while_loop_cond_graph_0 = while_loop_body_graph_0 = arg0_1 = arg1_1 = arg2_1 = None
    getitem = while_loop[0]
    getitem_1 = while_loop[1]
    getitem_2 = while_loop[2];  while_loop = None
    add = torch.ops.aten.add.Tensor(getitem, 1);  getitem = None
    return (add, getitem_1, getitem_2)
    """,  # noqa: B950
        )

    def _wrap_with_functionalize(self, fn, func_type):
        mode = None
        if func_type == "cpp":
            fn = CppFunctionalizeAPI().functionalize(fn)
        elif func_type == "python":
            fn = PythonFunctionalizeAPI().functionalize(fn)
            mode = FunctionalTensorMode()
        elif func_type == "functorch":
            fn = torch.func.functionalize(fn)
        else:
            assert func_type == "no"
        return fn, mode

    @parametrize("func_type", ["no", "cpp", "python", "functorch"])
    def test_while_loop_simple_functionalize_check_graph(self, func_type):
        fn, inp = WHILE_LOOP_TESTS["simple_with_mutation"]
        fn, mode = self._wrap_with_functionalize(fn, func_type)
        mode = mode if mode is not None else contextlib.nullcontext()
        with mode:
            graphs = self._check_tracing(fn, inp)
        if func_type == "no":
            self.assertExpectedInline(
                graphs["symbolic"].code.strip("\n"),
                """\
def forward(self, x_1):
    while_loop_cond_graph_0 = self.while_loop_cond_graph_0
    while_loop_body_graph_0 = self.while_loop_body_graph_0
    while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_0, while_loop_body_graph_0, (x_1,), ());  while_loop_cond_graph_0 = while_loop_body_graph_0 = x_1 = None
    getitem = while_loop[0];  while_loop = None
    return (getitem,)
    """,  # noqa: B950
            )
            self.assertExpectedInline(
                graphs["symbolic"].while_loop_cond_graph_0.code.strip("\n"),
                """\
def forward(self, arg0_1):
    clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
    add_ = torch.ops.aten.add_.Tensor(clone, 1);  clone = None
    add__1 = torch.ops.aten.add_.Tensor(add_, -1);  add_ = None
    sum_1 = torch.ops.aten.sum.default(add__1);  add__1 = None
    lt = torch.ops.aten.lt.Scalar(sum_1, 10);  sum_1 = None
    return lt
    """,
            )
            self.assertExpectedInline(
                graphs["symbolic"].while_loop_body_graph_0.code.strip("\n"),
                """\
def forward(self, arg0_1):
    clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
    add_ = torch.ops.aten.add_.Tensor(clone, 1);  clone = None
    add__1 = torch.ops.aten.add_.Tensor(add_, -1);  add_ = None
    add = torch.ops.aten.add.Tensor(add__1, 1);  add__1 = None
    return (add,)
    """,
            )
        elif func_type == "python":
            self.assertExpectedInline(
                graphs["symbolic"].code.strip("\n"),
                """\
def forward(self, arg0_1):
    while_loop_cond_graph_0 = self.while_loop_cond_graph_0
    while_loop_body_graph_0 = self.while_loop_body_graph_0
    while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_0, while_loop_body_graph_0, (arg0_1,), ());  while_loop_cond_graph_0 = while_loop_body_graph_0 = arg0_1 = None
    getitem = while_loop[0];  while_loop = None
    return (getitem,)
    """,  # noqa: B950
            )
            self.assertExpectedInline(
                graphs["symbolic"].while_loop_cond_graph_0.code.strip("\n"),
                """\
def forward(self, arg0_1):
    clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
    add = torch.ops.aten.add.Tensor(clone, 1);  clone = None
    add_1 = torch.ops.aten.add.Tensor(add, -1);  add = None
    sum_1 = torch.ops.aten.sum.default(add_1);  add_1 = None
    lt = torch.ops.aten.lt.Scalar(sum_1, 10);  sum_1 = None
    return lt
    """,
            )
            self.assertExpectedInline(
                graphs["symbolic"].while_loop_body_graph_0.code.strip("\n"),
                """\
def forward(self, arg0_1):
    clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
    add = torch.ops.aten.add.Tensor(clone, 1);  clone = None
    add_1 = torch.ops.aten.add.Tensor(add, -1);  add = None
    add_2 = torch.ops.aten.add.Tensor(add_1, 1);  add_1 = None
    return (add_2,)
    """,
            )
        else:
            self.assertExpectedInline(
                graphs["symbolic"].code.strip("\n"),
                """\
def forward(self, x_1):
    while_loop_cond_graph_0 = self.while_loop_cond_graph_0
    while_loop_body_graph_0 = self.while_loop_body_graph_0
    while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_0, while_loop_body_graph_0, (x_1,), ());  while_loop_cond_graph_0 = while_loop_body_graph_0 = x_1 = None
    getitem = while_loop[0];  while_loop = None
    return (getitem,)
    """,  # noqa: B950
            )
            self.assertExpectedInline(
                graphs["symbolic"].while_loop_cond_graph_0.code.strip("\n"),
                """\
def forward(self, arg0_1):
    clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
    add = torch.ops.aten.add.Tensor(clone, 1);  clone = None
    add_1 = torch.ops.aten.add.Tensor(add, -1);  add = None
    sum_1 = torch.ops.aten.sum.default(add_1);  add_1 = None
    lt = torch.ops.aten.lt.Scalar(sum_1, 10);  sum_1 = None
    return lt
    """,
            )
            self.assertExpectedInline(
                graphs["symbolic"].while_loop_body_graph_0.code.strip("\n"),
                """\
def forward(self, arg0_1):
    clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
    add = torch.ops.aten.add.Tensor(clone, 1);  clone = None
    add_1 = torch.ops.aten.add.Tensor(add, -1);  add = None
    add_2 = torch.ops.aten.add.Tensor(add_1, 1);  add_1 = None
    return (add_2,)
    """,
            )

    @parametrize("func_type", ["no", "cpp", "python", "functorch"])
    @parametrize("while_loop_test", list(WHILE_LOOP_TESTS.keys()))
    def test_while_loop_functionalize(self, func_type, while_loop_test):
        # simple_with_linear doesn't work becaue parameters and buffers
        # are not inputs so they're not wrapped by functionalization and tracing.
        if while_loop_test not in ("simple_with_linear", "nested_with_linear"):
            fn, inp = WHILE_LOOP_TESTS[while_loop_test]
            fn, mode = self._wrap_with_functionalize(fn, func_type)
            mode = mode if mode is not None else contextlib.nullcontext()
            with mode:
                self._check_tracing(fn, inp)

    @parametrize("while_loop_test", list(WHILE_LOOP_TESTS.keys()))
    def test_while_loop_tracing(self, while_loop_test):
        fn, inp = WHILE_LOOP_TESTS[while_loop_test]
        allow_non_fake_inputs = (
            False
            if while_loop_test not in ("simple_with_linear", "nested_with_linear")
            else True
        )
        self._check_tracing(fn, inp, allow_non_fake_inputs)

    @parametrize("backend", ["eager", "aot_eager"])
    @parametrize("while_loop_test", list(WHILE_LOOP_TESTS.keys()))
    def test_while_loop_compile(self, backend, while_loop_test):
        fn, inp = WHILE_LOOP_TESTS[while_loop_test]
        self._check_compile(fn, inp, backend=backend)

    @skipIfTorchDynamo("Graph is not captured by backend if test with dynamo")
    def test_while_loop_simple_with_linear_compile_check_graph(self):
        fn, inp = WHILE_LOOP_TESTS["simple_with_linear"]
        from torch._dynamo.testing import EagerAndRecordGraphs

        backend = EagerAndRecordGraphs()
        torch.compile(fn, backend=backend)(*inp)
        self.assertEqual(len(backend.graphs), 1)
        gm = backend.graphs[0]
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, L_iter_ : torch.Tensor, L_x_ : torch.Tensor):
    l_iter_ = L_iter_
    l_x_ = L_x_
    l__self___dec = self.L__self___dec
    l__self___linear_weight = self.L__self___linear_weight
    l__self___linear_bias = self.L__self___linear_bias
    cond_fn_0 = self.cond_fn_0
    body_fn_0 = self.body_fn_0
    while_loop = torch.ops.higher_order.while_loop(cond_fn_0, body_fn_0, (l_iter_, l_x_), (l__self___dec, l__self___linear_bias, l__self___linear_weight));  cond_fn_0 = body_fn_0 = l_iter_ = l_x_ = l__self___dec = l__self___linear_bias = l__self___linear_weight = None
    getitem = while_loop[0]
    getitem_1 = while_loop[1];  while_loop = None
    return (getitem, getitem_1)""",  # noqa: B950
        )
        self.assertExpectedInline(
            gm.cond_fn_0.code.strip(),
            """\
def forward(self, l_iter_, l_x_, l__self___dec_cond_fn, l__self___linear_bias_body_fn, l__self___linear_weight_body_fn):
    sub = l_iter_ - l__self___dec_cond_fn;  l_iter_ = l__self___dec_cond_fn = None
    gt = sub > 0;  sub = None
    return gt""",  # noqa: B950
        )
        self.assertExpectedInline(
            gm.body_fn_0.code.strip(),
            """\
def forward(self, l_iter_, l_x_, l__self___dec_cond_fn, l__self___linear_bias_body_fn, l__self___linear_weight_body_fn):
    sub = l_iter_ - 1;  l_iter_ = None
    linear = torch._C._nn.linear(l_x_, l__self___linear_weight_body_fn, l__self___linear_bias_body_fn);  l_x_ = l__self___linear_weight_body_fn = l__self___linear_bias_body_fn = None
    return (sub, linear)""",  # noqa: B950
        )

    def test_while_loop_nested2_traced(self):
        fn, inp = WHILE_LOOP_TESTS["nested2"]
        graphs = self._check_tracing(fn, inp)
        gm = graphs["symbolic"]
        outer_body = gm.while_loop_body_graph_0
        outer_cond = gm.while_loop_cond_graph_0
        inner_body = outer_body.while_loop_body_graph_0
        inner_cond = outer_body.while_loop_cond_graph_0
        self.assertExpectedInline(
            gm.code.strip("\n"),
            """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
    while_loop_cond_graph_0 = self.while_loop_cond_graph_0
    while_loop_body_graph_0 = self.while_loop_body_graph_0
    while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_0, while_loop_body_graph_0, (arg0_1, arg1_1, arg2_1, arg3_1), ());  while_loop_cond_graph_0 = while_loop_body_graph_0 = arg0_1 = arg1_1 = arg2_1 = arg3_1 = None
    getitem = while_loop[0]
    getitem_1 = while_loop[1]
    getitem_2 = while_loop[2]
    getitem_3 = while_loop[3];  while_loop = None
    return (getitem, getitem_1, getitem_2, getitem_3)
    """,  # noqa: B950
        )
        self.assertExpectedInline(
            outer_body.code.strip("\n"),
            """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
    while_loop_cond_graph_0 = self.while_loop_cond_graph_0
    while_loop_body_graph_0 = self.while_loop_body_graph_0
    while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_0, while_loop_body_graph_0, (arg0_1, arg1_1, arg2_1, arg3_1), ());  while_loop_cond_graph_0 = while_loop_body_graph_0 = arg0_1 = arg1_1 = arg2_1 = arg3_1 = None
    getitem = while_loop[0]
    getitem_1 = while_loop[1]
    getitem_2 = while_loop[2]
    getitem_3 = while_loop[3];  while_loop = None
    sub = torch.ops.aten.sub.Tensor(getitem, 1);  getitem = None
    clone = torch.ops.aten.clone.default(getitem_1);  getitem_1 = None
    mul = torch.ops.aten.mul.Tensor(getitem_2, 2);  getitem_2 = None
    div = torch.ops.aten.div.Tensor(getitem_3, 2);  getitem_3 = None
    return (sub, clone, mul, div)
    """,  # noqa: B950
        )
        self.assertExpectedInline(
            outer_body.code.strip("\n"),
            """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
    while_loop_cond_graph_0 = self.while_loop_cond_graph_0
    while_loop_body_graph_0 = self.while_loop_body_graph_0
    while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_0, while_loop_body_graph_0, (arg0_1, arg1_1, arg2_1, arg3_1), ());  while_loop_cond_graph_0 = while_loop_body_graph_0 = arg0_1 = arg1_1 = arg2_1 = arg3_1 = None
    getitem = while_loop[0]
    getitem_1 = while_loop[1]
    getitem_2 = while_loop[2]
    getitem_3 = while_loop[3];  while_loop = None
    sub = torch.ops.aten.sub.Tensor(getitem, 1);  getitem = None
    clone = torch.ops.aten.clone.default(getitem_1);  getitem_1 = None
    mul = torch.ops.aten.mul.Tensor(getitem_2, 2);  getitem_2 = None
    div = torch.ops.aten.div.Tensor(getitem_3, 2);  getitem_3 = None
    return (sub, clone, mul, div)
    """,  # noqa: B950
        )
        self.assertExpectedInline(
            inner_body.code.strip("\n"),
            """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
    clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
    sub = torch.ops.aten.sub.Tensor(arg1_1, 1);  arg1_1 = None
    add = torch.ops.aten.add.Tensor(arg2_1, 3.14);  arg2_1 = None
    sub_1 = torch.ops.aten.sub.Tensor(arg3_1, 2.71);  arg3_1 = None
    return (clone, sub, add, sub_1)
    """,
        )
        self.assertExpectedInline(
            inner_cond.code.strip("\n"),
            """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
    gt = torch.ops.aten.gt.Scalar(arg1_1, 0);  arg1_1 = None
    return gt
    """,
        )

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

        result_true_true = graph.forward(
            x, torch.tensor(True), torch.tensor(True)
        )  # True + True -> x * x
        result_true_false = graph.forward(
            x, torch.tensor(True), torch.tensor(False)
        )  # True + True -> x + x
        result_false_true = graph.forward(
            x, torch.tensor(False), torch.tensor(True)
        )  # False + either -> cos
        result_false_false = graph.forward(
            x, torch.tensor(False), torch.tensor(False)
        )  # False + either -> cos

        self.assertNotEqual(result_true_true, result_true_false)
        self.assertFalse(torch.allclose(result_false_true, result_true_true))

        self.assertEqual(result_false_true, result_false_false)

        self.assertEqual(result_true_true, (x * x) + x)
        self.assertEqual(result_true_false, x + x + x)

        self.assertEqual(result_false_true, torch.cos(x))

        graph = make_fx(f, tracing_mode="symbolic")(
            x, torch.tensor(False), torch.tensor(False)
        )
        self.assertEqual(
            graph(x, torch.tensor(True), torch.tensor(True)),
            f(x, torch.tensor(True), torch.tensor(True)),
        )

    def test_cond_functionalized_hah(self):
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

        graph_module = make_fx(torch.func.functionalize(f), tracing_mode="symbolic")(
            *example_inputs
        )
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
        gm_functional = make_fx(
            torch.func.functionalize(gm_non_functional), tracing_mode="real"
        )(inp)
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

        graph_module1 = make_fx(torch.func.functionalize(f), tracing_mode="symbolic")(
            *example_inputs
        )
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
        with self.assertRaisesRegex(
            UnsupportedAliasMutationException, "One of torch.cond branch"
        ):
            functional_f(*example_inputs)

        with self.assertRaisesRegex(
            UnsupportedAliasMutationException, "One of torch.cond branch"
        ):
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
        with self.assertRaisesRegex(
            UnsupportedAliasMutationException, "One of torch.cond branch"
        ):
            functional_f(*example_inputs)

        with self.assertRaisesRegex(
            UnsupportedAliasMutationException, "One of torch.cond branch"
        ):
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

        with self.assertRaisesRegex(
            UnsupportedAliasMutationException,
            "One of torch.cond branch might be aliasing",
        ):
            functional_f(*example_inputs)

        with self.assertRaisesRegex(
            UnsupportedAliasMutationException,
            "One of torch.cond branch might be aliasing",
        ):
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
        with self.assertRaisesRegex(
            UnsupportedAliasMutationException, "One of torch.cond branch"
        ):
            functional_f(*example_inputs)

        with self.assertRaisesRegex(
            UnsupportedAliasMutationException, "One of torch.cond branch"
        ):
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
        try:
            example_input_func = to_fun_old(example_input)
            torch._enable_functionalization(reapply_views=False)
            with self.assertRaisesRegex(
                UnsupportedAliasMutationException, "One of torch.cond branch"
            ):
                f(example_input_func)

            with self.assertRaisesRegex(
                UnsupportedAliasMutationException, "One of torch.cond branch"
            ):
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

        with self.assertRaisesRegex(
            UnsupportedAliasMutationException, "One of torch.cond branch"
        ):
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
            example_input_func = to_fun_old(example_input)
            torch._enable_functionalization(reapply_views=False)
            with self.assertRaisesRegex(
                UnsupportedAliasMutationException,
                "One of torch.cond branch might be aliasing",
            ):
                f(example_input_func)
        finally:
            torch._disable_functionalization()

        def f_wrapper(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                torch._enable_functionalization(reapply_views=False)
                try:
                    func_args = pytree.tree_map(
                        lambda x: torch._to_functional_tensor(x)
                        if isinstance(x, torch.Tensor)
                        else x,
                        args,
                    )
                    func_kwargs = pytree.tree_map(
                        lambda x: torch._to_functional_tensor(x)
                        if isinstance(x, torch.Tensor)
                        else x,
                        kwargs,
                    )
                    return func(*func_args, **func_kwargs)
                finally:
                    torch._disable_functionalization()

            return wrapper

        with self.assertRaisesRegex(
            UnsupportedAliasMutationException,
            "One of torch.cond branch might be aliasing",
        ):
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

        def f_wrapper(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                torch._enable_functionalization(reapply_views=False)
                try:
                    func_args = pytree.tree_map(
                        lambda x: to_fun_old(x) if isinstance(x, torch.Tensor) else x,
                        args,
                    )
                    func_kwargs = pytree.tree_map(
                        lambda x: to_fun_old(x) if isinstance(x, torch.Tensor) else x,
                        kwargs,
                    )
                    return pytree.tree_map(
                        from_fun_old, func(*func_args, **func_kwargs)
                    )
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
            return torch.add(torch.tensor([0.25, 0.25]), z)

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

        self.assertExpectedInline(
            graph.code.strip(),
            """\
def forward(self, x_1, pred_1, pred2_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    conditional = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, [x_1]);  pred_1 = true_graph_0 = false_graph_0 = None
    getitem = conditional[0];  conditional = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    conditional_1 = torch.ops.higher_order.cond(pred2_1, true_graph_1, false_graph_1, [x_1]);  pred2_1 = true_graph_1 = false_graph_1 = x_1 = None
    getitem_1 = conditional_1[0];  conditional_1 = None
    add = torch.ops.aten.add.Tensor(getitem, getitem_1);  getitem = getitem_1 = None
    return add""",  # noqa: B950
        )
        self.assertExpectedInline(
            graph.true_graph_0.code.strip(),
            """\
def forward(self, arg0_1):
    mul = torch.ops.aten.mul.Tensor(arg0_1, arg0_1);  arg0_1 = None
    return (mul,)""",
        )

    def test_raise_error_on_mismatch_type_size(self):
        def true_fn(x):
            return x.sin()

        def false_fn(x):
            return (x, x)

        def f(x, y):
            return cond(y, true_fn, false_fn, [x])

        x = torch.randn(4)
        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "Cond doesn't work unless it is captured completely with torch.compile",
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
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "Cond doesn't work unless it is captured completely with torch.compile",
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
        graph = make_fx(f, tracing_mode="fake")(
            x, torch.tensor(False), torch.tensor(False)
        )

        result_true_true = graph.forward(
            x, torch.tensor(True), torch.tensor(True)
        )  # True + True -> x * x
        result_true_false = graph.forward(
            x, torch.tensor(True), torch.tensor(False)
        )  # True + True -> x + x
        result_false_true = graph.forward(
            x, torch.tensor(False), torch.tensor(True)
        )  # False + either -> cos
        result_false_false = graph.forward(
            x, torch.tensor(False), torch.tensor(False)
        )  # False + either -> cos

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
            return torch.add(torch.tensor([0.25, 0.25]), z)

        def false_fn(k, _):
            return k.cos()

        def f(k, pred, pred2):
            return cond(pred, true_fn, false_fn, [k, pred2])

        x = torch.tensor([0.5, 0.5])
        graph = make_fx(f, tracing_mode="fake")(
            x, torch.tensor(False), torch.tensor(False)
        )

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
        graph = make_fx(f, tracing_mode="fake")(
            x, torch.tensor(False), torch.tensor(False)
        )

        self.assertExpectedInline(
            graph.code.strip(),
            """\
def forward(self, x_1, pred_1, pred2_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    conditional = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, [x_1]);  pred_1 = true_graph_0 = false_graph_0 = None
    getitem = conditional[0];  conditional = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    conditional_1 = torch.ops.higher_order.cond(pred2_1, true_graph_1, false_graph_1, [x_1]);  pred2_1 = true_graph_1 = false_graph_1 = x_1 = None
    getitem_1 = conditional_1[0];  conditional_1 = None
    add = torch.ops.aten.add.Tensor(getitem, getitem_1);  getitem = getitem_1 = None
    return add""",  # noqa: B950
        )
        self.assertExpectedInline(
            graph.true_graph_0.code.strip(),
            """\
def forward(self, arg0_1):
    mul = torch.ops.aten.mul.Tensor(arg0_1, arg0_1);  arg0_1 = None
    return (mul,)""",
        )

    def test_raise_error_on_mismatch_type_size_fake_tensor(self):
        def true_fn(x):
            return x.sin()

        def false_fn(x):
            return (x, x)

        def f(x, y):
            return cond(y, true_fn, false_fn, [x])

        x = torch.randn(4)
        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "Cond doesn't work unless it is captured completely with torch.compile",
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
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "Cond doesn't work unless it is captured completely with torch.compile",
        ):
            make_fx(f, tracing_mode="fake")(x, torch.tensor(False))

    def check_map_count(self, gm, op_count):
        i = 0
        for m in gm.modules():
            for node in m.graph.nodes:
                if (
                    node.op == "call_function"
                    and node.target == torch.ops.higher_order.map_impl
                ):
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
        gm = make_fx(g, tracing_mode="symbolic")(
            example_x, torch.ones(5), torch.ones(5)
        )
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
        gm = make_fx(g, tracing_mode="symbolic")(
            example_x, torch.ones(5), torch.ones(5)
        )
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

        gm = make_fx(g, tracing_mode="symbolic")(
            torch.ones(3, 4, 5, requires_grad=True), torch.ones(5, requires_grad=True)
        )
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
            flat_out = pytree.tree_leaves(out)
            flat_inp = pytree.tree_leaves((xs, y))
            requires_grad_inp = [inp for inp in flat_inp if inp.requires_grad]
            return torch.autograd.grad(
                flat_out, requires_grad_inp, [torch.ones_like(out) for out in flat_out]
            )

        gm = make_fx(g, tracing_mode="symbolic")(
            [torch.ones(3, 4, 5), torch.ones(3, 4, 5, requires_grad=True)],
            torch.ones(5, requires_grad=True),
        )
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
            flat_out = pytree.tree_leaves(out)
            flat_inp = pytree.tree_leaves((xs, y))
            requires_grad_inp = [inp for inp in flat_inp if inp.requires_grad]
            return torch.autograd.grad(
                flat_out, requires_grad_inp, [torch.ones_like(out) for out in flat_out]
            )

        traced_x = {
            "a": torch.ones(3, 4, 5, requires_grad=True),
            "b": torch.ones(3, 4, 5, requires_grad=True),
        }
        gm = make_fx(g, tracing_mode="symbolic")(
            traced_x, torch.ones(5, requires_grad=True)
        )
        x = {
            "a": torch.randn(4, 5, 6, requires_grad=True),
            "b": torch.ones(4, 5, 6, requires_grad=True),
        }
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
                    return pytree.tree_map(from_fun_old, func(*args, **kwargs))
                finally:
                    torch._disable_functionalization()

            return wrapper

        example_inputs = (
            torch.ones(3, 2, 4, requires_grad=True),
            torch.ones(2, 4, requires_grad=True),
        )
        gm = make_fx(f, tracing_mode="symbolic")(*example_inputs)
        fgm = make_fx(f_wrapper(f), tracing_mode="symbolic")(*example_inputs)
        xs = torch.ones(3, 4, 5, requires_grad=True)
        y = torch.ones(4, 5, requires_grad=True)

        self.assertEqual(gm(xs, y), f(xs, y))

        def count_mutable(gm):
            c = 0
            for node in gm.graph.nodes:
                if node.op == "call_function":
                    if node.target == torch.ops.higher_order.map_impl:
                        c += count_mutable(getattr(gm, str(node.args[0])))
                    elif schema := getattr(node.target, "_schema", None):
                        c += int(schema.is_mutable)
            return c

        self.assertEqual(count_mutable(fgm), 0)
        # One for forward, one for recomputation logic in backward
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

        gm = make_fx(torch.func.functionalize(f), tracing_mode="symbolic")(
            *example_inputs
        )
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
                    return pytree.tree_map(from_fun_old, func(*args, **kwargs))
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
        with self.assertRaisesRegex(
            UnsupportedAliasMutationException, "torch.map is mutating the input!"
        ):
            functional_f(*example_inputs)

    def test_map_functionalized_elem_mutation(self):
        def map_fn(x, y):
            x.add_(4)
            return x + y

        def f(xs, y):
            return control_flow.map(map_fn, xs, y)

        example_inputs = (torch.ones(3, 2, 4), torch.ones(4))
        functional_f = torch.func.functionalize(f)
        with self.assertRaisesRegex(
            UnsupportedAliasMutationException, "torch.map is mutating the input!"
        ):
            functional_f(*example_inputs)

    def test_cond_autograd_fail(self):
        def true_fn(x):
            return x.cos()

        def false_fn(x):
            return x.sin()

        def f(x, y):
            return control_flow.cond(x.shape[0] > 4, true_fn, false_fn, [y])

        example_inputs = (
            torch.ones(3, 2, 4, requires_grad=True),
            torch.ones(4, requires_grad=True),
        )
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
        with self.assertRaisesRegex(
            UnsupportedAliasMutationException, "torch.map is aliasing the input!"
        ):
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
        # The symbols in make_fx's shape_env should not be specialized.
        self.assertEqual(len(gm.shape_env.guards), 0)

        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x_1):
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0)
    eq = sym_size_int == 4;  sym_size_int = None
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    conditional = torch.ops.higher_order.cond(eq, true_graph_0, false_graph_0, [x_1]);  eq = true_graph_0 = false_graph_0 = x_1 = None
    getitem = conditional[0];  conditional = None
    return getitem""",  # noqa: B950
        )

        # We expect the traced graph module to work even if input size changes.
        x = torch.ones(4, 3, 2)
        self.assertEqual(gm(x), true_fn(x))
        self.assertEqual(foo(x), true_fn(x))

    def _check_closure_correctly_lifted(self, f, *, args, exp_res, exp_arg_num):
        assert isinstance(args, (tuple, list))
        self.assertEqual(f(*args), exp_res)
        gm = make_fx(f)(*args)
        self.assertEqual(gm(*args), exp_res)

        def cnt_placeholder(gm):
            return len([node for node in gm.graph.nodes if node.op == "placeholder"])

        placeholder_cnts = [cnt_placeholder(mod) for mod in gm.children()]
        self.assertTrue(all(cnt == exp_arg_num for cnt in placeholder_cnts))

    def _check_closure_correctly_lifted_with_mutation(
        self, f, closures_to_be_mutated, *, args, exp_arg_num
    ):
        exp_res = f(*args)
        self._check_closure_correctly_lifted(
            f, args=args, exp_res=exp_res, exp_arg_num=exp_arg_num
        )

        for closure in closures_to_be_mutated:
            closure.add(-1)
        new_exp_res = f(*args)

        self._check_closure_correctly_lifted(
            f, args=args, exp_res=new_exp_res, exp_arg_num=exp_arg_num
        )

    def test_cond_with_tensor_closure(self):
        a = torch.ones(2, 3)
        b = torch.ones(2, 3) + 1

        def true_fn(x):
            return x + a

        def false_fn(x):
            return x + b

        def foo(x):
            return cond(x.shape[0] == 4, true_fn, false_fn, [x])

        # expected branches takes [x, a, b] as input
        inp = torch.randn(2, 3)
        self._check_closure_correctly_lifted_with_mutation(
            foo, (a, b), args=(inp,), exp_arg_num=3
        )

    def test_cond_with_tensor_closure_graph_module(self):
        a = torch.ones(2, 3)
        b = torch.ones(2, 3) + 1

        def true_fn(x):
            return x + a

        def false_fn(x):
            return x + b

        def foo(x):
            return cond(x.shape[0] == 4, true_fn, false_fn, [x])

        # expected branches takes [x, a, b] as input
        inp = torch.randn(2, 3)

        gm = make_fx(foo)(inp)

        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    _tensor_constant0 = self._tensor_constant0
    _tensor_constant1 = self._tensor_constant1
    conditional = torch.ops.higher_order.cond(False, true_graph_0, false_graph_0, [x_1, _tensor_constant0, _tensor_constant1]);  true_graph_0 = false_graph_0 = x_1 = _tensor_constant0 = _tensor_constant1 = None
    getitem = conditional[0];  conditional = None
    return getitem""",  # noqa: B950
        )
        self.assertExpectedInline(
            gm.true_graph_0.code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1):
    add = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
    return (add,)""",
        )

    def test_cond_with_module_param_closure(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter(
                    "param", torch.nn.Parameter(torch.ones(2, 3), requires_grad=False)
                )
                self.register_buffer("buffer", torch.ones(2, 3) + 1)

        my_mode = Mod()

        def true_fn(x):
            return x + my_mode.param

        def false_fn(x):
            return x + my_mode.buffer

        def foo(x):
            return cond(x.shape[0] == 4, true_fn, false_fn, [x])

        inp = torch.ones(2, 3)
        # expected both branches takes (x, param, buffer)
        self._check_closure_correctly_lifted_with_mutation(
            foo, (my_mode.param, my_mode.buffer), args=(inp,), exp_arg_num=3
        )

    def test_cond_with_module_python_scalar_closure(self):
        def foo(x):
            a = torch.ones(1, 1)
            b = 1

            def true_fn(x):
                return x + a

            def false_fn(x):
                return x + b

            return cond(x.shape[0] == 4, true_fn, false_fn, [x])

        inp = torch.ones(2, 3)
        res = inp + 1
        # python scalar b is not lifted as input, so both branches take (x, a)
        self._check_closure_correctly_lifted(
            foo, args=(inp,), exp_res=res, exp_arg_num=2
        )

    def test_cond_nested_with_closure(self):
        a = torch.ones(1, 1)
        b = torch.ones(1, 1) + 1

        def inner_true_fn(x):
            return x + a

        def inner_false_fn(x):
            return x + b

        def foo(x):
            def true_fn(x):
                return cond(x.shape[0] == 2, inner_true_fn, inner_false_fn, [x])

            def false_fn(x):
                return cond(x.shape[0] > 4, inner_true_fn, inner_false_fn, [x])

            return cond(x.shape[0] == 4, true_fn, false_fn, [x])

        inp = torch.ones(2, 3)
        # For top-level cond, it take 3 arguments (x, a, b). Dynamo should
        # realize that the nonlocal variables are same for the true and false
        # branches, so it should de-dupe them.
        # For second-level conds, it takes (x, a, b)
        self._check_closure_correctly_lifted_with_mutation(
            foo, (a, b), args=(inp,), exp_arg_num=3
        )

    def test_cond_nested_with_closure_graph_module(self):
        a = torch.ones(1, 1)
        b = torch.ones(1, 1) + 1

        def inner_true_fn(x):
            return x + a

        def inner_false_fn(x):
            return x + b

        def foo(x):
            def true_fn(x):
                return cond(x.shape[0] == 2, inner_true_fn, inner_false_fn, [x])

            def false_fn(x):
                return cond(x.shape[0] > 4, inner_true_fn, inner_false_fn, [x])

            return cond(x.shape[0] == 4, true_fn, false_fn, [x])

    def test_map_unfunc_boolean_tensor_for_nested_map_cond(self):
        def map_fn(pred, x):
            def fn(x, pred):
                return control_flow.cond(pred, lambda x: x * 2, lambda x: x / 2, (x,))

            return control_flow.map(fn, x, pred)

        def f_wrapper(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                torch._enable_functionalization(reapply_views=False)
                try:
                    func_args = pytree.tree_map(
                        lambda x: to_fun_old(x) if isinstance(x, torch.Tensor) else x,
                        args,
                    )
                    func_kwargs = pytree.tree_map(
                        lambda x: to_fun_old(x) if isinstance(x, torch.Tensor) else x,
                        kwargs,
                    )
                    return pytree.tree_map(
                        from_fun_old, func(*func_args, **func_kwargs)
                    )
                finally:
                    torch._disable_functionalization()

            return wrapper

        gm = make_fx(f_wrapper(map_fn))(
            torch.tensor(True), torch.ones([2, 3], requires_grad=False)
        )
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, pred_1, x_1):
    body_graph_0 = self.body_graph_0
    map_impl = torch.ops.higher_order.map_impl(body_graph_0, [x_1], [pred_1]);  body_graph_0 = x_1 = pred_1 = None
    getitem = map_impl[0];  map_impl = None
    return getitem""",
        )
        self.assertExpectedInline(
            gm.body_graph_0.code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    conditional = torch.ops.higher_order.cond(arg1_1, true_graph_0, false_graph_0, [arg0_1]);  arg1_1 = true_graph_0 = false_graph_0 = arg0_1 = None
    getitem = conditional[0];  conditional = None
    return [getitem]""",  # noqa: B950
        )

    def test_cond_make_fx_preserve_stack_trace_for_nodes_in_subgraph(self):
        def true_fn(x):
            return x + x.cos()

        def false_fn(x):
            return x * x.sin()

        def foo(x):
            return cond(x.shape[0] == 4, true_fn, false_fn, (x,))

        inp = torch.randn([4, 3])
        gm, _ = torch._dynamo.export(foo)(inp)

        def run_with_interpreter(*args):
            with torch.fx.traceback.preserve_node_meta():
                return torch.fx.Interpreter(gm).run(*args)

        new_gm = make_fx(run_with_interpreter)(inp)

        checked_ops = {"add", "mul", "sin", "cos"}
        checked_meta = ["source_fn_stack", "stack_trace"]
        all_source_fns = collect_meta_for_filtered_nodes(gm, checked_ops, checked_meta)
        new_source_fns = collect_meta_for_filtered_nodes(
            new_gm, checked_ops, checked_meta
        )
        self.assertEqual(all_source_fns, new_source_fns)

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "triggers cache limit for foo and changes unique_graphs count.",
    )
    def test_cond_no_dynamo_cache_limit(self):
        torch._dynamo.reset()
        counters = torch._dynamo.utils.counters
        counters.clear()

        def foo(x, true_fn, false_fn):
            return cond(x.shape[0] == 4, true_fn, false_fn, (x,))

        inp = torch.ones(3, 4)
        exp_out = inp.sin()
        iter_n = torch._dynamo.config.cache_size_limit + 1

        # Need this because Dynamo checks lambda code ID not object itself.
        def make_dummy_fn(op):
            exec(f"temp = lambda x: x.{op}()")
            return locals()["temp"]

        for _ in range(iter_n):
            # each lambda has a different object id thus fails the guard
            self.assertEqual(
                foo(inp, make_dummy_fn("cos"), make_dummy_fn("sin")), exp_out
            )

        # each iteration captures a cond and a getitem from the tuple output
        self.assertEqual(counters["stats"]["calls_captured"], iter_n * 2)
        self.assertEqual(counters["stats"]["unique_graphs"], iter_n)

    def test_cond_with_consecutive_make_fx_symbolic(self):
        def true_fn(x):
            return x - x.cos()

        def false_fn(x):
            return x + x.sin()

        def foo(x):
            return cond(x.shape[0] == 4, true_fn, false_fn, [x])

        inps = (torch.ones(3, 4), torch.ones(3, 5), torch.ones(5, 4), torch.ones(5, 3))
        for inp in inps:
            gm = make_fx(foo, tracing_mode="symbolic")(torch.ones(3, 4))
            self.assertExpectedInline(
                gm.code.strip(),
                """\
def forward(self, x_1):
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0)
    eq = sym_size_int == 4;  sym_size_int = None
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    conditional = torch.ops.higher_order.cond(eq, true_graph_0, false_graph_0, [x_1]);  eq = true_graph_0 = false_graph_0 = x_1 = None
    getitem = conditional[0];  conditional = None
    return getitem""",  # noqa: B950
            )

            self.assertExpectedInline(
                gm.true_graph_0.code.strip(),
                """\
def forward(self, arg0_1):
    cos = torch.ops.aten.cos.default(arg0_1)
    sub = torch.ops.aten.sub.Tensor(arg0_1, cos);  arg0_1 = cos = None
    return (sub,)""",
            )

            self.assertExpectedInline(
                gm.false_graph_0.code.strip(),
                """\
def forward(self, arg0_1):
    sin = torch.ops.aten.sin.default(arg0_1)
    add = torch.ops.aten.add.Tensor(arg0_1, sin);  arg0_1 = sin = None
    return (add,)""",
            )

    def _create_test_fns_for_cond(
        self, pred, inner_most_fn, operands, closure_list, nested_level
    ):
        if nested_level == 0:
            if len(closure_list) > 0:

                def true_fn(*operands):
                    return inner_most_fn(*operands) + inner_most_fn(*closure_list)

                def false_fn(*operands):
                    return inner_most_fn(*operands) - inner_most_fn(*closure_list)

            else:

                def true_fn(*operands):
                    return inner_most_fn(*operands)

                def false_fn(*operands):
                    return inner_most_fn(*operands)

            def fn(*operands):
                if len(operands) == 0 and len(closure_list) == 0:
                    return torch.zeros(1)
                return cond(pred, true_fn, false_fn, operands)

            return operands, fn
        else:
            args, inner_fn = self._create_test_fns_for_cond(
                pred <= 0, inner_most_fn, operands, closure_list, nested_level - 1
            )

            def true_fn(*operands):
                return inner_most_fn(*operands) + inner_fn(*args)

            def false_fn(*operands):
                return inner_most_fn(*operands) - inner_fn(*args)

            def fn(*operands):
                if len(operands) == 0 and len(closure_list) == 0:
                    return torch.ones(1)
                return cond(pred, true_fn, false_fn, operands)

            return operands, fn

    def _init_predicate(self, pred_type):
        if pred_type == "bool":
            return True
        elif pred_type == "intTensor":
            return torch.tensor(1)
        elif pred_type == "floatTensor":
            return torch.tensor(1.0)
        elif pred_type == "boolTensor":
            return torch.tensor(False)
        else:
            raise NotImplementedError

    def _init_fn(self, inner_fn_type):
        if inner_fn_type == "function":
            return reduce_func
        elif inner_fn_type == "module":
            return ReduceMod()
        elif inner_fn_type == "object":
            return ReduceObj()
        else:
            raise NotImplementedError

    @parametrize("predType", ["bool", "intTensor", "floatTensor", "boolTensor"])
    @parametrize("innerFnType", ["function", "module", "object"])
    @parametrize("nOperands", [0, 1])
    @parametrize("nClosure", [0, 1])
    @parametrize("nesting", [0, 2])
    def test_cond_tracing_with_valid_inputs(
        self, predType, innerFnType, nOperands, nClosure, nesting
    ):
        pred = self._init_predicate(predType)
        inner_fn = self._init_fn(innerFnType)
        operands = [torch.ones(2, 3) + i for i in range(nOperands)]
        closure = [torch.ones(2, 3) - i for i in range(nClosure)]
        args, fn = self._create_test_fns_for_cond(
            pred, inner_fn, operands, closure, nesting
        )
        eager_res = fn(*args)
        for tracing_mode in ["symbolic", "fake", "real"]:
            # set _allow_non_fake_inputs = True to allow fake prop through closures
            with self.subTest(tracing_mode=tracing_mode):
                gm = make_fx(
                    fn, tracing_mode=tracing_mode, _allow_non_fake_inputs=True
                )(*args)
                self.assertEqual(gm(*args), eager_res)

    @parametrize("predType", ["boolTensor"])
    @parametrize("innerFnType", ["function", "module", "object"])
    @parametrize("nOperands", [1, 2])
    @parametrize("nClosure", [0, 1])
    @parametrize("nesting", [0])
    def test_cond_vmap(self, predType, innerFnType, nOperands, nClosure, nesting):
        pred = self._init_predicate(predType)
        inner_fn = self._init_fn(innerFnType)
        operands = [torch.ones(2, 3) + i for i in range(nOperands)]
        closure = [torch.ones(2, 3) - i for i in range(nClosure)]
        args, fn = self._create_test_fns_for_cond(
            pred, inner_fn, operands, closure, nesting
        )
        eager_res = fn(*args)
        out = torch.vmap(fn)(*args)
        if nClosure == 0:
            self.assertEqual(eager_res, out)
        else:
            self.assertEqual(eager_res, out[0])
            self.assertEqual(eager_res, out[1])

    def test_cond_vmap_simple(self):
        def fn(x):
            return torch.cond(
                pred=torch.tensor([True]),
                true_fn=lambda x: x + 100,
                false_fn=lambda x: x,
                operands=(x,),
            )

        a = torch.arange(15).reshape((3, 5))
        res = torch.vmap(fn, in_dims=(0,))(a)
        self.assertEqual(res.shape, (3, 5))
        self.assertEqual(res, a + 100)

    def test_cond_vmap_multiple_inputs(self):
        def fn(x, y):
            return torch.cond(
                pred=x.sum() < y.sum(),
                true_fn=lambda x, y: x + 100,
                false_fn=lambda x, y: y,
                operands=(x, y),
            )

        a = torch.arange(15).reshape(3, 5)
        b = torch.ones_like(a) + 3
        res = torch.vmap(fn, in_dims=(0, 0))(a, b)
        expected = torch.tensor(
            [[100, 101, 102, 103, 104], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4]]
        )
        self.assertEqual(res.shape, (3, 5))
        self.assertEqual(expected, res)

    def test_cond_vmap_single_input_with_closure(self):
        a = torch.ones((3, 5)) + 3
        c = torch.arange(5)

        def fn(x):
            return torch.cond(
                pred=torch.tensor([True]),
                true_fn=lambda x: x + c,
                false_fn=lambda x: x - c,
                operands=(x,),
            )

        res = torch.vmap(fn, in_dims=(0,))(
            a,
        )
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            res = torch.vmap(fn, in_dims=(0,))(
                a,
            )
        self.assertEqual(a + c, res)

    def test_cond_vmap_multiple_args_with_closure(self):
        a = torch.ones((3, 5), dtype=torch.int64) + 3
        b = torch.arange(15).reshape(3, 5)
        c = torch.arange(5)

        def fn(x, y):
            return torch.cond(
                pred=torch.tensor([False]),
                true_fn=lambda x, y: x + c,
                false_fn=lambda x, y: y - c,
                operands=(x, y),
            )

        res = torch.vmap(fn)(a, b)
        self.assertEqual(b - c, res)

    @parametrize("nClosure", [0, 1])
    def test_cond_vmap_multiple_outputs(self, nClosure):
        if nClosure:
            c = torch.ones(5, dtype=torch.int64) + 5

            def fn(x):
                return torch.cond(
                    pred=torch.tensor([True]),
                    true_fn=lambda x: (x + c, x - c),
                    false_fn=lambda x: (x, x),
                    operands=(x,),
                )

        else:

            def fn(x):
                return torch.cond(
                    pred=torch.tensor([True]),
                    true_fn=lambda x: (x + 1, x - 1),
                    false_fn=lambda x: (x, x),
                    operands=(x,),
                )

        a = torch.arange(15).reshape(3, 5)
        res = torch.vmap(fn)(
            a,
        )
        self.assertEqual(len(res), 2)
        if nClosure:
            self.assertEqual(res, (a + c, a - c))
        else:
            self.assertEqual(res, (a + 1, a - 1))

    def test_vmap_vmap(self):
        def fn(x):
            return torch.cond(
                pred=torch.tensor([True]),
                true_fn=lambda x: x + 1,
                false_fn=lambda x: x - 1,
                operands=(x,),
            )

        def wrapper(x):
            return torch.vmap(fn)(x)

        a = torch.ones((3, 4, 5))
        res = torch.vmap(wrapper)(a)
        self.assertEqual(res, a + 1)


instantiate_parametrized_tests(TestControlFlowTraced)

if __name__ == "__main__":
    run_tests()
