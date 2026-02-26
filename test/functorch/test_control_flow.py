# Owner(s): ["module: functorch"]
import contextlib
import functools
import unittest

import torch
import torch.utils._pytree as pytree
from functorch.experimental import control_flow
from functorch.experimental.control_flow import cond
from torch._dynamo.testing import (
    AotEagerAndRecordGraphs,
    EagerAndRecordGraphs,
    normalize_gm,
)
from torch._higher_order_ops.associative_scan import (
    _fake_associative_scan,
    associative_scan,
)
from torch._higher_order_ops.cudagraph_conditional_nodes import (
    ControlFlowOpWarmupDispatchMode,
    CUDAGraphCaptureControlFlowOpDispatchMode,
)
from torch._higher_order_ops.map import _fake_map
from torch._higher_order_ops.scan import _fake_scan, scan
from torch._higher_order_ops.schema import HopSchemaGenerator
from torch._higher_order_ops.while_loop import while_loop
from torch._subclasses.functional_tensor import (
    CppFunctionalizeAPI,
    FunctionalTensor,
    FunctionalTensorMode,
    PythonFunctionalizeAPI,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_cuda import SM70OrLater
from torch.testing._internal.common_quantization import skipIfNoDynamoSupport
from torch.testing._internal.common_utils import (
    decorateIf,
    instantiate_parametrized_tests,
    IS_WINDOWS,
    parametrize,
    requires_cuda,
    run_tests,
    skipIfCrossRef,
    skipIfTorchDynamo,
    TEST_CUDA_GRAPH_CONDITIONAL_NODES,
    TEST_WITH_CROSSREF,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
)


@contextlib.contextmanager
def check_cudagraphs_not_skipped(test_case):
    counters = torch._dynamo.utils.counters
    old_cudagraph_skips = counters["inductor"]["cudagraph_skips"]
    try:
        yield
    finally:
        # reset before the assert, because otherwise the reset is
        # skipped in case of assertion error
        new_cudagraph_skips = counters["inductor"]["cudagraph_skips"]
        counters["inductor"]["cudagraph_skips"] = old_cudagraph_skips
        test_case.assertEqual(
            counters["inductor"]["cudagraph_skips"], new_cudagraph_skips
        )


def _check_compile_cudagraph_backend(test_case, fn, args):
    # test cudagraphs backend
    cudagraphs_compiled_fn = torch.compile(fn, backend="cudagraphs")
    # We run 3 times.
    # This is what cuda graph trees does for the first 3 runs:
    # 1) run in eager mode, for warmup.
    # 2) do stream capture followed by graph replay.
    # 3 and beyond) do graph replay
    # So we need to get to iteration 3 to test all ways of running.
    outputs = []
    for i in range(3):
        with check_cudagraphs_not_skipped(test_case):
            torch.compiler.cudagraph_mark_step_begin()
            outputs.append(
                pytree.tree_map(
                    lambda x: x.detach().clone() if isinstance(x, torch.Tensor) else x,
                    cudagraphs_compiled_fn(*args),
                )
            )
    eager_res = fn(*args)
    for output in outputs:
        test_case.assertEqual(eager_res, output)


def _check_compile_many_backends_with_cudagraph(test_case, fn, args):
    for backend in ["eager_no_compile", "eager", "aot_eager"]:
        _check_compile_any_backend_with_cudagraph(test_case, fn, args, backend)


def _check_compile_any_backend_with_cudagraph(test_case, fn, args, backend):
    if backend == "eager_no_compile":
        compiled_fn = fn
    else:
        compiled_fn = torch.compile(fn, backend=backend)
    outputs = []

    side_stream = torch.cuda.Stream()

    with torch.cuda.stream(side_stream), ControlFlowOpWarmupDispatchMode():
        warmup_output = compiled_fn(*args)
    outputs.append(warmup_output)

    graph = torch.cuda.CUDAGraph()
    with (
        torch.cuda.graph(graph, stream=side_stream),
        CUDAGraphCaptureControlFlowOpDispatchMode(),
    ):
        captured_output = compiled_fn(*args)
    with torch.cuda.stream(side_stream):
        eager_res = fn(*args)
    graph.replay()
    outputs.append(
        pytree.tree_map(
            lambda x: x.clone() if isinstance(x, torch.Tensor) else x,
            captured_output,
        )
    )
    for output in outputs:
        test_case.assertEqual(eager_res, output)


# TODO: pull these helpers from AOTAutograd later
def to_fun(t):
    if isinstance(t, torch.Tensor):
        return FunctionalTensor.to_functional(t)
    return t


def from_fun(t):
    if not isinstance(t, FunctionalTensor):
        # quick sanity assert
        if isinstance(t, torch.Tensor):
            if torch._is_functional_tensor(t):
                raise AssertionError("Expected tensor to not be a functional tensor")
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
        if not torch._is_functional_tensor(t):
            raise AssertionError("Expected tensor to be a functional tensor")
        torch._sync(t)
        return torch._from_functional_tensor(t)
    return t


def _fake_while_loop(cond_fn, body_fn, operands):
    while cond_fn(*operands):
        operands = body_fn(*operands)
    return operands


def compile_mode_helper(fct, compile_mode):
    if compile_mode == "compile":
        return torch.compile(fct, fullgraph=True, dynamic=False)
    elif compile_mode == "compile_dynamic_shape":
        return torch.compile(fct, fullgraph=True, dynamic=True)
    elif compile_mode == "eager":
        return torch.compile(fct, fullgraph=True, backend="eager")
    else:
        return fct


ALIAS_FN = [
    lambda x: x,
    lambda x: x.view(-1),
    lambda x: x.reshape(-1),
    lambda x: x.squeeze(0),
    lambda x: x.unsqueeze(0),
    lambda x: x.transpose(0, 1),
    lambda x: x.flatten(),
    lambda x: x.expand(1, *x.size()),
]


def get_scan_combine_fn(name, associative=True, parameters=None):
    def add(x: torch.Tensor, y: torch.Tensor):
        return x + y

    def adds(x: torch.Tensor, y: torch.Tensor):
        return x + x, y + y

    def mul(x: torch.Tensor, y: torch.Tensor):
        return x * y

    def div(x: torch.Tensor, y: torch.Tensor):
        return x / y

    def s5_operator(x: torch.Tensor, y: torch.Tensor):
        A_i, Bu_i = x
        A_j, Bu_j = y
        return A_j * A_i, A_j * Bu_i + Bu_j

    def different_input_size_operator(x: torch.Tensor, y: torch.Tensor):
        x_o, dA_o, dB_o, C_o, y_o = x
        x_n, dA_n, dB_n, C_n, y_n = y

        x_new = x_n + x_o
        y_new = torch.einsum("bdn,bn->bd", x_new, C_n)

        return x_new, dA_n + 0.0, dB_n + 0.0, C_n + 0.0, y_new

    def tuple_fct(x, y):
        return (x[0] + y[0], x[1] * y[1])

    def complex_pointwise(x, y):
        return {
            "i": x["i"] * y["i"],
            "j": (
                [x["j"][0][0] * y["j"][0][0]],
                [{"o": x["j"][1][0]["o"] + y["j"][1][0]["o"]}],
            ),
        }

    def non_pointwise(x: torch.Tensor, y: torch.Tensor):
        W = torch.arange(4, dtype=torch.float, device=x.device).view(2, 2)
        return x @ W + y @ W

    def RNN(x: torch.Tensor, y: torch.Tensor):
        c_new = y @ parameters[0] + parameters[1]
        h_new = torch.tanh(c_new + x @ parameters[2] + parameters[3])
        return h_new, h_new.clone()

    def fct_c1_no_grad(x: torch.Tensor, y: torch.Tensor):
        h_new = torch.tanh(x[0] + x[1] + y)
        c2 = x[1] + y
        with torch.no_grad():
            c1 = x[0] + y
        return (c1, c2), h_new

    if name == "add":
        fct = add
    elif name == "adds":
        fct = adds
    elif name == "mul":
        fct = mul
    elif name == "div":
        fct = div
    elif name == "s5_operator":
        fct = s5_operator
    elif name == "different_input_size_operator":
        fct = different_input_size_operator
    elif name == "tuple_fct":
        fct = tuple_fct
    elif name == "complex_pointwise":
        fct = complex_pointwise
    elif name == "non_pointwise":
        fct = non_pointwise
    elif name == "RNN":
        fct = RNN
    elif name == "fct_c1_no_grad":
        fct = fct_c1_no_grad
    else:
        raise ValueError("Combine_fn name unknown!")

    if not associative:
        return lambda x, y: (fct(x, y), fct(x, y))
    else:
        return fct


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
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
            self.dec = torch.nn.Buffer(torch.tensor(1))

        def forward(self, iter, x):
            def cond_fn(it, x):
                return it - self.dec > 0

            def body_fn(it, x):
                return it - 1, self.linear(x)

            return while_loop(cond_fn, body_fn, (iter, x))

    class SimpleWithPytreeCarry(torch.nn.Module):
        def forward(self, it, pytree_input):
            def cond_fn(it, pytree_input):
                return it > 0

            def body_fn(it, pytree_input):
                x = pytree_input[0][0]
                y = pytree_input[1]["x"]
                z = pytree_input[1]["y"]
                new_x = y.sin()
                new_y = z.cos()
                new_z = x + 1
                return it - 1, ([new_x], {"x": new_y, "y": new_z})

            return while_loop(cond_fn, body_fn, (it, pytree_input))

    class NestedWithLinear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mod = SimpleWithLinear()
            self.outer_linear = torch.nn.Linear(2, 2)
            self.dec = torch.nn.Buffer(torch.tensor(1))

        def forward(self, iter, x):
            def cond_fn(it, x):
                return it - self.dec > 0

            def body_fn(it, x):
                return it - 1, self.outer_linear(self.mod(it, x)[1])

            return while_loop(cond_fn, body_fn, (iter, x))

    class PytreeIntCarry(torch.nn.Module):
        def forward(self, x):
            a = x.shape[0]
            b = x.shape[1]

            def cond_fn(shapes, const_int_dict, x):
                a, b = shapes
                c1, c2, c3 = const_int_dict["int_carry"]
                return c1 * c2 * c3 < a * b

            def body_fn(shapes, const_int_dict, x):
                a, b = shapes
                c1, c2, c3 = const_int_dict["int_carry"]
                return (
                    [a + 1, b + 1],
                    {"int_carry": (c1 + 1, c2 + 1, c3 + 1)},
                    x + 1,
                )

            carry = ([a, b], {"int_carry": (2, 2, 3)}, x.sin())
            out_shapes, out_it, out_x = while_loop(cond_fn, body_fn, carry)
            out_inc = pytree.tree_map(lambda x: x + 1, out_it)
            out_add = pytree.tree_map(lambda x: x + out_x, out_it)
            return (out_shapes, out_inc, out_add, out_x)

    class IntCarry(torch.nn.Module):
        def forward(self, x):
            def cond_fn(it, x):
                return it < x.shape[0]

            def body_fn(it, x):
                x_clone = x.clone()
                # Need these checks to select from x
                torch._check(it >= 0)
                torch._check(it < x.shape[0])
                x_clone.select(0, it).copy_(x_clone.select(0, it) + it)
                return it + 1, x_clone

            # We invoke the hop directly to avoid triggering dyanmo tracing
            out_it, out_x = torch.ops.higher_order.while_loop(
                cond_fn, body_fn, (0, x), tuple()
            )
            # We need torch._check to use it in torch.ones call
            torch._check(out_it > 0)
            return (
                out_it + 1,
                out_it + out_x,
                out_it < x.shape[0],
                torch.ones(out_it * 2),
            )

    class ConstAndSymIntOutput(torch.nn.Module):
        def forward(self, t):
            a = t.shape[0]
            b = t.shape[1]

            def cond_fn(a, b, c1, c2, c3, c0, u0, x):
                return c1 * c2 * c3 < a * b

            def body_fn(a, b, c1, c2, c3, c0, u0, x):
                return b, c1, c2, c3, a, 0, u0 + 1, x + 1

            carry = (a, b, 1, 1, 1, a + 1, t.sum().to(torch.int64).item(), t.sin())
            out_it = torch.ops.higher_order.while_loop(cond_fn, body_fn, carry, tuple())
            out_inc = pytree.tree_map(lambda x: x + 1, out_it)
            out_add = pytree.tree_map(lambda x: x + t, out_it)
            return out_inc, out_add

    nested2 = Nested()
    simple_with_linear = SimpleWithLinear()
    simple_with_pytree_carry = SimpleWithPytreeCarry()
    nested_with_linear = NestedWithLinear()
    int_carry = IntCarry()
    pytree_int_carry = PytreeIntCarry()
    const_and_symint_output = ConstAndSymIntOutput()

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
        "simple_with_pytree_carry": (
            simple_with_pytree_carry,
            (
                torch.tensor(3),
                ([torch.randn(3, 3)], {"x": torch.randn(3, 3), "y": torch.randn(3, 3)}),
            ),
        ),
        "int_carry": (int_carry, (torch.randn(2, 3),)),
        "pytree_int_carry": (
            pytree_int_carry,
            (torch.randn(2, 3),),
        ),
        "const_and_symint_output": (
            const_and_symint_output,
            (torch.randn(2, 3),),
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

    def check_autograd(self, result, result_exp, params):
        params_flatten = pytree.tree_leaves(params)
        result_flatten = pytree.tree_leaves(result)
        result_exp_flatten = pytree.tree_leaves(result_exp)
        grad_exp_init = [torch.ones_like(el) for el in result_exp_flatten]
        expected_grads = torch.autograd.grad(
            result_exp_flatten, params_flatten, grad_exp_init
        )
        grad_init = [torch.ones_like(el) for el in result_flatten]
        grads = torch.autograd.grad(result_flatten, params_flatten, grad_init)
        self.assertEqual(grads, expected_grads, atol=6e-05, rtol=6e-06)

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

    def test_cond_autograd_simple(self):
        def true_fn(x):
            return x.sin()

        def false_fn(x):
            return x.cos()

        for pred, fn in zip(
            [torch.tensor(False), torch.tensor(True)], [false_fn, true_fn]
        ):
            x = torch.randn(4, requires_grad=True)
            result = cond(pred, true_fn, false_fn, (x,))
            self.assertEqual(result, fn(x))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (x,), grad_out)
            expected_grads = torch.autograd.grad(fn(x), (x,), grad_out)
            self.assertEqual(expected_grads, grads)

        def f(pred, x):
            result = cond(pred, true_fn, false_fn, (x,))
            grad_out = torch.ones_like(result)
            return torch.autograd.grad(result, (x,), grad_out)

        gm = make_fx(f)(pred, x)

        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, pred_1, x_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (x_1,));  true_graph_0 = false_graph_0 = None
    getitem = cond[0];  cond = None
    ones_like = torch.ops.aten.ones_like.default(getitem, pin_memory = False);  getitem = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    cond_1 = torch.ops.higher_order.cond(pred_1, true_graph_1, false_graph_1, (x_1, ones_like));  pred_1 = true_graph_1 = false_graph_1 = x_1 = ones_like = None
    getitem_1 = cond_1[0];  cond_1 = None
    return (getitem_1,)""",  # noqa: B950
        )

    def test_cond_autograd_complex(self):
        def true_fn(x):
            return torch.abs((x**2).sin())

        def false_fn(x):
            return (x + 42).cos()

        for pred, fn in zip(
            [torch.tensor(False), torch.tensor(True)], [false_fn, true_fn]
        ):
            x = torch.randn(4, requires_grad=True)
            result = cond(pred, true_fn, false_fn, (x,))
            self.assertEqual(result, fn(x))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (x,), grad_out)
            expected_grads = torch.autograd.grad(fn(x), (x,), grad_out)
            self.assertEqual(expected_grads, grads)

        def f(pred, x):
            result = cond(pred, true_fn, false_fn, (x,))
            grad_out = torch.ones_like(result)
            return torch.autograd.grad(result, (x,), grad_out)

        gm = make_fx(f)(pred, x)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, pred_1, x_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (x_1,));  true_graph_0 = false_graph_0 = None
    getitem = cond[0];  cond = None
    ones_like = torch.ops.aten.ones_like.default(getitem, pin_memory = False);  getitem = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    cond_1 = torch.ops.higher_order.cond(pred_1, true_graph_1, false_graph_1, (x_1, ones_like));  pred_1 = true_graph_1 = false_graph_1 = x_1 = ones_like = None
    getitem_1 = cond_1[0];  cond_1 = None
    return (getitem_1,)""",  # noqa: B950
        )

    @skipIfTorchDynamo("Skip due to graph break when run with dynamo")
    def test_cond_autograd_nested(self):
        class Nested(torch.nn.Module):
            def forward(self, p0, p1, p2, a, b, c):
                def true_fn(x0, y0, z0):
                    def true_true_fn(x1, y1, z1):
                        return (x1 - y1 * z1) * 3.14

                    def true_false_fn(x1, y1, z1):
                        def true_false_true_fn(x2, y2, z2):
                            return (x2 * y2 * z2) / 2.71

                        def true_false_false_fn(x2, y2, z2):
                            return (x2 + y2 + z2) * 1.23

                        return torch.cond(
                            p2, true_false_true_fn, true_false_false_fn, [x1, y1, z1]
                        )

                    return torch.cond(p1, true_true_fn, true_false_fn, [x0, y0, z0])

                def false_fn(x0, y0, z0):
                    def false_true_fn(x1, y1, z1):
                        def false_true_true_fn(x2, y2, z2):
                            return (x2 - y2 - z2) + 1.23

                        def false_true_false_fn(x2, y2, z2):
                            return (x2 / y2 / z2) - 3.14

                        return torch.cond(
                            p2, false_true_true_fn, false_true_false_fn, [x1, y1, z1]
                        )

                    def false_false_fn(x1, y1, z1):
                        return (x1 - y1 * z1) / 2.71

                    return torch.cond(p1, false_true_fn, false_false_fn, [x0, y0, z0])

                return torch.cond(p0, true_fn, false_fn, [a, b, c])

        nn_module = Nested()

        def true_fn(x):
            return nn_module(
                torch.tensor(False), torch.tensor(True), torch.tensor(False), x, x, x
            )

        def false_fn(x):
            return nn_module(
                torch.tensor(True), torch.tensor(False), torch.tensor(True), x, x, x
            )

        x = torch.randn(4, requires_grad=True)

        for pred, fn in zip(
            [torch.tensor(False), torch.tensor(True)], [false_fn, true_fn]
        ):
            result = cond(pred, true_fn, false_fn, (x,))
            self.assertEqual(result, fn(x))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (x,), grad_out)
            expected_grads = torch.autograd.grad(fn(x), (x,), grad_out)
            self.assertEqual(expected_grads, grads)

    @skipIfTorchDynamo("Skip due to graph break when run with dynamo")
    def test_cond_autograd_mixed_require_grad(self):
        def true_fn(x, y, z):
            return x * y * z

        def false_fn(x, y, z):
            return x + y + z

        x = torch.randn(4, requires_grad=True)
        y = torch.randn(4, requires_grad=False)

        for pred, fn in zip(
            [torch.tensor(False), torch.tensor(True)], [false_fn, true_fn]
        ):
            result = cond(pred, true_fn, false_fn, (x, y, x))
            self.assertEqual(result, fn(x, y, x))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (x,), grad_out)
            expected_grads = torch.autograd.grad(fn(x, y, x), (x,), grad_out)
            self.assertEqual(expected_grads, grads)

        def f(pred, x, y, z):
            result = cond(pred, true_fn, false_fn, (x, y, z))
            grad_out = torch.ones_like(result)
            return torch.autograd.grad(result, (x,), grad_out)

        gm = make_fx(f)(pred, x, y, x)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, pred_1, x_1, y_1, z_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (z_1, y_1));  true_graph_0 = false_graph_0 = None
    getitem = cond[0];  cond = None
    ones_like = torch.ops.aten.ones_like.default(getitem, pin_memory = False);  getitem = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    cond_1 = torch.ops.higher_order.cond(pred_1, true_graph_1, false_graph_1, (z_1, y_1, ones_like));  pred_1 = true_graph_1 = false_graph_1 = z_1 = y_1 = ones_like = None
    getitem_1 = cond_1[0]
    getitem_2 = cond_1[1];  cond_1 = getitem_2 = None
    return (getitem_1,)""",  # noqa: B950
        )

    @skipIfTorchDynamo("Skip due to graph break when run with dynamo")
    def test_cond_autograd_grad_through_cond(self):
        nn_module = torch.nn.Linear(4, 4)

        def true_fn(x):
            return nn_module(x)

        def false_fn(X):
            return x * nn_module(x)

        x = torch.randn(4, requires_grad=True)

        for pred, fn in zip(
            [torch.tensor(False), torch.tensor(True)], [false_fn, true_fn]
        ):
            result = cond(pred, true_fn, false_fn, (x,))
            self.assertEqual(result, fn(x))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (nn_module.weight,), grad_out)
            expected_grads = torch.autograd.grad(
                fn(
                    x,
                ),
                (nn_module.weight,),
                grad_out,
            )
            self.assertEqual(expected_grads, grads)

        def f(pred, x):
            result = cond(pred, true_fn, false_fn, (x,))
            grad_out = torch.ones_like(result)
            return torch.autograd.grad(result, (nn_module.weight,), grad_out)

        # need to set _allow_non_fake_inputs = True because model parameters don't
        # get fakified.
        gm = make_fx(f, tracing_mode="symbolic", _allow_non_fake_inputs=True)(pred, x)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, pred_1, x_1):
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0)
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    _param_constant0 = self._param_constant0
    _param_constant1 = self._param_constant1
    _tensor_constant0 = self._tensor_constant0
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (_param_constant0, _param_constant1, x_1, sym_size_int, _tensor_constant0));  true_graph_0 = false_graph_0 = _param_constant0 = _param_constant1 = _tensor_constant0 = None
    getitem = cond[0];  cond = None
    ones_like = torch.ops.aten.ones_like.default(getitem, pin_memory = False);  getitem = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    _param_constant0_1 = self._param_constant0
    _param_constant1_1 = self._param_constant1
    _tensor_constant0_1 = self._tensor_constant0
    cond_1 = torch.ops.higher_order.cond(pred_1, true_graph_1, false_graph_1, (_param_constant0_1, _param_constant1_1, x_1, sym_size_int, _tensor_constant0_1, ones_like));  pred_1 = true_graph_1 = false_graph_1 = _param_constant0_1 = _param_constant1_1 = x_1 = sym_size_int = _tensor_constant0_1 = ones_like = None
    getitem_1 = cond_1[0];  getitem_1 = None
    getitem_2 = cond_1[1]
    getitem_3 = cond_1[2];  getitem_3 = None
    getitem_4 = cond_1[3];  cond_1 = getitem_4 = None
    return (getitem_2,)""",  # noqa: B950
        )

    def test_cond_in_forloop(self):
        def for_loop_fake(x):
            for _ in range(3):
                x = x * x + 1
            return x

        def for_loop_test(x):
            for i in range(3):
                pred = i < 3

                def true_fn(x):
                    return x * x + 1

                def false_fn(x):
                    return x

                x = cond(pred, true_fn, false_fn, (x,))

            return x

        x = torch.ones(4, requires_grad=True)
        x_new = for_loop_test(x)
        x_exp = for_loop_fake(x)

        self.assertEqual(x_new, x_exp)

        grad_out = torch.ones_like(x_new)
        grads = torch.autograd.grad(x_new, (x,), grad_out)
        expected_grads = torch.autograd.grad(x_exp, (x,), grad_out)
        self.assertEqual(expected_grads, grads)

        def f(x):
            x_new = for_loop_test(x)
            grad_out = torch.ones_like(x_new)
            return torch.autograd.grad(x_new, (x,), grad_out)

        gm = make_fx(f, tracing_mode="symbolic")(x)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x_1):
    mul = torch.ops.aten.mul.Tensor(x_1, x_1)
    add = torch.ops.aten.add.Tensor(mul, 1);  mul = None
    mul_1 = torch.ops.aten.mul.Tensor(add, add)
    add_1 = torch.ops.aten.add.Tensor(mul_1, 1);  mul_1 = None
    mul_2 = torch.ops.aten.mul.Tensor(add_1, add_1)
    add_2 = torch.ops.aten.add.Tensor(mul_2, 1);  mul_2 = None
    ones_like = torch.ops.aten.ones_like.default(add_2, pin_memory = False);  add_2 = None
    mul_3 = torch.ops.aten.mul.Tensor(ones_like, add_1)
    mul_4 = torch.ops.aten.mul.Tensor(ones_like, add_1);  ones_like = add_1 = None
    add_3 = torch.ops.aten.add.Tensor(mul_4, mul_3);  mul_4 = mul_3 = None
    mul_5 = torch.ops.aten.mul.Tensor(add_3, add)
    mul_6 = torch.ops.aten.mul.Tensor(add_3, add);  add_3 = add = None
    add_4 = torch.ops.aten.add.Tensor(mul_6, mul_5);  mul_6 = mul_5 = None
    mul_7 = torch.ops.aten.mul.Tensor(add_4, x_1)
    mul_8 = torch.ops.aten.mul.Tensor(add_4, x_1);  add_4 = x_1 = None
    add_5 = torch.ops.aten.add.Tensor(mul_8, mul_7);  mul_8 = mul_7 = None
    return (add_5,)""",  # noqa: B950
        )

    @skipIfTorchDynamo("Skip due to graph break when run with dynamo")
    def test_cond_autograd_pytree_not_all_inputs_used(self):
        def true_fn(x):
            return x["t"][0] + x["t"][1]["b"]

        def false_fn(x):
            return x["t"][0] * (x["t"][2][0] / x["t"][1]["b"])

        a = torch.randn(4, requires_grad=True)
        b = torch.randn(4, requires_grad=True)
        c = torch.randn(4, requires_grad=True)

        for pred, fn in zip(
            [torch.tensor(False), torch.tensor(True)], [false_fn, true_fn]
        ):
            result = cond(pred, true_fn, false_fn, ({"t": [a, {"b": b}, (c,)]},))
            self.assertEqual(result, fn({"t": [a, {"b": b}, (c,)]}))

            grad_out = torch.ones_like(result)
            if pred:
                with self.assertRaisesRegex(Exception, r"."):
                    grads = torch.autograd.grad(result, (a, b, c), grad_out)
                    expected_grads = torch.autograd.grad(
                        fn({"t": [a, {"b": b}, (c,)]}), (a, b, c), grad_out
                    )
                    self.assertEqual(expected_grads, grads)

        def f(pred, a, b, c):
            result = cond(pred, true_fn, false_fn, ({"t": [a, {"b": b}, (c,)]},))
            grad_out = torch.ones_like(result)
            return torch.autograd.grad(result, (a, b), grad_out)

        gm = make_fx(f, tracing_mode="symbolic", _allow_non_fake_inputs=True)(
            pred, a, b, c
        )
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, pred_1, a_1, b_1, c_1):
    sym_size_int = torch.ops.aten.sym_size.int(a_1, 0)
    sym_size_int_1 = torch.ops.aten.sym_size.int(b_1, 0)
    sym_size_int_2 = torch.ops.aten.sym_size.int(c_1, 0)
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (a_1, b_1, sym_size_int, sym_size_int_1, c_1, sym_size_int_2));  true_graph_0 = false_graph_0 = None
    getitem = cond[0];  cond = None
    ones_like = torch.ops.aten.ones_like.default(getitem, pin_memory = False);  getitem = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    cond_1 = torch.ops.higher_order.cond(pred_1, true_graph_1, false_graph_1, (a_1, b_1, sym_size_int, sym_size_int_1, c_1, sym_size_int_2, ones_like));  pred_1 = true_graph_1 = false_graph_1 = a_1 = b_1 = sym_size_int = sym_size_int_1 = c_1 = sym_size_int_2 = ones_like = None
    getitem_1 = cond_1[0]
    getitem_2 = cond_1[1]
    getitem_3 = cond_1[2];  cond_1 = getitem_3 = None
    return (getitem_1, getitem_2)""",  # noqa: B950
        )
        # Forward
        self.assertExpectedInline(
            gm.true_graph_0.code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1):
    add = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
    return (add,)""",
        )
        # Backward
        self.assertExpectedInline(
            gm.true_graph_1.code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1):
    add = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = add = None
    zeros_like = torch.ops.aten.zeros_like.default(arg4_1, pin_memory = False);  arg4_1 = None
    clone = torch.ops.aten.clone.default(arg6_1)
    clone_1 = torch.ops.aten.clone.default(arg6_1);  arg6_1 = None
    return [clone, clone_1, zeros_like]""",
        )

    def test_cond_autograd_pytree_input(self):
        def true_fn(x):
            return x["t"][0] + x["t"][1]["b"] * x["t"][2][0]

        def false_fn(x):
            return x["t"][0] * (x["t"][2][0] / x["t"][1]["b"])

        a = torch.randn(4, requires_grad=True)
        b = torch.randn(4, requires_grad=True)
        c = torch.randn(4, requires_grad=True)

        for pred, fn in zip(
            [torch.tensor(False), torch.tensor(True)], [false_fn, true_fn]
        ):
            result = cond(pred, true_fn, false_fn, ({"t": [a, {"b": b}, (c,)]},))
            self.assertEqual(result, fn({"t": [a, {"b": b}, (c,)]}))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (a, b), grad_out)
            expected_grads = torch.autograd.grad(
                fn({"t": [a, {"b": b}, (c,)]}), (a, b), grad_out
            )
            self.assertEqual(expected_grads, grads)

        def f(pred):
            result = cond(pred, true_fn, false_fn, ({"t": [a, {"b": b}, (c,)]},))
            grad_out = torch.ones_like(result)
            return torch.autograd.grad(result, (a, b), grad_out)

        # need to set _allow_non_fake_inputs = True because model parameters don't
        # get fakified.
        gm = make_fx(f, tracing_mode="symbolic", _allow_non_fake_inputs=True)(pred)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, pred_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    _tensor_constant0 = self._tensor_constant0
    _tensor_constant1 = self._tensor_constant1
    _tensor_constant2 = self._tensor_constant2
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (_tensor_constant0, _tensor_constant1, _tensor_constant2));  true_graph_0 = false_graph_0 = _tensor_constant0 = _tensor_constant1 = _tensor_constant2 = None
    getitem = cond[0];  cond = None
    ones_like = torch.ops.aten.ones_like.default(getitem, pin_memory = False);  getitem = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    _tensor_constant0_1 = self._tensor_constant0
    _tensor_constant1_1 = self._tensor_constant1
    _tensor_constant2_1 = self._tensor_constant2
    cond_1 = torch.ops.higher_order.cond(pred_1, true_graph_1, false_graph_1, (_tensor_constant0_1, _tensor_constant1_1, _tensor_constant2_1, ones_like));  pred_1 = true_graph_1 = false_graph_1 = _tensor_constant0_1 = _tensor_constant1_1 = _tensor_constant2_1 = ones_like = None
    getitem_1 = cond_1[0]
    getitem_2 = cond_1[1]
    getitem_3 = cond_1[2];  cond_1 = getitem_3 = None
    return (getitem_1, getitem_2)""",  # noqa: B950
        )

    def test_cond_autograd_different_pytree_output(self):
        def true_fn(x):
            return x["t"][0], {"r": x["t"][2][0] / x["t"][1]["b"]}, [x["t"][2][0]]

        def false_fn(x):
            return {"res": [x["t"][0] * x["t"][1]["b"], x["t"][2][0]]}

        a = torch.randn(4, requires_grad=True)
        b = torch.randn(4, requires_grad=True)
        c = torch.randn(4, requires_grad=True)

        for pred in [torch.tensor(False), torch.tensor(True)]:
            with self.assertRaisesRegex(
                torch._dynamo.exc.UncapturedHigherOrderOpError,
                r"Higher Order Operator: torch\.cond",
            ):
                cond(pred, true_fn, false_fn, ({"t": [a, {"b": b}, (c,)]},))

    @skipIfTorchDynamo("Skip due to graph break when run with dynamo")
    def test_cond_autograd_same_pytree_output(self):
        def true_fn(x):
            return {"res": [x["t"][0].clone(), (x["t"][2][0].clone(),)]}

        def false_fn(x):
            return {"res": [x["t"][1]["b"].clone(), (x["t"][2][0].clone(),)]}

        a = torch.randn(4, requires_grad=True)
        b = torch.randn(4, requires_grad=True)
        c = torch.randn(4, requires_grad=True)

        for pred, fn in zip(
            [torch.tensor(False), torch.tensor(True)], [false_fn, true_fn]
        ):
            result = cond(pred, true_fn, false_fn, ({"t": [a, {"b": b}, (c,)]},))
            result_exp = fn({"t": [a, {"b": b}, (c,)]})
            self.assertEqual(result, result_exp)

            result_flat, _ = pytree.tree_flatten(result)
            result_exp_flat, _ = pytree.tree_flatten(result_exp)

            grad_out = [torch.ones_like(g) for g in result_flat]
            expected_grads = torch.autograd.grad(result_exp_flat, (c,), grad_out)
            grads = torch.autograd.grad(result_flat, (c,), grad_out)
            self.assertEqual(expected_grads, grads)

        def f(pred):
            result = cond(pred, true_fn, false_fn, ({"t": [a, {"b": b}, (c,)]},))
            return result

        gm = make_fx(f, tracing_mode="real", _allow_non_fake_inputs=True)(pred)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, pred_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    _tensor_constant0 = self._tensor_constant0
    _tensor_constant1 = self._tensor_constant1
    _tensor_constant2 = self._tensor_constant2
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (_tensor_constant0, _tensor_constant1, _tensor_constant2));  pred_1 = true_graph_0 = false_graph_0 = _tensor_constant0 = _tensor_constant1 = _tensor_constant2 = None
    getitem = cond[0]
    getitem_1 = cond[1];  cond = None
    return {'res': [getitem, (getitem_1,)]}""",  # noqa: B950
        )

    @skipIfTorchDynamo("Skip due to graph break when run with dynamo")
    def test_cond_autograd_torch_nn_module(self):
        nn_module_true = torch.nn.Linear(4, 4)

        def true_fn(x):
            return nn_module_true(torch.abs((x**2).sin()))

        nn_module_false = torch.nn.GRUCell(4, 4)

        def false_fn(x):
            return nn_module_false((x + 42).cos())

        for pred, fn in zip(
            [torch.tensor(False), torch.tensor(True)], [false_fn, true_fn]
        ):
            x = torch.randn(4, requires_grad=True)
            result = cond(pred, true_fn, false_fn, (x,))
            self.assertEqual(result, fn(x))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (x,), grad_out)
            expected_grads = torch.autograd.grad(fn(x), (x,), grad_out)
            self.assertEqual(expected_grads, grads)

        def f(pred, x):
            result = cond(pred, true_fn, false_fn, (x,))
            grad_out = torch.ones_like(result)
            return torch.autograd.grad(result, (x,), grad_out)

        gm = make_fx(f)(pred, x)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, pred_1, x_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    _param_constant0 = self._param_constant0
    _param_constant1 = self._param_constant1
    _param_constant2 = self._param_constant2
    _param_constant3 = self._param_constant3
    _param_constant4 = self._param_constant4
    _param_constant5 = self._param_constant5
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (x_1, _param_constant0, _param_constant1, _param_constant2, _param_constant3, _param_constant4, _param_constant5));  true_graph_0 = false_graph_0 = _param_constant0 = _param_constant1 = _param_constant2 = _param_constant3 = _param_constant4 = _param_constant5 = None
    getitem = cond[0];  cond = None
    ones_like = torch.ops.aten.ones_like.default(getitem, pin_memory = False);  getitem = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    _param_constant0_1 = self._param_constant0
    _param_constant1_1 = self._param_constant1
    _param_constant2_1 = self._param_constant2
    _param_constant3_1 = self._param_constant3
    _param_constant4_1 = self._param_constant4
    _param_constant5_1 = self._param_constant5
    cond_1 = torch.ops.higher_order.cond(pred_1, true_graph_1, false_graph_1, (x_1, _param_constant0_1, _param_constant1_1, _param_constant2_1, _param_constant3_1, _param_constant4_1, _param_constant5_1, ones_like));  pred_1 = true_graph_1 = false_graph_1 = x_1 = _param_constant0_1 = _param_constant1_1 = _param_constant2_1 = _param_constant3_1 = _param_constant4_1 = _param_constant5_1 = ones_like = None
    getitem_1 = cond_1[0]
    getitem_2 = cond_1[1];  getitem_2 = None
    getitem_3 = cond_1[2];  getitem_3 = None
    getitem_4 = cond_1[3];  getitem_4 = None
    getitem_5 = cond_1[4];  getitem_5 = None
    getitem_6 = cond_1[5];  getitem_6 = None
    getitem_7 = cond_1[6];  cond_1 = getitem_7 = None
    return (getitem_1,)""",  # noqa: B950
        )

    def test_cond_autograd_user_nn_module(self):
        class User_nn_module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, input):
                return input * input

        nn_module_true = User_nn_module()

        def true_fn(x):
            return nn_module_true(torch.abs((x**2).sin()))

        nn_module_false = torch.nn.ReLU(inplace=False)

        def false_fn(x):
            return nn_module_false((x + 42).cos())

        for pred, fn in zip(
            [torch.tensor(False), torch.tensor(True)], [false_fn, true_fn]
        ):
            x = torch.randn(4, requires_grad=True)
            result = cond(pred, true_fn, false_fn, (x,))
            self.assertEqual(result, fn(x))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (x,), grad_out)
            expected_grads = torch.autograd.grad(fn(x), (x,), grad_out)
            self.assertEqual(expected_grads, grads)

        def f(pred, x):
            result = cond(pred, true_fn, false_fn, (x,))
            grad_out = torch.ones_like(result)
            return torch.autograd.grad(result, (x,), grad_out)

        gm = make_fx(f)(pred, x)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, pred_1, x_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (x_1,));  true_graph_0 = false_graph_0 = None
    getitem = cond[0];  cond = None
    ones_like = torch.ops.aten.ones_like.default(getitem, pin_memory = False);  getitem = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    cond_1 = torch.ops.higher_order.cond(pred_1, true_graph_1, false_graph_1, (x_1, ones_like));  pred_1 = true_graph_1 = false_graph_1 = x_1 = ones_like = None
    getitem_1 = cond_1[0];  cond_1 = None
    return (getitem_1,)""",  # noqa: B950
        )

    def test_cond_autograd_inner_fn(self):
        def true_fn(x):
            return torch.abs((x**2).sin())

        def false_fn(x):
            def inner_fn(x):
                return x**2

            return torch.abs(inner_fn(x).sin())

        x = torch.randn(4, requires_grad=True)
        pred = torch.tensor(False)
        fn = false_fn
        result_false = cond(pred, true_fn, false_fn, (x,))
        self.assertEqual(result_false, fn(x))

        grad_out = torch.ones_like(result_false)
        grads_false = torch.autograd.grad(result_false, (x,), grad_out)
        expected_grads = torch.autograd.grad(fn(x), (x,), grad_out)
        self.assertEqual(expected_grads, grads_false)

        pred = torch.tensor(True)
        fn = true_fn
        result_true = cond(pred, true_fn, false_fn, (x,))
        self.assertEqual(result_true, fn(x))
        self.assertEqual(result_false, result_true)

        grad_out = torch.ones_like(result_true)
        grads_true = torch.autograd.grad(result_true, (x,), grad_out)
        expected_grads = torch.autograd.grad(fn(x), (x,), grad_out)
        self.assertEqual(expected_grads, grads_true)
        self.assertEqual(grads_false, grads_true)

        def f(pred, x):
            result = cond(pred, true_fn, false_fn, (x,))
            grad_out = torch.ones_like(result)
            return torch.autograd.grad(result, (x,), grad_out)

        gm = make_fx(f)(pred, x)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, pred_1, x_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (x_1,));  true_graph_0 = false_graph_0 = None
    getitem = cond[0];  cond = None
    ones_like = torch.ops.aten.ones_like.default(getitem, pin_memory = False);  getitem = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    cond_1 = torch.ops.higher_order.cond(pred_1, true_graph_1, false_graph_1, (x_1, ones_like));  pred_1 = true_graph_1 = false_graph_1 = x_1 = ones_like = None
    getitem_1 = cond_1[0];  cond_1 = None
    return (getitem_1,)""",  # noqa: B950
        )

    def test_cond_autograd_inner_tensor(self):
        def true_fn(x):
            return torch.abs((x**2).sin())

        def false_fn(x):
            y = torch.ones(4, requires_grad=False) * 42
            return (x * y).cos()

        for pred, fn in zip(
            [torch.tensor(False), torch.tensor(True)], [false_fn, true_fn]
        ):
            x = torch.randn(4, requires_grad=True)
            result = cond(pred, true_fn, false_fn, (x,))
            self.assertEqual(result, fn(x))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (x,), grad_out)
            expected_grads = torch.autograd.grad(fn(x), (x,), grad_out)
            self.assertEqual(expected_grads, grads)

        def f(pred, x):
            result = cond(pred, true_fn, false_fn, (x,))
            grad_out = torch.ones_like(result)
            return torch.autograd.grad(result, (x,), grad_out)

        gm = make_fx(f)(pred, x)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, pred_1, x_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (x_1,));  true_graph_0 = false_graph_0 = None
    getitem = cond[0];  cond = None
    ones_like = torch.ops.aten.ones_like.default(getitem, pin_memory = False);  getitem = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    cond_1 = torch.ops.higher_order.cond(pred_1, true_graph_1, false_graph_1, (x_1, ones_like));  pred_1 = true_graph_1 = false_graph_1 = x_1 = ones_like = None
    getitem_1 = cond_1[0];  cond_1 = None
    return (getitem_1,)""",  # noqa: B950
        )

    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA.")
    def test_cond_autograd_gpu(self):
        def true_fn(x):
            return x.sin()

        def false_fn(x):
            return x.cos()

        for pred, fn in zip(
            [torch.tensor(False, device="cuda"), torch.tensor(True, device="cuda")],
            [false_fn, true_fn],
        ):
            x = torch.randn(4, requires_grad=True, device="cuda")
            result = cond(pred, true_fn, false_fn, (x,))
            self.assertEqual(result, fn(x))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (x,), grad_out)
            expected_grads = torch.autograd.grad(fn(x), (x,), grad_out)
            self.assertEqual(expected_grads, grads)

    def _test_cond_autograd(self, cond_fct, pred_fn, true_fn, false_fn, operands):
        from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata

        # This is a helper function that extracts the metadata from the tensor and
        # sets the requires_grad flag to false. This is needed as we compare the
        # metadata of the operands and the gradients
        def _extract_tensor_metadata_except_requires_grad(arg):
            metadata = _extract_tensor_metadata(arg)
            metadata = TensorMetadata(
                metadata.shape,
                metadata.dtype,
                False,
                metadata.stride,
                metadata.memory_format,
                metadata.is_quantized,
                metadata.qparams,
            )
            return metadata

        # Comparison of FWD path
        cond_outputs = cond_fct(pred_fn(*operands), true_fn, false_fn, operands)
        operands_forced_grad = [o.clone().detach() for o in operands]
        for o in operands_forced_grad:
            o.requires_grad = True
        cond_outputs_exp = (
            true_fn(*operands_forced_grad)
            if pred_fn(*operands_forced_grad)
            else false_fn(*operands_forced_grad)
        )
        self.assertEqual(cond_outputs, cond_outputs_exp)

        # Comparison of BWD path
        cond_inputs = [o for o in operands if o.requires_grad]
        cond_inputs_exp = [o for o in operands_forced_grad if o.requires_grad]

        # Check if at least some operators require grads
        if len(cond_inputs) > 0:
            grad_inputs = torch.autograd.grad(
                cond_outputs, cond_inputs, allow_unused=True, retain_graph=True
            )
            grad_inputs_exp = torch.autograd.grad(
                cond_outputs_exp,
                cond_inputs_exp,
                allow_unused=True,
                materialize_grads=True,
            )

            grad_exp_masked = [
                g for g, o in zip(grad_inputs_exp, operands) if o.requires_grad
            ]
            self.assertEqual(grad_exp_masked, grad_inputs)

            # Extraction and comparison of Metadata of operands and gradients
            operands_metadata = [
                _extract_tensor_metadata_except_requires_grad(o) for o in cond_inputs
            ]
            grad_metadata = [
                _extract_tensor_metadata_except_requires_grad(o) for o in grad_inputs
            ]
            self.assertTrue(
                all(op == g for op, g in zip(operands_metadata, grad_metadata))
            )

        return cond_outputs, cond_inputs

    @skipIfTorchDynamo("don't test compile on compile")
    @unittest.skipIf(not SM70OrLater, "triton")
    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA.")
    @parametrize("compile_mode", ["compile_dynamic_shape"])
    @parametrize("scalar", [False])
    def test_cond_autograd_zeros_unused_branch_complex_compile_fail(
        self, compile_mode, scalar
    ):
        device = torch.device("cuda")
        cond_fct = compile_mode_helper(torch.cond, compile_mode)

        autograd = [False, True, True, True, True]

        if scalar:
            # These operands work
            x = torch.randn((), device=device, requires_grad=bool(autograd[0]))
            w1 = torch.randn((), device=device, requires_grad=bool(autograd[1]))
            b1 = torch.randn((), device=device, requires_grad=bool(autograd[2]))
            w2 = torch.randn((), device=device, requires_grad=bool(autograd[3]))
            b2 = torch.randn((), device=device, requires_grad=bool(autograd[4]))
        else:
            # These operands do not work
            x = torch.randn(4, 5, device=device, requires_grad=bool(autograd[0]))
            w1 = torch.randn(2, 4, device=device, requires_grad=bool(autograd[1]))
            b1 = torch.randn(2, 1, device=device, requires_grad=bool(autograd[2]))
            w2 = torch.randn(2, 4, device=device, requires_grad=bool(autograd[3]))
            b2 = torch.randn(1, 5, device=device, requires_grad=bool(autograd[4]))

        operands = [x, w1, b1, w2, b2]

        def true_fn(x, w1, b1, w2, b2):
            if scalar:
                # This works
                return ((w1 * x + b1),)
            else:
                # This does not work
                return ((w1 @ x + b1).sum(),)

        def false_fn(x, w1, b1, w2, b2):
            if scalar:
                # This works
                return ((w2 * x + b2),)
            else:
                # This does not work
                return ((w2 @ x + b2).sum(),)

        def pred_fn(x, w1, b1, w2, b2):
            return x.mean() > 0

        cond_outputs, cond_inputs = self._test_cond_autograd(
            cond_fct, pred_fn, true_fn, false_fn, operands
        )

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
            RuntimeError,
            r"Higher Order Operator: torch\.ops\.higher_order\.map_impl",
        ):
            control_flow.map(f, x, y)

        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.UncapturedHigherOrderOpError,
            # "Expected all leaves to be of torch.Tensor type.*",
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.ops\.higher_order\.map_impl",
        ):
            control_flow.map(f1, x, y)

        # return None is OK
        control_flow.map(f2, x, y)

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

    def test_map_autograd_higher_order(self):
        from torch.autograd.functional import hessian as hes, jacobian as jac

        def f(x, y):
            return x.sin().cos() + y

        def wrapper_jac(x, y):
            return control_flow.map(f, x, y)

        def wrapper_jac_fake(x, y):
            return _fake_map(f, x, y)

        def wrapper_hes(x, y):
            return control_flow.map(f, x, y).sum()

        def wrapper_hes_fake(x, y):
            return _fake_map(f, x, y).sum()

        for g_fct, (wrap, wrap_fake) in [
            (jac, [wrapper_jac, wrapper_jac_fake]),
            (hes, [wrapper_hes, wrapper_hes_fake]),
        ]:
            xs = torch.ones(3, 2, 2, requires_grad=True)
            # Disable the gradient computation for y
            y = torch.ones(2, requires_grad=False)
            res = control_flow.map(f, xs, y)
            expected_res = _fake_map(f, xs, y)
            self.assertEqual(expected_res, res)

            expected_grads = g_fct(wrap_fake, (xs, y))
            grads = g_fct(wrap, (xs, y))
            self.assertEqual(expected_res, res)
            self.assertEqual(expected_grads, grads)

    def test_scan_y_less_ndim_then_dim(self):
        def combine_fn(carry, x):
            return carry @ x, (carry @ x).sum()

        init = torch.randn(4, 3)
        xs = torch.randn(3, 3, 2)
        dim = 2
        out = scan(combine_fn, init, xs, dim=dim)
        exp_out = _fake_scan(combine_fn, init, xs, dim=dim)
        self.assertEqual(out, exp_out)

    # TODO: provide an implementation for all compile modes and re-enable all test
    @skipIfTorchDynamo("don't test compile on compile")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_compile(self, reverse, compile_mode, device, autograd):
        def add2(x: torch.Tensor, y: torch.Tensor):
            return x * y, x + y

        x = torch.randn(3, 10, 2, device=device, requires_grad=autograd)

        scan_fct = compile_mode_helper(scan, compile_mode)

        for op, op_pt, init in [
            (
                get_scan_combine_fn("add", False),
                torch.cumsum,
                torch.zeros(10, 2, device=device, requires_grad=autograd),
            ),
            (
                get_scan_combine_fn("mul", False),
                torch.cumprod,
                torch.ones(10, 2, device=device, requires_grad=autograd),
            ),
        ]:
            result = scan_fct(op, init, x, dim=0, reverse=reverse)
            result_exp = _fake_scan(op, init=init, xs=x, dim=0, reverse=reverse)
            self.assertEqual(result, result_exp)
            if not reverse:
                result_exp_PT = op_pt(x, 0)
                self.assertEqual(result[1], result_exp_PT)

            if autograd:
                self.check_autograd(result, result_exp, (init, x))

        # Jax Examples
        x = torch.arange(0, 4, device=device, dtype=torch.int64)
        init = torch.zeros(1, device=device, dtype=torch.int64)
        cumsum1 = scan_fct(
            get_scan_combine_fn("add", False),
            init,
            x,
            dim=0,
            reverse=reverse,
        )
        cumsum_exp = _fake_scan(
            get_scan_combine_fn("add", False),
            init=init,
            xs=x,
            dim=0,
            reverse=reverse,
        )
        if not reverse:
            self.assertEqual(
                cumsum1[1],
                torch.tensor([[0.0], [1.0], [3.0], [6.0]], dtype=torch.int64),
            )
            self.assertEqual(cumsum1[0], torch.tensor([6.0], dtype=torch.int64))
        else:
            self.assertEqual(
                cumsum1[1],
                torch.tensor([[6.0], [6.0], [5.0], [3.0]], dtype=torch.int64),
            )
            self.assertEqual(cumsum1[0], torch.tensor([6.0], dtype=torch.int64))
        self.assertEqual(cumsum1, cumsum_exp)

        # Different carry computation as output computation
        x = torch.arange(1, 5, device=device, dtype=torch.int64)
        init = torch.ones(1, device=device, dtype=torch.int64)
        result = scan_fct(add2, init, x, dim=0, reverse=reverse)
        result_exp = _fake_scan(add2, init=init, xs=x, dim=0, reverse=reverse)
        if not reverse:
            self.assertEqual(
                result[1],
                torch.tensor([[2.0], [3.0], [5.0], [10.0]], dtype=torch.int64),
            )
            self.assertEqual(result[0], torch.tensor([24.0], dtype=torch.int64))
        else:
            self.assertEqual(
                result[1],
                torch.tensor([[25.0], [14.0], [7.0], [5.0]], dtype=torch.int64),
            )
            self.assertEqual(result[0], torch.tensor([24.0], dtype=torch.int64))
        self.assertEqual(result, result_exp)

        # Non associative operation
        x = torch.arange(
            0, 5, device=device, dtype=torch.float32, requires_grad=autograd
        )
        init = torch.ones(1, device=device, dtype=torch.float32, requires_grad=autograd)
        result = scan_fct(
            get_scan_combine_fn("div", False),
            init,
            x,
            dim=0,
            reverse=reverse,
        )
        result_exp = _fake_scan(
            get_scan_combine_fn("div", False),
            init=init,
            xs=x,
            dim=0,
            reverse=reverse,
        )
        self.assertEqual(result, result_exp)

        if autograd:
            self.check_autograd(result, result_exp, (init, x))

    # TODO: provide an implementation for all compile modes and re-enable all test
    @skipIfTorchDynamo("don't test compile on compile")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize(
        "dtype",
        [
            torch.float16,
            torch.float32,
            torch.int32,
            torch.int64,
            torch.complex64,
        ],
    )
    def test_scan_dtype(self, reverse, compile_mode, device, dtype):
        scan_fct = compile_mode_helper(scan, compile_mode)

        # Check all outputs and carries on the correct device and with torch.float32
        x = torch.randn(3, 10, 2, device=device).to(dtype=dtype)
        op, init = (
            get_scan_combine_fn("adds"),
            torch.zeros(10, 2, device=device, dtype=dtype),
        )
        result = scan_fct(op, init, x, dim=0, reverse=reverse)
        result_exp = _fake_scan(op, init=init, xs=x, dim=0, reverse=reverse)
        self.assertEqual(result, result_exp)
        self.assertEqual(
            [[r.device.type for r in res] for res in result],
            [[device.type for _ in res] for res in result],
        )
        self.assertEqual(
            [[r.dtype for r in res] for res in result],
            [[dtype for _ in res] for res in result],
        )

        # Check all outputs and carries on the correct device and
        # carry.dtype torch.float32 and output.dtype torch.float16
        x = torch.randn(3, 10, 2, device=device).to(dtype=dtype)
        op, init = (
            get_scan_combine_fn("adds"),
            torch.zeros(10, 2, device=device, dtype=torch.float32),
        )
        result = scan_fct(op, init, x, dim=0, reverse=reverse)
        result_exp = _fake_scan(op, init=init, xs=x, dim=0, reverse=reverse)
        self.assertEqual(result, result_exp)
        self.assertEqual(
            [[r.dtype for r in res] for res in result],
            [
                [torch.float32 for _ in range(len(result[0]))],
                [dtype for _ in range(len(result[1]))],
            ],
        )

        # Check all outputs and carries on the correct device and
        # carry.dtype torch.int64 and output.dtype torch.float32
        x = torch.randn(3, 10, 2, device=device)
        op, init = (
            get_scan_combine_fn("adds"),
            torch.zeros(10, 2, device=device, dtype=dtype),
        )
        result = scan_fct(op, init, x, dim=0, reverse=reverse)
        result_exp = _fake_scan(op, init=init, xs=x, dim=0, reverse=reverse)
        self.assertEqual(result, result_exp)
        self.assertEqual(
            [[r.dtype for r in res] for res in result],
            [
                [dtype for _ in range(len(result[0]))],
                [torch.float32 for _ in range(len(result[1]))],
            ],
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_dim(self, reverse, compile_mode, device, autograd):
        import random

        scan_fct = compile_mode_helper(scan, compile_mode)

        num_dims = [random.randint(2, 5) for _ in range(5)]
        for num_dim in num_dims:
            shapes = [random.randint(1, 10) for _ in range(num_dim)]
            rnd_scan_dim = random.randint(0, num_dim - 1)
            x = torch.randn(*shapes, device=device, requires_grad=autograd)
            init_shapes = shapes[:rnd_scan_dim] + shapes[rnd_scan_dim + 1 :]

            for op, op_pt, init in [
                (
                    get_scan_combine_fn("add", False),
                    torch.cumsum,
                    torch.zeros(*init_shapes, device=device, requires_grad=autograd),
                ),
                (
                    get_scan_combine_fn("mul", False),
                    torch.cumprod,
                    torch.ones(*init_shapes, device=device, requires_grad=autograd),
                ),
            ]:
                result = scan_fct(op, init, x, dim=rnd_scan_dim, reverse=reverse)
                result_exp = _fake_scan(
                    op, init=init, xs=x, dim=rnd_scan_dim, reverse=reverse
                )
                self.assertEqual(result, result_exp)
                if not reverse:
                    result_exp_PT = op_pt(x, rnd_scan_dim)
                    res_list = list(result)
                    res_list[1] = res_list[1].movedim(0, rnd_scan_dim)
                    self.assertEqual(res_list[1], result_exp_PT)

                if autograd:
                    self.check_autograd(result, result_exp, (init, x))

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_binary_operator(self, reverse, compile_mode, device, autograd):
        state_dim = 20
        timesteps = 10
        scan_fct = compile_mode_helper(scan, compile_mode)

        projected_inputs = torch.randn(
            timesteps, state_dim, requires_grad=autograd, device=device
        )
        A = torch.randn(state_dim, requires_grad=autograd, device=device)
        elements = (A.repeat((timesteps, 1)), projected_inputs)
        init = tuple(
            [
                torch.ones_like(
                    torch._ops.ops.aten.slice(elements[0], 0, 0, 1, 1),
                    requires_grad=autograd,
                )
            ]
            + [
                torch.zeros_like(
                    torch._ops.ops.aten.slice(projected_inputs, 0, 0, 1, 1),
                    requires_grad=autograd,
                )
            ]
        )

        init_clone = [i.clone() for i in init]
        init_clone2 = [i.clone() for i in init]
        elements_clone = [ele.clone() for ele in elements]
        elements_clone2 = [ele.clone() for ele in elements]
        result = scan_fct(
            get_scan_combine_fn("s5_operator", False),
            init_clone,
            elements_clone,
            reverse=reverse,
        )
        expected_result = _fake_scan(
            get_scan_combine_fn("s5_operator", False),
            init_clone2,
            elements_clone2,
            reverse=reverse,
        )
        self.assertEqual(result, expected_result)

        if autograd:
            result_flatten, _ = pytree.tree_flatten(result)
            result_exp_flatten, _ = pytree.tree_flatten(expected_result)

            grad_out = [torch.ones_like(el) for el in result_exp_flatten]
            expected_grads = torch.autograd.grad(
                result_exp_flatten, (*init_clone2, *elements_clone2), grad_out
            )
            grads = torch.autograd.grad(
                result_flatten, (*init_clone, *elements_clone), grad_out
            )
            self.assertEqual(grads, expected_grads)

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_tuple(self, reverse, compile_mode, device, autograd):
        x = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        y = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        inp = (x, y)
        init = tuple(torch._ops.ops.aten.slice(e, 0, 0, 1, 1) for e in inp)

        scan_fct = compile_mode_helper(scan, compile_mode)

        result_same = scan_fct(
            get_scan_combine_fn("tuple_fct", False),
            init,
            inp,
            dim=0,
            reverse=reverse,
        )
        expected_result = _fake_scan(
            get_scan_combine_fn("tuple_fct", False),
            init=init,
            xs=inp,
            dim=0,
            reverse=reverse,
        )
        self.assertEqual(result_same, expected_result)

        if autograd:
            self.check_autograd(result_same, expected_result, (init, inp))

        def fct_different_output_tuple(x, y):
            return ((x[0] + y[0], x[1] * y[1]), (x[1] * y[1]))

        inp = (x, y)
        init = tuple(torch._ops.ops.aten.slice(e, 0, 0, 1, 1) for e in inp)

        result_diff = scan(
            fct_different_output_tuple, init, inp, dim=0, reverse=reverse
        )
        expected_result = _fake_scan(
            fct_different_output_tuple, init=init, xs=inp, dim=0, reverse=reverse
        )
        self.assertEqual(result_diff, expected_result)
        self.assertEqual(result_diff[1], result_same[1][1])

        if autograd:
            self.check_autograd(result_diff, expected_result, (init, inp))

    def test_scan_wrong_pytree(self):
        # Init and input have same pytree
        def fct_wrong_pytree(x, y):
            return (
                {
                    "i": x["i"] * y["j"][0][0],
                    "k": torch.tensor(0.0),
                    "j": (
                        [x["j"][1][0]["o"].clone()],
                        [{"o": torch.sin(x["i"])}],
                    ),
                },
                {
                    "i": x["i"] * y["j"][0][0],
                    "k": torch.tensor(0.0),
                    "j": ([x["j"][1][0]["o"].clone()], [{"o": torch.sin(x["i"])}]),
                },
            )

        x = torch.randn(3, 2, 2)
        y = torch.randn(3, 2, 2)
        z = torch.randn(3, 2, 2)
        inp = {"i": x, "j": ([y], [{"o": z}])}
        inp_flat, inp_spec = pytree.tree_flatten(inp)
        init_flat = [torch._ops.ops.aten.slice(e, 0, 0, 1, 1) for e in inp_flat]
        init = pytree.tree_unflatten(init_flat, inp_spec)

        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.UncapturedHigherOrderOpError,
            # r"The tree structure of the inits and the carries are not identical.*",
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "Expected init and carry to have same number of outputs but got lhs.*",
        ):
            scan(fct_wrong_pytree, init, inp, dim=0)

    def test_scan_float_output(self):
        # Init and input have same pytree
        def fct_float_output(x, y):
            return 0.0, x + y

        x = torch.randn(3, 2, 2)
        init = torch._ops.ops.aten.slice(x, 0, 0, 1, 1)

        with self.assertRaisesRegex(
            # Should be:
            # torch._dynamo.exc.Unsupported,
            # "HigherOrderOperator body's output must consist of tensors or ints only"
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.ops\.higher_order\.scan",
        ):
            scan(fct_float_output, init, x, dim=0)

    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_complex_pytree(self, reverse, compile_mode, device, autograd):
        # Init and input have same pytree

        scan_fct = compile_mode_helper(scan, compile_mode)

        x = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        y = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        z = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        inp = {"i": x, "j": ([y], [{"o": z}])}
        inp_flat, inp_spec = pytree.tree_flatten(inp)
        init_flat = [torch._ops.ops.aten.slice(e, 0, 0, 1, 1) for e in inp_flat]
        init = pytree.tree_unflatten(init_flat, inp_spec)

        result = scan_fct(
            get_scan_combine_fn("complex_pointwise", False),
            init,
            inp,
            dim=0,
            reverse=reverse,
        )
        expected_result = _fake_scan(
            get_scan_combine_fn("complex_pointwise", False),
            init=init,
            xs=inp,
            dim=0,
            reverse=reverse,
        )
        self.assertEqual(result, expected_result)

        if autograd:
            self.check_autograd(result, expected_result, (init, inp))

    # TODO: Does not work because of the usage of vmap within associative_scan
    # The paT206899919 rameterization is commented out for the moment and the test is marked with expected fail
    # Fails with: AssertionError: scan is not an OpOverload
    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    def test_scan_associative_scan(self):
        combine_mode = "generic"
        compile_mode_scan = "compile"
        compile_mode_associative_scan = "none"
        reverse = True
        reverse_associative_scan = True
        device = torch.device("cuda")

        scan_fct = compile_mode_helper(scan, compile_mode_scan)
        associative_scan_fct = compile_mode_helper(
            associative_scan, compile_mode_associative_scan
        )
        init = torch.randn(10, 5, device=device)
        inp = torch.randn(3, 10, 5, device=device)

        def body(x, y):
            val = associative_scan_fct(
                get_scan_combine_fn("add", True),
                y,
                0,
                reverse=reverse_associative_scan,
                combine_mode=combine_mode,
            )
            return x + y, x + val

        result = scan_fct(body, init, inp, dim=0, reverse=reverse)
        expected_result = _fake_scan(
            body,
            init,
            inp,
            0,
            reverse=reverse,
        )

        self.assertEqual(result, expected_result)

    # TODO: provide an implementation for all compile modes and re-enable all test
    @skipIfTorchDynamo("don't test compile on compile")
    @requires_cuda
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_downstream_scan_matmul(self, compile_mode, reverse, device, autograd):
        inp = torch.randn(3, 10, 2, device=device, requires_grad=autograd)
        init = torch.randn(3, 2, device=device, requires_grad=autograd)

        for ind in range(2):
            # Chain with matmul
            def chain_fct(inp):
                W = torch.ones(2, 5, device=device)
                o = scan(
                    get_scan_combine_fn("add", False),
                    init,
                    inp,
                    dim=1,
                    reverse=reverse,
                )
                return o[ind] @ W

            fct_cmp = compile_mode_helper(chain_fct, compile_mode)

            expected_result = _fake_scan(
                get_scan_combine_fn("add", False),
                init=init,
                xs=inp,
                dim=1,
                reverse=reverse,
            )[ind] @ torch.ones(2, 5, device=device)
            result = fct_cmp(inp)
            self.assertEqual(result, expected_result)

            if autograd:
                self.check_autograd(result, expected_result, (init, inp))

    # TODO: provide an implementation for all compile modes and re-enable all test
    @skipIfTorchDynamo("don't test compile on compile")
    @requires_cuda
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_downstream_scan_scan_dim(
        self, compile_mode, reverse, device, autograd
    ):
        inp = torch.randn(3, 10, 2, device=device, requires_grad=autograd)
        init = torch.randn(3, 2, device=device, requires_grad=autograd)

        # Chain with scan on different dim
        init2 = torch.randn(1, 10, 2, device=device, requires_grad=autograd)

        def chain_fct_different_dim(inp):
            o1 = scan(
                get_scan_combine_fn("add", False),
                init,
                inp,
                dim=1,
                reverse=reverse,
            )
            o1 = pytree.tree_map(lambda t: t.movedim(0, 1), o1)
            o2 = scan(
                get_scan_combine_fn("add", False),
                init2,
                o1[1],
                dim=0,
                reverse=reverse,
            )
            return o2

        fct_cmp = compile_mode_helper(chain_fct_different_dim, compile_mode)

        xs = _fake_scan(
            get_scan_combine_fn("add", False),
            init=init,
            xs=inp,
            dim=1,
            reverse=reverse,
        )[1]
        xs = pytree.tree_map(lambda t: t.movedim(0, 1), xs)
        expected_result = _fake_scan(
            get_scan_combine_fn("add", False),
            init=init2,
            xs=xs,
            dim=0,
            reverse=reverse,
        )
        result = fct_cmp(inp)
        self.assertEqual(result, expected_result)

        if autograd:
            self.check_autograd(result, expected_result, (init, init2, inp))

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_non_pointwise(self, reverse, compile_mode, device, autograd):
        scan_fct = compile_mode_helper(scan, compile_mode)

        x = torch.randn(3, 10, 2, device=device, requires_grad=autograd)
        init = torch.randn(10, 2, device=device, requires_grad=autograd)
        expected_result = _fake_scan(
            get_scan_combine_fn("non_pointwise", False),
            init=init,
            xs=x,
            dim=0,
            reverse=reverse,
        )

        result = scan_fct(
            get_scan_combine_fn("non_pointwise", False),
            init,
            x,
            dim=0,
            reverse=reverse,
        )
        self.assertEqual(result, expected_result)

        if autograd:
            self.check_autograd(result, expected_result, (init, x))

    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    def test_scan_compile_cnt(self, reverse, device):
        dim = 1

        from torch._dynamo.testing import CompileCounter

        # Tests rely on automatic_dynamic = True
        with torch._dynamo.config.patch(automatic_dynamic_shapes=True):
            cnt = CompileCounter()
            x = torch.randn(3, 2, 5, device=device)
            init = torch.randn(3, 5, device=device)
            # First compilation step
            torch.compile(scan, backend=cnt)(
                get_scan_combine_fn("add", False),
                init,
                x,
                dim=dim,
                reverse=reverse,
            )
            self.assertEqual(cnt.frame_count, 1)

            x = torch.randn(3, 20, 5, device=device)
            init = torch.randn(3, 5, device=device)
            # Recompilation due to first different size
            torch.compile(scan, backend=cnt)(
                get_scan_combine_fn("add", False),
                init,
                x,
                dim=dim,
                reverse=reverse,
            )
            self.assertEqual(cnt.frame_count, 2)

            x = torch.randn(3, 40, 5, device=device)
            init = torch.randn(3, 5, device=device)
            # No recompilation, because of dynamic shape
            torch.compile(scan, backend=cnt)(
                get_scan_combine_fn("add", False),
                init,
                x,
                dim=dim,
                reverse=reverse,
            )
            self.assertEqual(cnt.frame_count, 2)

            x = torch.randn(3, 40, 5, device=device)
            init = torch.randn(3, 40, device=device)
            # Recompilation because of dim change
            torch.compile(scan, backend=cnt)(
                get_scan_combine_fn("add", False),
                init,
                x,
                dim=2,
                reverse=reverse,
            )
            self.assertEqual(cnt.frame_count, 3)

            x = torch.randn(3, 40, 20, device=device)
            init = torch.randn(3, 40, device=device)
            # Recompilation due to first different size on new dim
            torch.compile(scan, backend=cnt)(
                get_scan_combine_fn("add", False),
                init,
                x,
                dim=2,
                reverse=reverse,
            )
            self.assertEqual(cnt.frame_count, 4)

            x = torch.randn(3, 40, 40, device=device)
            init = torch.randn(3, 40, device=device)
            # No recompilation, because of dynamic shape on new dim
            torch.compile(scan, backend=cnt)(
                get_scan_combine_fn("add", False),
                init,
                x,
                dim=2,
                reverse=reverse,
            )
            self.assertEqual(cnt.frame_count, 4)

            x = torch.randn(3, 60, 40, device=device)
            init = torch.randn(3, 40, device=device)
            # Recompilation because of dim change
            torch.compile(scan, backend=cnt)(
                get_scan_combine_fn("add", False),
                init,
                x,
                dim=1,
                reverse=reverse,
            )
            self.assertEqual(cnt.frame_count, 5)

            x = torch.randn(3, 60, 40, device=device)
            init = torch.randn(3, 40, device=device)
            # Recompilation because of reverse change
            torch.compile(scan, backend=cnt)(
                get_scan_combine_fn("add", False),
                init,
                x,
                dim=1,
                reverse=not reverse,
            )
            self.assertEqual(cnt.frame_count, 6)

            x = torch.randn(3, 60, 40, device=device)
            init = torch.randn(3, 40, device=device)
            # No recompilation, as nothing changed
            torch.compile(scan, backend=cnt)(
                get_scan_combine_fn("add", False),
                init,
                x,
                dim=1,
                reverse=not reverse,
            )
            self.assertEqual(cnt.frame_count, 6)

            x = torch.randn(3, 120, 80, device=device)
            init = torch.randn(3, 80, device=device)
            # No recompilation, final test
            torch.compile(scan, backend=cnt)(
                get_scan_combine_fn("add", False),
                init,
                x,
                dim=1,
                reverse=reverse,
            )
            self.assertEqual(cnt.frame_count, 6)

    @skipIfTorchDynamo("don't test compile on compile")
    def test_scan_init_scanned_0(self):
        # Only init and no input
        x = torch.randn(3, 1, 2, device=torch.device("cpu"))
        init = torch.randn(3, 2, device=torch.device("cpu"))
        dim = 1

        # Scan dimension is 0
        init = torch._ops.ops.aten.slice(x, dim, 0, 1, 1)
        inp = torch._ops.ops.aten.slice(x, dim, 1, None, 1)
        with self.assertRaisesRegex(
            RuntimeError,
            "All xs leaves must at least have.*",
        ):
            scan(
                get_scan_combine_fn("add", False),
                init,
                inp,
                dim=dim,
            )

    @skipIfTorchDynamo("don't test compile on compile")
    def test_scan_init_non_tensor(self):
        x = torch.randn(3, 1, 2, device=torch.device("cpu"))
        dim = 1

        # Init is a float and not a tensor
        init = 1.0
        with self.assertRaisesRegex(RuntimeError, "All init leaves must be a Tensor.*"):
            scan(get_scan_combine_fn("add", False), init, x, dim=dim, reverse=False)

    @skipIfTorchDynamo("don't test compile on compile")
    def test_scan_init_wrong_shape(self):
        scan_fct = compile_mode_helper(scan, "none")

        # Only init and no input
        x = torch.randn(3, 1, 2)
        dim = 1

        # Init wrong shape (Other dim different)
        init = torch.randn(1, 2)
        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "Expected init and carry to have same metadata.*",
        ):
            scan_fct(
                get_scan_combine_fn("add", False),
                init,
                x,
                dim=dim,
            )

    @skipIfTorchDynamo("don't test compile on compile")
    def test_scan_init_wrong_pytree_init_longer_carry(self):
        def init_longer_carry(x: torch.Tensor, y: torch.Tensor):
            return x[0] + 1.0, y + 1.0

        scan_fct = compile_mode_helper(scan, "none")

        # Only init and no input
        x = torch.randn(3, 1, 2)
        dim = 1

        # Init wrong pytree
        init = (
            torch._ops.ops.aten.slice(x, dim, 0, 1, 1),
            torch._ops.ops.aten.slice(x, dim, 0, 1, 1),
        )

        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "Expected init and carry to have same number of outputs but got lhs.*",
        ):
            scan_fct(init_longer_carry, init, x, dim=dim)

    @skipIfTorchDynamo("don't test compile on compile")
    def test_scan_init_wrong_pytree_init_shorter_carry(self):
        def init_shorter_carry(x: torch.Tensor, y: torch.Tensor):
            return (x + 1, x + 2), x + 3

        scan_fct = compile_mode_helper(scan, "none")

        # Only init and no input
        x = torch.randn(3, 1, 2)
        dim = 1

        # Init wrong pytree
        init = torch._ops.ops.aten.slice(x, dim, 0, 1, 1)

        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.Unsupported,
            # The tree structure of the inits and the carries are not identical!
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "Expected init and carry to have same number of outputs but got lhs.*",
        ):
            scan_fct(init_shorter_carry, init, x, dim=dim)

    @skipIfTorchDynamo("don't test compile on compile")
    def test_scan_init_wrong_pytree_carry_shape(self):
        def wrong_carry_shape(x: torch.Tensor, y: torch.Tensor):
            return x[0, :], x + 3

        scan_fct = compile_mode_helper(scan, "none")

        # Only init and no input
        x = torch.randn(3, 1, 2)
        dim = 1

        # Init wrong pytree
        init = torch._ops.ops.aten.slice(x, dim, 0, 1, 1)

        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.Unsupported,
            # "Encountered aliasing during higher order op tracing for HOP.*"
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.ops\.higher_order\.scan",
        ):
            scan_fct(wrong_carry_shape, init, x, dim=dim)

    @skipIfTorchDynamo("don't test compile on compile")
    def test_scan_one_return(self):
        def no_carry(x: torch.Tensor, y: torch.Tensor):
            return x + 3

        scan_fct = compile_mode_helper(scan, "none")

        # Only init and no input
        x = torch.randn(3, 1, 2)
        dim = 1

        # Init wrong pytree
        init = torch._ops.ops.aten.slice(x, dim, 0, 1, 1)

        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.Unsupported,
            # combine_fn needs to produce two pytrees, one for the carries and one for the outputs.
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.ops\.higher_order\.scan",
        ):
            scan_fct(no_carry, init, x, dim=dim)

    @skipIfTorchDynamo("don't test compile on compile")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_init(self, reverse, compile_mode, device, autograd):
        scan_fct = compile_mode_helper(scan, compile_mode)

        # Only init and no input
        x = torch.randn(3, 1, 2, device=device, requires_grad=autograd)
        dim = 1
        op, op_pt = (get_scan_combine_fn("add", False), torch.cumsum)

        # Only init given
        init = torch._ops.ops.aten.slice(x, dim, 0, 1, 1)
        result = scan_fct(op, init, [], dim=dim, reverse=reverse)
        result_exp = _fake_scan(op, init=init, xs=[], dim=dim, reverse=reverse)
        result_init = scan_fct(op, init, [], dim=dim, reverse=reverse)
        self.assertEqual(result, result_exp)
        self.assertEqual(result_init, result_exp)
        self.assertEqual(result_init[0], init)

        if autograd:
            self.check_autograd(result, result_exp, (init,))

        x = torch.randn(3, 5, 2, device=device, requires_grad=autograd)
        dim = 0

        op, op_pt = (get_scan_combine_fn("add", False), torch.cumsum)
        inp = torch._ops.ops.aten.slice(x, dim, 1, None, 1)

        # Init tensor scalar
        init = torch.ones(1, device=device, requires_grad=autograd)

        def add_scalar_carry(x: torch.Tensor, y: torch.Tensor):
            return x + 1.0, x + y

        result_init = scan_fct(add_scalar_carry, init, inp, dim=dim, reverse=reverse)
        result_exp = _fake_scan(
            add_scalar_carry, init=init, xs=inp, dim=dim, reverse=reverse
        )
        self.assertEqual(result_init, result_exp)
        self.assertEqual(result_init[0], torch.tensor([3.0], device=device))

        if autograd:
            self.check_autograd(result_init, result_exp, (init, inp))

        # Init tensor entirely different shape than inp
        init = torch.randn(7, 8, device=device, requires_grad=autograd)

        def add_scalar_carry2(x: torch.Tensor, y: torch.Tensor):
            return x + 1.0, x[: y.shape[0], : y.shape[1]] + y

        result_init = scan_fct(add_scalar_carry2, init, inp, dim=dim, reverse=reverse)
        result_exp = _fake_scan(
            add_scalar_carry2, init=init, xs=inp, dim=dim, reverse=reverse
        )
        self.assertEqual(result_init, result_exp)

        # Init with two timestep on dim axis. Should work as y has always 1 on dim axis and
        # hence automatic broadcasting should work
        # I.e., the input shape is 2x5x2, but the carry at each iteration is 2x5x2,
        # thus the output of each iteration is 2x5x2, which results in the total output
        # to be 4x5x2
        init = torch._ops.ops.aten.slice(x, dim, 0, 2, 1)
        result_init = scan_fct(op, init, inp, dim=dim, reverse=reverse)
        result_exp = _fake_scan(op, init=init, xs=inp, dim=dim, reverse=reverse)
        self.assertEqual(result_init, result_exp)
        self.assertEqual(result_init[0].shape, torch.Size([2, 5, 2]))

        if autograd:
            self.check_autograd(result_init, result_exp, (init, inp))

        init = torch.tile(init, (1, 2, 1))

        def add_scalar_carry_sliced_out(x: torch.Tensor, y: torch.Tensor):
            return x + 1.0, x[:, :1, :] + y

        result_init = scan_fct(
            add_scalar_carry_sliced_out, init, inp, dim=dim, reverse=reverse
        )
        result_exp = _fake_scan(
            add_scalar_carry_sliced_out, init=init, xs=inp, dim=dim, reverse=reverse
        )
        self.assertEqual(result_init, result_exp)
        self.assertEqual(result_init[0].shape, torch.Size([2, 10, 2]))
        self.assertEqual(result_init[1].shape, torch.Size([2, 2, 5, 2]))

        if autograd:
            self.check_autograd(result_init, result_exp, (init, inp))

        # Correct case
        op, op_pt = (get_scan_combine_fn("add", False), torch.cumsum)
        x = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        init = torch.zeros(3, 2, device=device, requires_grad=autograd)
        dim = 2

        result = scan_fct(op, init, x, dim=dim, reverse=reverse)
        result_exp = _fake_scan(op, init=init, xs=x, dim=dim, reverse=reverse)

        self.assertEqual(result, result_exp)
        if not reverse:
            result_exp_PT = op_pt(x, dim)
            result = list(result)
            result[1] = pytree.tree_map(lambda t: torch.movedim(t, 0, dim), result[1])
            self.assertEqual(result[1], result_exp_PT)

        if autograd:
            self.check_autograd(result, result_exp, (init, x))

    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    def test_scan_init_wrong_pytree_complex(self, reverse, device):
        x = torch.randn(3, 2, 2, device=device)
        y = torch.randn(3, 2, 2, device=device)
        z = torch.randn(3, 2, 2, device=device)

        # Wrong pytree fed to the function
        init = {
            "i": torch._ops.ops.aten.slice(x, 0, 0, 1, 1),
            "j": (
                {"a": torch._ops.ops.aten.slice(x, 0, 0, 1, 1)},
                [torch._ops.ops.aten.slice(y, 0, 0, 1, 1)],
                [{"o": torch._ops.ops.aten.slice(z, 0, 0, 1, 1)}],
            ),
        }
        inp = {
            "i": torch._ops.ops.aten.slice(x, 0, 0, None, 1),
            "j": (
                [torch._ops.ops.aten.slice(y, 0, 0, None, 1)],
                [{"o": torch._ops.ops.aten.slice(z, 0, 0, None, 1)}],
            ),
        }
        with self.assertRaisesRegex(
            Exception,
            ".*",
        ):
            scan(
                get_scan_combine_fn("complex_pointwise", False),
                init,
                inp,
                dim=0,
                reverse=reverse,
            )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_init_pytree_complex(self, reverse, compile_mode, device, autograd):
        def fct_pointwise_different_output(x, y):
            return (
                {
                    "i": x["i"] * y["i"],
                    "j": (
                        [x["j"][0][0] * y["j"][0][0]],
                        [{"o": x["j"][1][0]["o"] + y["j"][1][0]["o"]}],
                    ),
                },
                (
                    y["i"] * 2,
                    {
                        "o": x["i"] * y["i"],
                        "j": (
                            [x["j"][0][0] * y["j"][0][0]],
                            [{"o": x["j"][1][0]["o"] + y["j"][1][0]["o"]}],
                        ),
                    },
                ),
            )

        def fct_pointwise_different_carry(x, y):
            return (
                {
                    "i": x["i"] * y["i"],
                    "j": (
                        x["i"] * 2,
                        [x["j"][1][0] * y["j"][0][0]],
                        [{"o": x["j"][2][0]["o"] + y["j"][1][0]["o"]}],
                    ),
                },
                (
                    y["i"] * 2,
                    {
                        "o": x["i"] * y["i"] + x["j"][0][0],
                        "j": (
                            [x["j"][1][0] * y["j"][0][0]],
                            [{"o": x["j"][2][0]["o"] + y["j"][1][0]["o"]}],
                        ),
                    },
                ),
            )

        scan_fct = compile_mode_helper(scan, compile_mode)

        x = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        y = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        z = torch.randn(3, 2, 2, device=device, requires_grad=autograd)

        if reverse:
            init_start, init_end = -1, None
            inp_start, inp_end = 0, -1
        else:
            init_start, init_end = 0, 1
            inp_start, inp_end = 1, None

        # Regular case
        init = {
            "i": torch._ops.ops.aten.slice(x, 0, init_start, init_end, 1),
            "j": (
                [torch._ops.ops.aten.slice(y, 0, init_start, init_end, 1)],
                [{"o": torch._ops.ops.aten.slice(z, 0, init_start, init_end, 1)}],
            ),
        }
        inp = {
            "i": torch._ops.ops.aten.slice(x, 0, inp_start, inp_end, 1),
            "j": (
                [torch._ops.ops.aten.slice(y, 0, inp_start, inp_end, 1)],
                [{"o": torch._ops.ops.aten.slice(z, 0, inp_start, inp_end, 1)}],
            ),
        }
        result = scan_fct(
            get_scan_combine_fn("complex_pointwise", False),
            init,
            inp,
            dim=0,
            reverse=reverse,
        )
        expected_result = _fake_scan(
            get_scan_combine_fn("complex_pointwise", False),
            init,
            inp,
            dim=0,
            reverse=reverse,
        )
        self.assertEqual(result, expected_result)

        if autograd:
            init_flat = pytree.tree_leaves(init)
            inp_flat = pytree.tree_leaves(inp)
            self.check_autograd(result, expected_result, (*init_flat, *inp_flat))

        # Pytree of output is different
        result = scan_fct(
            fct_pointwise_different_output, init, inp, dim=0, reverse=reverse
        )
        expected_result = _fake_scan(
            fct_pointwise_different_output, init=init, xs=inp, dim=0, reverse=reverse
        )
        self.assertEqual(result, expected_result)

        # Pytree of carry is different
        init = {
            "i": torch._ops.ops.aten.slice(x, 0, init_start, init_end, 1),
            "j": (
                torch._ops.ops.aten.slice(x, 0, init_start, init_end, 1),
                [torch._ops.ops.aten.slice(y, 0, init_start, init_end, 1)],
                [{"o": torch._ops.ops.aten.slice(z, 0, init_start, init_end, 1)}],
            ),
        }
        inp = {
            "i": torch._ops.ops.aten.slice(x, 0, inp_start, inp_end, 1),
            "j": (
                [torch._ops.ops.aten.slice(y, 0, inp_start, inp_end, 1)],
                [{"o": torch._ops.ops.aten.slice(z, 0, inp_start, inp_end, 1)}],
            ),
        }
        result = scan_fct(
            fct_pointwise_different_carry, init, inp, dim=0, reverse=reverse
        )
        expected_result = _fake_scan(
            fct_pointwise_different_carry, init=init, xs=inp, dim=0, reverse=reverse
        )
        self.assertEqual(result, expected_result)

        if autograd:
            init_flat = pytree.tree_leaves(init)
            inp_flat = pytree.tree_leaves(inp)
            self.check_autograd(result, expected_result, (*init_flat, *inp_flat))

    @skipIfTorchDynamo("don't test compile on compile")
    @skipIfNoDynamoSupport
    @skipIfCrossRef  # Arg order changes with crossref
    def test_scan_pytree_output(self):
        x = torch.randn(3, 10, 2, device=torch.device("cpu"))
        init = torch.randn(1, 10, 2, device=torch.device("cpu"))

        def f(fct, init, xs):
            return scan(fct, init, xs, dim=0, reverse=True)

        def combine_fn(init, x):
            a, b = (init[0] + x, init[1] - x)
            return (a, b), a - b

        # Check graph
        backend = EagerAndRecordGraphs()
        torch.compile(f, backend=backend)(combine_fn, (init, init.clone()), x)
        gm = backend.graphs[0]

        self.assertExpectedInline(
            normalize_gm(gm.print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_init_0_: "f32[1, 10, 2]", L_init_1_: "f32[1, 10, 2]", L_xs_: "f32[3, 10, 2]"):
        l_init_0_ = L_init_0_
        l_init_1_ = L_init_1_
        l_xs_ = L_xs_

        flip: "f32[3, 10, 2]" = torch.flip(l_xs_, [0]);  l_xs_ = None
        scan_combine_fn_0 = self.scan_combine_fn_0
        scan = torch.ops.higher_order.scan(scan_combine_fn_0, [l_init_0_, l_init_1_], [flip], []);  scan_combine_fn_0 = l_init_0_ = l_init_1_ = flip = None
        getitem: "f32[1, 10, 2]" = scan[0]
        getitem_1: "f32[1, 10, 2]" = scan[1]
        out: "f32[3, 1, 10, 2]" = scan[2];  scan = None
        out_1: "f32[3, 1, 10, 2]" = out.flip([0]);  out = None
        return (getitem, getitem_1, out_1)

    class scan_combine_fn_0(torch.nn.Module):
        def forward(self, child: "f32[1, 10, 2]", child_1: "f32[1, 10, 2]", child_2: "f32[10, 2]"):
            a: "f32[1, 10, 2]" = child + child_2;  child = None
            b: "f32[1, 10, 2]" = child_1 - child_2;  child_1 = child_2 = None

            child_3: "f32[1, 10, 2]" = a - b
            return [a, b, child_3]
""",  # noqa: B950
        )

    @skipIfTorchDynamo("Graph is not captured by backend if test with dynamo")
    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("autograd", [False, True])
    def test_scan_closure_RNN(self, compile_mode, autograd):
        dim = 1
        device = torch.device("cpu")
        scan_fct = compile_mode_helper(scan, compile_mode)

        rnn = torch.nn.RNN(
            input_size=5,
            hidden_size=7,
            batch_first=True,
        )
        rnn = rnn.to(device=device)
        x = torch.randn(3, 10, 5, device=device, requires_grad=autograd)
        h = torch.randn(3, 7, device=device, requires_grad=autograd)

        W_ih = rnn.weight_ih_l0.T.clone()
        b_ih = rnn.bias_ih_l0.clone()
        W_hh = rnn.weight_hh_l0.T.clone()
        b_hh = rnn.bias_hh_l0.clone()

        if not autograd:
            W_ih = W_ih.detach()
            b_ih = b_ih.detach()
            W_hh = W_hh.detach()
            b_hh = b_hh.detach()

        expected_result = rnn(x, torch.unsqueeze(h, 0))
        expected_result_out = expected_result[0]
        expected_result_state = expected_result[1][0, :]

        result = scan_fct(
            get_scan_combine_fn("RNN", True, parameters=[W_ih, b_ih, W_hh, b_hh]),
            h,
            x,
            dim=dim,
            reverse=False,
        )
        result_cmp = [result[0], torch.movedim(result[1], 0, dim)]
        self.assertEqual(result_cmp[0], expected_result_state)
        self.assertEqual(result_cmp[1], expected_result_out)

        if autograd:
            result_flat = pytree.tree_leaves(result)
            result_exp_flat = [expected_result_state, expected_result_out]

            grad_out_expected = [torch.ones_like(r) for r in result_exp_flat]
            expected_grads = torch.autograd.grad(
                result_exp_flat,
                (
                    h,
                    x,
                    rnn.weight_ih_l0,
                    rnn.bias_ih_l0,
                    rnn.weight_hh_l0,
                    rnn.bias_hh_l0,
                ),
                grad_out_expected,
            )
            expected_add_input_grads = list(expected_grads[2:])
            expected_grads = expected_grads[:2]

            grad_out = [torch.ones_like(r) for r in result]
            grads = torch.autograd.grad(
                result_flat, (h, x, W_ih, b_ih, W_hh, b_hh), grad_out
            )
            add_input_grads = list(grads[2:])
            add_input_grads[0] = add_input_grads[0].T
            add_input_grads[2] = add_input_grads[2].T
            grads = grads[:2]
            self.assertEqual(grads, expected_grads)
            self.assertEqual(add_input_grads, expected_add_input_grads)

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize(
        "partial_grad", ["xs", "init", "additional_inputs", "complex", "random"]
    )
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    def test_scan_closure_RNN_partial_autograd(
        self, reverse, compile_mode, partial_grad, device
    ):
        dim = 1
        scan_fct = compile_mode_helper(scan, compile_mode)

        # The first two booleans are the xs
        # The second two are the inits
        # The last four are the additional_inputs
        autograds = []

        if partial_grad == "xs":
            # xs tests
            autograds.append([True, False, True, True, True, True, True, True])
            autograds.append([False, False, True, True, True, True, True, True])
        elif partial_grad == "init":
            # init tests
            autograds.append([True, True, False, True, True, True, True, True])
            autograds.append([True, True, False, False, True, True, True, True])
        elif partial_grad == "additional_inputs":
            # additional input tests
            autograds.append([True, True, True, True, False, True, False, True])
            autograds.append([True, True, True, True, False, False, False, False])
        elif partial_grad == "complex":
            # complex cases
            autograds.append([True, False, False, False, False, False, False, True])
            autograds.append([False, False, True, True, False, False, False, True])
        elif partial_grad == "random":
            # random tests
            import random

            for _ in range(5):
                autograds.append([bool(random.randint(0, 1)) for _ in range(8)])

        for autograd in autograds:
            x = torch.randn(3, 10, 5, device=device, requires_grad=autograd[0])
            x1 = torch.randn(3, 10, 5, device=device, requires_grad=autograd[1])
            h = torch.randn(3, 7, device=device, requires_grad=autograd[2])
            h_1 = torch.randn(3, 7, device=device, requires_grad=autograd[3])
            W_ih = torch.randn(5, 7, device=device, requires_grad=autograd[4])
            b_ih = torch.randn(7, device=device, requires_grad=autograd[5])
            W_hh = torch.randn(7, 7, device=device, requires_grad=autograd[6])
            b_hh = torch.randn(7, device=device, requires_grad=autograd[7])

            params = [
                p
                for p, a in zip([x, x1, h, h_1, W_ih, b_ih, W_hh, b_hh], autograd)
                if a
            ]

            def RNN(x: torch.Tensor, y: torch.Tensor):
                c_new_0 = x[0] + 1
                c_new_1 = x[1] + 1
                h_new = (
                    torch.tanh(c_new_1 + x[0] @ W_hh + b_hh)
                    + y[0] @ W_ih
                    + y[1] @ W_ih
                    + b_ih
                    + x[1]
                )
                return (c_new_0, c_new_1), h_new

            inits = (h, h_1)
            result = scan_fct(RNN, inits, (x, x1), dim=dim, reverse=reverse)
            result_exp = _fake_scan(RNN, (h, h_1), (x, x1), dim=dim, reverse=reverse)
            self.assertEqual(result, result_exp)

            if autograd:
                result_flat = pytree.tree_leaves(result)
                result_exp_flat = pytree.tree_leaves(result_exp)
                exp_grad_mask = [bool(r.requires_grad) for r in result_exp_flat]
                self.check_autograd(
                    [r for r, m in zip(result_flat, exp_grad_mask) if m],
                    [r for r, m in zip(result_exp_flat, exp_grad_mask) if m],
                    params,
                )

    @requires_cuda
    @skipIfTorchDynamo("not a dynamo test")
    @unittest.skipIf(not SM70OrLater, "triton")
    @parametrize("layers", [1, 2, 3])
    @parametrize("device", ["cpu", "cuda"])
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_scan_multiple_layers_gradient(self, layers, device):
        import torch.nn as nn

        torch.manual_seed(1)

        LAYERS = layers
        BATCH_SIZE = 2
        SEQ_LEN = 5
        FEATURE_DIM = 10
        DEVICE = device

        class RNNLoop(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList(
                    [nn.Linear(FEATURE_DIM * 2, FEATURE_DIM) for _ in range(LAYERS)]
                )
                self.num_layers = LAYERS

            def forward(self, initial, inputs_sequence):
                B, T, _ = inputs_sequence.shape
                hs_list = initial
                all_out = []
                for t in range(T):
                    input = inputs_sequence[:, t, :]
                    for li, layer in enumerate(self.layers):
                        input_concat = torch.cat((hs_list[li], input), dim=-1)
                        update = layer(input_concat)
                        hs_list[li] = hs_list[li] + update
                        input = hs_list[li]

                    all_out.append(input)

                return torch.stack(all_out, dim=1)

        class RNNScanList(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList(
                    [nn.Linear(FEATURE_DIM * 2, FEATURE_DIM) for _ in range(LAYERS)]
                )
                self.num_layers = LAYERS

            def forward(self, initial, input_sequence):
                def step(carry, input):
                    hs_list = carry[:]
                    for li, layer in enumerate(self.layers):
                        h_prev_li = hs_list[li]
                        input_concat = torch.cat((h_prev_li, input), dim=-1)
                        update = layer(input_concat)
                        h_curr_li = h_prev_li + update
                        hs_list[li] = h_curr_li
                        input = h_curr_li
                    return [t.clone() for t in hs_list], input.clone()

                _, all_outputs_scan = scan(step, initial, input_sequence, dim=1)
                return all_outputs_scan.transpose(0, 1)

        class RNNScanTensor(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList(
                    [nn.Linear(FEATURE_DIM * 2, FEATURE_DIM) for _ in range(LAYERS)]
                )
                self.num_layers = LAYERS

            def forward(self, initial, input_sequence):
                def step(carry_tensor, xs_input):
                    input = xs_input
                    hs_tensor = carry_tensor
                    for li, layer in enumerate(self.layers):
                        current_h_prev_li_slice = hs_tensor[:, li, :]
                        input_concat = torch.cat(
                            (current_h_prev_li_slice, input), dim=-1
                        )
                        update = layer(input_concat)
                        h_curr_li = current_h_prev_li_slice + update
                        hs_tensor = hs_tensor.clone()
                        hs_tensor[:, li, :] = h_curr_li
                        input = h_curr_li
                    return hs_tensor.clone(), input.clone()

                hs_stacked = torch.stack(initial, dim=1)
                _, all_outputs_scan = scan(step, hs_stacked, input_sequence, dim=1)
                return all_outputs_scan.transpose(0, 1)

        def run_test_and_get_grads_loss(model, initial_hs, inputs):
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.zero_()

            current_initial_hs = [
                h.detach().clone().requires_grad_(h.requires_grad) for h in initial_hs
            ]
            current_inputs = (
                inputs.detach().clone().requires_grad_(inputs.requires_grad)
            )

            out = model(current_initial_hs, current_inputs)
            loss = out.sum()
            loss.backward()

            layer_grads = []
            for layer in model.layers:
                layer_grads.append(layer.weight.grad.clone())

            return layer_grads, loss

        torch.manual_seed(0)

        initial_hs_template = [
            torch.zeros(
                BATCH_SIZE, FEATURE_DIM, requires_grad=True, dtype=torch.float32
            ).to(DEVICE)
            for _ in range(LAYERS)
        ]
        inputs_template = torch.randn(
            BATCH_SIZE, SEQ_LEN, FEATURE_DIM, requires_grad=True, dtype=torch.float32
        ).to(DEVICE)

        # Test 3 models: RNNScanList, RNNScanTensor, RNNLoop
        models = [RNNScanList, RNNScanTensor, RNNLoop]

        for model_class in models:
            # Create uncompiled model
            model_uc = model_class().to(DEVICE)
            uncompiled_grads, uncompiled_loss = run_test_and_get_grads_loss(
                model_uc, initial_hs_template, inputs_template
            )

            # Create compiled model with same weights
            model_to_compile = model_class().to(DEVICE)
            model_to_compile.load_state_dict(model_uc.state_dict())
            compiled_model = torch.compile(model_to_compile)
            compiled_grads, compiled_loss = run_test_and_get_grads_loss(
                compiled_model, initial_hs_template, inputs_template
            )

            # Compare gradients for each layer
            for uncompiled_grad, compiled_grad in zip(uncompiled_grads, compiled_grads):
                self.assertEqual(
                    uncompiled_grad,
                    compiled_grad,
                )

            # Compare losses
            self.assertEqual(
                uncompiled_loss,
                compiled_loss,
            )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_closure_combine_fn_with_no_grad_init_carries_unequal_grad(
        self, reverse, compile_mode, device, autograd
    ):
        dim = 1
        scan_fct = compile_mode_helper(scan, compile_mode)
        x = torch.randn(3, 10, 7, device=device, requires_grad=autograd)
        h1 = torch.randn(3, 7, device=device, requires_grad=autograd)
        h2 = torch.randn(3, 7, device=device, requires_grad=autograd)

        result = scan_fct(
            get_scan_combine_fn("fct_c1_no_grad", True),
            (h1, h2),
            x,
            dim=dim,
            reverse=reverse,
        )
        result_exp = _fake_scan(
            get_scan_combine_fn("fct_c1_no_grad", True),
            (h1, h2),
            x,
            dim=dim,
            reverse=reverse,
        )
        self.assertEqual(result, result_exp)

        if autograd:
            # TODO: Ideally we should be able to select the results that require gradients like this
            # [leaf for leaf in pytree.tree_leaves(result) if leaf.requires_grad == True]
            # However, for the scan operator this does not work, as all outputs always have
            # grad_fn=<ScanAutogradOpBackward>
            res_req_grad_flat = pytree.tree_leaves(result)[1:]
            res_exp_req_grad_flat = pytree.tree_leaves(result_exp)[1:]
            self.check_autograd(res_req_grad_flat, res_exp_req_grad_flat, (x, h2))

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_closure_combine_fn_with_no_grad_init_carries_equal_grad(
        self, reverse, compile_mode, device, autograd
    ):
        dim = 1
        scan_fct = compile_mode_helper(scan, compile_mode)
        x = torch.randn(3, 10, 7, device=device, requires_grad=autograd)
        h1 = torch.randn(3, 7, device=device, requires_grad=False)
        h2 = torch.randn(3, 7, device=device, requires_grad=autograd)

        result = scan_fct(
            get_scan_combine_fn("fct_c1_no_grad", True),
            (h1, h2),
            x,
            dim=dim,
            reverse=reverse,
        )
        result_exp = _fake_scan(
            get_scan_combine_fn("fct_c1_no_grad", True),
            (h1, h2),
            x,
            dim=dim,
            reverse=reverse,
        )
        self.assertEqual(result, result_exp)

        if autograd:
            # TODO: Ideally we should be able to select the results that require gradients like this
            # [leaf for leaf in pytree.tree_leaves(result) if leaf.requires_grad == True]
            # However, for the scan operator this does not work, as all outputs always have
            # grad_fn=<ScanAutogradOpBackward>
            res_req_grad_flat = pytree.tree_leaves(result)[1:]
            res_exp_req_grad_flat = pytree.tree_leaves(result_exp)[1:]
            self.check_autograd(res_req_grad_flat, res_exp_req_grad_flat, (x, h2))

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_closure_combine_fn_with_no_grad_for_out(
        self, reverse, compile_mode, device, autograd
    ):
        dim = 1
        scan_fct = compile_mode_helper(scan, compile_mode)
        x = torch.randn(3, 10, 7, device=device, requires_grad=autograd)
        h1 = torch.randn(3, 7, device=device, requires_grad=autograd)
        h2 = torch.randn(3, 7, device=device, requires_grad=autograd)

        def fct_ys_no_grad(x: torch.Tensor, y: torch.Tensor):
            c1 = x[0] + y
            c2 = x[1] + y
            with torch.no_grad():
                h_new = torch.tanh(x[0] + x[1] + y)
            return (c1, c2), h_new

        result = scan_fct(fct_ys_no_grad, (h1, h2), x, dim=dim, reverse=reverse)
        result_exp = _fake_scan(fct_ys_no_grad, (h1, h2), x, dim=dim, reverse=reverse)
        self.assertEqual(result, result_exp)

        if autograd:
            self.check_autograd(result[0], result_exp[0], (x, h1, h2))

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_closure_combine_fn_with_no_grad_additional_inputs_partial(
        self, reverse, compile_mode, device, autograd
    ):
        dim = 1
        scan_fct = compile_mode_helper(scan, compile_mode)
        x = torch.randn(3, 10, 7, device=device, requires_grad=autograd)
        h = torch.randn(3, 7, device=device, requires_grad=autograd)
        W_ih = torch.randn(7, 7, device=device, requires_grad=autograd)
        b_ih = torch.randn(7, device=device, requires_grad=autograd)
        W_hh = torch.randn(7, 7, device=device, requires_grad=autograd)
        b_hh = torch.randn(7, device=device, requires_grad=autograd)

        def fct_no_grad_bhh_Whh(x: torch.Tensor, y: torch.Tensor):
            c_new = y @ W_ih + b_ih + x

            h_new = c_new + 1
            with torch.no_grad():
                h_new_no_grad = torch.tanh(x @ W_hh + b_hh)
            h_new2 = h_new + h_new_no_grad

            return c_new, h_new2

        result = scan_fct(fct_no_grad_bhh_Whh, h, x, dim=dim, reverse=reverse)
        result_exp = _fake_scan(fct_no_grad_bhh_Whh, h, x, dim=dim, reverse=reverse)
        self.assertEqual(result, result_exp)

        if autograd:
            self.check_autograd(result[1], result_exp[1], (h, x, W_ih, b_ih))

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_closure_combine_fn_with_no_grad_additional_inputs_all(
        self, reverse, compile_mode, device, autograd
    ):
        dim = 1
        scan_fct = compile_mode_helper(scan, compile_mode)
        x = torch.randn(3, 10, 7, device=device, requires_grad=autograd)
        h = torch.randn(3, 7, device=device, requires_grad=autograd)
        W_ih = torch.randn(7, 7, device=device, requires_grad=autograd)
        b_ih = torch.randn(7, device=device, requires_grad=autograd)
        W_hh = torch.randn(7, 7, device=device, requires_grad=autograd)
        b_hh = torch.randn(7, device=device, requires_grad=autograd)

        def fct_no_grad_bih_Wih_bhh_Whh(x: torch.Tensor, y: torch.Tensor):
            c_new = x + y
            h_new = c_new + x
            with torch.no_grad():
                c_new_no_grad = y @ W_ih + b_ih
                h_new_no_grad = torch.tanh(x @ W_hh + b_hh)
            c_new2 = c_new + c_new_no_grad
            h_new2 = h_new + h_new_no_grad
            return c_new2, h_new2

        result = scan_fct(fct_no_grad_bih_Wih_bhh_Whh, h, x, dim=dim, reverse=reverse)
        result_exp = _fake_scan(
            fct_no_grad_bih_Wih_bhh_Whh, h, x, dim=dim, reverse=reverse
        )
        self.assertEqual(result, result_exp)

        if autograd:
            self.check_autograd(result[1], result_exp[1], (h, x))

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_closure_combine_fn_carries_ys_same_grad(
        self, reverse, compile_mode, device, autograd
    ):
        dim = 1
        scan_fct = compile_mode_helper(scan, compile_mode)
        x = torch.randn(3, 10, 7, device=device, requires_grad=autograd)
        h = torch.randn(3, 7, device=device, requires_grad=autograd)
        W_ih = torch.randn(7, 7, device=device, requires_grad=autograd)
        b_ih = torch.randn(7, device=device, requires_grad=autograd)
        W_hh = torch.randn(7, 7, device=device, requires_grad=autograd)
        b_hh = torch.randn(7, device=device, requires_grad=autograd)

        def fct_no_grad_bih_Wih_bhh_Whh(x: torch.Tensor, y: torch.Tensor):
            c_new = x + y
            h_new = c_new + 1
            with torch.no_grad():
                c_new_no_grad = y @ W_ih + b_ih
                h_new_no_grad = torch.tanh(x @ W_hh + b_hh)
            c_new2 = c_new + c_new_no_grad
            h_new2 = h_new + h_new_no_grad
            return c_new2, h_new2

        result = scan_fct(fct_no_grad_bih_Wih_bhh_Whh, h, x, dim=dim, reverse=reverse)
        result_exp = _fake_scan(
            fct_no_grad_bih_Wih_bhh_Whh, h, x, dim=dim, reverse=reverse
        )
        self.assertEqual(result, result_exp)

        if autograd:
            self.check_autograd(result[1], result_exp[1], (h, x))

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_closure_nested(self, reverse, compile_mode, device, autograd):
        scan_fct = compile_mode_helper(scan, compile_mode)

        # Simple non-nested case
        x = torch.randn(3, 20, 5, device=device, requires_grad=autograd)
        h = torch.randn(3, 7, device=device, requires_grad=autograd)
        W = torch.randn(5, 7, device=device, requires_grad=autograd)
        b = torch.randn(7, device=device, requires_grad=autograd)

        def f1(x: torch.Tensor, y: torch.Tensor):
            c_new = y @ W + b
            h_new = torch.tanh(c_new + x)
            return c_new, h_new

        result = scan_fct(f1, h, x, dim=1, reverse=reverse)
        result_exp = _fake_scan(f1, h, x, dim=1, reverse=reverse)
        self.assertEqual(result, result_exp)

        if autograd:
            self.check_autograd(result, result_exp, (h, x, W, b))

        # Nested case
        def chain_fct(fct, f_1, f_2, xs, h_1, h_2):
            o1 = fct(
                f_1,
                h_1,
                xs,
                dim=1,
                reverse=reverse,
            )
            o2 = fct(
                f_2,
                h_2,
                o1[1],
                dim=0,
                reverse=reverse,
            )
            return o2

        x1 = torch.ones(3, 20, 5, device=device, requires_grad=autograd)
        h1 = torch.zeros(3, 7, device=device, requires_grad=autograd)
        h2 = torch.zeros(3, 3, device=device, requires_grad=autograd)
        W_1 = torch.randn(5, 7, device=device, requires_grad=autograd)
        b_1 = torch.randn(7, device=device, requires_grad=autograd)
        W_2 = torch.randn(7, 3, device=device, requires_grad=autograd)
        b_2 = torch.randn(3, device=device, requires_grad=autograd)

        def f1(x: torch.Tensor, y: torch.Tensor):
            c_new = y @ W_1 + b_1
            h_new = torch.tanh(c_new + x)
            return c_new, h_new

        def f2(x: torch.Tensor, y: torch.Tensor):
            c_new = y @ W_2 + b_2
            h_new = torch.tanh(c_new + x)
            return c_new, h_new

        result1 = chain_fct(scan_fct, f1, f2, x1, h1, h2)
        expected_result = chain_fct(_fake_scan, f1, f2, x1, h1, h2)
        self.assertEqual(result1, expected_result)

        if autograd:
            self.check_autograd(result1, expected_result, (h1, h2, x1, W_1, b_1))

        # Complex case
        x1 = torch.randn(3, 20, 3, device=device, requires_grad=autograd)
        h1 = torch.randn(3, 3, device=device, requires_grad=autograd)
        h2 = torch.randn(3, 3, device=device, requires_grad=autograd)
        W_1 = torch.randn(3, 3, device=device, requires_grad=autograd)
        b_1 = torch.randn(3, device=device, requires_grad=autograd)
        W_2 = torch.randn(3, 3, device=device, requires_grad=autograd)
        b_2 = torch.randn(3, device=device, requires_grad=autograd)

        def f1(x: torch.Tensor, y: torch.Tensor):
            c_new = y @ W_1 + b_1
            h_new = torch.tanh(c_new + x)
            return c_new, h_new

        def f2(x: torch.Tensor, y: torch.Tensor):
            c_new = y @ W_2 + b_2 * b_1 + y @ W_1
            h_new = torch.tanh(c_new + x)
            return c_new, h_new

        result1 = chain_fct(scan_fct, f1, f2, x1, h1, h2)
        expected_result = chain_fct(_fake_scan, f1, f2, x1, h1, h2)
        self.assertEqual(result1, expected_result)

        if autograd:
            self.check_autograd(
                result1, expected_result, (h1, h2, x1, W_1, b_1, W_2, b_2)
            )

    @skipIfNoDynamoSupport
    def test_scan_simple_graph_wrong_dtype(self):
        def add_wrong_dtype(x: torch.Tensor, y: torch.Tensor):
            return torch.ones_like(x + y, dtype=torch.int64), x + y

        x = torch.randn(3, 10, 2, device=torch.device("cpu"))
        init = torch.randn(1, 10, 2, device=torch.device("cpu"))

        def f(fct, init, xs):
            return scan(fct, init, xs, dim=0, reverse=True)

        # Wrong dtype
        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "Expected init and carry to have same metadata.*",
        ):
            f(add_wrong_dtype, init, x)

    @skipIfNoDynamoSupport
    @skipIfCrossRef  # Arg order changes with crossref
    def test_scan_simple_graph(self):
        x = torch.randn(3, 10, 2, device=torch.device("cpu"))
        init = torch.randn(1, 10, 2, device=torch.device("cpu"))

        def f(fct, init, xs):
            return scan(fct, init, xs, dim=0, reverse=True)

        # Correct case
        gm = make_fx(f, tracing_mode="symbolic")(
            get_scan_combine_fn("add", False), init, x
        )
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, fct_1, init_1, xs_1):
    flip = torch.ops.aten.flip.default(xs_1, [0])
    sym_size_int_1 = torch.ops.aten.sym_size.int(init_1, 1)
    sym_size_int_2 = torch.ops.aten.sym_size.int(init_1, 2)
    sym_size_int_3 = torch.ops.aten.sym_size.int(xs_1, 1)
    sym_size_int_4 = torch.ops.aten.sym_size.int(xs_1, 2);  xs_1 = None
    scan_combine_graph_0 = self.scan_combine_graph_0
    scan = torch.ops.higher_order.scan(scan_combine_graph_0, [init_1], [flip], (sym_size_int_1, sym_size_int_2, sym_size_int_3, sym_size_int_4));  scan_combine_graph_0 = init_1 = flip = sym_size_int_1 = sym_size_int_2 = sym_size_int_3 = sym_size_int_4 = None
    getitem = scan[0]
    getitem_1 = scan[1];  scan = None
    flip_1 = torch.ops.aten.flip.default(getitem_1, [0]);  getitem_1 = None
    return (getitem, flip_1)""",  # noqa: B950
        )

        # Check graph
        backend = EagerAndRecordGraphs()
        torch.compile(f, backend=backend)(get_scan_combine_fn("add", False), init, x)
        gm = backend.graphs[0]

        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, L_init_ : torch.Tensor, L_xs_ : torch.Tensor):
    l_init_ = L_init_
    l_xs_ = L_xs_
    flip = torch.flip(l_xs_, [0]);  l_xs_ = None
    scan_combine_fn_0 = self.scan_combine_fn_0
    scan = torch.ops.higher_order.scan(scan_combine_fn_0, [l_init_], [flip], []);  scan_combine_fn_0 = l_init_ = flip = None
    carry = scan[0]
    out = scan[1];  scan = None
    out_1 = out.flip([0]);  out = None
    return (carry, out_1)""",  # noqa: B950
        )

    @requires_cuda
    def test_scan_input_mutation(self):
        device = torch.device("cuda")

        def fct_input_mutation(x, y):
            x.add_(1)
            return x + y, x + y + 2

        x = torch.randn(3, 2, 2, device=device)
        init = torch.randn(2, 2, device=device)

        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.Unsupported,
            # "Encountered aliasing during higher order op tracing for HOP.*"
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.ops\.higher_order\.scan",
        ):
            scan(fct_input_mutation, init, x, dim=0)

    @requires_cuda
    def test_scan_input_carry_alias(self):
        device = torch.device("cuda")

        def fct_input_output_alias(x, y):
            return (x[0], x[1] + y[1]), (x[1] + y[1] + 1, x[1] + y[1] + 2)

        x = torch.randn(3, 2, 2, device=device)
        y = torch.randn(3, 2, 2, device=device)
        inp = (x, y)
        init = (torch.randn(2, 2, device=device), torch.randn(2, 2, device=device))

        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.Unsupported,
            # "Encountered aliasing during higher order op tracing for HOP.*"
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.ops\.higher_order\.scan",
        ):
            scan(fct_input_output_alias, init, inp, dim=0)

    @requires_cuda
    def test_scan_input_output_alias(self):
        device = torch.device("cuda")

        def fct_input_output_alias(x, y):
            return (x[0] + 1, x[1] + y[1]), (x[1], x[1] + y[1] + 2)

        x = torch.randn(3, 2, 2, device=device)
        y = torch.randn(3, 2, 2, device=device)
        inp = (x, y)
        init = (torch.randn(2, 2, device=device), torch.randn(2, 2, device=device))

        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.Unsupported,
            # "Encountered aliasing during higher order op tracing for HOP.*"
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.ops\.higher_order\.scan",
        ):
            scan(fct_input_output_alias, init, inp, dim=0)

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    def test_scan_carry_carry_alias(self):
        device = torch.device("cuda")

        def fct_carry_carry_alias(x, y):
            c = x[0] + y[1]
            return (c, c), (x[0] + y[1], x[0] + y[1] + 1)

        x = torch.randn(3, 2, 2, device=device)
        y = torch.randn(3, 2, 2, device=device)
        inp = (x, y)
        init = (torch.randn(2, 2, device=device), torch.randn(2, 2, device=device))

        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.Unsupported,
            # "Encountered aliasing during higher order op tracing for HOP.*"
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.ops\.higher_order\.scan",
        ):
            scan(fct_carry_carry_alias, init, inp, dim=0)

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    def test_scan_carry_output_alias(self):
        device = torch.device("cuda")

        def fct_carry_output_alias(x, y):
            c = x[0] + y[1]
            return (x[0] + y[1], c), (c, x[0] + y[1] + 1)

        x = torch.randn(3, 2, 2, device=device)
        y = torch.randn(3, 2, 2, device=device)
        inp = (x, y)
        init = (torch.randn(2, 2, device=device), torch.randn(2, 2, device=device))

        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.Unsupported,
            # "Encountered aliasing during higher order op tracing for HOP.*"
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.ops\.higher_order\.scan",
        ):
            scan(fct_carry_output_alias, init, inp, dim=0)


class AssociativeScanModels:
    @staticmethod
    def get_scan_fct(compile_mode, combine_mode):
        # Compile the associative_scan according to the provided compile_mode
        if compile_mode != "fake":
            assoc_scan_comp = compile_mode_helper(associative_scan, compile_mode)

            def scan_fct(combine_fn, xs, dim, reverse):
                return assoc_scan_comp(combine_fn, xs, dim, reverse, combine_mode)

        else:
            scan_fct = _fake_associative_scan
        return scan_fct

    class CombineFn(torch.nn.Module):
        def __init__(self, combine_fn, dim, reverse, combine_mode, compile_mode):
            super().__init__()

            self.scan_fct = AssociativeScanModels.get_scan_fct(
                compile_mode, combine_mode
            )
            self.combine_fn = combine_fn
            self.dim = dim
            self.reverse = reverse

        def forward(self, inputs):
            results = self.scan_fct(self.combine_fn, inputs, self.dim, self.reverse)
            return results

    class Simple(torch.nn.Module):
        def __init__(self, dim, reverse, combine_mode, compile_mode):
            super().__init__()

            kwargs = {
                "dim": dim,
                "reverse": reverse,
                "combine_mode": combine_mode,
                "compile_mode": compile_mode,
            }
            self.combine_fns = [
                AssociativeScanModels.CombineFn(
                    get_scan_combine_fn("add", True), **kwargs
                ),
                AssociativeScanModels.CombineFn(
                    get_scan_combine_fn("mul", True), **kwargs
                ),
            ]

        def forward(self, inputs):
            results = []
            for combine_fn in self.combine_fns:
                results.append(combine_fn(inputs))
            return results

    class ChainFn(torch.nn.Module):
        def __init__(self, combine_fn, dim, reverse, combine_mode, compile_mode):
            super().__init__()

            chain_len = len(combine_fn)
            kwargs = {
                "combine_fn": combine_fn,
                "dim": dim,
                "reverse": reverse,
                "combine_mode": combine_mode,
            }

            # Prepare the kwargs as a list.
            self.nested_tuple = []
            for ind in range(chain_len):
                kwargs_el = {}
                for key, val in kwargs.items():
                    # Check if val is a list and if it has the same length as combine_fn
                    # If so, then use the individual elements.
                    # If not, duplicate the first element.
                    if type(val) is list and len(val) == chain_len:
                        kwargs_el[key] = val[ind]
                    else:
                        kwargs_el[key] = val

                scan_fct = AssociativeScanModels.get_scan_fct(
                    compile_mode, kwargs_el["combine_mode"]
                )
                combine_fn = kwargs_el["combine_fn"]
                del kwargs_el["combine_fn"]
                del kwargs_el["combine_mode"]
                self.nested_tuple.append((combine_fn, scan_fct, kwargs_el))

        def forward(self, inputs):
            results = inputs
            for combine_fn, scan_fct, kwargs in self.nested_tuple:
                results = combine_fn(scan_fct, results, **kwargs)
            return results

    class NestedFn(torch.nn.Module):
        def forward(self, scan_fct, inputs, **kwargs):
            combine_fn = kwargs["combine_fn"]

            # Remove combine_fn from kwargs
            del kwargs["combine_fn"]

            results = scan_fct(combine_fn, inputs, **kwargs)

            return results


@unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
@skipIfNoDynamoSupport
class AssociativeScanTests(TestCase):
    def setUp(self):
        torch._dynamo.reset()
        super().setUp()

    def _check_autograd(self, result, result_exp, autograd_param):
        grad_param = [p for p in autograd_param if p.requires_grad]

        result_flatten, _ = pytree.tree_flatten(result)
        result_exp_flatten, _ = pytree.tree_flatten(result_exp)
        result_flatten = [r for r in result_flatten if r.requires_grad]
        result_exp_flatten = [r for r in result_exp_flatten if r.requires_grad]

        # Check the result and parameter lists
        if len(result_flatten) != len(result_exp_flatten):
            raise AssertionError(
                "The number of elements requiring gradients is different for the results and the expected results"
            )

        grad_exp_init = [torch.ones_like(el) for el in result_exp_flatten]
        expected_grads = torch.autograd.grad(
            result_exp_flatten, grad_param, grad_exp_init
        )
        grad_init = [torch.ones_like(el) for el in result_flatten]
        grads = torch.autograd.grad(result_flatten, grad_param, grad_init)

        self.assertEqual(grads, expected_grads, atol=6e-05, rtol=6e-06)

    def _run_test(self, model, model_fake, inputs, autograd_param=None):
        result = model(inputs)
        result_exp = model_fake(inputs)
        self.assertEqual(result, result_exp)

        if autograd_param is not None and any(
            par.requires_grad for par in autograd_param
        ):
            result_flat = pytree.tree_leaves(result)
            result_exp_flat = pytree.tree_leaves(result_exp)
            exp_grad_mask = [bool(r.requires_grad) for r in result_exp_flat]

            self._check_autograd(
                [r for r, m in zip(result_flat, exp_grad_mask) if m],
                [r for r, m in zip(result_exp_flat, exp_grad_mask) if m],
                autograd_param,
            )

        # Return the result of the functions under test for further investigations
        return result

    def _prepare_fake_kwargs(self, original_kwargs):
        kwargs_fake = original_kwargs.copy()
        kwargs_fake["compile_mode"] = "fake"
        return kwargs_fake

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("combine_mode", ["pointwise", "generic"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    # Skipping the combination of combine_mode=pointwise and device=cpu
    # as the current implementation of pointwise does only support CUDA device
    # Skipping the combination of combine_mode=pointwise and compile_mode=compile_dynamic_shape
    # as the current implementation does not support lifted arguments
    @decorateIf(
        unittest.skip,
        lambda params: (
            params["combine_mode"] == "pointwise"
            and (
                params["device"] == torch.device("cpu")
                or params["compile_mode"] == "compile_dynamic_shape"
            )
        ),
    )
    # # Skipping this combination as there is a CPP compilation failure that
    # # may be unrelated to associative_scan itself. There is a dedicated tests for
    # # this case below.
    # @decorateIf(
    #     unittest.skip,
    #     lambda params: (
    #         params["compile_mode"] == "compile_dynamic_shape"
    #         and params["combine_mode"] == "generic"
    #         and params["device"] == torch.device("cpu")
    #         and params["autograd"]
    #     ),
    # )
    def test_associative_scan_compile(
        self, combine_mode, reverse, compile_mode, device, autograd
    ):
        x = torch.randn(3, 10, 2, device=device, requires_grad=autograd)
        kwargs = {
            "dim": 0,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_mode": combine_mode,
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        results = self._run_test(
            model=AssociativeScanModels.Simple(**kwargs),
            model_fake=AssociativeScanModels.Simple(**kwargs_fake),
            inputs=x,
            autograd_param=None if not autograd else (x,),
        )

        if not reverse:
            results_torch = []
            for op_pt in [torch.cumsum, torch.cumprod]:
                results_torch.append(op_pt(x, 0))
            self.assertEqual(results, results_torch)

        # Jax Examples
        x = torch.arange(
            0, 4, device=device, dtype=torch.float32, requires_grad=autograd
        )
        kwargs = {
            "dim": 0,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": get_scan_combine_fn("add", True),
            "combine_mode": combine_mode,
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        result = self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=x,
            autograd_param=None if not autograd else (x,),
        )

        if not reverse:
            results_torch = torch.tensor([0.0, 1.0, 3.0, 6.0], dtype=torch.float32)
        else:
            results_torch = torch.tensor([6.0, 6.0, 5.0, 3.0], dtype=torch.float32)

        self.assertEqual(result, results_torch)

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("combine_mode", ["pointwise", "generic"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    # Skipping the combination of combine_mode=pointwise and device=cpu
    # as the current implementation of pointwise does only support CUDA device
    # Skipping the combination of combine_mode=pointwise and compile_mode=compile_dynamic_shape
    # as the current implementation does not support lifted arguments
    @decorateIf(
        unittest.skip,
        lambda params: (
            params["combine_mode"] == "pointwise"
            and (
                params["device"] == torch.device("cpu")
                or params["compile_mode"] == "compile_dynamic_shape"
            )
        ),
    )
    def test_associative_scan_dim(
        self, combine_mode, compile_mode, reverse, device, autograd
    ):
        import random

        random.seed(1234)

        num_dims = [random.randint(2, 5) for _ in range(4)]
        for num_dim in num_dims:
            # To avoid triggering automatic dynamic shape
            torch._dynamo.reset()
            shapes = [random.randint(1, 9) for _ in range(num_dim)]
            rnd_scan_dim = random.randint(0, num_dim - 1)
            x = torch.randn(*shapes, device=device, requires_grad=autograd)

            kwargs = {
                "dim": rnd_scan_dim,
                "reverse": reverse,
                "compile_mode": compile_mode,
                "combine_mode": combine_mode,
            }
            kwargs_fake = self._prepare_fake_kwargs(kwargs)
            results = self._run_test(
                model=AssociativeScanModels.Simple(**kwargs),
                model_fake=AssociativeScanModels.Simple(**kwargs_fake),
                inputs=x,
                autograd_param=None if not autograd else (x,),
            )

            if not reverse:
                results_torch = []
                for op_pt in [torch.cumsum, torch.cumprod]:
                    results_torch.append(op_pt(x, 0))
                self.assertEqual(results, results_torch)

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @unittest.expectedFailure
    def test_associative_scan_dim_shape_failure(self, compile_mode, combine_mode):
        num_dims = [2]
        for num_dim in num_dims:
            shapes = [9 for _ in range(num_dim)]
            rnd_scan_dim = 0
            x = torch.randn(*shapes, device=torch.device("cuda"))

            kwargs = {
                "dim": rnd_scan_dim,
                "reverse": True,
                "compile_mode": "compile",
                "combine_mode": "generic",
            }
            kwargs_fake = self._prepare_fake_kwargs(kwargs)
            self._run_test(
                model=AssociativeScanModels.Simple(**kwargs),
                model_fake=AssociativeScanModels.Simple(**kwargs_fake),
                inputs=x,
            )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("combine_mode", ["pointwise", "generic"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    # Skipping the combination of combine_mode=pointwise and device=cpu
    # as the current implementation of pointwise does only support CUDA device
    # Skipping the combination of combine_mode=pointwise and compile_mode=compile_dynamic_shape
    # as the current implementation does not support lifted arguments
    @decorateIf(
        unittest.skip,
        lambda params: (
            params["combine_mode"] == "pointwise"
            and (
                params["device"] == torch.device("cpu")
                or params["compile_mode"] == "compile_dynamic_shape"
            )
        ),
    )
    def test_associative_scan_tuple(
        self, compile_mode, combine_mode, reverse, device, autograd
    ):
        x = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        y = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        inp = (x, y)

        kwargs = {
            "dim": 0,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": get_scan_combine_fn("tuple_fct", True),
            "combine_mode": combine_mode,
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=inp,
            autograd_param=None if not autograd else inp,
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_associative_scan_expand_in_combine_fn(
        self, compile_mode, reverse, device, autograd
    ):
        x = torch.randn(3, 2, 2, device=device, requires_grad=autograd)

        def combine_fn(x, y):
            return x * torch.sum(y, -1).expand(x.shape)

        kwargs = {
            "dim": 0,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": combine_fn,
            "combine_mode": "generic",
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=x,
            autograd_param=None if not autograd else (x,),
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_associative_scan_non_contiguous_tensor(
        self, compile_mode, reverse, device, autograd
    ):
        x = (
            torch.arange(30, device=device, dtype=torch.float32, requires_grad=autograd)
            .view(10, 3)
            .t()
        )
        if x.is_contiguous():
            raise AssertionError("Expected x to not be contiguous")

        kwargs = {
            "dim": 0,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": get_scan_combine_fn("add", True),
            "combine_mode": "generic",
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=x,
            autograd_param=None if not autograd else (x,),
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("combine_mode", ["pointwise", "generic"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    # Skipping the combination of combine_mode=pointwise and device=cpu
    # as the current implementation of pointwise does only support CUDA device
    # Skipping the combination of combine_mode=pointwise and compile_mode=compile_dynamic_shape
    # as the current implementation does not support lifted arguments
    @decorateIf(
        unittest.skip,
        lambda params: (
            params["combine_mode"] == "pointwise"
            and (
                params["device"] == torch.device("cpu")
                or params["compile_mode"] == "compile_dynamic_shape"
            )
        ),
    )
    def test_associative_scan_complex_pytree(
        self, compile_mode, combine_mode, reverse, device, autograd
    ):
        x = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        y = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        z = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        inp = {"i": x, "j": ([y], [{"o": z}])}

        kwargs = {
            "dim": 0,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": get_scan_combine_fn("complex_pointwise", True),
            "combine_mode": combine_mode,
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=inp,
            autograd_param=None if not autograd else (x, y, z),
        )

    @skipIfTorchDynamo("don't test compile on compile")
    @skipIfNoDynamoSupport
    @skipIfCrossRef  # Arg order changes with crossref
    def test_associative_scan_pytree_output(self):
        x = (
            (
                torch.randn(3, 10, 2, device=torch.device("cpu")),
                (torch.randn(3, 10, 2, device=torch.device("cpu")),),
            ),
            torch.randn(3, 10, 2, device=torch.device("cpu")),
        )

        def f(fct, xs):
            return associative_scan(
                fct, xs, dim=0, reverse=True, combine_mode="generic"
            )

        def combine_fn(x: torch.Tensor, y: torch.Tensor):
            a, b = (x[0][0] + y[1], x[0][1][0] - y[1])
            return (a, (b,)), a - b

        # Check graph
        backend = EagerAndRecordGraphs()
        torch.compile(f, backend=backend)(combine_fn, x)
        gm = backend.graphs[0]

        self.assertExpectedInline(
            normalize_gm(gm.print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_xs_0_0_: "f32[3, 10, 2]", L_xs_0_1_0_: "f32[3, 10, 2]", L_xs_1_: "f32[3, 10, 2]"):
        l_xs_0_0_ = L_xs_0_0_
        l_xs_0_1_0_ = L_xs_0_1_0_
        l_xs_1_ = L_xs_1_

        elem: "f32[3, 10, 2]" = torch.movedim(l_xs_0_0_, 0, 0);  l_xs_0_0_ = None
        elem_1: "f32[3, 10, 2]" = torch.movedim(l_xs_0_1_0_, 0, 0);  l_xs_0_1_0_ = None
        elem_2: "f32[3, 10, 2]" = torch.movedim(l_xs_1_, 0, 0);  l_xs_1_ = None
        elem_3: "f32[3, 10, 2]" = torch.flip(elem, [0]);  elem = None
        elem_4: "f32[3, 10, 2]" = torch.flip(elem_1, [0]);  elem_1 = None
        elem_5: "f32[3, 10, 2]" = torch.flip(elem_2, [0]);  elem_2 = None
        child: "f32[1, 10, 2]" = torch.ops.aten.slice(elem_3, 0, 0, -1, 2)
        child_1: "f32[1, 10, 2]" = torch.ops.aten.slice(elem_4, 0, 0, -1, 2)
        child_2: "f32[1, 10, 2]" = torch.ops.aten.slice(elem_5, 0, 0, -1, 2)
        child_3: "f32[1, 10, 2]" = torch.ops.aten.slice(elem_3, 0, 1, None, 2)
        child_4: "f32[1, 10, 2]" = torch.ops.aten.slice(elem_4, 0, 1, None, 2)
        child_5: "f32[1, 10, 2]" = torch.ops.aten.slice(elem_5, 0, 1, None, 2)

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(1, 'error');  _vmap_increment_nesting = None

        _add_batch_dim: "f32[10, 2]" = torch._functorch.predispatch._add_batch_dim(child, 0, 1);  child = None
        _add_batch_dim_1: "f32[10, 2]" = torch._functorch.predispatch._add_batch_dim(child_1, 0, 1);  child_1 = None
        _add_batch_dim_2: "f32[10, 2]" = torch._functorch.predispatch._add_batch_dim(child_2, 0, 1);  child_2 = _add_batch_dim_2 = None
        _add_batch_dim_3: "f32[10, 2]" = torch._functorch.predispatch._add_batch_dim(child_3, 0, 1);  child_3 = _add_batch_dim_3 = None
        _add_batch_dim_4: "f32[10, 2]" = torch._functorch.predispatch._add_batch_dim(child_4, 0, 1);  child_4 = _add_batch_dim_4 = None
        _add_batch_dim_5: "f32[10, 2]" = torch._functorch.predispatch._add_batch_dim(child_5, 0, 1);  child_5 = None

        a: "f32[10, 2]" = _add_batch_dim + _add_batch_dim_5;  _add_batch_dim = None
        b: "f32[10, 2]" = _add_batch_dim_1 - _add_batch_dim_5;  _add_batch_dim_1 = _add_batch_dim_5 = None

        child_6: "f32[10, 2]" = a - b

        child_7: "f32[1, 10, 2]" = torch._functorch.predispatch._remove_batch_dim(a, 1, 1, 0);  a = None
        child_8: "f32[1, 10, 2]" = torch._functorch.predispatch._remove_batch_dim(b, 1, 1, 0);  b = None
        child_9: "f32[1, 10, 2]" = torch._functorch.predispatch._remove_batch_dim(child_6, 1, 1, 0);  child_6 = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

        child_10: "f32[1, 10, 2]" = torch.ops.aten.slice(elem_3, 0, 2, None, 2)
        child_11: "f32[1, 10, 2]" = torch.ops.aten.slice(elem_4, 0, 2, None, 2)
        child_12: "f32[1, 10, 2]" = torch.ops.aten.slice(elem_5, 0, 2, None, 2)

        lazy_load_decompositions_1 = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions_1 = None

        _vmap_increment_nesting_1 = torch._functorch.predispatch._vmap_increment_nesting(1, 'error');  _vmap_increment_nesting_1 = None

        _add_batch_dim_6: "f32[10, 2]" = torch._functorch.predispatch._add_batch_dim(child_7, 0, 1)
        _add_batch_dim_7: "f32[10, 2]" = torch._functorch.predispatch._add_batch_dim(child_8, 0, 1)
        _add_batch_dim_8: "f32[10, 2]" = torch._functorch.predispatch._add_batch_dim(child_9, 0, 1);  _add_batch_dim_8 = None
        _add_batch_dim_9: "f32[10, 2]" = torch._functorch.predispatch._add_batch_dim(child_10, 0, 1);  child_10 = _add_batch_dim_9 = None
        _add_batch_dim_10: "f32[10, 2]" = torch._functorch.predispatch._add_batch_dim(child_11, 0, 1);  child_11 = _add_batch_dim_10 = None
        _add_batch_dim_11: "f32[10, 2]" = torch._functorch.predispatch._add_batch_dim(child_12, 0, 1);  child_12 = None

        a_1: "f32[10, 2]" = _add_batch_dim_6 + _add_batch_dim_11;  _add_batch_dim_6 = None
        b_1: "f32[10, 2]" = _add_batch_dim_7 - _add_batch_dim_11;  _add_batch_dim_7 = _add_batch_dim_11 = None

        child_13: "f32[10, 2]" = a_1 - b_1

        child_14: "f32[1, 10, 2]" = torch._functorch.predispatch._remove_batch_dim(a_1, 1, 1, 0);  a_1 = None
        child_15: "f32[1, 10, 2]" = torch._functorch.predispatch._remove_batch_dim(b_1, 1, 1, 0);  b_1 = None
        child_16: "f32[1, 10, 2]" = torch._functorch.predispatch._remove_batch_dim(child_13, 1, 1, 0);  child_13 = None

        _vmap_decrement_nesting_1 = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting_1 = None

        slice_10: "f32[1, 10, 2]" = torch.ops.aten.slice(elem_3, 0, 0, 1);  elem_3 = None
        cat: "f32[2, 10, 2]" = torch.cat([slice_10, child_14], dim = 0);  slice_10 = child_14 = None
        slice_11: "f32[1, 10, 2]" = torch.ops.aten.slice(elem_4, 0, 0, 1);  elem_4 = None
        cat_1: "f32[2, 10, 2]" = torch.cat([slice_11, child_15], dim = 0);  slice_11 = child_15 = None
        slice_12: "f32[1, 10, 2]" = torch.ops.aten.slice(elem_5, 0, 0, 1);  elem_5 = None
        cat_2: "f32[2, 10, 2]" = torch.cat([slice_12, child_16], dim = 0);  slice_12 = child_16 = None

        b_2: "f32[2, 10, 2]" = torch._C._nn.pad(child_7, [0, 0, 0, 0, 0, 1], 'constant', None);  child_7 = None

        stacked: "f32[2, 2, 10, 2]" = torch.stack([cat, b_2], dim = 1);  cat = b_2 = None
        interleaved: "f32[4, 10, 2]" = torch.flatten(stacked, start_dim = 0, end_dim = 1);  stacked = None
        interleaved_1: "f32[3, 10, 2]" = torch.ops.aten.slice(interleaved, 0, 0, 3);  interleaved = None

        b_3: "f32[2, 10, 2]" = torch._C._nn.pad(child_8, [0, 0, 0, 0, 0, 1], 'constant', None);  child_8 = None

        stacked_1: "f32[2, 2, 10, 2]" = torch.stack([cat_1, b_3], dim = 1);  cat_1 = b_3 = None
        interleaved_2: "f32[4, 10, 2]" = torch.flatten(stacked_1, start_dim = 0, end_dim = 1);  stacked_1 = None
        interleaved_3: "f32[3, 10, 2]" = torch.ops.aten.slice(interleaved_2, 0, 0, 3);  interleaved_2 = None

        b_4: "f32[2, 10, 2]" = torch._C._nn.pad(child_9, [0, 0, 0, 0, 0, 1], 'constant', None);  child_9 = None

        stacked_2: "f32[2, 2, 10, 2]" = torch.stack([cat_2, b_4], dim = 1);  cat_2 = b_4 = None
        interleaved_4: "f32[4, 10, 2]" = torch.flatten(stacked_2, start_dim = 0, end_dim = 1);  stacked_2 = None
        interleaved_5: "f32[3, 10, 2]" = torch.ops.aten.slice(interleaved_4, 0, 0, 3);  interleaved_4 = None
        flip_3: "f32[3, 10, 2]" = interleaved_1.flip([0]);  interleaved_1 = None
        flip_4: "f32[3, 10, 2]" = interleaved_3.flip([0]);  interleaved_3 = None
        flip_5: "f32[3, 10, 2]" = interleaved_5.flip([0]);  interleaved_5 = None
        movedim_3: "f32[3, 10, 2]" = torch.movedim(flip_3, 0, 0);  flip_3 = None
        movedim_4: "f32[3, 10, 2]" = torch.movedim(flip_4, 0, 0);  flip_4 = None
        movedim_5: "f32[3, 10, 2]" = torch.movedim(flip_5, 0, 0);  flip_5 = None
        return (movedim_3, movedim_4, movedim_5)
""",  # noqa: B950
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("combine_mode", ["pointwise", "generic"])
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    # Skipping the combination of combine_mode=pointwise and device=cpu
    # as the current implementation of pointwise does only support CUDA device
    # Skipping the combination of combine_mode=pointwise and compile_mode=compile_dynamic_shape
    # as the current implementation does not support lifted arguments
    @decorateIf(
        unittest.skip,
        lambda params: (
            params["combine_mode"] == "pointwise"
            and (
                params["device"] == torch.device("cpu")
                or params["compile_mode"] == "compile_dynamic_shape"
            )
        ),
    )
    def test_associative_scan_downstream_scan_matmul(
        self, combine_mode, compile_mode, reverse, device, autograd
    ):
        def first_chain_fct(scan_fct, inp, **kwargs):
            o = scan_fct(get_scan_combine_fn("add", True), inp, **kwargs)
            return o

        def second_chain_fct(scan_fct, inp, **kwargs):
            W = torch.ones(2, 5, device=device)
            return inp @ W

        inp = torch.randn(3, 10, 2, device=device, requires_grad=autograd)
        kwargs = {
            "dim": 1,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": [first_chain_fct, second_chain_fct],
            "combine_mode": combine_mode,
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.ChainFn(**kwargs),
            model_fake=AssociativeScanModels.ChainFn(**kwargs_fake),
            inputs=inp,
            autograd_param=None if not autograd else (inp,),
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("combine_mode", ["pointwise", "generic"])
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    # Skipping the combination of combine_mode=pointwise and device=cpu
    # as the current implementation of pointwise does only support CUDA device
    # Skipping the combination of combine_mode=pointwise and compile_mode=compile_dynamic_shape
    # as the current implementation does not support lifted arguments
    @decorateIf(
        unittest.skip,
        lambda params: (
            params["combine_mode"] == "pointwise"
            and (
                params["device"] == torch.device("cpu")
                or params["compile_mode"] == "compile_dynamic_shape"
            )
        ),
    )
    def test_associative_scan_downstream_scan_scan(
        self, combine_mode, compile_mode, reverse, device, autograd
    ):
        def first_chain_fct(scan_fct, inp, **kwargs):
            o1 = scan_fct(get_scan_combine_fn("add", True), inp, **kwargs)
            return o1

        def second_chain_fct(scan_fct, inp, **kwargs):
            o2 = scan_fct(get_scan_combine_fn("add", True), inp, **kwargs)
            return o2

        inp = torch.randn(3, 10, 2, device=device, requires_grad=autograd)

        kwargs = {
            "dim": 1,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": [first_chain_fct, second_chain_fct],
            "combine_mode": combine_mode,
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.ChainFn(**kwargs),
            model_fake=AssociativeScanModels.ChainFn(**kwargs_fake),
            inputs=inp,
            autograd_param=None if not autograd else (inp,),
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("combine_mode", ["pointwise", "generic"])
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("reverse_first", [False, True])
    @parametrize("same_direction", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    # Skipping the combination of combine_mode=pointwise and device=cpu
    # as the current implementation of pointwise does only support CUDA device
    # Skipping the combination of combine_mode=pointwise and compile_mode=compile_dynamic_shape
    # as the current implementation does not support lifted arguments
    @decorateIf(
        unittest.skip,
        lambda params: (
            params["combine_mode"] == "pointwise"
            and (
                params["device"] == torch.device("cpu")
                or params["compile_mode"] == "compile_dynamic_shape"
            )
        ),
    )
    # Skipping the autograd=True because
    # associative_scan does currently not support gradients for lifted parameters
    @decorateIf(
        unittest.skip,
        lambda params: (params["combine_mode"] == "pointwise" and params["autograd"]),
    )
    def test_associative_scan_downstream_scan_scan_different_dim(
        self,
        combine_mode,
        compile_mode,
        reverse_first,
        same_direction,
        device,
        autograd,
    ):
        reverse_second = reverse_first if same_direction else not reverse_first

        def first_chain_fct(scan_fct, inp, **kwargs):
            o1 = scan_fct(get_scan_combine_fn("add", True), inp, **kwargs)
            return o1

        def second_chain_fct(scan_fct, inp, **kwargs):
            o2 = scan_fct(get_scan_combine_fn("add", True), inp, **kwargs)
            return o2

        inp = torch.randn(3, 10, 2, device=device, requires_grad=autograd)

        kwargs = {
            "dim": [1, 0],
            "reverse": [reverse_first, reverse_second],
            "compile_mode": compile_mode,
            "combine_fn": [first_chain_fct, second_chain_fct],
            "combine_mode": [combine_mode, combine_mode],
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.ChainFn(**kwargs),
            model_fake=AssociativeScanModels.ChainFn(**kwargs_fake),
            inputs=inp,
            autograd_param=None if not autograd else (inp,),
        )

    # TODO: Does not work because of the usage of vmap within associative_scan
    # TODO: Re-enable additional parameters again once this issues has been resolved
    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @unittest.expectedFailure
    def test_associative_scan_nested(self):
        combine_mode = "pointwise"
        compile_mode = "eager"
        reverse_first = False
        same_direction = False
        device = torch.device("cuda")

        reverse_second = reverse_first if same_direction else not reverse_first

        def first_nested_fct(x, y):
            y_new = associative_scan(
                second_nested_fct,
                y,
                0,
                reverse=reverse_second,
                combine_mode=combine_mode,
            )
            return x + y_new

        def first_nested_fct_fake(x, y):
            y_new = _fake_associative_scan(
                second_nested_fct, y, 0, reverse=reverse_second
            )
            return x + y_new

        def second_nested_fct(x, y):
            return x * y

        inp = torch.randn(3, 10, 2, device=device)

        kwargs = {
            "dim": 0,
            "reverse": reverse_first,
            "compile_mode": compile_mode,
            "combine_fn": first_nested_fct,
            "combine_mode": combine_mode,
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        kwargs_fake["combine_fn"] = first_nested_fct_fake
        self._run_test(
            model=AssociativeScanModels.NestedFn(**kwargs),
            model_fake=AssociativeScanModels.NestedFn(**kwargs_fake),
            inputs=inp,
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("loop_type", ["for"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_associative_scan_loop_in_combine_fn(
        self, compile_mode, loop_type, reverse, device, autograd
    ):
        def combine_fn(x, y):
            cnt = torch.zeros_like(y[0, :])
            if loop_type == "while":

                def cond_fn(ind, loop_val):
                    return (loop_val < 5)[0]

                def body_fn(ind, loop_val):
                    return ind + 1, loop_val + torch.abs(ind)

                new_ind, cnt = torch.while_loop(
                    cond_fn=cond_fn,
                    body_fn=body_fn,
                    carried_inputs=(
                        torch.zeros(1, dtype=torch.int32, device=cnt.device),
                        cnt,
                    ),
                )
            else:
                for ind in range(10):
                    cnt += torch.abs(y[ind])
            return x * cnt

        inp = torch.randn(3, 10, 1, device=device, requires_grad=autograd) * 2

        kwargs = {
            "dim": 0,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": combine_fn,
            "combine_mode": "generic",
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=inp,
            autograd_param=None if not autograd else (inp,),
        )

    # TODO: Does not work because of the usage of vmap within associative_scan
    # TODO: Re-enable additional parameters again once this issues has been resolved
    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @unittest.expectedFailure
    def test_associative_scan_loop_in_combine_fn_failure(self):
        compile_mode = "none"
        loop_type = "while"
        reverse = False
        device = torch.device("cuda")

        def combine_fn(x, y):
            _cnt = torch.zeros_like(y[0, :])
            if loop_type == "while":

                def cond_fn(ind, loop_val):
                    return (loop_val < 5)[0]

                def body_fn(ind, loop_val):
                    return ind + 1, loop_val + torch.abs(ind)

        inp = torch.randn(3, 10, 1, device=device) * 2

        kwargs = {
            "dim": 0,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": combine_fn,
            "combine_mode": "generic",
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=inp,
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    # Skipping the combination of compile_mode=compile_dynamic_shape
    # as the current implementation does not support lifted arguments
    @decorateIf(
        unittest.skip,
        lambda params: (
            params["device"] == torch.device("cpu")
            or params["compile_mode"] == "compile_dynamic_shape"
        ),
    )
    def test_associative_scan_cond_in_combine_fn(
        self, compile_mode, reverse, device, autograd
    ):
        def combine_fn(x, y):
            val = cond(torch.sum(y) > 0.0, lambda y: y.clone(), lambda y: 1.0 - y, (y,))
            return x * val

        inp = torch.randn(3, 10, 1, device=device, requires_grad=autograd)

        kwargs = {
            "dim": 0,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": combine_fn,
            "combine_mode": "generic",
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=inp,
            autograd_param=None if not autograd else (inp,),
        )

    # TODO: Does not work because of the usage of vmap within associative_scan
    # TODO: Re-enable additional parameters again once this issues has been resolved
    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @unittest.expectedFailure
    def test_associative_scan_map_in_combine_fn(self):
        compile_mode = "none"
        reverse = False
        device = torch.device("cuda")

        def combine_fn(x, y):
            def body(x, y):
                return x + y

            y_init = y[0]
            y_new = control_flow.map(body, y, y_init)
            return x * y_new

        inp = torch.randn(3, 10, 1, device=device)

        kwargs = {
            "dim": 0,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": combine_fn,
            "combine_mode": "generic",
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=inp,
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_associative_scan_vmap_in_combine_fn(
        self, compile_mode, reverse, device, autograd
    ):
        def combine_fn(x, y):
            def body(x):
                return x**2

            mapped_body = torch.vmap(body, 0, 0)
            y_new = mapped_body(y)
            return x + y_new

        inp = torch.randn(3, 10, 2, device=device, requires_grad=autograd)

        kwargs = {
            "dim": 0,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": combine_fn,
            "combine_mode": "generic",
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=inp,
            autograd_param=None if not autograd else (inp,),
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    # Skipping the combination of associative_scan and device=cpu
    # as the current implementation of pointwise does only support CUDA device
    @decorateIf(
        unittest.skip,
        lambda params: (params["device"] == torch.device("cpu")),
    )
    def test_associative_scan_non_pointwise_generic(
        self, reverse, compile_mode, device, autograd
    ):
        x = torch.randn(3, 10, 2, device=device, requires_grad=autograd)

        kwargs = {
            "dim": 0,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": get_scan_combine_fn("non_pointwise", True),
            "combine_mode": "generic",
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=x,
            autograd_param=None if not autograd else (x,),
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("combine_mode", ["pointwise", "generic"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    # Skipping the combination of combine_mode=pointwise and device=cpu
    # as the current implementation of pointwise does only support CUDA device
    # Skipping the combination of combine_mode=pointwise and compile_mode=compile_dynamic_shape
    # as the current implementation does not support lifted arguments
    @decorateIf(
        unittest.skip,
        lambda params: (
            params["combine_mode"] == "pointwise"
            and (
                params["device"] == torch.device("cpu")
                or params["compile_mode"] == "compile_dynamic_shape"
            )
        ),
    )
    def test_associative_scan_binary_operator(
        self, compile_mode, combine_mode, reverse, device, autograd
    ):
        state_dim = 20
        timesteps = 10
        projected_inputs = torch.randn(
            timesteps, state_dim, device=device, requires_grad=autograd
        )
        A = torch.randn(state_dim, device=device, requires_grad=autograd)
        elements = (A.repeat((timesteps, 1)), projected_inputs)

        kwargs = {
            "dim": 0,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": get_scan_combine_fn("s5_operator", True),
            "combine_mode": combine_mode,
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=elements,
            autograd_param=None if not autograd else elements,
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    def test_associative_scan_different_input_size(self, compile_mode, reverse, device):
        batch = 5
        hidden_dim = 3
        length = 10
        dstate = 7

        deltaA = torch.randn(
            (batch, hidden_dim, length, dstate), requires_grad=True, device=device
        )
        deltaB_u = torch.randn(
            (batch, hidden_dim, length, dstate), requires_grad=True, device=device
        )
        C = torch.randn((batch, dstate, length), requires_grad=True, device=device)
        x = torch.randn(
            (batch, hidden_dim, length, dstate), requires_grad=True, device=device
        )
        y = torch.randn((batch, hidden_dim, length), requires_grad=True, device=device)
        elements = (x, deltaA, deltaB_u, C, y)

        kwargs = {
            "dim": 2,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": get_scan_combine_fn("different_input_size_operator", True),
            "combine_mode": "generic",
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=elements,
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    def test_associative_scan_different_input_size_wrong_dim(self):
        batch = 5
        hidden_dim = 3
        length = 10
        dstate = 7

        deltaA = torch.randn(
            (batch, hidden_dim, length, dstate), device=torch.device("cuda")
        )
        deltaB_u = torch.randn(
            (batch, hidden_dim, length, dstate), device=torch.device("cuda")
        )
        C = torch.randn((batch, dstate, length), device=torch.device("cuda"))
        x = torch.randn(
            (batch, hidden_dim, length, dstate), device=torch.device("cuda")
        )
        y = torch.randn(
            (batch, hidden_dim, length, dstate), device=torch.device("cuda")
        )
        elements = (x, deltaA, deltaB_u, C, y)

        with self.assertRaisesRegex(
            ValueError,
            "All xs leaves must at least have.*",
        ):
            associative_scan(
                get_scan_combine_fn("different_input_size_operator", True),
                elements,
                3,
                combine_mode="pointwise",
            )

    @unittest.skipIf(not SM70OrLater, "triton")
    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA.")
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("combine_mode", ["pointwise", "generic"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    # Skipping the combine_mode=pointwise
    # as the current implementation of associative_scan lowering
    # does not support lifted arguments
    @decorateIf(
        unittest.skip,
        lambda params: (params["combine_mode"] == "pointwise"),
    )
    def test_associative_scan_freevars_simple(
        self, compile_mode, combine_mode, reverse, device, autograd
    ):
        H = torch.rand(2, device=device, requires_grad=autograd)

        def fct_freevars1(x: torch.Tensor, y: torch.Tensor):
            return x * H + y * 2

        def fct_freevars2(x: torch.Tensor, y: torch.Tensor):
            return x * H + y * H

        H1 = torch.rand(1, device=device, requires_grad=autograd)
        H2 = torch.rand(1, device=device, requires_grad=autograd)

        def fct_freevars3(x: torch.Tensor, y: torch.Tensor):
            return x * H1 + y * H2

        inp = torch.randn(3, 2, 2, device=device, requires_grad=autograd)

        for fct, param in [
            (fct_freevars1, (H,)),
            (fct_freevars2, (H,)),
            (fct_freevars3, (H1, H2)),
        ]:
            kwargs = {
                "dim": 0,
                "reverse": reverse,
                "compile_mode": compile_mode,
                "combine_fn": fct,
                "combine_mode": combine_mode,
            }
            kwargs_fake = self._prepare_fake_kwargs(kwargs)
            self._run_test(
                model=AssociativeScanModels.CombineFn(**kwargs),
                model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
                inputs=inp,
                autograd_param=None if not autograd else (inp, *param),
            )

    @unittest.skipIf(not SM70OrLater, "triton")
    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA.")
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("combine_mode", ["pointwise", "generic"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    # Skipping the combine_mode=pointwise
    # as the current implementation of associative_scan lowering
    # does not support lifted arguments
    @decorateIf(
        unittest.skip,
        lambda params: (params["combine_mode"] == "pointwise"),
    )
    def test_associative_scan_freevars_nested(
        self, compile_mode, combine_mode, reverse, device, autograd
    ):
        H1 = torch.rand(4, 5, device=device, requires_grad=autograd)
        H2 = torch.rand(4, 1, device=device, requires_grad=autograd)

        def fct_nested_outside(x: torch.Tensor, y: torch.Tensor):
            def inner(xi):
                return xi * H2

            ret = inner(y)
            return x + ret * H1

        def fct_nested_outside_fake(x: torch.Tensor, y: torch.Tensor):
            def inner(xi):
                return xi * H2

            ret = inner(y)
            return x + ret * H1

        # TODO: Using random tensors in the `combine_fn` triggers the vmap randomness error:
        # RuntimeError: vmap: called random operation while in randomness error mode.
        # Please either use the 'same' or 'different' randomness flags on vmap or perform the randomness operation out of vmap
        def fct_nested_inside(x: torch.Tensor, y: torch.Tensor):
            H2_i = torch.ones(4, 1, device=device) * 42

            def inner(xi):
                return xi * H2_i

            ret = inner(y)
            return x + ret * H1

        def fct_nested_inside_fake(x: torch.Tensor, y: torch.Tensor):
            H2_i = torch.ones(4, 1, device=device) * 42

            def inner(xi):
                return xi * H2_i

            ret = inner(y)
            return x + ret * H1

        inp = torch.randn(3, 4, 5, device=device, requires_grad=autograd)

        for fct, fct_fake, param in [
            (fct_nested_outside, fct_nested_outside_fake, (H1, H2)),
            (fct_nested_inside, fct_nested_inside_fake, ()),
        ]:
            kwargs = {
                "dim": 0,
                "reverse": reverse,
                "compile_mode": compile_mode,
                "combine_fn": fct,
                "combine_mode": combine_mode,
            }
            kwargs_fake = self._prepare_fake_kwargs(kwargs)
            kwargs_fake["combine_fn"] = fct_fake
            self._run_test(
                model=AssociativeScanModels.CombineFn(**kwargs),
                model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
                inputs=inp,
                autograd_param=None if not autograd else (inp, *param),
            )

    @unittest.skipIf(not SM70OrLater, "triton")
    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA.")
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("combine_mode", ["pointwise", "generic"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    # Skipping the combine_mode=pointwise
    # as the current implementation of associative_scan lowering
    # does not support lifted arguments
    @decorateIf(
        unittest.skip,
        lambda params: (params["combine_mode"] == "pointwise"),
    )
    def test_associative_scan_freevars_fct(
        self, compile_mode, combine_mode, reverse, device, autograd
    ):
        def additional_fct_no_add_inp(x, y):
            return x * y

        def fct_nested_outside(x: torch.Tensor, y: torch.Tensor):
            ret = additional_fct_no_add_inp(y, y)
            return x + ret

        inp = torch.randn(3, 4, 5, device=device, requires_grad=autograd)

        kwargs = {
            "dim": 0,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": fct_nested_outside,
            "combine_mode": combine_mode,
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=inp,
            autograd_param=None if not autograd else (inp,),
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA.")
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_associative_scan_freevars_fct_generic(
        self, compile_mode, reverse, device, autograd
    ):
        def additional_fct_no_add_inp(x, y):
            return x * y

        def fct_nested_outside(x: torch.Tensor, y: torch.Tensor):
            ret = associative_scan(
                additional_fct_no_add_inp, y, 1, combine_mode="generic"
            )
            return x + ret

        def fct_nested_outside_fake(x: torch.Tensor, y: torch.Tensor):
            ret = _fake_associative_scan(additional_fct_no_add_inp, y, 1)
            return x + ret

        inp = torch.randn(3, 4, 5, device=device, requires_grad=autograd)

        kwargs = {
            "dim": 0,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": fct_nested_outside,
            "combine_mode": "generic",
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        kwargs_fake["combine_fn"] = fct_nested_outside_fake
        self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=inp,
            autograd_param=None if not autograd else (inp,),
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA.")
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("combine_mode", ["pointwise", "generic"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    # Skipping the combine_mode=pointwise
    # as the current implementation of associative_scan lowering
    # does not support lifted arguments
    @decorateIf(
        unittest.skip,
        lambda params: (params["combine_mode"] == "pointwise"),
    )
    def test_associative_scan_freevars_shape_check(
        self, compile_mode, combine_mode, reverse, device, autograd
    ):
        H = torch.eye(2, device=device, requires_grad=True)

        def fct_freevars(x: torch.Tensor, y: torch.Tensor):
            return x @ H + y

        inp = torch.randn(2, 2, 3, device=device, requires_grad=True)

        kwargs = {
            "dim": 2,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": fct_freevars,
            "combine_mode": combine_mode,
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=inp,
            autograd_param=None if not autograd else (inp,),
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA.")
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("combine_mode", ["pointwise", "generic"])
    @parametrize("autograd", [False, True])
    # Skipping the combine_mode=pointwise
    # as the current implementation of associative_scan lowering
    # does not support lifted arguments
    @decorateIf(
        unittest.skip,
        lambda params: (params["combine_mode"] == "pointwise"),
    )
    def test_associative_scan_freevars_pytree(
        self, compile_mode, combine_mode, reverse, device, autograd
    ):
        xf = torch.randn(2, 2, device=device, requires_grad=autograd)
        yf = torch.randn(2, 2, device=device, requires_grad=autograd)
        zf = torch.randn(2, 2, device=device, requires_grad=autograd)
        inpf = {"i": xf, "j": ([yf], [{"o": zf}])}

        def fct_pointwise(x, y):
            return {
                "i": (x["i"] * y["i"]) + inpf["i"],
                "j": (
                    [(x["j"][0][0] * y["j"][0][0]) + inpf["j"][0][0]],
                    [
                        {
                            "o": (x["j"][1][0]["o"] + y["j"][1][0]["o"])
                            + inpf["j"][1][0]["o"]
                        }
                    ],
                ),
            }

        x = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        y = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        z = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        inp = {"i": x, "j": ([y], [{"o": z}])}

        kwargs = {
            "dim": 0,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": fct_pointwise,
            "combine_mode": combine_mode,
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=inp,
            autograd_param=None if not autograd else (*pytree.tree_leaves(inp),),
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("combine_mode", ["pointwise", "generic"])
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    # Skipping the combination of combine_mode=pointwise and device=cpu
    # as the current implementation of pointwise does only support CUDA device
    # Skipping the combination of combine_mode=pointwise and compile_mode=compile_dynamic_shape
    # as the current implementation does not support lifted arguments
    @decorateIf(
        unittest.skip,
        lambda params: (
            params["combine_mode"] == "pointwise"
            and (
                params["device"] == torch.device("cpu")
                or params["compile_mode"] == "compile_dynamic_shape"
                or torch.version.hip
            )
        ),
    )
    def test_associative_scan_partial_grad(
        self, combine_mode, compile_mode, reverse, device
    ):
        import random

        n_params = 6
        autograds = []
        autograds.append([True, True, True, True, True, True])
        autograds.append([False, False, False, False, False, False])
        autograds.append([False, True, False, False, False, False])
        for _ in range(5):
            autograds.append([bool(random.randint(0, 1)) for _ in range(n_params)])

        def mul2(x, y):
            return (*[xv * yv for xv, yv in zip(x, y)],)

        for a_grads in autograds:
            inp = tuple(
                [
                    torch.randn(10, 3, 2, device=device, requires_grad=a_grads[n])
                    for n in range(n_params)
                ]
            )

            kwargs = {
                "dim": 0,
                "reverse": reverse,
                "compile_mode": compile_mode,
                "combine_fn": mul2,
                "combine_mode": combine_mode,
            }
            kwargs_fake = self._prepare_fake_kwargs(kwargs)
            self._run_test(
                model=AssociativeScanModels.CombineFn(**kwargs),
                model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
                inputs=inp,
                autograd_param=inp,
            )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("combine_mode", ["pointwise", "generic"])
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    # Skipping the combination of combine_mode=pointwise and device=cpu
    # as the current implementation of pointwise does only support CUDA device
    # Skipping the combination of combine_mode=pointwise and compile_mode=compile_dynamic_shape
    # as the current implementation does not support lifted arguments
    @decorateIf(
        unittest.skip,
        lambda params: (
            params["combine_mode"] == "pointwise"
            and (
                params["device"] == torch.device("cpu")
                or params["compile_mode"] == "compile_dynamic_shape"
                or torch.version.hip
            )
        ),
    )
    def test_associative_scan_partial_grad_no_grad(
        self, combine_mode, compile_mode, reverse, device
    ):
        def mul_single_nograd(x, y):
            xy1 = x[0] * y[0]
            with torch.no_grad():
                xy2 = x[1] * y[1]
            return xy1, xy2

        inp = tuple(
            [torch.randn(10, 3, 2, device=device, requires_grad=True) for n in range(2)]
        )

        kwargs = {
            "dim": 0,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": mul_single_nograd,
            "combine_mode": combine_mode,
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=inp,
            autograd_param=inp[0:1],
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    def test_associative_scan_sparse_tensor(self):
        x = torch.tensor(
            [[[0.0, 0], [1.0, 2.0]], [[0.0, 0], [3.0, 4.0]], [[0.0, 0], [5.0, 6.0]]]
        ).to_sparse()

        with self.assertRaisesRegex(
            ValueError,
            "xs leaves must dense Tensors.*",
        ):
            associative_scan(
                get_scan_combine_fn("add", True), x, 0, combine_mode="generic"
            )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    def test_associative_scan_combine_fn_wrong_meta_in_combine_fn(self):
        device = torch.device("cuda")
        B, N, C, H, W = 3, 3, 2, 3, 3
        x = torch.randn(B, N, C, H, W, device=device)

        def fct_wrong_dtype(x, y):
            return (x + y).to(torch.int64)

        def fct_wrong_device(x, y):
            return (x + y).to(
                torch.device("cpu") if device.type == "cuda" else torch.device("cuda")
            )

        def fct_wrong_stride(x, y):
            return (x + y).to(memory_format=torch.channels_last)

        for fct in [fct_wrong_dtype, fct_wrong_device, fct_wrong_stride]:
            with self.assertRaisesRegex(
                torch._dynamo.exc.UncapturedHigherOrderOpError,
                "Expected initial_xs and combine_fn_output to have same metadata.*",
            ):
                associative_scan(fct, x, 0)

    @unittest.skipIf(not SM70OrLater, "triton")
    def test_associative_scan_wrong_pytree(self):
        def fct_wrong_pytree(x, y):
            return {
                "i": x["i"] * y["j"][0][0],
                "k": torch.tensor(0.0),
                "j": ([x["j"][1][0]["o"]], [{"o": torch.sin(x["i"])}]),
            }

        x = torch.randn(3, 2, 2)
        y = torch.randn(3, 2, 2)
        z = torch.randn(3, 2, 2)
        inp = {"i": x, "j": ([y], [{"o": z}])}

        with self.assertRaisesRegex(
            AssertionError,
            "Combine_fn received wrong number of arguments.*",
        ):
            associative_scan(fct_wrong_pytree, inp, 0, combine_mode="generic")

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    def test_associative_scan_non_pointwise(self):
        device = torch.device("cuda")
        x = torch.randn(3, 10, 2, device=device)
        with self.assertRaisesRegex(
            # Should be:
            RuntimeError,
            r"For combine_mode='pointwise', the combine_fn needs to be pointwise",
        ):
            associative_scan(
                get_scan_combine_fn("non_pointwise", True),
                x,
                0,
                combine_mode="pointwise",
            )

    @requires_cuda
    def test_associative_scan_input_mutation(self):
        device = torch.device("cuda")

        def fct_input_mutation(x, y):
            x.add_(1)
            return x + y

        x = torch.randn(3, 2, 2, device=device)

        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.Unsupported,
            # "Encountered aliasing during higher order op tracing for HOP.*"
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.ops\.higher_order\.associative_scan",
        ):
            associative_scan(fct_input_mutation, x, 0)

    @requires_cuda
    def test_associative_scan_input_output_alias(self):
        device = torch.device("cuda")

        def fct_input_output_alias(x, y):
            return x[0], x[1] + y[1]

        x = torch.randn(3, 2, 2, device=device)
        y = torch.randn(3, 2, 2, device=device)
        inp = (x, y)

        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.Unsupported,
            # "Encountered aliasing during higher order op tracing for HOP.*"
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.ops\.higher_order\.associative_scan",
        ):
            associative_scan(fct_input_output_alias, inp, 0)

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    def test_associative_scan_output_output_alias(self):
        device = torch.device("cuda")

        def fct_output_output_alias(x, y):
            c = x[0] + y[1]
            return c, c

        x = torch.randn(3, 2, 2, device=device)
        y = torch.randn(3, 2, 2, device=device)
        inp = (x, y)

        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.Unsupported,
            # "Encountered aliasing during higher order op tracing for HOP.*"
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.ops\.higher_order\.associative_scan",
        ):
            associative_scan(fct_output_output_alias, inp, 0)


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

    def _check_compile(self, fn, args, *, dynamic=False, backend="eager"):
        eager_res = fn(*args)
        compiled_fn = torch.compile(fn, backend=backend, dynamic=dynamic)
        self.assertEqual(compiled_fn(*args), eager_res)

    def _check_export(self, fn, args, *, strict=False, dynamic_shapes=None):
        eg_out = fn(*args)
        with torch._export.config.patch(use_new_tracer_experimental=True):
            ep = torch.export.export(
                fn, args, strict=strict, dynamic_shapes=dynamic_shapes
            )
        ep_out = ep.module()(*args)
        self.assertEqual(eg_out, ep_out)
        return ep

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

    @torch._dynamo.config.patch(trace_autograd_ops=True)
    @skipIfTorchDynamo("Graph is not captured by backend if test with dynamo")
    @skipIfCrossRef  # Arg order changes with crossref
    def test_cond_simple_with_linear_compile_check_graph(self):
        def true_fn(x):
            return x.sin()

        def false_fn(x):
            return x.cos()

        x = torch.randn(4, requires_grad=True)

        def f(pred, x):
            result = cond(pred, true_fn, false_fn, (x,))
            grad_out = torch.ones_like(result)
            return torch.autograd.grad(result, (x,), grad_out)

        backend = EagerAndRecordGraphs()
        torch.compile(f, backend=backend)(torch.tensor(False), x)
        # With autograd.grad tracing support, the entire function is traced
        # into a single graph instead of breaking into forward + backward graphs
        self.assertEqual(len(backend.graphs), 1)
        gm = backend.graphs[0]

        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, L_pred_ : torch.Tensor, L_x_ : torch.Tensor):
    l_pred_ = L_pred_
    l_x_ = L_x_
    cond_true_0 = self.cond_true_0
    cond_false_0 = self.cond_false_0
    cond = torch.ops.higher_order.cond(l_pred_, cond_true_0, cond_false_0, (l_x_,));  l_pred_ = cond_true_0 = cond_false_0 = None
    result = cond[0];  cond = None
    grad_out = torch.ones_like(result)
    grad = torch.autograd.grad(result, (l_x_,), grad_out);  result = l_x_ = grad_out = None
    getitem_1 = grad[0];  grad = None
    return (getitem_1,)""",  # noqa: B950
        )

    def test_while_loop_op_mismatch_in_meta(self):
        class Mod(torch.nn.Module):
            def forward(self, c, a, b):
                def cond_fn(c, a, b):
                    return c > 0

                def body_fn(c, a, b):
                    return c - 1, a.nonzero(), b.nonzero()

                return torch.ops.higher_order.while_loop(
                    cond_fn,
                    body_fn,
                    (c, a, b),
                    tuple(),
                )

        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "Expected carried_inputs and body_output to have same metadata but found",
        ):
            make_fx(Mod(), tracing_mode="fake")(
                torch.tensor(
                    0,
                ),
                torch.randn(2, 3),
                torch.randn(2, 3),
            )

    @unittest.skipIf(
        not TEST_CUDA_GRAPH_CONDITIONAL_NODES,
        "CUDA 12.4 or greater is required for CUDA Graphs with conditional nodes",
    )
    def test_cond_traced_not_nested_cudagraphs(self):
        def true_fn(x):
            return x.sin()

        def false_fn(x):
            return x.cos()

        def f(x, y):
            return cond(y, true_fn, false_fn, [x])

        x = torch.randn(4).cuda()
        true_pred = torch.tensor(True).cuda()
        false_pred = torch.tensor(False).cuda()

        _check_compile_cudagraph_backend(self, f, [x, true_pred])
        _check_compile_cudagraph_backend(self, f, [x, false_pred])
        _check_compile_many_backends_with_cudagraph(
            self,
            f,
            [x, true_pred],
        )
        _check_compile_many_backends_with_cudagraph(
            self,
            f,
            [x, false_pred],
        )

    @unittest.skipIf(
        not TEST_CUDA_GRAPH_CONDITIONAL_NODES,
        "CUDA 12.4 or greater is required for CUDA Graphs with conditional nodes",
    )
    def test_cond_pin_memory_cudagraphs(self):
        # Make sure that pinned host memory allocations get assigned
        # to a private pool correctly during stream capture, even
        # inside of conditional nodes.

        # Ideally, we would call torch.Tensor.pin_memory() directly,
        # but that is not allowed on fake tensors, so instead we call
        # an op whose C++ implementation used pinned memory
        # internally.
        sizes = [3, 4, 3]

        def true_fn(x):
            return torch.split_with_sizes_copy(x, sizes)

        def false_fn(x):
            return torch.split_with_sizes_copy(2 * x, sizes)

        def f(x, y):
            return cond(y, true_fn, false_fn, [x])

        x = torch.randn(10).cuda()
        true_pred = torch.tensor(True).cuda()
        false_pred = torch.tensor(False).cuda()

        _check_compile_cudagraph_backend(self, f, [x, true_pred])
        _check_compile_cudagraph_backend(self, f, [x, false_pred])
        _check_compile_many_backends_with_cudagraph(
            self,
            f,
            [x, true_pred],
        )
        _check_compile_many_backends_with_cudagraph(
            self,
            f,
            [x, false_pred],
        )

    @unittest.skipIf(
        not TEST_CUDA_GRAPH_CONDITIONAL_NODES,
        "CUDA 12.4 or greater is required for CUDA Graphs with conditional nodes",
    )
    def test_cond_traced_triply_nested_cudagraphs(self):
        def level3_true(x):
            return x.sin()

        def level3_false(x):
            return x.cos()

        def level2_true(x, p2):
            return cond(p2, level3_true, level3_false, [x])

        def level2_false(x, p2):
            return cond(p2, lambda x: x + 1, lambda x: x - 1, [x])

        def level1_true(x, p1, p2):
            return cond(p1, level2_true, level2_false, [x, p2])

        def level1_false(x, p1, p2):
            return cond(p1, level2_false, level2_true, [x, p2])

        def f(x, p0, p1, p2):
            return cond(p0, level1_true, level1_false, [x, p1, p2])

        x = torch.randn(4).cuda()

        test_inputs = [
            [
                x,
                torch.tensor(True).cuda(),
                torch.tensor(True).cuda(),
                torch.tensor(False).cuda(),
            ],
            [
                x,
                torch.tensor(False).cuda(),
                torch.tensor(False).cuda(),
                torch.tensor(True).cuda(),
            ],
        ]

        for args in test_inputs:
            _check_compile_cudagraph_backend(self, f, args)
            _check_compile_many_backends_with_cudagraph(self, f, args)

    @unittest.skipIf(
        not TEST_CUDA_GRAPH_CONDITIONAL_NODES,
        "CUDA 12.4 or greater is required for CUDA Graphs with conditional nodes",
    )
    def test_cond_traced_record_stream_reuse(self):
        torch.cuda.memory._set_allocator_settings(
            "graph_capture_record_stream_reuse:True"
        )
        try:
            predicate = torch.tensor(True, device="cuda")

            def true_fn():
                return torch.zeros(8, device="cuda"), torch.zeros(8, device="cuda")

            def false_fn():
                return torch.zeros(8, device="cuda"), torch.zeros(8, device="cuda")

            g = torch.cuda.CUDAGraph()
            with self.assertRaisesRegex(
                RuntimeError,
                "graph_capture_record_stream_reuse:True",
            ):
                with torch.cuda.graph(g), CUDAGraphCaptureControlFlowOpDispatchMode():
                    torch.cond(predicate, true_fn, false_fn, [])
        finally:
            torch.cuda.memory._set_allocator_settings(
                "graph_capture_record_stream_reuse:False"
            )

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

    def test_while_loop_pytree_carry(self):
        fn, inp = WHILE_LOOP_TESTS["simple_with_pytree_carry"]
        backend = EagerAndRecordGraphs()
        expected_res = fn(*inp)
        compiled_res = torch.compile(fn, backend=backend)(*inp)
        self.assertEqual(expected_res, compiled_res)
        # When test with torch dynamo, the graph is not captured because
        # it's traced together with the code before torch.compile
        if not TEST_WITH_TORCHDYNAMO:
            self.assertEqual(len(backend.graphs), 1)
            self.assertExpectedInline(
                backend.graphs[0].code.strip(),
                """\
def forward(self, L_it_ : torch.Tensor, L_pytree_input_0_0_ : torch.Tensor, L_pytree_input_1_x_ : torch.Tensor, L_pytree_input_1_y_ : torch.Tensor):
    l_it_ = L_it_
    l_pytree_input_0_0_ = L_pytree_input_0_0_
    l_pytree_input_1_x_ = L_pytree_input_1_x_
    l_pytree_input_1_y_ = L_pytree_input_1_y_
    cond_fn_0 = self.cond_fn_0
    body_fn_0 = self.body_fn_0
    while_loop = torch.ops.higher_order.while_loop(cond_fn_0, body_fn_0, (l_it_, l_pytree_input_0_0_, l_pytree_input_1_x_, l_pytree_input_1_y_), ());  cond_fn_0 = body_fn_0 = l_it_ = l_pytree_input_0_0_ = l_pytree_input_1_x_ = l_pytree_input_1_y_ = None
    getitem = while_loop[0]
    getitem_1 = while_loop[1]
    value = while_loop[2]
    value_1 = while_loop[3];  while_loop = None
    return (getitem, getitem_1, value, value_1)""",  # noqa: B950
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
            if func_type != "no":
                raise AssertionError(
                    f"Expected func_type to be 'no', got {func_type!r}"
                )
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
    # - "simple_with_linear" and "nested_with_linear" doesn't work because parameters and buffers
    #   are not inputs so they're not wrapped by functionalization and tracing.
    #
    # - make_fx tracing mode "real" fails for "int_carry", "pytree_int_carry" and "const_and_symint_output"
    #   because tensors are real but we unspecialize the ints with unbacked symints causing
    #   data dependent errors.
    #   Since this is not the common use path, we skip them for now.
    @parametrize(
        "while_loop_test",
        set(WHILE_LOOP_TESTS.keys())
        - {
            "simple_with_linear",
            "nested_with_linear",
            "int_carry",
            "pytree_int_carry",
            "const_and_symint_output",
        },
    )
    def test_while_loop_functionalize(self, func_type, while_loop_test):
        fn, inp = WHILE_LOOP_TESTS[while_loop_test]
        fn, mode = self._wrap_with_functionalize(fn, func_type)
        mode = mode if mode is not None else contextlib.nullcontext()
        with mode:
            self._check_tracing(fn, inp)

    # - make_fx tracing mode "real" fails for "int_carry", "pytree_int_carry" and "const_and_symint_output"
    #   because tensors are real but we unspecialize the ints with unbacked symints causing
    #   data dependent errors.
    #   Since this is not the common use path, we skip them for now.
    @parametrize(
        "while_loop_test",
        set(WHILE_LOOP_TESTS.keys())
        - {"int_carry", "pytree_int_carry", "const_and_symint_output"},
    )
    def test_while_loop_tracing(self, while_loop_test):
        fn, inp = WHILE_LOOP_TESTS[while_loop_test]
        allow_non_fake_inputs = while_loop_test in (
            "simple_with_linear",
            "nested_with_linear",
        )
        self._check_tracing(fn, inp, allow_non_fake_inputs)

    @parametrize("backend", ["eager", "aot_eager"])
    @parametrize("while_loop_test", list(WHILE_LOOP_TESTS.keys()))
    def test_while_loop_compile(self, backend, while_loop_test):
        fn, inp = WHILE_LOOP_TESTS[while_loop_test]
        self._check_compile(fn, inp, backend=backend)

    @skipIfTorchDynamo("Graph is not captured by backend if test with dynamo")
    @skipIfCrossRef  # Arg order changes with cross ref
    def test_while_loop_simple_with_linear_compile_check_graph(self):
        fn, inp = WHILE_LOOP_TESTS["simple_with_linear"]
        backend = EagerAndRecordGraphs()
        torch.compile(fn, backend=backend)(*inp)
        self.assertEqual(len(backend.graphs), 1)
        gm = backend.graphs[0]
        if torch._dynamo.config.inline_inbuilt_nn_modules:
            self.assertExpectedInline(
                normalize_gm(gm.print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_iter_: "i64[]", L_x_: "f32[2, 2]", L_self_buffers_dec_: "i64[]", L_self_modules_linear_parameters_weight_: "f32[2, 2]", L_self_modules_linear_parameters_bias_: "f32[2]"):
        l_iter_ = L_iter_
        l_x_ = L_x_
        l_self_buffers_dec_ = L_self_buffers_dec_
        l_self_modules_linear_parameters_weight_ = L_self_modules_linear_parameters_weight_
        l_self_modules_linear_parameters_bias_ = L_self_modules_linear_parameters_bias_

        cond_fn_0 = self.cond_fn_0
        body_fn_0 = self.body_fn_0
        while_loop = torch.ops.higher_order.while_loop(cond_fn_0, body_fn_0, (l_iter_, l_x_), (l_self_buffers_dec_, l_self_modules_linear_parameters_bias_, l_self_modules_linear_parameters_weight_));  cond_fn_0 = body_fn_0 = l_iter_ = l_x_ = l_self_buffers_dec_ = l_self_modules_linear_parameters_bias_ = l_self_modules_linear_parameters_weight_ = None
        getitem: "i64[]" = while_loop[0]
        getitem_1: "f32[2, 2]" = while_loop[1];  while_loop = None
        return (getitem, getitem_1)

    class cond_fn_0(torch.nn.Module):
        def forward(self, child: "i64[]", child_1: "f32[2, 2]", l_self_buffers_dec__cond_fn: "i64[]", l_self_modules_linear_parameters_bias__body_fn: "f32[2]", l_self_modules_linear_parameters_weight__body_fn: "f32[2, 2]"):
            sub: "i64[]" = child - l_self_buffers_dec__cond_fn;  child = l_self_buffers_dec__cond_fn = None
            gt: "b8[]" = sub > 0;  sub = None
            return gt

    class body_fn_0(torch.nn.Module):
        def forward(self, child_2: "i64[]", child_3: "f32[2, 2]", l_self_buffers_dec__cond_fn: "i64[]", l_self_modules_linear_parameters_bias__body_fn: "f32[2]", l_self_modules_linear_parameters_weight__body_fn: "f32[2, 2]"):
            child: "i64[]" = child_2 - 1;  child_2 = None
            child_4: "f32[2, 2]" = torch._C._nn.linear(child_3, l_self_modules_linear_parameters_weight__body_fn, l_self_modules_linear_parameters_bias__body_fn);  child_3 = l_self_modules_linear_parameters_weight__body_fn = l_self_modules_linear_parameters_bias__body_fn = None
            return (child, child_4)
""",  # noqa: B950
            )

    def test_while_loop_nested2_traced(self):
        fn, inp = WHILE_LOOP_TESTS["nested2"]
        graphs = self._check_tracing(fn, inp)
        gm = graphs["symbolic"]
        outer_body = gm.while_loop_body_graph_0
        inner_body = outer_body.while_loop_body_graph_0
        inner_cond = outer_body.while_loop_cond_graph_0
        self.assertExpectedInline(
            gm.code.strip("\n"),
            """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
    sym_size_int = torch.ops.aten.sym_size.int(arg3_1, 1)
    sym_size_int_1 = torch.ops.aten.sym_size.int(arg2_1, 1)
    sym_size_int_2 = torch.ops.aten.sym_size.int(arg2_1, 0)
    sym_size_int_3 = torch.ops.aten.sym_size.int(arg3_1, 0)
    while_loop_cond_graph_0 = self.while_loop_cond_graph_0
    while_loop_body_graph_0 = self.while_loop_body_graph_0
    while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_0, while_loop_body_graph_0, (arg0_1, arg1_1, arg2_1, arg3_1), (sym_size_int, sym_size_int_1, sym_size_int_2, sym_size_int_3));  while_loop_cond_graph_0 = while_loop_body_graph_0 = arg0_1 = arg1_1 = arg2_1 = arg3_1 = sym_size_int = sym_size_int_1 = sym_size_int_2 = sym_size_int_3 = None
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
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1):
    while_loop_cond_graph_0 = self.while_loop_cond_graph_0
    while_loop_body_graph_0 = self.while_loop_body_graph_0
    while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_0, while_loop_body_graph_0, (arg0_1, arg1_1, arg2_1, arg3_1), (arg7_1, arg7_1, arg7_1, arg7_1));  while_loop_cond_graph_0 = while_loop_body_graph_0 = arg0_1 = arg1_1 = arg2_1 = arg3_1 = arg7_1 = None
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
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1):
    while_loop_cond_graph_0 = self.while_loop_cond_graph_0
    while_loop_body_graph_0 = self.while_loop_body_graph_0
    while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_0, while_loop_body_graph_0, (arg0_1, arg1_1, arg2_1, arg3_1), (arg7_1, arg7_1, arg7_1, arg7_1));  while_loop_cond_graph_0 = while_loop_body_graph_0 = arg0_1 = arg1_1 = arg2_1 = arg3_1 = arg7_1 = None
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
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1):
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
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1):
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

        if TEST_CUDA_GRAPH_CONDITIONAL_NODES:
            x_cuda = x.cuda()
            true_pred = torch.tensor(True).cuda()

            _check_compile_cudagraph_backend(
                self,
                f,
                [x_cuda, true_pred, true_pred],
            )
            _check_compile_many_backends_with_cudagraph(
                self,
                f,
                [x_cuda, true_pred, true_pred],
            )

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

        def f_(x, y):
            return cond(y, true_fn, false_fn, [x])

        example_inputs = (torch.ones(4, 5),)
        functional_f = torch.func.functionalize(f)
        self.assertEqual(functional_f(*example_inputs), f(*example_inputs))

        graph_module = make_fx(torch.func.functionalize(f), tracing_mode="symbolic")(
            *example_inputs
        )
        self.assertEqual(graph_module(*example_inputs), f(*example_inputs))

        all_ops_in_true_branch = []
        for node in graph_module.true_graph_0.graph.nodes:
            if node.op == "call_function":
                all_ops_in_true_branch.append(node.target)

        self.assertFalse(any(op._schema.is_mutable for op in all_ops_in_true_branch))

        self.assertEqual(graph_module(*example_inputs), f(*example_inputs))

        if TEST_CUDA_GRAPH_CONDITIONAL_NODES:
            pred = torch.tensor(example_inputs[0].shape[0] == 1, device="cuda")
            _check_compile_cudagraph_backend(self, f_, [torch.ones(4, 5).cuda(), pred])
            _check_compile_many_backends_with_cudagraph(
                self, f_, [torch.ones(4, 5).cuda(), pred]
            )

    def test_cond_accepts_torch_function_as_inputs(self):
        a = torch.randn(3, 4)
        b = torch.randn(3, 4)

        def f(a, b):
            return cond(a.sum() > 0, torch.add, torch.mul, (a, b))

        gm = self._check_tracing(f, (a, b))["symbolic"]
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, a_1, b_1):
    sum_1 = torch.ops.aten.sum.default(a_1)
    gt = torch.ops.aten.gt.Scalar(sum_1, 0);  sum_1 = None
    sym_size_int = torch.ops.aten.sym_size.int(a_1, 1)
    sym_size_int_1 = torch.ops.aten.sym_size.int(b_1, 0)
    sym_size_int_2 = torch.ops.aten.sym_size.int(b_1, 1)
    sym_size_int_3 = torch.ops.aten.sym_size.int(a_1, 0)
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, (a_1, b_1, sym_size_int, sym_size_int_1, sym_size_int_2, sym_size_int_3));  gt = true_graph_0 = false_graph_0 = a_1 = b_1 = sym_size_int = sym_size_int_1 = sym_size_int_2 = sym_size_int_3 = None
    getitem = cond[0];  cond = None
    return getitem""",  # noqa: B950
        )
        self.assertExpectedInline(
            gm.true_graph_0.code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1):
    add = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
    return (add,)""",
        )
        self.assertExpectedInline(
            gm.false_graph_0.code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1):
    mul = torch.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
    return (mul,)""",
        )

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

    def test_cond_subgraph_same_shape_env_as_parent(self):
        def true_fn(x):
            return x.sin() + 10

        def false_fn(x):
            return x.cos() - 20

        def f(x, pred):
            y = cond(pred, true_fn, false_fn, [x])
            z = torch.add(y, y)
            return z

        symbolic_traced_graph = self._check_tracing(
            f, (torch.ones(4), torch.Tensor([True]))
        )["symbolic"]
        graph_shape_env = symbolic_traced_graph.shape_env

        def _node_shape_env_iter(gm):
            for node in symbolic_traced_graph.graph.nodes:
                if node.op == "call_function":
                    val = node.meta.get("val")
                    if isinstance(val, tuple):
                        for v in val:
                            yield v.fake_mode.shape_env
                    elif isinstance(val, torch.SymInt):
                        yield val.node.shape_env
                    else:
                        yield val.fake_mode.shape_env

        for shape_env in _node_shape_env_iter(symbolic_traced_graph):
            self.assertTrue(shape_env is graph_shape_env)

        for shape_env in _node_shape_env_iter(symbolic_traced_graph.true_graph_0):
            self.assertTrue(shape_env is graph_shape_env)

        for shape_env in _node_shape_env_iter(symbolic_traced_graph.false_graph_0):
            self.assertTrue(shape_env is graph_shape_env)

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

        graph_module = make_fx(torch.func.functionalize(f), tracing_mode="symbolic")(
            *example_inputs
        )
        self.assertEqual(graph_module(*example_inputs), f(*example_inputs))

        gm_true_true_branch = graph_module.true_graph_0.true_graph_0

        self.assertEqual(graph_module(*example_inputs), f(*example_inputs))

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
        # torch.cond inlines into one of the branches because the predicate
        # is a constant.
        gm = make_fx(torch.func.functionalize(f))(*example_inputs)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x_1):
    view = torch.ops.aten.view.default(x_1, [4, 5])
    add = torch.ops.aten.add.Tensor(view, 1);  view = None
    view_1 = torch.ops.aten.view.default(add, [4, 5]);  add = None
    view_2 = torch.ops.aten.view.default(view_1, [4, 5])
    sin = torch.ops.aten.sin.default(view_2);  view_2 = None
    sum_1 = torch.ops.aten.sum.default(sin);  sin = None
    copy_ = torch.ops.aten.copy_.default(x_1, view_1);  x_1 = view_1 = copy_ = None
    return sum_1""",
        )

        # torch.cond triggers the check of the branches because the predicate
        # is a SymBool.
        with self.assertRaisesRegex(
            torch._dynamo.exc.TorchRuntimeError,
            "cond_true might be modifying the input!",
        ):
            make_fx(torch.func.functionalize(f), tracing_mode="symbolic")(
                *example_inputs
            )

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
        gm = make_fx(torch.func.functionalize(f))(*example_inputs)
        # torch.cond inlines into one of the branches because the predicate
        # is a constant.
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x_1):
    view = torch.ops.aten.view.default(x_1, [5, 5])
    add = torch.ops.aten.add.Tensor(view, 1);  view = None
    view_1 = torch.ops.aten.view.default(add, [5, 5]);  add = None
    view_2 = torch.ops.aten.view.default(view_1, [5, 5])
    cos = torch.ops.aten.cos.default(view_2);  view_2 = None
    sum_1 = torch.ops.aten.sum.default(cos);  cos = None
    copy_ = torch.ops.aten.copy_.default(x_1, view_1);  x_1 = view_1 = copy_ = None
    return sum_1""",
        )

        # torch.cond triggers the check of the branches because the predicate
        # is a SymBool.
        with self.assertRaisesRegex(
            torch._dynamo.exc.TorchRuntimeError,
            "cond_false might be modifying the input!",
        ):
            make_fx(torch.func.functionalize(f), tracing_mode="symbolic")(
                *example_inputs
            )

    def test_cond_functionalized_output_alias_input(self):
        def true_fn(x):
            return x.clone()

        def false_fn(x):
            view_x = x.view(x.shape)
            return view_x

        def f(x):
            pred = x.shape[0] == 4
            return cond(pred, true_fn, false_fn, [x])

        example_inputs = (torch.ones(5, 5),)
        gm = make_fx(torch.func.functionalize(f))(*example_inputs)
        # torch.cond inlines into one of the branches because the predicate
        # is a constant.
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x_1):
    view = torch.ops.aten.view.default(x_1, [5, 5]);  x_1 = None
    return view""",
        )

        # torch.cond triggers the check of the branches because the predicate
        # is a SymBool.
        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.Unsupported,
            # "Encountered aliasing during higher order op tracing for HOP.*"
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.cond",
        ):
            make_fx(torch.func.functionalize(f), tracing_mode="symbolic")(
                *example_inputs
            )

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
        with self.assertRaisesRegex(
            torch._dynamo.exc.TorchRuntimeError,
            "cond_true might be modifying the input!",
        ):
            make_fx(torch.func.functionalize(f), tracing_mode="symbolic")(
                *example_inputs
            )

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
            f(example_input_func)

            with self.assertRaisesRegex(
                # Should be
                # torch._dynamo.exc.Unsupported,
                # "Encountered aliasing during higher order op tracing for HOP.*"
                torch._dynamo.exc.UncapturedHigherOrderOpError,
                r"Higher Order Operator: torch\.cond",
            ):
                make_fx(f, tracing_mode="symbolic")(example_input_func)
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
            # Should be
            # torch._dynamo.exc.Unsupported,
            # "Encountered aliasing during higher order op tracing for HOP.*"
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.cond",
        ):
            make_fx(f_wrapper(f), tracing_mode="symbolic")(example_input_func)

    def test_cond_functionalized_input_aliasing_with_aot_func(self):
        def true_fn(x):
            return x

        def false_fn(x):
            view_x = x.view(x.shape)
            return view_x

        def f(x):
            pred = x.sum() > 0
            return cond(pred, true_fn, false_fn, [x])

        example_input = torch.ones(5, 5)
        try:
            example_input_func = to_fun_old(example_input)
            torch._enable_functionalization(reapply_views=False)
            with self.assertRaisesRegex(
                # Should be
                # torch._dynamo.exc.Unsupported,
                # "Encountered aliasing during higher order op tracing for HOP.*"
                torch._dynamo.exc.UncapturedHigherOrderOpError,
                r"Higher Order Operator: torch\.cond",
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
            # Should be
            # torch._dynamo.exc.Unsupported,
            # "Encountered aliasing during higher order op tracing for HOP.*"
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.cond",
        ):
            make_fx(f_wrapper(f), tracing_mode="symbolic")(example_input)

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

        result_gm = make_fx(f_wrapper(f), tracing_mode="symbolic")(example_input)
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
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (x_1,));  pred_1 = true_graph_0 = false_graph_0 = None
    getitem = cond[0];  cond = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    cond_1 = torch.ops.higher_order.cond(pred2_1, true_graph_1, false_graph_1, (x_1,));  pred2_1 = true_graph_1 = false_graph_1 = x_1 = None
    getitem_1 = cond_1[0];  cond_1 = None
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

        if TEST_CUDA_GRAPH_CONDITIONAL_NODES:
            _check_compile_cudagraph_backend(
                self,
                f,
                [x.cuda(), torch.tensor(False).cuda(), torch.tensor(False).cuda()],
            )
            _check_compile_many_backends_with_cudagraph(
                self,
                f,
                [x.cuda(), torch.tensor(False).cuda(), torch.tensor(False).cuda()],
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
            # Should be
            # torch._dynamo.exc.Unsupported,
            # "Encountered aliasing during higher order op tracing for HOP.*"
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.cond",
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
            torch._dynamo.exc.TorchRuntimeError,
            "When merging two branches' output in torch.cond",
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
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (x_1,));  pred_1 = true_graph_0 = false_graph_0 = None
    getitem = cond[0];  cond = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    cond_1 = torch.ops.higher_order.cond(pred2_1, true_graph_1, false_graph_1, (x_1,));  pred2_1 = true_graph_1 = false_graph_1 = x_1 = None
    getitem_1 = cond_1[0];  cond_1 = None
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
            # Should be
            # torch._dynamo.exc.Unsupported,
            # "Encountered aliasing during higher order op tracing for HOP.*"
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.cond",
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
            torch._dynamo.exc.TorchRuntimeError,
            "When merging two branches' output in torch.cond",
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
            torch._dynamo.exc.TorchRuntimeError,
            "map might be modifying the input!",
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
            torch._dynamo.exc.TorchRuntimeError, "map might be modifying the input!"
        ):
            functional_f(*example_inputs)

    def test_cond_autograd_backward(self):
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
        f(*example_inputs).sum().backward()

        # Ensure no error is thrown when not running backward
        res = f(*example_inputs)

        # Ensure no error is thrown when not running backward
        res_compiled = torch.compile(f)(*example_inputs)
        self.assertEqual(res, res_compiled)

    @skipIfTorchDynamo("Skip because we're testing export")
    def test_cond_autograd_backward_inp_out_aliasing(self):
        from torch._dynamo.testing import AotEagerAndRecordGraphs

        backend = AotEagerAndRecordGraphs()

        def fn(x, y):
            return x + y

        def f(x, y):
            return control_flow.cond(x.sum() > 4, fn, fn, (x, y))

        example_inputs = (
            torch.ones(3, 4, requires_grad=True),
            torch.ones(3, 4, requires_grad=True),
        )
        res = f(*example_inputs)
        res.sum().backward()
        res_compiled = torch.compile(f, backend=backend)(*example_inputs)
        res_compiled.sum().backward()
        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.bw_graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[3, 4]", primals_2: "f32[3, 4]", gt: "b8[]", tangents_1: "f32[3, 4]"):
        true_graph_1 = self.true_graph_1
        false_graph_1 = self.false_graph_1
        cond_1 = torch.ops.higher_order.cond(gt, true_graph_1, false_graph_1, (primals_1, primals_2, tangents_1));  gt = true_graph_1 = false_graph_1 = primals_1 = primals_2 = tangents_1 = None
        getitem_1: "f32[3, 4]" = cond_1[0]
        getitem_2: "f32[3, 4]" = cond_1[1];  cond_1 = None
        return (getitem_1, getitem_2)

    class true_graph_1(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 4]", arg1_1: "f32[3, 4]", arg2_1: "f32[3, 4]"):
            clone: "f32[3, 4]" = torch.ops.aten.clone.default(arg2_1)
            clone_1: "f32[3, 4]" = torch.ops.aten.clone.default(arg2_1);  arg2_1 = None
            return [clone, clone_1]

    class false_graph_1(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 4]", arg1_1: "f32[3, 4]", arg2_1: "f32[3, 4]"):
            clone: "f32[3, 4]" = torch.ops.aten.clone.default(arg2_1)
            clone_1: "f32[3, 4]" = torch.ops.aten.clone.default(arg2_1);  arg2_1 = None
            return [clone, clone_1]
""",  # noqa: B950
            )
        self.assertEqual(res, res_compiled)

    @skipIfTorchDynamo("Skip because we're testing export")
    def test_cond_autograd_backward_out_out_aliasing(self):
        from torch._dynamo.testing import AotEagerAndRecordGraphs

        backend = AotEagerAndRecordGraphs()

        def fn(x, y):
            return (x + y).sin()

        def f(x, y):
            return control_flow.cond(x.sum() > 4, fn, fn, (x, y))

        example_inputs = (
            torch.ones(3, 4, requires_grad=True),
            torch.ones(3, 4, requires_grad=True),
        )
        res = f(*example_inputs)
        res.sum().backward()
        res_compiled = torch.compile(f, backend=backend)(*example_inputs)
        res_compiled.sum().backward()
        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.bw_graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[3, 4]", primals_2: "f32[3, 4]", gt: "b8[]", tangents_1: "f32[3, 4]"):
        true_graph_1 = self.true_graph_1
        false_graph_1 = self.false_graph_1
        cond_1 = torch.ops.higher_order.cond(gt, true_graph_1, false_graph_1, (primals_1, primals_2, tangents_1));  gt = true_graph_1 = false_graph_1 = primals_1 = primals_2 = tangents_1 = None
        getitem_1: "f32[3, 4]" = cond_1[0]
        getitem_2: "f32[3, 4]" = cond_1[1];  cond_1 = None
        return (getitem_1, getitem_2)

    class true_graph_1(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 4]", arg1_1: "f32[3, 4]", arg2_1: "f32[3, 4]"):
            add: "f32[3, 4]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
            cos: "f32[3, 4]" = torch.ops.aten.cos.default(add);  add = None
            mul: "f32[3, 4]" = torch.ops.aten.mul.Tensor(arg2_1, cos);  arg2_1 = cos = None
            clone: "f32[3, 4]" = torch.ops.aten.clone.default(mul)
            return [mul, clone]

    class false_graph_1(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 4]", arg1_1: "f32[3, 4]", arg2_1: "f32[3, 4]"):
            add: "f32[3, 4]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
            cos: "f32[3, 4]" = torch.ops.aten.cos.default(add);  add = None
            mul: "f32[3, 4]" = torch.ops.aten.mul.Tensor(arg2_1, cos);  arg2_1 = cos = None
            clone: "f32[3, 4]" = torch.ops.aten.clone.default(mul)
            return [mul, clone]
""",  # noqa: B950
            )
        self.assertEqual(res, res_compiled)

    def test_map_functionalized_elem_alias(self):
        def map_fn(x):
            x.view(x.shape)
            return x

        def f(xs):
            return control_flow.map(map_fn, xs)

        example_inputs = (torch.ones(3, 2, 4),)
        functional_f = torch.func.functionalize(f)
        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.Unsupported,
            # "Encountered aliasing during higher order op tracing.*"
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.ops\.higher_order\.map_impl",
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
    eq = sym_size_int == 4
    sym_size_int_1 = torch.ops.aten.sym_size.int(x_1, 1)
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(eq, true_graph_0, false_graph_0, (x_1, sym_size_int_1, sym_size_int));  eq = true_graph_0 = false_graph_0 = x_1 = sym_size_int_1 = sym_size_int = None
    getitem = cond[0];  cond = None
    return getitem""",  # noqa: B950
        )

        # We expect the traced graph module to work even if input size changes.
        x = torch.ones(4, 3, 2)
        self.assertEqual(gm(x), true_fn(x))
        self.assertEqual(foo(x), true_fn(x))

    def test_cond_with_unbacked_sym_pred(self):
        def foo(x):
            def true_fn(x):
                return x + x

            def false_fn(x):
                return x * x

            az = x.nonzero()
            return cond(az.shape[0] > 3, true_fn, false_fn, (x,))

        gm = make_fx(foo, tracing_mode="symbolic")(torch.randn(7))
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x_1):
    nonzero = torch.ops.aten.nonzero.default(x_1)
    sym_size_int = torch.ops.aten.sym_size.int(nonzero, 0);  nonzero = None
    gt = sym_size_int > 3;  sym_size_int = None
    sym_size_int_1 = torch.ops.aten.sym_size.int(x_1, 0)
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, (x_1, sym_size_int_1));  gt = true_graph_0 = false_graph_0 = x_1 = sym_size_int_1 = None
    getitem = cond[0];  cond = None
    return getitem""",  # noqa: B950
        )

    def _check_closure_correctly_lifted(self, f, *, args, exp_res, exp_arg_num):
        if not isinstance(args, (tuple, list)):
            raise AssertionError(f"Expected args to be tuple or list, got {type(args)}")
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

        gm = make_fx(foo, tracing_mode="symbolic", _allow_non_fake_inputs=True)(inp)

        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x_1):
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0)
    eq = sym_size_int == 4
    sym_size_int_1 = torch.ops.aten.sym_size.int(x_1, 1)
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    _tensor_constant0 = self._tensor_constant0
    _tensor_constant1 = self._tensor_constant1
    cond = torch.ops.higher_order.cond(eq, true_graph_0, false_graph_0, (x_1, _tensor_constant0, sym_size_int_1, sym_size_int, _tensor_constant1));  eq = true_graph_0 = false_graph_0 = x_1 = _tensor_constant0 = sym_size_int_1 = sym_size_int = _tensor_constant1 = None
    getitem = cond[0];  cond = None
    return getitem""",  # noqa: B950
        )
        self.assertExpectedInline(
            gm.true_graph_0.code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1):
    add = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
    return (add,)""",
        )

    def test_cond_with_module_param_closure(self):
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_parameter(
                    "param", torch.nn.Parameter(torch.ones(2, 3), requires_grad=False)
                )
                self.buffer = torch.nn.Buffer(torch.ones(2, 3) + 1)

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
    select_copy = torch.ops.aten.select_copy.int(x_1, 0, 0);  select_copy = None
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
    cond = torch.ops.higher_order.cond(arg1_1, true_graph_0, false_graph_0, (arg0_1,));  arg1_1 = true_graph_0 = false_graph_0 = arg0_1 = None
    getitem = cond[0];  cond = None
    return (getitem,)""",  # noqa: B950
        )

    @skipIfCrossRef  # Arg order changes with crossref
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
            return cond(x.sum() < 0, true_fn, false_fn, (x,))

        inp = torch.ones(3, 4)
        exp_out = inp.sin()
        iter_n = torch._dynamo.config.recompile_limit + 1

        # Need functions that cause recompilations
        def get_dummy_fns(str):
            def dummy_cos(x):
                return x.cos() + len(str) - len(str)

            def dummy_sin(x):
                return x.sin() + len(str) - len(str)

            return dummy_cos, dummy_sin

        for i in range(iter_n):
            # we fail guards each iter because `str(i)` is different
            self.assertEqual(foo(inp, *get_dummy_fns(str(i))), exp_out)

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
            gm = make_fx(foo, tracing_mode="symbolic")(inp)
            self.assertExpectedInline(
                gm.code.strip(),
                """\
def forward(self, x_1):
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0)
    eq = sym_size_int == 4
    sym_size_int_1 = torch.ops.aten.sym_size.int(x_1, 1)
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(eq, true_graph_0, false_graph_0, (x_1, sym_size_int_1, sym_size_int));  eq = true_graph_0 = false_graph_0 = x_1 = sym_size_int_1 = sym_size_int = None
    getitem = cond[0];  cond = None
    return getitem""",  # noqa: B950
            )

            self.assertExpectedInline(
                gm.true_graph_0.code.strip(),
                """\
def forward(self, arg0_1, arg1_1, arg2_1):
    cos = torch.ops.aten.cos.default(arg0_1)
    sub = torch.ops.aten.sub.Tensor(arg0_1, cos);  arg0_1 = cos = None
    return (sub,)""",
            )

            self.assertExpectedInline(
                gm.false_graph_0.code.strip(),
                """\
def forward(self, arg0_1, arg1_1, arg2_1):
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
                false_fn=lambda x: x.clone(),
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
                false_fn=lambda x, y: y.clone(),
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
                    false_fn=lambda x: (x.clone(), x.clone()),
                    operands=(x,),
                )

        else:

            def fn(x):
                return torch.cond(
                    pred=torch.tensor([True]),
                    true_fn=lambda x: (x + 1, x - 1),
                    false_fn=lambda x: (x.clone(), x.clone()),
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

    @parametrize("boolcond", [True, False])
    def test_vmap_vmap(self, boolcond):
        def fn(x):
            return torch.cond(
                pred=torch.tensor([True]) if not boolcond else True,
                true_fn=lambda x: x + 1,
                false_fn=lambda x: x - 1,
                operands=(x,),
            )

        def wrapper(x):
            return torch.vmap(fn)(x)

        a = torch.ones((3, 4, 5))
        res = torch.vmap(wrapper)(a)
        self.assertEqual(res, a + 1)

    def test_cond_trace_set__and_mutate_input(self):
        def f(a, tmp):
            a_view = a.view(-1)
            with torch.no_grad():
                a.set_(tmp)
                a_view.mul_(2)
            return a + tmp

        inp = torch.ones(3, 3, requires_grad=True)
        tmp = torch.ones(3, 3, requires_grad=True)
        # graph break: torch._dynamo.exc.Unsupported: call_function DelayGraphBreakVariable() [TensorVariable()] {}
        # due to set_
        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.cond",
        ):
            torch.cond(inp.sum() > 0, f, f, (inp, tmp))

    @skipIfCrossRef  # Arg order changes with crossref
    def test_cond_trace_set__and_mutate_intermediate(self):
        def f(a, tmp):
            a = a.clone()
            a_view = a.view(-1)
            tmp = tmp.clone()
            with torch.no_grad():
                a.set_(tmp)
                a_view.mul_(2)
            return a + tmp

        inp = torch.ones(3, 3, requires_grad=True)
        tmp = torch.ones(3, 3, requires_grad=True)

        class Mod(torch.nn.Module):
            def forward(self, inp: torch.Tensor, tmp: torch.Tensor) -> torch.Tensor:
                return torch.cond(inp.sum() > 0, f, f, (inp, tmp))

        with self.assertRaisesRegex(
            RuntimeError, "cannot mutate tensors with frozen storage"
        ):
            out = torch.compile(Mod(), backend="aot_eager")(inp, tmp)

        with self.assertRaisesRegex(
            RuntimeError, "cannot mutate tensors with frozen storage"
        ):
            out = torch.compile(Mod(), backend="inductor")(inp, tmp)

        backend = EagerAndRecordGraphs()
        out = torch.compile(Mod(), backend=backend)(inp, tmp)
        self.assertExpectedInline(
            backend.graphs[0].cond_true_0.code.strip("\n"),
            """\
def forward(self, l_inp_, l_tmp_):
    l_inp__1 = l_inp_
    l_tmp__1 = l_tmp_
    a = l_inp__1.clone();  l_inp__1 = None
    a_view = a.view(-1)
    tmp = l_tmp__1.clone();  l_tmp__1 = None
    _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None
    set_ = a.set_(tmp);  set_ = None
    mul_ = a_view.mul_(2);  a_view = mul_ = None
    _set_grad_enabled_1 = torch._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
    add = a + tmp;  a = tmp = None
    return (add,)
    """,
        )
        self.assertEqual(out, f(inp, tmp))

    @skipIfCrossRef  # Args get renamed to r in crossref mode
    @parametrize("requires_grad", [True, False])
    def test_cond_symint_operands(self, requires_grad):
        backend = EagerAndRecordGraphs()

        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num = 3

            def forward(self, a, b):
                return torch.cond(
                    pred=torch.tensor([True]),
                    true_fn=lambda a, b: a + b + self.num,
                    false_fn=lambda a, b: a - b - self.num,
                    operands=(a, b),
                )

        a = torch.ones(3, 3, requires_grad=requires_grad)
        b = torch.ones(3, 3, requires_grad=requires_grad)
        out = torch.compile(Mod(), backend=backend, dynamic=True)(a, b)
        self.assertEqual(out, Mod()(a, b))
        self.assertEqual(len(backend.graphs), 1)
        self.assertExpectedInline(
            backend.graphs[0].code.strip(),
            """\
def forward(self, s97 : torch.SymInt, L_a_ : torch.Tensor, L_b_ : torch.Tensor):
    l_a_ = L_a_
    l_b_ = L_b_
    tensor = torch.tensor([True])
    cond_true_0 = self.cond_true_0
    cond_false_0 = self.cond_false_0
    cond = torch.ops.higher_order.cond(tensor, cond_true_0, cond_false_0, (l_a_, l_b_, s97));  tensor = cond_true_0 = cond_false_0 = l_a_ = l_b_ = s97 = None
    getitem = cond[0];  cond = None
    return (getitem,)""",  # noqa: B950
        )

    def test_two_hops_not_sharing_code_obj(self):
        pred, args = torch.tensor(True), (torch.ones(3, 3),)

        def fn1(x):
            return x + 1

        def fn2(x):
            return x - 1

        from torch._dynamo.testing import CompileCounter

        # Tests rely on automatic_dynamic = True
        with torch._dynamo.config.patch(automatic_dynamic_shapes=True):
            cnt = CompileCounter()
            torch.compile(torch.cond, backend=cnt)(pred, fn1, fn2, args)
            self.assertEqual(cnt.frame_count, 1)

            args = (torch.randn(3, 3),)
            # No recompilation
            torch.compile(torch.cond, backend=cnt)(pred, fn1, fn2, args)
            self.assertEqual(cnt.frame_count, 1)

            def cond_fn(x):
                return x.sum() > 0

            args = (torch.randn(4, 4),)
            torch.compile(torch.while_loop, backend=cnt)(cond_fn, fn2, args)
            # recompilation
            self.assertEqual(cnt.frame_count, 2)

            args = (torch.randn(4, 4),)
            torch.compile(torch.while_loop, backend=cnt)(cond_fn, fn2, args)
            self.assertEqual(cnt.frame_count, 2)

            # With recompilation due to automatic dynamic
            # This also proves that while_loop doesn't share code obj with cond
            torch.compile(torch.cond, backend=cnt)(pred, fn1, fn2, (torch.randn(4, 4),))
            self.assertEqual(cnt.frame_count, 3)

    def test_hop_raises_if_not_overriding_call(self):
        class WrongHop(torch._ops.HigherOrderOperator):
            pass

        with self.assertRaisesRegex(TypeError, "WrongHop"):
            WrongHop("wrong_hop")

    def test_scan_functionalized(self):
        def f(init, xs):
            return scan(get_scan_combine_fn("add", False), init, xs, dim=1)

        example_inputs = torch.ones(5, 7, 4)
        example_init = torch.ones(5, 4)
        functional_f = torch.func.functionalize(f)
        self.assertEqual(
            functional_f(example_init, example_inputs), f(example_init, example_inputs)
        )

    def test_scan_functionalized_elem_mutation(self):
        def add1(x, y):
            x.add_(4)
            return x + y, x + y

        def f(init, xs):
            return scan(add1, init, xs, dim=1)

        example_inputs = torch.ones(5, 7, 4)
        example_init = torch.ones(5, 4)
        functional_f = torch.func.functionalize(f)
        with self.assertRaisesRegex(
            # TODO: Fix this so that the HOPs show similar errors for functionalization
            # This is the Exception with PYTORCH_TEST_WITH_DYNAMO=0
            # RuntimeError,
            # "torch.scan might be modifying the input!",
            # This is the Exception with PYTORCH_TEST_WITH_DYNAMO=1
            # torch._dynamo.exc.TorchDynamoException,
            # "Unexpected exception when running generated GraphModule.*"
            Exception,
            ".*",
        ):
            functional_f(example_init, example_inputs)

        def add2(x, y):
            y.add_(4)
            return x + y, x + y

        def f(init, xs):
            return scan(add2, init, xs, dim=1)

        functional_f = torch.func.functionalize(f)
        with self.assertRaisesRegex(
            # TODO: Fix this so that the HOPs show similar errors for functionalization
            # Should be
            # This is the Exception with PYTORCH_TEST_WITH_DYNAMO=0
            # RuntimeError,
            # "torch.scan might be modifying the input!",
            # This is the Exception with PYTORCH_TEST_WITH_DYNAMO=1
            # torch._dynamo.exc.TorchDynamoException,
            # "Unexpected exception when running generated GraphModule.*"
            Exception,
            ".*",
        ):
            functional_f(example_init, example_inputs)

    def test_scan_functionalized_elem_alias(self):
        def add(x, y):
            return x, x

        def f(init, xs):
            return scan(add, init, xs, dim=1)

        example_inputs = torch.ones(5, 7, 4)
        example_init = torch.ones(5, 4)
        functional_f = torch.func.functionalize(f)
        with self.assertRaisesRegex(
            # TODO: Fix this so that the HOPs show similar errors for functionalization
            # Should be
            # This is the Exception with PYTORCH_TEST_WITH_DYNAMO=0
            # torch._dynamo.exc.Unsupported,
            # "Encountered aliasing during higher order op tracing for HOP.*"
            # This is the Exception with PYTORCH_TEST_WITH_DYNAMO=1
            # torch._dynamo.exc.UncapturedHigherOrderOpError,
            # r"Higher Order Operator: torch\.ops\.higher_order\.scan",
            Exception,
            ".*",
        ):
            functional_f(example_init, example_inputs)

    @skipIfTorchDynamo("Graph is not captured by backend if test with dynamo")
    def test_scan_pytree_closure(self):
        param_buffer = ({"param": torch.randn(3, 3)}, (torch.randn(3),))

        def add(carry, x):
            ret = (carry @ param_buffer[0]["param"]) @ x + param_buffer[1][0]
            return ret, ret.sum()

        def f(init, xs):
            return scan(add, init, xs)

        init = torch.randn(4, 3)
        xs = torch.randn(3, 3, 3)

        backend = EagerAndRecordGraphs()
        eager_out = f(init, xs)
        compiled_out = torch.compile(f, backend=backend)(init, xs)
        exp_out = _fake_scan(add, init, xs)

        self.assertEqual(len(backend.graphs), 1)
        if TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                backend.graphs[0].code.strip(),
                """\
def forward(self, L_init_ : torch.Tensor, L_xs_ : torch.Tensor, L_add_closure_0_cell_contents_0_param_ : torch.Tensor, L_add_closure_0_cell_contents_1_0_ : torch.Tensor):
    l_init_ = L_init_
    l_xs_ = L_xs_
    l_add_closure_0_cell_contents_0_param_ = L_add_closure_0_cell_contents_0_param_
    l_add_closure_0_cell_contents_1_0_ = L_add_closure_0_cell_contents_1_0_
    scan_combine_fn_0 = self.scan_combine_fn_0
    scan = torch.ops.higher_order.scan(scan_combine_fn_0, [l_init_], [l_xs_], [l_add_closure_0_cell_contents_0_param_, l_add_closure_0_cell_contents_1_0_]);  scan_combine_fn_0 = l_init_ = l_xs_ = l_add_closure_0_cell_contents_0_param_ = l_add_closure_0_cell_contents_1_0_ = None
    carry = scan[0]
    out = scan[1];  scan = None
    return (carry, out)""",  # noqa: B950
            )
        else:
            self.assertExpectedInline(
                backend.graphs[0].code.strip(),
                """\
def forward(self, L_init_ : torch.Tensor, L_xs_ : torch.Tensor, L_add_closure_0_cell_contents_0_param_ : torch.Tensor, L_add_closure_0_cell_contents_1_0_ : torch.Tensor):
    l_init_ = L_init_
    l_xs_ = L_xs_
    l_add_closure_0_cell_contents_0_param_ = L_add_closure_0_cell_contents_0_param_
    l_add_closure_0_cell_contents_1_0_ = L_add_closure_0_cell_contents_1_0_
    scan_combine_fn_0 = self.scan_combine_fn_0
    scan = torch.ops.higher_order.scan(scan_combine_fn_0, [l_init_], [l_xs_], [l_add_closure_0_cell_contents_0_param_, l_add_closure_0_cell_contents_1_0_]);  scan_combine_fn_0 = l_init_ = l_xs_ = l_add_closure_0_cell_contents_0_param_ = l_add_closure_0_cell_contents_1_0_ = None
    carry = scan[0]
    out = scan[1];  scan = None
    return (carry, out)""",  # noqa: B950
            )
        self.assertEqual(eager_out, exp_out)
        self.assertEqual(compiled_out, exp_out)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_scan_in_vmap_simple(self):
        x = torch.randn(3, 4, 4)
        y = torch.randn(4, 2)
        zeros = torch.zeros(2, 3)

        def combine_fn(init, xs):
            return init.clone(), xs @ y

        def fn(scan_op, x, y):
            def inner_fn(zeros, x, y):
                x = x.view(2, 2, 4)

                return scan_op(
                    combine_fn,
                    zeros,
                    x,
                )

            return torch.vmap(inner_fn, in_dims=(1, 0, None))(zeros, x, y)

        out = fn(scan, x, y)
        compile_out = torch.compile(fn)(scan, x, y)
        exp = fn(_fake_scan, x, y)
        self.assertEqual(out, exp)
        self.assertEqual(out, compile_out)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_scan_in_vmap_complex_ops(self):
        # Test with various operations requiring shape reasoning
        x = torch.randn(4, 5, 3, 2)
        init = torch.randn(4, 3, 2)
        weight = torch.randn(3, 3)

        def combine_fn(carry, xs):
            # carry: (3, 2), xs: (3, 2)
            intermediate = torch.nn.functional.relu(carry)
            xs_t = xs.transpose(0, 1)  # (2, 3)
            result = xs_t @ weight  # (2, 3)
            new_carry = intermediate + result.transpose(0, 1)  # Back to (3, 2)
            output = torch.sin(carry).sum() + torch.cos(xs).mean()
            return new_carry, output

        def fn(scan_op, x, init):
            def inner_fn(x, init):
                return scan_op(combine_fn, init, x)

            return torch.vmap(inner_fn, in_dims=(0, 0))(x, init)

        out = fn(scan, x, init)
        compile_out = torch.compile(fn)(scan, x, init)
        exp = fn(_fake_scan, x, init)

        self.assertEqual(out, exp)
        self.assertEqual(compile_out, exp)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_scan_in_vmap_unbatched_x(self):
        # Test with various operations requiring shape reasoning
        x = torch.randn(5, 3, 2)
        init = torch.randn(4, 3, 2)
        weight = torch.randn(3, 3)

        def combine_fn(carry, xs):
            # carry: (3, 2), xs: (3, 2)
            intermediate = torch.nn.functional.relu(carry)
            xs_t = xs.transpose(0, 1)  # (2, 3)
            result = xs_t @ weight  # (2, 3)
            new_carry = intermediate + result.transpose(0, 1)  # Back to (3, 2)
            output = torch.sin(carry).sum() + torch.cos(xs).mean()
            return new_carry, output

        def fn(scan_op, x, init):
            def inner_fn(x, init):
                return scan_op(combine_fn, init, x)

            return torch.vmap(inner_fn, in_dims=(None, 0))(x, init)

        out = fn(scan, x, init)
        compile_out = torch.compile(fn)(scan, x, init)
        exp = fn(_fake_scan, x, init)

        self.assertEqual(out, exp)
        self.assertEqual(compile_out, exp)

    @skipIfTorchDynamo("not a dynamo test")
    def test_scan_in_vmap_unbatched_init_error(self):
        # Test with various operations requiring shape reasoning
        x = torch.randn(4, 5, 3, 2)
        init = torch.randn(4, 3, 2)
        weight = torch.randn(3, 3)

        def combine_fn(carry, xs):
            # carry: (3, 2), xs: (3, 2)
            intermediate = torch.nn.functional.relu(carry)
            xs_t = xs.transpose(0, 1)  # (2, 3)
            result = xs_t @ weight  # (2, 3)
            new_carry = intermediate + result.transpose(0, 1)  # Back to (3, 2)
            output = torch.sin(carry).sum() + torch.cos(xs).mean()
            return new_carry, output

        def vmap_fn(x, init):
            def fn(x, init):
                return scan(combine_fn, init, x)

            return torch.vmap(fn, in_dims=(0, None))(x, init)

        with self.assertRaisesRegex(
            RuntimeError,
            """The size of tensor a \\(4\\) must match the size of tensor b \\(2\\) at non-singleton dimension 4""",
        ):
            vmap_fn(x, init)

    @skipIfTorchDynamo("a vmap test, not a dynamo test")
    def test_vmap_closure_weight_error(self):
        init_batched = torch.randn(7, 2, 3)
        xs_batched = torch.randn(7, 5, 4)
        weight = torch.randn(7, 4, 3)

        def combine_fn(carry, xs):
            # carry: (2, 3), xs: (4,), weight: (4, 3)
            new_carry = carry + xs @ weight
            output = carry.sum()
            return new_carry, output

        def expected_fn(init, xs, weight):
            def fn(init, xs, weight):
                return _fake_scan(combine_fn, init, xs)

            return torch.vmap(fn, in_dims=(0, 0, 0))(init, xs, weight)

        # Note that even though weight is vampped but combine_fn is accessing
        # the closure weight instead of the wrapped out weight thus causing
        # a shape mismatch.
        with self.assertRaisesRegex(
            RuntimeError,
            """The size of tensor a \\(2\\) must match the size of tensor b \\(7\\) at non-singleton dimension 1""",
        ):
            expected_fn(init_batched, xs_batched, weight)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_scan_in_vmap_mixed_batch_dims(self):
        init = torch.randn(8, 5, 6)
        xs_batched = torch.randn(3, 6, 5, 8)
        scale = torch.randn([])

        def combine_fn(carry, xs):
            # carry: 8, 5
            # xs: 5, 8
            # new_carry: 8, 5
            new_carry = carry + (xs * scale).sum()
            output = xs @ carry
            return new_carry, output

        def fn(scan_op, init, xs):
            def inner_fn(init, xs):
                return scan_op(combine_fn, init, xs)

            return torch.vmap(inner_fn, in_dims=(2, 1))(init, xs)

        out = fn(scan, init, xs_batched)
        compile_out = torch.compile(fn)(scan, init, xs_batched)
        exp = fn(_fake_scan, init, xs_batched)

        self.assertEqual(out, exp)
        self.assertEqual(compile_out, exp)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_vmap_scan_vmap_scan_nested(self):
        # Outer batch: 3, inner batch: 4, outer scan: 5, inner scan: 6
        init = torch.randn(3, 4, 2, 8)
        xs_outer = torch.randn(3, 5, 4, 6, 2)

        def fn(scan_op, init, xs):
            def inner_combine(carry, xs):
                # carry: (2, 8), xs: (2,)
                new_carry = carry + xs.unsqueeze(-1)
                output = carry.sum(dim=0)  # (8,)
                return new_carry, output

            def outer_combine(init, xs):
                # carry: (4, 2, 8,), xs: (4, 6, 2)
                # xs has batch dimension 4 from outer vmap

                def inner_fn(init, xs):
                    # init: (2, 8)
                    # xs: (6, 2)
                    # final_carry: (2, 8)
                    # outputs: (6, 8)
                    final_carry, outputs = scan_op(inner_combine, init, xs)
                    return (final_carry.sum(0, keepdim=True) + outputs).sum(
                        dim=0
                    )  # (8,)

                inner_results = torch.vmap(inner_fn)(init, xs)  # (4, 8)
                new_carry = init + inner_results.mean(dim=0)  # (8,)
                output = inner_results.sum(dim=0)  # (8,)
                return new_carry.expand(*init.size()), output

            def vmap_inner_fn(init, xs):
                # init: (4, 2, 8)
                # xs: (5, 4, 6, 2)
                return scan_op(outer_combine, init, xs)

            return torch.vmap(vmap_inner_fn)(init, xs)

        out = fn(scan, init, xs_outer)
        compile_out = torch.compile(fn)(scan, init, xs_outer)
        exp = fn(_fake_scan, init, xs_outer)

        self.assertEqual(out, exp)
        self.assertEqual(compile_out, exp)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_scan_vmap_scan_nested(self):
        xs_outer = torch.randn(5, 3, 4, 2)
        init_outer = torch.randn(3, 8)

        def fn(scan_op, init, xs):
            def inner_combine_fake(carry, xs):
                # carry: 8
                # xs: 2
                new_carry = carry + xs.sum()
                output = carry * 2
                return new_carry, output

            def outer_combine_fake(carry, xs):
                # carry: 3, 8
                # xs: 3, 4, 2
                def inner_fn(carry_elem, xs_elem):
                    # carry_elem: 8
                    # xs: 4, 2
                    # final_carry: 8
                    # outputs.sum(0): 8
                    final_carry, outputs = _fake_scan(
                        inner_combine_fake, carry_elem, xs_elem
                    )
                    return outputs.sum(0), final_carry

                # result: (8,)
                # next_carry, (3, 8))
                result, next_carry = torch.vmap(inner_fn, in_dims=(0, 0))(carry, xs)
                output = result.sum(dim=0)
                return next_carry, output

            return scan_op(outer_combine_fake, init, xs)

        out = fn(scan, init_outer, xs_outer)
        compile_out = torch.compile(fn)(scan, init_outer, xs_outer)
        exp = fn(_fake_scan, init_outer, xs_outer)

        self.assertEqual(out, exp)
        self.assertEqual(compile_out, exp)

    @skipIfTorchDynamo("Skip because we're testing export")
    @parametrize("strict", [True, False])
    @parametrize("dynamic", [True, False])
    def test_while_loop_op_int_carry_export(self, strict, dynamic):
        m, args = WHILE_LOOP_TESTS["int_carry"]
        dynamic_shapes = {"x": {0: torch.export.Dim("dim_x")}} if dynamic else None
        ep = self._check_export(m, args, strict=strict, dynamic_shapes=dynamic_shapes)
        if not strict and dynamic:
            self.assertExpectedInline(
                normalize_gm(ep.module().print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, x):
        x: "f32[s77, 3]";

        x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
        _guards_fn = self._guards_fn(x);  _guards_fn = None
        sym_size_int_1: "Sym(s77)" = torch.ops.aten.sym_size.int(x, 0)

        while_loop_cond_graph_0 = self.while_loop_cond_graph_0
        while_loop_body_graph_0 = self.while_loop_body_graph_0
        while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_0, while_loop_body_graph_0, (0, x), ());  while_loop_cond_graph_0 = while_loop_body_graph_0 = x = None
        getitem_2: "Sym(u1)" = while_loop[0]
        ge: "Sym(u1 >= 1)" = getitem_2 >= 1
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u1 >= 1 on node 'ge'");  ge = _assert_scalar_default = None
        gt_1: "Sym(u1 > 0)" = getitem_2 > 0
        _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(gt_1, "Runtime assertion failed for expression 0 < u1 on node 'gt_1'");  gt_1 = _assert_scalar_default_1 = None
        getitem_1: "f32[s77, 3]" = while_loop[1];  while_loop = None

        add: "Sym(u1 + 1)" = getitem_2 + 1

        add_1: "f32[s77, 3]" = torch.ops.aten.add.Tensor(getitem_1, getitem_2);  getitem_1 = None

        lt: "Sym(u1 < s77)" = getitem_2 < sym_size_int_1;  sym_size_int_1 = None

        mul: "Sym(2*u1)" = getitem_2 * 2;  getitem_2 = None
        ones: "f32[2*u1]" = torch.ops.aten.ones.default([mul], device = device(type='cpu'), pin_memory = False);  mul = None
        return pytree.tree_unflatten((add, add_1, lt, ones), self._out_spec)

    class while_loop_cond_graph_0(torch.nn.Module):
        def forward(self, it_1: "Sym(u0)", x_1: "f32[s77, 3]"):
            sym_size_int_1: "Sym(s77)" = torch.ops.aten.sym_size.int(x_1, 0);  x_1 = None

            lt: "Sym(u0 < s77)" = it_1 < sym_size_int_1;  it_1 = sym_size_int_1 = None
            return lt

    class while_loop_body_graph_0(torch.nn.Module):
        def forward(self, it_1: "Sym(u0)", x_1: "f32[s77, 3]"):
            clone: "f32[s77, 3]" = torch.ops.aten.clone.default(x_1);  x_1 = None
            select: "f32[3]" = torch.ops.aten.select.int(clone, 0, it_1)
            select_1: "f32[3]" = torch.ops.aten.select.int(clone, 0, it_1)
            add: "f32[3]" = torch.ops.aten.add.Tensor(select_1, it_1);  select_1 = None
            copy_: "f32[3]" = torch.ops.aten.copy_.default(select, add);  select = add = copy_ = None
            add_1: "Sym(u0 + 1)" = it_1 + 1;  it_1 = None
            return (add_1, clone)
""",  # noqa: B950
            )

    @skipIfTorchDynamo("Graph is not captured correctly when test with dynamo")
    @parametrize("dynamic", [True, False])
    @parametrize("backend", ["eager", "aot_eager"])
    def test_while_loop_op_int_carry_compile(self, dynamic, backend):
        m, args = WHILE_LOOP_TESTS["int_carry"]
        if backend == "eager":
            backend = EagerAndRecordGraphs()
        self._check_compile(m, args, dynamic=dynamic, backend=backend)
        if (
            isinstance(backend, EagerAndRecordGraphs)
            and dynamic
            and not TEST_WITH_CROSSREF
        ):
            self.assertEqual(len(backend.graphs), 1)
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, s77: "Sym(s77)", s27: "Sym(s27)", L_x_: "f32[s77, s27]"):
        l_x_ = L_x_

        cond_fn_0 = self.cond_fn_0
        body_fn_0 = self.body_fn_0
        while_loop = torch.ops.higher_order.while_loop(cond_fn_0, body_fn_0, (0, l_x_), (s27, s77));  cond_fn_0 = body_fn_0 = l_x_ = s27 = None
        getitem_4: "Sym(u2)" = while_loop[0]
        ge: "Sym(u2 >= 1)" = getitem_4 >= 1
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u2 >= 1 on node 'ge'");  ge = _assert_scalar_default = None
        gt_1: "Sym(u2 > 0)" = getitem_4 > 0
        _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(gt_1, "Runtime assertion failed for expression 0 < u2 on node 'gt_1'");  gt_1 = _assert_scalar_default_1 = None
        out_x: "f32[s77, s27]" = while_loop[1];  while_loop = None

        gt: "Sym(u2 > 0)" = getitem_4 > 0
        _check = torch._check(gt);  gt = _check = None

        add: "Sym(u2 + 1)" = getitem_4 + 1

        add_1: "f32[s77, s27]" = getitem_4 + out_x;  out_x = None

        lt: "Sym(u2 < s77)" = getitem_4 < s77;  s77 = None

        mul: "Sym(2*u2)" = getitem_4 * 2;  getitem_4 = None
        ones: "f32[2*u2]" = torch.ones(mul);  mul = None
        return (add, add_1, lt, ones)

    class cond_fn_0(torch.nn.Module):
        def forward(self, unbacked_symint: "Sym(u0)", child: "f32[s77, s27]", s27: "Sym(s27)", s77: "Sym(s77)"):
            s27_1 = s27
            s77_1 = s77

            sym_size_int: "Sym(s77)" = torch.ops.aten.sym_size.int(child, 0)

            size = child.size();  child = size = None
            lt: "Sym(u0 < s77)" = unbacked_symint < sym_size_int;  unbacked_symint = sym_size_int = None
            return lt

    class body_fn_0(torch.nn.Module):
        def forward(self, unbacked_symint_0: "Sym(u1)", child_1: "f32[s77, s27]", s27: "Sym(s27)", s77: "Sym(s77)"):
            s27_1 = s27
            s77_1 = s77

            sym_size_int: "Sym(s77)" = torch.ops.aten.sym_size.int(child_1, 0)

            x_clone: "f32[s77, s27]" = child_1.clone()

            ge: "Sym(u1 >= 0)" = unbacked_symint_0 >= 0
            _check = torch._check(ge);  ge = _check = None

            size = child_1.size();  child_1 = size = None
            lt: "Sym(u1 < s77)" = unbacked_symint_0 < sym_size_int;  sym_size_int = None
            _check_1 = torch._check(lt);  lt = _check_1 = None

            select: "f32[s27]" = x_clone.select(0, unbacked_symint_0)
            select_1: "f32[s27]" = x_clone.select(0, unbacked_symint_0)
            add: "f32[s27]" = select_1 + unbacked_symint_0;  select_1 = None
            copy_: "f32[s27]" = select.copy_(add);  select = add = copy_ = None

            add_1: "Sym(u1 + 1)" = unbacked_symint_0 + 1;  unbacked_symint_0 = None
            return (add_1, x_clone)
""",  # noqa: B950
            )

    @skipIfTorchDynamo("Skip because we're testing export")
    @parametrize("strict", [True, False])
    @parametrize("dynamic", [True, False])
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_while_loop_op_constant_and_symint_output_export(self, strict, dynamic):
        m, args = WHILE_LOOP_TESTS["const_and_symint_output"]
        dynamic_shapes = {"t": {0: torch.export.Dim("dim_t")}} if dynamic else None
        ep = self._check_export(m, args, strict=strict, dynamic_shapes=dynamic_shapes)
        # strict or dynamic gives a slightly different graph
        if not strict and not dynamic:
            self.assertExpectedInline(
                normalize_gm(ep.module().print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, t):
        t: "f32[2, 3]";

        t, = fx_pytree.tree_flatten_spec(([t], {}), self._in_spec)
        _guards_fn = self._guards_fn(t);  _guards_fn = None

        sum_1: "f32[]" = torch.ops.aten.sum.default(t)
        _assert_tensor_metadata_default = torch.ops.aten._assert_tensor_metadata.default(sum_1, dtype = torch.float32, device = device(type='cpu'), layout = torch.strided);  _assert_tensor_metadata_default = None
        to: "i64[]" = torch.ops.aten.to.dtype(sum_1, torch.int64);  sum_1 = None
        item: "Sym(u0)" = torch.ops.aten.item.default(to);  to = None
        sin: "f32[2, 3]" = torch.ops.aten.sin.default(t)

        while_loop_cond_graph_0 = self.while_loop_cond_graph_0
        while_loop_body_graph_0 = self.while_loop_body_graph_0
        while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_0, while_loop_body_graph_0, (2, 3, 1, 1, 1, 3, item, sin), ());  while_loop_cond_graph_0 = while_loop_body_graph_0 = item = sin = None
        getitem_8: "Sym(u8)" = while_loop[0]
        getitem_9: "Sym(u9)" = while_loop[1]
        getitem_10: "Sym(u10)" = while_loop[2]
        getitem_11: "Sym(u11)" = while_loop[3]
        getitem_12: "Sym(u12)" = while_loop[4]
        getitem_13: "Sym(u13)" = while_loop[5]
        getitem_14: "Sym(u14)" = while_loop[6]
        getitem_7: "f32[2, 3]" = while_loop[7];  while_loop = None

        add: "Sym(u8 + 1)" = getitem_8 + 1
        add_1: "Sym(u9 + 1)" = getitem_9 + 1
        add_2: "Sym(u10 + 1)" = getitem_10 + 1
        add_3: "Sym(u11 + 1)" = getitem_11 + 1
        add_4: "Sym(u12 + 1)" = getitem_12 + 1
        add_5: "Sym(u13 + 1)" = getitem_13 + 1
        add_6: "Sym(u14 + 1)" = getitem_14 + 1
        add_7: "f32[2, 3]" = torch.ops.aten.add.Tensor(getitem_7, 1)

        add_8: "f32[2, 3]" = torch.ops.aten.add.Tensor(t, getitem_8);  getitem_8 = None
        add_9: "f32[2, 3]" = torch.ops.aten.add.Tensor(t, getitem_9);  getitem_9 = None
        add_10: "f32[2, 3]" = torch.ops.aten.add.Tensor(t, getitem_10);  getitem_10 = None
        add_11: "f32[2, 3]" = torch.ops.aten.add.Tensor(t, getitem_11);  getitem_11 = None
        add_12: "f32[2, 3]" = torch.ops.aten.add.Tensor(t, getitem_12);  getitem_12 = None
        add_13: "f32[2, 3]" = torch.ops.aten.add.Tensor(t, getitem_13);  getitem_13 = None
        add_14: "f32[2, 3]" = torch.ops.aten.add.Tensor(t, getitem_14);  getitem_14 = None
        add_15: "f32[2, 3]" = torch.ops.aten.add.Tensor(getitem_7, t);  getitem_7 = t = None
        return pytree.tree_unflatten((add, add_1, add_2, add_3, add_4, add_5, add_6, add_7, add_8, add_9, add_10, add_11, add_12, add_13, add_14, add_15), self._out_spec)

    class while_loop_cond_graph_0(torch.nn.Module):
        def forward(self, a_1: "Sym(u1)", b_1: "Sym(u2)", c1_1: "Sym(u3)", c2_1: "Sym(u4)", c3_1: "Sym(u5)", c0_1: "Sym(u6)", u0_1: "Sym(u7)", x_1: "f32[2, 3]"):
            mul: "Sym(u3*u4)" = c1_1 * c2_1;  c1_1 = c2_1 = None
            mul_1: "Sym(u3*u4*u5)" = mul * c3_1;  mul = c3_1 = None
            mul_2: "Sym(u1*u2)" = a_1 * b_1;  a_1 = b_1 = None
            lt: "Sym(u3*u4*u5 < u1*u2)" = mul_1 < mul_2;  mul_1 = mul_2 = None
            return lt

    class while_loop_body_graph_0(torch.nn.Module):
        def forward(self, a_1: "Sym(u1)", b_1: "Sym(u2)", c1_1: "Sym(u3)", c2_1: "Sym(u4)", c3_1: "Sym(u5)", c0_1: "Sym(u6)", u0_1: "Sym(u7)", x_1: "f32[2, 3]"):
            add: "Sym(u7 + 1)" = u0_1 + 1;  u0_1 = None
            add_1: "f32[2, 3]" = torch.ops.aten.add.Tensor(x_1, 1);  x_1 = None
            return (b_1, c1_1, c2_1, c3_1, a_1, 0, add, add_1)
""",  # noqa: B950
            )

    @skipIfTorchDynamo("Graph is not captured correctly when test with dynamo")
    @parametrize("dynamic", [True, False])
    @parametrize("backend", ["eager", "aot_eager"])
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_while_loop_op_constant_and_symint_output_compile(self, dynamic, backend):
        m, args = WHILE_LOOP_TESTS["const_and_symint_output"]
        if backend == "eager":
            backend = EagerAndRecordGraphs()
        self._check_compile(m, args, dynamic=dynamic, backend=backend)
        if (
            isinstance(backend, EagerAndRecordGraphs)
            # cross ref or dynamic gives a slightly different graph
            and not dynamic
            and not TEST_WITH_CROSSREF
        ):
            self.assertEqual(len(backend.graphs), 1)
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_t_: "f32[2, 3]"):
        l_t_ = L_t_

        sum_1: "f32[]" = l_t_.sum()
        to: "i64[]" = sum_1.to(torch.int64);  sum_1 = None
        item: "Sym(u0)" = to.item();  to = None
        sin: "f32[2, 3]" = l_t_.sin()

        cond_fn_0 = self.cond_fn_0
        body_fn_0 = self.body_fn_0
        while_loop = torch.ops.higher_order.while_loop(cond_fn_0, body_fn_0, (2, 3, 1, 1, 1, 3, item, sin), ());  cond_fn_0 = body_fn_0 = item = sin = None
        getitem_8: "Sym(u15)" = while_loop[0]
        getitem_9: "Sym(u16)" = while_loop[1]
        getitem_10: "Sym(u17)" = while_loop[2]
        getitem_11: "Sym(u18)" = while_loop[3]
        getitem_12: "Sym(u19)" = while_loop[4]
        getitem_13: "Sym(u20)" = while_loop[5]
        getitem_14: "Sym(u21)" = while_loop[6]
        getitem_7: "f32[2, 3]" = while_loop[7];  while_loop = None

        add: "Sym(u15 + 1)" = getitem_8 + 1
        add_1: "Sym(u16 + 1)" = getitem_9 + 1
        add_2: "Sym(u17 + 1)" = getitem_10 + 1
        add_3: "Sym(u18 + 1)" = getitem_11 + 1
        add_4: "Sym(u19 + 1)" = getitem_12 + 1
        add_5: "Sym(u20 + 1)" = getitem_13 + 1
        add_6: "Sym(u21 + 1)" = getitem_14 + 1
        add_7: "f32[2, 3]" = getitem_7 + 1

        add_8: "f32[2, 3]" = getitem_8 + l_t_;  getitem_8 = None
        add_9: "f32[2, 3]" = getitem_9 + l_t_;  getitem_9 = None
        add_10: "f32[2, 3]" = getitem_10 + l_t_;  getitem_10 = None
        add_11: "f32[2, 3]" = getitem_11 + l_t_;  getitem_11 = None
        add_12: "f32[2, 3]" = getitem_12 + l_t_;  getitem_12 = None
        add_13: "f32[2, 3]" = getitem_13 + l_t_;  getitem_13 = None
        add_14: "f32[2, 3]" = getitem_14 + l_t_;  getitem_14 = None
        add_15: "f32[2, 3]" = getitem_7 + l_t_;  getitem_7 = l_t_ = None
        return (add, add_1, add_2, add_3, add_4, add_5, add_6, add_7, add_8, add_9, add_10, add_11, add_12, add_13, add_14, add_15)

    class cond_fn_0(torch.nn.Module):
        def forward(self, unbacked_symint: "Sym(u1)", unbacked_symint_0: "Sym(u2)", unbacked_symint_1: "Sym(u3)", unbacked_symint_2: "Sym(u4)", unbacked_symint_3: "Sym(u5)", unbacked_symint_4: "Sym(u6)", unbacked_symint_5: "Sym(u7)", child: "f32[2, 3]"):
            mul: "Sym(u3*u4)" = unbacked_symint_1 * unbacked_symint_2;  unbacked_symint_1 = unbacked_symint_2 = None
            mul_1: "Sym(u3*u4*u5)" = mul * unbacked_symint_3;  mul = unbacked_symint_3 = None
            mul_2: "Sym(u1*u2)" = unbacked_symint * unbacked_symint_0;  unbacked_symint = unbacked_symint_0 = None
            lt: "Sym(u3*u4*u5 < u1*u2)" = mul_1 < mul_2;  mul_1 = mul_2 = None
            return lt

    class body_fn_0(torch.nn.Module):
        def forward(self, unbacked_symint_6: "Sym(u8)", unbacked_symint_7: "Sym(u9)", unbacked_symint_8: "Sym(u10)", unbacked_symint_9: "Sym(u11)", unbacked_symint_10: "Sym(u12)", unbacked_symint_11: "Sym(u13)", unbacked_symint_12: "Sym(u14)", child_1: "f32[2, 3]"):
            add: "Sym(u14 + 1)" = unbacked_symint_12 + 1;  unbacked_symint_12 = None
            child: "f32[2, 3]" = child_1 + 1;  child_1 = None
            return (unbacked_symint_7, unbacked_symint_8, unbacked_symint_9, unbacked_symint_10, unbacked_symint_6, 0, add, child)
""",  # noqa: B950
            )

    @skipIfTorchDynamo("Skip because we're testing export")
    @parametrize("strict", [True, False])
    @parametrize("dynamic", [True, False])
    def test_while_loop_op_pytree_int_carry_export(self, strict, dynamic):
        m, args = WHILE_LOOP_TESTS["pytree_int_carry"]
        dynamic_shapes = {"x": {0: torch.export.Dim("dim_x")}} if dynamic else None
        ep = self._check_export(m, args, strict=strict, dynamic_shapes=dynamic_shapes)
        if strict and dynamic and not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(ep.module().print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, x):
        x: "f32[s6, 3]";

        x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
        _guards_fn = self._guards_fn(x);  _guards_fn = None
        sym_size_int_1: "Sym(s6)" = torch.ops.aten.sym_size.int(x, 0)

        sin: "f32[s6, 3]" = torch.ops.aten.sin.default(x);  x = None

        while_loop_cond_graph_0 = self.while_loop_cond_graph_0
        while_loop_body_graph_0 = self.while_loop_body_graph_0
        while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_0, while_loop_body_graph_0, (sym_size_int_1, 3, 2, 2, 3, sin), ());  while_loop_cond_graph_0 = while_loop_body_graph_0 = sym_size_int_1 = sin = None
        getitem_6: "Sym(u10)" = while_loop[0]
        getitem_7: "Sym(u11)" = while_loop[1]
        getitem_8: "Sym(u12)" = while_loop[2]
        getitem_9: "Sym(u13)" = while_loop[3]
        getitem_10: "Sym(u14)" = while_loop[4]
        getitem_5: "f32[s6, 3]" = while_loop[5];  while_loop = None

        add: "Sym(u12 + 1)" = getitem_8 + 1
        add_1: "Sym(u13 + 1)" = getitem_9 + 1
        add_2: "Sym(u14 + 1)" = getitem_10 + 1

        add_3: "f32[s6, 3]" = torch.ops.aten.add.Tensor(getitem_5, getitem_8);  getitem_8 = None
        add_4: "f32[s6, 3]" = torch.ops.aten.add.Tensor(getitem_5, getitem_9);  getitem_9 = None
        add_5: "f32[s6, 3]" = torch.ops.aten.add.Tensor(getitem_5, getitem_10);  getitem_10 = None
        return pytree.tree_unflatten((getitem_6, getitem_7, add, add_1, add_2, add_3, add_4, add_5, getitem_5), self._out_spec)

    class while_loop_cond_graph_0(torch.nn.Module):
        def forward(self, arg0_1: "Sym(u15)", arg1_1: "Sym(u16)", arg2_1: "Sym(u17)", arg3_1: "Sym(u18)", arg4_1: "Sym(u19)", arg5_1: "f32[s6, 3]"):
            mul: "Sym(u17*u18)" = arg2_1 * arg3_1;  arg2_1 = arg3_1 = None
            mul_1: "Sym(u17*u18*u19)" = mul * arg4_1;  mul = arg4_1 = None
            mul_2: "Sym(u15*u16)" = arg0_1 * arg1_1;  arg0_1 = arg1_1 = None
            lt: "Sym(u17*u18*u19 < u15*u16)" = mul_1 < mul_2;  mul_1 = mul_2 = None
            return lt

    class while_loop_body_graph_0(torch.nn.Module):
        def forward(self, arg0_1: "Sym(u15)", arg1_1: "Sym(u16)", arg2_1: "Sym(u17)", arg3_1: "Sym(u18)", arg4_1: "Sym(u19)", arg5_1: "f32[s6, 3]"):
            add: "Sym(u15 + 1)" = arg0_1 + 1;  arg0_1 = None
            add_1: "Sym(u16 + 1)" = arg1_1 + 1;  arg1_1 = None

            add_2: "Sym(u17 + 1)" = arg2_1 + 1;  arg2_1 = None
            add_3: "Sym(u18 + 1)" = arg3_1 + 1;  arg3_1 = None
            add_4: "Sym(u19 + 1)" = arg4_1 + 1;  arg4_1 = None

            add_5: "f32[s6, 3]" = torch.ops.aten.add.Tensor(arg5_1, 1);  arg5_1 = None
            return (add, add_1, add_2, add_3, add_4, add_5)
""",  # noqa: B950
            )

    @skipIfTorchDynamo("Graph is not captured correctly when test with dynamo")
    @parametrize("dynamic", [True, False])
    @parametrize("backend", ["eager", "aot_eager"])
    def test_while_loop_op_pytree_int_carry_compile(self, dynamic, backend):
        m, args = WHILE_LOOP_TESTS["pytree_int_carry"]
        if backend == "eager":
            backend = EagerAndRecordGraphs()
        self._check_compile(m, args, dynamic=dynamic, backend=backend)
        if (
            isinstance(backend, EagerAndRecordGraphs)
            and dynamic
            and not TEST_WITH_CROSSREF
        ):
            self.assertEqual(len(backend.graphs), 1)
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, s77: "Sym(s77)", s27: "Sym(s27)", L_x_: "f32[s77, s27]"):
        l_x_ = L_x_

        child: "f32[s77, s27]" = l_x_.sin();  l_x_ = None

        cond_fn_0 = self.cond_fn_0
        body_fn_0 = self.body_fn_0
        while_loop = torch.ops.higher_order.while_loop(cond_fn_0, body_fn_0, (s77, s27, 2, 2, 3, child), (s27, s77));  cond_fn_0 = body_fn_0 = s77 = s27 = child = None
        getitem_10: "Sym(u10)" = while_loop[0]
        getitem_11: "Sym(u11)" = while_loop[1]
        getitem_12: "Sym(u12)" = while_loop[2]
        getitem_13: "Sym(u13)" = while_loop[3]
        getitem_14: "Sym(u14)" = while_loop[4]
        out_x: "f32[s77, s27]" = while_loop[5];  while_loop = None

        add: "Sym(u12 + 1)" = getitem_12 + 1
        add_1: "Sym(u13 + 1)" = getitem_13 + 1
        add_2: "Sym(u14 + 1)" = getitem_14 + 1

        add_3: "f32[s77, s27]" = getitem_12 + out_x;  getitem_12 = None
        add_4: "f32[s77, s27]" = getitem_13 + out_x;  getitem_13 = None
        add_5: "f32[s77, s27]" = getitem_14 + out_x;  getitem_14 = None
        return (getitem_10, getitem_11, add, add_1, add_2, add_3, add_4, add_5, out_x)

    class cond_fn_0(torch.nn.Module):
        def forward(self, unbacked_symint: "Sym(u0)", unbacked_symint_0: "Sym(u1)", unbacked_symint_1: "Sym(u2)", unbacked_symint_2: "Sym(u3)", unbacked_symint_3: "Sym(u4)", child_1: "f32[s77, s27]", s27: "Sym(s27)", s77: "Sym(s77)"):
            s27_1 = s27
            s77_1 = s77

            mul: "Sym(u2*u3)" = unbacked_symint_1 * unbacked_symint_2;  unbacked_symint_1 = unbacked_symint_2 = None
            mul_1: "Sym(u2*u3*u4)" = mul * unbacked_symint_3;  mul = unbacked_symint_3 = None
            mul_2: "Sym(u0*u1)" = unbacked_symint * unbacked_symint_0;  unbacked_symint = unbacked_symint_0 = None
            lt: "Sym(u2*u3*u4 < u0*u1)" = mul_1 < mul_2;  mul_1 = mul_2 = None
            return lt

    class body_fn_0(torch.nn.Module):
        def forward(self, unbacked_symint_4: "Sym(u5)", unbacked_symint_5: "Sym(u6)", unbacked_symint_6: "Sym(u7)", unbacked_symint_7: "Sym(u8)", unbacked_symint_8: "Sym(u9)", child_2: "f32[s77, s27]", s27: "Sym(s27)", s77: "Sym(s77)"):
            s27_1 = s27
            s77_1 = s77

            add: "Sym(u5 + 1)" = unbacked_symint_4 + 1;  unbacked_symint_4 = None
            add_1: "Sym(u6 + 1)" = unbacked_symint_5 + 1;  unbacked_symint_5 = None

            add_2: "Sym(u7 + 1)" = unbacked_symint_6 + 1;  unbacked_symint_6 = None
            add_3: "Sym(u8 + 1)" = unbacked_symint_7 + 1;  unbacked_symint_7 = None
            add_4: "Sym(u9 + 1)" = unbacked_symint_8 + 1;  unbacked_symint_8 = None

            child: "f32[s77, s27]" = child_2 + 1;  child_2 = None
            return (add, add_1, add_2, add_3, add_4, child)
""",  # noqa: B950
            )

    @parametrize("dynamic", [True, False])
    @parametrize("backend", ["eager", "aot_eager"])
    def test_compile_while_loop_stack_output(self, dynamic, backend):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                c = torch.tensor(0, dtype=torch.int64)

                def cond_fn(c, x):
                    return c < x.size(0)

                def body_fn(c, x):
                    return c + 1, self.linear(x)

                stacked_c, stacked_x = torch.ops.higher_order.while_loop_stack_output(
                    cond_fn, body_fn, (c, x), tuple()
                )
                return stacked_c, stacked_x

        x = torch.randn(3, 3)
        mod = Mod()
        compiled_out = torch.compile(mod, backend=backend, dynamic=dynamic)(x)
        self.assertEqual(len(compiled_out), 2)
        self.assertEqual(compiled_out[0].size(0), 3)
        self.assertEqual(compiled_out[1].size(0), 3)
        self.assertEqual(compiled_out, mod(x))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_while_loop_autograd_simple(self):
        backend = torch._dynamo.testing.AotEagerAndRecordGraphs()

        class ModEager(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                while x.sum() < 2:
                    x = x * x + 1 + self.linear(x)
                return x

        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                def cond_fn(x):
                    return x.sum() < 2

                def body_fn(x):
                    return x * x + 1 + self.linear(x)

                return torch._higher_order_ops.while_loop(cond_fn, body_fn, (x,))

        x = torch.randn(3, 3, requires_grad=True)
        x_clone = x.clone()
        mod = Mod()
        mod_eager = ModEager()
        # Copy weights from mod to mod_eager
        mod_eager.load_state_dict(mod.state_dict())
        compiled_out = torch.compile(mod, backend=backend, fullgraph=True)(x)
        exp_out = mod_eager(x_clone)
        compiled_out.sum().backward()
        exp_out.sum().backward()
        self.assertEqual(compiled_out, exp_out)
        eager_parameters = dict(mod_eager.named_parameters())
        compiled_parameters = dict(mod.named_parameters())
        for name, param in compiled_parameters.items():
            self.assertEqual(param, eager_parameters[name])
            self.assertEqual(param.grad, eager_parameters[name].grad)

        self.assertEqual(
            len(
                backend.fw_graphs[0].graph.find_nodes(
                    op="call_function",
                    target=torch.ops.higher_order.while_loop_stack_output,
                )
            ),
            1,
        )
        self.assertEqual(
            len(
                backend.bw_graphs[0].graph.find_nodes(
                    op="call_function", target=torch.ops.higher_order.while_loop
                )
            ),
            1,
        )
        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[3, 3]", primals_2: "f32[3, 3]", primals_3: "f32[3]"):
        while_loop_cond_graph_0 = self.while_loop_cond_graph_0
        while_loop_body_graph_0 = self.while_loop_body_graph_0
        while_loop_stack_output = torch.ops.higher_order.while_loop_stack_output(while_loop_cond_graph_0, while_loop_body_graph_0, (primals_1,), (primals_3, primals_2));  while_loop_cond_graph_0 = while_loop_body_graph_0 = None
        getitem: "f32[u2, 3, 3]" = while_loop_stack_output[0];  while_loop_stack_output = None
        select: "f32[3, 3]" = torch.ops.aten.select.int(getitem, 0, -1)
        unsqueeze: "f32[1, 3, 3]" = torch.ops.aten.unsqueeze.default(primals_1, 0);  primals_1 = None
        slice_1: "f32[u2 - 1, 3, 3]" = torch.ops.aten.slice.Tensor(getitem, 0, 0, -1);  getitem = None
        cat: "f32[u2, 3, 3]" = torch.ops.aten.cat.default([unsqueeze, slice_1]);  unsqueeze = slice_1 = None
        return (select, primals_2, primals_3, cat)

    class while_loop_cond_graph_0(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]", arg1_1: "f32[3]", arg2_1: "f32[3, 3]"):
            sum_1: "f32[]" = torch.ops.aten.sum.default(arg0_1);  arg0_1 = None
            lt: "b8[]" = torch.ops.aten.lt.Scalar(sum_1, 2);  sum_1 = None
            return lt

    class while_loop_body_graph_0(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]", arg1_1: "f32[3]", arg2_1: "f32[3, 3]"):
            mul: "f32[3, 3]" = torch.ops.aten.mul.Tensor(arg0_1, arg0_1)
            add: "f32[3, 3]" = torch.ops.aten.add.Tensor(mul, 1);  mul = None
            t: "f32[3, 3]" = torch.ops.aten.t.default(arg2_1);  arg2_1 = None
            addmm: "f32[3, 3]" = torch.ops.aten.addmm.default(arg1_1, arg0_1, t);  arg1_1 = arg0_1 = t = None
            add_1: "f32[3, 3]" = torch.ops.aten.add.Tensor(add, addmm);  add = addmm = None
            return (add_1,)
""",  # noqa: B950
            )

            self.assertExpectedInline(
                normalize_gm(backend.bw_graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_2: "f32[3, 3]", primals_3: "f32[3]", cat: "f32[u2, 3, 3]", tangents_1: "f32[3, 3]"):
        zeros: "i64[]" = torch.ops.aten.zeros.default([], dtype = torch.int64, device = device(type='cpu'), pin_memory = False)
        zeros_like: "f32[3]" = torch.ops.aten.zeros_like.default(primals_3, pin_memory = False)
        zeros_like_1: "f32[3, 3]" = torch.ops.aten.zeros_like.default(primals_2, pin_memory = False)
        while_loop_cond_graph_1 = self.while_loop_cond_graph_1
        while_loop_body_graph_1 = self.while_loop_body_graph_1
        while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_1, while_loop_body_graph_1, (zeros, tangents_1, zeros_like, zeros_like_1), (cat, primals_3, primals_2));  while_loop_cond_graph_1 = while_loop_body_graph_1 = zeros = tangents_1 = zeros_like = zeros_like_1 = cat = primals_3 = primals_2 = None
        getitem_2: "f32[3, 3]" = while_loop[1]
        getitem_3: "f32[3]" = while_loop[2]
        getitem_4: "f32[3, 3]" = while_loop[3];  while_loop = None
        return (getitem_2, getitem_4, getitem_3)

    class while_loop_cond_graph_1(torch.nn.Module):
        def forward(self, arg0_1: "i64[]", arg1_1: "f32[3, 3]", arg2_1: "f32[3]", arg3_1: "f32[3, 3]", arg4_1: "f32[u2, 3, 3]", arg5_1: "f32[3]", arg6_1: "f32[3, 3]"):
            sym_size_int_1: "Sym(u2)" = torch.ops.aten.sym_size.int(arg4_1, 0);  arg4_1 = None

            lt: "b8[]" = torch.ops.aten.lt.Scalar(arg0_1, sym_size_int_1);  arg0_1 = sym_size_int_1 = None
            return lt

    class while_loop_body_graph_1(torch.nn.Module):
        def forward(self, arg0_1: "i64[]", arg1_1: "f32[3, 3]", arg2_1: "f32[3]", arg3_1: "f32[3, 3]", arg4_1: "f32[u2, 3, 3]", arg5_1: "f32[3]", arg6_1: "f32[3, 3]"):
            sym_size_int_1: "Sym(u2)" = torch.ops.aten.sym_size.int(arg4_1, 0)

            rsub: "i64[]" = torch.ops.aten.rsub.Scalar(arg0_1, sym_size_int_1);  sym_size_int_1 = None
            sub_1: "i64[]" = torch.ops.aten.sub.Tensor(rsub, 1);  rsub = None
            _local_scalar_dense: "Sym(u7)" = torch.ops.aten._local_scalar_dense.default(sub_1);  sub_1 = None
            select: "f32[3, 3]" = torch.ops.aten.select.int(arg4_1, 0, _local_scalar_dense);  arg4_1 = _local_scalar_dense = None
            t: "f32[3, 3]" = torch.ops.aten.t.default(arg6_1);  arg6_1 = None
            t_1: "f32[3, 3]" = torch.ops.aten.t.default(t);  t = None
            mm: "f32[3, 3]" = torch.ops.aten.mm.default(arg1_1, t_1);  t_1 = None
            t_2: "f32[3, 3]" = torch.ops.aten.t.default(arg1_1)
            mm_1: "f32[3, 3]" = torch.ops.aten.mm.default(t_2, select);  t_2 = None
            t_3: "f32[3, 3]" = torch.ops.aten.t.default(mm_1);  mm_1 = None
            sum_1: "f32[1, 3]" = torch.ops.aten.sum.dim_IntList(arg1_1, [0], True)
            view: "f32[3]" = torch.ops.aten.view.default(sum_1, [3]);  sum_1 = None
            t_4: "f32[3, 3]" = torch.ops.aten.t.default(t_3);  t_3 = None
            mul_4: "f32[3, 3]" = torch.ops.aten.mul.Tensor(arg1_1, select)
            mul_5: "f32[3, 3]" = torch.ops.aten.mul.Tensor(arg1_1, select);  arg1_1 = select = None
            add_7: "f32[3, 3]" = torch.ops.aten.add.Tensor(mm, mul_5);  mm = mul_5 = None
            add_8: "f32[3, 3]" = torch.ops.aten.add.Tensor(add_7, mul_4);  add_7 = mul_4 = None
            add_9: "i64[]" = torch.ops.aten.add.Tensor(arg0_1, 1);  arg0_1 = None
            add_10: "f32[3]" = torch.ops.aten.add.Tensor(view, arg2_1);  view = arg2_1 = None
            add_11: "f32[3, 3]" = torch.ops.aten.add.Tensor(t_4, arg3_1);  t_4 = arg3_1 = None
            return (add_9, add_8, add_10, add_11)
""",  # noqa: B950
            )

    def test_input_output_alias(self):
        def fn(f, *args):
            return torch.cond(args[0].sum() > 0, f, f, args)

        x = torch.randn(2, 2)
        for f in ALIAS_FN:
            with self.assertRaisesRegex(
                # Should be
                # torch._dynamo.exc.Unsupported,
                # "Encountered aliasing during higher order op tracing for HOP.*"
                torch._dynamo.exc.UncapturedHigherOrderOpError,
                r"Higher Order Operator: torch\.cond",
            ):
                torch.compile(fn)(f, x)

    def test_input_input_alias(self):
        def fn(view_f, arg):
            def f(arg1, arg2):
                return arg1.cos(), arg2.sin()

            return torch.cond(arg.sum() > 0, f, f, (arg, view_f(arg)))

        x = torch.randn(2, 2)
        # ALIAS_FN[0] is an identical function, cond optimizes the duplication
        # as a result of auto lifting.
        for view_f in ALIAS_FN[1:]:
            with self.assertRaisesRegex(
                # Should be
                # torch._dynamo.exc.Unsupported,
                # "Encountered aliasing during higher order op tracing for HOP.*"
                torch._dynamo.exc.UncapturedHigherOrderOpError,
                r"Higher Order Operator: torch\.cond",
            ):
                torch.compile(fn)(view_f, x)

    @parametrize("inference_mode", [True, False])
    def test_input_mutation(self, inference_mode):
        def fn(view_f, *args):
            def mutate_f(x):
                v = view_f(x)
                v.add_(1)
                return v.sin()

            return torch.cond(args[0].sum() > 0, mutate_f, mutate_f, args)

        x = torch.randn(2, 2)
        for f in ALIAS_FN:
            with self.assertRaisesRegex(
                # Should be
                # torch._dynamo.exc.Unsupported,
                # "Encountered aliasing during higher order op tracing for HOP.*"
                torch._dynamo.exc.UncapturedHigherOrderOpError,
                r"Higher Order Operator: torch\.cond",
            ):
                torch.compile(fn)(f, x)

            with torch.inference_mode(inference_mode):
                if not inference_mode:
                    with self.assertRaisesRegex(
                        # Should be
                        # torch._dynamo.exc.Unsupported,
                        # "Encountered aliasing during higher order op tracing for HOP.*"
                        torch._dynamo.exc.UncapturedHigherOrderOpError,
                        r"Higher Order Operator: torch\.cond",
                    ):
                        torch.compile(fn)(f, x)
                else:
                    torch.compile(fn)(f, x)

    @requires_cuda
    @parametrize("device", ["cuda", "cpu"])
    def test_cond_input_mutation(self, device):
        predicate_true = torch.tensor(True, device=device)
        predicate_false = torch.tensor(False, device=device)
        org_data = torch.ones(2, 2, device=device)

        def fn(predicate, data):
            return torch.cond(
                predicate, lambda x: x + 1, lambda x: x.sin_().add_(2), [data]
            )

        with torch.no_grad():
            expected = org_data.sin() + 2
            data = org_data.clone()
            output = torch.compile(fn)(predicate_false, data)
            self.assertEqual(output, expected)
            self.assertIsNot(output, data)

            data = org_data.clone()
            output = torch.compile(fn)(predicate_true, data)
            self.assertEqual(output, org_data + 1)

    @skipIfTorchDynamo("Graph is not captured correctly when test with dynamo")
    def test_while_loop_unbacked_bindings(self):
        m, args = WHILE_LOOP_TESTS["pytree_int_carry"]
        backend = EagerAndRecordGraphs()
        self._check_compile(m, args, dynamic=True, backend=backend)
        self.assertEqual(len(backend.graphs), 1)
        while_loop_nodes = backend.graphs[0].graph.find_nodes(
            op="call_function", target=torch.ops.higher_order.while_loop
        )
        self.assertEqual(len(while_loop_nodes), 1)
        self.assertEqual(len(while_loop_nodes[0].meta.get("unbacked_bindings")), 5)

    # Return the .module() graph str result of non-strict export
    def _check_export_ret_graph_str(self, fn, args, dynamic_shapes=None) -> str:
        strict_ep = torch.export.export(
            fn, args, dynamic_shapes=dynamic_shapes, strict=True
        )
        non_strict_ep = torch.export.export(
            fn, args, dynamic_shapes=dynamic_shapes, strict=False
        )
        eager_res = fn(*args)
        self.assertEqual(strict_ep.module()(*args), eager_res)
        self.assertEqual(non_strict_ep.module()(*args), eager_res)
        return normalize_gm(non_strict_ep.module().print_readable(print_output=False))

    @skipIfTorchDynamo("Skip because dynamo cannot trace torch.export.")
    def test_cond_eager_run_with_item(self):
        class M(torch.nn.Module):
            def forward(self, a, b1, b2, c):
                def true_fn(x):
                    return x * b1.item()

                def false_fn(x):
                    return x * b2.item()

                r = torch.cond(a, true_fn, false_fn, (c,))
                return r * 2

        x = torch.randn(10, requires_grad=True)
        args = (
            torch.tensor(True),
            torch.tensor([3]),
            torch.tensor([4]),
            x,
        )
        model = M()
        torch.export.export(model, args, strict=True)
        graph_str = self._check_export_ret_graph_str(model, args, None)
        self.assertExpectedInline(
            graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, a, b1, b2, c):
        a: "b8[]"; b1: "i64[1]"; b2: "i64[1]"; c: "f32[10]";

        a, b1, b2, c, = fx_pytree.tree_flatten_spec(([a, b1, b2, c], {}), self._in_spec)
        _guards_fn = self._guards_fn(a, b1, b2, c);  _guards_fn = None

        true_graph_0 = self.true_graph_0
        false_graph_0 = self.false_graph_0
        cond = torch.ops.higher_order.cond(a, true_graph_0, false_graph_0, (c, b1, b2));  a = true_graph_0 = false_graph_0 = c = b1 = b2 = None
        getitem: "f32[10]" = cond[0];  cond = None

        mul: "f32[10]" = torch.ops.aten.mul.Tensor(getitem, 2);  getitem = None
        return pytree.tree_unflatten((mul,), self._out_spec)

    class true_graph_0(torch.nn.Module):
        def forward(self, c: "f32[10]", b1: "i64[1]", b2: "i64[1]"):
            item: "Sym(u0)" = torch.ops.aten.item.default(b1);  b1 = None

            mul: "f32[10]" = torch.ops.aten.mul.Tensor(c, item);  c = item = None
            return (mul,)

    class false_graph_0(torch.nn.Module):
        def forward(self, c: "f32[10]", b1: "i64[1]", b2: "i64[1]"):
            item: "Sym(u1)" = torch.ops.aten.item.default(b2);  b2 = None

            mul: "f32[10]" = torch.ops.aten.mul.Tensor(c, item);  c = item = None
            return (mul,)
""",  # noqa: B950
        )

    def test_cond_merge_graph_preserves_ph_meta(self):
        class M(torch.nn.Module):
            def forward(self, x, y, z):
                a = y.shape[0]
                b = z.shape[0]

                def true_fn(x):
                    return x + a

                def false_fn(x):
                    return x + b * z

                return torch.cond(x.sum() > 5, true_fn, false_fn, (x,))

        backend = EagerAndRecordGraphs()
        _ = torch.compile(M(), backend=backend)(
            torch.randn(3, 4), torch.randn(3, 4), torch.randn(3, 4)
        )
        self.assertEqual(len(backend.graphs), 1)
        gm = backend.graphs[0]
        subgraph_attr = gm.graph.find_nodes(op="get_attr")[0]
        subgm = getattr(gm, subgraph_attr.target)
        for ph in subgm.graph.find_nodes(op="placeholder"):
            self.assertTrue("example_value" in ph.meta)

    @skipIfTorchDynamo("Skip because dynamo cannot trace torch.export.")
    def test_cond_symint_closure(self):
        from torch.export import Dim

        class M(torch.nn.Module):
            def forward(self, x, y, z):
                a = y.shape[0]
                b = z.shape[0]

                def true_fn(x):
                    return x + a

                def false_fn(x):
                    return x + b * z

                # When exporting with non-strict: a and b are symints,
                # so torch.compile need to wrap and trace symint inputs.
                return torch.cond(x.shape[0] > 5, true_fn, false_fn, (x,))

        args = (torch.ones(3, 3), torch.ones(5), torch.ones(3, 3))
        model = M()
        dynamic_shapes = {"x": {0: Dim("d")}, "y": {0: Dim("d1")}, "z": {0: Dim("d")}}
        non_strict_graph_str = self._check_export_ret_graph_str(
            model, args, dynamic_shapes
        )
        self.assertExpectedInline(
            non_strict_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, x, y, z):
        x: "f32[s68, 3]"; y: "f32[s17]"; z: "f32[s68, 3]";

        x, y, z, = fx_pytree.tree_flatten_spec(([x, y, z], {}), self._in_spec)
        _guards_fn = self._guards_fn(x, y, z);  _guards_fn = None
        sym_size_int_4: "Sym(s17)" = torch.ops.aten.sym_size.int(y, 0);  y = None
        sym_size_int_5: "Sym(s68)" = torch.ops.aten.sym_size.int(z, 0)

        gt: "Sym(s68 > 5)" = sym_size_int_5 > 5

        true_graph_0 = self.true_graph_0
        false_graph_0 = self.false_graph_0
        cond = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, (x, sym_size_int_4, sym_size_int_5, z));  gt = true_graph_0 = false_graph_0 = x = sym_size_int_4 = sym_size_int_5 = z = None
        getitem: "f32[s68, 3]" = cond[0];  cond = None
        return pytree.tree_unflatten((getitem,), self._out_spec)

    class true_graph_0(torch.nn.Module):
        def forward(self, x: "f32[s68, 3]", sym_size_int_4: "Sym(s17)", sym_size_int_5: "Sym(s68)", z: "f32[s68, 3]"):
            add: "f32[s68, 3]" = torch.ops.aten.add.Tensor(x, sym_size_int_4);  x = sym_size_int_4 = None
            return (add,)

    class false_graph_0(torch.nn.Module):
        def forward(self, x: "f32[s68, 3]", sym_size_int_4: "Sym(s17)", sym_size_int_5: "Sym(s68)", z: "f32[s68, 3]"):
            mul: "f32[s68, 3]" = torch.ops.aten.mul.Tensor(z, sym_size_int_5);  z = sym_size_int_5 = None

            add: "f32[s68, 3]" = torch.ops.aten.add.Tensor(x, mul);  x = mul = None
            return (add,)
""",  # noqa: B950
        )

    # unbacked symint inputs are created during non-strict export,
    # which causes a graph break
    @unittest.expectedFailure
    def test_cond_unbacked_symint_closure(self):
        from torch.export import Dim

        class M(torch.nn.Module):
            def forward(self, x, y, z):
                a = y.shape[0]
                b = z.shape[0]
                # c is an unbacked symint in non-strict export
                c = y.sum().item()

                def true_fn(x):
                    return x + a + c

                def false_fn(x):
                    return x + b * z * c

                # When exporting with non-strict: a and b are symints,
                # so torch.compile need to wrap and trace symint inputs.
                return torch.cond(x.shape[0] > 5, true_fn, false_fn, (x,))

        args = (torch.ones(3, 3), torch.ones(5, dtype=torch.int32), torch.ones(3, 3))
        model = M()
        dynamic_shapes = {"x": {0: Dim("d")}, "y": {0: Dim("d1")}, "z": {0: Dim("d")}}
        _ = self._check_export_ret_graph_str(model, args, dynamic_shapes)

    @skipIfTorchDynamo(
        "Skip because _merge_output is not intended for dynamo to compile"
    )
    def test_merge_output(self):
        from torch._higher_order_ops.cond import _merge_output
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        # The shapes and strides are from raondomly generated pairs of tensors then swapaxes
        valid_test_cases = [
            # [(size1, stride1), (size2, stride2), (expected_stride, expected_size)]
            [((3,), (1,)), ((4,), (1,)), ("(u0,)", "(1,)")],
            [((1, 3), (3, 1)), ((3, 2), (2, 1)), ("(u0, u1)", "(u1, 1)")],
            [((2, 1), (1, 1)), ((7, 3), (3, 1)), ("(u0, u1)", "(u1, 1)")],
            [((5, 5), (1, 5)), ((4, 5), (1, 4)), ("(u0, 5)", "(1, u0)")],
            [
                ((7, 3, 1), (1, 7, 1)),
                ((4, 3, 3), (3, 12, 1)),
                ("(u0, 3, u1)", "(u1, u0*u1, 1)"),
            ],
            [
                ((5, 7, 4), (7, 1, 35)),
                ((7, 4, 4), (4, 1, 28)),
                ("(u0, u1, 4)", "(u1, 1, u0*u1)"),
            ],
            [
                ((1, 6, 3, 2), (36, 1, 6, 18)),
                ((4, 2, 2, 6), (24, 1, 2, 4)),
                ("(u0, u1, u2, u3)", "(u1*u2*u3, 1, u1, u1*u2)"),
            ],
            [
                ((6, 1, 6, 3), (18, 1, 1, 6)),
                ((2, 1, 3, 4), (12, 1, 1, 3)),
                ("(u0, 1, u1, u2)", "(u1*u2, 1, 1, u1)"),
            ],
            [
                ((3, 1, 2, 4, 1), (8, 8, 4, 1, 1)),
                ((2, 4, 1, 4, 1), (16, 4, 4, 1, 1)),
                ("(u0, u1, u2, 4, 1)", "(4*u1*u2, 4*u2, 4, 1, 1)"),
            ],
        ]

        def _inner(case):
            fake_mode = FakeTensorMode(shape_env=ShapeEnv())

            (size1, stride1), (size2, stride2), (merged_size, merged_stride) = case
            with fake_mode:
                t1 = torch.empty_strided(size1, stride1)
                t2 = torch.empty_strided(size2, stride2)
            out = _merge_output(t1, t2, fake_mode)
            self.assertEqual(str(tuple(out.size())), merged_size)
            self.assertEqual(str(tuple(out.stride())), merged_stride)

        for case in valid_test_cases:
            _inner(case)

        # The shapes and strides are from raondomly generated pairs of tensors then swapaxes
        invalid_test_cases = [
            # [(size1, stride1), (size2, stride2)]
            [((1,), (1,)), ((1,), (0,))],
            [
                ((1, 3), (1, 1)),
                ((5, 6), (6, 1)),
            ],  # t1 is not contiguous, t2 is contiguous
            [
                ((2, 1), (1, 1)),
                ((7, 3), (1, 3)),
            ],  # t1 is contiguous, t2 is not contiguous
            [
                ((5, 4), (4, 1)),
                ((5, 5), (1, 5)),
            ],  # t1 is contiguous, t2 is not contiguous
            [((7, 3, 1), (1, 7, 1)), ((4, 3, 3), (9, 1, 3))],  # layout is different
            [((5, 7, 4), (7, 1, 35)), ((7, 4, 4), (4, 28, 1))],  # layout is different
            [
                ((1, 6, 3, 2), (36, 1, 6, 18)),
                ((4, 1, 1, 6), (1, 4, 4, 4)),
            ],  # layout is different
            [
                ((6, 1, 6, 3), (18, 1, 1, 6)),
                ((1, 1, 1, 1), (1, 1, 1, 1)),
            ],  # layout is different
            [
                ((6, 1, 1, 6, 3), (3, 18, 18, 18, 1)),
                ((5, 1, 2, 1, 1), (2, 10, 1, 10, 1)),
            ],  # layout is different
        ]
        for case in invalid_test_cases:
            with self.assertRaisesRegex(Exception, r"."):
                _inner(case)

    @parametrize("dynamic", [True, False])
    @parametrize("backend", ["eager", "aot_eager"])
    def test_cond_mismatched_branch_output(self, dynamic, backend):
        class M(torch.nn.Module):
            def forward(self, x, y, z):
                a = y.shape[0]
                b = z.shape[0]

                def true_fn(x):
                    # clone the outputs so branches have the same storage_offset
                    return (x + a)[2:].clone()

                def false_fn(x):
                    # clone the outputs so branches have the same storage_offset
                    return (x + b * z)[:2].clone()

                ret = torch.cond(x.sum() > 0, true_fn, false_fn, (x,))
                return y.sum() - ret

        m = M()
        x, y, z = torch.randn(5, 4), torch.randn(5, 4), torch.randn(5, 4)
        out = m(x, y, z)
        if not (backend == "eager" and dynamic and not TEST_WITH_CROSSREF):
            compiled_out = torch.compile(
                m, backend=backend, dynamic=dynamic, fullgraph=True
            )(x, y, z)
            self.assertEqual(compiled_out, out)
        else:
            bk = EagerAndRecordGraphs()
            compiled_out = torch.compile(
                m, backend=bk, dynamic=dynamic, fullgraph=True
            )(x, y, z)
            self.assertEqual(compiled_out, out)
            self.assertExpectedInline(
                normalize_gm(bk.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, s17: "Sym(s17)", s94: "Sym(s94)", L_y_: "f32[s17, s94]", L_z_: "f32[s17, s94]", L_x_: "f32[s17, s94]"):
        l_y_ = L_y_
        l_z_ = L_z_
        l_x_ = L_x_

        sum_1: "f32[]" = l_x_.sum()
        gt: "b8[]" = sum_1 > 0;  sum_1 = None
        cond_true_0 = self.cond_true_0
        cond_false_0 = self.cond_false_0
        cond = torch.ops.higher_order.cond(gt, cond_true_0, cond_false_0, (l_x_, s94, s17, s17, l_z_));  gt = cond_true_0 = cond_false_0 = l_x_ = s94 = s17 = l_z_ = None
        getitem_5: "f32[u0, s94]" = cond[0]
        sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(getitem_5, 0);  getitem_5 = None
        ge: "Sym(u0 >= 0)" = sym_size_int >= 0;  sym_size_int = None
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
        ret: "f32[u0, s94]" = cond[0];  cond = None

        sum_2: "f32[]" = l_y_.sum();  l_y_ = None
        sub: "f32[u0, s94]" = sum_2 - ret;  sum_2 = ret = None
        return (sub,)

    class cond_true_0(torch.nn.Module):
        def forward(self, l_x_: "f32[s17, s94]", s94: "Sym(s94)", s17_true_branch: "Sym(s17)", getitem_2_false_branch: "Sym(s17)", l_z__false_branch: "f32[s17, s94]"):
            l_x__1 = l_x_
            s94_1 = s94

            add: "f32[s17, s94]" = l_x__1 + s17_true_branch;  l_x__1 = s17_true_branch = None
            getitem: "f32[s17 - 2, s94]" = add[slice(2, None, None)];  add = None
            clone: "f32[s17 - 2, s94]" = getitem.clone();  getitem = None
            return (clone,)

    class cond_false_0(torch.nn.Module):
        def forward(self, l_x_: "f32[s17, s94]", s94: "Sym(s94)", s17_true_branch: "Sym(s17)", getitem_2_false_branch: "Sym(s17)", l_z__false_branch: "f32[s17, s94]"):
            l_x__1 = l_x_
            s94_1 = s94

            mul: "f32[s17, s94]" = getitem_2_false_branch * l_z__false_branch;  getitem_2_false_branch = l_z__false_branch = None
            add: "f32[s17, s94]" = l_x__1 + mul;  l_x__1 = mul = None
            getitem: "f32[2, s94]" = add[slice(None, 2, None)];  add = None
            clone: "f32[2, s94]" = getitem.clone();  getitem = None
            return (clone,)
""",  # noqa: B950
            )

    @parametrize("dynamic", [True, False])
    @parametrize("backend", ["eager", "aot_eager"])
    def test_cond_mismatched_branch_strided_output(self, dynamic, backend):
        class M(torch.nn.Module):
            def forward(self, x, y):
                def true_fn(x, y):
                    return (
                        (x.swapaxes(-1, 0) + 1)
                        .unsqueeze(1)
                        .expand(-1, 5, -1, -1, -1, -1, -1),
                        torch.empty_strided((3, 3), (0, 1)),
                    )

                def false_fn(x, y):
                    return (
                        (y.swapaxes(-1, 0) + 1)
                        .unsqueeze(1)
                        .expand(-1, 4, -1, -1, -1, -1, -1),
                        torch.empty_strided((4, 5), (0, 1)),
                    )

                ret = torch.cond(x.sum() > 0, true_fn, false_fn, (x, y))
                return y.sum() + ret[0]

        m = M()
        x, y = torch.randn(1, 6, 1, 5, 4, 3), torch.randn(1, 4, 5, 1, 3, 8)
        out = m(x, y)
        compiled_out = torch.compile(
            m, backend=backend, dynamic=dynamic, fullgraph=True
        )(x, y)
        self.assertEqual(compiled_out, out)


class TestAutoFunctionalizeControlFlow(TestCase):
    def check(self, gen_fn, args, device, dynamic) -> torch.fx.GraphModule:
        args = pytree.tree_map(lambda t: t.to(device=device), args)

        def _clone(args):
            return [
                arg.clone() if isinstance(arg, torch.Tensor) else arg for arg in args
            ]

        def _new_fn():
            mod_or_fn = gen_fn()
            if isinstance(mod_or_fn, torch.nn.Module):
                mod_or_fn.to(device)
            return mod_or_fn

        # Only support input mutation in inference
        cloned_args = [_clone(args) for _ in range(3)]
        with torch.no_grad():
            exp = _new_fn()(*cloned_args[0])
        backend = AotEagerAndRecordGraphs()
        torch._dynamo.reset()
        with torch.no_grad():
            eager_out = torch.compile(
                _new_fn(), backend=backend, fullgraph=True, dynamic=dynamic
            )(*cloned_args[1])
        torch._dynamo.reset()
        with torch.no_grad():
            inductor_out = torch.compile(
                _new_fn(), backend="inductor", fullgraph=True, dynamic=dynamic
            )(*cloned_args[2])

        self.assertEqual(exp, eager_out)
        self.assertEqual(exp, inductor_out)
        self.assertEqual(cloned_args[0], cloned_args[1])
        self.assertEqual(cloned_args[0], cloned_args[2])
        return backend.fw_graphs[0]

    @requires_cuda
    @unittest.skipIf(not SM70OrLater, "triton")
    @parametrize("device", ["cuda", "cpu"])
    @parametrize("dynamic", [True, False])
    def test_cond_auto_functionalize_input_mutation(self, device, dynamic):
        class M(torch.nn.Module):
            def forward(self, x, y):
                def true_fn(x):
                    x.add_(1)
                    return x.sin()

                x = x.clone()
                ret = torch.cond(x.sum() > 0, true_fn, true_fn, (x,))
                return y + ret

        x, y = (
            torch.randn(3, 4, requires_grad=True),
            torch.randn(3, 4, requires_grad=True),
        )
        fw_gm = self.check(M, (x, y), device, dynamic)
        if not TEST_WITH_CROSSREF and not dynamic and device == "cuda":
            self.assertExpectedInline(
                normalize_gm(fw_gm.print_readable(print_output=False)),
                """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[3, 4]", arg1_1: "f32[3, 4]"):
        clone: "f32[3, 4]" = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None

        sum_1: "f32[]" = torch.ops.aten.sum.default(clone)
        gt: "b8[]" = torch.ops.aten.gt.Scalar(sum_1, 0);  sum_1 = None
        auto_functionalized_subgraph_0 = self.auto_functionalized_subgraph_0
        auto_functionalized_subgraph_1 = self.auto_functionalized_subgraph_1
        _tree_spec_constant0 = self._tree_spec_constant0
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.higher_order.cond, pred = gt, true_fn = auto_functionalized_subgraph_0, false_fn = auto_functionalized_subgraph_1, _operand0_base_index = 0, _all_bases = [clone], _op_schema = _tree_spec_constant0);  gt = auto_functionalized_subgraph_0 = auto_functionalized_subgraph_1 = clone = _tree_spec_constant0 = None
        getitem: "f32[3, 4]" = auto_functionalized_v2[0];  auto_functionalized_v2 = None

        add: "f32[3, 4]" = torch.ops.aten.add.Tensor(arg1_1, getitem);  arg1_1 = getitem = None
        return (add,)

    class auto_functionalized_subgraph_0(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 4]"):
            add: "f32[3, 4]" = torch.ops.aten.add.Tensor(arg0_1, 1)
            sin: "f32[3, 4]" = torch.ops.aten.sin.default(add)
            copy_: "f32[3, 4]" = torch.ops.aten.copy_.default(arg0_1, add);  arg0_1 = add = copy_ = None
            return (sin,)

    class auto_functionalized_subgraph_1(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 4]"):
            add: "f32[3, 4]" = torch.ops.aten.add.Tensor(arg0_1, 1)
            sin: "f32[3, 4]" = torch.ops.aten.sin.default(add)
            copy_: "f32[3, 4]" = torch.ops.aten.copy_.default(arg0_1, add);  arg0_1 = add = copy_ = None
            return (sin,)
""",  # noqa: B950
            )

    @requires_cuda
    @unittest.skipIf(not SM70OrLater, "triton")
    @parametrize("device", ["cuda", "cpu"])
    @parametrize("dynamic", [True, False])
    def test_cond_auto_functionalize_buffer_mutation(self, device, dynamic):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "buf", torch.ones(8, requires_grad=False, device=device)
                )

            def forward(self, p, x):
                def true_fn(x):
                    x.add_(1)
                    self.buf.add_(1)
                    return x + self.buf

                x = x.clone()
                out = torch.cond(p, true_fn, true_fn, (x,))
                return x + self.buf + out

        p, x = torch.tensor(True), torch.randn(1, requires_grad=True)
        fw_gm = self.check(M, (p, x), device, dynamic)
        if not TEST_WITH_CROSSREF and not dynamic and device == "cuda":
            self.assertExpectedInline(
                normalize_gm(fw_gm.print_readable(print_output=False)),
                """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[1]", arg1_1: "b8[]", arg2_1: "f32[8]"):
        clone: "f32[1]" = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None

        auto_functionalized_subgraph_0 = self.auto_functionalized_subgraph_0
        auto_functionalized_subgraph_1 = self.auto_functionalized_subgraph_1
        _tree_spec_constant0 = self._tree_spec_constant0
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.higher_order.cond, pred = arg1_1, true_fn = auto_functionalized_subgraph_0, false_fn = auto_functionalized_subgraph_1, _operand0_base_index = 0, _operand1_base_index = 1, _all_bases = [arg2_1, clone], _op_schema = _tree_spec_constant0);  arg1_1 = auto_functionalized_subgraph_0 = auto_functionalized_subgraph_1 = clone = _tree_spec_constant0 = None
        getitem: "f32[8]" = auto_functionalized_v2[0]
        getitem_1: "f32[8]" = auto_functionalized_v2[1]
        getitem_2: "f32[1]" = auto_functionalized_v2[2];  auto_functionalized_v2 = None

        add: "f32[8]" = torch.ops.aten.add.Tensor(getitem_2, getitem_1);  getitem_2 = None
        add_1: "f32[8]" = torch.ops.aten.add.Tensor(add, getitem);  add = getitem = None

        copy_: "f32[8]" = torch.ops.aten.copy_.default(arg2_1, getitem_1);  arg2_1 = getitem_1 = copy_ = None
        return (add_1,)

    class auto_functionalized_subgraph_0(torch.nn.Module):
        def forward(self, arg0_1: "f32[8]", arg1_1: "f32[1]"):
            add: "f32[1]" = torch.ops.aten.add.Tensor(arg1_1, 1)
            add_1: "f32[8]" = torch.ops.aten.add.Tensor(arg0_1, 1)
            add_2: "f32[8]" = torch.ops.aten.add.Tensor(add, add_1)
            copy_: "f32[8]" = torch.ops.aten.copy_.default(arg0_1, add_1);  arg0_1 = add_1 = copy_ = None
            copy__1: "f32[1]" = torch.ops.aten.copy_.default(arg1_1, add);  arg1_1 = add = copy__1 = None
            return (add_2,)

    class auto_functionalized_subgraph_1(torch.nn.Module):
        def forward(self, arg0_1: "f32[8]", arg1_1: "f32[1]"):
            add: "f32[1]" = torch.ops.aten.add.Tensor(arg1_1, 1)
            add_1: "f32[8]" = torch.ops.aten.add.Tensor(arg0_1, 1)
            add_2: "f32[8]" = torch.ops.aten.add.Tensor(add, add_1)
            copy_: "f32[8]" = torch.ops.aten.copy_.default(arg0_1, add_1);  arg0_1 = add_1 = copy_ = None
            copy__1: "f32[1]" = torch.ops.aten.copy_.default(arg1_1, add);  arg1_1 = add = copy__1 = None
            return (add_2,)
""",  # noqa: B950
            )

    @requires_cuda
    @unittest.skipIf(not SM70OrLater, "triton")
    @parametrize("device", ["cuda", "cpu"])
    @parametrize("dynamic", [True, False])
    def test_cond_auto_functionalize_union_input_mutation(self, device, dynamic):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.ones(4, 3, requires_grad=False))

            def forward(self, x, y):
                def true_fn(x):
                    x.add_(1)
                    return x.sin() @ self.buf

                def false_fn(x):
                    self.buf.add_(1)
                    return x.sin() @ self.buf

                x = x.clone()
                ret = torch.cond(x.sum() > 0, true_fn, false_fn, (x,))
                return y + ret + x.sum() + self.buf.sum()

        x, y = (
            torch.randn(3, 4, requires_grad=False),
            torch.randn(1, requires_grad=False),
        )
        fw_gm = self.check(M, (x, y), device, dynamic)
        if not TEST_WITH_CROSSREF and not dynamic and device == "cuda":
            self.assertExpectedInline(
                normalize_gm(fw_gm.print_readable(print_output=False)),
                """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[3, 4]", arg1_1: "f32[4, 3]", arg2_1: "f32[1]"):
        clone: "f32[3, 4]" = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None

        sum_1: "f32[]" = torch.ops.aten.sum.default(clone)
        gt: "b8[]" = torch.ops.aten.gt.Scalar(sum_1, 0);  sum_1 = None
        auto_functionalized_subgraph_0 = self.auto_functionalized_subgraph_0
        auto_functionalized_subgraph_1 = self.auto_functionalized_subgraph_1
        _tree_spec_constant0 = self._tree_spec_constant0
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.higher_order.cond, pred = gt, true_fn = auto_functionalized_subgraph_0, false_fn = auto_functionalized_subgraph_1, _operand0_base_index = 0, _operand1_base_index = 1, _all_bases = [arg1_1, clone], _op_schema = _tree_spec_constant0);  gt = auto_functionalized_subgraph_0 = auto_functionalized_subgraph_1 = clone = _tree_spec_constant0 = None
        getitem: "f32[3, 3]" = auto_functionalized_v2[0]
        getitem_1: "f32[4, 3]" = auto_functionalized_v2[1]
        getitem_2: "f32[3, 4]" = auto_functionalized_v2[2];  auto_functionalized_v2 = None

        add: "f32[3, 3]" = torch.ops.aten.add.Tensor(arg2_1, getitem);  arg2_1 = getitem = None
        sum_2: "f32[]" = torch.ops.aten.sum.default(getitem_2);  getitem_2 = None
        add_1: "f32[3, 3]" = torch.ops.aten.add.Tensor(add, sum_2);  add = sum_2 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(getitem_1)
        add_2: "f32[3, 3]" = torch.ops.aten.add.Tensor(add_1, sum_3);  add_1 = sum_3 = None

        copy_: "f32[4, 3]" = torch.ops.aten.copy_.default(arg1_1, getitem_1);  arg1_1 = getitem_1 = copy_ = None
        return (add_2,)

    class auto_functionalized_subgraph_0(torch.nn.Module):
        def forward(self, arg0_1: "f32[4, 3]", arg1_1: "f32[3, 4]"):
            add: "f32[3, 4]" = torch.ops.aten.add.Tensor(arg1_1, 1)
            sin: "f32[3, 4]" = torch.ops.aten.sin.default(add)
            mm: "f32[3, 3]" = torch.ops.aten.mm.default(sin, arg0_1);  sin = arg0_1 = None
            copy_: "f32[3, 4]" = torch.ops.aten.copy_.default(arg1_1, add);  arg1_1 = add = copy_ = None
            return (mm,)

    class auto_functionalized_subgraph_1(torch.nn.Module):
        def forward(self, arg0_1: "f32[4, 3]", arg1_1: "f32[3, 4]"):
            add: "f32[4, 3]" = torch.ops.aten.add.Tensor(arg0_1, 1)
            sin: "f32[3, 4]" = torch.ops.aten.sin.default(arg1_1);  arg1_1 = None
            mm: "f32[3, 3]" = torch.ops.aten.mm.default(sin, add);  sin = None
            copy_: "f32[4, 3]" = torch.ops.aten.copy_.default(arg0_1, add);  arg0_1 = add = copy_ = None
            return (mm,)
""",  # noqa: B950
            )


_hop_schema_test_schema_types = [
    "bool",
    "int",
    "float",
    "str",
    "Tensor",
    "SymInt",
    "SymBool",
    "GraphModule",
    "ScriptObj",
]


@skipIfTorchDynamo("We don't expect users to torch.compile hop schema generation.")
@unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
class TestHopSchema(TestCase):
    def _get_example_val(self, ty: str):
        from torch.fx.experimental.sym_node import SymNode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        def create_symtype(cls, pytype, shape_env, val):
            from torch._dynamo.source import ConstantSource

            symbol = shape_env.create_symbol(
                val,
                source=ConstantSource(
                    f"__testing_hop_schema{len(shape_env.backed_var_to_val)}"
                ),
            )
            return cls(SymNode(symbol, shape_env, pytype, hint=val))

        if ty == "bool":
            return True
        elif ty == "int":
            return 1
        elif ty == "float":
            return 1.0
        elif ty == "str":
            return "foo"
        elif ty == "Tensor":
            return torch.tensor(1)
        elif ty == "SymInt":
            shape_env = ShapeEnv()
            return create_symtype(torch.SymInt, int, shape_env, 1)
        elif ty == "SymBool":
            shape_env = ShapeEnv()
            return create_symtype(torch.SymBool, bool, shape_env, True)
        elif ty == "GraphModule":

            def f(x):
                return x.sin()

            return make_fx(f)(torch.ones(1))
        elif ty == "ScriptObj":
            from torch.testing._internal.torchbind_impls import (
                init_torchbind_implementations,
            )

            init_torchbind_implementations()
            foo = torch.classes._TorchScriptTesting._Foo(3, 4)
            return foo
        else:
            raise NotImplementedError(ty)

    @parametrize("schema_type", _hop_schema_test_schema_types)
    def test_type_gen(self, schema_type):
        from torchgen.gen_schema_utils import TypeGen

        example_val = self._get_example_val(schema_type)
        ty = TypeGen.from_example(example_val)
        # Test the generated type can be parsed
        self.assertEqual(ty.parse(str(ty)), ty)

    @parametrize("schema_type", _hop_schema_test_schema_types)
    def test_list_gen(self, schema_type):
        from torchgen.gen_schema_utils import TypeGen

        example_val = self._get_example_val(schema_type)
        li1 = [example_val]
        ty1 = TypeGen.from_example(li1)
        ty2 = TypeGen.from_example(li1)
        self.assertEqual(ty1.parse(str(ty1)), ty1)
        self.assertEqual(ty2.parse(str(ty2)), ty2)

    def test_function_schema_gen(self):
        from torchgen.gen_schema_utils import FunctionSchemaGen

        inps = [
            (schema_type + "_v", self._get_example_val(schema_type))
            for schema_type in _hop_schema_test_schema_types
        ]
        schema1 = FunctionSchemaGen.from_example("test_op1", inps, torch.ones(1))
        schema2 = FunctionSchemaGen.from_example(
            "test_op2",
            inps,
            [
                torch.ones(1),
            ],
        )
        schema3 = FunctionSchemaGen.from_example(
            "test_op3", inps, [torch.ones(1), torch.ones(1)]
        )
        self.assertExpectedInline(
            str(schema1),
            """test_op1(bool bool_v, int int_v, float float_v, str str_v, Tensor Tensor_v, SymInt SymInt_v, SymBool SymBool_v, GraphModule GraphModule_v, __torch__.torch.classes._Foo ScriptObj_v) -> Tensor""",  # noqa: B950
        )
        self.assertExpectedInline(
            str(schema2),
            """test_op2(bool bool_v, int int_v, float float_v, str str_v, Tensor Tensor_v, SymInt SymInt_v, SymBool SymBool_v, GraphModule GraphModule_v, __torch__.torch.classes._Foo ScriptObj_v) -> Tensor""",  # noqa: B950
        )
        self.assertExpectedInline(
            str(schema3),
            """test_op3(bool bool_v, int int_v, float float_v, str str_v, Tensor Tensor_v, SymInt SymInt_v, SymBool SymBool_v, GraphModule GraphModule_v, __torch__.torch.classes._Foo ScriptObj_v) -> (Tensor, Tensor)""",  # noqa: B950,
        )
        self.assertEqual(schema1.parse(str(schema1)), schema1)
        self.assertEqual(schema2.parse(str(schema2)), schema2)
        self.assertEqual(schema3.parse(str(schema3)), schema3)

    def test_schema_tree_spec(self):
        schema_gen = HopSchemaGenerator(torch.ops.higher_order.cond)
        args = (torch.randn(3, 4), torch.randn(2, 3))
        with self.assertRaisesRegex(
            RuntimeError, "Please only add flattened inputs to the hop schema"
        ):
            schema_gen.add_arg("tuple_args", args)

        for i, arg in enumerate(args):
            schema_gen.add_arg(f"tuple_args{i}", arg)
        schema_gen.add_schema_tree_spec(pytree.tree_flatten(args)[1])
        flat_schema = schema_gen.gen_schema()
        self.assertExpectedInline(
            str(flat_schema), """cond(Tensor tuple_args0, Tensor tuple_args1) -> ()"""
        )

    def test_cond_gen_schema_tensor_inputs(self):
        schema = torch.ops.higher_order.cond.gen_schema(
            torch.tensor(True),
            lambda x: x.sin(),
            lambda x: x.cos(),
            (torch.randn(3, 4),),
        )
        self.assertExpectedInline(
            str(schema),
            """cond(Tensor pred, Any true_fn, Any false_fn, Tensor operand0) -> ((Tensor))""",
        )

    def test_cond_gen_schema_symbool_inputs(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        with fake_mode, fake_mode.shape_env.ignore_fresh_unbacked_symbols():
            sym_bool = torch.randn(3, 4).nonzero().size(0) == 0

        schema = torch.ops.higher_order.cond.gen_schema(
            sym_bool,
            lambda x: x.sin(),
            lambda x: x.cos(),
            (torch.randn(3, 4),),
        )
        self.assertExpectedInline(
            str(schema),
            """cond(SymBool pred, Any true_fn, Any false_fn, Tensor operand0) -> ((Tensor))""",
        )

    def test_while_loop_gen_schema_tensor_inputs(self):
        def cond_fn(x, y):
            return x.sum() < 10

        def body_fn(x, y):
            return x + 1, y.sin()

        schema = torch.ops.higher_order.while_loop.gen_schema(
            cond_fn,
            body_fn,
            (torch.randn(3, 4), torch.randn(2, 3)),
            (),
        )
        self.assertExpectedInline(
            str(schema),
            """while_loop(Any cond_fn, Any body_fn, Tensor carried_input0, Tensor carried_input1) -> (Tensor, Tensor)""",
        )

    def test_while_loop_gen_schema_with_additional_inputs(self):
        def cond_fn(x, y, z):
            return x.sum() < z

        def body_fn(x, y, z):
            return x + 1, y.sin()

        schema = torch.ops.higher_order.while_loop.gen_schema(
            cond_fn,
            body_fn,
            (torch.randn(3, 4), torch.randn(2, 3)),
            (torch.tensor(10),),
        )
        self.assertExpectedInline(
            str(schema),
            """while_loop(Any cond_fn, Any body_fn, Tensor carried_input0, Tensor carried_input1, Tensor additional_input0) -> (Tensor, Tensor)""",  # noqa: B950
        )

    def test_scan_gen_schema_tensor_inputs(self):
        def combine_fn(carry, x):
            return carry + x, carry * x

        schema = torch.ops.higher_order.scan.gen_schema(
            combine_fn,
            (torch.randn(3, 4),),
            (torch.randn(5, 3, 4),),
            (),
        )
        self.assertExpectedInline(
            str(schema),
            """scan(Any combine_fn, Tensor init0, Tensor xs0) -> (Tensor, Tensor)""",
        )

    def test_scan_gen_schema_with_additional_inputs(self):
        def combine_fn(carry, x, scale):
            return carry + x * scale, carry * x

        schema = torch.ops.higher_order.scan.gen_schema(
            combine_fn,
            (torch.randn(3, 4),),
            (torch.randn(5, 3, 4),),
            (torch.tensor(2.0),),
        )
        self.assertExpectedInline(
            str(schema),
            """scan(Any combine_fn, Tensor init0, Tensor xs0, Tensor additional_input0) -> (Tensor, Tensor)""",  # noqa: B950
        )

    def test_scan_gen_schema_multiple_inputs(self):
        def combine_fn(carry1, carry2, x1, x2):
            return carry1 + x1, carry2 * x2, carry1 - x1, carry2 + x2

        schema = torch.ops.higher_order.scan.gen_schema(
            combine_fn,
            (torch.randn(3, 4), torch.randn(2, 3)),
            (torch.randn(5, 3, 4), torch.randn(5, 2, 3)),
            (),
        )
        self.assertExpectedInline(
            str(schema),
            """scan(Any combine_fn, Tensor init0, Tensor init1, Tensor xs0, Tensor xs1) -> (Tensor, Tensor, Tensor, Tensor)""",  # noqa: B950
        )

    def test_associative_scan_gen_schema_tensor_inputs(self):
        def combine_fn(x, y):
            return x + y

        schema = torch.ops.higher_order.associative_scan.gen_schema(
            combine_fn,
            (torch.randn(5, 3, 4),),
            (),
        )
        self.assertExpectedInline(
            str(schema),
            """associative_scan(Any combine_fn, Tensor xs0) -> ((Tensor))""",
        )

    def test_associative_scan_gen_schema_with_additional_inputs(self):
        def combine_fn(x, y, scale):
            return x * y * scale

        schema = torch.ops.higher_order.associative_scan.gen_schema(
            combine_fn,
            (torch.randn(5, 3, 4),),
            (torch.tensor(2.0),),
        )
        self.assertExpectedInline(
            str(schema),
            """associative_scan(Any combine_fn, Tensor xs0, Tensor additional_input0) -> ((Tensor))""",
        )

    def test_associative_scan_gen_schema_multiple_inputs(self):
        def combine_fn(x1, x2, y1, y2):
            return x1 + y1, x2 * y2

        schema = torch.ops.higher_order.associative_scan.gen_schema(
            combine_fn,
            (torch.randn(5, 3, 4), torch.randn(5, 2, 3)),
            (),
        )
        self.assertExpectedInline(
            str(schema),
            """associative_scan(Any combine_fn, Tensor xs0, Tensor xs1) -> (Tensor, Tensor)""",
        )

    def test_while_loop_gen_schema_with_int_carries(self):
        def cond_fn(x, y, z, c):
            return x < y

        def body_fn(x, y, z, c):
            return x + 1, y - 1, z.sin(), c + x

        schema = torch.ops.higher_order.while_loop.gen_schema(
            cond_fn,
            body_fn,
            (2, 10, torch.randn(2, 3)),
            (torch.tensor(10),),
        )
        self.assertExpectedInline(
            str(schema),
            """while_loop(Any cond_fn, Any body_fn, int carried_input0, int carried_input1, Tensor carried_input2, Tensor additional_input0) -> (int, int, Tensor, Tensor)""",  # noqa: B950
        )

    def test_while_loop_gen_schema_with_input_mutation(self):
        def cond_fn(x, y, z, c):
            return x < y

        def body_fn(x, y, z, c):
            x.add_(1)
            y.sub_(1)
            z.sin_()
            c.add_(x)
            return x, y, z

        c = torch.randn(3, 3)

        schema = torch.ops.higher_order.while_loop.gen_schema(
            cond_fn,
            body_fn,
            (torch.randn(3, 3), torch.randn(3, 3), torch.randn(3, 3)),
            (c,),
        )
        self.assertExpectedInline(
            str(schema),
            """while_loop(Any cond_fn, Any body_fn, Tensor(a2!) carried_input0, Tensor(a3!) carried_input1, Tensor(a4!) carried_input2, Tensor(a5!) additional_input0) -> (Tensor, Tensor, Tensor)""",  # noqa: B950
        )


class DynamicCondModel(torch.nn.Module):
    def __init__(self, input_size=16, hidden_size=64, output_size=10):
        super().__init__()
        self.fc1_0 = torch.nn.Linear(input_size, hidden_size)
        self.fc1_1 = torch.nn.Linear(input_size, 32)
        self.fc1_2 = torch.nn.Linear(32, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        def true_fn(x):
            return self.fc1_0(x)

        def false_fn(x):
            x = self.fc1_1(x)
            return self.fc1_2(x)

        pred = x.sum() > 0
        x = cond(pred, true_fn, false_fn, [x])

        x = self.relu(x)
        x = self.fc2(x)

        return x


@unittest.skipIf(
    not TEST_CUDA_GRAPH_CONDITIONAL_NODES,
    "CUDA 12.4 or greater is required for CUDA Graphs with conditional nodes",
)
class TestControlFlowNN(TestCase):
    def test_cond_in_NN(self):
        model = DynamicCondModel().cuda()

        def autograd_test(x):
            model.zero_grad(set_to_none=True)
            output = model(x)
            loss = output.sum()
            loss.backward()
            grads = [p.grad for p in model.parameters()]
            return (output, loss, grads)

        x = torch.randn(16, device="cuda")

        _check_compile_many_backends_with_cudagraph(self, autograd_test, [x])
        _check_compile_cudagraph_backend(self, autograd_test, [x])


@unittest.skipIf(
    not TEST_CUDA_GRAPH_CONDITIONAL_NODES,
    "CUDA 12.4 or greater is required for CUDA Graphs with conditional nodes",
)
class TestControlFlowAndRNG(TestCase):
    @parametrize("rng_func", ["custom_generator", "default_generator"])
    def test_rng_with_conditional_nodes_errors(self, rng_func):
        pred = torch.tensor(True, device="cuda")
        x = torch.ones(10, dtype=torch.float32, device="cuda")

        if rng_func == "custom_generator":
            self.skipTest(
                "randn() currently does not work with a generator argument in dynamo."
            )
            generator = torch.Generator("cuda")

            def custom_generator(x):
                return x + torch.randn(
                    *x.shape, generator=generator, dtype=x.dtype, device=x.device
                )

            rng_func = custom_generator
        elif rng_func == "default_generator":

            def default_generator(x):
                return x + torch.randn(*x.shape, dtype=x.dtype, device=x.device)

            rng_func = default_generator

        def func(pred, x):
            return torch.cond(pred, rng_func, lambda x: 2 * x, [x])

        compiled_func = torch.compile(func, backend="cudagraphs")
        with self.assertRaisesRegex(
            RuntimeError,
            "RNG within data-dependent conditional nodes is not supported yet",
        ):
            compiled_func(pred, x)

    def test_rng_outside_conditional_nodes_does_not_error(self):
        pred = torch.tensor(True, device="cuda")
        x = torch.ones(10, dtype=torch.float32, device="cuda")

        def func(pred, x):
            y = torch.cond(pred, lambda t: 2 * t, lambda t: 3 * t, [x])
            return y + torch.randn(*y.shape, dtype=y.dtype, device=y.device)

        compiled_func = torch.compile(func, backend="cudagraphs")
        for _ in range(3):
            out = compiled_func(pred, x)
            self.assertEqual(out.shape, x.shape)


instantiate_parametrized_tests(TestHopSchema)
instantiate_parametrized_tests(TestControlFlowTraced)
instantiate_parametrized_tests(TestAutoFunctionalizeControlFlow)

instantiate_parametrized_tests(TestControlFlow)
instantiate_parametrized_tests(AssociativeScanTests)

instantiate_parametrized_tests(TestControlFlowAndRNG)

if __name__ == "__main__":
    run_tests()
