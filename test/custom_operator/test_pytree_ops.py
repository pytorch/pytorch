# Owner(s): ["module: custom-operators"]

from dataclasses import dataclass

import torch
import torch.utils._pytree as pytree
from torch._dynamo.testing import AotEagerAndRecordGraphs
from torch._library.infer_schema import infer_schema
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


@dataclass
class Point:
    x: torch.Tensor
    y: torch.Tensor


pytree.register_dataclass(Point)


class TestPytreeOps(TestCase):
    def setUp(self):
        self.lib = torch.library.Library("_TestPytreeOps", "FRAGMENT")  # noqa: TOR901
        super().setUp()

    def tearDown(self):
        self.lib._destroy()
        super().tearDown()

    def test_schema_inference_list_types_before_pytree(self):
        def fn_list_tensor(
            list_tensor: list[torch.Tensor],
            list_int: list[int],
            list_float: list[float],
            list_bool: list[bool],
            pytree_list: list,
        ) -> torch.Tensor:
            return list_tensor[0]

        schema = infer_schema(fn_list_tensor, mutates_args=())
        self.assertEqual(
            schema,
            "(Tensor[] list_tensor, SymInt[] list_int, float[] list_float, bool[] list_bool, builtins.list pytree_list) -> Tensor",
        )

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_dict_input(self):
        # Use define/impl API instead of custom_op
        torch.library.define(
            "_TestPytreeOps::dict_op",
            "(builtins.dict d, Tensor t) -> Tensor",
            lib=self.lib,
        )

        @torch.library.impl("_TestPytreeOps::dict_op", "CPU", lib=self.lib)
        def dict_op_impl(d: dict, t: torch.Tensor) -> torch.Tensor:
            return torch.sin(d["x"] - d["y"] + t)

        d = {"x": torch.randn(2, 3), "y": torch.randn(2, 3)}
        t = torch.randn(2, 3)
        y = torch.ops._TestPytreeOps.dict_op(d, t)
        self.assertEqual(y, torch.sin(d["x"] - d["y"] + t))

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_list_input(self):
        @torch.library.custom_op("_TestPytreeOps::list_op", mutates_args=())
        def foo(lst: list, t: torch.Tensor) -> torch.Tensor:
            return torch.sin(lst[0] + lst[1] + t)

        lst = [torch.randn(2, 3), torch.randn(2, 3)]
        t = torch.randn(2, 3)
        y = torch.ops._TestPytreeOps.list_op(lst, t)
        self.assertEqual(y, torch.sin(lst[0] + lst[1] + t))

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_dataclass_input(self):
        @torch.library.custom_op("_TestPytreeOps::dataclass_op", mutates_args=())
        def dataclass_op_impl(a: Point) -> torch.Tensor:
            return torch.sqrt(torch.sum((a.x - a.y) ** 2))

        x = Point(x=torch.randn(2, 3), y=torch.randn(2, 3))
        y = torch.ops._TestPytreeOps.dataclass_op(x)
        self.assertEqual(y, torch.sqrt(torch.sum((x.x - x.y) ** 2)))

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_tuple_input(self):
        @torch.library.custom_op("_TestPytreeOps::tuple_op", mutates_args=())
        def foo(tup: tuple, t: torch.Tensor) -> torch.Tensor:
            return torch.cos(tup[0] * tup[1] + t)

        tup = (torch.randn(2, 3), torch.randn(2, 3))
        t = torch.randn(2, 3)
        y = torch.ops._TestPytreeOps.tuple_op(tup, t)
        self.assertEqual(y, torch.cos(tup[0] * tup[1] + t))

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_nested_pytree_input(self):
        @torch.library.custom_op("_TestPytreeOps::nested_op", mutates_args=())
        def foo(d: dict) -> torch.Tensor:
            return torch.sin(d["a"][0] + d["b"]["x"])

        d = {
            "a": [torch.randn(2, 3), torch.randn(2, 3)],
            "b": {"x": torch.randn(2, 3), "y": torch.randn(2, 3)},
        }
        y = torch.ops._TestPytreeOps.nested_op(d)
        self.assertEqual(y, torch.sin(d["a"][0] + d["b"]["x"]))

    def test_compile(self):
        @torch.library.custom_op("mylib::foo", mutates_args=())
        def foo(p: Point, t: torch.Tensor) -> torch.Tensor:
            return torch.sin(p.x - p.y + t)

        @foo.register_fake
        def _(p: Point, t: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(t)

        p = Point(x=torch.randn(2, 3), y=torch.randn(2, 3))
        t = torch.randn(2, 3)

        backend = AotEagerAndRecordGraphs()

        def fn(p, t):
            return torch.ops.mylib.foo(p, t)

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        self.assertEqual(compiled_fn(p, t), fn(p, t))

        actual_graph = torch._dynamo.testing.normalize_gm(
            backend.graphs[0].print_readable(print_output=False)
        )
        self.assertExpectedInline(
            actual_graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_t_: "f32[2, 3]", L_p_x: "f32[2, 3]", L_p_y: "f32[2, 3]"):
        l_t_ = L_t_
        l_p_x = L_p_x
        l_p_y = L_p_y

        mylib_foo_input_spec : torch.utils._pytree.TreeSpec = self.mylib_foo_input_spec
        flat_apply: "f32[2, 3]" = torch.ops.higher_order.flat_apply(torch.ops.mylib.foo.default, mylib_foo_input_spec, l_p_x, l_p_y, l_t_);  mylib_foo_input_spec = l_p_x = l_p_y = l_t_ = None
        return (flat_apply,)
""",  # noqa: B950
        )

        actual_graph = torch._dynamo.testing.normalize_gm(
            backend.fw_graphs[0].print_readable(print_output=False)
        )
        self.assertExpectedInline(
            actual_graph,
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 3]", arg1_1: "f32[2, 3]", arg2_1: "f32[2, 3]"):
        _tree_spec_constant0 = self._tree_spec_constant0
        flat_apply_foo: "f32[2, 3]" = torch.ops.higher_order.flat_apply(torch.ops.mylib.foo.default, _tree_spec_constant0, arg1_1, arg2_1, arg0_1);  _tree_spec_constant0 = arg1_1 = arg2_1 = arg0_1 = None
        return (flat_apply_foo,)
""",  # noqa: B950
        )

        compiled_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        self.assertEqual(compiled_fn(p, t), fn(p, t))

    def test_compile_backward(self):
        @torch.library.custom_op("mylib::backward_foo", mutates_args=())
        def backward_foo(p: Point, t: torch.Tensor) -> torch.Tensor:
            return p.x * p.y + t

        @backward_foo.register_fake
        def _(p: Point, t: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(t)

        def setup_context(ctx, inputs, output):
            p, t = inputs
            ctx.save_for_backward(p.x, p.y)

        def backward(ctx, grad_output):
            x, y = ctx.saved_tensors
            return None, grad_output

        backward_foo.register_autograd(backward, setup_context=setup_context)

        p = Point(x=torch.randn(2, 3), y=torch.randn(2, 3))
        t = torch.randn(2, 3, requires_grad=True)

        @torch.compile(backend="inductor", fullgraph=True)
        def fn(p, t):
            return torch.ops.mylib.backward_foo(p, t)

        result = fn(p, t)
        expected = p.x * p.y + t
        self.assertEqual(result, expected)

        grad_out = torch.ones_like(result)
        result.backward(grad_out)
        self.assertEqual(t.grad, grad_out)

    def test_autograd_function_with_pytree_backward(self):
        @torch.library.custom_op("mylib::pytree_backward_op", mutates_args=())
        def pytree_backward_op(p: Point, grad: torch.Tensor) -> torch.Tensor:
            return grad * (p.x + p.y)

        @pytree_backward_op.register_fake
        def _(p: Point, grad: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(grad)

        class PytreeBackwardFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, p):
                ctx.p = p
                return x * 2

            @staticmethod
            def backward(ctx, grad_output):
                grad_input = torch.ops.mylib.pytree_backward_op(ctx.p, grad_output)
                return grad_input, None

        p = Point(x=torch.randn(2, 3), y=torch.randn(2, 3))
        x = torch.randn(2, 3, requires_grad=True)

        result = PytreeBackwardFunction.apply(x, p)
        grad_out = torch.ones_like(result)
        result.backward(grad_out)

        x2 = torch.randn(2, 3, requires_grad=True)
        backend = AotEagerAndRecordGraphs()

        @torch.compile(backend=backend, fullgraph=True)
        def fn(x, p):
            return PytreeBackwardFunction.apply(x, p)

        result = fn(x2, p)
        result.backward(grad_out)
        self.assertEqual(x2.grad, x.grad)

        self.assertExpectedInline(
            backend.bw_graphs[0].code.strip(),
            """\
def forward(self, primals_2, primals_3, tangents_1):
    _tree_spec_constant0 = self._tree_spec_constant0
    flat_apply_pytree_backward_op = torch.ops.higher_order.flat_apply(torch.ops.mylib.pytree_backward_op.default, _tree_spec_constant0, primals_2, primals_3, tangents_1);  _tree_spec_constant0 = primals_2 = primals_3 = tangents_1 = None
    return (flat_apply_pytree_backward_op, None, None)""",  # noqa: B950
        )


if __name__ == "__main__":
    run_tests()
