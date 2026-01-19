# Owner(s): ["module: dynamo", "module: higher order operators"]
import re
from dataclasses import dataclass

import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch._dynamo.testing import (
    AotEagerAndRecordGraphs,
    EagerAndRecordGraphs,
    extract_graph,
    normalize_gm,
    remove_trailing_space,
)
from torch._higher_order_ops.flat_apply import (
    flat_apply,
    func_to_graphable,
    is_graphable,
    to_graphable,
)
from torch.testing._internal.common_utils import skipIfTorchDynamo
from torch.testing._internal.dynamo_pytree_test_utils import PytreeRegisteringTestCase


def distance(a, b, norm):
    if norm.typ == "l2":
        return torch.sqrt((a.x - b.x).pow(2) + (a.y - b.y).pow(2))
    elif norm.typ == "l1":
        return (a.x - b.x).abs() + (a.y - b.y).abs()


@dataclass(frozen=True)
class Norm:
    typ: str

    def __fx_repr__(self):
        return f"Norm(typ={self.typ!r})", {"Norm": Norm}


torch._library.opaque_object.register_opaque_type(Norm, typ="value")


@dataclass
class Point:
    x: Tensor
    y: Tensor


pytree.register_dataclass(Point)


@dataclass
class InputData:
    count: int
    values: Tensor


torch.utils._pytree.register_dataclass(InputData)


@dataclass
class InputInvalid:
    count: int
    values: Tensor


# No pytree for InputInvalid


@dataclass
class OutputData:
    result1: Tensor
    result2: Tensor


torch.utils._pytree.register_dataclass(OutputData)


@dataclass
class OutputInvalid:
    result1: Tensor
    result2: Tensor


# No pytree for OutputInvalid


class FlatApplyTests(PytreeRegisteringTestCase):
    def test_simple(self):
        tensor = torch.tensor

        a = Point(tensor(0.0), tensor(0.0))
        b = Point(tensor(3.0), tensor(4.0))
        norm = Norm("l2")

        args = (a, b)
        kwargs = {"norm": norm}

        empty_list, func_spec = func_to_graphable(distance)
        self.assertEqual(empty_list, [])

        flat_args, in_spec = to_graphable((args, kwargs))

        for arg in flat_args:
            self.assertTrue(is_graphable(arg))

        # Test flat_apply returns same thing as original function
        result = flat_apply(func_spec, in_spec, *flat_args)
        self.assertEqual(result, distance(*args, **kwargs))

    def test_non_tensor_output(self):
        tensor = torch.tensor

        a = Point(tensor(0.0), tensor(0.0))
        b = Point(tensor(3.0), tensor(4.0))

        args = (a, b)
        kwargs = {}

        def f(a, b):
            return [a.x + 1, (b.x + 2, [a.y + 3, 4.0], "5"), 6 + b.y]

        empty_list, func_spec = func_to_graphable(f)
        self.assertEqual(empty_list, [])

        flat_args, in_spec = to_graphable((args, kwargs))

        for arg in flat_args:
            self.assertTrue(is_graphable(arg))

        # Test flat_apply returns same thing as original function
        result = flat_apply(func_spec, in_spec, *flat_args)
        self.assertEqual(result, f(*args, **kwargs))

    def test_nonstrict_trace_dynamo_graph(self):
        class Point:
            x: Tensor
            y: Tensor

            def __init__(self, x, y):
                self.x = x
                self.y = y

        class PointTensor:
            p: Point
            t: Tensor

            def __init__(self, p, t):
                self.p = p
                self.t = t

        self.register_pytree_node(
            PointTensor,
            lambda pt: ((pt.p, pt.t), ()),
            lambda pt, _: PointTensor(pt[0], pt[1]),
            serialized_type_name=f"{PointTensor.__module__}.{PointTensor.__qualname__}",
        )

        self.register_pytree_node(
            Point,
            lambda p: ((p.x, p.y), ()),
            lambda xy, _: Point(xy[0], xy[1]),
            serialized_type_name=f"{Point.__module__}.{Point.__qualname__}",
        )

        def trace_point(p):
            torch._dynamo.graph_break()
            return p.x * p.y

        @torch._dynamo.nonstrict_trace
        def trace_point_tensor(pt):
            torch._dynamo.graph_break()
            return pt.t + trace_point(pt.p)

        backend = EagerAndRecordGraphs()

        @torch.compile(fullgraph=True, backend=backend)
        def fn(x, y):
            p = Point(x, y)
            t = x + y
            pt = PointTensor(p, t)
            res = trace_point_tensor(pt)
            return res

        fn(torch.randn(10), torch.randn(10))
        self.assertExpectedInline(
            normalize_gm(backend.graphs[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[10]", L_y_: "f32[10]"):
        l_x_ = L_x_
        l_y_ = L_y_

        t: "f32[10]" = l_x_ + l_y_

        trace_point_tensor_spec : torch.utils._pytree.TreeSpec = self.trace_point_tensor_spec
        trace_point_tensor_input_spec : torch.utils._pytree.TreeSpec = self.trace_point_tensor_input_spec
        flat_apply_capture = torch__dynamo_variables_torch_flat_apply_capture(trace_point_tensor_spec, trace_point_tensor_input_spec, l_x_, l_y_, t);  trace_point_tensor_spec = trace_point_tensor_input_spec = l_x_ = l_y_ = t = None
        res: "f32[10]" = flat_apply_capture[0];  flat_apply_capture = None
        return (res,)
""",  # NOQA: B950
        )

    def test_nonstrict_trace_captured_tensor_post_aot_graph(self):
        cst = torch.ones(1)

        @torch._dynamo.nonstrict_trace
        def trace_me(x, y):
            torch._dynamo.graph_break()
            return x * y + cst

        backend = AotEagerAndRecordGraphs()

        @torch.compile(fullgraph=True, backend=backend)
        def fn(x, y):
            return trace_me(x, y)

        fn(torch.randn(10), torch.randn(10))
        self.assertExpectedInline(
            normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[10]", arg1_1: "f32[10]"):
        mul: "f32[10]" = torch.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        _tensor_constant0: "f32[1]" = self._tensor_constant0
        add: "f32[10]" = torch.ops.aten.add.Tensor(mul, _tensor_constant0);  mul = _tensor_constant0 = None
        return (add,)
""",  # NOQA: B950
        )


@skipIfTorchDynamo("Not a suitable dynamo wrapped test")
class TestInputOutput(PytreeRegisteringTestCase):
    def test_simple(self):
        a = 4
        b = torch.randn(4, 4)

        @torch._dynamo.nonstrict_trace
        def gn(i_count: int, i_values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            output_tensor = a + b * i_count * i_values

            result1 = output_tensor + i_count
            result2 = output_tensor * (i_count + 1)

            return output_tensor, result1, result2

        def fn(i: InputData) -> Tensor:
            x = torch.sin(i.values)
            y, z_result1, z_result2 = gn(i.count, i.values)
            return x + y + z_result1 + z_result2

        count = 5
        values = torch.randn(4, 4)
        i = InputData(count, values)
        ref = fn(i)
        opt_fn = torch.compile(lambda i: fn(i), backend="aot_eager", fullgraph=True)
        res = opt_fn(i)
        self.assertEqual(ref, res)

        _, gms, _, _ = extract_graph(lambda i: fn(i), i)
        self.assertExpectedInline(
            print_graph(gms[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_i_values: "f32[4, 4]"):
        l_i_values = L_i_values

        # code: x = torch.sin(i.values)
        x: "f32[4, 4]" = torch.sin(l_i_values)

        # code: y, z_result1, z_result2 = gn(i.count, i.values)
        gn_spec : torch.utils._pytree.TreeSpec = self.gn_spec
        gn_input_spec : torch.utils._pytree.TreeSpec = self.gn_input_spec
        flat_apply_capture = torch__dynamo_variables_torch_flat_apply_capture(gn_spec, gn_input_spec, 5, l_i_values);  gn_spec = gn_input_spec = l_i_values = None
        y: "f32[4, 4]" = flat_apply_capture[0]
        z_result1: "f32[4, 4]" = flat_apply_capture[1]
        z_result2: "f32[4, 4]" = flat_apply_capture[2];  flat_apply_capture = None

        # code: return x + y + z_result1 + z_result2
        add: "f32[4, 4]" = x + y;  x = y = None
        add_1: "f32[4, 4]" = add + z_result1;  add = z_result1 = None
        add_2: "f32[4, 4]" = add_1 + z_result2;  add_1 = z_result2 = None
        return (add_2,)
""",  # NOQA: B950
        )

    def test_dataclass_input(self):
        a = 4
        b = torch.randn(4, 4)

        @torch._dynamo.nonstrict_trace
        def gn(i: InputData) -> tuple[Tensor, Tensor, Tensor]:
            output_tensor = a + b * i.count * i.values

            result1 = output_tensor + i.count
            result2 = output_tensor * (i.count + 1)

            return output_tensor, result1, result2

        def fn(i: InputData) -> Tensor:
            x = torch.sin(i.values)
            y, z_result1, z_result2 = gn(i)
            return x + y + z_result1 + z_result2

        count = 5
        values = torch.randn(4, 4)
        i = InputData(count, values)
        ref = fn(i)
        opt_fn = torch.compile(lambda i: fn(i), backend="aot_eager", fullgraph=True)
        res = opt_fn(i)
        self.assertEqual(ref, res)

        _, gms, _, _ = extract_graph(lambda i: fn(i), i)
        self.assertExpectedInline(
            print_graph(gms[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_i_values: "f32[4, 4]"):
        l_i_values = L_i_values

        # code: x = torch.sin(i.values)
        x: "f32[4, 4]" = torch.sin(l_i_values)

        # code: y, z_result1, z_result2 = gn(i)
        gn_spec : torch.utils._pytree.TreeSpec = self.gn_spec
        gn_input_spec : torch.utils._pytree.TreeSpec = self.gn_input_spec
        flat_apply_capture = torch__dynamo_variables_torch_flat_apply_capture(gn_spec, gn_input_spec, 5, l_i_values);  gn_spec = gn_input_spec = l_i_values = None
        y: "f32[4, 4]" = flat_apply_capture[0]
        z_result1: "f32[4, 4]" = flat_apply_capture[1]
        z_result2: "f32[4, 4]" = flat_apply_capture[2];  flat_apply_capture = None

        # code: return x + y + z_result1 + z_result2
        add: "f32[4, 4]" = x + y;  x = y = None
        add_1: "f32[4, 4]" = add + z_result1;  add = z_result1 = None
        add_2: "f32[4, 4]" = add_1 + z_result2;  add_1 = z_result2 = None
        return (add_2,)
""",  # NOQA: B950
        )

    def test_invalid_input(self):
        a = 4
        b = torch.randn(4, 4)

        @torch._dynamo.nonstrict_trace
        def gn(i: InputInvalid) -> tuple[Tensor, Tensor, Tensor]:
            output_tensor = a + b * i.count * i.values

            result1 = output_tensor + i.count
            result2 = output_tensor * (i.count + 1)

            return output_tensor, result1, result2

        def fn(i: InputInvalid) -> Tensor:
            x = torch.sin(i.values)
            y, z_result1, z_result2 = gn(i)
            return x + y + z_result1 + z_result2

        count = 5
        values = torch.randn(4, 4)
        i = InputInvalid(count, values)
        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Invalid input type for nonstrict_trace-ed function",
        ):
            opt_fn(i)

    def test_dataclass_output(self):
        a = 4
        b = torch.randn(4, 4)

        @torch._dynamo.nonstrict_trace
        def gn(i_count: int, i_values: Tensor) -> tuple[Tensor, OutputData]:
            output_tensor = a + b * i_count * i_values

            result1 = output_tensor + i_count
            result2 = output_tensor * (i_count + 1)
            out = OutputData(result1, result2)

            return output_tensor, out

        def fn(i: InputData) -> Tensor:
            x = torch.sin(i.values)
            y, z = gn(i.count, i.values)
            return x + y + z.result1 + z.result2

        count = 5
        values = torch.randn(4, 4)
        i = InputData(count, values)
        ref = fn(i)
        opt_fn = torch.compile(lambda i: fn(i), backend="aot_eager", fullgraph=True)
        res = opt_fn(i)
        self.assertEqual(ref, res)

        _, gms, _, _ = extract_graph(lambda i: fn(i), i)
        self.assertExpectedInline(
            print_graph(gms[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_i_values: "f32[4, 4]"):
        l_i_values = L_i_values

        # code: x = torch.sin(i.values)
        x: "f32[4, 4]" = torch.sin(l_i_values)

        # code: y, z = gn(i.count, i.values)
        gn_spec : torch.utils._pytree.TreeSpec = self.gn_spec
        gn_input_spec : torch.utils._pytree.TreeSpec = self.gn_input_spec
        flat_apply_capture = torch__dynamo_variables_torch_flat_apply_capture(gn_spec, gn_input_spec, 5, l_i_values);  gn_spec = gn_input_spec = l_i_values = None
        y: "f32[4, 4]" = flat_apply_capture[0]
        value: "f32[4, 4]" = flat_apply_capture[1]
        value_1: "f32[4, 4]" = flat_apply_capture[2];  flat_apply_capture = None

        # code: return x + y + z.result1 + z.result2
        add: "f32[4, 4]" = x + y;  x = y = None
        add_1: "f32[4, 4]" = add + value;  add = value = None
        add_2: "f32[4, 4]" = add_1 + value_1;  add_1 = value_1 = None
        return (add_2,)
""",  # NOQA: B950
        )

    def test_invalid_output(self):
        a = 4
        b = torch.randn(4, 4)

        @torch._dynamo.nonstrict_trace
        def gn(i_count: int, i_values: Tensor) -> tuple[Tensor, OutputInvalid]:
            output_tensor = a + b * i_count * i_values

            result1 = output_tensor + i_count
            result2 = output_tensor * (i_count + 1)
            out = OutputInvalid(result1, result2)

            return output_tensor, out

        def fn(i: InputData) -> Tensor:
            x = torch.sin(i.values)
            y, z = gn(i.count, i.values)
            return x + y + z.result1 + z.result2

        count = 5
        values = torch.randn(4, 4)
        i = InputData(count, values)
        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Unsupported output type for nonstrict_trace-ed function",
        ):
            opt_fn(i)


def remove_file_comment(gm_str: str) -> str:
    return remove_trailing_space(re.sub(r"# File.*, code:", "# code:", gm_str))


def print_graph(graph: torch.fx.GraphModule) -> str:
    return remove_file_comment(graph.print_readable(print_output=False))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
