# Owner(s): ["module: dynamo"]
# flake8: noqa: B950
import torch
import torch.fx
from torch._dynamo.graph_deduplication import _flatten_args_kwargs
from torch._dynamo.graph_utils import _detect_cycles
from torch._dynamo.test_case import TestCase
from torch._dynamo.testing import AotEagerAndRecordGraphs, normalize_gm


def extract_graph(fn, *args, **kwargs):
    backend = AotEagerAndRecordGraphs()
    result = torch.compile(backend=backend)(fn)(*args, **kwargs)
    return result, backend.graphs, backend.fw_graphs


def graph_str(gm):
    return normalize_gm(gm.print_readable(print_output=False))


class GraphDededuplicationTests(TestCase):
    def run_and_return_graphs(self, fn, *args, **kwargs):
        with torch._dynamo.config.patch("use_graph_deduplication", True):
            return extract_graph(fn, *args, **kwargs)

    def test_single_subgraph(self):
        def inner_fn(x, y):
            x0 = x + 1
            y0 = y + 2
            z = x0.sum() + y0.sum()
            return z

        def fn(x, y):
            _o0 = inner_fn(x, y)
            o1 = torch.sin(y)
            o2 = inner_fn(x, o1)
            o3 = inner_fn(x, y)
            o4 = o3 * o3
            return o2 * o4

        x = torch.rand(10, 10, requires_grad=True)
        y = torch.rand(10, 20, requires_grad=True)
        x_clone = x.clone().requires_grad_(True)
        y_clone = y.clone().requires_grad_(True)

        ref_result = fn(x, y)
        result, graphs, fw_graphs = self.run_and_return_graphs(fn, x_clone, y_clone)

        torch.allclose(ref_result, result)
        ref_result.sum().backward()
        result.sum().backward()

        self.assertEqual(len(graphs), 1)
        self.assertEqual(len(fw_graphs), 1)
        self.assertExpectedInline(
            graph_str(graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[10, 10]", L_y_: "f32[10, 20]"):
        subgraph_0 = self.subgraph_0
        l_x_ = L_x_
        l_y_ = L_y_

        o1: "f32[10, 20]" = torch.sin(l_y_)

        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_, l_y_);  invoke_subgraph = None
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_, o1);  o1 = None

        getitem_1: "f32[]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None

        invoke_subgraph_2 = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_, l_y_);  subgraph_0 = l_x_ = l_y_ = None

        getitem_2: "f32[]" = invoke_subgraph_2[0];  invoke_subgraph_2 = None

        o4: "f32[]" = getitem_2 * getitem_2;  getitem_2 = None

        mul_1: "f32[]" = getitem_1 * o4;  getitem_1 = o4 = None
        return (mul_1,)

    class subgraph_0(torch.nn.Module):
        def forward(self, subgraph_input_l_x_, subgraph_input_l_y_):
            x0: "f32[10, 10]" = subgraph_input_l_x_ + 1;  subgraph_input_l_x_ = None

            y0: "f32[10, 20]" = subgraph_input_l_y_ + 2;  subgraph_input_l_y_ = None

            sum_1: "f32[]" = x0.sum();  x0 = None
            sum_2: "f32[]" = y0.sum();  y0 = None
            z: "f32[]" = sum_1 + sum_2;  sum_1 = sum_2 = None
            return (z,)
""",
        )

        self.assertExpectedInline(
            graph_str(fw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[10, 10]", primals_2: "f32[10, 20]"):
        sin: "f32[10, 20]" = torch.ops.aten.sin.default(primals_2)

        partitioned_fw_subgraph_0_0 = self.partitioned_fw_subgraph_0_0
        invoke_subgraph_5 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_0, 'partitioned_fw_subgraph_0_0', primals_1, sin);  partitioned_fw_subgraph_0_0 = sin = None
        getitem_1: "f32[]" = invoke_subgraph_5[0];  invoke_subgraph_5 = None
        partitioned_fw_subgraph_0_1 = self.partitioned_fw_subgraph_0_0
        invoke_subgraph_7 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_1, 'partitioned_fw_subgraph_0_0', primals_1, primals_2);  partitioned_fw_subgraph_0_1 = primals_1 = None
        getitem_2: "f32[]" = invoke_subgraph_7[0];  invoke_subgraph_7 = None

        mul: "f32[]" = torch.ops.aten.mul.Tensor(getitem_2, getitem_2)

        mul_1: "f32[]" = torch.ops.aten.mul.Tensor(getitem_1, mul);  mul = None
        return (mul_1, primals_2, getitem_1, getitem_2)

    class partitioned_fw_subgraph_0_0(torch.nn.Module):
        def forward(self, primals_0: "f32[10, 10]", primals_1: "f32[10, 20]"):
            add: "f32[10, 10]" = torch.ops.aten.add.Tensor(primals_0, 1);  primals_0 = None

            add_1: "f32[10, 20]" = torch.ops.aten.add.Tensor(primals_1, 2);  primals_1 = None

            sum_1: "f32[]" = torch.ops.aten.sum.default(add);  add = None
            sum_2: "f32[]" = torch.ops.aten.sum.default(add_1);  add_1 = None
            add_2: "f32[]" = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
            return (add_2,)
""",
        )

    def test_single_subgraph2(self):
        def fn(x):
            x0 = x + 2
            o = inner_fn(x0)
            o = torch.cos(o)
            o = inner_fn(o)
            return torch.sin(o)

        def inner_fn(x):
            o = x * 7
            o += 1
            o += 2
            return o

        x = torch.rand(10, 10, requires_grad=True)
        x_clone = x.clone().requires_grad_(True)

        ref_result = fn(x)
        result, graphs, fw_graphs = self.run_and_return_graphs(fn, x_clone)

        torch.allclose(ref_result, result)
        ref_result.sum().backward()
        result.sum().backward()
        self.assertEqual(len(graphs), 1)
        self.assertEqual(len(fw_graphs), 1)
        self.assertExpectedInline(
            graph_str(graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[10, 10]"):
        subgraph_0 = self.subgraph_0
        l_x_ = L_x_

        x0: "f32[10, 10]" = l_x_ + 2;  l_x_ = None

        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', x0);  x0 = None

        getitem: "f32[10, 10]" = invoke_subgraph[0];  invoke_subgraph = None

        o_3: "f32[10, 10]" = torch.cos(getitem);  getitem = None

        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', o_3);  subgraph_0 = o_3 = None

        getitem_1: "f32[10, 10]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None

        sin: "f32[10, 10]" = torch.sin(getitem_1);  getitem_1 = None
        return (sin,)

    class subgraph_0(torch.nn.Module):
        def forward(self, subgraph_input_x0):
            o: "f32[10, 10]" = subgraph_input_x0 * 7;  subgraph_input_x0 = None

            o += 1;  o_1: "f32[10, 10]" = o;  o = None

            o_1 += 2;  o_2: "f32[10, 10]" = o_1;  o_1 = None
            return (o_2,)
""",
        )
        self.assertExpectedInline(
            graph_str(fw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[10, 10]"):
        add: "f32[10, 10]" = torch.ops.aten.add.Tensor(primals_1, 2);  primals_1 = None

        partitioned_fw_subgraph_0_0 = self.partitioned_fw_subgraph_0_0
        invoke_subgraph_4 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_0, 'partitioned_fw_subgraph_0_0', add);  partitioned_fw_subgraph_0_0 = add = None
        getitem: "f32[10, 10]" = invoke_subgraph_4[0];  invoke_subgraph_4 = None

        cos: "f32[10, 10]" = torch.ops.aten.cos.default(getitem)

        partitioned_fw_subgraph_0_1 = self.partitioned_fw_subgraph_0_0
        invoke_subgraph_6 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_1, 'partitioned_fw_subgraph_0_0', cos);  partitioned_fw_subgraph_0_1 = cos = None
        getitem_1: "f32[10, 10]" = invoke_subgraph_6[0];  invoke_subgraph_6 = None

        sin: "f32[10, 10]" = torch.ops.aten.sin.default(getitem_1)
        cos_1: "f32[10, 10]" = torch.ops.aten.cos.default(getitem_1);  getitem_1 = None

        sin_1: "f32[10, 10]" = torch.ops.aten.sin.default(getitem);  getitem = None
        neg: "f32[10, 10]" = torch.ops.aten.neg.default(sin_1);  sin_1 = None
        return (sin, cos_1, neg)

    class partitioned_fw_subgraph_0_0(torch.nn.Module):
        def forward(self, primals_0: "f32[10, 10]"):
            mul: "f32[10, 10]" = torch.ops.aten.mul.Tensor(primals_0, 7);  primals_0 = None

            add: "f32[10, 10]" = torch.ops.aten.add.Tensor(mul, 1);  mul = None

            add_1: "f32[10, 10]" = torch.ops.aten.add.Tensor(add, 2);  add = None
            return (add_1,)
""",
        )

    def test_multiple_subgraphs(self):
        def inner_fn(x, y):
            x1 = x + 1
            y1 = y + 2
            z = x1.sum() + y1.sum()
            return z

        def inner_fn2(a, b):
            a0 = a + 2
            b0 = b + 3
            c = a0 * b0.cos().sum()
            return c

        def fn(x, y):
            x0 = torch.cos(x)
            y0 = torch.sin(y)
            o1 = inner_fn2(x0, y0)
            o0 = inner_fn(x, y)
            o1 = torch.sin(o0)
            o2 = inner_fn(x, y0)
            o3 = inner_fn2(x0, y0)
            o4 = inner_fn(x, y)
            return o1 * o2 * o3 + o4

        x = torch.rand(10, 10, requires_grad=True)
        y = torch.rand(10, 20, requires_grad=True)
        x_clone = x.clone().requires_grad_(True)
        y_clone = y.clone().requires_grad_(True)

        ref_result = fn(x, y)
        result, graphs, fw_graphs = self.run_and_return_graphs(fn, x_clone, y_clone)

        torch.allclose(ref_result, result)
        ref_result.sum().backward()
        result.sum().backward()
        self.assertEqual(len(graphs), 1)
        self.assertEqual(len(fw_graphs), 1)

        self.assertExpectedInline(
            graph_str(graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[10, 10]", L_y_: "f32[10, 20]"):
        subgraph_1 = self.subgraph_1
        subgraph_0 = self.subgraph_0
        l_x_ = L_x_
        l_y_ = L_y_

        x0: "f32[10, 10]" = torch.cos(l_x_)

        y0: "f32[10, 20]" = torch.sin(l_y_)

        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_, l_y_)

        getitem: "f32[]" = invoke_subgraph[0];  invoke_subgraph = None

        o1: "f32[]" = torch.sin(getitem);  getitem = None

        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_, y0)

        getitem_1: "f32[]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None

        mul_2: "f32[]" = o1 * getitem_1;  o1 = getitem_1 = None

        invoke_subgraph_2 = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_, l_y_);  subgraph_0 = l_x_ = l_y_ = None

        getitem_2: "f32[]" = invoke_subgraph_2[0];  invoke_subgraph_2 = None

        invoke_subgraph_3 = torch.ops.higher_order.invoke_subgraph(subgraph_1, 'subgraph_1', x0, y0);  invoke_subgraph_3 = None
        invoke_subgraph_4 = torch.ops.higher_order.invoke_subgraph(subgraph_1, 'subgraph_1', x0, y0);  subgraph_1 = x0 = y0 = None

        getitem_4: "f32[10, 10]" = invoke_subgraph_4[0];  invoke_subgraph_4 = None

        mul_3: "f32[10, 10]" = mul_2 * getitem_4;  mul_2 = getitem_4 = None
        add_13: "f32[10, 10]" = mul_3 + getitem_2;  mul_3 = getitem_2 = None
        return (add_13,)

    class subgraph_1(torch.nn.Module):
        def forward(self, subgraph_input_x0, subgraph_input_y0):
            a0: "f32[10, 10]" = subgraph_input_x0 + 2;  subgraph_input_x0 = None

            b0: "f32[10, 20]" = subgraph_input_y0 + 3;  subgraph_input_y0 = None

            cos_1: "f32[10, 20]" = b0.cos();  b0 = None
            sum_1: "f32[]" = cos_1.sum();  cos_1 = None
            c: "f32[10, 10]" = a0 * sum_1;  a0 = sum_1 = None
            return (c,)

    class subgraph_0(torch.nn.Module):
        def forward(self, subgraph_input_l_x_, subgraph_input_l_y_):
            x1: "f32[10, 10]" = subgraph_input_l_x_ + 1;  subgraph_input_l_x_ = None

            y1: "f32[10, 20]" = subgraph_input_l_y_ + 2;  subgraph_input_l_y_ = None

            sum_2: "f32[]" = x1.sum();  x1 = None
            sum_3: "f32[]" = y1.sum();  y1 = None
            z: "f32[]" = sum_2 + sum_3;  sum_2 = sum_3 = None
            return (z,)
""",
        )
        self.assertExpectedInline(
            graph_str(fw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[10, 10]", primals_2: "f32[10, 20]"):
        cos: "f32[10, 10]" = torch.ops.aten.cos.default(primals_1)

        sin: "f32[10, 20]" = torch.ops.aten.sin.default(primals_2)

        partitioned_fw_subgraph_0_0 = self.partitioned_fw_subgraph_0_0
        invoke_subgraph_9 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_0, 'partitioned_fw_subgraph_0_0', primals_1, primals_2);  partitioned_fw_subgraph_0_0 = None
        getitem: "f32[]" = invoke_subgraph_9[0];  invoke_subgraph_9 = None

        sin_1: "f32[]" = torch.ops.aten.sin.default(getitem)

        partitioned_fw_subgraph_0_1 = self.partitioned_fw_subgraph_0_0
        invoke_subgraph_11 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_1, 'partitioned_fw_subgraph_0_0', primals_1, sin);  partitioned_fw_subgraph_0_1 = None
        getitem_1: "f32[]" = invoke_subgraph_11[0];  invoke_subgraph_11 = None

        mul: "f32[]" = torch.ops.aten.mul.Tensor(sin_1, getitem_1);  sin_1 = None

        partitioned_fw_subgraph_0_2 = self.partitioned_fw_subgraph_0_0
        invoke_subgraph_13 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_2, 'partitioned_fw_subgraph_0_0', primals_1, primals_2);  partitioned_fw_subgraph_0_2 = None
        getitem_2: "f32[]" = invoke_subgraph_13[0];  invoke_subgraph_13 = None
        partitioned_fw_subgraph_1_0 = self.partitioned_fw_subgraph_1_0
        invoke_subgraph_15 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_1_0, 'partitioned_fw_subgraph_1_0', cos, sin);  partitioned_fw_subgraph_1_0 = cos = sin = None
        getitem_19: "f32[]" = invoke_subgraph_15[3]
        getitem_18: "f32[10, 20]" = invoke_subgraph_15[2]
        getitem_17: "f32[10, 10]" = invoke_subgraph_15[1]
        getitem_4: "f32[10, 10]" = invoke_subgraph_15[0];  invoke_subgraph_15 = None

        mul_1: "f32[10, 10]" = torch.ops.aten.mul.Tensor(mul, getitem_4);  mul = None
        add: "f32[10, 10]" = torch.ops.aten.add.Tensor(mul_1, getitem_2);  mul_1 = getitem_2 = None
        return (add, primals_1, primals_2, getitem, getitem_1, getitem_19, getitem_18, getitem_17, getitem_4)

    class partitioned_fw_subgraph_0_0(torch.nn.Module):
        def forward(self, primals_0: "f32[10, 10]", primals_1: "f32[10, 20]"):
            add: "f32[10, 10]" = torch.ops.aten.add.Tensor(primals_0, 1);  primals_0 = None

            add_1: "f32[10, 20]" = torch.ops.aten.add.Tensor(primals_1, 2);  primals_1 = None

            sum_1: "f32[]" = torch.ops.aten.sum.default(add);  add = None
            sum_2: "f32[]" = torch.ops.aten.sum.default(add_1);  add_1 = None
            add_2: "f32[]" = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
            return (add_2,)

    class partitioned_fw_subgraph_1_0(torch.nn.Module):
        def forward(self, primals_0: "f32[10, 10]", primals_1: "f32[10, 20]"):
            add: "f32[10, 10]" = torch.ops.aten.add.Tensor(primals_0, 2)

            add_1: "f32[10, 20]" = torch.ops.aten.add.Tensor(primals_1, 3)

            cos: "f32[10, 20]" = torch.ops.aten.cos.default(add_1);  add_1 = None
            sum_1: "f32[]" = torch.ops.aten.sum.default(cos);  cos = None
            mul: "f32[10, 10]" = torch.ops.aten.mul.Tensor(add, sum_1);  add = None
            return (mul, primals_0, primals_1, sum_1)
""",
        )

    def test_dependent_subgraphs(self):
        def inner_fn(x, y):
            x0 = x + 1
            y0 = y + 2
            z = x0.sum() + y0.sum()
            return z

        def fn(x, y):
            o0 = inner_fn(x, y)
            o1 = inner_fn(x, o0)
            return o1

        x = torch.rand(10, 10, requires_grad=True)
        y = torch.rand(10, 20, requires_grad=True)
        x_clone = x.clone().requires_grad_(True)
        y_clone = y.clone().requires_grad_(True)

        ref_result = fn(x, y)
        result, graphs, fw_graphs = self.run_and_return_graphs(fn, x_clone, y_clone)

        torch.allclose(ref_result, result)
        ref_result.sum().backward()
        result.sum().backward()
        self.assertEqual(len(graphs), 1)
        self.assertEqual(len(fw_graphs), 1)
        self.assertExpectedInline(
            graph_str(fw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[10, 10]", primals_2: "f32[10, 20]"):
        add: "f32[10, 20]" = torch.ops.aten.add.Tensor(primals_2, 2);  primals_2 = None

        sum_1: "f32[]" = torch.ops.aten.sum.default(add);  add = None

        partitioned_fw_subgraph_0_0 = self.partitioned_fw_subgraph_0_0
        invoke_subgraph_4 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_0, 'partitioned_fw_subgraph_0_0', primals_1, sum_1);  partitioned_fw_subgraph_0_0 = sum_1 = None
        getitem: "f32[]" = invoke_subgraph_4[0];  invoke_subgraph_4 = None

        add_1: "f32[]" = torch.ops.aten.add.Tensor(getitem, 2);  getitem = None

        sum_2: "f32[]" = torch.ops.aten.sum.default(add_1);  add_1 = None

        partitioned_fw_subgraph_0_1 = self.partitioned_fw_subgraph_0_0
        invoke_subgraph_6 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_1, 'partitioned_fw_subgraph_0_0', primals_1, sum_2);  partitioned_fw_subgraph_0_1 = primals_1 = sum_2 = None
        getitem_1: "f32[]" = invoke_subgraph_6[0];  invoke_subgraph_6 = None
        return (getitem_1,)

    class partitioned_fw_subgraph_0_0(torch.nn.Module):
        def forward(self, primals_0: "f32[10, 10]", primals_1: "f32[]"):
            add: "f32[10, 10]" = torch.ops.aten.add.Tensor(primals_0, 1);  primals_0 = None

            sum_1: "f32[]" = torch.ops.aten.sum.default(add);  add = None
            add_1: "f32[]" = torch.ops.aten.add.Tensor(sum_1, primals_1);  sum_1 = primals_1 = None
            return (add_1,)
""",
        )

    def test_input_mutation(self):
        def inner_fn(x, y):
            x0 = x + 1
            y0 = y + 2
            z = x0.sum() + y0.sum()
            return z

        def inner_fn2(x, y):
            x0 = x + 1
            y0 = y + 1
            x.add_(x0)
            y.add_(y0)
            return x.sum() + y.sum()

        def fn(x, y):
            x0 = torch.sin(x)
            _y0 = torch.cos(y)
            # o0 = inner_fn(x0, y0)
            # o1 = inner_fn(x0, o0)
            o2 = inner_fn2(x0, y)
            o3 = inner_fn2(x0.clone(), y.clone())
            return o2 + o3

        x = torch.rand(10, 10, requires_grad=False)
        y = torch.rand(10, 20, requires_grad=False)
        x_clone = x.clone()
        y_clone = y.clone()

        ref_result = fn(x, y)
        result, graphs, fw_graphs = self.run_and_return_graphs(fn, x_clone, y_clone)

        torch.allclose(ref_result, result)
        self.assertEqual(len(graphs), 1)
        self.assertEqual(len(fw_graphs), 1)
        self.assertExpectedInline(
            graph_str(fw_graphs[0]),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[10, 10]", arg1_1: "f32[10, 20]"):
        sin: "f32[10, 10]" = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None

        add: "f32[10, 10]" = torch.ops.aten.add.Tensor(sin, 1)

        add_1: "f32[10, 20]" = torch.ops.aten.add.Tensor(arg1_1, 1)

        add_2: "f32[10, 10]" = torch.ops.aten.add.Tensor(sin, add);  sin = add = None

        add_3: "f32[10, 20]" = torch.ops.aten.add.Tensor(arg1_1, add_1);  add_1 = None

        clone: "f32[10, 10]" = torch.ops.aten.clone.default(add_2)
        clone_1: "f32[10, 20]" = torch.ops.aten.clone.default(add_3)

        add_4: "f32[10, 10]" = torch.ops.aten.add.Tensor(clone, 1)

        add_5: "f32[10, 20]" = torch.ops.aten.add.Tensor(clone_1, 1)

        add_6: "f32[10, 10]" = torch.ops.aten.add.Tensor(clone, add_4);  clone = add_4 = None

        add_7: "f32[10, 20]" = torch.ops.aten.add.Tensor(clone_1, add_5);  clone_1 = add_5 = None

        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'subgraph_0', add_2, add_3);  repeated_subgraph0 = add_2 = None
        getitem: "f32[]" = invoke_subgraph[0];  invoke_subgraph = None
        repeated_subgraph0_1 = self.repeated_subgraph0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0_1, 'subgraph_0', add_6, add_7);  repeated_subgraph0_1 = add_6 = add_7 = None
        getitem_1: "f32[]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None

        add_8: "f32[]" = torch.ops.aten.add.Tensor(getitem, getitem_1);  getitem = getitem_1 = None

        copy_: "f32[10, 20]" = torch.ops.aten.copy_.default(arg1_1, add_3);  arg1_1 = add_3 = copy_ = None
        return (add_8,)

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[10, 10]", arg1_1: "f32[10, 20]"):
            sum_1: "f32[]" = torch.ops.aten.sum.default(arg0_1);  arg0_1 = None
            sum_2: "f32[]" = torch.ops.aten.sum.default(arg1_1);  arg1_1 = None
            add: "f32[]" = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
            return (add,)
""",
        )

    def test_input_aliasing(self):
        def inner_fn(x, y):
            x0 = x.view(x.size())
            return x0.view(x.size())

        def inner_fn2(x, y):
            x = x * 2
            y = y * 2
            return x.sum() + y.sum()

        def fn(x, y):
            o0 = inner_fn(x, y)
            o1 = inner_fn(x, y)
            o2 = inner_fn2(x, y)
            o3 = inner_fn2(x, y)
            return o0 + o1 + o2.sum() + o3.sum()

        x = torch.rand(10, 10, requires_grad=False)
        y = torch.rand(10, 20, requires_grad=False)
        x_clone = x.clone()
        y_clone = y.clone()

        ref_result = fn(x, y)
        result, graphs, fw_graphs = self.run_and_return_graphs(fn, x_clone, y_clone)

        torch.allclose(ref_result, result)
        self.assertEqual(len(graphs), 1)
        self.assertEqual(len(fw_graphs), 1)
        self.assertExpectedInline(
            graph_str(fw_graphs[0]),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[10, 10]", arg1_1: "f32[10, 20]"):
        view: "f32[10, 10]" = torch.ops.aten.view.default(arg0_1, [10, 10])

        view_1: "f32[10, 10]" = torch.ops.aten.view.default(view, [10, 10]);  view = None

        view_2: "f32[10, 10]" = torch.ops.aten.view.default(arg0_1, [10, 10])

        view_3: "f32[10, 10]" = torch.ops.aten.view.default(view_2, [10, 10]);  view_2 = None

        add: "f32[10, 10]" = torch.ops.aten.add.Tensor(view_1, view_3);  view_1 = view_3 = None

        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'subgraph_0', arg0_1, arg1_1);  repeated_subgraph0 = None
        getitem: "f32[]" = invoke_subgraph[0];  invoke_subgraph = None

        sum_1: "f32[]" = torch.ops.aten.sum.default(getitem);  getitem = None
        add_1: "f32[10, 10]" = torch.ops.aten.add.Tensor(add, sum_1);  add = sum_1 = None

        repeated_subgraph0_1 = self.repeated_subgraph0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0_1, 'subgraph_0', arg0_1, arg1_1);  repeated_subgraph0_1 = arg0_1 = arg1_1 = None
        getitem_1: "f32[]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None

        sum_2: "f32[]" = torch.ops.aten.sum.default(getitem_1);  getitem_1 = None
        add_2: "f32[10, 10]" = torch.ops.aten.add.Tensor(add_1, sum_2);  add_1 = sum_2 = None
        return (add_2,)

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[10, 10]", arg1_1: "f32[10, 20]"):
            mul: "f32[10, 10]" = torch.ops.aten.mul.Tensor(arg0_1, 2);  arg0_1 = None

            mul_1: "f32[10, 20]" = torch.ops.aten.mul.Tensor(arg1_1, 2);  arg1_1 = None

            sum_1: "f32[]" = torch.ops.aten.sum.default(mul);  mul = None
            sum_2: "f32[]" = torch.ops.aten.sum.default(mul_1);  mul_1 = None
            add: "f32[]" = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
            return (add,)
""",
        )

    def test_flatten_with_slices(self):
        tree = [{"x": 3}, ["x", slice(1, 2, 3), 1], [4, 5, 6, [slice(3, 4, 5)]]]
        out = _flatten_args_kwargs(tree)
        self.assertExpectedInline(
            str(out), """[3, 'x', 1, 2, 3, 1, 4, 5, 6, 3, 4, 5]"""
        )

    def test_cycle_detection_no_cycle(self):
        def fn(x, y):
            x0 = x + 1
            y0 = y + 2
            z = x0.sum() + y0.sum()
            return z

        x = torch.rand(10, 10, requires_grad=False)
        y = torch.rand(10, 20, requires_grad=False)

        _, _, fw_graphs = self.run_and_return_graphs(fn, x, y)
        mod = fw_graphs[0]
        self.assertExpectedInline(_detect_cycles(mod.graph), """no cycle detected""")

    def test_cycle_detection_simple(self):
        def fn(x, y):
            x0 = x + 1
            y0 = y + 2
            z = x0.sum() + y0.sum()
            return z

        x = torch.rand(10, 10, requires_grad=False)
        y = torch.rand(10, 20, requires_grad=False)

        _, _, fw_graphs = self.run_and_return_graphs(fn, x, y)
        mod = fw_graphs[0]
        add_node = next(n for n in mod.graph.nodes if n.name == "add")
        add_2 = next(n for n in mod.graph.nodes if n.name == "add_2")
        args = add_node.args
        add_node.args = (args[0], add_2)
        self.assertExpectedInline(
            _detect_cycles(mod.graph),
            """cycle detected in path: deque([arg0_1, add, sum_1, add_2, add])""",
        )

    def test_cycle_detection_complex(self):
        def inner_fn(x, y):
            x0 = x.view(x.size())
            return x0.view(x.size())

        def inner_fn2(x, y):
            x = x * 2
            y = y * 2
            return x.sum() + y.sum()

        def fn(x, y):
            o0 = inner_fn(x, y)
            o1 = inner_fn(x, y)
            o2 = inner_fn2(x, y)
            o3 = inner_fn2(x, y)
            return o0 + o1 + o2.sum() + o3.sum()

        x = torch.rand(10, 10, requires_grad=False)
        y = torch.rand(10, 20, requires_grad=False)
        x_clone = x.clone()
        y_clone = y.clone()

        _, _, fw_graphs = self.run_and_return_graphs(fn, x_clone, y_clone)
        mod = fw_graphs[0]
        invoke_subgraph_node = next(
            n for n in mod.graph.nodes if n.name == "invoke_subgraph"
        )
        add_2 = next(n for n in mod.graph.nodes if n.name == "add_2")
        args = invoke_subgraph_node.args
        invoke_subgraph_node.args = (add_2, args[1])
        self.assertExpectedInline(
            _detect_cycles(mod.graph),
            """cycle detected in path: deque([arg0_1, invoke_subgraph_1, getitem_1, sum_2, add_2, invoke_subgraph, getitem, sum_1, add_1, add_2])""",
        )

    def test_autocast_ordering(self):
        from torch._dynamo.graph_deduplication import (
            _populate_additional_deps,
            _stable_topological_sort,
        )

        def inner_fn(x, y):
            x0 = x.view(x.size())
            return x0.view(x.size())

        def inner_fn2(x, y):
            x = x * 2
            y = y * 2
            return x.sum() + y.sum()

        def fn(x, y):
            o0 = inner_fn(x, y)
            o1 = inner_fn(x, y)
            o2 = inner_fn2(x, y)
            o3 = inner_fn2(x, y)
            return o0 + o1 + o2.sum() + o3.sum()

        x = torch.rand(10, 10, requires_grad=False)
        y = torch.rand(10, 20, requires_grad=False)
        x_clone = x.clone()
        y_clone = y.clone()

        _, _, fw_graphs = self.run_and_return_graphs(fn, x_clone, y_clone)
        mod = fw_graphs[0]

        def get_node(name):
            return next(n for n in mod.graph.nodes if n.name == name)

        sum_1 = get_node("sum_1")
        enter_autocast = mod.graph.call_function(torch.amp._enter_autocast)
        sum_1.append(enter_autocast)
        sum_2 = get_node("sum_2")
        exit_autocast = mod.graph.call_function(torch.amp._exit_autocast)
        sum_2.append(exit_autocast)
        additional_deps = _populate_additional_deps(mod.graph)
        invoke_subgraph = get_node("invoke_subgraph")
        invoke_subgraph.append(enter_autocast)
        getitem_1 = get_node("getitem_1")
        getitem_1.append(exit_autocast)
        self.assertExpectedInline(
            graph_str(mod),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[10, 10]", arg1_1: "f32[10, 20]"):
        view: "f32[10, 10]" = torch.ops.aten.view.default(arg0_1, [10, 10])

        view_1: "f32[10, 10]" = torch.ops.aten.view.default(view, [10, 10]);  view = None

        view_2: "f32[10, 10]" = torch.ops.aten.view.default(arg0_1, [10, 10])

        view_3: "f32[10, 10]" = torch.ops.aten.view.default(view_2, [10, 10]);  view_2 = None

        add: "f32[10, 10]" = torch.ops.aten.add.Tensor(view_1, view_3);  view_1 = view_3 = None

        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'subgraph_0', arg0_1, arg1_1);  repeated_subgraph0 = None
        _enter_autocast = torch.amp.autocast_mode._enter_autocast();  _enter_autocast = None
        getitem: "f32[]" = invoke_subgraph[0];  invoke_subgraph = None

        sum_1: "f32[]" = torch.ops.aten.sum.default(getitem);  getitem = None
        add_1: "f32[10, 10]" = torch.ops.aten.add.Tensor(add, sum_1);  add = sum_1 = None

        repeated_subgraph0_1 = self.repeated_subgraph0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0_1, 'subgraph_0', arg0_1, arg1_1);  repeated_subgraph0_1 = arg0_1 = arg1_1 = None
        getitem_1: "f32[]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        _exit_autocast = torch.amp.autocast_mode._exit_autocast();  _exit_autocast = None

        sum_2: "f32[]" = torch.ops.aten.sum.default(getitem_1);  getitem_1 = None
        add_2: "f32[10, 10]" = torch.ops.aten.add.Tensor(add_1, sum_2);  add_1 = sum_2 = None
        return (add_2,)

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[10, 10]", arg1_1: "f32[10, 20]"):
            mul: "f32[10, 10]" = torch.ops.aten.mul.Tensor(arg0_1, 2);  arg0_1 = None

            mul_1: "f32[10, 20]" = torch.ops.aten.mul.Tensor(arg1_1, 2);  arg1_1 = None

            sum_1: "f32[]" = torch.ops.aten.sum.default(mul);  mul = None
            sum_2: "f32[]" = torch.ops.aten.sum.default(mul_1);  mul_1 = None
            add: "f32[]" = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
            return (add,)
""",
        )
        _stable_topological_sort(mod.graph, additional_deps)
        self.assertExpectedInline(
            graph_str(mod),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[10, 10]", arg1_1: "f32[10, 20]"):
        view: "f32[10, 10]" = torch.ops.aten.view.default(arg0_1, [10, 10])

        view_1: "f32[10, 10]" = torch.ops.aten.view.default(view, [10, 10]);  view = None

        view_2: "f32[10, 10]" = torch.ops.aten.view.default(arg0_1, [10, 10])

        view_3: "f32[10, 10]" = torch.ops.aten.view.default(view_2, [10, 10]);  view_2 = None

        add: "f32[10, 10]" = torch.ops.aten.add.Tensor(view_1, view_3);  view_1 = view_3 = None

        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'subgraph_0', arg0_1, arg1_1);  repeated_subgraph0 = None
        getitem: "f32[]" = invoke_subgraph[0];  invoke_subgraph = None

        sum_1: "f32[]" = torch.ops.aten.sum.default(getitem);  getitem = None

        _enter_autocast = torch.amp.autocast_mode._enter_autocast();  _enter_autocast = None

        add_1: "f32[10, 10]" = torch.ops.aten.add.Tensor(add, sum_1);  add = sum_1 = None

        repeated_subgraph0_1 = self.repeated_subgraph0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0_1, 'subgraph_0', arg0_1, arg1_1);  repeated_subgraph0_1 = arg0_1 = arg1_1 = None
        getitem_1: "f32[]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None

        sum_2: "f32[]" = torch.ops.aten.sum.default(getitem_1);  getitem_1 = None

        _exit_autocast = torch.amp.autocast_mode._exit_autocast();  _exit_autocast = None

        add_2: "f32[10, 10]" = torch.ops.aten.add.Tensor(add_1, sum_2);  add_1 = sum_2 = None
        return (add_2,)

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[10, 10]", arg1_1: "f32[10, 20]"):
            mul: "f32[10, 10]" = torch.ops.aten.mul.Tensor(arg0_1, 2);  arg0_1 = None

            mul_1: "f32[10, 20]" = torch.ops.aten.mul.Tensor(arg1_1, 2);  arg1_1 = None

            sum_1: "f32[]" = torch.ops.aten.sum.default(mul);  mul = None
            sum_2: "f32[]" = torch.ops.aten.sum.default(mul_1);  mul_1 = None
            add: "f32[]" = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
            return (add,)
""",
        )

    def test_output_nodes_last(self):
        from torch._dynamo.graph_deduplication import _stable_topological_sort

        def inner_fn(x, y):
            x0 = x.view(x.size())
            return x0.view(x.size())

        def inner_fn2(x, y):
            x = x * 2
            y = y * 2
            return x.sum() + y.sum()

        def fn(x, y):
            o0 = inner_fn(x, y)
            o1 = inner_fn(x, y)
            o2 = inner_fn2(x, y)
            o3 = inner_fn2(x, y)
            return o0 + o1 + o2.sum() + o3.sum()

        x = torch.rand(10, 10, requires_grad=False)
        y = torch.rand(10, 20, requires_grad=False)
        x_clone = x.clone()
        y_clone = y.clone()

        _, _, fw_graphs = self.run_and_return_graphs(fn, x_clone, y_clone)
        mod = fw_graphs[0]
        output = next(n for n in mod.graph.nodes if n.op == "output")
        add_2 = next(n for n in mod.graph.nodes if n.name == "sum_2")
        add_2.append(output)

        self.assertExpectedInline(
            graph_str(mod),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[10, 10]", arg1_1: "f32[10, 20]"):
        view: "f32[10, 10]" = torch.ops.aten.view.default(arg0_1, [10, 10])

        view_1: "f32[10, 10]" = torch.ops.aten.view.default(view, [10, 10]);  view = None

        view_2: "f32[10, 10]" = torch.ops.aten.view.default(arg0_1, [10, 10])

        view_3: "f32[10, 10]" = torch.ops.aten.view.default(view_2, [10, 10]);  view_2 = None

        add: "f32[10, 10]" = torch.ops.aten.add.Tensor(view_1, view_3);  view_1 = view_3 = None

        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'subgraph_0', arg0_1, arg1_1);  repeated_subgraph0 = None
        getitem: "f32[]" = invoke_subgraph[0];  invoke_subgraph = None

        sum_1: "f32[]" = torch.ops.aten.sum.default(getitem);  getitem = None
        add_1: "f32[10, 10]" = torch.ops.aten.add.Tensor(add, sum_1);  add = sum_1 = None

        repeated_subgraph0_1 = self.repeated_subgraph0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0_1, 'subgraph_0', arg0_1, arg1_1);  repeated_subgraph0_1 = arg0_1 = arg1_1 = None
        getitem_1: "f32[]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None

        sum_2: "f32[]" = torch.ops.aten.sum.default(getitem_1);  getitem_1 = None
        return (add_2,)
        add_2: "f32[10, 10]" = torch.ops.aten.add.Tensor(add_1, sum_2);  add_1 = sum_2 = None

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[10, 10]", arg1_1: "f32[10, 20]"):
            mul: "f32[10, 10]" = torch.ops.aten.mul.Tensor(arg0_1, 2);  arg0_1 = None

            mul_1: "f32[10, 20]" = torch.ops.aten.mul.Tensor(arg1_1, 2);  arg1_1 = None

            sum_1: "f32[]" = torch.ops.aten.sum.default(mul);  mul = None
            sum_2: "f32[]" = torch.ops.aten.sum.default(mul_1);  mul_1 = None
            add: "f32[]" = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
            return (add,)
""",
        )
        _stable_topological_sort(mod.graph, {})
        self.assertExpectedInline(
            graph_str(mod),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[10, 10]", arg1_1: "f32[10, 20]"):
        view: "f32[10, 10]" = torch.ops.aten.view.default(arg0_1, [10, 10])

        view_1: "f32[10, 10]" = torch.ops.aten.view.default(view, [10, 10]);  view = None

        view_2: "f32[10, 10]" = torch.ops.aten.view.default(arg0_1, [10, 10])

        view_3: "f32[10, 10]" = torch.ops.aten.view.default(view_2, [10, 10]);  view_2 = None

        add: "f32[10, 10]" = torch.ops.aten.add.Tensor(view_1, view_3);  view_1 = view_3 = None

        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'subgraph_0', arg0_1, arg1_1);  repeated_subgraph0 = None
        getitem: "f32[]" = invoke_subgraph[0];  invoke_subgraph = None

        sum_1: "f32[]" = torch.ops.aten.sum.default(getitem);  getitem = None
        add_1: "f32[10, 10]" = torch.ops.aten.add.Tensor(add, sum_1);  add = sum_1 = None

        repeated_subgraph0_1 = self.repeated_subgraph0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0_1, 'subgraph_0', arg0_1, arg1_1);  repeated_subgraph0_1 = arg0_1 = arg1_1 = None
        getitem_1: "f32[]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None

        sum_2: "f32[]" = torch.ops.aten.sum.default(getitem_1);  getitem_1 = None
        add_2: "f32[10, 10]" = torch.ops.aten.add.Tensor(add_1, sum_2);  add_1 = sum_2 = None
        return (add_2,)

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[10, 10]", arg1_1: "f32[10, 20]"):
            mul: "f32[10, 10]" = torch.ops.aten.mul.Tensor(arg0_1, 2);  arg0_1 = None

            mul_1: "f32[10, 20]" = torch.ops.aten.mul.Tensor(arg1_1, 2);  arg1_1 = None

            sum_1: "f32[]" = torch.ops.aten.sum.default(mul);  mul = None
            sum_2: "f32[]" = torch.ops.aten.sum.default(mul_1);  mul_1 = None
            add: "f32[]" = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
            return (add,)
""",
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
