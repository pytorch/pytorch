# Owner(s): ["oncall: export"]
# flake8: noqa
import unittest
from typing import Dict, List, Tuple

import torch
import torch._dynamo
from torch._dynamo.test_case import run_tests, TestCase
from torch._export.wrappers import _mark_strict_experimental
from torch._functorch.aot_autograd import aot_export_module
from torch.export._trace import _convert_ts_to_export_experimental
from torch.export.experimental import _export_forward_backward
from torch.testing import FileCheck


@unittest.skipIf(not torch._dynamo.is_dynamo_supported(), "dynamo isn't supported")
class TestExperiment(TestCase):
    def test_with_buffer_as_submodule(self):
        @_mark_strict_experimental
        class B(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buffer1 = torch.nn.Buffer(torch.ones(3))

            def forward(self, x):
                y = x + 2
                y.add_(4)
                # this doesnt' work today with HOO
                # self.buffer1.add_(6)
                buffer_updated = self.buffer1 + 6
                return x.sum() + y.sum() + buffer_updated.sum()

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.submodule = B()

            def forward(self, x):
                x_v2 = x.sin()
                return (self.submodule(x_v2), x + 3)

        inp = torch.randn(3)
        ep = torch.export.export(M(), (inp,), strict=False)
        self.assertExpectedInline(
            str(ep.graph_module.code.strip()),
            """\
def forward(self, b_submodule_buffer1, x):
    sin = torch.ops.aten.sin.default(x)
    strict_graph_0 = self.strict_graph_0
    strict_mode = torch.ops.higher_order.strict_mode(strict_graph_0, (sin, b_submodule_buffer1));  strict_graph_0 = sin = b_submodule_buffer1 = None
    getitem_2 = strict_mode[0];  strict_mode = None
    add = torch.ops.aten.add.Tensor(x, 3);  x = None
    return (getitem_2, add)""",
        )

        self.assertExpectedInline(
            str(ep.graph_module.strict_graph_0.code.strip()),
            """\
def forward(self, arg0_1, arg1_1):
    add = torch.ops.aten.add.Tensor(arg0_1, 2)
    add_1 = torch.ops.aten.add.Tensor(add, 4);  add = None
    add_2 = torch.ops.aten.add.Tensor(arg1_1, 6);  arg1_1 = None
    sum_1 = torch.ops.aten.sum.default(arg0_1);  arg0_1 = None
    sum_2 = torch.ops.aten.sum.default(add_1);  add_1 = None
    add_3 = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
    sum_3 = torch.ops.aten.sum.default(add_2);  add_2 = None
    add_4 = torch.ops.aten.add.Tensor(add_3, sum_3);  add_3 = sum_3 = None
    return (add_4,)""",
        )

        eager_mod = M()
        ep = torch.export.export(eager_mod, (inp,), strict=True)

        graph_res_1, graph_res_2 = ep.module()(inp)
        eager_res_1, eager_res_2 = eager_mod(inp)

        self.assertTrue(torch.allclose(graph_res_2, eager_res_2))
        self.assertTrue(torch.allclose(graph_res_1, eager_res_1))

        graph_res_1, graph_res_2 = ep.module()(inp)
        eager_res_1, eager_res_2 = eager_mod(inp)

        self.assertTrue(torch.allclose(graph_res_2, eager_res_2))
        self.assertTrue(torch.allclose(graph_res_1, eager_res_1))

    def test_mark_strict_with_container_type(self):
        @_mark_strict_experimental
        class B(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                x0 = x[0][0]
                return x0.sum()

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.submodule = B()

            def forward(self, x):
                return self.submodule(x)

        inp = ((torch.randn(3),),)
        with self.assertRaisesRegex(
            RuntimeError, "strict_mode HOO doesn't work unless"
        ):
            ep = torch.export.export(M(), inp, strict=False)

    def test_torchscript_module_export(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x.cos() + x.sin()

        model_to_trace = M()
        inps = (torch.randn(4, 4),)
        traced_module_by_torchscript = torch.jit.trace(M(), example_inputs=inps)

        exported_module = _convert_ts_to_export_experimental(
            traced_module_by_torchscript, inps
        )

        self.assertTrue(torch.allclose(exported_module(*inps), model_to_trace(*inps)))

    def test_torchscript_module_export_single_input(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x.cos() + x.sin()

        model_to_trace = M()
        inps = torch.randn(4, 4)
        traced_module_by_torchscript = torch.jit.trace(M(), example_inputs=inps)

        exported_module = _convert_ts_to_export_experimental(
            traced_module_by_torchscript, inps
        )

        self.assertTrue(torch.allclose(exported_module(inps), model_to_trace(inps)))

    def test_torchscript_module_export_various_inputs_with_annotated_input_names(self):
        def _check_equality_and_annotations(m_func, inps):
            # Original module.
            model_to_trace = m_func()

            # ExportedProgram from TorchScript module.
            traced_module_by_torchscript = torch.jit.trace(
                m_func(), example_inputs=inps
            )
            exported_module = _convert_ts_to_export_experimental(
                traced_module_by_torchscript, inps
            )

            # ExportedProgram from original module.
            original_exported_module = torch.export.export(m_func(), inps)

            # Check whether input annotations are the same as tracing the original module.
            orig_ph_name_list = [
                n.name
                for n in original_exported_module.graph.nodes
                if n.op == "placeholder"
            ]
            ph_name_list = [
                n.name for n in exported_module.graph.nodes if n.op == "placeholder"
            ]
            self.assertEqual(orig_ph_name_list, ph_name_list)

            # Check results equality.
            self.assertTrue(
                torch.allclose(exported_module(*inps), model_to_trace(*inps))
            )

        # Tuple
        class MTuple(torch.nn.Module):
            def forward(self, x: Tuple[torch.Tensor]):
                return x[0] + x[1]

        _check_equality_and_annotations(MTuple, ((torch.randn(4), torch.randn(4)),))

        # List
        class MList(torch.nn.Module):
            def forward(self, x: List[torch.Tensor]):
                return x[0] + x[1]

        _check_equality_and_annotations(MList, ([torch.randn(4), torch.randn(4)],))

        # Dict
        class MDict(torch.nn.Module):
            def forward(self, x: Dict[str, torch.Tensor]):
                return x["0"] + x["1"]

        _check_equality_and_annotations(
            MDict, ({"0": torch.randn(4), "1": torch.randn(4)},)
        )

    def test_joint_basic(self) -> None:
        class Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)
                self.loss = torch.nn.CrossEntropyLoss()

            def forward(self, x):
                return self.loss(
                    self.linear(x).softmax(dim=0), torch.tensor([1.0, 0.0, 0.0])
                )

        m = Module()
        example_inputs = (torch.randn(3),)
        m(*example_inputs)
        ep = torch.export._trace._export(m, example_inputs, pre_dispatch=True)
        joint_ep = _export_forward_backward(ep)
        self.assertExpectedInline(
            str(joint_ep.graph_module.code).strip(),
            """\
def forward(self, p_linear_weight, p_linear_bias, c_lifted_tensor_0, x):
    view = torch.ops.aten.view.default(x, [1, 3]);  x = None
    permute = torch.ops.aten.permute.default(p_linear_weight, [1, 0]);  p_linear_weight = None
    addmm = torch.ops.aten.addmm.default(p_linear_bias, view, permute);  p_linear_bias = permute = None
    view_1 = torch.ops.aten.view.default(addmm, [3]);  addmm = None
    _softmax = torch.ops.aten._softmax.default(view_1, 0, False);  view_1 = None
    alias = torch.ops.aten.alias.default(_softmax)
    alias_1 = torch.ops.aten.alias.default(alias);  alias = None
    clone = torch.ops.aten.clone.default(c_lifted_tensor_0);  c_lifted_tensor_0 = None
    alias_2 = torch.ops.aten.alias.default(clone);  clone = None
    alias_3 = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    alias_4 = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    _log_softmax = torch.ops.aten._log_softmax.default(_softmax, 0, False);  _softmax = None
    alias_5 = torch.ops.aten.alias.default(_log_softmax)
    alias_6 = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    mul = torch.ops.aten.mul.Tensor(_log_softmax, alias_4);  _log_softmax = None
    sum_1 = torch.ops.aten.sum.dim_IntList(mul, []);  mul = None
    neg = torch.ops.aten.neg.default(sum_1);  sum_1 = None
    div = torch.ops.aten.div.Scalar(neg, 1);  neg = None
    full_like = torch.ops.aten.full_like.default(div, 1, pin_memory = False, memory_format = torch.preserve_format)
    div_1 = torch.ops.aten.div.Scalar(full_like, 1);  full_like = None
    neg_1 = torch.ops.aten.neg.default(div_1);  div_1 = None
    expand = torch.ops.aten.expand.default(neg_1, [3]);  neg_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(expand, alias_4);  expand = alias_4 = None
    alias_7 = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    alias_8 = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    exp = torch.ops.aten.exp.default(alias_8);  alias_8 = None
    sum_2 = torch.ops.aten.sum.dim_IntList(mul_1, [0], True)
    mul_2 = torch.ops.aten.mul.Tensor(exp, sum_2);  exp = sum_2 = None
    sub = torch.ops.aten.sub.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    alias_9 = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    alias_10 = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    mul_3 = torch.ops.aten.mul.Tensor(sub, alias_10);  sub = None
    sum_3 = torch.ops.aten.sum.dim_IntList(mul_3, [0], True)
    mul_4 = torch.ops.aten.mul.Tensor(alias_10, sum_3);  alias_10 = sum_3 = None
    sub_1 = torch.ops.aten.sub.Tensor(mul_3, mul_4);  mul_3 = mul_4 = None
    view_2 = torch.ops.aten.view.default(sub_1, [1, 3]);  sub_1 = None
    permute_1 = torch.ops.aten.permute.default(view_2, [1, 0])
    mm = torch.ops.aten.mm.default(permute_1, view);  permute_1 = view = None
    permute_2 = torch.ops.aten.permute.default(mm, [1, 0]);  mm = None
    sum_4 = torch.ops.aten.sum.dim_IntList(view_2, [0], True);  view_2 = None
    view_3 = torch.ops.aten.view.default(sum_4, [3]);  sum_4 = None
    permute_3 = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    return (div, permute_3, view_3)""",
        )
        ep = joint_ep.run_decompositions()
        self.assertExpectedInline(
            str(ep.graph_module.code).strip(),
            """\
def forward(self, p_linear_weight, p_linear_bias, c_lifted_tensor_0, x):
    view = torch.ops.aten.view.default(x, [1, 3]);  x = None
    permute = torch.ops.aten.permute.default(p_linear_weight, [1, 0]);  p_linear_weight = None
    addmm = torch.ops.aten.addmm.default(p_linear_bias, view, permute);  p_linear_bias = permute = None
    view_1 = torch.ops.aten.view.default(addmm, [3]);  addmm = None
    _softmax = torch.ops.aten._softmax.default(view_1, 0, False);  view_1 = None
    alias = torch.ops.aten.alias.default(_softmax)
    alias_1 = torch.ops.aten.alias.default(alias);  alias = None
    clone = torch.ops.aten.clone.default(c_lifted_tensor_0);  c_lifted_tensor_0 = None
    alias_2 = torch.ops.aten.alias.default(clone);  clone = None
    alias_3 = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    alias_4 = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    _log_softmax = torch.ops.aten._log_softmax.default(_softmax, 0, False);  _softmax = None
    alias_5 = torch.ops.aten.alias.default(_log_softmax)
    alias_6 = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    mul = torch.ops.aten.mul.Tensor(_log_softmax, alias_4);  _log_softmax = None
    sum_1 = torch.ops.aten.sum.dim_IntList(mul, []);  mul = None
    neg = torch.ops.aten.neg.default(sum_1);  sum_1 = None
    div = torch.ops.aten.div.Scalar(neg, 1);  neg = None
    full_like = torch.ops.aten.full_like.default(div, 1, pin_memory = False, memory_format = torch.preserve_format)
    div_1 = torch.ops.aten.div.Scalar(full_like, 1);  full_like = None
    neg_1 = torch.ops.aten.neg.default(div_1);  div_1 = None
    expand = torch.ops.aten.expand.default(neg_1, [3]);  neg_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(expand, alias_4);  expand = alias_4 = None
    alias_7 = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    alias_8 = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    exp = torch.ops.aten.exp.default(alias_8);  alias_8 = None
    sum_2 = torch.ops.aten.sum.dim_IntList(mul_1, [0], True)
    mul_2 = torch.ops.aten.mul.Tensor(exp, sum_2);  exp = sum_2 = None
    sub = torch.ops.aten.sub.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    alias_9 = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    alias_10 = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    mul_3 = torch.ops.aten.mul.Tensor(sub, alias_10);  sub = None
    sum_3 = torch.ops.aten.sum.dim_IntList(mul_3, [0], True)
    mul_4 = torch.ops.aten.mul.Tensor(alias_10, sum_3);  alias_10 = sum_3 = None
    sub_1 = torch.ops.aten.sub.Tensor(mul_3, mul_4);  mul_3 = mul_4 = None
    view_2 = torch.ops.aten.view.default(sub_1, [1, 3]);  sub_1 = None
    permute_1 = torch.ops.aten.permute.default(view_2, [1, 0])
    mm = torch.ops.aten.mm.default(permute_1, view);  permute_1 = view = None
    permute_2 = torch.ops.aten.permute.default(mm, [1, 0]);  mm = None
    sum_4 = torch.ops.aten.sum.dim_IntList(view_2, [0], True);  view_2 = None
    view_3 = torch.ops.aten.view.default(sum_4, [3]);  sum_4 = None
    permute_3 = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    return (div, permute_3, view_3)""",
        )

    def test_joint_dynamic(self) -> None:
        from torch.export import Dim

        class Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.y = torch.nn.Parameter(torch.randn(3))

            def forward(self, x):
                x = torch.ones(x.shape[0], 3)
                return (self.y + x).sum()

        m = Module()
        example_inputs = (torch.randn(3),)
        m(*example_inputs)
        ep = torch.export._trace._export(
            m, example_inputs, pre_dispatch=True, dynamic_shapes={"x": {0: Dim("x0")}}
        )
        joint_ep = _export_forward_backward(ep)


if __name__ == "__main__":
    run_tests()
