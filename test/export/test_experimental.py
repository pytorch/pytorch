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
            def __init__(self):
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
            def __init__(self):
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
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x0 = x[0][0]
                return x0.sum()

        class M(torch.nn.Module):
            def __init__(self):
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
            def __init__(self):
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
        print(joint_ep)

        """
        ExportedProgram:
            class GraphModule(torch.nn.Module):
                def forward(self, arg0_1: "f32[3, 3]", arg1_1: "f32[3]", arg2_1: "f32[3]", arg3_1: "f32[3]"):
                    # No stacktrace found for following nodes
                    view: "f32[1, 3]" = torch.ops.aten.view.default(arg3_1, [1, 3]);  arg3_1 = None
                    t: "f32[3, 3]" = torch.ops.aten.t.default(arg0_1);  arg0_1 = None
                    addmm: "f32[1, 3]" = torch.ops.aten.addmm.default(arg1_1, view, t);  arg1_1 = t = None
                    view_1: "f32[3]" = torch.ops.aten.view.default(addmm, [3]);  addmm = None
                    _softmax: "f32[3]" = torch.ops.aten._softmax.default(view_1, 0, False);  view_1 = None
                    detach_1: "f32[3]" = torch.ops.aten.detach.default(_softmax)
                    clone: "f32[3]" = torch.ops.aten.clone.default(arg2_1);  arg2_1 = None
                    detach_5: "f32[3]" = torch.ops.aten.detach.default(clone);  clone = None
                    _log_softmax: "f32[3]" = torch.ops.aten._log_softmax.default(_softmax, 0, False);  _softmax = None
                    detach_12: "f32[3]" = torch.ops.aten.detach.default(_log_softmax)
                    mul: "f32[3]" = torch.ops.aten.mul.Tensor(_log_softmax, detach_5);  _log_softmax = None
                    sum_1: "f32[]" = torch.ops.aten.sum.default(mul);  mul = None
                    neg: "f32[]" = torch.ops.aten.neg.default(sum_1);  sum_1 = None
                    div: "f32[]" = torch.ops.aten.div.Scalar(neg, 1);  neg = None
                    ones_like: "f32[]" = torch.ops.aten.ones_like.default(div, pin_memory = False, memory_format = torch.preserve_format)
                    div_1: "f32[]" = torch.ops.aten.div.Scalar(ones_like, 1);  ones_like = None
                    neg_1: "f32[]" = torch.ops.aten.neg.default(div_1);  div_1 = None
                    expand: "f32[3]" = torch.ops.aten.expand.default(neg_1, [3]);  neg_1 = None
                    mul_1: "f32[3]" = torch.ops.aten.mul.Tensor(expand, detach_5);  expand = detach_5 = None
                    _log_softmax_backward_data: "f32[3]" = torch.ops.aten._log_softmax_backward_data.default(mul_1, detach_12, 0, torch.float32);  mul_1 = detach_12 = None
                    _softmax_backward_data: "f32[3]" = torch.ops.aten._softmax_backward_data.default(_log_softmax_backward_data, detach_1, 0, torch.float32);  _log_softmax_backward_data = detach_1 = None
                    view_2: "f32[1, 3]" = torch.ops.aten.view.default(_softmax_backward_data, [1, 3]);  _softmax_backward_data = None
                    t_1: "f32[3, 1]" = torch.ops.aten.t.default(view_2)
                    mm: "f32[3, 3]" = torch.ops.aten.mm.default(t_1, view);  t_1 = view = None
                    t_2: "f32[3, 3]" = torch.ops.aten.t.default(mm);  mm = None
                    sum_2: "f32[1, 3]" = torch.ops.aten.sum.dim_IntList(view_2, [0], True);  view_2 = None
                    view_3: "f32[3]" = torch.ops.aten.view.default(sum_2, [3]);  sum_2 = None
                    t_3: "f32[3, 3]" = torch.ops.aten.t.default(t_2);  t_2 = None
                    return (div, t_3, view_3)

        Graph signature: ExportGraphSignature(
            input_specs=[
                InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='arg0_1'), target='linear.weight', persistent=None),
                InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='arg1_1'), target='linear.bias', persistent=None),
                InputSpec(kind=<InputKind.CONSTANT_TENSOR: 4>, arg=TensorArgument(name='arg2_1'), target='lifted_tensor_0', persistent=None),
                InputSpec(kind=<InputKind.USER_INPUT: 1>, arg=TensorArgument(name='arg3_1'), target=None, persistent=None)
            ],
            output_specs=[
                OutputSpec(kind=<OutputKind.USER_OUTPUT: 1>, arg=TensorArgument(name='div'), target=None),
                OutputSpec(kind=<OutputKind.GRADIENT_TO_PARAMETER: 4>, arg=TensorArgument(name='t_3'), target='linear.weight'),
                OutputSpec(kind=<OutputKind.GRADIENT_TO_PARAMETER: 4>, arg=TensorArgument(name='view_3'), target='linear.bias')
            ]
        )
        Range constraints: {}
        """

    def test_joint_dynamic(self) -> None:
        from torch.export import Dim

        class Module(torch.nn.Module):
            def __init__(self):
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
