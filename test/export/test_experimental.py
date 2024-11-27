# Owner(s): ["oncall: export"]
# flake8: noqa
import unittest
from typing import Dict, List, Tuple

import torch
import torch._dynamo
from torch._dynamo.test_case import run_tests, TestCase
from torch._functorch.aot_autograd import aot_export_module
from torch.export import export, export_for_training
from torch.export._trace import _convert_ts_to_export_experimental
from torch.export.experimental import _export_forward_backward
from torch.testing import FileCheck


@unittest.skipIf(not torch._dynamo.is_dynamo_supported(), "dynamo isn't supported")
class TestExperiment(TestCase):
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
            original_exported_module = torch.export.export_for_training(m_func(), inps)

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
        ep = torch.export.export_for_training(m, example_inputs)
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
    _log_softmax = torch.ops.aten._log_softmax.default(_softmax, 0, False);  _softmax = None
    alias_2 = torch.ops.aten.alias.default(_log_softmax)
    alias_3 = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    mul = torch.ops.aten.mul.Tensor(_log_softmax, clone);  _log_softmax = None
    sum_1 = torch.ops.aten.sum.dim_IntList(mul, []);  mul = None
    neg = torch.ops.aten.neg.default(sum_1);  sum_1 = None
    div = torch.ops.aten.div.Scalar(neg, 1);  neg = None
    full_like = torch.ops.aten.full_like.default(div, 1, pin_memory = False, memory_format = torch.preserve_format)
    div_1 = torch.ops.aten.div.Scalar(full_like, 1);  full_like = None
    neg_1 = torch.ops.aten.neg.default(div_1);  div_1 = None
    expand = torch.ops.aten.expand.default(neg_1, [3]);  neg_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(expand, clone);  expand = clone = None
    alias_4 = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    alias_5 = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    exp = torch.ops.aten.exp.default(alias_5);  alias_5 = None
    sum_2 = torch.ops.aten.sum.dim_IntList(mul_1, [0], True)
    mul_2 = torch.ops.aten.mul.Tensor(exp, sum_2);  exp = sum_2 = None
    sub = torch.ops.aten.sub.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    alias_6 = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    alias_7 = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    mul_3 = torch.ops.aten.mul.Tensor(sub, alias_7);  sub = None
    sum_3 = torch.ops.aten.sum.dim_IntList(mul_3, [0], True)
    mul_4 = torch.ops.aten.mul.Tensor(alias_7, sum_3);  alias_7 = sum_3 = None
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
    _log_softmax = torch.ops.aten._log_softmax.default(_softmax, 0, False);  _softmax = None
    alias_2 = torch.ops.aten.alias.default(_log_softmax)
    alias_3 = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    mul = torch.ops.aten.mul.Tensor(_log_softmax, clone);  _log_softmax = None
    sum_1 = torch.ops.aten.sum.dim_IntList(mul, []);  mul = None
    neg = torch.ops.aten.neg.default(sum_1);  sum_1 = None
    div = torch.ops.aten.div.Scalar(neg, 1);  neg = None
    full_like = torch.ops.aten.full_like.default(div, 1, pin_memory = False, memory_format = torch.preserve_format)
    div_1 = torch.ops.aten.div.Scalar(full_like, 1);  full_like = None
    neg_1 = torch.ops.aten.neg.default(div_1);  div_1 = None
    expand = torch.ops.aten.expand.default(neg_1, [3]);  neg_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(expand, clone);  expand = clone = None
    alias_4 = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    alias_5 = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    exp = torch.ops.aten.exp.default(alias_5);  alias_5 = None
    sum_2 = torch.ops.aten.sum.dim_IntList(mul_1, [0], True)
    mul_2 = torch.ops.aten.mul.Tensor(exp, sum_2);  exp = sum_2 = None
    sub = torch.ops.aten.sub.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    alias_6 = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    alias_7 = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    mul_3 = torch.ops.aten.mul.Tensor(sub, alias_7);  sub = None
    sum_3 = torch.ops.aten.sum.dim_IntList(mul_3, [0], True)
    mul_4 = torch.ops.aten.mul.Tensor(alias_7, sum_3);  alias_7 = sum_3 = None
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
        ep = torch.export.export_for_training(
            m, example_inputs, dynamic_shapes={"x": {0: Dim("x0")}}
        )
        joint_ep = _export_forward_backward(ep)

    def test_joint_cifar10_backwards(self) -> None:
        import torch.nn as nn
        import torch.nn.functional as F

        # From Pytorch's CIFAR10 example:
        # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(16 * 5 * 5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)
                self.loss = nn.CrossEntropyLoss()

            def forward(self, x, labels):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = torch.flatten(x, 1)  # flatten all dimensions except batch
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return self.loss(x, labels)

        net = Net()
        x = torch.randn(4, 3, 32, 32)
        labels = torch.ones(4, dtype=torch.int64)
        inputs = (x, labels)

        ep = export_for_training(net, inputs)
        ep = _export_forward_backward(ep)

    def test_joint_loss_index(self):
        from torch.export.graph_signature import OutputKind

        class Foo(torch.nn.Module):
            def __init__(self, index):
                super().__init__()
                self.l = torch.nn.Linear(4, 4)
                self.index = index

            def forward(self, x):
                x = self.l(x)
                x = x.sum()
                if self.index == 0:
                    return x, -x.detach()
                else:
                    return x.detach(), x

        inputs = (torch.randn(4, 4),)
        for i in [0, 1]:
            ep = export_for_training(Foo(i), inputs)
            ep_joint = _export_forward_backward(ep, joint_loss_index=i)
            for j, spec in enumerate(ep_joint.graph_signature.output_specs):
                if i == j:
                    self.assertTrue(spec.kind == OutputKind.LOSS_OUTPUT)
                else:
                    self.assertTrue(spec.kind != OutputKind.LOSS_OUTPUT)


if __name__ == "__main__":
    run_tests()
