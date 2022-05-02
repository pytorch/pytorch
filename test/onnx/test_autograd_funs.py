# Owner(s): ["module: onnx"]

from test_pytorch_common import TestCase, run_tests

import torch
import torch.onnx
from torch.onnx import (utils,
                        OperatorExportTypes,
                        TrainingMode,
                        register_custom_op_symbolic,
                        unregister_custom_op_symbolic)
from torch.onnx.symbolic_helper import (_set_opset_version,
                                        _set_operator_export_type,
                                        _set_onnx_shape_inference,
                                        _unpack_list,
                                        parse_args)
import torch.utils.cpp_extension
from autograd_helper import CustomFunction as CustomFunction2
from test_pytorch_common import (skipIfUnsupportedMinOpsetVersion,
                                 skipIfUnsupportedMaxOpsetVersion)
from verify import verify

import torchvision

import onnx

import io
import copy
import unittest

skip = unittest.skip


class _BaseTestCase(TestCase):

    def setUp(self):
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

    def _model_to_graph(self, model, input,
                        do_constant_folding=True,
                        training=TrainingMode.EVAL,
                        operator_export_type=OperatorExportTypes.ONNX,
                        input_names=None,
                        dynamic_axes=None,
                        inline_autograd=False):
        if training == torch.onnx.TrainingMode.TRAINING:
            model.train()
        elif training == torch.onnx.TrainingMode.EVAL:
            model.eval()
        utils._validate_dynamic_axes(dynamic_axes, model, None, None)
        graph, params_dict, torch_out = utils._model_to_graph(model, input,
                                                              do_constant_folding=do_constant_folding,
                                                              _disable_torch_constant_prop=True,
                                                              operator_export_type=operator_export_type,
                                                              training=training,
                                                              input_names=input_names,
                                                              dynamic_axes=dynamic_axes,
                                                              inline_autograd=inline_autograd)
        return graph, params_dict, torch_out

class TestAutogradFuns_opset9(_BaseTestCase):
    opset_version = 9


    def test_single_output(self):
        class SingleOut(torch.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                result = i.exp()
                result = result.log()
                ctx.save_for_backward(result)
                return result

            @staticmethod
            def backward(ctx, grad_output):
                result, = ctx.saved_tensors
                return grad_output * result

        class Caller(torch.nn.Module):
            def forward(self, input):
                result = input + 5
                return SingleOut.apply(result) + 3

        model = Caller()
        input = torch.ones(1)

        graph, _, __ = self._model_to_graph(model, (input, ),
                                            input_names=["x"],
                                            dynamic_axes={"x": [0]},
                                            inline_autograd=True)
        iter = graph.nodes()
        self.assertEqual(next(iter).kind(), "onnx::Constant")
        self.assertEqual(next(iter).kind(), "onnx::Add")
        self.assertEqual(next(iter).kind(), "onnx::Exp")
        self.assertEqual(next(iter).kind(), "onnx::Log")

        graph_non_autograd, _, __ = self._model_to_graph(model, (input, ),
                                                         input_names=["x"],
                                                         dynamic_axes={"x": [0]})
        iter_na = graph_non_autograd.nodes()
        self.assertEqual(next(iter_na).kind(), "onnx::Constant")
        self.assertEqual(next(iter_na).kind(), "onnx::Add")
        python_op_node = next(iter_na)
        self.assertEqual(python_op_node.kind(), "prim::PythonOp")
        self.assertEqual(python_op_node.hasAttribute("Subgraph"), True)                              


    def test_multi_output(self):
        class MultiOut(torch.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                result_exp = i.exp()
                result_log = result_exp.log()
                ctx.save_for_backward(result_exp, result_log)
                return result_exp, result_log

            @staticmethod
            def backward(ctx, grad_output):
                result, = ctx.saved_tensors
                return grad_output * result

        class Caller(torch.nn.Module):
            def forward(self, input):
                return MultiOut.apply(input)

        model = Caller()
        input = torch.ones(1, 5)

        graph, _, __ = self._model_to_graph(model, (input, ),
                                            input_names=["x"],
                                            dynamic_axes={"x": [0, 1]},
                                            inline_autograd=True)
        iter = graph.nodes()
        self.assertEqual(next(iter).kind(), "onnx::Exp")
        self.assertEqual(next(iter).kind(), "onnx::Log")

        graph_non_autograd, _, __ = self._model_to_graph(model, (input, ),
                                                         input_names=["x"],
                                                         dynamic_axes={"x": [0]})
        iter_na = graph_non_autograd.nodes()
        python_op_node = next(iter_na)
        self.assertEqual(python_op_node.kind(), "prim::PythonOp")
        self.assertEqual(python_op_node.hasAttribute("Subgraph"), True)   


    def test_nested_autograd(self):
        class Child(torch.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                result = i.log()
                result_log = result.log()
                ctx.save_for_backward(result_log)
                return result_log

            @staticmethod
            def backward(ctx, grad_output):
                result, = ctx.saved_tensors
                return grad_output * result

        class Parent(torch.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                result_exp = i.exp()
                result_log = Child.apply(result_exp)
                ctx.save_for_backward(result_exp, result_log)
                return result_exp, result_log

            @staticmethod
            def backward(ctx, grad_output):
                result, = ctx.saved_tensors
                return grad_output * result

        class Caller(torch.nn.Module):
            def forward(self, input):
                return Parent.apply(input)

        model = Caller()
        input = torch.ones(1, 5)

        graph, _, __ = self._model_to_graph(model, (input, ),
                                            input_names=["x"],
                                            dynamic_axes={"x": [0, 1]},
                                            inline_autograd=True)
        iter = graph.nodes()
        self.assertEqual(next(iter).kind(), "onnx::Exp")
        self.assertEqual(next(iter).kind(), "onnx::Log")
        self.assertEqual(next(iter).kind(), "onnx::Log")

        graph_non_autograd, _, __ = self._model_to_graph(model, (input, ),
                                                         input_names=["x"],
                                                         dynamic_axes={"x": [0]})
        iter_na = graph_non_autograd.nodes()
        python_op_node = next(iter_na)
        self.assertEqual(python_op_node.kind(), "prim::PythonOp")
        self.assertEqual(python_op_node.hasAttribute("Subgraph"), True)


    def test_grad_mutliply(self):
        class GradMultiply(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, scale):
                ctx.scale = scale
                res = x.new(x)
                ctx.mark_shared_storage((x, res))
                return res

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output * ctx.scale, None

        class Caller(torch.nn.Module):
            def forward(self, input, scale):
                return GradMultiply.apply(input, scale)

        model = Caller()
        input = torch.ones(1, 5)
        scale = torch.ones(1)

        graph, _, __ = self._model_to_graph(model, (input, scale),
                                            input_names=["x", "y"],
                                            dynamic_axes={"x": [0, 1]},
                                            inline_autograd=True)
        iter = graph.nodes()
        self.assertEqual(next(iter).kind(), "onnx::Identity")
        
        graph_non_autograd, _, __ = self._model_to_graph(model, (input, scale),
                                                         input_names=["x", "y"],
                                                         dynamic_axes={"x": [0, 1]})
        iter_na = graph_non_autograd.nodes()
        python_op_node = next(iter_na)
        self.assertEqual(python_op_node.kind(), "prim::PythonOp")
        self.assertEqual(python_op_node.hasAttribute("Subgraph"), True)


    def test_aten_unsupported(self):
        class Erf(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                erf_out = torch.special.erf(x)
                ctx.save_for_backward(erf_out)
                return erf_out

            @staticmethod
            def backward(ctx, grad_output):
                result = ctx.saved_tensors
                return torch.special.erfinv(result), None

        class Caller(torch.nn.Module):
            def forward(self, input):
                return Erf.apply(input)

        model = Caller()
        input = torch.ones(1, 5)

        graph_non_autograd, _, __ = self._model_to_graph(model, (input, ),
                                                         input_names=["x"],
                                                         dynamic_axes={"x": [0, 1]})

        iter_na = graph_non_autograd.nodes()
        python_op_node = next(iter_na)
        self.assertEqual(python_op_node.kind(), "prim::PythonOp")
        self.assertEqual(python_op_node.hasAttribute("Subgraph"), True)

        graph, _, __ = self._model_to_graph(model, (input, ),
                                            input_names=["x"],
                                            dynamic_axes={"x": [0, 1]},
                                            inline_autograd=True)
        iter = graph.nodes()
        self.assertEqual(next(iter).kind(), "aten::special_erf")


class TestAutogradFuns_opset10(TestAutogradFuns_opset9):
    opset_version = 10


class TestAutogradFuns_opset11(TestAutogradFuns_opset9):
    opset_version = 11


class TestAutogradFuns_opset12(TestAutogradFuns_opset9):
    opset_version = 12


class TestAutogradFuns_opset13(TestAutogradFuns_opset9):
    opset_version = 13


class TestAutogradFuns_opset14(TestAutogradFuns_opset9):
    opset_version = 14


class TestAutogradFuns_opset15(TestAutogradFuns_opset9):
    opset_version = 15


if __name__ == "__main__":
    run_tests()
