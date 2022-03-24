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
                        dynamic_axes=None):
        if training == torch.onnx.TrainingMode.TRAINING:
            model.train()
        elif training == torch.onnx.TrainingMode.EVAL:
            model.eval()
        # Need disable onnx_shape_inference for this test because it puts const node to initializers.
        _set_onnx_shape_inference(False)
        utils._validate_dynamic_axes(dynamic_axes, model, None, None)
        graph, params_dict, torch_out = utils._model_to_graph(model, input,
                                                              do_constant_folding=do_constant_folding,
                                                              _disable_torch_constant_prop=True,
                                                              operator_export_type=operator_export_type,
                                                              training=training,
                                                              input_names=input_names,
                                                              dynamic_axes=dynamic_axes)
        _set_onnx_shape_inference(True)
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
                                            dynamic_axes={"x": [0]})
        iter = graph.nodes()
        self.assertEqual(next(iter).kind(), "onnx::Constant")
        self.assertEqual(next(iter).kind(), "onnx::Add")
        self.assertEqual(next(iter).kind(), "onnx::Exp")
        self.assertEqual(next(iter).kind(), "onnx::Log")


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
                                            dynamic_axes={"x": [0, 1]})
        iter = graph.nodes()
        self.assertEqual(next(iter).kind(), "onnx::Exp")
        self.assertEqual(next(iter).kind(), "onnx::Log")


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
