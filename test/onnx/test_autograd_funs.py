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

from test_pytorch_onnx_onnxruntime import run_model_test

import torchvision

import onnx

import io
import copy
import unittest

skip = unittest.skip

class TestAutogradFuns(unittest.TestCase):
    opset_version = 9
    keep_initializers_as_inputs = False
    onnx_shape_inference = True

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
        run_model_test(self, model, input_args=(input,))
    
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
        run_model_test(self, model, input_args=(input,))

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
        run_model_test(self, model, input_args=(input,))

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


if __name__ == "__main__":
    unittest.main()
