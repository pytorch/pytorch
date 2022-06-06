# Owner(s): ["module: onnx"]

import torch
from torch.onnx._globals import GLOBALS

from test_pytorch_onnx_onnxruntime import run_model_test

import unittest

class TestAutogradFuns(unittest.TestCase):
    opset_version = GLOBALS.export_onnx_opset_version
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

    # Skip test as torch.erf() is not supported
    # This is a proof of concept to show error message originates from
    # unsupported operator rather than missing symbolic method
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
        try:
            run_model_test(self, model, input_args=(input,))
        except Exception:
            raise unittest.SkipTest("Unsupported operator")

    def test_inline_and_symbolic(self):
        class Exp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                ctx.save_for_backward(input)
                return i.exp()

            @staticmethod
            def symbolic(g, input):
                return g.op("Exp", input)

        class LogLog(torch.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                ctx.save_for_backward(input)
                return i.log().log()

        class Caller(torch.nn.Module):
            def forward(self, input):
                exp_result = Exp.apply(input)
                return LogLog.apply(exp_result)

        model = Caller()
        input = torch.ones(1)
        run_model_test(self, model, input_args=(input,))


if __name__ == "__main__":
    unittest.main()
