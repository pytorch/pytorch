import torch
import torch.jit
import torch.onnx
import torch.nn as nn
from torch.nn import Module
import unittest
from torch.autograd import Variable, Function
from common import TestCase, run_tests
import io

try:
    import onnx
    import google.protobuf.text_format
    HAS_TOFFEE = True
except ImportError:
    HAS_TOFFEE = False

onnx_only = unittest.skipIf(not HAS_TOFFEE, "no onnx support library")


def export_to_string(model, inputs, *args, **kwargs):
    f = io.BytesIO()
    torch.onnx.export(model, inputs, f, *args, **kwargs)
    return f.getvalue()


class FuncModule(Module):
    def __init__(self, f):
        super(FuncModule, self).__init__()
        self.f = f

    def forward(self, *args):
        return self.f(*args)


@onnx_only
class TestONNX(TestCase):
    maxDiff = None

    def assertONNXExpected(self, binary_pb, subname=None):
        model_def = onnx.ModelProto.FromString(binary_pb)
        self.assertExpected(google.protobuf.text_format.MessageToString(model_def, float_format='.15g'), subname)

    def test_basic(self):
        x = Variable(torch.Tensor([0.4]), requires_grad=True)
        y = Variable(torch.Tensor([0.7]), requires_grad=True)

        trace = torch._C._tracer_enter((x, y), 0)
        z = -torch.sigmoid(torch.tanh(x * (x + y)))
        torch._C._tracer_exit((z,))
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_onnx(trace)
        self.assertONNXExpected(trace.export())

    def test_view(self):
        x = Variable(torch.Tensor([0]), requires_grad=True)
        trace, _ = torch.jit.trace(lambda x: x.view(1, 1), x)
        torch._C._jit_pass_onnx(trace)
        self.assertONNXExpected(trace.export())

    def test_transpose(self):
        x = Variable(torch.Tensor([[0, 1], [2, 3]]), requires_grad=True)
        trace, _ = torch.jit.trace(lambda x: x.transpose(0, 1).transpose(1, 0), x)
        torch._C._jit_pass_onnx(trace)
        self.assertONNXExpected(trace.export())

    def test_concat(self):
        # volatile is of particular interest; it caused a segfault
        # with the exporter
        x = Variable(torch.randn(2, 3), volatile=True)
        y = Variable(torch.randn(2, 3), volatile=True)
        self.assertONNXExpected(export_to_string(FuncModule(lambda inputs: torch.cat(inputs, 1)), ((x, y),)))

    def test_permute(self):
        x = Variable(torch.Tensor([[[[[[0]]]]]]), requires_grad=True)
        self.assertONNXExpected(export_to_string(FuncModule(lambda x: x.permute(0, 1, 4, 2, 5, 3)), (x, )))

    def test_params(self):
        x = Variable(torch.Tensor([[1, 2], [3, 4]]), requires_grad=True)
        y = Variable(torch.Tensor([[1, 2], [3, 4]]), requires_grad=True)
        trace, _ = torch.jit.trace(lambda x, y: -torch.sigmoid(torch.tanh(x * (x + y))), (x, y))
        initializers = [x.data]
        torch._C._jit_pass_onnx(trace)
        self.assertONNXExpected(trace.export(initializers))

    def test_non_float_params(self):
        x = Variable(torch.LongTensor([[1, 2], [3, 4]]), requires_grad=True)
        y = Variable(torch.LongTensor([[1, 2], [3, 4]]), requires_grad=True)
        trace, _ = torch.jit.trace(lambda x, y: x * y + x, (x, y))
        initializers = [x.data]
        torch._C._jit_pass_onnx(trace)
        self.assertONNXExpected(trace.export(initializers))

    def test_symbolic_mismatch(self):
        class MyFun(Function):
            @staticmethod
            def symbolic(g, x):
                # The inside of this function should never be invoked, because
                # we will fail due to an argument mismatch first.
                assert False

            @staticmethod
            def forward(ctx, x, y):
                return x + y

        x = Variable(torch.randn(2, 2).fill_(1.0))
        y = Variable(torch.randn(2, 2).fill_(1.0))
        with self.assertRaisesRegex(TypeError, "occurred when translating MyFun"):
            export_to_string(FuncModule(MyFun().apply), (x, y))

    # TODO: Do an nn style test for these
    def test_batchnorm(self):
        x = Variable(torch.randn(2, 2).fill_(1.0), requires_grad=True)
        self.assertONNXExpected(export_to_string(nn.BatchNorm2d(2), x))

    def test_batchnorm_training(self):
        x = Variable(torch.randn(2, 2).fill_(1.0), requires_grad=True)
        self.assertONNXExpected(export_to_string(nn.BatchNorm2d(2), x, training=True))

    def test_conv(self):
        x = Variable(torch.randn(20, 16, 50, 40).fill_(1.0), requires_grad=True)
        self.assertONNXExpected(export_to_string(nn.Conv2d(16, 13, 3, bias=False), x))

    def test_maxpool(self):
        x = Variable(torch.randn(20, 16, 50))
        self.assertONNXExpected(export_to_string(nn.MaxPool1d(3, stride=2), x))

    def test_at_op(self):
        x = Variable(torch.randn(3, 4))

        class MyFun(Function):

            @staticmethod
            def symbolic(g, x):
                return g.at("add", x, x)

            @staticmethod
            def forward(ctx, x):
                return x + x

        class MyModule(Module):

            def forward(self, x):
                return MyFun.apply(x)

        self.assertONNXExpected(export_to_string(MyModule(), x))


if __name__ == '__main__':
    run_tests()
