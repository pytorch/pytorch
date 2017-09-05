import torch
import torch.jit
import torch.onnx
import torch.nn as nn
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


@onnx_only
class TestONNX(TestCase):
    maxDiff = None

    def assertONNXExpected(self, binary_pb, subname=None):
        graph_def = onnx.GraphProto.FromString(binary_pb)
        self.assertExpected(google.protobuf.text_format.MessageToString(graph_def, float_format='.15g'), subname)

    def test_basic(self):
        x = Variable(torch.Tensor([0.4]), requires_grad=True)
        y = Variable(torch.Tensor([0.7]), requires_grad=True)

        trace = torch._C._tracer_enter((x, y))
        z = -torch.sigmoid(torch.tanh(x * (x + y)))
        torch._C._tracer_exit((z,))
        torch._C._jit_pass_lint(trace)
        self.assertONNXExpected(trace.export(False))

    def test_view(self):
        x = Variable(torch.Tensor([0]), requires_grad=True)
        trace, _ = torch.jit.record_trace(lambda x: x.view(1, 1), x)
        self.assertONNXExpected(trace.export(False))

    def test_transpose(self):
        x = Variable(torch.Tensor([[0, 1], [2, 3]]), requires_grad=True)
        trace, _ = torch.jit.record_trace(lambda x: x.transpose(0, 1).transpose(1, 0), x)
        self.assertONNXExpected(trace.export(False))

    def test_permute(self):
        x = Variable(torch.Tensor([[[[[[0]]]]]]), requires_grad=True)
        trace, _ = torch.jit.record_trace(lambda x: x.permute(0, 1, 4, 2, 5, 3), x)
        self.assertONNXExpected(trace.export(False))

    def test_params(self):
        x = Variable(torch.Tensor([[1, 2], [3, 4]]), requires_grad=True)
        y = Variable(torch.Tensor([[1, 2], [3, 4]]), requires_grad=True)
        trace, _ = torch.jit.record_trace(lambda x, y: -torch.sigmoid(torch.tanh(x * (x + y))), x, y)
        initializers = [x.data]
        self.assertONNXExpected(trace.export(initializers, False))

    def test_non_float_params(self):
        x = Variable(torch.LongTensor([[1, 2], [3, 4]]), requires_grad=True)
        y = Variable(torch.LongTensor([[1, 2], [3, 4]]), requires_grad=True)
        trace, _ = torch.jit.record_trace(lambda x, y: x * y + x, x, y)
        initializers = [x.data]
        self.assertONNXExpected(trace.export(initializers, False))

    # TODO: Do an nn style test for these
    def test_batchnorm(self):
        x = Variable(torch.randn(2, 2).fill_(1.0), requires_grad=True)
        self.assertONNXExpected(export_to_string(nn.BatchNorm2d(2), x))

    def test_conv(self):
        x = Variable(torch.randn(20, 16, 50, 40).fill_(1.0), requires_grad=True)
        self.assertONNXExpected(export_to_string(nn.Conv2d(16, 13, 3, bias=False), x))

    def test_maxpool(self):
        x = Variable(torch.randn(20, 16, 50))
        self.assertONNXExpected(export_to_string(nn.MaxPool1d(3, stride=2), x))

if __name__ == '__main__':
    run_tests()
