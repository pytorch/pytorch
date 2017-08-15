import torch
import torch.jit
import torch.nn as nn
import unittest
import math
from torch.autograd import Variable, Function
from common import TestCase, run_tests


class TestJit(TestCase):
    maxDiff = None

    def test_simple(self):
        a = x = Variable(torch.Tensor([0.4]), requires_grad=True)
        b = y = Variable(torch.Tensor([0.7]), requires_grad=True)

        trace, (x, y) = torch._C._tracer_enter((x, y))
        z = torch.sigmoid(torch.tanh(x * (x + y)))
        z, = torch._C._tracer_exit((z,))
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_init(trace)
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_fuse(trace)
        torch._C._jit_pass_lint(trace)

        self.assertExpected(str(trace))

    def test_export(self):
        x = Variable(torch.Tensor([0.4]), requires_grad=True)
        y = Variable(torch.Tensor([0.7]), requires_grad=True)

        trace, (x, y) = torch._C._tracer_enter((x, y))
        z = -torch.sigmoid(torch.tanh(x * (x + y)))
        torch._C._tracer_exit((z,))
        torch._C._jit_pass_lint(trace)
        self.assertExpected(torch._C._jit_pass_export(trace))

    def test_lstm(self):
        # Careful: don't use fused backend (enabled with CUDA)
        # Pasted from test_LSTM_cell
        input = Variable(torch.randn(3, 10))
        hx = Variable(torch.randn(3, 20))
        cx = Variable(torch.randn(3, 20))
        trace, _ = torch.jit.record_trace(
            nn.LSTMCell(10, 20), input, (hx, cx))
        print(str(trace))
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_init(trace)
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_fuse(trace)
        torch._C._jit_pass_lint(trace)
        self.assertExpected(str(trace))

    def test_function_as_argument(self):
        # Careful: don't use fused backend (enabled with CUDA)
        # Pasted from test_LSTM_cell
        input = Variable(torch.randn(3, 10))
        hx = Variable(torch.randn(3, 20))
        cx = Variable(torch.randn(3, 20))
        lstm = nn.LSTMCell(10, 20)

        def a_function(a, b):
            return lstm(a, b)
        trace, _ = torch.jit.record_trace(
            a_function, input, (hx, cx), parameters=lstm.parameters())
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_init(trace)
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_fuse(trace)
        torch._C._jit_pass_lint(trace)
        self.assertExpected(str(trace))

    def test_verify(self):
        x = Variable(torch.Tensor([0.4]), requires_grad=True)
        y = Variable(torch.Tensor([0.7]), requires_grad=True)

        def doit(x, y):
            return torch.sigmoid(torch.tanh(x * (x + y)))

        traced = torch.jit.traced(
            doit, enabled=True, verify=True, time=True, optimize=False)
        z = traced(x, y)
        z2 = traced(x, y)
        self.assertEqual(z, torch.sigmoid(torch.tanh(x * (x + y))))
        self.assertEqual(z, z2)

    def test_traced_function(self):
        x = Variable(torch.Tensor([0.4]), requires_grad=True)
        y = Variable(torch.Tensor([0.7]), requires_grad=True)

        def doit(x, y):
            return torch.sigmoid(torch.tanh(x * (x + y)))

        traced = torch.jit.traced(doit)
        z = traced(x, y)
        z2 = traced(x, y)
        self.assertEqual(z, torch.sigmoid(torch.tanh(x * (x + y))))
        self.assertEqual(z, z2)

    def test_disabled_traced_function(self):
        x = Variable(torch.Tensor([0.4]), requires_grad=True)
        y = Variable(torch.Tensor([0.7]), requires_grad=True)

        def doit(x, y):
            return torch.sigmoid(torch.tanh(x * (x + y)))

        traced = torch.jit.traced(doit, enabled=False)
        z = traced(x, y)
        z2 = traced(x, y)
        self.assertEqual(z, torch.sigmoid(torch.tanh(x * (x + y))))
        self.assertEqual(z, z2)

    def test_traced_module(self):
        input = Variable(torch.randn(3, 10))
        hx = Variable(torch.randn(3, 20))
        cx = Variable(torch.randn(3, 20))
        lstm = nn.LSTMCell(10, 20)
        lstm = torch.jit.traced(lstm, verify=True)

        out = lstm(input, (hx, cx))
        out2 = lstm(input, (hx, cx))
        self.assertEqual(out, out2)

    def test_alexnet(self):

        inplace = False
        class AlexNet(nn.Module):

            def __init__(self, num_classes=1000):
                super(AlexNet, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                    nn.ReLU(inplace=inplace),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.Conv2d(64, 192, kernel_size=5, padding=2),
                    nn.ReLU(inplace=inplace),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.Conv2d(192, 384, kernel_size=3, padding=1),
                    nn.ReLU(inplace=inplace),
                    nn.Conv2d(384, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=inplace),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=inplace),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                )
                self.classifier = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(256 * 6 * 6, 4096),
                    nn.ReLU(inplace=inplace),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=inplace),
                    nn.Linear(4096, num_classes),
                )

            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), 256 * 6 * 6)
                x = self.classifier(x)
                return x

        x = Variable(torch.randn(10, 3, 224, 224).fill_(1.0), requires_grad=True)
        trace, _ = torch.jit.record_trace(AlexNet(), x)
        print(str(trace))
        self.assertExpected(str(trace))
        self.assertExpected(torch._C._jit_pass_export(trace), "pbtxt")

    def test_vgg(self):
        inplace = False
        cfg = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }

        class VGG(nn.Module):

            def __init__(self, features, num_classes=1000):
                super(VGG, self).__init__()
                self.features = features
                self.classifier = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(inplace=inplace),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=inplace),
                    nn.Dropout(),
                    nn.Linear(4096, num_classes),
                )
                self._initialize_weights()

            def _initialize_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                        m.weight.data.normal_(0, math.sqrt(2. / n))
                        if m.bias is not None:
                            m.bias.data.zero_()
                    elif isinstance(m, nn.BatchNorm2d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()
                    elif isinstance(m, nn.Linear):
                        n = m.weight.size(1)
                        m.weight.data.normal_(0, 0.01)
                        m.bias.data.zero_()

            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x

        def make_layers(cfg, batch_norm=False):
            layers = []
            in_channels = 3
            for v in cfg:
                if v == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=inplace)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=inplace)]
                    in_channels = v
            return nn.Sequential(*layers)

        # VGG 16-layer model (configuration "D")
        x = Variable(torch.randn(10, 3, 224, 224).fill_(1.0), requires_grad=True)
        vgg16 = VGG(make_layers(cfg['D']))
        trace, _ = torch.jit.record_trace(vgg16, x)
        print(str(trace))
        self.assertExpected(str(trace), "16")
        self.assertExpected(torch._C._jit_pass_export(trace), "16-pbtxt")

        # # VGG 16-layer model (configuration "D") with batch normalization
        # x = Variable(torch.randn(10, 3, 224, 224).fill_(1.0), requires_grad=True)
        # vgg16_bn = VGG(make_layers(cfg['D'], batch_norm=True))
        # trace, _ = torch.jit.record_trace(vgg16_bn, x)
        # print(str(trace))
        # self.assertExpected(str(trace), "16_bn")
        # self.assertExpected(torch._C._jit_pass_export(trace), "16_bn-pbtxt")

        # VGG 19-layer model (configuration "E")
        x = Variable(torch.randn(10, 3, 224, 224).fill_(1.0), requires_grad=True)
        vgg19 = VGG(make_layers(cfg['E']))
        trace, _ = torch.jit.record_trace(vgg19, x)
        print(str(trace))
        self.assertExpected(str(trace), "19")
        self.assertExpected(torch._C._jit_pass_export(trace), "19-pbtxt")

        # # VGG 19-layer model (configuration 'E') with batch normalization
        # x = Variable(torch.randn(10, 3, 224, 224).fill_(1.0), requires_grad=True)
        # vgg19_bn = VGG(make_layers(cfg['E'], batch_norm=True))
        # trace, _ = torch.jit.record_trace(vgg19_bn, x)
        # print(str(trace))
        # self.assertExpected(str(trace), "19_bn")
        # self.assertExpected(torch._C._jit_pass_export(trace), "19_bn-pbtxt")

    def test_autograd_closure(self):
        a = x = Variable(torch.Tensor([0.4]), requires_grad=True)
        b = y = Variable(torch.Tensor([0.7]), requires_grad=True)

        trace, (x, y) = torch._C._tracer_enter((x, y))

        z, _ = torch.max(x * (x + y), 0)
        w = torch.abs(x * x * x + y)

        z, w = torch._C._tracer_exit((z, w))
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_init(trace)
        torch._C._jit_pass_lint(trace)
        closure = torch._C._jit_createAutogradClosure(trace)
        z2, w2 = Variable._execution_engine.run_forward(closure, (a, b))
        self.assertEqual(z, z2)
        self.assertEqual(w, w2)

    def test_constant(self):
        a = x = Variable(torch.randn(2, 2), requires_grad=True)

        trace, (x,) = torch._C._tracer_enter((x,))

        y = Variable(torch.diag(torch.Tensor([2, 2])))
        z = x.matmul(y)

        z, = torch._C._tracer_exit((z,))
        closure = torch._C._jit_createAutogradClosure(trace)

        z2, = Variable._execution_engine.run_forward(closure, (a,))
        self.assertEqual(z, z2)

        y.data.fill_(1000)  # make sure the data has been cloned

        a2 = Variable(torch.ones(2, 2) * 2, requires_grad=True)
        z3, = Variable._execution_engine.run_forward(closure, (a2,))
        self.assertEqual(z3.data, torch.ones(2, 2) * 4)

    def test_c_function(self):
        x = Variable(torch.randn(1, 3, 10, 10))
        m = nn.Conv2d(3, 8, 3, 1)

        trace, new_vars = torch._C._tracer_enter((x,) + tuple(m.parameters()))
        x = new_vars[0]
        y = m(x)
        _ = torch._C._tracer_exit((y,))
        self.assertExpected(str(trace))

    def test_legacy_fail(self):

        class Legacy(Function):
            def forward(self, x):
                return x

            def backward(self, grad_output):
                return grad_output
        a = x = Variable(torch.Tensor([0]), requires_grad=True)
        trace, (x,) = torch._C._tracer_enter((x,))
        self.assertRaises(RuntimeError, lambda: Legacy()(x))
        x, = torch._C._tracer_exit((x,))

    @unittest.skip("in-place is not supported")
    def test_inplace_transplant(self):
        a = x = Variable(torch.Tensor([0]), requires_grad=True)
        trace, (x,) = torch._C._tracer_enter((x,))
        y = x.clone()
        y.add_(2)
        y.add_(3)
        y, = torch._C._tracer_exit((y,))
        self.assertExpected(str(trace))

    def test_backward(self):
        a = Variable(torch.randn(2, 2), requires_grad=True)
        b = Variable(torch.randn(2, 2), requires_grad=True)

        x = a
        y = a * b

        trace, (x, y) = torch._C._tracer_enter((x, y))
        z = y * 2 * x
        z, = torch._C._tracer_exit((z,))
        torch._C._jit_pass_lint(trace)

        grad, = torch.autograd.grad(z, x, Variable(
            torch.ones(2, 2), requires_grad=True), create_graph=True)
        torch._C._jit_pass_lint(trace)

        # Run dead code elimination to remove unused trace nodes
        torch._C._jit_pass_dco(trace)
        self.assertExpected(str(trace))

    def test_cpp(self):
        torch._C._jit_run_cpp_tests()


if __name__ == '__main__':

    run_tests()
