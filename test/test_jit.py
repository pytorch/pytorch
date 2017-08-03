import torch
import torch.jit
import torch.nn as nn
import unittest
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

    def test_lstm(self):
        # Careful: don't use fused backend (enabled with CUDA)
        # Pasted from test_LSTM_cell
        input = Variable(torch.randn(3, 10))
        hx = Variable(torch.randn(3, 20))
        cx = Variable(torch.randn(3, 20))
        lstm = torch.jit.trace_model(nn.LSTMCell(10, 20))
        trace, _ = lstm(input, (hx, cx))
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_init(trace)
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_fuse(trace)
        torch._C._jit_pass_lint(trace)
        self.assertExpected(str(trace))

    @unittest.skip("in-place is not supported")
    def test_alexnet(self):

        class AlexNet(nn.Module):

            def __init__(self, num_classes=1000):
                super(AlexNet, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.Conv2d(64, 192, kernel_size=5, padding=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.Conv2d(192, 384, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(384, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                )
                self.classifier = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(256 * 6 * 6, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Linear(4096, num_classes),
                )

            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), 256 * 6 * 6)
                x = self.classifier(x)
                return x

        model = torch.jit.trace_model(AlexNet())
        x = Variable(torch.randn(10, 3, 224, 224), requires_grad=True)
        trace, _ = model(x)
        self.assertExpected(str(trace))

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

        grad, = torch.autograd.grad(z, x, Variable(torch.ones(2, 2), requires_grad=True), create_graph=True)
        torch._C._jit_pass_lint(trace)

        # Run dead code elimination to remove unused trace nodes
        torch._C._jit_pass_dco(trace)
        self.assertExpected(str(trace))

    def test_cpp(self):
        torch._C._jit_run_cpp_tests()


if __name__ == '__main__':

    run_tests()
