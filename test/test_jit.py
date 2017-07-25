import torch
import torch.jit
import torch.nn as nn
import unittest
from torch.autograd import Variable
from common import TestCase, run_tests


class TestJit(TestCase):
    def test_simple(self):
        x = Variable(torch.Tensor([0.4]), requires_grad=True)
        y = Variable(torch.Tensor([0.7]), requires_grad=True)

        torch._C._tracer_enter((x, y))
        z = torch.sigmoid(torch.tanh(x * (x + y)))
        trace = torch._C._tracer_exit((z,))
        torch._C._jit_optim_fuse(trace)

        self.assertExpected(str(trace))
        return

        # Re-enable this when the interpreter is back
        zs = z._execution_engine.run_forward(trace, (x, y))
        self.assertEqual(z, zs)

        # TODO: test that backwards works correctly

    def test_lstm(self):
        # Careful: don't use fused backend (enabled with CUDA)
        # Pasted from test_LSTM_cell
        input = Variable(torch.randn(3, 10))
        hx = Variable(torch.randn(3, 20))
        cx = Variable(torch.randn(3, 20))
        lstm = torch.jit.trace_model(nn.LSTMCell(10, 20))
        trace, _ = lstm(input, (hx, cx))
        torch._C._jit_optim_fuse(trace)
        self.assertExpected(str(trace))

    def test_autograd_closure(self):
        x = Variable(torch.Tensor([0.4]), requires_grad=True)
        y = Variable(torch.Tensor([0.7]), requires_grad=True)

        torch._C._tracer_enter((x, y))

        z = torch.sigmoid(torch.tanh(x * (x + y)))
        w = torch.abs(x * x * x + y)

        trace = torch._C._tracer_exit((z, w))
        closure = torch._C._jit_createAutogradClosure(trace)
        z2, w2 = Variable._execution_engine.run_forward(closure, (x, y))
        self.assertEqual(z, z2)
        self.assertEqual(w, w2)


if __name__ == '__main__':
    run_tests()
