import torch
import torch.jit
from torch.autograd import Variable
from common import TestCase, run_tests


class TestJit(TestCase):
    def test_simple(self):
        x = Variable(torch.Tensor([0.4]), requires_grad=True)
        y = Variable(torch.Tensor([0.7]), requires_grad=True)

        torch._C._tracer_enter((x, y))
        z = torch.sigmoid(torch.tanh(x * (x + y)))
        trace = torch._C._tracer_exit((z,))

        # TODO: Do something more automated here
        print(trace)
        return

        # Re-enable this when the interpreter is back
        zs = z._execution_engine.run_forward(trace, (x, y))
        self.assertEqual(z, zs)

        # TODO: test that backwards works correctly

if __name__ == '__main__':
    run_tests()
