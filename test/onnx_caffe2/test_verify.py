# Owner(s): ["module: onnx"]

import caffe2.python.onnx.backend as backend
import torch
from torch.autograd import Function
from torch.nn import Module, Parameter
from torch.testing._internal import common_utils
from verify import verify


class TestVerify(common_utils.TestCase):
    maxDiff = None

    def assertVerifyExpectFail(self, *args, **kwargs):
        try:
            verify(*args, **kwargs)
        except AssertionError as e:
            if str(e):
                # substring a small piece of string because the exact message
                # depends on system's formatting settings
                # self.assertExpected(str(e)[:60])
                # NB: why we comment out the above check? because numpy keeps
                # changing the error format, and we have to keep updating the
                # expect files let's relax this constraint
                return
            else:
                raise
        # Don't put this in the try block; the AssertionError will catch it
        self.assertTrue(False, msg="verify() did not fail when expected to")

    def test_result_different(self):
        class BrokenAdd(Function):
            @staticmethod
            def symbolic(g, a, b):
                return g.op("Add", a, b)

            @staticmethod
            def forward(ctx, a, b):
                return a.sub(b)  # yahaha! you found me!

        class MyModel(Module):
            def forward(self, x, y):
                return BrokenAdd().apply(x, y)

        x = torch.tensor([1, 2])
        y = torch.tensor([3, 4])
        self.assertVerifyExpectFail(MyModel(), (x, y), backend)

    def test_jumbled_params(self):
        class MyModel(Module):
            def forward(self, x):
                y = x * x
                self.param = Parameter(torch.tensor([2.0]))
                return y

        x = torch.tensor([1, 2])
        with self.assertRaisesRegex(RuntimeError, "state_dict changed"):
            verify(MyModel(), x, backend)

    def test_dynamic_model_structure(self):
        class MyModel(Module):
            def __init__(self):
                super().__init__()
                self.iters = 0

            def forward(self, x):
                if self.iters % 2 == 0:
                    r = x * x
                else:
                    r = x + x
                self.iters += 1
                return r

        x = torch.tensor([1, 2])
        self.assertVerifyExpectFail(MyModel(), x, backend)

    def test_embedded_constant_difference(self):
        class MyModel(Module):
            def __init__(self):
                super().__init__()
                self.iters = 0

            def forward(self, x):
                r = x[self.iters % 2]
                self.iters += 1
                return r

        x = torch.tensor([[1, 2], [3, 4]])
        self.assertVerifyExpectFail(MyModel(), x, backend)

    def test_explicit_test_args(self):
        class MyModel(Module):
            def forward(self, x):
                if x.data.sum() == 1.0:
                    return x + x
                else:
                    return x * x

        x = torch.tensor([[6, 2]])
        y = torch.tensor([[2, -1]])
        self.assertVerifyExpectFail(MyModel(), x, backend, test_args=[(y,)])


if __name__ == "__main__":
    common_utils.run_tests()
