# Owner(s): ["oncall: jit"]

import torch
from torch.testing._internal.common_utils import skipIfTorchDynamo
from torch.testing._internal.jit_utils import JitTestCase


if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


@skipIfTorchDynamo("Not a suitable test for TorchDynamo")
class TestTraceAutograd(JitTestCase):
    def validateTrace(self, model):
        inputs = torch.FloatTensor(2, 3, 4).uniform_(0, 10)
        traced = torch.jit.trace(model, inputs)
        self.assertEqual(abs(traced(inputs) - model(inputs)).sum(), 0.0)

    def test_autograd_sqrt(self):
        class Model(torch.nn.Module):
            def forward(self, inputs):
                x = inputs[0].detach().requires_grad_()
                with torch.enable_grad():
                    loss = torch.sqrt(x).sum()
                    return torch.autograd.grad([loss], [x])[0]

        self.validateTrace(Model())

    def test_autograd_rsqrt(self):
        class Model(torch.nn.Module):
            def forward(self, inputs):
                x = inputs[0].detach().requires_grad_()
                with torch.enable_grad():
                    loss = torch.rsqrt(x).sum()
                    return torch.autograd.grad([loss], [x])[0]

        self.validateTrace(Model())

    def test_autograd_pow(self):
        class Model(torch.nn.Module):
            def forward(self, inputs):
                x = inputs[0].detach().requires_grad_()
                with torch.enable_grad():
                    loss = torch.pow(x, 2).sum()
                    return torch.autograd.grad([loss], [x])[0]

        self.validateTrace(Model())

    def test_autograd_mean(self):
        class Model(torch.nn.Module):
            def forward(self, inputs):
                x = inputs[0].detach().requires_grad_()
                with torch.enable_grad():
                    loss = torch.mean(x).sum()
                    return torch.autograd.grad([loss], [x])[0]

        self.validateTrace(Model())

    def test_autograd_matmul(self):
        class Model(torch.nn.Module):
            def forward(self, inputs):
                x = inputs[0].detach().requires_grad_()
                with torch.enable_grad():
                    loss = torch.matmul(x, x.transpose(-2, -1)).sum()
                    return torch.autograd.grad([loss], [x])[0]

        self.validateTrace(Model())
