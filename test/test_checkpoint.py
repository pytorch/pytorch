import torch
import torch.nn as nn
import math, unittest
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint_sequential
from common import TestCase, run_tests


class TestCheckpoint(TestCase):

    def custom(self):
        def custom_forward(start, end, modules):
            def forward_extended(*inputs):
                input = inputs[0]
                for j in range(start, end + 1):
                    input = modules[j](input)
                return input
            return forward_extended
        return custom_forward

    def test_checkpoint(self):
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
            nn.ReLU()
        )

        x = Variable(torch.randn(1, 100), requires_grad=True)

        # not checkpointed
        original = model
        out = original(x)
        out_not_checkpointed = out.data.clone()
        original.zero_grad()
        out.sum().backward()
        grad_not_checkpointed = {}
        for name, param in model.named_parameters():
            grad_not_checkpointed[name] = param.grad.data.clone()

        # checkpointed
        chunks = 2
        modules = [module for k, module in model._modules.items()]
        input_var = Variable(x.data, requires_grad=True)
        out = checkpoint_sequential(modules, chunks, self.custom(), input_var)
        out_checkpointed = out.data.clone()
        model.zero_grad()
        out.sum().backward()
        grad_checkpointed = {}
        for name, param in model.named_parameters():
            grad_checkpointed[name] = param.grad.data.clone()

        # compare the output and parameters gradients
        self.assertEqual(out_checkpointed, out_not_checkpointed)
        for name in grad_checkpointed:
            self.assertEqual(grad_checkpointed[name], grad_not_checkpointed[name])


if __name__ == '__main__':
    run_tests()
