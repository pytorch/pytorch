import torch
import torch.nn as nn
import math, unittest
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint_sequential


class TestCheckpoint(unittest.TestCase):

    def test_checkpoint_valid(self):
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
            nn.ReLU()
        )

        input_var = Variable(torch.randn(1, 100), requires_grad=True)

        # checkpointed
        chunks = 2
        modules = [module for k, module in model._modules.items()]
        out = checkpoint_sequential(modules, chunks, input_var)
        with self.assertRaises(AssertionError):
            torch.autograd.grad(
                outputs=[out], grad_outputs=[torch.ones(1, 5)], inputs=[input_var], create_graph=True
            )

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
        out = checkpoint_sequential(modules, chunks, input_var)
        out_checkpointed = out.data.clone()
        model.zero_grad()
        out.sum().backward()
        grad_checkpointed = {}
        for name, param in model.named_parameters():
            grad_checkpointed[name] = param.grad.data.clone()

        # compare the output and parameters gradients
        print("==> Output difference: {}".format(
            (out_checkpointed - out_not_checkpointed).abs().sum()))
        # self.assertEqual(out_checkpointed, out_not_checkpointed)
        print("==> Checking Params grad difference")
        for name in grad_checkpointed:
            print("Diff for {} is {}".format(
                name,
                (grad_checkpointed[name] - grad_not_checkpointed[name]).abs().sum()
            ))
            # self.assertEqual(grad_checkpointed[name], grad_not_checkpointed[name])


if __name__ == '__main__':
    unittest.main()
