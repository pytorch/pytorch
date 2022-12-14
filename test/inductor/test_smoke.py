# Owner(s): ["module: inductor"]
import logging
import unittest

import torch
import torch._dynamo as torchdynamo
import torch._inductor.config as torchinductor_config

torchdynamo.config.log_level = logging.INFO
torchdynamo.config.verbose = True
torchinductor_config.debug = True


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = torch.nn.Linear(1, 6)
        self.l2 = torch.nn.Linear(6, 1)

    def forward(self, x=None):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x


class SmokeTest(unittest.TestCase):
    def test_mlp(self):
        mlp = torchdynamo.optimize("inductor")(MLP().cuda())
        for _ in range(3):
            mlp(torch.randn(1, device="cuda"))
