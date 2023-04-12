import torch

import torch.library
import torch.nn as nn

from torch.distributed._spmd import compile

from torch.distributed._spmd.parallel_mode import DataParallel
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.net1 = nn.Linear(50, 32)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(32, 8)

    def forward(self, x):
        return torch.sigmoid(self.net2(self.relu(self.net1(x))))

    def reset_parameters(self, *args, **kwargs):
        self.net1.reset_parameters()
        self.net2.reset_parameters()


# simple train step definition, just an example
def train_step(model, optim, train_batch):
    def loss_fn(out, labels):
        return (out - labels).sum()

    optim.zero_grad()
    inputs, labels = train_batch

    out = model(inputs)
    loss = loss_fn(out, labels)
    loss.backward()
    optim.step()
    return loss


class TestDataParallel(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @with_comms
    def test_replicate(self):
        model = SimpleMLP().cuda(self.rank)
        # model = SimpleMLP()
        optim = torch.optim.SGD(model.parameters(), lr=0.1)
        # optim = torch.optim.SGD(model.parameters(), lr=0.1, foreach=True)
        # optim = torch.optim.Adam(model.parameters(), lr=0.1, fused=True)
        # train_batch = (
        #     torch.randn(256, 50).to(self.rank),
        #     torch.randn(256, 8).to(self.rank),
        # )
        # train_batch = (
        #     torch.randn(256, 50),
        #     torch.randn(256, 8),
        # )
        # train_batch = (
        #     torch.randn(128, 50),
        #     torch.randn(128, 8),
        # )
        train_batch = (
            torch.randn(128, 50).to(self.rank),
            torch.randn(128, 8).to(self.rank),
        )

        # need one step to warm up optimizers
        # train_step(model, optim, train_batch)

        # compile the model
        # compiled = compile(parallel_mode=DataParallel("replicate"))(train_step)
        compiled = compile(parallel_mode=DataParallel("replicate"))(train_step)

        compiled(model, optim, train_batch)
        for param in model.parameters():
            print(param.shape, type(param))
        # run the model

    @with_comms
    def test_fully_shard(self):
        model = SimpleMLP().cuda(self.rank)
        # model = SimpleMLP()
        optim = torch.optim.SGD(model.parameters(), lr=0.1)
        # optim = torch.optim.SGD(model.parameters(), lr=0.1, foreach=True)
        # optim = torch.optim.Adam(model.parameters(), lr=0.1, fused=True)
        # train_batch = (
        #     torch.randn(256, 50).to(self.rank),
        #     torch.randn(256, 8).to(self.rank),
        # )
        # train_batch = (
        #     torch.randn(256, 50),
        #     torch.randn(256, 8),
        # )
        # train_batch = (
        #     torch.randn(128, 50),
        #     torch.randn(128, 8),
        # )
        train_batch = (
            torch.randn(128, 50).to(self.rank),
            torch.randn(128, 8).to(self.rank),
        )

        # need one step to warm up optimizers
        # train_step(model, optim, train_batch)

        # compile the model
        # compiled = compile(parallel_mode=DataParallel("replicate"))(train_step)
        compiled = compile(parallel_mode=DataParallel("fully_shard"))(train_step)

        compiled(model, optim, train_batch)
        for param in model.parameters():
            print(param.shape, type(param))
        # run the model
