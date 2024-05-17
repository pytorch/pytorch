# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
# This file is a model zoo for testing torch.distributed.pipelining.
import torch
from torch.distributed.pipelining import pipe_split


class ExampleCode(torch.nn.Module):
    default_dhid = 512
    default_batch_size = 256

    def __init__(self, d_hid: int = default_dhid):
        super().__init__()
        self.mm_param0 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param1 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.register_buffer("cval", torch.randn((d_hid,), requires_grad=False))
        self.lin0 = torch.nn.Linear(d_hid, d_hid)
        self.lin1 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x, y=torch.zeros(default_batch_size, default_dhid)):
        x = torch.mm(x, self.mm_param0)
        x = x + y
        x = torch.relu(x)
        # try passing a value that doesn't require_grad across skip boundaries
        a_constant = self.cval.clone()
        x = self.lin0(x)
        pipe_split()
        x = torch.relu(x) + a_constant
        x = torch.mm(x, self.mm_param1)
        x = self.lin1(x)
        x = torch.relu(x)
        return x


# MLP Layer
class MLPModule(torch.nn.Module):
    def __init__(self, d_hid):
        super().__init__()
        self.net1 = torch.nn.Linear(d_hid, d_hid)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        x = self.net1(x)
        x = self.relu(x)
        x = self.net2(x)
        return x


# Multi-MLP model
class MultiMLP(torch.nn.Module):
    def __init__(self, d_hid):
        super().__init__()
        self.mlp0 = MLPModule(d_hid)
        self.mlp1 = MLPModule(d_hid)

    def forward(self, x):
        x = self.mlp0(x)
        pipe_split()
        x = self.mlp1(x)
        return x
