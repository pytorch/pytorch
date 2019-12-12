#!/usr/bin/env python3

import torch
import unittest

from caffe2.python import workspace


class PlusTwoNet(torch.nn.Module):
    def forward(self, x):
        x = torch.ops._caffe2.AliasWithName(x, "input")
        y = x + 1
        y = torch.ops._caffe2.AliasWithName(y, "intermediate")
        z = y + 1
        z = torch.ops._caffe2.AliasWithName(z, "output")
        return z


class TestAliasWithNameOp(unittest.TestCase):
    device = "cuda" if workspace.has_cuda_support else "cpu"

    def test_alias_with_name_op(self):
        plus_two_net = PlusTwoNet()

        test_vector = [
            torch.Tensor([42]),
            torch.Tensor(1, 2, 3),
            torch.Tensor(5, 2, 3),
            torch.Tensor(2, 2).to(torch.int64),
        ]

        for x in test_vector:
            x = x.to(self.device)
            y = plus_two_net(x)
            torch.testing.assert_allclose(x + 2, y)

    def test_alias_is_in_place(self):
        x = torch.Tensor([3, 42]).to(self.device)
        y = torch.ops._caffe2.AliasWithName(x, "new_name")
        x[1] = 6
        torch.testing.assert_allclose(x, torch.Tensor([3, 6]).to(self.device))
        # y should also change because y is alias of x
        torch.testing.assert_allclose(y, torch.Tensor([3, 6]).to(self.device))
