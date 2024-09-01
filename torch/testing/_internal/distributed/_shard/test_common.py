# mypy: allow-untyped-defs

import torch
import torch.nn as nn

from torch.distributed._shard.sharded_tensor import ShardedTensor


class SimpleMegatronLM(nn.Module):
    def __init__(self, linear_size, rank=None, dtype=torch.float32):
        super().__init__()
        self.fc1 = nn.Linear(*linear_size[0], dtype=dtype)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(*linear_size[1], dtype=dtype)
        if rank is not None:
            self.fc1.cuda(rank)
            self.fc2.cuda(rank)

    def forward(self, inp):
        return self.fc2(self.gelu(self.fc1(inp)))

    def get_weights(self):
        if isinstance(self.fc1.weight, ShardedTensor):
            weight1 = self.fc1.weight.local_tensor()
        else:
            weight1 = self.fc1.weight

        if isinstance(self.fc2.weight, ShardedTensor):
            weight2 = self.fc2.weight.local_tensor()
        else:
            weight2 = self.fc2.weight

        return (weight1, weight2)

    def get_biases(self):
        return (self.fc1.bias, self.fc2.bias)

    def get_weight_grads(self):
        return (self.fc1.weight.grad, self.fc2.weight.grad)

    def get_bias_grads(self):
        return (self.fc1.bias.grad, self.fc2.bias.grad)
