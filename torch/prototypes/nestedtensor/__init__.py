import torch
from . import nested

print("1")

torch.nn.functional.embedding = nested.embedding_monkey
torch.nn.functional.dropout = nested.dropout_monkey
torch.nn.functional.cross_entropy = nested.cross_entropy_monkey
torch.nn.functional.linear = nested.linear_monkey
torch.nn.modules.LSTM.forward = nested.nn_lstm_forward_monkey

torch.nestedtensor = nested.make_nested_tensor
torch.cat = nested.cat
torch.stack = nested.stack

print("2")

NestedTensor = nested.NestedTensor

print("3")

__all__ = ["NestedTensor"]
