import torch
from . import nested
from .codegen import tensorextension

torch.nn.functional.embedding = nested.embedding_monkey
torch.nn.functional.dropout = nested.dropout_monkey
torch.nn.functional.cross_entropy = nested.cross_entropy_monkey
torch.nn.functional.linear = nested.linear_monkey
torch.nn.modules.LSTM.forward = nested.nn_lstm_forward_monkey

torch.nestedtensor = nested.make_nested_tensor
torch.cat = nested.cat

NestedTensor = nested.NestedTensor

# The contract is that func only works with torch.Tensor
def f(func, output, input1, input2):
    func(output.tensor, input1.tensor, input2.tensor)


# It's important to note here that output.a needs to be a ByteTensor
def f1(func, output, input1, input2):
    output.a = output.a.type(torch.uint8)
    func(output.tensor, input1.tensor, input2.tensor)


def f2(func, output, input1):
    func(output.tensor, input1.tensor)

NestedTensor = tensorextension.add_pointwise_binary_functions(NestedTensor, f)
NestedTensor = tensorextension.add_pointwise_comparison_functions(NestedTensor, f1)
NestedTensor = tensorextension.add_pointwise_unary_functions(NestedTensor, f2)

import pprint
pprint.pprint(dir(NestedTensor))

__all__ = ["NestedTensor"]
