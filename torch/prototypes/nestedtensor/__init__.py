import torch
from . import nested
from .codegen import tensorextension

torch.nestedtensor = nested.make_nested_tensor
NestedTensor = nested.NestedTensor

# The contract is that func only works with torch.Tensor
def f(func, output, input1, input2):
    assert len(output) == len(input1)
    assert len(input1) == len(input2)
    for i in range(len(output)):
        func(output.tensors[i], input1.tensors[i], input2.tensors[i])


def f1(func, output, input1):
    assert len(output) == len(input1)
    for i in range(len(output)):
        func(output.tensors[i], input1.tensors[i])


NestedTensor = tensorextension.add_pointwise_binary_functions(NestedTensor, f)
NestedTensor = tensorextension.add_pointwise_comparison_functions(NestedTensor, f)
NestedTensor = tensorextension.add_pointwise_unary_functions(NestedTensor, f1)

__all__ = ["NestedTensor"]
