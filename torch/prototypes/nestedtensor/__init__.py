import torch
from . import nested
from .codegen import tensorextension

torch.nn.functional.embedding = nested.embedding_monkey

torch.nestedtensor = nested.make_nested_tensor
NestedTensor = nested.NestedTensor

# The contract is that func only works with torch.Tensor
def f(func, output, input1, input2):
    assert len(output) == len(input1)
    assert len(input1) == len(input2)
    for i in range(len(output)):
        # NOTE: We are disabling broadcasting for now
        assert output.tensors[i].size() == input1.tensors[i].size()
        assert input2.tensors[i].size() == input1.tensors[i].size()
        func(output.tensors[i], input1.tensors[i], input2.tensors[i])

def f1(func, output, input1, input2):
    assert len(output) == len(input1)
    assert len(input1) == len(input2)
    output = output.to(torch.uint8)
    for i in range(len(output)):
        # NOTE: We are disabling broadcasting for now
        assert output.tensors[i].size() == input1.tensors[i].size()
        assert input2.tensors[i].size() == input1.tensors[i].size()
        func(output.tensors[i], input1.tensors[i], input2.tensors[i])


def f2(func, output, input1):
    assert len(output) == len(input1)
    for i in range(len(output)):
        # NOTE: We are disabling broadcasting for now
        assert output.tensors[i].size() == input1.tensors[i].size()
        func(output.tensors[i], input1.tensors[i])


NestedTensor = tensorextension.add_pointwise_binary_functions(NestedTensor, f)
NestedTensor = tensorextension.add_pointwise_comparison_functions(NestedTensor, f1)
NestedTensor = tensorextension.add_pointwise_unary_functions(NestedTensor, f2)

__all__ = ["NestedTensor"]
