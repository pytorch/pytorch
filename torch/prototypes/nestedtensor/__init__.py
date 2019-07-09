import torch
from . import nested
from .codegen import tensorextension

torch.nn.functional.embedding = nested.embedding_monkey

torch.nestedtensor = nested.make_nested_tensor
NestedTensor = nested.NestedTensor

# The contract is that func only works with torch.Tensor
def f(func, input1, input2, out=None):
    assert len(args) == 2
    if len(kwargs):
        assert list(kwargs.keys()) == ['out']
    output = kwargs.get('out')
    assert len(output) == len(input1)
    assert len(input1) == len(input2)
    if out is None:
        out = input1.clone()
    for i in range(len(output)):
        # NOTE: We are disabling broadcasting for now
        assert output.tensors[i].size() == input1.tensors[i].size()
        assert input2.tensors[i].size() == input1.tensors[i].size()
        func(input1.tensors[i], input2.tensors[i], out=out.tensors[i])
    return output

def f1(func, output, input1, input2):
    assert len(output) == len(input1)
    assert len(input1) == len(input2)
    if out is None:
        out = input1.clone()
    for i in range(len(output)):
        output.tensors[i] = output.tensors[i].to(torch.uint8)
    for i in range(len(output)):
        # NOTE: We are disabling broadcasting for now
        assert output.tensors[i].size() == input1.tensors[i].size()
        assert input2.tensors[i].size() == input1.tensors[i].size()
        func(input1.tensors[i], input2.tensors[i], out=out.tensors[i])
    return output


def f2(func, output, input1):
    assert len(output) == len(input1)
    for i in range(len(output)):
        # NOTE: We are disabling broadcasting for now
        assert output.tensors[i].size() == input1.tensors[i].size()
        func(output.tensors[i], input1.tensors[i])


torch, NestedTensor = tensorextension.add_pointwise_binary_functions(torch, NestedTensor, f)
torch, NestedTensor = tensorextension.add_pointwise_comparison_functions(torch, NestedTensor, f1)
NestedTensor = tensorextension.add_pointwise_unary_functions(NestedTensor, f2)

__all__ = ["NestedTensor"]
