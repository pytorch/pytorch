import torch
from . import nested
from .codegen import tensorextension

torch.nn.functional.embedding = nested.embedding_monkey

torch.nestedtensor = nested.make_nested_tensor
NestedTensor = nested.NestedTensor

def _create_out(input1, out):
    if out is None:
        out = input1.clone()
    assert len(out) == len(input1)
    return out

# The contract is that func only works with torch.Tensor
def f(func, input1, input2, out=None):
    out = _create_out(input1, out)
    assert len(input1) == len(input2)
    for i in range(len(out)):
        # NOTE: We are disabling broadcasting for now
        assert out.tensors[i].size() == input1.tensors[i].size()
        assert input2.tensors[i].size() == input1.tensors[i].size()
        func(input1.tensors[i], input2.tensors[i], out=out.tensors[i])
    return out


def f1(func, input1, input2, out=None):
    out = _create_out(input1, out)
    assert len(input1) == len(input2)
    for i in range(len(out)):
        out.tensors[i] = out.tensors[i].to(torch.uint8)
    for i in range(len(out)):
        # NOTE: We are disabling broadcasting for now
        assert out.tensors[i].size() == input1.tensors[i].size()
        assert input2.tensors[i].size() == input1.tensors[i].size()
        func(input1.tensors[i], input2.tensors[i], out=out.tensors[i])
    return out


def f2(func, input1, out=None):
    out = _create_out(input1, out)
    for i in range(len(out)):
        # NOTE: We are disabling broadcasting for now
        assert out.tensors[i].size() == input1.tensors[i].size()
        func(out.tensors[i], input1.tensors[i])


torch, NestedTensor = tensorextension.add_pointwise_unary_functions(torch, NestedTensor, f2)
torch, NestedTensor = tensorextension.add_pointwise_binary_functions(torch, NestedTensor, f)
torch, NestedTensor = tensorextension.add_pointwise_comparison_functions(torch, NestedTensor, f1)

__all__ = ["NestedTensor"]
