import torch
from . import nested
import torch.prototypes.nestedtensor.codegen as codegen

# from torch.utils.cpp_extension import load
# tensor_list = load(name="tensor_list", sources=["torch/prototypes/nestedtensor/csrc/generated/tensor_list.cpp"])

def is_available():
    return hasattr(torch._C, "_tensor_list_init")

if is_available() and not torch._C._tensor_list_init():
    raise RuntimeError("Failed to initialize PyTorch distributed support")

NestedTensor = nested.NestedTensor

def _create_out(input1, out, dtype=None):
    if out is None:
        if dtype is None:
            dtype = input1.dtype
        out_tensors = []
        for tensor in input1.tensors:
            out_tensors.append(torch.empty_like(tensor, dtype=dtype))
        out = NestedTensor(out_tensors)
    assert len(out) == len(input1)
    return out


def _unary(func_name, func, input1, out=None):
    out = _create_out(input1, out)
    for i in range(len(out)):
        # NOTE: We are disabling broadcasting for now
        assert out.tensors[i].size() == input1.tensors[i].size()
    list_func = getattr(tensor_list, func_name)
    list_func(input1.tensors, out.tensors)
    return out


# The contract is that func only works with torch.Tensor
def _binary(func_name, func, input1, input2, out=None):
    out = _create_out(input1, out)
    assert len(input1) == len(input2)
    for i in range(len(out)):
        # NOTE: We are disabling broadcasting for now
        assert out.tensors[i].size() == input1.tensors[i].size()
        assert input2.tensors[i].size() == input1.tensors[i].size()
    list_func = getattr(tensor_list, func_name)
    list_func(input1.tensors, input2.tensors, out.tensors)
    return out


def _comparison(func_name, func, input1, input2, out=None):
    out = _create_out(input1, out, dtype=torch.uint8)
    assert len(input1) == len(input2)
    for i in range(len(out)):
        # NOTE: We are disabling broadcasting for now
        assert out.tensors[i].size() == input1.tensors[i].size()
        assert input2.tensors[i].size() == input1.tensors[i].size()
    list_func = getattr(tensor_list, func_name)
    list_func(input1.tensors, input2.tensors, out.tensors)
    return out

class Module:
    pass

import torch
torch, NestedTensor = codegen.add_pointwise_unary_functions(torch, NestedTensor, _unary)
torch, NestedTensor = codegen.add_pointwise_binary_functions(torch, NestedTensor, _binary)
torch, NestedTensor = codegen.add_pointwise_comparison_functions(torch, NestedTensor, _comparison)
torch.nestedtensor = nested.make_nested_tensor

__all__ = ["NestedTensor"]
