import torch
from . import nested
import torch.tensortypes.nestedtensor.codegen as codegen

NestedTensor = nested.NestedTensor


def _nary_gen(out_dtype=None):
    # Follows signature of torch nary functions
    def _nary(func_name, func, *inputs, out=None):
        # NOTE: We are disabling broadcasting for now.
        for i in range(1, len(inputs)):
            for j in range(len(inputs[i])):
                assert inputs[0].tensors[j].size() == inputs[i].tensors[j].size()
        if out is None:
            out_tensors = []
            for i in range(len(inputs[0])):
                out_tensor = func(*list(map(lambda x: x.tensors[i], inputs)))
                if out_dtype is not None:
                    out_tensor = out_tensor.to(out_dtype)
                out_tensors.append(out_tensor)
            return NestedTensor(out_tensors)
        else:
            # NOTE: We are disabling broadcasting for now.
            for i in range(len(out)):
                assert out.tensors[i].size() == inputs[0].tensors[i].size()
            if out_dtype is not None:
                out = out.to(out_dtype)
            for i in range(len(inputs[0])):
                func(*list(map(lambda x: x.tensors[i], inputs)), out=out.tensors[i])
            return out
    return _nary


torch, NestedTensor = codegen.add_pointwise_unary_functions(torch, NestedTensor, _nary_gen())
torch, NestedTensor = codegen.add_pointwise_binary_functions(torch, NestedTensor, _nary_gen())
torch, NestedTensor = codegen.add_pointwise_comparison_functions(torch, NestedTensor, _nary_gen(torch.uint8))
torch.nestedtensor = nested.make_nested_tensor
torch.as_nestedtensor = nested.as_nestedtensor
