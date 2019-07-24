import os
import torch

from . import nested
from . import codegen

def _nary_gen(out_dtype=None):
    # Follows signature of torch nary functions
    def _nary(*args, **kwargs):
        func_name = args[0]
        func = args[1]
        inputs = args[2:]
        out = kwargs.get('out', None)
        # NOTE: We are disabling broadcasting for now. These checks introduce a lot of overhead.
        for i in range(1, len(inputs)):
            for j in range(len(inputs[i])):
                assert inputs[0]._tensors[j].dim() == inputs[i]._tensors[j].dim()
        if out is None:
            out_tensors = []
            for i in range(len(inputs[0])):
                out_tensor = func(*list(map(lambda x: x._tensors[i], inputs)))
                if out_dtype is not None:
                    out_tensor = out_tensor.to(out_dtype)
                out_tensors.append(out_tensor)
            return nested.NestedTensor(out_tensors)
        else:
            # NOTE: We are disabling broadcasting for now. These checks introduce a lot of overhead.
            for i in range(len(out)):
                assert out._tensors[i].dim() == inputs[0]._tensors[i].dim()
            if out_dtype is not None:
                out = out.to(out_dtype)
            if all(nested_tensor.is_contiguous() for nested_tensor in inputs):
                func(*list(map(lambda x: x.buffer_, inputs)), out=out.buffer_)
            else:
                for i in range(len(inputs[0])):
                    func(*list(map(lambda x: x._tensors[i], inputs)), out=out._tensors[i])
            return out
    return _nary


# NOTE: This is inefficient! The functions that are being overwritten in torch
# are being replaced by functions with very inefficient dispatch mechanisms to add
# support for NestedTensor to torch.
nested, nested.NestedTensor = codegen.add_pointwise_unary_functions(torch.nested, NestedTensor, _nary_gen())
nested, nested.NestedTensor = codegen.add_pointwise_binary_functions(torch.nested, NestedTensor, _nary_gen())
nested, nested.NestedTensor = codegen.add_pointwise_comparison_functions(torch.nested, NestedTensor, _nary_gen(torch.uint8))

torch.nested.nn.functional.conv2d = nested.conv2d
torch.nested.nn.functional.relu = nested.relu
torch.nested.max_pool2d = nested.max_pool2d
