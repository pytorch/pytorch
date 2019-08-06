import os
import torch

from . import nested
from . import codegen

def _nested_apply_return(f):
    def decorator(self, *args, **kwargs):
        if not(torch.is_tensor(self) or is_nested_tensor(self)):
            raise ValueError("First argument must be Tensor or NestedTensor")
        if self.nested_dim == 1:
            return f(self, *args, **kwargs)
        else:
            components = self.unbind()
            if not all(components[0].nested_dim == component.nested_dim for component in components):
                raise ValueError("All NestedTensors must have the same nested dimension")
            return NestedTensor([f(component, *args, **kwargs) for component in components])
    return decorator

def _nary_gen(out_dtype=None):
    # Follows signature of torch nary functions
    def _nary(*args, **kwargs):
        func_name = args[0]
        func = args[1]
        inputs = args[2:]
        out = kwargs.get('out', None)

        def _nary_tensors(inputs, out):
            # NOTE: We are disabling broadcasting for now. These checks introduce a lot of overhead.
            # for i in range(1, len(inputs)):
            #     for j in range(len(inputs[i])):
            #         assert inputs[0]._tensors[j].dim() == inputs[i]._tensors[j].dim()
            if out is None:
                out_tensors = []
                for i in range(len(inputs[0])):
                    inp_tensors = []
                    for x in inputs:
                        inp_tensors.append(x._tensors[i])
                    out_tensor = func(*inp_tensors)
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

        if torch.is_tensor(inputs[0]):
            return _nary_tensors(inputs, out)
        else:
            unbound_inputs = [inp.unbind() for inp in inputs]
            unbound_out = out.unbind()
            result = []
            for i in range(len(inputs[0])):
                result.append(_nary_tensors(*([unbound_input[i] for unbound_input in unbound_inputs] + [out[i]])))
            return nested.NestedTensor(result)

    return _nary


# NOTE: This is inefficient! The functions that are being overwritten in torch
# are being replaced by functions with very inefficient dispatch mechanisms to add
# support for NestedTensor to torch.

def monkey_patch(module):
    module.is_nested_tensor = nested.is_nested_tensor
    module.nested_tensor = nested.nested_tensor
    module.NestedTensor = nested.NestedTensor
    module.as_nested_tensor = nested.as_nested_tensor

    module, nested.NestedTensor = codegen.add_pointwise_unary_functions(module, nested.NestedTensor, _nary_gen())
    module, nested.NestedTensor = codegen.add_pointwise_binary_functions(module, nested.NestedTensor, _nary_gen())
    module, nested.NestedTensor = codegen.add_pointwise_comparison_functions(module, nested.NestedTensor, _nary_gen(torch.uint8))

    module.nn.functional.interpolate = nested.interpolate
    module.nn.functional.conv2d = nested.conv2d
    module.nn.functional.relu = nested.relu

    module.max_pool2d = nested.max_pool2d

    return module

__all__ = ["monkey_patch"]
