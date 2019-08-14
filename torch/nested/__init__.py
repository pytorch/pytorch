import os
import torch

from . import nested
from . import codegen


# NOTE: This is inefficient! The functions that are being overwritten in torch
# are being replaced by functions with very inefficient dispatch mechanisms to add
# support for NestedTensor to torch.

def monkey_patch(module):
    module.is_nested_tensor = nested.is_nested_tensor
    module.nested_tensor = nested.nested_tensor
    module.NestedTensor = nested.NestedTensor
    module.as_nested_tensor = nested.as_nested_tensor
    module.tensor_mask_to_nested_tensor = nested.tensor_mask_to_nested_tensor

    module, nested.NestedTensor = codegen.add_pointwise_unary_functions(module, nested.NestedTensor, nested._nary_gen())
    module, nested.NestedTensor = codegen.add_pointwise_binary_functions(
        module, nested.NestedTensor, nested._nary_gen())
    module, nested.NestedTensor = codegen.add_pointwise_comparison_functions(
        module, nested.NestedTensor, nested._nary_gen(torch.uint8))

    module.nn.functional.interpolate = nested.interpolate
    module.nn.functional.conv2d = nested.conv2d
    module.nn.functional.relu = nested.relu

    module.max_pool2d = nested.max_pool2d

    return module


__all__ = ["monkey_patch"]
