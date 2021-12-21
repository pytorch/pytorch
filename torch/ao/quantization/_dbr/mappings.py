import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.quantized as nnq
toq = torch.ops.quantized
from torch.ao.quantization.quantization_mappings import (
    DEFAULT_STATIC_QUANT_MODULE_MAPPINGS,
    DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS,
)

import operator
from typing import Callable

# TODO(future PR): reuse all of these with existing quantization mappings

fp32_to_int8_fun_mapping = {
    torch.Tensor.add: torch.ops.quantized.add,
    torch.Tensor.add_: torch.ops.quantized.add,
    torch.add: torch.ops.quantized.add,
    operator.add: torch.ops.quantized.add,
    operator.iadd: torch.ops.quantized.add,
    torch.Tensor.mul: torch.ops.quantized.mul,
    torch.mul: torch.ops.quantized.mul,
    operator.mul: torch.ops.quantized.mul,
    torch.cat: torch.ops.quantized.cat,
    F.conv2d: torch.ops.quantized.conv2d,
    F.linear: toq.linear,
}

# TODO: enforce that functions in fp32_to_int8_fun_mapping must both be
# in functions_supported_by_quantization
functions_supported_by_quantization = set([
    torch.Tensor.add,
    torch.Tensor.add_,
    torch.Tensor.mul,
    torch.add,
    torch.mul,
    torch.cat,
    # adding for MobileNetV2, will need a better place for these
    torch.nn.functional.adaptive_avg_pool2d,
    F.hardsigmoid,
    torch.flatten,
    toq.add,
    toq.mul,
    toq.cat,
    F.conv2d,
    toq.conv2d,
    F.dropout,
    torch.relu,
    F.relu,
    F.linear,
    toq.linear,
])

module_types_supported_by_quantization = set()
module_types_supported_by_quantization |= \
    set(DEFAULT_STATIC_QUANT_MODULE_MAPPINGS.keys())
module_types_supported_by_quantization |= \
    set(DEFAULT_STATIC_QUANT_MODULE_MAPPINGS.values())
module_types_supported_by_quantization |= \
    set(DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS.keys())
module_types_supported_by_quantization |= \
    set(DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS.values())
module_types_supported_by_quantization |= set([
    # these are quantizeable modules which do not need swaps
    nn.ReLU,
    nn.Dropout,
    nn.Identity,
])
module_types_supported_by_quantization -= set([
    # TODO(future PR): enable DBR quantization for embeddings
    nn.Embedding,
    nnq.Embedding,
    nn.EmbeddingBag,
    nnq.EmbeddingBag,
])


# These can work in either fp32 or quint8, without the need for observation
# TODO: better name
module_types_supported_by_quantization_preserves_dtype = set([
    nn.Identity,
    nn.Dropout,
])

functions_supported_by_quantization_preserves_dtype = set([
    F.dropout,
])

add_and_mul_ops = set([
    torch.add,
    torch.Tensor.add,
    torch.Tensor.add_,
    torch.mul,
    torch.Tensor.mul,
])

# TODO(future): reuse global mapping
known_module_fusion_patterns = [
    (torch.nn.Conv2d, torch.nn.ReLU),
    (torch.nn.Conv2d, torch.nn.BatchNorm2d),
]

binary_related_ops = (
    (torch.add, torch.Tensor.add),
    (torch.add, torch.Tensor.add_),
    (torch.Tensor.add, torch.Tensor.add_),
    (torch.mul, torch.Tensor.mul),
    (torch.mul, torch.Tensor.mul_),
    (torch.Tensor.mul, torch.Tensor.mul_),
)

# TODO(future PR): reuse global mapping
a_related_to_b = set()
for a, b in binary_related_ops:
    a_related_to_b.add((a, b))
    a_related_to_b.add((b, a))
for a, b in DEFAULT_STATIC_QUANT_MODULE_MAPPINGS.items():
    a_related_to_b.add((a, b))
    a_related_to_b.add((b, a))
for a, b in DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS.items():
    a_related_to_b.add((a, b))
    a_related_to_b.add((b, a))
for a, b in fp32_to_int8_fun_mapping.items():
    a_related_to_b.add((a, b))
    a_related_to_b.add((b, a))

def ops_are_related(
    cur_op: Callable,
    expected_op_type: Callable,
    type_is_module: bool,
) -> bool:
    # if isinstance(cur_op, torch.nn.Module):
    if type_is_module:
        cur_op = type(cur_op)
    return cur_op == expected_op_type or \
        (cur_op, expected_op_type) in a_related_to_b

# validity checks
# TODO: move these out
for m in module_types_supported_by_quantization_preserves_dtype:
    assert m in module_types_supported_by_quantization, \
        f"{m} needs to be added to module_types_supported_by_quantization"

for f in functions_supported_by_quantization_preserves_dtype:
    assert f in functions_supported_by_quantization, \
        f"{f} needs to be added to functions_supported_by_quantization"
