import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.quantized as nnq
import torch.nn.intrinsic as nni
import torch.nn.intrinsic.quantized as nniq
toq = torch.ops.quantized

import operator

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
    torch.flatten,
    toq.add,
    toq.mul,
    toq.cat,
    F.conv2d,
    F.dropout,
])

module_types_supported_by_quantization = set([
    nn.Conv2d,
    nnq.Conv2d,
    nn.intrinsic.modules.fused.ConvReLU2d,
    nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d,
    nn.BatchNorm2d,
    nn.ReLU,
    # TODO(future PR): detect inplace modifications by torch functions
    nn.ReLU6,
    nn.Linear,
    nnq.Linear,
    nn.Dropout,
    nn.Identity,
    nn.LeakyReLU,
    nn.LayerNorm,
    nnq.LayerNorm,
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

# TODO(future PR): reuse existing mapping
q_mod_to_float_mod_mapping = {
    nnq.Conv2d: nn.Conv2d,
    nniq.ConvReLU2d: nni.ConvReLU2d,
    nnq.ReLU6: nn.ReLU6,
    nnq.Linear: nn.Linear,
    nnq.LeakyReLU: nn.LeakyReLU,
    nnq.LayerNorm: nn.LayerNorm,
}

add_and_mul_ops = set([
    torch.add,
    torch.Tensor.add,
    torch.Tensor.add_,
    torch.mul,
    torch.Tensor.mul,
])

# validity checks
# TODO: move these out
for m in module_types_supported_by_quantization_preserves_dtype:
    assert m in module_types_supported_by_quantization, \
        f"{m} needs to be added to module_types_supported_by_quantization"

for f in functions_supported_by_quantization_preserves_dtype:
    assert f in functions_supported_by_quantization, \
        f"{f} needs to be added to functions_supported_by_quantization"
