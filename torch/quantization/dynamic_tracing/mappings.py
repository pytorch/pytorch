import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.nn.intrinsic as nni
import torch.nn.intrinsic.quantized as nniq
import operator

# TODO(future PR): reuse all of these with existing quantization mappings

fp32_to_int8_fun_mapping = {
    torch.Tensor.add: torch.ops.quantized.add,
    torch.add: torch.ops.quantized.add,
    operator.add: torch.ops.quantized.add,
    torch.Tensor.mul: torch.ops.quantized.mul,
    torch.mul: torch.ops.quantized.mul,
    operator.mul: torch.ops.quantized.mul,
    torch.cat: torch.ops.quantized.cat,
}

functions_supported_by_quantization = set([
    torch.Tensor.add, torch.Tensor.mul, torch.add, torch.mul, torch.cat,
])

module_types_supported_by_quantization = set([
    nn.Conv2d,
    nnq.Conv2d,
    nn.intrinsic.modules.fused.ConvReLU2d,
    nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d,
    nn.ReLU,
])

# TODO(future PR): reuse existing mapping
q_mod_to_float_mod_mapping = {
    nnq.Conv2d: nn.Conv2d,
    nniq.ConvReLU2d: nni.ConvReLU2d,
}
