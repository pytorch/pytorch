import torch
import tensorrt as trt
from torch.fx.experimental.fx2trt.converter_registry import tensorrt_converter

from .converter_utils import get_dyn_range, get_inputs_from_args_and_kwargs


quantize_per_tensor_inputs = ["input", "scale", "zero_point", "dtype"]


@tensorrt_converter("dequantize")
@tensorrt_converter(torch.dequantize)
@tensorrt_converter(torch.nn.quantized.modules.DeQuantize)
def dequantize(network, submod, args, kwargs, layer_name):
    input_val = args[0]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(f'Dequantize received input {input_val} that is not part '
                           'of the TensorRT region!')

    return input_val


@tensorrt_converter(torch.quantize_per_tensor)
@tensorrt_converter(torch.nn.quantized.modules.Quantize)
def quantize(network, submod, args, kwargs, layer_name):
    # If submod is not nn.Module then it's quantize_per_tensor
    if not isinstance(submod, torch.nn.Module):
        input_val, scale, zero_point, dtype = get_inputs_from_args_and_kwargs(args, kwargs, quantize_per_tensor_inputs)
    else:
        input_val = args[0]
        scale = submod.scale
        zero_point = submod.zero_point
        dtype = submod.dtype

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(f'Quantize received input {input_val} that is not part '
                           'of the TensorRT region!')

    if dtype != torch.quint8:
        raise RuntimeError(f"Only support torch.quint8 quantized type for activation, get {dtype}.")

    input_val.dynamic_range = get_dyn_range(scale, zero_point, dtype)
    return input_val


@tensorrt_converter(torch.nn.modules.linear.Identity)
def identity(network, submod, args, kwargs, layer_name):
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(f'Identity received input {input_val} that is not part '
                           'of the TensorRT region!')

    return input_val
