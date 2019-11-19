from torch.onnx.symbolic_helper import parse_args
import torch.onnx.symbolic_helper as sym_help
import torch.onnx.symbolic_registry as sym_registry
import importlib
from inspect import getmembers, isfunction

def register_quantized_ops(domain, version):
    # Register all the non-quantized ops
    sym_registry.register_version('', version)
    # Register all quantized ops
    module = importlib.import_module('torch.onnx.symbolic_caffe2')
    sym_registry._symbolic_versions['caffe2'] = module
    quant_version_ops = getmembers(sym_registry._symbolic_versions['caffe2'])
    for op in quant_version_ops:
        if isfunction(op[1]) and not sym_registry.is_registered_op(op[0], domain, version):
            aten_q_ops = ['relu', '_empty_affine_quantized', 'dequantize', 'quantize_per_tensor', 'upsample_nearest2d', 'clamp', 'avg_pool2d', 'slice']
            if op[0] in aten_q_ops:
                sym_registry.register_op(op[0], op[1], '', version)
            sym_registry.register_op(op[0], op[1], domain, version)

def nchw2nhwc(g, input):
    quantized_input = input in sym_help._quantized_ops
    if quantized_input:
        quant_args = {
            "Y_scale_f": input.node()["Y_scale"],
            "Y_zero_point_i": input.node()["Y_zero_point"],
        }
        input = g.op("_caffe2::Int8Dequantize", input)
    input = g.op("_caffe2::NCHW2NHWC", input)
    if quantized_input:
        input = g.op("_caffe2::Int8Quantize", input, **quant_args)
    return input

def nhwc2nchw(g, output, scale, zero_point):
    dequant = g.op("_caffe2::Int8Dequantize", output)
    output = g.op("_caffe2::NHWC2NCHW", dequant)
    quant_args = {
        "Y_scale_f": scale,
        "Y_zero_point_i": zero_point,
    }
    output = g.op("_caffe2::Int8Quantize", output, **quant_args)
    return output

def linear_prepack(g, weight, bias):
    # Mapping to a dummy caffe2 prepack node.
    # During the onnx -> c2 conversion we can look up original weight and bias
    # from this node
    output = g.op("_caffe2::WeightPrepack", weight, bias)
    sym_help._quantized_ops.add(output)
    return output

@parse_args('v', 'v', 'v', 'f', 'i')
def linear(g, input, weight, bias, scale, zero_point):
    kwargs = {
        "Y_scale_f": scale,
        "Y_zero_point_i": zero_point,
    }
    output = g.op("_caffe2::Int8FC", input, weight, bias, **kwargs)
    sym_help._quantized_ops.add(output)
    return output

def conv_prepack(g, input, weight, bias, stride, padding, dilation, groups):
    # Mapping to a dummy caffe2 prepack node.
    # During the onnx -> c2 conversion we can look up original weight and bias
    # from this node
    output = g.op("_caffe2::WeightPrepack", input, weight, bias)
    sym_help._quantized_ops.add(output)
    return output

@parse_args('v', 'v', 'v', 'is', 'is', 'is', 'i', 'f', 'i')
def conv2d(g, input, weight, bias, stride, padding, dilation, groups, scale, zero_point):
    kernel_size = weight.node()["shape"][1:3]
    kwargs = {
        "strides_i": stride,
        "pads_i": padding + padding,
        "dilations_i": dilation,
        "group_i": groups,
        "kernels_i": kernel_size,
        "order_s": "NHWC",
        "Y_scale_f": scale,
        "Y_zero_point_i": zero_point,
    }

    input = nchw2nhwc(g, input)
    output = g.op("_caffe2::Int8Conv", input, weight, bias, **kwargs)
    output = nhwc2nchw(g, output, scale, zero_point)

    sym_help._quantized_ops.add(output)
    return output

@parse_args('v', 'v', 'v', 'is', 'is', 'is', 'i', 'f', 'i')
def conv2d_relu(g, input, weight, bias, stride, padding, dilation, groups, scale, zero_point):
    kernel_size = weight.node()["shape"][1:3]
    kwargs = {
        "strides_i": stride,
        "pads_i": padding + padding,
        "dilations_i": dilation,
        "group_i": groups,
        "kernels_i": kernel_size,
        "order_s": "NHWC",
        "Y_scale_f": scale,
        "Y_zero_point_i": zero_point,
    }

    input = nchw2nhwc(g, input)
    output = g.op("_caffe2::Int8ConvRelu", input, weight, bias, **kwargs)
    output = nhwc2nchw(g, output, scale, zero_point)

    sym_help._quantized_ops.add(output)
    return output

@parse_args('v', 'v', 'f', 'i')
def add(g, input_a, input_b, scale, zero_point):
    kwargs = {
        "Y_scale_f": scale,
        "Y_zero_point_i": zero_point,
    }
    output = g.op("_caffe2::Int8Add", input_a, input_b, **kwargs)
    sym_help._quantized_ops.add(output)
    return output

@parse_args('v', 'is')
def upsample_nearest2d(g, input, output_size):
    if input not in sym_help._quantized_ops:
        from torch.onnx.symbolic_opset9 import upsample_nearest2d
        return upsample_nearest2d(g, input, output_size)

    # FIXME hard-coded input sizes, as we cannot get input size of quantized tensor
    input_sizes = [24, 32]
    dim = 4
    scales = [1. if i < 2 else
                       float(output_size[-(dim - i)]) / float(input_sizes[-(dim - i)])
                       for i in range(0, dim)]
    kwargs = {
        "width_scale_f": scales[2],
        "height_scale_f": scales[3],
        "Y_scale_f": input.node()["Y_scale"],
        "Y_zero_point_i": input.node()["Y_zero_point"],
    }

    input = nchw2nhwc(g, input)
    output = g.op("_caffe2::Int8ResizeNearest", input, **kwargs)
    output = nhwc2nchw(g, output, input.node()["Y_scale"], input.node()["Y_zero_point"])
    sym_help._quantized_ops.add(output)
    return output

@parse_args('v')
def relu(g, input):
    if input not in sym_help._quantized_ops:
        from torch.onnx.symbolic_opset9 import relu
        return relu(g, input)
    kwargs = {
        "Y_scale_f": input.node()["Y_scale"],
        "Y_zero_point_i": input.node()["Y_zero_point"],
    }
    output = g.op("_caffe2::Int8Relu", input, **kwargs)
    sym_help._quantized_ops.add(output)
    return output

@parse_args('v', 'f', 'i', 't')
def quantize_per_tensor(g, input, scale, zero_point, dtype):
    kwargs = {
        "Y_scale_f": scale,
        "Y_zero_point_i": zero_point,
    }
    output = g.op("_caffe2::Int8Quantize", input, **kwargs)
    sym_help._quantized_ops.add(output)
    return output

@parse_args('v')
def dequantize(g, input):
    return g.op("_caffe2::Int8Dequantize", input)

@parse_args('v', 't', 't', 't', 't', 't', 't', 't')
def _empty_affine_quantized(g, input, shape, scale, zero_point, dtype, pin_memory, memory_format, layout):
    return input

# FIXME hack to convert clamp operator to caffe2.
# The op defined in sym_opset9 does not convert the args correctly
@parse_args('v', 'f', 'f')
def clamp(g, input, min, max):
    # min or max may be None that we need to dispatch to
    # Clip separately, as ONNX does not have None syntax

    kwargs = {
        "min_f": min,
        "max_f": max,
    }
    return g.op("_caffe2::Clip", input, **kwargs)

@parse_args('v', 'is', 'is', 'is', 'i', 'i', 'none')
def avg_pool2d(g, input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override=None):
    if input not in sym_help._quantized_ops:
        from torch.onnx.symbolic_opset9 import avg_pool2d
        return avg_pool2d(g, input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
    kwargs = {
        "strides_i": stride,
        "pads_i": padding + padding,
        "kernel_i": kernel_size[0],
        "order_s": "NHWC",
        "Y_scale_f": input.node()["Y_scale"],
        "Y_zero_point_i": input.node()["Y_zero_point"],
    }
    input = nchw2nhwc(g, input)
    output = g.op("_caffe2::Int8AveragePool", input, **kwargs)
    output = nhwc2nchw(g, output, input.node()["Y_scale"], input.node()["Y_zero_point"])
    sym_help._quantized_ops.add(output)
    return output

@parse_args('v', 'v', 'v', 'v', 'i')
def slice(g, input, dim, start, end, step):
    if input not in sym_help._quantized_ops:
        from torch.onnx.symbolic_opset9 import slice
        return slice(g, input, dim, start, end, step)

    start = sym_help._parse_arg(start, 'i')
    end = sym_help._parse_arg(end, 'i')
    kwargs = {
        "starts_i": start,
        "ends_i": end,
        "Y_scale_f": input.node()["Y_scale"],
        "Y_zero_point_i": input.node()["Y_zero_point"],
    }
    output = g.op("_caffe2::Int8Slice", input, **kwargs)
    sym_help._quantized_ops.add(output)
    return output
