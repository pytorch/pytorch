from torch.onnx.symbolic_helper import parse_args
import torch.onnx.symbolic_helper as sym_help

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
    kwargs = {
        "Y_scale_f": scale,
        "Y_zero_point_i": zero_point,
    }
    output = g.op("_caffe2::Int8Conv", input, weight, bias, stride, padding, dilation, groups, **kwargs)
    sym_help._quantized_ops.add(output)
    return output

@parse_args('v', 'v', 'v', 'is', 'is', 'is', 'i', 'f', 'i')
def conv2d_relu(g, input, weight, bias, stride, padding, dilation, groups, scale, zero_point):
    kwargs = {
        "Y_scale_f": scale,
        "Y_zero_point_i": zero_point,
    }
    output = g.op("_caffe2::Int8ConvRelu", input, weight, bias, stride, padding, dilation, groups, **kwargs)
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

def upsample_nearest_2d(g, input, size, scale_factor, mode, align_corners):
    output = g.op("_caffe2::Int8ResizeNearest", input, scale_factor, scale_factor)
    sym_help._quantized_ops.add(output)
    return output

@parse_args('v')
def relu(g, input):
    output = g.op("_caffe2::Int8Relu", input)
    sym_help._quantized_ops.add(output)
    return output

@parse_args('v', 'f', 'i')
def quantize_per_tensor(g, input, scale, zero_point):
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
