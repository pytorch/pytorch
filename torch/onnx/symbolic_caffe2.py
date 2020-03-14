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
            aten_q_ops = ['relu', '_empty_affine_quantized', 'dequantize',
                          'quantize_per_tensor', 'upsample_nearest2d', 'avg_pool2d', 'reshape', 'slice', 'cat', 'max_pool2d']
            if op[0] in aten_q_ops:
                sym_registry.register_op(op[0], op[1], '', version)
            sym_registry.register_op(op[0], op[1], domain, version)

def _permute_helper(g, input, axes):
    quant_args = {
        "axes_i": axes,
        "Y_scale_f": input.node()["Y_scale"],
        "Y_zero_point_i": input.node()["Y_zero_point"],
    }
    output = g.op("_caffe2::Int8Transpose", input, **quant_args)
    sym_help._quantized_ops.add(output)
    return output

def nchw2nhwc(g, input):
    axes = [0, 2, 3, 1]
    return _permute_helper(g, input, axes)

def nhwc2nchw(g, input):
    axes = [0, 3, 1, 2]
    return _permute_helper(g, input, axes)

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
    output = g.op("_caffe2::Int8Conv", input, weight, bias, **kwargs)
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
    output = g.op("_caffe2::Int8ConvRelu", input, weight, bias, **kwargs)
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

def upsample_nearest2d(g, input, output_size, align_corners=None, scales_h=None, scales_w=None):
    if input not in sym_help._quantized_ops:
        from torch.onnx.symbolic_opset9 import upsample_nearest2d as upsample_nearest2d_impl
        return upsample_nearest2d_impl(g, input, output_size, align_corners)

    output_size = sym_help._parse_arg(output_size, 'is')
    kwargs = {
        "output_size_i": output_size,
        "Y_scale_f": input.node()["Y_scale"],
        "Y_zero_point_i": input.node()["Y_zero_point"],
    }
    input = nchw2nhwc(g, input)
    output = g.op("_caffe2::Int8ResizeNearest", input, **kwargs)
    output = nhwc2nchw(g, output)
    sym_help._quantized_ops.add(output)
    return output
@parse_args('v', 'is', 'is', 'is', 'is', 'i')
def max_pool2d(g, input, kernel_size, stride, padding, dilation, ceil_mode):
    if input not in sym_help._quantized_ops:
        from torch.onnx.symbolic_opset9 import max_pool2d
        return max_pool2d(g, input, kernel_size, stride, padding, dilation, ceil_mode)
    kwargs = {
        "strides_i": stride,
        "pads_i": padding + padding,
        "kernel_i": kernel_size[0],
        "order_s": "NHWC",
        "Y_scale_f": input.node()["Y_scale"],
        "Y_zero_point_i": input.node()["Y_zero_point"],
    }
    input = nchw2nhwc(g, input)
    output = g.op("_caffe2::Int8MaxPool", input, **kwargs)
    output = nhwc2nchw(g, output)
    sym_help._quantized_ops.add(output)
    return output

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
    output = nhwc2nchw(g, output)
    sym_help._quantized_ops.add(output)
    return output

def reshape(g, input, shape):
    if input not in sym_help._quantized_ops:
        from torch.onnx.symbolic_opset9 import reshape
        return reshape(g, input, shape)

    kwargs = {
        "Y_scale_f": input.node()["Y_scale"],
        "Y_zero_point_i": input.node()["Y_zero_point"],
    }
    output = g.op("_caffe2::Int8Reshape", input, shape, **kwargs)
    sym_help._quantized_ops.add(output)
    return output

@parse_args('v', 'v', 'v', 'v', 'i')
def slice(g, input, dim, start, end, step):
    if input not in sym_help._quantized_ops:
        from torch.onnx.symbolic_opset9 import slice
        return slice(g, input, dim, start, end, step)

    if step != 1:
        raise RuntimeError("ONNX quantized slice export only works for step 1.")
    start = sym_help._parse_arg(start, 'i')
    end = sym_help._parse_arg(end, 'i')
    dim = sym_help._parse_arg(dim, 'i')

    kwargs = {
        "start_idx_i": start,
        "end_idx_i": end,
        "dim_i": dim,
        "Y_scale_f": input.node()["Y_scale"],
        "Y_zero_point_i": input.node()["Y_zero_point"],
    }
    output = g.op("_caffe2::Int8Slice", input, **kwargs)
    sym_help._quantized_ops.add(output)
    return output

def cat(g, tensor_list, dim, scale=None, zero_point=None):
    tensors = sym_help._unpack_list(tensor_list)
    input = tensors[0]
    if input not in sym_help._quantized_ops:
        from torch.onnx.symbolic_opset9 import cat
        return cat(g, tensor_list, dim)

    dim = sym_help._parse_arg(dim, 'i')
    kwargs = {
        "Y_scale_f": tensors[0].node()["Y_scale"],
        "Y_zero_point_i": tensors[0].node()["Y_zero_point"],
    }
    output = g.op("_caffe2::Int8Concat", *tensors, axis_i=dim, **kwargs)
    sym_help._quantized_ops.add(output)
    return output
