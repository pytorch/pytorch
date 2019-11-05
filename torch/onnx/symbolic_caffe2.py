import importlib
import torch.onnx.symbolic_registry as sym_registry
from inspect import getmembers, isfunction
from torch.onnx.symbolic_helper import parse_args

def register_quantized_ops(domain, version):
    # Register all the non-quantized ops
    sym_registry.register_version('', version)
    # Register all quantized ops
    module = importlib.import_module('torch.onnx.symbolic_caffe2')
    sym_registry._symbolic_versions['caffe2'] = module
    quant_version_ops = getmembers(sym_registry._symbolic_versions['caffe2'])
    for op in quant_version_ops:
        if isfunction(op[1]) and not sym_registry.is_registered_op(op[0], domain, version):
            sym_registry.register_op(op[0], op[1], domain, version)

def linear_prepack(g, input, weight):
    return input

def linear(g, input, weight, scale, zero_point):
    return g.op("_caffe2::Int8FC", input, weight, scale, zero_point)

def conv_prepack(g, input, weight, stride, padding, dilation, groups):
    return input

def conv2d(g, input, weight, stride, padding, dilation, groups, scale, zero_point):
    return g.op("_caffe2::Int8Conv", input, weight, stride, padding, dilation, groups, scale, zero_point)

def conv2d_relu(g, input, weight, stride, padding, dilation, groups, scale, zero_point):
    return g.op("_caffe2::Int8ConvRelu", input, weight, stride, padding, dilation, groups, scale, zero_point)

@parse_args('v', 'v', 'f', 'i')
def add(g, input_a, input_b, scale, zero_point):
    kwargs = {
        "Y_scale_f": scale,
        "Y_zero_point_i": zero_point,
    }
    return g.op("_caffe2::Int8Add", input_a, input_b, **kwargs)

def upsample_nearest_2d(g, input, size, scale_factor, mode, align_corners):
    return g.op("_caffe2::Int8ResizeNearest", input, scale_factor, scale_factor)

def relu(g, input, scale, zero_point):
    return g.op("_caffe2::Int8Relu", input, scale, zero_point)
