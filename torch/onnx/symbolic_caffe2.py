# mypy: allow-untyped-defs
import importlib
import inspect

from torch.onnx import symbolic_helper, symbolic_opset9 as opset9
from torch.onnx._internal import jit_utils, registration


def register_quantized_ops(domain: str, version: int):
    # Register all quantized ops
    module = importlib.import_module("torch.onnx.symbolic_caffe2")
    quant_version_ops = inspect.getmembers(module)
    aten_q_ops = {
        "relu",
        "_empty_affine_quantized",
        "dequantize",
        "quantize_per_tensor",
        "upsample_nearest2d",
        "avg_pool2d",
        "reshape",
        "slice",
        "cat",
        "max_pool2d",
        "sigmoid",
    }
    for op, func in quant_version_ops:
        name = f"{domain}::{op}"
        if inspect.isfunction(func) and not registration.registry.is_registered_op(
            name, version
        ):
            if op in aten_q_ops:
                # Override the builtin aten ops
                registration.registry.register(
                    f"aten::{op}", version, func, custom=True
                )
            registration.registry.register(name, version, func)


def _permute_helper(g: jit_utils.GraphContext, input, axes):
    quant_args = {
        "axes_i": axes,
        "Y_scale_f": symbolic_helper._node_get(input.node(), "Y_scale"),
        "Y_zero_point_i": symbolic_helper._node_get(input.node(), "Y_zero_point"),
    }
    output = g.op("_caffe2::Int8Transpose", input, **quant_args)
    symbolic_helper._quantized_ops.add(output)
    return output


def nchw2nhwc(g: jit_utils.GraphContext, input):
    axes = [0, 2, 3, 1]
    return _permute_helper(g, input, axes)


def nhwc2nchw(g: jit_utils.GraphContext, input):
    axes = [0, 3, 1, 2]
    return _permute_helper(g, input, axes)


def linear_prepack(g: jit_utils.GraphContext, weight, bias):
    # Mapping to a dummy caffe2 prepack node.
    # During the onnx -> c2 conversion we can look up original weight and bias
    # from this node
    output = g.op("_caffe2::WeightPrepack", weight, bias)
    symbolic_helper._quantized_ops.add(output)
    return output


@symbolic_helper.parse_args("v", "v", "v", "f", "i")
def linear(g: jit_utils.GraphContext, input, weight, bias, scale, zero_point):
    kwargs = {
        "Y_scale_f": scale,
        "Y_zero_point_i": zero_point,
    }
    output = g.op("_caffe2::Int8FC", input, weight, bias, **kwargs)
    symbolic_helper._quantized_ops.add(output)
    return output


def conv_prepack(
    g: jit_utils.GraphContext, input, weight, bias, stride, padding, dilation, groups
):
    # Mapping to a dummy caffe2 prepack node.
    # During the onnx -> c2 conversion we can look up original weight and bias
    # from this node
    output = g.op("_caffe2::WeightPrepack", input, weight, bias)
    symbolic_helper._quantized_ops.add(output)
    return output


@symbolic_helper.parse_args("v", "v", "v", "is", "is", "is", "i", "f", "i")
def conv2d(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    scale,
    zero_point,
):
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
    symbolic_helper._quantized_ops.add(output)
    return output


@symbolic_helper.parse_args("v", "v", "v", "is", "is", "is", "i", "f", "i")
def conv2d_relu(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    scale,
    zero_point,
):
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
    symbolic_helper._quantized_ops.add(output)
    return output


@symbolic_helper.parse_args("v", "v", "f", "i")
def add(g: jit_utils.GraphContext, input_a, input_b, scale, zero_point):
    kwargs = {
        "Y_scale_f": scale,
        "Y_zero_point_i": zero_point,
    }
    output = g.op("_caffe2::Int8Add", input_a, input_b, **kwargs)
    symbolic_helper._quantized_ops.add(output)
    return output


@symbolic_helper.parse_args("v")
def relu(g: jit_utils.GraphContext, input):
    if input not in symbolic_helper._quantized_ops:
        return opset9.relu(g, input)
    kwargs = {
        "Y_scale_f": symbolic_helper._node_get(input.node(), "Y_scale"),
        "Y_zero_point_i": symbolic_helper._node_get(input.node(), "Y_zero_point"),
    }
    output = g.op("_caffe2::Int8Relu", input, **kwargs)
    symbolic_helper._quantized_ops.add(output)
    return output


@symbolic_helper.parse_args("v", "f", "i", "t")
def quantize_per_tensor(g: jit_utils.GraphContext, input, scale, zero_point, dtype):
    kwargs = {
        "Y_scale_f": scale,
        "Y_zero_point_i": zero_point,
    }
    output = g.op("_caffe2::Int8Quantize", input, **kwargs)
    symbolic_helper._quantized_ops.add(output)
    return output


@symbolic_helper.parse_args("v")
def dequantize(g: jit_utils.GraphContext, input):
    return g.op("_caffe2::Int8Dequantize", input)


@symbolic_helper.parse_args("v", "t", "t", "t", "t", "t", "t", "t")
def _empty_affine_quantized(
    g: jit_utils.GraphContext,
    input,
    shape,
    scale,
    zero_point,
    dtype,
    pin_memory,
    memory_format,
    layout,
):
    return input


def upsample_nearest2d(
    g: jit_utils.GraphContext,
    input,
    output_size,
    align_corners=None,
    scales_h=None,
    scales_w=None,
):
    if input not in symbolic_helper._quantized_ops:
        return opset9.upsample_nearest2d(g, input, output_size, align_corners)  # type: ignore[attr-defined]

    output_size = symbolic_helper._parse_arg(output_size, "is")
    kwargs = {
        "output_size_i": output_size,
        "Y_scale_f": symbolic_helper._node_get(input.node(), "Y_scale"),
        "Y_zero_point_i": symbolic_helper._node_get(input.node(), "Y_zero_point"),
    }
    input = nchw2nhwc(g, input)
    output = g.op("_caffe2::Int8ResizeNearest", input, **kwargs)
    output = nhwc2nchw(g, output)
    symbolic_helper._quantized_ops.add(output)
    return output


@symbolic_helper.parse_args("v", "is", "is", "is", "is", "i")
def max_pool2d(
    g: jit_utils.GraphContext,
    input,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
):
    if input not in symbolic_helper._quantized_ops:
        return opset9.max_pool2d(  # type: ignore[attr-defined]
            g, input, kernel_size, stride, padding, dilation, ceil_mode
        )
    kwargs = {
        "strides_i": stride,
        "pads_i": padding + padding,
        "kernel_i": kernel_size[0],
        "order_s": "NHWC",
        "Y_scale_f": symbolic_helper._node_get(input.node(), "Y_scale"),
        "Y_zero_point_i": symbolic_helper._node_get(input.node(), "Y_zero_point"),
    }
    input = nchw2nhwc(g, input)
    output = g.op("_caffe2::Int8MaxPool", input, **kwargs)
    output = nhwc2nchw(g, output)
    symbolic_helper._quantized_ops.add(output)
    return output


@symbolic_helper.parse_args("v", "is", "is", "is", "i", "i", "none")
def avg_pool2d(
    g: jit_utils.GraphContext,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override=None,
):
    if input not in symbolic_helper._quantized_ops:
        return opset9.avg_pool2d(  # type: ignore[attr-defined]
            g,
            input,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )
    kwargs = {
        "strides_i": stride,
        "pads_i": padding + padding,
        "kernel_i": kernel_size[0],
        "order_s": "NHWC",
        "Y_scale_f": symbolic_helper._node_get(input.node(), "Y_scale"),
        "Y_zero_point_i": symbolic_helper._node_get(input.node(), "Y_zero_point"),
    }
    input = nchw2nhwc(g, input)
    output = g.op("_caffe2::Int8AveragePool", input, **kwargs)
    output = nhwc2nchw(g, output)
    symbolic_helper._quantized_ops.add(output)
    return output


def reshape(g: jit_utils.GraphContext, input, shape):
    if input not in symbolic_helper._quantized_ops:
        return opset9.reshape(g, input, shape)

    kwargs = {
        "Y_scale_f": symbolic_helper._node_get(input.node(), "Y_scale"),
        "Y_zero_point_i": symbolic_helper._node_get(input.node(), "Y_zero_point"),
    }
    output = g.op("_caffe2::Int8Reshape", input, shape, **kwargs)
    symbolic_helper._quantized_ops.add(output)
    return output


@symbolic_helper.parse_args("v", "v", "v", "v", "i")
def slice(g: jit_utils.GraphContext, input, dim, start, end, step):
    if input not in symbolic_helper._quantized_ops:
        return opset9.slice(g, input, dim, start, end, step)

    if step != 1:
        raise RuntimeError("ONNX quantized slice export only works for step 1.")
    start = symbolic_helper._parse_arg(start, "i")
    end = symbolic_helper._parse_arg(end, "i")
    dim = symbolic_helper._parse_arg(dim, "i")

    kwargs = {
        "start_idx_i": start,
        "end_idx_i": end,
        "dim_i": dim,
        "Y_scale_f": symbolic_helper._node_get(input.node(), "Y_scale"),
        "Y_zero_point_i": symbolic_helper._node_get(input.node(), "Y_zero_point"),
    }
    output = g.op("_caffe2::Int8Slice", input, **kwargs)
    symbolic_helper._quantized_ops.add(output)
    return output


def cat(g: jit_utils.GraphContext, tensor_list, dim, scale=None, zero_point=None):
    tensors = symbolic_helper._unpack_list(tensor_list)
    input = tensors[0]
    if input not in symbolic_helper._quantized_ops:
        return opset9.cat(g, tensor_list, dim)

    dim = symbolic_helper._parse_arg(dim, "i")
    kwargs = {
        "Y_scale_f": tensors[0].node()["Y_scale"],
        "Y_zero_point_i": tensors[0].node()["Y_zero_point"],
    }
    output = g.op("_caffe2::Int8Concat", *tensors, axis_i=dim, **kwargs)
    symbolic_helper._quantized_ops.add(output)
    return output


@symbolic_helper.parse_args("v")
def sigmoid(g: jit_utils.GraphContext, input):
    if input not in symbolic_helper._quantized_ops:
        return opset9.sigmoid(g, input)
    # Caffe2 expects the output scale to be 1/2^8
    # and output zero_point to be 0 (quint8 type)
    out_scale = 1.0 / 256
    zero_point = 0
    kwargs = {
        "Y_scale_f": out_scale,
        "Y_zero_point_i": zero_point,
    }
    output = g.op("_caffe2::Int8Sigmoid", input, **kwargs)
    symbolic_helper._quantized_ops.add(output)
    return output
