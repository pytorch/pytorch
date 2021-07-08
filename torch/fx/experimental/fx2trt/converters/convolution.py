import torch
import tensorrt as trt
from torch.fx.experimental.fx2trt.fx2trt import tensorrt_converter

from .helper_functions import extend_attr_to_tuple, mark_as_int8_layer, to_numpy, get_dyn_range

def common_conv(network, mod, dimension, input_val, layer_name, is_quantized):
    if mod.padding_mode != "zeros":
        raise RuntimeError(f"Only support padding mode: zeros, got {mod.padding_mode}.")

    kernel_size = extend_attr_to_tuple(mod, "kernel_size", dimension)
    stride = extend_attr_to_tuple(mod, "stride", dimension)
    padding = extend_attr_to_tuple(mod, "padding", dimension)
    dilation = extend_attr_to_tuple(mod, "dilation", dimension)

    kernel = to_numpy(mod.weight() if is_quantized else mod.weight)
    bias = to_numpy(mod.bias() if is_quantized else mod.bias)

    layer = network.add_convolution(
        input=input_val,
        num_output_maps=mod.out_channels,
        kernel_shape=kernel_size,
        kernel=kernel,
        bias=bias,
    )
    layer.name = layer_name
    layer.stride = stride
    layer.padding = padding
    layer.dilation = dilation
    layer.num_groups = mod.groups

    if is_quantized:
        # Assume the dtype of activation is torch.quint8
        mark_as_int8_layer(layer, get_dyn_range(mod.scale, mod.zero_point, torch.quint8))

    return layer.get_output(0)


def common_conv_relu(network, mod, dimension, input_val, layer_name, is_quantized):
    conv_output = common_conv(
        network,
        mod,
        dimension=2,
        input_val=input_val,
        layer_name=f"{layer_name}_conv",
        is_quantized=is_quantized,
    )

    layer = network.add_activation(
        input=conv_output, type=trt.ActivationType.RELU)
    layer.name = f"{layer_name}_relu"

    if is_quantized:
        mark_as_int8_layer(layer, conv_output.dynamic_range)

    return layer.get_output(0)


@tensorrt_converter(torch.nn.modules.conv.Conv2d)
def conv2d(network, submod, args, kwargs, layer_name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(f"Conv2d received input {input_val} that is not part "
                           "of the TensorRT region!")

    return common_conv(network, submod, dimension=2, input_val=input_val, layer_name=layer_name, is_quantized=False)


@tensorrt_converter(torch.nn.quantized.modules.conv.Conv2d)
def quantized_conv2d(network, submod, args, kwargs, layer_name):
    input_val = args[0]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(f'Quantized Conv2d received input {input_val} that is not part '
                           'of the TensorRT region!')

    return common_conv(network, submod, dimension=2, input_val=input_val, layer_name=layer_name, is_quantized=True)


@tensorrt_converter(torch.nn.intrinsic.quantized.modules.ConvReLU2d)
def quantized_conv_relu2d(network, submod, args, kwargs, layer_name):
    input_val = args[0]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(f'Quantized ConvReLU2d received input {input_val} that is not part '
                           'of the TensorRT region!')

    return common_conv_relu(network, submod, dimension=2, input_val=input_val, layer_name=f"{layer_name}_conv", is_quantized=True)
