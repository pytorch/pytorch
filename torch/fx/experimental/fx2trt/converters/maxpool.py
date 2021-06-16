import torch
import tensorrt as trt
from torch.fx.experimental.fx2trt.fx2trt import tensorrt_converter

from .helper_functions import mark_as_int8_layer, extend_attr_to_tuple

def common_maxpool(network, mod, dimension, input_val, layer_name):
    kernel_size = extend_attr_to_tuple(mod, "kernel_size", dimension)
    stride = extend_attr_to_tuple(mod, "stride", dimension)
    padding = extend_attr_to_tuple(mod, "padding", dimension)

    layer = network.add_pooling(
        input=input_val, type=trt.PoolingType.MAX, window_size=kernel_size)

    layer.stride = stride
    layer.padding = padding
    layer.name = layer_name

    if mod.ceil_mode:
        layer.padding_mode = trt.PaddingMode.EXPLICIT_ROUND_UP

    if input_val.dynamic_range:
        mark_as_int8_layer(layer, input_val.dynamic_range)

    return layer.get_output(0)


@tensorrt_converter(torch.nn.modules.pooling.MaxPool2d)
def maxpool2d(network, submod, args, kwargs, layer_name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(f"MaxPool2d received input {input_val} that is not part "
                           "of the TensorRT region!")

    return common_maxpool(network, submod, dimension=2, input_val=input_val, layer_name=layer_name)
