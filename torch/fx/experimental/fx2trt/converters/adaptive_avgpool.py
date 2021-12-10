import torch
import tensorrt as trt
from torch.fx.experimental.fx2trt.converter_registry import tensorrt_converter

from .converter_utils import mark_as_int8_layer, extend_mod_attr_to_tuple

@tensorrt_converter(torch.nn.modules.pooling.AdaptiveAvgPool2d)
def adaptive_avgpool2d(network, submod, args, kwargs, name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(f"AdaptiveAvgPool2d received input {input_val} that is not part "
                           "of the TensorRT region!")

    output_size = extend_mod_attr_to_tuple(submod, "output_size", 2)
    stride = (input_val.shape[-2] // output_size[-2], input_val.shape[-1] // output_size[-1])
    kernel_size = stride
    layer = network.add_pooling(
        input=input_val, type=trt.PoolingType.AVERAGE, window_size=kernel_size)
    layer.stride = stride
    layer.name = name

    if input_val.dynamic_range:
        mark_as_int8_layer(layer, input_val.dynamic_range)

    return layer.get_output(0)
