import operator
import torch
import tensorrt as trt
from torch.fx.experimental.fx2trt.fx2trt import tensorrt_converter

from .helper_functions import get_dyn_range, mark_as_int8_layer

@tensorrt_converter(torch.mul)
@tensorrt_converter(operator.mul)
def mul(network, target, args, kwargs, layer_name):
    # operator.mul
    if len(kwargs) == 0:
        lhs_val, rhs_val = args
    else:
        # torch.mul
        lhs_val, rhs_val = kwargs["input"], kwargs["other"]

    if not all(isinstance(arg, trt.tensorrt.ITensor) for arg in [lhs_val, rhs_val]):
        raise RuntimeError('mul() received an input that is not part of the TensorRT region!')

    layer = network.add_elementwise(lhs_val, rhs_val, trt.ElementWiseOperation.PROD)
    layer.name = layer_name

    return layer.get_output(0)


@tensorrt_converter(torch.ops.quantized.mul)
def quantized_mul(network, target, args, kwargs, layer_name):
    assert len(args) == 0
    lhs_val, rhs_val = kwargs["qa"], kwargs["qb"]

    if not all(isinstance(i, trt.tensorrt.ITensor) for i in [lhs_val, rhs_val]):
        raise RuntimeError('Quantized mul received an input that is not part of the TensorRT region!')

    layer = network.add_elementwise(lhs_val, rhs_val, trt.ElementWiseOperation.PROD)
    layer.name = layer_name
    dyn_range = get_dyn_range(kwargs["scale"], kwargs["zero_point"], torch.quint8)
    mark_as_int8_layer(layer, dyn_range)

    return layer.get_output(0)
