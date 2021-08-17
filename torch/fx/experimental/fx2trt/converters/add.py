import operator
import torch
import tensorrt as trt
from torch.fx.experimental.fx2trt.fx2trt import tensorrt_converter

from .helper_functions import get_dyn_range, mark_as_int8_layer

@tensorrt_converter(operator.add)
@tensorrt_converter(torch.add)
def add(network, target, args, kwargs, layer_name):
    # operator.add
    if len(kwargs) == 0:
        lhs_val, rhs_val = args
    else:
        # torch.add
        lhs_val, rhs_val = kwargs["input"], kwargs["other"]
        assert kwargs["alpha"] == 1

    if not all(isinstance(arg, trt.tensorrt.ITensor) for arg in [lhs_val, rhs_val]):
        raise RuntimeError("add() received an input that is not part of the TensorRT region!")

    layer = network.add_elementwise(lhs_val, rhs_val, trt.ElementWiseOperation.SUM)
    layer.name = layer_name

    return layer.get_output(0)


@tensorrt_converter(torch.ops.quantized.add)
def quantized_add(network, target, args, kwargs, layer_name):
    lhs_val, rhs_val = kwargs["qa"], kwargs["qb"]

    if not all(isinstance(i, trt.tensorrt.ITensor) for i in [lhs_val, rhs_val]):
        raise RuntimeError('Quantized add received an input that is not part of the TensorRT region!')

    layer = network.add_elementwise(lhs_val, rhs_val, trt.ElementWiseOperation.SUM)
    layer.name = layer_name
    dyn_range = get_dyn_range(kwargs["scale"], kwargs["zero_point"], torch.quint8)
    mark_as_int8_layer(layer, dyn_range)

    return layer.get_output(0)


@tensorrt_converter(torch.ops.quantized.add_relu)
def quantized_add_relu(network, submod, args, kwargs, layer_name):
    lhs_val, rhs_val = kwargs["qa"], kwargs["qb"]

    if not all(isinstance(i, trt.tensorrt.ITensor) for i in [lhs_val, rhs_val]):
        raise RuntimeError('Quantized add_relu received an input that is not part of the TensorRT region!')

    layer = network.add_elementwise(lhs_val, rhs_val, trt.ElementWiseOperation.SUM)
    layer.name = f"{layer_name}_add"
    dyn_range = get_dyn_range(kwargs["scale"], kwargs["zero_point"], torch.quint8)
    mark_as_int8_layer(layer, dyn_range)

    layer = network.add_activation(
        input=layer.get_output(0), type=trt.ActivationType.RELU)
    layer.name = f"{layer_name}_relu"
    mark_as_int8_layer(layer, dyn_range)

    return layer.get_output(0)
