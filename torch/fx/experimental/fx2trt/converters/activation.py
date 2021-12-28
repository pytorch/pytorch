import torch
import numpy as np
import tensorrt as trt
from torch.fx.experimental.fx2trt.converter_registry import tensorrt_converter

from .converter_utils import mark_as_int8_layer

def common_activation(network, mod, input_val, activation_type, activation_dyn_range_fn, layer_name):
    layer = network.add_activation(
        input=input_val, type=activation_type)
    layer.name = layer_name

    if input_val.dynamic_range:
        dyn_range = activation_dyn_range_fn(input_val.dynamic_range)
        mark_as_int8_layer(layer, dyn_range)

    return layer.get_output(0)


@tensorrt_converter(torch.nn.functional.relu)
@tensorrt_converter(torch.nn.modules.activation.ReLU)
def relu(network, submod, args, kwargs, layer_name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(f"ReLU received input {input_val} that is not part "
                           "of the TensorRT region!")

    def activation_dyn_range_fn(dyn_range):
        return max(0, dyn_range[0]), max(0, dyn_range[1])

    return common_activation(network, submod, input_val, trt.ActivationType.RELU, activation_dyn_range_fn, layer_name)


@tensorrt_converter(torch.nn.modules.activation.Sigmoid)
def sigmoid(network, submod, args, kwargs, layer_name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0
    input_val = kwargs['input']

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(f'Sigmoid received input {input_val} that is not part '
                           'of the TensorRT region!')

    def activation_dyn_range_fn(dyn_range):
        def sigmoid_fn(x):
            return 1 / (1 + np.exp(-x))
        return sigmoid_fn(dyn_range[0]), sigmoid_fn(dyn_range[1])

    return common_activation(network, submod, input_val, trt.ActivationType.SIGMOID, activation_dyn_range_fn, layer_name)
