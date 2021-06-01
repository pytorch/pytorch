import operator
import torch
import tensorrt as trt
import numpy as np

from torch.fx.experimental.fx2trt.fx2trt import tensorrt_converter, torch_dtype_to_trt


def process_attr(submod: torch.nn.Module, name: str, size: int):
    val = getattr(submod, name)
    if not isinstance(val, tuple):
        val = (val,) * size
    return val


def to_numpy(tensor):
    if tensor.is_quantized:
        tensor = tensor.dequantize()
    return tensor.cpu().detach().contiguous().numpy()


def get_dyn_range(scale, zero_point, dtype):
    """
    Get the dynamic range of a tensor based on its scale, zero_point and dtype.
    """
    if dtype == torch.quint8:
        min_val, max_val = 0, 255
    elif dtype == torch.qint8:
        min_val, max_val = -128, 127
    else:
        raise RuntimeError(f"Unsupported quantized dtype {dtype}")

    return (min_val - zero_point) * scale, (max_val - zero_point) * scale


def mark_as_int8_layer(layer, dynamic_range):
    """
    Set the precision of a layer to int8 as well as the type of its first output.
    Also set the dynamic range of its first output.
    """
    if layer.type not in {trt.LayerType.SHUFFLE, trt.LayerType.CONCATENATION, trt.LayerType.CONSTANT, trt.LayerType.SHAPE}:
        layer.precision = trt.int8

    for i in range(layer.num_outputs):
        # set_output_type is different from output_val.dtype
        # https://fburl.com/5k49sl9g
        layer.set_output_type(i, trt.int8)
        output_val = layer.get_output(i)
        output_val.dynamic_range = dynamic_range
        # output_val.dtype = trt.int8


@tensorrt_converter(torch.nn.modules.conv.Conv2d)
def torch_nn_modules_conv_Conv2d(network, submod, args, kwargs, name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(f"Conv2d received input {input_val} that is not part "
                           "of the TensorRT region!")

    kernel_size = process_attr(submod, "kernel_size", 2)
    stride = process_attr(submod, "stride", 2)
    padding = process_attr(submod, "padding", 2)
    dilation = process_attr(submod, "dilation", 2)

    kernel = to_numpy(submod.weight)

    bias = trt.Weights(torch_dtype_to_trt(submod.weight.dtype))
    if submod.bias is not None:
        bias = to_numpy(submod.bias)

    layer = network.add_convolution(
        input=input_val,
        num_output_maps=submod.out_channels,
        kernel_shape=kernel_size,
        kernel=kernel,
        bias=bias,
    )
    layer.name = name
    layer.stride = stride
    layer.padding = padding
    layer.dilation = dilation

    if submod.groups is not None:
        layer.num_groups = submod.groups

    return layer.get_output(0)


@tensorrt_converter(torch.nn.modules.batchnorm.BatchNorm2d)
def torch_nn_modules_batchnorm_BatchNorm2d(network, submod, args, kwargs, name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(f"BatchNorm2d received input {input_val} that is not part "
                           "of the TensorRT region!")

    scale = to_numpy(submod.weight) / np.sqrt(
        to_numpy(submod.running_var) + submod.eps
    )
    bias = (
        to_numpy(submod.bias)
        - to_numpy(submod.running_mean) * scale
    )
    power = np.ones_like(scale)

    layer = network.add_scale(input_val, trt.ScaleMode.CHANNEL, bias, scale, power)
    layer.name = name
    return layer.get_output(0)


@tensorrt_converter(torch.nn.functional.relu)
@tensorrt_converter(torch.nn.modules.activation.ReLU)
def torch_nn_modules_activation_ReLU(network, submod, args, kwargs, name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(f"ReLU received input {input_val} that is not part "
                           "of the TensorRT region!")

    layer = network.add_activation(
        input=input_val, type=trt.ActivationType.RELU)
    layer.name = name

    if input_val.dynamic_range:
        dyn_range = (max(0, input_val.dynamic_range[0]), max(0, input_val.dynamic_range[1]))
        mark_as_int8_layer(layer, dyn_range)

    return layer.get_output(0)


@tensorrt_converter(torch.nn.modules.activation.Sigmoid)
def torch_nn_modules_activation_Sigmoid(network, submod, args, kwargs, name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0
    input_val = kwargs['input']

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(f'Sigmoid received input {input_val} that is not part '
                           'of the TensorRT region!')

    layer = network.add_activation(
        input=input_val, type=trt.ActivationType.SIGMOID)
    layer.name = name

    if input_val.dynamic_range:
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        min_range, max_range = input_val.dynamic_range
        dyn_range = (sigmoid(min_range), sigmoid(max_range))
        mark_as_int8_layer(layer, dyn_range)

    return layer.get_output(0)


@tensorrt_converter(torch.nn.modules.pooling.MaxPool2d)
def torch_nn_modules_pooling_MaxPool2d(network, submod, args, kwargs, name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(f"MaxPool2d received input {input_val} that is not part "
                           "of the TensorRT region!")

    kernel_size = process_attr(submod, "kernel_size", 2)
    stride = process_attr(submod, "stride", 2)
    padding = process_attr(submod, "padding", 2)
    ceil_mode = submod.ceil_mode

    layer = network.add_pooling(
        input=input_val, type=trt.PoolingType.MAX, window_size=kernel_size)

    layer.stride = stride
    layer.padding = padding
    layer.name = name

    if ceil_mode:
        layer.padding_mode = trt.PaddingMode.EXPLICIT_ROUND_UP

    if input_val.dynamic_range:
        mark_as_int8_layer(layer, input_val.dynamic_range)

    return layer.get_output(0)


def elementwise_binary_op(network, lhs_val, rhs_val, name, op_type, dynamic_range=None):
    # TODO: broadcast
    # https://github.com/NVIDIA-AI-IOT/torch2trt/blob/44977a94cb087fe521421802e9df12a5ac3ceb3f/torch2trt/torch2trt.py#L168
    layer = network.add_elementwise(lhs_val, rhs_val, op_type)
    layer.name = name

    if dynamic_range:
        mark_as_int8_layer(layer, dynamic_range)

    return layer.get_output(0)


@tensorrt_converter(operator.add)
@tensorrt_converter(torch.add)
def torch_add(network, target, args, kwargs, name):
    if len(kwargs) != 0:
        raise RuntimeError("`out` parameter on torch.add not supported!")

    assert len(args) == 2
    if not all(isinstance(arg, trt.tensorrt.ITensor) for arg in args):
        raise RuntimeError("add() received an input that is not part of the TensorRT region!")

    lhs_val, rhs_val = args
    return elementwise_binary_op(
        network,
        lhs_val,
        rhs_val,
        name,
        trt.ElementWiseOperation.SUM,
    )


@tensorrt_converter(torch.ops.quantized.add)
def torch_ops_quantized_add(network, target, args, kwargs, name):
    assert len(args) == 0
    lhs_val, rhs_val = kwargs["qa"], kwargs["qb"]

    if not all(isinstance(i, trt.tensorrt.ITensor) for i in [lhs_val, rhs_val]):
        raise RuntimeError('Quantized add received an input that is not part of the TensorRT region!')

    return elementwise_binary_op(
        network,
        lhs_val,
        rhs_val,
        name,
        trt.ElementWiseOperation.SUM,
        get_dyn_range(kwargs["scale"], kwargs["zero_point"], torch.quint8)
    )


@tensorrt_converter(torch.mul)
@tensorrt_converter(operator.mul)
def torch_mul(network, target, args, kwargs, name):
    assert len(args) == 2
    if not all(isinstance(arg, trt.tensorrt.ITensor) for arg in args):
        raise RuntimeError('mul() received an input that is not part of the TensorRT region!')

    lhs_val, rhs_val = args
    return elementwise_binary_op(
        network,
        lhs_val,
        rhs_val,
        name,
        trt.ElementWiseOperation.PROD,
    )


@tensorrt_converter(torch.ops.quantized.mul)
def torch_ops_quantized_mul(network, target, args, kwargs, name):
    assert len(args) == 0
    lhs_val, rhs_val = kwargs["qa"], kwargs["qb"]

    if not all(isinstance(i, trt.tensorrt.ITensor) for i in [lhs_val, rhs_val]):
        raise RuntimeError('Quantized mul received an input that is not part of the TensorRT region!')

    return elementwise_binary_op(
        network,
        lhs_val,
        rhs_val,
        name,
        trt.ElementWiseOperation.PROD,
        get_dyn_range(kwargs["scale"], kwargs["zero_point"], torch.quint8)
    )


@tensorrt_converter(torch.nn.modules.pooling.AdaptiveAvgPool2d)
def torch_nn_modules_pooling_AdaptiveAvgPool2d(network, submod, args, kwargs, name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(f"AdaptiveAvgPool2d received input {input_val} that is not part "
                           "of the TensorRT region!")

    output_size = process_attr(submod, "output_size", 2)
    stride = (input_val.shape[-2] // output_size[-2], input_val.shape[-1] // output_size[-1])
    kernel_size = stride
    layer = network.add_pooling(
        input=input_val, type=trt.PoolingType.AVERAGE, window_size=kernel_size)
    layer.stride = stride
    layer.name = name

    if input_val.dynamic_range:
        mark_as_int8_layer(layer, input_val.dynamic_range)

    return layer.get_output(0)


@tensorrt_converter(torch.flatten)
def torch_flatten(network, target, args, kwargs, name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(f"Flatten received input {input_val} that is not part "
                           "of the TensorRT region!")

    # For trt shape we don"t have batch dim
    start_dim = kwargs["start_dim"] - 1
    end_dim = len(input_val.shape) if kwargs["end_dim"] == -1 else kwargs["end_dim"] - 1

    assert start_dim >= 0, "Expect non negtive start_dim, this probably due to flatten batch dim."

    new_shape = []
    flatten_dim = 1
    for i, dim in enumerate(input_val.shape):
        if i < start_dim:
            new_shape.append(dim)
        elif i > end_dim:
            new_shape.append(flatten_dim)
            new_shape.append(dim)
        else:
            flatten_dim *= dim

    if end_dim == len(input_val.shape):
        new_shape.append(flatten_dim)

    layer = network.add_shuffle(input_val)
    layer.reshape_dims = tuple(new_shape)
    layer.name = name

    if input_val.dynamic_range:
        mark_as_int8_layer(layer, input_val.dynamic_range)

    return layer.get_output(0)


@tensorrt_converter(torch.nn.modules.linear.Linear)
def torch_nn_modules_linear_Linear(network, submod, args, kwargs, name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(f"Linear received input {input_val} that is not part "
                           "of the TensorRT region!")

    layer = network.add_shuffle(input_val)
    layer.reshape_dims = tuple(input_val.shape) + (1, 1)
    layer.name = f"{name}_pre_shuffle"

    bias = trt.Weights(torch_dtype_to_trt(submod.weight.dtype))
    if submod.bias is not None:
        bias = to_numpy(submod.bias)

    # add fully connected
    layer = network.add_fully_connected(
        input=layer.get_output(0),
        num_outputs=submod.out_features,
        kernel=to_numpy(submod.weight),
        bias=bias
    )
    layer.name = f"{name}_linear"

    # reshape back
    layer = network.add_shuffle(layer.get_output(0))
    layer.reshape_dims = tuple(input_val.shape[:-1]) + (submod.out_features,)
    layer.name = f"{name}_post_shuffle"
    return layer.get_output(0)


@tensorrt_converter(torch.quantize_per_tensor)
def torch_quanitze_per_tensor(network, submod, args, kwargs, name):
    input_val = args[0]
    scale = args[1].item()
    zero_point = args[2].item()
    dtype = args[3]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(f'quantize_per_tensor received input {input_val} that is not part '
                           'of the TensorRT region!')

    input_val.dynamic_range = get_dyn_range(scale, zero_point, dtype)
    return input_val


@tensorrt_converter(torch.nn.quantized.modules.Quantize)
def torch_nn_quantized_modules_modules_Quantize(network, submod, args, kwargs, name):
    input_val = args[0]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(f'Quantize received input {input_val} that is not part '
                           'of the TensorRT region!')

    input_val.dynamic_range = get_dyn_range(submod.scale, submod.zero_point, submod.dtype)
    return input_val


@tensorrt_converter(torch.nn.quantized.modules.linear.Linear)
def torch_nn_quantized_modules_linear_Linear(network, submod, args, kwargs, name):
    input_val = args[0]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(f'Quantized Linear received input {input_val} that is not part '
                           'of the TensorRT region!')

    layer = network.add_shuffle(input_val)
    layer.reshape_dims = tuple(input_val.shape) + (1, 1)
    layer.name = f'{name}_pre_shuffle'
    mark_as_int8_layer(layer, input_val.dynamic_range)

    bias = None
    if submod.bias() is not None:
        bias = to_numpy(submod.bias())

    # add fully connected
    layer = network.add_fully_connected(
        input=layer.get_output(0),
        num_outputs=submod.out_features,
        kernel=to_numpy(submod.weight()),
        bias=bias
    )
    layer.name = f'{name}_linear'
    dyn_range = get_dyn_range(submod.scale, submod.zero_point, torch.quint8)
    mark_as_int8_layer(layer, dyn_range)

    # reshape back to N
    layer = network.add_shuffle(layer.get_output(0))
    layer.reshape_dims = tuple(input_val.shape[:-1]) + (submod.out_features,)
    layer.name = f'{name}_post_shuffle'
    mark_as_int8_layer(layer, dyn_range)

    return layer.get_output(0)


@tensorrt_converter(torch.nn.quantized.modules.conv.Conv2d)
def torch_nn_quantized_modules_conv_Conv2d(network, submod, args, kwargs, name):
    input_val = args[0]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(f'Quantized Conv2d received input {input_val} that is not part '
                           'of the TensorRT region!')

    kernel_size = process_attr(submod, 'kernel_size', 2)
    stride = process_attr(submod, 'stride', 2)
    padding = process_attr(submod, 'padding', 2)
    dilation = process_attr(submod, 'dilation', 2)

    kernel = to_numpy(submod.weight())

    bias = None
    if submod.bias() is not None:
        bias = to_numpy(submod.bias())

    layer = network.add_convolution(
        input=input_val,
        num_output_maps=submod.out_channels,
        kernel_shape=kernel_size,
        kernel=kernel,
        bias=bias,
    )
    layer.name = name
    layer.stride = stride
    layer.padding = padding
    layer.dilation = dilation

    if submod.groups is not None:
        layer.num_groups = submod.groups

    mark_as_int8_layer(layer, get_dyn_range(submod.scale, submod.zero_point, torch.quint8))

    return layer.get_output(0)


@tensorrt_converter("dequantize")
@tensorrt_converter(torch.nn.quantized.modules.DeQuantize)
def tensor_dequantize(network, submod, args, kwargs, name):
    input_val = args[0]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(f'dequantize received input {input_val} that is not part '
                           'of the TensorRT region!')

    return input_val


@tensorrt_converter(torch.nn.quantized.modules.batchnorm.BatchNorm2d)
def torch_nn_quantized_modules_batchnorm_BatchNorm2d(network, submod, args, kwargs, name):
    input_val = args[0]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(f'Quantized BatchNorm2d received input {input_val} that is not part '
                           'of the TensorRT region!')

    scale = to_numpy(submod.weight) / np.sqrt(
        to_numpy(submod.running_var) + submod.eps
    )
    bias = (
        to_numpy(submod.bias) - to_numpy(submod.running_mean) * scale
    )
    power = np.ones_like(scale)

    layer = network.add_scale(input_val, trt.ScaleMode.CHANNEL, bias, scale, power)
    layer.name = name
    mark_as_int8_layer(layer, get_dyn_range(submod.scale, submod.zero_point, torch.quint8))

    return layer.get_output(0)


@tensorrt_converter(torch.ops.quantized.add_relu)
def torch_ops_quantized_add_relu(network, submod, args, kwargs, name):
    assert len(args) == 0

    lhs_val, rhs_val = kwargs["qa"], kwargs["qb"]

    if not all(isinstance(i, trt.tensorrt.ITensor) for i in [lhs_val, rhs_val]):
        raise RuntimeError('Quantized add_relu received an input that is not part of the TensorRT region!')

    layer = network.add_elementwise(lhs_val, rhs_val, trt.ElementWiseOperation.SUM)
    layer.name = f"{name}_add"
    dyn_range = get_dyn_range(kwargs["scale"], kwargs["zero_point"], torch.quint8)
    mark_as_int8_layer(layer, dyn_range)

    layer = network.add_activation(
        input=layer.get_output(0), type=trt.ActivationType.RELU)
    layer.name = f"{name}_relu"
    mark_as_int8_layer(layer, dyn_range)

    return layer.get_output(0)


@tensorrt_converter(torch.nn.intrinsic.quantized.modules.ConvReLU2d)
def torch_nn_intrinsic_quantized_modules_ConvReLU2d(network, submod, args, kwargs, name):
    input_val = args[0]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(f'Quantized Conv2d received input {input_val} that is not part '
                           'of the TensorRT region!')

    kernel_size = process_attr(submod, 'kernel_size', 2)
    stride = process_attr(submod, 'stride', 2)
    padding = process_attr(submod, 'padding', 2)
    dilation = process_attr(submod, 'dilation', 2)

    kernel = to_numpy(submod.weight())

    bias = None
    if submod.bias() is not None:
        bias = to_numpy(submod.bias())

    layer = network.add_convolution(
        input=input_val,
        num_output_maps=submod.out_channels,
        kernel_shape=kernel_size,
        kernel=kernel,
        bias=bias,
    )
    layer.name = f"{name}_conv"
    layer.stride = stride
    layer.padding = padding
    layer.dilation = dilation

    if submod.groups is not None:
        layer.num_groups = submod.groups

    dyn_range = get_dyn_range(submod.scale, submod.zero_point, torch.quint8)
    mark_as_int8_layer(layer, dyn_range)

    layer = network.add_activation(
        input=layer.get_output(0), type=trt.ActivationType.RELU)
    layer.name = f"{name}_relu"
    mark_as_int8_layer(layer, dyn_range)

    return layer.get_output(0)


@tensorrt_converter(torch.nn.modules.linear.Identity)
def torch_nn_modules_linear_Identity(network, submod, args, kwargs, name):
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(f'Identity received input {input_val} that is not part '
                           'of the TensorRT region!')

    return input_val
