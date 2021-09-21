import math
import operator

import torch.fx.experimental.fx_acc.acc_ops as acc_ops
import torch.fx.experimental.fx_acc.acc_utils as acc_utils
import numpy as np
import tensorrt as trt
import torch
from torch.fx.experimental.fx2trt.fx2trt import (
    tensorrt_converter,
    torch_dtype_from_trt,
    get_dynamic_dims,
)
from typing import Optional


def to_numpy(tensor: Optional[torch.Tensor]):
    """
    Convert a PyTorch Tensor to a Numpy Array.
    """
    if tensor is None:
        return tensor

    if tensor.is_quantized:
        tensor = tensor.dequantize()

    assert isinstance(tensor, torch.Tensor), f"to_numpy can't be called on None or a torch.Tensor, got: {tensor}"

    return tensor.cpu().detach().contiguous().numpy()


def has_dynamic_shape(shape):
    return any(s == -1 for s in shape)


def get_axes_for_reduce_op(dim, has_implicit_batch_dimension):
    if isinstance(dim, int):
        dim = (dim,)

    if has_implicit_batch_dimension:
        assert 0 not in dim, "Can't reduce over batch dimension when it's implicit."

    axes = 0
    for d in dim:
        axes |= 1 << (d - (1 if has_implicit_batch_dimension else 0))

    return axes


def create_constant(network, tensor, name, dtype):
    if isinstance(tensor, int):
        tensor = torch.IntTensor([tensor])

    if isinstance(tensor, float):
        tensor = torch.Tensor([tensor])

    if dtype:
        tensor = tensor.to(dtype)

    constant = network.add_constant(tensor.shape, to_numpy(tensor))
    constant.name = name
    return constant.get_output(0)


def get_trt_tensor(network, input_val, name, dtype=None):
    if isinstance(input_val, (torch.Tensor, int, float)):
        return create_constant(network, input_val, name, dtype)
    elif not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"Received input {input_val} of name {name} that "
            "is not part of the TensorRT region!"
        )
    else:
        return input_val


def append_ones(network, input, name, num_prepend_ones):
    layer = network.add_shuffle(input)

    if has_dynamic_shape(input.shape):
        input_shape_layer = network.add_shape(input)
        input_shape_layer.name = f"{name}_broadcast_orig_shape"
        prepend_shape_layer = network.add_constant(
            (num_prepend_ones,), np.ones((num_prepend_ones,), dtype=np.int32)
        )
        prepend_shape_layer.name = f"{name}_broadcast_prepend_ones"
        reshape_dim_layer = network.add_concatenation(
            [prepend_shape_layer.get_output(0), input_shape_layer.get_output(0)]
        )
        reshape_dim_layer.axis = 0
        reshape_dim_layer.name = f"{name}_broadcast_final_shape"
        layer.set_input(1, reshape_dim_layer.get_output(0))
    else:
        layer.reshape_dims = (1,) * num_prepend_ones + tuple(input.shape)

    layer.name = name
    return layer.get_output(0)


def broadcast(network, a, b, a_name, b_name, preset_diff=0):
    """
    Broadcast two TensorRT tensors to the same number of dimensions by
    prepending 1s to the tensor with less number of dimensions.

    Args:
        network: TensorRT network object.
        a: A TensorRT tensor.
        b: A TensorRT tensor.
        a_name: Name of tensor a.
        b_name: Name of tensor b.
        preset_diff: The difference of number of dimensions after broadcast.
            A positive number means after broadcast, tensor `a` would have
            `preset_diff` more dimensions than `b`. This is used in matmul,
            since we need to broadcast tensors but not always to the same
            number of dimension. The reason is that matmul supports Matrix
            x Vector and in this case broadcasted vector should have 1 less
            number of dimensions than the matrix tensor.
    """
    a_shape = tuple(a.shape)
    b_shape = tuple(b.shape)

    diff = len(a_shape) - len(b_shape) - preset_diff
    if diff > 0:
        b = append_ones(network, b, f"{b_name}_broadcast", diff)
    elif diff < 0:
        a = append_ones(network, a, f"{a_name}_broadcast", -diff)

    return a, b

def add_binary_elementwise_layer(network, lhs_val, rhs_val, op_type, name):
    """
    This function adds a TensorRT elementwise layer. We only allow at most one
    operand to not be a trt tensor, otherwise, we should const fold it first.
    If any operand is not a trt tensor, we make it a trt constant layer which
    has the same type as the other trt tensor. Then we broadcast these two inputs
    to have the same number of dimensions.

    Limitation:
        If we are using implicit batch dim mode, the operand that is not a trt
    tensor are not allowed to have larger ranks than the trt tensor operand.

    Args:
        network: TensorRT network object.
        lhs_val: Left operand of the binary operation. Could be a TensorRT tensor,
            a PyTorch tensor or a simple value.
        rhs_val: Right operand of the binary operation. Similar to lhs_val.
        op_type: Type of the TensorRT elementwise binary operation.
        name: The name we want to assign to the created TensorRT layer.

    Returns:
        The output of TensorRT elementwise layer.
    """
    dtype = None
    is_lhs_trt_tensor = False
    is_rhs_trt_tensor = False
    if isinstance(lhs_val, trt.tensorrt.ITensor):
        dtype = torch_dtype_from_trt(lhs_val.dtype)
        is_lhs_trt_tensor = True
    if isinstance(rhs_val, trt.tensorrt.ITensor):
        dtype = torch_dtype_from_trt(rhs_val.dtype)
        is_rhs_trt_tensor = True
    if not is_lhs_trt_tensor and not is_rhs_trt_tensor:
        raise RuntimeError(f"Both operands of the binary elementwise op {name}"
                           "are constant. In this case, please consider constant fold the model first.")

    lhs_val = get_trt_tensor(network, lhs_val, f"{name}_lhs", dtype)
    rhs_val = get_trt_tensor(network, rhs_val, f"{name}_rhs", dtype)

    # Check the limitation in the doc string.
    if network.has_implicit_batch_dimension:
        if is_lhs_trt_tensor and not is_rhs_trt_tensor:
            assert len(lhs_val.shape) >= len(rhs_val.shape)
        elif not is_lhs_trt_tensor and is_rhs_trt_tensor:
            assert len(rhs_val.shape) >= len(lhs_val.shape)

    lhs_val, rhs_val = broadcast(
        network, lhs_val, rhs_val, f"{name}_lhs", f"{name}_rhs"
    )
    layer = network.add_elementwise(lhs_val, rhs_val, op_type)
    layer.name = name
    return layer.get_output(0)


def add_unary_layer(network, input_val, operation_type, name):
    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"{operation_type} received input {input_val} that is not part "
            "of the TensorRT region!"
        )
    layer = network.add_unary(input_val, operation_type)
    layer.name = name
    return layer.get_output(0)


def add_activation_layer(network, input_val, operation_type, name):
    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"{operation_type} received input {input_val} that is not part "
            "of the TensorRT region!"
        )
    layer = network.add_activation(input_val, operation_type)
    layer.name = name
    return layer.get_output(0)


def add_transpose_layer(
    network, input_val, dim_0, dim_1, name, ignore_implicit_batch=False
):
    """Adds a transpose layer to the TensorRT network
    Args:
        network: TensorRT Network object
        input_val: tensorrt.ITensor
        dim_0, dim_1: dimensions for transpose, e.g. dim_0=1, dim_1=0 means transpose
        the first two dimensions
        name: Name of the layer
        ignore_implicit_batch: activations might have implicit batch, but weights do
        not, when this is True, we'll ignore the implicit batch and use the dimension
        argument as is
    Returns:
        output TensorRT ITensor from the transpose layer
    """
    if not ignore_implicit_batch and network.has_implicit_batch_dimension:
        assert (
            dim_0 != 0 and dim_1 != 0
        ), "It's not allowed to call transpose on non-constant when batch dim is implicit!"
        dim_0 -= 1
        dim_1 -= 1

    permutation = list(range(len(input_val.shape)))
    permutation[dim_0] = dim_1
    permutation[dim_1] = dim_0

    layer = network.add_shuffle(input_val)
    layer.second_transpose = tuple(permutation)
    layer.name = name
    return layer.get_output(0)

def add_matrix_multiply_layer(network, input_val, other_val, name, transpose_input=False, transpose_other=False):
    """ Adds a matrix multiply layer to the TensorRT network
    Args:
        network: TensorRT Network
        input_val: input matrix/vector TensorRT ITensor
        other_val: another input matrix/vector TensorRT ITensor
        name: Name of the matrix multiply layer
        transpose_input: boolean indicating whether to transpose the input_val Tensor or not
        transpose_other: boolean indicaiton whether to transpose the other_val Tensor or not
    Returns:
        output TensorRT ITensor from the matrix multiply layer
    """
    input_matrix_op = other_matrix_op = trt.MatrixOperation.NONE
    preset_diff = 0

    if len(input_val.shape) == 1:
        assert not transpose_input, "can't transpose input vector"
        preset_diff -= 1
        input_matrix_op = trt.MatrixOperation.VECTOR
    elif transpose_input:
        input_matrix_op = trt.MatrixOperation.TRANSPOSE

    if len(other_val.shape) == 1:
        assert not transpose_input, "can't transpose other vector"
        preset_diff += 1
        other_matrix_op = trt.MatrixOperation.VECTOR
    elif transpose_other:
        other_matrix_op = trt.MatrixOperation.TRANSPOSE


    input_val, other_val = broadcast(network, input_val, other_val, f"{name}_input", f"{name}_other", preset_diff)
    layer = network.add_matrix_multiply(input_val, input_matrix_op, other_val, other_matrix_op)
    layer.name = name
    return layer.get_output(0)

def process_attr(val, num_elem):
    if not isinstance(val, tuple):
        val = (val,) * num_elem
    return val


@tensorrt_converter(acc_ops.conv2d)
def acc_ops_conv2d(network, target, args, kwargs, name):
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"Conv2d received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    if has_dynamic_shape(input_val.shape):
        assert input_val.shape[1] != -1, "Channel dim can't be dynamic for convolution."

    # for now we'll assume bias is constant Tensor or None,
    # and bias being ITensor is not supported in TensorRT api
    # right now
    bias = to_numpy(kwargs["bias"])

    if network.has_explicit_precision:
        weight = get_trt_tensor(network, kwargs["weight"], f"{name}_weight")
        weight_shape = tuple(kwargs["weight"].shape)
        # will need to use uninitialized weight and set it later to support
        # ITensor weights
        dummy_weight = trt.Weights()

        layer = network.add_convolution(
            input=input_val,
            num_output_maps=weight.shape[0],
            kernel_shape=weight.shape[2:],
            kernel=dummy_weight,
            bias=bias,
        )

        layer.set_input(1, weight)
    else:
        weight = to_numpy(kwargs["weight"])
        layer = network.add_convolution(
            input=input_val,
            num_output_maps=weight.shape[0],
            kernel_shape=weight.shape[2:],
            kernel=weight,
            bias=bias,
        )

    layer.name = name
    layer.stride = kwargs["stride"]
    layer.padding = kwargs["padding"]
    layer.dilation = kwargs["dilation"]
    if kwargs["groups"] is not None:
        layer.num_groups = kwargs["groups"]

    return layer.get_output(0)


@tensorrt_converter(acc_ops.flatten)
def acc_ops_flatten(network, target, args, kwargs, name):
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"flatten received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    num_dims = len(input_val.shape) + (1 if network.has_implicit_batch_dimension else 0)
    start_dim = (kwargs["start_dim"] if "start_dim" in kwargs else 0) % num_dims
    end_dim = (kwargs["end_dim"] if "end_dim" in kwargs else -1) % num_dims

    if network.has_implicit_batch_dimension:
        assert start_dim != 0, "Can't flatten batch dimension when it's implicit."
        start_dim -= 1
        end_dim -= 1

    layer = network.add_shuffle(input_val)
    layer.name = name

    # If there're dynamic shapes then we need to use shape layers
    # to figure out the final shape after flatten. We first slice
    # the input shape to three parts:
    #   1. dimensions before start_dim
    #   2. dimensions between start_dim and end_dim
    #   3. dimensions after end_dim
    # Part 1 and 3 might not exist if start_dim is 0 or end_dim is
    # last dim. Then we do a reduced multiplication over part 2 to
    # get flattened dim. Finally, we concatenate the three parts to
    # get the final shape.
    if has_dynamic_shape(input_val.shape):
        input_shape_layer = network.add_shape(input_val)
        input_shape_layer.name = f"{name}_orig_shape"

        final_shapes = []

        # Shapes before start_dim
        if start_dim > 0:
            prefix_shape_layer = network.add_slice(
                input_shape_layer.get_output(0),
                start=(0,),
                shape=(start_dim,),
                stride=(1,),
            )
            prefix_shape_layer.name = f"{name}_pre_shape"
            final_shapes.append(prefix_shape_layer.get_output(0))

        flatten_shape_layer = network.add_slice(
            input_shape_layer.get_output(0),
            start=(start_dim,),
            shape=(end_dim - start_dim + 1,),
            stride=(1,),
        )
        flatten_shape_layer.name = f"{name}_need_flatten"
        flatten_shape_layer = network.add_reduce(
            flatten_shape_layer.get_output(0),
            trt.ReduceOperation.PROD,
            axes=get_axes_for_reduce_op(0, False),
            keep_dims=True,
        )
        flatten_shape_layer.name = f"{name}_flatten_dim"
        final_shapes.append(flatten_shape_layer.get_output(0))

        # Shapes after start_dim
        if end_dim < len(input_val.shape) - 1:
            suffix_shape_layer = network.add_slice(
                input_shape_layer.get_output(0),
                start=(end_dim + 1,),
                shape=(len(input_val.shape) - end_dim - 1,),
                stride=(1,),
            )
            suffix_shape_layer.name = f"{name}_suffix_shape"
            final_shapes.append(suffix_shape_layer.get_output(0))

        final_shape_layer = network.add_concatenation(final_shapes)
        final_shape_layer.axis = 0
        final_shape_layer.name = f"{name}_final_shape"
        layer.set_input(1, final_shape_layer.get_output(0))
    else:
        final_shape = []
        flatten_dim = 1
        for i, s in enumerate(input_val.shape):
            if i >= start_dim and i <= end_dim:
                flatten_dim *= s
            elif i == end_dim + 1:
                final_shape.append(flatten_dim)
                final_shape.append(s)
            else:
                final_shape.append(s)
        if end_dim == len(input_val.shape) - 1:
            final_shape.append(flatten_dim)

        layer.reshape_dims = tuple(final_shape)

    return layer.get_output(0)


# For implicit batch dim mode, we use this to represent batch dim if we
# ever trying to retrieve it via size() and we hope it will fail hard if
# it's used somewhere else.
IMPLICIT_BATCH_DIM = -999


@tensorrt_converter(acc_ops.size)
def acc_ops_size(network, target, args, kwargs, name):
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"size received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    if not has_dynamic_shape(input_val.shape):
        if network.has_implicit_batch_dimension:
            return torch.Size((IMPLICIT_BATCH_DIM,) + tuple(input_val.shape))
        return torch.Size(input_val.shape)

    layer = network.add_shape(input_val)
    layer.name = name
    return layer.get_output(0)


@tensorrt_converter(acc_ops.batch_norm)
def acc_ops_batch_norm(network, target, args, kwargs, name):
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"BatchNorm2d received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    if has_dynamic_shape(input_val.shape):
        assert input_val.shape[1] != -1, "Channel dim can't be dynamic for batch norm."

    scale = to_numpy(kwargs["weight"]) / np.sqrt(
        to_numpy(kwargs["running_var"]) + kwargs["eps"]
    )
    bias = (
        to_numpy(kwargs["bias"])
        - to_numpy(kwargs["running_mean"]) * scale
    )
    power = np.ones_like(scale)

    layer = network.add_scale(input_val, trt.ScaleMode.CHANNEL, bias, scale, power)
    layer.name = name

    return layer.get_output(0)

@tensorrt_converter(acc_ops.layer_norm)
def acc_ops_layer_norm(network, target, args, kwargs, name):
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(f"LayerNorm received input {input_val} that is not part "
                           "of the TensorRT region!")

    shape = kwargs["weight"].shape
    broadcasted_shape = (1,) * (len(input_val.shape) - len(shape)) + shape
    gamma = to_numpy(kwargs["weight"].reshape(*shape))
    beta = to_numpy(kwargs["bias"].reshape(*shape))
    eps = kwargs["eps"]
    normalized_shape = kwargs["normalized_shape"]

    axes = 0
    for d in range(len(normalized_shape)):
        axes |= 1 << (len(input_val.shape) - d - 1)

    # E[x]
    mean_expected_layer = network.add_reduce(input_val, trt.ReduceOperation.AVG, axes, keep_dims=True)
    mean_expected_layer.name = f"{name}_mean_expected"
    # X-E[x]
    sub_trt = add_binary_elementwise_layer(
        network, input_val, mean_expected_layer.get_output(0), trt.ElementWiseOperation.SUB, f"{name}_sub"
    )
    # Variance = mean(pow(x_sub_mean,2))
    pow_tensor = network.add_constant(
        (1,) * len(input_val.shape), trt.Weights(np.ascontiguousarray([2.0], dtype=np.float32))
    )
    pow_tensor.name = f"{name}_power"
    pow_var = add_binary_elementwise_layer(
        network, sub_trt, pow_tensor.get_output(0), trt.ElementWiseOperation.POW, f"{name}_pow_var"
    )
    mean_trt_layer = network.add_reduce(pow_var, trt.ReduceOperation.AVG, axes, keep_dims=True)
    mean_trt_layer.name = f"{name}_mean"
    # Variance + eps
    eps_tensor = network.add_constant(
        (1,) * len(input_val.shape), trt.Weights(np.ascontiguousarray([eps], dtype=np.float32))
    )
    eps_tensor.name = f"{name}_eps"
    add_trt = add_binary_elementwise_layer(
        network, mean_trt_layer.get_output(0), eps_tensor.get_output(0), trt.ElementWiseOperation.SUM, f"{name}_add"
    )
    # SQRT((Var + eps))
    sqrt_trt = add_unary_layer(network, add_trt, trt.UnaryOperation.SQRT, f"{name}_sqrt")
    # (x - E[x]) / sqrt((var + eps))
    div_trt = add_binary_elementwise_layer(network, sub_trt, sqrt_trt, trt.ElementWiseOperation.DIV, f"{name}_div_trt")

    gamma_tensor = network.add_constant(gamma.shape, trt.Weights(np.ascontiguousarray(gamma)))
    gamma_tensor.name = f"{name}_gamma"
    beta_tensor = network.add_constant(gamma.shape, trt.Weights(np.ascontiguousarray(beta)))
    beta_tensor.name = f"{name}_beta"
    # y * gamma + beta
    scale_layer = add_binary_elementwise_layer(
        network, div_trt, gamma_tensor.get_output(0), trt.ElementWiseOperation.PROD, f"{name}_scale"
    )
    return add_binary_elementwise_layer(
        network, scale_layer, beta_tensor.get_output(0), trt.ElementWiseOperation.SUM, name
    )

@tensorrt_converter(acc_ops.softmax)
def acc_ops_softmax(network, target, args, kwargs, name):
    input_val = kwargs["input"]
    dim = kwargs["dim"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"softmax received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    # Used to get dim when dim is None. Copied from PyTorch softmax implementation.
    def get_softmax_dim(ndim):
        if ndim == 0 or ndim == 1 or ndim == 3:
            ret = 0
        else:
            ret = 1
        return ret

    if dim is None:
        dim = get_softmax_dim(
            len(input_val.shape)
            if not network.has_implicit_batch_dimension
            else len(input_val.shape) + 1
        )

    if network.has_implicit_batch_dimension:
        assert dim != 0, "Can't apply softmax on batch dimension when it's implicit."
        dim = (dim % (len(input_val.shape) + 1)) - 1

    layer = network.add_softmax(input_val)
    layer.axes = 1 << dim
    layer.name = name
    return layer.get_output(0)


@tensorrt_converter(acc_ops.relu)
def acc_ops_relu(network, target, args, kwargs, name):
    input_val = kwargs["input"]
    operation_type = trt.ActivationType.RELU
    return add_activation_layer(network, input_val, operation_type, name)


@tensorrt_converter(acc_ops.sin)
def acc_ops_sin(network, target, args, kwargs, name):
    input_val = kwargs["input"]
    operation_type = trt.UnaryOperation.SIN
    return add_unary_layer(network, input_val, operation_type, name)


@tensorrt_converter(acc_ops.cos)
def acc_ops_cos(network, target, args, kwargs, name):
    input_val = kwargs["input"]
    operation_type = trt.UnaryOperation.COS
    return add_unary_layer(network, input_val, operation_type, name)


@tensorrt_converter(acc_ops.tan)
def acc_ops_tan(network, target, args, kwargs, name):
    input_val = kwargs["input"]
    operation_type = trt.UnaryOperation.TAN
    return add_unary_layer(network, input_val, operation_type, name)


@tensorrt_converter(acc_ops.sinh)
def acc_ops_sinh(network, target, args, kwargs, name):
    input_val = kwargs["input"]
    operation_type = trt.UnaryOperation.SINH
    return add_unary_layer(network, input_val, operation_type, name)


@tensorrt_converter(acc_ops.cosh)
def acc_ops_cosh(network, target, args, kwargs, name):
    input_val = kwargs["input"]
    operation_type = trt.UnaryOperation.COSH
    return add_unary_layer(network, input_val, operation_type, name)


@tensorrt_converter(acc_ops.tanh)
def acc_ops_tanh(network, target, args, kwargs, name):
    input_val = kwargs["input"]
    operation_type = trt.ActivationType.TANH
    return add_activation_layer(network, input_val, operation_type, name)


@tensorrt_converter(acc_ops.asin)
def acc_ops_asin(network, target, args, kwargs, name):
    input_val = kwargs["input"]
    operation_type = trt.UnaryOperation.ASIN
    return add_unary_layer(network, input_val, operation_type, name)


@tensorrt_converter(acc_ops.acos)
def acc_ops_acos(network, target, args, kwargs, name):
    input_val = kwargs["input"]
    operation_type = trt.UnaryOperation.ACOS
    return add_unary_layer(network, input_val, operation_type, name)


@tensorrt_converter(acc_ops.atan)
def acc_ops_atan(network, target, args, kwargs, name):
    input_val = kwargs["input"]
    operation_type = trt.UnaryOperation.ATAN
    return add_unary_layer(network, input_val, operation_type, name)


@tensorrt_converter(acc_ops.exp)
def acc_ops_exp(network, target, args, kwargs, name):
    input_val = kwargs["input"]
    operation_type = trt.UnaryOperation.EXP
    return add_unary_layer(network, input_val, operation_type, name)


@tensorrt_converter(acc_ops.log)
def acc_ops_log(network, target, args, kwargs, name):
    input_val = kwargs["input"]
    operation_type = trt.UnaryOperation.LOG
    return add_unary_layer(network, input_val, operation_type, name)


@tensorrt_converter(acc_ops.sqrt)
def acc_ops_sqrt(network, target, args, kwargs, name):
    input_val = kwargs["input"]
    operation_type = trt.UnaryOperation.SQRT
    return add_unary_layer(network, input_val, operation_type, name)


@tensorrt_converter(acc_ops.reciprocal)
def acc_ops_reciprocal(network, target, args, kwargs, name):
    input_val = kwargs["input"]
    operation_type = trt.UnaryOperation.RECIP
    return add_unary_layer(network, input_val, operation_type, name)


@tensorrt_converter(acc_ops.abs)
def acc_ops_abs(network, target, args, kwargs, name):
    input_val = kwargs["input"]
    operation_type = trt.UnaryOperation.ABS
    return add_unary_layer(network, input_val, operation_type, name)


@tensorrt_converter(acc_ops.neg)
def acc_ops_neg(network, target, args, kwargs, name):
    input_val = kwargs["input"]
    operation_type = trt.UnaryOperation.NEG
    return add_unary_layer(network, input_val, operation_type, name)


@tensorrt_converter(acc_ops.floor)
def acc_ops_floor(network, target, args, kwargs, name):
    input_val = kwargs["input"]
    operation_type = trt.UnaryOperation.FLOOR
    return add_unary_layer(network, input_val, operation_type, name)


@tensorrt_converter(acc_ops.ceil)
def acc_ops_ceil(network, target, args, kwargs, name):
    input_val = kwargs["input"]
    operation_type = trt.UnaryOperation.CEIL
    return add_unary_layer(network, input_val, operation_type, name)


@tensorrt_converter(acc_ops.sum)
def acc_ops_sum(network, target, args, kwargs, name):
    input_val = kwargs["input"]
    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"sum received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    # If dim is specified, then we are computing reduced sum over certain dimensions.
    # Otherwise, we are dong summation over all elements, which is only supported in
    # explicit batch dimension.
    if "dim" not in kwargs:
        assert (
            not network.has_implicit_batch_dimension
        ), "Do not support sum all the elements for implicit batch."
        dim = range(0, len(input_val.shape))
    else:
        dim = kwargs["dim"]

    keepdim = False if "keepdim" not in kwargs else kwargs["keepdim"]
    layer = network.add_reduce(
        input_val,
        trt.ReduceOperation.SUM,
        get_axes_for_reduce_op(dim, network.has_implicit_batch_dimension),
        keepdim,
    )
    layer.name = name
    return layer.get_output(0)


def add_acc_ops_full_reduce(network, target, args, kwargs, name, reduce_op):
    input_val = kwargs["input"]
    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"max received input {input_val} that is not part "
            "of the TensorRT region!"
        )
    assert (
        not network.has_implicit_batch_dimension
    ), "Do not support max over all the elements for implicit batch."

    dim = range(len(input_val.shape))

    layer = network.add_reduce(
        input_val,
        reduce_op,
        get_axes_for_reduce_op(dim, network.has_implicit_batch_dimension),
        False,
    )
    layer.name = name
    return layer.get_output(0)

def add_acc_ops_dim_reduce(network, target, args, kwargs, name, reduce_op):
    new_kwargs = kwargs.copy()
    new_kwargs['k'] = 1


    if reduce_op == trt.ReduceOperation.MAX:
        new_kwargs['largest'] = True
    elif reduce_op == trt.ReduceOperation.MIN:
        new_kwargs['largest'] = False
    new_kwargs['sorted'] = False


    (topk_out0, topk_out1) = acc_ops_topk(network, target, args, new_kwargs, name + "_topk")

    topk_out0.name = f"{name}_topk0"
    topk_out1.name = f"{name}_topk1"

    if 'keepdim' in new_kwargs and new_kwargs['keepdim']:
        return (topk_out0, topk_out1)

    dim = new_kwargs['dim']
    if network.has_implicit_batch_dimension:
        assert dim != 0, "can't reduce on dim == 0 when network has implicit batch dimension"
        # we remove the first dim in the shape tuple when it is implicit
        dim -= 1
    input_val = topk_out0
    shape = input_val.shape

    output_shape = []
    for i, s in enumerate(shape):
        if i == dim and s == 1:
            continue
        output_shape.append(s)

    shuffle_layer0 = network.add_shuffle(input_val)
    shuffle_layer0.reshape_dims = tuple(output_shape)
    shuffle_layer0.name = name + '_shuffle0'

    input_val = topk_out1
    shape = input_val.shape

    shuffle_layer1 = network.add_shuffle(input_val)
    shuffle_layer1.reshape_dims = tuple(output_shape)
    shuffle_layer1.name = name + '_shuffle1'


    return (shuffle_layer0.get_output(0), shuffle_layer1.get_output(0))

@tensorrt_converter(acc_ops.max_full_reduce)
def acc_ops_max_full_reduce(network, target, args, kwargs, name):
    return add_acc_ops_full_reduce(network, target, args, kwargs, name, trt.ReduceOperation.MAX)

@tensorrt_converter(acc_ops.min_full_reduce)
def acc_ops_min_full_reduce(network, target, args, kwargs, name):
    return add_acc_ops_full_reduce(network, target, args, kwargs, name, trt.ReduceOperation.MIN)

@tensorrt_converter(acc_ops.max_dim_reduce)
def acc_ops_max_dim_reduce(network, target, args, kwargs, name):
    return add_acc_ops_dim_reduce(network, target, args, kwargs, name, trt.ReduceOperation.MAX)

@tensorrt_converter(acc_ops.min_dim_reduce)
def acc_ops_min_dim_reduce(network, target, args, kwargs, name):
    return add_acc_ops_dim_reduce(network, target, args, kwargs, name, trt.ReduceOperation.MIN)

@tensorrt_converter(acc_ops.maximum)
def acc_ops_maximum(network, target, args, kwargs, name):
    return add_binary_elementwise_layer(
        network, kwargs["input"], kwargs["other"], trt.ElementWiseOperation.MAX, name
    )

@tensorrt_converter(acc_ops.minimum)
def acc_ops_minimum(network, target, args, kwargs, name):
    return add_binary_elementwise_layer(
        network, kwargs["input"], kwargs["other"], trt.ElementWiseOperation.MIN, name
    )

@tensorrt_converter(acc_ops.max_pool2d)
def acc_ops_max_pool2d(network, target, args, kwargs, name):
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"MaxPool2d received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    kernel_size = process_attr(kwargs["kernel_size"], 2)
    stride = process_attr(kwargs["stride"], 2)
    padding = process_attr(kwargs["padding"], 2)
    dilation = process_attr(kwargs["dilation"], 2)
    ceil_mode = kwargs["ceil_mode"]

    if dilation != (1, 1):
        raise RuntimeError(
            f"Only support dilation=(1, 1) for maxpool, but got {dilation}"
        )

    layer = network.add_pooling(
        input=input_val, type=trt.PoolingType.MAX, window_size=kernel_size
    )
    layer.stride = stride
    layer.padding = padding
    layer.name = name

    if ceil_mode:
        layer.padding_mode = trt.PaddingMode.EXPLICIT_ROUND_UP

    return layer.get_output(0)


@tensorrt_converter(acc_ops.squeeze)
def acc_ops_squeeze(network, target, args, kwargs, name):
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"squeeze received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    dim = kwargs["dim"] if "dim" in kwargs else None
    # Squeeze with dim=None would only work in explicit batch dim mode without any dynamic
    # dim, which is a very rare case. For now we just claim not supporting dim=None.
    assert dim is not None, "We don't support dim=None right now."

    dim = dim % (len(input_val.shape) + (1 if network.has_implicit_batch_dimension else 0))
    if network.has_implicit_batch_dimension:
        assert dim != 0, "We don't support squeeze batch dim when it's implicit."
        dim -= 1

    assert input_val.shape[dim] != -1, "We don't support squeeze dynamic dim."
    assert (
        len(get_dynamic_dims(input_val.shape)) <= 1
    ), "Currently more than one dynamic dim for input to squeeze is not supported."

    output_shape = []
    for i, s in enumerate(input_val.shape):
        if i == dim and s == 1:
            continue
        output_shape.append(s)
    layer = network.add_shuffle(input_val)
    layer.reshape_dims = tuple(output_shape)
    layer.name = name
    return layer.get_output(0)


@tensorrt_converter(acc_ops.add)
def acc_ops_add(network, target, args, kwargs, name):
    return add_binary_elementwise_layer(
        network, kwargs["input"], kwargs["other"], trt.ElementWiseOperation.SUM, name
    )


@tensorrt_converter(acc_ops.sub)
def acc_ops_sub(network, target, args, kwargs, name):
    return add_binary_elementwise_layer(
        network, kwargs["input"], kwargs["other"], trt.ElementWiseOperation.SUB, name
    )


@tensorrt_converter(acc_ops.div)
def acc_ops_div(network, target, args, kwargs, name):
    return add_binary_elementwise_layer(
        network, kwargs["input"], kwargs["other"], trt.ElementWiseOperation.DIV, name
    )


@tensorrt_converter(acc_ops.mul)
def acc_ops_mul(network, target, args, kwargs, name):
    return add_binary_elementwise_layer(
        network, kwargs["input"], kwargs["other"], trt.ElementWiseOperation.PROD, name
    )

@tensorrt_converter(acc_ops.pow)
def acc_ops_pow(network, target, args, kwargs, name):
    return add_binary_elementwise_layer(
        network, kwargs["input"], kwargs["exponent"], trt.ElementWiseOperation.POW, name
    )

@tensorrt_converter(acc_ops.unsqueeze)
def acc_ops_unsqueeze(network, target, args, kwargs, name):
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(f"unsqueeze received input {input_val} that is not part "
                           "of the TensorRT region!")

    dim = kwargs["dim"]
    if network.has_implicit_batch_dimension:
        assert dim != 0
        dim -= 1

    assert len(get_dynamic_dims(input_val.shape)) <= 1, "Currently we don't support unsqueeze with more than one dynamic dims."
    layer = network.add_shuffle(input_val)
    layer.reshape_dims = tuple(input_val.shape)[:dim] + (1,) + tuple(input_val.shape)[dim:]
    layer.name = name
    return layer.get_output(0)

@tensorrt_converter(acc_ops.topk)
def acc_ops_topk(network, target, args, kwargs, name):
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(f"topk received input {input_val} that is not part "
                           "of the TensorRT region!")

    if kwargs["sorted"] and kwargs["k"] != 1:
        raise RuntimeError("Currently we don't support sorted=True in topk.")

    if not network.has_implicit_batch_dimension and len(input_val.shape) <= 1:
        raise RuntimeError("At least 2 dimensions are required for input to topk.")

    num_dims = len(input_val.shape) + (1 if network.has_implicit_batch_dimension else 0)
    k = kwargs["k"]
    dim = (kwargs["dim"] if kwargs["dim"] is not None else -1) % num_dims
    operation = trt.TopKOperation.MAX if kwargs["largest"] else trt.TopKOperation.MIN
    layer = network.add_topk(
        input_val, operation, k, get_axes_for_reduce_op(dim, network.has_implicit_batch_dimension)
    )
    layer.name = name
    return layer.get_output(0), layer.get_output(1)

@tensorrt_converter(acc_ops.adaptive_avg_pool2d)
def acc_ops_adaptive_avg_pool2d(network, target, args, kwargs, name):
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"AdaptiveAvgPool2d received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    assert (
        input_val.shape[-1] != -1 and input_val.shape[-1] != -1
    ), "AdaptiveAvgPool2d currently doesn't support dynamic shapes for last two dims."
    output_size = kwargs["output_size"]

    for input_dim, output_dim in zip(input_val.shape[-2:], output_size):
        if input_dim % output_dim != 0:
            raise RuntimeError(
                "For AdaptiveAvgPool, input dim has to be integer multiple of output dim."
                f"Got input dim {input_dim}, output dim {output_dim}"
            )

    stride = (
        input_val.shape[-2] // output_size[0],
        input_val.shape[-1] // output_size[1],
    )
    kernel_size = (
        input_val.shape[-2] - (output_size[0] - 1) * stride[0],
        input_val.shape[-1] - (output_size[1] - 1) * stride[1],
    )
    layer = network.add_pooling(
        input=input_val, type=trt.PoolingType.AVERAGE, window_size=kernel_size
    )
    layer.stride = stride
    layer.name = name

    return layer.get_output(0)


@tensorrt_converter(acc_ops.avg_pool2d)
def acc_ops_avg_pool2d(network, target, args, kwargs, name):
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"AvgPool2d received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    kernel_size = process_attr(kwargs["kernel_size"], 2)
    stride = process_attr(kwargs["stride"], 2)
    padding = process_attr(kwargs["padding"], 2)
    ceil_mode = kwargs["ceil_mode"]
    count_include_pad = kwargs["count_include_pad"]
    divisor_override = kwargs["divisor_override"]

    if divisor_override:
        raise RuntimeError("TensorRT does not support divisor_override.")

    layer = network.add_pooling(
        input=input_val, type=trt.PoolingType.AVERAGE, window_size=kernel_size
    )
    layer.stride = stride
    layer.padding = padding
    layer.average_count_excludes_padding = False if count_include_pad else True

    if ceil_mode:
        layer.padding_mode = trt.PaddingMode.EXPLICIT_ROUND_UP

    return layer.get_output(0)


@tensorrt_converter(acc_ops.reshape)
def acc_ops_reshape(network, target, args, kwargs, name):
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"Reshape received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    shape = acc_utils.get_field_from_acc_out_ty(kwargs["acc_out_ty"], "shape")
    if network.has_implicit_batch_dimension:
        shape = shape[1:]

    layer = network.add_shuffle(input_val)

    if all(isinstance(s, int) for s in shape):
        layer.reshape_dims = tuple(shape)
    else:
        # Convert all the dimensions to trt Tensors.
        trt_shape = []

        for i, s in enumerate(shape):
            if isinstance(s, trt.tensorrt.ITensor):
                if len(s.shape) == 0:
                    s = append_ones(network, s, f"{name}_{i}", 1)
                trt_shape.append(s)
            else:
                trt_shape.append(
                    get_trt_tensor(network, s, f"{name}_{i}")
                )

        shape_layer = network.add_concatenation(inputs=trt_shape)
        shape_layer.axis = 0
        shape_layer.name = f"{name}_output_shape"
        layer.set_input(1, shape_layer.get_output(0))

    layer.name = name
    return layer.get_output(0)

@tensorrt_converter(acc_ops.slice_tensor)
def acc_ops_slice_tensor(network, target, args, kwargs, name):
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(f"slice_tensor received input {input_val} that is not part "
                           "of the TensorRT region!")

    dims = kwargs["dims"]
    if network.has_implicit_batch_dimension:
        if not len(dims):
            raise RuntimeError("dim argument cannot be empty!")
        if any([dim == 0 for dim in dims]):
            raise RuntimeError(
                f"We do not support slice_tensor at batch dim when it's implicit, got {dims}!"
            )
        dims = [d - 1 for d in dims]
    else:
        raise RuntimeError("We don't support slice_tensor with explicit batch dimension yet!")

    start = [0] * len(input_val.shape)
    stride = [1] * len(start)
    output_shape = list(input_val.shape)
    starts = kwargs["starts"]
    stops = kwargs["stops"]
    steps = kwargs["steps"]

    for i, dim in enumerate(dims):
        start[dim] = starts[i]
        stride[dim] = steps[i]
        output_shape[dim] = (stops[i] - start[i]) // steps[i]

    layer = network.add_slice(input_val, start=start, shape=output_shape, stride=stride)
    layer.name = name
    return layer.get_output(0)

@tensorrt_converter(acc_ops.split)
def acc_ops_split(network, target, args, kwargs, name):
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(f"split received input {input_val} that is not part "
                           "of the TensorRT region!")

    dim = kwargs["dim"]
    if network.has_implicit_batch_dimension:
        assert dim != 0, "Can't split on batch dim when it's implicit!"
        dim -= 1
    else:
        raise RuntimeError("We don't support split with explicit batch dimension yet!")

    split_size = kwargs["split_size"]
    start = [0] * len(input_val.shape)
    stride = [1] * len(start)
    offset = 0
    num_splits = (input_val.shape[dim] + split_size - 1) // split_size
    if num_splits < 1:
        raise RuntimeError(f"Invalid split: {input_val.shape[dim]} wuth split_size={split_size}")

    max_offset = input_val.shape[dim]
    # add slice layers
    output = []
    for i in range(num_splits):
        shape = list(input_val.shape)
        shape[dim] = min(split_size, max_offset - offset)
        start[dim] = offset
        layer = network.add_slice(input_val, start=start, shape=shape, stride=stride)
        offset += split_size
        layer.name = f"{name}_{i}"
        output.append(layer.get_output(0))
    return output

@tensorrt_converter(acc_ops.linear)
def acc_ops_linear(network, target, args, kwargs, name):
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"Linear received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    dynamic_dims = get_dynamic_dims(input_val.shape)
    assert len(dynamic_dims) < 2 and input_val.shape[-1] != -1, (
        "Currently we only support one dynmaic "
        "dim for linear and it can't be the last dim."
    )

    # TODO: Need to benchmark the performance of lowering linear as fully_connected versus
    # lowering as matmul + add. TensorRT documentation suggests to always lower it as
    # matmul + add but we found in some cases this results in performance regression compared
    # with lowering to fully_connected layer.
    layer = network.add_shuffle(input_val)
    layer.reshape_dims = tuple(input_val.shape) + (1, 1)
    layer.name = f"{name}_pre_shuffle"
    bias = to_numpy(kwargs["bias"])

    if network.has_explicit_precision:
        weight = get_trt_tensor(network, kwargs["weight"], f"{name}_weight")
        # will need to use uninitialized weight and set it later to support
        # ITensor weights
        dummy_weight = trt.Weights()

        # add fully connected
        layer = network.add_fully_connected(
            input=layer.get_output(0),
            num_outputs=weight.shape[0],
            kernel=dummy_weight,
            bias=bias,
        )
        layer.set_input(1, weight)
    else:
        weight = to_numpy(kwargs["weight"])
        layer = network.add_fully_connected(
            input=layer.get_output(0),
            num_outputs=weight.shape[0],
            kernel=weight,
            bias=bias,
        )
    layer.name = f"{name}_linear"

    # reshape back
    layer = network.add_shuffle(layer.get_output(0))
    layer.reshape_dims = tuple(input_val.shape[:-1]) + (kwargs["weight"].shape[0],)
    layer.name = f"{name}_post_shuffle"

    return layer.get_output(0)

def add_clamp(network, input, val, op):
    acc_ops_clamp_shape = (1,) * len(input.shape)  # broadcast all dimensions
    acc_ops_clamp_tensor = (
        val
        * torch.ones(acc_ops_clamp_shape, dtype=torch_dtype_from_trt(input.dtype))
        .cpu()
        .numpy()
    )
    acc_ops_clamp_trt = network.add_constant(acc_ops_clamp_shape, acc_ops_clamp_tensor)
    layer = network.add_elementwise(input, acc_ops_clamp_trt.get_output(0), op)

    return layer


@tensorrt_converter(acc_ops.clamp)
def acc_ops_clamp(network, target, args, kwargs, name):
    input_val = kwargs["input"]
    min_val = kwargs["min"]
    max_val = kwargs["max"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"Clamp received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    if min_val is not None:
        clamp_min_layer = add_clamp(
            network, input_val, min_val, trt.ElementWiseOperation.MAX
        )
        clamp_min_layer.name = f"{name}_clamp_min"
        input_val = clamp_min_layer.get_output(0)
    if max_val is not None:
        clamp_max_layer = add_clamp(
            network, input_val, max_val, trt.ElementWiseOperation.MIN
        )
        clamp_max_layer.name = f"{name}_clamp_max"
        input_val = clamp_max_layer.get_output(0)

    return input_val

@tensorrt_converter(acc_ops.tuple_construct)
def acc_ops_tuple_construct(network, target, args, kwargs, name):
    return kwargs["tensors"]


@tensorrt_converter(acc_ops.contiguous)
def acc_ops_contiguous(network, target, args, kwargs, name):
    return kwargs["input"]


@tensorrt_converter(acc_ops.getitem)
def acc_ops_getitem(network, target, args, kwargs, name):
    input_val = kwargs["input"]
    slices = kwargs["idx"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        return operator.getitem(input_val, slices)

    assert not has_dynamic_shape(
        input_val.shape
    ), "Currently we don't support slicing tensor if it has dynamic shape."

    def num_slice_types(slices):
        """
        Gather the number of slice in getitem slices.
        """
        num_slice = 0
        for s in slices:
            if isinstance(s, slice) or isinstance(s, int):
                num_slice += 1
        return num_slice

    def slice_to_trt_params(py_slice, dim_size):
        """
        Convert python slice to TensorRT slice layer parameters.
        """
        start = (py_slice.start % dim_size) if py_slice.start else 0
        stride = py_slice.step if py_slice.step else 1
        stop = (py_slice.stop % dim_size) if py_slice.stop else dim_size
        size = math.ceil((stop - start) * 1.0 / stride)
        return start, size, stride

    if not isinstance(slices, tuple):
        slices = (slices,)

    if network.has_implicit_batch_dimension:
        # Raise an error if it's trying to subscript batch dimension unless it's
        # slice(None, None, None).
        batch_subscript = slices[0]
        if batch_subscript != slice(None, None, None):
            raise RuntimeError(
                f"{name}: Can't subscript batch dimension when it's implicit. Got {slices}"
            )

        # Remove batch_dim subscript
        slices = slices[1:]

    # Replace ellipsis with expanded slices.
    # Compute the number of dim ellipsis represent.
    num_ellipsis = len(input_val.shape) - num_slice_types(slices)
    new_slices = []
    for s in slices:
        if s == Ellipsis:
            while num_ellipsis > 0:
                new_slices.append(slice(None, None, None))
                num_ellipsis -= 1
        else:
            new_slices.append(s)
    slices = new_slices

    # Build trt slice layer params
    start = []
    size = []
    stride = []

    i = 0
    for s in slices:
        if s is None:
            continue

        if isinstance(s, slice):
            params = slice_to_trt_params(s, input_val.shape[i])
            start.append(params[0])
            size.append(params[1])
            stride.append(params[2])
        else:
            start.append(s % input_val.shape[i])
            size.append(1)
            stride.append(1)
        i += 1

    while i < len(input_val.shape):
        start.append(0)
        size.append(input_val.shape[i])
        stride.append(1)
        i += 1

    layer = network.add_slice(
        input=input_val,
        start=start,
        shape=size,
        stride=stride,
    )
    layer.name = name

    # Add shuffle layer to insert dimensions for 'None' and remove dimensions for 'int'.
    if any(not isinstance(s, slice) for s in slices):
        slice_out = layer.get_output(0)
        layer = network.add_shuffle(slice_out)
        final_shape = []
        original_idx = 0
        for s in slices:
            # If it's a slice, keep the dim.
            if isinstance(s, slice):
                final_shape.append(slice_out.shape[original_idx])
                original_idx += 1
            # If it's None, extend the dim.
            elif s is None:
                final_shape.append(1)
            # If it's a int, remove the dim.
            else:
                original_idx += 1
        layer.reshape_dims = tuple(final_shape) + tuple(slice_out.shape)[original_idx:]

    return layer.get_output(0)


@tensorrt_converter(acc_ops.cat)
def acc_ops_cat(network, target, args, kwargs, name):
    tensors = kwargs["tensors"]

    if any(not isinstance(t, trt.tensorrt.ITensor) for t in tensors):
        raise RuntimeError(
            f"cat received inputs {tensors} that is not part " "of the TensorRT region!"
        )

    layer = network.add_concatenation(inputs=tensors)
    layer.axis = kwargs["dim"] - (1 if network.has_implicit_batch_dimension else 0)
    layer.name = name
    return layer.get_output(0)


@tensorrt_converter(acc_ops.matmul)
def acc_ops_matmul(network, target, args, kwargs, name):
    input_val = get_trt_tensor(network, kwargs["input"], f"{name}_input")
    other_val = get_trt_tensor(network, kwargs["other"], f"{name}_other")

    for i in [input_val, other_val]:
        if not isinstance(i, trt.tensorrt.ITensor):
            raise RuntimeError(
                f"matmul received input {i} that is not part " "of the TensorRT region!"
            )

    return add_matrix_multiply_layer(network, input_val, other_val, name)

@tensorrt_converter(acc_ops.sigmoid)
def acc_ops_sigmoid(network, target, args, kwargs, name):
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"Sigmoid received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    layer = network.add_activation(input=input_val, type=trt.ActivationType.SIGMOID)
    layer.name = name
    return layer.get_output(0)


@tensorrt_converter(acc_ops.permute)
def acc_ops_permute(network, target, args, kwargs, name):
    input_val = kwargs["input"]
    permutation = kwargs["permutation"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"permute received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    if network.has_implicit_batch_dimension:
        assert permutation[0] == 0, "Can't permute batch dimension when it's implicit."
        permutation = [i - 1 for i in permutation[1:]]

    layer = network.add_shuffle(input_val)
    layer.second_transpose = tuple(permutation)
    layer.name = name
    return layer.get_output(0)

@tensorrt_converter(acc_ops.quantize_per_tensor)
def acc_ops_quantize_per_tensor(network, target, args, kwargs, name):
    input_val = get_trt_tensor(network, kwargs["input"], f"{name}_input")


    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(f"{name} received input {input_val} that is not part "
                           "of the TensorRT region!")

    q_scale = acc_utils.get_field_from_acc_out_ty(kwargs["acc_out_ty"], "q_scale")
    q_zero_point = acc_utils.get_field_from_acc_out_ty(kwargs["acc_out_ty"], "q_zero_point")
    dtype = acc_utils.get_field_from_acc_out_ty(kwargs["acc_out_ty"], "dtype")
    if dtype not in (torch.quint8, torch.qint8, torch.qint32):
        raise RuntimeError("Only support (torch.quint8, torch.qint8, torch.qint32) "
                           f"quantized type in quantize_per_tensor, get {dtype}.")

    if q_zero_point != 0:
        raise RuntimeError(f"Only support zero_point == 0, get {q_zero_point}")

    scale_layer = network.add_constant((1,), trt.Weights(np.ascontiguousarray([float(q_scale)], dtype=np.float32)))
    scale_layer.name = input_val.name + ".quant.scale"
    scale = scale_layer.get_output(0)
    # assert trt.__version__ > "8.0", "Explicit quantize op is only supported in "
    # "TensorRT 8.0 or above, current TensorRT version:" + trt.__version__
    layer = network.add_quantize(input=input_val, scale=scale)
    layer.axis = 0
    layer.name = input_val.name + ".quant"
    return layer.get_output(0)

@tensorrt_converter(acc_ops.dequantize)
def acc_ops_dequantize(network, target, args, kwargs, name):
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(f"{name} received input {input_val} that is not part "
                           "of the TensorRT region!")

    q_scale = acc_utils.get_field_from_acc_out_ty(kwargs["input_tensor_meta"], "q_scale")
    q_zero_point = acc_utils.get_field_from_acc_out_ty(kwargs["input_tensor_meta"], "q_zero_point")
    dtype = acc_utils.get_field_from_acc_out_ty(kwargs["input_tensor_meta"], "dtype")

    if dtype not in (torch.quint8, torch.qint8, torch.qint32):
        raise RuntimeError("Only support (torch.quint8, torch.qint8, torch.qint32) "
                           f"quantized type in dequantize, get {dtype}.")

    if q_zero_point != 0:
        raise RuntimeError(f"Only support zero_point == 0, get {q_zero_point}")

    scale_layer = network.add_constant((1,), trt.Weights(np.ascontiguousarray([q_scale], dtype=np.float32)))
    scale_layer.name = input_val.name + ".dequant.scale"
    scale = scale_layer.get_output(0)
    # assert trt.__version__ > "8.0", "Explicit dequantize op is only supported in "
    # "TensorRT 8.0 or above, current TensorRT version:" + trt.__version__
    layer = network.add_dequantize(input=input_val, scale=scale)
    layer.name = input_val.name + ".dequant"
    layer.axis = 0
    return layer.get_output(0)
