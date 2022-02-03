import operator
import warnings
from typing import Any, Tuple, Sequence, Union, List, Optional, Dict, Callable

import numpy as np
import tensorrt as trt
import torch
from torch.fx.node import Target, Argument
from torch.fx.experimental.fx2trt.types import *  # noqa: F403
from torch.fx.experimental.fx2trt.utils import torch_dtype_from_trt


def get_trt_plugin(
    plugin_name: str,
    field_collection: List[TRTPluginFieldCollection],
    version: str,
    plugin_namespace: str = ""
) -> TRTPlugin:
    """
    Get a TensorRT plugin based on the given parameters.

    Args:
        plugin_name (str): Name of the plugin.
        field_collection (List[TRTPluginFieldCollection]): Parameters that needed
            to create a plugin using the plugin creator.
        version (str): Version of the plugin.
        plugin_namespace (str): Namespace of the plugin.

    Returns:
        A TensorRT plugin that can be added to TensorRT network as Plugin layer.
    """
    plugin_registry = trt.get_plugin_registry()
    plugin_creator = plugin_registry.get_plugin_creator(plugin_name, version, plugin_namespace)
    assert plugin_creator, f"Unabled to find plugin creator with name {plugin_name}"
    plugin = plugin_creator.create_plugin(name=plugin_name, field_collection=field_collection)

    assert plugin is not None, f"Plugin: {plugin_name} could not be fetched"
    return plugin


def get_positive_dim(dim: int, dim_size: int) -> int:
    """
    Given an integer number that represents a dimension in the array,
    transform it to a positive integer dim if it's negative. Otherwise, do
    nothing.

    Args:
        dim (int): A integer number that represents a dimension in an array.
        dim_size (int): The size of the dimension in the array.

    Returns:
        A positive integer that represent the same dimension as the given dim.
    """
    if dim < 0:
        return dim % dim_size
    return dim


def set_layer_name(layer: TRTLayer, target: Target, name: str) -> None:
    """
    Set the TensorRT layer name to "[TensorRT Layer Type]_[Original Op Name]_[FX Node Name with Suffix]"

    Args:
        layer (TRTLayer): A TensorRT layer of which we want to set the name.
        target (Target): A fx node.target. For call_function node, it's the function that
            the node represents.
        name (str): Consists of fx node.name with optional suffix.
    """
    target_name = target if isinstance(target, str) else f"acc_ops.{target.__name__}"
    layer.name = f"[{layer.type.name}]-[{target_name}]-[{name}]"


def extend_attr_to_tuple(
    val: Any,
    num_elem: int,
) -> Tuple[Any, ...]:
    """
    If `val` is not a tuple or a list, then we make a tuple of size `num_elem` by
    replicating `val` `num_elem` times.

    Args:
        val (Any): Value that we want to process.

    Returns:
        A tuple.
    """
    if not isinstance(val, (tuple, list)):
        val = (val,) * num_elem
    return val


def extend_mod_attr_to_tuple(mod: torch.nn.Module, name: str, size: int):
    """
    Extend an attribute of `mod` that named `name` to a tuple of `size`.
    """
    val = getattr(mod, name)
    return extend_attr_to_tuple(val, size)


def to_numpy(tensor: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    """
    Convert a PyTorch Tensor to a Numpy Array. If the tensor is
    quantized it will be dequantized first.

    Args:
        tensor (Optional[torch.Tensor]): A PyTorch tensor or None.

    Returns:
        A Numpy array.
    """

    if tensor is None:
        return tensor

    assert isinstance(tensor, torch.Tensor), f"to_numpy can only be called on None or a torch.Tensor, got: {tensor}"
    if tensor.is_quantized:
        tensor = tensor.dequantize()

    return tensor.cpu().detach().contiguous().numpy()


def has_dynamic_shape(shape: Shape) -> bool:
    """
    Determine if the given shape has dynamic dim. i.e. if there're -1 in shape.

    Args:
        shape (Shape): Shape of a tensor. Essentially is a sequence of integers.

    Returns:
        A boolean value indicates whether there's dynamic dim in the shape.
    """
    return any(s == -1 for s in shape)


def get_axes_for_reduce_op(
    dim: Union[int, Sequence[int]],
    has_implicit_batch_dimension: bool,
) -> int:
    """
    TensorRT reduce layer relies on the binary representation of axes to
    determine which dims to reduce. For example, if we want to reduce on
    dim 1 and 2 then axes should be 6(110).

    Args:
        dim (Union[int, Sequence[int]]): An integer or a sequence of integers
            that will be used to generate axes for TensorRT.
        has_implicit_batch_dimension (bool): Whether the TensorRT network is
            using implicit batch dimension.

    Returns:
        An integer which binary form can be used as axes for TensorRT reduce
        layer.
    """
    if isinstance(dim, int):
        dim = (dim,)

    if has_implicit_batch_dimension:
        assert 0 not in dim, "Can't reduce over batch dimension when it's implicit."

    axes = 0
    for d in dim:
        axes |= 1 << (d - (1 if has_implicit_batch_dimension else 0))

    return axes


def create_constant(
    network: TRTNetwork,
    value: Union[int, float, torch.Tensor],
    name: str,
    dtype: Optional[torch.dtype],
) -> TRTTensor:
    """
    Add a TensorRT constant layer whose value is `value` to `network`.

    Args:
        network (TRTNetwork): A TensorRT network to which we want to add
            a constant layer.
        value (Union[int, float, torch.Tensor]): A literal value or a PyTorch tensor
            that will be used as value of the added TensorRT Constant layer.
        name (str): Name of the added TensorRT Constant layer.
        dtype (Optional[torch.dtype]): If a dtype is given, we will convert the type
            of the given `value` to this dtype.

    Returns:
        A TensorRT ITensor that represents the given value.
    """
    if isinstance(value, int):
        value = torch.IntTensor([value])

    if isinstance(value, float):
        value = torch.Tensor([value])

    if dtype:
        value = value.to(dtype)

    constant = network.add_constant(value.shape, to_numpy(value))
    constant.name = name
    return constant.get_output(0)


def get_trt_tensor(
    network: TRTNetwork,
    input_val: Any,
    name: str,
    dtype: Optional[torch.dtype] = None
) -> TRTTensor:
    """
    Given a value of random type, we try to convert it to a TensorRT ITensor.
    An runtime error is raised if we're not able to do that.

    Args:
        network (TRTNetwork): A TensorRT network. If we want to
            add a TensorRT Constant layer, we will add it to this network.
        input_val (Any): An value that we want to convert to a TensorRT ITensor.
        name (str): The name of the created TensorRT Constant layer if there's
            one.
        dtype (Optional[torch.dtype]): If dtype is provided, the given value
            will be converted to this dtype.

    Returns:
        A TensorRT ITensor that represents the given value.
    """
    if isinstance(input_val, (torch.Tensor, int, float)):
        return create_constant(network, input_val, name, dtype)
    elif not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"Received input {input_val} of name {name} that "
            "is not part of the TensorRT region!"
        )
    else:
        return input_val


def prepend_ones(
    network: TRTNetwork,
    tensor: TRTTensor,
    name: str,
    num_prepend_ones: int,
) -> TRTTensor:
    """
    Prepend 1s to the shape of TensorRT ITensor `tensor`.

    Args:
        network (TRTNetwork): The TensorRT network that `tensor`
            belongs to.
        tensor (TRTTensor): A TensorRT tensor.
        name (str): Name of the TensorRT Shuffle layer which is used to prepend
            1s.
        num_prepend_ones (int): Number of 1s that will be prepend.

    Returns:
        A Tensorrt ITensor which contains the same value as `tensor` but with
        more 1s prepended to the beginning of `tensor` shape.
    """
    layer = network.add_shuffle(tensor)

    # If there're dynamic dim in tensor's shape, we need to use shape layer to
    # compute the final shape.
    if has_dynamic_shape(tensor.shape):
        tensor_shape_layer = network.add_shape(tensor)
        tensor_shape_layer.name = f"{name}_broadcast_orig_shape"
        prepend_shape_layer = network.add_constant(
            (num_prepend_ones,), np.ones((num_prepend_ones,), dtype=np.int32)
        )
        prepend_shape_layer.name = f"{name}_broadcast_prepend_ones"
        reshape_dim_layer = network.add_concatenation(
            [prepend_shape_layer.get_output(0), tensor_shape_layer.get_output(0)]
        )
        reshape_dim_layer.axis = 0
        reshape_dim_layer.name = f"{name}_broadcast_final_shape"
        layer.set_input(1, reshape_dim_layer.get_output(0))
    else:
        layer.reshape_dims = (1,) * num_prepend_ones + tuple(tensor.shape)

    layer.name = name
    return layer.get_output(0)


def broadcast(
    network: TRTNetwork,
    a: TRTTensor,
    b: TRTTensor,
    a_name: str,
    b_name: str,
    preset_diff: int = 0
) -> Tuple[TRTTensor, TRTTensor]:
    """
    Broadcast two TensorRT tensors to the same number of dimensions by
    prepending 1s to the tensor with less number of dimensions.

    Args:
        network (TRTNetwork): TensorRT network object.
        a (TRTTensor): A TensorRT ITensor.
        b (TRTTensor): A TensorRT ITensor.
        a_name (str): Name of tensor a.
        b_name (str): Name of tensor b.
        preset_diff (int): The difference of number of dimensions after broadcast.
            A positive number means after broadcast, tensor `a` would have `preset_diff`
            more dimensions than `b`. This is used in matmul, since we need to broadcast
            tensors but not always to the same number of dimension. The reason is that
            matmul supports Matrix x Vector and in this case broadcasted vector should
            have 1 less number of dimensions than the matrix tensor.

    Returns:
        Two TensorRT ITensors that are broadcasted to the same number of dimensions.
    """
    a_shape = tuple(a.shape)
    b_shape = tuple(b.shape)

    diff = len(a_shape) - len(b_shape) - preset_diff
    if diff > 0:
        b = prepend_ones(network, b, f"{b_name}_broadcast", diff)
    elif diff < 0:
        a = prepend_ones(network, a, f"{a_name}_broadcast", -diff)

    return a, b


def add_binary_elementwise_layer(
    network: TRTNetwork,
    lhs_val: Union[int, float, TRTTensor, torch.Tensor],
    rhs_val: Union[int, float, TRTTensor, torch.Tensor],
    op_type: trt.ElementWiseOperation,
    target: Target,
    name: str
) -> TRTTensor:
    """
    This function adds a TensorRT elementwise layer. We allow both operands to be
    constant (not a trt tensor) because in implicit batch dimension mode, we could
    introduce constant via .size() op. Other scenario should be const folded first.
    If any operand is not a trt tensor, we make it a trt constant layer which has
    the same type as the other trt tensor. Then we broadcast these two inputs to
    have the same number of dimensions.

    Limitation:
        If we are using implicit batch dim mode, the operand that is not a trt
    tensor are not allowed to have larger ranks than the trt tensor operand.

    Args:
        network (TRTNetwork): TensorRT network object.
        lhs_val (TRTTensor): Left operand of the binary operation. Could
            be a TensorRT tensor, a PyTorch tensor or a simple value.
        rhs_val (TRTTensor): Right operand of the binary operation. Similar
            to lhs_val.
        op_type (trt.ElementWiseOperation): Type of the TensorRT elementwise binary operation.
        target (Target): Target of fx node.
        name (str): The name we want to assign to the created TensorRT layer.

    Returns:
        The output of TensorRT Elementwise layer.
    """
    dtype = None
    is_lhs_trt_tensor = False
    is_rhs_trt_tensor = False
    if isinstance(lhs_val, TRTTensor):
        dtype = torch_dtype_from_trt(lhs_val.dtype)
        is_lhs_trt_tensor = True
    if isinstance(rhs_val, TRTTensor):
        dtype = torch_dtype_from_trt(rhs_val.dtype)
        is_rhs_trt_tensor = True
    if not is_lhs_trt_tensor and not is_rhs_trt_tensor:
        warnings.warn(f"Both operands of the binary elementwise op {name} "
                      "are constant. In this case, please consider constant fold the model first.")
        return get_python_op_from_trt_elementwise_op(op_type)(lhs_val, rhs_val)

    lhs_val = get_trt_tensor(network, lhs_val, f"{name}_lhs", dtype)
    rhs_val = get_trt_tensor(network, rhs_val, f"{name}_rhs", dtype)

    # Check the limitation in the doc string.
    if network.has_implicit_batch_dimension:
        if is_lhs_trt_tensor and not is_rhs_trt_tensor:
            assert len(lhs_val.shape) >= len(rhs_val.shape), f"{lhs_val.shape} >= {rhs_val.shape}"
        elif not is_lhs_trt_tensor and is_rhs_trt_tensor:
            assert len(rhs_val.shape) >= len(lhs_val.shape), f"{rhs_val.shape} >= {lhs_val.shape}"

    lhs_val, rhs_val = broadcast(
        network, lhs_val, rhs_val, f"{name}_lhs", f"{name}_rhs"
    )
    layer = network.add_elementwise(lhs_val, rhs_val, op_type)
    set_layer_name(layer, target, name)
    return layer.get_output(0)


def add_unary_layer(
    network: TRTNetwork,
    input_val: TRTTensor,
    operation_type: trt.UnaryOperation,
    target: Target,
    name: str,
) -> TRTTensor:
    """
    Add a TensorRT Unary layer to `network`.

    Args:
        network (TRTNetwork): TensorRT network object.
        input_val (TRTTensor): Input to the unary op. Must be a TensorRT tensor.
        op_type (trt.ElementWiseOperation): Type of the TensorRT unary operation.
        target (Target): Target of fx node.
        name (str): The name we want to assign to the created TensorRT layer.

    Returns:
        The output of TensorRT Unary layer.
    """
    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"{operation_type} received input {input_val} that is not part "
            "of the TensorRT region!"
        )
    layer = network.add_unary(input_val, operation_type)
    set_layer_name(layer, target, name)
    return layer.get_output(0)


def add_activation_layer(
    network: TRTNetwork,
    input_val: TRTTensor,
    operation_type: trt.ActivationType,
    target: Target,
    name: str,
    alpha: Optional[Any] = None,
    beta: Optional[Any] = None,
) -> TRTTensor:
    """
    Add a TensorRT Activation layer to `network`.

    Args:
        network (TRTNetwork): TensorRT network object.
        input_val (TRTTensor): Input to the activation op.
            Must be a TensorRT tensor.
        op_type (trt.ElementWiseOperation): Type of the TensorRT activation
            operation.
        target (Target): Target of fx node.
        name (str): The name we want to assign to the created TensorRT layer.
        alpha (Optional[Any]): If not None, we will use it to set the alpha
            attribute of the created TensorRT activation layer.
        beta (Optional[Any]): If not None, we will use it to set the beta
            attribute of the created TensorRT activation layer.

    Returns:
        The output of TensorRT Activation layer.
    """
    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"{operation_type} received input {input_val} that is not part "
            "of the TensorRT region!"
        )
    layer = network.add_activation(input_val, operation_type)
    if alpha is not None:
        layer.alpha = alpha
    if beta is not None:
        layer.beta = beta
    set_layer_name(layer, target, name)
    return layer.get_output(0)


def add_reduce_layer(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    operation_type: trt.ActivationType,
    name: str,
) -> TRTTensor:
    """
    Add a TensorRT Reduce layer to `network`.

    Args:
        network (TRTNetwork): TensorRT network object.
        target (Target): Target of fx node.
        args (Tuple[Argument, ...]): Args of the fx node.
        kwargs (Dict[str, Argument]): Kwargs of the fx node.
        operation_type (trt.ElementWiseOperation): Type of the TensorRT activation
            operation.
        name (str): The name we want to assign to the created TensorRT layer.

    Returns:
        The output of TensorRT Reduce layer.
    """
    input_val = kwargs["input"]
    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"{name} received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    # If dim is specified, then the op is reducing over certain dimensions.
    # Otherwise, it's reducing over all elements, which is only supported in
    # explicit batch dimension.
    if "dim" not in kwargs:
        assert (
            not network.has_implicit_batch_dimension
        ), f"We don't support reduce({name}) over all the elements if batch dim is implicit."
        dim = range(0, len(input_val.shape))
    else:
        dim = kwargs["dim"]  # type: ignore[assignment]

    keepdim = False if "keepdim" not in kwargs else kwargs["keepdim"]
    layer = network.add_reduce(
        input_val,
        operation_type,
        get_axes_for_reduce_op(dim, network.has_implicit_batch_dimension),
        keepdim,
    )
    set_layer_name(layer, target, name)
    return layer.get_output(0)


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
        output_val = layer.get_output(i)
        output_val.dynamic_range = dynamic_range
        layer.set_output_type(i, trt.int8)
        # output_val.dtype = trt.int8


def get_inputs_from_args_and_kwargs(args, kwargs, input_names):
    inputs = []
    for i, key in enumerate(input_names):
        if key not in kwargs:
            inputs.append(args[i])
        else:
            inputs.append(kwargs[key])
    return inputs


def sign(
    network: TRTNetwork,
    input_val: TRTTensor,
    target: Target,
    name: str
) -> TRTTensor:
    """
    Sign is calculated as below:
       x = input
       sign = (exp(x) // exp(abs(x))) * 2 - 1
       For positive number and 0, (exp(x) // exp(abs(x))) yield 1; for negative number, (exp(x) // exp(abs(x))) yield 0.
       With multiply 2, the value become 2(for pos and 0) and 0(for neg).
       Finally minus 1, the value become 1(for pos and 0) and -1(for neg).

    Args:
        network (TRTNetwork): TensorRT network object.
        input_val (TRTTensor): The input tensor.
        target (Target): fx node target.
        name (str): Name of the fx node with optional suffix.

    Returns:
        A TensorRT tensor represent the result of sign operator.
    """
    input_exp_output = add_unary_layer(network, input_val, trt.UnaryOperation.EXP, target, f"{name}_prod_exp")
    input_abs_output = add_unary_layer(network, input_val, trt.UnaryOperation.ABS, target, f"{name}_prod_abs")
    input_abs_exp_output = add_unary_layer(network, input_abs_output, trt.UnaryOperation.EXP, target, f"{name}_prod_abs_exp")
    floor_div_output = add_binary_elementwise_layer(network, input_exp_output, input_abs_exp_output,
                                                    trt.ElementWiseOperation.FLOOR_DIV, target, f"{name}_exp_floor_div")
    double_floor_div_output = add_binary_elementwise_layer(network, floor_div_output, 2,
                                                           trt.ElementWiseOperation.PROD, target, f"{name}_floor_div*2")
    return add_binary_elementwise_layer(network, double_floor_div_output, 1,
                                        trt.ElementWiseOperation.SUB, target, f"{name}_sign")


def trunc_div(
    input: TRTTensor,
    other: TRTTensor,
    network: TRTNetwork,
    target: Target,
    name: str
) -> TRTTensor:
    """
    Perform trunc divide on Tensor, result of divide will be round toward zero.
    This means for positive number, it will be floor round; for negative number,
    it will be ceil round. Example: [2.1, 0.8, -3.2] -> [2, 0, -3].

    Args:
        input: divisor.
        other: dividend.
        network: INetworkDefinition.
        target: node target.
        name: namespace for the op

    Returns:
        A TensorRT tensor represent the result of trunc divide.
    """
    prod_output = add_binary_elementwise_layer(network, input, other, trt.ElementWiseOperation.PROD, target, f"{name}_prod")
    sign_output = sign(network, prod_output, target, name)

    # Convert constant input into ITensor for UnaryOperation
    if not isinstance(input, trt.tensorrt.ITensor):
        input = get_trt_tensor(network, input, f"{name}_input")
    if not isinstance(other, trt.tensorrt.ITensor):
        other = get_trt_tensor(network, other, f"{name}_other", dtype=torch_dtype_from_trt(input.dtype))

    abs_input_output = add_unary_layer(network, input, trt.UnaryOperation.ABS, target, f"{name}_abs_input")
    abs_other_output = add_unary_layer(network, other, trt.UnaryOperation.ABS, target, f"{name}_abs_other")
    abs_floor_output = add_binary_elementwise_layer(network, abs_input_output, abs_other_output,
                                                    trt.ElementWiseOperation.FLOOR_DIV, target, f"{name}_floor_div")
    output = add_binary_elementwise_layer(network, abs_floor_output, sign_output,
                                          trt.ElementWiseOperation.PROD, target, f"{name}_output")

    return output


def get_python_op_from_trt_elementwise_op(trt_op: TRTElementWiseOp) -> Callable[[Any, Any], Any]:
    if trt_op == trt.ElementWiseOperation.SUM:
        return operator.add
    elif trt_op == trt.ElementWiseOperation.PROD:
        return operator.mul
    elif trt_op == trt.ElementWiseOperation.SUB:
        return operator.sub
    elif trt_op == trt.ElementWiseOperation.DIV:
        return operator.truediv
    elif trt_op == trt.ElementWiseOperation.FLOOR_DIV:
        return operator.floordiv
    else:
        raise RuntimeError(f"{trt_op} is not supported yet!")
