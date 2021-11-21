from typing import Callable, Any, Tuple, Sequence, Union, List, Optional

import numpy as np
import tensorrt as trt
import torch
from torch.fx.experimental.fx2trt.fx2trt import torch_dtype_from_trt

Target = Union[Callable[..., Any], str]
ShapeType = Union[Sequence[int], trt.Dims]


def get_trt_plugin(
    plugin_name: str,
    field_collection: List[trt.PluginFieldCollection],
    version: str,
    plugin_namespace: str = ""
) -> trt.IPluginV2:
    """
    Get a TensorRT plugin based on the given parameters.

    Args:
        plugin_name (str): Name of the plugin.
        field_collection (List[trt.PluginFieldCollection]): Parameters that needed
            to create a plugin using the plugin creator.
        version (str): Version of the plugin.
        plugin_namespace (str): Namespace of the plugin.

    Returns:
        A TensorRT plugin that can be added to TensorRT network as Plugin layer.
    """
    plugin_registry = trt.get_plugin_registry()
    plugin_creator = plugin_registry.get_plugin_creator(plugin_name, version, plugin_namespace)
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


def set_layer_name(layer: trt.ILayer, target: Target, name: str) -> None:
    """
    Set the TensorRT layer name to "[TensorRT Layer Type]_[Original Op Name]_[FX Node Name with Suffix]"

    Args:
        layer (trt.ILayer): A TensorRT layer of which we want to set the name.
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
    If `val` is not a tuple, then we make a tuple of size `num_elem` by
    replicating `val` `num_elem` times.

    Args:
        val (Any): Value that we want to process.

    Returns:
        A tuple.
    """
    if not isinstance(val, tuple):
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


def has_dynamic_shape(shape: ShapeType) -> bool:
    """
    Determine if the given shape has dynamic dim. i.e. if there're -1 in shape.

    Args:
        shape (ShapeType): Shape of a tensor. Essentially is a sequence of integers.

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
    network: trt.INetworkDefinition,
    value: Union[int, float, torch.Tensor],
    name: str,
    dtype: Optional[torch.dtype],
) -> trt.tensorrt.ITensor:
    """
    Add a TensorRT constant layer whose value is `value` to `network`.

    Args:
        network (trt.INetworkDefinition): A TensorRT network to which we want to add
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
    network: trt.INetworkDefinition,
    input_val: Any,
    name: str,
    dtype: Optional[torch.dtype] = None
) -> trt.tensorrt.ITensor:
    """
    Given a value of random type, we try to convert it to a TensorRT ITensor.
    An runtime error is raised if we're not able to do that.

    Args:
        network (trt.INetworkDefinition): A TensorRT network. If we want to
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
    elif not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"Received input {input_val} of name {name} that "
            "is not part of the TensorRT region!"
        )
    else:
        return input_val


def prepend_ones(
    network: trt.INetworkDefinition,
    tensor: trt.tensorrt.ITensor,
    name: str,
    num_prepend_ones: int,
) -> trt.tensorrt.ITensor:
    """
    Prepend 1s to the shape of TensorRT ITensor `tensor`.

    Args:
        network (trt.INetworkDefinition): The TensorRT network that `tensor`
            belongs to.
        tensor (trt.tensorrt.ITensor): A TensorRT tensor.
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
    network: trt.INetworkDefinition,
    a: trt.tensorrt.ITensor,
    b: trt.tensorrt.ITensor,
    a_name: str,
    b_name: str,
    preset_diff: int = 0
) -> Tuple[trt.tensorrt.ITensor, trt.tensorrt.ITensor]:
    """
    Broadcast two TensorRT tensors to the same number of dimensions by
    prepending 1s to the tensor with less number of dimensions.

    Args:
        network (trt.INetworkDefinition): TensorRT network object.
        a (trt.tensorrt.ITensor): A TensorRT ITensor.
        b (trt.tensorrt.ITensor): A TensorRT ITensor.
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
    network: trt.INetworkDefinition,
    lhs_val: Union[int, float, trt.tensorrt.ITensor, torch.Tensor],
    rhs_val: Union[int, float, trt.tensorrt.ITensor, torch.Tensor],
    op_type: trt.ElementWiseOperation,
    target: Target,
    name: str
) -> trt.tensorrt.ITensor:
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
        network (trt.INetworkDefinition): TensorRT network object.
        lhs_val (trt.tensorrt.ITensor): Left operand of the binary operation. Could
            be a TensorRT tensor, a PyTorch tensor or a simple value.
        rhs_val (trt.tensorrt.ITensor): Right operand of the binary operation. Similar
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
    network: trt.INetworkDefinition,
    input_val: trt.tensorrt.ITensor,
    operation_type: trt.UnaryOperation,
    target: Target,
    name: str,
) -> trt.tensorrt.ITensor:
    """
    Add a TensorRT Unary layer to `network`.

    Args:
        network (trt.INetworkDefinition): TensorRT network object.
        input_val (trt.tensorrt.ITensor): Input to the unary op. Must be a TensorRT tensor.
        op_type (trt.ElementWiseOperation): Type of the TensorRT unary operation.
        target (Target): Target of fx node.
        name (str): The name we want to assign to the created TensorRT layer.

    Returns:
        The output of TensorRT Unary layer.
    """
    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"{operation_type} received input {input_val} that is not part "
            "of the TensorRT region!"
        )
    layer = network.add_unary(input_val, operation_type)
    set_layer_name(layer, target, name)
    return layer.get_output(0)


def add_activation_layer(
    network: trt.INetworkDefinition,
    input_val: trt.tensorrt.ITensor,
    operation_type: trt.ActivationType,
    target: Target,
    name: str,
    alpha: Optional[Any] = None,
    beta: Optional[Any] = None,
) -> trt.tensorrt.ITensor:
    """
    Add a TensorRT Activation layer to `network`.

    Args:
        network (trt.INetworkDefinition): TensorRT network object.
        input_val (trt.tensorrt.ITensor): Input to the activation op.
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
    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"{operation_type} received input {input_val} that is not part "
            "of the TensorRT region!"
        )
    layer = network.add_activation(input_val, operation_type)
    if alpha:
        layer.alpha = alpha
    if beta:
        layer.beta = beta
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
