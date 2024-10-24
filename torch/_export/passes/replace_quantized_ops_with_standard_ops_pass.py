# mypy: allow-untyped-defs
import logging
import operator
from typing import List, Optional, Tuple, Union

import torch
import torch.export._trace
from torch._ops import OpOverload
from torch.ao.quantization.fx._decomposed import (
    dequantize_per_channel,
    dequantize_per_tensor,
    quantize_per_tensor,
)
from torch.ao.quantization.utils import calculate_qmin_qmax
from torch.fx.graph_module import _assign_attr


log = logging.getLogger(__name__)

# Those values will need to be carried over multiple operators.
_INPUT_Q_DTYPE: Optional[Union[torch.dtype, torch.fx.Node]] = None
_SCALE: Optional[Union[float, torch.fx.Node]] = None
_ZERO_POINT: Optional[Union[float, torch.fx.Node]] = None


def int_to_valid_dtype(val: int) -> torch.dtype:
    from torch._export.converter import _TORCH_ENUM_TO_DTYPE  # No circular import.

    if isinstance(val, torch.dtype):
        return val
    dtype = _TORCH_ENUM_TO_DTYPE[val]
    if dtype == torch.quint8:
        return torch.uint8
    elif dtype == torch.qint8:
        return torch.int8
    return dtype


def fx_enum_to_dtype(gm: torch.fx.GraphModule, val: int) -> torch.fx.Node:
    return gm.graph.call_function(int_to_valid_dtype, (val,))


def insert_quantized_node(
    gm: torch.fx.GraphModule,
    val_node: torch.fx.Node,
    scale_node: Union[float, torch.fx.Node],
    zero_point_node: Union[float, torch.fx.Node],
    qmin_node: Union[float, int, torch.fx.Node],
    qmax_node: Union[float, int, torch.fx.Node],
    dtype_node: Union[torch.dtype, torch.fx.Node],
    qscheme: Optional[torch.qscheme],
) -> torch.fx.Node:
    return gm.graph.call_function(
        quantize_per_tensor,
        (
            val_node,
            scale_node,
            zero_point_node,
            qmin_node,
            qmax_node,
            dtype_node,
        ),
    )


def get_dequantized(
    val: torch.Tensor,
    scale: Union[float, torch.Tensor],
    zero_point: Union[float, torch.Tensor],
    qmin: Union[float, int],
    qmax: Union[float, int],
    dtype: torch.dtype,
    axis: Optional[int],
    qscheme: Optional[torch.qscheme],
) -> torch.Tensor:
    if qscheme is torch.per_tensor_affine:
        return dequantize_per_tensor(
            val,
            scale,
            zero_point,
            qmin,
            qmax,
            dtype,
        )
    elif qscheme is torch.per_channel_affine:
        return dequantize_per_channel(
            val,
            scale,
            zero_point,
            axis,
            qmin,
            qmax,
            dtype,
        )
    else:
        raise RuntimeError(f"Unsupported dequantization scheme: {qscheme}")


def insert_dequantized_node(
    gm: torch.fx.GraphModule,
    val_node: torch.fx.Node,
    scale_node: Union[float, torch.fx.Node],
    zero_point_node: Union[float, torch.fx.Node],
    qmin_node: Union[float, int, torch.fx.Node],
    qmax_node: Union[float, int, torch.fx.Node],
    dtype_node: Union[torch.dtype, torch.fx.Node],
    axis_node: Optional[Union[int, torch.fx.Node]],
    qscheme: Optional[torch.qscheme],
) -> torch.fx.Node:
    if qscheme is torch.per_tensor_affine:
        return gm.graph.call_function(
            dequantize_per_tensor,
            (
                val_node,
                scale_node,
                zero_point_node,
                qmin_node,
                qmax_node,
                dtype_node,
            ),
        )
    elif qscheme is torch.per_channel_affine:
        return gm.graph.call_function(
            dequantize_per_channel,
            (
                val_node,
                scale_node,
                zero_point_node,
                axis_node,
                qmin_node,
                qmax_node,
                dtype_node,
            ),
        )
    else:
        raise RuntimeError(f"Unsupported dequantization scheme: {qscheme}")


def get_qmin_qmax(dtype: torch.dtype) -> Tuple[Union[int, float], Union[int, float]]:
    return calculate_qmin_qmax(None, None, False, dtype, False)  # type: ignore[arg-type]


def insert_qmin_qmax_node(
    gm: torch.fx.GraphModule, dtype_node: Union[torch.dtype, torch.fx.Node]
) -> Tuple[torch.fx.Node, torch.fx.Node]:
    q_min_max_node = gm.graph.call_function(
        calculate_qmin_qmax, (None, None, False, dtype_node, False)
    )
    qmin_node = gm.graph.call_function(operator.getitem, (q_min_max_node, 0))
    qmax_node = gm.graph.call_function(operator.getitem, (q_min_max_node, 1))
    return qmin_node, qmax_node


def get_script_object(
    gm: torch.nn.Module, node: torch.fx.Node
) -> torch._C.ScriptObject:
    assert isinstance(node, torch.fx.Node)
    assert node.op == "get_attr"
    attr_name = node.target
    assert isinstance(attr_name, str)

    mod = gm
    for attr in attr_name.split("."):
        mod = getattr(mod, attr)
    assert isinstance(mod, torch._C.ScriptObject)
    return mod


def insert_weight_and_bias_get_attr_node_from_get_attr_to_scriptobject(
    gm: torch.fx.GraphModule,
    param_node: torch.fx.Node,
) -> Tuple[torch.fx.Node, Optional[torch.fx.Node]]:
    """Directly inline tensor from a get_attr fx node."""
    mod = get_script_object(gm, param_node)
    w_qtensor, b_qtensor = mod.unpack()  # type: ignore[attr-defined]
    w_attr_name, b_attr_name = (
        f"dequantized_{param_node.target}_w",
        f"dequantized_{param_node.target}_b",
    )
    return insert_weight_and_bias_get_attr_node(
        gm, w_qtensor, b_qtensor, w_attr_name, b_attr_name
    )


def insert_weight_and_bias_get_attr_node_from_get_attr_to_qtensor(
    gm: torch.fx.GraphModule,
    get_attr_to_weight_node: torch.fx.Node,
    get_attr_to_bias_node: Optional[torch.fx.Node],
) -> Tuple[torch.fx.Node, Optional[torch.fx.Node]]:
    assert isinstance(get_attr_to_weight_node.target, str)
    w_qtensor = getattr(gm, get_attr_to_weight_node.target)
    w_attr_name = f"dequantized_{get_attr_to_weight_node.target}_w"

    if get_attr_to_bias_node is not None:
        assert isinstance(get_attr_to_bias_node.target, str)
        b_qtensor = getattr(gm, get_attr_to_bias_node.target)
        b_attr_name = f"dequantized_{get_attr_to_bias_node.target}_b"
    else:
        b_qtensor, b_attr_name = None, ""

    return insert_weight_and_bias_get_attr_node(
        gm, w_qtensor, b_qtensor, w_attr_name, b_attr_name
    )


def insert_weight_and_bias_get_attr_node(
    gm: torch.fx.GraphModule,
    w_qtensor: torch.Tensor,
    b_qtensor: Optional[torch.Tensor],
    w_attr_name: str,
    b_attr_name: str,
) -> Tuple[torch.fx.Node, Optional[torch.fx.Node]]:
    w_tensor = get_tensor_from_qtensor(w_qtensor)
    _assign_attr(w_tensor, gm, w_attr_name)
    w_tensor_attr = gm.graph.get_attr(w_attr_name)

    if b_qtensor is not None:
        b_tensor = get_tensor_from_qtensor(b_qtensor, dequant=False)
        _assign_attr(b_tensor, gm, b_attr_name)
        b_tensor_attr = gm.graph.get_attr(b_attr_name)
    else:
        b_tensor_attr = None

    return w_tensor_attr, b_tensor_attr


def get_tensor_from_qtensor(
    qtensor: torch.Tensor, dequant: bool = True
) -> torch.Tensor:
    # Manual conversion because qint8 is not used anymore.
    if qtensor.dtype in [torch.qint8, torch.quint8]:
        tensor = qtensor.int_repr()
    else:
        tensor = qtensor

    # Weights need dequantization with scaling and zero_point adjustment, but
    # bias does not need that.
    if dequant:
        qscheme = qtensor.qscheme()
        if qscheme == torch.per_channel_affine:
            scale, zero_point, axis = (
                qtensor.q_per_channel_scales(),
                qtensor.q_per_channel_zero_points(),
                qtensor.q_per_channel_axis(),
            )
        else:
            scale, zero_point, axis = (
                qtensor.q_scale(),  # type: ignore[assignment]
                qtensor.q_zero_point(),  # type: ignore[assignment]
                None,
            )
        dtype = tensor.dtype
        qmin, qmax = get_qmin_qmax(dtype)
        return get_dequantized(
            tensor, scale, zero_point, qmin, qmax, dtype, axis, qscheme
        )
    return tensor


def insert_fused_activation_node(
    gm: torch.fx.GraphModule, opname: str, fx_node: torch.fx.Node
) -> torch.fx.Node:
    if opname in ["conv1d_relu", "conv2d_relu", "linear_relu", "add_relu", "mul_relu"]:
        fx_node = gm.graph.call_function(torch.ops.aten.relu, (fx_node,))
    return fx_node


def _conv1d_op_with_squeeze(
    inp: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
) -> torch.Tensor:
    # In quantized version, conv1d is emulated using conv2d with squeeze and unsqueeze
    # operations before and after the conv2d operation to match the dimension of weights.
    # Reference: https://github.com/pytorch/pytorch/blob/eca0cb0fbe84bb0a34fa94afe261bceecd52c436/aten/src/ATen/native/quantized/cpu/qconv.cpp#L1827  # noqa: B950
    s_inp = torch.ops.aten.unsqueeze(inp, 2)
    conv1d_res = torch.ops.aten.conv2d(
        s_inp,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
    )
    uns_conv1d_res = torch.ops.aten.squeeze(conv1d_res, 2)
    return uns_conv1d_res


def _transform_conv_with_packedparam(gm: torch.fx.GraphModule, node: torch.fx.Node):
    """Conv specfic transformation function."""
    assert isinstance(node.target, torch._ops.OpOverload)
    opname = node.target._opname
    scale_node, zero_point_node = node.args[2], node.args[3]

    op_f = (
        torch.ops.aten.conv2d
        if opname in ["conv2d", "conv2d_relu"]
        else _conv1d_op_with_squeeze
    )

    inp_node, param_node = node.args[0], node.args[1]
    assert isinstance(inp_node, torch.fx.Node)
    assert isinstance(param_node, torch.fx.Node)

    if param_node.op == "call_function":
        # Using Conv2dPrepackParam from conv_prepack.
        # We directly skip the packing call and inline weights and bias.
        w_node, b_node = param_node.args[0], param_node.args[1]
        assert isinstance(w_node, torch.fx.Node)
        assert b_node is None or isinstance(b_node, torch.fx.Node)
        (
            param_0,
            param_1,
        ) = insert_weight_and_bias_get_attr_node_from_get_attr_to_qtensor(
            gm, w_node, b_node
        )
        op_res_node = gm.graph.call_function(
            op_f, (inp_node, param_0, param_1, *param_node.args[2:])
        )
    else:
        # Using ConvPrepackedParam.
        param = get_script_object(gm, param_node)
        (
            param_0,
            param_1,
        ) = insert_weight_and_bias_get_attr_node_from_get_attr_to_scriptobject(
            gm, param_node
        )  # type: ignore[assignment]
        op_res_node = gm.graph.call_function(
            op_f,
            (
                inp_node,
                param_0,
                param_1,
                param.stride(),  # type: ignore[attr-defined]
                param.padding(),  # type: ignore[attr-defined]
                param.dilation(),  # type: ignore[attr-defined]
                param.groups(),  # type: ignore[attr-defined]
            ),
        )
    return op_res_node, scale_node, zero_point_node


def _transform_linear_with_packedparam(gm: torch.fx.GraphModule, node: torch.fx.Node):
    """Linear specfic transformation function."""
    scale_node, zero_point_node = node.args[2], node.args[3]

    inp_node, param_node = node.args[0], node.args[1]
    assert isinstance(inp_node, torch.fx.Node)
    assert isinstance(param_node, torch.fx.Node)

    if param_node.op == "call_function":
        # Using LinearPrepackParam from linear_prepack.
        # We directly skip the packing call and inline weights and bias.
        w_node, b_node = param_node.args[0], param_node.args[1]
        assert isinstance(w_node, torch.fx.Node)
        assert b_node is None or isinstance(b_node, torch.fx.Node)
        (
            param_0,
            param_1,
        ) = insert_weight_and_bias_get_attr_node_from_get_attr_to_qtensor(
            gm, w_node, b_node
        )
        op_res_node = gm.graph.call_function(
            torch.ops.aten.linear, (inp_node, param_0, param_1, *param_node.args[2:])
        )
    else:
        # Using LinearPackedParams.
        (
            param_0,
            param_1,
        ) = insert_weight_and_bias_get_attr_node_from_get_attr_to_scriptobject(
            gm, param_node
        )  # type: ignore[assignment]
        op_res_node = gm.graph.call_function(
            torch.ops.aten.linear, (inp_node, param_0, param_1)
        )
    return op_res_node, scale_node, zero_point_node


def _transform_op_where_last_two_arguments_are_scale_and_zero_point(
    gm: torch.fx.GraphModule, node: torch.fx.Node
):
    """
    This transformation function can be used for function where the last two
    parameters are scale and zero point. Additionally, the function's parameters
    do not need any unpacking.
    """
    to_standard_op = {
        "mul": torch.ops.aten.mul,
        "mul_relu": torch.ops.aten.mul,
        "add": torch.ops.aten.add,
        "add_relu": torch.ops.aten.add,
        "softmax": torch.ops.aten.softmax,
        "cat": torch.ops.aten.cat,
        "hardswish": torch.ops.aten.hardswish,
    }

    assert isinstance(node.target, torch._ops.OpOverload)
    opname, args = node.target._opname, node.args
    scale_node, zero_point_node = args[-2], args[-1]
    op_res_node = gm.graph.call_function(to_standard_op[opname], tuple(args[:-2]))
    return op_res_node, scale_node, zero_point_node


def _transform_scalar_arithmetic(gm: torch.fx.GraphModule, node: torch.fx.Node):
    """Transform scalar overload for basic arithmetic."""
    to_standard_op = {
        "mul": torch.ops.aten.mul.Scalar,
        "add": torch.ops.aten.add.Scalar,
    }
    assert isinstance(node.target, torch._ops.OpOverload)
    opname, args = node.target._opname, node.args
    op_res_node = gm.graph.call_function(to_standard_op[opname], args)
    return op_res_node, _SCALE, _ZERO_POINT


def _transform_prepacked_op(gm: torch.fx.GraphModule, node: torch.fx.Node):
    """
    Transformation for functions under prepacked namespace, where they share
    the same handling logic that [...]OpContext contains all parameters.
    """
    assert isinstance(node.target, torch._ops.OpOverload)
    opname, args = node.target._opname, node.args
    op_f = None
    if opname == "conv2d_clamp_run":
        op_f = torch.ops.aten.conv2d
    elif opname == "linear_clamp_run":
        op_f = torch.ops.aten.linear
    else:
        raise RuntimeError(f"Invalid operator {opname}")

    assert isinstance(args[1], torch.fx.Node)
    so = get_script_object(gm, args[1])

    func_args = []
    func_args += [args[0]]
    func_args += so.unpack()[:2]  # type: ignore[attr-defined]
    if opname == "conv2d_clamp_run":
        func_args += torch.ops.prepacked.unpack_prepacked_sizes_conv2d(so)[2:]

    op_res_node = gm.graph.call_function(op_f, tuple(func_args))
    return op_res_node


def _transform_batch_norm(gm: torch.fx.GraphModule, node: torch.fx.Node):
    args = node.args
    scale_node, zero_point_node = args[-2], args[-1]
    op_res_node = gm.graph.call_function(
        torch.ops.aten.native_batch_norm, (*args[:-3], False, 0.1, args[-3])
    )
    op_res_node = gm.graph.call_function(operator.getitem, (op_res_node, 0))
    return op_res_node, scale_node, zero_point_node


def fx_transform_quantized_op_to_standard_op(
    gm: torch.fx.GraphModule, node: torch.fx.Node
) -> torch.fx.Node:
    global _SCALE, _ZERO_POINT, _INPUT_Q_DTYPE

    assert isinstance(node.target, torch._ops.OpOverload)
    opname, overload = node.target._opname, node.target._overloadname

    key = f"{opname}.{overload}"
    opname_to_transform_f = {
        "conv1d.new": _transform_conv_with_packedparam,
        "conv1d_relu.new": _transform_conv_with_packedparam,
        "conv1d.default": _transform_conv_with_packedparam,
        "conv1d_relu.default": _transform_conv_with_packedparam,
        "conv2d.new": _transform_conv_with_packedparam,
        "conv2d_relu.new": _transform_conv_with_packedparam,
        "conv2d.default": _transform_conv_with_packedparam,
        "conv2d_relu.default": _transform_conv_with_packedparam,
        "linear.default": _transform_linear_with_packedparam,
        "linear_relu.default": _transform_linear_with_packedparam,
        "add.default": _transform_op_where_last_two_arguments_are_scale_and_zero_point,
        "add_relu.default": _transform_op_where_last_two_arguments_are_scale_and_zero_point,
        "mul.default": _transform_op_where_last_two_arguments_are_scale_and_zero_point,
        "mul_relu.default": _transform_op_where_last_two_arguments_are_scale_and_zero_point,
        "softmax.default": _transform_op_where_last_two_arguments_are_scale_and_zero_point,
        "cat.default": _transform_op_where_last_two_arguments_are_scale_and_zero_point,
        "hardswish.default": _transform_op_where_last_two_arguments_are_scale_and_zero_point,
        "batch_norm2d.default": _transform_batch_norm,
        "mul.Scalar": _transform_scalar_arithmetic,
        "add.Scalar": _transform_scalar_arithmetic,
    }

    if f"{key}" not in opname_to_transform_f:
        raise RuntimeError(f"Unsupported quantized op during transformation: {key}")

    op_res_node, scale_node, zero_point_node = opname_to_transform_f[f"{key}"](gm, node)

    # Add fused activation layer.
    op_res_node = insert_fused_activation_node(gm, opname, op_res_node)
    _SCALE, _ZERO_POINT = scale_node, zero_point_node

    assert _INPUT_Q_DTYPE is not None
    qmin_node, qmax_node = insert_qmin_qmax_node(gm, _INPUT_Q_DTYPE)
    q_fx_node = insert_quantized_node(
        gm,
        op_res_node,
        scale_node,
        zero_point_node,
        qmin_node,
        qmax_node,
        _INPUT_Q_DTYPE,
        torch.per_tensor_affine,
    )
    dq_fx_node = insert_dequantized_node(
        gm,
        q_fx_node,
        scale_node,
        zero_point_node,
        qmin_node,
        qmax_node,
        _INPUT_Q_DTYPE,
        None,
        torch.per_tensor_affine,
    )
    return dq_fx_node


def replace_quantized_ops_with_standard_ops(gm: torch.fx.GraphModule):
    """
    Replace legacy quantized ops (aten.quantize_per_tensor, quantized.conv) with
    PT2 ops (quantize_decomposed.quantize_per_tensor, aten.conv).

    Before:    x || -> aten.q        || -> quantized.conv2d     || -> quantized.linear    || -> aten.dq || -> y

    After:     x || -> qd.q -> qd.dq || -> aten.conv2d -> qd.q -> qd.dq || aten.linear -> qd.q -> qd.dq || -> y

    (qd == quantized_decomposed library, q = quantize, dq = dequantize)
                                          ^
                                          |
                getattr(w), getattr(b) from Conv2dParamPrepack

    During each iteration, the transformation spits out the transformed operator, its quantized output,
    and its dequantized value together. We did this because dequantization need to use the
    scale and zero point parameters from the quantization to recover the approximate original value. After each
    iteration, the new dequantization node will be used as the input to the next node (e.g., dq2 -> linear).

    For operators like conv2d and linear, their weights and bias are packed in a quantized format in the ScriptObject.
    During the transformation, we unpack those objects, get their dequantized tensor, populate those
    as attributes to the module, and use getattr to access them.

    One exception in the transformation is conv_prepack and linear_prepack. Those calls pack
    weight and bias constant tensors into ScriptObject, which are then used by subsequent conv2d or linear calls.
    During transformation, we directly skip transforming conv_prepack or linear_prepack. We check whether ScriptObject to the
    quantized::conv2d or linear is from conv_prepack or linear_prepack. If it is, we then inline those parameters
    to the operator by converting them to a getattr fx.node.

    For prepacked::conv2d_clamp_run and prepacked::linear_clamp_run, we directly convert them to aten.conv2d and aten.linear
    without the need of doing de/quantization.

    Three global variables defined are _INPUT_Q_DTYPE, _SCALE, _ZERO_POINT. _INPUT_Q_DTYPE determines the de/quantization
    data type, which is the same across the entire program, but it only shows up in the very first quantization
    call. _SCALE and _ZERO_POINT are used only when operators do not have those specified. E.g., mul.Scalar.
    """

    global _INPUT_Q_DTYPE

    quantized = False

    last_quantized_node = None
    for node in gm.graph.nodes:
        if isinstance(node.target, OpOverload):
            with gm.graph.inserting_before(node):
                namespace, opname = node.target.namespace, node.target._opname
                if namespace == "quantized" and opname not in [
                    "conv_prepack",
                    "linear_prepack",
                ]:
                    quantized = True
                    fx_node = fx_transform_quantized_op_to_standard_op(gm, node)
                    node.replace_all_uses_with(fx_node)
                    last_quantized_node = fx_node
                elif namespace == "prepacked":
                    quantized = True
                    fx_node = _transform_prepacked_op(gm, node)
                    node.replace_all_uses_with(fx_node)
                    last_quantized_node = fx_node
                elif namespace == "aten" and opname == "quantize_per_tensor":
                    inp_node, scale_node, zero_point_node, dtype_node = node.args
                    dtype_node = fx_enum_to_dtype(gm, dtype_node)
                    _INPUT_Q_DTYPE = dtype_node
                    qmin_node, qmax_node = insert_qmin_qmax_node(gm, dtype_node)
                    q_fx_node = insert_quantized_node(
                        gm,
                        inp_node,
                        scale_node,
                        zero_point_node,
                        qmin_node,
                        qmax_node,
                        dtype_node,
                        torch.per_tensor_affine,
                    )
                    dq_fx_node = insert_dequantized_node(
                        gm,
                        q_fx_node,
                        scale_node,
                        zero_point_node,
                        qmin_node,
                        qmax_node,
                        dtype_node,
                        None,
                        torch.per_tensor_affine,
                    )
                    node.replace_all_uses_with(dq_fx_node)
                    last_quantized_node = dq_fx_node
                elif namespace == "aten" and opname == "dequantize":
                    assert last_quantized_node is not None
                    node.replace_all_uses_with(last_quantized_node)
                else:
                    last_quantized_node = node

    # Post-processing again to remove legacy ScriptObjects and quantizated tensors
    # stored as attributes or in the buffer. This is used to clean up the GraphModule
    # to not trigger tracing errors like missing __obj_flatten__ functions.
    def _clean_attr(mod: torch.nn.Module):
        for submod in mod.modules():
            attr_names_to_clean = set()
            for k, v in submod.__dict__.items():
                if isinstance(v, torch.ScriptObject):
                    attr_names_to_clean.add(k)
                if k == "_buffers":
                    buffer_name_to_clean = set()
                    for b_name, b_value in v.items():
                        if isinstance(b_value, torch.Tensor) and b_value.dtype in [
                            torch.qint8,
                            torch.quint8,
                        ]:
                            buffer_name_to_clean.add(b_name)
                    for b_name in buffer_name_to_clean:
                        v.pop(b_name, None)
            for attr_name in attr_names_to_clean:
                delattr(submod, attr_name)

    if quantized:
        """
        TODO: SetAttr + quantized ops will result incorrect program. This flag is used to temporarily
        bypass test cases.

        The deadcode elimination pass is needed to remove legacy quantized ops. Otherwise, retracing
        will throw errors. However, the current way of SetAttr does inplace update to attributes, so
        this pass regard them as dead code and remove them. Below is an example of GraphModule before
        and after the dead code elimination pass.

        class GraphModule(torch.nn.Module):
            def forward(self, x_1):
                # No stacktrace found for following nodes
                data = self.data;  data = None
                data_1 = self.data
                add_tensor = torch.ops.aten.add.Tensor(data_1, x_1, alpha = 1);  data_1 = None
                data_2 = self.data
                copy_ = torch_Tensor_copy_(data_2, add_tensor);  data_2 = add_tensor = copy_ = None
                data_3 = self.data
                add_tensor_1 = torch.ops.aten.add.Tensor(x_1, data_3, alpha = 1);  x_1 = data_3 = None
                return add_tensor_1

        class GraphModule(torch.nn.Module):
            def forward(self, x_1):
                # No stacktrace found for following nodes
                data_3 = self.data
                add_tensor_1 = torch.ops.aten.add.Tensor(x_1, data_3, alpha = 1);  x_1 = data_3 = None
                return add_tensor_1
        """
        gm.graph.eliminate_dead_code()
        _clean_attr(gm)
