import logging
import random
import weakref

import functorch

import torch
from torch import _prims
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode
from torch.overrides import TorchFunctionMode
from .utils import is_cpu_device
from . import config
from .utils import decode_device, is_cpu_device
import copy
from torch._dynamo.utils import detect_fake_mode
from torch.fx.passes.shape_prop import ShapeProp


def is_quantized_graph_module(gm: torch.fx.GraphModule):
    found_quantize = False
    quantize_ops = (
        torch.ops.quantized_decomposed.quantize_per_tensor,
        torch.ops.quantized_decomposed.quantize_per_channel,
        torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
    )
    for node in gm.graph.nodes:
        if node.target in quantize_ops:
            found_quantize = True
            break
    return found_quantize


def _quantize_and_replace_weight(
    gm: torch.fx.GraphModule, dq_per_channel_node: torch.fx.Node
):
    # pattern: w - q - dq - weighted op
    q_per_channel_node = dq_per_channel_node.args[0]
    weight_node = q_per_channel_node.args[0]
    w_attr_name = weight_node.target
    weight = getattr(gm, w_attr_name)

    assert isinstance(weight, torch.Tensor), "Cannot find weight for quantization"
    if weight.is_quantized:
        return
    quantize_args = (
        getattr(gm, n.target) if isinstance(n, torch.fx.Node) else n
        for n in q_per_channel_node.args
    )
    q_arg_list = list(quantize_args)
    q_arg_tuple = tuple(q_arg_list)
    weight_int8 = torch.nn.parameter.Parameter(
        torch.ops.quantized_decomposed.quantize_per_channel(*q_arg_tuple),
        requires_grad=False,
    )

    qw_attr_name = w_attr_name + "_quant"
    setattr(gm, qw_attr_name, weight_int8)
    weight_node.target = qw_attr_name
    gm.graph.owning_module._buffers[qw_attr_name] = weight_int8
    delattr(gm, w_attr_name)
    q_per_channel_node.replace_all_uses_with(weight_node)
    gm.graph.erase_node(q_per_channel_node)


def pre_quantize_weights(gm: torch.fx.GraphModule):
    # pattern: w - q - dq - weighted op
    aten = torch.ops.aten
    decomposed = torch.ops.quantized_decomposed
    for node in gm.graph.nodes:
        dq_per_channel_node = None
        if node.target == aten.convolution.default:
            # conv args = (x, w, ...)
            dq_per_channel_node = node.args[1]
        if dq_per_channel_node is not None:
            assert (
                dq_per_channel_node.target == decomposed.dequantize_per_channel
            ), "Cannot find the dequantize op for weight"
            _quantize_and_replace_weight(gm, dq_per_channel_node)
    gm.graph.lint()
    gm.recompile()


def fuse_reference_quantized_conv_unary(gm: torch.fx.GraphModule):
    """
    Replace pattern:
    # dequantize_per_channel -
    # dequantize_per_tensor  - conv - post_op - quantize_per_tensor
    into new pattern:
    # torch.ops.quantized.conv_unary(post_op = post_op)
    """
    aten = torch.ops.aten
    quantized_decomposed = torch.ops.quantized_decomposed
    convolution = aten.convolution.default
    relu = aten.relu.default
    relu_ = aten.relu_.default
    quantize_per_tensor = quantized_decomposed.quantize_per_tensor
    dequantize_per_tensor = quantized_decomposed.dequantize_per_tensor
    dequantize_per_channel = quantized_decomposed.dequantize_per_channel

    unary_post_ops = {
        "relu": relu,
        "relu_": relu_,
    }
    for name, unary_post_op in unary_post_ops.items():
        for node in gm.graph.nodes:
            if node.target is convolution:
                (
                    x,
                    w,
                    bias,
                    stride,
                    padding,
                    dilation,
                    is_transposed,
                    out_padding,
                    groups,
                ) = node.args
                assert (
                    x.target == dequantize_per_tensor
                ), "input's node should be dequantize_per_tensor"
                assert (
                    w.target == dequantize_per_channel
                ), "weight's node should be dequantize_per_channel"
                (qx, x_scale, x_zp, x_quant_min, x_quant_max, x_dtype) = x.args
                (qw, w_scale, w_zp, w_axis, w_quant_min, w_quant_max, w_dtype) = w.args
                if (
                    len(list(node.users)) != 1
                    or len(list(x.users)) != 1
                    or len(list(w.users)) != 1
                ):
                    # Ensure met these criteria
                    # 1. the conv node has only 1 user node
                    # 2. the dequantize_per_tensor node as activation input of conv has only 1 user node
                    # 3. the dequantize_per_channel node as weight input of conv has only 1 user node
                    continue

                if list(node.users)[0].target is unary_post_op:
                    # conv relu fusion
                    unary_op_to_be_fused = list(node.users)[0]
                    if (
                        len(list(unary_op_to_be_fused.users)) != 1
                        or list(unary_op_to_be_fused.users)[0].target
                        != quantize_per_tensor
                    ):
                        # Ensure met these criteria
                        # 1. the unary op has only 1 user node.
                        # 2. this user node should be quantize_per_tensor.
                        continue
                    quant_per_tensor_node = list(unary_op_to_be_fused.users)[0]
                else:
                    # Not meet fusion pattern: the op after conv is not unary_op to be fused
                    continue

                (
                    y,
                    y_scale,
                    y_zp,
                    y_quant_min,
                    y_quant_max,
                    y_dtype,
                ) = quant_per_tensor_node.args
                with gm.graph.inserting_after(quant_per_tensor_node):
                    args = (
                        qx,
                        x_scale,
                        x_zp,
                        qw,
                        w_scale,
                        w_zp,
                        w_axis,
                        bias,
                        stride,
                        padding,
                        dilation,
                        groups,
                        y_scale,
                        y_zp,
                        name,
                    )
                    new_conv_node = gm.graph.call_function(
                        torch.ops.quantized.conv_unary, args=args
                    )
                # Copy node meta
                new_conv_node.meta = copy.copy(quant_per_tensor_node.meta)
                quant_per_tensor_node.replace_all_uses_with(new_conv_node)

                gm.graph.erase_node(quant_per_tensor_node)  # erase quantize_per_tensor
                gm.graph.erase_node(unary_op_to_be_fused)  # erase unary_op
                gm.graph.erase_node(node)  # erase conv
                gm.graph.erase_node(w)  # erase dequantize_per_channel
                gm.graph.erase_node(x)  # erase dequantize_per_tensor

    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return gm


def fuse_single_reference_quantized_conv(gm: torch.fx.GraphModule):
    """
    Single reference conv without post op fusion
    Replace pattern:
    # dequantize_per_channel -
    # dequantize_per_tensor  - conv - quantize_per_tensor
    into new pattern:
    # torch.ops.quantized.conv_unary(post_op = None)
    """
    aten = torch.ops.aten
    quantized_decomposed = torch.ops.quantized_decomposed
    convolution = aten.convolution.default
    quantize_per_tensor = quantized_decomposed.quantize_per_tensor
    dequantize_per_tensor = quantized_decomposed.dequantize_per_tensor
    dequantize_per_channel = quantized_decomposed.dequantize_per_channel

    for node in gm.graph.nodes:
        if node.target is convolution:
            (
                x,
                w,
                bias,
                stride,
                padding,
                dilation,
                is_transposed,
                out_padding,
                groups,
            ) = node.args
            assert (
                x.target == dequantize_per_tensor
            ), "input's node should be dequantize_per_tensor"
            assert (
                w.target == dequantize_per_channel
            ), "weight's node should be dequantize_per_channel"
            (qx, x_scale, x_zp, x_quant_min, x_quant_max, x_dtype) = x.args
            (qw, w_scale, w_zp, w_axis, w_quant_min, w_quant_max, w_dtype) = w.args
            if (
                len(list(node.users)) != 1
                or len(list(x.users)) != 1
                or len(list(w.users)) != 1
            ):
                # Ensure met these criteria
                # 1. the conv node has only 1 user node
                # 2. the dequantize_per_tensor node as activation input of conv has only 1 user node
                # 3. the dequantize_per_channel node as weight input of conv has only 1 user node
                continue

            if list(node.users)[0].target is quantize_per_tensor:
                # Single conv without post op
                quant_per_tensor_node = list(node.users)[0]
            else:
                # Not meet fusion pattern: the op after conv is not unary_op to be fused or quantize_per_tensor
                continue

            (
                y,
                y_scale,
                y_zp,
                y_quant_min,
                y_quant_max,
                y_dtype,
            ) = quant_per_tensor_node.args
            with gm.graph.inserting_after(quant_per_tensor_node):
                args = (
                    qx,
                    x_scale,
                    x_zp,
                    qw,
                    w_scale,
                    w_zp,
                    w_axis,
                    bias,
                    stride,
                    padding,
                    dilation,
                    groups,
                    y_scale,
                    y_zp,
                    "none",
                )
                new_conv_node = gm.graph.call_function(
                    torch.ops.quantized.conv_unary, args=args
                )
            # Copy node meta
            new_conv_node.meta = copy.copy(quant_per_tensor_node.meta)
            quant_per_tensor_node.replace_all_uses_with(new_conv_node)

            gm.graph.erase_node(quant_per_tensor_node)  # erase quantize_per_tensor
            gm.graph.erase_node(node)  # erase conv
            gm.graph.erase_node(w)  # erase dequantize_per_channel
            gm.graph.erase_node(x)  # erase dequantize_per_tensor

    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return gm


def fuse_reference_quantized_conv(gm: torch.fx.GraphModule):
    gm = fuse_reference_quantized_conv_unary(gm)
    gm = fuse_single_reference_quantized_conv(gm)
    return gm


def _insert_packed_weight_bias(
    gm: torch.fx.GraphModule,
    weight_node: torch.fx.Node,
    bias_node: torch.fx.Node,
    packed_weight: torch.Tensor,
    packed_bias: torch.Tensor,
):
    w_attr_name = weight_node.target
    w_packed_attr_name = w_attr_name + "_packed"
    gm.graph.owning_module._buffers[w_packed_attr_name] = packed_weight
    setattr(gm, w_packed_attr_name, gm.graph.owning_module._buffers[w_packed_attr_name])
    weight_node.target = w_packed_attr_name
    delattr(gm, w_attr_name)
    # q_per_channel_node.replace_all_uses_with(weight_node)
    # gm.graph.erase_node(q_per_channel_node)
    # Replace the original bias with packed bias
    if bias_node is not None:
        b_attr_name = bias_node.target
        b_pack_attr_name = b_attr_name + "_packed"
        gm.graph.owning_module._buffers[b_pack_attr_name] = packed_bias
        setattr(gm, b_pack_attr_name, gm.graph.owning_module._buffers[b_pack_attr_name])
        bias_node.target = b_pack_attr_name
        delattr(gm, b_attr_name)


def _prepack_conv_weight(gm: torch.fx.GraphModule):
    # Assume dq - conv (- relu) - q is already fused and `w - q` is replaced by qw
    decomposed = torch.ops.quantized_decomposed
    for node in gm.graph.nodes:
        if node.target == torch.ops.quantized.conv_unary:
            # node args = (qx, x_scale, x_zp,
            #              qw, w_scale, w_zp, w_axis,
            #              bias, stride, padding, dilation, groups,
            #              y_scale, y_zp, unary_post_op_name)
            weight_node = node.args[3]
            assert hasattr(gm, weight_node.target) and isinstance(
                getattr(gm, weight_node.target), torch.Tensor
            ), "Cannot find quantized weight of convolution"
            weight_int8 = getattr(gm, weight_node.target)
            bias_node = node.args[7]
            # Prepack weight into an MKLDNN tensor of dtype int8
            w_scales = getattr(gm, node.args[4].target)
            x_shape = node.args[0].meta.get("tensor_meta").shape
            x_scale = getattr(gm, node.args[1].target)
            x_zp = getattr(gm, node.args[2].target)
            bias = getattr(gm, bias_node.target) if bias_node is not None else None
            stride = node.args[8]
            padding = node.args[9]
            dilation = node.args[10]
            groups = node.args[11]
            packed_weight, packed_bias = torch.ops.quantized.conv_prepack_cpu_tensor(
                weight_int8,
                w_scales,
                x_shape,
                x_scale,
                x_zp,
                bias,
                stride,
                padding,
                dilation,
                groups,
            )
            # Replace the original weight with packed weight
            _insert_packed_weight_bias(
                gm, weight_node, bias_node, packed_weight, packed_bias
            )
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return gm


def prepack_weight_in_graph(gm: torch.fx.GraphModule):
    gm = _prepack_conv_weight(gm)
    return gm


def fuse_quantization(gm: torch.fx.GraphModule, example_inputs):
    # skip if gm is not a quantized graph module
    if not (is_quantized_graph_module(gm) and is_cpu_device(example_inputs)):
        return gm
    gm.graph.eliminate_dead_code()
    gm.recompile()
    # Fuse `quant_per_channel - weight` and replace the original fp32 weight with quantized one
    pre_quantize_weights(gm)

    # To store input shapes on the graph
    # Get shape by node.meta.get("tensor_meta").shape
    fake_mode = detect_fake_mode(example_inputs)
    ShapeProp(gm, fake_mode=fake_mode).propagate(*example_inputs)

    # Fuse `dq - op (- post ops) - q` to quantized op
    gm = fuse_reference_quantized_conv(gm)

    # Reorder quantized weight to desired format for oneDNN kernel
    # After that, weight is a MKLDNN tensor and it replaces the original one in graph
    gm = prepack_weight_in_graph(gm)

    return gm
