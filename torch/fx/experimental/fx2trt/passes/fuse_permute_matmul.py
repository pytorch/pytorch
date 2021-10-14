import warnings
import torch
import torch.fx
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
from typing import Any


def trt_transposed_matmul(lhs: torch.Tensor, rhs: torch.Tensor, lhs_transposed: bool, rhs_transposed: bool):
    if lhs_transposed:
        lhs = lhs.transpose(-1, -2)
    if rhs_transposed:
        rhs = rhs.transpose(-1, -2)
    return torch.matmul(lhs, rhs)


def trt_transposed_linear(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    return torch.matmul(input.transpose(-1, -2), weight.t()) + bias


def check_permute(node: torch.fx.Node):
    ranks = len(node.meta["tensor_meta"].shape)
    permutation = list(i % ranks for i in node.kwargs["permutation"])  # type: ignore[union-attr]
    allowed_permutation = list(i for i in range(ranks))
    allowed_permutation[-1] = ranks - 2
    allowed_permutation[-2] = ranks - 1
    return permutation == allowed_permutation


def has_right_users(node: torch.fx.Node, op: Any) -> bool:
    return all([u.target == op for u in node.users.keys()])


def fuse_permute_linear(gm: torch.fx.GraphModule):
    """
    Fuse pattern like permute + linear if permute is transposing the last two dimension.
    """
    for node in gm.graph.nodes:
        if node.target == acc_ops.linear:
            inp = node.kwargs["input"]
            if inp.target == acc_ops.permute and check_permute(inp) and has_right_users(inp, acc_ops.linear):
                inp = inp.kwargs["input"]
                weight = node.kwargs["weight"]
                bias = node.kwargs["bias"]
                with gm.graph.inserting_before(node):
                    fused_node = gm.graph.call_function(trt_transposed_linear, args=(inp, weight, bias))
                    node.replace_all_uses_with(fused_node)

    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return gm


def fuse_permute_matmul(gm: torch.fx.GraphModule):
    """
    Fuse pattern like permute + matmul if permute is transposing the last two dimension.
    """
    for node in gm.graph.nodes:
        if node.target == acc_ops.matmul:
            lhs, rhs = node.kwargs["input"], node.kwargs["other"]
            lhs_transposed = rhs_tranposed = False
            skip = False

            if lhs.target == acc_ops.permute and check_permute(lhs):
                if not has_right_users(lhs, acc_ops.matmul):
                    skip = True
                else:
                    lhs_transposed = True
                    lhs = lhs.kwargs["input"]

            if rhs.target == acc_ops.permute and check_permute(rhs):
                if not has_right_users(rhs, acc_ops.matmul):
                    skip = True
                else:
                    rhs_tranposed = True
                    rhs = rhs.kwargs["input"]

            if (not skip) and (lhs_transposed or rhs_tranposed):
                with gm.graph.inserting_before(node):
                    fused_node = gm.graph.call_function(trt_transposed_matmul, args=(lhs, rhs, lhs_transposed, rhs_tranposed))
                node.replace_all_uses_with(fused_node)

    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return gm


try:
    import tensorrt as trt
    from torch.fx.experimental.fx2trt.fx2trt import tensorrt_converter
    from torch.fx.experimental.fx2trt.converters.acc_ops_converters import get_trt_tensor, add_binary_elementwise_layer, broadcast
except Exception as e:
    warnings.warn(f"Unable to import TensorRT related libraries.: {e}")
else:
    @tensorrt_converter(trt_transposed_matmul)
    def trt_transposed_matmul_converter(network, target, args, kwargs, name):
        lhs, rhs, lhs_transposed, rhs_transposed = args

        layer = network.add_matrix_multiply(
            lhs,
            trt.MatrixOperation.TRANSPOSE if lhs_transposed else trt.MatrixOperation.NONE,
            rhs,
            trt.MatrixOperation.TRANSPOSE if rhs_transposed else trt.MatrixOperation.NONE,
        )
        layer.name = name
        return layer.get_output(0)

    @tensorrt_converter(trt_transposed_linear)
    def trt_transposed_linear_converter(network, target, args, kwargs, name):
        input, weight, bias = args

        weight = get_trt_tensor(network, weight.t(), f"{name}_weight")
        bias = get_trt_tensor(network, bias.reshape(1, -1), f"{name}_bias")

        input, weight = broadcast(network, input, weight, f"{input.name}_broadcast", f"{weight.name}_broadcast")
        layer = network.add_matrix_multiply(
            input,
            trt.MatrixOperation.TRANSPOSE,
            weight,
            trt.MatrixOperation.NONE,
        )
        layer.name = f"{name}_mm"
        return add_binary_elementwise_layer(
            network, layer.get_output(0), bias, trt.ElementWiseOperation.SUM, f"{name}_add"
        )
