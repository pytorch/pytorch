import warnings

import torch
import torch.fx
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
from torch.fx.experimental.fx_acc.acc_utils import (
    get_attr,
)

def fuse_sparse_matmul_add(gm: torch.fx.GraphModule):
    """
    Replace acc_ops.matmul + acc_ops.add with acc_ops.linear
    TRT8.2 can take advantage of structured sparsity (2:4), but the graph needs contain a single FC layer.
    Later versions of TRT should work with matmul.

    Example before:
    def forward(self, x):
        a = self.a
        b = self.b
        addmm_mm = torch.fx.experimental.fx_acc.acc_ops.matmul(input = a, other = b);  a = b = None
        addmm_add = torch.fx.experimental.fx_acc.acc_ops.add(input = addmm_mm, other = x);  addmm_mm = x = None
        return addmm_add

    After:
    def forward(self, x):
        a = self.a
        b = self.b
        linear_1 = torch.fx.experimental.fx_acc.acc_ops.linear(input = a, weight = b, bias = x);  a = b = x = None
        return linear_1
    """
    counter = 0
    for node in gm.graph.nodes:
        if node.target != acc_ops.add:
            continue
        add_node = node
        bias = add_node.kwargs["other"]

        if bias.op != "get_attr":
            continue
        # test that bias tensor is one-dimensional, should correspond to shape (out_features)
        if get_attr(bias).dim() > 1:
            continue

        node = add_node.kwargs["input"]
        if node.target != acc_ops.matmul:
            continue
        matmul_node = node
        a = matmul_node.kwargs["input"]

        node = matmul_node.kwargs["other"]
        if node.op != "get_attr":
            continue

        get_attr_node = node
        weight = get_attr(get_attr_node)
        # TODO: verify that weight comply with TRT structured sparsity requirements:
        # For each output channel and for each spatial pixel in the kernel weights,
        # every 4 input channels must have at least 2 zeros.

        # test that weight tensor is two-dimensional, should correspond to shape (out_features, in_features)
        if weight.dim() != 2:
            continue

        weight_t = weight.transpose(0, 1)
        weight_t_name = "weight_t_tensor_" + str(counter)
        gm.register_buffer(weight_t_name, weight_t)
        counter += 1

        with gm.graph.inserting_before(add_node):
            weight_t_attr = gm.graph.get_attr(weight_t_name)
            fused_node = gm.graph.call_function(acc_ops.linear, kwargs={"input": a, "weight": weight_t_attr, "bias": bias})
        add_node.replace_all_uses_with(fused_node)

    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return gm

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


def fuse_permute_linear(gm: torch.fx.GraphModule):
    """
    Fuse pattern like permute + linear if permute is transposing the last two dimension.
    """
    for node in gm.graph.nodes:
        if node.target == acc_ops.linear:
            inp = node.kwargs["input"]
            if inp.target == acc_ops.permute and check_permute(inp):
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
                lhs_transposed = True
                lhs = lhs.kwargs["input"]

            if rhs.target == acc_ops.permute and check_permute(rhs):
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


def fuse_unsqueeze_cat_sum(gm: torch.fx.GraphModule):
    for node in gm.graph.nodes:
        if node.target != acc_ops.sum:
            continue
        prev_node = node.kwargs["input"]
        if prev_node.target != acc_ops.cat or prev_node.kwargs["dim"] != 0:
            continue
        cat_inputs = prev_node.kwargs["tensors"]
        valid_pass = True
        for i in cat_inputs:
            if i.target != acc_ops.unsqueeze or i.kwargs["dim"] != 0:
                valid_pass = False
                break

        if not valid_pass:
            continue
        input_val = [i.kwargs["input"] for i in cat_inputs]

        with gm.graph.inserting_before(node):
            left = input_val[0]
            for i in range(1, len(input_val)):
                right = input_val[i]
                fused_node = gm.graph.call_function(acc_ops.add, kwargs={"input": left, "other": right})
                left = fused_node
        node.replace_all_uses_with(fused_node)

    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return gm



try:
    import tensorrt as trt
    from torch.fx.experimental.fx2trt.converter_registry import tensorrt_converter
    from torch.fx.experimental.fx2trt.converters.acc_ops_converters import (
        get_trt_tensor,
        add_binary_elementwise_layer,
        broadcast,
        set_layer_name,
    )
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
        set_layer_name(layer, target, name)
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
        set_layer_name(layer, target, f"{name}_mm")
        return add_binary_elementwise_layer(
            network, layer.get_output(0), bias, trt.ElementWiseOperation.SUM, target, f"{name}_add"
        )
