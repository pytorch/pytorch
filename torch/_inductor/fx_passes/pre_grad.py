import copy
import logging
from typing import List, Optional

import torch
import torch.nn as nn
from torch._dynamo.utils import detect_fake_mode
from torch.fx.experimental.optimization import (
    matches_module_pattern,
    replace_node_module,
)
from torch.fx.passes.shape_prop import ShapeProp
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_conv_bn_weights

from .. import config

from ..fx_utils import matches_module_function_pattern
from ..pattern_matcher import (
    init_once_fakemode,
    PatternMatcherPass,
    stable_topological_sort,
)
from ..utils import is_cpu_device

log = logging.getLogger(__name__)

normalization_pass = PatternMatcherPass(prevent_match_across_mutations=True)
merge_splits_pass = PatternMatcherPass(prevent_match_across_mutations=True)
split_cat_pass = PatternMatcherPass(prevent_match_across_mutations=True)
unbind_stack_pass = PatternMatcherPass(prevent_match_across_mutations=True)

pattern_matcher_passes: List[PatternMatcherPass] = [
    normalization_pass,
    merge_splits_pass,
    split_cat_pass,
    unbind_stack_pass,
]


@init_once_fakemode
def lazy_init():
    from . import split_cat  # noqa: F401

    if config.is_fbcode():
        from .fb import split_cat as split_cat_fb  # noqa: F401


def pre_grad_passes(gm, example_inputs):
    """
    Apply passes on the input FX graph using Torch IR.

    WARNING:
    The IR before grad is not functional or normalized, so it is harder
    to write passes on this IR.  Passes must be safe with respect to
    aliasing and mutation and need to handle all possible arg schemas.

    Consider adding a new pass to post_grad.py or joint_graph.py which
    are after functionalization and normalization.
    """

    if config.pattern_matcher:
        lazy_init()
        gm = fuse_fx(gm, example_inputs)
        for pattern_matcher_pass in pattern_matcher_passes:
            pattern_matcher_pass.apply(gm.graph)

    stable_topological_sort(gm.graph)
    gm.graph.lint()
    gm.recompile()

    return gm


def fuse_fx(gm: torch.fx.GraphModule, example_inputs):
    is_cpu = is_cpu_device(example_inputs)

    fake_mode = detect_fake_mode(example_inputs)

    gm = sink_cat_after_pointwise(gm)
    if config.permute_fusion and not is_cpu:
        # For linear permute fusion, we need to check input info to identify
        # and perform proper permutation/transpose
        ShapeProp(gm, fake_mode=fake_mode).propagate(*example_inputs)
        gm = linear_permute_fusion(gm)
        gm = permute_linear_fusion(gm)
        gm = permute_matmul_fusion(gm)

    # make sure the autograd is disabled.
    if torch.is_grad_enabled():
        return gm
    if not is_cpu:
        return gm
    gm = remove_identity(gm)
    gm = fuse_conv_bn(gm)
    return gm


def fetch_attr(target: str, mod):
    target_atoms = target.split(".")
    attr_itr = mod
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(
                f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}"
            )
        attr_itr = getattr(attr_itr, atom)
    return attr_itr


def remove_identity(gm: torch.fx.GraphModule):
    """
    Removes all identity layers from the module.
    """

    class IdentityRemover(torch.fx.Transformer):
        def call_module(self, target, args, kwargs):
            if isinstance(self.submodules[target], nn.Identity):
                assert len(args) == 1
                return args[0]
            else:
                return super().call_module(target, args, kwargs)

    return IdentityRemover(gm).transform()


def fuse_conv_bn(gm: torch.fx.GraphModule, inplace=False):
    """
    Fuses Convolution/BN layers for inference purposes.
    """
    modules_patterns = [
        (torch.nn.Conv1d, torch.nn.BatchNorm1d),
        (torch.nn.Conv2d, torch.nn.BatchNorm2d),
        (torch.nn.Conv3d, torch.nn.BatchNorm3d),
    ]
    module_function_patterns = [
        (torch.nn.Conv1d, F.batch_norm),
        (torch.nn.Conv2d, F.batch_norm),
        (torch.nn.Conv3d, F.batch_norm),
    ]
    modules = dict(gm.named_modules())
    for pattern in modules_patterns:
        for node in gm.graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                    continue
                conv = modules[node.args[0].target]
                bn = modules[node.target]
                eval_mode = all(not n.training for n in [conv, bn])
                if not eval_mode:
                    continue
                if not bn.track_running_stats:
                    continue
                fused_conv = fuse_conv_bn_eval(conv, bn)
                replace_node_module(node.args[0], modules, fused_conv)
                node.replace_all_uses_with(node.args[0])
                gm.graph.erase_node(node)
    gm.graph.lint()
    for pattern in module_function_patterns:
        for node in gm.graph.nodes:
            if matches_module_function_pattern(pattern, node, modules):
                # TODO: support kwargs.
                if len(node.args) != 8:
                    continue
                conv = modules[node.args[0].target]
                bn_training = node.args[5]
                bn_eps = node.args[7]
                if conv.training or bn_training:
                    continue
                if type(bn_eps) is not float:
                    continue
                bn_args_is_constant = all(
                    n.op == "get_attr" and len(n.users) == 1 for n in node.args[1:5]
                )
                if not bn_args_is_constant:
                    continue
                bn_running_mean = fetch_attr(node.args[1].target, gm)
                bn_running_var = fetch_attr(node.args[2].target, gm)
                bn_weight = fetch_attr(node.args[3].target, gm)
                bn_bias = fetch_attr(node.args[4].target, gm)
                if bn_running_mean is None or bn_running_var is None:
                    continue
                fused_conv = copy.deepcopy(conv)
                fused_conv.weight, fused_conv.bias = fuse_conv_bn_weights(
                    fused_conv.weight,
                    fused_conv.bias,
                    bn_running_mean,
                    bn_running_var,
                    bn_eps,
                    bn_weight,
                    bn_bias,
                )
                replace_node_module(node.args[0], modules, fused_conv)
                node.replace_all_uses_with(node.args[0])
                gm.graph.erase_node(node)
    gm.graph.lint()
    gm.recompile()

    return gm


class NormalizedLinearNode:
    def __init__(self, node: torch.fx.Node) -> None:
        assert node.op == "call_function"
        assert node.target in [torch.nn.functional.linear]
        self.node: torch.fx.Node = node

    def get_input(self) -> torch.fx.Node:
        if len(self.node.args) > 0:
            return self.node.args[0]
        else:
            return self.node.kwargs["input"]

    def get_weight(self) -> torch.fx.Node:
        if len(self.node.args) > 1:
            return self.node.args[1]
        else:
            return self.node.kwargs["weight"]

    def get_bias(self) -> torch.fx.Node:
        if len(self.node.args) > 2:
            return self.node.args[2]
        else:
            return self.node.kwargs["bias"] if "bias" in self.node.kwargs else None


class NormalizedMatmulNode:
    def __init__(self, node: torch.fx.Node) -> None:
        assert node.op == "call_function"
        assert node.target in [torch.bmm, torch.matmul]
        self.node: torch.fx.Node = node

    def get_input(self) -> torch.fx.Node:
        if len(self.node.args) > 0:
            return self.node.args[0]
        else:
            return self.node.kwargs["input"]

    def get_other(self) -> torch.fx.Node:
        if len(self.node.args) > 1:
            return self.node.args[1]
        else:
            return self.node.kwargs["other"]


def check_permute(node: torch.fx.Node):
    ranks = len(node.meta["tensor_meta"].shape)
    if len(node.args) > 3:
        permutation = [node.args[i] % ranks for i in range(1, ranks + 1)]
    elif (
        "permutation" in node.kwargs
        and node.kwargs["permutation"] is not None
        and len(node.kwargs["permutation"]) > 2
    ):
        permutation = [i % ranks for i in node.kwargs["permutation"]]
    else:
        return False
    allowed_permutation = list(range(ranks))
    allowed_permutation[-1] = ranks - 2
    allowed_permutation[-2] = ranks - 1
    return permutation == allowed_permutation


def sink_cat_after_pointwise(module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    def one_user(node):
        users = list(node.users)
        return users[0] if len(users) == 1 else None

    def is_view(node):
        view = {"view"}
        return node.op == "call_method" and node.target in view

    def is_pointwise_unary(node):
        pointwise = {torch.relu, torch.tanh, "relu", "tanh"}
        return node.op in {"call_function", "call_method"} and node.target in pointwise

    g = module.graph
    for node in g.nodes:
        if node.op != "call_function" or node.target != torch.cat:
            continue

        cat_or_view = node
        while True:
            user = one_user(cat_or_view)
            if not user or not is_view(user):
                break
            cat_or_view = user

        if user and is_pointwise_unary(user):
            with g.inserting_before(node):

                def cat_args(tensors, dim=0):
                    return tensors, dim

                tensors, dim = cat_args(*node.args, **node.kwargs)
                new_tensors = [
                    g.create_node(user.op, user.target, args=(arg,), kwargs=user.kwargs)
                    for arg in tensors
                ]
                new_cat = g.create_node(
                    "call_function", torch.cat, args=(new_tensors, dim)
                )
                user.replace_all_uses_with(cat_or_view)
                node.replace_all_uses_with(new_cat)
                g.erase_node(user)
                g.erase_node(node)
    g.lint()
    module.recompile()
    return module


def linear_permute_fusion(module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for node in module.graph.nodes:
        if (
            node.op == "call_method"
            and node.target == "permute"
            and check_permute(node)
        ):
            if len(node.args) > 0:
                input_node = node.args[0]
            else:
                input_node = node.kwargs["input"]
            if (
                input_node.op == "call_function"
                and input_node.target == torch.nn.functional.linear
            ):
                normalized = NormalizedLinearNode(input_node)
                input = normalized.get_input()
                weight = normalized.get_weight()
                bias = normalized.get_bias()
                with module.graph.inserting_before(node):
                    fused_node = module.graph.call_function(
                        linear_transpose, args=(input, weight, bias)
                    )
                    node.replace_all_uses_with(fused_node)
                    module.graph.erase_node(node)
                    if len(input_node.users) == 0:
                        module.graph.erase_node(input_node)

    module.graph.lint()
    module.recompile()
    return module


# Y1 = X * W^T + bias
# Y2 = Y1.permute(0, 2, 1)
# ---->
# Y2 = (W * X^T + bias.unsqueeze(-1))^T
def linear_transpose(
    input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    if bias is None:
        return torch.matmul(weight, input.transpose(-1, -2))
    return torch.matmul(weight, input.transpose(-1, -2)) + bias.unsqueeze(-1)


def permute_linear_fusion(module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for node in module.graph.nodes:
        if node.op == "call_function" and node.target == torch.nn.functional.linear:
            if len(node.args) > 0:
                input_node = node.args[0]
            else:
                input_node = node.kwargs["input"]
            if (
                input_node.op == "call_method"
                and input_node.target == "permute"
                and check_permute(input_node)
            ):
                normalized = NormalizedLinearNode(node)
                if len(input_node.args) > 0:
                    input = input_node.args[0]
                else:
                    input = input_node.kwargs["input"]
                weight = normalized.get_weight()
                bias = normalized.get_bias()
                with module.graph.inserting_before(node):
                    fused_node = module.graph.call_function(
                        transpose_linear, args=(input, weight, bias)
                    )
                    node.replace_all_uses_with(fused_node)
                    module.graph.erase_node(node)
                    if len(input_node.users) == 0:
                        module.graph.erase_node(input_node)

    module.graph.lint()
    module.recompile()
    return module


def permute_matmul_fusion(module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for node in module.graph.nodes:
        if node.op == "call_function" and (
            node.target == torch.bmm or node.target == torch.matmul
        ):
            normalized = NormalizedMatmulNode(node)
            input_A_node = normalized.get_input()
            input_B_node = normalized.get_other()
            input_A = input_A_node
            input_B = input_B_node
            Atrans = Btrans = False
            if (
                input_A_node.op == "call_method"
                and input_A_node.target == "permute"
                and check_permute(input_A_node)
            ):
                Atrans = True
                if len(input_A_node.args) > 0:
                    input_A = input_A_node.args[0]
                else:
                    input_A = input_A_node.kwargs["input"]

            if (
                input_B_node.op == "call_method"
                and input_B_node.target == "permute"
                and check_permute(input_B_node)
            ):
                Btrans = True
                if len(input_B_node.args) > 0:
                    input_B = input_B_node.args[0]
                else:
                    input_B = input_B_node.kwargs["input"]

            if Atrans or Btrans:
                with module.graph.inserting_before(node):
                    fused_node = module.graph.call_function(
                        transpose_matmul,
                        args=(input_A, input_B, Atrans, Btrans),
                    )
                node.replace_all_uses_with(fused_node)
                module.graph.erase_node(node)
                if Atrans and len(input_A_node.users) == 0:
                    module.graph.erase_node(input_A_node)
                if Btrans and len(input_B_node.users) == 0:
                    module.graph.erase_node(input_B_node)

    module.graph.lint()
    module.recompile()
    return module


# X1 = X.permute(0, 2, 1)
# Y1 = X1 * W1^T + bias1
# ---->
# Y2 = X1.transpose(-1, -2) * W1^T + bias1
def transpose_linear(
    input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    if bias is None:
        return torch.matmul(input.transpose(-1, -2), weight.t())
    return torch.matmul(input.transpose(-1, -2), weight.t()) + bias


def transpose_matmul(A: torch.Tensor, B: torch.Tensor, Atrans: bool, Btrans: bool):
    if Atrans:
        A = A.transpose(-1, -2)
    if Btrans:
        B = B.transpose(-1, -2)
    return torch.matmul(A, B)
