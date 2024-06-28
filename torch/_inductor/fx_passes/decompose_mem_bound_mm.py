# mypy: allow-untyped-defs
import logging
from typing import List

import torch
from torch import Tensor
from torch._dynamo.utils import counters

from .. import config

from ..pattern_matcher import Arg, CallFunction, Match, register_graph_pattern
from .split_cat import construct_pattern_matcher_pass

aten = torch.ops.aten
log = logging.getLogger(__name__)

# TODO: need a better strategy for decomposing mm
MIN_FIRST_DIMENSION_DECOMPOSITION = 10240
MAX_OTHER_DIMENSION_DECOMPOSITION = 32

min_first_dimension_decomposition = MIN_FIRST_DIMENSION_DECOMPOSITION
max_other_dimention_decomposition = MAX_OTHER_DIMENSION_DECOMPOSITION
if "decompose_mm_pass" in config.post_grad_fusion_options:
    min_first_dimension_decomposition = config.post_grad_fusion_options[
        "decompose_mm_pass"
    ].get("min_first_dimension_decomposition", MIN_FIRST_DIMENSION_DECOMPOSITION)
    max_other_dimention_decomposition = config.post_grad_fusion_options[
        "decompose_mm_pass"
    ].get("max_other_dimention_decomposition", MAX_OTHER_DIMENSION_DECOMPOSITION)


def check_device(a: Tensor, b: Tensor) -> bool:
    return a.is_cuda and b.is_cuda


def realize_inputs(inputs: List[torch.fx.Node]):
    for inp in inputs:
        if isinstance(inp, torch.fx.node.Node):
            inp.meta["inductor_realize_to_strides"] = True


def should_decompose_bmm(mat1, mat2) -> bool:
    if is_node_meta_valid(mat1) and is_node_meta_valid(mat2):
        mat1 = mat1.meta["val"]
        mat2 = mat2.meta["val"]
    else:
        return False
    if not check_device(mat1, mat2):
        return False
    else:
        if len(mat1.shape) != 3 or len(mat2.shape) != 3:
            return False
        if mat1.shape[0] < min_first_dimension_decomposition:
            return False
        # 2 of m, n, k must be <= MAX_OTHER_DIMENSION_DECOMPOSITION
        if (mat1.shape[1] < max_other_dimention_decomposition) + (
            mat1.shape[2] < max_other_dimention_decomposition
        ) + (mat2.shape[2] < max_other_dimention_decomposition) < 2:
            return False
    return True


def should_decompose_mm(mat1, mat2) -> bool:
    if is_node_meta_valid(mat1) and is_node_meta_valid(mat2):
        mat1 = mat1.meta["val"]
        mat2 = mat2.meta["val"]
    else:
        return False
    return (
        check_device(mat1, mat2)
        and len(mat1.shape) == 2
        and len(mat2.shape) == 2
        and mat1.shape[0] >= min_first_dimension_decomposition
        and mat2.shape[0] < max_other_dimention_decomposition
        and mat2.shape[1] < max_other_dimention_decomposition
    )


def is_node_meta_valid(node: torch.fx.Node):
    return "val" in node.meta


def print_decompose_pattern(match: Match, inputs: List[torch.fx.Node]):
    node = match.nodes[-1]
    log.debug(
        "Decompose %s with input shape: %s",
        node.target,
        ", ".join(
            str(input.meta["val"].shape) if "val" in input.meta else "None"
            for input in inputs
        ),
    )


@register_graph_pattern(
    CallFunction(aten.bmm, Arg(), Arg()),
    pass_dict=construct_pattern_matcher_pass("decompose_mm_pass"),
)
def decompose_bmm(match: Match, mat1: torch.fx.Node, mat2: torch.fx.Node):
    def repl(mat1, mat2):
        return torch.sum(mat1[:, :, :, None] * mat2[:, None, :, :], dim=-2)

    if should_decompose_bmm(mat1, mat2):
        counters["inductor"]["decompose_bmm"] += 1
        match.replace_by_example(repl, [mat1, mat2])
        print_decompose_pattern(match, [mat1, mat2])
        realize_inputs([mat1, mat2])
    return


@register_graph_pattern(
    CallFunction(aten.addmm, Arg(), Arg(), Arg()),
    pass_dict=construct_pattern_matcher_pass("decompose_mm_pass"),
)
def decompose_addmm(
    match: Match,
    mat1: torch.fx.Node,
    mat2: torch.fx.Node,
    mat3: torch.fx.Node,
):
    def repl(mat1, mat2, mat3):
        return torch.sum(mat2[:, :, None] * mat3[None, :, :], dim=-2) + mat1

    if should_decompose_mm(mat2, mat3):
        counters["inductor"]["decompose_addmm"] += 1
        match.replace_by_example(repl, [mat1, mat2, mat3])
        print_decompose_pattern(match, [mat1, mat2, mat3])
        realize_inputs([mat1, mat2, mat3])
    return


@register_graph_pattern(
    CallFunction(aten.mm, Arg(), Arg()),
    pass_dict=construct_pattern_matcher_pass("decompose_mm_pass"),
)
def decompose_mm(
    match: Match,
    mat1: torch.fx.Node,
    mat2: torch.fx.Node,
):
    def repl(mat1, mat2):
        return torch.sum(mat1[:, :, None] * mat2[None, :, :], dim=-2)

    if should_decompose_mm(mat1, mat2):
        counters["inductor"]["decompose_mm"] += 1
        match.replace_by_example(repl, [mat1, mat2])
        print_decompose_pattern(match, [mat1, mat2])
        realize_inputs([mat1, mat2])
    return
