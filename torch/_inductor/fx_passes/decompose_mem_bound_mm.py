import logging
from typing import List, Optional

import torch
from torch import Tensor
from torch._dynamo.utils import counters
from torch._inductor import utils

from ..pattern_matcher import (
    Arg,
    CallFunction,
    config_flag,
    Ignored,
    Match,
    register_graph_pattern,
)
from .post_grad import decompose_mm_pass

aten = torch.ops.aten
log = logging.getLogger(__name__)

# TODO: need a better strategy for decomposing mm
MIN_FIRST_DIMENSION_DECOMPOSITION = 10240
MAX_OTHER_DIMENSION_DECOMPOSITION = 32


def check_device(a: Tensor, b: Tensor) -> bool:
    return a.is_cuda and b.is_cuda


def should_decompose_common(
    mat1: Tensor, mat2: Tensor, input: Optional[Tensor] = None
) -> bool:
    return (
        torch._inductor.config.decompose_mem_bound_mm
        and check_device(mat1, mat2)
        and not utils.any_is_symbolic(mat1, mat2, input)
    )


def should_decompose_bmm(mat1, mat2) -> bool:
    if is_node_meta_valid(mat1) and is_node_meta_valid(mat2):
        mat1 = mat1.meta["val"]
        mat2 = mat2.meta["val"]
    else:
        return False
    if not should_decompose_common(mat1, mat2):
        return False
    else:
        if len(mat1.shape) != 3 or len(mat2.shape) != 3:
            return False
        if mat1.shape[0] < MIN_FIRST_DIMENSION_DECOMPOSITION:
            return False
        # 2 of m, n, k must be <= MAX_OTHER_DIMENSION_DECOMPOSITION
        if (mat1.shape[1] < MAX_OTHER_DIMENSION_DECOMPOSITION) + (
            mat1.shape[2] < MAX_OTHER_DIMENSION_DECOMPOSITION
        ) + (mat2.shape[2] < MAX_OTHER_DIMENSION_DECOMPOSITION) < 2:
            return False
    return True


def should_decompose_mm(mat1, mat2) -> bool:
    if is_node_meta_valid(mat1) and is_node_meta_valid(mat2):
        mat1 = mat1.meta["val"]
        mat2 = mat2.meta["val"]
    else:
        return False
    return (
        should_decompose_common(mat1, mat2)
        and len(mat1.shape) == 2
        and len(mat2.shape) == 2
        and mat1.shape[0] >= MIN_FIRST_DIMENSION_DECOMPOSITION
        and mat2.shape[0] < MAX_OTHER_DIMENSION_DECOMPOSITION
        and mat2.shape[1] < MAX_OTHER_DIMENSION_DECOMPOSITION
    )


def should_decompose_mmt(mat1, mat2) -> bool:
    if is_node_meta_valid(mat1) and is_node_meta_valid(mat2):
        mat1 = mat1.meta["val"]
        mat2 = mat2.meta["val"]
    else:
        return False
    return (
        should_decompose_common(mat1, mat2)
        and len(mat1.shape) == 2
        and len(mat2.shape) == 2
        and mat1.shape[0] >= MIN_FIRST_DIMENSION_DECOMPOSITION
        and mat1.shape[1] < MAX_OTHER_DIMENSION_DECOMPOSITION
        and mat2.shape[1] < MAX_OTHER_DIMENSION_DECOMPOSITION
    )


def should_decompose_mm_largek(mat1, mat2) -> bool:
    if is_node_meta_valid(mat1) and is_node_meta_valid(mat2):
        mat1 = mat1.meta["val"]
        mat2 = mat2.meta["val"]
    else:
        return False
    return (
        should_decompose_common(mat1, mat2)
        and len(mat1.shape) == 2
        and len(mat2.shape) == 2
        and mat1.shape[1] >= MIN_FIRST_DIMENSION_DECOMPOSITION
        and mat1.shape[0] < MAX_OTHER_DIMENSION_DECOMPOSITION
        and mat2.shape[1] < MAX_OTHER_DIMENSION_DECOMPOSITION
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
    pass_dict=decompose_mm_pass,
    extra_check=config_flag("decompose_mem_bound_mm"),
)
def decompose_bmm(match: Match, mat1: torch.fx.Node, mat2: torch.fx.Node):
    def repl(mat1, mat2):
        return torch.sum(mat1[:, :, :, None] * mat2[:, None, :, :], dim=-2)

    if should_decompose_bmm(mat1, mat2):
        counters["inductor"]["decompose_bmm"] += 1
        match.replace_by_example(repl, [mat1, mat2])
        print_decompose_pattern(match, [mat1, mat2])
    return


@register_graph_pattern(
    CallFunction(aten.addmm, Arg(), Arg(), Arg()),
    pass_dict=decompose_mm_pass,
    extra_check=config_flag("decompose_mem_bound_mm"),
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
    return


@register_graph_pattern(
    CallFunction(aten.mm, CallFunction(aten.permute, Arg(), Ignored()), Arg()),
    pass_dict=decompose_mm_pass,
    extra_check=config_flag("decompose_mem_bound_mm"),
)
def decompose_mmt(
    match: Match,
    mat1: torch.fx.Node,
    mat2: torch.fx.Node,
):
    def repl(mat1, mat2):
        return torch.sum(mat1[:, :, None] * mat2[:, None, :], dim=0)

    if should_decompose_mmt(mat1, mat2):
        counters["inductor"]["decompose_mmt"] += 1
        match.replace_by_example(repl, [mat1, mat2])
        print_decompose_pattern(match, [mat1, mat2])
    return


@register_graph_pattern(
    CallFunction(aten.mm, Arg(), Arg()),
    pass_dict=decompose_mm_pass,
    extra_check=config_flag("decompose_mem_bound_mm"),
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
    return


@register_graph_pattern(
    CallFunction(aten.mm, Arg(), Arg()),
    pass_dict=decompose_mm_pass,
    extra_check=config_flag("decompose_mem_bound_mm"),
)
def decompose_mm_large_k(
    match: Match,
    mat1: torch.fx.Node,
    mat2: torch.fx.Node,
):
    def repl(mat1, mat2):
        mat1 = mat1.permute(1, 0)
        return torch.sum(mat1[:, :, None] * mat2[:, None, :], dim=0)

    if should_decompose_mm_largek(mat1, mat2):
        counters["inductor"]["decompose_mm_large_k"] += 1
        match.replace_by_example(repl, [mat1, mat2])
        print_decompose_pattern(match, [mat1, mat2])
    return
