# mypy: allow-untyped-defs
import logging
from typing import Optional

import torch

from torch._dynamo.utils import counters, is_node_meta_valid

from .. import config

from ..kernel.quack.quack_rmsnorm import rmsnorm as quack_rmsnorm
from ..pattern_matcher import (
    CallFunctionVarArgs,
    Match,
    MULTIPLE,
    register_graph_pattern,
)
from .split_cat import construct_pattern_matcher_pass

log = logging.getLogger(__name__)


def check_device(inputs: list[torch.Tensor], device="cuda") -> bool:

    return all(input.device.type == device for input in inputs)


def should_replace_norm(inputs: list[Optional[torch.fx.Node]]) -> bool:
    inputs = [input for input in inputs if input is not None]

    if not all(is_node_meta_valid(input) for input in inputs):
        return False

    return check_device([input.meta["example_value"] for input in inputs])


def print_norm_pattern(match: Match, inputs: list[Optional[torch.fx.Node]]):
    node = match.nodes[-1]
    log.debug(
        "replace layer_norm %s with input shape: %s",
        node.target,
        ", ".join(
            str(input.meta["example_value"].shape) if input is not None else "None"
            for input in inputs
        ),
    )


@register_graph_pattern(
    CallFunctionVarArgs([torch.nn.functional.rms_norm, torch.rms_norm], users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("use_custom_rmsnorm_kernel_pass"),
)
def rms_norm_replacement(
    match: Match,
    input: torch.fx.Node,
    normalized_shape: list[int],
    weight: Optional[torch.fx.Node] = None,
    eps: Optional[float] = None,
):
    def repl(input, weight, eps):
        if config.pre_grad_fusion_options["use_custom_rmsnorm_kernel_pass"].get("quack", False):
            out, _ = quack_rmsnorm(input, weight, eps=eps)
            return out

    if weight is None:
        # quack rmsnorm does not support weight is None
        return False
    
    if should_replace_norm([input, weight]):
        counters["inductor"]["use_custom_rmsnorm_kernel_pass"] += 1
        if eps is None:
            eps = torch.finfo(input.meta["example_value"].dtype).eps
        match.replace_by_example(repl, [input, weight, eps])
        print_norm_pattern(match, [input, weight])
