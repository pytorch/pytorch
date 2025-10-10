# mypy: allow-untyped-defs
import logging
from typing import Optional

import torch
from torch import Tensor
from torch._dynamo.utils import counters, is_node_meta_valid

from .. import config
from ..pattern_matcher import (
    CallFunctionVarArgs,
    Match,
    MULTIPLE,
    register_graph_pattern,
)
from .split_cat import construct_pattern_matcher_pass


log = logging.getLogger(__name__)


def check_device(a: Tensor, b: Optional[Tensor], device="cuda") -> bool:
    # Collect all non-None tensors
    tensors = [t for t in (a, b) if t is not None]
    # Check device type for each tensor
    return all(t.device.type == device for t in tensors)


def should_replace_norm(
    input: torch.fx.Node,
    weight: Optional[torch.fx.Node] = None,
) -> bool:
    if not is_node_meta_valid(input) or not is_node_meta_valid(weight):
        return False
    input = input.meta["example_value"]
    weight = weight.meta["example_value"] if weight is not None else None

    return check_device(input, weight)


def print_norm_pattern(match: Match, inputs: list[torch.fx.Node | None]):
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
    CallFunctionVarArgs(torch.nn.functional.layer_norm, users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("replace_layer_norm_pass"),
)
def layer_norm_replacement(
    match: Match,
    input: torch.fx.Node,
    normalized_shape: list[int],
    weight: Optional[torch.fx.Node] = None,
    bias: Optional[torch.fx.Node] = None,
    eps: Optional[float] = 1e-5,
):
    if config.pre_grad_fusion_options["replace_layer_norm_pass"].get(
        "rmsnorm", False
    ) and should_replace_norm(input, weight):
        graph = match.graph
        layer_norm_node = match.nodes[-1]
        kwargs = layer_norm_node.kwargs
        # rmsnorm does not use bias
        if "eps" not in kwargs:
            new_kwargs = {"weight": kwargs.get("weight", None)}
        else:
            new_kwargs = {
                "weight": kwargs.get("weight", None), "eps": kwargs.get("eps", 1e-5)
            }
        with graph.inserting_after(layer_norm_node):
            mrs_norm_node = graph.call_function(
                torch.nn.functional.rms_norm,
                args=layer_norm_node.args,
                kwargs=new_kwargs,
            )
            layer_norm_node.replace_all_uses_with(mrs_norm_node)
            mrs_norm_node.meta.update(layer_norm_node.meta)
            graph.erase_node(layer_norm_node)
        counters["inductor"]["replace_layer_norm_pass"] += 1
        print_norm_pattern(match, [input, weight])
