import torch
from torch.fx import Node
from typing import (
    List,
    Callable,
)

def _is_share_obs_or_fq_op(op: Callable) -> bool:
    return op in [
        torch.ops.aten.hardtanh.default,
        torch.ops.aten.mean.default,
        torch.ops.aten.mean.dim,
        torch.ops.aten.adaptive_avg_pool2d.default,
        torch.ops.aten.view_copy.default,
        torch.ops.aten.view.default,
    ]

def propagate_annotation(model: torch.fx.GraphModule) -> None:
    for n in model.graph.nodes:
        if n.op != "call_function" or not _is_share_obs_or_fq_op(n.target):
            continue

        prev_node = n.args[0]
        if not isinstance(prev_node, Node):
            continue

        target_dtype_info = prev_node.meta.get("target_dtype_info", None)
        if not target_dtype_info:
            continue

        output_act_obs_or_fq_ctr = target_dtype_info.get("output_act_obs_or_fq_ctr", None)
        if not output_act_obs_or_fq_ctr:
            continue

        # make sure current node is not annotated
        if "target_dtype_info" in n.meta and n.meta["target_dtype_info"].get("_annotated", False):
            continue

        # propagate the previous output_act_obs_or_fq to the current node
        n.meta["target_dtype_info"] = {
            "input_act_obs_or_fq_ctr": output_act_obs_or_fq_ctr,
            "output_act_obs_or_fq_ctr": output_act_obs_or_fq_ctr,
            "input_output_share_observers": True,
            "_annotated": True,
        }
