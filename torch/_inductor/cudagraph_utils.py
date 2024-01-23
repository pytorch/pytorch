from typing import Optional

import torch
from torch._inductor.codecache import CompiledFxGraph


def get_mutating_use_stack_trace(placeholder_node) -> Optional[str]:
    # reinplaced uses might have a single, non-copy_ use
    if len(placeholder_node.users) == 1:
        return next(placeholder_node.users).meta.get("stack_trace", None)

    for use in placeholder_node.users:
        if use.target == torch.ops.aten.copy_.default:
            if stack_trace := use.meta.get("stack_trace", None):
                return stack_trace

    return None


def format_default_skip_message(reason: str) -> str:
    return f"skipping cudagraphs due to {reason}"


def check_for_mutation(
    gm: torch.fx.GraphModule, compiled_graph: CompiledFxGraph, num_fixed: int
) -> Optional[str]:
    default_msg = format_default_skip_message("mutated inputs")

    # doesnt work for non-trees because the warmup run would apply mutation twice
    if torch._inductor.config.triton.cudagraph_trees:
        # checking if mutation is only on parameters/static inputs
        mutation_indices = [
            idx for idx in compiled_graph.mutated_input_idxs if idx >= num_fixed
        ]
        has_mutation = len(mutation_indices) != 0

        if not has_mutation:
            return None

        stack_trace: Optional[str] = ""
        placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]

        for idx in mutation_indices:
            placeholder = placeholders[idx]
            if stack_trace := get_mutating_use_stack_trace(placeholder):
                break

        if stack_trace:
            msg = f"skipping cudagraphs due to mutaton on input. Found from : \n {stack_trace}"
            return msg

        return default_msg

    else:
        has_mutation = len(compiled_graph.mutated_inputs) != 0
        return None if not has_mutation else default_msg
