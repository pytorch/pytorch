from typing import Any, Dict, List, Optional

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


def get_use_stack_trace(node) -> Optional[str]:
    for use in node.users:
        if stack_trace := use.meta.get("stack_trace", None):
            return stack_trace
    return None


def check_multiple_devices(
    device_node_mapping: Dict[torch.device, torch.fx.Node]
) -> Optional[str]:
    if cpu_node := device_node_mapping.get(torch.device("cpu")):
        if stack_trace := get_use_stack_trace(cpu_node):
            return format_default_skip_message(
                f"cpu device. Found from : \n {stack_trace}"
            )

    if (
        len(device_node_mapping) == 1
        and next(iter(device_node_mapping.keys())).type == "cuda"
    ):
        return None

    keys_repr = (repr(key) for key in device_node_mapping.keys())
    return format_default_skip_message(f"multiple devices: {', '.join(keys_repr)}")


def check_lowering_cudagraph_checks(
    device_node_mapping: Dict[torch.device, torch.fx.Node]
) -> Optional[str]:
    return check_multiple_devices(device_node_mapping)


def check_for_incompatible_cudagraph_ops(gm):
    forbidden_set = {
        "aten._fused_moving_avg_obs_fq_helper.default",
        "aten._fused_moving_avg_obs_fq_helper_functional.default",
        "aten.multinomial.default",
        "fbgemm.dense_to_jagged.default",
        "fbgemm.jagged_to_padded_dense.default",
        "run_and_save_rng_state",
        "run_with_rng_state",
        "aten._local_scalar_dense",
    }
    if torch.are_deterministic_algorithms_enabled():
        forbidden_set.update(
            {
                "aten._unsafe_index_put.default",
                "aten.index_put.default",
                "aten.index_put_.default",
                "aten.scatter.src",
                "aten.scatter.reduce",
                "aten.scatter.value_reduce",
                "aten.scatter_add_",
                "aten.scatter_add.default",
                "aten.scatter_reduce.two",
                "aten.scatter_reduce_.two",
                "aten.scatter_reduce.two_out",
            }
        )

    for node in gm.graph.nodes:
        if str(node.target) in forbidden_set:
            if stack_trace := node.meta.get("stack_trace", None):
                return format_default_skip_message(
                    f"incompatible ops. Found from {stack_trace}"
                )

            return format_default_skip_message("incompatible ops")

    return None


def check_post_lowering_cudagraph_disable_reason(
    example_inputs: List[Any],
    gm: torch.fx.GraphModule,
    compiled_graph: CompiledFxGraph,
    num_fixed: int,
):
    if has_mutation_str := check_for_mutation(gm, compiled_graph, num_fixed):
        return has_mutation_str

    if incompat_op_str := check_for_incompatible_cudagraph_ops(gm):
        return incompat_op_str

    complex_memory_overlap_inputs = any(
        torch._inductor.compile_fx.complex_memory_overlap(t)
        for t in example_inputs
        if isinstance(t, torch.Tensor)
    )

    if complex_memory_overlap_inputs:
        return "skipping cudagraphs due to complex memory overlap"

    non_tensor_inputs = not all(
        isinstance(t, (torch.Tensor, torch.SymInt)) for t in example_inputs
    )

    if non_tensor_inputs:
        return "skipping cudagraphs due to non-Tensor inputs"

    return None
