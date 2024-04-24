import dataclasses
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch


@dataclasses.dataclass(frozen=True)
class FunctionID:
    "Unique counter of a function wrapped in cudagraphify_impl"
    id: int


@dataclasses.dataclass(frozen=True)
class WrappedFunction:
    """
    Represents a function that you want to record for CUDA graph replay,
    with a little more metadata so we can identify if we have an applicable
    CUDA graph in our CUDA graph tree for it.
    """

    model: Callable[..., Any]
    static_input_idxs: List[int]
    id: FunctionID
    constants: Tuple[torch.Tensor, ...]
    placeholders: List[torch.fx.Node]
    mutated_input_idxs: List[int]


def get_placeholders(graph: torch.fx.Graph) -> List[torch.fx.Node]:
    return [node for node in graph.nodes if node.op == "placeholder"]


def get_mutating_use_stack_trace(placeholder_node: torch.fx.Node) -> Optional[str]:
    # reinplaced uses might have a single, non-copy_ use
    if len(placeholder_node.users) == 1:
        return next(iter(placeholder_node.users)).meta.get("stack_trace", None)

    for use in placeholder_node.users:
        if use.target == torch.ops.aten.copy_.default:
            if stack_trace := use.meta.get("stack_trace", None):
                return stack_trace

    return None


def format_default_skip_message(reason: str) -> str:
    return f"skipping cudagraphs due to {reason}"


def get_mutation_stack_trace(
    placeholders: List[torch.fx.Node], mutation_indices: Iterable[int]
) -> str:
    stack_trace: Optional[str] = ""

    for idx in mutation_indices:
        placeholder = placeholders[idx]
        if stack_trace := get_mutating_use_stack_trace(placeholder):
            break

    if stack_trace:
        msg = f"skipping cudagraphs due to mutation on input. Found from : \n {stack_trace}"
        return msg

    return format_default_skip_message("mutated inputs")


def check_for_mutation(
    func: WrappedFunction,
    inputs: List[torch.Tensor],
    is_cuda_graph_recorded_tensor: Callable[[torch.Tensor], bool],
) -> Optional[str]:
    default_msg = format_default_skip_message("mutated inputs")

    # doesnt work for non-trees because the warmup run would apply mutation twice
    if torch._inductor.config.triton.cudagraph_trees:
        # checking if mutation is only on parameters/static inputs
        mutation_indices = [
            idx
            for idx in func.mutated_input_idxs
            if not (
                idx in func.static_input_idxs
                or is_cuda_graph_recorded_tensor(inputs[idx])
            )
        ]
        has_mutation = len(mutation_indices) != 0
        if not has_mutation:
            return None

        return get_mutation_stack_trace(func.placeholders, mutation_indices)

    else:
        has_mutation = len(func.mutated_input_idxs) != 0
        return None if not has_mutation else default_msg


def get_use_stack_trace(node) -> Optional[str]:
    for use in node.users:
        if stack_trace := use.meta.get("stack_trace", None):
            return stack_trace
    return None


def check_multiple_devices_or_any_cpu_nodes(
    device_node_mapping: Dict[torch.device, torch.fx.Node]
) -> Optional[str]:
    if cpu_node := device_node_mapping.get(torch.device("cpu")):
        if stack_trace := get_use_stack_trace(cpu_node):
            return format_default_skip_message(
                f"cpu device. Found from : \n {stack_trace}"
            )

        return format_default_skip_message("cpu device")

    if (
        len(device_node_mapping) == 1
        and next(iter(device_node_mapping.keys())).type == "cuda"
    ):
        return None

    keys_repr = (repr(key) for key in device_node_mapping.keys())
    return format_default_skip_message(f"multiple devices: {', '.join(keys_repr)}")


def check_lowering_disable_cudagraph(
    device_node_mapping: Dict[torch.device, torch.fx.Node]
):
    return check_multiple_devices_or_any_cpu_nodes(device_node_mapping)


@dataclasses.dataclass
class BoxedDeviceIndex:
    value: Optional[int]

    def set(self, device_idx: Optional[int]):
        assert device_idx is None or isinstance(device_idx, int)
        self.value = device_idx


def check_for_mutation_ignore_cuda_graph_managed_tensor(
    gm: torch.fx.GraphModule, compiled_graph, num_fixed: int
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
        placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
        return get_mutation_stack_trace(placeholders, mutation_indices)

    else:
        has_mutation = len(compiled_graph.mutated_inputs) != 0
        return None if not has_mutation else default_msg
