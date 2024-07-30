# mypy: allow-untyped-defs
from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch._dynamo.utils import counters
from torch._inductor.utils import InputType

perf_hint_log = torch._logging.getArtifactLogger(__name__, "perf_hints")


OutputType = List[Optional[Union[int, torch.Tensor]]]
ModelType = Callable[[List[InputType]], OutputType]


@dataclasses.dataclass(frozen=True)
class FunctionID:
    "Unique counter of a function wrapped in cudagraphify_impl"
    id: int


@dataclasses.dataclass(frozen=True)
class PlaceholderInfo:
    """
    A serializable version of torch.fx.Node that contains information
    pertinent to placeholder stack traces. We use these in logging and error messages
    related to cudagraphs, and will cache these results.
    """

    name: str
    stack_trace: Optional[str]
    # This field is recursive, but never cyclic (since a node never uses itself)
    users: List[PlaceholderInfo]
    mutating_use_stack_trace: Optional[str]


@dataclasses.dataclass(frozen=True)
class WrappedFunction:
    """
    Represents a function that you want to record for CUDA graph replay,
    with a little more metadata so we can identify if we have an applicable
    CUDA graph in our CUDA graph tree for it.
    """

    model: Callable[..., Any]
    static_input_idxs: Sequence[int]
    id: FunctionID
    constants: Tuple[torch.Tensor, ...]
    placeholders: Sequence[PlaceholderInfo]
    mutated_input_idxs: Sequence[int]


def get_mutating_use_stack_trace_from_node(
    placeholder_node: torch.fx.Node,
) -> Optional[str]:
    # reinplaced uses might have a single, non-copy_ use
    if len(placeholder_node.users) == 1:
        return next(iter(placeholder_node.users)).meta.get("stack_trace", None)

    for use in placeholder_node.users:
        if use.target == torch.ops.aten.copy_.default:
            if stack_trace := use.meta.get("stack_trace", None):
                return stack_trace

    return None


def get_mutating_use_stack_trace(placeholder_info: PlaceholderInfo) -> Optional[str]:
    return placeholder_info.mutating_use_stack_trace


def to_placeholder_info(placeholder_node: torch.fx.Node) -> PlaceholderInfo:
    name = placeholder_node.name
    stack_trace = placeholder_node.meta.get("stack_trace", None)
    users = []
    mutating_use_stack_trace = None
    # Only recurse to users once, since we only care about user's stack traces
    if placeholder_node.op == "placeholder":
        users = [to_placeholder_info(i) for i in placeholder_node.users]
        mutating_use_stack_trace = get_mutating_use_stack_trace_from_node(
            placeholder_node
        )

    return PlaceholderInfo(name, stack_trace, users, mutating_use_stack_trace)


def get_placeholder_info(graph: torch.fx.Graph) -> List[PlaceholderInfo]:
    return [
        to_placeholder_info(node) for node in graph.nodes if node.op == "placeholder"
    ]


def format_default_skip_message(reason: str) -> str:
    return f"skipping cudagraphs due to {reason}"


def get_mutation_stack_trace(
    placeholders: Sequence[PlaceholderInfo], mutation_indices: Sequence[int]
) -> str:
    stack_trace: Optional[str] = ""

    for idx in mutation_indices:
        placeholder = placeholders[idx]
        if stack_trace := get_mutating_use_stack_trace(placeholder):
            break

    msg = format_default_skip_message(
        f"mutated inputs ({len(mutation_indices)} instances)"
    )
    if stack_trace:
        return f"{msg}. Found from : \n {stack_trace}"

    return msg


def check_for_mutation(
    func: WrappedFunction,
    inputs: List[InputType],
    is_cuda_graph_recorded_tensor: Callable[[torch.Tensor], bool],
) -> Optional[str]:
    # doesnt work for non-trees because the warmup run would apply mutation twice
    if torch._inductor.config.triton.cudagraph_trees:
        # checking if mutation is only on parameters/static inputs
        mutation_indices: Sequence[int] = [
            idx
            for idx in func.mutated_input_idxs
            if not (
                idx in func.static_input_idxs
                or is_cuda_graph_recorded_tensor(inputs[idx])  # type: ignore[arg-type]
            )
        ]
    else:
        mutation_indices = func.mutated_input_idxs

    return (
        get_mutation_stack_trace(func.placeholders, mutation_indices)
        if mutation_indices
        else None
    )


def _get_use_stack_trace(node) -> Optional[str]:
    for use in node.users:
        if stack_trace := use.meta.get("stack_trace", None):
            return stack_trace
    return None


def check_multiple_devices_or_any_cpu_nodes(
    device_node_mapping: Dict[torch.device, torch.fx.Node]
) -> Optional[str]:
    if cpu_node := device_node_mapping.get(torch.device("cpu")):
        msg = f"cpu device ({cpu_node.name})"
        if stack_trace := _get_use_stack_trace(cpu_node):
            return format_default_skip_message(f"{msg}. Found from : \n {stack_trace}")

        return format_default_skip_message(msg)

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


def log_cudagraph_skip_and_bump_counter(msg):
    perf_hint_log.warning(msg)
    counters["inductor"]["cudagraph_skips"] += 1


@dataclasses.dataclass
class BoxedDeviceIndex:
    value: Optional[int]

    def set(self, device_idx: Optional[int]):
        assert device_idx is None or isinstance(device_idx, int)
        self.value = device_idx


def check_for_mutation_ignore_cuda_graph_managed_tensor(
    gm: torch.fx.GraphModule, compiled_graph, static_input_idxs: Sequence[int]
) -> Optional[str]:
    default_msg = format_default_skip_message("mutated inputs")

    # doesnt work for non-trees because the warmup run would apply mutation twice
    if torch._inductor.config.triton.cudagraph_trees:
        unique_idxs = set(static_input_idxs)
        # checking if mutation is only on parameters/static inputs
        mutation_indices = [
            idx for idx in compiled_graph.mutated_input_idxs if idx not in unique_idxs
        ]
        has_mutation = len(mutation_indices) != 0
        if not has_mutation:
            return None
        placeholders = get_placeholder_info(gm.graph)
        return get_mutation_stack_trace(placeholders, mutation_indices)

    else:
        has_mutation = len(compiled_graph.mutated_inputs) != 0
        return None if not has_mutation else default_msg


def get_placeholder_stack_trace(placeholder: PlaceholderInfo) -> Optional[str]:
    """
    Gets the first non-empty stack trace of a placeholder or its users.
    """
    if placeholder.stack_trace:
        return placeholder.stack_trace

    for user in placeholder.users:
        if user.stack_trace:
            return user.stack_trace

    return None


class CheckInvariantStatus(Enum):
    # Check invariant succeeded
    SUCCESS = 1

    # Previously managed data pointers are not stable
    CudagraphManagedIdxMismatch = 2

    # Static tensor input addresses are not stable
    StaticInputIdxMismatch = 3

    # Expected dead indices before graph are live
    ExpectedDeadIndicesBeforeGraphMismatch = 4

    def __str__(self):
        if self.name == "CudagraphManagedIdxMismatch":
            return "cudagraph managed tensor data pointer changed"
        elif self.name == "StaticInputIdxMismatch":
            return "static input data pointer changed"
        elif self.name == "ExpectedDeadIndicesBeforeGraphMismatch":
            return "expected dead indices before graph are live"
        else:
            return f"{self.name}: {self.value}"


def log_data_ptr_mismatch(
    placeholders: Sequence[PlaceholderInfo],
    inputs: List[InputType],
    recorded_data_ptr: Sequence[Optional[int]],
    target_idxs: Sequence[int],
    mismatch: CheckInvariantStatus,
) -> str:
    """
    Logs the mismatch between input data pointers and recorded data pointers.
    This checks only idxs in target_idxs.
    """
    assert len(inputs) == len(recorded_data_ptr) and len(inputs) == len(
        placeholders
    ), "length mismatch between inputs, recorded_data_ptr, and placeholders"

    t_tensors = [inputs[i] for i in target_idxs]
    t_data_ptrs = [recorded_data_ptr[i] for i in target_idxs]
    error_msg = f"{mismatch}.\n"
    for i, (tensor, data_ptr) in enumerate(zip(t_tensors, t_data_ptrs)):
        assert isinstance(tensor, torch.Tensor)
        index = target_idxs[i]
        if tensor.data_ptr() != data_ptr:
            placeholder = placeholders[index]
            error_msg = (
                f"{error_msg}input name: {placeholder.name}. "
                f"data pointer changed from {data_ptr} to {tensor.data_ptr()}. "
                f"input stack trace: {get_placeholder_stack_trace(placeholder)}\n"
            )
    return error_msg


def maybe_warning_due_to_dynamic_shape(
    fn_cache: Dict[Tuple[int, ...], Callable[..., Any]],
    new_int_key: Any,
):
    num_cudagraphs = len(fn_cache.keys()) + 1

    def warn_msg():
        return (
            "CUDAGraph supports dynamic shapes by recording a new graph for each "
            "distinct input size. Recording too many CUDAGraphs may lead to "
            f"extra overhead. We have observed {num_cudagraphs} distinct sizes. "
            "Please consider the following options for better performance: "
            "a) padding inputs to a few fixed number of shapes; or b) set "
            "torch._inductor.config.triton.cudagraph_skip_dynamic_graphs=True. "
            "Set torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit=None "
            "to silence this warning."
        )

    if (
        torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit
        and num_cudagraphs
        > torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit
    ):
        perf_hint_log.warning(warn_msg())
