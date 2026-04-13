# mypy: disallow-untyped-defs
from __future__ import annotations

import dataclasses
from collections.abc import Callable
from enum import Enum
from typing import Any, TYPE_CHECKING, TypeVar

import torch
from torch._dynamo.utils import counters, get_metrics_context
from torch._inductor.utils import GraphPartitionMap, InputType
from torch._subclasses.fake_tensor import get_plain_tensors, is_fake
from torch.utils._ordered_set import OrderedSet

from .utils import is_using_cudagraph_partition


if TYPE_CHECKING:
    from collections.abc import Sequence, Set as AbstractSet

    from torch._inductor.output_code import OutputCode

_OC = TypeVar("_OC", bound="OutputCode")


cudagraphs_log = torch._logging.getArtifactLogger(__name__, "cudagraphs")
static_inputs_log = torch._logging.getArtifactLogger(
    __name__, "cudagraph_static_inputs"
)


OutputType = list[int | torch.Tensor | None]
ModelType = Callable[[list[InputType]], OutputType]


class CUDAGraphPolicy:
    """Pluggable policy controlling CUDA graph wrapping in Inductor's post_compile.

    Override methods to customize:
      - HOW compiled functions are cudagraph-wrapped (cudagraphify)
      - WHETHER inner CompiledFxGraphs should be wrapped (should_wrap)
      - OUTER wrapping of compound outputs like RegionalOutputCode (wrap_output)

    Set via ``torch._inductor.config.cudagraph_policy``.  When ``None``
    (the default), the existing built-in behaviour is used unchanged.

    Example usage::

        class MyCUDAGraphPolicy(CUDAGraphPolicy):
            def cudagraphify(self, model, example_inputs, static_input_idxs, **kwargs):
                return my_custom_wrapper(model, example_inputs, static_input_idxs)


        with torch._inductor.config.patch("cudagraph_policy", MyCUDAGraphPolicy()):
            compiled_fn = deserialize_artifacts(...)
    """

    def cudagraphify(
        self,
        model: Callable[..., Any],
        example_inputs: Sequence[InputType],
        static_input_idxs: Sequence[int],
        *,
        device_index: int,
        is_backward: bool,
        is_inference: bool,
        **kwargs: Any,
    ) -> Callable[..., Any]:
        """Wrap a single compiled callable with CUDA graph capture/replay.

        Called by ``cudagraph_post_compile`` for each ``CompiledFxGraph``.
        The default delegates to ``compile_fx.cudagraphify`` (cudagraph_trees).

        ``example_inputs`` are the example inputs at post_compile time.
        The default implementation does not forward them because
        ``compile_fx.cudagraphify`` defers graph recording to the first
        real call via an inner closure.  Subclasses that need the
        example inputs for warmup or static-input detection may use them.

        When ``config.graph_partition=True``, setting a CUDAGraphPolicy
        bypasses ``cudagraph_partition_post_compile`` (which wraps each
        partition individually) and routes through ``cudagraph_post_compile``
        instead, so this method wraps the *entire* callable, not individual
        partitions.  Subclasses that need per-partition control should
        handle partitioning internally.
        """
        from torch._inductor.compile_fx import cudagraphify

        return cudagraphify(
            model,
            static_input_idxs,
            device_index=device_index,
            is_backward=is_backward,
            is_inference=is_inference,
            **kwargs,
        )

    def should_wrap(self, compiled_graph: OutputCode) -> bool:
        """Whether to apply cudagraph wrapping to this CompiledFxGraph.

        Called for each inner ``CompiledFxGraph`` during ``post_compile``.
        Return ``False`` to skip wrapping (e.g. when wrapping at the outer
        level via ``wrap_output`` instead).

        Default: ``True`` (wrap everything, same as current behaviour).
        """
        return True

    def wrap_output(self, output_code: _OC) -> _OC:
        """Optional outer-level wrapping after inner post_compile completes.

        Called by ``_compile_fx_inner``, ``BundledOutputCodeLoadable.post_compile``,
        and ``FxGraphCacheLoadable.post_compile`` on the ``OutputCode`` returned
        from ``post_compile``.  Subclasses that only want to wrap specific
        output types should check ``isinstance`` and return the input
        unchanged for types they don't handle.

        Default: identity (no outer wrapping).
        """
        return output_code


@dataclasses.dataclass(frozen=True, slots=True)
class FunctionID:
    "Unique counter of a function wrapped in cudagraphify_impl"

    id: int


@dataclasses.dataclass(frozen=True, slots=True)
class PlaceholderInfo:
    """
    A serializable version of torch.fx.Node that contains information
    pertinent to placeholder stack traces. We use these in logging and error messages
    related to cudagraphs, and will cache these results.
    """

    name: str
    stack_trace: str | None
    # This field is recursive, but never cyclic (since a node never uses itself)
    users: list[PlaceholderInfo]
    mutating_use_stack_trace: str | None


@dataclasses.dataclass(frozen=True, slots=True)
class WrappedFunction:
    """
    Represents a function that you want to record for CUDA graph replay,
    with a little more metadata so we can identify if we have an applicable
    CUDA graph in our CUDA graph tree for it.
    """

    model: Callable[..., Any]
    static_input_idxs: Sequence[int]
    id: FunctionID
    constants: tuple[torch.Tensor, ...]
    placeholders: Sequence[PlaceholderInfo]
    mutated_input_idxs: Sequence[int]


def get_mutating_use_stack_trace_from_node(
    placeholder_node: torch.fx.Node,
) -> str | None:
    # reinplaced uses might have a single, non-copy_ use
    if len(placeholder_node.users) == 1:
        return next(iter(placeholder_node.users)).meta.get("stack_trace", None)

    for use in placeholder_node.users:
        if use.target is torch.ops.aten.copy_.default:
            if stack_trace := use.meta.get("stack_trace", None):
                return stack_trace

    return None


def get_mutating_use_stack_trace(placeholder_info: PlaceholderInfo) -> str | None:
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


def get_placeholder_info(graph: torch.fx.Graph) -> list[PlaceholderInfo]:
    return [
        to_placeholder_info(node) for node in graph.nodes if node.op == "placeholder"
    ]


def format_default_skip_message(reason: str) -> str:
    return f"skipping cudagraphs due to {reason}"


def get_mutation_stack_trace(
    placeholders: Sequence[PlaceholderInfo],
    mutation_indices: AbstractSet[int] | Sequence[int],
) -> str:
    stack_trace: str | None = ""

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
    inputs: list[InputType],
    is_cuda_graph_recorded_tensor: Callable[[torch.Tensor], bool],
) -> str | None:
    # doesn't work for non-trees because the warmup run would apply mutation twice
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

    static_inputs_log.debug(
        "check mutation static input indices: %s", func.static_input_idxs
    )
    static_inputs_log.debug("check mutation mutation indices: %s", mutation_indices)

    return (
        get_mutation_stack_trace(func.placeholders, mutation_indices)
        if mutation_indices
        else None
    )


def _get_use_stack_trace(node: torch.fx.Node) -> str | None:
    for use in node.users:
        if stack_trace := use.meta.get("stack_trace", None):
            return stack_trace
    return None


def check_multiple_devices_or_any_cpu_nodes(
    device_node_mapping: dict[torch.device, torch.fx.Node],
) -> str | None:
    # meta tensors are supported since there is no compute
    device_node_mapping.pop(torch.device("meta"), None)

    # dynamo cudagraph does not support graph partition
    if is_using_cudagraph_partition():
        # graph partition supports splitting on cpu op. So we can ignore cpu nodes.
        device_node_mapping.pop(torch.device("cpu"), None)

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

    keys_repr = (repr(key) for key in device_node_mapping)
    return format_default_skip_message(f"multiple devices: {', '.join(keys_repr)}")


def check_lowering_disable_cudagraph(
    device_node_mapping: dict[torch.device, torch.fx.Node],
) -> str | None:
    return check_multiple_devices_or_any_cpu_nodes(device_node_mapping)


def log_cudagraph_skip_and_bump_counter(msg: str) -> None:
    cudagraphs_log.warning(msg)
    counters["inductor"]["cudagraph_skips"] += 1

    if torch._inductor.config.triton.cudagraph_or_error:
        raise RuntimeError(msg)

    metrics_context = get_metrics_context()
    if metrics_context.in_progress():
        metrics_context.set("cudagraph_skip_reason", msg, overwrite=True)


@dataclasses.dataclass
class BoxedDeviceIndex:
    value: int | None

    def set(self, device_idx: int | None) -> None:
        assert device_idx is None or isinstance(device_idx, int)
        self.value = device_idx


def check_for_mutation_ignore_cuda_graph_managed_tensor(
    gm: torch.fx.GraphModule,
    mutated_inputs: OrderedSet[str],
    mutated_input_idxs: OrderedSet[int],
    static_input_idxs: Sequence[int],
) -> str | None:
    default_msg = format_default_skip_message("mutated inputs")

    # doesn't work for non-trees because the warmup run would apply mutation twice
    if torch._inductor.config.triton.cudagraph_trees:
        unique_idxs = OrderedSet(static_input_idxs)
        # checking if mutation is only on parameters/static inputs
        mutation_indices = [idx for idx in mutated_input_idxs if idx not in unique_idxs]
        has_mutation = len(mutation_indices) != 0
        if not has_mutation:
            return None
        placeholders = get_placeholder_info(gm.graph)
        return get_mutation_stack_trace(placeholders, mutation_indices)

    else:
        has_mutation = len(mutated_inputs) != 0
        return None if not has_mutation else default_msg


def get_placeholder_stack_trace(placeholder: PlaceholderInfo) -> str | None:
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

    def __str__(self) -> str:
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
    inputs: list[InputType],
    recorded_data_ptr: Sequence[int | None],
    target_idxs: Sequence[int],
    mismatch: CheckInvariantStatus,
) -> str:
    """
    Logs the mismatch between input data pointers and recorded data pointers.
    This checks only idxs in target_idxs.
    """
    assert len(inputs) == len(recorded_data_ptr) and len(inputs) == len(placeholders), (
        "length mismatch between inputs, recorded_data_ptr, and placeholders"
    )

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
    fn_cache: dict[tuple[int, ...], Callable[..., Any]],
    new_int_key: Any,
) -> bool:
    num_cudagraphs = len(fn_cache.keys()) + 1

    def warn_msg() -> str:
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
        cudagraphs_log.warning(warn_msg())
        return True

    return False


@dataclasses.dataclass(frozen=True)
class CudagraphCachedInfo:
    """
    Info needed to realign inputs
    """

    placeholders: Sequence[PlaceholderInfo]
    stack_traces: list[str | None]
    cudagraph_fail_reasons: list[str]


@dataclasses.dataclass(frozen=True)
class CudagraphMetadata:
    """
    Metadata for recording a CUDA graph.
    """

    placeholders: Sequence[PlaceholderInfo]
    static_input_idxs: OrderedSet[int]
    mutated_input_idxs: OrderedSet[int]
    stack_traces: list[str | None]
    constants: dict[str, torch.Tensor]


def get_partition_cudagraph_metadata(
    partition_map: GraphPartitionMap,
    metadata: CudagraphMetadata,
) -> CudagraphMetadata:
    """
    Convert the cudagraph metadata at the graph level to the graph partition level,
    given the graph partition info (i.e., mapping from partition input/output index
    to graph input/output index).
    """

    partition_placeholders = []
    partition_static_input_idxs: OrderedSet[int] = OrderedSet()
    partition_mutated_input_idxs: OrderedSet[int] = OrderedSet()
    for partition_input_idx, graph_input_idx in enumerate(
        partition_map.input_index_mapping
    ):
        if graph_input_idx in metadata.static_input_idxs:
            partition_static_input_idxs.add(partition_input_idx)

        if graph_input_idx in metadata.mutated_input_idxs:
            partition_mutated_input_idxs.add(partition_input_idx)

        if graph_input_idx is not None:
            placeholder = metadata.placeholders[graph_input_idx]
        else:
            # create a dummy placeholder info since this partition input is not a graph input
            placeholder = PlaceholderInfo(
                name=f"partition_{partition_map.id}_placeholder_{partition_input_idx}",
                stack_trace=None,
                users=[],
                mutating_use_stack_trace=None,
            )
        partition_placeholders.append(placeholder)

    partition_stack_traces = []
    for graph_output_idx in partition_map.output_index_mapping:
        if graph_output_idx is not None:
            partition_stack_traces.append(metadata.stack_traces[graph_output_idx])
        else:
            partition_stack_traces.append(None)

    partition_constants = {
        name: metadata.constants[name] for name in partition_map.constant_names
    }

    return CudagraphMetadata(
        partition_placeholders,
        partition_static_input_idxs,
        partition_mutated_input_idxs,
        partition_stack_traces,
        partition_constants,
    )


def collect_cuda_data_ptrs(obj: object) -> OrderedSet[int]:
    """Debug helper that collects the data pointers of all CUDA tensors in the object."""
    if not isinstance(obj, torch.Tensor):
        return OrderedSet()

    ptrs: OrderedSet[int] = OrderedSet()
    for base in get_plain_tensors(obj, out=[]):
        if type(base) is not torch.Tensor:
            continue
        if is_fake(base) or base.is_meta or base.device.type != "cuda":
            continue
        try:
            ptrs.add(base.data_ptr())
        except Exception:
            pass
    return ptrs
