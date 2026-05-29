from __future__ import annotations

import bisect
import ctypes
import dataclasses
import gc
import itertools
import operator
import os
import pprint
import sys
from typing import Any, TYPE_CHECKING

import cuda.bindings.driver as cuda_driver  # pyrefly: ignore [missing-import]
import cuda.bindings.nvrtc as nvrtc  # pyrefly: ignore [missing-import]
import cuda.bindings.runtime as cuda_runtime  # pyrefly: ignore [missing-import]

import torch
from torch._dynamo.utils import preserve_rng_state
from torch._inductor import config
from torch._inductor.compile_fx import (
    copy_misaligned_inputs,
    get_expanded_dims,
    get_input_idxs_to_check,
    index_expanded_dims,
    remove_unaligned_input_idxs,
    static_input,
)
from torch._inductor.cudagraph_utils import (
    log_cudagraph_skip_and_bump_counter,
    maybe_warning_due_to_dynamic_shape,
    ModelType,
    OutputType,
    PlaceholderInfo,
)
from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch._guards import CompileId
    from torch._inductor.utils import InputType


log = torch._logging.getArtifactLogger(__name__, "cudagraphs")

StackTraces = list[str | None]


def max_alignment(value: int) -> int:
    """
    Returns the largest power-of-two divisor (alignment) of the given integer.
    Equivalent to the maximum byte alignment the address supports.
    """
    if value == 0:
        return 0
    highest_possible_alignment = (
        value & -value
    )  # Isolates the least significant set bit
    return highest_possible_alignment
    # if highest_possible_alignment > upper_bound:
    #     return upper_bound
    # else:
    #     return highest_possible_alignment


def nbytes_underlying_storage(tensor: torch.Tensor):
    if tensor.numel() == 0:
        return 0
    max_index = sum((s - 1) * st for s, st in zip(tensor.shape, tensor.stride()))
    covered_bytes = (max_index + 1) * tensor.element_size()
    if tensor.is_contiguous():
        assert tensor.nbytes == covered_bytes
    return covered_bytes


@dataclasses.dataclass(frozen=True)
class NonTensorOutputSpec:
    value: int | None


@dataclasses.dataclass(frozen=True)
class EmptyTensorOutputSpec:
    device: torch.device
    dtype: torch.dtype
    stride: tuple[int, ...]
    size: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class InputAliasOutputSpec:
    input_idx: int
    offset_from_input: int
    device: torch.device
    dtype: torch.dtype
    stride: tuple[int, ...]
    size: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class SegmentOutputSpec:
    segment_idx: int
    storage_offset: int
    device: torch.device
    dtype: torch.dtype
    stride: tuple[int, ...]
    size: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class InputSlotSpec:
    input_idx: int
    segment_idx: int
    storage_offset: int
    device: torch.device
    dtype: torch.dtype
    stride: tuple[int, ...]
    size: tuple[int, ...]
    expanded_dims: list[int]


@dataclasses.dataclass(frozen=True)
class CapturedInputInfo:
    input_idx: int
    data_ptr: int
    nbytes: int
    itemsize: int
    device: torch.device
    dtype: torch.dtype
    stride: tuple[int, ...]
    size: tuple[int, ...]
    expanded_dims: list[int]
    is_static: bool


OutputSpec = (
    NonTensorOutputSpec
    | EmptyTensorOutputSpec
    | InputAliasOutputSpec
    | SegmentOutputSpec
)


def _direct_tensor_pointer_key(
    inputs: list[InputType], static_input_idxs: Sequence[int]
) -> tuple[tuple[int, str, int | None, int], ...]:
    static_input_idxs_set = OrderedSet(static_input_idxs)
    pointer_key: list[tuple[int, str, int | None, int]] = []
    for idx, value in enumerate(inputs):
        if not isinstance(value, torch.Tensor):
            continue
        if idx in static_input_idxs_set or not value.is_cuda:
            pointer_key.append(
                (
                    idx,
                    value.device.type,
                    value.device.index,
                    value.data_ptr(),
                )
            )
    return tuple(pointer_key)


def cudagraphify_impl(
    model: ModelType,
    # Some inputs are ints, while others are SymInts, hmmm....
    inputs: list[InputType],
    static_input_idxs: Sequence[int],
    *args: Any,
    **kwargs: Any,
) -> ModelType:
    """Cache parameterized CUDA graph runners for dynamic input pointer sets."""
    fn_cache: dict[
        tuple[Any, tuple[tuple[int, str, int | None, int], ...]], Callable[..., Any]
    ] = {}
    shape_cache: dict[Any, Callable[..., Any]] = {}
    warmed_up_keys: OrderedSet[Any] = OrderedSet()

    # Detect int inputs: we need to index on these
    int_key = [i for i, v in enumerate(inputs) if isinstance(v, int)]
    get_ints: Any = operator.itemgetter(*int_key) if int_key else lambda _: None

    has_warn = False

    del inputs

    def deferred_cudagraphify(inputs: list[InputType]) -> OutputType:
        nonlocal has_warn

        int_key = get_ints(inputs)
        check_input_idxs = get_input_idxs_to_check(inputs, static_input_idxs)
        new_static_input_idxs = remove_unaligned_input_idxs(inputs, static_input_idxs)
        for idx in OrderedSet(static_input_idxs) - OrderedSet(new_static_input_idxs):
            input = inputs[idx]
            if not isinstance(input, torch.Tensor):
                continue
            print(
                "WARNING: cudagraph_digraphs static input is misaligned; "
                "treating it as non-static "
                f"idx={idx} data_ptr={input.data_ptr()} "
                f"alignment={max_alignment(input.data_ptr())}"
            )
        copy_misaligned_inputs(inputs, check_input_idxs)

        pointer_key = _direct_tensor_pointer_key(inputs, new_static_input_idxs)
        cache_key = (int_key, pointer_key)
        fn = fn_cache.get(cache_key)
        if fn is not None:
            return fn(inputs)

        if int_key not in warmed_up_keys:
            warmed_up_keys.add(int_key)
            return model(list(inputs))

        if int_key is None:
            log.info("recording cudagraph tree for graph without symints")
        else:
            log.info("recording cudagraph tree for symint key %s", int_key)

        if not has_warn:
            has_warn = maybe_warning_due_to_dynamic_shape(shape_cache, int_key)

        try:
            fn, out = cudagraphify(
                model, inputs, new_static_input_idxs, *args, **kwargs
            )
        except RuntimeError as e:
            error = str(e)
            unsupported_graph = (
                "Captured CUDA graphs" in error
                or "Failed to identify CUDA graph" in error
                or "CUDA graph pointer did not map consistently" in error
                or "memcpy nodes should have been removed" in error
                or "memset nodes should have been removed" in error
            )
            if not unsupported_graph:
                raise

            log_cudagraph_skip_and_bump_counter(
                "skipping parameterized cudagraph replay because graph "
                f"metadata could not be made dynamic: {error}"
            )

            def fallback_no_cudagraph(inputs: list[InputType]) -> OutputType:
                return model(list(inputs))

            fn_cache[cache_key] = fallback_no_cudagraph
            shape_cache[int_key] = fallback_no_cudagraph
            return fallback_no_cudagraph(inputs)

        fn_cache[cache_key] = fn
        shape_cache[int_key] = fn

        return out

    return deferred_cudagraphify


def find_overlaps(intervals, assert_should_not_happen=False):
    """
    Given a list of intervals represented as (start, length),
    print each pair of intervals that overlap.
    """
    n = len(intervals)
    for i in range(n):
        start1, len1 = intervals[i]
        end1 = start1 + len1
        for j in range(i + 1, n):
            start2, len2 = intervals[j]
            end2 = start2 + len2

            # Check if [start1, end1) overlaps [start2, end2)
            if start1 < end2 and start2 < end1:
                if assert_should_not_happen:
                    raise AssertionError(
                        "Overlapping dynamic tensor intervals: "
                        f"{i}=({start1}, {len1}), {j}=({start2}, {len2})"
                    )
                log.debug(
                    "Overlapping dynamic tensor intervals: %s=(%s, %s), %s=(%s, %s)",
                    i,
                    start1,
                    len1,
                    j,
                    start2,
                    len2,
                )


def _has_active_memory(segment: dict[str, Any]) -> bool:
    return int(segment.get("active_size", 0)) != 0 or any(
        block.get("state") != "inactive" for block in segment.get("blocks", ())
    )


def _begin_capture_memory_history_recording() -> bool:
    history_was_enabled = torch._C._cuda_isHistoryEnabled()
    history_max_entries = int(
        os.environ.get(
            "TORCHINDUCTOR_CUDAGRAPH_CAPTURE_MEMORY_HISTORY_MAX_ENTRIES",
            "1000000",
        )
    )
    torch.cuda.memory._record_memory_history(
        "all",
        context=None,
        max_entries=history_max_entries,
        clear_history=True,
    )
    return history_was_enabled


def _end_capture_memory_history_recording(history_was_enabled: bool) -> None:
    if not history_was_enabled:
        # Exact previous memory history settings are not exposed. Leave
        # caller-enabled history alone, and only disable when history was
        # initially off. Disabling clears the trace, so callers must take any
        # trace-bearing snapshot before invoking this.
        torch.cuda.memory._record_memory_history(None)


def _raw_memory_snapshot(pool_id: tuple[int, int], include_traces: bool) -> Any:
    return torch._C._cuda_memorySnapshot((pool_id[0], pool_id[1], include_traces))


def _block_ranges_from_memory_snapshot(
    memory_snapshot: list[dict[str, Any]],
) -> tuple[list[int], list[int], list[torch.device], list[int]]:
    address_starts: list[int] = []
    sizes: list[int] = []
    devices: list[torch.device] = []
    segment_idxs: list[int] = []
    for segment_idx, segment_snapshot in enumerate(memory_snapshot):
        segment_start = int(segment_snapshot["address"])
        segment_size = int(segment_snapshot["total_size"])
        segment_device = torch.device("cuda", int(segment_snapshot["device"]))
        block_address = segment_start
        blocks = segment_snapshot.get("blocks", ())
        if not blocks:
            if segment_size:
                address_starts.append(segment_start)
                sizes.append(segment_size)
                devices.append(segment_device)
                segment_idxs.append(segment_idx)
            continue
        for block in blocks:
            block_size = int(block["size"])
            if block_size:
                address_starts.append(int(block.get("address", block_address)))
                sizes.append(block_size)
                devices.append(segment_device)
                segment_idxs.append(segment_idx)
            block_address += block_size
        assert block_address == segment_start + segment_size
    return address_starts, sizes, devices, segment_idxs


def _merge_overlapping_allocation_ranges(
    address_starts: list[int],
    sizes: list[int],
    devices: list[torch.device],
    segment_idxs: list[int],
) -> tuple[list[int], list[int], list[torch.device], list[int]]:
    order = sorted(range(len(address_starts)), key=lambda idx: address_starts[idx])
    merged_starts: list[int] = []
    merged_sizes: list[int] = []
    merged_devices: list[torch.device] = []
    merged_segment_idxs: list[int] = []
    for idx in order:
        start = address_starts[idx]
        size = sizes[idx]
        if size == 0:
            continue
        device = devices[idx]
        segment_idx = segment_idxs[idx]
        end = start + size
        if not merged_starts or start >= merged_starts[-1] + merged_sizes[-1]:
            merged_starts.append(start)
            merged_sizes.append(size)
            merged_devices.append(device)
            merged_segment_idxs.append(segment_idx)
            continue
        if device != merged_devices[-1]:
            raise RuntimeError("CUDA graph dynamic allocation ranges overlap")
        merged_end = max(merged_starts[-1] + merged_sizes[-1], end)
        merged_sizes[-1] = merged_end - merged_starts[-1]
        if merged_segment_idxs[-1] < 0:
            merged_segment_idxs[-1] = segment_idx
    return merged_starts, merged_sizes, merged_devices, merged_segment_idxs


def cudagraphify(
    model: ModelType,
    inputs: list[InputType],
    static_input_idxs: Sequence[int] = (),
    *,
    device_index: int,
    is_backward: bool,
    is_inference: bool,
    stack_traces: StackTraces | None = None,
    constants: tuple[
        torch.Tensor, ...
    ] = (),  # TODO: How is constants different from inputs contained in static_input_idxs?
    placeholders: tuple[PlaceholderInfo, ...] = (),
    mutated_input_idxs: tuple[int, ...] = (),
    compile_id: CompileId | None = None,
) -> tuple[ModelType, OutputType]:
    # TODO: we should fail if this fxgraph ever inspects data_ptr()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    static_input_idxs_set = OrderedSet(static_input_idxs)
    graphs: list[torch.cuda.CUDAGraph] = []
    dynamic_tensors_list: list[list[tuple[int, int]]] = []
    capture_keepalive: list[Any] = []
    capture_pool_keepalive: list[Any] = []
    input_pool_keepalive: list[Any] = []
    capture_inputs: list[InputType] = []
    static_outputs: tuple[torch.Tensor | int | None, ...] = ()
    segment_address_starts: list[int] = []
    segment_sizes: list[int] = []
    segment_devices: list[torch.device] = []
    segment_input_idxs: list[int | None] = []
    output_specs: list[OutputSpec] = []
    input_slot_specs: list[InputSlotSpec] = []

    copy_outputs = os.environ.get("TORCHINDUCTOR_CUDAGRAPHS_COPY_OUTPUTS", "never")
    if copy_outputs not in ("backward_only", "forward_only", "always", "never"):
        raise RuntimeError(
            "TORCHINDUCTOR_CUDAGRAPHS_COPY_OUTPUTS must be one of "
            "'backward_only', 'forward_only', 'always', or 'never', "
            f"but got {copy_outputs!r}"
        )
    copy_replay_outputs = (
        copy_outputs == "always"
        or (copy_outputs == "backward_only" and is_backward)
        or (copy_outputs == "forward_only" and not is_backward)
    )
    allocate_blocks_env = os.environ.get("TORCHINDUCTOR_CUDAGRAPHS_ALLOCATE_BLOCKS")
    if allocate_blocks_env is None:
        allocate_blocks = False
    elif allocate_blocks_env in ("0", "1"):
        allocate_blocks = allocate_blocks_env == "1"
    else:
        raise RuntimeError(
            "TORCHINDUCTOR_CUDAGRAPHS_ALLOCATE_BLOCKS must be '0' or '1', "
            f"got {allocate_blocks_env!r}"
        )

    reserved_mem_before_captures = torch.cuda.memory_reserved(device_index)

    for i in range(2):
        # We want to make sure that a workspace is allocated into the
        # cuda graph's private pool. We don't want to simply use the
        # workspace created during warmup, which may die later.
        torch._C._cuda_clearCublasWorkspaces()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        mem_allocator = graph.get_mem_allocator()
        capture_pool = torch.cuda.MemPool(mem_allocator)
        capture_pool_keepalive.append(capture_pool)
        if config.triton.cudagraphs_separate_input_pool:
            input_pool = torch.cuda.MemPool(mem_allocator)
            input_pool_keepalive.append(input_pool)
        else:
            input_pool = capture_pool

        history_was_enabled: bool | None = None
        if allocate_blocks:
            history_was_enabled = _begin_capture_memory_history_recording()

        with torch.cuda.stream(stream):
            with torch.cuda.use_mem_pool(input_pool):
                old_value = torch._C._get_deterministic_fill_uninitialized_memory()
                torch._C._set_deterministic_fill_uninitialized_memory(False)
                try:
                    capture_inputs = [
                        (
                            x
                            if (
                                not isinstance(x, torch.Tensor)
                                or not x.is_cuda
                                or idx in static_input_idxs_set
                            )
                            else static_input(x)
                        )
                        for idx, x in enumerate(inputs)
                    ]
                finally:
                    torch._C._set_deterministic_fill_uninitialized_memory(old_value)

        captured_input_infos: list[CapturedInputInfo] = []
        for idx, capture_input in enumerate(capture_inputs):
            if not isinstance(capture_input, torch.Tensor):
                continue
            if not capture_input.is_cuda:
                continue
            captured_input_infos.append(
                CapturedInputInfo(
                    input_idx=idx,
                    data_ptr=capture_input.data_ptr(),
                    nbytes=nbytes_underlying_storage(capture_input),
                    itemsize=capture_input.itemsize,
                    device=capture_input.device,
                    dtype=capture_input.dtype,
                    stride=tuple(capture_input.stride()),
                    size=tuple(capture_input.size()),
                    expanded_dims=get_expanded_dims(capture_input),
                    is_static=idx in static_input_idxs_set,
                )
            )
            if idx not in static_input_idxs_set:
                alignment = max_alignment(capture_input.data_ptr())
                assert alignment == 0 or alignment >= 256, (
                    "Capture input alignment is less than expected"
                )

        with torch.cuda.stream(stream):
            copy_dsts: list[torch.Tensor] = []
            copy_srcs: list[torch.Tensor] = []
            for idx, (capture_input, original_input) in enumerate(
                zip(capture_inputs, inputs)
            ):
                if (
                    idx in static_input_idxs_set
                    or not isinstance(capture_input, torch.Tensor)
                    or not capture_input.is_cuda
                ):
                    continue
                assert isinstance(original_input, torch.Tensor)
                copy_dsts.append(
                    index_expanded_dims(capture_input, get_expanded_dims(capture_input))
                )
                copy_srcs.append(
                    index_expanded_dims(
                        original_input, get_expanded_dims(capture_input)
                    )
                )
            if copy_dsts:
                torch._foreach_copy_(copy_dsts, copy_srcs)
        copy_dsts = []
        copy_srcs = []
        capture_input = None
        original_input = None

        with (
            preserve_rng_state(),
            torch.cuda.graph(
                graph,
                pool=torch.cuda._POOL_HANDLE(capture_pool.id),
                stream=stream,
                capture_error_mode="thread_local",
            ),
        ):
            model_outputs = model(capture_inputs)

        if not config.triton.cudagraphs_preserve_memops:
            replace_memops_with_kernels(
                cuda_runtime.cudaGraph_t(init_value=graph.raw_cuda_graph())
            )

        if not isinstance(model_outputs, (list, tuple)):
            static_outputs = (model_outputs,)
        else:
            static_outputs = tuple(model_outputs)
        capture_keepalive.append(static_outputs)

        graph_pool_id = graph.pool()
        graph_memory_snapshot_result: dict[str, Any] | None = None
        if allocate_blocks:
            graph_memory_snapshot_result = _raw_memory_snapshot(
                graph_pool_id, include_traces=True
            )
            graph_memory_snapshot: list[dict[str, Any]] = graph_memory_snapshot_result[
                "segments"
            ]
        else:
            graph_memory_snapshot = torch.cuda.memory_snapshot(
                graph_pool_id, include_traces=True
            )
        if input_pool.id == graph.pool():
            memory_snapshot = graph_memory_snapshot
            input_memory_snapshot = graph_memory_snapshot
        else:
            input_memory_snapshot: list[dict[str, Any]] = torch.cuda.memory_snapshot(
                input_pool.id, include_traces=True
            )
            memory_snapshot = graph_memory_snapshot

        if history_was_enabled is not None:
            _end_capture_memory_history_recording(history_was_enabled)

        segment_address_starts = [
            int(segment_snapshot["address"]) for segment_snapshot in memory_snapshot
        ]
        segment_sizes = [
            int(segment_snapshot["total_size"]) for segment_snapshot in memory_snapshot
        ]
        segment_devices = [
            torch.device("cuda", int(segment_snapshot["device"]))
            for segment_snapshot in memory_snapshot
        ]
        segment_idxs_sorted_by_address = sorted(
            range(len(segment_address_starts)),
            key=lambda idx: segment_address_starts[idx],
        )
        segment_address_starts_sorted = [
            segment_address_starts[idx] for idx in segment_idxs_sorted_by_address
        ]

        def lookup_segment_idx(ptr: int) -> int:
            sorted_idx = bisect.bisect(segment_address_starts_sorted, ptr) - 1
            if sorted_idx == -1:
                return -1
            segment_idx = segment_idxs_sorted_by_address[sorted_idx]
            if ptr < segment_address_starts[segment_idx] + segment_sizes[segment_idx]:
                return segment_idx
            return -1

        if allocate_blocks:
            (
                allocation_address_starts,
                allocation_sizes,
                allocation_devices,
                allocation_segment_idxs,
            ) = _block_ranges_from_memory_snapshot(memory_snapshot)
            assert graph_memory_snapshot_result is not None
            synthetic_segment_idx = len(segment_address_starts)
            graph_pool_id_tuple = tuple(graph_pool_id)
            for device_idx, trace in enumerate(
                graph_memory_snapshot_result.get("device_traces", ())
            ):
                device = torch.device("cuda", device_idx)
                for trace_entry in trace:
                    if trace_entry.get("action") != "alloc":
                        continue
                    if tuple(trace_entry.get("pool_id", (0, 0))) != graph_pool_id_tuple:
                        continue
                    ptr = int(trace_entry["addr"])
                    size = int(trace_entry["size"])
                    if size == 0:
                        continue
                    segment_idx = lookup_segment_idx(ptr)
                    if segment_idx == -1:
                        segment_idx = synthetic_segment_idx
                        synthetic_segment_idx += 1
                    allocation_address_starts.append(ptr)
                    allocation_sizes.append(size)
                    allocation_devices.append(device)
                    allocation_segment_idxs.append(segment_idx)
            (
                allocation_address_starts,
                allocation_sizes,
                allocation_devices,
                allocation_segment_idxs,
            ) = _merge_overlapping_allocation_ranges(
                allocation_address_starts,
                allocation_sizes,
                allocation_devices,
                allocation_segment_idxs,
            )
        else:
            allocation_address_starts = list(segment_address_starts)
            allocation_sizes = list(segment_sizes)
            allocation_devices = list(segment_devices)
            allocation_segment_idxs = list(range(len(segment_sizes)))

        allocation_idxs_by_segment: dict[int, list[int]] = {}
        for allocation_idx, segment_idx in enumerate(allocation_segment_idxs):
            allocation_idxs_by_segment.setdefault(segment_idx, []).append(
                allocation_idx
            )
        allocation_idxs_sorted_by_address = sorted(
            range(len(allocation_address_starts)),
            key=lambda idx: allocation_address_starts[idx],
        )
        allocation_address_starts_sorted = [
            allocation_address_starts[idx] for idx in allocation_idxs_sorted_by_address
        ]

        def lookup_allocation_idx(ptr: int) -> int:
            sorted_idx = bisect.bisect(allocation_address_starts_sorted, ptr) - 1
            if sorted_idx == -1:
                return -1
            allocation_idx = allocation_idxs_sorted_by_address[sorted_idx]
            if (
                ptr
                < allocation_address_starts[allocation_idx]
                + allocation_sizes[allocation_idx]
            ):
                return allocation_idx
            return -1

        input_segment_address_starts = [
            int(segment_snapshot["address"])
            for segment_snapshot in input_memory_snapshot
        ]
        input_segment_sizes = [
            int(segment_snapshot["total_size"])
            for segment_snapshot in input_memory_snapshot
        ]
        input_segment_idxs_sorted_by_address = sorted(
            range(len(input_segment_address_starts)),
            key=lambda idx: input_segment_address_starts[idx],
        )
        input_segment_address_starts_sorted = [
            input_segment_address_starts[idx]
            for idx in input_segment_idxs_sorted_by_address
        ]

        def lookup_input_segment_idx(ptr: int) -> int:
            sorted_idx = bisect.bisect(input_segment_address_starts_sorted, ptr) - 1
            if sorted_idx == -1:
                return -1
            segment_idx = input_segment_idxs_sorted_by_address[sorted_idx]
            if (
                ptr
                < input_segment_address_starts[segment_idx]
                + input_segment_sizes[segment_idx]
            ):
                return segment_idx
            return -1

        captured_input_ranges: list[tuple[int, int, int]] = []
        direct_input_infos: list[CapturedInputInfo] = []
        input_segment_idxs: dict[int, list[int]] = {}
        input_slot_specs = []
        for input_info in captured_input_infos:
            if input_info.nbytes == 0:
                continue
            if input_info.is_static:
                captured_input_ranges.append(
                    (
                        input_info.input_idx,
                        input_info.data_ptr,
                        input_info.nbytes,
                    )
                )
                continue
            if config.triton.cudagraphs_separate_input_pool:
                input_segment_idx = lookup_input_segment_idx(input_info.data_ptr)
                if input_segment_idx == -1:
                    raise RuntimeError(
                        "Non-static CUDA graph input was not allocated in the "
                        f"input capture pool {input_info.device}"
                    )
                input_storage_offset = (
                    input_info.data_ptr
                    - input_segment_address_starts[input_segment_idx]
                )
                assert (
                    input_storage_offset + input_info.nbytes
                    <= input_segment_sizes[input_segment_idx]
                )
                direct_input_infos.append(input_info)
                captured_input_ranges.append(
                    (
                        input_info.input_idx,
                        input_info.data_ptr,
                        input_info.nbytes,
                    )
                )
                continue
            allocation_idx = lookup_allocation_idx(input_info.data_ptr)
            if allocation_idx == -1:
                raise RuntimeError(
                    "Non-static CUDA graph input was not allocated in the "
                    f"capture pool {input_info.device}"
                )
            storage_offset = (
                input_info.data_ptr - allocation_address_starts[allocation_idx]
            )
            assert storage_offset % input_info.itemsize == 0
            if storage_offset + input_info.nbytes > allocation_sizes[allocation_idx]:
                raise RuntimeError(
                    "Non-static CUDA graph input spans multiple replay "
                    "allocations: "
                    f"input_idx={input_info.input_idx} "
                    f"data_ptr={input_info.data_ptr} "
                    f"nbytes={input_info.nbytes} "
                    f"allocation_idx={allocation_idx} "
                    f"allocation_start={allocation_address_starts[allocation_idx]} "
                    f"allocation_size={allocation_sizes[allocation_idx]} "
                    f"storage_offset={storage_offset}"
                )
            segment_idx = allocation_segment_idxs[allocation_idx]
            input_segment_idxs.setdefault(segment_idx, []).append(input_info.input_idx)
            input_slot_specs.append(
                InputSlotSpec(
                    input_idx=input_info.input_idx,
                    segment_idx=allocation_idx,
                    storage_offset=storage_offset // input_info.itemsize,
                    device=input_info.device,
                    dtype=input_info.dtype,
                    stride=input_info.stride,
                    size=input_info.size,
                    expanded_dims=input_info.expanded_dims,
                )
            )

        output_specs = []
        output_segment_idxs: dict[int, list[int]] = {}

        # I need to map each segment index to all output tensors, I think.
        for static_output_idx, static_output in enumerate(static_outputs):
            if not isinstance(static_output, torch.Tensor):
                assert static_output is None or isinstance(static_output, int)
                output_specs.append(NonTensorOutputSpec(static_output))
                continue
            if static_output.data_ptr() == 0:
                assert static_output.nbytes == 0
                output_specs.append(
                    EmptyTensorOutputSpec(
                        device=static_output.device,
                        dtype=static_output.dtype,
                        stride=tuple(static_output.stride()),
                        size=tuple(static_output.size()),
                    )
                )
                continue
            assert static_output.is_cuda, (
                "I suppose non cuda outputs are allowed, but I would like to "
                "catch them explicitly for now"
            )
            input_alias_spec: InputAliasOutputSpec | None = None
            for input_idx, input_data_ptr, input_nbytes in captured_input_ranges:
                if (
                    input_data_ptr <= static_output.data_ptr()
                    and static_output.data_ptr()
                    + nbytes_underlying_storage(static_output)
                    <= input_data_ptr + input_nbytes
                ):
                    assert input_alias_spec is None, (
                        "CUDA graph inputs should never share a buffer "
                        "during stream capture!!!"
                    )
                    offset_from_input = static_output.data_ptr() - input_data_ptr
                    assert offset_from_input % static_output.itemsize == 0
                    offset_from_input = offset_from_input // static_output.itemsize
                    input_alias_spec = InputAliasOutputSpec(
                        input_idx=input_idx,
                        offset_from_input=offset_from_input,
                        device=static_output.device,
                        dtype=static_output.dtype,
                        stride=tuple(static_output.stride()),
                        size=tuple(static_output.size()),
                    )
            if input_alias_spec is not None:
                output_specs.append(input_alias_spec)
                continue

            allocation_idx = lookup_allocation_idx(static_output.data_ptr())
            if allocation_idx == -1:
                # In this case, the output must be part of a
                # non-dynamic input tensor (which we should
                # verify!). In that situation, the output tensor
                # should always have the same output address across
                # runs.
                raise AssertionError(
                    static_output_idx,
                    static_output.data_ptr(),
                    nbytes_underlying_storage(static_output),
                )
            storage_offset = (
                static_output.data_ptr() - allocation_address_starts[allocation_idx]
            )
            assert storage_offset < allocation_sizes[allocation_idx]
            assert (
                nbytes_underlying_storage(static_output)
                <= allocation_sizes[allocation_idx]
            )
            assert (
                storage_offset + nbytes_underlying_storage(static_output)
                <= allocation_sizes[allocation_idx]
            )
            assert storage_offset % static_output.itemsize == 0
            segment_idx = allocation_segment_idxs[allocation_idx]
            output_segment_idxs.setdefault(segment_idx, []).append(static_output_idx)
            output_specs.append(
                SegmentOutputSpec(
                    segment_idx=allocation_idx,
                    storage_offset=storage_offset // static_output.itemsize,
                    device=static_output.device,
                    dtype=static_output.dtype,
                    stride=tuple(static_output.stride()),
                    size=tuple(static_output.size()),
                )
            )

        input_segments = sorted(
            input_segment_idxs, key=lambda idx: tuple(input_segment_idxs[idx])
        )
        output_segments = sorted(
            (idx for idx in output_segment_idxs if idx not in input_segment_idxs),
            key=lambda idx: tuple(output_segment_idxs[idx]),
        )
        all_replay_segment_idxs = sorted(allocation_idxs_by_segment)

        def segment_order_key(idx: int) -> tuple[int, tuple[Any, ...]]:
            if idx < len(segment_sizes):
                size = segment_sizes[idx]
                block_key = tuple(
                    (
                        block.get("size"),
                        block.get("requested_size", 0),
                        block.get("state"),
                    )
                    for block in memory_snapshot[idx].get("blocks", ())
                )
                return size, block_key
            allocation_total = sum(
                allocation_sizes[allocation_idx]
                for allocation_idx in allocation_idxs_by_segment[idx]
            )
            return allocation_total, ()

        remaining_segments = sorted(
            (
                idx
                for idx in all_replay_segment_idxs
                if idx not in input_segment_idxs and idx not in output_segment_idxs
            ),
            key=segment_order_key,
        )
        segment_order = input_segments + output_segments + remaining_segments
        allocation_order = [
            allocation_idx
            for segment_idx in segment_order
            for allocation_idx in allocation_idxs_by_segment[segment_idx]
        ]
        num_direct_input_allocations = len(direct_input_infos)
        old_to_new_allocation_idx = {
            old_idx: num_direct_input_allocations + new_idx
            for new_idx, old_idx in enumerate(allocation_order)
        }
        segment_address_starts = [
            input_info.data_ptr for input_info in direct_input_infos
        ] + [allocation_address_starts[idx] for idx in allocation_order]
        segment_sizes = [input_info.nbytes for input_info in direct_input_infos] + [
            allocation_sizes[idx] for idx in allocation_order
        ]
        segment_devices = [input_info.device for input_info in direct_input_infos] + [
            allocation_devices[idx] for idx in allocation_order
        ]
        segment_input_idxs = [
            input_info.input_idx for input_info in direct_input_infos
        ] + [None for _ in allocation_order]
        input_slot_specs = [
            dataclasses.replace(
                input_slot_spec,
                segment_idx=old_to_new_allocation_idx[input_slot_spec.segment_idx],
            )
            for input_slot_spec in input_slot_specs
        ]
        output_specs = [
            dataclasses.replace(
                output_spec,
                segment_idx=old_to_new_allocation_idx[output_spec.segment_idx],
            )
            if isinstance(output_spec, SegmentOutputSpec)
            else output_spec
            for output_spec in output_specs
        ]

        find_overlaps(list(zip(segment_address_starts, segment_sizes)), True)
        dynamic_tensor_allocations = list(zip(segment_address_starts, segment_sizes))

        dynamic_tensors_list.append(dynamic_tensor_allocations)
        graphs.append(graph)

    for graph_idx, captured_graph in enumerate(graphs):
        pool_id = captured_graph.pool()
        # print(
        #     "GALVEZ: torch.cuda.memory_snapshot("
        #     f"pool_id={pool_id}, include_traces=True) "
        #     f"for cudagraph_digraphs graph {graph_idx}"
        # )
        # pprint.pprint(
        #     torch.cuda.memory_snapshot(pool_id, include_traces=True),
        #     sort_dicts=False,
        # )
    graph = graphs[1]
    find_overlaps(dynamic_tensors_list[0], True)
    find_overlaps(dynamic_tensors_list[1], True)
    find_overlaps(dynamic_tensors_list[0] + dynamic_tensors_list[1], True)
    graph_runner = make_dynamic_graph_runner(
        graph,
        dynamic_tensors_list[1],
        graphs[0],
        dynamic_tensors_list[0],
    )

    num_inputs = len(inputs)
    output_alias_input_idxs = OrderedSet(
        output_spec.input_idx
        for output_spec in output_specs
        if isinstance(output_spec, InputAliasOutputSpec)
    )
    copyback_input_idxs = OrderedSet(mutated_input_idxs) | output_alias_input_idxs

    capture_keepalive.clear()
    capture_inputs = []
    static_outputs = ()
    model_outputs = None
    static_output = None
    capture_input = None
    copy_dsts = []
    copy_srcs = []
    original_input = None
    capture_pool = None
    input_pool = None
    gc.collect()
    for graph_idx, captured_graph in enumerate(graphs):
        pool_id = captured_graph.pool()
        pool_snapshot = torch.cuda.memory_snapshot(pool_id, include_traces=True)
        active_segments = [
            segment for segment in pool_snapshot if _has_active_memory(segment)
        ]
        print(
            "GALVEZ: cudagraph_digraphs pool memory snapshot "
            f"graph_idx={graph_idx} pool_id={pool_id} "
            f"num_segments={len(pool_snapshot)} "
            f"num_active_segments={len(active_segments)}"
        )
        # print.pprint(pool_snapshot, sort_dicts=False)
        if active_segments:
            print(
                "GALVEZ: WARNING cudagraph_digraphs active pool segments before "
                f"release_pool_memory graph_idx={graph_idx} pool_id={pool_id}"
            )
            pprint.pprint(active_segments, sort_dicts=False)
        captured_graph.release_pool_memory()
    mem_allocator = None
    capture_pool_keepalive.clear()
    input_pool_keepalive.clear()
    gc.collect()
    # torch.cuda.synchronize() # TODO: is this necessary? I don't think it is...
    # gc.collect()
    # torch.cuda.empty_cache()

    # I need to call this only after the private memory pools' segments are no longer active.
    torch.cuda.reset_peak_memory_stats(device_index)
    reserved_mem_after_capture_cleanup = torch.cuda.memory_reserved(device_index)
    reserved_mem_delta_after_capture_cleanup = (
        reserved_mem_after_capture_cleanup - reserved_mem_before_captures
    )
    print(
        "GALVEZ: cudagraph_digraphs_reserved_mem_after_capture_cleanup "
        f"before_gb={reserved_mem_before_captures / 10**9} "
        f"after_gb={reserved_mem_after_capture_cleanup / 10**9} "
        f"delta_gb={reserved_mem_delta_after_capture_cleanup / 10**9}"
    )

    def run(new_inputs: list[InputType]) -> OutputType:
        """Replay the graph with new input and output storage pointers."""
        assert num_inputs == len(new_inputs)

        dynamic_tensors: list[torch.Tensor] = []

        for size, device, input_idx in zip(
            segment_sizes, segment_devices, segment_input_idxs
        ):
            if input_idx is None:
                dynamic_tensors.append(
                    torch.empty(size, dtype=torch.int8, device=device)
                )
                continue
            new_input = new_inputs[input_idx]
            if not isinstance(new_input, torch.Tensor) or not new_input.is_cuda:
                raise RuntimeError(
                    "CUDA graph input pointer update expected replay input "
                    f"{input_idx} to be a CUDA tensor, got {new_input!r}"
                )
            if nbytes_underlying_storage(new_input) < size:
                raise RuntimeError(
                    "CUDA graph replay input storage is smaller than the "
                    f"captured input storage for input {input_idx}: "
                    f"{nbytes_underlying_storage(new_input)} < {size}"
                )
            dynamic_tensors.append(new_input)

        copy_dsts: list[torch.Tensor] = []
        copy_srcs: list[torch.Tensor] = []
        copyback_dsts: list[torch.Tensor] = []
        copyback_srcs: list[torch.Tensor] = []

        torch.cuda.nvtx.range_push("push inputs")
        for input_slot_spec in input_slot_specs:
            new_input = new_inputs[input_slot_spec.input_idx]
            assert isinstance(new_input, torch.Tensor)
            if not new_input.is_cuda:
                raise RuntimeError(
                    "Non-static CUDA graph input was recorded as a CUDA tensor, "
                    f"but replay input {input_slot_spec.input_idx} is on "
                    f"{new_input.device}"
                )
            storage_tensor = dynamic_tensors[input_slot_spec.segment_idx]
            input_slot = torch.empty(
                (), device=input_slot_spec.device, dtype=input_slot_spec.dtype
            )
            input_slot.set_(
                storage_tensor.untyped_storage(),
                storage_offset=input_slot_spec.storage_offset,
                stride=input_slot_spec.stride,
                size=input_slot_spec.size,
            )
            input_copy_dst = index_expanded_dims(
                input_slot, input_slot_spec.expanded_dims
            )
            input_copy_src = index_expanded_dims(
                new_input, input_slot_spec.expanded_dims
            )
            copy_dsts.append(input_copy_dst)
            copy_srcs.append(input_copy_src)
            if input_slot_spec.input_idx in copyback_input_idxs:
                copyback_dsts.append(input_copy_src)
                copyback_srcs.append(input_copy_dst)
        if copy_dsts:
            torch._foreach_copy_(copy_dsts, copy_srcs)
        torch.cuda.nvtx.range_pop()

        graph_runner.replay(dynamic_tensors)

        if copyback_dsts:
            torch._foreach_copy_(copyback_dsts, copyback_srcs)

        outputs: OutputType = []

        for output_spec in output_specs:
            if isinstance(output_spec, NonTensorOutputSpec):
                outputs.append(output_spec.value)
                continue
            if isinstance(output_spec, EmptyTensorOutputSpec):
                outputs.append(
                    torch.empty_strided(
                        output_spec.size,
                        output_spec.stride,
                        device=output_spec.device,
                        dtype=output_spec.dtype,
                    )
                )
                continue
            if isinstance(output_spec, InputAliasOutputSpec):
                input_tensor = new_inputs[output_spec.input_idx]
                assert isinstance(input_tensor, torch.Tensor)
                true_output_tensor = torch.empty(
                    (), device=output_spec.device, dtype=output_spec.dtype
                )
                input_offset_from_storage = (
                    input_tensor.data_ptr() - input_tensor.untyped_storage().data_ptr()
                )
                assert input_offset_from_storage % input_tensor.itemsize == 0
                input_offset_from_storage = (
                    input_offset_from_storage // input_tensor.itemsize
                )
                true_output_tensor.set_(
                    input_tensor.untyped_storage(),
                    storage_offset=(
                        output_spec.offset_from_input + input_offset_from_storage
                    ),
                    stride=output_spec.stride,
                    size=output_spec.size,
                )

                outputs.append(true_output_tensor)
                continue
            assert isinstance(output_spec, SegmentOutputSpec)
            storage_tensor = dynamic_tensors[output_spec.segment_idx]
            true_output_tensor = torch.empty(
                (), device=output_spec.device, dtype=output_spec.dtype
            )

            true_output_tensor.set_(
                storage_tensor.untyped_storage(),
                storage_offset=output_spec.storage_offset,
                stride=output_spec.stride,
                size=output_spec.size,
            )
            if copy_replay_outputs:
                compact_output_tensor = torch.empty_strided(
                    output_spec.size,
                    output_spec.stride,
                    device=output_spec.device,
                    dtype=output_spec.dtype,
                )
                compact_output_tensor.copy_(true_output_tensor)
                true_output_tensor = compact_output_tensor
            outputs.append(true_output_tensor)

        return outputs

    return run, run(list(inputs))


def cuda_python_error_check(function_call_output):
    """Makes calls to cuda-python's cuda runtime functions more
    pythonic by throwing an exception if they return a status
    which is not cudaSuccess
    """
    import cuda.bindings  # type: ignore[import]

    error, *others = function_call_output
    if (
        isinstance(error, cuda.bindings.runtime.cudaError_t)
        and error != cuda.bindings.runtime.cudaError_t.cudaSuccess
    ):
        raise ValueError(f"CUDA failure! {error}")
    elif (
        isinstance(error, cuda.bindings.driver.CUresult)
        and error != cuda.bindings.driver.CUresult.CUDA_SUCCESS
    ):
        raise ValueError(f"CUDA failure! {error}")
    elif (
        isinstance(error, cuda.bindings.nvrtc.nvrtcResult)
        and error != cuda.bindings.nvrtc.nvrtcResult.NVRTC_SUCCESS
    ):
        raise ValueError(f"NVRTC failure! {error}")
    elif len(others) == 1:
        return others[0]
    else:
        return tuple(others)


@dataclasses.dataclass(frozen=True)
class DynamicAllocation:
    ptr: int
    size: int
    alloc_idx: int


@dataclasses.dataclass(frozen=True)
class KernelParamPointerUpdate:
    param_index: int
    param_byte_offset: int
    alloc_idx: int
    alloc_offset: int


@dataclasses.dataclass(frozen=True)
class GraphPointerUpdate:
    alloc_idx: int
    alloc_offset: int


@dataclasses.dataclass(frozen=True)
class MemcpyNodeUpdate:
    node: int
    src_ptr: int
    src_pitch: int
    src_xsize: int
    src_ysize: int
    src_pos: tuple[int, int, int]
    dst_ptr: int
    dst_pitch: int
    dst_xsize: int
    dst_ysize: int
    dst_pos: tuple[int, int, int]
    extent: tuple[int, int, int]
    src_update: GraphPointerUpdate | None
    dst_update: GraphPointerUpdate | None

    def to_params(self, actual_data_ptrs: list[int]) -> Any:
        src_ptr = self.src_ptr
        if self.src_update is not None:
            src_ptr = (
                actual_data_ptrs[self.src_update.alloc_idx]
                + self.src_update.alloc_offset
            )
        dst_ptr = self.dst_ptr
        if self.dst_update is not None:
            dst_ptr = (
                actual_data_ptrs[self.dst_update.alloc_idx]
                + self.dst_update.alloc_offset
            )

        params = cuda_driver.CUDA_MEMCPY3D()
        params.srcXInBytes, params.srcY, params.srcZ = self.src_pos
        params.srcMemoryType = cuda_driver.CUmemorytype.CU_MEMORYTYPE_DEVICE
        params.srcDevice = cuda_driver.CUdeviceptr(init_value=src_ptr)
        params.srcPitch = self.src_pitch
        params.srcHeight = self.src_ysize
        params.dstXInBytes, params.dstY, params.dstZ = self.dst_pos
        params.dstMemoryType = cuda_driver.CUmemorytype.CU_MEMORYTYPE_DEVICE
        params.dstDevice = cuda_driver.CUdeviceptr(init_value=dst_ptr)
        params.dstPitch = self.dst_pitch
        params.dstHeight = self.dst_ysize
        params.WidthInBytes, params.Height, params.Depth = self.extent
        return params


@dataclasses.dataclass(frozen=True)
class MemsetNodeUpdate:
    node: int
    dst: int
    pitch: int
    value: int
    element_size: int
    width: int
    height: int
    dst_update: GraphPointerUpdate

    def to_params(self, actual_data_ptrs: list[int]) -> Any:
        params = cuda_driver.CUDA_MEMSET_NODE_PARAMS()
        params.dst = cuda_driver.CUdeviceptr(
            init_value=(
                actual_data_ptrs[self.dst_update.alloc_idx]
                + self.dst_update.alloc_offset
            )
        )
        params.pitch = self.pitch
        params.value = self.value
        params.elementSize = self.element_size
        params.width = self.width
        params.height = self.height
        return params


class PythonDynamicCUDAGraph:
    def __init__(
        self,
        graph: torch.cuda.CUDAGraph,
        device_nodes: list[int],
        param_offsets: list[int],
        alloc_indices: list[int],
        alloc_offsets: list[int],
        memcpy_node_updates: list[MemcpyNodeUpdate],
        memset_node_updates: list[MemsetNodeUpdate],
    ) -> None:
        self.graph = graph
        device_dynamic_tensor_indices = sorted(OrderedSet(alloc_indices))
        device_dynamic_tensor_idx_map = {
            old_idx: new_idx
            for new_idx, old_idx in enumerate(device_dynamic_tensor_indices)
        }
        self.device_nodes = torch.tensor(device_nodes, dtype=torch.int64, device="cpu")
        self.param_offsets = torch.tensor(
            param_offsets, dtype=torch.int64, device="cpu"
        )
        self.alloc_indices = torch.tensor(
            [device_dynamic_tensor_idx_map[idx] for idx in alloc_indices],
            dtype=torch.int64,
            device="cpu",
        )
        self.alloc_offsets = torch.tensor(
            alloc_offsets, dtype=torch.int64, device="cpu"
        )
        self.device_dynamic_tensor_indices = device_dynamic_tensor_indices
        self.memcpy_node_updates = memcpy_node_updates
        self.memset_node_updates = memset_node_updates

    def replay(self, dynamic_tensors: list[torch.Tensor]) -> None:
        if self.memcpy_node_updates or self.memset_node_updates:
            actual_data_ptrs = [tensor.data_ptr() for tensor in dynamic_tensors]
            graph_exec = cuda_driver.CUgraphExec(
                init_value=self.graph.raw_cuda_graph_exec()
            )
            ctx = cuda_python_error_check(cuda_driver.cuCtxGetCurrent())
            for update in self.memcpy_node_updates:
                cuda_python_error_check(
                    cuda_driver.cuGraphExecMemcpyNodeSetParams(
                        graph_exec,
                        cuda_driver.CUgraphNode(init_value=update.node),
                        update.to_params(actual_data_ptrs),
                        ctx,
                    )
                )
            for update in self.memset_node_updates:
                cuda_python_error_check(
                    cuda_driver.cuGraphExecMemsetNodeSetParams(
                        graph_exec,
                        cuda_driver.CUgraphNode(init_value=update.node),
                        update.to_params(actual_data_ptrs),
                        ctx,
                    )
                )
        if self.device_nodes.numel():
            device_dynamic_tensors = [
                dynamic_tensors[idx] for idx in self.device_dynamic_tensor_indices
            ]
            for idx, tensor in zip(
                self.device_dynamic_tensor_indices, device_dynamic_tensors
            ):
                if not tensor.is_cuda:
                    raise RuntimeError(
                        "CUDA graph kernel pointer update references dynamic "
                        f"tensor allocation {idx}, but replay tensor is on "
                        f"{tensor.device}"
                    )
            torch._C._cuda_graph_apply_device_kernel_node_updates(
                self.device_nodes,
                self.param_offsets,
                self.alloc_indices,
                self.alloc_offsets,
                device_dynamic_tensors,
            )
        self.graph.replay()


def _create_and_sort_allocations(
    dynamic_tensors: list[tuple[int, int]],
) -> tuple[list[DynamicAllocation], list[DynamicAllocation], list[int]]:
    allocations = [
        DynamicAllocation(ptr=ptr, size=size, alloc_idx=i)
        for i, (ptr, size) in enumerate(dynamic_tensors)
    ]
    sorted_allocations = sorted(allocations, key=lambda alloc: alloc.ptr)
    for prev, cur in itertools.pairwise(sorted_allocations):
        if prev.ptr + prev.size > cur.ptr:
            raise RuntimeError("Dynamic tensors may not overlap")
    return allocations, sorted_allocations, [alloc.ptr for alloc in sorted_allocations]


def _check_allocation_within_graph(
    ptr: int,
    sorted_allocations: list[DynamicAllocation],
    allocation_starts: list[int],
) -> tuple[int, int] | None:
    if ptr == 0:
        return None
    idx = bisect.bisect_right(allocation_starts, ptr) - 1
    if idx < 0:
        return None
    allocation = sorted_allocations[idx]
    if ptr < allocation.ptr + allocation.size:
        return allocation.alloc_idx, ptr - allocation.ptr
    return None


def _dynamic_update_for_graph_pointer(
    ptr: int,
    other_ptr: int,
    sorted_allocations: list[DynamicAllocation],
    allocation_starts: list[int],
    other_sorted_allocations: list[DynamicAllocation],
    other_allocation_starts: list[int],
) -> GraphPointerUpdate | None:
    if ptr == other_ptr:
        return None
    result = _check_allocation_within_graph(ptr, sorted_allocations, allocation_starts)
    other_result = _check_allocation_within_graph(
        other_ptr, other_sorted_allocations, other_allocation_starts
    )
    if result is None and other_result is None:
        return None
    if result is None or other_result is None or result != other_result:
        raise RuntimeError(
            "CUDA graph pointer did not map consistently across captures"
        )
    alloc_idx, alloc_offset = result
    return GraphPointerUpdate(alloc_idx=alloc_idx, alloc_offset=alloc_offset)


def _graph_nodes(graph: Any) -> list[Any]:
    _, num_nodes = cuda_python_error_check(cuda_runtime.cudaGraphGetNodes(graph))
    if num_nodes == 0:
        return []
    nodes, _ = cuda_python_error_check(cuda_runtime.cudaGraphGetNodes(graph, num_nodes))
    return list(nodes)


def _kernel_param_infos(func: Any) -> list[tuple[int, int]]:
    if hasattr(cuda_driver, "cuFuncGetParamCount"):
        param_count = cuda_python_error_check(cuda_driver.cuFuncGetParamCount(func))
        return [
            cuda_python_error_check(cuda_driver.cuFuncGetParamInfo(func, param_index))
            for param_index in range(param_count)
        ]

    param_infos: list[tuple[int, int]] = []
    for param_index in itertools.count():
        result = cuda_driver.cuFuncGetParamInfo(func, param_index)
        error, *others = result
        if error == cuda_driver.CUresult.CUDA_ERROR_INVALID_VALUE:
            return param_infos
        param_offset, param_size = cuda_python_error_check(result)
        param_infos.append((param_offset, param_size))

    raise AssertionError("unreachable")


def _kernel_name(func: Any) -> str:
    name = cuda_python_error_check(cuda_driver.cuFuncGetName(func))
    if isinstance(name, bytes):
        return name.decode(errors="replace")
    return str(name)


def _kernel_arg_buffers(
    params: Any, param_infos: list[tuple[int, int]]
) -> tuple[bytes, ...]:
    if not param_infos:
        return ()

    if params.extra:
        packed_arg_buffer = _packed_kernel_arg_buffer_from_extra(params)
        buffers = []
        for param_offset, param_size in param_infos:
            end_offset = param_offset + param_size
            if end_offset > len(packed_arg_buffer):
                raise RuntimeError(
                    "CUDA graph kernel extra buffer is smaller than its params"
                )
            buffers.append(packed_arg_buffer[param_offset:end_offset])
        return tuple(buffers)

    if not params.kernelParams:
        raise RuntimeError(
            "CUDA graph kernel nodes without kernelParams are not supported"
        )

    arg_ptrs = (ctypes.c_void_p * len(param_infos)).from_address(params.kernelParams)
    buffers = []
    for param_index, (_, param_size) in enumerate(param_infos):
        arg_ptr = arg_ptrs[param_index]
        if not arg_ptr:
            raise RuntimeError("CUDA graph kernel parameter pointer is null")
        buffers.append(ctypes.string_at(arg_ptr, param_size))
    return tuple(buffers)


def _packed_kernel_arg_buffer_from_extra(params: Any) -> bytes:
    pointer_size = ctypes.sizeof(ctypes.c_void_p)
    extra = int(params.extra)
    buffer_ptr = None
    buffer_size = None

    for entry_idx in range(0, 16, 2):
        key = ctypes.c_void_p.from_address(extra + entry_idx * pointer_size).value
        if key in (None, int(cuda_driver.CU_LAUNCH_PARAM_END_AS_INT)):
            break

        value = ctypes.c_void_p.from_address(
            extra + (entry_idx + 1) * pointer_size
        ).value
        if key == int(cuda_driver.CU_LAUNCH_PARAM_BUFFER_POINTER_AS_INT):
            buffer_ptr = value
        elif key == int(cuda_driver.CU_LAUNCH_PARAM_BUFFER_SIZE_AS_INT):
            if value is None:
                raise RuntimeError("CUDA graph kernel extra buffer size is null")
            buffer_size = ctypes.c_size_t.from_address(value).value
    else:
        raise RuntimeError("CUDA graph kernel extra list is malformed")

    if buffer_ptr is None or buffer_size is None:
        raise RuntimeError("CUDA graph kernel extra buffer is missing")
    return ctypes.string_at(buffer_ptr, buffer_size)


def _dynamic_pointer_updates_for_kernel(
    arg_buffers: tuple[bytes, ...],
    other_arg_buffers: tuple[bytes, ...],
    sorted_allocations: list[DynamicAllocation],
    allocation_starts: list[int],
    other_sorted_allocations: list[DynamicAllocation],
    other_allocation_starts: list[int],
    *,
    debug_node_idx: int | None = None,
    debug_name: str | None = None,
) -> list[KernelParamPointerUpdate]:
    pointer_size = ctypes.sizeof(ctypes.c_void_p)
    updates = []
    for param_index, (arg_buffer, other_arg_buffer) in enumerate(
        zip(arg_buffers, other_arg_buffers)
    ):
        if len(arg_buffer) != len(other_arg_buffer):
            raise RuntimeError("CUDA graph kernel parameter sizes differ")
        for param_byte_offset in range(
            0, len(arg_buffer) - pointer_size + 1, pointer_size
        ):
            ptr = int.from_bytes(
                arg_buffer[param_byte_offset : param_byte_offset + pointer_size],
                sys.byteorder,
            )
            other_ptr = int.from_bytes(
                other_arg_buffer[param_byte_offset : param_byte_offset + pointer_size],
                sys.byteorder,
            )
            if ptr == other_ptr:
                continue
            result = _check_allocation_within_graph(
                ptr, sorted_allocations, allocation_starts
            )
            other_result = _check_allocation_within_graph(
                other_ptr, other_sorted_allocations, other_allocation_starts
            )
            if result is not None or other_result is not None:
                # Packed kernel argument structs can contain non-pointer 64-bit
                # fields that happen to fall inside a capture's allocation
                # range. The captures can assign the same logical allocation to
                # different segment indices, so require agreement on offset but
                # use the allocation index from the graph we will replay.
                if result is None or other_result is None:
                    debug_path = os.environ.get(
                        "TORCHINDUCTOR_CUDAGRAPH_DIGRAPHS_DEBUG_FILE"
                    )
                    if debug_path:
                        with open(debug_path, "a") as debug_file:
                            debug_file.write(
                                "ignored_one_sided_match "
                                f"node={debug_node_idx} "
                                f"name={debug_name} "
                                f"param={param_index} "
                                f"offset=0x{param_byte_offset:x} "
                                f"ptr=0x{ptr:x} other_ptr=0x{other_ptr:x} "
                                f"match={result} other_match={other_result}\n"
                            )
                    continue
                if result[1] != other_result[1]:
                    debug_path = os.environ.get(
                        "TORCHINDUCTOR_CUDAGRAPH_DIGRAPHS_DEBUG_FILE"
                    )
                    if debug_path:
                        with open(debug_path, "a") as debug_file:
                            debug_file.write(
                                "ignored_mismatch "
                                f"node={debug_node_idx} "
                                f"name={debug_name} "
                                f"param={param_index} "
                                f"offset=0x{param_byte_offset:x} "
                                f"ptr=0x{ptr:x} other_ptr=0x{other_ptr:x} "
                                f"match={result} other_match={other_result}\n"
                            )
                    raise RuntimeError(
                        "CUDA graph pointer did not map consistently across captures"
                    )
                alloc_idx, alloc_offset = result
                updates.append(
                    KernelParamPointerUpdate(
                        param_index=param_index,
                        param_byte_offset=param_byte_offset,
                        alloc_idx=alloc_idx,
                        alloc_offset=alloc_offset,
                    )
                )
    return updates


def _dump_kernel_pointer_debug(
    node_idx: int,
    name: str,
    params: Any,
    param_infos: list[tuple[int, int]],
    arg_buffers: tuple[bytes, ...],
    other_arg_buffers: tuple[bytes, ...],
    pointer_updates: list[KernelParamPointerUpdate],
    sorted_allocations: list[DynamicAllocation],
    allocation_starts: list[int],
    other_sorted_allocations: list[DynamicAllocation],
    other_allocation_starts: list[int],
) -> None:
    debug_path = os.environ.get("TORCHINDUCTOR_CUDAGRAPH_DIGRAPHS_DEBUG_FILE")
    if not debug_path:
        return
    pointer_size = ctypes.sizeof(ctypes.c_void_p)
    lines = [
        f"node={node_idx}",
        f"name={name}",
        f"grid=({params.gridDimX}, {params.gridDimY}, {params.gridDimZ})",
        f"block=({params.blockDimX}, {params.blockDimY}, {params.blockDimZ})",
        f"param_infos={param_infos}",
        "pointer_updates="
        + repr(
            [
                (
                    update.param_index,
                    param_infos[update.param_index][0] + update.param_byte_offset,
                    update.alloc_idx,
                    update.alloc_offset,
                )
                for update in pointer_updates
            ]
        ),
    ]
    selected_offsets = (
        0x0,
        0x8,
        0x10,
        0x18,
        0x30,
        0x180,
        0x188,
        0x1A8,
        0x1B0,
        0x1B8,
        0x1E0,
        0x208,
        0x210,
        0x218,
        0x228,
    )
    lines.append("selected_slots=")
    for param_index, arg_buffer in enumerate(arg_buffers):
        for param_byte_offset in selected_offsets:
            if param_byte_offset + pointer_size <= len(arg_buffer):
                ptr = int.from_bytes(
                    arg_buffer[param_byte_offset : param_byte_offset + pointer_size],
                    sys.byteorder,
                )
                result = _check_allocation_within_graph(
                    ptr, sorted_allocations, allocation_starts
                )
                lines.append(
                    "  "
                    f"param={param_index} "
                    f"offset=0x{param_byte_offset:x} "
                    f"value=0x{ptr:x} match={result}"
                )
    lines.append("changed_8byte_slots=")
    for param_index, (arg_buffer, other_arg_buffer) in enumerate(
        zip(arg_buffers, other_arg_buffers)
    ):
        param_offset, _ = param_infos[param_index]
        for param_byte_offset in range(
            0, len(arg_buffer) - pointer_size + 1, pointer_size
        ):
            slot = arg_buffer[param_byte_offset : param_byte_offset + pointer_size]
            other_slot = other_arg_buffer[
                param_byte_offset : param_byte_offset + pointer_size
            ]
            if slot == other_slot:
                continue
            ptr = int.from_bytes(slot, sys.byteorder)
            other_ptr = int.from_bytes(other_slot, sys.byteorder)
            result = _check_allocation_within_graph(
                ptr, sorted_allocations, allocation_starts
            )
            other_result = _check_allocation_within_graph(
                other_ptr, other_sorted_allocations, other_allocation_starts
            )
            lines.append(
                "  "
                f"param={param_index} "
                f"constant_offset=0x{param_offset + param_byte_offset:x} "
                f"value=0x{ptr:x} other_value=0x{other_ptr:x} "
                f"match={result} other_match={other_result}"
            )
    lines.append("matched_pointers=")
    for param_index, arg_buffer in enumerate(arg_buffers):
        param_offset, _ = param_infos[param_index]
        for param_byte_offset in range(
            0, len(arg_buffer) - pointer_size + 1, pointer_size
        ):
            ptr = int.from_bytes(
                arg_buffer[param_byte_offset : param_byte_offset + pointer_size],
                sys.byteorder,
            )
            result = _check_allocation_within_graph(
                ptr, sorted_allocations, allocation_starts
            )
            if result:
                alloc_idx, alloc_offset = result
                lines.append(
                    "  "
                    f"param={param_index} "
                    f"constant_offset=0x{param_offset + param_byte_offset:x} "
                    f"ptr=0x{ptr:x} alloc_idx={alloc_idx} alloc_offset={alloc_offset}"
                )
    with open(debug_path, "a") as debug_file:
        debug_file.write("\n".join(lines))
        debug_file.write("\n\n")


def _make_device_updatable_kernel_node(node: Any) -> int:
    attr_value = cuda_runtime.cudaKernelNodeAttrValue()
    attr_value.deviceUpdatableKernelNode.deviceUpdatable = 1
    cuda_python_error_check(
        cuda_runtime.cudaGraphKernelNodeSetAttribute(
            node,
            cuda_runtime.cudaKernelNodeAttrID.cudaLaunchAttributeDeviceUpdatableKernelNode,
            attr_value,
        )
    )
    dev_node = int(attr_value.deviceUpdatableKernelNode.devNode)
    if dev_node == 0:
        raise RuntimeError("CUDA did not return a device-updatable kernel node")
    return dev_node


def _check_memcpy_params_supported(params: Any) -> None:
    if int(params.srcArray) != 0 or int(params.dstArray) != 0:
        raise RuntimeError("CUDA array memcpy nodes are not supported")
    if params.kind not in (
        cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
        cuda_runtime.cudaMemcpyKind.cudaMemcpyDefault,
    ):
        raise RuntimeError(f"Unsupported CUDA graph memcpy kind: {params.kind}")
    assert params.extent.height == 1 and params.extent.depth == 1, (
        "Expected 1D CUDA graph memcpy node, got "
        f"width={params.extent.width}, "
        f"height={params.extent.height}, "
        f"depth={params.extent.depth}"
    )
    assert (
        params.srcPos.y == 0
        and params.srcPos.z == 0
        and params.dstPos.y == 0
        and params.dstPos.z == 0
    ), (
        "Expected 1D CUDA graph memcpy node positions, got "
        f"srcPos=({params.srcPos.x}, {params.srcPos.y}, "
        f"{params.srcPos.z}), dstPos=({params.dstPos.x}, "
        f"{params.dstPos.y}, {params.dstPos.z})"
    )


def _memcpy_metadata_key(params: Any) -> tuple[Any, ...]:
    return (
        params.kind,
        params.srcPtr.pitch,
        params.srcPtr.xsize,
        params.srcPtr.ysize,
        params.srcPos.x,
        params.srcPos.y,
        params.srcPos.z,
        params.dstPtr.pitch,
        params.dstPtr.xsize,
        params.dstPtr.ysize,
        params.dstPos.x,
        params.dstPos.y,
        params.dstPos.z,
        params.extent.width,
        params.extent.height,
        params.extent.depth,
    )


def _memcpy_node_update(
    node: Any,
    params: Any,
    other_params: Any,
    sorted_allocations: list[DynamicAllocation],
    allocation_starts: list[int],
    other_sorted_allocations: list[DynamicAllocation],
    other_allocation_starts: list[int],
) -> MemcpyNodeUpdate | None:
    _check_memcpy_params_supported(params)
    _check_memcpy_params_supported(other_params)
    if _memcpy_metadata_key(params) != _memcpy_metadata_key(other_params):
        raise RuntimeError("Captured CUDA graph memcpy params differ")

    src_update = _dynamic_update_for_graph_pointer(
        int(params.srcPtr.ptr),
        int(other_params.srcPtr.ptr),
        sorted_allocations,
        allocation_starts,
        other_sorted_allocations,
        other_allocation_starts,
    )
    dst_update = _dynamic_update_for_graph_pointer(
        int(params.dstPtr.ptr),
        int(other_params.dstPtr.ptr),
        sorted_allocations,
        allocation_starts,
        other_sorted_allocations,
        other_allocation_starts,
    )
    if src_update is None and dst_update is None:
        return None
    return MemcpyNodeUpdate(
        node=int(node),
        src_ptr=int(params.srcPtr.ptr),
        src_pitch=params.srcPtr.pitch,
        src_xsize=params.srcPtr.xsize,
        src_ysize=params.srcPtr.ysize,
        src_pos=(params.srcPos.x, params.srcPos.y, params.srcPos.z),
        dst_ptr=int(params.dstPtr.ptr),
        dst_pitch=params.dstPtr.pitch,
        dst_xsize=params.dstPtr.xsize,
        dst_ysize=params.dstPtr.ysize,
        dst_pos=(params.dstPos.x, params.dstPos.y, params.dstPos.z),
        extent=(params.extent.width, params.extent.height, params.extent.depth),
        src_update=src_update,
        dst_update=dst_update,
    )


def _check_memset_params_supported(params: Any) -> None:
    assert params.height == 1, (
        "Expected 1D CUDA graph memset node, got "
        f"width={params.width}, height={params.height}, pitch={params.pitch}, "
        f"elementSize={params.elementSize}"
    )
    if params.elementSize not in (1, 2, 4):
        raise RuntimeError(
            f"Unsupported CUDA graph memset element size: {params.elementSize}"
        )


def _memset_metadata_key(params: Any) -> tuple[int, int, int, int, int]:
    return (
        params.pitch,
        params.value,
        params.elementSize,
        params.width,
        params.height,
    )


def _memset_node_update(
    node: Any,
    params: Any,
    other_params: Any,
    sorted_allocations: list[DynamicAllocation],
    allocation_starts: list[int],
    other_sorted_allocations: list[DynamicAllocation],
    other_allocation_starts: list[int],
) -> MemsetNodeUpdate | None:
    _check_memset_params_supported(params)
    _check_memset_params_supported(other_params)
    if _memset_metadata_key(params) != _memset_metadata_key(other_params):
        raise RuntimeError("Captured CUDA graph memset params differ")

    dst_update = _dynamic_update_for_graph_pointer(
        int(params.dst),
        int(other_params.dst),
        sorted_allocations,
        allocation_starts,
        other_sorted_allocations,
        other_allocation_starts,
    )
    if dst_update is None:
        return None
    return MemsetNodeUpdate(
        node=int(node),
        dst=int(params.dst),
        pitch=params.pitch,
        value=params.value,
        element_size=params.elementSize,
        width=params.width,
        height=params.height,
        dst_update=dst_update,
    )


def make_dynamic_graph_runner(
    graph: torch.cuda.CUDAGraph,
    dynamic_tensors: list[tuple[int, int]],
    other_graph: torch.cuda.CUDAGraph,
    other_dynamic_tensors: list[tuple[int, int]],
) -> PythonDynamicCUDAGraph:
    """Build a replay helper that updates dynamic pointers in a captured graph."""
    _, sorted_allocations, allocation_starts = _create_and_sort_allocations(
        dynamic_tensors
    )
    (
        _,
        other_sorted_allocations,
        other_allocation_starts,
    ) = _create_and_sort_allocations(other_dynamic_tensors)

    graph_handle = cuda_runtime.cudaGraph_t(init_value=graph.raw_cuda_graph())
    other_graph_handle = cuda_runtime.cudaGraph_t(
        init_value=other_graph.raw_cuda_graph()
    )
    nodes = _graph_nodes(graph_handle)
    other_nodes = _graph_nodes(other_graph_handle)
    if len(nodes) != len(other_nodes):
        raise RuntimeError("Captured CUDA graphs are not topologically identical")

    device_nodes: list[int] = []
    param_offsets: list[int] = []
    alloc_indices: list[int] = []
    alloc_offsets: list[int] = []
    memcpy_node_updates: list[MemcpyNodeUpdate] = []
    memset_node_updates: list[MemsetNodeUpdate] = []
    allowed_node_types = OrderedSet(
        [
            cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeEmpty,
            cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeWaitEvent,
            cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeHost,
            cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeEventRecord,
            cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeExtSemaphoreSignal,
            cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeExtSemaphoreWait,
            cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeMemAlloc,
            cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeMemFree,
        ]
    )

    for node_idx, (node, other_node) in enumerate(zip(nodes, other_nodes)):
        node_type = cuda_python_error_check(cuda_runtime.cudaGraphNodeGetType(node))
        other_node_type = cuda_python_error_check(
            cuda_runtime.cudaGraphNodeGetType(other_node)
        )
        if node_type != other_node_type:
            raise RuntimeError("Captured CUDA graphs have different node types")

        if node_type == cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeKernel:
            params = cuda_python_error_check(
                cuda_driver.cuGraphKernelNodeGetParams(node)
            )
            other_params = cuda_python_error_check(
                cuda_driver.cuGraphKernelNodeGetParams(other_node)
            )
            param_infos = _kernel_param_infos(params.func)
            other_param_infos = _kernel_param_infos(other_params.func)
            if param_infos != other_param_infos:
                raise RuntimeError("Captured CUDA graph kernel params differ")

            arg_buffers = _kernel_arg_buffers(params, param_infos)
            other_arg_buffers = _kernel_arg_buffers(other_params, other_param_infos)
            name = _kernel_name(params.func)
            try:
                pointer_updates = _dynamic_pointer_updates_for_kernel(
                    arg_buffers,
                    other_arg_buffers,
                    sorted_allocations,
                    allocation_starts,
                    other_sorted_allocations,
                    other_allocation_starts,
                    debug_node_idx=node_idx,
                    debug_name=name,
                )
            except RuntimeError as e:
                _dump_kernel_pointer_debug(
                    node_idx,
                    name,
                    params,
                    param_infos,
                    arg_buffers,
                    other_arg_buffers,
                    [],
                    sorted_allocations,
                    allocation_starts,
                    other_sorted_allocations,
                    other_allocation_starts,
                )
                raise RuntimeError(
                    "Failed to identify CUDA graph kernel pointer updates "
                    f"for node_idx={node_idx} kernel={name} "
                    f"grid=({params.gridDimX}, {params.gridDimY}, {params.gridDimZ}) "
                    f"block=({params.blockDimX}, {params.blockDimY}, {params.blockDimZ})"
                ) from e
            if os.environ.get("TORCHINDUCTOR_CUDAGRAPH_DIGRAPHS_DEBUG_FILE"):
                if "cublas" in name:
                    _dump_kernel_pointer_debug(
                        node_idx,
                        name,
                        params,
                        param_infos,
                        arg_buffers,
                        other_arg_buffers,
                        pointer_updates,
                        sorted_allocations,
                        allocation_starts,
                        other_sorted_allocations,
                        other_allocation_starts,
                    )
            if pointer_updates:
                dev_node = _make_device_updatable_kernel_node(node)
                for update in pointer_updates:
                    param_offset, _ = param_infos[update.param_index]
                    device_nodes.append(dev_node)
                    param_offsets.append(param_offset + update.param_byte_offset)
                    alloc_indices.append(update.alloc_idx)
                    alloc_offsets.append(update.alloc_offset)
        elif node_type == cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeMemcpy:
            if not config.triton.cudagraphs_preserve_memops:
                raise RuntimeError("memcpy nodes should have been removed")
            memcpy_update = _memcpy_node_update(
                node,
                cuda_python_error_check(
                    cuda_runtime.cudaGraphMemcpyNodeGetParams(node)
                ),
                cuda_python_error_check(
                    cuda_runtime.cudaGraphMemcpyNodeGetParams(other_node)
                ),
                sorted_allocations,
                allocation_starts,
                other_sorted_allocations,
                other_allocation_starts,
            )
            if memcpy_update is not None:
                memcpy_node_updates.append(memcpy_update)
        elif node_type == cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeMemset:
            if not config.triton.cudagraphs_preserve_memops:
                raise RuntimeError("memset nodes should have been removed")
            memset_update = _memset_node_update(
                node,
                cuda_python_error_check(
                    cuda_runtime.cudaGraphMemsetNodeGetParams(node)
                ),
                cuda_python_error_check(
                    cuda_runtime.cudaGraphMemsetNodeGetParams(other_node)
                ),
                sorted_allocations,
                allocation_starts,
                other_sorted_allocations,
                other_allocation_starts,
            )
            if memset_update is not None:
                memset_node_updates.append(memset_update)
        elif node_type in (
            cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeGraph,
            cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeConditional,
        ):
            raise RuntimeError("nested CUDA graph nodes are not supported")
        elif node_type not in allowed_node_types:
            raise RuntimeError(f"Unsupported CUDA graph node type: {node_type}")

    graph.instantiate()
    if device_nodes:
        stream = torch.cuda.current_stream()
        cuda_python_error_check(
            cuda_runtime.cudaGraphUpload(
                cuda_runtime.cudaGraphExec_t(init_value=graph.raw_cuda_graph_exec()),
                cuda_runtime.cudaStream_t(init_value=stream.cuda_stream),
            )
        )
        stream.synchronize()
    return PythonDynamicCUDAGraph(
        graph,
        device_nodes,
        param_offsets,
        alloc_indices,
        alloc_offsets,
        memcpy_node_updates,
        memset_node_updates,
    )


memcpy_kernel = None
memset_kernel = None


def replace_memops_with_kernels(graph: cuda_runtime.cudaGraph_t) -> None:
    """
    Replace all memcpy and memset nodes in a CUDA graph with equivalent kernel implementations.

    Args:
        graph: The CUDA graph to modify
    """
    # Get all nodes in the graph
    _, num_nodes = cuda_python_error_check(cuda_runtime.cudaGraphGetNodes(graph))
    if num_nodes == 0:
        return
    nodes, _ = cuda_python_error_check(cuda_runtime.cudaGraphGetNodes(graph, num_nodes))

    # Compile kernels for memcpy and memset
    global memcpy_kernel
    global memset_kernel
    if memcpy_kernel is None:
        memcpy_kernel, _ = compile_memcpy_kernel()
    if memset_kernel is None:
        memset_kernel, _ = compile_memset_kernel()

    # Process each node in the graph
    for node in nodes:
        # Get node type
        node_type = cuda_python_error_check(cuda_runtime.cudaGraphNodeGetType(node))

        if node_type == cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeMemcpy:
            replace_memcpy_with_kernel(graph, node, memcpy_kernel)

        elif node_type == cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeMemset:
            replace_memset_with_kernel(graph, node, memset_kernel)


def compile_memcpy_kernel() -> tuple[cuda_runtime.cudaFunction_t, str]:
    """Compile the memcpy kernel with NVRTC"""
    kernel_source = """
    extern "C" __global__ void custom_memcpy_kernel(
        void* dst,
        const void* src,
        size_t row_bytes,
        size_t height,
        size_t depth,
        size_t dst_pitch,
        size_t src_pitch,
        size_t dst_slice_pitch,
        size_t src_slice_pitch) {
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;
        size_t count = row_bytes * height * depth;

        char* dst_ptr = (char*)dst;
        const char* src_ptr = (const char*)src;

        for (size_t i = tid; i < count; i += stride) {
            size_t x = i % row_bytes;
            size_t row = i / row_bytes;
            size_t y = row % height;
            size_t z = row / height;
            dst_ptr[z * dst_slice_pitch + y * dst_pitch + x] =
                src_ptr[z * src_slice_pitch + y * src_pitch + x];
        }
    }
    """

    # Compile with NVRTC
    prog = cuda_python_error_check(
        nvrtc.nvrtcCreateProgram(kernel_source.encode(), b"memcpy_kernel.cu", 0, [], [])
    )
    cuda_python_error_check(nvrtc.nvrtcCompileProgram(prog, 0, []))

    # Get PTX and write to file
    ptx_size = cuda_python_error_check(nvrtc.nvrtcGetPTXSize(prog))
    ptx = b" " * ptx_size
    cuda_python_error_check(nvrtc.nvrtcGetPTX(prog, ptx))

    # Load the kernel
    module = cuda_python_error_check(cuda_driver.cuModuleLoadData(ptx))
    kernel = cuda_python_error_check(
        cuda_driver.cuModuleGetFunction(module, b"custom_memcpy_kernel")
    )

    return kernel, "custom_memcpy_kernel"


def compile_memset_kernel() -> tuple[cuda_runtime.cudaFunction_t, str]:
    """Compile the memset kernel with NVRTC"""
    kernel_source = """
    extern "C" __global__ void custom_memset_kernel(
        void* dst,
        unsigned int value,
        size_t width,
        size_t height,
        size_t pitch,
        unsigned int element_size) {
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;
        size_t count = width * height;

        char* dst_ptr = (char*)dst;

        for (size_t i = tid; i < count; i += stride) {
            size_t x = i % width;
            size_t y = i / width;
            char* ptr = dst_ptr + y * pitch + x * element_size;
            if (element_size == 1) {
                *((unsigned char*)ptr) = (unsigned char)value;
            } else if (element_size == 2) {
                *((unsigned short*)ptr) = (unsigned short)value;
            } else {
                *((unsigned int*)ptr) = value;
            }
        }
    }
    """

    # Compile with NVRTC
    prog = cuda_python_error_check(
        nvrtc.nvrtcCreateProgram(kernel_source.encode(), b"memset_kernel.cu", 0, [], [])
    )
    cuda_python_error_check(nvrtc.nvrtcCompileProgram(prog, 0, []))

    # Get PTX and write to file
    ptx_size = cuda_python_error_check(nvrtc.nvrtcGetPTXSize(prog))
    ptx = b" " * ptx_size
    cuda_python_error_check(nvrtc.nvrtcGetPTX(prog, ptx))

    # Load the kernel
    module = cuda_python_error_check(cuda_driver.cuModuleLoadData(ptx))
    kernel = cuda_python_error_check(
        cuda_driver.cuModuleGetFunction(module, b"custom_memset_kernel")
    )

    return kernel, "custom_memset_kernel"


def replace_memcpy_with_kernel(graph, memcpy_node, kernel):
    """Replace a memcpy node with an equivalent kernel node"""
    # Get the memcpy node parameters
    memcpy_params = cuda_python_error_check(
        cuda_runtime.cudaGraphMemcpyNodeGetParams(memcpy_node)
    )
    if int(memcpy_params.srcArray) != 0 or int(memcpy_params.dstArray) != 0:
        raise RuntimeError("CUDA array memcpy nodes are not supported")
    if memcpy_params.kind not in (
        cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
        cuda_runtime.cudaMemcpyKind.cudaMemcpyDefault,
    ):
        raise RuntimeError(f"Unsupported CUDA graph memcpy kind: {memcpy_params.kind}")
    assert memcpy_params.extent.height == 1 and memcpy_params.extent.depth == 1, (
        "Expected 1D CUDA graph memcpy node, got "
        f"width={memcpy_params.extent.width}, "
        f"height={memcpy_params.extent.height}, "
        f"depth={memcpy_params.extent.depth}"
    )
    assert (
        memcpy_params.srcPos.y == 0
        and memcpy_params.srcPos.z == 0
        and memcpy_params.dstPos.y == 0
        and memcpy_params.dstPos.z == 0
    ), (
        "Expected 1D CUDA graph memcpy node positions, got "
        f"srcPos=({memcpy_params.srcPos.x}, {memcpy_params.srcPos.y}, "
        f"{memcpy_params.srcPos.z}), dstPos=({memcpy_params.dstPos.x}, "
        f"{memcpy_params.dstPos.y}, {memcpy_params.dstPos.z})"
    )

    # Create parameters for kernel node
    kernel_params = cuda_driver.CUDA_KERNEL_NODE_PARAMS()

    # Set up kernel execution parameters
    row_bytes = memcpy_params.extent.width
    height = memcpy_params.extent.height
    depth = memcpy_params.extent.depth
    size = row_bytes * height * depth

    # Simple heuristic for grid and block size
    block_size = 256
    grid_size = (size + block_size - 1) // block_size
    grid_size = min(grid_size, 65535)  # Ensure grid size is within limits

    kernel_params.gridDimX = grid_size
    kernel_params.gridDimY = 1
    kernel_params.gridDimZ = 1
    kernel_params.blockDimX = block_size
    kernel_params.blockDimY = 1
    kernel_params.blockDimZ = 1
    kernel_params.sharedMemBytes = 0

    # Setup kernel arguments
    dst_pitch = memcpy_params.dstPtr.pitch or row_bytes
    src_pitch = memcpy_params.srcPtr.pitch or row_bytes
    dst_ysize = memcpy_params.dstPtr.ysize or height
    src_ysize = memcpy_params.srcPtr.ysize or height
    dst_slice_pitch = dst_pitch * dst_ysize
    src_slice_pitch = src_pitch * src_ysize
    dst_ptr = (
        int(memcpy_params.dstPtr.ptr)
        + memcpy_params.dstPos.x
        + memcpy_params.dstPos.y * dst_pitch
        + memcpy_params.dstPos.z * dst_slice_pitch
    )
    src_ptr = (
        int(memcpy_params.srcPtr.ptr)
        + memcpy_params.srcPos.x
        + memcpy_params.srcPos.y * src_pitch
        + memcpy_params.srcPos.z * src_slice_pitch
    )

    kernel_params.kernelParams = (
        (
            dst_ptr,
            src_ptr,
            row_bytes,
            height,
            depth,
            dst_pitch,
            src_pitch,
            dst_slice_pitch,
            src_slice_pitch,
        ),
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ),
    )
    kernel_params.func = kernel

    # Create the new kernel node
    new_node = cuda_python_error_check(
        cuda_driver.cuGraphAddKernelNode(graph, [], 0, kernel_params)
    )

    replace_node_in_graph(graph, memcpy_node, new_node)


def replace_memset_with_kernel(graph, memset_node, kernel):
    """Replace a memset node with an equivalent kernel node"""
    # Get the memset node parameters
    memset_params = cuda_python_error_check(
        cuda_runtime.cudaGraphMemsetNodeGetParams(memset_node)
    )

    # Create parameters for kernel node
    kernel_params = cuda_driver.CUDA_KERNEL_NODE_PARAMS()

    # Calculate total size based on memset params
    width = memset_params.width
    height = memset_params.height
    element_size = memset_params.elementSize
    assert height == 1, (
        "Expected 1D CUDA graph memset node, got "
        f"width={width}, height={height}, pitch={memset_params.pitch}, "
        f"elementSize={element_size}"
    )
    if element_size not in (1, 2, 4):
        raise RuntimeError(
            f"Unsupported CUDA graph memset element size: {element_size}"
        )
    pitch = memset_params.pitch or width * element_size
    size = width * height

    # Simple heuristic for grid and block size
    block_size = 256
    grid_size = (size + block_size - 1) // block_size
    grid_size = min(grid_size, 65535)  # Ensure grid size is within limits

    kernel_params.gridDimX = grid_size
    kernel_params.gridDimY = 1
    kernel_params.gridDimZ = 1
    kernel_params.blockDimX = block_size
    kernel_params.blockDimY = 1
    kernel_params.blockDimZ = 1
    kernel_params.sharedMemBytes = 0

    # Setup kernel arguments
    dst_ptr = memset_params.dst
    value = memset_params.value

    # Create kernel args (dst, value, size)
    kernel_params.kernelParams = (
        (dst_ptr, value, width, height, pitch, element_size),
        (
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint,
        ),
    )
    # This crashes when you use
    # cuda_runtime.cudaGraphKernelNodeParams, so we use the driver API
    # instead.
    kernel_params.func = kernel

    # Create the new kernel node
    new_node = cuda_python_error_check(
        cuda_driver.cuGraphAddKernelNode(graph, [], 0, kernel_params)
    )

    replace_node_in_graph(graph, memset_node, new_node)


def _cuda_graph_node_links(query: Callable[..., Any], node: Any) -> list[Any]:
    result = cuda_python_error_check(query(node))
    if len(result) == 2:
        nodes, num_nodes = result
    else:
        nodes, _edge_data, num_nodes = result
    if num_nodes == 0:
        return []

    result = cuda_python_error_check(query(node, num_nodes))
    if len(result) == 2:
        nodes, num_nodes = result
    else:
        nodes, _edge_data, num_nodes = result
    return list(nodes[:num_nodes])


def _cuda_graph_node_link_queries() -> tuple[Callable[..., Any], Callable[..., Any]]:
    runtime_version = cuda_python_error_check(cuda_runtime.cudaRuntimeGetVersion())
    if runtime_version >= 13000:
        return (
            cuda_runtime.cudaGraphNodeGetDependencies,
            cuda_runtime.cudaGraphNodeGetDependentNodes,
        )
    return (
        cuda_runtime.cudaGraphNodeGetDependencies_v2,
        cuda_runtime.cudaGraphNodeGetDependentNodes_v2,
    )


def _cuda_graph_add_dependencies(
    graph: Any,
    from_nodes: list[Any],
    to_nodes: list[Any],
    num_dependencies: int,
) -> None:
    runtime_version = cuda_python_error_check(cuda_runtime.cudaRuntimeGetVersion())
    if runtime_version >= 13000:
        cuda_python_error_check(
            cuda_runtime.cudaGraphAddDependencies(
                graph, from_nodes, to_nodes, None, num_dependencies
            )
        )
    else:
        cuda_python_error_check(
            cuda_runtime.cudaGraphAddDependencies(
                graph, from_nodes, to_nodes, num_dependencies
            )
        )


def replace_node_in_graph(graph, old_node, new_node):
    """
    Replace a node in a CUDA graph by removing the old node and adding the new one.

    Args:
        graph (cudaGraph_t): The CUDA graph to modify
        old_node (cudaGraphNode_t): The node to remove from the graph
        new_node (cudaGraphNode_t): The node to add to the graph
    """
    get_dependencies, get_dependent_nodes = _cuda_graph_node_link_queries()
    dependencies = _cuda_graph_node_links(get_dependencies, old_node)
    dependent_nodes = _cuda_graph_node_links(get_dependent_nodes, old_node)

    # Remove the old node from the graph
    cuda_python_error_check(cuda_runtime.cudaGraphDestroyNode(old_node))

    # TODO: preserve non-default edge data once cuda-python deep-copies it.
    # See NVIDIA/cuda-python#1804.

    # Add the new node to the graph with the same dependencies as the old node
    if len(dependencies) > 0:
        _cuda_graph_add_dependencies(
            graph,
            dependencies,
            [new_node] * len(dependencies),
            len(dependencies),
        )

    # Add the dependencies from the new node to the old node's dependent nodes
    if len(dependent_nodes) > 0:
        _cuda_graph_add_dependencies(
            graph,
            [new_node] * len(dependent_nodes),
            dependent_nodes,
            len(dependent_nodes),
        )
