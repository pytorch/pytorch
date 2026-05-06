from __future__ import annotations

import bisect
import ctypes
import dataclasses
import itertools
import operator
import os
import sys
from typing import Any, TYPE_CHECKING

import cuda.bindings.driver as cuda_driver  # pyrefly: ignore [missing-import]
import cuda.bindings.nvrtc as nvrtc  # pyrefly: ignore [missing-import]
import cuda.bindings.runtime as cuda_runtime  # pyrefly: ignore [missing-import]

import torch
from torch._dynamo.utils import preserve_rng_state
from torch._inductor.compile_fx import (
    copy_misaligned_inputs,
    get_input_idxs_to_check,
    remove_unaligned_input_idxs,
    static_input,
)
from torch._inductor.cudagraph_utils import (
    maybe_warning_due_to_dynamic_shape,
    ModelType,
    OutputType,
    PlaceholderInfo,
)
from torch._inductor.utils import clone_preserve_strides
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


def cudagraphify_impl(
    model: ModelType,
    # Some inputs are ints, while others are SymInts, hmmm....
    inputs: list[InputType],
    static_input_idxs: Sequence[int],
    *args: Any,
    **kwargs: Any,
) -> ModelType:
    fn_cache: dict[tuple[int, ...], Callable[..., Any]] = {}

    # Detect int inputs: we need to index on these
    int_key = [i for i, v in enumerate(inputs) if isinstance(v, int)]
    get_ints: Any = operator.itemgetter(*int_key) if int_key else lambda _: None

    has_warn = False

    del inputs

    def deferred_cudagraphify(inputs: list[InputType]) -> OutputType:
        nonlocal has_warn

        int_key = get_ints(inputs)
        fn = fn_cache.get(int_key)
        if fn is not None:
            return fn(inputs)

        if int_key is None:
            log.info("recording cudagraph tree for graph without symints")
        else:
            log.info("recording cudagraph tree for symint key %s", int_key)

        if not has_warn:
            has_warn = maybe_warning_due_to_dynamic_shape(fn_cache, int_key)

        # first get indices we need to check to align, then update our static inputs,
        # and finally copy
        check_input_idxs = get_input_idxs_to_check(inputs, static_input_idxs)
        new_static_input_idxs = remove_unaligned_input_idxs(inputs, static_input_idxs)
        copy_misaligned_inputs(inputs, check_input_idxs)

        # here is the real cudagraphify. If I understand correctly, it
        # will run eagerly first, and then do stream capture. That
        # works, right?
        fn, out = cudagraphify(model, inputs, new_static_input_idxs, *args, **kwargs)
        fn_cache[int_key] = fn

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
    with torch.cuda.stream(stream):
        outputs = model(list(inputs))

    graphs: list[torch.cuda.CUDAGraph] = []
    dynamic_tensors_list: list[list[tuple[int, int]]] = []
    capture_keepalive: list[Any] = []
    static_inputs: list[InputType] = []
    static_outputs: tuple[torch.Tensor | int | None, ...] = ()
    segment_address_starts: list[int] = []
    segment_sizes: list[int] = []
    segment_devices: list[torch.device] = []
    segment_idx_containing_this_output_tensor: list[int | None] = []
    static_output_idx_to_input_idx_and_offset: dict[int, tuple[int, int]] = {}
    non_tensor_output_idxs: OrderedSet[int] = OrderedSet()
    empty_tensor_idxs_to_devices: dict[int, torch.device] = {}

    for i in range(2):
        torch._C._cuda_clearCublasWorkspaces()
        graph = torch.cuda.CUDAGraph(keep_graph=True)

        assert len(static_input_idxs) == 0

        old_value = torch._C._get_deterministic_fill_uninitialized_memory()
        torch._C._set_deterministic_fill_uninitialized_memory(False)
        try:
            static_inputs = [
                (
                    x
                    if not isinstance(x, torch.Tensor)
                    else static_input(
                        x
                    )  # this guarantees an alignment of 256 bytes. Hmmm...
                )
                for x in inputs
            ]
            capture_keepalive.append(static_inputs)
        finally:
            torch._C._set_deterministic_fill_uninitialized_memory(old_value)

        for si in static_inputs:
            if not isinstance(si, torch.Tensor):
                continue
            alignment = max_alignment(si.data_ptr())
            # cuda train AlbertForMaskedLM has a CPU tensor for
            # some reason, which doesn't have the 256 byte
            # alignment you would expect.
            assert (si.is_cuda and alignment == 0 or alignment >= 256) or si.is_cpu, (
                "Static input alignment is less than expected"
            )

        static_inputs_new_list = list(static_inputs)
        with (
            preserve_rng_state(),
            torch.cuda.graph(
                graph,
                stream=stream,
                capture_error_mode="thread_local",
            ),
        ):
            model_outputs = model(static_inputs_new_list)

        replace_memops_with_kernels(
            cuda_runtime.cudaGraph_t(init_value=graph.raw_cuda_graph())
        )

        if not isinstance(model_outputs, (list, tuple)):
            static_outputs = (model_outputs,)
        else:
            static_outputs = tuple(model_outputs)
        capture_keepalive.append(static_outputs)

        memory_snapshot: list[dict[str, Any]] = torch.cuda.memory_snapshot(graph.pool())
        memory_snapshot.sort(key=lambda x: x["address"])

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

        segment_idx_containing_this_output_tensor = []

        static_output_idx_to_input_idx_and_offset = {}

        static_inputs_only_tensors: list[tuple[int, int]] = [
            (input.data_ptr(), nbytes_underlying_storage(input))
            for idx, input in enumerate(static_inputs)
            if isinstance(input, torch.Tensor) and input.is_cuda
        ]

        non_tensor_output_idxs = OrderedSet()
        empty_tensor_idxs_to_devices = {}

        # I need to map each segment index to all output tensors, I think.
        for static_output_idx, static_output in enumerate(static_outputs):
            if not isinstance(static_output, torch.Tensor):
                assert static_output is None or isinstance(static_output, int)
                non_tensor_output_idxs.add(static_output_idx)
                segment_idx_containing_this_output_tensor.append(None)
                continue
            if static_output.data_ptr() == 0:
                assert static_output.nbytes == 0
                empty_tensor_idxs_to_devices[static_output_idx] = static_output.device
                segment_idx_containing_this_output_tensor.append(None)
                continue
            assert static_output.is_cuda, (
                "I suppose non cuda outputs are allowed, but I would like to "
                "catch them explicitly for now"
            )
            segment_idx = (
                bisect.bisect(segment_address_starts, static_output.data_ptr()) - 1
            )
            not_found = (
                segment_idx == -1
                or not static_output.data_ptr()
                < segment_address_starts[segment_idx] + segment_sizes[segment_idx]
            )
            if not_found:
                segment_idx_containing_this_output_tensor.append(-1)
                for i, (static_input_data_ptr, static_input_nbytes) in enumerate(
                    static_inputs_only_tensors
                ):
                    if (
                        static_input_data_ptr <= static_output.data_ptr()
                        and static_output.data_ptr()
                        + nbytes_underlying_storage(static_output)
                        <= static_input_data_ptr + static_input_nbytes
                    ):
                        assert (
                            static_output_idx
                            not in static_output_idx_to_input_idx_and_offset
                        ), (
                            "Static inputs should never share a buffer "
                            "during stream capture!!!"
                        )
                        # TODO: define the behavior when inputs alias each other.
                        offset_from_input = (
                            static_output.data_ptr() - static_input_data_ptr
                        )
                        assert offset_from_input % static_output.itemsize == 0
                        offset_from_input = offset_from_input // static_output.itemsize
                        static_output_idx_to_input_idx_and_offset[static_output_idx] = (
                            i,
                            offset_from_input,
                        )
                # In this case, the output must be part of a
                # non-dynamic input tensor (which we should
                # verify!). In that situation, the output tensor
                # should always have the same output address across
                # runs.
                assert static_output_idx in static_output_idx_to_input_idx_and_offset, (
                    static_output_idx,
                    static_output.data_ptr(),
                    nbytes_underlying_storage(static_output),
                )
                continue
            segment_idx_containing_this_output_tensor.append(segment_idx)

        find_overlaps(list(static_inputs_only_tensors), True)
        dynamic_tensor_allocations = list(static_inputs_only_tensors)
        find_overlaps(list(zip(segment_address_starts, segment_sizes)), True)
        dynamic_tensor_allocations.extend(zip(segment_address_starts, segment_sizes))

        dynamic_tensors_list.append(dynamic_tensor_allocations)
        graphs.append(graph)

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

    assert len(static_input_idxs) == 0

    dynamic_input_idxs: OrderedSet[int] = OrderedSet()
    for idx, static_input_value in enumerate(static_inputs):
        if (
            idx not in static_input_idxs
            and isinstance(static_input_value, torch.Tensor)
            and static_input_value.is_cuda
        ):
            dynamic_input_idxs.add(idx)

    def run(new_inputs: list[InputType]) -> OutputType:
        """Replay the graph with new input and output storage pointers."""
        assert len(static_inputs) == len(new_inputs)

        dynamic_tensors: list[torch.Tensor] = []

        old_tensors: list[torch.Tensor] = []
        new_tensors: list[torch.Tensor] = []

        # We need to speed this up somehow. Maybe I can keep the
        # parameters as static inputs. I just need to make sure that
        # tangents are not considered static, IIUC. There are way too
        # many inputs, and this loop ends up taking >100 microseconds,
        # which makes parameterized cuda graph launch slower than
        # cudagraph trees in the end.
        torch.cuda.nvtx.range_push("push inputs")
        for idx in dynamic_input_idxs:
            new_input = new_inputs[idx]
            assert isinstance(new_input, torch.Tensor)
            # TODO: This isn't quite right, is it? It's possible that
            # a CUDA kernel may expect a larger alignment than the
            # dynamic input may have.  We can get an upper bound on
            # the alignment of a tensor, though I suppose we can
            # assume that the alignment is always at least 256 bytes,
            # since cudaMalloc() is guaranteed to return a pointer
            # with at least that much alignment, and each tensor
            # returned by pytorch has at least 512 byte alignment
            # inside that buffer.
            # So the mod 16 here should be really be modulo the
            # original tensor's alignment, for safety.
            if new_input.data_ptr() % 16 == 0:
                dynamic_tensors.append(new_input)
            else:
                dynamic_tensors.append(clone_preserve_strides(new_input))
                new_tensors.append(dynamic_tensors[-1])
                old_tensors.append(new_input)
        torch.cuda.nvtx.range_pop()

        dynamic_tensors.extend(
            torch.empty(size, dtype=torch.int8, device=device)
            for size, device in zip(segment_sizes, segment_devices)
        )

        graph_runner.replay(dynamic_tensors)

        if old_tensors:
            # Copy data back. I need to do this only if the input is
            # marked as mutable or if an output aliases the input
            torch._foreach_copy_(old_tensors, new_tensors)

        new_inputs_only_tensors = [
            input
            for input in new_inputs
            if isinstance(input, torch.Tensor) and input.is_cuda
        ]

        outputs: OutputType = []

        for i, static_output in enumerate(static_outputs):
            if i in non_tensor_output_idxs:
                outputs.append(static_output)
                continue
            if i in empty_tensor_idxs_to_devices:
                outputs.append(torch.empty(0, device=empty_tensor_idxs_to_devices[i]))
                continue
            containing_segment_idx = segment_idx_containing_this_output_tensor[i]
            assert containing_segment_idx is not None
            assert isinstance(static_output, torch.Tensor)
            if containing_segment_idx == -1:
                input_idx, offset_from_input = (
                    static_output_idx_to_input_idx_and_offset[i]
                )
                input_tensor = new_inputs_only_tensors[input_idx]
                true_output_tensor = torch.empty(
                    (), device=static_output.device, dtype=static_output.dtype
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
                    storage_offset=offset_from_input + input_offset_from_storage,
                    stride=static_output.stride(),
                    size=static_output.size(),
                )

                outputs.append(true_output_tensor)
                continue
            storage_tensor = dynamic_tensors[
                len(dynamic_input_idxs) + containing_segment_idx
            ]
            storage_offset = (
                static_output.data_ptr()
                - segment_address_starts[containing_segment_idx]
            )
            assert storage_offset < segment_sizes[containing_segment_idx]
            assert (
                nbytes_underlying_storage(static_output)
                <= segment_sizes[containing_segment_idx]
            )
            assert (
                storage_offset + nbytes_underlying_storage(static_output)
                <= segment_sizes[containing_segment_idx]
            )
            true_output_tensor = torch.empty(
                (), device=static_output.device, dtype=static_output.dtype
            )

            # This is the storage_offset in bytes, but set_ requires
            # an offset in terms of the dtype of the tensor
            assert storage_offset % static_output.itemsize == 0
            storage_offset_itemsize = storage_offset // static_output.itemsize

            true_output_tensor.set_(
                storage_tensor.untyped_storage(),
                storage_offset=storage_offset_itemsize,
                stride=static_output.stride(),
                size=static_output.size(),
            )
            outputs.append(true_output_tensor)

        return outputs

    return run, outputs


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
class KernelNodeParamsTemplate:
    func: int
    gridDimX: int
    gridDimY: int
    gridDimZ: int
    blockDimX: int
    blockDimY: int
    blockDimZ: int
    sharedMemBytes: int
    kern: int
    ctx: int

    @staticmethod
    def from_params(params: Any) -> KernelNodeParamsTemplate:
        if int(params.func) == 0 and int(params.kern) == 0:
            raise RuntimeError("CUDA graph kernel node has neither func nor kern set")
        return KernelNodeParamsTemplate(
            func=int(params.func),
            gridDimX=params.gridDimX,
            gridDimY=params.gridDimY,
            gridDimZ=params.gridDimZ,
            blockDimX=params.blockDimX,
            blockDimY=params.blockDimY,
            blockDimZ=params.blockDimZ,
            sharedMemBytes=params.sharedMemBytes,
            kern=int(params.kern),
            ctx=int(params.ctx),
        )

    def to_params(
        self, arg_buffers: Sequence[bytes | bytearray]
    ) -> tuple[Any, list[Any]]:
        params = cuda_driver.CUDA_KERNEL_NODE_PARAMS()
        params.gridDimX = self.gridDimX
        params.gridDimY = self.gridDimY
        params.gridDimZ = self.gridDimZ
        params.blockDimX = self.blockDimX
        params.blockDimY = self.blockDimY
        params.blockDimZ = self.blockDimZ
        params.sharedMemBytes = self.sharedMemBytes
        params.extra = 0

        keepalive: list[Any] = []
        if self.func != 0:
            func = cuda_driver.CUfunction(init_value=self.func)
            keepalive.append(func)
            params.func = func
        elif self.kern != 0:
            kern = cuda_driver.CUkernel(init_value=self.kern)
            keepalive.append(kern)
            params.kern = kern
            if self.ctx != 0:
                ctx = cuda_driver.CUcontext(init_value=self.ctx)
                keepalive.append(ctx)
                params.ctx = ctx
        else:
            raise RuntimeError("CUDA graph kernel node has neither func nor kern set")

        arg_ptrs = (ctypes.c_void_p * len(arg_buffers))()
        for i, arg_buffer in enumerate(arg_buffers):
            buffer = ctypes.create_string_buffer(bytes(arg_buffer), len(arg_buffer))
            keepalive.append(buffer)
            arg_ptrs[i] = ctypes.addressof(buffer)

        keepalive.append(arg_ptrs)
        params.kernelParams = ctypes.addressof(arg_ptrs)
        return params, keepalive


@dataclasses.dataclass(frozen=True)
class KernelNodeUpdate:
    node: Any
    params_template: KernelNodeParamsTemplate
    original_arg_buffers: tuple[bytes, ...]
    pointer_updates: tuple[KernelParamPointerUpdate, ...]

    def to_params(self, actual_data_ptrs: list[int]) -> tuple[Any, list[Any]]:
        arg_buffers = [bytearray(buffer) for buffer in self.original_arg_buffers]
        for update in self.pointer_updates:
            new_ptr = actual_data_ptrs[update.alloc_idx] + update.alloc_offset
            arg_buffer = arg_buffers[update.param_index]
            arg_buffer[
                update.param_byte_offset : update.param_byte_offset
                + ctypes.sizeof(ctypes.c_void_p)
            ] = new_ptr.to_bytes(ctypes.sizeof(ctypes.c_void_p), sys.byteorder)
        return self.params_template.to_params(arg_buffers)


class PythonDynamicCUDAGraph:
    def __init__(
        self,
        graph: torch.cuda.CUDAGraph,
        kernel_node_updates: list[KernelNodeUpdate],
    ) -> None:
        self.graph = graph
        self.kernel_node_updates = kernel_node_updates
        self._last_param_keepalive: list[Any] = []

    def replay(self, dynamic_tensors: list[torch.Tensor]) -> None:
        actual_data_ptrs = [tensor.data_ptr() for tensor in dynamic_tensors]
        keepalive: list[Any] = []
        graph_exec = cuda_driver.CUgraphExec(
            init_value=self.graph.raw_cuda_graph_exec()
        )

        for update in self.kernel_node_updates:
            params, params_keepalive = update.to_params(actual_data_ptrs)
            keepalive.extend(params_keepalive)
            cuda_python_error_check(
                cuda_driver.cuGraphExecKernelNodeSetParams(
                    graph_exec, update.node, params
                )
            )

        self._last_param_keepalive = keepalive
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


def _graph_nodes(graph: Any) -> list[Any]:
    _, num_nodes = cuda_python_error_check(cuda_runtime.cudaGraphGetNodes(graph))
    if num_nodes == 0:
        return []
    nodes, _ = cuda_python_error_check(cuda_runtime.cudaGraphGetNodes(graph, num_nodes))
    return list(nodes)


def _kernel_param_infos(func: Any) -> list[tuple[int, int]]:
    param_count = cuda_python_error_check(cuda_driver.cuFuncGetParamCount(func))
    return [
        cuda_python_error_check(cuda_driver.cuFuncGetParamInfo(func, param_index))
        for param_index in range(param_count)
    ]


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
            result = _check_allocation_within_graph(
                ptr, sorted_allocations, allocation_starts
            )
            other_result = _check_allocation_within_graph(
                other_ptr, other_sorted_allocations, other_allocation_starts
            )
            if result and other_result:
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

    kernel_node_updates = []
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
            pointer_updates = _dynamic_pointer_updates_for_kernel(
                arg_buffers,
                other_arg_buffers,
                sorted_allocations,
                allocation_starts,
                other_sorted_allocations,
                other_allocation_starts,
            )
            if os.environ.get("TORCHINDUCTOR_CUDAGRAPH_DIGRAPHS_DEBUG_FILE"):
                name = _kernel_name(params.func)
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
                kernel_node_updates.append(
                    KernelNodeUpdate(
                        node=node,
                        params_template=KernelNodeParamsTemplate.from_params(params),
                        original_arg_buffers=arg_buffers,
                        pointer_updates=tuple(pointer_updates),
                    )
                )
        elif node_type in (
            cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeMemcpy,
            cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeMemset,
        ):
            raise RuntimeError("memcpy and memset nodes should have been removed")
        elif node_type in (
            cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeGraph,
            cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeConditional,
        ):
            raise RuntimeError("nested CUDA graph nodes are not supported")
        elif node_type not in allowed_node_types:
            raise RuntimeError(f"Unsupported CUDA graph node type: {node_type}")

    graph.instantiate()
    return PythonDynamicCUDAGraph(graph, kernel_node_updates)


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
    nodes, _edge_data, num_nodes = cuda_python_error_check(query(node))
    if num_nodes == 0:
        return []

    nodes, _edge_data, num_nodes = cuda_python_error_check(query(node, num_nodes))
    return list(nodes[:num_nodes])


def replace_node_in_graph(graph, old_node, new_node):
    """
    Replace a node in a CUDA graph by removing the old node and adding the new one.

    Args:
        graph (cudaGraph_t): The CUDA graph to modify
        old_node (cudaGraphNode_t): The node to remove from the graph
        new_node (cudaGraphNode_t): The node to add to the graph
    """
    dependencies = _cuda_graph_node_links(
        cuda_runtime.cudaGraphNodeGetDependencies, old_node
    )
    dependent_nodes = _cuda_graph_node_links(
        cuda_runtime.cudaGraphNodeGetDependentNodes, old_node
    )

    # Remove the old node from the graph
    cuda_python_error_check(cuda_runtime.cudaGraphDestroyNode(old_node))

    # TODO: preserve non-default edge data once cuda-python deep-copies it.
    # See NVIDIA/cuda-python#1804.

    # Add the new node to the graph with the same dependencies as the old node
    if len(dependencies) > 0:
        cuda_python_error_check(
            cuda_runtime.cudaGraphAddDependencies(
                graph,
                dependencies,
                [new_node] * len(dependencies),
                None,
                len(dependencies),
            )
        )

    # Add the dependencies from the new node to the old node's dependent nodes
    if len(dependent_nodes) > 0:
        cuda_python_error_check(
            cuda_runtime.cudaGraphAddDependencies(
                graph,
                [new_node] * len(dependent_nodes),
                dependent_nodes,
                None,
                len(dependent_nodes),
            )
        )
