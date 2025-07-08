from __future__ import annotations

import cuda.bindings.runtime as cudart

import bisect
import contextlib
import dataclasses
import functools
import gc
import itertools
import operator
import sys
import threading
import traceback
import warnings
import weakref
from collections import defaultdict
from contextlib import AbstractContextManager
from enum import auto, Enum
from typing import Any, Callable, cast, Optional, TYPE_CHECKING, TypeVar, Union

import torch.fx
from torch import Tensor
from torch._dynamo.callback import CallbackTrigger
from torch._dynamo.mutation_guard import GenerationTracker
from torch._dynamo.utils import counters, dynamo_timed, preserve_rng_state
from torch._inductor.compile_fx import (
    align_inputs_from_check_idxs,
    copy_misaligned_inputs,
    get_expanded_dims,
    get_input_idxs_to_check,
    index_expanded_dims,
    remove_unaligned_input_idxs,
    static_input,
)
from torch._inductor.cudagraph_utils import (
    check_for_mutation,
    CheckInvariantStatus,
    FunctionID,
    log_cudagraph_skip_and_bump_counter,
    log_data_ptr_mismatch,
    maybe_warning_due_to_dynamic_shape,
    ModelType,
    OutputType,
    PlaceholderInfo,
    WrappedFunction,
)
from torch.multiprocessing.reductions import StorageWeakRef
from torch.storage import UntypedStorage
from torch.utils import _pytree as pytree
from torch.utils._ordered_set import OrderedSet
from torch.utils.weak import TensorWeakRef


if TYPE_CHECKING:
    from collections.abc import Generator, Iterator, Sequence

    from torch._guards import CompileId
    from torch._inductor.utils import InputType
    from torch.types import _bool

StorageWeakRefPointer = int
StorageDataPtr = int
NBytes = int
S = TypeVar("S", bound="StorageWeakRefWrapper")


if torch.backends.cuda.is_built():
    from torch._C import (
        _cuda_CUDAAllocator_AllocatorState as AllocatorState,
        _set_cached_tensors_enabled as _set_cached_tensors_enabled,
    )
else:

    class AllocatorState:  # type: ignore[no-redef]
        pass

    def _set_cached_tensors_enabled(enabled: _bool) -> None:
        pass


log = torch._logging.getArtifactLogger(__name__, "cudagraphs")


from . import config

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

        # Okay, so these form a key that we use to create different
        # cuda graphs.
        int_key = get_ints(inputs)
        fn = fn_cache.get(int_key)
        if fn is not None:
            # print("GALVEZ: getting cached cudagraph!")
            return fn(inputs)

        # Yeah, recording a graph key
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
                print(f"Overlap found between:")
                print(f"  Interval {i}: start = {start1}, length = {len1}")
                print(f"  Interval {j}: start = {start2}, length = {len2}")
                print()
                if assert_should_not_happen:
                    assert False

def cudagraphify(
    model: ModelType,
    inputs: list[InputType],
    static_input_idxs: Sequence[int] = (),
    *,
    device_index: int,
    is_backward: bool,
    is_inference: bool,
    stack_traces: Optional[StackTraces] = None,
    constants: tuple[torch.Tensor, ...] = (),
    placeholders: tuple[PlaceholderInfo, ...] = (),
    mutated_input_idxs: tuple[int, ...] = (),
    compile_id: Optional[CompileId] = None,
) -> tuple[ModelType, OutputType]:
    # inps_expanded_dims = [
    #     get_expanded_dims(x) if idx not in static_input_idxs else []
    #     for idx, x in enumerate(inputs)
    # ]

    print("GALVEZ:cudagraph_digraphs.py cudagraphify")

    graph = torch.cuda.CUDAGraph(keep_graph=True)
    dynamic_graph_arguments = True
    mem_allocator = graph.get_mem_allocator()
    input_pool = torch.cuda.MemPool(mem_allocator)

    torch.cuda.memory._record_memory_history(True)

    with torch.cuda.use_mem_pool(input_pool):
        static_inputs = [
            (
                x
                if not isinstance(x, torch.Tensor)
                else static_input(x)
                if idx not in static_input_idxs
                else x.detach() # Interesting to detatch these...
            )
            for idx, x in enumerate(inputs)
        ]

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    print("GALVEZ: inputs=", len(inputs))
    with torch.cuda.stream(stream):
        outputs = model(inputs)

    print("GALVEZ: outputs=", len(outputs))

    torch.cuda.synchronize()

    # Commeting pool.id fixes everythin for me. Hmm... The
    # "inactive" part is a bit worrisome.Can I inspect to see when
    # segments become inactive?

    print("GALVEZ: input memory snapshots:")
    input_memory_snapshot = torch.cuda.memory_snapshot(input_pool.id)
    input_memory_snapshot.sort(key=lambda x: x['address'])
    import pprint
    pprint.pprint(input_memory_snapshot)

    pool = torch.cuda.MemPool(mem_allocator)
    with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local",
                          dynamic_graph=dynamic_graph_arguments, pool=pool.id):
        # Can't I get the underlying fxgraph here? No.
        static_outputs = model(list(static_inputs))

    replace_memops_with_kernels(cudart.cudaGraph_t(init_value=graph.raw_cuda_graph()))

    print("GALVEZ: output memory snapshots:")
    memory_snapshot = torch.cuda.memory_snapshot(pool.id)
    import pprint
    pprint.pprint(memory_snapshot)

    # import ipdb; ipdb.set_trace()

    # TODO: Why this? I should really be iterating over the outptuts as if they are a pytree...
    if not isinstance(static_outputs, (list, tuple)):
        static_outputs = (static_outputs,)

    assert graph.pool() == pool.id

    memory_snapshot = torch.cuda.memory_snapshot(graph.pool())
    # memory_snapshot = input_memory_snapshot + memory_snapshot
    memory_snapshot.sort(key=lambda x: x['address'])

    segment_address_starts = [segment_snapshot['address'] for segment_snapshot in memory_snapshot]
    segment_sizes = [segment_snapshot['total_size'] for segment_snapshot in memory_snapshot]
    segment_devices = [torch.device("cuda", segment_snapshot['device']) for segment_snapshot in memory_snapshot]

    containing_segment_idxs = OrderedSet()
    segment_idx_containing_this_output_tensor = []
    dynamic_idx_containing_this_output_tensor = []

    # static_inputs_only_tensors_possible_overlaps = [input for idx, input in enumerate(static_inputs) if idx not in static_input_idxs and isinstance(input, torch.Tensor) and input.is_cuda]

    static_inputs_only_tensors = [(input.data_ptr(), input.nbytes) for idx, input in enumerate(static_inputs) if idx not in static_input_idxs and isinstance(input, torch.Tensor) and input.is_cuda]

    # Maybe the simplest strategy is to simply find all of the
    # segments corresponding to at least one input tensor. Collect them in
    # an OrderedSet.

    input_segment_address_starts = [segment_snapshot['address'] for segment_snapshot in input_memory_snapshot]
    input_segment_sizes = [segment_snapshot['total_size'] for segment_snapshot in input_memory_snapshot]

    # static_inputs_only_tensors = []
    # dynamic_input_idxs = OrderedSet()
    # # does_index_need_underlying_segment = set()
    # for i, input in enumerate(static_inputs_only_tensors_possible_overlaps):
    #     start1 = input.data_ptr()
    #     end1 = input.data_ptr() + input.nbytes
    #     is_contained_within_other_tensor = False
    #     for j in range(i + 1, len(static_inputs_only_tensors_possible_overlaps)):
    #         input2 = static_inputs_only_tensors_possible_overlaps[j]
    #         start2 = input2.data_ptr()
    #         end2 = input2.data_ptr() + input2.nbytes
    #         if start1 >= start2 and end1 <= end2:
    #             # input1 is contained within input2
    #             # break because we will get to input2 later in the outer loop
    #             is_contained_within_other_tensor = True
    #             break
    #         elif start1 < end2 and start2 < end1:
    #             # input2 and input2 overlap
    #             assert bisect.bisect(input_segment_address_starts, start1) == bisect.bisect(input_segment_address_starts, start2)
    #             segment_idx = bisect.bisect(input_segment_address_starts, start1) - 1
    #             static_inputs_only_tensors.append((input_segment_address_starts[segment_idx], input_segment_sizes[segment_idx]))
    #             raise ValueError("Overlap without containment of inputs not handled yet.")
    #             break

    #     if not is_contained_within_other_tensor:
    #         static_inputs_only_tensors.append((input.data_ptr(), input.nbytes))
    #         dynamic_input_idxs.add(i)

    # I need to figure out what to do when two inputs overlap during
    # stream capture but do not overlap during replay...
    output_tensors_tuples = [(output.data_ptr(), output.nbytes) for idx, output in enumerate(static_outputs)]

    print("GALVEZ input overlaps")
    find_overlaps(list(static_inputs_only_tensors), True)
    print("GALVEZ input output overlaps")
    find_overlaps(static_inputs_only_tensors + output_tensors_tuples)
    print("GALVEZ: static_inputs_only_tensors", static_inputs_only_tensors)
    print("GALVEZ: output_tensors_tuples", output_tensors_tuples)


    static_output_idx_to_input_idx_and_offset = {}

    # I need to map each segment index to all output tensors, I think.
    for static_output_idx, static_output in enumerate(static_outputs):
        segment_idx = bisect.bisect(segment_address_starts, static_output.data_ptr()) - 1
        # assert segment_idx != -1, "Found an output address with no underlying allocation. This is likely due to it being a view of an input tensor"
        if segment_idx == -1:
            dynamic_idx_containing_this_output_tensor.append(-1)
            segment_idx_containing_this_output_tensor.append(-1)
            # TODO: Figure out which input this falls into.
            for i, (static_input_data_ptr, static_input_nbytes) in enumerate(static_inputs_only_tensors):
                if (static_input_data_ptr <= static_output.data_ptr() and
                    static_output.data_ptr() + static_output.nbytes <= static_input_data_ptr + static_input_nbytes
                    ):
                    assert not static_output_idx in static_output_idx_to_input_idx_and_offset, "GALVEZ: inputs sharing a buffer!!!"
                    # TODO: We need to think about what to do when inputs are aliasing each other!
                    offset_from_input = static_output.data_ptr() - static_input_data_ptr
                    static_output_idx_to_input_idx_and_offset[static_output_idx] = (i, offset_from_input)
                assert static_output_idx in static_output_idx_to_input_idx_and_offset
            continue
        if segment_idx in containing_segment_idxs:
            for i, containing_segment_idx in enumerate(containing_segment_idxs):
                if segment_idx == containing_segment_idx:
                    dynamic_idx_containing_this_output_tensor.append(i)
                    break
        else:
            dynamic_idx_containing_this_output_tensor.append(len(containing_segment_idxs))
        containing_segment_idxs.add(segment_idx)
        segment_idx_containing_this_output_tensor.append(segment_idx)

    static_output_segment_tensors = []

    for segment_idx in containing_segment_idxs:
        static_output_segment_tensors.append((segment_address_starts[segment_idx], segment_sizes[segment_idx]))

    torch.cuda.synchronize()

    # does this include tensors? I think so...
    dynamic_tensors = list(static_inputs_only_tensors) + static_output_segment_tensors

    print("GALVEZ: static_output_segment_tensors= ", static_output_segment_tensors)
    print("GALVEZ: containing_segment_idxs= ", containing_segment_idxs)

    print("GALVEZ output overlaps")
    find_overlaps(list(static_output_segment_tensors))
    print("GALVEZ total overlaps")
    find_overlaps(dynamic_tensors)

    graph.become_dynamic(dynamic_tensors)

    dynamic_input_idxs = OrderedSet([idx for idx in range(len(static_inputs)) if idx not in static_input_idxs and isinstance(static_inputs[idx], torch.Tensor)])

    torch.cuda.memory._record_memory_history(False)

    def run(new_inputs):
        assert len(static_inputs) == len(new_inputs)

        dynamic_tensors = []

        for idx in dynamic_input_idxs:
            dynamic_tensors.append(new_inputs[idx])

        # torch.cuda.nvtx.range_push("Dynamic outputs creation")
        for segment_idx in containing_segment_idxs:
            containing_segment_size_bytes = segment_sizes[segment_idx]
            containing_segment_device = segment_devices[segment_idx]

            storage_tensor = torch.empty(containing_segment_size_bytes, dtype=torch.int8, device=containing_segment_device)
            dynamic_tensors.append(storage_tensor)
        # torch.cuda.nvtx.range_pop()

        graph.replay_dynamic(dynamic_tensors)

        output_tensors = []

        for i, static_output in enumerate(static_outputs):
            dynamic_segment_idx = dynamic_idx_containing_this_output_tensor[i]
            containing_segment_idx = segment_idx_containing_this_output_tensor[i]
            if dynamic_segment_idx == -1:
                assert containing_segment_idx == -1
                input_idx, offset_from_input = static_output_idx_to_input_idx_and_offset[i]
                true_output_tensor = torch.empty((), device=static_output.device, dtype=static_output.dtype)
                true_output_tensor.set_(dynamic_tensors[input_idx].untyped_storage(),
                                        storage_offset=offset_from_input,
                                        stride=static_output.stride(),
                                        size=static_output.size())
                output_tensors.append(true_output_tensor)
                continue
            # print("GALVEZ:", i, dynamic_segment_idx, containing_segment_idx, len(dynamic_tensors), len(dynamic_input_idxs))
            storage_tensor = dynamic_tensors[len(dynamic_input_idxs) + dynamic_segment_idx]
            storage_offset = static_output.data_ptr() - segment_address_starts[containing_segment_idx]
            true_output_tensor = torch.empty((), device=static_output.device, dtype=static_output.dtype)
            # print("GALVEZ:", static_output.size(), static_output.stride(), storage_offset)
            true_output_tensor.set_(storage_tensor.untyped_storage(),
                                    storage_offset=storage_offset,
                                    stride=static_output.stride(),
                                    size=static_output.size())
            output_tensors.append(true_output_tensor)

        return output_tensors

    return run, outputs

import ctypes
import tempfile
from typing import Dict, List, Tuple

import cuda.bindings.runtime as cuda_runtime
import cuda.bindings.driver  as cuda_driver
import cuda.nvrtc as nvrtc

def cuda_python_error_check(function_call_output):
    """Makes calls to cuda-python's cuda runtime functions more
    pythonic by throwing an exception if they return a status
    which is not cudaSuccess
    """
    import cuda.bindings  # type: ignore[import]

    error, *others = function_call_output
    if (isinstance(error, cuda.bindings.runtime.cudaError_t)
        and error != cuda.bindings.runtime.cudaError_t.cudaSuccess):
        raise ValueError(f"CUDA failure! {error}")
    elif (isinstance(error, cuda.bindings.driver.CUresult)
          and error != cuda.bindings.driver.CUresult.CUDA_SUCCESS):
        raise ValueError(f"CUDA failure! {error}")
    elif (isinstance(error, cuda.bindings.nvrtc.nvrtcResult)
          and error != cuda.bindings.nvrtc.nvrtcResult.NVRTC_SUCCESS):
        raise ValueError(f"NVRTC failure! {error}")
    elif len(others) == 1:
        return others[0]
    else:
        return tuple(others)

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
    memcpy_kernel, memcpy_func = compile_memcpy_kernel()
    memset_kernel, memset_func = compile_memset_kernel()
    
    # Process each node in the graph
    for node in nodes:
        # Get node type
        node_type = cuda_python_error_check(cuda_runtime.cudaGraphNodeGetType(node))
        
        if node_type == cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeMemcpy:
            replace_memcpy_with_kernel(graph, node, memcpy_kernel, memcpy_func)
        
        elif node_type == cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeMemset:
            replace_memset_with_kernel(graph, node, memset_kernel, memset_func)

def compile_memcpy_kernel() -> Tuple[cuda_runtime.cudaFunction_t, str]:
    """Compile the memcpy kernel with NVRTC"""
    kernel_source = """
    extern "C" __global__ void custom_memcpy_kernel(void* dst, const void* src, size_t count) {
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;
        
        // Copy data as bytes
        char* dst_ptr = (char*)dst;
        const char* src_ptr = (const char*)src;
        
        for (size_t i = tid; i < count; i += stride) {
            dst_ptr[i] = src_ptr[i];
        }
    }
    """
    
    # Create a temp file for the PTX
    # with tempfile.NamedTemporaryFile(suffix='.ptx', delete=False) as ptx_file:
    #     ptx_path = ptx_file.name
    
    # Compile with NVRTC
    prog = cuda_python_error_check(nvrtc.nvrtcCreateProgram(kernel_source.encode(), b"memcpy_kernel.cu", 0, [], []))
    cuda_python_error_check(nvrtc.nvrtcCompileProgram(prog, 0, []))
    
    # Get PTX and write to file
    ptx_size = cuda_python_error_check(nvrtc.nvrtcGetPTXSize(prog))
    ptx = b' ' * ptx_size
    cuda_python_error_check(nvrtc.nvrtcGetPTX(prog, ptx))
    
    # with open(ptx_path, 'wb') as f:
    #     f.write(ptx)
    
    # Load the kernel
    module = cuda_python_error_check(cuda_driver.cuModuleLoadData(ptx))
    kernel = cuda_python_error_check(cuda_driver.cuModuleGetFunction(module, b"custom_memcpy_kernel"))
    
    return kernel, "custom_memcpy_kernel"

def compile_memset_kernel() -> Tuple[cuda_runtime.cudaFunction_t, str]:
    """Compile the memset kernel with NVRTC"""
    kernel_source = """
    extern "C" __global__ void custom_memset_kernel(void* dst, int value, size_t count) {
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;
        
        // Set memory as bytes
        unsigned char* dst_ptr = (unsigned char*)dst;
        unsigned char byte_value = (unsigned char)value;
        
        for (size_t i = tid; i < count; i += stride) {
            dst_ptr[i] = byte_value;
        }
    }
    """
    
    # Create a temp file for the PTX
    with tempfile.NamedTemporaryFile(suffix='.ptx', delete=False) as ptx_file:
        ptx_path = ptx_file.name
    
    # Compile with NVRTC
    prog = cuda_python_error_check(nvrtc.nvrtcCreateProgram(kernel_source.encode(), b"memset_kernel.cu", 0, [], []))
    cuda_python_error_check(nvrtc.nvrtcCompileProgram(prog, 0, []))
    
    # Get PTX and write to file
    ptx_size = cuda_python_error_check(nvrtc.nvrtcGetPTXSize(prog))
    ptx = b' ' * ptx_size
    cuda_python_error_check(nvrtc.nvrtcGetPTX(prog, ptx))

    # Load the kernel
    module = cuda_python_error_check(cuda_driver.cuModuleLoadData(ptx))
    kernel = cuda_python_error_check(cuda_driver.cuModuleGetFunction(module, b"custom_memset_kernel"))
    
    return kernel, "custom_memset_kernel"

def replace_memcpy_with_kernel(graph, memcpy_node, kernel, func_name):
    """Replace a memcpy node with an equivalent kernel node"""
    # Get the memcpy node parameters
    memcpy_params = cuda_python_error_check(cuda_runtime.cudaGraphMemcpyNodeGetParams(memcpy_node))
    
    # Get dependencies of the memcpy node
    _, num_dependencies = cuda_python_error_check(cuda_runtime.cudaGraphNodeGetDependencies(memcpy_node))
    if num_dependencies == 0:
        dependencies = []
    else:
        dependencies, _ = cuda_python_error_check(cuda_runtime.cudaGraphNodeGetDependencies(memcpy_node, num_dependencies))
    
    # Create parameters for kernel node
    kernel_params = cuda_driver.CUDA_KERNEL_NODE_PARAMS()
    
    # Set up kernel execution parameters
    size = memcpy_params.extent.width * memcpy_params.extent.height * memcpy_params.extent.depth
    
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
    dst_ptr = memcpy_params.dstPtr.ptr
    src_ptr = memcpy_params.srcPtr.ptr
    
    kernel_params.kernelParams = (
        (dst_ptr, src_ptr, size),
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t),
    )
    kernel_params.func = kernel
    
    # Create the new kernel node
    new_node = cuda_python_error_check(cuda_driver.cuGraphAddKernelNode(
        graph, dependencies, num_dependencies, kernel_params))
    
    # Remove the old memcpy node
    cuda_python_error_check(cuda_runtime.cudaGraphDestroyNode(memcpy_node))

def replace_memset_with_kernel(graph, memset_node, kernel, func_name):
    """Replace a memset node with an equivalent kernel node"""
    # Get the memset node parameters
    memset_params = cuda_python_error_check(cuda_runtime.cudaGraphMemsetNodeGetParams(memset_node))
    
    # Get dependencies of the memset node
    _, num_dependencies = cuda_python_error_check(cuda_runtime.cudaGraphNodeGetDependencies(memset_node))
    if num_dependencies == 0:
        dependencies = []
    else:
        dependencies, _ = cuda_python_error_check(cuda_runtime.cudaGraphNodeGetDependencies(memset_node, num_dependencies))
    
    # Create parameters for kernel node
    kernel_params = cuda_driver.CUDA_KERNEL_NODE_PARAMS()
    
    # Calculate total size based on memset params
    width = memset_params.width
    height = memset_params.height
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
        (dst_ptr, value, size),
        (ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t)
    )
    # This crashes when you use
    # cuda_runtime.cudaGraphKernelNodeParams, so we use the driver API
    # instead.
    kernel_params.func = kernel
    
    # Create the new kernel node
    new_node = cuda_python_error_check(cuda_driver.cuGraphAddKernelNode(
        graph, dependencies, num_dependencies, kernel_params))
    
    # Remove the old memset node
    cuda_python_error_check(cuda_runtime.cudaGraphDestroyNode(memset_node))
