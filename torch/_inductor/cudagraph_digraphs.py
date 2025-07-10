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
    constants: tuple[torch.Tensor, ...] = (), # TODO: How is constants different from inputs contained in static_input_idxs?
    placeholders: tuple[PlaceholderInfo, ...] = (),
    mutated_input_idxs: tuple[int, ...] = (),
    compile_id: Optional[CompileId] = None,
) -> tuple[ModelType, OutputType]:
    # TODO: we should fail if this fxgraph ever inspects data_ptr()

    print("GALVEZ:cudagraph_digraphs.py cudagraphify")

    graph = torch.cuda.CUDAGraph(keep_graph=True)
    dynamic_graph_arguments = True
    mem_allocator = graph.get_mem_allocator()
    input_pool = torch.cuda.MemPool(mem_allocator)

    # torch.cuda.memory._record_memory_history(True)

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
    with torch.cuda.stream(stream):
        outputs = model(inputs)

    # Note: not really any need for this.
    torch.cuda.synchronize()

    pool = torch.cuda.MemPool(mem_allocator)
    with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local",
                          dynamic_graph=dynamic_graph_arguments, pool=pool.id):
        # Can't I get the underlying fxgraph here? No.
        static_outputs = model(list(static_inputs))

    replace_memops_with_kernels(cudart.cudaGraph_t(init_value=graph.raw_cuda_graph()))

    if not isinstance(static_outputs, (list, tuple)):
        static_outputs = (static_outputs,)

    assert graph.pool() == pool.id

    memory_snapshot = torch.cuda.memory_snapshot(graph.pool())
    memory_snapshot.sort(key=lambda x: x['address'])

    segment_address_starts = [segment_snapshot['address'] for segment_snapshot in memory_snapshot]
    segment_sizes = [segment_snapshot['total_size'] for segment_snapshot in memory_snapshot]
    segment_devices = [torch.device("cuda", segment_snapshot['device']) for segment_snapshot in memory_snapshot]

    containing_segment_idxs = OrderedSet()
    segment_idx_containing_this_output_tensor = []
    dynamic_idx_containing_this_output_tensor = []

    static_output_idx_to_input_idx_and_offset = {}

    # print("GALVEZ:input_memory_snapshot")
    # input_memory_snapshot = torch.cuda.memory_snapshot(input_pool.id)
    # import pprint
    # pprint.pprint(input_memory_snapshot)
    # print("GALVEZ:output_and_temp_memory_snapshot")
    # output_and_temp_memory_snapshot = torch.cuda.memory_snapshot(graph.pool())
    # import pprint
    # pprint.pprint(output_and_temp_memory_snapshot)
    # print("GALVEZ:total_memory_snapshot")
    # total_memory_snapshot = torch.cuda.memory_snapshot()
    # import pprint
    # pprint.pprint(total_memory_snapshot)

    static_inputs_only_tensors = [(input.data_ptr(), input.nbytes) for idx, input in enumerate(static_inputs) if isinstance(input, torch.Tensor) and input.is_cuda]
    static_inputs_only_dynamic_tensors = [(input.data_ptr(), input.nbytes) for idx, input in enumerate(static_inputs) if idx not in static_input_idxs and isinstance(input, torch.Tensor) and input.is_cuda]

    non_tensor_output_idxs = set()

    # I need to map each segment index to all output tensors, I think.
    for static_output_idx, static_output in enumerate(static_outputs):
        if not isinstance(static_output, torch.Tensor):
            non_tensor_output_idxs.add(static_output_idx)
            dynamic_idx_containing_this_output_tensor.append(None)
            segment_idx_containing_this_output_tensor.append(None)
            # containing_segment_idxs.add(None)
            continue
        assert static_output.is_cuda, "I suppose non cuda outputs are allowed, but I would like to catch them explicitly for now"
        segment_idx = bisect.bisect(segment_address_starts, static_output.data_ptr()) - 1
        not_found = (segment_idx == -1 or not static_output.data_ptr() < segment_address_starts[segment_idx] + segment_sizes[segment_idx])
        # assert segment_idx != -1, "Found an output address with no underlying allocation. This is likely due to it being a view of an input tensor"
        if not_found:
            dynamic_idx_containing_this_output_tensor.append(-1)
            segment_idx_containing_this_output_tensor.append(-1)
            # print("output:", static_output.data_ptr(), static_output.nbytes)
            for i, (static_input_data_ptr, static_input_nbytes) in enumerate(static_inputs_only_tensors):
                # print("input:", static_input_data_ptr, static_input_nbytes)
                if (static_input_data_ptr <= static_output.data_ptr() and
                    static_output.data_ptr() + static_output.nbytes <= static_input_data_ptr + static_input_nbytes
                    ):
                    assert not static_output_idx in static_output_idx_to_input_idx_and_offset, "GALVEZ: static inputs should never share a buffer during stream capture!!!"
                    # TODO: We need to think about what to do when inputs are aliasing each other!
                    offset_from_input = static_output.data_ptr() - static_input_data_ptr
                    static_output_idx_to_input_idx_and_offset[static_output_idx] = (i, offset_from_input)
            # In this case, the output must be part of a
            # non-dynamic input tensor (which we should
            # verify!). In that situation, the output tensor
            # should always have the same output address across
            # runs.
            assert static_output_idx in static_output_idx_to_input_idx_and_offset, (static_output_idx, static_output.data_ptr(), static_output.nbytes)
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

    dynamic_tensors = list(static_inputs_only_dynamic_tensors) + static_output_segment_tensors

    find_overlaps(list(static_output_segment_tensors), True)
    print("GALVEZ total overlaps")
    find_overlaps(dynamic_tensors)

    graph.become_dynamic(dynamic_tensors)

    dynamic_input_idxs = OrderedSet([idx for idx in range(len(static_inputs)) if idx not in static_input_idxs and isinstance(static_inputs[idx], torch.Tensor)])

    def run(new_inputs):
        assert len(static_inputs) == len(new_inputs)

        dynamic_tensors = []

        for idx in dynamic_input_idxs:
            dynamic_tensors.append(new_inputs[idx])

        for segment_idx in containing_segment_idxs:
            containing_segment_size_bytes = segment_sizes[segment_idx]
            containing_segment_device = segment_devices[segment_idx]

            storage_tensor = torch.empty(containing_segment_size_bytes, dtype=torch.int8, device=containing_segment_device)
            # For debugging purposes only
            # storage_tensor[:] = 0
            dynamic_tensors.append(storage_tensor)

        graph.replay_dynamic(dynamic_tensors)

        outputs = []

        for i, static_output in enumerate(static_outputs):
            if i in non_tensor_output_idxs:
                outputs.append(static_output)
            dynamic_segment_idx = dynamic_idx_containing_this_output_tensor[i]
            containing_segment_idx = segment_idx_containing_this_output_tensor[i]
            if dynamic_segment_idx == -1:
                assert containing_segment_idx == -1
                input_idx, offset_from_input = static_output_idx_to_input_idx_and_offset[i]
                input_tensor = new_inputs[input_idx]
                true_output_tensor = torch.empty((), device=static_output.device, dtype=static_output.dtype)
                true_output_tensor.set_(input_tensor.untyped_storage(),
                                        storage_offset=offset_from_input,
                                        stride=static_output.stride(),
                                        size=static_output.size())
                outputs.append(true_output_tensor)
                continue
            storage_tensor = dynamic_tensors[len(dynamic_input_idxs) + dynamic_segment_idx]
            storage_offset = static_output.data_ptr() - segment_address_starts[containing_segment_idx]
            try:
                assert storage_offset < segment_sizes[containing_segment_idx]
            except AssertionError:
                import ipdb; ipdb.set_trace()
            assert static_output.nbytes <= segment_sizes[containing_segment_idx]
            assert storage_offset + static_output.nbytes <= segment_sizes[containing_segment_idx]
            true_output_tensor = torch.empty((), device=static_output.device, dtype=static_output.dtype)

            # This is the storage_offset in bytes, but set_ requires
            # an offset in terms of the dtype of the tensor
            assert storage_offset % static_output.itemsize == 0
            storage_offset_itemsize = storage_offset // static_output.itemsize

            # try:
            true_output_tensor.set_(storage_tensor.untyped_storage(),
                                    storage_offset=storage_offset_itemsize,
                                    stride=static_output.stride(),
                                    size=static_output.size())
            # except RuntimeError as e:
            #     # import ipdb; ipdb.set_trace()
            #     raise
            outputs.append(true_output_tensor)

        return outputs

    graph.debug_dump("cudagraph_edited.dot")
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

    replace_node_in_graph(graph, memcpy_node, new_node)

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
    # print("GALVEZ: memset params", memset_params)
    assert memset_params.pitch == 0
    assert memset_params.elementSize == 1
    
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
        graph, [], 0, kernel_params))

    replace_node_in_graph(graph, memset_node, new_node)

import cuda.bindings.runtime

def replace_node_in_graph(graph, old_node, new_node):
    """
    Replace a node in a CUDA graph by removing the old node and adding the new one.
    
    Args:
        graph (cudaGraph_t): The CUDA graph to modify
        old_node (cudaGraphNode_t): The node to remove from the graph
        new_node (cudaGraphNode_t): The node to add to the graph
    """
    # Get dependencies of the old node
    _, num_dependencies = cuda_python_error_check(cuda.bindings.runtime.cudaGraphNodeGetDependencies(old_node))
    if num_dependencies == 0:
        dependencies = []
    else:
        dependencies, _ = cuda_python_error_check(cuda.bindings.runtime.cudaGraphNodeGetDependencies(old_node, num_dependencies))
    
    # Get dependent nodes of the old node
    _, num_dependents = cuda_python_error_check(cuda.bindings.runtime.cudaGraphNodeGetDependentNodes(old_node))
    if num_dependents == 0:
        dependent_nodes = []
    else:
        dependent_nodes, _ = cuda_python_error_check(cuda.bindings.runtime.cudaGraphNodeGetDependentNodes(old_node, num_dependents))
    
    # Remove the old node from the graph
    cuda_python_error_check(cuda.bindings.runtime.cudaGraphDestroyNode(old_node))

    # TODO: call cudaGraphGetEdges_v2 to get the edge data. Right now, we are losing it
    
    # Add the new node to the graph with the same dependencies as the old node
    # import ipdb; ipdb.set_trace()
    if len(dependencies) > 0:
        cuda_python_error_check(cuda.bindings.runtime.cudaGraphAddDependencies(
            graph,
            dependencies,
            [new_node] * len(dependencies),
            len(dependencies)
        ))
    
    # Add the dependencies from the new node to the old node's dependent nodes
    if len(dependent_nodes) > 0:
        cuda_python_error_check(cuda.bindings.runtime.cudaGraphAddDependencies(
            graph,
            [new_node] * len(dependent_nodes),
            dependent_nodes,
            len(dependent_nodes)
        ))
