from __future__ import annotations

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
    pool = torch.cuda.MemPool(mem_allocator)

    torch.cuda.memory._record_memory_history(True)

    with torch.cuda.use_mem_pool(pool):
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

    print("GALVEZ: input + output memory snapshots:")
    memory_snapshot = torch.cuda.memory_snapshot(pool.id)
    import pprint
    pprint.pprint(memory_snapshot)

    with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local",
                          dynamic_graph=dynamic_graph_arguments, pool=pool.id):
        # Can't I get the underlying fxgraph here? No.
        static_outputs = model(list(static_inputs))

    print("GALVEZ: input + output memory snapshots:")
    memory_snapshot = torch.cuda.memory_snapshot(pool.id)
    import pprint
    pprint.pprint(memory_snapshot)

    # import ipdb; ipdb.set_trace()

    # TODO: Why this? I should really be iterating over the outptuts as if they are a pytree...
    if not isinstance(static_outputs, (list, tuple)):
        static_outputs = (static_outputs,)

    memory_snapshot = torch.cuda.memory_snapshot(graph.pool())
    memory_snapshot.sort(key=lambda x: x['address'])

    segment_address_starts = [segment_snapshot['address'] for segment_snapshot in memory_snapshot]
    segment_sizes = [segment_snapshot['total_size'] for segment_snapshot in memory_snapshot]
    segment_devices = [torch.device("cuda", segment_snapshot['device']) for segment_snapshot in memory_snapshot]

    containing_segment_idxs = OrderedSet()
    segment_idx_containing_this_output_tensor = []
    dynamic_idx_containing_this_output_tensor = []

    # I need to map each segment index to all output tensors, I think.
    for static_output in static_outputs:
        segment_idx = bisect.bisect(segment_address_starts, static_output.data_ptr()) - 1
        if segment_idx in containing_segment_idxs:
            for i, containing_segment_idx in enumerate(containing_segment_idxs):
                if segment_idx == containing_segment_idx:
                    dynamic_idx_containing_this_output_tensor.append(i)
                    break
        else:
            dynamic_idx_containing_this_output_tensor.append(len(containing_segment_idxs))
        containing_segment_idxs.add(segment_idx)
        segment_idx_containing_this_output_tensor.append(segment_idx)

    # assert len(containing_segment_idxs) == len(set(containing_segment_idxs)), "GALVEZ: detected output pointers sharing a segment!"

    static_output_segment_tensors = []

    for segment_idx in containing_segment_idxs:
        static_output_segment_tensors.append((segment_address_starts[segment_idx], segment_sizes[segment_idx]))

    torch.cuda.synchronize()

    # does this include tensors? I think so...
    static_inputs_only_tensors = [(input.data_ptr(), input.nbytes) for idx, input in enumerate(static_inputs) if idx not in static_input_idxs and isinstance(input, torch.Tensor) and input.is_cuda]
    dynamic_tensors = list(static_inputs_only_tensors) + static_output_segment_tensors
    graph.become_dynamic(dynamic_tensors)

    dynamic_input_idxs = OrderedSet([idx for idx in range(len(static_inputs)) if idx not in static_input_idxs and isinstance(static_inputs[idx], torch.Tensor)])

    def run(new_inputs):
        assert len(static_inputs) == len(new_inputs)

        dynamic_tensors = []

        for idx in dynamic_input_idxs:
            dynamic_tensors.append(new_inputs[idx])

        torch.cuda.nvtx.range_push("Dynamic outputs creation")
        for segment_idx in containing_segment_idxs:
            containing_segment_size_bytes = segment_sizes[segment_idx]
            containing_segment_device = segment_devices[segment_idx]

            storage_tensor = torch.empty(containing_segment_size_bytes, dtype=torch.int8, device=containing_segment_device)
            dynamic_tensors.append(storage_tensor)
        torch.cuda.nvtx.range_pop()

        graph.replay_dynamic(dynamic_tensors)

        output_tensors = []

        for i, static_output in enumerate(static_outputs):
            dynamic_segment_idx = dynamic_idx_containing_this_output_tensor[i]
            containing_segment_idx = segment_idx_containing_this_output_tensor[i]
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
