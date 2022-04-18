import collections

import torch
import torch.distributed as dist



def _verify_param_shape_across_processes(process_group, tensors, logger=None):
    return dist._verify_params_across_processes(process_group, tensors, logger)


def _recursive_to(inputs, target_gpu, use_side_stream_for_tensor_copies):
    r"""
    Recursively moves input to the target_gpu.
    """
    # Define within to avoid ciruclar import
    from ._functions import _get_stream
    from .scatter_gather import is_namedtuple
    def to_map(obj):
        if isinstance(obj, torch.Tensor):
            if obj.device == torch.device("cuda", target_gpu):
                return (obj,)
            if not use_side_stream_for_tensor_copies:
                return (obj.to(target_gpu),)
            else:
                # Perform CPU -> GPU copies in a background stream. This code is
                # motivated from similar logic in torch/nn/parallel/_functions.py
                stream = _get_stream(target_gpu)
                with torch.cuda.stream(stream):
                    output = obj.to(target_gpu)
                # synchronize with the copy stream
                with torch.cuda.device(target_gpu):
                    current_stream = torch.cuda.current_stream()
                    # Sync the current stream with the copy stream
                    current_stream.wait_stream(stream)
                    # Ensure tensor memory is not reused until work on
                    # main stream is complete
                    output.record_stream(current_stream)
                return (output,)
        if is_namedtuple(obj):
            return [type(obj)(*args) for args in zip(*map(to_map, obj))]
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(to_map, obj)))
        if isinstance(obj, str):
            # Needs to be checked, otherwise it's taken as a sequence infinitely.
            # This is because the elements of a string are also strings, and so on.
            return [obj]
        if isinstance(obj, collections.abc.Sequence) and len(obj) > 0:
            try:
                return [type(obj)(i) for i in zip(*map(to_map, obj))]
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [list(i) for i in zip(*map(to_map, obj))]
        if isinstance(obj, collections.abc.Mapping) and len(obj) > 0:
            try:
                return [type(obj)(i) for i in zip(*map(to_map, obj.items()))]
            except TypeError:
                # The mapping type may not support `__init__(iterable)`.
                return [dict(i) for i in zip(*map(to_map, obj.items()))]
        return [obj]

    # Avoid reference cycle
    try:
        res = to_map(inputs)
    finally:
        to_map = None
    return res


def _to_kwargs(inputs, kwargs, device_id, use_side_stream_for_tensor_copies):
    inputs = (
        _recursive_to(inputs, device_id, use_side_stream_for_tensor_copies)
        if inputs
        else []
    )
    kwargs = (
        _recursive_to(kwargs, device_id, use_side_stream_for_tensor_copies)
        if kwargs
        else []
    )
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


def _sync_params_and_buffers(
    module,
    process_group,
    broadcast_bucket_size,
    rank,
    params_and_buffers_to_ignore,
):
    """
    Syncs ``module``'s parameters and buffers state so that all ranks contain
    the same module state across all ranks. Note that this API assumes that all
    parameter shapes are consistent before running the synchronization. This can
    be checked with ``verify_param_shape_across_processes``.
    """
    module_states = []
    for name, param in module.named_parameters():
        if name not in params_and_buffers_to_ignore:
            module_states.append(param.detach())

    for name, buffer in module.named_buffers():
        if name not in params_and_buffers_to_ignore:
            module_states.append(buffer.detach())

    if len(module_states) > 0:
        dist._broadcast_coalesced(
            process_group, module_states, broadcast_bucket_size, rank
        )
