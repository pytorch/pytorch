import torch
import torch.distributed as dist
from torch.nn.parallel._functions import _get_stream
from torch.nn.parallel.scatter_gather import (  # type: ignore[attr-defined]
    _is_namedtuple
)
from typing import Any, Dict, List, Tuple

__all__ = []  # type: ignore[var-annotated]

def _pack_kwargs(*args: Any, **kwargs: Any) -> Tuple[Tuple[Any, ...], Tuple[str, ...]]:
    """
    Turn argument list into separate key list and value list (unpack_kwargs does the opposite)
    Inspiration: https://github.com/facebookresearch/fairscale/blob/eeb6684/fairscale/internal/containers.py#L70
    Usage::

        kwarg_keys, flat_args = pack_kwargs(1, 2, a=3, b=4)
        assert kwarg_keys == ("a", "b")
        assert flat_args == (1, 2, 3, 4)
        args, kwargs = unpack_kwargs(kwarg_keys, flat_args)
        assert args == (1, 2)
        assert kwargs == {"a": 3, "b": 4}
    Returns:
        Tuple[Tuple[Any, ...], Tuple[str, ...]]: The first tuple element gives
        gives both positional args and kwarg values, where the positional args
        proceed kwarg values and kwarg values are ordered consistently with the
        kwarg keys. The second tuple element gives the kwarg keys.
        The second tuple element's length is at most the first tuple element's length.
    """
    kwarg_keys: List[str] = []
    flat_args: List[Any] = list(args)
    for k, v in kwargs.items():
        kwarg_keys.append(k)
        flat_args.append(v)

    return tuple(flat_args), tuple(kwarg_keys)


def _unpack_kwargs(flat_args: Tuple[Any, ...], kwarg_keys: Tuple[str, ...]) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """See _pack_kwargs."""
    assert len(kwarg_keys) <= len(flat_args), f"too many keys {len(kwarg_keys)} vs. {len(flat_args)}"
    if len(kwarg_keys) == 0:
        return flat_args, {}
    args = flat_args[: -len(kwarg_keys)]
    kwargs = {k: v for k, v in zip(kwarg_keys, flat_args[-len(kwarg_keys) :])}
    return args, kwargs

def _recursive_to(inputs, target_gpu, use_side_stream_for_tensor_copies):
    r"""
    Recursively moves input to the target_gpu.
    """

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
                    output.record_stream(current_stream)  # type: ignore[arg-type]
                return (output,)
        if _is_namedtuple(obj):
            return [type(obj)(*args) for args in zip(*map(to_map, obj))]
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(to_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return [list(i) for i in zip(*map(to_map, obj))]
        if isinstance(obj, dict) and len(obj) > 0:
            return [type(obj)(i) for i in zip(*map(to_map, obj.items()))]
        return [obj]

    # Avoid reference cycle
    try:
        res = to_map(inputs)
    finally:
        to_map = None  # type: ignore[assignment]
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

def _verify_param_shape_across_processes(process_group, tensors, logger=None):
    return dist._verify_params_across_processes(process_group, tensors, logger)

def _sync_module_states(
    module,
    process_group,
    broadcast_bucket_size,
    src,
    params_and_buffers_to_ignore,
):
    """
    Syncs ``module``'s parameters and buffers state so that all ranks contain
    the same module state across all ranks. Note that this API assumes that all
    parameter shapes are consistent before running the synchronization. This can
    be checked with ``_verify_param_shape_across_processes``.
    """
    module_states = []
    for name, param in module.named_parameters():
        if name not in params_and_buffers_to_ignore:
            module_states.append(param.detach())

    for name, buffer in module.named_buffers():
        if name not in params_and_buffers_to_ignore:
            module_states.append(buffer.detach())

    _sync_params_and_buffers(
        process_group,
        module_states,
        broadcast_bucket_size,
        src
    )

def _sync_params_and_buffers(
    process_group: dist.ProcessGroup,
    module_states: List[torch.Tensor],
    broadcast_bucket_size: int,
    src: int,
):
    """
    Synchronizes ``module_states`` (list of tensors) across all processes by
    broadcasting them from rank 0.
    """
    if len(module_states) > 0:
        dist._broadcast_coalesced(
            process_group, module_states, broadcast_bucket_size, src
        )

def _replace_by_prefix(
    state_dict: Dict[str, Any],
    old_prefix: str,
    new_prefix: str,
) -> None:
    """
    Replace all keys that match a given old_prefix with a new_prefix (in-place).

    Usage::

        state_dict = {"layer.xyz": torch.tensor(1)}
        replace_by_prefix_(state_dict, "layer.", "module.layer.")
        assert state_dict == {"module.layer.xyz": torch.tensor(1)}
    """
    if old_prefix == new_prefix:
        raise ValueError("old_prefix and new_prefix must be distinct")
    for key in list(state_dict.keys()):
        if not key.startswith(old_prefix):
            continue
        new_key = new_prefix + key[len(old_prefix) :]
        state_dict[new_key] = state_dict[key]
        del state_dict[key]
