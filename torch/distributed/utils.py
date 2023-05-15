import dataclasses
import traceback
from typing import Any, Callable, Dict, List, OrderedDict, Set, Tuple, Union

import torch
import torch.distributed as dist
from torch.nn.parallel._functions import _get_stream
from torch.nn.parallel.scatter_gather import (  # type: ignore[attr-defined]
    _is_namedtuple,
)
from torch.nn.utils.rnn import PackedSequence

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


def _unpack_kwargs(
    flat_args: Tuple[Any, ...], kwarg_keys: Tuple[str, ...]
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """See _pack_kwargs."""
    assert len(kwarg_keys) <= len(
        flat_args
    ), f"too many keys {len(kwarg_keys)} vs. {len(flat_args)}"
    if len(kwarg_keys) == 0:
        return flat_args, {}
    args = flat_args[: -len(kwarg_keys)]
    kwargs = dict(zip(kwarg_keys, flat_args[-len(kwarg_keys) :]))
    return args, kwargs


def _recursive_to(inputs, target_device, use_side_stream_for_tensor_copies):
    r"""
    Recursively moves input to the target_device.
    """

    def to_map(obj):
        if isinstance(obj, (torch.Tensor, PackedSequence)):
            device = obj.data.device if isinstance(obj, PackedSequence) else obj.device
            if device == target_device:
                return (obj,)
            if not use_side_stream_for_tensor_copies:
                return (obj.to(target_device),)
            else:
                # If the custom module is not registered to torch, stream is not used for acceleration
                device_mod = getattr(torch, device.type, None)
                if device.type == "cpu" or device_mod is None:
                    return (obj.to(target_device),)
                # Perform CPU -> target_device copies in a background stream. This code is
                # motivated from similar logic in torch/nn/parallel/_functions.py
                stream = _get_stream(target_device)
                with device_mod.stream(stream):
                    output = obj.to(target_device)
                # synchronize with the copy stream
                with torch.cuda.device(target_device.index):
                    current_stream = device_mod.current_stream()
                    # Sync the current stream with the copy stream
                    current_stream.wait_stream(stream)
                    # Ensure tensor memory is not reused until work on
                    # main stream is complete
                    if isinstance(obj, PackedSequence):
                        output.data.record_stream(current_stream)  # type: ignore[arg-type]
                    else:
                        assert isinstance(output, torch.Tensor)
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


def _p_assert(cond: Any, s: str, raise_assertion_error: bool = True) -> None:
    """This is used as an alternate to ``assert`` when in the backward context
    to print the error message ``s`` since otherwise, it is swallowed."""
    if not cond:
        print(s)
        traceback.print_stack()
        if raise_assertion_error:
            raise AssertionError(s)


def _alloc_storage(tensor: torch.Tensor, size: torch.Size) -> bool:
    """
    Allocate storage for ``tensor`` with the given size.

    Returns:
        bool: ``True`` if this method allocated storage and ``False`` if the
        storage was already allocated.
    """
    with torch.no_grad():
        already_allocated = tensor._typed_storage()._size() == size.numel()
        if not already_allocated:
            tensor_storage_size = tensor._typed_storage()._size()
            _p_assert(
                tensor_storage_size == 0,
                f"Tensor storage should have been resized to be 0 but got {tensor_storage_size}",
            )
            tensor._typed_storage()._resize_(size.numel())
        return not already_allocated


def _free_storage(tensor: torch.Tensor) -> bool:
    """
    Frees the underlying storage of ``tensor``.

    Returns:
        bool: ``True`` if the method freed the storage and ``False`` if the
        storage was already freed.
    """
    with torch.no_grad():
        already_freed = tensor._typed_storage()._size() == 0
        if not already_freed:
            _p_assert(
                tensor.storage_offset() == 0,
                "Freeing a tensor's storage is unsafe when it is not the sole occupant\n"
                f"storage offset: {tensor.storage_offset()}\n"
                f"storage size: {tensor._typed_storage()._size()}\n"
                f"tensor shape: {tensor.shape}",
            )
            tensor._typed_storage()._resize_(0)
        return not already_freed


def _apply_to_tensors(
    fn: Callable,
    container: Union[torch.Tensor, Dict, List, Tuple, Set, OrderedDict, PackedSequence],
) -> Any:
    """Recursively apply to all tensor in different kinds of container types."""

    def apply(
        x: Union[torch.Tensor, Dict, List, Tuple, Set, OrderedDict, PackedSequence]
    ) -> Any:
        if torch.is_tensor(x):
            return fn(x)
        elif hasattr(x, "__dataclass_fields__"):
            dc = dataclasses.replace(x)
            for f in dataclasses.fields(dc):
                name = f.name
                setattr(dc, name, apply(getattr(dc, name)))
            return dc
        elif isinstance(x, OrderedDict):
            od = x.__class__()
            for key, value in x.items():
                od[key] = apply(value)
            return od
        elif isinstance(x, PackedSequence):
            apply(x.data)
            return x
        elif isinstance(x, dict):
            return {key: apply(value) for key, value in x.items()}
        elif _is_namedtuple(x):
            res = (apply(el) for el in x)
            return type(x)(*res)
        elif isinstance(x, (list, tuple, set)):
            return type(x)(apply(el) for el in x)
        else:
            return x

    return apply(container)


def _to_kwargs(inputs, kwargs, target_device, use_side_stream_for_tensor_copies):
    inputs = (
        _recursive_to(inputs, target_device, use_side_stream_for_tensor_copies)
        if inputs
        else []
    )
    kwargs = (
        _recursive_to(kwargs, target_device, use_side_stream_for_tensor_copies)
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
    broadcast_buffers=True,
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

    if broadcast_buffers:
        for name, buffer in module.named_buffers():
            if name not in params_and_buffers_to_ignore:
                module_states.append(buffer.detach())

    _sync_params_and_buffers(process_group, module_states, broadcast_bucket_size, src)


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
