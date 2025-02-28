# mypy: allow-untyped-defs
import dataclasses
import traceback
from collections import OrderedDict
from collections.abc import Container
from typing import Any, Callable, Optional, overload, TypeVar

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.utils.rnn import PackedSequence


__all__ = []  # type: ignore[var-annotated]


def _pack_kwargs(*args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], tuple[str, ...]]:
    """
    Turn argument list into separate key list and value list (unpack_kwargs does the opposite).

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
    kwarg_keys: list[str] = []
    flat_args: list[Any] = list(args)
    for k, v in kwargs.items():
        kwarg_keys.append(k)
        flat_args.append(v)

    return tuple(flat_args), tuple(kwarg_keys)


def _cast_forward_inputs(
    dtype: Optional[torch.dtype],
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """
    Cast floating point tensors in ``args`` and ``kwargs`` to ``input_dtype``.

    This respects the existing ``requires_grad`` on the tensors.
    """
    if dtype is None:
        return args, kwargs

    def cast_fn(x: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(x) or x.dtype == dtype:
            return x
        return x.to(dtype)

    return (_apply_to_tensors(cast_fn, args), _apply_to_tensors(cast_fn, kwargs))


def _unpack_kwargs(
    flat_args: tuple[Any, ...], kwarg_keys: tuple[str, ...]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """See _pack_kwargs."""
    assert len(kwarg_keys) <= len(flat_args), (
        f"too many keys {len(kwarg_keys)} vs. {len(flat_args)}"
    )
    if len(kwarg_keys) == 0:
        return flat_args, {}
    args = flat_args[: -len(kwarg_keys)]
    kwargs = dict(zip(kwarg_keys, flat_args[-len(kwarg_keys) :]))
    return args, kwargs


S = TypeVar("S", dict, list, tuple)
T = TypeVar("T", torch.Tensor, PackedSequence)


@overload
def _recursive_to(
    inputs: S, target_device: torch.device, use_side_stream_for_tensor_copies: bool
) -> list[S]: ...


@overload
def _recursive_to(
    inputs: T, target_device: torch.device, use_side_stream_for_tensor_copies: bool
) -> tuple[T]: ...


def _recursive_to(inputs, target_device, use_side_stream_for_tensor_copies):
    r"""Recursively moves input to the target_device."""

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

                from torch.nn.parallel._functions import _get_stream

                # Perform CPU -> target_device copies in a background stream. This code is
                # motivated from similar logic in torch/nn/parallel/_functions.py
                stream = _get_stream(target_device)
                with device_mod.stream(stream):
                    output = obj.to(target_device)
                # synchronize with the copy stream
                with device_mod.device(target_device.index):
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

        from torch.nn.parallel.scatter_gather import _is_namedtuple

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
    """Alternate to ``assert`` when in the backward context to print the error message ``s`` since otherwise, it is swallowed."""
    if not cond:
        print(s)
        traceback.print_stack()
        if raise_assertion_error:
            raise AssertionError(s)


def _alloc_storage(tensor: torch.Tensor, size: torch.Size) -> None:
    """
    Allocate storage for ``tensor`` with the given size.

    Returns:
        bool: ``True`` if this method allocated storage and ``False`` if the
        storage was already allocated.
    """
    with torch.no_grad():
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            already_allocated = tensor._typed_storage()._size() == size.numel()
            if not already_allocated:
                tensor_storage_size = tensor._typed_storage()._size()
                _p_assert(
                    tensor_storage_size == 0,
                    "Tensor storage should have been resized to be 0 but got PLACEHOLDEr",
                )
                tensor._typed_storage()._resize_(size.numel())


def _free_storage(tensor: torch.Tensor):
    """
    Frees the underlying storage of ``tensor``.

    Returns:
        bool: ``True`` if the method freed the storage and ``False`` if the
        storage was already freed.
    """
    with torch.no_grad():
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
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


Q = TypeVar("Q")
R = TypeVar("R", dict, list, tuple, set, OrderedDict, PackedSequence, Any)


@overload
def _apply_to_tensors(
    fn: Callable[[torch.Tensor], Q], container: torch.Tensor
) -> Q: ...


@overload
def _apply_to_tensors(fn: Callable[[torch.Tensor], Any], container: R) -> R: ...


def _apply_to_tensors(fn, container):
    """Recursively apply to all tensor in different kinds of container types."""

    def apply(x):
        from torch.nn.parallel.scatter_gather import _is_namedtuple

        if isinstance(x, torch.Tensor):
            return fn(x)
        elif hasattr(x, "__dataclass_fields__"):
            dc = dataclasses.replace(x)
            changes = {
                f.name: apply(getattr(dc, f.name)) for f in dataclasses.fields(dc)
            }
            return dataclasses.replace(dc, **changes)
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


def _to_kwargs(
    inputs: tuple[Any, ...],
    kwargs: Optional[dict[str, Any]],
    target_device: torch.device,
    use_side_stream_for_tensor_copies: bool,
) -> tuple[tuple[Any, ...], tuple[dict[str, Any], ...]]:
    moved_inputs = (
        _recursive_to(inputs, target_device, use_side_stream_for_tensor_copies)
        if inputs
        else []
    )
    moved_kwargs = (
        _recursive_to(kwargs, target_device, use_side_stream_for_tensor_copies)
        if kwargs
        else []
    )
    if len(moved_inputs) < len(moved_kwargs):
        moved_inputs.extend([() for _ in range(len(moved_kwargs) - len(inputs))])
    elif len(moved_kwargs) < len(moved_inputs):
        moved_kwargs.extend([{} for _ in range(len(moved_inputs) - len(moved_kwargs))])
    return tuple(moved_inputs), tuple(moved_kwargs)


def _verify_param_shape_across_processes(
    process_group: dist.ProcessGroup,
    tensors: list[torch.Tensor],
    logger: Optional["dist.Logger"] = None,
):
    return dist._verify_params_across_processes(process_group, tensors, logger)


def _sync_module_states(
    module: nn.Module,
    process_group: dist.ProcessGroup,
    broadcast_bucket_size: int,
    src: int,
    params_and_buffers_to_ignore: Container[str],
    broadcast_buffers: bool = True,
) -> None:
    """
    Sync ``module``'s parameters and buffers state.

    Syncs ``module``'s parameters and buffers state so that all ranks contain
    the same module state across all ranks. Note that this API assumes that all
    parameter shapes are consistent before running the synchronization. This can
    be checked with ``_verify_param_shape_across_processes``.
    """
    module_states: list[torch.Tensor] = []
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
    module_states: list[torch.Tensor],
    broadcast_bucket_size: int,
    src: int,
) -> None:
    """Synchronize ``module_states`` (list of tensors) across all processes by broadcasting them from rank 0."""
    if len(module_states) > 0:
        dist._broadcast_coalesced(
            process_group, module_states, broadcast_bucket_size, src
        )


def _replace_by_prefix(
    state_dict: dict[str, Any],
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


def _data_ptr_allocated(tensor: torch.Tensor) -> bool:
    return tensor.untyped_storage().data_ptr() > 0


def _get_root_modules(modules: list[nn.Module]) -> list[nn.Module]:
    """
    Returns the modules in ``modules`` that are root modules (i.e.
    parent-less) with respect to the set ``modules``. In other words, these
    are the modules in ``modules`` that are the not child of any other
    module in ``modules``.
    """
    root_modules: list[nn.Module] = []
    module_to_modules: dict[nn.Module, set[nn.Module]] = {
        module: set(module.modules()) for module in modules
    }
    for candidate_module in modules:
        is_root_module = True
        for module, _modules in module_to_modules.items():
            is_child_module = (
                candidate_module is not module and candidate_module in _modules
            )
            if is_child_module:
                is_root_module = False
                break
        if is_root_module:
            root_modules.append(candidate_module)
    return root_modules
