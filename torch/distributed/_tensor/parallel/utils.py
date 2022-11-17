import functools

import torch
from torch.distributed._tensor import DeviceMesh, DTensor
from typing import Callable, Optional, Union

_Prepare_Input_Func_Type = Callable[
    [Union[torch.Tensor, DTensor], Optional[DeviceMesh], Optional[int]],
    DTensor,
]

_Prepare_Output_Func_Type = Callable[
    [DTensor, Optional[DeviceMesh], Optional[int]], Union[torch.Tensor, DTensor]
]


def _prepare_input_validate(
    _prepare_input_func: _Prepare_Input_Func_Type,
) -> _Prepare_Input_Func_Type:
    """
    Inject common validation logics for `_prepare_input` funcs via this
    decorator, including verifying that input needs to be either
    a :class:`Tensor` or :class:`DTensor` and only 1D :class:`DeviceMesh`
    is passed in.

    Args:
        _prepare_input_func (Callable): The func we want to inject the
            validation into.

    Returns:
        func (Callable): Same input function with validation logic added.

    Example::
        >>> @_prepare_input_validate
        >>> def make_input_shard_1d(args, kwargs):
        >>>   ...
        >>>
        >>> input = torch.rand(...)
        >>> dtensor = make_input_shard_1d(input, device_mesh, 1)
        >>> # This will call '_prepare_input_validate' first
    """

    @functools.wraps(_prepare_input_func)
    def wrapper(*args, **kwargs):  # pyre-ignore[2, 3]
        assert len(args) >= 1, "_prepare_input needs at least one arg."
        input = args[0]
        device_mesh = None if len(args) < 2 else args[1]

        if device_mesh is None:
            if isinstance(input, DTensor):
                device_mesh = input.device_mesh
                args = (*args[:1], device_mesh, *args[2:])  # pyre-ignore[60]
            else:
                raise RuntimeError(
                    "device_mesh is not passed nor can be inferred"
                )
        if device_mesh.ndim != 1:
            raise RuntimeError(
                f"device_mesh has dims {device_mesh.ndim} but expcted to be 1 for input."
            )
        return _prepare_input_func(*args, **kwargs)

    return wrapper


def _prepare_output_validate(
    _prepare_output_func: _Prepare_Output_Func_Type,
) -> _Prepare_Output_Func_Type:
    """
    Inject common validation logics for _prepare_output funcs via this
    decorator, including verifying that output needs to be a DTensor
    and only 1D Device Mesh is passed in.
    Example::
        >>> @_prepare_output_validate
        >>> def make_output_shard_1d(args, kwargs):
        >>>   ...
        >>>
        >>> dt = distribute(tensor, device_mesh, [Shard(0)])
        >>> make_output_shard_1d(dt, device_mesh, 1)
        >>> # This will call '_prepare_output_validate' first
    Args:
        _prepare_output_func (Callable): The func we want to inject the
            validation into.
    Return:
        func (Callable): Same input func with validation logic added.
    """

    @functools.wraps(_prepare_output_func)
    def wrapper(*args, **kwargs):  # pyre-ignore[2, 3]
        assert len(args) >= 1, "_prepare_output needs at least one arg."
        output = args[0]
        assert isinstance(
            output, DTensor
        ), f"Expect output of Tensor Parallel to be a DTensor, but found {type(output)}."
        if len(args) < 2 or args[1] is None:
            device_mesh = output.device_mesh
            args = (*args[:1], device_mesh, *args[2:])  # pyre-ignore[60]
        else:
            device_mesh = args[1]

        assert (
            device_mesh.ndim == 1
        ), f"device_mesh has dims {device_mesh.ndim} but expcted to be 1 for output."
        return _prepare_output_func(*args, **kwargs)

    return wrapper
