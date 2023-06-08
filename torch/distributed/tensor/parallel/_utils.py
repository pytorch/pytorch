import functools
from typing import Callable, Optional, Union

import torch
from torch.distributed._tensor import DeviceMesh, DTensor

_PrepareInputType = Callable[
    [Union[torch.Tensor, DTensor], Optional[DeviceMesh], Optional[int]], DTensor
]

_PrepareOutputType = Callable[
    [DTensor, Optional[DeviceMesh], Optional[int]], Union[torch.Tensor, DTensor]
]


def _prepare_input_validate(
    _prepare_input_func: _PrepareInputType,
) -> _PrepareInputType:
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
        >>> # xdoctest: +SKIP(failing)
        >>> @_prepare_input_validate
        >>> def make_input_shard_1d(args, kwargs):
        >>>   ...
        >>>
        >>> # xdoctest: +SKIP(failing)
        >>> input = torch.rand(...)
        >>> dtensor = make_input_shard_1d(input, device_mesh, 1)
        >>> # This will call '_prepare_input_validate' first
    """

    @functools.wraps(_prepare_input_func)
    def wrapper(*args, **kwargs):  # pyre-ignore[2, 3]
        assert len(args) >= 1, "_prepare_input needs at least one arg."
        input = args[0]
        if isinstance(input, (list, tuple)):
            input = input[0]
            args = (input, *args[1:])
        device_mesh = None if len(args) < 2 else args[1]

        if device_mesh is None:
            if isinstance(input, DTensor):
                device_mesh = input.device_mesh
                args = (*args[:1], device_mesh, *args[2:])  # pyre-ignore[60]
            else:
                raise RuntimeError("device_mesh is not passed nor can be inferred")
        if device_mesh.ndim != 1:
            raise RuntimeError(
                f"device_mesh has dims {device_mesh.ndim} but expected to be 1"
                " for input."
            )
        return _prepare_input_func(*args, **kwargs)

    return wrapper


def _prepare_output_validate(
    _prepare_output_func: _PrepareOutputType,
) -> _PrepareOutputType:
    """
    Inject common validation logics for _prepare_output funcs via this
    decorator, including verifying that output needs to be a DTensor
    and only 1D Device Mesh is passed in.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> @_prepare_output_validate
        >>> def make_output_shard_1d(args, kwargs):
        >>>   ...
        >>>
        >>> # xdoctest: +SKIP(failing)
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
        assert isinstance(output, DTensor), (
            "Expect output of Tensor Parallel to be a DTensor, but found"
            f" {type(output)}."
        )
        if len(args) < 2 or args[1] is None:
            device_mesh = output.device_mesh
            args = (*args[:1], device_mesh, *args[2:])  # pyre-ignore[60]
        else:
            device_mesh = args[1]

        assert device_mesh.ndim == 1, (
            f"device_mesh has dims {device_mesh.ndim} but expected to be 1 for"
            " output."
        )
        return _prepare_output_func(*args, **kwargs)

    return wrapper


def _create_1d_device_mesh(device_mesh: DeviceMesh, tp_mesh_dim: int = 0) -> DeviceMesh:
    """
    This function converts a N-D ``device_mesh`` into a 1D ``device_mesh``
    for 1D Tensor Parallelism.

    Args:
        device_mesh (DeviceMesh):
            :class:``DeviceMesh`` object which describes the mesh topology
            of devices for the DTensor.
        tp_mesh_dim (int):
            the dimension of ``device_mesh`` where we perform
            Tensor Parallelism on.

    Return:
        device_mesh (DeviceMesh): 1-D :class:``DeviceMesh`` object that
            Tensor Parallelism operates on.
    """
    assert tp_mesh_dim < device_mesh.ndim and tp_mesh_dim >= -device_mesh.ndim, (
        f"Expect tp_mesh_dim within range [{-device_mesh.ndim},"
        f" {device_mesh.ndim}), but found {tp_mesh_dim}."
    )

    if device_mesh.ndim == 1:
        return device_mesh

    # swap the current dim to the last dim then reshape to flatten out other
    # dims, so we can just extract the list of ranks which contains cur_rank.
    cur_rank = device_mesh.get_rank()
    pg_ranks_by_dim = device_mesh.mesh.swapdims(-1, tp_mesh_dim).reshape(
        -1, device_mesh.mesh.size(tp_mesh_dim)
    )
    for mesh_1d in pg_ranks_by_dim:
        sub_mesh = DeviceMesh(device_mesh.device_type, mesh_1d, _init_process_groups=False)
        if cur_rank in mesh_1d:
            res_sub_mesh = sub_mesh

    sub_pg = device_mesh.get_dim_groups()[tp_mesh_dim]
    res_sub_mesh._dim_groups = [sub_pg]
    return res_sub_mesh
