# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Optional, Sequence

# Import all builtin dist tensor ops
import torch
import torch.distributed._tensor.ops
from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.api import distribute_module, distribute_tensor, DTensor
from torch.distributed._tensor.device_mesh import DeviceMesh, mesh_resources
from torch.distributed._tensor.placement_types import Placement, Replicate, Shard

# All public APIs from dtensor package
__all__ = [
    "DTensor",
    "DeviceMesh",
    "distribute_tensor",
    "distribute_module",
    "Shard",
    "Replicate",
]


def _dtensor_init_helper(
    init_op,
    size: torch.Size,
    device_mesh=None,
    placements=None,
    **kwargs,
) -> DTensor:
    # if device_mesh is None, use the one from mesh resources
    device_mesh = device_mesh or mesh_resources.get_current_mesh()
    kwargs["device"] = device_mesh.device_type

    # set default placements to replicated if not specified
    placements = placements or [Replicate() for _ in range(device_mesh.ndim)]

    # check device_mesh againts placements
    assert device_mesh.ndim == len(
        placements
    ), "mesh dimension does not match the length of placements"

    assert kwargs["layout"] == torch.strided, "layout value not supported!"
    torch_stride = torch._prims_common.make_contiguous_strides_for(size)

    # get local tensor shape
    local_shape = compute_local_shape(size, device_mesh, placements)
    # initialize the local tensor
    if len(local_shape) == 0:
        local_tensor = torch.empty(0, **kwargs)
    elif init_op == torch.full:
        fill_value = kwargs.pop("fill_value", 0)
        local_tensor = init_op(local_shape, fill_value, **kwargs)
    else:
        local_tensor = init_op(local_shape, **kwargs)

    return DTensor(
        local_tensor=local_tensor,
        device_mesh=device_mesh,
        placements=placements,
        shape=size,
        dtype=local_tensor.dtype,
        stride=torch_stride,
        requires_grad=kwargs["requires_grad"],
    )


def _normalize_to_torch_size(size) -> torch.Size:
    # convert Union[Tuple[int], Tuple[Sequence[int]]] to torch.Size
    # normalize the size argument
    if len(size) == 1 and isinstance(size[0], Sequence):
        torch_size = size[0]
    else:
        torch_size = list(size)
    return torch.Size(torch_size)


def ones(
    *size,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    requires_grad: bool = False,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with the scalar value 1, with the shape defined
    by the variable argument ``size``.

    Args:
        size (int...): a sequence of integers defining the shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: ones(1,2,3..) or ones([1,2,3..]) or ones((1,2,3..))

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
        layout (:class:`torch.layout`, optional): the desired layout of returned DTensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``, ``_Partial``

    Returns:
        A :class:`DTensor` object on each rank
    """
    torch_size = _normalize_to_torch_size(size)

    return _dtensor_init_helper(
        torch.ones,
        torch_size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        device_mesh=device_mesh,
        placements=placements,
    )


def empty(
    *size,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    requires_grad: bool = False,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with uninitialized data. The shape of the :class:`DTensor`
    is defined by the variable argument ``size``.

    Args:
        size (int...): a sequence of integers defining the shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: empty(1,2,3..) or empty([1,2,3..]) or empty((1,2,3..))

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).\
        layout (:class:`torch.layout`, optional): the desired layout of returned :class:`DTensor`.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``, ``_Partial``

    Returns:
        A :class:`DTensor` object on each rank
    """
    torch_size = _normalize_to_torch_size(size)

    return _dtensor_init_helper(
        torch.empty,
        torch_size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        device_mesh=device_mesh,
        placements=placements,
    )


def full(
    size,
    fill_value,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    requires_grad: bool = False,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with ``fill_value``. The scalar value type should match
        ``device_mesh.device_type``.

    Args:
        size (int...): a sequence of integers defining the shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: ones(1,2,3..) or ones([1,2,3..]) or ones((1,2,3..))
        fill_value(Scalar): the value to fill the output tensor with.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
        layout (:class:`torch.layout`, optional): the desired layout of returned DTensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks.
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``, ``_Partial``.

    Returns:
        A :class:`DTensor` object on each rank
    """
    torch_size = _normalize_to_torch_size(size)

    return _dtensor_init_helper(
        torch.full,
        torch_size,
        fill_value=fill_value,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        device_mesh=device_mesh,
        placements=placements,
    )


def zeros(
    *size,
    requires_grad: bool = False,
    dtype: torch.dtype = None,
    layout: torch.layout = torch.strided,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with the scalar value 0.

    Args:
        size (int...): a sequence of integers defining the shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: zeros(1,2,3..) or zeros([1,2,3..]) or zeros((1,2,3..))
    Keyword args:
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
        layout (:class:`torch.layout`, optional): the desired layout of returned :class:`DTensor`.
            Default: ``torch.strided``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``, ``_Partial``

    Returns:
        A :class:`DTensor` object on each rank
    """
    torch_size = _normalize_to_torch_size(size)

    return _dtensor_init_helper(
        torch.zeros,
        torch_size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        device_mesh=device_mesh,
        placements=placements,
    )
