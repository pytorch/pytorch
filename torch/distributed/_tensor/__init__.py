# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Optional, Sequence

# Import all builtin dist tensor ops
import torch
import torch.distributed._tensor.ops
from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.api import distribute_module, distribute_tensor, DTensor
from torch.distributed._tensor.device_mesh import DeviceMesh, get_global_device_mesh
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


def _dtensor_init_helper(init_op, *size, **kwargs) -> DTensor:
    # get default device mesh if there's nothing specified
    device_mesh = kwargs.pop("device_mesh", None)
    if device_mesh is None:
        device_mesh = get_global_device_mesh()

    # set default placements to replicated if not specified
    placements = kwargs.pop("placements", None)
    if placements is None:
        placements = [Replicate() for _ in range(device_mesh.ndim)]

    # check device_mesh againts placements
    assert device_mesh.ndim == len(
        placements
    ), "mesh dimension does not match the length of placements"

    # normalize the size argument
    if len(size) == 1 and isinstance(size[0], Sequence):
        torch_size = size[0]
    else:
        torch_size = list(size)
    torch_size = torch.Size(torch_size)
    assert kwargs["layout"] == torch.strided, "layout value not supported!"
    torch_stride = torch._prims_common.make_contiguous_strides_for(torch_size)

    # get local tensor shape
    local_shape = compute_local_shape(torch_size, device_mesh, placements)
    # initialize the local tensor
    if len(local_shape) == 0:
        local_tensor = torch.empty(0, **kwargs)
    else:
        local_tensor = init_op(local_shape, **kwargs)

    return DTensor(
        local_tensor=local_tensor,
        device_mesh=device_mesh,
        placements=placements,
        shape=torch_size,
        dtype=local_tensor.dtype,
        stride=torch_stride,
        requires_grad=kwargs["requires_grad"],
    )


def ones(
    *size,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    layout: torch.layout = torch.strided,
    requires_grad: bool = False,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with the scalar value 1, with the shape defined
    by the variable argument ``size``.

    Args:
        size (int...): a sequence of integers defining the shape of the output DTensor.
        Can be a variable number of arguments or a collection like a list or tuple.
        E.g.: ones(1,2,3..) or ones([1,2,3..]) or ones((1,2,3..))

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned DTensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
        device (:class:`torch.device`, optional): the desired device of returned DTensor.
            Default: if ``None``, uses the current device for the default tensor type
            (see :func:`torch.set_default_tensor_type`). device will be the CPU for CPU
            tensor types and the current CUDA device for CUDA tensor types.
        layout (:class:`torch.layout`, optional): the desired layout of returned DTensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks
        placement: a sequence of :class:`Placement` type: Shard, Replicate, _Partial

    Returns:
        A :class:`DTensor` object on each rank
    """
    return _dtensor_init_helper(
        torch.ones,
        *size,
        dtype=dtype,
        device=device,
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
        size (int...): a sequence of integers defining the shape of the output
            Dtensor. Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: zeros(1,2,3..) or zeros([1,2,3..]) or zeros((1,2,3..))
    Keyword args:
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks
        placement: a sequence of :class:`Placement` type: Shard, Replicate, _Partial

    Returns:
        A :class:`DTensor` object on each rank
    """
    return _dtensor_init_helper(
        torch.zeros,
        *size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        device_mesh=device_mesh,
        placements=placements,
    )
