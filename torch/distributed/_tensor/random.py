# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from torch import Tensor

from torch.distributed._tensor import DeviceMesh


def set_rng_state(new_state: Tensor, device_mesh: DeviceMesh) -> None:
    """Sets the random number generator state of the specified device mesh.

    Args:
        new_state (torch.ByteTensor): The desired state.
        device_mesh (:class:`DeviceMesh`): The device mesh to set the RNG state.

    Returns:
        None

    .. warning::
        Current implementation only supports a GPU device mesh.
        If ``device_mesh`` is a sub-mesh and the calling rank is not a part of it,
        `set_rng_state` will not set its GPU device's generator state.
    """
    if device_mesh is not None and device_mesh.get_coordinate() is not None:
        # the current rank is in mesh
        if device_mesh.device_type == "cuda":
            torch.cuda.set_rng_state(new_state)
        else:
            raise NotImplementedError(
                f"DTensor randomness only supports cuda device type, but got {device_mesh.device_type}"
            )


def get_rng_state(device_mesh: DeviceMesh) -> Tensor:
    """Returns the random number generator state of the calling rank as a ByteTensor.

    Args:
        device_mesh (:class:`DeviceMesh`): The device mesh to return the RNG state of.

    Returns:
        A :class:`Tensor` object that contains the random number generator state.

    .. warning::
        Current implementation only supports a GPU device mesh.
        If ``device_mesh`` is a sub-mesh and the calling rank is not a part of it,
        `get_rng_state` still returns its GPU device's generator state.
    """
    assert (
        device_mesh is not None
    ), f"expect a DeviceMesh parameter but {device_mesh} was passed in."

    if device_mesh.device_type == "cuda":
        return torch.cuda.get_rng_state()
    else:
        raise NotImplementedError(
            f"DTensor randomness only supports cuda device type, but got {device_mesh.device_type}"
        )


def manual_seed(seed: int, device_mesh: DeviceMesh) -> None:
    """Sets the seed for generating random numbers for the calling rank.

    Args:
        seed (int): The desired seed.
        device_mesh (:class:`DeviceMesh`): The device mesh to set the seed.

    Returns:
        None

    .. warning::
        When calling this function, :func:`manual_seed` must be called from all ranks of the
        default `ProcessGroup` even if some ranks may not be a part of the `device_mesh`,
        with the same `seed` value.
        If ``device_mesh`` is a sub-mesh and the calling rank is not a part of it,
        `manual_seed` will not set its GPU device's generator seed.
        Current implementation only supports a GPU device mesh.
    """
    assert (
        device_mesh is not None
    ), f"expect a DeviceMesh parameter but {device_mesh} was passed in."

    import torch.distributed as dist

    # broadcast the seed from rank 0 over the default PG
    object_list = [seed] * dist.get_world_size()
    dist.all_gather_object(object_list, seed)
    for rank, object in enumerate(object_list):
        if seed != int(object):
            raise RuntimeError(
                f"calling manual_seed function over {device_mesh} but received different seed values on ranks:",
                f"seed on rank {dist.get_rank()} is {seed}, and seed on rank {rank} is {object}!",
            )

    # the current rank is in mesh
    if device_mesh.get_coordinate() is not None:
        if device_mesh.device_type == "cuda":
            torch.cuda.manual_seed(seed)
        else:
            raise NotImplementedError(
                f"DTensor randomness only supports cuda device type, but got {device_mesh.device_type}"
            )


def _set_offset(new_offset: int, device_mesh: DeviceMesh) -> None:
    """Sets the random number generator state offset for the calling rank.

    Args:
        new_offset (int): The desired random number generator state offset.
        device_mesh (:class:`DeviceMesh`): The device mesh to set the offset.

    Returns:
        None

    .. warning::
        Current implementation only supports a GPU device mesh.
        Different offset values can be passed in on different ranks so that each rank
        can generate different random numbers in following rand calls.
        If ``device_mesh`` is a sub-mesh and the calling rank is not a part of it,
        `_set_offset` will not set its GPU device's generator offset.
    """
    if device_mesh is not None and device_mesh.get_coordinate() is not None:
        # the current rank is in mesh
        if device_mesh.device_type == "cuda":
            # source: https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp
            # the RNG state tensor returned from torch.cuda.get_rng_state() is a ByteTensor
            # first 200 * sizeof(4120) bytes in tensor are 0xFF
            # next sizeof(uint64_t) bytes are the random seed
            # last sizeof(int64_t) bytes are the offset
            state = get_rng_state(device_mesh)
            offset = state[-8:].view(torch.int64)
            offset[0] = new_offset
            set_rng_state(state, device_mesh)
        else:
            raise NotImplementedError(
                f"DTensor randomness only supports cuda device type, "
                f"but got {device_mesh.device_type}"
            )
