# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Optional

import torch
from torch import Tensor

from torch.distributed._tensor import DeviceMesh

def set_rng_state(new_state: Tensor, device_mesh: DeviceMesh) -> None:
    r"""
    """
    if device_mesh is not None and device_mesh.get_coordinate() is not None:
        # the current rank is in mesh
        if device_mesh.device_type == "cuda":
            torch.cuda.set_rng_state(new_state)
        else:
            raise NotImplementedError(
                f"DTensor randomness only supports cuda device type, "
                f"but got {device_mesh.device_type}"
            )

def get_rng_state(device_mesh: DeviceMesh) -> Optional[Tensor]:
    r"""
    """
    if device_mesh is None or device_mesh.get_coordinate() is None:
        return None
    else:
        if device_mesh.device_type == "cuda":
            return torch.cuda.get_rng_state()
        else:
            raise NotImplementedError(
                f"DTensor randomness only supports cuda device type, "
                f"but got {device_mesh.device_type}"
            )

def manual_seed(seed:int, device_mesh: DeviceMesh) -> None:
    r"""
    """
    if device_mesh is not None and device_mesh.get_coordinate() is not None:
        # the current rank is in mesh
        if device_mesh.device_type == "cuda":
            torch.cuda.manual_seed(seed)
        else:
            raise NotImplementedError(
                f"DTensor randomness only supports cuda device type, "
                f"but got {device_mesh.device_type}"
            )

def manual_seed_all(seed:int, device_mesh: DeviceMesh) -> None:
    r"""
    """
    if device_mesh is not None and device_mesh.get_coordinate() is not None:
        # the current rank is in mesh
        # perform extra check on seed value. seed must be consistent within mesh
        if device_mesh.device_type == "cuda":
            torch.cuda.manual_seed(seed)
        else:
            raise NotImplementedError(
                f"DTensor randomness only supports cuda device type, "
                f"but got {device_mesh.device_type}"
            )

def _set_offset(new_offset: int, device_mesh: DeviceMesh) -> None:
    if device_mesh is not None and device_mesh.get_coordinate() is not None:
        # the current rank is in mesh
        if device_mesh.device_type == "cuda":
            state = get_rng_state(device_mesh)
            offset = state[-8:].view(torch.int64)
            offset[0] = new_offset
            set_rng_state(state, device_mesh)
        else:
            raise NotImplementedError(
                f"DTensor randomness only supports cuda device type, "
                f"but got {device_mesh.device_type}"
            )