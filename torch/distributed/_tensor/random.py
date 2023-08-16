# Copyright (c) Meta Platforms, Inc. and affiliates
import contextlib
import warnings
from typing import Dict, List, Optional

import torch
import torch.distributed as dist

from torch import Tensor
from torch.distributed._tensor.device_mesh import _get_device_handle, DeviceMesh
from torch.distributed._tensor.placement_types import DTensorSpec, Shard


_rng_tracker: Optional["RNGStateTracker"] = None


def is_rng_supported_mesh(device_mesh: DeviceMesh) -> bool:
    """Checks if the current device of `device_mesh` supports DTensor's random APIs.
    Currently DTensor Random APIs only supports cuda/cuda-like devices. We suggest
    users call this API to test the availability before using our random APIs.

    Args:
        device_mesh (:class:`DeviceMesh`): The device mesh on which we check if the
            random ops APIs are supported.

    Returns:
        A bool value. True if `device_mesh` supports DTensor Random APIs; False otherwise.

    .. warning::
        Currently we only support correct RNG on cuda/cuda-like devices.
    """
    device_handle = _get_device_handle(device_mesh.device_type)
    if device_handle and hasattr(device_handle, "set_rng_state"):
        return True
    else:
        warnings.warn(
            f"DTensor random operators may not have complete support on {device_mesh.device_type} device mesh"
        )
        return False


def manual_seed(seed: int, device_mesh: DeviceMesh, tp_dim: int = 0) -> None:
    """Sets the seed for generating random numbers for the calling rank.

    Args:
        seed (int): The desired seed.
        device_mesh (:class:`DeviceMesh`): The device mesh to set the seed.
        tp_dim (int, optional): The mesh dimension where to apply Tensor Parallel
            Default: 0

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
    device_handle = _get_device_handle(device_mesh.device_type)
    if not device_handle:
        raise NotImplementedError(
            f"DTensor randomness only supports cuda/cuda-like device type, but got {device_mesh.device_type}"
        )

    # allgather the seed over the default PG
    object_list = [seed] * dist.get_world_size()
    dist.all_gather_object(object_list, seed)
    for rank, object in enumerate(object_list):
        if seed != int(object):
            raise RuntimeError(
                f"calling manual_seed function over {device_mesh} but received different seed values on ranks:",
                f"seed on rank {dist.get_rank()} is {seed}, and seed on rank {rank} is {object}!",
            )
    # instantiate a RNG tracker if haven't. By default DTensor uses an
    # OffsetBasedRNGTracker to perform random operators.
    global _rng_tracker
    if not _rng_tracker:
        _rng_tracker = OffsetBasedRNGTracker(device_mesh.device_type)

    # the current rank is in mesh
    if device_mesh.get_coordinate() is not None:
        if isinstance(_rng_tracker, TensorParallelRNGTracker):
            _rng_tracker._manual_seed(device_mesh, seed, tp_dim)
        elif isinstance(_rng_tracker, OffsetBasedRNGTracker):
            _rng_tracker._manual_seed(seed)
        else:
            raise RuntimeError(
                f"Unknown type of cuda RNG state tracker: _rng_tracker = {_rng_tracker}"
            )


class RNGStateTracker:
    """
    RNGStateTracker stores Random Number Generator (RNG) state (a ByteTensor object)
    in a dict, mapping from a corresponding tag to each state tensor. It also provides
    a set of convenient utility methods to help access/modify the state tensors. The most
    important interface is _distribute_region which will be used when DTensor executes
    a random op (an operator that calls RNG).
    """

    def __init__(self, device_type: str = "cuda"):
        self._device_type = device_type
        self._device_handle = _get_device_handle(device_type)
        if not (self._device_handle and self._device_handle.is_available()):
            raise RuntimeError(
                f"{self.__class__.__name__} instantiation requires the presence of CUDA/CUDA-like device"
            )

        self._states: Dict[str, Tensor] = {}
        self._devices = [self._device_handle.current_device()]
        self._use_distribute_region = True

    @property
    def rng_states(self) -> Dict[str, Tensor]:
        return self._states

    @property
    def distribute_region_enabled(self) -> bool:
        return self._use_distribute_region

    @distribute_region_enabled.setter
    def distribute_region_enabled(self, value) -> None:
        self._use_distribute_region = value

    def rng_state_is_sync(self, name) -> bool:
        return name in self.rng_states

    def get_seed(self, name: str) -> int:
        if name not in self.rng_states:
            raise RuntimeError(
                f"{self.__class__.__name__} does not have random state for {name}"
            )

        seed_tensor = (self.rng_states[name])[0:8].view(dtype=torch.int64)
        return int(seed_tensor.item())

    def set_seed(self, name: str, seed: int) -> None:
        seed_tensor = torch.tensor([seed]).view(torch.uint8)
        offset_tensor = torch.tensor([0]).view(torch.uint8)
        self.rng_states[name] = torch.cat([seed_tensor, offset_tensor])

    def _distribute_region(self, spec: DTensorSpec):
        pass


class OffsetBasedRNGTracker(RNGStateTracker):
    """
    This subclass of `RNGStateTracker` defines the default policy of how RNG states
    should be shared and synchronized among all ranks to respect the semantics of DTensor
    random operators.
    """

    def __init__(self, device_type: str = "cuda"):
        super().__init__(device_type)
        # synchronize RNG state using rank 0's current one
        rng_state = self._device_handle.get_rng_state().to(device_type)
        dist.broadcast(rng_state, 0)
        self.rng_states["parallel-rng"] = rng_state.to("cpu")

    def _manual_seed(self, parallel_seed: int) -> None:
        self.set_seed("parallel-rng", parallel_seed)

    @contextlib.contextmanager
    def _distribute_region(self, spec: DTensorSpec):
        # check if the parallel rng state has been synchronized or not
        if not self.rng_state_is_sync("parallel-rng"):
            raise RuntimeError(
                "OffsetBasedRNGTracker requires the random state to be synchronized "
                "before entering into a distribute region!"
            )

        if self.distribute_region_enabled:
            old_offset = self.get_offset("parallel-rng")
            self._set_pre_op_offset(spec)
            with torch.random.fork_rng(self._devices, device_type=self._device_type):
                self._device_handle.set_rng_state(self.rng_states["parallel-rng"])
                try:
                    yield  # execute the region code
                finally:
                    # update offset to synchronize among ranks
                    self._set_post_op_offset(spec, old_offset)
        else:
            yield

    def get_offset(self, name: str) -> int:
        if name not in self.rng_states:
            raise RuntimeError(
                f"{self.__class__.__name__} does not have random state for {name}"
            )

        offset_tensor = (self.rng_states[name])[8:].view(dtype=torch.int64)
        return int(offset_tensor.item())

    def set_offset(self, name: str, offset: int) -> None:
        if name not in self.rng_states:
            raise RuntimeError(
                f"{self.__class__.__name__} does not have random state for {name}"
            )

        seed_tensor = (self.rng_states[name])[0:8]
        offset_tensor = torch.tensor([offset]).view(torch.uint8)
        self.rng_states[name] = torch.cat([seed_tensor, offset_tensor])

    def _set_pre_op_offset(self, spec: DTensorSpec) -> None:
        """Set the starting RNG offset for current device's local shard before actual
        op execution. The pre_op_offset value should start from the current RNG offset
        and increment by the size of local shard until it reaches the size of the whole
        DTensor. For different ranks that hold the same DTensor shard, their pre_op_offset
        will be the same.

        Args:
            spec (:class:`DTensorSpec`): the spec of the DTensor object on which
                we prepare the offset for running random ops.

        Returns:
            None

        .. warning::
            Note that, current implementation does not consider DTensor's continguity.

        Example:
            take a DTensor of shape [8, 16] as an example. Assume that the DTensor
            is placed on a device mesh with placements ([Shard(1), Replicate(), Shard(0)]),
            and the mesh is:
                [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
            ``spec.mesh.get_coordinate()`` provides the coordinate of the current rank
            in the mesh. For example, the coordinate of rank 5 is (1, 0, 1).

            Another concept to introduce besides rank coordinate is shard coordinate.
            Each rank holds a local shard of the DTensor. In the example, the DTensor
            is partitioned into 4 [4, 8] shards. The first shard has 2 replicas and
            rank 0 (coord (0, 0, 0)) and rank 2 (coord (0, 1, 0)) have 1 replica each.
            That being said, the local shard on rank 0 and rank 2 correspond to the same
            shard of the DTensor. To denote each DTensor shard, we use a shard coordinate
            (in the example, it will be a tuple (i, j) where shard (i, j) has the slice
            DTensor[4 * i : 4 * (i + 1), 8 * j : 8 * (j + 1)], 0 <= i < 2, 0 <= j < 2).

            Once we have rank coordinate and shard coordinate, we can calculate on each rank
            what shard of the DTensor the rank holds, with the help of dim_map. The dim_map
            of the above DTensor is [2, 0] so the shard coordinate of a rank with rank coord
            (x, y, z) is simply (z, x) by taking(rank_coord[dim_map[0]],rank_coord[dim_map[1]]).
            Following this calculation,
            rank 0 and rank 2 holds the shard of coord (0, 0);
            rank 1 and rank 3 holds the shard of coord (0, 1);
            rank 4 and rank 6 holds the shard of coord (1, 0);
            rank 5 and rank 7 holds the shard of coord (1, 1);

            The last value to calculate before obtaining the starting offset is the shard linear index.
            The starting offset for each rank will be its shard_linear_index * local_tensor_numel.
        """
        dtensor_shape = spec.shape
        mesh = spec.mesh
        dim_map = spec.dim_map

        # Compute shard coordinate:
        # The coordinate on each tensor dim is a tuple (idx, range)
        # If a DTensor is partitioned on its dim i into n shards, and the current rank
        # holds the j-th, then its shard coordinate will be (idx=j, range=n) on dim i
        coordinate = mesh.get_coordinate()
        assert coordinate is not None
        shard_coord = [
            coordinate[mesh_dim] if mesh_dim >= 0 else 0 for mesh_dim in dim_map
        ]
        shard_size = [
            mesh.size(mesh_dim) if mesh_dim >= 0 else 1 for mesh_dim in dim_map
        ]

        # compute shard linear index
        shard_linear_idx = self._calc_shard_linear_idx(shard_coord, shard_size)

        # compute starting offset using the first shard's size
        local_size_on_rank_0 = list(dtensor_shape)
        for idx, placement in enumerate(spec.placements):
            if isinstance(placement, Shard):
                mesh_dim_size = mesh.size(idx)
                shard_dim = placement.dim
                local_size_on_rank_0[shard_dim] = placement._local_shard_size_on_dim(
                    dtensor_shape[shard_dim],
                    mesh_dim_size,
                    0,
                    return_offset=False,
                )[0]

        from torch.distributed._tensor.ops.utils import prod

        local_size = prod(local_size_on_rank_0)

        # get current RNG offset
        current_offset = self.get_offset("parallel-rng")

        # pytorch: offset must be multiple of 4
        # source: aten/src/ATen/cuda/CUDAGeneratorImpl.cpp
        offset_incr = (shard_linear_idx * local_size + 3) // 4 * 4
        self.set_offset("parallel-rng", current_offset + offset_incr)

    def _set_post_op_offset(self, spec: DTensorSpec, old_offset: int) -> None:
        """Sets the RNG to a synchronized state after running the local random op. Every
        rank should set its RNG offset to `old_offset + DTensor.numel()` where old_offset is
        the offset before calling `set_pre_op_offset` i.e. the offset before running DTensor
        random ops.

        Args:
            spec (:class:`DTensorSpec`): the spec of the DTensor object on which
                we post-process the offset for running random ops.

        Returns:
            None
        """
        dtensor_shape = spec.shape

        from torch.distributed._tensor.ops.utils import prod

        numel = prod(dtensor_shape)
        # pytorch: offset must be multiple of 4
        # source: aten/src/ATen/cuda/CUDAGeneratorImpl.cpp
        numel = (numel + 3) // 4 * 4
        self.set_offset("parallel-rng", old_offset + numel)

    def _calc_shard_linear_idx(
        self, shard_coord: List[int], shard_size: List[int]
    ) -> int:
        # compute shard linear index
        shard_linear_idx = 0
        shard_coord_stride = 1
        for idx, size in zip(reversed(shard_coord), reversed(shard_size)):
            shard_linear_idx += idx * shard_coord_stride
            shard_coord_stride *= size

        return shard_linear_idx


class TensorParallelRNGTracker(RNGStateTracker):
    def __init__(self, device_type: str = "cuda"):
        super().__init__(device_type)
        # copy the default RNG state
        self.rng_states["tensor-parallel-rng"] = self._device_handle.get_rng_state()

    def _manual_seed(
        self,
        device_mesh: DeviceMesh,
        base_seed: int = 1234,
        tp_dim: int = 0,
    ):
        coordinate = device_mesh.get_coordinate()
        assert coordinate is not None
        tensor_parallel_rank = coordinate[tp_dim]
        # this magic number 2718 comes from Megatron's code
        # (https://github.com/NVIDIA/Megatron-LM/blob/060415572f4365a2e895f8036c4e37dad0efbdf5/megatron/core/tensor_parallel/random.py#L162-L163)
        MegatronMagicNum = 2718
        tensor_parallel_seed = base_seed + MegatronMagicNum + tensor_parallel_rank
        self.set_seed("tensor-parallel-rng", tensor_parallel_seed)

    @contextlib.contextmanager
    def _distribute_region(self, spec: DTensorSpec):
        # check if the tensor parallel rng state has been synchronized or not
        if not self.rng_state_is_sync("tensor-parallel-rng"):
            raise RuntimeError(
                "TensorParallelRNGTracker requires the random state to be synchronized "
                "before entering into a distribute region!"
            )

        if self.distribute_region_enabled:
            with torch.random.fork_rng(self._devices, device_type=self._device_type):
                self._device_handle.set_rng_state(
                    self.rng_states["tensor-parallel-rng"]
                )
                try:
                    yield
                finally:
                    self.rng_states[
                        "tensor-parallel-rng"
                    ] = self._device_handle.get_rng_state()
        else:
            yield
