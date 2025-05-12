# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import contextlib
import warnings
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.device_mesh import _get_device_handle, DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor.placement_types import Shard


__all__ = [
    "is_rng_supported_mesh",
    "manual_seed",
    "OffsetBasedRNGTracker",
]

_rng_tracker: Optional["_RNGStateTracker"] = None


def is_rng_supported_mesh(device_mesh: DeviceMesh) -> bool:
    """Checks if the current device of ``device_mesh`` supports DTensor's random APIs.
    Currently DTensor Random APIs only supports cuda/cuda-like devices. We suggest
    users call this API to test the availability before using our random APIs.

    Args:
        device_mesh (:class:`DeviceMesh`): The device mesh on which we check if the
            random ops APIs are supported.

    Returns:
        A bool value. True if ``device_mesh`` supports DTensor Random APIs; False otherwise.

    .. warning::
        Currently we only support correct RNG on cuda/cuda-like devices.
    """
    device_handle = _get_device_handle(device_mesh.device_type)
    if device_handle and hasattr(device_handle, "set_rng_state"):
        return True
    else:
        # TODO: Logs way too much
        warnings.warn(
            f"DTensor random operators may not have complete support on {device_mesh.device_type} device mesh"
        )
        return False


def manual_seed(seed: int, device_mesh: DeviceMesh) -> None:
    """Sets the seed for generating random numbers for the calling rank.

    Args:
        seed (int): The desired seed.
        device_mesh (:class:`DeviceMesh`): The device mesh to set the seed. It is
            required that the ``device_mesh`` include the calling rank. This is
            to ensure that the SPMD region maintains a synchronous RNG state, which
            means no ranks should be initialized with values other than ``seed``.

    Returns:
        None

    .. warning::
        :func:`manual_seed` does not check the ``seed`` value correctness. Users must
        ensure on their own that the value passed in is the desired ``seed`` for ranks
        within ``device_mesh``.
        If ``device_mesh`` is a sub-mesh and the calling rank is not a part of it,
        ``manual_seed`` will throw an error.
        Current implementation only supports a GPU device mesh.
    """
    if not is_rng_supported_mesh(device_mesh):
        warnings.warn(
            "DTensor manual_seed() may not have complete support "
            f"on {device_mesh.device_type} device mesh"
        )
        return

    # instantiate a RNG tracker if haven't. By default DTensor uses an
    # OffsetBasedRNGTracker to perform random operators.
    global _rng_tracker
    if not _rng_tracker:
        _rng_tracker = OffsetBasedRNGTracker(device_mesh, run_state_sync=False)

    # the current rank is in mesh
    if device_mesh.get_coordinate() is not None:
        _rng_tracker._manual_seed(seed)
    else:
        raise RuntimeError(
            "manual_seed requires the current rank to be a part of the device mesh "
            "otherwise DTensor RNG state on the rank will not be initialized and "
            "the behavior of DTensor random ops is undefined."
        )


class _RNGStateTracker:
    """
    _RNGStateTracker stores Random Number Generator (RNG) state (a ByteTensor object)
    in a dict, mapping from a corresponding tag to each state tensor. It also provides
    a set of convenient utility methods to help access/modify the state tensors. The most
    important interface is _distribute_region which will be used when DTensor executes
    a random op (an operator that calls RNG).
    """

    def __init__(self, device: torch.device):
        self._device = device
        self._device_handle = _get_device_handle(self._device.type)
        if not (self._device_handle and self._device_handle.is_available()):
            raise RuntimeError(
                f"{self.__class__.__name__} instantiation requires the presence of "
                f"{device.type} device but couldn't find."
            )

        self._states: dict[str, Tensor] = {}
        self._use_distribute_region = True

    @property
    def rng_states(self) -> dict[str, Tensor]:
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
        seed_tensor = torch.tensor([seed], dtype=torch.uint64, device="cpu").view(
            torch.uint8
        )
        offset_tensor = torch.tensor([0], dtype=torch.uint64, device="cpu").view(
            torch.uint8
        )
        self.rng_states[name] = torch.cat([seed_tensor, offset_tensor])

    def _distribute_region(self, spec: DTensorSpec):
        pass

    def _manual_seed(self, parallel_seed: int) -> None:
        pass


class OffsetBasedRNGTracker(_RNGStateTracker):
    """
    This subclass of ``_RNGStateTracker`` defines the default policy of how RNG states
    should be shared and synchronized among all ranks to respect the semantics of DTensor
    random operators.

    note: _RNGStateTracker only supports cuda/cuda-like device.
    """

    def __init__(
        self,
        device_mesh: DeviceMesh,
        run_state_sync: bool = True,
    ):
        super().__init__(_resolve_device(device_mesh=device_mesh))
        assert self._device_handle is not None
        # DTensor RNG tracker so far only supports CUDA/CUDA-like devices
        if self._device.type == "cpu":
            raise RuntimeError(
                f"{self.__class__.__name__} instantiation requires the presence of "
                f"CUDA/CUDA-like/XPU device. Got {self._device.type} instead."
            )

        rng_state = self._device_handle.get_rng_state().to(self._device)
        if run_state_sync:
            # synchronize RNG state using rank 0's current one
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
            with torch.random.fork_rng(
                devices=[self._device], device_type=self._device.type
            ):
                assert self._device_handle is not None
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
        offset_tensor = torch.tensor([offset], dtype=torch.uint64, device="cpu").view(
            torch.uint8
        )
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
        # note: dim_map does not allow double sharding which is the FSDP(fully_shard)+TP
        # case. Replace the custom logic with dim_map once we support it.
        dim_map: list[Union[int, list[int]]] = [-1] * spec.ndim
        for i, placement in enumerate(spec.placements):
            if isinstance(placement, Shard):
                shard_dim = placement.dim
                if dim_map[shard_dim] == -1:
                    dim_map[shard_dim] = [i]
                else:
                    mesh_dim_list = dim_map[shard_dim]
                    assert isinstance(mesh_dim_list, list)
                    mesh_dim_list.append(i)

        # Compute shard coordinate:
        # The coordinate on each tensor dim is a tuple (idx, range)
        # If a DTensor is partitioned on its dim i into n shards, and the current rank
        # holds the j-th, then its shard coordinate will be (idx=j, range=n) on dim i
        mesh_coordinate = mesh.get_coordinate()
        assert mesh_coordinate is not None
        mesh_size = mesh.shape
        shard_idx_by_dim = []
        total_num_shards_by_dim = []  # total number of shards on each tensor dim
        for mesh_dim in dim_map:
            shard_idx = 0
            total_num_shards = 1
            # the tensor dim is sharded on more than 1 mesh dim
            if isinstance(mesh_dim, list):
                rank_coord = [mesh_coordinate[d] for d in mesh_dim]
                num_shards = [mesh_size[d] for d in mesh_dim]
                # compute the shard idx and total number of shards
                for idx, size in zip(rank_coord, num_shards):
                    shard_idx = shard_idx * size + idx
                    total_num_shards *= size

            shard_idx_by_dim.append(shard_idx)
            total_num_shards_by_dim.append(total_num_shards)

        # compute shard linear index
        shard_linear_idx = self._calc_shard_linear_idx(
            shard_idx_by_dim, total_num_shards_by_dim
        )

        # compute starting offset using the first shard's size
        local_size_on_rank_0 = list(dtensor_shape)
        for idx, placement in enumerate(spec.placements):
            if isinstance(placement, Shard):
                mesh_dim_size = mesh.size(idx)
                shard_dim = placement.dim
                local_size_on_rank_0[shard_dim], _ = (
                    placement._local_shard_size_and_offset(
                        dtensor_shape[shard_dim],
                        mesh_dim_size,
                        0,
                    )
                )

        from torch.distributed.tensor._ops.utils import prod

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

        from torch.distributed.tensor._ops.utils import prod

        numel = prod(dtensor_shape)
        # pytorch: offset must be multiple of 4
        # source: aten/src/ATen/cuda/CUDAGeneratorImpl.cpp
        numel = (numel + 3) // 4 * 4
        self.set_offset("parallel-rng", old_offset + numel)

    def _calc_shard_linear_idx(
        self, shard_coord: list[int], shard_size: list[int]
    ) -> int:
        # compute shard linear index
        shard_linear_idx = 0
        shard_coord_stride = 1
        for idx, size in zip(reversed(shard_coord), reversed(shard_size)):
            shard_linear_idx += idx * shard_coord_stride
            shard_coord_stride *= size

        return shard_linear_idx


def _resolve_device(device_mesh: DeviceMesh) -> torch.device:
    device_type = device_mesh.device_type
    device_handle = _get_device_handle(device_type)
    assert device_handle is not None
    device_idx = device_mesh.get_rank() % device_handle.device_count()
    return torch.device(f"{device_type}:{device_idx:d}")
