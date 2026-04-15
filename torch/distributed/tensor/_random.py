# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import contextlib
import warnings
from collections.abc import Sequence
from logging import getLogger
from typing import Optional

import torch
from torch.distributed._local_tensor import maybe_run_for_local_tensor
from torch.distributed.device_mesh import _get_device_handle, DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.types import IntLikeType


logger = getLogger(__name__)

__all__ = [
    "is_rng_supported_mesh",
    "manual_seed",
    "OffsetBasedRNGTracker",
    "StatelessRNGTracker",
    "set_rng_tracker",
    "get_rng_tracker",
]

_rng_tracker: Optional["_RNGStateTracker"] = None


def set_rng_tracker(tracker: "_RNGStateTracker") -> None:
    """Set the active DTensor RNG tracker.

    Controls how random ops (``uniform_``, ``normal_``, ``native_dropout``,
    etc.) behave on DTensors.  The default is
    :class:`OffsetBasedRNGTracker`; pass a :class:`StatelessRNGTracker` for
    key-based stateless generation.

    Args:
        tracker: An instance of :class:`OffsetBasedRNGTracker` or
            :class:`StatelessRNGTracker`.

    Example::

        >>> import torch.func._random as random
        >>> from torch.distributed.tensor._random import (
        ...     StatelessRNGTracker, set_rng_tracker,
        ... )
        >>> key = random.key(42, device="cuda")  # doctest: +SKIP
        >>> mesh = init_device_mesh("cuda", (4,))  # doctest: +SKIP
        >>> set_rng_tracker(StatelessRNGTracker(key, mesh))  # doctest: +SKIP
    """
    global _rng_tracker
    _rng_tracker = tracker


def get_rng_tracker() -> Optional["_RNGStateTracker"]:
    """Return the currently active DTensor RNG tracker, or ``None``."""
    return _rng_tracker


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
            f"DTensor random operators may not have complete support on {device_mesh.device_type} device mesh",
            stacklevel=2,
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
            f"on {device_mesh.device_type} device mesh",
            stacklevel=2,
        )
        return

    # TODO: deprecate this API, but also need to ensure we disable broadcast for PP case, and that's currently
    # bundled together with this API.  See torchtitan/distributed/utils.py:set_determinism
    # warnings.warn(
    #     "DTensor manual_seed() is deprecated, since DTensor no longer maintains a separate copy of generator state. "
    #     "Use `torch.manual_seed` instead"
    # )
    # Note: we still need to ensure setting `run_state_sync=False` to support the pp case

    # instantiate a RNG tracker if haven't. By default DTensor uses an
    # OffsetBasedRNGTracker to perform random operators.
    global _rng_tracker
    if not _rng_tracker:
        _rng_tracker = OffsetBasedRNGTracker(device_mesh, run_state_sync=False)

    if device_mesh.get_coordinate() is None:
        raise RuntimeError(
            "manual_seed requires the current rank to be a part of the device mesh "
            "otherwise DTensor RNG state on the rank will not be initialized and "
            "the behavior of DTensor random ops is undefined."
        )

    # DTensor no longer maintains a copy of rng state. manual seed on dtensor is the same thing
    # as manual seed on torch.
    #
    # torch.manual_seed will handle LocalTensor mode correctly by
    # iterating through all ranks if seed is a LocalIntNode.
    torch.manual_seed(seed)


class _PhiloxState:
    """
    Convenience accessor for interpreting the packed bits of (seed: uint64, offset: uint64) in the philox state,
    which for some reason is actually exposed as a size-16 uint8 tensor.

    The state is always moved to .cpu since it is necessary for it to be on CPU before applying it back to a generator.
    """

    def __init__(self, state: torch.Tensor):
        self._state = state.to("cpu")

    @property
    def state(self):
        return self._state

    @property
    def offset(self) -> torch.Tensor:
        return self._state[8:].view(dtype=torch.int64)

    @offset.setter
    def offset(self, offset: torch.Tensor) -> None:
        if offset.numel() != 1:
            raise AssertionError
        self._state[8:] = offset.view(torch.uint8)

    @property
    def seed(self) -> torch.Tensor:
        return self._state[:8].view(dtype=torch.uint64)

    @seed.setter
    def seed(self, seed: torch.Tensor) -> None:
        if seed.numel() != 1:
            raise AssertionError
        self._state[:8] = seed.view(torch.uint8)


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
        self._use_distribute_region = True

    @property
    def distribute_region_enabled(self) -> bool:
        return self._use_distribute_region

    @distribute_region_enabled.setter
    def distribute_region_enabled(self, value) -> None:
        self._use_distribute_region = value

    def _distribute_region(
        self, spec: DTensorSpec, generator: torch.Generator | None = None
    ):
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
        if self._device_handle is None:
            raise AssertionError
        # DTensor RNG tracker so far only supports CUDA/CUDA-like devices
        if self._device.type == "cpu":
            raise RuntimeError(
                f"{self.__class__.__name__} instantiation requires the presence of "
                f"CUDA/CUDA-like/XPU device. Got {self._device.type} instead."
            )

        if run_state_sync:
            rng_state = self._get_device_state()
            # synchronize RNG state using rank 0's current one
            torch.distributed.broadcast(rng_state, 0)
            my_rng_state = self._get_device_state()
            if not all(my_rng_state == rng_state):
                logger.warning(
                    "DTensor is synchronizing RNG states of every rank with the state from rank 0. "
                    "This behavior is deprecated. "
                    "Please call `torch.manual_seed()` on every rank that participates in SPMD DTensor Operations with "
                    "the same seed. If using Pipeline Parallelism, each pipeling state would use a different seed, "
                    "but all ranks belonging to one pipeline stage would use the same seed."
                )
            self._set_device_state(rng_state)

    def _get_device_state(self) -> torch.Tensor:
        if self._device.type == "hpu":
            self._device_handle.set_rng_ctx("philox")
        rng_state = self._device_handle.get_rng_state().to(self._device)
        if self._device.type == "hpu":
            self._device_handle.unset_rng_ctx("philox")
        return rng_state

    def _set_device_state(self, state: torch.Tensor):
        # It seems that the underlying generator wants a cpu tensor but the dtensor code expects `_get_device_state`
        # to convert to a 'device' tensor, probably because we may use it with our backend comms for sync/debug
        # for now, we just convert back to cpu here to make sure it always works.
        if self._device.type == "hpu":
            self._device_handle.set_rng_ctx("philox")
        self._device_handle.set_rng_state(state.to("cpu"))
        if self._device.type == "hpu":
            self._device_handle.unset_rng_ctx("philox")

    @contextlib.contextmanager
    def _distribute_region(
        self, spec: DTensorSpec, generator: torch.Generator | None = None
    ):
        from torch.distributed._local_tensor import maybe_enable_local_tracker

        if local_tracker_context := maybe_enable_local_tracker(
            self._device.type, self.distribute_region_enabled, spec, generator
        ):
            with local_tracker_context:
                yield
            return

        # regular (non-LocalTensor) mode
        if generator is not None:
            # This is a little hacky, but for any user-passed generator, we store its state under a unique key,
            # not because we need to keep a copy of it but because its the easiest way to make it work with the
            # existing set/get APIs. We also ensure we remove it from rng_states after each _distribute_region.
            state = _PhiloxState(generator.get_state())
        else:
            state = _PhiloxState(self._get_device_state())

        if self.distribute_region_enabled:
            if self._device.type == "hpu":
                self._device_handle.set_rng_ctx("philox")
            old_offset = state.offset.clone()
            self._set_pre_op_offset(state, spec)
            with torch.random.fork_rng(
                devices=[self._device], device_type=self._device.type
            ):
                if self._device_handle is None:
                    raise AssertionError
                self._device_handle.set_rng_state(state.state)
                try:
                    yield  # execute the region code
                finally:
                    # update offset to synchronize among ranks
                    self._set_post_op_offset(state, spec, old_offset)
            if self._device.type == "hpu":
                self._device_handle.unset_rng_ctx("philox")
        else:
            yield

        if generator is not None:
            # ensure we (a) propagate the state advancement back to the user's RNG so its visible and impacts any future
            # usage of that RNG (dtensor or non-dtensor), (b) drop it from our own cache so that if the user updates
            # the seed value in their rng and uses it with DTensor again, we always use the latest value
            generator.set_state(state.state)
        else:
            self._set_device_state(state.state)

    def _set_pre_op_offset(self, state: _PhiloxState, spec: DTensorSpec) -> None:
        """Set the starting RNG offset for current device's local shard before actual
        op execution. The pre_op_offset value should start from the current RNG offset
        and increment by the size of local shard until it reaches the size of the whole
        DTensor. For different ranks that hold the same DTensor shard, their pre_op_offset
        will be the same.

        Args:
            state (:class:`Tensor`): The generator state to modify
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
        start_offset_incr, _ = self._compute_rng_offsets(spec)
        state.offset = state.offset + start_offset_incr

    def _set_post_op_offset(
        self, state: _PhiloxState, spec: DTensorSpec, old_offset: torch.Tensor
    ) -> None:
        """Sets the RNG to a synchronized state after running the local random op. Every
        rank should set its RNG offset to `old_offset + DTensor.numel()` where old_offset is
        the offset before calling `set_pre_op_offset` i.e. the offset before running DTensor
        random ops.

        Args:
            state (:class:`Tensor`): The generator state to modify.
            spec (:class:`DTensorSpec`): the spec of the DTensor object on which
                we post-process the offset for running random ops.

        Returns:
            None
        """
        _, end_offset_incr = self._compute_rng_offsets(spec)
        state.offset = old_offset + end_offset_incr

    def _compute_rng_offsets(self, spec: DTensorSpec) -> tuple[int, int]:
        """Compute the RNG offset increments for a distributed random op.

        These values are derived from mesh topology, placements, and tensor shape,
        and are static for a given compiled graph. They can be burned into the graph
        as integer constants rather than keeping the DTensorSpec around at runtime.

        Returns:
            (start_offset_incr, end_offset_incr) — both aligned to multiples of 4.
        """
        from torch.distributed.tensor._ops.utils import prod

        mesh = spec.mesh
        mesh_coordinate = [mesh._sym_get_coordinate(i) for i in range(mesh.ndim)]

        shard_idx_by_dim, total_num_shards_by_dim = _calc_shard_info(
            mesh_coordinate, spec
        )
        shard_linear_idx = self._calc_shard_linear_idx(
            shard_idx_by_dim, total_num_shards_by_dim
        )
        local_size = prod(_calc_first_shard_size(spec))
        # pytorch: offset must be multiple of 4
        # source: aten/src/ATen/cuda/CUDAGeneratorImpl.cpp
        start_offset_incr = (shard_linear_idx * local_size + 3) // 4 * 4
        end_offset_incr = (prod(spec.shape) + 3) // 4 * 4

        return start_offset_incr, end_offset_incr

    def _calc_shard_linear_idx(
        self, shard_coord: Sequence[IntLikeType], shard_size: Sequence[IntLikeType]
    ) -> IntLikeType:
        return _calc_shard_linear_idx(shard_coord, shard_size)


def _calc_first_shard_size(spec: DTensorSpec) -> list[int]:
    local_size_on_rank_0 = list(spec.shape)
    for idx, placement in enumerate(spec.placements):
        if isinstance(placement, Shard | _StridedShard):
            mesh_dim_size = spec.mesh.size(idx)
            shard_dim = placement.dim
            local_size_on_rank_0[shard_dim], _ = placement._local_shard_size_and_offset(
                spec.shape[shard_dim],
                mesh_dim_size,
                0,
            )
    return local_size_on_rank_0


def _calc_shard_info(
    mesh_coordinate: Sequence[IntLikeType], spec: DTensorSpec
) -> tuple[list[IntLikeType], list[IntLikeType]]:
    mesh = spec.mesh
    # note: dim_map does not allow double sharding which is the FSDP(fully_shard)+TP
    # case. Replace the custom logic with dim_map once we support it.
    dim_map: list[int | list[int]] = [-1] * spec.ndim
    for i, placement in enumerate(spec.placements):
        if isinstance(placement, Shard | _StridedShard):
            shard_dim = placement.dim
            if dim_map[shard_dim] == -1:
                dim_map[shard_dim] = [i]
            else:
                mesh_dim_list = dim_map[shard_dim]
                if not isinstance(mesh_dim_list, list):
                    raise AssertionError
                mesh_dim_list.append(i)

    # Compute shard coordinate:
    # The coordinate on each tensor dim is a tuple (idx, range)
    # If a DTensor is partitioned on its dim i into n shards, and the current rank
    # holds the j-th, then its shard coordinate will be (idx=j, range=n) on dim i
    mesh_size = mesh.shape
    shard_idx_by_dim = []
    total_num_shards_by_dim: list[
        IntLikeType
    ] = []  # total number of shards on each tensor dim
    for mesh_dim in dim_map:
        shard_idx: IntLikeType = 0
        total_num_shards: IntLikeType = 1
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
    return shard_idx_by_dim, total_num_shards_by_dim


def _calc_shard_linear_idx(
    shard_coord: Sequence[IntLikeType], shard_size: Sequence[IntLikeType]
) -> IntLikeType:
    # compute shard linear index
    shard_linear_idx: IntLikeType = 0
    shard_coord_stride: IntLikeType = 1
    for idx, size in zip(reversed(shard_coord), reversed(shard_size)):
        shard_linear_idx += idx * shard_coord_stride
        shard_coord_stride *= size

    return shard_linear_idx


def _resolve_device(device_mesh: DeviceMesh) -> torch.device:
    device_type = device_mesh.device_type
    device_handle = _get_device_handle(device_type)
    if device_handle is None:
        raise AssertionError
    device_idx = device_mesh.get_rank() % device_handle.device_count()

    @maybe_run_for_local_tensor
    def get_device(device_idx):
        return torch.device(f"{device_type}:{device_idx:d}")

    return get_device(device_idx)


def _placements_to_splits(
    global_shape: tuple[int, ...],
    placements: tuple[Placement, ...],
    mesh: DeviceMesh,
) -> tuple[int, ...]:
    """Convert DTensor placements + mesh to the ``splits`` tuple expected by
    :func:`torch.func._random.unbind`.

    For each tensor dimension, the split count is the product of mesh sizes
    along all mesh dimensions that shard that tensor dimension.  Replicated
    mesh dimensions contribute a factor of 1 (no split).
    """
    splits = [1] * len(global_shape)
    for mesh_dim, placement in enumerate(placements):
        if isinstance(placement, Shard):
            splits[placement.dim] *= mesh.size(mesh_dim)
        elif isinstance(placement, Replicate):
            pass
        elif isinstance(placement, Partial):
            raise ValueError(
                "StatelessRNGTracker does not support Partial placements"
            )
        else:
            raise ValueError(f"Unsupported placement type: {type(placement)}")
    return tuple(splits)


def _my_shard_indices(
    ndim: int,
    placements: tuple[Placement, ...],
    mesh: DeviceMesh,
    coordinate: list[int],
) -> tuple[int, ...]:
    """Return this rank's index into the unbind grid for each tensor dimension.

    For each tensor dimension, if it is sharded across one or more mesh
    dimensions, the index is computed as a composite row-major index over
    those mesh dimensions.  Non-sharded dimensions get index 0.
    """
    dim_to_mesh_dims: dict[int, list[int]] = {}
    for mesh_dim, placement in enumerate(placements):
        if isinstance(placement, Shard):
            dim_to_mesh_dims.setdefault(placement.dim, []).append(mesh_dim)

    indices: list[int] = []
    for d in range(ndim):
        if d in dim_to_mesh_dims:
            idx = 0
            for mesh_dim in dim_to_mesh_dims[d]:
                idx = idx * mesh.size(mesh_dim) + coordinate[mesh_dim]
            indices.append(idx)
        else:
            indices.append(0)
    return tuple(indices)


class StatelessRNGTracker(_RNGStateTracker):
    """Stateless RNG tracker for DTensor random generation.

    Set as the active DTensor RNG tracker via
    ``torch.distributed.tensor._random._rng_tracker = tracker``, then
    standard ops like ``dtensor.uniform_()`` or ``torch.randn`` on DTensors
    are automatically handled.  Internally the tracker derives a per-op key
    from a root key + auto-incrementing counter, computes shard offsets, and
    sets the Philox generator state so each rank gets the correct slice.

    For explicit generation with reconstruction guarantee (SPMD), use
    :func:`torch.func._random.sharded_uniform` /
    :func:`torch.func._random.sharded_normal` instead.

    Args:
        key: A PRNG key from :func:`torch.func._random.key`.
        device_mesh: The device mesh for distributed generation.

    Example::

        >>> import torch
        >>> import torch.func._random as random
        >>> from torch.distributed.tensor import _random as dt_random
        >>> from torch.distributed.tensor._random import StatelessRNGTracker
        >>> mesh = init_device_mesh("cuda", (4,))  # doctest: +SKIP
        >>> key = random.key(42, device="cuda")  # doctest: +SKIP
        >>> dt_random._rng_tracker = StatelessRNGTracker(key, mesh)  # doctest: +SKIP
    """

    def __init__(
        self, key: torch.Tensor, device_mesh: DeviceMesh
    ) -> None:
        super().__init__(_resolve_device(device_mesh=device_mesh))
        self._mesh = device_mesh
        self._coordinate = device_mesh.get_coordinate()
        if self._coordinate is None:
            raise RuntimeError(
                "StatelessRNGTracker requires the current rank to be part of "
                "the device mesh"
            )
        self._key = key
        self._counter = 0

    @property
    def key(self) -> torch.Tensor:
        return self._key

    @key.setter
    def key(self, value: torch.Tensor) -> None:
        self._key = value
        self._counter = 0

    def _get_device_state(self) -> torch.Tensor:
        return self._device_handle.get_rng_state().to(self._device)

    # -- dispatch-compatible path (dtensor.uniform_, torch.randn, etc.) --

    @contextlib.contextmanager
    def _distribute_region(
        self, spec: DTensorSpec, generator: torch.Generator | None = None
    ):
        """Context manager called by DTensor dispatch for random ops.

        Derives a per-op key from the root key + counter, computes the shard
        offset from the spec, and sets the Philox generator state so the
        standard op produces the correct values for this rank's shard.
        """
        import torch.func._random as stateless_random

        # Derive a unique key for this op.
        op_key = stateless_random.fold_in(self._key, self._counter)
        self._counter += 1

        # Extract (seed, offset) from the derived key as raw bytes,
        # then set them directly on the Philox state to avoid
        # signed/unsigned overflow issues with large uint64 values.
        key_bytes = op_key.cpu().view(torch.uint8)
        derived_seed_bytes = key_bytes[:8]
        derived_offset_bytes = key_bytes[8:]

        # Compute this shard's offset increment (same logic as offset-based).
        start_offset_incr, _ = self._compute_rng_offsets(spec)

        # Build a Philox state with the derived seed and adjusted offset.
        if generator is not None:
            state = _PhiloxState(generator.get_state())
        else:
            state = _PhiloxState(self._get_device_state())

        state.seed = derived_seed_bytes.view(torch.uint64)
        # CUDA requires offset to be a multiple of 4. Align the base offset
        # first, then add the shard increment (already aligned) to preserve
        # shard differentiation.
        raw_offset = derived_offset_bytes.view(torch.int64).item()
        # Work in unsigned space: mask to uint64 range.
        unsigned_offset = raw_offset & ((1 << 64) - 1)
        aligned_offset = (unsigned_offset // 4) * 4 + start_offset_incr
        state.offset = torch.tensor([aligned_offset & ((1 << 63) - 1)], dtype=torch.int64)

        with torch.random.fork_rng(
            devices=[self._device], device_type=self._device.type
        ):
            self._device_handle.set_rng_state(state.state)
            yield

        if generator is not None:
            generator.set_state(state.state)

    def _run_random_op(
        self,
        spec: DTensorSpec,
        op_call: object,
        local_tensor_args: tuple,
        local_kwargs: dict,
    ) -> tuple[object | None, bool]:
        """Fill a local tensor with parallelism-invariant random values.

        Generates the full (global) tensor using a single non-batched PRNG
        key derived from the root key + counter, then extracts this rank's
        local shard.  Because every rank derives the same key for the same
        op, the global random values are identical regardless of how the
        tensor is partitioned — making weight initialisation independent of
        TP/DP degree.

        Supports ``Shard``, ``_StridedShard``, and ``Replicate`` placements.

        Returns ``(result, True)`` on success, or ``(None, False)`` if the
        caller should fall back to :meth:`_distribute_region`.
        """
        import torch.func._random as stateless_random

        # Only handle the in-place ops used by weight init.
        if op_call not in (
            torch.ops.aten.uniform_.default,
            torch.ops.aten.normal_.default,
        ):
            return None, False

        # Reject Partial or unknown placements.
        for p in spec.placements:
            if isinstance(p, Partial):
                return None, False
            if not isinstance(p, (Shard, _StridedShard, Replicate)):
                return None, False

        local_tensor = local_tensor_args[0]

        # Derive a unique key for this op.
        op_key = stateless_random.fold_in(self._key, self._counter)
        self._counter += 1

        # Generate the full (global) tensor using a non-batched key so that
        # the random values depend only on (root_key, counter), not on the
        # parallelism configuration.
        full = torch.empty(
            spec.shape, dtype=local_tensor.dtype, device=local_tensor.device
        )
        if op_call == torch.ops.aten.uniform_.default:
            low = float(local_tensor_args[1]) if len(local_tensor_args) > 1 else 0.0
            high = float(local_tensor_args[2]) if len(local_tensor_args) > 2 else 1.0
            stateless_random.uniform_(op_key, full, low=low, high=high)
        elif op_call == torch.ops.aten.normal_.default:
            mean = float(local_tensor_args[1]) if len(local_tensor_args) > 1 else 0.0
            std = float(local_tensor_args[2]) if len(local_tensor_args) > 2 else 1.0
            stateless_random.normal_(op_key, full, mean=mean, std=std)

        # Extract this rank's local shard from the full tensor.  We wrap
        # the full tensor as a Replicated DTensor and redistribute to the
        # target placements, letting DTensor's own logic handle _StridedShard,
        # double-sharding, and any other exotic placement combination.
        from torch.distributed.tensor import DTensor

        rep_placements = [Replicate()] * spec.mesh.ndim
        full_dt = DTensor.from_local(full, spec.mesh, rep_placements, run_check=False)
        target_dt = full_dt.redistribute(spec.mesh, spec.placements)
        local_tensor.copy_(target_dt._local_tensor)
        del full, full_dt, target_dt
        return local_tensor, True

    def _compute_rng_offsets(self, spec: DTensorSpec) -> tuple[int, int]:
        """Compute shard offset increments (same formula as offset-based)."""
        from torch.distributed.tensor._ops.utils import prod

        mesh = spec.mesh
        mesh_coordinate = [mesh._sym_get_coordinate(i) for i in range(mesh.ndim)]
        shard_idx_by_dim, total_num_shards_by_dim = _calc_shard_info(
            mesh_coordinate, spec
        )
        shard_linear_idx = _calc_shard_linear_idx(
            shard_idx_by_dim, total_num_shards_by_dim
        )
        local_size = prod(_calc_first_shard_size(spec))
        start_offset_incr = (shard_linear_idx * local_size + 3) // 4 * 4
        end_offset_incr = (prod(spec.shape) + 3) // 4 * 4
        return start_offset_incr, end_offset_incr

