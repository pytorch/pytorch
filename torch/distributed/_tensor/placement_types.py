# Copyright (c) Meta Platforms, Inc. and affiliates

from dataclasses import dataclass
from typing import Optional, List, Sequence, Tuple, cast

import torch
import torch.distributed.distributed_c10d as c10d
from torch.distributed._spmd.comm_tensor import CommTensor

from torch.distributed._tensor.device_mesh import DeviceMesh


class Placement(object):
    # base class Placement type

    # convenient utils to check for placement types
    def is_shard(self, dim: Optional[int] = None) -> bool:
        if dim is not None and isinstance(self, Shard):
            return self.dim == dim
        else:
            return isinstance(self, Shard)

    def is_replicate(self) -> bool:
        return isinstance(self, Replicate)

    def is_partial(self) -> bool:
        return isinstance(self, _Partial)


@dataclass
class Shard(Placement):
    # shard placement, shard on a dim
    dim: int

    def _split_tensor(
        self,
        tensor: torch.Tensor,
        num_chunks: int,
        *,
        with_padding: bool = True,
        contiguous: bool = True,
    ) -> Tuple[List[torch.Tensor], int]:
        # NOTE: For with_padding option, we pad the tensor on each rank before calling
        # the collectives (i.e. scatter/all_gather, etc.). This is because for gloo
        # backend, it does not support uneven collectives, nccl supports some, but
        # it might be slow compared to even size collective, we need to pad tensor
        # before really calling the collective, and unpad/narrow it afterwards
        # TODO: consider if we should remove this logic once ProcessGroupGloo
        # support uneven list, and collective perfomance on par
        assert (
            self.dim <= tensor.ndim
        ), f"Sharding dim {self.dim} greater than tensor ndim {tensor.ndim}"
        assert (
            tensor.size(self.dim) >= num_chunks
        ), f"Tensors to be sharded on dim {self.dim} must be at least as large as the number of devices in that dimension {num_chunks}"
        # split tensor over dimension `dim` into n slices with padding if necessary
        tensor_list = list(tensor.tensor_split(num_chunks, self.dim))
        idx_start_to_pad = tensor.size(self.dim) % num_chunks
        if with_padding or contiguous:
            shard_list = []
            for i, shard in enumerate(tensor_list):
                if (
                    with_padding
                    and idx_start_to_pad != 0
                    and i >= idx_start_to_pad
                ):
                    shard = self._pad_tensor(shard)
                # input tensors are expected to be congtiguous by the collective backend
                shard = shard.contiguous() if contiguous else shard
                shard_list.append(shard)
            return shard_list, idx_start_to_pad
        else:
            return tensor_list, idx_start_to_pad

    def _pad_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        # pad tensor by 1 on the shard dim
        pad = [0, 0] * (tensor.ndim - self.dim)
        pad[-1] = 1
        return torch.nn.functional.pad(tensor, pad)

    def _unpad_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        # unpad tensor by 1 on the shard dim
        return tensor.narrow(
            self.dim, start=0, length=tensor.size(self.dim) - 1
        )

    def _local_shard_size_on_dim(
        self,
        size_on_dim: int,
        num_chunks: int,
        rank: int,
        return_offset: bool = False,
    ) -> Tuple[int, int]:
        """
        returns the local shard size and offset on a given tensor dim
        """
        assert (
            size_on_dim >= num_chunks
        ), f"Size to be sharded on dim {self.dim} must be at least as large as the number of devices in that dimension {num_chunks}"
        split_size, pad_idx = divmod(size_on_dim, num_chunks)
        local_shard_size = (
            split_size + 1 if pad_idx != 0 and rank < pad_idx else split_size
        )
        local_offset_on_dim = -1
        if return_offset:
            local_offset_on_dim = (
                rank * split_size + pad_idx if rank >= pad_idx else rank
            )
        return (local_shard_size, local_offset_on_dim)

    def _shard_tensor(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        """
        shard and scatter a tensor on a mesh dimension (use coordinate
        0 on the mesh dimension as source of truth)
        """
        my_coordinate = mesh.get_coordinate_on_dim(mesh_dim)
        num_chunks = mesh.size(dim=mesh_dim)
        # TODO: what should happen if rank is not in the mesh?
        # see issue https://github.com/pytorch/tau/pull/492
        assert (
            my_coordinate is not None
        ), "Rank if not part of mesh"  # TODO: figure out behavior here
        scatter_list, pad_idx = self._split_tensor(
            tensor, num_chunks, with_padding=True, contiguous=True
        )
        output = torch.empty_like(scatter_list[my_coordinate])
        mesh.scatter(output, scatter_list, mesh_dim=mesh_dim)

        if pad_idx != 0 and my_coordinate >= pad_idx:
            output = self._unpad_tensor(output)
        return output

    def _reduce_shard_tensor(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        reduce_op: c10d.ReduceOp,
        mesh_dim: int,
    ) -> torch.Tensor:
        """
        reduce and scatter a tensor on a mesh dimension
        """
        my_coordinate = mesh.get_coordinate_on_dim(mesh_dim)
        num_chunks = mesh.size(dim=mesh_dim)
        # TODO: what should happen if rank is not in the mesh?
        # see issue https://github.com/pytorch/tau/pull/492
        assert (
            my_coordinate is not None
        ), "Rank if not part of mesh"  # TODO: figure out behavior here
        scattered_list, pad_idx = self._split_tensor(
            tensor, num_chunks, with_padding=True, contiguous=True
        )
        # wrap with comm tensor
        scattered_list = [CommTensor(t) for t in scattered_list]
        output = torch.empty_like(scattered_list[my_coordinate])
        mesh.reduce_scatter(
            CommTensor(output),
            scattered_list,  # pyre-ignore[6]
            op=reduce_op,
            mesh_dim=mesh_dim,
        )
        if pad_idx != 0 and my_coordinate >= pad_idx:
            output = self._unpad_tensor(output)
        return output

    def _to_replicate_tensor(
        self,
        local_tensor: torch.Tensor,
        size: torch.Size,
        mesh: DeviceMesh,
        mesh_dim: int,
    ) -> torch.Tensor:
        """
        This function all_gather all shards and return a tensor that
        is replicated on the previously sharded mesh dimension
        """
        my_coordinate = mesh.get_coordinate_on_dim(mesh_dim)
        num_chunks = mesh.size(dim=mesh_dim)
        # TODO: what should happen if rank is not in the mesh?
        # see issue https://github.com/pytorch/tau/pull/492
        assert (
            my_coordinate is not None
        ), "Rank if not part of mesh"  # TODO: figure out behavior here
        # check if it needs to pad input tensor before all_gather
        pad_idx = size[self.dim] % num_chunks
        if pad_idx != 0 and my_coordinate >= pad_idx:
            local_tensor = self._pad_tensor(local_tensor).contiguous()

        gathered_list = []
        # N.B. CommTensor does not change eager mode behavior. During tracing, it
        # makes sure communication result is properly waited before subsequent
        # read operations.
        for _ in range(num_chunks):
            gathered_list.append(
                CommTensor(
                    torch.empty_like(
                        local_tensor,
                        memory_format=torch.contiguous_format,
                    )
                )
            )

        mesh.all_gather(gathered_list, CommTensor(local_tensor.contiguous()), mesh_dim=mesh_dim)  # type: ignore[arg-type]
        # unpad the tensor if the input tensor was padded
        if pad_idx != 0:
            gathered_list = [
                self._unpad_tensor(gathered_tensor)  # type: ignore[misc]
                if i >= pad_idx
                else gathered_tensor
                for i, gathered_tensor in enumerate(gathered_list)
            ]
        return torch.cat(gathered_list, dim=self.dim)  # type: ignore[arg-type]


@dataclass
class Replicate(Placement):
    # replicate placement
    pass


@dataclass
class _Partial(Placement):
    # This is a default partial placement with element-wise reduce op
    # when doing reduction it follows the contract of `_to_replicate`
    # and `_to_shard` to do the reduction and convert the local tensor
    # to the corresponding state (replicate or shard)
    #
    # We can implement custom reductions as needed by subclassing this
    # class and override those contracts.
    reduce_op: c10d.ReduceOp.RedOpType = c10d.ReduceOp.RedOpType.SUM  # type: ignore[attr-defined]

    def _to_replicate(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        # out-of-place all_reduce to replicate, since the current partial DTensor
        # might get used by other ops as well, so we can't inplace modify it
        cloned_local = CommTensor(
            tensor.clone(memory_format=torch.contiguous_format)
        )
        mesh.all_reduce(
            cloned_local, c10d.ReduceOp(self.reduce_op), mesh_dim=mesh_dim  # type: ignore[call-arg]
        )
        return cloned_local

    def _to_shard(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        shard_spec: Placement,
    ) -> torch.Tensor:
        # by default call reduce_shard_tensor of the shard_spec.
        shard_spec = cast(Shard, shard_spec)
        return shard_spec._reduce_shard_tensor(
            tensor, mesh, c10d.ReduceOp(self.reduce_op), mesh_dim  # type: ignore[call-arg]
        )


# used internally to propagate the placements
@dataclass
class DTensorSpec(object):
    mesh: DeviceMesh
    placements: Sequence[Placement]
    # shape of the current dist tensor, this will be set upon
    # construction of the DTensor, prop rule could read it, and
    # would need to set in output spec when calculate the output
    # sharding
    shape: torch.Size
    # ndim of the current dist tensor, if passed in, this would be
    # validated with shape, if not passed in, will be generated from
    # the shape
    ndim: int = -1

    def __post_init__(self) -> None:
        if self.ndim == -1:
            self.ndim = len(self.shape)

    @property
    def dim_map(self) -> List[int]:
        """
        dim_map is a property we derive from `placements` of
        the distributed tensor. It simply return a list of ints
        where dim_map[i] denotes the sharding mapping to the mesh
        dimension, and len(dim_map) == dist_tensor.ndim
        dim_map[i] = -1: means tensor dim i replicate on mesh
        dim_map[i] = j: means tensor dim i shard on mesh dim j

        For example, we have a dist tensor that have the shape of
        [18, 20, 30], and device_mesh([0, 1, 2, 3]), placements:
        [Shard(1)], the dim_map of this placement would be:
        [-1, 1, -1]. This representation is pretty helpful during
        sharding propagation where we could know exactly each
        tensor dimension is sharded or not.

        Note that if placements contains `_Partial`, we have to
        explicitly deal with it, so that when we create a DTensorSpec
        with dim_map, we could properly record the pending sums.
        """
        # dims mapping of dist tensor sharding
        # return size of tensor ndim, -1 represent replicate
        # and int >=0 represent shard on that device mesh dim
        r = [-1] * self.ndim
        for i, placement in enumerate(self.placements):
            if placement.is_shard():
                shard_dim = cast(Shard, placement).dim
                if r[shard_dim] > -1:
                    raise ValueError(
                        f"Tensor dim {shard_dim} is already sharded on mesh dim {r[shard_dim]},"
                        " DTensor operator implementation does not support things like hybrid"
                        " sharding strategies yet (i.e. [Shard(0), Shard(0)])"
                    )
                r[shard_dim] = i
        return r

    @property
    def sums(self) -> List[int]:
        """
        sums is a property we derive from `placements` of the
        distributed tensor. It simply return a list of ints where
        sums[i] denotes the pending sum (partial) on mesh dim i
        """
        return [
            idx
            for idx, placement in enumerate(self.placements)
            if placement.is_partial()
        ]

    @property
    def local_shape(self) -> Tuple[int, ...]:
        """
        Compute the shape of a local shard of the given DTensor on its current
        coordinate of the mesh.
        """
        assert (
            self.shape is not None
        ), "DTensorSpec does not contain global shape."
        local_shape = list(self.shape)  # start with global shape
        for idx, placement in enumerate(self.placements):
            mesh_dim_size = self.mesh.size(idx)
            my_coordinate = self.mesh.get_coordinate_on_dim(idx)
            assert my_coordinate is not None, "Rank not part of mesh!"
            if isinstance(placement, Shard):
                shard_dim = placement.dim
                assert (
                    shard_dim < self.ndim
                ), f"Sharding dim {shard_dim} greater than tensor ndim {self.ndim}"
                local_shard_size, _ = placement._local_shard_size_on_dim(
                    local_shape[shard_dim], mesh_dim_size, my_coordinate
                )
                assert isinstance(local_shard_size, int)
                local_shape[shard_dim] = local_shard_size
        return tuple(local_shape)

    @property
    def local_offsets(self) -> Tuple[int, ...]:
        """
        Compute the offsets of a local shard of the given DTensor on its current
        global rank. This is mostly used by distributed checkpointing to know the
        exact offsets of the local shard.
        """
        assert (
            self.shape is not None
        ), "DTensorSpec does not contain global shape."
        local_offsets = [0] * self.ndim
        local_shape = list(self.shape)

        for idx, placement in enumerate(self.placements):
            mesh_dim_size = self.mesh.size(idx)
            my_coordinate = self.mesh.get_coordinate_on_dim(idx)
            assert my_coordinate is not None, "Rank not part of mesh!"
            if isinstance(placement, Shard):
                shard_dim = placement.dim
                assert (
                    shard_dim < self.ndim
                ), f"Sharding dim {shard_dim} greater than tensor ndim {self.ndim}"
                shard_size, shard_offset = placement._local_shard_size_on_dim(
                    local_shape[shard_dim],
                    mesh_dim_size,
                    my_coordinate,
                    return_offset=True,
                )
                local_shape[shard_dim] = shard_size
                local_offsets[shard_dim] = shard_offset
        return tuple(local_offsets)

    @classmethod
    def from_dim_map(
        cls,
        mesh: DeviceMesh,
        dim_map: List[int],
        sums: List[int],
        shape: torch.Size,
    ) -> "DTensorSpec":
        """
        Construct a DTensorSpec from dim_map list and pending sum.

        Args:
            mesh (class:`DeviceMesh`): device mesh to be used in the DTensorSpec
            dim_map (List[int]): a list of integer that represents sharding on each
                tensor dimension, see `dim_map` property doc for details
            sums (List[int]): a list of integer that represents the dist tensor have
                pending sum on which device mesh dimension.
            shape (torch.Size): shape of the DTensor associated with this spec.

        Return:
            a class:`DTensorSpec` object
        """
        # by default replicate on device mesh dims
        placements: List[Placement] = [Replicate() for _ in range(mesh.ndim)]

        # find all mesh dims that need pending reductions
        for s in sums:
            placements[s] = _Partial()

        for i, m in enumerate(dim_map):
            if m >= 0:
                placement = placements[m]
                if placement.is_shard():
                    placement = cast(Shard, placement)
                    raise RuntimeError(
                        f"DeviceMesh dimension cann't be mapped to two dimension of the same tensor: {i} and {placement.dim}"
                    )
                elif placement.is_partial():
                    raise RuntimeError(
                        f"DeviceMesh dimension {m} cannot be both shard and partial!"
                    )
                placements[m] = Shard(i)

        return cls(mesh, placements, shape=shape, ndim=len(dim_map))
