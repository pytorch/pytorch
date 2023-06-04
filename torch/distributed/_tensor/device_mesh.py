# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
from typing import List, Optional, TYPE_CHECKING, Union

import torch
import torch.distributed._functional_collectives as funcol

from torch.distributed.distributed_c10d import (
    _get_default_group,
    all_gather,
    all_to_all,
    broadcast,
    get_global_rank,
    get_rank,
    get_world_size,
    GroupMember,
    init_process_group,
    is_initialized,
    new_group,
    ProcessGroup,
    ReduceOp,
    scatter,
    Work,
)


logger = logging.getLogger(__name__)

# only import numpy typing when type checking
if TYPE_CHECKING:
    try:
        from numpy.typing import ArrayLike
    except ImportError:
        logger.warning(
            "DeviceMesh requires numpy >= 1.21 to be installed for type checking"
        )


class _MeshEnv(object):
    def __init__(self) -> None:
        self.mesh_stack: List[DeviceMesh] = []

    def get_current_mesh(self) -> "DeviceMesh":
        if len(self.mesh_stack) == 0:
            raise RuntimeError("No device mesh is currently active!")
        return self.mesh_stack[-1]


mesh_resources: _MeshEnv = _MeshEnv()


class DeviceMesh(object):
    """
    DeviceMesh represents a mesh of devices, where layout of devices could be
    represented as a n-d dimension array, and each value of the n-d dimensional
    array is the global id of the default process group ranks.

    DeviceMesh could be used to describe the layout of devices across the cluster,
    and serves as a proxy for communication among the device lists within the cluster.

    We use the default ProcessGroup in this DeviceMesh class to implement proper
    communications. Note that we also add collective wrappers in this class. This is
    used to decouple detailed communication backend with the underlying
    DTensor implementation.

    DeviceMesh can be used as a context manager.
    Args:
        device_type (str): device type of the mesh. Currently supports: cpu, cuda.
        mesh (ndarray): could be a multi-dimension array or an integer tensor that
            describes the layout of devices, the ids are global ids of the
            default process group.

    Returns:
        A :class:`DeviceMesh` object

    Example (2 host with 4 GPUs each):
        ```
        # The following program runs on each process/rank in SPMD manner.
        # initialize device mesh as (2, 4) to represent the topology
        # of cross-host(dim 0), and within-host (dim 1)
        mesh = DeviceMesh(device_type="cuda",
                          mesh=[
                            [0, 1, 2, 3],
                            [4, 5, 6, 7]
                          ])
        ```
        A reduction over the first dimension of mesh will reduce across
        columns (0, 4), .. and (3, 7), a reduction over the second dimension
        of mesh reduces across rows (0, 1, 2, 3) and (4, 5, 6, 7)

    """

    device_type: str
    mesh: torch.Tensor

    def __init__(
        self,
        device_type: str,
        mesh: Union[torch.Tensor, "ArrayLike"],
        *,
        _init_process_groups: bool = True,
    ) -> None:
        self.device_type = device_type
        self.mesh = (
            mesh.detach()
            if isinstance(mesh, torch.Tensor)
            else torch.tensor(mesh, dtype=torch.int)
        )
        # always try to create default (world) pg, even if it is not initialized
        # already. The world pg is used for device mesh identity (rank) on each
        # process (we need to know if the current global rank is in the mesh or not)
        self._get_or_create_default_group()
        if _init_process_groups:
            self._dim_groups = self._init_process_groups()

    def _get_or_create_default_group(self):
        default_initialized = is_initialized()
        if not default_initialized:
            init_process_group()

        world_size = get_world_size()
        if self.mesh.numel() > world_size:
            raise RuntimeError(
                f"Mesh should not be bigger than default world size, but found {self.mesh.numel()} ranks!"
            )

        # TODO: if user want to pass pg_options, offer a way to do it
        if not default_initialized and self.device_type == "cuda":
            # automatically set the current cuda device base on num of gpu devices available in each host
            # NOTE: This device selection would only work for homogeneous hardware.
            num_gpus_per_host = torch.cuda.device_count()
            if world_size % num_gpus_per_host != 0:
                raise RuntimeError(
                    f"DeviceMesh only support homogeneous hardware, but found "
                    f"{world_size} ranks and {num_gpus_per_host} cuda devices!"
                )
            torch.cuda.set_device(get_rank() % num_gpus_per_host)
        # TODO (xilunwu): to perform DTensor random ops, we need to ensure all ranks in mesh is initialized
        # with the same random seed. The seed to use will be the current seed on rank 0. We store this seed
        # as an attribute of device mesh for future use. However, the detail is still TBD how we gonna use
        # this attribute, so we will implement this logic once we figure out the answer.
        self._seed = torch.initial_seed()

        # calculate the coordinates of the current global rank on the mesh
        rank_coords = (self.mesh == get_rank()).nonzero()
        assert rank_coords.size(0) in (0, 1)
        self._coordinate_on_dim: Optional[List[int]] = (
            rank_coords[0].tolist() if rank_coords.size(0) > 0 else None
        )
        return _get_default_group()

    def _init_process_groups(self):
        default_pg = _get_default_group()
        # check mesh tensor validity
        unique_mesh_values = self.mesh.unique(sorted=True)
        if unique_mesh_values.numel() != self.mesh.numel():
            raise RuntimeError(
                f"DeviceMesh cannot have duplicate values, but found {self.mesh.tolist()}"
            )
        # validate that all calling ranks pass in the same `mesh` argument.
        self_mesh = self.mesh.to(self.device_type)
        mesh_list = [self_mesh.clone() for _ in range(get_world_size())]
        all_gather(mesh_list, self_mesh)
        for other_rank, other_mesh in enumerate(mesh_list):
            if not torch.equal(self_mesh, other_mesh):
                raise RuntimeError(
                    f"DeviceMesh initialization does not allow different mesh argument:"
                    f"rank {get_rank()} has mesh {self_mesh} while rank {other_rank}"
                    f"has mesh {other_mesh}!"
                )

        # groups created by dimension, each dimension should have exact
        # one valid process group per rank
        dim_groups: List[ProcessGroup] = []

        if self.mesh.ndim == 1 and len(unique_mesh_values) == get_world_size():
            # if the mesh is the same as world_pg, we just append the default
            # pg to the first dim groups, as new_group cannot have the exact
            # same ranks as world
            dim_groups.append(default_pg)
        else:
            # create sub pgs base on the mesh argument specified
            for dim in range(self.mesh.ndim):
                # swap the current dim to the last dim
                # then reshape to flatten out other dims
                pg_ranks_by_dim = self.mesh.swapdims(-1, dim).reshape(
                    -1, self.mesh.size(dim)
                )
                # multi-dim mesh, create subgroups by looping over the pg_ranks
                # for each dim and append the groups
                for dim_mesh in pg_ranks_by_dim:
                    subgroup_ranks = dim_mesh.tolist()
                    # call new_group regardless of the current rank in the
                    # pg or not, it's required that all ranks participate
                    # in subgroup construction
                    new_subgroup = new_group(ranks=subgroup_ranks)
                    # only add to dim_groups if the current rank in the subgroup
                    if self.get_rank() in subgroup_ranks:
                        if len(dim_groups) > dim:
                            raise RuntimeError(
                                f"Each device mesh dimension should get only one process group, but got {self.get_rank} "
                                f"in {subgroup_ranks}!"
                            )
                        dim_groups.append(new_subgroup)
        return dim_groups

    def __enter__(self) -> "DeviceMesh":
        # set this mesh as the current mesh in mesh env
        mesh_resources.mesh_stack.append(self)
        return self

    # pyre-fixme[2]: Parameter must be annotated.
    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        # pop this mesh from mesh env
        mesh_resources.mesh_stack.pop()

    def __repr__(self) -> str:
        return f"DeviceMesh:({self.mesh.tolist()})"

    def __hash__(self):
        return hash((self.mesh, id(self)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DeviceMesh):
            return False
        if id(self) == id(other):
            return True
        return self.mesh.equal(other.mesh)

    def get_dim_groups(self) -> List[ProcessGroup]:
        if not hasattr(self, "_dim_groups"):
            raise RuntimeError("DeviceMesh process groups not initialized!")
        return self._dim_groups

    def size(self, dim: Optional[int] = None) -> int:
        return self.mesh.numel() if dim is None else self.mesh.size(dim)

    @property
    def ndim(self) -> int:
        return self.mesh.ndim

    def get_rank(self) -> int:
        return get_rank()

    def get_coordinate(self) -> Optional[List[int]]:
        """
        Return the relative indices of this rank relative to all
        dimensions of the mesh. If this rank is not part of the mesh, return None.
        """
        return self._coordinate_on_dim if self._coordinate_on_dim else None

    def scatter(
        self,
        output: torch.Tensor,
        scatter_list: List[torch.Tensor],
        mesh_dim: int = 0,
        async_op: bool = False,
    ) -> Optional[Work]:
        """
        scatter a list of tensors to a device mesh dimension. We by default
        use the first rank of the mesh dimension as the source of truth, i.e
        for a 2d mesh [[0, 1], [2, 3]], if we scatter on mesh_dim = 1, we will
        scatter the tensor list on rank 0 to rank 0/1, and tensor list on rank
        2 to rank 2/3.

        Args:
            output (torch.Tensor): the tensor to receive the scattered list.
            scatter_list (List[torch.Tensor]): the tensor list to be scattered.
            mesh_dim (int, optional): indicate which mesh dimension we want
                to scatter on, we by default choose the first rank on the
                mesh dimension as source of truth.

        Returns:
            A :class:`Work` object
        """
        # TODO: Ideally we should use the meta tensor way
        # (to register a meta kernel for the collective op)
        # so that it would avoid the communication. Need to
        # remove the check below once that is done.
        if output.is_meta:
            return None
        dim_group = self._dim_groups[mesh_dim]
        # src need to be global rank
        src_for_dim = 0
        if dim_group is not GroupMember.WORLD:
            src_for_dim = get_global_rank(dim_group, 0)

        if src_for_dim == get_rank():
            fut = scatter(
                output,
                scatter_list=scatter_list,
                src=src_for_dim,
                group=dim_group,
                async_op=async_op,
            )
        else:
            fut = scatter(
                output,
                scatter_list=None,
                src=src_for_dim,
                group=dim_group,
                async_op=async_op,
            )

        return fut

    def broadcast(
        self,
        tensor: torch.Tensor,
        mesh_dim: int = 0,
        async_op: bool = False,
    ) -> Optional[Work]:
        """
        broadcast the tensor to a device mesh dimension. We by default
        use the first rank of the mesh dimension as the source of truth, i.e
        for a 2d mesh [[0, 1], [2, 3]], if we broadcast on mesh_dim = 1, we will
        broadcast the tensor on rank 0 to rank 0/1, and tensor on rank 2
        to rank 2/3.

        Args:
            tensor (torch.Tensor): tensor to broadcast.
            mesh_dim (int, optional): indicate which mesh dimension we want
                to scatter on, we by default choose the first rank on the
                mesh dimension as source of truth.

        Returns:
            A :class:`Work` object
        """
        # TODO: Ideally we should use the meta tensor way
        # (to register a meta kernel for the collective op)
        # so that it would avoid the communication. Need to
        # remove the check below once that is done.
        if tensor.is_meta:
            return None
        dim_group = self._dim_groups[mesh_dim]
        # src need to be global rank
        src_for_dim = 0
        if dim_group is not GroupMember.WORLD:
            src_for_dim = get_global_rank(dim_group, 0)

        return broadcast(tensor, src=src_for_dim, group=dim_group, async_op=async_op)

    def all_gather(
        self,
        tensor: torch.Tensor,
        mesh_dim: int = 0,
        gather_dim: int = 0,
    ) -> torch.Tensor:
        """
        all_gather the tensor on each rank to a bigger tensor on a
        device mesh dimension.

        Args:
            tensor (torch.Tensor): tensor to be gathered on each rank.
            mesh_dim (int, optional): indicate which mesh dimension we want
                to scatter on, we by default choose the first rank on the
                mesh dimension as source of truth.
            gather_dim (int, optional): Dimension to concatenate the resulting tensor.

        Returns:
            A :class:`AsyncCollectiveTensor` object
        """
        dim_group = self._dim_groups[mesh_dim]
        return funcol.all_gather_tensor(tensor, gather_dim=gather_dim, group=dim_group)

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: ReduceOp.RedOpType = ReduceOp.SUM,
        mesh_dim: int = 0,
    ) -> torch.Tensor:
        """
        all_reduce the tensor on each rank on a device mesh dimension, and
        return an output tensor on each rank after all_reduce.

        Args:
            tensor (torch.Tensor): tensor to be all_reduced on each rank.
            op (:class:`torch.distributed.distributed_c10d.ReduceOp, optional):
                the reduction op of all_reduce (i.e. ReduceOp.SUM)
            mesh_dim (int, optional): indicate which mesh dimension we want
                to reduce on.

        Returns:
            A :class:`AsyncCollectiveTensor` object
        """
        return funcol.all_reduce(tensor, reduceOp=op.name, group=(self, mesh_dim))

    def reduce_scatter(
        self,
        input: torch.Tensor,
        op: ReduceOp.RedOpType = ReduceOp.SUM,
        mesh_dim: int = 0,
        scatter_dim: int = 0,
    ) -> torch.Tensor:
        """
        reduce the input on each rank on a device mesh dimension, and scatter
        the results as output tensor on each rank.

        Args:
            input (torch.Tensor): tensor to be reduced and scattered
                and scattered on each rank.
            op (:class:`torch.distributed.distributed_c10d.ReduceOp, optional):
                the reduction op of reduce_scatter (i.e. ReduceOp.SUM)
            mesh_dim (int, optional): indicate which mesh dimension we want
                to scatter on.

        Returns:
            A :class:`torch.Tensor` object
        """
        if self.device_type == "cpu":
            # cpu::gloo backend does not have reduce_scatter we fallback to do all_reduce
            # + local chunk
            logger.warning(
                "ProcessGroupGloo does not support reduce_scatter, falling back with all reduce!"
            )
            dim_group = self._dim_groups[mesh_dim]
            group_size = get_world_size(dim_group)
            group_rank = get_rank(dim_group)
            if scatter_dim != 0:
                tensor_list = torch.chunk(input, group_size, dim=scatter_dim)
                input = torch.cat(tensor_list)

            flat_tensor = funcol.all_reduce(input, reduceOp=op.name, group=dim_group)
            chunks = flat_tensor.chunk(group_size, dim=0)
            scatter_tensor = chunks[group_rank]
        else:
            dim_group = self._dim_groups[mesh_dim]
            scatter_tensor = funcol.reduce_scatter_tensor(
                input, reduceOp=op.name, scatter_dim=scatter_dim, group=dim_group
            )

        return scatter_tensor

    # TODO: test uneven split on GLOO and NCCL
    def all_to_all(
        self,
        output_tensor_list: List[torch.Tensor],
        input_tensor_list: List[torch.Tensor],
        mesh_dim: int = 0,
        async_op: bool = False,
    ) -> Optional[Work]:
        dim_group = self._dim_groups[mesh_dim]

        work = None
        # no direct dist.all_to_all support on 'gloo' so we manually do scatters
        if self.device_type == "cpu":
            logger.warning(
                "ProcessGroupGloo does not support all_to_all, falling back with scatters!"
            )
            # TODO: pull the handle of uneven case in #492
            dim_group_size = get_world_size(dim_group)
            for i in range(dim_group_size):
                # src need to be global rank
                src_for_dim = i
                if dim_group is not GroupMember.WORLD:
                    src_for_dim = get_global_rank(dim_group, i)

                work = scatter(
                    output_tensor_list[i],
                    input_tensor_list if self.get_rank() == src_for_dim else [],
                    group=dim_group,
                    src=src_for_dim,
                    async_op=async_op,
                )
        else:
            work = all_to_all(
                output_tensor_list,
                input_tensor_list,
                dim_group,
                async_op=async_op,
            )
        return work
