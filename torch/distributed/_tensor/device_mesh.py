# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
from typing import List, Optional, Tuple, TYPE_CHECKING, Union

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol

from torch.distributed.distributed_c10d import (
    _find_pg_by_ranks_and_tag,
    _get_default_group,
    _get_group_tag,
    get_rank,
    get_world_size,
    init_process_group,
    is_initialized,
    new_group,
    ProcessGroup,
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


class _MeshEnv:
    def __init__(self) -> None:
        self.mesh_stack: List[DeviceMesh] = []

    def get_current_mesh(self) -> "DeviceMesh":
        if len(self.mesh_stack) == 0:
            raise RuntimeError("No device mesh is currently active!")
        return self.mesh_stack[-1]


mesh_resources: _MeshEnv = _MeshEnv()


def _get_device_handle(device_type: str = "cuda"):
    """
    Get the module corresponding to the device_type which is cuda or cuda-like device.
    For example, when the device_type is cuda, the module `torch.cuda` is returned.
    Return None when device_type is cpu or there is no corresponding module,
    otherwise return the corresponding module.
    """
    return getattr(torch, device_type, None) if device_type != "cpu" else None


class DeviceMesh:
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
        device_type (str): device type of the mesh. Currently supports: cpu, cuda/cuda-like.
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
        mesh_dim_names: Optional[Tuple[str]] = None,
        _init_process_groups: bool = True,
        _validate_mesh: bool = True,
    ) -> None:
        self.device_type = device_type
        self.mesh = (
            mesh.detach()
            if isinstance(mesh, torch.Tensor)
            else torch.tensor(mesh, dtype=torch.int)
        )
        self.mesh_dim_names = mesh_dim_names
        # always try to create default (world) pg, even if it is not initialized
        # already. The world pg is used for device mesh identity (rank) on each
        # process (we need to know if the current global rank is in the mesh or not)
        self._get_or_create_default_group()
        if _init_process_groups:
            self._init_process_groups(_validate_mesh)

    def _get_or_create_default_group(self):
        default_initialized = is_initialized()
        if not default_initialized:
            init_process_group()

        world_size = get_world_size()
        if self.mesh.numel() > world_size:
            raise RuntimeError(
                f"Mesh should not be bigger than default world size, but found {self.mesh.numel()} ranks!"
            )

        device_handle = _get_device_handle(self.device_type)
        # TODO: if user want to pass pg_options, offer a way to do it
        if not default_initialized and device_handle:
            # automatically set the current cuda/cuda-like device base on num of gpu devices available in each host
            # NOTE: This device selection would only work for homogeneous hardware.
            num_devices_per_host = device_handle.device_count()
            if world_size % num_devices_per_host != 0:
                raise RuntimeError(
                    f"DeviceMesh only support homogeneous hardware, but found "
                    f"{world_size} ranks and {num_devices_per_host} {self.device_type} devices!"
                )
            device_handle.set_device(get_rank() % num_devices_per_host)

        # calculate the coordinates of the current global rank on the mesh
        rank_coords = (self.mesh == get_rank()).nonzero()
        assert rank_coords.size(0) in (0, 1)
        self._coordinate_on_dim: Optional[List[int]] = (
            rank_coords[0].tolist() if rank_coords.size(0) > 0 else None
        )
        return _get_default_group()

    def _validate_mesh(self):
        # check mesh tensor validity
        unique_mesh_values = self.mesh.unique(sorted=True)
        if unique_mesh_values.numel() != self.mesh.numel():
            raise RuntimeError(
                f"DeviceMesh cannot have duplicate values, but found {self.mesh.tolist()}"
            )

        # validate that all calling ranks pass in the same `mesh` argument.
        self_mesh = self.mesh.to(self.device_type)
        mesh_tensor = funcol.all_gather_tensor(
            self_mesh, gather_dim=0, group=_get_default_group()
        )
        mesh_tensor_chunked = torch.chunk(mesh_tensor, get_world_size())
        for other_rank, other_mesh in enumerate(mesh_tensor_chunked):
            if not torch.equal(self_mesh, other_mesh):
                raise RuntimeError(
                    f"DeviceMesh initialization does not allow different mesh argument:"
                    f"rank {get_rank()} has mesh {self_mesh} while rank {other_rank}"
                    f"has mesh {other_mesh}!"
                )

    def _init_process_groups(self, _validate_mesh):
        if _validate_mesh:
            self._validate_mesh()

        # group tag/ranks associated with each mesh dimension, each mesh dimension should
        # have one sub-group per rank
        dim_group_infos: List[Tuple[str, List[int]]] = []

        if self.mesh.ndim == 1 and self.mesh.numel() == get_world_size():
            # if the mesh is the same as world_pg, we just append the default
            # pg to the first dim groups, as new_group cannot have the exact
            # same ranks as world
            dim_group_infos.append(
                (_get_group_tag(_get_default_group()), list(range(get_world_size())))
            )
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
                    dim_group = new_group(ranks=subgroup_ranks)
                    # only add to dim_groups if the current rank in the subgroup
                    if self.get_rank() in subgroup_ranks:
                        if len(dim_group_infos) > dim:
                            raise RuntimeError(
                                f"Each device mesh dimension should get only one process group, but got {self.get_rank} "
                                f"in {subgroup_ranks}!"
                            )
                        dim_group_infos.append(
                            (_get_group_tag(dim_group), subgroup_ranks)
                        )
        self._dim_group_infos = dim_group_infos

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

    def get_dim_groups(
        self, mesh_dim: Optional[int] = None
    ) -> Union[ProcessGroup, List[ProcessGroup]]:
        if not hasattr(self, "_dim_group_infos"):
            raise RuntimeError("DeviceMesh process groups not initialized!")
        if mesh_dim is not None:
            return _find_pg_by_ranks_and_tag(*self._dim_group_infos[mesh_dim])
        else:
            dim_groups = []
            for mesh_dim in range(self.mesh.ndim):
                dim_groups.append(
                    _find_pg_by_ranks_and_tag(*self._dim_group_infos[mesh_dim])
                )
            return dim_groups

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


def init_device_mesh(
    device_type: str,
    mesh_shape: Tuple[int],
    *,
    mesh_dim_names: Optional[Tuple[str]] = None,
) -> DeviceMesh:
    """
    Initializes a `DeviceMesh` based on `device_type`, `mesh_shape`, and `mesh_dim_names` parameters.
    This creates a DeviceMesh with a mesh layout of n-d dimensional array, n being the len(mesh_shape)
    and ith dimension being in size mesh_shape[i]. If mesh_dim_names is provided, each dimension is
    labeled as mesh_dim_names[i].


    Args:
        device_type (str): device type of the mesh. Currently supports: cpu, cuda/cuda-like.
        mesh_shape: Tuple[int]: A tuple describes the dimension of the multi-dimesnion array
        that describes the layout of devices.
    Kwargs:
        mesh_dim_names: Optional[Tuple[str]]: A tuple of mesh dim names to be assigned to each dimension
        of the multi-dimensional array that describes the layout of devices. Its length must match the length
        of `mesh_shape`.

    Returns:
        A :class:`DeviceMesh` object

    .. note: If no process group is found, init_device_mesh will initialize distributed process group/groups
    behind the scene, which are requried for distributed communications.

    Example:
        >>> # xdoctest: +SKIP
        >>> two_d_mesh = init_device_mesh("cuda", mesh_shape=(2, 8), mesh_dim_names=("dp", "tp"))
        >>> two_d_mesh = init_device_mesh("cuda", mesh_shape=(2, -1), mesh_dim_names=("dp", "tp"))
    """
    if mesh_dim_names is not None and len(mesh_shape) != len(mesh_dim_names):
        raise RuntimeError(
            f"Please provide a mesh_dim_name to each mesh_dim! Found {len(mesh_dim_names)} instead of {len(mesh_shape)}."
        )

    mesh = torch.arange(dist.get_world_size()).view(mesh_shape)
    device_mesh = DeviceMesh(
        device_type=device_type,
        mesh=mesh,
        mesh_dim_names=mesh_dim_names,
    )

    return device_mesh
