# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import torch

from torch.distributed import is_available

__all__ = ["init_device_mesh", "DeviceMesh"]


if not is_available():
    import sys

    # We need to create the stubs when distributed is not available.
    # Otherwise, we would fail the doc tests (```./.ci/pytorch/docs-test.sh```),
    # since it would try to import ``torch.distributed.device_mesh`` or
    # ``torch.distributed.init_device_mesh`` but cannot find them.

    class _DeviceMeshStub:
        pass

    def _init_device_mesh_stub():
        pass

    sys.modules["torch.distributed.device_mesh"].DeviceMesh = _DeviceMeshStub  # type: ignore[attr-defined]
    sys.modules[
        "torch.distributed.device_mesh"
    ].init_device_mesh = _init_device_mesh_stub  # type: ignore[attr-defined]


else:
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
            self.child_to_parent_mapping: Dict[DeviceMesh, DeviceMesh] = {}

        def get_current_mesh(self) -> "DeviceMesh":
            if len(self.mesh_stack) == 0:
                raise RuntimeError("No device mesh is currently active!")
            return self.mesh_stack[-1]

        def create_child_mesh(
            self, device_mesh: "DeviceMesh", mesh_dim: int, mesh_dim_name: str
        ) -> "DeviceMesh":
            # swap the current dim to the last dim then reshape to flatten out other
            # dims, so we can just extract the list of ranks which contains cur_rank.
            cur_rank = device_mesh.get_rank()
            pg_ranks_by_dim = device_mesh.mesh.swapdims(-1, mesh_dim).reshape(
                -1, device_mesh.mesh.size(mesh_dim)
            )

            for mesh_1d in pg_ranks_by_dim:
                sub_mesh = DeviceMesh(
                    device_mesh.device_type,
                    mesh_1d,
                    mesh_dim_names=(mesh_dim_name,),
                    _init_process_groups=False,
                )
                if cur_rank in mesh_1d:
                    res_sub_mesh = sub_mesh

            res_sub_mesh._dim_group_infos = [device_mesh._dim_group_infos[mesh_dim]]
            # Assign the current DeviceMesh as the parent of the child DeviceMesh.
            self.child_to_parent_mapping[res_sub_mesh] = device_mesh
            return res_sub_mesh

        def get_parent_mesh(self, device_mesh: "DeviceMesh") -> Optional["DeviceMesh"]:
            return self.child_to_parent_mapping.get(device_mesh, None)

        def get_parent_mesh_dim(self, device_mesh: "DeviceMesh") -> Optional[int]:
            """
            Return the index of the mesh dim in the parent mesh.
            The device_mesh passed in needs to be sliced out from a parent mesh.
            """
            parent_mesh = self.get_parent_mesh(device_mesh)
            child_mesh_dim_names = device_mesh.mesh_dim_names
            if parent_mesh and child_mesh_dim_names:
                assert (
                    len(child_mesh_dim_names) == 1
                ), "The child mesh can only be a 1D mesh."
                child_mesh_dim_name = child_mesh_dim_names[0]
                if parent_mesh.mesh_dim_names:
                    return parent_mesh._get_mesh_dim_by_name(child_mesh_dim_name)
            return None

        @staticmethod
        def num_devices_per_host(device_type: str) -> int:
            return _get_device_handle(device_type).device_count()

        @staticmethod
        def num_hosts(device_type: str) -> int:
            # ProcessGroup can't tell us this info so we have to infer it, assume
            # homogeneous hardware for now
            return get_world_size() // _MeshEnv.num_devices_per_host(device_type)

    _mesh_resources: _MeshEnv = _MeshEnv()

    def _get_device_handle(device_type: str = "cuda"):
        """
        Get the module corresponding to the device_type which is cuda or cuda-like device.
        For example, when the device_type is cuda, the module `torch.cuda` is returned.
        Return None when there is no corresponding module for device_type, otherwise
        return the corresponding module.
        """
        return getattr(torch, device_type, None)

    class DeviceMesh:
        """
        DeviceMesh represents a mesh of devices, where layout of devices could be
        represented as a n-d dimension array, and each value of the n-d dimensional
        array is the global id of the default process group ranks.

        DeviceMesh could be used to describe the layout of devices across the cluster,
        and serves as a proxy for communication among the device lists within the cluster.

        DeviceMesh can be used as a context manager.

        .. note::
            DeviceMesh follows SPMD programming model, which means the same PyTorch Python program
            is running on all processes/ranks in the cluster. Therefore, users need to make sure the
            `mesh` array (which describes the layout of devices) should be identical across all ranks.
            Inconsistent `mesh` will lead to silent hang.

        Args:
            device_type (str): The device type of the mesh. Currently supports: "cpu", "cuda/cuda-like".
            mesh (ndarray): A multi-dimensional array or an integer tensor describing the layout
                of devices, where the IDs are global IDs of the default process group.

        Returns:
            DeviceMesh: A :class:`DeviceMesh` object representing the device layout.

        The following program runs on each process/rank in an SPMD manner. In this example, we have 2
        hosts with 4 GPUs each.
        A reduction over the first dimension of mesh will reduce across
        columns (0, 4), .. and (3, 7), a reduction over the second dimension
        of mesh reduces across rows (0, 1, 2, 3) and (4, 5, 6, 7).

        Example::
            >>> # xdoctest: +SKIP("no rank")
            >>> from torch.distributed.device_mesh import DeviceMesh
            >>>
            >>> # Initialize device mesh as (2, 4) to represent the topology
            >>> # of cross-host(dim 0), and within-host (dim 1).
            >>> mesh = DeviceMesh(device_type="cuda", mesh=[[0, 1, 2, 3],[4, 5, 6, 7]])
        """

        device_type: str
        mesh: torch.Tensor
        mesh_dim_names: Optional[Tuple[str, ...]]

        def __init__(
            self,
            device_type: str,
            mesh: Union[torch.Tensor, "ArrayLike"],
            *,
            mesh_dim_names: Optional[Tuple[str, ...]] = None,
            _init_process_groups: bool = True,
        ) -> None:
            self.device_type = device_type
            self.mesh = (
                mesh.detach()
                if isinstance(mesh, torch.Tensor)
                else torch.tensor(mesh, dtype=torch.int)
            )
            self.mesh_dim_names = mesh_dim_names

            # private field to pre-generate DeviceMesh's hash
            self._flatten_mesh_list = tuple(self.mesh.flatten().tolist())
            self._hash = hash((self._flatten_mesh_list, self.mesh.shape, id(self)))

            # Skip process group initialization if xla device.
            # TODO(yeounoh) implement DeviceMesh backend and register XLA backend.
            if device_type != "xla":
                # always try to create default (world) pg, even if it is not initialized
                # already. The world pg is used for device mesh identity (rank) on each
                # process (we need to know if the current global rank is in the mesh or not).
                self._get_or_create_default_group()
                if _init_process_groups:
                    self._init_process_groups()

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
                if (
                    world_size > num_devices_per_host
                    and world_size % num_devices_per_host != 0
                ):
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

        def _init_process_groups(self):
            # group tag/ranks associated with each mesh dimension, each mesh dimension should
            # have one sub-group per rank
            dim_group_infos: List[Tuple[str, List[int]]] = []

            if self.mesh.ndim == 1 and self.mesh.numel() == get_world_size():
                # if the mesh is the same as world_pg, we just append the default
                # pg to the first dim groups, as new_group cannot have the exact
                # same ranks as world
                dim_group_infos.append(
                    (
                        _get_group_tag(_get_default_group()),
                        list(range(get_world_size())),
                    )
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
            _mesh_resources.mesh_stack.append(self)
            return self

        # pyre-fixme[2]: Parameter must be annotated.
        def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
            # pop this mesh from mesh env
            _mesh_resources.mesh_stack.pop()

        def __repr__(self) -> str:
            return f"DeviceMesh({self.mesh.tolist()})"

        def __hash__(self):
            return self._hash

        def __eq__(self, other: object) -> bool:
            if not isinstance(other, DeviceMesh):
                return False
            if id(self.mesh) == id(other.mesh):
                return True
            return (
                self.mesh.shape == other.mesh.shape
                and self._flatten_mesh_list == other._flatten_mesh_list
            )

        def __getitem__(self, mesh_dim_name: str) -> "DeviceMesh":
            """
            Slice the current DeviceMesh based on the mesh_dim_name given to create a child
            DeviceMesh.

            Args:
                mesh_dim_name (str): the name of the mesh dimension of the parent DeviceMesh
                to create a child DeviceMesh for.
            Returns:
                A :class:`DeviceMesh` object

            The following program runs on each process/rank in an SPMD manner. In this example, we have 2
            hosts with 4 GPUs each.
            Calling mesh["tp"] on rank 0, 1, 2, 3 would return a 1D child DeviceMesh:([0, 1, 2, 3]).
            Calling mesh["tp"] on rank 4, 5, 6, 7 would return a 1D child DeviceMesh:([4, 5, 6, 7]).
            Calling mesh["dp"] on rank 0, 4 would return a 1D child DeviceMesh:([0, 4]).
            Calling mesh["dp"] on rank 1, 5 would return a 1D child DeviceMesh:([1, 5]).
            Calling mesh["dp"] on rank 2, 6 would return a 1D child DeviceMesh:([2, 6]).
            Calling mesh["dp"] on rank 3, 7 would return a 1D child DeviceMesh:([3, 7]).

            Example::
                >>> # xdoctest: +SKIP("no rank")
                >>> from torch.distributed.device_mesh import DeviceMesh
                >>>
                >>> # Initialize device mesh as (2, 4) to represent the topology
                >>> # of cross-host(dim 0), and within-host (dim 1).
                >>> mesh = DeviceMesh(device_type="cuda", mesh=[[0, 1, 2, 3],[4, 5, 6, 7]])
            """
            if self.mesh.ndim <= 1:
                raise RuntimeError(
                    f"Cannot slice a DeviceMesh with {self.mesh.ndim} dimension."
                )
            mesh_dim = self._get_mesh_dim_by_name(mesh_dim_name)
            submesh = _mesh_resources.create_child_mesh(self, mesh_dim, mesh_dim_name)

            return submesh

        def get_group(
            self, mesh_dim: Optional[Union[int, str]] = None
        ) -> Union[ProcessGroup, List[ProcessGroup]]:
            """
            Returns a list of ProcessGroups corresponding to the mesh dimensions, or
            returns a single ProcessGroup if mesh_dim is specified or the given mesh has
            only one mesh dimension.

            Args:
                mesh_dim (str/int, optional): it can be the name of the mesh dimension or the index
                of the mesh dimension. Default is None.

            Returns:
                A list of :class:`ProcessGroup` object when `mesh_dim` is not specified for
                a DeviceMesh with more than 1 dimension; otherwise, returns a single
                :class:`ProcessGroup` object.
            """
            if not hasattr(self, "_dim_group_infos"):
                raise RuntimeError("DeviceMesh process groups not initialized!")

            if self.mesh.ndim == 1:
                return _find_pg_by_ranks_and_tag(*self._dim_group_infos[0])

            if mesh_dim is not None:
                if isinstance(mesh_dim, str):
                    mesh_dim = self._get_mesh_dim_by_name(mesh_dim)
                return _find_pg_by_ranks_and_tag(*self._dim_group_infos[mesh_dim])
            else:
                dim_groups = []
                for ith_dim in range(self.mesh.ndim):
                    dim_groups.append(
                        _find_pg_by_ranks_and_tag(*self._dim_group_infos[ith_dim])
                    )
                return dim_groups

        def size(self, mesh_dim: Optional[int] = None) -> int:
            return self.mesh.numel() if mesh_dim is None else self.mesh.size(mesh_dim)

        @property
        def ndim(self) -> int:
            return self.mesh.ndim

        @property
        def shape(self) -> Tuple[int, ...]:
            return tuple(self.mesh.shape)

        def get_rank(self) -> int:
            """
            Returns the current global rank.
            """
            return get_rank()

        def get_local_rank(self, mesh_dim: Optional[Union[int, str]] = None) -> int:
            """
            Returns the local rank of the given mesh_dim of the DeviceMesh.

            Args:
                mesh_dim (str/int, optional): it can be the name of the mesh dimension or the index
                of the mesh dimension. Default is None.

            Returns:
                An integer denotes the local rank.

            The following program runs on each process/rank in an SPMD manner. In this example, we have 2
            hosts with 4 GPUs each.
            Calling mesh_2d.get_local_rank(mesh_dim=0) on rank 0, 1, 2, 3 would return 0.
            Calling mesh_2d.get_local_rank(mesh_dim=0) on rank 4, 5, 6, 7 would return 1.
            Calling mesh_2d.get_local_rank(mesh_dim=1) on rank 0, 4 would return 0.
            Calling mesh_2d.get_local_rank(mesh_dim=1) on rank 1, 5 would return 1.
            Calling mesh_2d.get_local_rank(mesh_dim=1) on rank 2, 6 would return 2.
            Calling mesh_2d.get_local_rank(mesh_dim=1) on rank 3, 7 would return 3.

            Example::
                >>> # xdoctest: +SKIP("no rank")
                >>> from torch.distributed.device_mesh import DeviceMesh
                >>>
                >>> # Initialize device mesh as (2, 4) to represent the topology
                >>> # of cross-host(dim 0), and within-host (dim 1).
                >>> mesh = DeviceMesh(device_type="cuda", mesh=[[0, 1, 2, 3],[4, 5, 6, 7]])
            """
            if self.ndim > 1 and mesh_dim is None:
                raise RuntimeError(
                    f"Found the DeviceMesh have {self.mesh.ndim} dimensions",
                    "Optional kwarg `mesh_dim` needs to be specified when device_mesh.ndim > 1.",
                )
            elif mesh_dim is None:
                mesh_dim = 0

            mesh_dim_group = self.get_group(mesh_dim)  # type: ignore[arg-type]
            assert isinstance(
                mesh_dim_group, ProcessGroup
            ), "We expect ProcessGroup before calling `get_rank`!"
            return get_rank(mesh_dim_group)  # type: ignore[arg-type]

        def get_coordinate(self) -> Optional[List[int]]:
            """
            Return the relative indices of this rank relative to all
            dimensions of the mesh. If this rank is not part of the mesh, return None.
            """
            return self._coordinate_on_dim if self._coordinate_on_dim else None

        def _get_mesh_dim_by_name(self, mesh_dim_name: str) -> int:
            if self.mesh_dim_names is None or len(self.mesh_dim_names) == 0:
                raise KeyError(
                    "No `mesh_dim_names` found.",
                )
            if mesh_dim_name not in self.mesh_dim_names:
                raise KeyError(
                    f"Mesh dimension '{mesh_dim_name}' does not exist.",
                    f"Available mesh dimensions are: {self.mesh_dim_names}",
                )
            return self.mesh_dim_names.index(mesh_dim_name)  # type: ignore[union-attr]

    def init_device_mesh(
        device_type: str,
        mesh_shape: Tuple[int, ...],
        *,
        mesh_dim_names: Optional[Tuple[str, ...]] = None,
    ) -> DeviceMesh:
        """
        Initializes a `DeviceMesh` based on `device_type`, `mesh_shape`, and `mesh_dim_names` parameters.

        This creates a DeviceMesh with an n-dimensional array layout, where `n` is the length of `mesh_shape`.
        If `mesh_dim_names` is provided, each dimension is labeled as `mesh_dim_names[i]`.

        .. note::
            `init_device_mesh` follows SPMD programming model, meaning the same PyTorch Python program
            runs on all processes/ranks in the cluster. Ensure `mesh_shape` (the dimensions of the nD array
            describing device layout) is identical across all ranks. Inconsistent `mesh_shape` may lead to hanging.

        .. note::
            If no process group is found, init_device_mesh will initialize distributed process group/groups
            required for distributed communications behind the scene.

        Args:
            device_type (str): The device type of the mesh. Currently supports: "cpu", "cuda/cuda-like".
            mesh_shape (Tuple[int]): A tuple defining the dimensions of the multi-dimensional array
                describing the layout of devices.
            mesh_dim_names (Tuple[str], optional): A tuple of mesh dimension names to assign to each dimension
                of the multi-dimensional array describing the layout of devices. Its length must match the length
                of `mesh_shape`. Each string in `mesh_dim_names` must be unique.

        Returns:
            DeviceMesh: A :class:`DeviceMesh` object representing the device layout.

        Example::
            >>> # xdoctest: +SKIP("no rank")
            >>> from torch.distributed.device_mesh import init_device_mesh
            >>>
            >>> mesh_1d = init_device_mesh("cuda", mesh_shape=(8,))
            >>> mesh_2d = init_device_mesh("cuda", mesh_shape=(2, 8), mesh_dim_names=("dp", "tp"))

        """
        if mesh_dim_names is not None:
            if len(set(mesh_dim_names)) != len(mesh_dim_names):
                raise RuntimeError(
                    "Each mesh_dim_name must be unique.",
                    f"Found repeated mesh_dim_name in mesh_dim_names {mesh_dim_names}",
                )

            if len(mesh_shape) != len(mesh_dim_names):
                raise RuntimeError(
                    "mesh_shape and mesh_dim_names should have same length!",
                    f"Found len(mesh_dim_names): {len(mesh_dim_names)} and len(mesh_shape):{len(mesh_shape)}.",
                )

        mesh = torch.arange(math.prod(mesh_shape)).view(mesh_shape)
        device_mesh = DeviceMesh(
            device_type=device_type,
            mesh=mesh,
            mesh_dim_names=mesh_dim_names,
        )

        return device_mesh
