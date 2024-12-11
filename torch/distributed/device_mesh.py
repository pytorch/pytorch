# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import math
import threading
from functools import reduce
from itertools import chain
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import torch
from torch.distributed import is_available
from torch.utils._typing_utils import not_none


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
    from torch._C._distributed_c10d import Backend as C10dBackend
    from torch.distributed.distributed_c10d import (
        _find_pg_by_ranks_and_tag,
        _get_default_group,
        _get_group_tag,
        get_backend,
        get_process_group_ranks,
        get_rank,
        get_world_size,
        init_process_group,
        is_initialized,
        new_group,
        ProcessGroup,
        split_group,
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

    class _MeshEnv(threading.local):
        def __init__(self) -> None:
            self.mesh_stack: List[DeviceMesh] = []
            self.child_to_root_mapping: Dict[DeviceMesh, DeviceMesh] = {}
            self.mesh_dim_group_options: Dict[
                int, Tuple[str, Optional[C10dBackend.Options]]
            ] = {}
            self.root_to_flatten_mapping: Dict[DeviceMesh, Dict[str, DeviceMesh]] = {}
            # Record flatten mesh name to its mesh dim index in root mesh.
            self.flatten_name_to_root_dims: Dict[
                DeviceMesh, Dict[str, Tuple[int, ...]]
            ] = {}

        def get_current_mesh(self) -> "DeviceMesh":
            if len(self.mesh_stack) == 0:
                raise RuntimeError("No device mesh is currently active!")
            return self.mesh_stack[-1]

        def create_sub_mesh(
            self,
            device_mesh: "DeviceMesh",
            submesh_dim_names: Tuple[str, ...],
            submesh_dims: List[Tuple[int, ...]],
        ) -> "DeviceMesh":
            # Get the submesh dim size from the submesh_dims.
            # For example, if we have a 3D mesh with mesh_shape (2, 2, 2) mesh_dim_names ("dp", "cp", "tp") and we want
            # to slice out mesh["dp_cp"], then submesh_dims = [(0, 1), (2,)] and submesh_dim_size = [2 * 2, 2] = [4, 2].
            # If we want to slice out mesh["dp", "cp"], then submesh_dims = [(0,), (1,)] and submesh_dim_size = [2, 2].
            slice_dim_size = [
                reduce(
                    lambda x, y: x * device_mesh.mesh.size(y),
                    mesh_dim,
                    1,
                )
                for mesh_dim in submesh_dims
            ]

            mesh_tensor = device_mesh.mesh
            # slice_dim_idx could be differnt from submesh_dims, as we may need to flatten out some dims.
            slice_dim_idx = []
            slice_dim_group_info = []
            # keep track of the number of dims that have been flattened so we can get the correct slice_dim_idx in the
            # flattened mesh tensor.
            num_dims_flatten = 0
            for mesh_dim_indices, mesh_dim_name in zip(submesh_dims, submesh_dim_names):
                # Currently, this only allows slicing out a contiguous flattened dim.
                # TODO: we need to handle reconstructing a non-contiguous flattened dim.
                if len(mesh_dim_indices) > 1:
                    # We need to move the start_dim and end_dim to the left if some dims are already flattened.
                    mesh_tensor = mesh_tensor.flatten(
                        start_dim=mesh_dim_indices[0] - num_dims_flatten,
                        end_dim=mesh_dim_indices[-1] - num_dims_flatten,
                    )
                    # If some dims are already flattened, we need to adjust the slice_dim_idx accordingly.
                    # For example, if the submesh_dims = [(0, 1), (2,), (3, 4)] with 0-1 flattened and 3-4 flattened,
                    # then the final slice_dim_idx should be [0, 1, 2].
                    slice_dim_idx.append(mesh_dim_indices[0] - num_dims_flatten)
                    num_dims_flatten += len(mesh_dim_indices) - 1
                    slice_dim_group_info.append(
                        self.root_to_flatten_mapping[device_mesh][
                            mesh_dim_name
                        ]._dim_group_infos[0]
                    )
                else:
                    slice_dim_idx.append(mesh_dim_indices[0] - num_dims_flatten)
                    slice_dim_group_info.append(
                        device_mesh._dim_group_infos[mesh_dim_indices[0]]
                    )

            # mesh_tensor has already been flattened if needed. So mesh_tensor.ndim <= device_mesh.mesh.ndim now.
            mesh_dims_remained_idx = list(range(mesh_tensor.ndim))
            for idx in slice_dim_idx:
                mesh_dims_remained_idx.remove(idx)

            # pg_ranks_by_dim is the size of [number of local ranks of the outermost submesh dimension, *slice_dim_idx]
            # This means on each local rank of the outermost slice mesh dim, we have a tensor of submesh size with
            # the pg ranks of the submesh. From this, we can extract the submesh mesh tensor contains the current rank.
            pg_ranks_by_dim = mesh_tensor.permute(
                *mesh_dims_remained_idx, *slice_dim_idx
            ).reshape(-1, *slice_dim_size)

            cur_rank = device_mesh.get_rank()
            for mesh_nd in pg_ranks_by_dim:
                submesh = DeviceMesh(
                    device_mesh.device_type,
                    mesh_nd,
                    mesh_dim_names=submesh_dim_names,
                    _init_backend=False,
                )
                if cur_rank in mesh_nd:
                    res_submesh = submesh

            res_submesh._dim_group_infos = slice_dim_group_info  # type: ignore[possibly-undefined]
            self.child_to_root_mapping[res_submesh] = device_mesh

            return res_submesh

        def create_flatten_mesh(
            self, device_mesh: "DeviceMesh", mesh_dim_name: Optional[str] = None
        ) -> "DeviceMesh":
            root_mesh = _mesh_resources.get_root_mesh(device_mesh)

            flatten_dims_in_root = [
                not_none(root_mesh.mesh_dim_names).index(flattened_mesh_dim_name)
                for flattened_mesh_dim_name in not_none(device_mesh.mesh_dim_names)
            ]

            if not mesh_dim_name:
                mesh_dim_name = "_".join(
                    [
                        not_none(root_mesh.mesh_dim_names)[dim]
                        for dim in flatten_dims_in_root
                    ]
                )

            # Check whether the mesh_dim_name for flattened mesh is valid.
            self.flatten_name_to_root_dims.setdefault(root_mesh, {})
            invalid_dim_names = chain(
                *list(not_none(root_mesh.mesh_dim_names)),
                *self.flatten_name_to_root_dims[root_mesh].keys(),
            )
            if mesh_dim_name in invalid_dim_names:
                raise RuntimeError(
                    f"{mesh_dim_name} already exists for submesh of the {root_mesh}. ",
                    f"The mesh_dim_names of submesh and flattened mesh are {invalid_dim_names}. "
                    f"Please specify another valid mesh_dim_name.",
                )

            # Quick return if the flatten mesh has been created before.
            # TODO: If we decide to restrict flatten initialization once, we should remove
            # this check and throw an error if the flatten mesh is already created before.
            if (
                root_mesh in self.root_to_flatten_mapping
                and mesh_dim_name in self.root_to_flatten_mapping[root_mesh]
            ):
                return self.root_to_flatten_mapping[root_mesh][mesh_dim_name]

            flattened_mesh_dim_size = math.prod(device_mesh.mesh.size())

            remained_dims_in_root = list(range(root_mesh.mesh.ndim))
            for flatten_dim_in_root in flatten_dims_in_root:
                remained_dims_in_root.remove(flatten_dim_in_root)

            pg_ranks_by_dim = root_mesh.mesh.permute(
                *remained_dims_in_root, *flatten_dims_in_root
            ).reshape(-1, flattened_mesh_dim_size)

            cur_rank = root_mesh.get_rank()
            for mesh_nd in pg_ranks_by_dim:
                # need to init backend here since the flattened pg doesn't exist in root mesh.
                flattened_mesh = DeviceMesh(
                    root_mesh.device_type,
                    mesh_nd,
                    mesh_dim_names=(mesh_dim_name,),
                )
                if cur_rank in mesh_nd:
                    res_flattened_mesh = flattened_mesh
            self.child_to_root_mapping[res_flattened_mesh] = root_mesh  # type: ignore[possibly-undefined]
            self.root_to_flatten_mapping.setdefault(root_mesh, {})[mesh_dim_name] = res_flattened_mesh  # type: ignore[possibly-undefined]
            self.flatten_name_to_root_dims[root_mesh][mesh_dim_name] = tuple(flatten_dims_in_root)  # type: ignore[possibly-undefined]

            return res_flattened_mesh

        def get_root_mesh(self, device_mesh: "DeviceMesh") -> "DeviceMesh":
            # If a mesh could not be found in the child_to_root_mapping, it is a root mesh itself.
            # A root mesh is not created through slicing.
            # We considers the root mesh of a root mesh is itself.
            root_mesh = self.child_to_root_mapping.get(device_mesh, None)
            return device_mesh if not root_mesh else root_mesh

        def get_root_mesh_dim(self, device_mesh: "DeviceMesh") -> Optional[int]:
            """
            Returns the index of the mesh dim in the root mesh.
            The device_mesh passed in needs to be sliced out from the root mesh
            or submesh of the root mesh.
            """
            root_mesh = self.get_root_mesh(device_mesh)
            child_mesh_dim_names = device_mesh.mesh_dim_names
            if root_mesh and child_mesh_dim_names:
                assert (
                    len(child_mesh_dim_names) == 1
                ), "The submesh can only be a 1D mesh."
                child_mesh_dim_name = child_mesh_dim_names[0]
                return self.get_mesh_dim_by_name(root_mesh, child_mesh_dim_name)
            return None

        @staticmethod
        def num_devices_per_host(device_type: str) -> int:
            return _get_device_handle(device_type).device_count()

        @staticmethod
        def num_hosts(device_type: str) -> int:
            # ProcessGroup can't tell us this info so we have to infer it, assume
            # homogeneous hardware for now
            return get_world_size() // _MeshEnv.num_devices_per_host(device_type)

        def get_mesh_dim_by_name(
            self, device_mesh: "DeviceMesh", mesh_dim_name: str
        ) -> int:
            if (
                device_mesh.mesh_dim_names is None
                or len(device_mesh.mesh_dim_names) == 0
            ):
                raise KeyError(
                    "No `mesh_dim_names` found.",
                )
            if mesh_dim_name not in device_mesh.mesh_dim_names:
                raise KeyError(
                    f"Mesh dimension '{mesh_dim_name}' does not exist.",
                    f"Available mesh dimensions are: mesh_dim_names={device_mesh.mesh_dim_names}",
                )
            return not_none(device_mesh.mesh_dim_names.index(mesh_dim_name))

        def _set_mesh_dim_group_options(
            self,
            dim: int,
            backend: str,
            pg_options: Optional[C10dBackend.Options] = None,
        ) -> None:
            self.mesh_dim_group_options[dim] = (backend, pg_options)

        def _get_slice_mesh_dims(
            self, device_mesh, mesh_dim_names
        ) -> List[Tuple[int, ...]]:
            """
            Validate whether the mesh_dim_names is valid for slicing the given device_mesh.
            If valid, return dim indexes of the slice mesh in the device mesh.
            """
            if device_mesh != self.get_root_mesh(device_mesh):
                raise RuntimeError("Cannot create a submesh from a submesh.")

            # The slice mesh_dim_names should consist either the device_mesh's mesh_dim_names
            # or its flattened mesh's mesh_dim_names.
            self.flatten_name_to_root_dims.setdefault(device_mesh, {})
            flatten_name_to_root_dims = self.flatten_name_to_root_dims[device_mesh]
            valid_mesh_dim_names = [
                *device_mesh.mesh_dim_names,
                *flatten_name_to_root_dims,
            ]

            if not all(
                mesh_dim_name in valid_mesh_dim_names
                for mesh_dim_name in mesh_dim_names
            ):
                raise KeyError(
                    f"Invalid mesh_dim_names {mesh_dim_names} specified. "
                    f"Valid mesh_dim_names are {valid_mesh_dim_names}."
                )

            # Validate the order of the slice mesh dim indices.
            # This needs to be in ascending order.
            curr_idx = -1
            slice_mesh_dims = []
            for mesh_dim_name in mesh_dim_names:
                if mesh_dim_name in flatten_name_to_root_dims:
                    mesh_indices = flatten_name_to_root_dims[mesh_dim_name]
                    # TODO: this doesn't allow non-contiguous slicing with flatten dim yet. next_idx
                    # should be mesh_indices[0] once we support non-contiguous slicing with flatten dim.
                    next_idx = mesh_indices[-1]
                    slice_mesh_dims.append(mesh_indices)
                else:
                    next_idx = device_mesh.mesh_dim_names.index(mesh_dim_name)
                    slice_mesh_dims.append((next_idx,))
                if next_idx <= curr_idx:
                    raise KeyError(
                        f"Invalid mesh_dim_names {mesh_dim_names} specified. ",
                        f"Found mesh dim indices to slice: {slice_mesh_dims}. ",
                        "Mesh dim indices should be in ascending order.",
                    )
                curr_idx = next_idx

            return slice_mesh_dims

        def _get_all_submeshes(
            self, device_mesh: "DeviceMesh", mesh_dim_name: str
        ) -> List["DeviceMesh"]:
            """
            Return all the submeshes of a given mesh dimension of the device mesh.
            """
            mesh_dim = self.get_mesh_dim_by_name(device_mesh, mesh_dim_name)
            pg_ranks_by_dim = device_mesh.mesh.swapdims(-1, mesh_dim).reshape(
                -1, device_mesh.mesh.size(mesh_dim)
            )

            cur_rank = device_mesh.get_rank()
            res_submeshes = []
            for mesh_1d in pg_ranks_by_dim:
                submesh = DeviceMesh(
                    device_mesh.device_type,
                    mesh_1d,
                    mesh_dim_names=(mesh_dim_name,),
                    _init_backend=False,
                )
                submesh._dim_group_infos = (
                    [device_mesh._dim_group_infos[mesh_dim]]
                    if cur_rank in mesh_1d
                    else []
                )
                res_submeshes.append(submesh)

            return res_submeshes

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
            _init_backend: bool = True,
        ) -> None:
            self.device_type = device_type
            if isinstance(mesh, torch.Tensor) and mesh.device.type != "cpu":
                raise ValueError(f"`mesh` must be a CPU tensor, got {mesh}")
            self.mesh = (
                mesh.detach().to(dtype=torch.int)
                if isinstance(mesh, torch.Tensor)
                else torch.tensor(mesh, device="cpu", dtype=torch.int)
            )
            self.mesh_dim_names = tuple(mesh_dim_names) if mesh_dim_names else None

            # private field to pre-generate DeviceMesh's hash
            self._flatten_mesh_list = tuple(self.mesh.flatten().tolist())
            self._thread_id = None

            # Skip process group initialization if xla device or init backend is False
            # TODO(yeounoh) implement DeviceMesh backend and register XLA backend.
            if device_type != "xla":
                # always try to create default (world) pg, even if it is not initialized
                # already. The world pg is used for device mesh identity (rank) on each
                # process (we need to know if the current global rank is in the mesh or not).
                if _init_backend:
                    self._get_or_create_default_group()
                    self._init_process_groups()

                if is_initialized() and get_backend() == "threaded":
                    self._thread_id = threading.get_ident()

                # calculate the coordinates of the current global rank on the mesh
                rank_coords = (self.mesh == get_rank()).nonzero()
                assert rank_coords.size(0) in (0, 1)
                self._coordinate_on_dim: Optional[List[int]] = (
                    rank_coords[0].tolist() if rank_coords.size(0) > 0 else None
                )

        def _get_or_create_default_group(self):
            default_initialized = is_initialized()
            if not default_initialized:
                init_process_group()

            world_size = get_world_size()
            if self.mesh.numel() > world_size:
                raise RuntimeError(
                    f"Mesh should not be bigger than default world size {world_size}, but found {self.mesh.numel()} ranks!"
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

            return _get_default_group()

        def _init_process_groups(self):
            # tag/ranks/group_name associated with each mesh dimension, each
            # mesh dimension should have one sub-group per rank
            #
            # TODO(yifu): remove tag and ranks once we fully migrate to native
            # functional collectives. See details in:
            # https://github.com/pytorch/pytorch/issues/93173#issuecomment-1907095208
            dim_group_infos: List[Tuple[str, List[int], str]] = []
            default_group = _get_default_group()

            if self.mesh.ndim == 1 and self.mesh.numel() == get_world_size():
                # Append the default pg to the first dim groups only if the default pg is compatible with `self.device_type`.
                # Otherwise, create new pg.
                ranks = list(range(get_world_size()))
                dim_group = (
                    new_group(
                        backend="cpu:gloo,cuda:nccl",
                        ranks=ranks,
                        group_desc="mesh_default",
                    )
                    if torch.cuda.is_available()
                    and get_backend(default_group) == "gloo"
                    else default_group
                )
                dim_group_infos.append(
                    (
                        _get_group_tag(dim_group),
                        ranks,
                        dim_group.group_name,
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

                    # Respect dim group options specified via _MeshEnv.set_dim_group_options().
                    # Inherit from the parent group if no options are specified for the group.
                    if dim in _mesh_resources.mesh_dim_group_options:
                        (
                            backend,
                            pg_options,
                        ) = _mesh_resources.mesh_dim_group_options[dim]
                    else:
                        backend, pg_options = None, None

                    # If we have a 2D mesh with mesh_dim_names ("dp", "tp"), the group description
                    # of the subgroups would be `mesh_dim_dp` and `mesh_name_tp`.
                    # If the mesh doesn't not have a mesh_dim_names, then the group description of the
                    # subgroup would be `mesh_dim_0` and `mesh_dim_1`.
                    group_desc = (
                        f"mesh_{self.mesh_dim_names[dim]}"
                        if self.mesh_dim_names
                        else f"mesh_dim_{dim}"
                    )

                    # If bound_device_id exists, it means the nccl communicator has been eagerly initialized
                    # so that we can use `split_group` to create subgroups through `ncclCommSplit`.
                    # In this case, we only need to make one API call (`split_group``) for the subgroup creation
                    # for each mesh dimension. In a 2 * 4 mesh, we only need to make 2 API calls per ranks to create
                    # all the subgroups.
                    # Otherwise, we need to make more than one API call (`new_group`) for subgroup creations. The
                    # numbers of API calls are equal to the number of subgroups for each mesh dimension. In a 2 * 4
                    # mesh, we need to make 2 + 4 = 6 API calls per ranks to create all the subgroups.
                    dim_group = None
                    if (
                        bound_device_id := getattr(
                            default_group, "bound_device_id", None
                        )
                    ) is not None:
                        dim_group = split_group(
                            parent_pg=default_group,
                            pg_options=pg_options,
                            split_ranks=pg_ranks_by_dim.tolist(),
                            group_desc=group_desc,
                        )

                    # If the subgroup has been already created through `split_group`, we simply loop over `pg_ranks_by_dim`
                    # and append the `(group_tag, subgroup_ranks, and group_name)` tuple to the `dim_group_infos` list when
                    # the current rank is in the subgroup.
                    # Otherwise, we use `new_group` instead of `split_group` to create subgroups by looping over `pg_ranks_by_dim`
                    # along with appending information to the `dim_group_infos` list whenever necessary.
                    for dim_mesh in pg_ranks_by_dim:
                        subgroup_ranks = dim_mesh.tolist()

                        # We temporarily revert the re-use subgroup, since it breaks two internal tests.
                        # Temporarily reverting to resolve test timeout while root-causing.
                        # TODO: Add two tests to cover internal tests scenarios and re-enable reuse subgroup if exists.
                        if bound_device_id is None:
                            dim_group = new_group(
                                ranks=subgroup_ranks,
                                backend=backend,
                                pg_options=pg_options,
                                group_desc=group_desc,
                            )

                        # only add to dim_groups if the current rank in the subgroup
                        if self.get_rank() in subgroup_ranks:
                            if len(dim_group_infos) > dim:
                                raise RuntimeError(
                                    f"Each device mesh dimension should get only one process group, but got {self.get_rank()} "
                                    f"in {subgroup_ranks}!"
                                )
                            dim_group_infos.append(
                                (
                                    _get_group_tag(not_none(dim_group)),
                                    subgroup_ranks,
                                    dim_group.group_name,
                                )
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
            device_mesh_repr = (
                f"DeviceMesh('{self.device_type}', {self.mesh.tolist()})"
                if not self.mesh_dim_names
                else f"DeviceMesh('{self.device_type}', {self.mesh.tolist()}, mesh_dim_names={self.mesh_dim_names})"
            )
            return device_mesh_repr

        def __hash__(self):
            # lazily compute hash
            self._hash = getattr(self, "_hash", None)
            if not self._hash:
                self._hash = hash(
                    (
                        self._flatten_mesh_list,
                        self.mesh.shape,
                        self.device_type,
                        self.mesh_dim_names,
                        self._thread_id,
                    )
                )
            return self._hash

        def __eq__(self, other: object) -> bool:
            if not isinstance(other, DeviceMesh):
                return False
            if id(self) == id(other):
                return True
            else:
                return (
                    self._flatten_mesh_list == other._flatten_mesh_list
                    and self.mesh.shape == other.mesh.shape
                    and self.device_type == other.device_type
                    and self.mesh_dim_names == other.mesh_dim_names
                    and self._thread_id == other._thread_id
                )

        def __getitem__(
            self, mesh_dim_names: Union[str, Tuple[str, ...]]
        ) -> "DeviceMesh":
            """
            Slice the current DeviceMesh based on the mesh_dim_names given to create a submesh.
            The submesh created consists of the dimensions and the communicators indicated by
            ``mesh_dim_names``

            Args:
                mesh_dim_names (Union[str, Tuple[str]]): the name or the tuple of names of the
                mesh dimension of the DeviceMesh to create the submesh for.
            Returns:
                A :class:`DeviceMesh` object

            The following program runs on each process/rank in an SPMD manner in a world size of 8.
            In the first example:
                Calling mesh_2d["tp"] on rank 0, 1, 2, 3 returns a 1D submesh of DeviceMesh:([0, 1, 2, 3]).
                Calling mesh_2d["tp"] on rank 4, 5, 6, 7 returns a 1D submesh of  DeviceMesh:([4, 5, 6, 7]).
                Calling mesh_2d["dp"] on rank 0, 4 returns a 1D submesh of  DeviceMesh:([0, 4]).
                Calling mesh_2d["dp"] on rank 1, 5 returns a 1D submesh of  DeviceMesh:([1, 5]).
                Calling mesh_2d["dp"] on rank 2, 6 returns a 1D submesh of  DeviceMesh:([2, 6]).
                Calling mesh_2d["dp"] on rank 3, 7 returns a 1D submesh of  DeviceMesh:([3, 7]).

            In the second example:
                Calling mesh_3d["dp", "cp"] on rank 0, 1, 4, 5 returns a 2D submesh of DeviceMesh:([[0, 1], [4, 5]]).
                Calling mesh_3d["dp", "cp"] on rank 2, 3, 6, 7 returns a 2D submesh of DeviceMesh:([[2, 3], [6, 7]]).
                Calling mesh_3d["cp", "dp"] on rank 0, 1, 4, 5 returns a 2D submesh of DeviceMesh:([[0, 4], [1, 5]]).
                Calling mesh_3d["cp", "dp"] on rank 2, 3, 6, 7 returns a 2D submesh of DeviceMesh:([[2, 6], [3, 7]]).

            Example::
                >>> # xdoctest: +SKIP("no rank")
                >>> from torch.distributed.device_mesh import DeviceMesh
                >>>
                >>> # Initialize a 2D device mesh as (2, 4) to represent the topology
                >>> # of cross-host(dim 0), and within-host (dim 1).
                >>> mesh_2d = init_device_mesh(device_type="cuda", (2,4), mesh_dim_names=("dp", "tp"))
                >>> tp_mesh = mesh_2d["tp"]
                >>> dp_mesh = mesh_2d["dp"]
                >>>
                >>> # Initialize a 3D mesh.
                >>> mesh_3d = init_device_mesh(device_type="cuda", (2,2,2), mesh_dim_names=("dp", "pp", "cp"))
                >>> # The order of the mesh_dim_names provided deteremines the order of dimensions in the submesh.
                >>> dp_cp_mesh = mesh_3d["dp", "cp"]
                >>> cp_dp_mesh = mesh_3d["cp", "dp"]
            """
            if not self.mesh_dim_names:
                raise RuntimeError("Cannot slice a DeviceMesh without mesh_dim_names!")

            mesh_dim_names = (
                (mesh_dim_names,) if isinstance(mesh_dim_names, str) else mesh_dim_names
            )

            if mesh_dim_names == self.mesh_dim_names:
                return self
            else:
                slice_mesh_dims = _mesh_resources._get_slice_mesh_dims(
                    self, mesh_dim_names
                )
                # When using FakeTensorMode to trace the model, `create_sub_mesh()` will
                # fail as it will require a real tensor to manipulate.
                # `unset_fake_temporarily()` will allow us to materialize the tensors
                # within `_mesh_resources`, which should not affect modling.
                #
                # Note that this should be orthogonal to torch.compile(). But whether
                # we can compile device_mesh `slicing` (no graph break) is not verified
                # yet and need a follow-up,
                # TODO: compiler + device_mesh slicing.
                with torch._subclasses.fake_tensor.unset_fake_temporarily():
                    submesh = _mesh_resources.create_sub_mesh(
                        self, mesh_dim_names, slice_mesh_dims
                    )
                return submesh

        def get_group(self, mesh_dim: Optional[Union[int, str]] = None) -> ProcessGroup:
            """
            Returns the single ProcessGroup specified by mesh_dim, or, if mesh_dim is not specified and the
            DeviceMesh is 1-dimensional, returns the only ProcessGroup in the mesh.

            Args:
                mesh_dim (str/int, optional): it can be the name of the mesh dimension or the index
                of the mesh dimension. Default is None.

            Returns:
                A :class:`ProcessGroup` object.
            """
            if not hasattr(self, "_dim_group_infos"):
                raise RuntimeError("DeviceMesh process groups not initialized!")

            if self.mesh.ndim > 1 and mesh_dim is None:
                raise RuntimeError(
                    f"Found the DeviceMesh have {self.mesh.ndim} dimensions",
                    "Optional kwarg `mesh_dim` needs to be specified when device_mesh.ndim > 1.",
                    "If you want to get the list of all the ProcessGroups in the DeviceMesh,"
                    "please use `get_all_groups()` instead.",
                )

            # Quick return if the current device_mesh is a 1D mesh.
            if self.mesh.ndim == 1 and mesh_dim is None:
                return not_none(
                    _find_pg_by_ranks_and_tag(*self._dim_group_infos[0][:2])  # type: ignore[index]
                )

            root_mesh = _mesh_resources.get_root_mesh(self)
            root_to_flatten_mapping = _mesh_resources.root_to_flatten_mapping.get(
                root_mesh, None
            )
            if root_to_flatten_mapping and mesh_dim in root_to_flatten_mapping.keys():
                dim_group_infos = root_to_flatten_mapping[mesh_dim]._dim_group_infos[0][:2]  # type: ignore[index]
                return not_none(_find_pg_by_ranks_and_tag(*dim_group_infos))
            else:
                mesh_dim = (
                    _mesh_resources.get_mesh_dim_by_name(self, mesh_dim)
                    if isinstance(mesh_dim, str)
                    else mesh_dim
                )
                return not_none(
                    _find_pg_by_ranks_and_tag(*self._dim_group_infos[mesh_dim][:2])  # type: ignore[index]
                )

        def get_all_groups(self) -> List[ProcessGroup]:
            """
            Returns a list of ProcessGroups for all mesh dimensions.

            Returns:
                A list of :class:`ProcessGroup` object.
            """
            return [self.get_group(i) for i in range(self.mesh.ndim)]

        @staticmethod
        def from_group(
            group: Union[ProcessGroup, List[ProcessGroup]],
            device_type: str,
            mesh: Optional[Union[torch.Tensor, "ArrayLike"]] = None,
            *,
            mesh_dim_names: Optional[Tuple[str, ...]] = None,
        ) -> "DeviceMesh":
            """
            Constructs a :class:`DeviceMesh` with ``device_type`` from an
            existing :class:`ProcessGroup`.

            The constructed device mesh has number of dimensions equal to the
            number of groups passed. If more than one group is passed, then the
            ``mesh`` argument is required.
            """
            if isinstance(group, ProcessGroup):
                group_ranks = get_process_group_ranks(group)
                if (
                    isinstance(mesh, torch.Tensor) and mesh.tolist() != group_ranks
                ) or (
                    mesh is not None
                    and not isinstance(mesh, torch.Tensor)
                    and mesh != group_ranks
                ):
                    raise ValueError(
                        f"Invalid mesh {str(mesh)} for ProcessGroup with ranks {group_ranks}"
                    )
                mesh = torch.tensor(group_ranks, device="cpu", dtype=torch.int)
                device_mesh = DeviceMesh(
                    device_type,
                    mesh,
                    mesh_dim_names=mesh_dim_names,
                    _init_backend=False,
                )
                device_mesh._dim_group_infos = [
                    (_get_group_tag(group), group_ranks, group.group_name)
                ]
                return device_mesh
            groups = list(group)
            if len(groups) == 0:
                raise ValueError("Expects at least one ProcessGroup to be passed")
            if mesh is None:
                raise ValueError("Must pass mesh if passing multiple ProcessGroups")
            mesh = (
                mesh.detach().to(dtype=torch.int, device="cpu")
                if isinstance(mesh, torch.Tensor)
                else torch.tensor(mesh, device="cpu", dtype=torch.int)
            )
            if mesh.ndim != len(groups):
                raise ValueError(
                    "Expects mesh with ndim equal to number of ProcessGroups but got "
                    f"mesh {mesh.tolist()} and {len(groups)} ProcessGroups"
                )
            device_mesh = DeviceMesh(
                device_type, mesh, mesh_dim_names=mesh_dim_names, _init_backend=False
            )
            device_mesh._dim_group_infos = [
                (
                    _get_group_tag(group),
                    get_process_group_ranks(group),
                    group.group_name,
                )
                for group in groups
            ]
            return device_mesh

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

            mesh_dim_group = not_none(self.get_group(mesh_dim))
            assert isinstance(
                mesh_dim_group, ProcessGroup
            ), "We expect ProcessGroup before calling `get_rank`!"
            return not_none(get_rank(mesh_dim_group))

        def get_coordinate(self) -> Optional[List[int]]:
            """
            Return the relative indices of this rank relative to all
            dimensions of the mesh. If this rank is not part of the mesh, return None.
            """
            return self._coordinate_on_dim if self._coordinate_on_dim else None

        def _flatten(self, mesh_dim_name: Optional[str] = None) -> "DeviceMesh":
            """
            Returns a 1D DeviceMesh by flattening the current DeviceMesh.

            If no mesh_dim_name is provided, the default is a string concatentaing the mesh_dim_names of the
            given submesh with each mesh_dim_name separated by "_". For example, if we have a 3D mesh
            DeviceMesh([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], mesh_dim_names=("dp", "cp", "tp")), calling
            mesh_3d["dp", "cp"]._flatten() will create a 1D submesh DeviceMesh([0, 1, 2, 3], mesh_dim_names=("dp_cp",))
            on rank 0, 1, 2, 3 and a 1D submesh DeviceMesh([4, 5, 6, 7], mesh_dim_names=("dp_cp",)) on rank 4, 5, 6, 7.

            After the flattened dimension is created, to access the flattened dimesnion in mesh_3d, one can use the
            existing slicing method to obtain the flattened mesh through calling mesh_3d["dp_cp"].
            """
            if not self.mesh_dim_names:
                raise RuntimeError(
                    "Cannot flatten a DeviceMesh without mesh_dim_names!"
                )

            return _mesh_resources.create_flatten_mesh(self, mesh_dim_name)

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
                Passing in a device type with a GPU index, such as "cuda:0", is not allowed.
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

        # assume valid device types are all letters
        if device_type and not device_type.isalpha():
            raise RuntimeError(
                f"Device type with index is not supported but got {device_type}. ",
                "If you maintained a 'torch.device' object, it's recommended to pass in 'device.type'.",
            )

        # Always initialize the mesh's tensor on CPU, regardless of what the
        # external device type has been set to be (e.g. meta)
        with torch.device("cpu"):
            mesh = torch.arange(math.prod(mesh_shape), dtype=torch.int).view(mesh_shape)
        device_mesh = DeviceMesh(
            device_type=device_type,
            mesh=mesh,
            mesh_dim_names=mesh_dim_names,
        )

        return device_mesh
