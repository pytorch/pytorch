# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import math
import os
import threading
import warnings
from collections.abc import Iterator
from itertools import zip_longest
from typing import Optional, TYPE_CHECKING, Union

import torch
from torch.distributed import is_available
from torch.distributed._cute_layout import _Layout, init_layouts_from_mesh
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
        _get_default_group,
        _resolve_process_group,
        get_backend,
        get_process_group_ranks,
        get_rank,
        get_world_size,
        GroupMember,
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
            self.mesh_stack: list[DeviceMesh] = []
            self.child_to_root_mapping: dict[DeviceMesh, DeviceMesh] = {}
            self.mesh_dim_group_options: dict[
                int, tuple[Optional[str], Optional[C10dBackend.Options]]
            ] = {}

        def get_current_mesh(self) -> "DeviceMesh":
            if len(self.mesh_stack) == 0:
                raise RuntimeError("No device mesh is currently active!")
            return self.mesh_stack[-1]

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
                assert len(child_mesh_dim_names) == 1, (
                    "The submesh can only be a 1D mesh."
                )
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
            backend: Optional[str],
            pg_options: Optional[C10dBackend.Options] = None,
        ) -> None:
            self.mesh_dim_group_options[dim] = (backend, pg_options)

        def _get_all_submeshes(
            self, device_mesh: "DeviceMesh", mesh_dim_name: str
        ) -> list["DeviceMesh"]:
            """
            Return all the submeshes of a given mesh dimension of the device mesh.
            """
            mesh_dim = self.get_mesh_dim_by_name(device_mesh, mesh_dim_name)
            layout = device_mesh._layouts[mesh_dim]
            pg_ranks_by_dim = layout.layout_to_global_ranks(device_mesh.size())
            cur_rank = device_mesh.get_rank()
            res_submeshes = []
            for mesh_1d in pg_ranks_by_dim:
                submesh = DeviceMesh(
                    device_mesh.device_type,
                    mesh_1d,
                    mesh_dim_names=(mesh_dim_name,),
                    _init_backend=False,
                )
                submesh._backend = device_mesh._backend
                submesh._layouts = device_mesh._layouts if cur_rank in mesh_1d else ()
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

    class _DeviceMeshBackend:
        def __init__(
            self,
            device_type: str,
        ) -> None:
            """
            _DeviceMeshBackend acts as a class which handles all the mechanical bookkeepings
            for the backend initialization of the DeviceMesh and all the following transformations.

            Also we only use a singleton for one DeviceMesh universe, which means all the DeviceMesh
            objects share the same backend initialization if they are transformed from the same device mesh.
            If users initialize a new DeviceMesh object from scratch, we will create a new singleton.
            """
            self.device_type = device_type
            self.layouts_to_groups: dict[_Layout, str] = {}
            self.names_to_layouts: dict[str, _Layout] = {}

            # always try to create default (world) pg, even if it is not initialized
            # already. The world pg is used for device mesh identity (rank) on each
            # process (we need to know if the current global rank is in the mesh or not).
            self._setup_world_group_and_device()

        def _setup_world_group_and_device(self):
            default_initialized = is_initialized()
            # TODO: think about how to allow pg options to be passed to world group
            # or mesh dimension groups
            if not default_initialized:
                init_process_group()

            world_size = get_world_size()

            # ONLY set the device if the current device is not initialized, if user already
            # set the device before DeviceMesh init, we respect the user's choice.
            device_handle = _get_device_handle(self.device_type)
            if device_handle and not device_handle.is_initialized():
                # auto set the cuda/cuda-like device only if user has not set it, if there's LOCAL_RANK
                # env variable from launchers, we use it to set the device.
                if "LOCAL_RANK" in os.environ:
                    local_rank = int(os.environ["LOCAL_RANK"])
                    logger.info(
                        "Setting default device for the current process based on LOCAL_RANK=%s",
                        local_rank,
                    )
                    device_handle.set_device(local_rank)
                else:
                    warnings.warn(
                        "It seems like you did not set/select the default device for the current process before the DeviceMesh "
                        "initialization or use a launcher (i.e. torchrun) which populates `LOCAL_RANK` environment variable. "
                        "It is recommended to set the current device for the process BEFORE the DeviceMesh initialization so that "
                        "the underlying communicator (i.e. NCCL) can be initialized properly. "
                        "Given that the current process has no default device selected, DeviceMesh will use a heuristic to set the "
                        "device_id via `global_rank % num_devices_per_host`, assuming homogeneous hardware cluster. "
                    )
                    # heuristic to set the current cuda/cuda-like device base on num of gpu devices available in each host
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

        def _maybe_create_backend(
            self,
            layout: _Layout,
            dim: int,
            name: Optional[str] = None,
            backend_override: tuple[Optional[str], Optional[C10dBackend.Options]] = (
                None,
                None,
            ),
            group: Optional[ProcessGroup] = None,
        ) -> None:
            """
            Creates or retrieves a process group backend for a given layout and dimension.

            This method manages the mapping between layouts, dimension names, and process groups.
            It ensures that each layout has a corresponding process group, and each named dimension
            maps to exactly one layout. If the layout already has a process group, it validates
            the existing mappings and returns early.

            Args:
                layout (_Layout): The layout object representing the communication pattern
                dim (int): The dimension index in the device mesh
                name (Optional[str]): Optional name for the mesh dimension
                backend_override (tuple[Optional[str], Optional[C10dBackend.Options]]):
                    Optional backend type and options to override default settings
                group (Optional[ProcessGroup]): Optional existing process group to use instead of creating a new one

            Note:
                - If a layout already has a process group, any backend_override or explicit group will be ignored
                - If a name is provided, it will be mapped to the layout for future reference
                - For world-size layouts with no overrides, it will try to use the default process group
                - For other layouts, it creates appropriate subgroups using either split_group (more efficient and it only
                available for nccl communicators) or new_group
                - Because process group itself is not serializable, we only store the group name in the mapping
            """
            # Every mesh_dim_name will only map to one and only one layout.
            # When the layout has backend initiated, we want to ensure that
            if layout in self.layouts_to_groups:
                if name is not None:
                    assert name in self.names_to_layouts, (
                        f"dim_name {name} has not been mapped to any layout"
                    )
                    assert self.names_to_layouts[name] == layout, (
                        f"dim_name {name} has been mapped to another layout"
                    )
                if backend_override != (None, None):
                    warnings.warn(
                        f"Group for {layout} ({name=}) already exists, ignoring backend override"
                    )
                if group is not None:
                    warnings.warn(
                        f"Group for {layout} ({name=}) already exists, ignoring explicit group"
                    )
                return

            # Let mesh_dim_name maps to the current layout whose backend has not been initialized.
            # If mesh_dim_name has already mapped to another layout, we throw an error.
            if name is not None:
                assert name not in self.names_to_layouts, (
                    f"Mesh dim name {name} has been mapped to other backend initiated layout already"
                )
                self.names_to_layouts[name] = layout

            # When user explicitly pass in a process group, we directly reuse that PG as backend rather
            # than creating a new one.
            if group is not None:
                self.layouts_to_groups[layout] = group.group_name
                return

            default_group = _get_default_group()

            if layout == _Layout(((get_world_size(), 1),)) and backend_override == (
                None,
                None,
            ):
                # Append the default pg to the first dim groups only if the default pg is compatible with `self.device_type`.
                # Otherwise, create new pg.
                ranks = list(range(get_world_size()))
                group = (
                    new_group(
                        backend="cpu:gloo,cuda:nccl",
                        ranks=ranks,
                        group_desc="mesh_default",
                    )
                    if torch.cuda.is_available()
                    and get_backend(default_group) == "gloo"
                    else default_group
                )
            else:
                # Generate the pg_ranks_by_dim for pg_creation from the layout.
                pg_ranks_by_dim = layout.layout_to_global_ranks(get_world_size())
                backend, pg_options = backend_override

                # If we have a 2D mesh with mesh_dim_names ("dp", "tp"), the group description
                # of the subgroups would be `mesh_dim_dp` and `mesh_name_tp`.
                # If the mesh doesn't not have a mesh_dim_names, then the group description of the
                # subgroup would be `mesh_dim_0` and `mesh_dim_1`.
                group_desc = f"mesh_{name}" if name is not None else f"mesh_dim_{dim}"

                # If bound_device_id exists, it means the nccl communicator has been eagerly initialized
                # so that we can use `split_group` to create subgroups through `ncclCommSplit`.
                # In this case, we only need to make one API call (`split_group``) for the subgroup creation
                # for each mesh dimension. In a 2 * 4 mesh, we only need to make 2 API calls per ranks to create
                # all the subgroups.
                # Otherwise, we need to make more than one API call (`new_group`) for subgroup creations. The
                # numbers of API calls are equal to the number of subgroups for each mesh dimension. In a 2 * 4
                # mesh, we need to make 2 + 4 = 6 API calls per ranks to create all the subgroups.
                group = None
                if (
                    getattr(default_group, "bound_device_id", None) is not None
                    and torch.cuda.is_available()
                    and (
                        backend is None
                        or default_group._get_backend(torch.device("cuda")).name()
                        == backend
                    )
                ):
                    group = split_group(
                        parent_pg=default_group,
                        pg_options=pg_options,
                        split_ranks=pg_ranks_by_dim,
                        group_desc=group_desc,
                    )
                else:
                    # If the subgroup has been already created through `split_group`, we simply loop over `pg_ranks_by_dim`
                    # and append the `group_name` to the `dim_group_names` list when the current rank is in the subgroup.
                    # Otherwise, we use `new_group` instead of `split_group` to create subgroups by looping over `pg_ranks_by_dim`
                    # along with appending information to the `dim_group_names` list whenever necessary.
                    for subgroup_ranks in pg_ranks_by_dim:
                        # We temporarily revert the reuse subgroup, since it breaks two internal tests.
                        # Temporarily reverting to resolve test timeout while root-causing.
                        # TODO: Add two tests to cover internal tests scenarios and re-enable reuse subgroup if exists.
                        maybe_group = new_group(
                            ranks=subgroup_ranks,
                            backend=backend,
                            pg_options=pg_options,
                            group_desc=group_desc,
                        )
                        if maybe_group != GroupMember.NON_GROUP_MEMBER:
                            assert group is None
                            group = maybe_group

            assert group is not None
            self.layouts_to_groups[layout] = group.group_name

    class DeviceMesh:
        """
        DeviceMesh represents a mesh of devices, where layout of devices could be
        represented as a n-d dimension array, and each value of the n-d dimensional
        array is the global id of the default process group ranks.

        DeviceMesh could be used to setup the N dimensional device connections across the cluster,
        and manage the ProcessGroups for N dimensional parallelisms. Communications could happen on
        each dimension of the DeviceMesh separately. DeviceMesh respects the device that user selects
        already (i.e. if user call `torch.cuda.set_device` before the DeviceMesh initialization),
        and will select/set the device for the current process if user does not set the device
        beforehand. Note that manual device selection should happen BEFORE the DeviceMesh initialization.

        DeviceMesh can also be used as a context manager when using together with DTensor APIs.

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

        # TODO: to make existing public fields private and add some methods/properties for bc.
        device_type: str
        mesh: torch.Tensor
        mesh_dim_names: Optional[tuple[str, ...]]
        _layouts: tuple[_Layout, ...]
        _backend: Optional[_DeviceMeshBackend] = None

        def __init__(
            self,
            device_type: str,
            mesh: Union[torch.Tensor, "ArrayLike"],
            *,
            mesh_dim_names: Optional[tuple[str, ...]] = None,
            backend_override: Optional[
                tuple[tuple[Optional[str], Optional[C10dBackend.Options]], ...]
            ] = None,
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
            # Internal bookkeeping for the device mesh.
            self._layouts = init_layouts_from_mesh(self.mesh.size(), self.mesh.stride())

            # Skip process group initialization if xla device or init backend is False
            # TODO(yeounoh) implement DeviceMesh backend and register XLA backend.
            if not _init_backend or device_type == "xla":
                return

            world_size = get_world_size()
            if self.mesh.numel() > world_size:
                raise RuntimeError(
                    f"Mesh should not be bigger than default world size {world_size}, but found {self.mesh.numel()} ranks!"
                )
            self._backend = _DeviceMeshBackend(device_type)
            if backend_override is None:
                backend_override = ((None, None),) * self.mesh.ndim

            for i, layout in enumerate(self._layouts):
                backend_override_ = backend_override[i]
                global_override = _mesh_resources.mesh_dim_group_options.get(
                    i, (None, None)
                )
                if backend_override_ == (None, None):
                    backend_override_ = global_override
                elif global_override != (None, None):
                    raise RuntimeError(
                        f"Dimension {i} present both in the backend_override argument "
                        "and via _mesh_resources._set_mesh_dim_group_options"
                    )
                self._backend._maybe_create_backend(
                    layout,
                    i,
                    self.mesh_dim_names[i] if self.mesh_dim_names else None,
                    backend_override=backend_override_,
                )

        @staticmethod
        def _from_backend(
            device_type: str,
            backend: _DeviceMeshBackend,
            layouts: tuple[_Layout, ...],
            cur_rank: int,
            *,
            dim_names: Optional[Union[str, tuple[str, ...]]] = None,
        ) -> "DeviceMesh":
            """
            Creates a DeviceMesh from an existing backend and layouts. This will create a new device mesh
            from a same backend universe or transformed from (e.g, flatten, etc.). Although CuTe layout makes
            bookkeeping lot easier, we still need to reconstruct the global DeviceMesh mesh tensor from layouts. Especially
            for a list of layouts, we need to view it as a single flattened layout and view as size of list of numel of each
            layout.

            For example, if we have a layouts of ((2,4), (2,2)), we need to view it as a flattened layout of (4,2) and
            view it as a single mesh tensor of (2,2,2) if the world size is 8 like:
            [
               [[0, 2],
               [1, 3]],
               [[4, 6],
                [5, 7]]
            ]
            Rank 0 and 2 will get [[0, 2]]
            Rank 1 and 3 will get [[1, 3]]
            Rank 4 and 6 will get [[4, 6]]
            Rank 5 and 7 will get [[5, 7]]


            Args:
                device_type (str): The device type for the mesh (e.g., "cuda", "cpu")
                backend (_DeviceMeshBackend): Existing backend to use for the new mesh
                layouts (tuple[_Layout, ...]): Tuple of layout objects defining the mesh structure
                cur_rank (int): Current global rank to determine which part of the mesh this rank belongs to
                dim_names (Optional[Union[str, tuple[str, ...]]]): Names for the mesh dimensions

            Returns:
                DeviceMesh: A new DeviceMesh object with backend and layouts configured

            Note:
                This is an internal method primarily used for creating submeshes when slicing
                or transforming an existing DeviceMesh.
            """
            # Extract sizes and strides from layouts
            size_l, stride_l, mesh_size = [], [], []
            for layout in layouts:
                size_l.extend(layout.sizes)
                mesh_size.append(layout.numel())
                stride_l.extend(layout.strides)

            # Create combined layout and get ranks
            layout = _Layout(tuple(zip(size_l, stride_l)))
            pg_ranks_by_dim = layout.layout_to_global_ranks(not_none(get_world_size()))

            # Create tensor representation of the mesh
            tensor = torch.tensor(pg_ranks_by_dim, device="cpu", dtype=torch.int).view(
                -1, *mesh_size
            )

            # Find the mesh containing current rank
            nd_mesh = None
            for ndm in tensor:
                if cur_rank in ndm:
                    nd_mesh = ndm
                    break
            assert nd_mesh is not None, (
                f"Could not find the mesh containing the current rank {cur_rank}"
            )

            # Create device mesh without initializing backend
            device_mesh = DeviceMesh(
                device_type,
                nd_mesh,
                _init_backend=False,
            )

            # Set backend and layouts
            device_mesh._backend = backend
            device_mesh._layouts = layouts

            # Set dimension names if provided, we also need to ensure it is a tuple
            # so it is hashable
            if dim_names:
                device_mesh.mesh_dim_names = (
                    (dim_names,) if isinstance(dim_names, str) else tuple(dim_names)
                )

            return device_mesh

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
                f"({', '.join(f'{k}={v}' for k, v in zip(self.mesh_dim_names, self.mesh.shape))})"
                if self.mesh_dim_names
                else f"{tuple(self.mesh.shape)}"
            )
            device_mesh_repr = f"DeviceMesh({device_mesh_repr}, device: '{self.device_type}', stride: {self.mesh.stride()}"
            # We only print the mesh tensor if the debug mode is turned on.
            if os.environ.get("TORCH_DISTRIBUTED_DEBUG", "") == "DETAIL":
                device_mesh_repr += f", Mesh: {self.mesh.tolist()}"
            return f"{device_mesh_repr})"

        def __hash__(self):
            # lazily compute hash
            self._hash = getattr(self, "_hash", None)
            if not self._hash:
                self._hash = hash(
                    (
                        id(self._backend),
                        self._layouts,
                        self.mesh.shape,
                        self.device_type,
                        self.mesh_dim_names,
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
                    id(self._backend) == id(other._backend)
                    and self._layouts == other._layouts
                    and self.mesh.shape == other.mesh.shape
                    and self.device_type == other.device_type
                    and self.mesh_dim_names == other.mesh_dim_names
                )

        def __getitem__(
            self, mesh_dim_names: Union[str, tuple[str, ...]]
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
            if not self._backend:
                raise NotImplementedError(
                    "Slicing a DeviceMesh without backend initialized is not supported!"
                )

            mesh_dim_names = (
                (mesh_dim_names,) if isinstance(mesh_dim_names, str) else mesh_dim_names
            )

            if mesh_dim_names == self.mesh_dim_names:
                return self
            else:
                if not set(mesh_dim_names) <= self._backend.names_to_layouts.keys():
                    raise KeyError(
                        f"Invalid mesh_dim_names {mesh_dim_names} specified."
                    )

                layouts_sliced = [
                    self._backend.names_to_layouts[n] for n in mesh_dim_names
                ]
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
                    # When users sliced dim_names outside from current mesh, we will check whether
                    # there is layout overlap. Eventually we will just directly throw error here because
                    # we will deprecate the slicing of flattened dim_name from root mesh.
                    if set(self._layouts) < set(layouts_sliced):
                        sizes = tuple(x for l in layouts_sliced for x in l.sizes)
                        strides = tuple(x for l in layouts_sliced for x in l.strides)
                        base = torch.empty(sizes, dtype=torch.uint8)
                        t = torch.as_strided(base, size=sizes, stride=strides)
                        if torch._debug_has_internal_overlap(t):
                            raise RuntimeError(
                                f"slicing overlapping dim_names {mesh_dim_names} is not allowed"
                            )
                    res_mesh = DeviceMesh._from_backend(
                        self.device_type,
                        self._backend,
                        tuple(layouts_sliced),
                        self.get_rank(),
                        dim_names=mesh_dim_names,
                    )
                _mesh_resources.child_to_root_mapping[res_mesh] = (
                    _mesh_resources.get_root_mesh(self)
                )

                return res_mesh

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
            if not self._backend or not all(
                layout in self._backend.layouts_to_groups for layout in self._layouts
            ):
                raise RuntimeError("DeviceMesh backend not initialized!")

            if self.ndim > 1 and mesh_dim is None:
                raise RuntimeError(
                    f"Found the DeviceMesh have {self.ndim} dimensions",
                    "Optional kwarg `mesh_dim` needs to be specified when device_mesh.ndim > 1.",
                    "If you want to get the list of all the ProcessGroups in the DeviceMesh,"
                    "please use `get_all_groups()` instead.",
                )

            # Quick return if the current device_mesh is a 1D mesh.
            if self.ndim == 1 and mesh_dim is None:
                return not_none(
                    _resolve_process_group(
                        self._backend.layouts_to_groups[self._layouts[0]]
                    )
                )

            if isinstance(mesh_dim, str):
                if not self.mesh_dim_names:
                    raise ValueError(
                        "Cannot get group by name on a DeviceMesh without names"
                    )
                if mesh_dim not in self._backend.names_to_layouts:
                    raise ValueError(
                        f"Invalid named dim {mesh_dim!r} for DeviceMesh with names {self.mesh_dim_names}"
                    )
            elif isinstance(mesh_dim, int):
                if mesh_dim >= self.ndim:
                    raise ValueError(
                        f"Invalid mesh_dim {mesh_dim} for DeviceMesh with {self.ndim} dimensions"
                    )

            if isinstance(mesh_dim, str):
                layout = self._backend.names_to_layouts[mesh_dim]
            else:
                layout = self._layouts[not_none(mesh_dim)]

            return not_none(
                _resolve_process_group(self._backend.layouts_to_groups[layout])
            )

        def get_all_groups(self) -> list[ProcessGroup]:
            """
            Returns a list of ProcessGroups for all mesh dimensions.

            Returns:
                A list of :class:`ProcessGroup` object.
            """
            return [self.get_group(i) for i in range(self.ndim)]

        @staticmethod
        def from_group(
            group: Union[ProcessGroup, list[ProcessGroup]],
            device_type: str,
            mesh: Optional[Union[torch.Tensor, "ArrayLike"]] = None,
            *,
            mesh_dim_names: Optional[tuple[str, ...]] = None,
        ) -> "DeviceMesh":
            """
            Constructs a :class:`DeviceMesh` with ``device_type`` from an
            existing :class:`ProcessGroup` or a list of existing :class:`ProcessGroup`.

            The constructed device mesh has number of dimensions equal to the
            number of groups passed. For example, if a single process group is passed in,
            the resulted DeviceMesh is a 1D mesh. If a list of 2 process groups is passed in,
            the resulted DeviceMesh is a 2D mesh.

            If more than one group is passed, then the ``mesh`` and ``mesh_dim_names`` arguments
            are required. The order of the process groups passed in determines the topology of
            the mesh. For example, the first process group will be the 0th dimension of the DeviceMesh.
            The `mesh` tensor passed in must have the same number of dimensions as the number of process
            groups passed in, and the order of the dimensions in the `mesh` tensor must match the order
            in the process groups passed in.

            Args:
                group (ProcessGroup or list[ProcessGroup]): the existing ProcessGroup
                    or a list of existing ProcessGroups.
                device_type (str): The device type of the mesh. Currently supports: "cpu",
                    "cuda/cuda-like". Passing in a device type with a GPU index, such as "cuda:0",
                    is not allowed.
                mesh (torch.Tensor or ArrayLike, optional): A multi-dimensional array or an
                    integer tensor describing the layout of devices, where the IDs are global IDs
                    of the default process group. Default is None.
                mesh_dim_names (tuple[str], optional): A tuple of mesh dimension names to assign
                    to each dimension of the multi-dimensional array describing the layout of devices.
                    Its length must match the length of `mesh_shape`. Each string in `mesh_dim_names`
                    must be unique. Default is None.

            Returns:
                DeviceMesh: A :class:`DeviceMesh` object representing the device layout.
            """

            # 1D scenario
            if isinstance(group, ProcessGroup):
                group_ranks = get_process_group_ranks(group)
                mesh_list = (
                    mesh.tolist()
                    if isinstance(mesh, torch.Tensor)
                    else list(mesh or [])  # type: ignore[arg-type]
                )
                if mesh_list and mesh_list != group_ranks:
                    raise ValueError(
                        f"Invalid mesh {mesh_list} for ProcessGroup with ranks {group_ranks}"
                    )
                mesh = torch.tensor(group_ranks, device="cpu", dtype=torch.int)
                group = [group]

            # nD scenario
            if len(group) == 0:
                raise ValueError("Expects at least one ProcessGroup to be passed")
            if mesh is None:
                raise ValueError("Must pass mesh if passing multiple ProcessGroups")
            if mesh_dim_names is None and len(group) > 1:
                raise ValueError(
                    "Must pass mesh_dim_names if passing multiple ProcessGroups"
                )
            mesh = (
                mesh.detach().to(dtype=torch.int, device="cpu")
                if isinstance(mesh, torch.Tensor)
                else torch.tensor(mesh, device="cpu", dtype=torch.int)
            )
            if mesh.ndim != len(group):
                raise ValueError(
                    "Expects mesh with ndim equal to number of ProcessGroups but got "
                    f"mesh {mesh.tolist()} and {len(group)} ProcessGroups"
                )

            backend = _DeviceMeshBackend(device_type)
            layouts = init_layouts_from_mesh(mesh.size(), mesh.stride())
            if len(layouts) != len(group):
                raise ValueError(
                    f"zip arguments must have equal lengths for layouts {layouts} and groups {group}"
                )
            for i, (l, g) in enumerate(zip(layouts, group)):
                name = mesh_dim_names[i] if mesh_dim_names else None
                backend._maybe_create_backend(l, i, name, group=g)

            return DeviceMesh._from_backend(
                device_type, backend, layouts, get_rank(), dim_names=mesh_dim_names
            )

        def size(self, mesh_dim: Optional[int] = None) -> int:
            return (
                self.shape[mesh_dim] if mesh_dim is not None else math.prod(self.shape)
            )

        @property
        def ndim(self) -> int:
            return len(self._layouts)

        @property
        def shape(self) -> tuple[int, ...]:
            return tuple(l.numel() for l in self._layouts)

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
                    f"Found the DeviceMesh have {self.ndim} dimensions",
                    "Optional kwarg `mesh_dim` needs to be specified when device_mesh.ndim > 1.",
                )
            elif mesh_dim is None:
                mesh_dim = 0

            mesh_dim_group = not_none(self.get_group(mesh_dim))
            assert isinstance(mesh_dim_group, ProcessGroup), (
                "We expect ProcessGroup before calling `get_rank`!"
            )
            return not_none(get_rank(mesh_dim_group))

        def get_coordinate(self) -> Optional[list[int]]:
            """
            Return the relative indices of this rank relative to all
            dimensions of the mesh. If this rank is not part of the mesh, return None.
            """
            ranks = [get_rank(pg) for pg in self.get_all_groups()]
            return ranks if all(r != -1 for r in ranks) else None

        def _flatten(
            self,
            mesh_dim_name: Optional[str] = None,
            backend_override: Union[
                None, str, C10dBackend.Options, tuple[str, C10dBackend.Options]
            ] = None,
        ) -> "DeviceMesh":
            """
            Returns a 1D DeviceMesh by flattening the current DeviceMesh.

            If no mesh_dim_name is provided, the default is a string concatenating the mesh_dim_names of the
            given submesh with each mesh_dim_name separated by "_". For example, if we have a 3D mesh
            DeviceMesh([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], mesh_dim_names=("dp", "cp", "tp")), calling
            mesh_3d["dp", "cp"]._flatten() will create a 1D submesh DeviceMesh([0, 1, 2, 3], mesh_dim_names=("dp_cp",))
            on rank 0, 1, 2, 3 and a 1D submesh DeviceMesh([4, 5, 6, 7], mesh_dim_names=("dp_cp",)) on rank 4, 5, 6, 7.

            After the flattened dimension is created, to access the flattened dimension in mesh_3d, one can use the
            existing slicing method to obtain the flattened mesh through calling mesh_3d["dp_cp"].
            """
            if not self.mesh_dim_names:
                raise RuntimeError(
                    "Cannot flatten a DeviceMesh without mesh_dim_names!"
                )

            if not self._backend:
                raise NotImplementedError(
                    "flatten a device mesh without backend initialized is not supported!"
                )

            if not mesh_dim_name:
                mesh_dim_name = "_".join(not_none(self.mesh_dim_names))

            global_override = _mesh_resources.mesh_dim_group_options.get(
                0, (None, None)
            )
            if backend_override is None:
                backend_override_tuple = global_override
            else:
                (backend_override_tuple,) = _normalize_backend_override(
                    {0: backend_override}, 1
                )
                if backend_override_tuple == (None, None):
                    backend_override_tuple = global_override
                elif global_override != (None, None):
                    raise RuntimeError(
                        "Dimension 0 present both in the backend_override argument "
                        "and via _mesh_resources._set_mesh_dim_group_options"
                    )

            flattened_layout = _Layout(
                tuple(
                    sorted(
                        [ss for l in self._layouts for ss in l.sizes_and_strides],
                        key=lambda x: x[1],
                        reverse=True,
                    )
                )
            ).coalesce()

            self._backend._maybe_create_backend(
                flattened_layout,
                0,
                mesh_dim_name,
                backend_override=backend_override_tuple,
            )

            return DeviceMesh._from_backend(
                self.device_type,
                self._backend,
                (flattened_layout,),
                self.get_rank(),
                dim_names=(mesh_dim_name,) if mesh_dim_name else None,
            )

    def _normalize_backend_override(
        backend_override: dict[
            Union[int, str],
            Union[str, C10dBackend.Options, tuple[str, C10dBackend.Options]],
        ],
        ndim: int,
        mesh_dim_names: Optional[tuple[str, ...]] = None,
    ) -> Iterator[tuple[Optional[str], Optional[C10dBackend.Options]]]:
        if mesh_dim_names is None:
            mesh_dim_names = ()
        for dim_idx, dim_name in zip_longest(range(ndim), mesh_dim_names):
            if dim_name is not None and dim_name in backend_override:
                if dim_idx in backend_override:
                    raise RuntimeError(
                        f"Found redundant dim index {dim_idx} and "
                        f"name {dim_name} in backend_override"
                    )
                val = backend_override.pop(dim_name)
            elif dim_idx in backend_override:
                val = backend_override.pop(dim_idx)
            else:
                yield (None, None)
                continue

            if isinstance(val, str):
                yield (val, None)
            elif isinstance(val, C10dBackend.Options):
                yield (None, val)
            else:
                yield val

        if backend_override:
            raise RuntimeError(
                f"Found invalid keys in backend_override: got {list(backend_override.keys())}, "
                f"expected integers in range [0, {ndim}) or one of {mesh_dim_names}"
            )

    def init_device_mesh(
        device_type: str,
        mesh_shape: tuple[int, ...],
        *,
        mesh_dim_names: Optional[tuple[str, ...]] = None,
        backend_override: Optional[
            dict[
                Union[int, str],
                Union[str, C10dBackend.Options, tuple[str, C10dBackend.Options]],
            ]
        ] = None,
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
            device_type (str): The device type of the mesh. Currently supports: "cpu", "cuda/cuda-like", "xpu".
                Passing in a device type with a GPU index, such as "cuda:0", is not allowed.
            mesh_shape (Tuple[int]): A tuple defining the dimensions of the multi-dimensional array
                describing the layout of devices.
            mesh_dim_names (Tuple[str], optional): A tuple of mesh dimension names to assign to each dimension
                of the multi-dimensional array describing the layout of devices. Its length must match the length
                of `mesh_shape`. Each string in `mesh_dim_names` must be unique.
            backend_override (Dict[int | str, tuple[str, Options] | str | Options], optional): Overrides for some or all of
                the ProcessGroups that will be created for each mesh dimension. Each key can be either the index of a
                dimension or its name (if mesh_dim_names is provided). Each value can be a tuple containing the name
                of the backend and its options, or just one of these two components (in which case the other will be
                set to its default value).

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

        if backend_override is not None:
            backend_override_tuple = tuple(
                _normalize_backend_override(
                    backend_override, len(mesh_shape), mesh_dim_names
                )
            )
        else:
            backend_override_tuple = None

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
            backend_override=backend_override_tuple,
        )

        return device_mesh
