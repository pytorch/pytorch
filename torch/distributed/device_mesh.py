# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import os
import threading
import warnings
from collections.abc import Iterator
from itertools import zip_longest
from typing import Optional, TYPE_CHECKING, Union

import torch
from torch.distributed import is_available
from torch.distributed._mesh_layout import _MeshLayout
from torch.distributed._pycute import IntTuple, is_int, suffix_product
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

    BackendConfig = tuple[Optional[str], Optional[C10dBackend.Options]]
    torch.serialization.add_safe_globals([_MeshLayout])

    class _MeshEnv(threading.local):
        def __init__(self) -> None:
            self.mesh_stack: list[DeviceMesh] = []

        def get_current_mesh(self) -> "DeviceMesh":
            if len(self.mesh_stack) == 0:
                raise RuntimeError("No device mesh is currently active!")
            return self.mesh_stack[-1]

        # TODO: to remove it once we move all use cases into new API.
        def get_root_mesh(self, device_mesh: "DeviceMesh") -> "DeviceMesh":
            # If a mesh could not be found in the child_to_root_mapping, it is a root mesh itself.
            # A root mesh is not created through slicing.
            # We considers the root mesh of a root mesh is itself.
            # We keep this function for backward compatibility.
            warnings.warn(
                "This get_root_mesh API will be deprecated soon."
                "Please use `get_root_mesh` inside DeviceMesh instead.",
                stacklevel=2,
            )
            if not device_mesh:
                return device_mesh
            return device_mesh._get_root_mesh()

        @staticmethod
        def num_devices_per_host(device_type: str) -> int:
            return _get_device_handle(device_type).device_count()

        @staticmethod
        def num_hosts(device_type: str) -> int:
            # ProcessGroup can't tell us this info so we have to infer it, assume
            # homogeneous hardware for now
            return get_world_size() // _MeshEnv.num_devices_per_host(device_type)

        # TODO: to remove it once we move all use cases into new API.
        # We keep this API for backward compatibility.
        def _get_all_submeshes(
            self, device_mesh: "DeviceMesh", mesh_dim_name: str
        ) -> list["DeviceMesh"]:
            warnings.warn(
                "This _get_all_submeshes API will be deprecated soon."
                "Please use `_get_all_submeshes` inside DeviceMesh instead.",
                stacklevel=2,
            )
            return device_mesh._get_all_submeshes(mesh_dim_name)

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
            _rank (int): (experimental/internal)
                The global rank of the current process. If not provided, it will
                be inferred from the default process group.

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

        _device_type: str
        _rank_map: torch.Tensor
        _mesh_dim_names: Optional[tuple[str, ...]]
        _layout: _MeshLayout
        _root_mesh: Optional["DeviceMesh"] = None
        # Record flatten mesh name to its flattened mesh in root mesh.
        _flatten_mapping: dict[str, "DeviceMesh"]

        def __init__(
            self,
            device_type: str,
            mesh: Optional[Union[torch.Tensor, "ArrayLike"]] = None,
            *,
            mesh_dim_names: Optional[tuple[str, ...]] = None,
            backend_override: Optional[tuple[BackendConfig, ...]] = None,
            _init_backend: bool = True,
            _rank: Optional[int] = None,
            _layout: Optional[_MeshLayout] = None,
            _rank_map: Optional[torch.Tensor] = None,
            _root_mesh: Optional["DeviceMesh"] = None,
        ) -> None:
            # no-op in OSS, logs API usage metrics in meta-internal runs
            torch._C._log_api_usage_once(
                "torch.distributed.device_mesh.DeviceMesh.__init__"
            )
            if mesh is not None:
                if _layout is not None or _rank_map is not None:
                    raise TypeError(
                        "Cannot provide _layout and/or _rank_map if passing explicit mesh"
                    )
                if isinstance(mesh, torch.Tensor) and mesh.device.type != "cpu":
                    raise ValueError(f"`mesh` must be a CPU tensor, got {mesh}")
                mesh_tensor = (
                    mesh.detach().to(dtype=torch.int).contiguous()
                    if isinstance(mesh, torch.Tensor)
                    else torch.tensor(mesh, device="cpu", dtype=torch.int)
                )
                _layout = _MeshLayout(mesh_tensor.size(), mesh_tensor.stride())
                _rank_map = mesh_tensor.flatten()
            else:
                if _layout is None or _rank_map is None:
                    raise TypeError(
                        "The mesh argument is required except for PRIVATE USAGE ONLY!"
                    )

            assert _layout.check_non_overlap(), (
                "Please use a non-overlapping layout when creating a DeviceMesh."
            )
            assert _rank_map.ndim == 1, "The rank map must be 1-dimensional"
            assert _rank_map.is_contiguous(), "The rank map must be contiguous"
            assert _rank_map.numel() >= _layout.cosize(), (
                f"The rank map contains {_rank_map.numel()} element, "
                f"which isn't large enough for layout {_layout}"
            )

            self._device_type = device_type
            self._layout = _layout
            self._rank_map = _rank_map
            self._mesh_dim_names = tuple(mesh_dim_names) if mesh_dim_names else None
            self._root_mesh = _root_mesh

            if backend_override is None:
                backend_override = ((None, None),) * len(self._layout)
            elif len(backend_override) != len(self._layout):
                raise ValueError(
                    f"backend_override should have the same length as the number of mesh dimensions, "
                    f"but got {len(backend_override)} and {len(self._layout)}."
                )
            # Internal bookkeeping for the device mesh.
            self._layout = (
                _layout
                if _layout
                else _MeshLayout(self.mesh.size(), self.mesh.stride())
            )
            if not self._layout.check_non_overlap():
                raise AssertionError(
                    "Please use a non-overlapping layout when creating a DeviceMesh."
                )
            # Because we still need to support slicing of flattened dim from root mesh, so we don't check stride here.
            if self._layout.numel() != self.mesh.numel():
                raise AssertionError(
                    "Please use a valid layout when creating a DeviceMesh."
                    f"The layout {self._layout} is not consistent with the mesh size {self.mesh.size()}."
                )

            # private field to pre-generate DeviceMesh's hash
            self._flatten_mesh_list = tuple(self.mesh.flatten().tolist())
            self._thread_id = None
            # Initialize instance-specific flatten mapping
            self._flatten_mapping = {}

            # Skip process group initialization if xla device or init backend is False
            # TODO(yeounoh) implement DeviceMesh backend and register XLA backend.
            self._thread_id = None
            if device_type != "xla":
                # always try to create default (world) pg, even if it is not initialized
                # already. The world pg is used for device mesh identity (rank) on each
                # process (we need to know if the current global rank is in the mesh or not).
                if _init_backend:
                    self._setup_world_group_and_device()
                    self._dim_group_names = self._init_process_groups(
                        self._layout,
                        self._rank_map,
                        self._mesh_dim_names,
                        backend_override,
                    )

                if is_initialized() and get_backend() == "threaded":
                    # pyrefly: ignore [bad-assignment]
                    self._thread_id = threading.get_ident()

                if _rank is None:
                    _rank = get_rank()

                # calculate the coordinates of the current global rank on the mesh
                rank_coords = (self.mesh == _rank).nonzero()
                if rank_coords.size(0) not in (0, 1):
                    raise AssertionError(
                        f"rank_coords.size(0) must be 0 or 1, got {rank_coords.size(0)}"
                    )
                self._coordinate_on_dim: Optional[list[int]] = (
                    rank_coords[0].tolist() if rank_coords.size(0) > 0 else None
                )

            # private field to pre-generate DeviceMesh's hash
            self._flatten_rank_map = tuple(self._rank_map.tolist())
            # Initialize instance-specific flatten mapping
            self._flatten_mapping = {}

        @property
        def device_type(self) -> str:
            """Returns the device type of the mesh."""
            return self._device_type

        @property
        def mesh(self) -> torch.Tensor:
            """Returns the tensor representing the layout of devices."""
            full_mesh = self._layout.remap_to_tensor(self._rank_map)
            if full_mesh.size(0) == 1:
                return full_mesh[0]
            my_coords = (full_mesh == get_rank()).nonzero()
            if my_coords.size(0) > 0:
                return full_mesh[my_coords[0, 0]]
            raise RuntimeError(
                "In order to get the mesh Tensor of a DeviceMesh it needs to "
                "either have all its original dimensions (e.g., no slicing) "
                "or it needs to contain the local rank"
            )

        @property
        def mesh_dim_names(self) -> Optional[tuple[str, ...]]:
            """Returns the names of mesh dimensions."""
            return self._mesh_dim_names

        def _setup_world_group_and_device(self):
            default_initialized = is_initialized()
            # TODO: think about how to allow pg options to be passed to world group
            # or mesh dimension groups
            if not default_initialized:
                init_process_group()

            world_size = get_world_size()
            if self._layout.numel() > world_size:
                raise RuntimeError(
                    f"Mesh should not be bigger than default world size {world_size}, but found {self._layout.numel()} ranks!"
                )

            # ONLY set the device if the current device is not initialized, if user already
            # set the device before DeviceMesh init, we respect the user's choice.
            device_handle = _get_device_handle(self._device_type)
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
                        "device_id via `global_rank % num_devices_per_host`, assuming homogeneous hardware cluster. ",
                        stacklevel=2,
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
                            f"{world_size} ranks and {num_devices_per_host} {self._device_type} devices!"
                        )
                    device_handle.set_device(get_rank() % num_devices_per_host)

            return _get_default_group()

        @staticmethod
        def _init_one_process_group(
            sub_layout: _MeshLayout,
            rank_map: torch.Tensor,
            dim_name: str,
            backend_override: BackendConfig,
        ) -> Optional[str]:
            # Generate a 2D global mesh tensor for the current dim for PG creation.
            pg_ranks_by_dim = sub_layout.nest().remap_to_tensor(rank_map)
            backend, pg_options = backend_override
            # We need to explicitly pass in timeout when specified in option, otherwise
            # the default timeout will be used to override the timeout set in option.
            # TODO: remove this once we have fixed inside c10d level.
            timeout = pg_options._timeout if pg_options else None

            # If we have a 2D mesh with mesh_dim_names ("dp", "tp"), the group description
            # of the subgroups would be `mesh_dim_dp` and `mesh_name_tp`.
            # If the mesh doesn't have a mesh_dim_names, then the group description of the
            # subgroup would be `mesh_dim_0` and `mesh_dim_1`.
            group_desc = f"mesh_{dim_name}"

            dim_group = None
            default_group = _get_default_group()

            # Early return if there is only one sub_layout in the mesh layout.
            if sub_layout.numel() == get_world_size() and backend_override == (
                None,
                None,
            ):
                # Append the default pg to the first dim groups only if the default pg is compatible with `self._device_type`.
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
                return dim_group.group_name  # type: ignore[union-attr]

            # If bound_device_id exists, it means the nccl communicator has been eagerly initialized
            # so that we can use `split_group` to create subgroups through `ncclCommSplit`.
            # In this case, we only need to make one API call (`split_group``) for the subgroup creation
            # for each mesh dimension. In a 2 * 4 mesh, we only need to make two API calls per ranks to create
            # all the subgroups.
            # Otherwise, we need to make more than one API call (`new_group`) for subgroup creations. The
            # numbers of API calls are equal to the number of subgroups for each mesh dimension. In a 2 * 4
            # mesh, we need to make two API calls per ranks to create all the subgroups.
            if (
                getattr(default_group, "bound_device_id", None) is not None
                and torch.cuda.is_available()
                and (
                    backend is None
                    or default_group._get_backend(torch.device("cuda")).name()
                    == backend
                )
            ):
                dim_group = split_group(
                    parent_pg=default_group,
                    timeout=timeout,
                    pg_options=pg_options,
                    split_ranks=pg_ranks_by_dim.tolist(),
                    group_desc=group_desc,
                )
                return dim_group.group_name  # type: ignore[union-attr]

            # If the subgroup has been already created through `split_group`, we simply loop over `pg_ranks_by_dim`
            # and append the `group_name` to the `dim_group_names` list when the current rank is in the subgroup.
            # Otherwise, we use `new_group` instead of `split_group` to create subgroups by looping over `pg_ranks_by_dim`
            # along with appending information to the `dim_group_names` list whenever necessary.
            pg_name = None
            for dim_mesh in pg_ranks_by_dim:
                subgroup_ranks = dim_mesh.tolist()
                dim_group = new_group(
                    ranks=subgroup_ranks,
                    timeout=timeout,
                    backend=backend,
                    pg_options=pg_options,
                    group_desc=group_desc,
                )

                # only add to dim_groups if the current rank in the subgroup
                if get_rank() in subgroup_ranks:
                    if pg_name is not None:
                        raise RuntimeError(
                            f"Each device mesh dimension should get only one process group, but got {get_rank()} "
                            f"in {subgroup_ranks}!"
                        )
                    pg_name = dim_group.group_name
            return pg_name

        @staticmethod
        def _init_process_groups(
            layout: _MeshLayout,
            rank_map: torch.Tensor,
            mesh_dim_names: Optional[tuple[str, ...]],
            backend_override: tuple[BackendConfig, ...],
        ) -> list[str]:
            # group_name associated with each mesh dimension, each
            # mesh dimension should have one sub-group per rank
            dim_group_names: list[str] = []
            # create sub pgs base on the mesh argument specified
            for dim in range(len(layout)):
                dim_name = mesh_dim_names[dim] if mesh_dim_names else f"dim_{dim}"
                dim_group_names.append(
                    DeviceMesh._init_one_process_group(  # type: ignore[arg-type]
                        layout[dim], rank_map, dim_name, backend_override[dim]
                    )
                )
            if any(n is None for n in dim_group_names):
                assert all(n is None for n in dim_group_names)
                return []
            return dim_group_names

        def _get_root_mesh(self) -> "DeviceMesh":
            return self._root_mesh if self._root_mesh else self

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
                f"({', '.join(f'{k}={v}' for k, v in zip(self._mesh_dim_names, self._layout.top_level_sizes))})"
                if self._mesh_dim_names
                else f"{self._layout.top_level_sizes}"
            )
            device_mesh_repr = f"DeviceMesh({device_mesh_repr}, '{self.device_type}', stride={self._layout.strides}"
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
                        self._flatten_rank_map,
                        self._layout,
                        self._device_type,
                        self._mesh_dim_names,
                        self._thread_id,
                    )
                )
            return self._hash

        def __eq__(self, other: object) -> bool:
            if self is other:
                return True
            if not isinstance(other, DeviceMesh):
                return False
            return (
                self._flatten_rank_map == other._flatten_rank_map
                and self._layout == other._layout
                and self._device_type == other._device_type
                and self._mesh_dim_names == other._mesh_dim_names
                and self._thread_id == other._thread_id
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
            if not self._mesh_dim_names:
                raise RuntimeError("Cannot slice a DeviceMesh without mesh_dim_names!")

            mesh_dim_names = (
                (mesh_dim_names,) if isinstance(mesh_dim_names, str) else mesh_dim_names
            )

            if mesh_dim_names == self._mesh_dim_names:
                return self
            else:
                sliced_mesh_layout = self._get_slice_mesh_layout(mesh_dim_names)
                # When using FakeTensorMode to trace the model, `_create_sub_mesh()` will
                # fail as it will require a real tensor to manipulate.
                # `unset_fake_temporarily()` will allow us to materialize the tensors
                # within `_create_sub_mesh`, which should not affect modling.
                #
                # Note that this should be orthogonal to torch.compile(). But whether
                # we can compile device_mesh `slicing` (no graph break) is not verified
                # yet and need a follow-up,
                # TODO: compiler + device_mesh slicing.
                with torch._subclasses.fake_tensor.unset_fake_temporarily():
                    submesh = self._create_sub_mesh(sliced_mesh_layout, mesh_dim_names)
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
            if not hasattr(self, "_dim_group_names"):
                raise RuntimeError("DeviceMesh process groups not initialized!")

            if len(self._layout) > 1 and mesh_dim is None:
                raise RuntimeError(
                    f"Found the DeviceMesh have {len(self._layout)} dimensions",
                    "Optional kwarg `mesh_dim` needs to be specified when device_mesh.ndim > 1.",
                    "If you want to get the list of all the ProcessGroups in the DeviceMesh,"
                    "please use `get_all_groups()` instead.",
                )

            # Quick return if the current device_mesh is a 1D mesh.
            if len(self._layout) == 1 and mesh_dim is None:
                return not_none(_resolve_process_group(self._dim_group_names[0]))

            root_mesh = self._get_root_mesh()
            root_to_flatten_mapping = root_mesh._flatten_mapping
            if root_to_flatten_mapping and mesh_dim in root_to_flatten_mapping.keys():
                dim_group_name = root_to_flatten_mapping[
                    mesh_dim  # type: ignore[index]
                ]._dim_group_names[0]
                return not_none(_resolve_process_group(dim_group_name))
            else:
                mesh_dim = (
                    self._get_mesh_dim_by_name(mesh_dim)
                    if isinstance(mesh_dim, str)
                    else mesh_dim
                )
                if not isinstance(mesh_dim, int):
                    raise AssertionError(
                        f"mesh_dim must be an int, got {type(mesh_dim)}"
                    )
                return not_none(_resolve_process_group(self._dim_group_names[mesh_dim]))

        def get_all_groups(self) -> list[ProcessGroup]:
            """
            Returns a list of ProcessGroups for all mesh dimensions.

            Returns:
                A list of :class:`ProcessGroup` object.
            """
            return [self.get_group(i) for i in range(len(self._layout))]

        def _create_sub_mesh(
            self,
            layout: _MeshLayout,
            submesh_dim_names: tuple[str, ...],
        ) -> "DeviceMesh":
            root_mesh = self._get_root_mesh()
            slice_dim_group_name = []
            for name in submesh_dim_names:
                if name in not_none(self._mesh_dim_names):
                    slice_dim_group_name.append(
                        self._dim_group_names[  # type: ignore[has-type]
                            not_none(self._mesh_dim_names).index(name)
                        ]
                    )
                else:
                    # If device_mesh is not root_mesh, we already throw error in _get_slice_mesh_layout
                    # Since we will deprecate the slicing of flattened dim_name from root mesh soon,
                    # we don't want to optimize the code furthermore.
                    flatten_mesh = self._flatten_mapping[name]
                    slice_dim_group_name.append(
                        flatten_mesh._dim_group_names[  # type: ignore[has-type]
                            not_none(flatten_mesh._mesh_dim_names).index(name)
                        ]
                    )
            res_submesh = DeviceMesh(
                self._device_type,
                _layout=layout,
                _rank_map=root_mesh._rank_map,
                mesh_dim_names=submesh_dim_names,
                _root_mesh=root_mesh,
                _init_backend=False,
            )
            res_submesh._dim_group_names = slice_dim_group_name
            return res_submesh

        def _create_flatten_mesh(
            self,
            mesh_dim_name: Optional[str] = None,
            backend_override: BackendConfig = (None, None),
        ) -> "DeviceMesh":
            root_mesh = self._get_root_mesh()

            if not mesh_dim_name:
                mesh_dim_name = "_".join(not_none(self._mesh_dim_names))

            # Flatten a 1D device mesh into its original mesh_dim_name will return itself.
            if self.ndim == 1 and mesh_dim_name in not_none(self._mesh_dim_names):
                return self

            # Check whether the mesh_dim_name for flattened mesh is valid.
            invalid_dim_names = not_none(root_mesh._mesh_dim_names)
            if mesh_dim_name in invalid_dim_names:
                raise ValueError(
                    f"{mesh_dim_name} already exists for submesh of the {root_mesh}. ",
                    f"The mesh_dim_names of submesh and flattened mesh are {invalid_dim_names}. "
                    f"Please specify another valid mesh_dim_name.",
                )

            flattened_mesh_layout = self._layout.coalesce()
            if len(flattened_mesh_layout) > 1:
                flattened_mesh_layout = flattened_mesh_layout.nest()
            # Quick return if the flatten mesh has been created before.
            if mesh_dim_name in root_mesh._flatten_mapping:
                if (
                    flattened_mesh_layout
                    == root_mesh._flatten_mapping[mesh_dim_name]._layout
                ):
                    return root_mesh._flatten_mapping[mesh_dim_name]
                else:
                    raise ValueError(
                        f"Flatten mesh with mesh_dim_name {mesh_dim_name} has been created before, "
                        f"Please specify another valid mesh_dim_name."
                    )

            res_flattened_mesh = DeviceMesh(
                root_mesh._device_type,
                _layout=flattened_mesh_layout,
                _rank_map=root_mesh._rank_map,
                mesh_dim_names=(mesh_dim_name,),
                _root_mesh=root_mesh,
                backend_override=(backend_override,),
            )
            root_mesh._flatten_mapping[mesh_dim_name] = res_flattened_mesh

            return res_flattened_mesh

        def _get_root_mesh_dim(self) -> Optional[int]:
            """
            Returns the index of the mesh dim in the root mesh.
            The device_mesh passed in needs to be sliced out from the root mesh
            or submesh of the root mesh.
            """
            root_mesh = self._get_root_mesh()
            child_mesh_dim_names = self._mesh_dim_names
            if root_mesh and child_mesh_dim_names:
                if len(child_mesh_dim_names) != 1:
                    raise AssertionError("The submesh can only be a 1D mesh.")
                child_mesh_dim_name = child_mesh_dim_names[0]
                return root_mesh._get_mesh_dim_by_name(child_mesh_dim_name)
            return None

        def _get_mesh_dim_by_name(self, mesh_dim_name: str) -> int:
            if self._mesh_dim_names is None or len(self._mesh_dim_names) == 0:
                raise KeyError(
                    "No `mesh_dim_names` found.",
                )
            if mesh_dim_name not in self._mesh_dim_names:
                raise KeyError(
                    f"Mesh dimension '{mesh_dim_name}' does not exist.",
                    f"Available mesh dimensions are: mesh_dim_names={self._mesh_dim_names}",
                )
            return not_none(self._mesh_dim_names.index(mesh_dim_name))

        def _get_slice_mesh_layout(
            self, mesh_dim_names: tuple[str, ...]
        ) -> _MeshLayout:
            """
            Validate whether the mesh_dim_names is valid for slicing the given device_mesh.
            If valid, return dim indexes of the slice mesh in the device mesh.
            """
            slice_from_root = True
            if self != self._get_root_mesh():
                slice_from_root = False

            # The slice mesh_dim_names should consist either the current device_mesh's mesh_dim_names
            # or its flattened mesh's mesh_dim_names if it's root_mesh.
            flatten_name_to_root_layout = (
                {
                    key: mesh._layout
                    for key, mesh in self._get_root_mesh()._flatten_mapping.items()
                }
                if slice_from_root
                else {}
            )
            valid_mesh_dim_names = [
                *not_none(self._mesh_dim_names),
                *flatten_name_to_root_layout,
            ]

            if not all(
                mesh_dim_name in valid_mesh_dim_names
                for mesh_dim_name in mesh_dim_names
            ):
                raise KeyError(
                    f"Invalid mesh_dim_names {mesh_dim_names} specified. "
                    f"Valid mesh_dim_names are {valid_mesh_dim_names}."
                )

            layout_sliced = []
            for name in mesh_dim_names:
                if name in not_none(self._mesh_dim_names):
                    layout_sliced.append(
                        self._layout[not_none(self._mesh_dim_names).index(name)]
                    )
                elif name in flatten_name_to_root_layout:
                    warnings.warn(
                        "Slicing a flattened dim from root mesh will be deprecated in PT 2.11. "
                        "Users need to bookkeep the flattened mesh directly. ",
                        stacklevel=2,
                    )
                    layout_sliced.append(flatten_name_to_root_layout[name])

            sliced_sizes = tuple(l.sizes for l in layout_sliced)
            sliced_strides = tuple(l.strides for l in layout_sliced)

            # The check below is from DeviceMesh's implementation before adopting CuTe layout for internal
            # bookkeeping and it can be removed but we need to define what is the expected behavior.
            # TODO: Remove the below check and define the expected behavior.
            # Validate the order of the slice mesh dim indices.
            # This needs to be in ascending order.
            pre_stride = -1
            for stride in reversed(sliced_strides):
                # Note that with CuTe layout, we can support slicing flattened non-contiguous mesh dims with no problem.
                # But this will make this behavior complicated so we decided to not support it for now.
                if not is_int(stride):
                    raise NotImplementedError(
                        "Currently, this only allows slicing out a contiguous flattened dim."
                    )
                if stride < pre_stride:
                    raise KeyError(
                        f"Invalid mesh_dim_names {mesh_dim_names} specified. "
                        "Mesh dim indices should be in ascending order."
                    )
                pre_stride = stride

            # When users sliced dim_names outside from current mesh, we will check whether
            # there is layout overlap.
            # TODO: Eventually we will just directly throw error here because
            # we will deprecate the slicing of flattened dim_name from root mesh.
            layout_sliced = _MeshLayout(sliced_sizes, sliced_strides)
            if not layout_sliced.check_non_overlap():
                raise RuntimeError(
                    f"Slicing overlapping dim_names {mesh_dim_names} is not allowed."
                )

            return layout_sliced

        # TODO: to make this use case by other components public API in the future.
        def _get_all_submeshes(self, mesh_dim_name: str) -> list["DeviceMesh"]:
            """
            Return all the submeshes of a given mesh dimension of the device mesh.
            """
            mesh_dim = self._get_mesh_dim_by_name(mesh_dim_name)
            layout = self._layout[mesh_dim]
            pg_ranks_by_dim = layout.remap_to_tensor(self._rank_map)
            cur_rank = self.get_rank()
            res_submeshes = []
            for mesh_1d in pg_ranks_by_dim:
                submesh = DeviceMesh(
                    self._device_type,
                    mesh_1d,
                    mesh_dim_names=(mesh_dim_name,),
                    _init_backend=False,
                )
                submesh._dim_group_names = (  # type: ignore[has-type]
                    [self._dim_group_names[mesh_dim]]  # type: ignore[has-type]
                    if cur_rank in mesh_1d
                    else []
                )
                res_submeshes.append(submesh)

            return res_submeshes

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
                device_mesh._dim_group_names = [group.group_name]
                return device_mesh

            # nD scenario
            groups = list(group)
            if len(groups) == 0:
                raise ValueError("Expects at least one ProcessGroup to be passed")
            if mesh is None:
                raise ValueError("Must pass mesh if passing multiple ProcessGroups")
            if mesh_dim_names is None:
                raise ValueError(
                    "Must pass mesh_dim_names if passing multiple ProcessGroups"
                )
            # When init a DeviceMesh with multiple ProcessGroups directly, we need to make sure
            # the mesh tensor is contiguous. Otherwise, the layout we inferred from the mesh tensor
            # will have larger span than the actual tensor. This is just internal implementation detail
            # and does not affect user facing behavior.
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
            device_mesh._dim_group_names = [group.group_name for group in groups]
            return device_mesh

        def size(self, mesh_dim: Optional[int] = None) -> int:
            if mesh_dim is not None:
                return self._layout[mesh_dim].numel()
            return self._layout.numel()

        @property
        def ndim(self) -> int:
            return len(self._layout)

        @property
        def shape(self) -> tuple[int, ...]:
            return self._layout.top_level_sizes

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
                    f"Found the DeviceMesh have {len(self._layout)} dimensions",
                    "Optional kwarg `mesh_dim` needs to be specified when device_mesh.ndim > 1.",
                )
            elif mesh_dim is None:
                mesh_dim = 0

            mesh_dim_group = not_none(self.get_group(mesh_dim))
            if not isinstance(mesh_dim_group, ProcessGroup):
                raise AssertionError(
                    "We expect ProcessGroup before calling `get_rank`!"
                )
            return not_none(get_rank(mesh_dim_group))

        def get_coordinate(self) -> Optional[list[int]]:
            """
            Return the relative indices of this rank relative to all
            dimensions of the mesh. If this rank is not part of the mesh, return None.
            """
            return self._coordinate_on_dim if self._coordinate_on_dim else None

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
            mesh_3d["dp", "cp"]._flatten() will create a 1D submesh DeviceMesh([0, 2, 4, 6], mesh_dim_names=("dp_cp",))
            on rank 0, 2, 4, 6 and a 1D submesh DeviceMesh([1, 3, 5, 7], mesh_dim_names=("dp_cp",)) on rank 1, 3, 5, 7.

            After the flattened dimension is created, to access the flattened dimension in mesh_3d, one can use the
            existing slicing method to obtain the flattened mesh through calling mesh_3d["dp_cp"].
            """
            if not self._mesh_dim_names:
                raise RuntimeError(
                    "Cannot flatten a DeviceMesh without mesh_dim_names!"
                )

            if backend_override is not None:
                (backend_override_tuple,) = _normalize_backend_override(
                    {0: backend_override}, 1
                )
            else:
                backend_override_tuple = (None, None)

            return self._create_flatten_mesh(mesh_dim_name, backend_override_tuple)

        def _create_unflatten_mesh(
            self,
            dim: int,
            mesh_sizes: tuple[int, ...],
            mesh_dim_names: tuple[str, ...],
            backend_override: tuple[
                tuple[Optional[str], Optional[C10dBackend.Options]], ...
            ] = ((None, None),),
        ) -> "DeviceMesh":
            inner_layout = _MeshLayout(tuple(mesh_sizes), suffix_product(mesh_sizes))

            if inner_layout.numel() != self._layout[dim].numel():
                raise ValueError(
                    f"The product of {mesh_sizes=} is {inner_layout.numel()}, "
                    f"but the original dimension at dim={dim} has size {self._layout[dim].numel()}. "
                    f"These must be equal for unflatten to work correctly."
                )

            partial_layout = self._layout[dim].composition(inner_layout)
            unflattened_layout = self._layout.splice(dim, dim + 1, partial_layout)
            unflattened_mesh_dim_names = list(not_none(self.mesh_dim_names))
            unflattened_mesh_dim_names[dim : dim + 1] = list(mesh_dim_names)

            root_mesh = self._get_root_mesh()
            res_mesh = DeviceMesh(
                self.device_type,
                _layout=unflattened_layout,
                _rank_map=root_mesh._rank_map,
                mesh_dim_names=tuple(unflattened_mesh_dim_names),
                _root_mesh=root_mesh,
                _init_backend=False,
            )

            # If original mesh has initiated its backend, we need to initialize the backend
            # of unflatten dims as well.
            # TODO: To make backend init more efficient with cute layout representation and support
            # per dim backend init.
            if hasattr(self, "_dim_group_names"):
                dim_group_names = self._dim_group_names.copy()
                dim_group_names[dim : dim + 1] = self._init_process_groups(
                    partial_layout,
                    root_mesh._rank_map,
                    mesh_dim_names,
                    backend_override,
                )
                res_mesh._dim_group_names = dim_group_names

            return res_mesh

        def _unflatten(
            self,
            dim: Union[int, str],
            mesh_sizes: tuple[int, ...],
            mesh_dim_names: tuple[str, ...],
            backend_override: Optional[
                dict[
                    str,
                    Union[str, C10dBackend.Options, tuple[str, C10dBackend.Options]],
                ]
            ] = None,
        ) -> "DeviceMesh":
            """
            Returns a DeviceMesh by unflatten the current DeviceMesh.

            This api can be used to unflatten a N-D DeviceMesh into N-1+len(mesh_sizes)-D meshes or submeshes.
            The dim is the dimension to be unflattened which can be either a string or an integer.

            The mesh_sizes is a tuple which specifies the shape of the mesh unflatten into for the given dim.
            The mesh_dim_names is a list of strings which specifies the names of the dimensions of the mesh unflatten into.
            Its length must match the length of mesh_sizes.

            For example, if we have a 1D mesh DeviceMesh([0, 1, 2, 3, 4, 5, 6, 7], mesh_dim_names=("world")),
            calling mesh_1d._unflatten(0, (2, 2, 4), ["dp", "pp", "tp"]) will create a 3D mesh
            DeviceMesh([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], mesh_dim_names=("dp", "cp", "tp")).

            Note that after calling the unflatten, there is no access to the unflattened dimension in mesh_1d, one can only
            use the newly unflattened mesh to slice out the unflattened mesh dims.
            """
            if isinstance(dim, int) and dim >= self.ndim:
                raise ValueError(
                    f"dim {dim} specified in `_unflatten` is out of range {self.ndim}"
                )
            elif isinstance(dim, str) and dim in not_none(self.mesh_dim_names):
                raise ValueError(
                    f"dim {dim} specified in `_unflatten` is not in {self.mesh_dim_names}"
                )

            if len(mesh_sizes) != len(mesh_dim_names):
                raise RuntimeError(
                    "mesh_dim_names must have same length as mesh_sizes in _unflatten!"
                )

            if isinstance(dim, str):
                dim = not_none(self.mesh_dim_names).index(dim)

            if backend_override is not None:
                backend_override_tuple = tuple(
                    _normalize_backend_override(
                        backend_override,  # type: ignore[arg-type]
                        len(mesh_sizes),
                        mesh_dim_names,
                    )
                )
            else:
                backend_override_tuple = ((None, None),) * len(mesh_dim_names)

            return self._create_unflatten_mesh(
                dim,
                mesh_sizes,
                mesh_dim_names,
                backend_override_tuple,
            )

        @staticmethod
        def _concatenate(device_mesh_list: list["DeviceMesh"]) -> "DeviceMesh":
            concat_dim_names: list[str] = []
            concat_sizes: list[IntTuple] = []
            concat_strides: list[IntTuple] = []
            concat_dim_group_name: list[str] = []
            flatten_rank_map = device_mesh_list[0]._flatten_rank_map
            for dm in device_mesh_list:
                for i in range(len(dm._layout)):
                    concat_sizes.append(dm._layout[i].sizes)
                    concat_strides.append(dm._layout[i].strides)
                concat_dim_names.extend(not_none(dm.mesh_dim_names))
                concat_dim_group_name.extend(not_none(dm._dim_group_names))
                # Concatenate device mesh having different root mesh tensors are meaningless
                # because the concatenated indices should be indexed by the same root mesh tensor.
                if dm._flatten_rank_map != flatten_rank_map:
                    raise RuntimeError(
                        "Cannot concatenate DeviceMeshes derived from different device meshs"
                    )
            concat_mesh_layout = _MeshLayout(tuple(concat_sizes), tuple(concat_strides))
            if not concat_mesh_layout.check_non_overlap():
                raise RuntimeError(
                    f"Cannot concatenate overlapping meshes: {device_mesh_list}"
                )
            res_mesh = DeviceMesh(
                device_mesh_list[0].device_type,
                _layout=concat_mesh_layout,
                _rank_map=device_mesh_list[0]._rank_map,
                mesh_dim_names=tuple(concat_dim_names),
                _root_mesh=device_mesh_list[0]._get_root_mesh(),
                _init_backend=False,
            )
            res_mesh._dim_group_names = concat_dim_group_name
            return res_mesh

    def _normalize_backend_override(
        backend_override: dict[
            Union[int, str],
            Union[str, C10dBackend.Options, tuple[str, C10dBackend.Options]],
        ],
        ndim: int,
        mesh_dim_names: Optional[tuple[str, ...]] = None,
    ) -> Iterator[BackendConfig]:
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

        layout = _MeshLayout(tuple(mesh_shape), suffix_product(tuple(mesh_shape)))
        # Always initialize the (identity) rank map on CPU, regardless of what the
        # external device type has been set to be (e.g. meta)
        with torch.device("cpu"):
            rank_map = torch.arange(layout.numel(), dtype=torch.int)
        device_mesh = DeviceMesh(
            device_type=device_type,
            _layout=layout,
            _rank_map=rank_map,
            mesh_dim_names=mesh_dim_names,
            backend_override=backend_override_tuple,
        )

        return device_mesh
