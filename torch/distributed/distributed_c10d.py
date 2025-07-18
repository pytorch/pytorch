# mypy: allow-untyped-defs
"""Distributed Collective Communication (c10d)."""

import collections.abc
import contextlib
import ctypes
import hashlib
import io
import itertools
import logging
import os
import pickle
import sys
import time
import warnings
from collections import namedtuple
from datetime import timedelta
from typing import Any, Callable, Optional, TYPE_CHECKING, Union
from typing_extensions import deprecated

import torch
from torch._C import _DistStoreError as DistStoreError
from torch._C._distributed_c10d import (
    _DistributedBackendOptions,
    _register_process_group,
    _resolve_process_group,
    _unregister_all_process_groups,
    _unregister_process_group,
    AllgatherOptions,
    AllreduceCoalescedOptions,
    AllreduceOptions,
    AllToAllOptions,
    BarrierOptions,
    BroadcastOptions,
    DebugLevel,
    GatherOptions,
    get_debug_level,
    PrefixStore,
    ProcessGroup,
    ReduceOp,
    ReduceOptions,
    ReduceScatterOptions,
    ScatterOptions,
    Store,
    Work,
)
from torch._utils_internal import set_pytorch_distributed_envs_from_justknobs
from torch.monitor import _WaitCounter
from torch.overrides import handle_torch_function, has_torch_function
from torch.utils._typing_utils import not_none

from .c10d_logger import _exception_logger, _time_logger
from .constants import default_pg_nccl_timeout, default_pg_timeout
from .rendezvous import register_rendezvous_handler, rendezvous  # noqa: F401


__all__ = [
    "Backend",
    "BackendConfig",
    "GroupMember",
    "P2POp",
    "all_gather",
    "all_gather_coalesced",
    "all_gather_object",
    "all_reduce",
    "all_reduce_coalesced",
    "all_to_all",
    "all_to_all_single",
    "barrier",
    "batch_isend_irecv",
    "broadcast",
    "send_object_list",
    "recv_object_list",
    "broadcast_object_list",
    "destroy_process_group",
    "gather",
    "gather_object",
    "get_backend_config",
    "get_backend",
    "get_default_backend_for_device",
    "get_rank",
    "get_world_size",
    "get_pg_count",
    "group",
    "init_process_group",
    "irecv",
    "is_gloo_available",
    "is_initialized",
    "is_mpi_available",
    "is_backend_available",
    "is_nccl_available",
    "is_torchelastic_launched",
    "is_ucc_available",
    "is_xccl_available",
    "isend",
    "monitored_barrier",
    "new_group",
    "new_subgroups",
    "new_subgroups_by_enumeration",
    "recv",
    "reduce",
    "reduce_scatter",
    "scatter",
    "scatter_object_list",
    "send",
    "supports_complex",
    "AllreduceCoalescedOptions",
    "AllreduceOptions",
    "AllToAllOptions",
    "BarrierOptions",
    "BroadcastOptions",
    "GatherOptions",
    "PrefixStore",
    "ProcessGroup",
    "ReduceOp",
    "ReduceOptions",
    "ReduceScatterOptions",
    "ScatterOptions",
    "Store",
    "DebugLevel",
    "get_debug_level",
    "Work",
    "default_pg_timeout",
    "get_group_rank",
    "get_global_rank",
    "get_process_group_ranks",
    "reduce_op",
    "all_gather_into_tensor",
    "reduce_scatter_tensor",
    "get_node_local_rank",
    "split_group",
]

_MPI_AVAILABLE = True
_NCCL_AVAILABLE = True
_GLOO_AVAILABLE = True
_UCC_AVAILABLE = True
_XCCL_AVAILABLE = True

_pickler = pickle.Pickler
_unpickler = pickle.Unpickler


# Change __module__ of all imported types from torch._C._distributed_c10d that are public
def _export_c_types() -> None:
    _public_types_to_change_module = [
        AllreduceCoalescedOptions,
        AllreduceOptions,
        AllToAllOptions,
        BarrierOptions,
        BroadcastOptions,
        GatherOptions,
        PrefixStore,
        ProcessGroup,
        ReduceOp,
        ReduceOptions,
        ReduceScatterOptions,
        ScatterOptions,
        Store,
        DebugLevel,
        get_debug_level,
        Work,
    ]
    for type in _public_types_to_change_module:
        type.__module__ = "torch.distributed.distributed_c10d"


_export_c_types()

try:
    from torch._C._distributed_c10d import ProcessGroupMPI

    ProcessGroupMPI.__module__ = "torch.distributed.distributed_c10d"
    __all__ += ["ProcessGroupMPI"]
except ImportError:
    _MPI_AVAILABLE = False

try:
    from torch._C._distributed_c10d import ProcessGroupNCCL

    ProcessGroupNCCL.__module__ = "torch.distributed.distributed_c10d"
    __all__ += ["ProcessGroupNCCL"]
except ImportError:
    _NCCL_AVAILABLE = False

try:
    from torch._C._distributed_c10d import _ProcessGroupWrapper, ProcessGroupGloo

    ProcessGroupGloo.__module__ = "torch.distributed.distributed_c10d"
    __all__ += ["ProcessGroupGloo"]
except ImportError:
    _GLOO_AVAILABLE = False

try:
    from torch._C._distributed_c10d import ProcessGroupUCC

    ProcessGroupUCC.__module__ = "torch.distributed.distributed_c10d"
    __all__ += ["ProcessGroupUCC"]
except ImportError:
    _UCC_AVAILABLE = False

try:
    from torch._C._distributed_c10d import ProcessGroupXCCL

    ProcessGroupXCCL.__module__ = "torch.distributed.distributed_c10d"
    __all__ += ["ProcessGroupXCCL"]
except ImportError:
    _XCCL_AVAILABLE = False

logger = logging.getLogger(__name__)

PG_WRAPPER_STORE_PREFIX = "pg_wrapper"


# Some reduce ops are not supported by complex numbers and will result in an error.
# We currently provide complex support to the distributed API by viewing
# complex tensors as real (torch.view_as_real), meaning that calling
# these unsupported ops will return garbage values rather than error out.
# (e.g. max(2+3i, 3+2i) = 3+3i)
# We'd like calls to unsupported ops to error out accordingly,
# rather than returning garbage values.
def supports_complex(reduceOp: ReduceOp) -> bool:
    """Return true if reduce ops is supported. False otherwise."""
    denyList = [
        ReduceOp.MAX,
        ReduceOp.MIN,
        ReduceOp.PRODUCT,
        ReduceOp.BAND,
        ReduceOp.BOR,
        ReduceOp.BXOR,
    ]
    return reduceOp not in denyList


# TODO refactor into enum/strenum
class Backend(str):  # noqa: SLOT000
    """
    An enum-like class for backends.

    Available backends: GLOO, NCCL, UCC, MPI, XCCL, and other registered backends.

    The values of this class are lowercase strings, e.g., ``"gloo"``. They can
    be accessed as attributes, e.g., ``Backend.NCCL``.

    This class can be directly called to parse the string, e.g.,
    ``Backend(backend_str)`` will check if ``backend_str`` is valid, and
    return the parsed lowercase string if so. It also accepts uppercase strings,
    e.g., ``Backend("GLOO")`` returns ``"gloo"``.

    .. note:: The entry ``Backend.UNDEFINED`` is present but only used as
              initial value of some fields. Users should neither use it directly
              nor assume its existence.
    """

    UNDEFINED = "undefined"
    GLOO = "gloo"
    NCCL = "nccl"
    UCC = "ucc"
    MPI = "mpi"
    XCCL = "xccl"

    _BackendPlugin = namedtuple("_BackendPlugin", ["creator_fn", "extended_api"])

    _plugins: dict[str, _BackendPlugin] = {}

    backend_list = [UNDEFINED, GLOO, NCCL, XCCL, UCC, MPI]

    # 3rd-party devices can register the default backend support here
    default_device_backend_map: dict[str, str] = {
        "cpu": GLOO,
        "cuda": NCCL,
        "xpu": XCCL,
        "mps": GLOO,
    }

    backend_capability: dict[str, list[str]] = {
        GLOO: ["cpu", "cuda"],
        NCCL: ["cuda"],
        XCCL: ["xpu"],
        UCC: ["cpu", "cuda"],
        MPI: ["cpu", "cuda"],
    }

    backend_type_map: dict[str, ProcessGroup.BackendType] = {
        UNDEFINED: ProcessGroup.BackendType.UNDEFINED,
        GLOO: ProcessGroup.BackendType.GLOO,
        NCCL: ProcessGroup.BackendType.NCCL,
        XCCL: ProcessGroup.BackendType.XCCL,
        UCC: ProcessGroup.BackendType.UCC,
        MPI: ProcessGroup.BackendType.MPI,
    }

    def __new__(cls, name: str):
        """Create and return a new instance of the class."""
        if not isinstance(name, str):
            raise ValueError("Backend constructor parameter must be string-ish")
        value = getattr(Backend, name.upper(), Backend.UNDEFINED)

        if value == Backend.UNDEFINED:
            value = name.lower()
        return value

    @classmethod
    def register_backend(
        cls,
        name,
        func,
        extended_api=False,
        devices: Optional[Union[str, list[str]]] = None,
    ) -> None:
        """
        Register a new backend with the given name and instantiating function.

        This class method is used by 3rd party ``ProcessGroup`` extension to
        register new backends.

        Args:
            name (str): Backend name of the ``ProcessGroup`` extension. It
                        should match the one in ``init_process_group()``.
            func (function): Function handler that instantiates the backend.
                             The function should be implemented in the backend
                             extension and takes four arguments, including
                             ``store``, ``rank``, ``world_size``, and ``timeout``.
            extended_api (bool, optional): Whether the backend supports extended argument structure.
                                           Default: ``False``. If set to ``True``, the backend
                                           will get an instance of ``c10d::DistributedBackendOptions``, and
                                           a process group options object as defined by the backend implementation.
            device (str or list of str, optional): device type this backend
                            supports, e.g. "cpu", "cuda", etc. If `None`,
                            assuming both "cpu" and "cuda"

        .. note:: This support of 3rd party backend is experimental and subject to change.

        """
        # This takes care of CUSTOM Out-of-tree backend types, update in backend_list indicates availability
        if not hasattr(Backend, name.upper()):
            setattr(Backend, name.upper(), name.lower())
        if name.lower() not in Backend.backend_list:
            Backend.backend_list.append(name.lower())

        if devices is not None:
            for device in devices:
                if device not in Backend.default_device_backend_map:
                    Backend.default_device_backend_map[device] = name.lower()
        Backend.backend_type_map[name.lower()] = ProcessGroup.BackendType.CUSTOM

        # Update device capability matrix in Backend class
        if devices is None:
            # This is more of a backward support for groups like `threaded`:
            # assume default devices "cpu" and "cuda", but warn
            warnings.warn(
                f"Device capability of {name} unspecified, assuming `cpu` and "
                "`cuda`. Please specify it via the `devices` argument of "
                "`register_backend`."
            )
            Backend.backend_capability[name.lower()] = ["cpu", "cuda"]
        elif isinstance(devices, str):
            # Single device string specified. Simply convert to list.
            Backend.backend_capability[name.lower()] = [devices]
        else:
            Backend.backend_capability[name.lower()] = devices

        Backend._plugins[name.upper()] = Backend._BackendPlugin(func, extended_api)


class BackendConfig:
    """Backend configuration class."""

    def __init__(self, backend: Backend):
        """Init."""
        self.device_backend_map: dict[str, Backend] = {}
        backend = str(backend)

        if backend == Backend.UNDEFINED:
            # Detect the accelerator on the machine. If no accelerator is
            # available, it returns CPU.
            device_type = torch._C._get_accelerator().type
            try:
                backend_str = Backend.default_device_backend_map[device_type]
                self.device_backend_map[device_type] = Backend(backend_str)
            except KeyError:
                raise ValueError(
                    f"We detected accelerator {device_type} on your machine. "
                    f"But we don't know which communication backend to use for this accelerator. "
                    f"Please specify the `backend` argument in the `init_process_group` call."
                ) from None
        elif backend.lower() in Backend.backend_list:
            # Cases for when backend is a single string (without device types)
            # e.g. "nccl", "gloo", "ucc", "mpi"
            supported_devices = Backend.backend_capability[backend.lower()]
            backend_val = Backend(backend)
            self.device_backend_map = dict.fromkeys(supported_devices, backend_val)
        elif ":" in backend.lower():
            # Backend specified in "device:backend" format
            # make sure the backend string is in the correct format
            # "{device_type1}:{backend1},{device_type2}:{backend2}"
            # e.g. "cpu:gloo,cuda:nccl"
            backend_str_error_message = f"""The custom backend string argument is invalid: {backend}.
                Custom backend string is an experimental feature where the backend string must be in the format:
                "<device_type1>:<backend1>,<device_type2>:<backend2>...". e.g. 'cpu:gloo,cuda:nccl'"""

            # parse the backend string and populate the device_backend_map
            for device_backend_pair_str in backend.lower().split(","):
                device_backend_pair = device_backend_pair_str.split(":")
                if len(device_backend_pair) != 2:
                    raise ValueError(
                        f"Invalid device:backend pairing: \
                                     {device_backend_pair_str}. {backend_str_error_message}"
                    )
                device, backend = device_backend_pair
                if device in self.device_backend_map:
                    raise ValueError(
                        f"Duplicate device type {device} \
                                     in backend string: {backend}. {backend_str_error_message}"
                    )
                self.device_backend_map[device] = Backend(backend)
        else:
            # User specified a single backend name whose device capability is
            # unknown, assuming it can support the default devices of PyTorch
            # (cpu and cuda)
            warnings.warn(
                f"Device capability of {backend} unknown, assuming `cpu` and "
                "`cuda`. You can specify it in `device:backend` format in "
                "`init_process_group` call."
            )
            backend_val = Backend(backend)
            self.device_backend_map = {
                "cpu": backend_val,
                "cuda": backend_val,
                "xpu": backend_val,
            }

        logger.info("Using backend config: %s", self.device_backend_map)

    def __repr__(self):
        """Return all the device:backend pairs separated by commas."""
        return ",".join(
            f"{device}:{backend}" for device, backend in self.device_backend_map.items()
        )

    def get_device_backend_map(self) -> dict[str, Backend]:
        """Return backend map of the device."""
        return self.device_backend_map


class _reduce_op:
    r"""
    Deprecated enum-like class.

    For reduction operations: ``SUM``, ``PRODUCT``, ``MIN``, and ``MAX``.

    :class:`~torch.distributed.ReduceOp` is recommended to use instead.
    """

    def __init__(self) -> None:
        # __members__ is a dict storing key-value pairs for enum classes
        for k, v in ReduceOp.RedOpType.__members__.items():
            setattr(self, k, v)
        self.__members__ = ReduceOp.RedOpType.__members__

    @deprecated(
        "`torch.distributed.reduce_op` is deprecated, "
        "please use `torch.distributed.ReduceOp` instead",
        category=FutureWarning,
    )
    def __getattribute__(self, key):
        return object.__getattribute__(self, key)


reduce_op = _reduce_op()


class P2POp:
    """
    A class to build point-to-point operations for ``batch_isend_irecv``.

    This class builds the type of P2P operation, communication buffer, peer rank,
    Process Group, and tag. Instances of this class will be passed to
    ``batch_isend_irecv`` for point-to-point communications.

    Args:
        op (Callable): A function to send data to or receive data from a peer process.
            The type of ``op`` is either ``torch.distributed.isend`` or
            ``torch.distributed.irecv``.
        tensor (Tensor): Tensor to send or receive.
        peer (int, optional): Destination or source rank.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        tag (int, optional): Tag to match send with recv.
        group_peer (int, optional): Destination or source rank.
    """

    def __init__(
        self,
        op: Callable,
        tensor: torch.Tensor,
        peer: Optional[int] = None,
        group: Optional[ProcessGroup] = None,
        tag: int = 0,
        group_peer: Optional[int] = None,
    ):
        """Init."""
        self.op = op
        self.tensor = tensor
        self.group = _group_or_default_group(group)
        self.peer = _canonicalize_group_rank(
            self.group, peer, group_peer, return_global=True
        )
        self.tag = tag
        self.group_peer = _canonicalize_group_rank(self.group, peer, group_peer)

    def __new__(
        cls,
        op: Callable,
        tensor: torch.Tensor,
        peer: Optional[int] = None,
        group: Optional[ProcessGroup] = None,
        tag: int = 0,
        group_peer: Optional[int] = None,
    ):
        """Create and return a new instance of the class."""
        _check_op(op)
        _check_single_tensor(tensor, "tensor")

        return object.__new__(cls)

    def __repr__(self):
        my_group_rank = get_rank(self.group)
        op_name = self.op.__name__
        group_name = self.group.group_name if self.group else "default_pg"
        if "send" in op_name:
            s = my_group_rank
            d = self.group_peer
        elif "recv" in op_name:
            s = self.group_peer
            d = my_group_rank
        else:
            return super().__repr__()

        return f"P2POp({op_name} pg={group_name}, group_src={s}, group_dst={d},  {self.tensor.shape}, {self.tensor.dtype})"


class _CollOp:
    """
    A class to capture collective operations.

    Args:
        op (Callable): A collective function, e.g. ``torch.distributed.all_reduce``.
        tensor (Tensor): Tensor to operate on.
        dst_tensor (Tensor, optional): Provided when source and destination tensors are not the same.
        redop (ReduceOp, optional): reduce operation.
        root (int, optional): root of broadcast or reduce.
    """

    def __init__(
        self,
        op: Callable,
        tensor: torch.Tensor,
        dst_tensor: Optional[torch.Tensor] = None,
        redop: Optional[ReduceOp] = None,
        root: Optional[int] = None,
    ):
        self.op = op
        self.tensor = tensor
        self.dst_tensor = dst_tensor
        self.redop = redop
        self.root = root


# DO NOT USE THESE FIELDS DIRECTLY.
# Use them through the _world object to make sure the _world override mechanism
_pg_map: dict[ProcessGroup, tuple[str, Store]] = {}
_pg_names: dict[ProcessGroup, str] = {}
_pg_group_ranks: dict[ProcessGroup, dict[int, int]] = {}
# For a pg, it is a map from ProcessGroup to BackendConfig
_pg_backend_config: dict[ProcessGroup, str] = {}
_group_count = 0
_tags_to_pg: dict[str, list[ProcessGroup]] = {}
_pg_to_tag: dict[ProcessGroup, str] = {}
_backend: Optional[str] = None


class _World:
    """
    Container class for c10d process group state.

    This is used during registration and lookup of PG state.

    .. warning:: This is an experimental API intended to expose the inner workings
       of c10d and is subject to change..
    """

    def __init__(self) -> None:
        self._default_pg = None
        self._pg_coalesce_state: dict[ProcessGroup, list[_CollOp]] = {}

    @property
    def default_pg(self) -> Optional[ProcessGroup]:
        """
        Process group that includes all ranks of the cluster.

        This default ProcessGroup is used by c10d APIs when a ProcessGroup is needed
        but None is provided.
        """
        return self._default_pg

    @default_pg.setter
    def default_pg(self, value) -> None:
        self._default_pg = value

    @property
    def pg_map(self) -> dict[ProcessGroup, tuple[str, Store]]:
        """
        Provide Mapping from ProcessGroup to backend name and store.

        For NCCL and GLOO pg, it is a map from ProcessGroup to (Backend, Store)
        For MPI pg, it is a map from ProcessGroup to (Backend, None)

        TODO don't expose the map, expose fine grained ops
        """
        global _pg_map
        return _pg_map

    @property
    def pg_names(self) -> dict[ProcessGroup, str]:
        """
        Process group's names, map from ProcessGroup to str.

        TODO don't expose the map, expose fine grained ops
        """
        global _pg_names
        return _pg_names

    @property
    def pg_group_ranks(self) -> dict[ProcessGroup, dict[int, int]]:
        """
        Process group's global rank to local rank mapping.

        TODO don't expose the map, expose fine grained ops
        """
        global _pg_group_ranks
        return _pg_group_ranks

    @property
    def pg_backend_config(self) -> dict[ProcessGroup, str]:
        """
        Process group's backend config.

        TODO don't expose the map, expose fine grained ops
        """
        global _pg_backend_config
        return _pg_backend_config

    @property
    def group_count(self) -> int:
        """
        Process group count for default naming.

        TODO don't expose group_count, use something else instead
        """
        global _group_count
        return _group_count

    @group_count.setter
    def group_count(self, value: int) -> None:
        """Use to compute the name of ProcessGroups when using global synchronization."""
        global _group_count
        _group_count = value

    @property
    def tags_to_pg(self) -> dict[str, list[ProcessGroup]]:
        global _tags_to_pg
        return _tags_to_pg

    @property
    def pg_to_tag(self) -> dict[ProcessGroup, str]:
        global _pg_to_tag
        return _pg_to_tag

    @property
    def pg_coalesce_state(self) -> dict[ProcessGroup, list[_CollOp]]:
        return self._pg_coalesce_state

    @property
    def pg_config_info(self) -> list[dict[str, Any]]:
        """
        Return a list of dict with process groups and backends.

        Along with their unique IDs and configurations (types and ranks).
        """
        config_info: list[dict[str, Any]] = []
        default_pg_size = _get_group_size(None)
        for pg in self.pg_map.keys():
            ranks = self.pg_group_ranks[pg]
            config_info.append(
                {
                    "pg_name": self.pg_names[pg],
                    "pg_desc": pg.group_desc,
                    "backend_config": self.pg_backend_config[pg],
                    "ranks": (
                        list(ranks.keys()) if len(ranks) != default_pg_size else []
                    ),  # 'ranks' is an empty list when all ranks are involved in a pg
                    "group_size": len(ranks),
                    "group_count": self.group_count,
                }
            )
        return config_info


_world = _World()
"""Holds the singleton instance of ``_World`` used by c10. Experimental extension point to override it"""


class _WorldMeta(type):
    """
    Meta class of ``group`` and ``GroupMember``.

    Allows them to have the class property ``WORLD``.
    """

    # Points to the default PG once initialized.
    @property
    def WORLD(cls) -> Optional[ProcessGroup]:
        return _world.default_pg

    @WORLD.setter
    def WORLD(cls, pg: Optional[ProcessGroup]):
        _world.default_pg = pg


class group(metaclass=_WorldMeta):
    """Group class. Placeholder."""


class GroupMember(metaclass=_WorldMeta):
    """Group member class."""

    NON_GROUP_MEMBER = -100


def _get_default_timeout(backend: Backend) -> timedelta:
    # see note on nccl vs other backend timeout (constants.py)
    if backend == Backend.NCCL:
        if not isinstance(default_pg_nccl_timeout, timedelta):
            # TODO moco benchmark on CPU initializes pgnccl backend today, triggered this assert in CI before it was
            # changed to be a warning.  We should fix the moco model.
            warnings.warn(
                "Attempted to get default timeout for nccl backend, but NCCL support is not compiled"
            )
            return default_pg_timeout
        return default_pg_nccl_timeout
    else:
        return default_pg_timeout


def _check_valid_timeout(timeout: Any) -> None:
    if not isinstance(timeout, timedelta):
        raise TypeError(
            f"Expected timeout argument to be of type datetime.timedelta, got {timeout}"
        )


# Default process group state
_default_pg_init_method: Optional[str] = None

STORE_BASED_BARRIER_PREFIX = "store_based_barrier_key"


def _get_object_coll_device(group: Optional[ProcessGroup] = None) -> str:
    """
    .. note:: This is an internal helper and does not have backward
        compatibility, please use with caution.

    Return the device type to use with ``group`` for object collectives or
    barrier.

    There are selection rules:
        1. If user specifies exactly one backend in ``init_process_group`` call:
            use that backend
        2. Else if user specifies multiple "device:backend" pairs in init_process_group:
            If "cpu" is among those pairs, use "cpu" (because the object is in cpu memory);
            Otherwise, use the first backend (sort of a random pick).

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    Returns:
        str: The device type to use for object collective with ``group``.

    """
    group = group or _get_default_group()

    if not isinstance(group, ProcessGroup):
        warnings.warn(
            f"You are using a Backend {type(group)} as a ProcessGroup. "
            "This usage is deprecated since PyTorch 2.0. Please use a public API "
            "of PyTorch Distributed instead.",
        )
        # Provide backward compatibility to cases where `group` passed in is
        # actually a Backend (like `ProcessGroupGloo`) rather than a
        # `ProcessGroup` in PT 2.0 sense
        if isinstance(group, ProcessGroupGloo):
            # RPC uses Gloo for object collectives
            return "cpu"
        else:
            raise ValueError(f"Expecting a ProcessGroup, but got a {type(group)}.")

    """
    ``group._device_types`` is a property pybind that returns the devices
    ("cpu", "cuda", etc) supported by ``group``. Can be multiple if the
    ``group`` supports multiple devices.
    """
    devices = group._device_types

    if len(devices) == 1:
        # User fixed exactly one backend in `init_process_group`
        return devices[0].type
    elif len(devices) == 0:
        # No backend has been registered with this PG (maybe because no
        # collective has been run?) We pick cpu as the default and hopefully
        # this would lazily init Gloo or other available cpu backend.
        return "cpu"
    elif torch.device("cpu") in devices:
        # There are multiple backends in this PG and cpu is among them.
        # cpu is preferred as the object is in cpu memory. No need for device
        # copy.
        return "cpu"
    else:
        # No cpu in the backend list. Randomly pick the first backend
        return devices[0].type


def _get_pg_default_device(group: Optional[ProcessGroup] = None) -> torch.device:
    """
    .. note:: This method will be deprecated, it only stays for
        backward-compatiblity reason. Alternatives:

        - If you need to find a device for object collectives, please use
        `_get_object_coll_device(group)`.

        - If you need to query the device types supported by group, please use
        `_device_capability(group)`.

    Return the device type registered with ``group``.

    For example, if `init_process_group("nccl", ...)` was called, the returned
    value would be `torch.device("cuda")`.

    Errors out if no device has been registered.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    Returns:
        torch.device: The device type registered with ``group``.
    """

    warnings.warn(
        "`_get_pg_default_device` will be deprecated, it only stays for "
        "backward-compatiblity reason. If you need to find a device for object "
        "collectives, please use `_get_object_coll_device`. If you need to query "
        "the device types supported by group, please use "
        "`_device_capability(group)`. "
    )
    group = group or _get_default_group()

    if not isinstance(group, ProcessGroup):
        # Provide backward compatibility to cases where `group` passed in is
        # actually a Backend (like `ProcessGroupGloo`) rather than a
        # `ProcessGroup` in PT 2.0 sense
        warnings.warn(
            f"You are using a Backend {type(group)} as a ProcessGroup. "
            "This usage is deprecated since PyTorch 2.0. Please use a public API "
            "of PyTorch Distributed instead.",
            FutureWarning,
            stacklevel=3,
        )
        # Most users create Gloo with private API for object collectives
        return torch.device("cpu")

    """
    ``group._device_types`` is a property pybind that returns the devices
    ("cpu", "cuda", etc) supported by ``group``. Can be multiple if the
    ``group`` supports multiple devices.
    """
    devices = group._device_types

    if len(devices) == 1:
        # User fixed exactly one backend in `init_process_group`
        return devices[0]
    elif len(devices) == 0:
        raise RuntimeError(
            "Default device not found, because no backend has been registered "
            "with this ProcessGroup."
        )
    else:
        # There are multiple backends in this PG.
        if torch.device("cpu") in devices:
            rv = torch.device("cpu")
        else:
            rv = devices[0]
        warnings.warn(
            "Multiple backends are registered with this ProcessGroup. We cannot "
            f"determine which one is the default. Returning {rv}. "
            "Please consider using other APIs."
        )
        return rv


def _device_capability(group: Optional[ProcessGroup] = None) -> list[str]:
    """
    Return the device type(s) supported by ``group``.

    Args:
        group (ProcessGroup, optional): The process group to query. If None,
            the default process group will be used.

    Returns:
        List[str]: A list of device types supported by ``group``.
    """
    group = group or _get_default_group()
    return [device.type for device in group._device_types]


@_time_logger
def _store_based_barrier(
    rank,
    store,
    group_name,
    rendezvous_count,
    timeout,
    logging_interval=timedelta(seconds=10),
) -> None:
    """
    Store based barrier for synchronizing processes.

    Barrier based on store which is used for synchronizing processes after
    ``init_process_group`` or ``new_group``. Intended to be used only with
    those two methods and is not a generic alternative to ``barrier()``.
    """
    store_key = f"{STORE_BASED_BARRIER_PREFIX}:{group_name}"
    store.add(store_key, 1)
    logger.debug("Added key: %s to store for rank: %s", store_key, rank)

    # Now wait for all workers to check in with the store.
    world_size = rendezvous_count
    worker_count = store.add(store_key, 0)

    last_worker_key = f"{store_key}:last_worker"
    if worker_count == world_size:
        store.set(last_worker_key, "1")

    # adjust the timeout to be at least 10secs + 1sec per thousand ranks to reduce the odds of timeout
    # this value was empirically found while scale testing.
    logging_interval = max(logging_interval, timedelta(seconds=10 + world_size / 1000))

    start = time.time()
    while True:
        try:
            # This will throw an exception after the logging_interval in which we print out
            # the status of the group or time out officially, throwing runtime error
            store.wait([last_worker_key], logging_interval)
            break
        except RuntimeError as e:
            worker_count = store.add(store_key, 0)
            # Print status periodically to keep track.
            logger.debug(
                "Waiting in store based barrier to initialize process group for %s seconds"
                "rank: %s, key: %s (world_size=%s, num_workers_joined=%s, timeout=%s error=%s)",
                time.time() - start,
                rank,
                store_key,
                world_size,
                worker_count,
                timeout,
                e,
            )

            if timedelta(seconds=(time.time() - start)) > timeout:
                raise DistStoreError(  # noqa: B904
                    "Timed out initializing process group in store based barrier on "
                    f"rank {rank}, for key: {store_key} (world_size={world_size}, "
                    f"num_workers_joined={worker_count}, timeout={timeout} error={e})"
                )

    logger.info(
        "Rank %s: Completed store-based barrier for key:%s with %s nodes.",
        rank,
        store_key,
        world_size,
    )


def _rank_not_in_group(group: Optional[ProcessGroup]) -> bool:
    """Check if the current process's rank is not in a given group."""
    if group is None:
        return False
    return group == GroupMember.NON_GROUP_MEMBER


def _warn_not_in_group(op_name) -> None:
    global_rank = -1 if GroupMember.WORLD is None else GroupMember.WORLD.rank()
    warnings.warn(
        f"Running {op_name} on global rank {global_rank} which does not "
        "belong to the given group."
    )


def get_group_rank(group: ProcessGroup, global_rank: int) -> int:
    """
    Translate a global rank into a group rank.

    ``global_rank`` must be part of ``group`` otherwise this raises RuntimeError.

    Args:
        group (ProcessGroup): ProcessGroup to find the relative rank.
        global_rank (int): Global rank to query.

    Returns:
        Group rank of ``global_rank`` relative to ``group``

    N.B. calling this function on the default process group returns identity
    """
    if group is GroupMember.WORLD:
        return global_rank
    if group not in _world.pg_group_ranks:
        raise ValueError(
            f"Group {group} is not registered, please create group with torch.distributed.new_group API"
        )
    group_ranks = _world.pg_group_ranks[group]
    if global_rank not in group_ranks:
        raise ValueError(f"Global rank {global_rank} is not part of group {group}")

    return group_ranks[global_rank]


def get_global_rank(group: ProcessGroup, group_rank: int) -> int:
    """
    Translate a group rank into a global rank.

    ``group_rank`` must be part of `group` otherwise this raises RuntimeError.

    Args:
        group (ProcessGroup): ProcessGroup to find the global rank from.
        group_rank (int): Group rank to query.

    Returns:
        Global rank of ``group_rank`` relative to ``group``

    N.B. calling this function on the default process group returns identity
    """
    if group is GroupMember.WORLD:
        return group_rank
    if group not in _world.pg_group_ranks:
        raise ValueError(
            f"Group {group} is not registered, please create group with torch.distributed.new_group API"
        )
    for rank, grp_rank in _world.pg_group_ranks[group].items():
        if grp_rank == group_rank:
            return rank
    raise ValueError(f"Group rank {group_rank} is not part of group {group}")


# TODO: remove this once the ecosystem moves away from it.
@deprecated(
    "`torch.distributed.distributed_c10d._get_global_rank` is deprecated, "
    "please use `torch.distributed.distributed_c10d.get_global_rank` instead",
    category=FutureWarning,
)
def _get_global_rank(group, rank) -> int:
    """Use get_global_rank as this method is deprecated."""
    return get_global_rank(group, rank)


def get_process_group_ranks(group: Optional[ProcessGroup]) -> list[int]:
    """
    Get all ranks associated with ``group``.

    Args:
        group (Optional[ProcessGroup]): ProcessGroup to get all ranks from.
            If None, the default process group will be used.

    Returns:
        List of global ranks ordered by group rank.
    """
    return list(_world.pg_group_ranks[group or _get_default_group()].keys())


def _get_group_size(group) -> int:
    """Get a given group's world size."""
    if group is GroupMember.WORLD or group is None:
        default_pg = _get_default_group()
        return default_pg.size()
    return group.size()


def _get_group_size_by_name(group_name: str) -> int:
    group = _resolve_process_group(group_name)
    return group.size()


def _resolve_group_name_by_ranks_and_tag(ranks: list[int], tag: str) -> str:
    # TODO(yifu): remove this function once ranks + tag is not a supported
    # identifier for process group for functional collectives.
    group = _find_pg_by_ranks_and_tag(tag, ranks)
    if group is None:
        raise ValueError("")
    return group.group_name


def _check_single_tensor(param, param_name) -> None:
    """Check that the parameter ``param_name`` is a single tensor."""
    if not isinstance(param, torch.Tensor):
        raise TypeError(
            f"""Invalid function argument. Expected parameter `{param_name}` of type torch.Tensor
             but got {type(param)} instead."""
        )


def _check_tensor_list(param, param_name) -> None:
    """Check that the parameter ``param_name`` is a list of tensors."""
    if not isinstance(param, list):
        raise TypeError(
            f"""Invalid function argument. Expected parameter `{param_name}` of type List[torch.Tensor]
             but got {type(param)} instead."""
        )
    elif not all(isinstance(p, torch.Tensor) for p in param):
        raise TypeError(
            f"""Invalid function argument. Expected parameter `{param_name}` of type List[torch.Tensor]
             but got {type(param)} with elements of type {[type(p) for p in param]}."""
        )


def _group_or_default_group(group: Optional[ProcessGroup] = None) -> ProcessGroup:
    if group is None or group is GroupMember.WORLD:
        group = _get_default_group()
    return group


def _canonicalize_group_rank(
    group: ProcessGroup,
    global_rank: Optional[int] = None,
    group_rank: Optional[int] = None,
    return_global: bool = False,
) -> int:
    """
    Helper method to take _either_ a global rank or a group rank and produce a group rank.

    If 'return_global' is true, produce a global rank instead of a group rank.
    """

    if group_rank is not None:
        if global_rank is not None:
            raise ValueError("Can't specify both group_rank and global_rank")
        if return_global:
            return get_global_rank(group, group_rank)
    else:
        if global_rank is None:
            raise ValueError("Must specify global_rank or group_rank")
        if return_global:
            return global_rank
        group_rank = get_group_rank(group, global_rank)
    return group_rank


def _check_not_self_rank(group: ProcessGroup, rank: int, rank_type: str):
    if group.rank() == rank:
        raise ValueError(
            f"Invalid {rank_type} rank: {rank_type} rank should not be the same as "
            "the rank of the current process."
        )


def _as_iterable(obj) -> collections.abc.Iterable:
    return obj if isinstance(obj, list) else (obj,)


def _ensure_all_tensors_same_dtype(*tensors) -> None:
    last_dtype = None
    for tensor in itertools.chain.from_iterable(map(_as_iterable, tensors)):
        tensor_dtype = tensor.dtype
        # Mixing complex and its element type is allowed
        if tensor_dtype.is_complex:
            tensor_dtype = (
                torch.float32 if tensor_dtype == torch.complex64 else torch.complex128
            )

        if last_dtype is None:
            last_dtype = tensor_dtype
        else:
            if last_dtype != tensor_dtype:
                raise ValueError(
                    "Invalid usage of tensors with different dtypes"
                    f"Found {last_dtype} and  {tensor.dtype}"
                )


def _check_op(op) -> None:
    """Check that the ``op`` is either isend or irecv."""
    if op not in [isend, irecv]:
        raise ValueError(
            "Invalid ``op``. Expected ``op`` "
            "to be of type ``torch.distributed.isend`` or "
            "``torch.distributed.irecv``."
        )


def _check_p2p_op_list(p2p_op_list) -> None:
    """
    Check that the ``p2p_op_list`` is a list of P2POp instances.

    Also, check that all ops use the same group.
    """
    if not isinstance(p2p_op_list, list) or not all(
        isinstance(p2p_op, P2POp) for p2p_op in p2p_op_list
    ):
        raise ValueError(
            "Invalid ``p2p_op_list``. Each op is expected to "
            "to be of type ``torch.distributed.P2POp``."
        )

    group = p2p_op_list[0].group
    if not all(group == p2p_op.group for p2p_op in p2p_op_list):
        raise ValueError("All ops need to use the same group.")


def is_mpi_available() -> bool:
    """Check if the MPI backend is available."""
    return _MPI_AVAILABLE


def is_nccl_available() -> bool:
    """Check if the NCCL backend is available."""
    return _NCCL_AVAILABLE


def is_gloo_available() -> bool:
    """Check if the Gloo backend is available."""
    return _GLOO_AVAILABLE


def is_ucc_available() -> bool:
    """Check if the UCC backend is available."""
    return _UCC_AVAILABLE


def is_xccl_available() -> bool:
    """Check if the XCCL backend is available."""
    return _XCCL_AVAILABLE


def is_backend_available(backend: str) -> bool:
    """
    Check backend availability.

    Checks if the given backend is available and supports the built-in backends or
    third-party backends through function ``Backend.register_backend``.

    Args:
        backend (str): Backend name.
    Returns:
        bool: Returns true if the backend is available otherwise false.
    """
    # If the backend has an ``is_backend_available`` function, return the result of that function directly
    available_func = getattr(torch.distributed, f"is_{backend.lower()}_available", None)
    if available_func:
        return available_func()

    return backend.lower() in Backend.backend_list


def is_initialized() -> bool:
    """Check if the default process group has been initialized."""
    return GroupMember.WORLD is not None


def is_torchelastic_launched() -> bool:
    """
    Check whether this process was launched with ``torch.distributed.elastic`` (aka torchelastic).

    The existence of ``TORCHELASTIC_RUN_ID`` environment
    variable is used as a proxy to determine whether the current process
    was launched with torchelastic. This is a reasonable proxy since
    ``TORCHELASTIC_RUN_ID`` maps to the rendezvous id which is always a
    non-null value indicating the job id for peer discovery purposes..
    """
    return os.getenv("TORCHELASTIC_RUN_ID") is not None


def _is_barrier_after_init() -> int:
    # Environment variable to control whether process group should perform a
    # barrier after its init. Default value is 0, i.e. no barrier. If you
    # experience issue with this setting, you may set
    # `TORCH_DIST_INIT_BARRIER=1` to add the barrier.
    return int(os.getenv("TORCH_DIST_INIT_BARRIER", "0"))


def _get_default_group() -> ProcessGroup:
    """Get the default process group created by init_process_group."""
    if not is_initialized():
        raise ValueError(
            "Default process group has not been initialized, "
            "please make sure to call init_process_group."
        )
    if TYPE_CHECKING:
        return not_none(GroupMember.WORLD)
    else:
        return GroupMember.WORLD


def _get_default_store() -> Store:
    """Get the default store created by init_process_group."""
    if not is_initialized():
        raise ValueError(
            "Default process group has not been initialized, "
            "please make sure to call init_process_group."
        )
    default_pg = _get_default_group()
    _, default_store = _world.pg_map[default_pg]
    return default_store


def _update_default_pg(pg) -> None:
    _world.default_pg = pg
    rank = pg.rank() if pg is not None and pg != GroupMember.NON_GROUP_MEMBER else -1
    torch._C._distributed_c10d._set_global_rank(rank)


def get_backend_config(group: Optional[ProcessGroup] = None) -> str:
    """
    Return the backend configuration of the given process group.

    Args:
        group (ProcessGroup, optional): The process group to work on. The
            default is the general main process group. If another specific group
            is specified, the calling process must be part of :attr:`group`.

    Returns:
        The backend configuration of the given process group as a lower case string.

    """
    pg = group or _get_default_group()
    if _rank_not_in_group(pg):
        raise ValueError("Invalid process group specified")
    backend_config = _world.pg_backend_config.get(pg)
    return str(not_none(backend_config))


def get_backend(group: Optional[ProcessGroup] = None) -> Backend:
    """
    Return the backend of the given process group.

    Args:
        group (ProcessGroup, optional): The process group to work on. The
            default is the general main process group. If another specific group
            is specified, the calling process must be part of :attr:`group`.

    Returns:
        The backend of the given process group as a lower case string.

    """
    pg = group or _get_default_group()
    if _rank_not_in_group(pg):
        raise ValueError("Invalid process group specified")

    pg_store = _world.pg_map.get(pg, None)
    if pg_store is None:
        raise ValueError(
            f"Process group {pg} is not initialized in the world group map. Please initialize the group first."
        )

    return Backend(not_none(pg_store)[0])


def get_default_backend_for_device(device: Union[str, torch.device]) -> str:
    """
    Return the default backend for the given device.

    Args:
        device (Union[str, torch.device]): The device to get the default backend for.

    Returns:
        The default backend for the given device as a lower case string.

    """
    if isinstance(device, torch.device):
        device_str = device.type
    else:
        device_str = torch.device(device).type

    backend = Backend.default_device_backend_map.get(device_str)
    if backend is None:
        raise ValueError(f"Default backend not registered for device : {device}")

    return backend


def _get_process_group_uid(pg: ProcessGroup) -> int:
    backend = None
    try:
        backend = pg._get_backend(torch.device("cuda"))
    except RuntimeError:
        pass
    if is_nccl_available() and isinstance(backend, ProcessGroupNCCL):
        return backend.uid
    return -1


def _get_pg_config(group: Optional[ProcessGroup] = None) -> dict[str, Any]:
    """
    Return the pg configuration of the given process group.

    """
    pg = group or _get_default_group()
    return {
        "pg_name": _get_process_group_name(pg),
        "pg_desc": pg.group_desc,
        "backend_config": get_backend_config(pg),
        "pg_size": _get_group_size(pg),
        "ranks": get_process_group_ranks(pg),
    }


def _get_all_pg_configs() -> list[dict[str, Any]]:
    """
    Return the pg configuration of all the process groups.

    """
    config_info: list[dict[str, Any]] = [
        _get_pg_config(pg) for pg in _world.pg_map.keys()
    ]
    return config_info


def get_pg_count() -> int:
    """
    Return the number of process groups.

    """
    return _world.group_count


def get_node_local_rank(fallback_rank: Optional[int] = None) -> int:
    """
    Return the local rank of the current process relative to the node.

    Semantically, this is a useful concept for mapping processes to devices.
    For example, on a node with 8 accelerator you could use the node local rank to decide
    which accelerator device to bind the process to.

    In practice, the actual assignment of node local ranks is handled by the process launcher outside of pytorch,
    and communicated via the `LOCAL_RANK` environment variable.

    Torchrun will automatically populate `LOCAL_RANK`, but other launchers may not.  If `LOCAL_RANK` is unspecified,
    this API will fall back to the provided kwarg 'fallback_rank' if specified, otherwise it will raise an error. The
    intent is to allow writing an application that runs either in single or multi device contexts without error.

    """
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    elif fallback_rank is not None:
        return int(fallback_rank)
    raise RuntimeError(
        "LOCAL_RANK is not in the environment. Consider passing fallback_rank to allow `get_node_local_rank` to work, "
        "assuming you are not running in a multi-device context and want the code to run locally instead."
    )


def _add_ephemeral_timeout_for_all_pgs(timeout: timedelta) -> None:
    """
    This API adds an ephemeral timeout extension for all PGs locally
    on one rank. The timeout gets reset when the first collective issued
    after API called finished.
    NOTE: We only support to set timeout for cuda backends for now.
    NOTE: While this feature
    provides flexibility in specific scenarios, it introduces statefulness
    to timeout setting. Therefore, it is advisable to use this API sparingly
    and consider alternative approaches, such as directly setting the timeout
    or utilizing a barrier collective (one can set any timeout to the barrier),
    whenever feasible.

    Args:
        timeout (timedelta): The delta of timeout to extend.

    Returns:
        None.
    """
    for pg in _world.pg_map.keys():
        devices = pg._device_types
        if torch.device("cuda") in devices:
            backend = pg._get_backend(torch.device("cuda"))
            if is_nccl_available() and isinstance(backend, ProcessGroupNCCL):
                backend._add_ephemeral_timeout(timeout)


def _set_pg_timeout(timeout: timedelta, group: Optional[ProcessGroup] = None) -> None:
    """
    Set the timeout for the given process group when users want to use a different timeout instead of
    default values.

    Args:
        timeout (timedelta): Timeout for operations executed against the process group which
            users want to set. Default value is 10 minutes for NCCL and 30 minutes for other backends.
            This is the duration after which collectives will be aborted asynchronously and the process will crash.
            This is done since CUDA execution is async and it is no longer safe to continue executing user code since
            failed async NCCL operations might result in subsequent CUDA operations running on corrupted data.
            When TORCH_NCCL_BLOCKING_WAIT is set, the process will block and wait for this timeout.

        group (ProcessGroup, optional): The process group to work on. The
            default is the general main process group. If another specific group
            is specified, the calling process must be part of :attr:`group`.

    Returns:
        None
    """
    if group is None:
        group = _get_default_group()
    if _rank_not_in_group(group):
        raise ValueError("Invalid process group specified")
    assert isinstance(group, ProcessGroup)
    devices = group._device_types
    backends = set()
    if torch.device("cpu") in devices and is_gloo_available():
        backend = group._get_backend(torch.device("cpu"))
        if isinstance(backend, ProcessGroupGloo):
            backends.add(backend)
    if torch.device("cuda") in devices:
        backend = group._get_backend(torch.device("cuda"))
        if is_nccl_available() and isinstance(backend, ProcessGroupNCCL):
            backends.add(backend)  # type: ignore[arg-type]
        elif is_gloo_available() and isinstance(backend, ProcessGroupGloo):
            backends.add(backend)  # type: ignore[arg-type]
    if len(backends) == 0:
        warnings.warn("Set timeout is now only supported for either nccl or gloo.")
    for backend in backends:
        backend._set_default_timeout(timeout)


@_exception_logger
@_time_logger
def init_process_group(
    backend: Optional[str] = None,
    init_method: Optional[str] = None,
    timeout: Optional[timedelta] = None,
    world_size: int = -1,
    rank: int = -1,
    store: Optional[Store] = None,
    group_name: str = "",
    pg_options: Optional[Any] = None,
    device_id: Optional[Union[torch.device, int]] = None,
) -> None:
    """
    Initialize the default distributed process group.

    This will also initialize the distributed package.

    There are 2 main ways to initialize a process group:
        1. Specify ``store``, ``rank``, and ``world_size`` explicitly.
        2. Specify ``init_method`` (a URL string) which indicates where/how
           to discover peers. Optionally specify ``rank`` and ``world_size``,
           or encode all required parameters in the URL and omit them.

    If neither is specified, ``init_method`` is assumed to be "env://".


    Args:
        backend (str or Backend, optional): The backend to use. Depending on
            build-time configurations, valid values include ``mpi``, ``gloo``,
            ``nccl``, ``ucc``, or one that is registered by a third-party
            plugin.
            Since 2.6, if ``backend`` is not provided, c10d will use a backend
            registered for the device type indicated by the `device_id` kwarg
            (if provided). The known default registrations today are: ``nccl``
            for ``cuda``, ``gloo`` for ``cpu``.
            If neither ``backend`` nor ``device_id`` is provided, c10d will
            detect the accelerator on the run-time machine and use a backend
            registered for that detected accelerator (or ``cpu``).
            This field can be given as a lowercase string (e.g., ``"gloo"``),
            which can also be accessed via :class:`Backend` attributes (e.g.,
            ``Backend.GLOO``).
            If using multiple processes per machine with ``nccl`` backend, each
            process must have exclusive access to every GPU it uses, as sharing
            GPUs between processes can result in deadlock or NCCL invalid usage.
            ``ucc`` backend is experimental.
            Default backend for the device can be queried with
            :func:`get_default_backend_for_device`.
        init_method (str, optional): URL specifying how to initialize the
                                     process group. Default is "env://" if no
                                     ``init_method`` or ``store`` is specified.
                                     Mutually exclusive with ``store``.
        world_size (int, optional): Number of processes participating in
                                    the job. Required if ``store`` is specified.
        rank (int, optional): Rank of the current process (it should be a
                              number between 0 and ``world_size``-1).
                              Required if ``store`` is specified.
        store(Store, optional): Key/value store accessible to all workers, used
                                to exchange connection/address information.
                                Mutually exclusive with ``init_method``.
        timeout (timedelta, optional): Timeout for operations executed against
            the process group. Default value is 10 minutes for NCCL and 30 minutes for other backends.
            This is the duration after which collectives will be aborted asynchronously and the process will crash.
            This is done since CUDA execution is async and it is no longer safe to continue executing user code since
            failed async NCCL operations might result in subsequent CUDA operations running on corrupted data.
            When TORCH_NCCL_BLOCKING_WAIT is set, the process will block and wait for this timeout.

        group_name (str, optional, deprecated): Group name. This argument is ignored
        pg_options (ProcessGroupOptions, optional): process group options
            specifying what additional options need to be passed in during
            the construction of specific process groups. As of now, the only
            options we support is ``ProcessGroupNCCL.Options`` for the ``nccl``
            backend, ``is_high_priority_stream`` can be specified so that
            the nccl backend can pick up high priority cuda streams when
            there're compute kernels waiting. For other available options to config nccl,
            See https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#ncclconfig-t
        device_id (torch.device | int, optional): a single, specific device
            this process will work on, allowing for backend-specific
            optimizations.  Currently this has two effects, only under
            NCCL: the communicator is immediately formed (calling
            ``ncclCommInit*`` immediately rather than the normal lazy
            call) and sub-groups will use ``ncclCommSplit`` when
            possible to avoid unnecessary overhead of group creation. If you
            want to know NCCL initialization error early, you can also use this
            field. If an `int` is provided, the API assumes that the accelerator
            type at compile time will be used.

    .. note:: To enable ``backend == Backend.MPI``, PyTorch needs to be built from source
        on a system that supports MPI.

    .. note:: Support for multiple backends is experimental. Currently when no backend is
        specified, both ``gloo`` and ``nccl`` backends will be created. The ``gloo`` backend
        will be used for collectives with CPU tensors and the ``nccl`` backend will be used
        for collectives with CUDA tensors. A custom backend can be specified by passing in
        a string with format "<device_type>:<backend_name>,<device_type>:<backend_name>", e.g.
        "cpu:gloo,cuda:custom_backend".

    """

    global _world

    global _backend
    global _default_pg_init_method

    if GroupMember.WORLD is not None:
        raise ValueError("trying to initialize the default process group twice!")

    set_pytorch_distributed_envs_from_justknobs()

    # Depending on the import order, some trace_rules functions may be evaluated
    # during the import phase. In such a case, these functions may not correctly
    # add the distributed related rules due to import circular dependency.
    # We need to clear the lru_cache during the runtime to ensure the correctness
    # of these trace_rules.
    #
    # Since this API must be called before all distributed code being compiled,
    # clearing the cache here should be safe.
    if "torch._dynamo" in sys.modules:
        torch._dynamo.trace_rules.clear_lru_cache()

    assert (store is None) or (init_method is None), (
        "Cannot specify both init_method and store."
    )

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    # Get the compile-time accelerator type.
    # None indicates no accelerator support.
    acc = torch.accelerator.current_accelerator()

    # Auto complete device id
    if isinstance(device_id, int):
        if acc is None:
            raise ValueError(
                "device_id is an int, but no accelerator support is found from the current compilation. "
                "Please use a different compiled version that supports your accelerator."
            )
        device_id = torch.device(acc.type, device_id)

    # Sanity check device_id
    if device_id is not None and device_id.type != "cpu":
        # Type
        if acc is None or device_id.type != acc.type:
            raise ValueError(
                f"device_id {device_id} does not match the current compilation's accelerator support: {acc}. "
                "Please use a different compiled version that supports your accelerator."
            )
        # Index
        if device_id.index is None:
            raise ValueError("Please use a device_id with index.")
        # Range
        if device_id.index >= torch.accelerator.device_count():
            raise ValueError(
                f"device_id {device_id} is out of range. Please use a device index less than "
                f"the number of accelerators available: {torch.accelerator.device_count()}."
            )

    logger.info("Using device: %s", device_id)

    # If user did not provide a backend string but provided a device id, e.g.
    # >>> init_process_group(device_id=device)
    # we try to figure out the backend name based on the device type.
    if backend is None and device_id is not None:
        # Note: 3rd-party devices can register default backend through the
        # default map below.
        backend = Backend.default_device_backend_map.get(device_id.type)

    # If we still cannot figure it out, e.g.
    # >>> init_process_group()
    # we set it to `undefined` and rely on lazy init.
    if backend is None:
        backend = "undefined"

    # Convert string into `Backend` type
    backend = Backend(backend)

    if timeout is None:
        timeout = _get_default_timeout(backend)

    _check_valid_timeout(timeout)

    """
    Group name is not visible to users unless they access
    internals of c10d. This means we can ignore the value
    they provide as it not exposed in a public way.
    """
    group_name = _process_group_name([], use_hashed_name=False)
    if backend == Backend.MPI:
        if world_size != -1 or rank != -1:
            warnings.warn(
                f"For MPI backend, world_size ({world_size}) and rank ({rank}) "
                "are ignored since they are assigned by the "
                "MPI runtime."
            )

        default_pg, _ = _new_process_group_helper(
            -1,
            -1,
            [],
            backend,
            Store(),  # Placeholder value since store cannot be None
            group_name,
            timeout=timeout,
            group_desc="default_pg",
        )
        _update_default_pg(default_pg)
    else:
        # backward compatible API
        if store is None:
            rendezvous_iterator = rendezvous(
                not_none(init_method), rank, world_size, timeout=timeout
            )
            store, rank, world_size = next(rendezvous_iterator)
            store.set_timeout(timeout)

            # Use a PrefixStore to avoid accidental overrides of keys used by
            # different systems (e.g. RPC) in case the store is multi-tenant.
            store = PrefixStore("default_pg", store)

        default_pg, _ = _new_process_group_helper(
            world_size,
            rank,
            [],
            backend,
            store,
            group_name,
            backend_options=pg_options,
            timeout=timeout,
            device_id=device_id,
            group_desc="default_pg",
        )
        _update_default_pg(default_pg)

    _world.pg_group_ranks[GroupMember.WORLD] = {  # type: ignore[index]
        i: i
        for i in range(GroupMember.WORLD.size())  # type: ignore[attr-defined]
    }
    _backend = _world.pg_map[not_none(GroupMember.WORLD)][0]
    _default_pg_init_method = init_method

    old_hook = sys.excepthook
    excepthook_prefix = f"[rank{get_rank()}]"

    def _distributed_excepthook(*args):
        old_stderr = sys.stderr
        sys.stderr = buf = io.StringIO()
        try:
            old_hook(*args)
        finally:
            sys.stderr = old_stderr
        msg = buf.getvalue()
        msg = "\n".join(
            f"{excepthook_prefix}: {s}" if s != "" else "" for s in msg.split("\n")
        )
        sys.stderr.write(msg)
        sys.stderr.flush()

    sys.excepthook = _distributed_excepthook

    if _is_barrier_after_init() == 1:
        # barrier at the end to ensure that once we return from this method, all
        # process groups including global variables (if any) are updated
        # correctly on all ranks.
        # Update 04/2023: for large-scale runs, this barrier (esp. store-based
        # barrier) may be costly and/or unscalable. Also, in a lot of cases,
        # these barriers may be unnecessary, as proven by a green CI after
        # removal. An environment variable `TORCH_DIST_INIT_BARRIER` has been
        # added which enables this barrier only when set to 1.
        logger.debug(
            "Performing barrier after ProcessGroup initialization since "
            "TORCH_DIST_INIT_BARRIER = 1"
        )
        if backend == Backend.MPI:
            # MPI backend doesn't use store.
            barrier()
        else:
            # Use store based barrier here since barrier() used a bunch of
            # default devices and messes up NCCL internal state.
            _store_based_barrier(rank, store, group_name, world_size, timeout)


def _get_split_source(pg):
    split_from = None
    if pg.bound_device_id:
        split_from = pg._get_backend(pg.bound_device_id)
    elif pg is _world.default_pg:
        try:
            split_from = pg._get_backend(torch.device("cuda"))
        except RuntimeError:
            # no cuda device associated with this backend
            pass

    if not split_from or not split_from.supports_splitting:
        return None

    # If necessary, find a backend to split from by peeling process
    # group wrappers from our potentially wrapped process group.
    while _GLOO_AVAILABLE and isinstance(split_from, _ProcessGroupWrapper):
        split_from = split_from.wrapped_pg

    return split_from


def _new_process_group_helper(
    group_size,
    group_rank,
    global_ranks_in_group,
    backend,
    store,
    group_name,
    backend_options=None,
    timeout=None,
    pg_tag=None,
    device_id=None,
    group_desc=None,
):
    """
    Create a new distributed process group.

    This function must be called by ALL processes in the global group, even if
    the calling process is not part of the newly created group. In that case,
    this function returns GroupMember.NON_GROUP_MEMBER.

    This function is called with ``global_ranks_in_group == []`` for the default group.
    """
    global _world

    if group_name in _world.pg_names.values():
        raise ValueError(
            "The specified group name has already been "
            "created, please use a different group name"
        )

    if device_id is not None and (device_id.index is None or device_id.type == "cpu"):
        raise ValueError(
            "init_process_group device_id parameter must be an accelerator with an index"
        )

    # Note: _new_process_group_helper is only called from init_process_group, which always provides a timeout value
    _check_valid_timeout(timeout)

    if pg_tag not in [None, ""]:
        # creating with the same tag and rank set results in the same underlying PG
        existing_group = _find_pg_by_ranks_and_tag(pg_tag, global_ranks_in_group)
        if existing_group:
            _, prefix_store = _world.pg_map[existing_group]
            return existing_group, prefix_store

    group_desc = "undefined" if group_desc is None else group_desc

    # The list of group ranks is empty if we're creating the default group.
    is_default_group = len(global_ranks_in_group) == 0

    # nccl and potentially other backends allow creation of
    # communicators based on pre-existing ones, which can save
    # initialization time.  Due to lazy initialization of
    # communicators in some backends, we have to be careful and only
    # split when we *know* the default PG has already started communicator initialization.
    # We know this if we have bound a device id to the default pg (eager initialized).
    if is_initialized() and _get_default_group().bound_device_id:
        split_from = _get_split_source(_get_default_group())
    else:
        split_from = None

    # If this is a subgroup (which means group_ranks is specified),
    # we check if the current process is a member of the new group.
    if not is_default_group:
        global_rank = _get_default_group().rank()
        if global_rank not in global_ranks_in_group:
            # If we are using `ncclCommSplit` (or similar split from
            # other APIs) to create the communicator, we will need to
            # call `ncclCommSplit` on *all* ranks in this new group's
            # parent group, even those not in the new group.  This is
            # a requirement of the NCCL API as otherwise we would get
            # out of sync.
            if split_from:
                split_from.perform_nocolor_split(_get_default_group().bound_device_id)
            return GroupMember.NON_GROUP_MEMBER, None

    prefix_store = PrefixStore(f"{group_name}/", store)
    # The backend for PG will be set later based on what's inside BackendConfig
    # and timeout are set in each backend's option.
    pg: ProcessGroup = ProcessGroup(
        prefix_store,
        group_rank,
        group_size,
    )
    backend_config = BackendConfig(backend)
    # Set the default backend when single backend is passed in.
    if "," not in str(backend) and ":" not in str(backend):
        assert backend in Backend.backend_type_map, f"Unknown backend type {backend}"
        if backend == Backend.UNDEFINED:
            # Currently when backend is UNDEFINED, both ``gloo`` and ``nccl`` backends
            # will be created, we use nccl(if cuda is available) or gloo as default
            # backend so we can correctly call getDefaultBackend which in ProcessGroup.
            if Backend.NCCL in backend_config.get_device_backend_map().values():
                pg._set_default_backend(ProcessGroup.BackendType.NCCL)
            else:
                pg._set_default_backend(ProcessGroup.BackendType.GLOO)
        else:
            pg._set_default_backend(Backend.backend_type_map[backend])
    # In order to correctly call pg._has_hooks(), we should set the default backend
    # when multi backend is passed in
    else:
        if Backend.NCCL in backend_config.device_backend_map.values():
            pg._set_default_backend(ProcessGroup.BackendType.NCCL)
        elif Backend._plugins.keys():
            custom_backend = next(iter(Backend._plugins.keys()))
            if custom_backend in backend_config.device_backend_map.values():
                pg._set_default_backend(ProcessGroup.BackendType.CUSTOM)
        else:
            pg._set_default_backend(ProcessGroup.BackendType.GLOO)

    if device_id:
        pg.bound_device_id = device_id
    backend_class: torch._C._distributed_c10d.Backend
    for device, backend_str in backend_config.get_device_backend_map().items():
        # Use the group name as prefix in the default store, such that
        # a single store can be reused by multiple groups.
        backend_prefix_store = PrefixStore(f"{device}/", prefix_store)

        if backend_str == Backend.MPI:
            if not is_mpi_available():
                raise RuntimeError(
                    "Distributed package doesn't have MPI built in."
                    " MPI is only included if you build PyTorch from"
                    " source on a host that has MPI installed."
                )
            backend_class = ProcessGroupMPI.create(global_ranks_in_group)
            backend_type = ProcessGroup.BackendType.MPI
            if not backend_class:
                return GroupMember.NON_GROUP_MEMBER, None
            # create new process group with accurate rank and size
            if pg.rank() == -1 and pg.size() == -1:
                pg = ProcessGroup(
                    backend_prefix_store,
                    backend_class.rank(),
                    backend_class.size(),
                )
                pg._set_default_backend(backend_type)
        elif backend_str == Backend.GLOO:
            # TODO: remove this check after lazy initialization is supported
            # if pg_options is not None:
            #     raise RuntimeError("GLOO options not supported")
            if not is_gloo_available():
                raise RuntimeError("Distributed package doesn't have Gloo built in")
            backend_class = ProcessGroupGloo(
                backend_prefix_store, group_rank, group_size, timeout=timeout
            )
            backend_class.options.global_ranks_in_group = global_ranks_in_group
            backend_class.options.group_name = group_name
            backend_type = ProcessGroup.BackendType.GLOO
        elif backend_str == Backend.NCCL:
            if not is_nccl_available():
                raise RuntimeError("Distributed package doesn't have NCCL built in")
            if backend_options is not None:
                assert isinstance(backend_options, ProcessGroupNCCL.Options), (
                    "Expected backend_options argument to be of type ProcessGroupNCCL.Options"
                )
                if backend_options._timeout != timeout:
                    warnings.warn(
                        "backend_options._timeout was specified, "
                        "but timeout kwarg has a default value that will always override it. "
                    )
            else:
                # default backend_options for NCCL
                backend_options = ProcessGroupNCCL.Options()
                backend_options.is_high_priority_stream = False
            backend_options._timeout = timeout

            if split_from:
                backend_options.split_from = split_from
                backend_options.split_color = _process_group_color(
                    global_ranks_in_group
                )
            backend_options.global_ranks_in_group = global_ranks_in_group
            backend_options.group_name = group_name
            backend_class = ProcessGroupNCCL(
                backend_prefix_store, group_rank, group_size, backend_options
            )
            backend_type = ProcessGroup.BackendType.NCCL
        elif backend_str == Backend.UCC and is_ucc_available():
            # TODO: once UCC plugin is fully deprecated, remove
            # is_ucc_available() from above elif-condition and raise
            # RuntimeError if is_ucc_available() returns false.

            backend_class = ProcessGroupUCC(
                backend_prefix_store, group_rank, group_size, timeout=timeout
            )
            backend_type = ProcessGroup.BackendType.UCC
        elif backend_str == Backend.XCCL:
            if not is_xccl_available():
                raise RuntimeError("Distributed package doesn't have XCCL built in")
            backend_class = ProcessGroupXCCL(
                backend_prefix_store, group_rank, group_size
            )
            backend_type = ProcessGroup.BackendType.XCCL
        else:
            assert backend_str.upper() in Backend._plugins, (
                f"Unknown c10d backend type {backend_str.upper()}"
            )

            backend_plugin = Backend._plugins[backend_str.upper()]
            creator_fn = backend_plugin.creator_fn
            extended_api = backend_plugin.extended_api
            backend_type = ProcessGroup.BackendType.CUSTOM

            if not extended_api:
                backend_class = creator_fn(
                    backend_prefix_store, group_rank, group_size, timeout
                )
            else:
                dist_backend_opts = _DistributedBackendOptions()
                dist_backend_opts.store = backend_prefix_store
                dist_backend_opts.group_rank = group_rank
                dist_backend_opts.group_size = group_size
                dist_backend_opts.timeout = timeout
                dist_backend_opts.group_id = group_name
                dist_backend_opts.global_ranks_in_group = global_ranks_in_group

                backend_class = creator_fn(dist_backend_opts, backend_options)

        # Set sequence numbers for gloo and nccl backends.
        if backend_str == Backend.GLOO:
            assert isinstance(backend_class, ProcessGroupGloo)
            backend_class._set_sequence_number_for_group()
        elif backend_str == Backend.NCCL:
            assert isinstance(backend_class, ProcessGroupNCCL)
            backend_class._set_sequence_number_for_group()

        # If the type is a subclass of ProcessGroup then return this process group immediately
        # TODO: This defaults to the old behavior for PythonProcessGroups which overwrites the
        # ProcessGroup instance
        if issubclass(type(backend_class), ProcessGroup):
            pg = backend_class  # type: ignore[assignment]
            break

        # Process group wrapper initialization for supported PGs when TORCH_DISTRIBUTED_DEBUG is set
        if (
            backend_str in [Backend.GLOO, Backend.NCCL, Backend.UCC]
            or backend_str.upper() in Backend._plugins
        ):
            # In debug mode and if GLOO is available, wrap in a wrapper PG that
            # enables enhanced collective checking for debuggability.
            if get_debug_level() == DebugLevel.DETAIL:
                if not _GLOO_AVAILABLE:
                    logger.info(
                        """TORCH_DISTRIBUTED_DEBUG was set to DETAIL, but
                                GLOO is not available. Build with Gloo to
                                create a wrapper process group in debug mode
                                to aid collective desynchronization debugging."""
                    )
                else:
                    backend_class = _create_process_group_wrapper(
                        wrapped_pg=backend_class,
                        store_prefix=group_name,
                        store=backend_prefix_store,
                        rank=group_rank,
                        world_size=group_size,
                        timeout=timeout,
                    )

        # register only a single backend when all get_device_backend_map values are the same
        if len(set(backend_config.get_device_backend_map().values())) == 1:
            for device in backend_config.get_device_backend_map().keys():
                pg._register_backend(torch.device(device), backend_type, backend_class)

            # break out of outer loop to not create any more backends
            break

        pg._register_backend(torch.device(device), backend_type, backend_class)

    # set group_name and group_dsec to backend
    assert group_name is not None
    assert group_desc is not None
    pg._set_group_name(group_name)
    pg._set_group_desc(group_desc)

    if device_id and pg._get_backend(device_id).supports_splitting:
        eager_backend = pg._get_backend(device_id)
        eager_backend.eager_connect_single_device(device_id)

    # update global state
    _world.pg_map[pg] = (backend, prefix_store)
    _world.pg_names[pg] = group_name
    _register_process_group(group_name, pg)

    _world.pg_backend_config[pg] = str(backend_config)
    # "" is the default tag for user PGs
    if pg_tag in [None, ""]:
        pg_tag = f"ptd:{group_name}"
        _world.tags_to_pg.setdefault("", []).append(pg)
    else:
        pg_tag = f"user:{pg_tag}"

    _world.tags_to_pg.setdefault(pg_tag, []).append(pg)
    _world.pg_to_tag[pg] = pg_tag
    return pg, prefix_store


def destroy_process_group(group: Optional[ProcessGroup] = None):
    """
    Destroy a given process group, and deinitialize the distributed package.

    Args:
        group (ProcessGroup, optional): The process group to be destroyed, if
                                        group.WORLD is given, all process
                                        groups including the default one will
                                        be destroyed.
    """
    global _world

    if group == GroupMember.NON_GROUP_MEMBER:
        return

    if group is None:
        pg = GroupMember.WORLD
    else:
        pg = group

    assert pg is not None
    if _world.pg_map.get(pg, None) is None:
        raise ValueError("Invalid process group specified")

    # When users register Python onCompletion hooks, those hooks will run on a
    # different thread than the main thread. Today, the ProcessGroup dtor does
    # wait for that thread. However, the dtor might finish after the Python
    # Interpreter exits. After that grabbing the GIL for the Python hook will crash.
    # We can either revive the interpreter when running hooks or keep the main one
    # alive until all works and hooks are done. The current implementation does the
    # latter. Therefore, we explicitly call _wait_for_pending_works() here to wait
    # for the pending hooks to finish.
    if type(pg) == ProcessGroup and pg._has_hooks():
        pg._wait_for_pending_works()

    if group is None or group == GroupMember.WORLD:
        # shutdown all backends in the order of pg names. shutting down in order because
        # ncclCommAbort() was a 'collective' call in some versions of NCCL.
        for pg_to_shutdown in sorted(
            _world.pg_names, key=lambda x: _world.pg_names[x], reverse=True
        ):
            pg_to_shutdown.shutdown()

        _update_default_pg(None)
        _world.pg_map.clear()
        _world.pg_names.clear()
        _world.pg_group_ranks.clear()
        _world.pg_backend_config.clear()
        _world.pg_to_tag.clear()
        _world.tags_to_pg.clear()
        _world.pg_coalesce_state.clear()
        _unregister_all_process_groups()

        # when process group doesn't have an explicit name (only WORLD (default)
        # process group can have an explicit name), we use global _world.group_count
        # to generate the name. We need to reset the counter on destruction to
        # allow consistent value to be generated when we re-create process
        # groups after some trainers recover from failure
        #
        # We only reset this when WORLD is being destroyed because if this
        # process group is in good state, we aren't dealing with failures.
        _world.group_count = 0
    else:
        pg.shutdown()
        del _world.pg_map[pg]
        del _world.pg_names[pg]
        del _world.pg_group_ranks[pg]
        del _world.pg_backend_config[pg]
        if pg in _world.pg_coalesce_state.keys():
            warnings.warn(
                "Some coalesced collectives haven't been launched when "
                "ProcessGroup is destroyed. They will be cleaned."
            )
            del _world.pg_coalesce_state[pg]

        tag = _world.pg_to_tag.get(pg)
        del _world.pg_to_tag[pg]
        if tag is not None:
            try:
                _world.tags_to_pg[tag].remove(pg)
                if tag.startswith("ptd:"):
                    _world.tags_to_pg[""].remove(pg)
            except Exception:
                pass
        _unregister_process_group(pg.group_name)


def _abort_process_group(group: Optional[ProcessGroup] = None):
    """
    Abort a given process group. If group.WORLD (i.e. `None`) is given, all
    process groups including the default one will be aborted.

    Args:
        group (ProcessGroup, optional): The process group to be aborted.

    .. note:: this API is experimental and currently only works with the NCCL
        backend.

    .. note:: this API should be used with `TORCH_NCCL_ASYNC_ERROR_HANDLING`
        turned off (i.e. set to 0). Otherwise, ProcessGroupNCCL's watchdog may
        automatically handle errors or timeouts for you including aborting the
        ProcessGroup.
    """
    global _world

    if group == GroupMember.NON_GROUP_MEMBER:
        return

    pg = group or GroupMember.WORLD

    assert pg is not None
    if _world.pg_map.get(pg, None) is None:
        raise ValueError("Invalid process group specified or has been destroyed.")

    try:
        backend = pg._get_backend(torch.device("cuda"))
    except RuntimeError:
        backend = None

    if group is None or group == GroupMember.WORLD:
        # Abort all backends within a ncclGroupStart|End semantic.
        # This ensures that different NCCL communicators' abort calls won't
        # deadlock each other.
        # For details, please see: https://github.com/pytorch/pytorch/issues/119797
        if is_nccl_available() and isinstance(backend, ProcessGroupNCCL):
            backend._group_start()
        for pg_to_abort in sorted(
            _world.pg_names, key=lambda x: _world.pg_names[x], reverse=True
        ):
            pg_to_abort.abort()
        if is_nccl_available() and isinstance(backend, ProcessGroupNCCL):
            backend._group_end()

        _update_default_pg(None)
        _world.pg_map.clear()
        _world.pg_names.clear()
        _world.pg_group_ranks.clear()
        _world.pg_backend_config.clear()
        _world.pg_to_tag.clear()
        _world.tags_to_pg.clear()
        _world.pg_coalesce_state.clear()
        _unregister_all_process_groups()

        # when process group doesn't have an explicit name (only WORLD (default)
        # process group can have an explicit name), we use global _world.group_count
        # to generate the name. We need to reset the counter on destruction to
        # allow consistent value to be generated when we re-create process
        # groups after some trainers recover from failure
        #
        # We only reset this when WORLD is being destroyed because if this
        # process group is in good state, we aren't dealing with failures.
        _world.group_count = 0
    else:
        pg.abort()
        del _world.pg_map[pg]
        del _world.pg_names[pg]
        del _world.pg_group_ranks[pg]
        del _world.pg_backend_config[pg]
        if pg in _world.pg_coalesce_state.keys():
            warnings.warn(
                "Some coalesced collectives haven't been launched when "
                "ProcessGroup is aborted. They will be cleaned."
            )
            del _world.pg_coalesce_state[pg]

        tag = _world.pg_to_tag.get(pg)
        del _world.pg_to_tag[pg]
        if tag is not None:
            try:
                _world.tags_to_pg[tag].remove(pg)
                if tag.startswith("ptd:"):
                    _world.tags_to_pg[""].remove(pg)
            except Exception:
                pass
        _unregister_process_group(pg.group_name)


def get_rank(group: Optional[ProcessGroup] = None) -> int:
    """
    Return the rank of the current process in the provided ``group``, default otherwise.

    Rank is a unique identifier assigned to each process within a distributed
    process group. They are always consecutive integers ranging from 0 to
    ``world_size``.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    Returns:
        The rank of the process group
        -1, if not part of the group

    """
    if _rank_not_in_group(group):
        return -1

    default_pg = _get_default_group()
    if group is None or group is GroupMember.WORLD:
        return default_pg.rank()

    return get_group_rank(group, default_pg.rank())


def get_world_size(group: Optional[ProcessGroup] = None) -> int:
    """
    Return the number of processes in the current process group.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    Returns:
        The world size of the process group
        -1, if not part of the group

    """
    if _rank_not_in_group(group):
        return -1

    return _get_group_size(group)


def isend(
    tensor: torch.Tensor,
    dst: Optional[int] = None,
    group: Optional[ProcessGroup] = None,
    tag: int = 0,
    group_dst: Optional[int] = None,
) -> Optional[Work]:
    """
    Send a tensor asynchronously.

    .. warning::
        Modifying ``tensor`` before the request completes causes undefined
        behavior.

    .. warning::
        ``tag`` is not supported with the NCCL backend.

    Unlike send, which is blocking, isend allows src == dst rank, i.e. send to self.

    Args:
        tensor (Tensor): Tensor to send.
        dst (int): Destination rank on global process group (regardless of ``group`` argument)
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        tag (int, optional): Tag to match send with remote recv
        group_dst (int, optional): Destination rank on ``group``.  Invalid to specify both ``dst`` and ``group_dst``

    Returns:
        A distributed request object.
        None, if not part of the group

    """
    group = _group_or_default_group(group)
    group_dst = _canonicalize_group_rank(group, dst, group_dst)
    _check_single_tensor(tensor, "tensor")
    if _rank_not_in_group(group):
        _warn_not_in_group("isend")
        return None

    if tensor.is_complex():
        tensor = torch.view_as_real(tensor)

    return group.send([tensor], group_dst, tag)


def irecv(
    tensor: torch.Tensor,
    src: Optional[int] = None,
    group: Optional[ProcessGroup] = None,
    tag: int = 0,
    group_src: Optional[int] = None,
) -> Optional[Work]:
    """
    Receives a tensor asynchronously.

    .. warning::
        ``tag`` is not supported with the NCCL backend.

    Unlike recv, which is blocking, irecv allows src == dst rank, i.e. recv from self.

    Args:
        tensor (Tensor): Tensor to fill with received data.
        src (int, optional): Source rank on global process group (regardless of ``group`` argument).
            Will receive from any process if unspecified.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        tag (int, optional): Tag to match recv with remote send
        group_src (int, optional): Destination rank on ``group``.  Invalid to specify both ``src`` and ``group_src``.

    Returns:
        A distributed request object.
        None, if not part of the group

    """
    _check_single_tensor(tensor, "tensor")
    if _rank_not_in_group(group):
        _warn_not_in_group("irecv")
        return None

    if tensor.is_complex():
        tensor = torch.view_as_real(tensor)

    group = _group_or_default_group(group)
    if src is None and group_src is None:
        return group.recv_anysource([tensor], tag)
    else:
        group_src = _canonicalize_group_rank(group, src, group_src)
        return group.recv([tensor], group_src, tag)


@_exception_logger
def send(
    tensor: torch.Tensor,
    dst: Optional[int] = None,
    group: Optional[ProcessGroup] = None,
    tag: int = 0,
    group_dst: Optional[int] = None,
) -> None:
    """
    Send a tensor synchronously.

    .. warning::
        ``tag`` is not supported with the NCCL backend.

    Args:
        tensor (Tensor): Tensor to send.
        dst (int): Destination rank on global process group (regardless of ``group`` argument).
            Destination rank should not be the same as the rank of the current process.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        tag (int, optional): Tag to match send with remote recv
        group_dst (int, optional): Destination rank on ``group``.  Invalid to specify both ``dst`` and ``group_dst``.

    """
    group = _group_or_default_group(group)
    group_dst = _canonicalize_group_rank(group, dst, group_dst)
    _check_not_self_rank(group, group_dst, "destination")
    work = isend(tensor, group=group, tag=tag, group_dst=group_dst)
    if work is not None:
        work.wait()


@_exception_logger
def recv(
    tensor: torch.Tensor,
    src: Optional[int] = None,
    group: Optional[ProcessGroup] = None,
    tag: int = 0,
    group_src: Optional[int] = None,
) -> int:
    """
    Receives a tensor synchronously.

    .. warning::
        ``tag`` is not supported with the NCCL backend.

    Args:
        tensor (Tensor): Tensor to fill with received data.
        src (int, optional): Source rank on global process group (regardless of ``group`` argument).
            Will receive from any process if unspecified.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        tag (int, optional): Tag to match recv with remote send
        group_src (int, optional): Destination rank on ``group``.  Invalid to specify both ``src`` and ``group_src``.
    Returns:
        Sender rank
        -1, if not part of the group

    """
    work = irecv(tensor, src=src, group=group, tag=tag, group_src=group_src)
    if work is None:
        return -1
    work.wait()
    if src is None:
        if group_src is None:
            group_src = work._source_rank()
        group = _group_or_default_group(group)
        _check_not_self_rank(group, group_src, "source")
        src = get_global_rank(group, group_src)
    return src


class _IllegalWork(Work):
    def __getattribute__(self, name):
        if name in [
            "is_success",
            "exception",
            "wait",
            "source_rank",
            "_source_rank",
            "result",
            "synchronize",
        ]:
            raise ValueError(f"Illegal to call {name} on IllegalWork object")


class _CoalescingManager:
    def __init__(self) -> None:
        self.works: list[Work] = []

    def append(self, work: Optional[Work] = None):
        if work:
            self.works.append(work)

    def wait(self):
        for work in self.works:
            work.wait()


@contextlib.contextmanager
def _coalescing_manager(
    group: Optional[ProcessGroup] = None,
    device: Optional[torch.device] = None,
    async_ops: bool = False,
):
    """
    Context manager used to coalesce collectives or P2P operations when possible.

    Args:
        group (`ProcessGroup`, optional): The process group to work on. If None,
            the default process group will be used.
        device (`torch.device`, optional): Default is None, set to a device if
            there isn't a `**_coalesced` implementation by the backend.
        async_ops (`bool`, optional): whether the coalesced ops are async ops.

    Examples:
        >>> # xdoctest: +SKIP("no rank")
        >>> # Synchronous ops
        >>> with _coalescing_manager():
        >>>     for i in range(num_colls):
        >>>         dist.all_reduce(tensors[i])
        >>> # Asynchronous ops
        >>> with _coalescing_manager(async_ops=True) as cm:
        >>>     for i in range(num_colls):
        >>>         dist.all_reduce(tensors[i])
        >>> cm.wait()

    .. warning::
       :func:`_coalescing_manager` currently do not support coalescing
       all-reduces with different reduce operators, e.g.  `ReduceOp.SUM` mixed
       with `ReduceOp.PRODUCT`.
    """
    group = group or _get_default_group()
    op_list = _world.pg_coalesce_state.setdefault(group, [])
    if op_list:
        raise ValueError(
            "ProcessGroup has non-empty op list at the start of coalescing"
        )
    if device:
        group._start_coalescing(device)
    cm = _CoalescingManager()
    yield cm
    work = None
    op_list = _world.pg_coalesce_state.pop(group)
    if op_list:
        # Collectives supporting "Fast Path" coalescing are captured.
        # See implementation in corresponding collective APIs.
        # Currently supported:
        # - coalesced `all_reduce`
        # - coalesced `all_gather_into_tensor`
        # - coalesced `reduce_scatter_tensor`
        op0 = op_list[0].op
        if op0 == all_reduce:
            tensors = [op.tensor for op in op_list]
            all_reduce_opts = AllreduceCoalescedOptions()
            all_reduce_opts.reduceOp = not_none(op_list[0].redop)
            all_reduce_opts.asyncOp = async_ops
            work = group.allreduce_coalesced(tensors, all_reduce_opts)
        elif op0 == all_gather_into_tensor:
            inputs = []
            outputs = []
            for op in op_list:
                inputs.append(op.tensor)
                outputs.append(not_none(op.dst_tensor))
            all_gather_opts = AllgatherOptions()
            all_gather_opts.asyncOp = async_ops
            work = group.allgather_into_tensor_coalesced(outputs, inputs)
        elif op0 == reduce_scatter_tensor:
            inputs = []
            outputs = []
            for op in op_list:
                inputs.append(op.tensor)
                outputs.append(not_none(op.dst_tensor))
            reduce_opts = ReduceScatterOptions()
            reduce_opts.reduceOp = not_none(op_list[0].redop)
            reduce_opts.asyncOp = async_ops
            work = group.reduce_scatter_tensor_coalesced(outputs, inputs, reduce_opts)
        else:
            raise AssertionError(
                f"Coalescing manager does not support fast-path coalescing of {op0}, "
                f"yet {op0} is still recorded in op list. This is an internal error of c10d."
            )

    if device:
        # Old style of letting each coll inside the context manager to call into C++ counterpart via python binding
        work = group._end_coalescing(device)

    if async_ops:
        cm.append(work)
    elif (
        work is not None
    ):  # Backward compatible with backends that don't sync at CPP level
        work.wait()
    # Otherwise, the backend has sync'ed at CPP level


class _TimeEstimator:
    def __init__(self) -> None:
        self.estimated_time: Optional[float] = None


@contextlib.contextmanager
def _time_estimator(
    group: Optional[ProcessGroup] = None,
    device: Optional[torch.device] = None,
):
    """
    Context manager used to estimate time of collectives.
    Within the context manager, nothing is actually run and the backend just simulates
    the collective time only.

    Args:
        group (`ProcessGroup`, optional): The process group to work on. If None,
            the default process group will be used.
        device (`torch.device`, optional): Default is None, set to a device if
            there isn't a `**_coalesced` implementation by the backend.

    Examples:
        >>> # xdoctest: +SKIP("no rank")
        >>> # Synchronous ops
        >>> with _time_estimator() as cm:
        >>>     for i in range(num_colls):
        >>>         dist.all_reduce(tensors[i])
        >>> # estimate time is stored in cm.estimated_time

    .. warning::
       :func:`_time_estimator` currently only support NCCL backend but it can
       easily be extended to other backends.

       Also a NCCL communicator needs to be created because only with a real communicator can we do accurate estimation.
       The communicator internally has knowledge about the links it runs on
       (e.g. intra-node or inter-node, whether the links are NVLink or PCI-e or IB).
    """
    # TODO: We need to also support torch inductor for the time estimator.
    group = group or _get_default_group()
    device = device or _get_pg_default_device(group)
    backend = group._get_backend(device)
    if not backend.supports_time_estimate:
        raise NotImplementedError(
            f"collective time estimator is not supported in the current version of backend {backend}"
        )
    backend._start_time_estimate()  # type: ignore[attr-defined]
    cm = _TimeEstimator()
    yield cm
    cm.estimated_time = backend._end_time_estimate()  # type: ignore[attr-defined]


def batch_isend_irecv(p2p_op_list: list[P2POp]) -> list[Work]:
    """
    Send or Receive a batch of tensors asynchronously and return a list of requests.

    Process each of the operations in ``p2p_op_list`` and return the corresponding
    requests. NCCL, Gloo, and UCC backend are currently supported.

    Args:
        p2p_op_list: A list of point-to-point operations(type of each operator is
            ``torch.distributed.P2POp``). The order of the isend/irecv in the list
            matters and it needs to match with corresponding isend/irecv on the
            remote end.

    Returns:
        A list of distributed request objects returned by calling the corresponding
        op in the op_list.

    Examples:
        >>> # xdoctest: +SKIP("no rank")
        >>> send_tensor = torch.arange(2, dtype=torch.float32) + 2 * rank
        >>> recv_tensor = torch.randn(2, dtype=torch.float32)
        >>> send_op = dist.P2POp(dist.isend, send_tensor, (rank + 1) % world_size)
        >>> recv_op = dist.P2POp(
        ...     dist.irecv, recv_tensor, (rank - 1 + world_size) % world_size
        ... )
        >>> reqs = batch_isend_irecv([send_op, recv_op])
        >>> for req in reqs:
        >>>     req.wait()
        >>> recv_tensor
        tensor([2, 3])     # Rank 0
        tensor([0, 1])     # Rank 1

    .. note:: Note that when this API is used with the NCCL PG backend, users must set
        the current GPU device with `torch.cuda.set_device`, otherwise it will
        lead to unexpected hang issues.

        In addition, if this API is the first collective call in the ``group``
        passed to ``dist.P2POp``, all ranks of the ``group`` must participate in
        this API call; otherwise, the behavior is undefined. If this API call is
        not the first collective call in the ``group``, batched P2P operations
        involving only a subset of ranks of the ``group`` are allowed.
    """
    _check_p2p_op_list(p2p_op_list)
    group = p2p_op_list[0].group
    if group is None:
        group = _get_default_group()
    device = p2p_op_list[0].tensor.device

    def peer_kwarg(op: P2POp) -> dict[str, int]:
        key = "group_dst" if op.op == isend else "group_src"
        return {key: op.group_peer}

    if type(group) == ProcessGroup and group._get_backend(device).supports_coalescing:
        # NCCL style coalescing
        with _coalescing_manager(group, device, async_ops=True) as cm:
            for p2p_op in p2p_op_list:
                p2p_op.op(
                    p2p_op.tensor,
                    group=p2p_op.group,
                    tag=p2p_op.tag,
                    **peer_kwarg(p2p_op),
                )

        return cm.works
    else:
        # backend not support coalescing
        reqs = []
        for p2p_op in p2p_op_list:
            work = p2p_op.op(
                p2p_op.tensor,
                group=p2p_op.group,
                tag=p2p_op.tag,
                **peer_kwarg(p2p_op),
            )
            if work:
                reqs.append(work)
        return reqs


@_exception_logger
def broadcast(
    tensor: torch.Tensor,
    src: Optional[int] = None,
    group: Optional[ProcessGroup] = None,
    async_op: bool = False,
    group_src: Optional[int] = None,
):
    """
    Broadcasts the tensor to the whole group.

    ``tensor`` must have the same number of elements in all processes
    participating in the collective.

    Args:
        tensor (Tensor): Data to be sent if ``src`` is the rank of current
            process, and tensor to be used to save received data otherwise.
        src (int): Source rank on global process group (regardless of ``group`` argument).
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op
        group_src (int): Source rank on ``group``.  Must specify one of ``group_src``
            and ``src`` but not both.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    group = _group_or_default_group(group)
    group_src = _canonicalize_group_rank(group, src, group_src, return_global=False)
    _check_single_tensor(tensor, "tensor")
    if _rank_not_in_group(group):
        _warn_not_in_group("broadcast")
        return

    opts = BroadcastOptions()
    opts.rootRank = group_src
    opts.rootTensor = 0
    opts.asyncOp = async_op
    if tensor.is_complex():
        tensor = torch.view_as_real(tensor)
    work = group.broadcast([tensor], opts)
    if async_op:
        return work
    elif (
        work is not None
    ):  # Backward compatible with backends that don't sync at CPP level
        work.wait()
    # Otherwise, the backend has sync'ed at CPP level


@_exception_logger
def all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False):
    """
    Reduces the tensor data across all machines in a way that all get the final result.

    After the call ``tensor`` is going to be bitwise identical in all processes.

    Complex tensors are supported.

    Args:
        tensor (Tensor): Input and output of the collective. The function
            operates in-place.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    Examples:
        >>> # xdoctest: +SKIP("no rank")
        >>> # All tensors below are of torch.int64 type.
        >>> # We have 2 process groups, 2 ranks.
        >>> device = torch.device(f"cuda:{rank}")
        >>> tensor = torch.arange(2, dtype=torch.int64, device=device) + 1 + 2 * rank
        >>> tensor
        tensor([1, 2], device='cuda:0') # Rank 0
        tensor([3, 4], device='cuda:1') # Rank 1
        >>> dist.all_reduce(tensor, op=ReduceOp.SUM)
        >>> tensor
        tensor([4, 6], device='cuda:0') # Rank 0
        tensor([4, 6], device='cuda:1') # Rank 1

        >>> # All tensors below are of torch.cfloat type.
        >>> # We have 2 process groups, 2 ranks.
        >>> tensor = torch.tensor(
        ...     [1 + 1j, 2 + 2j], dtype=torch.cfloat, device=device
        ... ) + 2 * rank * (1 + 1j)
        >>> tensor
        tensor([1.+1.j, 2.+2.j], device='cuda:0') # Rank 0
        tensor([3.+3.j, 4.+4.j], device='cuda:1') # Rank 1
        >>> dist.all_reduce(tensor, op=ReduceOp.SUM)
        >>> tensor
        tensor([4.+4.j, 6.+6.j], device='cuda:0') # Rank 0
        tensor([4.+4.j, 6.+6.j], device='cuda:1') # Rank 1

    """
    # Dynamo has built-in logic to map legacy distributed ops to functional collectives.
    # Let's redirect to a torch function mode that can mimic this logic outside Dynamo
    # (e.g., non-strict export implements such a torch function mode).
    relevant_args = (tensor,)
    if has_torch_function(relevant_args):
        return handle_torch_function(
            all_reduce,
            relevant_args,
            tensor,
            op=op,
            group=group,
            async_op=async_op,
        )

    _check_single_tensor(tensor, "tensor")
    if _rank_not_in_group(group):
        _warn_not_in_group("all_reduce")
        return

    if tensor.is_complex():
        if not supports_complex(op):
            raise ValueError(f"all_reduce does not support {op} on complex tensors")
        tensor = torch.view_as_real(tensor)

    opts = AllreduceOptions()
    opts.reduceOp = op
    opts.asyncOp = async_op
    if group is None:
        group = _get_default_group()

    if group in _world.pg_coalesce_state.keys():
        # We are in coalescing context, do not issue single operation, just append a collective representation
        coll = _CollOp(all_reduce, tensor, None, op, None)
        _world.pg_coalesce_state[group].append(coll)
        if async_op:
            return _IllegalWork()
        else:
            return None

    work = group.allreduce([tensor], opts)

    if async_op:
        return work
    elif (
        work is not None
    ):  # Backward compatible with backends that don't sync at CPP level
        work.wait()
    # Otherwise, the backend has sync'ed at CPP level


@_exception_logger
@deprecated(
    "`torch.distributed.all_reduce_coalesced` will be deprecated. If you must "
    "use it, please revisit our documentation later at "
    "https://pytorch.org/docs/main/distributed.html#collective-functions",
    category=FutureWarning,
)
def all_reduce_coalesced(tensors, op=ReduceOp.SUM, group=None, async_op=False):
    """
    WARNING: at this time individual shape checking is not implemented across nodes.

    For example, if the rank 0 node passes [torch.rand(4), torch.rand(2)] and the
    rank 1 node passes [torch.rand(2), torch.rand(2), torch.rand(2)], the allreduce
    operation will proceed without complaint and return erroneous outputs. This lack
    of shape checking results in significant performance improvements but users of this
    function should take extra care to ensure that each node passes in tensors whose
    shapes match across nodes.

    Reduces each tensor in tensors (residing on the same device) across all machines
    in such a way that all get the final result.

    After the call each tensor in tensors is going to bitwise identical
    in all processes.

    Complex tensors are supported.

    Args:
        tensors (Union[List[Tensor], Tensor]): Input and output of the collective.
            The function operates in-place.
        op (Optional[ReduceOp]): One of the values from
            ``torch.distributed.ReduceOp`` enum. Specifies an operation used for
            element-wise reductions.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (Optional[bool]): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    """
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    _check_tensor_list(tensors, "tensor")
    _ensure_all_tensors_same_dtype(tensors)
    if _rank_not_in_group(group):
        _warn_not_in_group("all_reduce_coalesced")
        return

    if any(t.is_complex() for t in tensors) and not supports_complex(op):
        raise ValueError(f"all_reduce does not support {op} on complex tensors")

    tensors = [t if not t.is_complex() else torch.view_as_real(t) for t in tensors]

    opts = AllreduceCoalescedOptions()
    opts.reduceOp = op
    opts.asyncOp = async_op
    group = group or _get_default_group()
    work = group.allreduce_coalesced(tensors, opts)

    if async_op:
        return work.get_future()
    elif (
        work is not None
    ):  # Backward compatible with backends that don't sync at CPP level
        work.wait()
    # Otherwise, the backend has sync'ed at CPP level


@_exception_logger
def reduce(
    tensor: torch.Tensor,
    dst: Optional[int] = None,
    op=ReduceOp.SUM,
    group: Optional[ProcessGroup] = None,
    async_op: bool = False,
    group_dst: Optional[int] = None,
):
    """
    Reduces the tensor data across all machines.

    Only the process with rank ``dst`` is going to receive the final result.

    Args:
        tensor (Tensor): Input and output of the collective. The function
            operates in-place.
        dst (int): Destination rank on global process group (regardless of ``group`` argument)
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op
        group_dst (int): Destination rank on ``group``.  Must specify one of ``group_dst``
            and ``dst`` but not both.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    group = _group_or_default_group(group)
    group_dst = _canonicalize_group_rank(group, dst, group_dst, return_global=False)
    _check_single_tensor(tensor, "tensor")
    if _rank_not_in_group(group):
        _warn_not_in_group("reduce")
        return

    opts = ReduceOptions()
    opts.reduceOp = op
    opts.rootRank = group_dst
    opts.asyncOp = async_op
    work = group.reduce([tensor], opts)
    if async_op:
        return work
    elif (
        work is not None
    ):  # Backward compatible with backends that don't sync at CPP level
        work.wait()
    # Otherwise, the backend has sync'ed at CPP level


def _object_to_tensor(obj, device, group):
    with _WaitCounter("pytorch.wait_counter.c10d._object_to_tensor").guard():
        f = io.BytesIO()
        _pickler(f).dump(obj)
        byte_storage = torch.ByteStorage._from_buffer(f.getvalue())  # type: ignore[attr-defined]
        # Do not replace `torch.ByteTensor` or `torch.LongTensor` with torch.tensor and specifying dtype.
        # Otherwise, it will cause 100X slowdown.
        # See: https://github.com/pytorch/pytorch/issues/65696
        byte_tensor = torch.ByteTensor(byte_storage).to(device)
        if get_debug_level() == DebugLevel.DETAIL and is_nccl_available():
            backend = get_backend(group)
            if backend == Backend.NCCL:
                hash = torch._C._distributed_c10d._hash_tensors([byte_tensor])
                logger.warning(
                    "_object_to_tensor size: %s hash value: %s",
                    byte_tensor.numel(),
                    hash,
                )
        local_size = torch.LongTensor([byte_tensor.numel()]).to(device)
        return byte_tensor, local_size


def _tensor_to_object(tensor, tensor_size, group):
    with _WaitCounter("pytorch.wait_counter.c10d._tensor_to_object").guard():
        if get_debug_level() == DebugLevel.DETAIL and is_nccl_available():
            backend = get_backend(group)
            if backend == Backend.NCCL:
                hash = torch._C._distributed_c10d._hash_tensors([tensor])
                logger.warning(
                    "_tensor_to_object size: %s hash value: %s", tensor.numel(), hash
                )
        tensor = tensor.cpu()
        buf = tensor.numpy().tobytes()[:tensor_size]
        return _unpickler(io.BytesIO(buf)).load()


@_exception_logger
def all_gather_object(object_list, obj, group=None):
    """
    Gathers picklable objects from the whole group into a list.

    Similar to :func:`all_gather`, but Python objects can be passed in.
    Note that the object must be picklable in order to be gathered.

    Args:
        object_list (list[Any]): Output list. It should be correctly sized as the
            size of the group for this collective and will contain the output.
        obj (Any): Pickable Python object to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.

    Returns:
        None. If the calling rank is part of this group, the output of the
        collective will be populated into the input ``object_list``. If the
        calling rank is not part of the group, the passed in ``object_list`` will
        be unmodified.

    .. note:: Note that this API differs slightly from the :func:`all_gather`
        collective since it does not provide an ``async_op`` handle and thus
        will be a blocking call.

    .. note:: For NCCL-based processed groups, internal tensor representations
        of objects must be moved to the GPU device before communication takes
        place. In this case, the device used is given by
        ``torch.cuda.current_device()`` and it is the user's responsibility to
        ensure that this is set so that each rank has an individual GPU, via
        ``torch.cuda.set_device()``.

    .. warning::
        Object collectives have a number of serious performance and scalability
        limitations.  See :ref:`object_collectives` for details.

    .. warning::
        :func:`all_gather_object` uses ``pickle`` module implicitly, which is
        known to be insecure. It is possible to construct malicious pickle data
        which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    .. warning::
        Calling :func:`all_gather_object` with GPU tensors is not well supported
        and inefficient as it incurs GPU -> CPU transfer since tensors would be
        pickled. Please consider using :func:`all_gather` instead.

    Example::
        >>> # xdoctest: +SKIP("need process group init")
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> # Assumes world_size of 3.
        >>> gather_objects = ["foo", 12, {1: 2}] # any picklable object
        >>> output = [None for _ in gather_objects]
        >>> dist.all_gather_object(output, gather_objects[dist.get_rank()])
        >>> output
        ['foo', 12, {1: 2}]
    """
    if _rank_not_in_group(group):
        _warn_not_in_group("all_gather_object")
        return

    current_device = _get_object_coll_device(group)
    input_tensor, local_size = _object_to_tensor(obj, current_device, group)

    # Gather all local sizes. This is so that we can find the max size, and index
    # until the correct size when deserializing the tensors.
    group_size = get_world_size(group=group)
    object_sizes_tensor = torch.zeros(
        group_size, dtype=torch.long, device=current_device
    )
    object_size_list = [
        object_sizes_tensor[i].unsqueeze(dim=0) for i in range(group_size)
    ]
    # Allgather tensor sizes
    all_gather(object_size_list, local_size, group=group)
    max_object_size = int(max(object_size_list).item())  # type: ignore[type-var]
    # Resize tensor to max size across all ranks.
    input_tensor.resize_(max_object_size)
    coalesced_output_tensor = torch.empty(
        max_object_size * group_size, dtype=torch.uint8, device=current_device
    )
    # Output tensors are nonoverlapping views of coalesced_output_tensor
    output_tensors = [
        coalesced_output_tensor[max_object_size * i : max_object_size * (i + 1)]
        for i in range(group_size)
    ]
    all_gather(output_tensors, input_tensor, group=group)
    # Deserialize outputs back to object.
    for i, tensor in enumerate(output_tensors):
        tensor = tensor.type(torch.uint8)
        tensor_size = object_size_list[i]
        object_list[i] = _tensor_to_object(tensor, tensor_size, group)


@_exception_logger
def gather_object(
    obj: Any,
    object_gather_list: Optional[list[Any]] = None,
    dst: Optional[int] = None,
    group: Optional[ProcessGroup] = None,
    group_dst: Optional[int] = None,
):
    """
    Gathers picklable objects from the whole group in a single process.

    Similar to :func:`gather`, but Python objects can be passed in. Note that the
    object must be picklable in order to be gathered.

    Args:
        obj (Any): Input object. Must be picklable.
        object_gather_list (list[Any]): Output list. On the ``dst`` rank, it
            should be correctly sized as the size of the group for this
            collective and will contain the output. Must be ``None`` on non-dst
            ranks. (default is ``None``)
        dst (int, optional): Destination rank on global process group (regardless of ``group`` argument).
            (If both ``dst`` and ``group_dst`` are None, default is global rank 0)
        group: (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.
        group_dst (int, optional): Destination rank on ``group``.  Invalid to specify both ``dst`` and ``group_dst``

    Returns:
        None. On the ``dst`` rank, ``object_gather_list`` will contain the
        output of the collective.

    .. note:: Note that this API differs slightly from the gather collective
        since it does not provide an async_op handle and thus will be a blocking
        call.

    .. note:: For NCCL-based processed groups, internal tensor representations
        of objects must be moved to the GPU device before communication takes
        place. In this case, the device used is given by
        ``torch.cuda.current_device()`` and it is the user's responsibility to
        ensure that this is set so that each rank has an individual GPU, via
        ``torch.cuda.set_device()``.

    .. warning::
        Object collectives have a number of serious performance and scalability
        limitations.  See :ref:`object_collectives` for details.

    .. warning::
        :func:`gather_object` uses ``pickle`` module implicitly, which is
        known to be insecure. It is possible to construct malicious pickle data
        which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    .. warning::
        Calling :func:`gather_object` with GPU tensors is not well supported
        and inefficient as it incurs GPU -> CPU transfer since tensors would be
        pickled. Please consider using :func:`gather` instead.

    Example::
        >>> # xdoctest: +SKIP("need process group init")
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> # Assumes world_size of 3.
        >>> gather_objects = ["foo", 12, {1: 2}] # any picklable object
        >>> output = [None for _ in gather_objects]
        >>> dist.gather_object(
        ...     gather_objects[dist.get_rank()],
        ...     output if dist.get_rank() == 0 else None,
        ...     dst=0
        ... )
        >>> # On rank 0
        >>> output
        ['foo', 12, {1: 2}]
    """
    group = _group_or_default_group(group)
    if dst is None and group_dst is None:
        dst = 0
    group_dst = _canonicalize_group_rank(group, dst, group_dst, return_global=False)
    if _rank_not_in_group(group):
        _warn_not_in_group("gather_object")
        return

    # Ensure object_gather_list is specified appropriately.
    my_group_rank = group.rank()
    _validate_output_list_for_rank(my_group_rank, group_dst, object_gather_list)
    current_device = _get_object_coll_device(group)
    input_tensor, local_size = _object_to_tensor(obj, current_device, group)

    # Gather all local sizes. This is so that we can find the max size, and index
    # until the correct size when deserializing the tensors.
    group_size = get_world_size(group=group)
    object_sizes_tensor = torch.zeros(
        group_size, dtype=torch.long, device=current_device
    )
    object_size_list = [
        object_sizes_tensor[i].unsqueeze(dim=0) for i in range(group_size)
    ]
    # Allgather tensor sizes. An all-gather is needed here despite this being a
    # gather, since each rank needs to broadcast a tensor of the same (maximal)
    # size.
    all_gather(object_size_list, local_size, group=group)
    max_object_size = int(max(object_size_list).item())  # type: ignore[type-var]
    # Resize tensor to max size across all ranks.
    input_tensor.resize_(max_object_size)
    # Avoid populating output tensors if the result won't be gathered on this rank.
    if my_group_rank == group_dst:
        coalesced_output_tensor = torch.empty(
            max_object_size * group_size, dtype=torch.uint8, device=current_device
        )
        # Output tensors are nonoverlapping views of coalesced_output_tensor
        output_tensors = [
            coalesced_output_tensor[max_object_size * i : max_object_size * (i + 1)]
            for i in range(group_size)
        ]
    # All ranks call gather with equal-sized tensors.
    gather(
        input_tensor,
        gather_list=output_tensors if my_group_rank == group_dst else None,  # type: ignore[possibly-undefined]
        group_dst=group_dst,
        group=group,
    )
    if my_group_rank != group_dst:
        return

    assert object_gather_list is not None, "Must provide object_gather_list on dst rank"
    for i, tensor in enumerate(output_tensors):
        tensor = tensor.type(torch.uint8)
        tensor_size = object_size_list[i]
        object_gather_list[i] = _tensor_to_object(tensor, tensor_size, group)


@_exception_logger
def send_object_list(
    object_list: list[Any],
    dst: Optional[int] = None,
    group: Optional[ProcessGroup] = None,
    device: Optional[torch.device] = None,
    group_dst: Optional[int] = None,
):
    """
    Sends picklable objects in ``object_list`` synchronously.

    Similar to :func:`send`, but Python objects can be passed in.
    Note that all objects in ``object_list`` must be picklable in order to be
    sent.

    Args:
        object_list (List[Any]): List of input objects to sent.
            Each object must be picklable. Receiver must provide lists of equal sizes.
        dst (int): Destination rank to send ``object_list`` to.
            Destination rank is based on global process group (regardless of ``group`` argument)
        group: (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.
        device (``torch.device``, optional): If not None, the objects are
            serialized and converted to tensors which are moved to the
            ``device`` before sending. Default is ``None``.
        group_dst (int, optional): Destination rank on ``group``.
            Must specify one of ``dst`` and ``group_dst`` but not both
    Returns:
        ``None``.

    .. note:: For NCCL-based process groups, internal tensor representations
        of objects must be moved to the GPU device before communication takes
        place. In this case, the device used is given by
        ``torch.cuda.current_device()`` and it is the user's responsibility to
        ensure that this is set so that each rank has an individual GPU, via
        ``torch.cuda.set_device()``.

    .. warning::
        Object collectives have a number of serious performance and scalability
        limitations.  See :ref:`object_collectives` for details.

    .. warning::
        :func:`send_object_list` uses ``pickle`` module implicitly, which
        is known to be insecure. It is possible to construct malicious pickle
        data which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    .. warning::
        Calling :func:`send_object_list` with GPU tensors is not well supported
        and inefficient as it incurs GPU -> CPU transfer since tensors would be
        pickled. Please consider using :func:`send` instead.

    Example::
        >>> # xdoctest: +SKIP("need process group init")
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> # Assumes backend is not NCCL
        >>> device = torch.device("cpu")
        >>> if dist.get_rank() == 0:
        >>>     # Assumes world_size of 2.
        >>>     objects = ["foo", 12, {1: 2}] # any picklable object
        >>>     dist.send_object_list(objects, dst=1, device=device)
        >>> else:
        >>>     objects = [None, None, None]
        >>>     dist.recv_object_list(objects, src=0, device=device)
        >>> objects
        ['foo', 12, {1: 2}]
    """
    group = _group_or_default_group(group)
    group_dst = _canonicalize_group_rank(group, dst, group_dst)
    _check_not_self_rank(group, group_dst, "destination")

    if _rank_not_in_group(group):
        _warn_not_in_group("send_object_list")
        return

    # Current device selection.
    # To preserve backwards compatibility, ``device`` is default to ``None``
    # in which case we run current logic of device selection, i.e.
    # ``current_device`` is CUDA if backend is NCCL otherwise CPU device. In the
    # case it is not ``None`` we move the size and object tensors to be
    # sent to this device.
    current_device = device or _get_object_coll_device(group)
    # Serialize object_list elements to tensors on src rank.
    tensor_list, size_list = zip(
        *[_object_to_tensor(obj, current_device, group) for obj in object_list]
    )
    object_sizes_tensor = torch.cat(size_list)

    # Send object sizes
    send(object_sizes_tensor, group_dst=group_dst, group=group)

    # Concatenate and send serialized object tensors
    # Note: torch.cat will do an extra memory copy to the current device, if the tensor_list
    # has only one element, we can skip the copy.
    if len(tensor_list) == 1:  # type: ignore[possibly-undefined]
        object_tensor = tensor_list[0]
    else:
        object_tensor = torch.cat(tensor_list)

    send(object_tensor, group_dst=group_dst, group=group)


@_exception_logger
def recv_object_list(
    object_list: list[Any],
    src: Optional[int] = None,
    group: Optional[ProcessGroup] = None,
    device: Optional[torch.device] = None,
    group_src: Optional[int] = None,
):
    """
    Receives picklable objects in ``object_list`` synchronously.

    Similar to :func:`recv`, but can receive Python objects.

    Args:
        object_list (List[Any]): List of objects to receive into.
            Must provide a list of sizes equal to the size of the list being sent.
        src (int, optional): Source rank from which to recv ``object_list``.
            Source rank is based on global process group (regardless of ``group`` argument)
            Will receive from any rank if set to None. Default is ``None``.
        group: (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.
        device (``torch.device``, optional): If not None, receives on this device.
            Default is ``None``.
        group_src (int, optional): Destination rank on ``group``.  Invalid to specify both ``src`` and ``group_src``.

    Returns:
        Sender rank. -1 if rank is not part of the group. If rank is part of the group,
        ``object_list`` will contain the sent objects from ``src`` rank.

    .. note:: For NCCL-based process groups, internal tensor representations
        of objects must be moved to the GPU device before communication takes
        place. In this case, the device used is given by
        ``torch.cuda.current_device()`` and it is the user's responsibility to
        ensure that this is set so that each rank has an individual GPU, via
        ``torch.cuda.set_device()``.

    .. warning::
        Object collectives have a number of serious performance and scalability
        limitations.  See :ref:`object_collectives` for details.

    .. warning::
        :func:`recv_object_list` uses ``pickle`` module implicitly, which
        is known to be insecure. It is possible to construct malicious pickle
        data which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    .. warning::
        Calling :func:`recv_object_list` with GPU tensors is not well supported
        and inefficient as it incurs GPU -> CPU transfer since tensors would be
        pickled. Please consider using :func:`recv` instead.

    Example::
        >>> # xdoctest: +SKIP("need process group init")
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> # Assumes backend is not NCCL
        >>> device = torch.device("cpu")
        >>> if dist.get_rank() == 0:
        >>>     # Assumes world_size of 2.
        >>>     objects = ["foo", 12, {1: 2}] # any picklable object
        >>>     dist.send_object_list(objects, dst=1, device=device)
        >>> else:
        >>>     objects = [None, None, None]
        >>>     dist.recv_object_list(objects, src=0, device=device)
        >>> objects
        ['foo', 12, {1: 2}]
    """
    if _rank_not_in_group(group):
        _warn_not_in_group("recv_object_list")
        return -1

    # Current device selection.
    # To preserve backwards compatibility, ``device`` is default to ``None``
    # in which case we run current logic of device selection, i.e.
    # ``current_device`` is CUDA if backend is NCCL otherwise CPU device. In the
    # case it is not ``None`` we move the size and object tensors to be
    # received to this device.
    current_device = device or _get_object_coll_device(group)
    object_sizes_tensor = torch.empty(
        len(object_list), dtype=torch.long, device=current_device
    )

    # Receive object sizes
    rank_sizes = recv(object_sizes_tensor, src=src, group=group, group_src=group_src)

    # Tensor to receive serialized objects into.
    object_tensor = torch.empty(  # type: ignore[call-overload]
        torch.sum(object_sizes_tensor).item(),  # type: ignore[arg-type]
        dtype=torch.uint8,
        device=current_device,
    )

    rank_objects = recv(object_tensor, src=src, group=group, group_src=group_src)
    assert rank_sizes == rank_objects, (
        "Mismatch in return ranks for object sizes and objects."
    )
    # Deserialize objects using their stored sizes.
    offset = 0
    for i, obj_size in enumerate(object_sizes_tensor):
        obj_view = object_tensor[offset : offset + obj_size]
        obj_view = obj_view.type(torch.uint8)
        offset += obj_size
        object_list[i] = _tensor_to_object(obj_view, obj_size, group)
    return rank_objects


@_exception_logger
def broadcast_object_list(
    object_list: list[Any],
    src: Optional[int] = None,
    group: Optional[ProcessGroup] = None,
    device: Optional[torch.device] = None,
    group_src: Optional[int] = None,
):
    """
    Broadcasts picklable objects in ``object_list`` to the whole group.

    Similar to :func:`broadcast`, but Python objects can be passed in.
    Note that all objects in ``object_list`` must be picklable in order to be
    broadcasted.

    Args:
        object_list (List[Any]): List of input objects to broadcast.
            Each object must be picklable. Only objects on the ``src`` rank will
            be broadcast, but each rank must provide lists of equal sizes.
        src (int): Source rank from which to broadcast ``object_list``.
            Source rank is based on global process group (regardless of ``group`` argument)
        group: (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.
        device (``torch.device``, optional): If not None, the objects are
            serialized and converted to tensors which are moved to the
            ``device`` before broadcasting. Default is ``None``.
        group_src (int): Source rank on ``group``.  Must not specify one of ``group_src``
            and ``src`` but not both.

    Returns:
        ``None``. If rank is part of the group, ``object_list`` will contain the
        broadcasted objects from ``src`` rank.

    .. note:: For NCCL-based process groups, internal tensor representations
        of objects must be moved to the GPU device before communication takes
        place. In this case, the device used is given by
        ``torch.cuda.current_device()`` and it is the user's responsibility to
        ensure that this is set so that each rank has an individual GPU, via
        ``torch.cuda.set_device()``.

    .. note:: Note that this API differs slightly from the :func:`broadcast`
        collective since it does not provide an ``async_op`` handle and thus
        will be a blocking call.

    .. warning::
        Object collectives have a number of serious performance and scalability
        limitations.  See :ref:`object_collectives` for details.

    .. warning::
        :func:`broadcast_object_list` uses ``pickle`` module implicitly, which
        is known to be insecure. It is possible to construct malicious pickle
        data which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    .. warning::
        Calling :func:`broadcast_object_list` with GPU tensors is not well supported
        and inefficient as it incurs GPU -> CPU transfer since tensors would be
        pickled. Please consider using :func:`broadcast` instead.

    Example::
        >>> # xdoctest: +SKIP("need process group init")
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> if dist.get_rank() == 0:
        >>>     # Assumes world_size of 3.
        >>>     objects = ["foo", 12, {1: 2}] # any picklable object
        >>> else:
        >>>     objects = [None, None, None]
        >>> # Assumes backend is not NCCL
        >>> device = torch.device("cpu")
        >>> dist.broadcast_object_list(objects, src=0, device=device)
        >>> objects
        ['foo', 12, {1: 2}]
    """
    group = _group_or_default_group(group)
    if src is None and group_src is None:
        src = 0
    group_src = _canonicalize_group_rank(group, src, group_src, return_global=False)
    if _rank_not_in_group(group):
        _warn_not_in_group("broadcast_object_list")
        return

    # Current device selection.
    # To preserve backwards compatibility, ``device`` is default to ``None``
    # in which case we run current logic of device selection, i.e.
    # ``current_device`` is CUDA if backend is NCCL otherwise CPU device. In the
    # case it is not ``None`` we move the size and object tensors to be
    # broadcasted to this device.
    current_device = device or _get_object_coll_device(group)
    my_group_rank = group.rank()
    # Serialize object_list elements to tensors on src rank.
    if my_group_rank == group_src:
        tensor_list, size_list = zip(
            *[_object_to_tensor(obj, current_device, group) for obj in object_list]
        )
        object_sizes_tensor = torch.cat(size_list)
    else:
        object_sizes_tensor = torch.empty(
            len(object_list), dtype=torch.long, device=current_device
        )

    # Broadcast object sizes
    broadcast(object_sizes_tensor, group_src=group_src, group=group)

    # Concatenate and broadcast serialized object tensors
    # Note: torch.cat will do an extra memory copy to the current device, if the tensor_list
    # has only one element, we can skip the copy.
    if my_group_rank == group_src:
        if len(tensor_list) == 1:  # type: ignore[possibly-undefined]
            object_tensor = tensor_list[0]
        else:
            object_tensor = torch.cat(tensor_list)
    else:
        object_tensor = torch.empty(  # type: ignore[call-overload]
            torch.sum(object_sizes_tensor).item(),  # type: ignore[arg-type]
            dtype=torch.uint8,
            device=current_device,
        )

    broadcast(object_tensor, group_src=group_src, group=group)
    # Deserialize objects using their stored sizes.
    offset = 0
    if my_group_rank != group_src:
        for i, obj_size in enumerate(object_sizes_tensor):
            obj_view = object_tensor[offset : offset + obj_size]
            obj_view = obj_view.type(torch.uint8)
            offset += obj_size
            object_list[i] = _tensor_to_object(obj_view, obj_size, group)


@_exception_logger
def scatter_object_list(
    scatter_object_output_list: list[Any],
    scatter_object_input_list: Optional[list[Any]] = None,
    src: Optional[int] = None,
    group: Optional[ProcessGroup] = None,
    group_src: Optional[int] = None,
):
    """
    Scatters picklable objects in ``scatter_object_input_list`` to the whole group.

    Similar to :func:`scatter`, but Python objects can be passed in. On
    each rank, the scattered object will be stored as the first element of
    ``scatter_object_output_list``. Note that all objects in
    ``scatter_object_input_list`` must be picklable in order to be scattered.

    Args:
        scatter_object_output_list (List[Any]): Non-empty list whose first
            element will store the object scattered to this rank.
        scatter_object_input_list (List[Any], optional): List of input objects to scatter.
            Each object must be picklable. Only objects on the ``src`` rank will
            be scattered, and the argument can be ``None`` for non-src ranks.
        src (int): Source rank from which to scatter ``scatter_object_input_list``.
            Source rank is based on global process group (regardless of ``group`` argument).
            (If both ``src`` and ``group_src`` are None, default is global rank 0)
        group: (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.
        group_src (int, optional): Source rank on ``group``.  Invalid to specify both ``src`` and ``group_src``

    Returns:
        ``None``. If rank is part of the group, ``scatter_object_output_list``
        will have its first element set to the scattered object for this rank.

    .. note:: Note that this API differs slightly from the scatter collective
        since it does not provide an ``async_op`` handle and thus will be a
        blocking call.

    .. warning::
        Object collectives have a number of serious performance and scalability
        limitations.  See :ref:`object_collectives` for details.

    .. warning::
        :func:`scatter_object_list` uses ``pickle`` module implicitly, which
        is known to be insecure. It is possible to construct malicious pickle
        data which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    .. warning::
        Calling :func:`scatter_object_list` with GPU tensors is not well supported
        and inefficient as it incurs GPU -> CPU transfer since tensors would be
        pickled. Please consider using :func:`scatter` instead.

    Example::
        >>> # xdoctest: +SKIP("need process group init")
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> if dist.get_rank() == 0:
        >>>     # Assumes world_size of 3.
        >>>     objects = ["foo", 12, {1: 2}] # any picklable object
        >>> else:
        >>>     # Can be any list on non-src ranks, elements are not used.
        >>>     objects = [None, None, None]
        >>> output_list = [None]
        >>> dist.scatter_object_list(output_list, objects, src=0)
        >>> # Rank i gets objects[i]. For example, on rank 2:
        >>> output_list
        [{1: 2}]
    """
    group = _group_or_default_group(group)
    if src is None and group_src is None:
        src = 0
    group_src = _canonicalize_group_rank(group, src, group_src, return_global=False)
    if _rank_not_in_group(group):
        _warn_not_in_group("scatter_object_list")
        return

    if (
        not isinstance(scatter_object_output_list, list)
        or len(scatter_object_output_list) < 1
    ):
        raise ValueError(
            "Expected argument scatter_object_output_list to be a list of size at least 1."
        )

    my_group_rank = group.rank()
    pg_device = _get_object_coll_device(group)
    if my_group_rank == group_src:
        if scatter_object_input_list is None:
            raise ValueError(
                "source rank must provide non-None scatter_object_input_list"
            )
        tensor_list, tensor_sizes = zip(
            *[
                _object_to_tensor(obj, pg_device, group)
                for obj in scatter_object_input_list
            ]
        )
        tensor_list, tensor_sizes = list(tensor_list), list(tensor_sizes)

        # Src rank broadcasts the maximum tensor size. This is because all ranks are
        # expected to call into scatter() with equal-sized tensors.
        max_tensor_size = max(tensor_sizes)  # type: ignore[possibly-undefined]
        for tensor in tensor_list:  # type: ignore[possibly-undefined]
            tensor.resize_(max_tensor_size)
    else:
        max_tensor_size = torch.tensor([0], dtype=torch.long, device=pg_device)
    broadcast(max_tensor_size, group_src=group_src, group=group)

    # Scatter actual serialized objects
    output_tensor = torch.empty(
        max_tensor_size.item(), dtype=torch.uint8, device=pg_device
    )
    scatter(
        output_tensor,
        scatter_list=None if my_group_rank != group_src else tensor_list,  # type: ignore[possibly-undefined]
        group_src=group_src,
        group=group,
    )

    # Scatter per-object sizes to trim tensors when deserializing back to object
    obj_tensor_size = torch.tensor([0], dtype=torch.long, device=pg_device)
    scatter(
        obj_tensor_size,
        scatter_list=None if my_group_rank != group_src else tensor_sizes,  # type: ignore[possibly-undefined]
        group_src=group_src,
        group=group,
    )

    # Deserialize back to object
    scatter_object_output_list[0] = _tensor_to_object(
        output_tensor, obj_tensor_size, group
    )


@_exception_logger
def all_gather(tensor_list, tensor, group=None, async_op=False):
    """
    Gathers tensors from the whole group in a list.

    Complex and uneven sized tensors are supported.

    Args:
        tensor_list (list[Tensor]): Output list. It should contain
            correctly-sized tensors to be used for output of the collective.
            Uneven sized tensors are supported.
        tensor (Tensor): Tensor to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    Examples:
        >>> # xdoctest: +SKIP("need process group init")
        >>> # All tensors below are of torch.int64 dtype.
        >>> # We have 2 process groups, 2 ranks.
        >>> device = torch.device(f"cuda:{rank}")
        >>> tensor_list = [
        ...     torch.zeros(2, dtype=torch.int64, device=device) for _ in range(2)
        ... ]
        >>> tensor_list
        [tensor([0, 0], device='cuda:0'), tensor([0, 0], device='cuda:0')] # Rank 0
        [tensor([0, 0], device='cuda:1'), tensor([0, 0], device='cuda:1')] # Rank 1
        >>> tensor = torch.arange(2, dtype=torch.int64, device=device) + 1 + 2 * rank
        >>> tensor
        tensor([1, 2], device='cuda:0') # Rank 0
        tensor([3, 4], device='cuda:1') # Rank 1
        >>> dist.all_gather(tensor_list, tensor)
        >>> tensor_list
        [tensor([1, 2], device='cuda:0'), tensor([3, 4], device='cuda:0')] # Rank 0
        [tensor([1, 2], device='cuda:1'), tensor([3, 4], device='cuda:1')] # Rank 1

        >>> # All tensors below are of torch.cfloat dtype.
        >>> # We have 2 process groups, 2 ranks.
        >>> tensor_list = [
        ...     torch.zeros(2, dtype=torch.cfloat, device=device) for _ in range(2)
        ... ]
        >>> tensor_list
        [tensor([0.+0.j, 0.+0.j], device='cuda:0'), tensor([0.+0.j, 0.+0.j], device='cuda:0')] # Rank 0
        [tensor([0.+0.j, 0.+0.j], device='cuda:1'), tensor([0.+0.j, 0.+0.j], device='cuda:1')] # Rank 1
        >>> tensor = torch.tensor(
        ...     [1 + 1j, 2 + 2j], dtype=torch.cfloat, device=device
        ... ) + 2 * rank * (1 + 1j)
        >>> tensor
        tensor([1.+1.j, 2.+2.j], device='cuda:0') # Rank 0
        tensor([3.+3.j, 4.+4.j], device='cuda:1') # Rank 1
        >>> dist.all_gather(tensor_list, tensor)
        >>> tensor_list
        [tensor([1.+1.j, 2.+2.j], device='cuda:0'), tensor([3.+3.j, 4.+4.j], device='cuda:0')] # Rank 0
        [tensor([1.+1.j, 2.+2.j], device='cuda:1'), tensor([3.+3.j, 4.+4.j], device='cuda:1')] # Rank 1

    """
    # Dynamo has built-in logic to map legacy distributed ops to functional collectives.
    # Let's redirect to a torch function mode that can mimic this logic outside Dynamo
    # (e.g., non-strict export implements such a torch function mode).
    relevant_args = (tensor,)
    if has_torch_function(relevant_args):
        return handle_torch_function(
            all_gather,
            relevant_args,
            tensor_list,
            tensor,
            group=group,
            async_op=async_op,
        )

    _check_tensor_list(tensor_list, "tensor_list")
    _check_single_tensor(tensor, "tensor")
    _ensure_all_tensors_same_dtype(tensor_list, tensor)
    if _rank_not_in_group(group):
        _warn_not_in_group("all_gather")
        return

    tensor_list = [
        t if not t.is_complex() else torch.view_as_real(t) for t in tensor_list
    ]
    tensor = tensor if not tensor.is_complex() else torch.view_as_real(tensor)

    group = group or _get_default_group()
    opts = AllgatherOptions()
    opts.asyncOp = async_op
    work = group.allgather([tensor_list], [tensor], opts)

    if async_op:
        return work
    elif (
        work is not None
    ):  # Backward compatible with backends that don't sync at CPP level
        work.wait()
    # Otherwise, the backend has sync'ed at CPP level


@_exception_logger
def all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False):
    """
    Gather tensors from all ranks and put them in a single output tensor.

    This function requires all tensors to be the same size on each process.

    Args:
        output_tensor (Tensor): Output tensor to accommodate tensor elements
            from all ranks. It must be correctly sized to have one of the
            following forms:
            (i) a concatenation of all the input tensors along the primary
            dimension; for definition of "concatenation", see ``torch.cat()``;
            (ii) a stack of all the input tensors along the primary dimension;
            for definition of "stack", see ``torch.stack()``.
            Examples below may better explain the supported output forms.
        input_tensor (Tensor): Tensor to be gathered from current rank.
            Different from the ``all_gather`` API, the input tensors in this
            API must have the same size across all ranks.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    Examples:
        >>> # xdoctest: +SKIP("need process group init")
        >>> # All tensors below are of torch.int64 dtype and on CUDA devices.
        >>> # We have two ranks.
        >>> device = torch.device(f"cuda:{rank}")
        >>> tensor_in = torch.arange(2, dtype=torch.int64, device=device) + 1 + 2 * rank
        >>> tensor_in
        tensor([1, 2], device='cuda:0') # Rank 0
        tensor([3, 4], device='cuda:1') # Rank 1
        >>> # Output in concatenation form
        >>> tensor_out = torch.zeros(world_size * 2, dtype=torch.int64, device=device)
        >>> dist.all_gather_into_tensor(tensor_out, tensor_in)
        >>> tensor_out
        tensor([1, 2, 3, 4], device='cuda:0') # Rank 0
        tensor([1, 2, 3, 4], device='cuda:1') # Rank 1
        >>> # Output in stack form
        >>> tensor_out2 = torch.zeros(world_size, 2, dtype=torch.int64, device=device)
        >>> dist.all_gather_into_tensor(tensor_out2, tensor_in)
        >>> tensor_out2
        tensor([[1, 2],
                [3, 4]], device='cuda:0') # Rank 0
        tensor([[1, 2],
                [3, 4]], device='cuda:1') # Rank 1
    """
    # Dynamo has built-in logic to map legacy distributed ops to functional collectives.
    # Let's redirect to a torch function mode that can mimic this logic outside Dynamo
    # (e.g., non-strict export implements such a torch function mode).
    relevant_args = (input_tensor,)
    if has_torch_function(relevant_args):
        return handle_torch_function(
            all_gather_into_tensor,
            relevant_args,
            output_tensor,
            input_tensor,
            group=group,
            async_op=async_op,
        )

    _check_single_tensor(input_tensor, "input_tensor")
    _check_single_tensor(output_tensor, "output_tensor")
    if _rank_not_in_group(group):
        _warn_not_in_group("all_gather_into_tensor")
        return

    output_tensor = (
        output_tensor
        if not output_tensor.is_complex()
        else torch.view_as_real(output_tensor)
    )
    input_tensor = (
        input_tensor
        if not input_tensor.is_complex()
        else torch.view_as_real(input_tensor)
    )

    opts = AllgatherOptions()
    opts.asyncOp = async_op

    group = group or _get_default_group()

    if group in _world.pg_coalesce_state.keys():
        # We are in coalescing context, do not issue single operation, just append a collective representation
        coll = _CollOp(all_gather_into_tensor, input_tensor, output_tensor)
        _world.pg_coalesce_state[group].append(coll)
        if async_op:
            return _IllegalWork()
        else:
            return None

    work = group._allgather_base(output_tensor, input_tensor, opts)

    if async_op:
        return work
    elif (
        work is not None
    ):  # Backward compatible with backends that don't sync at CPP level
        work.wait()
    # Otherwise, the backend has sync'ed at CPP level


@_exception_logger
@deprecated(
    "`torch.distributed._all_gather_base` is a private function and will be deprecated. "
    "Please use `torch.distributed.all_gather_into_tensor` instead.",
    category=FutureWarning,
)
def _all_gather_base(output_tensor, input_tensor, group=None, async_op=False):
    """
    Single tensor all gather. Gathers a single tensor from all ranks, and puts them in a single output tensor.

    Args:
        output_tensor (Tensor): Output tensor. It should contain
            correctly-sized tensors to be used for output of the collective.
        input_tensor (Tensor): Tensor to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    .. warning::
        `_all_gather_base` is a private function. Users should use
        `all_gather_into_tensor` instead.

    """
    return all_gather_into_tensor(output_tensor, input_tensor, group, async_op)


@_exception_logger
@deprecated(
    "`torch.distributed.all_gather_coalesced` will be deprecated. If you must use it, "
    "please revisit our documentation later at "
    "https://pytorch.org/docs/main/distributed.html#collective-functions",
    category=FutureWarning,
)
def all_gather_coalesced(
    output_tensor_lists, input_tensor_list, group=None, async_op=False
):
    """
    Gathers input tensors from the whole group in a list in a coalesced manner.

    Complex tensors are supported.

    Args:
        output_tensor_lists (list[list[Tensor]]): Output list. It should contain
            correctly-sized tensors to be used for output of the collective.
        input_tensor_list (list[Tensor]): Tensors to be broadcast from
            current process. At least one tensor has to be non empty.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    Example:
        we have 2 process groups, 2 ranks.
        rank 0 passes:
            input_tensor_list = [[[1, 1], [1, 1]], [2], [3, 3]]
            output_tensor_lists =
               [[[[-1, -1], [-1, -1]], [-1], [-1, -1]],
                [[[-1, -1], [-1, -1]], [-1], [-1, -1]]]
        rank 1 passes:
            input_tensor_list = [[[3, 3], [3, 3]], [5], [1, 1]]
            output_tensor_lists =
               [[[[-1, -1], [-1, -1]], [-1], [-1, -1]],
                [[[-1, -1], [-1, -1]], [-1], [-1, -1]]]
        both rank 0 and 1 get:
            output_tensor_lists =
               [[[1, 1], [1, 1]], [2], [3, 3]],
                [[3, 3], [3, 3]], [5], [1, 1]]].

    WARNING: at this time individual shape checking is not implemented across nodes.
    For example, if the rank 0 node passes [torch.rand(4), torch.rand(2)] and the
    rank 1 node passes [torch.rand(2), torch.rand(2), torch.rand(2)], the
    all_gather_coalesced operation will proceed without complaint and return
    erroneous outputs. This lack of shape checking results in significant
    performance improvements but users of this function should take extra care
    to ensure that each node passes in tensors whose shapes match across nodes.
    """
    # We only check basic compatibility with C++ params here, C++ code will
    # do shape and type checking.
    if _rank_not_in_group(group):
        _warn_not_in_group("all_gather_coalesced")
        return
    _check_tensor_list(input_tensor_list, "input_tensor_list")
    _ensure_all_tensors_same_dtype(input_tensor_list)
    if not isinstance(output_tensor_lists, list):
        raise TypeError(
            "Invalid function argument: output_tensor_lists should be a list"
        )
    for output_tensor_list in output_tensor_lists:
        _check_tensor_list(output_tensor_list, "output_tensor_lists")
        _ensure_all_tensors_same_dtype(output_tensor_list)

    output_tensor_lists = [
        [t if not t.is_complex() else torch.view_as_real(t) for t in l]
        for l in output_tensor_lists
    ]
    input_tensor_list = [
        t if not t.is_complex() else torch.view_as_real(t) for t in input_tensor_list
    ]

    group = group or _get_default_group()
    opts = AllgatherOptions()
    opts.asyncOp = async_op
    work = group.allgather_coalesced(output_tensor_lists, input_tensor_list, opts)

    if async_op:
        return work.get_future()
    elif (
        work is not None
    ):  # Backward compatible with backends that don't sync at CPP level
        work.wait()
    # Otherwise, the backend has sync'ed at CPP level


def _validate_output_list_for_rank(my_rank, dst, gather_list):
    if dst == my_rank:
        if not gather_list:
            raise ValueError(
                "Argument ``gather_list`` must be specified on destination rank."
            )
    elif gather_list:
        raise ValueError(
            "Argument ``gather_list`` must NOT be specified on non-destination ranks."
        )


@_exception_logger
def gather(
    tensor: torch.Tensor,
    gather_list: Optional[list[torch.Tensor]] = None,
    dst: Optional[int] = None,
    group: Optional[ProcessGroup] = None,
    async_op: bool = False,
    group_dst: Optional[int] = None,
):
    """
    Gathers a list of tensors in a single process.

    This function requires all tensors to be the same size on each process.

    Args:
        tensor (Tensor): Input tensor.
        gather_list (list[Tensor], optional): List of appropriately,
            same-sized tensors to use for gathered data
            (default is None, must be specified on the destination rank)
        dst (int, optional): Destination rank on global process group (regardless of ``group`` argument).
            (If both ``dst`` and ``group_dst`` are None, default is global rank 0)
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op
        group_dst (int, optional): Destination rank on ``group``.  Invalid to specify both ``dst`` and ``group_dst``

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    .. note:: Note that all Tensors in gather_list must have the same size.

    Example::
        >>> # xdoctest: +SKIP("no rank")
        >>> # We have 2 process groups, 2 ranks.
        >>> tensor_size = 2
        >>> device = torch.device(f'cuda:{rank}')
        >>> tensor = torch.ones(tensor_size, device=device) + rank
        >>> if dist.get_rank() == 0:
        >>>     gather_list = [torch.zeros_like(tensor, device=device) for i in range(2)]
        >>> else:
        >>>     gather_list = None
        >>> dist.gather(tensor, gather_list, dst=0)
        >>> # Rank 0 gets gathered data.
        >>> gather_list
        [tensor([1., 1.], device='cuda:0'), tensor([2., 2.], device='cuda:0')] # Rank 0
        None                                                                   # Rank 1

    """
    _check_single_tensor(tensor, "tensor")

    # Parameter ``gather_list`` may be left unspecified on non-dst ranks.
    if gather_list:
        _check_tensor_list(gather_list, "gather_list")
    else:
        gather_list = []
    _ensure_all_tensors_same_dtype(tensor, gather_list)
    group = _group_or_default_group(group)
    if _rank_not_in_group(group):
        _warn_not_in_group("gather")
        return
    if dst is None and group_dst is None:
        dst = 0
    group_dst = _canonicalize_group_rank(group, dst, group_dst, return_global=False)
    my_group_rank = group.rank()
    _validate_output_list_for_rank(my_group_rank, group_dst, gather_list)
    output_tensors = [gather_list] if group_dst == my_group_rank else []
    input_tensors = [tensor]

    opts = GatherOptions()
    opts.rootRank = group_dst
    opts.asyncOp = async_op
    work = group.gather(output_tensors, input_tensors, opts)

    if async_op:
        return work
    elif (
        work is not None
    ):  # Backward compatible with backends that don't sync at CPP level
        work.wait()
    # Otherwise, the backend has sync'ed at CPP level


@_exception_logger
def scatter(
    tensor: torch.Tensor,
    scatter_list: Optional[list[torch.Tensor]] = None,
    src: Optional[int] = None,
    group: Optional[ProcessGroup] = None,
    async_op: bool = False,
    group_src: Optional[int] = None,
):
    """
    Scatters a list of tensors to all processes in a group.

    Each process will receive exactly one tensor and store its data in the
    ``tensor`` argument.

    Complex tensors are supported.

    Args:
        tensor (Tensor): Output tensor.
        scatter_list (list[Tensor]): List of tensors to scatter (default is
            None, must be specified on the source rank)
        src (int): Source rank on global process group (regardless of ``group`` argument).
            (If both ``src`` and ``group_src`` are None, default is global rank 0)
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op
        group_src (int, optional): Source rank on ``group``.  Invalid to specify both ``src`` and ``group_src``

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    .. note:: Note that all Tensors in scatter_list must have the same size.

    Example::
        >>> # xdoctest: +SKIP("need process group init")
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> tensor_size = 2
        >>> device = torch.device(f'cuda:{rank}')
        >>> output_tensor = torch.zeros(tensor_size, device=device)
        >>> if dist.get_rank() == 0:
        >>>     # Assumes world_size of 2.
        >>>     # Only tensors, all of which must be the same size.
        >>>     t_ones = torch.ones(tensor_size, device=device)
        >>>     t_fives = torch.ones(tensor_size, device=device) * 5
        >>>     scatter_list = [t_ones, t_fives]
        >>> else:
        >>>     scatter_list = None
        >>> dist.scatter(output_tensor, scatter_list, src=0)
        >>> # Rank i gets scatter_list[i].
        >>> output_tensor
        tensor([1., 1.], device='cuda:0') # Rank 0
        tensor([5., 5.], device='cuda:1') # Rank 1

    """
    _check_single_tensor(tensor, "tensor")
    # Parameter ``scatter_list`` may be left unspecified on non-src ranks.
    if scatter_list:
        _check_tensor_list(scatter_list, "scatter_list")
    else:
        scatter_list = []
    _ensure_all_tensors_same_dtype(tensor, scatter_list)
    group = _group_or_default_group(group)
    if src is None and group_src is None:
        src = 0
    group_src = _canonicalize_group_rank(group, src, group_src, return_global=False)
    if _rank_not_in_group(group):
        _warn_not_in_group("scatter")
        return
    scatter_list = [
        t if not t.is_complex() else torch.view_as_real(t) for t in scatter_list
    ]
    tensor = tensor if not tensor.is_complex() else torch.view_as_real(tensor)

    my_group_rank = group.rank()
    if group_src == my_group_rank:
        if not scatter_list:
            raise ValueError(
                "Argument ``scatter_list`` must be specified on source rank."
            )
        input_tensors = [scatter_list]
        output_tensors = [tensor]
    else:
        if scatter_list:
            raise ValueError(
                "Argument ``scatter_list`` must NOT be specified on non-source ranks."
            )
        input_tensors = []
        output_tensors = [tensor]

    opts = ScatterOptions()
    opts.rootRank = group_src
    opts.asyncOp = async_op
    work = group.scatter(output_tensors, input_tensors, opts)

    if async_op:
        return work
    elif (
        work is not None
    ):  # Backward compatible with backends that don't sync at CPP level
        work.wait()
    # Otherwise, the backend has sync'ed at CPP level


@_exception_logger
def reduce_scatter(output, input_list, op=ReduceOp.SUM, group=None, async_op=False):
    """
    Reduces, then scatters a list of tensors to all processes in a group.

    Args:
        output (Tensor): Output tensor.
        input_list (list[Tensor]): List of tensors to reduce and scatter.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    """
    _check_single_tensor(output, "output")
    _check_tensor_list(input_list, "input_list")
    _ensure_all_tensors_same_dtype(output, input_list)
    if _rank_not_in_group(group):
        _warn_not_in_group("reduce_scatter")
        return

    opts = ReduceScatterOptions()
    opts.reduceOp = op
    opts.asyncOp = async_op

    group = group or _get_default_group()
    work = group.reduce_scatter([output], [input_list], opts)

    if async_op:
        return work
    elif (
        work is not None
    ):  # Backward compatible with backends that don't sync at CPP level
        work.wait()
    # Otherwise, the backend has sync'ed at CPP level


@_exception_logger
def reduce_scatter_tensor(output, input, op=ReduceOp.SUM, group=None, async_op=False):
    """
    Reduces, then scatters a tensor to all ranks in a group.

    Args:
        output (Tensor): Output tensor. It should have the same size across all
            ranks.
        input (Tensor): Input tensor to be reduced and scattered. Its size
            should be output tensor size times the world size. The input tensor
            can have one of the following shapes:
            (i) a concatenation of the output tensors along the primary
            dimension, or
            (ii) a stack of the output tensors along the primary dimension.
            For definition of "concatenation", see ``torch.cat()``.
            For definition of "stack", see ``torch.stack()``.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    Examples:
        >>> # xdoctest: +SKIP("need process group init")
        >>> # All tensors below are of torch.int64 dtype and on CUDA devices.
        >>> # We have two ranks.
        >>> device = torch.device(f"cuda:{rank}")
        >>> tensor_out = torch.zeros(2, dtype=torch.int64, device=device)
        >>> # Input in concatenation form
        >>> tensor_in = torch.arange(world_size * 2, dtype=torch.int64, device=device)
        >>> tensor_in
        tensor([0, 1, 2, 3], device='cuda:0') # Rank 0
        tensor([0, 1, 2, 3], device='cuda:1') # Rank 1
        >>> dist.reduce_scatter_tensor(tensor_out, tensor_in)
        >>> tensor_out
        tensor([0, 2], device='cuda:0') # Rank 0
        tensor([4, 6], device='cuda:1') # Rank 1
        >>> # Input in stack form
        >>> tensor_in = torch.reshape(tensor_in, (world_size, 2))
        >>> tensor_in
        tensor([[0, 1],
                [2, 3]], device='cuda:0') # Rank 0
        tensor([[0, 1],
                [2, 3]], device='cuda:1') # Rank 1
        >>> dist.reduce_scatter_tensor(tensor_out, tensor_in)
        >>> tensor_out
        tensor([0, 2], device='cuda:0') # Rank 0
        tensor([4, 6], device='cuda:1') # Rank 1

    """
    # Dynamo has built-in logic to map legacy distributed ops to functional collectives.
    # Let's redirect to a torch function mode that can mimic this logic outside Dynamo
    # (e.g., non-strict export implements such a torch function mode).
    relevant_args = (input,)
    if has_torch_function(relevant_args):
        return handle_torch_function(
            reduce_scatter_tensor,
            relevant_args,
            output,
            input,
            op=op,
            group=group,
            async_op=async_op,
        )

    _check_single_tensor(output, "output")
    _check_single_tensor(input, "input")

    if _rank_not_in_group(group):
        _warn_not_in_group("reduce_scatter_tensor")
        return

    opts = ReduceScatterOptions()
    opts.reduceOp = op
    opts.asyncOp = async_op

    group = group or _get_default_group()

    # Check if we are in coalescing context
    # If we are, do not issue single operation, just append a collective representation
    if group in _world.pg_coalesce_state.keys():
        coll = _CollOp(reduce_scatter_tensor, input, output, op, None)
        _world.pg_coalesce_state[group].append(coll)
        if async_op:
            return _IllegalWork()
        else:
            return None

    work = group._reduce_scatter_base(output, input, opts)

    if async_op:
        return work
    elif (
        work is not None
    ):  # Backward compatible with backends that don't sync at CPP level
        work.wait()
    # Otherwise, the backend has sync'ed at CPP level


@deprecated(
    "`torch.distributed._reduce_scatter_base` is a private function and will be deprecated. "
    "Please use `torch.distributed.reduce_scatter_tensor` instead.",
    category=FutureWarning,
)
def _reduce_scatter_base(output, input, op=ReduceOp.SUM, group=None, async_op=False):
    """
    Reduces, then scatters a flattened tensor to all processes in a group.

    Args:
        output (Tensor): Output tensor.
        input (Tensor): Input tensor that is of size output tensor size times world size
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    .. warning::
        `_reduce_scatter_base` is a private function. Users should use
        `reduce_scatter_tensor` instead.

    """
    return reduce_scatter_tensor(output, input, op, group, async_op)


@_exception_logger
def all_to_all_single(
    output,
    input,
    output_split_sizes=None,
    input_split_sizes=None,
    group=None,
    async_op=False,
):
    """
    Split input tensor and then scatter the split list to all processes in a group.

    Later the received tensors are concatenated from all the processes in the group
    and returned as a single output tensor.

    Complex tensors are supported.

    Args:
        output (Tensor): Gathered concatenated output tensor.
        input (Tensor): Input tensor to scatter.
        output_split_sizes: (list[Int], optional): Output split sizes for dim 0
            if specified None or empty, dim 0 of ``output`` tensor must divide
            equally by ``world_size``.
        input_split_sizes: (list[Int], optional): Input split sizes for dim 0
            if specified None or empty, dim 0 of ``input`` tensor must divide
            equally by ``world_size``.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    .. warning::
        `all_to_all_single` is experimental and subject to change.

    Examples:
        >>> # xdoctest: +SKIP("Undefined rank")
        >>> input = torch.arange(4) + rank * 4
        >>> input
        tensor([0, 1, 2, 3])     # Rank 0
        tensor([4, 5, 6, 7])     # Rank 1
        tensor([8, 9, 10, 11])   # Rank 2
        tensor([12, 13, 14, 15]) # Rank 3
        >>> output = torch.empty([4], dtype=torch.int64)
        >>> dist.all_to_all_single(output, input)
        >>> output
        tensor([0, 4, 8, 12])    # Rank 0
        tensor([1, 5, 9, 13])    # Rank 1
        tensor([2, 6, 10, 14])   # Rank 2
        tensor([3, 7, 11, 15])   # Rank 3

        >>> # Essentially, it is similar to following operation:
        >>> scatter_list = list(input.chunk(world_size))
        >>> gather_list = list(output.chunk(world_size))
        >>> for i in range(world_size):
        >>>     dist.scatter(gather_list[i], scatter_list if i == rank else [], src = i)

        >>> # Another example with uneven split
        >>> input
        tensor([0, 1, 2, 3, 4, 5])                                       # Rank 0
        tensor([10, 11, 12, 13, 14, 15, 16, 17, 18])                     # Rank 1
        tensor([20, 21, 22, 23, 24])                                     # Rank 2
        tensor([30, 31, 32, 33, 34, 35, 36])                             # Rank 3
        >>> input_splits
        [2, 2, 1, 1]                                                     # Rank 0
        [3, 2, 2, 2]                                                     # Rank 1
        [2, 1, 1, 1]                                                     # Rank 2
        [2, 2, 2, 1]                                                     # Rank 3
        >>> output_splits
        [2, 3, 2, 2]                                                     # Rank 0
        [2, 2, 1, 2]                                                     # Rank 1
        [1, 2, 1, 2]                                                     # Rank 2
        [1, 2, 1, 1]                                                     # Rank 3
        >>> output = ...
        >>> dist.all_to_all_single(output, input, output_splits, input_splits)
        >>> output
        tensor([ 0,  1, 10, 11, 12, 20, 21, 30, 31])                     # Rank 0
        tensor([ 2,  3, 13, 14, 22, 32, 33])                             # Rank 1
        tensor([ 4, 15, 16, 23, 34, 35])                                 # Rank 2
        tensor([ 5, 17, 18, 24, 36])                                     # Rank 3


        >>> # Another example with tensors of torch.cfloat type.
        >>> input = torch.tensor(
        ...     [1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], dtype=torch.cfloat
        ... ) + 4 * rank * (1 + 1j)
        >>> input
        tensor([1+1j, 2+2j, 3+3j, 4+4j])                                # Rank 0
        tensor([5+5j, 6+6j, 7+7j, 8+8j])                                # Rank 1
        tensor([9+9j, 10+10j, 11+11j, 12+12j])                          # Rank 2
        tensor([13+13j, 14+14j, 15+15j, 16+16j])                        # Rank 3
        >>> output = torch.empty([4], dtype=torch.int64)
        >>> dist.all_to_all_single(output, input)
        >>> output
        tensor([1+1j, 5+5j, 9+9j, 13+13j])                              # Rank 0
        tensor([2+2j, 6+6j, 10+10j, 14+14j])                            # Rank 1
        tensor([3+3j, 7+7j, 11+11j, 15+15j])                            # Rank 2
        tensor([4+4j, 8+8j, 12+12j, 16+16j])                            # Rank 3
    """
    # Dynamo has built-in logic to map legacy distributed ops to functional collectives.
    # Let's redirect to a torch function mode that can mimic this logic outside Dynamo
    # (e.g., non-strict export implements such a torch function mode).
    relevant_args = (input,)
    if has_torch_function(relevant_args):
        return handle_torch_function(
            all_to_all_single,
            relevant_args,
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=async_op,
        )

    if _rank_not_in_group(group):
        _warn_not_in_group("all_to_all_single")
        return

    opts = AllToAllOptions()
    opts.asyncOp = async_op
    _check_single_tensor(output, "output")
    _check_single_tensor(input, "input")
    _ensure_all_tensors_same_dtype(output, input)

    if input.is_complex():
        input = torch.view_as_real(input)
    if output.is_complex():
        output = torch.view_as_real(output)

    output_split_sizes = [] if output_split_sizes is None else output_split_sizes
    input_split_sizes = [] if input_split_sizes is None else input_split_sizes

    group = group or _get_default_group()
    work = group.alltoall_base(
        output, input, output_split_sizes, input_split_sizes, opts
    )

    if async_op:
        return work
    elif (
        work is not None
    ):  # Backward compatible with backends that don't sync at CPP level
        work.wait()
    # Otherwise, the backend has sync'ed at CPP level


@_exception_logger
def all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False):
    """
    Scatters list of input tensors to all processes in a group and return gathered list of tensors in output list.

    Complex tensors are supported.

    Args:
        output_tensor_list (list[Tensor]): List of tensors to be gathered one
            per rank.
        input_tensor_list (list[Tensor]): List of tensors to scatter one per rank.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    .. warning::
        `all_to_all` is experimental and subject to change.

    Examples:
        >>> # xdoctest: +SKIP("Undefined rank")
        >>> input = torch.arange(4) + rank * 4
        >>> input = list(input.chunk(4))
        >>> input
        [tensor([0]), tensor([1]), tensor([2]), tensor([3])]     # Rank 0
        [tensor([4]), tensor([5]), tensor([6]), tensor([7])]     # Rank 1
        [tensor([8]), tensor([9]), tensor([10]), tensor([11])]   # Rank 2
        [tensor([12]), tensor([13]), tensor([14]), tensor([15])] # Rank 3
        >>> output = list(torch.empty([4], dtype=torch.int64).chunk(4))
        >>> dist.all_to_all(output, input)
        >>> output
        [tensor([0]), tensor([4]), tensor([8]), tensor([12])]    # Rank 0
        [tensor([1]), tensor([5]), tensor([9]), tensor([13])]    # Rank 1
        [tensor([2]), tensor([6]), tensor([10]), tensor([14])]   # Rank 2
        [tensor([3]), tensor([7]), tensor([11]), tensor([15])]   # Rank 3

        >>> # Essentially, it is similar to following operation:
        >>> scatter_list = input
        >>> gather_list = output
        >>> for i in range(world_size):
        >>>     dist.scatter(gather_list[i], scatter_list if i == rank else [], src=i)

        >>> input
        tensor([0, 1, 2, 3, 4, 5])                                       # Rank 0
        tensor([10, 11, 12, 13, 14, 15, 16, 17, 18])                     # Rank 1
        tensor([20, 21, 22, 23, 24])                                     # Rank 2
        tensor([30, 31, 32, 33, 34, 35, 36])                             # Rank 3
        >>> input_splits
        [2, 2, 1, 1]                                                     # Rank 0
        [3, 2, 2, 2]                                                     # Rank 1
        [2, 1, 1, 1]                                                     # Rank 2
        [2, 2, 2, 1]                                                     # Rank 3
        >>> output_splits
        [2, 3, 2, 2]                                                     # Rank 0
        [2, 2, 1, 2]                                                     # Rank 1
        [1, 2, 1, 2]                                                     # Rank 2
        [1, 2, 1, 1]                                                     # Rank 3
        >>> input = list(input.split(input_splits))
        >>> input
        [tensor([0, 1]), tensor([2, 3]), tensor([4]), tensor([5])]                   # Rank 0
        [tensor([10, 11, 12]), tensor([13, 14]), tensor([15, 16]), tensor([17, 18])] # Rank 1
        [tensor([20, 21]), tensor([22]), tensor([23]), tensor([24])]                 # Rank 2
        [tensor([30, 31]), tensor([32, 33]), tensor([34, 35]), tensor([36])]         # Rank 3
        >>> output = ...
        >>> dist.all_to_all(output, input)
        >>> output
        [tensor([0, 1]), tensor([10, 11, 12]), tensor([20, 21]), tensor([30, 31])]   # Rank 0
        [tensor([2, 3]), tensor([13, 14]), tensor([22]), tensor([32, 33])]           # Rank 1
        [tensor([4]), tensor([15, 16]), tensor([23]), tensor([34, 35])]              # Rank 2
        [tensor([5]), tensor([17, 18]), tensor([24]), tensor([36])]                  # Rank 3

        >>> # Another example with tensors of torch.cfloat type.
        >>> input = torch.tensor(
        ...     [1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], dtype=torch.cfloat
        ... ) + 4 * rank * (1 + 1j)
        >>> input = list(input.chunk(4))
        >>> input
        [tensor([1+1j]), tensor([2+2j]), tensor([3+3j]), tensor([4+4j])]            # Rank 0
        [tensor([5+5j]), tensor([6+6j]), tensor([7+7j]), tensor([8+8j])]            # Rank 1
        [tensor([9+9j]), tensor([10+10j]), tensor([11+11j]), tensor([12+12j])]      # Rank 2
        [tensor([13+13j]), tensor([14+14j]), tensor([15+15j]), tensor([16+16j])]    # Rank 3
        >>> output = list(torch.empty([4], dtype=torch.int64).chunk(4))
        >>> dist.all_to_all(output, input)
        >>> output
        [tensor([1+1j]), tensor([5+5j]), tensor([9+9j]), tensor([13+13j])]          # Rank 0
        [tensor([2+2j]), tensor([6+6j]), tensor([10+10j]), tensor([14+14j])]        # Rank 1
        [tensor([3+3j]), tensor([7+7j]), tensor([11+11j]), tensor([15+15j])]        # Rank 2
        [tensor([4+4j]), tensor([8+8j]), tensor([12+12j]), tensor([16+16j])]        # Rank 3

    """
    if _rank_not_in_group(group):
        _warn_not_in_group("all_to_all")
        return

    opts = AllToAllOptions()
    opts.asyncOp = async_op
    _check_tensor_list(output_tensor_list, "output_tensor_list")
    _check_tensor_list(input_tensor_list, "input_tensor_list")
    _ensure_all_tensors_same_dtype(output_tensor_list, input_tensor_list)

    input_tensor_list = [
        t if not t.is_complex() else torch.view_as_real(t) for t in input_tensor_list
    ]
    output_tensor_list = [
        t if not t.is_complex() else torch.view_as_real(t) for t in output_tensor_list
    ]

    group = group or _get_default_group()
    work = group.alltoall(output_tensor_list, input_tensor_list, opts)

    if async_op:
        return work
    elif (
        work is not None
    ):  # Backward compatible with backends that don't sync at CPP level
        work.wait()
    # Otherwise, the backend has sync'ed at CPP level


@_exception_logger
def barrier(
    group: Optional[ProcessGroup] = GroupMember.WORLD, async_op=False, device_ids=None
):
    """
    Synchronize all processes.

    This collective blocks processes until the whole group enters this function,
    if async_op is False, or if async work handle is called on wait().

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op
        device_ids ([int], optional): List of device/GPU ids. Only one id is expected.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    .. note:: `ProcessGroupNCCL` now blocks the cpu thread till the completion of the barrier collective.
    """
    group = group or _get_default_group()

    if _rank_not_in_group(group):
        _warn_not_in_group("barrier")
        return

    opts = BarrierOptions()
    opts.asyncOp = async_op
    # Detect the accelerator on the machine. If no accelerator is available, it
    # returns CPU.
    device = torch._C._get_accelerator()
    if isinstance(device_ids, list):
        opts.device_ids = device_ids
        # use only the first device id
        opts.device = torch.device(device.type, device_ids[0])
    elif getattr(group, "bound_device_id", None) is not None:
        # Use device id from `init_process_group(device_id=...)`
        opts.device = group.bound_device_id  # type: ignore[assignment]
    elif device.type == "cpu" or _get_object_coll_device(group) == "cpu":
        opts.device = torch.device("cpu")
    else:
        # Use the current device set by the user. If user did not set any, this
        # may use default device 0, causing issues like hang or all processes
        # creating context on device 0.
        opts.device = device
        warnings.warn(  # warn only once
            "No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user. "
        )

    work = group.barrier(opts=opts)

    if async_op:
        return work
    elif (
        work is not None
    ):  # Backward compatible with backends that don't sync at CPP level
        work.wait()
    # Otherwise, the backend has sync'ed at CPP level


def monitored_barrier(
    group: Optional[ProcessGroup] = GroupMember.WORLD,
    timeout=None,
    wait_all_ranks=False,
):
    """
    Synchronize processes similar to ``torch.distributed.barrier``, but consider a configurable timeout.

    It is able to report ranks that did not pass this barrier within the provided timeout.
    Specifically, for non-zero ranks, will block until a send/recv is processed from rank 0.
    Rank 0 will block until all send /recv from other ranks are processed, and will report
    failures for ranks that failed to respond in time. Note that if one rank does not reach the
    monitored_barrier (for example due to a hang), all other ranks would fail in monitored_barrier.

    This collective will block all processes/ranks in the group, until the
    whole group exits the function successfully, making it useful for debugging
    and synchronizing. However, it can have a performance impact and should only
    be used for debugging or scenarios that require full synchronization points
    on the host-side. For debugging purposes, this barrier can be inserted
    before the application's collective calls to check if any ranks are
    desynchronized.

    .. note:: Note that this collective is only supported with the GLOO backend.

    Args:
        group (ProcessGroup, optional): The process group to work on. If
            ``None``, the default process group will be used.
        timeout (datetime.timedelta, optional): Timeout for monitored_barrier.
            If ``None``, the default process group timeout will be used.
        wait_all_ranks (bool, optional): Whether to collect all failed ranks or
            not. By default, this is ``False`` and ``monitored_barrier`` on rank 0
            will throw on the first failed rank it encounters in order to fail
            fast. By setting ``wait_all_ranks=True`` ``monitored_barrier`` will
            collect all failed ranks and throw an error containing information
            about all failed ranks.

    Returns:
        ``None``.

    Example::
        >>> # xdoctest: +SKIP("need process group init")
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> if dist.get_rank() != 1:
        >>>     dist.monitored_barrier() # Raises exception indicating that
        >>> # rank 1 did not call into monitored_barrier.
        >>> # Example with wait_all_ranks=True
        >>> if dist.get_rank() == 0:
        >>>     dist.monitored_barrier(wait_all_ranks=True) # Raises exception
        >>> # indicating that ranks 1, 2, ... world_size - 1 did not call into
        >>> # monitored_barrier.
    """
    # Need to call rank not in group before using the group, otherwise
    # "Invalid process group" error is raised.
    if _rank_not_in_group(group):
        _warn_not_in_group("monitored_barrier")
        return

    if get_backend(group) != Backend.GLOO:
        raise ValueError("monitored_barrier is only implemented for GLOO backend.")

    if timeout is None:
        timeout = _get_default_timeout(get_backend(group))
    elif isinstance(timeout, float):
        # TODO(whc) apparently some existing test case for monitored_barrier passes in a timeout in float format?
        warnings.warn(
            "Please specify timeout arg as a timedelta. "
            f"Converting current value of {timeout} assuming it represents seconds",
        )
        timeout = timedelta(seconds=timeout)

    _check_valid_timeout(timeout)

    group_to_use = _get_default_group() if group is None else group
    return group_to_use.monitored_barrier(  # type:ignore[attr-defined]
        timeout, wait_all_ranks=wait_all_ranks
    )


def _create_process_group_wrapper(
    wrapped_pg: torch._C._distributed_c10d.Backend,
    store_prefix: str,
    store: Store,
    rank: int,
    world_size: int,
    timeout: timedelta = default_pg_timeout,
):
    assert _GLOO_AVAILABLE, "ProcessGroupWrapper unsupported without GLOO backend."

    # (whc) this appears to be just for the gloo backend? if so, `default_pg_timeout` is appropriate...

    # Create a separate prefix store for the helper process group.
    prefix = f"{PG_WRAPPER_STORE_PREFIX}:{store_prefix}"
    store = PrefixStore(prefix, store)
    helper_pg = ProcessGroupGloo(store, rank, world_size, timeout=timeout)
    # Wrap the underlying pg with ProcessGroupWrapper.
    wrapped_pg = _ProcessGroupWrapper(wrapped_pg, helper_pg)
    return wrapped_pg


# helper function for deterministically hashing a list of ranks to a unique
# string
def _hash_ranks_to_str(ranks: list[int]) -> str:
    rank_join: str = "_".join(map(str, ranks))
    # In case there is already a PG with the same rank composition
    unique_str = "_".join([rank_join, str(len(_world.pg_names))])
    return hashlib.sha1(bytes(unique_str, "utf-8"), usedforsecurity=False).hexdigest()


# Takes a list of ranks and computes an integer color
def _process_group_color(ranks: list[int]) -> int:
    # Convert list to tuple to make it hashable
    ranks = tuple(ranks)
    hash_value = hash(ranks)
    # Split color must be:
    # - a non-negative integer;
    # - a type compatible with C's int because we are pybinding to the latter.
    # Thus, we limit the hash value within c_int's max value.
    max_c_int = 2 ** (ctypes.sizeof(ctypes.c_int) * 8 - 1)
    color = abs(hash_value) % max_c_int
    return color


def _process_group_name(ranks, use_hashed_name):
    # Create name for a process group.
    global _world
    if use_hashed_name:
        pg_name = _hash_ranks_to_str(ranks)
    else:
        pg_name = str(_world.group_count)
        _world.group_count += 1
    # TODO: why is group count incremented only in the else path?
    return pg_name


def _get_backend_from_str(backend: Optional[str] = None) -> Backend:
    # Default to the same backend as the global process group
    #  if backend is not specified.
    if not backend:
        backend = get_backend(_get_default_group())
    return Backend(backend)


def _is_safe_to_split() -> bool:
    """
    Checks if it is safe to split the any process group in the world.
    This is only safe if the default pg has a bound device id, otherwise
    users must be aware that a pg is only splittable after the first collective is
    issued.
    """
    return False if _get_default_group().bound_device_id is None else True


@_time_logger
def split_group(
    parent_pg: Optional[ProcessGroup] = None,
    split_ranks: Optional[list] = None,
    timeout: Optional[timedelta] = None,
    pg_options: Optional[Any] = None,
    group_desc: Optional[str] = None,
) -> Optional[ProcessGroup]:
    """
    Create a new process group split from the given parent process group.

    warning:: This is an experimental API. Only the ``NCCL`` and custom plugin backends
    are supported. Other backends will raise an error.
    Users of this API must guarantee that all ranks in the parent group enter this API call,
    and the split of the sub groups is the same across all ranks in the parent group.

    Args:
        parent_pg (ProcessGroup, optional): The parent process group. If None,
            the default process group will be used. Users need to guarantee that
            the parent group is fully initialized (e.g, communicators are initialized)
        split_ranks (list[list[int]]): the split ranks, which is a list of list of ranks.
            Users need to make sure the validity of the split ranks such that one
            split (represented by one inner list of ints) does not overlap with any other split.
            Note that the ranks in each split is the group rank (instead of global rank)
            in the parent pg. For example, if the parent group has 4 ranks, and split_ranks can be
            [[0, 1], [2, 3]]. Note [[0,1]] is also a valid split, in which case ranks 2, 3 would
            return a non-group member.
        timeout (timedelta, optional): see `init_process_group` for details and default value.
        pg_options (ProcessGroupOptions, optional): Additional options need to be passed in during
            the construction of specific process groups. i.e.``is_high_priority_stream``
            can be specified so that process group can pick up high priority cuda streams.
        group_desc (str, optional): a string to describe the process group.

    Returns:
        ProcessGroup if the current rank is within one split/subgroup given by split_ranks,
        or None if the current rank is not part of any split_ranks`.

    """
    # check inputs
    if split_ranks is None:
        raise ValueError("split_ranks cannot be None")

    global _world
    default_pg = _get_default_group()
    device_id = default_pg.bound_device_id
    if not device_id:
        raise RuntimeError(
            "No device associated with the default pg, not safe to split any process groups"
        )
    _default_backend, default_store = _world.pg_map[default_pg]
    global_rank = default_pg.rank()
    global_world_size = default_pg.size()

    if not parent_pg:
        parent_pg = default_pg
    if parent_pg not in _world.pg_group_ranks:
        raise ValueError(f"Group {parent_pg} is not registered")

    parent_global_to_group_ranks = _world.pg_group_ranks[parent_pg]
    parent_group_to_global_ranks = {
        group_rank: global_rank
        for global_rank, group_rank in parent_global_to_group_ranks.items()
    }

    if global_rank not in parent_global_to_group_ranks:
        raise ValueError(
            f"Global rank {global_rank} is not part of the parent group {parent_pg}"
        )

    parent_group_rank = parent_global_to_group_ranks[global_rank]
    parent_backend = parent_pg._get_backend(torch.device("cuda"))

    # if the parent backend does not support splitting, raise error
    # currently this API only support NCCL backend
    if not parent_backend or not parent_backend.supports_splitting:
        raise RuntimeError(
            "No backend for the parent process group or its backend does not support splitting"
        )

    # set the group_desc before the color or no_cloor split
    group_desc = (
        f"{parent_pg.group_desc}:split:{parent_backend.comm_split_count()}"  # type: ignore[attr-defined]
        if group_desc is None
        else group_desc
    )

    parent_backend_str, _ = _world.pg_map[parent_pg]
    # same type of backend as the parent process group
    backend = Backend(parent_backend_str)
    backend_config = BackendConfig(backend)

    if pg_options is None:
        # default pg_options same as the parent process group
        pg_options = parent_backend.options

    # this timeout defaulting/validation is used for all the new_groups/new_subgroups variants,
    # which may just pass their timeout value (or None)
    if timeout is None:
        timeout = _get_default_timeout(backend)
    _check_valid_timeout(timeout)

    # find my group of ranks and my group local rank in split_ranks
    my_group = None
    group_rank = -1

    for split_group in split_ranks:
        if len(split_group) == 0:
            raise ValueError("the split group cannot be empty")
        if len(split_group) > global_world_size:
            raise ValueError(
                "the split group's size should be less or equal to the world_size set by init_process_group"
            )
        if len(split_group) != len(set(split_group)):
            raise ValueError("the split group cannot have duplicate ranks")
        split_group = sorted(split_group)
        if parent_group_rank in split_group:
            my_group = split_group
            group_rank = split_group.index(parent_group_rank)
            break
    # if my rank does not belong to any sub group,
    # no_color split should be called
    if my_group is None or group_rank == -1:
        parent_backend.perform_nocolor_split(device_id)  # type: ignore[attr-defined]
        return None

    group_name = _process_group_name(my_group, use_hashed_name=False)
    global_ranks_in_my_group = [parent_group_to_global_ranks[rank] for rank in my_group]

    prefix_store = PrefixStore(f"{group_name}/", default_store)
    # We register the backend after initializing and timeout is set in pg_options.
    pg: ProcessGroup = ProcessGroup(
        prefix_store,
        group_rank,
        len(my_group),
    )
    pg.bound_device_id = device_id  # type: ignore[union-attr]
    pg_options._timeout = timeout  # type: ignore[union-attr]
    pg_options.split_from = parent_backend  # type: ignore[union-attr]
    pg_options.split_color = _process_group_color(my_group)  # type: ignore[union-attr]
    pg_options.global_ranks_in_group = global_ranks_in_my_group  # type: ignore[union-attr]
    pg_options.group_name = group_name  # type: ignore[union-attr]

    if parent_backend_str == Backend.NCCL:
        backend_type = ProcessGroup.BackendType.NCCL
        if not isinstance(pg_options, ProcessGroupNCCL.Options):
            raise RuntimeError(
                "Expected pg_options argument to be of type ProcessGroupNCCL.Options"
            )
        backend_class = ProcessGroupNCCL(
            prefix_store, group_rank, len(my_group), pg_options
        )
    else:
        assert parent_backend_str.upper() in Backend._plugins, (
            f"Unknown c10d backend type {parent_backend_str.upper()}"
        )
        backend_plugin = Backend._plugins[parent_backend_str.upper()]
        creator_fn = backend_plugin.creator_fn
        extended_api = backend_plugin.extended_api
        backend_type = ProcessGroup.BackendType.CUSTOM
        if not extended_api:
            backend_class = creator_fn(prefix_store, group_rank, len(my_group), timeout)
        else:
            dist_backend_opts = _DistributedBackendOptions()
            dist_backend_opts.store = prefix_store
            dist_backend_opts.group_rank = group_rank
            dist_backend_opts.group_size = len(my_group)
            backend_class = creator_fn(dist_backend_opts, pg_options)

    pg._set_default_backend(backend_type)
    backend_class._set_sequence_number_for_group()

    pg._register_backend(torch.device("cuda"), backend_type, backend_class)

    # set group_name and group_desc to backend
    assert group_name is not None
    assert group_desc is not None
    pg._set_group_name(group_name)
    pg._set_group_desc(group_desc)

    # always eagerly initialize the backend in split_group
    eager_backend = pg._get_backend(device_id)
    eager_backend.eager_connect_single_device(device_id)

    # update global state
    _world.pg_map[pg] = (backend, prefix_store)
    _world.pg_names[pg] = group_name
    _register_process_group(group_name, pg)
    _world.pg_backend_config[pg] = str(backend_config)
    pg_tag = f"ptd:{group_name}"
    _world.tags_to_pg.setdefault(pg_tag, []).append(pg)
    _world.pg_to_tag[pg] = pg_tag

    # Create the global rank to group rank mapping
    _world.pg_group_ranks[pg] = {
        global_rank: group_rank
        for group_rank, global_rank in enumerate(global_ranks_in_my_group)
    }

    return pg


@_time_logger
def new_group(
    ranks=None,
    timeout=None,
    backend=None,
    pg_options=None,
    use_local_synchronization=False,
    group_desc=None,
    device_id: Optional[torch.device] = None,
):
    """
    Create a new distributed group.

    This function requires that all processes in the main group (i.e. all
    processes that are part of the distributed job) enter this function, even
    if they are not going to be members of the group. Additionally, groups
    should be created in the same order in all processes.

    .. warning::
        Safe concurrent usage:
        When using multiple process groups with the ``NCCL`` backend, the user
        must ensure a globally consistent execution order of collectives across
        ranks.

        If multiple threads within a process issue collectives, explicit
        synchronization is necessary to ensure consistent ordering.

        When using async variants of torch.distributed communication APIs,
        a work object is returned and the communication kernel is
        enqueued on a separate CUDA stream, allowing overlap of communication
        and computation. Once one or more async ops have been issued on one process
        group, they must be synchronized with other cuda streams by calling `work.wait()`
        before using another process group.

        See `Using multiple NCCL communicators concurrently
        <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#using-multiple-nccl-communicators-concurrently>`
        for more details.

    Args:
        ranks (list[int]): List of ranks of group members. If ``None``, will be
            set to all ranks. Default is ``None``.
        timeout (timedelta, optional): see `init_process_group` for details and default value.
        backend (str or Backend, optional): The backend to use. Depending on
            build-time configurations, valid values are ``gloo`` and ``nccl``.
            By default uses the same backend as the global group. This field
            should be given as a lowercase string (e.g., ``"gloo"``), which can
            also be accessed via :class:`Backend` attributes (e.g.,
            ``Backend.GLOO``). If ``None`` is passed in, the backend
            corresponding to the default process group will be used. Default is
            ``None``.
        pg_options (ProcessGroupOptions, optional): process group options
            specifying what additional options need to be passed in during
            the construction of specific process groups. i.e. for the ``nccl``
            backend, ``is_high_priority_stream`` can be specified so that
            process group can pick up high priority cuda streams. For other available options to config nccl,
            See https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#ncclconfig-tuse_local_synchronization
            (bool, optional): perform a group-local barrier at the end of the process group creation.
            This is different in that non-member ranks don't need to call into API and don't
            join the barrier.
        group_desc (str, optional): a string to describe the process group.
        device_id (torch.device, optional): a single, specific device
            to "bind" this process to,  The `new_group` call will try to initialize
            a communication backend immediately for the device if this field is given.

    Returns:
        A handle of distributed group that can be given to collective calls or
        GroupMember.NON_GROUP_MEMBER if the rank is not part of ``ranks``.

    N.B. use_local_synchronization doesn't work with MPI.

    N.B. While use_local_synchronization=True can be significantly faster with larger
    clusters and small process groups, care must be taken since it changes cluster behavior
    as non-member ranks don't join the group barrier().

    N.B. use_local_synchronization=True can lead to deadlocks when each rank creates
    multiple overlapping process groups. To avoid that, make sure all ranks follow the
    same global creation order.
    """
    return _new_group_with_tag(
        ranks,
        timeout,
        backend,
        pg_options,
        None,
        use_local_synchronization=use_local_synchronization,
        group_desc=group_desc,
        device_id=device_id,
    )


def _new_group_with_tag(
    ranks=None,
    timeout=None,
    backend=None,
    backend_options=None,
    pg_tag=None,
    use_local_synchronization=False,
    group_desc=None,
    device_id: Optional[torch.device] = None,
):
    """
    Variant of ``new_group`` that exposes tag creation.

    :: N.B. The mechanism is experimental and tied to the functional collectives effort, see
    ``torch.distributed._functional_collectives`` for reference on how to use it.
    """
    global _world

    default_pg = _get_default_group()
    if device_id is None:
        device_id = default_pg.bound_device_id
    elif default_pg.bound_device_id is not None:
        assert device_id == default_pg.bound_device_id, (
            "Mismatched bound device between new pg and the default pg."
        )
    default_backend, default_store = _world.pg_map[default_pg]
    global_rank = default_pg.rank()
    global_world_size = default_pg.size()

    # Default to the same backend as the global process group
    # if the backend is not specified.
    if not backend:
        backend = default_backend
    backend = Backend(backend)

    # this timeout defaulting/validation is used for all the new_groups/new_subgroups variants,
    # which may just pass their timeout value (or None)
    if timeout is None:
        timeout = _get_default_timeout(backend)
    _check_valid_timeout(timeout)

    if use_local_synchronization:
        # MPI backend doesn't have have a way for us to perform a partial sync
        if backend == Backend.MPI:
            raise ValueError(
                "MPI backend doesn't support use_local_synchronization=True"
            )
        if ranks is not None and get_rank() not in ranks:
            return None

    # checks the input ranks
    if ranks is not None:
        ranks = sorted(ranks)
        group_world_size = len(ranks)
        if group_world_size > global_world_size:
            raise ValueError(
                "the new group's world size should be less or "
                "equal to the world size set by "
                "init_process_group"
            )
        # check ranks' sanity
        for rank in ranks:
            if rank < 0 or rank >= global_world_size:
                raise ValueError(
                    "The new group's rank should be within "
                    "the world_size set by init_process_group"
                )
        if global_rank in ranks:
            group_rank = ranks.index(global_rank)
        else:
            group_rank = None
    else:
        ranks = list(range(global_world_size))
        group_world_size = global_world_size
        group_rank = global_rank

    group_name = _process_group_name(ranks, use_hashed_name=use_local_synchronization)

    pg, pg_store = _new_process_group_helper(
        group_world_size,
        group_rank,
        ranks,
        backend,
        default_store,
        group_name,
        backend_options=backend_options,
        timeout=timeout,
        pg_tag=pg_tag,
        device_id=device_id,
        group_desc=group_desc,
    )

    # Create the global rank to group rank mapping
    _world.pg_group_ranks[pg] = {
        global_rank: group_rank for group_rank, global_rank in enumerate(ranks)
    }

    if _is_barrier_after_init() == 1:
        # barrier at the end to ensure that once we return from this method, all
        # process groups including global variables (if any) are updated
        # correctly on all ranks.
        # Update 04/2023: for large-scale runs, this barrier (esp. store-based
        # barrier) may be costly and/or unscalable. Also, in a lot of cases,
        # these barriers may be unnecessary, as proven by a green CI after
        # removal. An environment variable `TORCH_DIST_INIT_BARRIER` has been
        # added which enables this barrier only when set to 1.
        logger.info(
            "Performing barrier after ProcessGroup initialization since "
            "TORCH_DIST_INIT_BARRIER = 1"
        )
        if backend == Backend.MPI:
            # MPI doesn't have store.
            barrier()
        else:
            barrier_store = pg_store if use_local_synchronization else default_store
            world_size = len(ranks) if use_local_synchronization else get_world_size()
            # Use store based barrier here since barrier() used a bunch of
            # default devices and messes up NCCL internal state.
            _store_based_barrier(
                global_rank, barrier_store, group_name, world_size, timeout
            )

    return pg


def new_subgroups(
    group_size=None,
    group=None,
    timeout=None,
    backend=None,
    pg_options=None,
    group_desc=None,
):
    """
    Create subgroups of equal size.

    By default, it creates intra-machine subgroups,
    where each of which contains all the ranks of a machine, based on the assumption
    that each machine has the same number of devices.

    This is a convenience API that calls ``new_group`` to generate multiple subgroups.
    It requires that all processes in the main group (i.e. all
    processes that are part of the distributed job) enter this function, even
    if they are not going to be members of the group.

    .. warning::
        If ``group_size`` is passed in, the world size must be divisible by ``group_size``.
        If no ``group_size`` is passed in, it believe that you are creating a group based
        on CUDA and determining the group size by number of CUDA devices, and if not all
        the machines have the same number of devices, the subgroup division will be
        different across nodes and can cause unexpected behaviors. Therefore, if you are
        creating a subgroup that does not depend on CUDA (such as Gloo on CPU), please
        pass in ``group_size`` correctly.

    .. warning::
        See warning `Safe concurrent usage` for `new_group` API for important details about
        using multiple process groups concurrently in a safe manner.

    Args:
        group_size (int, optional): The size of each subgroup. If ``None``,
            the default subgroup size is equal to the number of devices on each machine,
            based on the assumption that each machine has exactly the same
            number of devices. Default is ``None``.
        group (ProcessGroup, optional): The process group to work on. If
            ``None``, the default process group will be used. Default is ``None``.
        timeout (timedelta, optional): see `init_process_group` for details and default value.
        backend (str or Backend, optional): The backend to use. Depending on
            build-time configurations, valid values are ``gloo`` and ``nccl``.
            By default uses the same backend as the global group. This field
            should be given as a lowercase string (e.g., ``"gloo"``), which can
            also be accessed via :class:`Backend` attributes (e.g.,
            ``Backend.GLOO``). If ``None`` is passed in, the backend
            corresponding to the default process group will be used. Default is
            ``None``.
        pg_options (ProcessGroupOptions, optional): process group options
            specifying what additional options need to be passed in during
            the construction of specific process groups. i.e. for the ``nccl``
            backend, ``is_high_priority_stream`` can be specified so that
            process group can pick up high priority cuda streams.
        group_desc (str, optional): A string describing the group. Each subgroup will
            inherit its group_desc

    Returns:
        The subgroup containing the current rank, and all the subgroups used for cleanup.

    Examples:
        >>> # Create intra-machine subgroups.
        >>> # xdoctest: +SKIP("need process group init")
        >>> cur_subgroup, subgroups = dist.new_subgroups()
        >>> # Allreduce within the machine.
        >>> rank = dist.get_rank()
        >>> tensor = torch.ones(1, device=rank) * rank
        >>> dist.all_reduce(tensor, group=cur_subgroup)
        >>> tensor
        tensor([28])  # Assume 8 CUDA devices per machine.  28 is sum(range(8)).
        >>> # Cleanup.
        >>> for subgroup in subgroups:
        >>>     dist.destroy_process_group(subgroup)
    """
    if group_size is None:
        if not torch.cuda.is_available():
            raise ValueError(
                "Default group size only takes effect when CUDA is available."
                "If your subgroup using a backend that does not depend on CUDA,"
                "please pass in 'group_size' correctly."
            )
        group_size = torch.cuda.device_count()
    if group_size <= 0:
        raise ValueError(f"The arg 'group_size' ({group_size}) must be positive")

    world_size = get_world_size(group=group)
    if world_size < group_size:
        raise ValueError(
            f"The arg 'group_size' ({group_size}) must not exceed the world size ({world_size})"
        )
    if world_size % group_size != 0:
        raise ValueError(
            f"The world size ({world_size}) must be divisible by '{group_size=}'"
        )

    # TODO: Use itertools.batched(get_process_group_ranks(group=group), group_size) instead when Python 3.12 is supported.
    ranks = get_process_group_ranks(group=group)
    ranks_per_subgroup_list = [
        ranks[i : i + group_size] for i in range(0, len(ranks), group_size)
    ]
    return new_subgroups_by_enumeration(
        ranks_per_subgroup_list,
        timeout=timeout,
        backend=backend,
        pg_options=pg_options,
        group_desc=group_desc,
    )


def new_subgroups_by_enumeration(
    ranks_per_subgroup_list,
    timeout=None,
    backend=None,
    pg_options=None,
    group_desc=None,
):
    """
    Create subgroups by dividing the global world.

    The division is specified by a nested list of ranks. The subgroups cannot have
    overlap, and some ranks may not have to be in any subgroup.

    This is a convenience API that calls ``new_group`` to generate multiple subgroups.
    It requires that all processes in the main group (i.e. all
    processes that are part of the distributed job) enter this function, even
    if they are not going to be members of the group.

    .. warning::
        See warning `Safe concurrent usage` for `new_group` API for important details about
        using multiple process groups concurrently in a safe manner.

    Args:
        ranks_per_subgroup_list (list[list[int]]): A nested list of ranks of
            group members.
        timeout (timedelta, optional): see `init_process_group` for details and default value.
        backend (str or Backend, optional): The backend to use. Depending on
             build-time configurations, valid values are ``gloo`` and ``nccl``.
             By default uses the same backend as the global group. This field
             should be given as a lowercase string (e.g., ``"gloo"``), which can
             also be accessed via :class:`Backend` attributes (e.g.,
             ``Backend.GLOO``). If ``None`` is passed in, the backend
             corresponding to the default process group will be used. Default is
             ``None``.
        pg_options (ProcessGroupOptions, optional): process group options
            specifying what additional options need to be passed in during
            the construction of specific process groups. i.e. for the ``nccl``
            backend, ``is_high_priority_stream`` can be specified so that
            process group can pick up high priority cuda streams.
        group_desc (str, optional): A string describing the group. Each subgroup will
            inherit its group_desc.

    Returns:
        The subgroup containing the current rank, and all the subgroups used for cleanup.

    Examples:
        >>> # Create two subgroups, where each has 2 processes.
        >>> # xdoctest: +SKIP("need process group init")
        >>> cur_subgroup, subgroups = dist.new_subgroups(ranks=[[0, 2], [1, 3]])
        >>> rank = dist.get_rank()
        >>> tensor = torch.ones(1, device=rank) * rank
        >>> dist.all_reduce(tensor, group=cur_subgroup)
        >>> tensor
        tensor([2])     # Subgroup 0: ranks 0 and 2
        tensor([4])     # Subgroup 1: ranks 1 and 3
    """
    if ranks_per_subgroup_list is None or len(ranks_per_subgroup_list) == 0:
        raise ValueError("The arg 'ranks_per_subgroup_list' cannot be empty")

    subgroups = []
    cur_subgroup = None
    # Create a mapping from rank to subgroup to check if there is any subgroup overlap.
    rank_to_ranks_dict = {}  # type: ignore[var-annotated]
    for ranks in ranks_per_subgroup_list:
        subgroup = new_group(
            ranks=ranks,
            timeout=timeout,
            backend=backend,
            pg_options=pg_options,
            group_desc=group_desc,
        )
        subgroups.append(subgroup)
        my_rank = get_rank()
        for rank in ranks:
            if rank in rank_to_ranks_dict:
                raise ValueError(
                    f"Rank {rank} has appeared in both subgroup {rank_to_ranks_dict[rank]} and {ranks}"
                )
            rank_to_ranks_dict[rank] = ranks
            if my_rank == rank:
                cur_subgroup = subgroup
                logger.info("Rank %s is assigned to subgroup %s", rank, ranks)

    return cur_subgroup, subgroups


def _find_pg_by_ranks_and_tag(tag: str, ranks: list[int]) -> Optional[ProcessGroup]:
    if len(tag) > 0 and not tag.startswith("ptd:") and not tag.startswith("user:"):
        tag = f"user:{tag}"

    for group in _world.tags_to_pg.get(tag, []):
        if group.size() != len(ranks):
            continue

        group_ranks = get_process_group_ranks(group)
        good = all(r in group_ranks for r in ranks)
        if good:
            return group
    return None


def _find_or_create_pg_by_ranks_and_tag(
    tag: str, ranks: list[int], stride: int
) -> ProcessGroup:
    assert len(ranks) % stride == 0, (
        f"Ranks length ({len(ranks)}) must be divisible by stride ({stride})"
    )

    my_rank = get_rank()
    my_ranks = None

    if stride == len(ranks):
        my_ranks = ranks.copy()
        assert my_rank in my_ranks, "rankset doesn't include the current node"
    else:
        for i in range(0, len(ranks), stride):
            rank_set = ranks[i : i + stride]
            if my_rank in rank_set:
                my_ranks = rank_set
        assert my_ranks is not None, "rankset doesn't include the current node"

    my_ranks = sorted(my_ranks)

    pg = _find_pg_by_ranks_and_tag(tag, my_ranks)
    if pg is not None:
        return pg
    if tag == "":
        raise ValueError("Cannot automatically create PG with empty tag")
    # TODO copy settings and timeout from default PG
    return _new_group_with_tag(my_ranks, pg_tag=tag)


def _get_group_tag(pg: ProcessGroup) -> str:
    """Return the tag associated with ``pg``."""
    tag = _world.pg_to_tag[pg]
    tag = tag.removeprefix("user:")
    return tag


def _get_process_group_name(pg: ProcessGroup) -> str:
    return _world.pg_names.get(pg, "None")


def _get_process_group_store(pg: ProcessGroup) -> Store:
    return _world.pg_map[pg][1]
