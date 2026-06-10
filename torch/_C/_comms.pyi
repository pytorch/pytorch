#!/usr/bin/env python3
# pyre-strict
# Copyright (c) Meta Platforms, Inc. and affiliates.

from collections.abc import Callable
from datetime import timedelta
from enum import auto, Enum
from typing import Any, TypeAlias

InitHandle = str

class RedOpType(Enum):
    SUM = auto()
    PRODUCT = auto()
    MIN = auto()
    MAX = auto()
    BAND = auto()
    BOR = auto()
    BXOR = auto()
    PREMUL_SUM = auto()
    AVG = auto()

class ReduceOp:
    SUM: ReduceOp = ...
    PRODUCT: ReduceOp = ...
    MIN: ReduceOp = ...
    MAX: ReduceOp = ...
    BAND: ReduceOp = ...
    BOR: ReduceOp = ...
    BXOR: ReduceOp = ...
    AVG: ReduceOp = ...
    @staticmethod
    def PREMUL_SUM(factor: Any) -> ReduceOp: ...
    @property
    def type(self) -> RedOpType: ...

class OpName(Enum):
    """Collective operation name for hooks."""

    send = auto()
    recv = auto()
    broadcast = auto()
    all_reduce = auto()
    reduce = auto()
    all_gather = auto()
    all_gather_v = auto()
    all_gather_single = auto()
    reduce_scatter = auto()
    reduce_scatter_v = auto()
    reduce_scatter_single = auto()
    all_to_all_single = auto()
    all_to_all_v_single = auto()
    all_to_all = auto()
    barrier = auto()
    scatter = auto()
    gather = auto()
    gather_single = auto()
    split = auto()
    new_window = auto()
    finalize = auto()

class RemovableHandle:
    """Handle for removing a registered hook."""

    def remove(self) -> None: ...

# Per-collective pre-hook args (passed as typed 3rd argument to pre-hooks)
class SendPreHookArgs:
    @property
    def tensor(self) -> Any: ...
    @property
    def peer(self) -> int: ...
    @property
    def async_op(self) -> bool: ...

class RecvPreHookArgs:
    @property
    def tensor(self) -> Any: ...
    @property
    def peer(self) -> int: ...
    @property
    def async_op(self) -> bool: ...

class BroadcastPreHookArgs:
    @property
    def tensor(self) -> Any: ...
    @property
    def root(self) -> int: ...
    @property
    def async_op(self) -> bool: ...

class AllReducePreHookArgs:
    @property
    def tensor(self) -> Any: ...
    @property
    def async_op(self) -> bool: ...

class ReducePreHookArgs:
    @property
    def tensor(self) -> Any: ...
    @property
    def root(self) -> int: ...
    @property
    def async_op(self) -> bool: ...

class AllGatherPreHookArgs:
    @property
    def input(self) -> Any: ...
    @property
    def output(self) -> Any: ...
    @property
    def async_op(self) -> bool: ...

class AllGatherVPreHookArgs:
    @property
    def input(self) -> Any: ...
    @property
    def output(self) -> Any: ...
    @property
    def async_op(self) -> bool: ...

class AllGatherSinglePreHookArgs:
    @property
    def input(self) -> Any: ...
    @property
    def output(self) -> Any: ...
    @property
    def async_op(self) -> bool: ...

class ReduceScatterPreHookArgs:
    @property
    def input(self) -> Any: ...
    @property
    def output(self) -> Any: ...
    @property
    def async_op(self) -> bool: ...

class ReduceScatterVPreHookArgs:
    @property
    def input(self) -> Any: ...
    @property
    def output(self) -> Any: ...
    @property
    def async_op(self) -> bool: ...

class ReduceScatterSinglePreHookArgs:
    @property
    def input(self) -> Any: ...
    @property
    def output(self) -> Any: ...
    @property
    def async_op(self) -> bool: ...

class AllToAllSinglePreHookArgs:
    @property
    def input(self) -> Any: ...
    @property
    def output(self) -> Any: ...
    @property
    def async_op(self) -> bool: ...

class AllToAllVSinglePreHookArgs:
    @property
    def input(self) -> Any: ...
    @property
    def output(self) -> Any: ...
    @property
    def async_op(self) -> bool: ...

class AllToAllPreHookArgs:
    @property
    def input(self) -> Any: ...
    @property
    def output(self) -> Any: ...
    @property
    def async_op(self) -> bool: ...

class BarrierPreHookArgs:
    @property
    def async_op(self) -> bool: ...

class ScatterPreHookArgs:
    @property
    def output(self) -> Any: ...
    @property
    def input(self) -> Any: ...
    @property
    def root(self) -> int: ...
    @property
    def async_op(self) -> bool: ...

class GatherPreHookArgs:
    @property
    def input(self) -> Any: ...
    @property
    def output(self) -> Any: ...
    @property
    def root(self) -> int: ...
    @property
    def async_op(self) -> bool: ...

class GatherSinglePreHookArgs:
    @property
    def input(self) -> Any: ...
    @property
    def output(self) -> Any: ...
    @property
    def root(self) -> int: ...
    @property
    def async_op(self) -> bool: ...

class SplitPreHookArgs:
    @property
    def ranks(self) -> list[int]: ...
    @property
    def name(self) -> str: ...

class NewWindowPreHookArgs: ...
class FinalizePreHookArgs: ...

# Pre-hook args union type
PreHookArgs: TypeAlias = (
    SendPreHookArgs
    | RecvPreHookArgs
    | BroadcastPreHookArgs
    | AllReducePreHookArgs
    | ReducePreHookArgs
    | AllGatherPreHookArgs
    | AllGatherVPreHookArgs
    | AllGatherSinglePreHookArgs
    | ReduceScatterPreHookArgs
    | ReduceScatterVPreHookArgs
    | ReduceScatterSinglePreHookArgs
    | AllToAllSinglePreHookArgs
    | AllToAllVSinglePreHookArgs
    | AllToAllPreHookArgs
    | BarrierPreHookArgs
    | ScatterPreHookArgs
    | GatherPreHookArgs
    | GatherSinglePreHookArgs
    | SplitPreHookArgs
    | NewWindowPreHookArgs
    | FinalizePreHookArgs
)

# Per-collective post-hook args
class CollectivePostHookArgs: ...
class SendPostHookArgs(CollectivePostHookArgs): ...
class RecvPostHookArgs(CollectivePostHookArgs): ...
class BroadcastPostHookArgs(CollectivePostHookArgs): ...
class AllReducePostHookArgs(CollectivePostHookArgs): ...
class ReducePostHookArgs(CollectivePostHookArgs): ...
class AllGatherPostHookArgs(CollectivePostHookArgs): ...
class AllGatherVPostHookArgs(CollectivePostHookArgs): ...
class AllGatherSinglePostHookArgs(CollectivePostHookArgs): ...
class ReduceScatterPostHookArgs(CollectivePostHookArgs): ...
class ReduceScatterVPostHookArgs(CollectivePostHookArgs): ...
class ReduceScatterSinglePostHookArgs(CollectivePostHookArgs): ...
class AllToAllSinglePostHookArgs(CollectivePostHookArgs): ...
class AllToAllVSinglePostHookArgs(CollectivePostHookArgs): ...
class AllToAllPostHookArgs(CollectivePostHookArgs): ...
class BarrierPostHookArgs(CollectivePostHookArgs): ...
class ScatterPostHookArgs(CollectivePostHookArgs): ...
class GatherPostHookArgs(CollectivePostHookArgs): ...
class GatherSinglePostHookArgs(CollectivePostHookArgs): ...
class SplitPostHookArgs: ...
class NewWindowPostHookArgs: ...
class FinalizePostHookArgs: ...

PostHookArgs: TypeAlias = (
    SendPostHookArgs
    | RecvPostHookArgs
    | BroadcastPostHookArgs
    | AllReducePostHookArgs
    | ReducePostHookArgs
    | AllGatherPostHookArgs
    | AllGatherVPostHookArgs
    | AllGatherSinglePostHookArgs
    | ReduceScatterPostHookArgs
    | ReduceScatterVPostHookArgs
    | ReduceScatterSinglePostHookArgs
    | AllToAllSinglePostHookArgs
    | AllToAllVSinglePostHookArgs
    | AllToAllPostHookArgs
    | BarrierPostHookArgs
    | ScatterPostHookArgs
    | GatherPostHookArgs
    | GatherSinglePostHookArgs
    | SplitPostHookArgs
    | NewWindowPostHookArgs
    | FinalizePostHookArgs
)

class CommOptions:
    abort_process_on_timeout_or_error: bool
    timeout: timedelta
    store: Any
    name: str
    enable_reconfigure: bool
    hints: dict[str, str]
    def __init__(self) -> None: ...

class SendOptions:
    def __init__(self) -> None: ...
    timeout: timedelta
    hints: dict[str, str]

class RecvOptions:
    def __init__(self) -> None: ...
    timeout: timedelta
    hints: dict[str, str]

class BatchP2POptions:
    def __init__(self) -> None: ...
    timeout: timedelta
    hints: dict[str, str]

class ReconfigureOptions:
    """Options for the reconfigure() fault tolerance API."""

    uuid: int
    init_handles: list[InitHandle] | set[InitHandle]
    timeout: timedelta | None
    hints: dict[str, str]
    def __init__(
        self,
        uuid: int = ...,
        init_handles: list[InitHandle] | set[InitHandle] = ...,
        timeout: timedelta | None = None,
        hints: dict[str, str] | None = None,
    ) -> None: ...

class BroadcastOptions:
    def __init__(self) -> None: ...
    timeout: timedelta
    hints: dict[str, str]

class AllReduceOptions:
    def __init__(self) -> None: ...
    timeout: timedelta
    hints: dict[str, str]

class ReduceOptions:
    def __init__(self) -> None: ...
    timeout: timedelta
    hints: dict[str, str]

class AllGatherOptions:
    def __init__(self) -> None: ...
    timeout: timedelta
    hints: dict[str, str]

class AllGatherSingleOptions:
    def __init__(self) -> None: ...
    timeout: timedelta
    hints: dict[str, str]

class ReduceScatterOptions:
    def __init__(self) -> None: ...
    timeout: timedelta
    hints: dict[str, str]

class ReduceScatterSingleOptions:
    def __init__(self) -> None: ...
    timeout: timedelta
    hints: dict[str, str]

class AllToAllOptions:
    def __init__(self) -> None: ...
    timeout: timedelta
    hints: dict[str, str]

class AllToAllSingleOptions:
    def __init__(self) -> None: ...
    timeout: timedelta
    hints: dict[str, str]

class AllToAllvSingleOptions:
    def __init__(self) -> None: ...
    timeout: timedelta
    hints: dict[str, str]

class BarrierOptions:
    def __init__(self) -> None: ...
    timeout: timedelta
    hints: dict[str, str]

class ScatterOptions:
    def __init__(self) -> None: ...
    timeout: timedelta
    hints: dict[str, str]

class GatherOptions:
    def __init__(self) -> None: ...
    timeout: timedelta
    hints: dict[str, str]

class AllGatherPInitOptions:
    def __init__(self) -> None: ...
    timeout: timedelta
    hints: dict[str, str]

class AllGatherPExecOptions:
    def __init__(self) -> None: ...
    timeout: timedelta
    hints: dict[str, str]

# Opaque handle type for persistent AllGather
AllGatherPHandle: TypeAlias = Any

class WorkStatus(Enum):
    NOT_STARTED = auto()
    INPROGRESS = auto()
    COMPLETED = auto()
    TIMEDOUT = auto()
    ERROR = auto()

class TorchWork:
    def is_completed(self) -> bool: ...
    def wait(self) -> None: ...
    def wait_blocking(self) -> None: ...
    def _set_status(self, status: WorkStatus) -> None: ...

class TorchCommWinAccessType(Enum):
    WIN_ACCESS_TYPE_UNIFIED = auto()
    WIN_ACCESS_TYPE_SEPARATE = auto()

class TorchCommWindowAttr:
    def __init__(self) -> None: ...
    access_type: TorchCommWinAccessType

class TorchCommWindow:
    @property
    def dtype(self) -> Any: ...
    @property
    def shape(self) -> list[int]: ...
    @property
    def device(self) -> Any: ...
    def get_size(self) -> int: ...
    def get_tensor(self) -> Any | None: ...
    def tensor_register(
        self,
        tensor: Any,
        owning: bool = True,
    ) -> None: ...
    def tensor_deregister(
        self,
    ) -> None: ...
    def put(
        self,
        tensor: Any,
        dst_rank: int,
        target_offset_nelems: int,
        async_op: bool,
        hints: dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def signal(
        self,
        dst_rank: int,
        async_op: bool,
        hints: dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def wait_signal(
        self,
        peer_rank: int,
        async_op: bool,
        hints: dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def map_remote_tensor(
        self,
        rank: int,
    ) -> Any: ...
    def get_attr(self, peer_rank: int) -> TorchCommWindowAttr: ...

class P2POpType(Enum):
    SEND = auto()
    RECV = auto()

class P2POp:
    type: P2POpType
    tensor: Any
    peer: int
    def __init__(self, type: P2POpType, tensor: Any, peer: int) -> None: ...

class BatchSendRecv:
    ops: list[P2POp]
    def send(self, tensor: Any, dst: int) -> None: ...
    def recv(self, tensor: Any, src: int) -> None: ...
    def issue(self, async_op: bool, options: BatchP2POptions = ...) -> TorchWork: ...

class TorchComm:
    def finalize(self) -> None: ...
    def get_rank(self) -> int: ...
    def get_size(self) -> int: ...
    def get_name(self) -> str: ...
    def get_options(self) -> CommOptions: ...
    def get_device(self) -> Any: ...
    def get_backend(self) -> str: ...
    def get_backend_version(self) -> str: ...
    def get_backend_impl(self) -> TorchCommBackend: ...
    def unsafe_get_backend(self) -> TorchCommBackend: ...
    def get_init_handle(self) -> InitHandle: ...
    def reconfigure(
        self,
        uuid: int,
        init_handles: list[InitHandle] | set[InitHandle],
        timeout: timedelta | None = None,
        hints: dict[str, str] | None = None,
    ) -> TorchWork: ...
    def abort(self) -> None: ...
    def is_abort_supported(self) -> bool: ...
    def is_aborted(self) -> bool: ...
    def send(
        self,
        tensor: Any,
        dst: int,
        async_op: bool,
        hints: dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def recv(
        self,
        tensor: Any,
        src: int,
        async_op: bool,
        hints: dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def broadcast(
        self,
        tensor: Any,
        root: int,
        async_op: bool,
        hints: dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def all_reduce(
        self,
        tensor: Any,
        op: ReduceOp,
        async_op: bool,
        hints: dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def reduce(
        self,
        tensor: Any,
        root: int,
        op: ReduceOp,
        async_op: bool,
        hints: dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def all_gather(
        self,
        tensor_list: list[Any],
        tensor: Any,
        async_op: bool,
        hints: dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def all_gather_v(
        self,
        tensor_list: list[Any],
        tensor: Any,
        async_op: bool,
        hints: dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def all_gather_single(
        self,
        output: Any,
        input: Any,
        async_op: bool,
        hints: dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def reduce_scatter(
        self,
        output: Any,
        input_list: list[Any],
        op: ReduceOp,
        async_op: bool,
        hints: dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def reduce_scatter_v(
        self,
        output: Any,
        input_list: list[Any],
        op: ReduceOp,
        async_op: bool,
        hints: dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def reduce_scatter_single(
        self,
        output: Any,
        input: Any,
        op: ReduceOp,
        async_op: bool,
        hints: dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def all_to_all_single(
        self,
        output: Any,
        input: Any,
        async_op: bool,
        hints: dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def all_to_all_v_single(
        self,
        output: Any,
        input: Any,
        output_split_sizes: list[int],
        input_split_sizes: list[int],
        async_op: bool,
        hints: dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def all_to_all(
        self,
        output_tensor_list: list[Any],
        input_tensor_list: list[Any],
        async_op: bool,
        hints: dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def barrier(
        self,
        async_op: bool,
        hints: dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def scatter(
        self,
        output_tensor: Any,
        input_tensor_list: list[Any],
        root: int,
        async_op: bool,
        hints: dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def gather(
        self,
        output_tensor_list: list[Any],
        input_tensor: Any,
        root: int,
        async_op: bool,
        hints: dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def gather_single(
        self,
        output: Any,
        input: Any,
        root: int,
        async_op: bool,
        hints: dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def split(
        self,
        ranks: list[int],
        name: str,
        hints: dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchComm: ...
    def batch_op_create(self) -> BatchSendRecv: ...
    def new_window(self, tensor: Any | None = None) -> TorchCommWindow: ...
    def all_gather_p_init(
        self,
        output: Any,
        hints: dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> AllGatherPHandle: ...
    def all_gather_p_exec(
        self,
        handle: AllGatherPHandle,
        input: Any,
        async_op: bool,
        hints: dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def all_gather_p_free(
        self,
        handle: AllGatherPHandle,
    ) -> None: ...
    def get_mem_allocator(self) -> Any: ...
    @property
    def mem_allocator(self) -> Any: ...
    def register_pre_hook(
        self, callback: Callable[[OpName, int, PreHookArgs], None]
    ) -> RemovableHandle: ...
    def register_post_hook(
        self, callback: Callable[[int, PostHookArgs], None]
    ) -> RemovableHandle: ...
    def register_abort_hook(self, callback: Callable[[], None]) -> RemovableHandle: ...
    def tensor_register(self, tensor: Any) -> None: ...
    def tensor_deregister(self, tensor: Any) -> None: ...

def new_comm(
    backend: str,
    device: Any,
    abort_process_on_timeout_or_error: bool | None = ...,
    timeout: timedelta | None = ...,
    store: Any | None = ...,
    name: str | None = ...,
    enable_reconfigure: bool = False,
    hints: dict[str, str] | None = ...,
) -> TorchComm: ...

class _BackendWrapper:
    def __init__(self, comm: TorchComm) -> None: ...
    def get_comm(self) -> TorchComm: ...
    def get_mem_allocator(self) -> Any: ...

def get_mem_allocator(backend: str) -> Any: ...

class TorchCommBackend:
    def __init__(self) -> None: ...

def register_backend(name: str, backend_class: type[TorchCommBackend]) -> None: ...
def _is_backend_registered(name: str) -> bool: ...
