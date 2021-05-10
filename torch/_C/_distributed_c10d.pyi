from datetime import timedelta
from enum import Enum
from typing import Optional, List, Any, Tuple, overload

from torch import Tensor

# This module is defined in torch/csrc/distributed/c10d/init.cpp

_DEFAULT_FIRST_BUCKET_BYTES: int
_DEFAULT_NO_TIMEOUT: timedelta
_DEFAULT_PG_TIMEOUT: timedelta

class BuiltinCommHookType(Enum):
    ALLREDUCE = ...
    FP16_COMPRESS = ...

def _register_comm_hook(reducer: Reducer, state: Any, comm_hook: Any): ...
def _register_builtin_comm_hook(
    reducer: Reducer, comm_hook_type: BuiltinCommHookType
): ...

class GradBucket:
    def __init__(
        self,
        index: int,
        tensor: Tensor,
        offsets: List[int],
        lengths: List[int],
        sizes_list: List[Tuple[int]],
    ): ...
    def get_index(self) -> int: ...
    def get_tensor(self) -> Tensor: ...
    def get_per_parameter_tensors(self) -> List[Tensor]: ...
    def is_the_last_bucket_to_allreduce(self) -> bool: ...
    def set_tensor(self, tensor: Tensor) -> None: ...

class Reducer:
    def __init__(
        self,
        replicas: List[List[Tensor]],
        bucket_indices: List[List[int]],
        process_group: ProcessGroup,
        expect_sparse_gradients: List[List[bool]],
        bucket_bytes_cap: int,
        find_unused_parameters: bool,
        gradient_as_bucket_view: bool,
    ): ...
    def initialize_buckets(self, bucket_indices: List[List[int]]): ...
    ...

class Logger:
    def __init__(self, reducer: Reducer): ...
    def set_construction_data_and_log(
        self,
        module_name: str,
        device_ids: List[int],
        output_device: int,
        broadcast_buffers: bool,
    ): ...
    ...

def _get_debug_mode(): ...

class _DistributedDebugLevel(Enum):
    OFF = ...
    INFO = ...
    DETAIL = ...

class ReduceOp(Enum):
    SUM = ...
    PRODUCT = ...
    MIN = ...
    MAX = ...
    BAND = ...
    BOR = ...
    BXOR = ...
    UNUSED = ...

class BroadcastOptions:
    rootRank: int
    rootTensor: int
    timeout: timedelta

class AllreduceOptions:
    reduceOp: ReduceOp
    timeout: timedelta

class AllreduceCoalescedOptions(AllreduceOptions): ...

class ReduceOptions:
    reduceOp: ReduceOp
    rootRank: int
    rootTensor: int
    timeout: timedelta

class AllGatherOptions:
    timeout: timedelta

class GatherOptions:
    rootRank: int
    timeout: timedelta

class ScatterOptions:
    rootRank: int
    timeout: timedelta

class ReduceScatterOptions:
    reduceOp: ReduceOp
    timeout: timedelta

class BarrierOptions:
    device_ids: List[int]
    timeout: timedelta

class AllToAllOptions:
    timeout: timedelta

class Store:
    def set(self, key: str, value: str): ...
    def get(self, key: str) -> bytes: ...
    def add(self, key: str, value: int) -> int: ...
    def compare_set(self, key: str, expected_value: str, desired_value: str) -> bytes: ...
    def delete_key(self, key: str) -> bool: ...
    def num_keys(self) -> int: ...
    def set_timeout(self, timeout: timedelta): ...
    @overload
    def wait(self, keys: List[str]): ...
    @overload
    def wait(self, keys: List[str], timeout: timedelta): ...

class FileStore(Store):
    def __init__(self, path: str, numWorkers: int): ...

class HashStore(Store):
    def __init__(self): ...

class TCPStore(Store):
    def __init__(
        self,
        host_name: str,
        port: int,
        world_size: int = ...,
        is_master: bool = ...,
        timeout: timedelta = ...,
        wait_for_workers: bool = ...
    ): ...

class PrefixStore(Store):
    def __init__(self, prefix: str, store: Store): ...

class Work:
    def is_completed(self) -> bool: ...
    def is_success(self) -> bool: ...
    def exception(self) -> Any: ...
    def wait(self, timeout: timedelta = _DEFAULT_NO_TIMEOUT) -> bool: ...
    def source_rank(self) -> int: ...
    def _source_rank(self) -> int: ...
    def result(self) -> List[Tensor]: ...
    def synchronize(self): ...
    ...

class ProcessGroup:
    class Options: ...
    def __init__(self): ...
    def rank(self) -> int: ...
    def size(self) -> int: ...
    @overload
    def broadcast(
        self,
        tensors: List[Tensor],
        opts=BroadcastOptions(),
    ) -> Work: ...
    @overload
    def broadcast(
        self,
        tensor: Tensor,
        root: int,
    ) -> Work: ...
    @overload
    def allreduce(
        self,
        tensors: List[Tensor],
        opts: AllreduceOptions = AllreduceOptions(),
    ) -> Work: ...
    @overload
    def allreduce(
        self,
        tensors: List[Tensor],
        op=ReduceOp.SUM,
    ) -> Work: ...
    @overload
    def allreduce(
        self,
        tensor: Tensor,
        op=ReduceOp.SUM,
    ) -> Work: ...
    def allreduce_coalesced(
        self,
        tensors: List[Tensor],
        opts=AllreduceCoalescedOptions(),
    ) -> Work: ...
    @overload
    def reduce(
        self,
        tensors: List[Tensor],
        opts=ReduceOptions(),
    ) -> Work: ...
    @overload
    def reduce(
        self,
        tensor: Tensor,
        root: int,
        op=ReduceOp.SUM,
    ) -> Work: ...
    @overload
    def allgather(
        self,
        output_tensors: List[List[Tensor]],
        input_tensors: List[Tensor],
        opts=AllGatherOptions(),
    ) -> Work: ...
    @overload
    def allgather(
        self,
        output_tensors: List[Tensor],
        input_tensor: Tensor,
    ) -> Work: ...
    def _allgather_base(
        self,
        output: Tensor,
        input: Tensor,
        opts = AllGatherOptions(),
    ) -> Work: ...
    def allgather_coalesced(
        self,
        output_lists: List[List[Tensor]],
        input_list: List[Tensor],
        opts=AllGatherOptions(),
    ) -> Work: ...
    @overload
    def gather(
        self,
        output_tensors: List[List[Tensor]],
        input_tensors: List[Tensor],
        opts=GatherOptions(),
    ) -> Work: ...
    @overload
    def gather(
        self,
        output_tensors: List[Tensor],
        input_tensor: Tensor,
        root: int,
    ) -> Work: ...
    @overload
    def scatter(
        self,
        output_tensors: List[Tensor],
        input_tensors: List[List[Tensor]],
        opts=ScatterOptions(),
    ) -> Work: ...
    @overload
    def scatter(
        self,
        output_tensor: Tensor,
        input_tensors: List[Tensor],
        root: int,
    ) -> Work: ...
    @overload
    def reduce_scatter(
        self,
        output_tensors: List[Tensor],
        input_tensors: List[List[Tensor]],
        opts=ReduceScatterOptions(),
    ) -> Work: ...
    @overload
    def reduce_scatter(
        self,
        output_tensors: Tensor,
        input_tensor: List[Tensor],
    ) -> Work: ...
    @overload
    def alltoall_base(
        self,
        output_tensor: Tensor,
        input_tensor: Tensor,
        output_split_sizes: List[int],
        input_split_sizes: List[int],
        opts=AllToAllOptions(),
    ) -> Work: ...
    @overload
    def alltoall_base(
        self,
        output: Tensor,
        input: Tensor,
        output_split_sizes: List[int],
        input_split_sizes: List[int],
    ) -> Work: ...
    @overload
    def alltoall(
        self,
        output_tensor: List[Tensor],
        input_tensor: List[Tensor],
        opts=AllToAllOptions(),
    ) -> Work: ...
    @overload
    def alltoall(
        self,
        output: List[Tensor],
        input: List[Tensor],
    ) -> Work: ...
    def send(
        self,
        tensors: List[Tensor],
        dstRank: int,
        tag: int,
    ) -> Work: ...
    def recv(
        self,
        tensors: List[Tensor],
        srcRank: int,
        tag: int,
    ) -> Work: ...
    def recv_anysource(self, tensors: List[Tensor], tag: int) -> Work: ...
    def barrier(self, opts=BarrierOptions()) -> Work: ...

class ProcessGroupRoundRobin(ProcessGroup): ...

def _round_robin_process_groups(
    process_groups: List[ProcessGroup],
) -> ProcessGroupRoundRobin: ...

class ProcessGroupGloo(ProcessGroup):
    class Device: ...
    class Options: ...
    def __init__(
        self,
        store: Store,
        rank: int,
        size: int,
        timeout: timedelta,
    ): ...
    @staticmethod
    def create_device(hostname=str(), interface=str()) -> Device: ...
    ...
    @staticmethod
    def create_default_device() -> Device: ...
    ...

class ProcessGroupNCCL(ProcessGroup):
    class Options: ...
    def __init__(
        self,
        store: Store,
        rank: int,
        size: int,
        timeout: timedelta,
    ): ...
    @staticmethod
    def _group_start() -> None: ...
    @staticmethod
    def _group_end() -> None: ...
    ...

class ProcessGroupMPI(ProcessGroup):
    def __init__(
        self,
        rank: int,
        size: int,
        pgComm: int,
    ): ...
    @staticmethod
    def create(ranks: List[int]) -> ProcessGroupMPI: ...

def _compute_bucket_assignment_by_size(
    tensors: List[Tensor],
    bucket_size: int,
    expect_sparse_gradient: List[bool],
    tensor_indices: List[int],
) -> List[List[int]]: ...
def _broadcast_coalesced(
    process_group: ProcessGroup,
    tensors: List[Tensor],
    buffer_size: int,
    src: int,
): ...
def _test_python_store(store: Store): ...
def _verify_model_across_ranks(
    process_group: ProcessGroup, replicas: List[List[Tensor]]
): ...
