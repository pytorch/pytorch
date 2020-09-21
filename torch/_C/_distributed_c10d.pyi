from torch import Tensor
from torch.nn import Parameter
from enum import Enum
from typing import Optional, List, Any
from datetime import timedelta

# distributed/c10d/init.cpp
class _GradBucket:
    def __init__(self, tensors: List[Tensor]): ...
    def get_tensors(self) -> List[Tensor]: ...
class Work:
    def wait(self) -> bool: ...
    def source_rank(self) -> int: ...
class Store:
    kDefaultTimeout: timedelta
    kNoTimeout: timedelta
    def set(self, key: str, value: str): ...
    def get(self, key: str) -> bytes: ...
    def add(self, key: str, value: int) -> int: ...
class PrefixStore(Store):
    def __init__(
        self,
        prefix: str,
        store: Store
    ): ...
class FileStore(Store):
    def __init__(
        self,
        path: str,
        numWorkers: int
    ): ...
class TCPStore(Store):
    def __init__(
        self,
        masterAddr: Optional[str],
        masterPort: int,
        numWorkers: int,
        isServer: bool,
        timeout: timedelta
    ): ...
class ReduceOp(Enum):
    SUM = 0
    PRODUCT = 1
    MIN = 2
    MAX = 3
    BAND = 4
    BOR = 5
    BXOR = 6
    UNUSED = 7
class AllGatherOptions:
    timeout: timedelta
class AllreduceOptions:
    reduceOp: ReduceOp
    timeout: timedelta
class AllreduceCoalescedOptions(AllreduceOptions): ...
class AllToAllOptions:
    timeout: timedelta
class BroadcastOptions:
    rootRank: int
    rootTensor: int
    timeout: timedelta
class GatherOptions:
    rootRank: int
    timeout: timedelta
class ReduceOptions:
    reduceOp: ReduceOp
    rootRank: int
    rootTensor: int
    timeout: timedelta
class ReduceScatterOptions:
    reduceOp: ReduceOp
    timeout: timedelta
class ScatterOptions:
    rootRank: int
    timeout: timedelta
class BarrierOptions: ...
class ProcessGroup:
    def __init__(self): ...
    def rank(self) -> int: ...
    def size(self) -> int: ...
    def allgather(
        self,
        outputTensors: List[List[Tensor]],
        inputTensors: List[Tensor],
        opts: AllGatherOptions = AllGatherOptions(),
    ) -> Work: ...
    def allgather_coalesced(
        self,
        outputTensors: List[List[Tensor]],
        inputTensors: List[Tensor],
        opts: AllGatherOptions = AllGatherOptions(),
    ) -> Work: ...
    def gather(
        self,
        outputTensors: List[List[Tensor]],
        inputTensors: List[Tensor],
        opts: GatherOptions = GatherOptions(),
    ) -> Work: ...
    def scatter(
        self,
        outputTensors: List[Tensor],
        inputTensors: List[List[Tensor]],
        opts: ScatterOptions = ScatterOptions(),
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
    def recv_anysource(
        self,
        tensors: List[Tensor],
        tag: int
    ) -> Work: ...
    def reduce_scatter(
        self,
        outputTensors: List[Tensor],
        inputTensors: List[List[Tensor]],
        opts: ReduceScatterOptions = ReduceScatterOptions(),
    ) -> Work: ...
    def alltoall_base(
        self,
        outputTensor: Tensor,
        inputTensor: Tensor,
        outputSplitSizes: List[int],
        inputSplitSizes: List[int],
        opts: AllToAllOptions = AllToAllOptions(),
    ) -> Work: ...
    def alltoall(
        self,
        outputTensor: List[Tensor],
        inputTensor: List[Tensor],
        opts: AllToAllOptions = AllToAllOptions(),
    ) -> Work: ...
    def broadcast(
        self,
        tensors: List[Tensor],
        opts: BroadcastOptions = BroadcastOptions(),
    ) -> Work: ...
    def barrier(
        self,
        opts: BarrierOptions = BarrierOptions()
    ) -> Work: ...
    def reduce(
        self,
        tensors: List[Tensor],
        opts: ReduceOptions = ReduceOptions(),
    ) -> Work: ...
    def allreduce(
        self,
        tensors: List[Tensor],
        opts: AllreduceOptions = AllreduceOptions(),
    ) -> Work:  ...
    def allreduce_coalesced(
        self,
        tensors: List[Tensor],
        opts: AllreduceCoalescedOptions = AllreduceCoalescedOptions(),
    ) -> Work: ...
class ProcessGroupMPI(ProcessGroup):
    def __init__(
        self,
        rank: int,
        size: int,
        pgComm: int,
    ): ...
    @staticmethod
    def create(ranks: List[int]) -> ProcessGroupMPI: ...
class ProcessGroupNCCL(ProcessGroup):
    def __init__(
        self,
        store: Store,
        rank: int,
        size: int,
        timeout: timedelta,
    ): ...
class ProcessGroupGloo(ProcessGroup):
    def __init__(
        self,
        store: Store,
        rank: int,
        size: int,
        timeout: timedelta,
    ): ...
def _compute_bucket_assignment_by_size(
    tensors: List[Tensor],
    bucket_size: int,
    expect_sparse_gradient: List[bool],
    tensor_indices: List[int]) -> List[List[int]]: ...
def _broadcast_coalesced(
    process_group: ProcessGroup,
    tensors: List[Tensor],
    buffer_size: int,
    rank: int,
): ...
class Reducer:
    def __init__(
        self,
        replicas: List[List[Parameter]],
        bucket_indices: List[List[int]],
        process_group: ProcessGroup,
        expect_sparse_gradients: List[List[bool]],
        bucket_bytes_cap: int,
        find_unused_parameters: bool,
    ): ...
def _register_comm_hook(reducer: Reducer, state: Any, comm_hook: Any): ...
_DEFAULT_FIRST_BUCKET_BYTES: int
