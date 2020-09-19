from enum import Enum
from typing import Optional
from datetime import timedelta

# distributed/c10d/init.cpp
class ProcessGroup:
    ...
class Store:
    kDefaultTimeout: timedelta
    kNoTimeout: timedelta
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
