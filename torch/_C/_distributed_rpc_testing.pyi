from ._distributed_c10d import ProcessGroup, Store
from ._distributed_rpc import ProcessGroupAgent, ProcessGroupRpcBackendOptions, WorkerInfo
from typing import List, Dict, overload
from datetime import timedelta

# This module is defined in torch/csrc/distributed/rpc/testing/init.cpp

class FaultyProcessGroupRpcBackendOptions(ProcessGroupRpcBackendOptions):
    def __init__(
        self,
        num_send_recv_threads: int,
        rpc_timeout: float,
        init_method: str,
        messages_to_fail: List[str],
        messages_to_delay: Dict[str, float],
        num_fail_sends: int
    ): ...
    num_send_recv_threads: int
    messages_to_fail: List[str]
    messages_to_delay: Dict[str, float]
    num_fail_sends: int

class FaultyProcessGroupAgent(ProcessGroupAgent):
    def __init__(
        self,
        store: Store,
        name: str,
        process_group: ProcessGroup,
        num_send_recv_threads: int,
        rpc_timeout: timedelta,
        messages_to_fail: List[str],
        messages_to_delay: Dict[str, float],
        num_fail_sends: int
    ): ...
    def join(self): ...
    def shutdown(self): ...
    @overload
    def get_worker_info(self) -> WorkerInfo: ...
    @overload
    def get_worker_info(self, workerName: str) -> WorkerInfo: ...
    @overload
    def get_worker_info(self, id: int) -> WorkerInfo: ...
    def get_worker_infos(self) -> List[WorkerInfo]: ...
