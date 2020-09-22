from ._distributed_c10d import ProcessGroup
from ._distributed_rpc import ProcessGroupAgent
from datetime import timedelta
from typing import List, Dict

# distributed/rpc/testing/init.cpp
class FaultyProcessGroupRpcBackendOptions:
    def __init__(
        self,
        num_send_recv_threads: int,
        rpc_timeout: float,
        init_method: str,
        messages_to_fail: List[str],
        messages_to_delay: Dict[str, float],
        num_fail_sends: int
    ): ...
class FaultyProcessGroupAgent(ProcessGroupAgent):
    def __init__(
        self,
        name: str,
        process_group: ProcessGroup,
        num_send_recv_threads: int,
        rpc_timeout: timedelta,
        messages_to_fail: List[str],
        messages_to_delay: Dict[str, float],
        num_fail_sends: int
    ): ...
