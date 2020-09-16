
from typing import List, Dict
from datetime import timedelta
import torch
from torch.distributed.distributed_c10d import ProcessGroup

# mypy declarations
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
class FaultyProcessGroupAgent:
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

def is_available():
    return hasattr(torch._C, "_faulty_agent_init")


if is_available() and not torch._C._faulty_agent_init():
    raise RuntimeError("Failed to initialize torch.distributed.rpc._testing")

if is_available():
    # Registers FAULTY_PROCESS_GROUP RPC backend.
    from . import faulty_agent_backend_registry
