import torch
from ._distributed_c10d import ProcessGroup, Store
from ._distributed_rpc import (
    _TensorPipeRpcBackendOptionsBase,
    TensorPipeAgent,
    WorkerInfo,
)
from typing import List, Dict, overload
from datetime import timedelta

# This module is defined in torch/csrc/distributed/rpc/testing/init.cpp

class FaultyTensorPipeRpcBackendOptions(_TensorPipeRpcBackendOptionsBase):
    def __init__(
        self,
        num_worker_threads: int,
        rpc_timeout: float,
        init_method: str,
        messages_to_fail: List[str],
        messages_to_delay: Dict[str, float],
        num_fail_sends: int,
    ): ...
    num_send_recv_threads: int
    messages_to_fail: List[str]
    messages_to_delay: Dict[str, float]
    num_fail_sends: int

class FaultyTensorPipeAgent(TensorPipeAgent):
    def __init__(
        self,
        store: Store,
        name: str,
        rank: int,
        world_size: int,
        process_group: ProcessGroup,
        options: FaultyTensorPipeRpcBackendOptions,
        reverse_device_maps: Dict[str, Dict[torch.device, torch.device]],
        devices: List[torch.device],
    ): ...
