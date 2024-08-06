import torch
from torch._C._distributed_c10d import Store
from torch._C._distributed_rpc import _TensorPipeRpcBackendOptionsBase, TensorPipeAgent

# This module is defined in torch/csrc/distributed/rpc/testing/init.cpp

class FaultyTensorPipeRpcBackendOptions(_TensorPipeRpcBackendOptionsBase):
    def __init__(
        self,
        num_worker_threads: int,
        rpc_timeout: float,
        init_method: str,
        messages_to_fail: list[str],
        messages_to_delay: dict[str, float],
        num_fail_sends: int,
    ) -> None: ...
    num_send_recv_threads: int
    messages_to_fail: list[str]
    messages_to_delay: dict[str, float]
    num_fail_sends: int

class FaultyTensorPipeAgent(TensorPipeAgent):
    def __init__(
        self,
        store: Store,
        name: str,
        rank: int,
        world_size: int,
        options: FaultyTensorPipeRpcBackendOptions,
        reverse_device_maps: dict[str, dict[torch.device, torch.device]],
        devices: list[torch.device],
    ) -> None: ...
