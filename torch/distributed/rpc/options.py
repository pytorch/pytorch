from . import _TensorPipeRpcBackendOptionsBase
from . import constants as rpc_contants

import torch

from typing import Dict


class TensorPipeRpcBackendOptions(_TensorPipeRpcBackendOptionsBase):

    def __init__(
        self,
        num_worker_threads: int = rpc_contants.DEFAULT_NUM_WORKER_THREADS,
        _transports: List = None,
        _channels: List = None,
        rpc_timeout: float = rpc_contants.DEFAULT_RPC_TIMEOUT_SEC,
        init_method: str = rpc_contants.DEFAULT_INIT_METHOD
    ):
        super().__init__(
            num_worker_threads,
            _transports,
            _channels,
            rpc_timeout,
            init_method
        )

    def set_device_map(self, to: str, device_map: Dict):
        device_index_map = {}
        curr_device_maps = super().device_maps
        for k in device_map:
            v = device_map[k]
            k, v = torch.device(k), torch.device(v)
            if k.type != 'cuda' or v.type != 'cuda':
                raise ValueError(
                    "`set_device_map` only supports CUDA devices, "
                    f"but got device pair {k}: {v}"

                )
            if to in curr_device_maps and k.index in curr_device_maps[to]:
                curr_v = super().device_maps[to][k.index]
                raise ValueError(
                    "`set_device_map` only supports 1-to-1 mapping, "
                    f"trying to map {k} to {v} and {curr_v}"
                )
            device_index_map[k.index] = v.index
        super().set_device_map(to, device_index_map)
