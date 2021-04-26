from torch._C._distributed_rpc import _TensorPipeRpcBackendOptionsBase
from . import constants as rpc_contants

import torch

from typing import Dict, List, Optional, Union


DeviceType = Union[int, str, torch.device]


def _to_device_index(device: DeviceType) -> int:
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(
            "`set_devices` expect a list of CUDA devices, but got "
            f"device type {device.type}."
        )
    return device.index


def _to_device_index_map(device_map: Dict[DeviceType, DeviceType]) -> Dict[int, int]:
    device_index_map : Dict[int, int] = {}
    reverse_map : Dict[int, int] = {}
    for k in device_map:
        v = device_map[k]
        k, v = torch.device(k), torch.device(v)
        if k.type != 'cuda' or v.type != 'cuda':
            raise ValueError(
                "`set_device_map` only supports CUDA devices, "
                f"but got device pair {k}: {v}"

            )
        if v.index in reverse_map:
            raise ValueError(
                "`device_map` only supports 1-to-1 mapping, "
                f"trying to map {k} and {reverse_map[v.index]} to {v.index}"
            )
        device_index_map[k.index] = v.index
        reverse_map[v.index] = k.index
    return device_index_map


def _to_device_index_list(devices: List[DeviceType]) -> List[int]:
    return list(map(_to_device_index, devices))



class TensorPipeRpcBackendOptions(_TensorPipeRpcBackendOptionsBase):
    r"""
    The backend options for
    :class:`~torch.distributed.rpc.TensorPipeAgent`, derived from
    :class:`~torch.distributed.rpc.RpcBackendOptions`.

    Args:
        num_worker_threads (int, optional): The number of threads in the
            thread-pool used by
            :class:`~torch.distributed.rpc.TensorPipeAgent` to execute
            requests (default: 16).
        rpc_timeout (float, optional): The default timeout, in seconds,
            for RPC requests (default: 60 seconds). If the RPC has not
            completed in this timeframe, an exception indicating so will
            be raised. Callers can override this timeout for individual
            RPCs in :meth:`~torch.distributed.rpc.rpc_sync` and
            :meth:`~torch.distributed.rpc.rpc_async` if necessary.
        init_method (str, optional): The URL to initialize the distributed
            store used for rendezvous. It takes any value accepted for the
            same argument of :meth:`~torch.distributed.init_process_group`
            (default: ``env://``).
        device_maps (Dict[str, Dict], optional): Device placement mappings from
            this worker to the callee. Key is the callee worker name and value
            the dictionary (``Dict`` of ``int``, ``str``, or ``torch.device``)
            that maps this worker's devices to the callee worker's devices.
            (default: ``None``)
        devices (List[int, str, or ``torch.device``], optional): all local
            CUDA devices used by RPC agent. By Default, it will be initialized
            to all local devices from its own ``device_maps`` and corresponding
            devices from its peers' ``device_maps``. When processing CUDA RPC
            requests, the agent will properly synchronize CUDA streams for
            all devices in this ``List``.
    """
    def __init__(
        self,
        *,
        num_worker_threads: int = rpc_contants.DEFAULT_NUM_WORKER_THREADS,
        rpc_timeout: float = rpc_contants.DEFAULT_RPC_TIMEOUT_SEC,
        init_method: str = rpc_contants.DEFAULT_INIT_METHOD,
        device_maps: Optional[Dict[str, Dict[DeviceType, DeviceType]]] = None,
        devices: Optional[List[DeviceType]] = None,
        _transports: List = None,
        _channels: List = None,
    ):
        device_index_maps = (
            {} if device_maps is None else
            {k : _to_device_index_map(v) for k, v in device_maps.items()}
        )
        device_index_list = (
            [] if devices is None else
            _to_device_index_list(devices)
        )
        super().__init__(
            num_worker_threads,
            _transports,
            _channels,
            rpc_timeout,
            init_method,
            device_index_maps,
            device_index_list,
        )

    def set_device_map(self, to: str, device_map: Dict[DeviceType, DeviceType]):
        r"""
        Set device mapping between each RPC caller and callee pair. This
        function can be called multiple times to incrementally add
        device placement configurations.

        Args:
            worker_name (str): Callee name.
            device_map (Dict of int, str, or torch.device): Device placement
                mappings from this worker to the callee. This map must be
                invertible.

        Example::
            >>> # both workers
            >>> def add(x, y):
            >>>     print(x)  # tensor([1., 1.], device='cuda:1')
            >>>     return x + y, (x + y).to(2)
            >>>
            >>> # on worker 0
            >>> options = TensorPipeRpcBackendOptions(
            >>>     num_worker_threads=8,
            >>>     device_maps={"worker1": {0: 1}}
            >>>     # maps worker0's cuda:0 to worker1's cuda:1
            >>> )
            >>> options.set_device_map("worker1", {1: 2})
            >>> # maps worker0's cuda:1 to worker1's cuda:2
            >>>
            >>> rpc.init_rpc(
            >>>     "worker0",
            >>>     rank=0,
            >>>     world_size=2,
            >>>     backend=rpc.BackendType.TENSORPIPE,
            >>>     rpc_backend_options=options
            >>> )
            >>>
            >>> x = torch.ones(2)
            >>> rets = rpc.rpc_sync("worker1", add, args=(x.to(0), 1))
            >>> # The first argument will be moved to cuda:1 on worker1. When
            >>> # sending the return value back, it will follow the invert of
            >>> # the device map, and hence will be moved back to cuda:0 and
            >>> # cuda:1 on worker0
            >>> print(rets[0])  # tensor([2., 2.], device='cuda:0')
            >>> print(rets[1])  # tensor([2., 2.], device='cuda:1')
        """
        device_index_map = _to_device_index_map(device_map)
        curr_device_maps = super().device_maps

        if to in curr_device_maps:
            for k, v in device_index_map.items():
                if k in curr_device_maps[to] and v != curr_device_maps[to][k]:
                    raise ValueError(
                        "`set_device_map` only supports 1-to-1 mapping, trying"
                        f" to map {k} to {v} and {curr_device_maps[to][k]}"
                    )

        super()._set_device_map(to, device_index_map)

    def set_devices(self, devices: List[DeviceType]):
        r"""
        Set local devices used by the TensorPipe RPC agent. When processing
        CUDA RPC requests, the TensorPipe RPC agent will properly synchronize
        CUDA streams for all devices in this ``List``.

        Args:
            devices (List of int, str, or torch.device): local devices used by
                the TensorPipe RPC agent.
        """
        self.devices = _to_device_index_list(devices)
