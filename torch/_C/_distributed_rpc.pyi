from typing import Iterator, Tuple, Dict, Optional, List, Any, overload, Callable, Type, Generic, TypeVar
from datetime import timedelta
import torch
from ._autograd import ProfilerConfig, ProfilerState
from ._distributed_c10d import ProcessGroup, Store

# mypy type annotations
# distributed/rpc/init.cpp
class _TensorPipeRpcBackendOptionsBase:
    num_worker_threads: int
    _transports: Optional[List[str]]
    _channels: Optional[List[str]]
    _init_method: str
    rpc_timeout: timedelta
    device_maps: Dict[str, Dict[int, int]]
    @overload
    def __init__(
        self,
        num_worker_threads: int,
        _transports: Optional[List],
        _channels: Optional[List],
        rpc_timeout: timedelta,
        init_method: str,
        device_maps: Dict[str, Dict[int, int]]): ...
    @overload
    def __init__(
            self,
            rpc_timeout: timedelta,
            init_method: str,
            num_worker_threads: int,
            _transports: Optional[List],
            _channels: Optional[List]
        ): ...
    def set_device_map(self, to: str, device_map: Dict[str, Dict[int, int]]): ...
class ProcessGroupAgent:
    def __init__(
        self,
        worker_name: str,
        pg: Any,
        numSendRecvThreads: int,
        rpcTimeout: timedelta
    ): ...
class TensorPipeAgent:
    def __init__(
        self,
        store: Store,
        name: str,
        worker_id: int,
        world_size: int,
        pg: ProcessGroup,
        opts: _TensorPipeRpcBackendOptionsBase,
    ): ...
class ProcessGroupRpcBackendOptions:
    def __init__(
        self,
        num_send_recv_threads: int,
        rpc_timeout: float,
        init_method: str
    ): ...
# distributed/rpc/remote_profiler_manager.h
class RemoteProfilerManager:
    @staticmethod
    def set_current_profiling_key(key: str): ...
class WorkerInfo:
    name: str
    worker_id: int
    def __init__(self, name: str, worker_id: int): ...
class _RpcAgent:
    def join(self): ...
    @overload
    def get_worker_info(self, work_name) -> WorkerInfo: ...
    @overload
    def get_worker_info(self) -> WorkerInfo: ...
    def get_worker_infos(self) -> List[WorkerInfo]: ...
    def get_debug_info(self) -> Dict[str, str]: ...
    def shutdown(self): ...
# distributed/rpc/py_rref.h
class PyRRef:
    @staticmethod
    def _deserialize(tp: Tuple) -> 'PyRRef': ...
# distributed/rpc/rpc_agent.h
class RpcBackendOptions:
    rpcTimeoutSeconds: float
    initMethod: str
# distributed/autograd/profiler.h
class Event:
    name: str
def _disable_server_process_global_profiler() -> List[List[List[Event]]]: ...
def _enable_server_process_global_profiler(new_config: ProfilerConfig): ...
# distributed/rpc/init.cpp
def _set_profiler_node_id(default_node_id: int) -> None: ...
def _enable_jit_rref_pickle() -> None: ...
def _disable_jit_rref_pickle() -> None: ...
def _set_and_start_rpc_agent(agent: _RpcAgent) -> None: ...
def _is_current_rpc_agent_set() -> bool: ...
def _rref_context_get_debug_info() -> Dict[str, str]: ...
def _set_rpc_timeout(timeout: float): ...
def get_rpc_timeout() -> float: ...
def enable_gil_profiling(flag: bool): ...
_DEFAULT_NUM_SEND_RECV_THREADS: int
_DEFAULT_INIT_METHOD: str
_DEFAULT_NUM_WORKER_THREADS: int
_UNSET_RPC_TIMEOUT: bool
_DEFAULT_RPC_TIMEOUT_SEC: timedelta
# distributed/rpc/python_functions.h
def _cleanup_python_rpc_handler(): ...
def _delete_all_user_and_unforked_owner_rrefs(): ...
def _destroy_rref_context(ignore_rref_leak: bool): ...
def _get_current_rpc_agent()-> _RpcAgent: ...
def _invoke_remote_builtin(
    dst: WorkerInfo,
    op_name: str,
    rpc_timeout_seconds: float,
    *args: Any,
    **kwargs: Any
    ): ...
def _invoke_remote_python_udf(
    dst: WorkerInfo,
    qualified_name_str: str,
    rpc_timeout_seconds: float,
    is_async_execution: bool,
    *args: Any,
    **kwargs: Any
    ): ...
def _invoke_remote_torchscript(
    dst: WorkerInfo,
    qualified_name_str: str,
    rpc_timeout_seconds: float,
    is_async_execution: bool,
    *args: Any,
    **kwargs: Any
    ): ...
def _invoke_rpc_builtin(
    dst: WorkerInfo,
    qualified_name_str: str,
    rpc_timeout_seconds: float,
    *args: Any,
    **kwargs: Any
    ): ...
def _invoke_rpc_python_udf(
    dst: WorkerInfo,
    qualified_name_str: str,
    tensors: List[torch.Tensor],
    rpc_timeout_seconds: float,
    is_async_execution: bool
    ): ...
def _invoke_rpc_torchscript(
    dstWorkerName: str,
    qualifiedNameStr: str,
    rpcTimeoutSeconds: float,
    isAsyncExecution: bool,
    *args: Any,
    **kwargs: Any
    ): ...
def _reset_current_rpc_agent(): ...
