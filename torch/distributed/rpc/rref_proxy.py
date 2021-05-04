from functools import partial

from . import functions

import torch
from .constants import UNSET_RPC_TIMEOUT

def _local_invoke(rref, func_name, args, kwargs):
    return getattr(rref.local_value(), func_name)(*args, **kwargs)

@functions.async_execution
def _local_invoke_async_execution(rref, func_name, args, kwargs):
    return getattr(rref.local_value(), func_name)(*args, **kwargs)

def _invoke_rpc(rref, rpc_api, func_name, timeout, *args, **kwargs):
    # Since rref._get_type can potentially issue an RPC, it should respect the
    # passed in timeout here.
    rref_type = rref._get_type(timeout=timeout)

    _invoke_func = _local_invoke
    # Bypass ScriptModules when checking for async function attribute.
    bypass_type = issubclass(rref_type, torch.jit.ScriptModule) or issubclass(
        rref_type, torch._C.ScriptModule
    )
    if not bypass_type:
        func = getattr(rref_type, func_name)
        if hasattr(func, "_wrapped_async_rpc_function"):
            _invoke_func = _local_invoke_async_execution

    return rpc_api(
        rref.owner(),
        _invoke_func,
        args=(rref, func_name, args, kwargs),
        timeout=timeout
    )

# This class manages proxied RPC API calls for RRefs. It is entirely used from
# C++ (see python_rpc_handler.cpp).
class RRefProxy:
    def __init__(self, rref, rpc_api, timeout=UNSET_RPC_TIMEOUT):
        self.rref = rref
        self.rpc_api = rpc_api
        self.rpc_timeout = timeout

    def __getattr__(self, func_name):
        return partial(_invoke_rpc, self.rref, self.rpc_api, func_name, self.rpc_timeout)
