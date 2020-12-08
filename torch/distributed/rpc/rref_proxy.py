from functools import partial

from . import functions

import torch

def _local_invoke(rref, func_name, args, kwargs):
    return getattr(rref.local_value(), func_name)(*args, **kwargs)

@functions.async_execution
def _local_invoke_async_execution(rref, func_name, args, kwargs):
    return getattr(rref.local_value(), func_name)(*args, **kwargs)

def _invoke_rpc(rref, rpc_api, func_name, *args, **kwargs):
    rref_type = rref._get_type()

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
        args=(rref, func_name, args, kwargs)
    )


class RRefProxy:
    def __init__(self, rref, rpc_api):
        self.rref = rref
        self.rpc_api = rpc_api

    def __getattr__(self, func_name):
        return partial(_invoke_rpc, self.rref, self.rpc_api, func_name)
