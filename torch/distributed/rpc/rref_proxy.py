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
    # if rpc_sync, allow get_type to block, else, add invoke logic as a then()
    # callback.
    block_on_type = rpc_api in [torch.distributed.rpc.rpc_sync, torch.distributed.rpc.remote]
    rref_type = rref._get_type(timeout=timeout, blocking=block_on_type)
    # rref_type = rref_type.wait()
    should_wait_on_rref_type = isinstance(rref_type, torch._C.Future)
    # Helper function to allow invoke to be run as a chained callback instead of
    # inline.

    def invoke_on_owner(rref_type):
        _invoke_func = _local_invoke
        if should_wait_on_rref_type:
            try:
                rref_type = rref_type.wait()
            except BaseException as err:
                # Future corresponding to rref type had an error, propagate
                # error instead of issuing RPC.
                ret = torch.futures.Future()
                ret.set_exception(err)
                return ret

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

    # If we don't need to wait on the rref type (i.e. was cached) or are not
    # running rpc_async, invoke the RPC API inline
    if not should_wait_on_rref_type or block_on_type:
        return invoke_on_owner(rref_type)
    else:
        # Create a future whose result is the result of the above rpc_api()
        # future. We don't simply return rref_type.then(cb) since the user would
        # have to call wait() twice to access the future result.
        ret_fut = torch.futures.Future()

        def set_result_from_rpc(rpc_fut):
            # Propagate RPC result (or error) to ret_fut.
            try:
                rpc_result = rpc_fut.wait()
                ret_fut.set_result(rpc_result)
            except BaseException as err:
                ret_fut.set_exception(err)
                assert ret_fut.done()

        def cb(rref_type_fut):
            rpc_fut = invoke_on_owner(rref_type_fut)
            rpc_fut.add_done_callback(set_result_from_rpc)

        rref_type.add_done_callback(cb)
        return ret_fut


# This class manages proxied RPC API calls for RRefs. It is entirely used from
# C++ (see python_rpc_handler.cpp).
class RRefProxy:
    def __init__(self, rref, rpc_api, timeout=UNSET_RPC_TIMEOUT):
        self.rref = rref
        self.rpc_api = rpc_api
        self.rpc_timeout = timeout

    def __getattr__(self, func_name):
        return partial(_invoke_rpc, self.rref, self.rpc_api, func_name, self.rpc_timeout)
