import enum
from functools import partial

from .api import rpc_sync, rpc_async, remote, get_worker_info


def _local_invoke(rref, func_name, args, kwargs):
    return getattr(rref.local_value(), func_name)(*args, **kwargs)


def _invoke_rpc_sync(rref, rpc_api, func_name, *args, **kwargs):
    return rpc_api(
        rref.owner(),
        _local_invoke,
        args = (rref, func_name, args, kwargs)
    )


class RRefProxy:
    def __init__(self, rref, rpc_api):
        self.rref = rref
        self.rpc_api = rpc_api

    def __getattr__(self, func_name):
        return partial(_invoke_rpc_sync, self.rref, self.rpc_api, func_name)


