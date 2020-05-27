#!/usr/bin/python3

REMOTE_MODULE_TEMPLATE = """from typing import Tuple, Any

import torch
import torch.distributed.rpc as rpc
from torch import Tensor, nn
from torch._jit_internal import Future, RRef


{module_interface_cls}


{jit_script_decorator}
def _remote_forward(module_rref: RRef[module_interface_cls], {arg_types}){arrow_and_return_type}:  # noqa
    module = module_rref.local_value()
    return module.forward({args}, {kwargs})


class _RemoteModule(nn.Module):  # noqa
    module_rref: RRef[module_interface_cls]  # noqa
    # Users reply on this field to know if this generated RemoteModule is TorchScript-able.
    is_scriptable: bool
    # Users can specify a unique name for the generated RemoteModule.
    # For example, a RemoteModule represents a shard of a EmbeddingBag.
    # In the case of re-sharding after resuming from a checkpoint.
    # The shard can be referenced using this unique name.
    global_unique_name: str

    def __init__(self, module_rref, is_scriptable, global_unique_name):
        super().__init__()
        self.module_rref = module_rref
        self.is_scriptable = is_scriptable
        self.global_unique_name = global_unique_name

    {jit_export_decorator}
    def forward_async(self, {arg_types}){arrow_and_future_return_type}:  # noqa
        args = (self.module_rref, {args})
        kwargs = {{{kwargs}}}
        return rpc.rpc_async(
            self.module_rref.owner(),
            _remote_forward,
            args,
            kwargs,
        )

    {jit_export_decorator}
    def forward(self, {arg_types}){arrow_and_return_type}:  # noqa
        args = (self.module_rref, {args})
        kwargs = {{{kwargs}}}
        ret_fut = rpc.rpc_async(
            self.module_rref.owner(),
            _remote_forward,
            args,
            kwargs,
        )
        return ret_fut.wait()


{jit_interface_decorator}
class _RemoteModuleInterface:
    def forward_async(self, {arg_types}){arrow_and_future_return_type}:  # noqa
        pass

    def forward(self, {arg_types}){arrow_and_return_type}:  # noqa
        pass
"""
