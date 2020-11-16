#!/usr/bin/python3

REMOTE_MODULE_TEMPLATE = """from typing import *

import torch
import torch.distributed.rpc as rpc
from torch import Tensor
from torch._jit_internal import Future
from torch.distributed.rpc import RRef


{assign_module_interface_cls}


{jit_script_decorator}
def _remote_forward(module_rref: RRef[module_interface_cls], {arg_types}){arrow_and_return_type}:  # noqa
    module = module_rref.local_value()
    return module.forward({args}, {kwargs})


def forward_async(self, {arg_types}){arrow_and_future_return_type}:  # noqa
    args = (self.module_rref, {args})
    kwargs = {{{kwargs}}}
    return rpc.rpc_async(
        self.module_rref.owner(),
        _remote_forward,
        args,
        kwargs,
    )


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


_generated_methods = [
    forward_async,
    forward,
]
"""
