#!/usr/bin/python3

REMOTE_MODULE_TEMPLATE = """from typing import *

import torch
import torch.distributed.rpc as rpc
from torch import Tensor
from torch._jit_internal import Future
from torch.distributed.rpc import RRef
from typing import List, Tuple, Any


{assign_module_interface_cls}


{jit_script_decorator}
# WARNING: If the module is on a cuda device, any CPU tensor stored in a nested ``arg`` or `kwargs``,
# will not be implicitly moved to the same cuda device.
def _remote_forward(
    module_rref: RRef[module_interface_cls], device: str, {arg_types}){arrow_and_return_type}:
    module = module_rref.local_value()
    device = torch.device(device)

    return module.forward({args}, {kwargs})

    if device.type != "cuda":
        return module.forward({args}, {kwargs})

    # If the module is on a cuda device,
    # move any CPU tensor in args or kwargs to the same cuda device.
    # Since torch script does not support generator expression,
    # have to use concatenation instead of
    # ``list(i.to(device) if isinstance(i, Tensor) else i for i in *args)``.
    args = ({args},)
    out_args: List[Any] = []
    for arg in args:
        out_args.append((arg.to(device),) if isinstance(arg, Tensor) else (arg,))

    kwargs = {{{kwargs}}}
    for k, v in kwargs.items():
        if isinstance(v, Tensor):
            kwargs[k] = kwargs[k].to(device)

    # Since only CPU tensors are allowed to send over wire,
    # need to move any GPU tensor to CPU in the output.
    # Since torch script does not support generator expression,
    # have to use concatenation instead of
    # ``tuple(i.cpu() if isinstance(i, Tensor) else i for i in module.forward(*out_args, {kwargs}))``.
    # TODO: Once process group RPC backend is deprecated,
    # and a device map is explicitly provided to TensorPipe backend,
    # we can leave the forward output on CUDA device and avoid the post-processing here.
    ret: Tuple[Any] = ()
    for i in module.forward(*out_args, {kwargs}):
        i = (i.cpu(),) if isinstance(i, Tensor) else (i,)
        ret = ret + i
    return ret


def forward_async(self, {arg_types}){arrow_and_future_return_type}:
    args = (self.module_rref, self.device, {args})
    kwargs = {{{kwargs}}}
    return rpc.rpc_async(
        self.module_rref.owner(),
        _remote_forward,
        args,
        kwargs,
    )


def forward(self, {arg_types}){arrow_and_return_type}:
    args = (self.module_rref, self.device, {args})
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
