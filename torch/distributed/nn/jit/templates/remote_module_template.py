#!/usr/bin/python3

REMOTE_MODULE_TEMPLATE = """from typing import *

import torch
import torch.distributed.rpc as rpc
from torch import Tensor
from torch._jit_internal import Future
from torch.distributed.rpc import RRef
from typing import Tuple


{assign_module_interface_cls}


{jit_script_decorator}
def _remote_forward(
    module_rref: RRef[module_interface_cls], device: str, is_device_map_set: bool, {arg_types}){arrow_and_return_type}:
    module = module_rref.local_value()
    device = torch.device(device)

    if device.type != "cuda":
        return module.forward({args}, {kwargs})

    # FIXME: Remote this line after fixing test_load_di_parts in
    # torch/fb/training_toolkit/applications/sparse_nn/batch_distributed_inference/tests:batch_distributed_inference_test
    # The error is that:
    # 1) If annotation of Tuple[()] is used for out_args and ret, then the test fails because
    # "Variable 'ret' previously has type Tuple[()] but is now being assigned to a value of type Tuple[Tensor]".
    # 2) If annotation of Tuple[Any] is used for out_args and ret, the the test fails because
    # "Variable 'ret' is annotated with type Tuple[Any] but is being assigned to a value of type Tuple[()]".
    # 3) If there is no annotation, for out_args and ret, the the test fails because
    # "Variable 'ret' previously has type Tuple[()] but is now being assigned to a value of type Tuple[Tensor]".
    return module.forward({args}, {kwargs})

    # If the module is on a cuda device,
    # move any CPU tensor in args or kwargs to the same cuda device.
    # Since torch script does not support generator expression,
    # have to use concatenation instead of
    # ``tuple(i.to(device) if isinstance(i, Tensor) else i for i in *args)``.
    args = ({args},)
    out_args: Tuple[()] = ()
    for arg in args:
        arg = (arg.to(device),) if isinstance(arg, Tensor) else (arg,)
        out_args = out_args + arg

    kwargs = {{{kwargs}}}
    for k, v in kwargs.items():
        if isinstance(v, Tensor):
            kwargs[k] = kwargs[k].to(device)

    if is_device_map_set:
        return module.forward(*out_args, {kwargs})

    # If the device map is empty, then only CPU tensors are allowed to send over wire,
    # so have to move any GPU tensor to CPU in the output.
    # Since torch script does not support generator expression,
    # have to use concatenation instead of
    # ``tuple(i.cpu() if isinstance(i, Tensor) else i for i in module.forward(*out_args, {kwargs}))``.
    ret: Tuple[()] = ()
    for i in module.forward(*out_args, {kwargs}):
        i = (i.cpu(),) if isinstance(i, Tensor) else (i,)
        ret = ret + i
    return ret


def forward_async(self, {arg_types}){arrow_and_future_return_type}:
    args = (self.module_rref, self.device, self.is_device_map_set, {args})
    kwargs = {{{kwargs}}}
    return rpc.rpc_async(
        self.module_rref.owner(),
        _remote_forward,
        args,
        kwargs,
    )


def forward(self, {arg_types}){arrow_and_return_type}:
    args = (self.module_rref, self.device, self.is_device_map_set, {args})
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
