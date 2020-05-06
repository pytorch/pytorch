#!/usr/bin/python3
import uuid
from typing import Any, Callable, Dict, Tuple

import torch
import torch.distributed.rpc as rpc
from torch.distributed.nn.jit import instantiator


def _gen_global_unique_name():
    return f"{uuid.uuid4().hex}"


def _instantiate_template(module_interface_cls, is_scriptable):
    # Instantiate _RemoteModule class template on the local side.
    generated_module = instantiator.instantiate_remote_module_template(
        module_interface_cls, is_scriptable
    )

    remote_module_cls = generated_module._RemoteModule
    remote_module_interface_cls = generated_module._RemoteModuleInterface

    return remote_module_cls, remote_module_interface_cls


def _script_module_creator_wrapper(module_creator, module_interface_cls, args, kwargs):
    module = module_creator(*args, **kwargs)
    script_module = torch.jit.script(module)
    return rpc.RRef(script_module, module_interface_cls)


def RemoteModule(
    to: str,
    module_creator: Callable,
    args: Tuple = None,
    kwargs: Dict[str, Any] = None,
    global_unique_name: str = None,
    module_interface_cls: Any = None,
):
    """
        The returned RemoteModule instance will have auto-generated
        forward_async / forward methods, check the implementation in
        remote_module.py.template,
        the arguments they take are the same as the forward method
        of the module created by the module_creator.

    Args:
        to: str.
        module_creator: Callable. The return type of the callable
            must be a class decorated with @torch.jit.interface.
    """
    # Sanity checks.
    assert rpc._is_current_rpc_agent_set(), "RemoteModule only works in RPC."

    # Default arguments preperation.
    args = args if args is not None else ()
    kwargs = kwargs if kwargs is not None else {}
    global_unique_name = (  # Assign a global name for the module to be created.
        global_unique_name
        if global_unique_name is not None
        else _gen_global_unique_name()
    )

    # Infer module_interface type.
    module_interface_cls = instantiator.infer_module_interface_cls(
        module_creator, module_interface_cls
    )
    is_scriptable = getattr(module_interface_cls, "__torch_script_interface__", False)

    # Instantiate template on remote side.
    fut = rpc.rpc_async(
        to, _instantiate_template, (module_interface_cls, is_scriptable)
    )

    # Instantiate template on local side.
    remote_module_cls, _remote_module_interface_cls = _instantiate_template(module_interface_cls, is_scriptable)

    # Create the module on the remote side.
    if is_scriptable:
        module_rref = rpc.rpc_sync(
            to,
            _script_module_creator_wrapper,
            (module_creator, module_interface_cls, args, kwargs),
        )
    else:
        module_rref = rpc.remote(to, module_creator, args, kwargs)

    # Create remote_module_cls instance on local side.
    remote_module = remote_module_cls(module_rref, is_scriptable, global_unique_name)

    fut.wait()
    return remote_module
