# Copyright (c) Meta Platforms, Inc. and affiliates

import collections
import importlib
import types
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh


_original_functions: Dict[Callable, Callable] = {}
_wrapper_functions: Dict[Callable, Callable] = {}
_replaced_objs: collections.defaultdict[
    Callable, Set[Tuple[types.ModuleType, str]]
] = collections.defaultdict(set)


def distribute_function(
    fn: Callable,
    fn_module: types.ModuleType,
    fn_callers: List[nn.Module],
    device_mesh: DeviceMesh,
    input_fn: Optional[Callable] = None,
    output_fn: Optional[Callable] = None,
) -> None:
    """
    ``distribute_function`` is an experimental API that allows users to "distribute"
    the inputs and outputs of a function. Similar to ``distribute_module``, this API
    installs hooks to the ``fn`` to convert the inputs and outputs. There are two
    major differences between ``distribute_function`` and ``distribute_module``.
    First, a function does not have parammeters and buffers, as a result,
    ``distribute_function`` itself won't convert any tensors but simply install the
    input and output hooks.  The tnesor conversion will happen in the hooks.
    Another difference is an nn.Module subclass can have several instances and each
    instance be fed into ``distribute_module`` independently with affecting other
    instance. On the other hand, function is a singleton object. So if a function
    is distributed by ``distribute_function`` all subsequent calls to the function
    will invoke the installed hooks.

    Args:
        fn (Callable): the function to be distributed.
        fn_module (types.ModuleType): the Python module that the function is declared.
            e.g., if ``fn`` is ``torch.nn.functional.scaled_dot_product_attention``,
            ``fn_module`` is ``torch.nn.functional``.
        fn_callers (nn.Module): the nn.Module that calls ``fn``.
        device_mesh (:class:`DeviceMesh`): the device mesh that will be used by the
            input and output hooks to distribute the tensors.
        input_fn (Optioinal[Callable]): the hook to distribute or convert the input
            arguments of ``fn``.
        output_fn (Optioinal[Callable]): the hook to distribute or convert the output
            arguments of ``fn``.
    """

    def wrapper(target_fn, input_fn, output_fn):
        def inner_fn(*args, **kwargs):
            if input_fn is not None:
                args, kwargs = input_fn(device_mesh, *args, **kwargs)
            output = target_fn(*args, **kwargs)
            if output_fn is not None:
                output = output_fn(device_mesh, output)
            return output

        return inner_fn

    def setattr_(module, obj_name, obj, new_obj):
        setattr(module, obj_name, new_obj)
        global _replaced_objs
        _replaced_objs[obj].add((module, obj_name))

    global _original_functions
    global _wrapper_functions
    if fn in _original_functions:
        wrapper_func = _original_functions[fn]
        original_func = fn
    elif fn in _wrapper_functions:
        wrapper_func = fn
        original_func = _wrapper_functions[fn]
    else:
        original_func = fn
        wrapper_func = wrapper(fn, input_fn, output_fn)
        setattr_(fn_module, fn.__name__, fn, wrapper_func)

    for nn_module in fn_callers:
        fn_caller_module = importlib.import_module(nn_module.__module__)
        for obj_name in dir(fn_caller_module):
            obj = getattr(fn_caller_module, obj_name)
            if obj == original_func:
                setattr_(fn_caller_module, obj_name, obj, wrapper_func)
