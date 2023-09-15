import builtins
import collections
import copy
import functools
import inspect
import itertools
import math
import operator
import types
import warnings

from typing import cast, Dict, Optional, Set

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


import torch
import torch._functorch.deprecated as deprecated_func
from torch.fx._symbolic_trace import is_fx_tracing

from . import config
from .external_utils import is_compiling
from .utils import is_safe_constant, NP_SUPPORTED_MODULES

"""
A note on allowed functions:

Dynamo consults this file to determine if a particular function/module
is allowed to appear as a node in its fx output.

If a function is disallowed, it may either be traced-through, or skipped.

Trace-through means dynamo will continue to trace the interior code for
the function/module rather than stopping at its boundary and recording it
as a node in the fx graph. Whether tracing through or allowing, the functionality
of the function/module is part of the dynamo graph.  Caveat: if tracing through,
any interior operation could trigger its own graph-break.

Skips are determined by (torch/_dynamo/skipfiles.py) - see "a note on
skipfiles" there.
"""


def make_function_id_set(lazy_initializer):
    """
    Track a set of `id()`s of objects which are either allowed or not
    allowed to go into the generated FX graph.  Use to test for torch.*,
    numpy.*, builtins.*, etc.

    Support user modification to permit customization of what can be
    added to the graph and what will cause a graph break.
    """

    class FunctionIdSet:
        function_ids: Optional[Set[int]] = None
        function_names: Optional[Dict[int, str]] = None

        def __call__(self):
            if self.function_ids is None:
                value = lazy_initializer()
                if isinstance(value, dict):
                    self.function_ids = set(value.keys())
                    self.function_names = value
                else:
                    assert isinstance(value, set)
                    self.function_ids = value
            return self.function_ids

        def get_name(self, idx: int, default: str):
            self()  # lazy init
            return self.function_names.get(idx, default)

        def add(self, idx: int):
            self()  # lazy init
            self.function_ids.add(idx)

        def remove(self, idx: int):
            if idx in self():
                self.function_ids.remove(idx)

        def __contains__(self, idx: int):
            return idx in self()

    return FunctionIdSet()


@make_function_id_set
def _disallowed_function_ids():
    remove = [
        True,
        False,
        None,
        collections.OrderedDict,
        copy.copy,
        copy.deepcopy,
        inspect.signature,
        math.__package__,
        torch.__builtins__,
        torch.autocast_decrement_nesting,
        torch.autocast_increment_nesting,
        torch.autograd.grad,
        torch.clear_autocast_cache,
        torch.cuda.current_device,
        torch.cuda.set_device,
        torch.distributions.constraints.is_dependent,
        torch.distributions.normal.Normal,
        torch.inference_mode,
        torch.jit.isinstance,
        torch.set_anomaly_enabled,
        torch.set_autocast_cache_enabled,
        torch.set_autocast_cpu_dtype,
        torch.set_autocast_cpu_enabled,
        torch.set_autocast_enabled,
        torch.set_autocast_gpu_dtype,
        warnings.warn,
        torch._C._dynamo.eval_frame.unsupported,
        torch.Tensor.__init__,
    ]
    if torch.distributed.is_available():
        from torch.distributed import _functional_collectives

        config.skipfiles_inline_module_allowlist.add(_functional_collectives)

    # extract all dtypes from torch
    dtypes = [
        obj for obj in torch.__dict__.values() if isinstance(obj, type(torch.float32))
    ]
    remove += dtypes
    storage = [
        obj
        for obj in torch.__dict__.values()
        if isinstance(obj, type(torch.FloatStorage))
    ]
    remove += storage

    # Distributed APIs don't work well with torch.compile.
    if torch.distributed.is_available():
        remove.extend(
            torch.distributed.distributed_c10d.dynamo_unsupported_distributed_c10d_ops
        )

    return {id(x) for x in remove}


@make_function_id_set
def _allowed_function_ids():
    """
    Walk torch.* and get the ids of all the stuff in it
    """
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed")
    torch_object_ids = dict()

    def _is_allowed_module_prefix(obj):
        allowed_modules = ("torch", "math")
        # torch.nn.modules.rnn is disallowed because these modules internally
        # flatten their parameters.  This flattening process will call
        # Tensor.set_ with a Storage, and Storages cannot be traced with
        # AOTAutograd; so we need to graph-break. To ensure this, we inline
        # these functions, rather than keep them opaque-ly in the graph.
        disallowed_modules = (
            "torch.optim.",
            "torch.utils._foreach_utils",  # omit the period so we match all the functions in this module
            "torch.nn.modules.rnn.",
            "torch._dynamo.",
            "torch._C._dynamo.",
            "torch._inductor.",
            "torch._C.inductor.",
            "torch.fx.",
            "torch.distributed.fsdp.",
            "torch.distributed._tensor.",
        )
        allowed_modules_dot = tuple([x + "." for x in allowed_modules])
        module = inspect.getmodule(obj)
        if module is None:
            return False

        mod_name = module.__name__

        if any(mod_name.startswith(m) for m in disallowed_modules):
            return False

        return mod_name in allowed_modules or mod_name.startswith(allowed_modules_dot)

    def _find_torch_objects(module):
        if any(
            module.__name__.startswith(mod_name)
            for mod_name in config.allowed_functions_module_string_ignorelist
        ):
            return
        torch_object_ids[id(module)] = module.__name__
        for name, obj in list(module.__dict__.items()):
            if id(obj) not in torch_object_ids:
                # Dynamo allows all builtins into the graph and does not attempt
                # to introspect into them. We don't want to allow instances of
                # HigherOrderOperator into the graph all the time (Dynamo needs
                # to introspect the body functions of these HigherOrderOperator
                # first, decide they are safe, and then allow them into the graph).
                # So we exclude HigherOrderOperator from being a builtin.
                import torch._ops

                if isinstance(obj, torch._ops.HigherOrderOperator):
                    continue

                # We want to trace through `grad` and `vmap`
                if obj in (
                    torch.func.grad,
                    deprecated_func.grad,
                    torch.func.vmap,
                    deprecated_func.vmap,
                ):
                    continue

                if isinstance(obj, types.ModuleType):
                    if obj.__name__.startswith("torch.") and _is_allowed_module_prefix(
                        obj
                    ):
                        torch_object_ids[id(obj)] = f"{module.__name__}.{name}"
                        _find_torch_objects(obj)
                elif _is_allowed_module_prefix(obj):
                    torch_object_ids[id(obj)] = f"{module.__name__}.{name}"
                elif inspect.getmodule(obj) is None and not is_safe_constant(obj):
                    torch_object_ids[id(obj)] = f"{module.__name__}.{name}"

    _find_torch_objects(torch)
    _find_torch_objects(math)

    # torch.Tensor.{fn}
    for name in dir(torch.Tensor):
        method = getattr(torch.Tensor, name)
        if isinstance(
            method, (types.MethodDescriptorType, types.WrapperDescriptorType)
        ):
            torch_object_ids[id(method)] = f"torch.Tensor.{name}"

    for idx in _disallowed_function_ids():
        if idx in torch_object_ids:
            del torch_object_ids[idx]

    for extra in (is_fx_tracing, is_compiling):
        torch_object_ids[id(extra)] = f"{extra.__module__}.{extra.__name__}"

    return torch_object_ids


@make_function_id_set
def _allowed_user_defined_function_ids():
    rv = {}
    return rv


@make_function_id_set
def _builtin_function_ids():
    rv = {
        id(v): f"builtins.{k}"
        for k, v in builtins.__dict__.items()
        if not k.startswith("_") and callable(v)
    }
    rv.update(
        {
            id(v): f"operator.{k}"
            for k, v in operator.__dict__.items()
            if not k.startswith("_") and callable(v)
        }
    )
    rv.update(
        {id(v): f"functools.{v.__name__}" for v in (itertools.chain, itertools.islice)}
    )
    rv.update({id(cast): "typing.cast"})
    rv[id(functools.reduce)] = "functools.reduce"
    return rv


@make_function_id_set
def _numpy_function_ids():
    rv = dict()
    for mod in NP_SUPPORTED_MODULES:
        rv.update(
            {
                id(v): f"{mod.__name__}.{k}"
                for k, v in mod.__dict__.items()
                if callable(v)
                and (getattr(v, "__module__", None) or mod.__name__) == mod.__name__
            }
        )
    return rv


@make_function_id_set
def _builtin_constant_ids():
    """
    Collects constant builtins by eliminating callable items.
    """
    rv = {
        id(v): f"builtins.{k}"
        for k, v in builtins.__dict__.items()
        if not k.startswith("_") and not callable(v)
    }
    return rv


def is_allowed(obj):
    """Is this safe to trace like torch.add ?"""
    # torch.ops is populated lazily so we don't necessarily have them in
    # _allowed_function_ids.  Figure it out by testing the type instead
    # in those cases
    if id(obj) in _disallowed_function_ids:
        return False

    return id(obj) in _allowed_function_ids or isinstance(
        obj,
        (torch._ops.OpOverloadPacket, torch._ops.OpOverload, torch._ops._OpNamespace),
    )


def is_user_defined_allowed(obj):
    return id(obj) in _allowed_user_defined_function_ids


def torch_get_name(obj, default):
    """Convert a torch.* function to a string"""
    return _allowed_function_ids.get_name(id(obj), default)


def is_builtin_callable(obj):
    return id(obj) in _builtin_function_ids


def is_builtin_constant(obj):
    return id(obj) in _builtin_constant_ids


def is_numpy(obj):
    if np is None:
        return False
    return isinstance(obj, np.ndarray) or id(obj) in _numpy_function_ids
