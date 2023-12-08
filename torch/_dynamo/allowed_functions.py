import builtins
import collections
import copy
import dataclasses
import functools
import inspect
import itertools
import math
import operator
import sys
import types
import warnings

from collections import defaultdict
from typing import Any, Callable, cast, Dict, List, Optional, Set, Union

np: Optional[types.ModuleType] = None
try:
    import numpy as np
except ModuleNotFoundError:
    pass


import torch
import torch._functorch.deprecated as deprecated_func
from torch.fx._symbolic_trace import is_fx_tracing

from . import config
from .external_utils import is_compiling
from .utils import hashable, is_safe_constant, NP_SUPPORTED_MODULES

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


class FunctionIdSet:
    """
    Track a set of `id()`s of objects which are either allowed or not
    allowed to go into the generated FX graph.  Use to test for torch.*,
    numpy.*, builtins.*, etc.

    Support user modification to permit customization of what can be
    added to the graph and what will cause a graph break.
    """

    function_ids: Optional[Set[int]] = None
    function_names: Optional[Dict[int, str]] = None

    def __init__(self, lazy_initializer: Callable[[], Union[Dict[int, str], Set[int]]]):
        self.lazy_initializer = lazy_initializer

    def __call__(self):
        if self.function_ids is None:
            value = self.lazy_initializer()
            if isinstance(value, dict):
                self.function_ids = set(value.keys())
                self.function_names = value
            else:
                assert isinstance(value, set)
                self.function_ids = value
        return self.function_ids

    def get_name(self, idx: int, default: str):
        self()  # lazy init
        assert self.function_names is not None
        return self.function_names.get(idx, default)

    def add(self, idx: int):
        function_ids = self()  # lazy init
        function_ids.add(idx)

    def remove(self, idx: int):
        function_ids = self()
        if idx in function_ids:
            function_ids.remove(idx)

    def __contains__(self, idx: int):
        return idx in self()


@FunctionIdSet
def _disallowed_function_ids() -> Set[int]:
    remove: List[Any] = [
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
        torch.resize_as_,
        torch._tensor._convert,
    ]

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


# Helper function to dump the torch name rule map generated based on
# the heuristic defined in gen_allowed_objs_and_ids.
def dump_allowed_torch_name_rule_map() -> None:
    m = gen_allowed_objs_and_ids(record=True, c_binding_only=False).name_rule_map
    for k, v in m.items():
        print(f'"{k}": {v.__name__},')


@dataclasses.dataclass
class AllowedObjects:
    """
    Track the objects, object id - name pairs, and name - dynamo wrapping rule pairs
    from the heuristic defined in `gen_allowed_objs_and_ids`.
    TODO: Remove the overalp/duplication between these fields
    after allowed_functions refactor is done.
    """

    object_ids: Dict[int, str]
    ctx_mamager_classes: Set[Any]
    c_binding_in_graph_functions: Set[Any]
    non_c_binding_in_graph_functions: Set[Any]
    name_rule_map: Dict[str, Any]


def gen_allowed_objs_and_ids(record=False, c_binding_only=True) -> AllowedObjects:
    """
    Walk torch.* and get the ids of all the stuff in it
    """
    from .variables import TorchCtxManagerClassVariable, TorchInGraphFunctionVariable

    warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed")
    torch_object_ids = dict()
    ctx_mamager_classes = set()
    c_binding_in_graph_functions = set()
    non_c_binding_in_graph_functions = set()
    torch_name_rule_map = dict()

    # Add obj to ctx_mamager_classes set if it's a torch context manager class.
    # This is used to generate the ctx manager class list based on heuristic.
    def heuristic_record_if_ctx_manager(obj, module, name):
        if (
            issubclass(type(obj), type)
            and hasattr(obj, "__enter__")
            and hasattr(obj, "__exit__")
        ):
            torch_name_rule_map[
                f"{module.__name__}.{name}"
            ] = TorchCtxManagerClassVariable
            ctx_mamager_classes.add(obj)

    # In some platforms, these functions were loaded as classes instead of functions.
    # To mitigate these weired cases, we need this special check.
    def is_special_functions(obj):
        return hashable(obj) and obj in {
            torch._C._cuda_isCurrentStreamCapturing,
            torch._C._graph_pool_handle,
        }

    # Add obj to c_binding_in_graph_functions set or non_c_binding_in_graph_functions set
    # if it's a torch function or method.
    # This is used to generate the in graph function list based on heuristic.
    def heuristic_record_if_in_graph_function(obj, module, name):
        try:
            if hasattr(obj, "__wrapped__"):
                obj = obj.__wrapped__
        except Exception:
            pass
        if isinstance(
            obj,
            (
                types.FunctionType,
                types.MethodType,
                types.BuiltinFunctionType,
                types.MethodDescriptorType,
                types.WrapperDescriptorType,
            ),
        ) or is_special_functions(obj):
            torch_name_rule_map[
                f"{module.__name__}.{name}"
            ] = TorchInGraphFunctionVariable
            if c_binding_only:
                if not hasattr(obj, "__code__"):
                    c_binding_in_graph_functions.add(obj)
            else:
                if hasattr(obj, "__code__"):
                    non_c_binding_in_graph_functions.add(obj)
                else:
                    c_binding_in_graph_functions.add(obj)

    def _is_allowed_module_prefix(obj):
        allowed_modules = ("torch", "math")
        # torch.nn.modules.rnn is disallowed because these modules internally
        # flatten their parameters.  This flattening process will call
        # Tensor.set_ with a Storage, and Storages cannot be traced with
        # AOTAutograd; so we need to graph-break. To ensure this, we inline
        # these functions, rather than keep them opaque-ly in the graph.
        disallowed_modules = [
            "torch.optim",
            "torch.nn.modules.rnn",
            "torch._dynamo",
            "torch._C._dynamo",
            "torch._inductor",
            "torch._C.inductor",
            "torch.fx",
            "torch._C._autograd",
            "torch._C._cudart",
            "torch._C._distributed_autograd",
            "torch._C._distributed_c10d",
            "torch._C._distributed_rpc",
            "torch._C._functorch",
            "torch._C._monitor",
            "torch._C._nvtx",
            "torch._C._lazy",
            "torch._C._profiler",
            "torch.__config__",
            "torch._custom_op",
            "torch._dispatch",
            "torch._export",
            "torch._functorch.make_functional",
            "torch._functorch.compile_utils",
            "torch._functorch.partitioners",
            "torch._functorch.aot_autograd",
            "torch._functorch.compilers",
            "torch._functorch.fx_minifier",
            "torch.autograd.profiler_util",
            "torch.autograd.profiler",
            "torch._jit_internal",
            "torch._library",
            "torch._lobpcg",
            "torch._logging",
            "torch._meta_registrations",
            "torch._namedtensor_internals",
            "torch._numpy",
            "torch._sources",
            "torch._subclasses",
            "torch._tensor",
            "torch._tensor_str",
            "torch._utils",
            "torch._utils_internal",
            "torch._vmap_internals",
            "torch.compiler",
            "torch.distributed",
            "torch.export",
            "torch.hub",
            "torch.jit",
            "torch.library",
            "torch.masked.maskedtensor",
            "torch.nn.init",
            "torch.nn.modules.module",
            "torch.nn.parallel",
            "torch.nn.utils",
            "torch.multiprocessing",
            "torch.onnx",
            "torch.overrides",
            "torch.package",
            "torch.profiler",
            "torch.serialization",
            "torch.storage",
            "torch.utils",
        ]
        if config.trace_distributed:
            disallowed_modules.append("torch.distributed.")

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
                    torch.nn.functional.triplet_margin_with_distance_loss,
                    torch.cond,
                ):
                    continue

                if isinstance(obj, types.ModuleType):
                    if obj.__name__.startswith("torch.") and _is_allowed_module_prefix(
                        obj
                    ):
                        torch_object_ids[id(obj)] = f"{module.__name__}.{name}"
                        _find_torch_objects(obj)
                elif _is_allowed_module_prefix(obj):
                    if record:
                        heuristic_record_if_ctx_manager(obj, module, name)
                        heuristic_record_if_in_graph_function(obj, module, name)
                    torch_object_ids[id(obj)] = f"{module.__name__}.{name}"
                elif inspect.getmodule(obj) is None and not is_safe_constant(obj):
                    if record:
                        heuristic_record_if_ctx_manager(obj, module, name)
                        heuristic_record_if_in_graph_function(obj, module, name)
                    torch_object_ids[id(obj)] = f"{module.__name__}.{name}"

    _find_torch_objects(torch)
    _find_torch_objects(math)

    if config.trace_distributed:
        from torch.distributed import _functional_collectives_impl as fci

        for f in [
            fci._all_gather_into_tensor,
            fci._all_reduce,
            fci._reduce_scatter_tensor,
            fci._all_reduce_coalesced,
            fci._all_gather_into_tensor_coalesced,
            fci._reduce_scatter_tensor_coalesced,
        ]:
            torch_object_ids[id(f)] = repr(f)

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

    return AllowedObjects(
        torch_object_ids,
        ctx_mamager_classes,
        c_binding_in_graph_functions,
        non_c_binding_in_graph_functions,
        torch_name_rule_map,
    )


@FunctionIdSet
def _allowed_function_ids() -> Dict[int, str]:
    return gen_allowed_objs_and_ids().object_ids


@FunctionIdSet
def _allowed_user_defined_function_ids() -> Dict[int, str]:
    rv: Dict[int, str] = {}
    return rv


@FunctionIdSet
def _builtin_function_ids() -> Dict[int, str]:
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
    rv.update(
        {
            id(cast): "typing.cast",
            id(functools.reduce): "functools.reduce",
            id(copy.deepcopy): "copy.deepcopy",
        }
    )
    return rv


@FunctionIdSet
def _numpy_function_ids() -> Dict[int, str]:
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


@FunctionIdSet
def _builtin_constant_ids() -> Dict[int, str]:
    """
    Collects constant builtins by eliminating callable items.
    """
    rv = {
        id(v): f"builtins.{k}"
        for k, v in builtins.__dict__.items()
        if not k.startswith("_") and not callable(v)
    }
    return rv


_lazy_module_init: Dict[str, List[Callable[[], None]]] = defaultdict(list)


def add_module_init_func(name: str, init_func: Callable[[], None]) -> None:
    """Register a module without eagerly importing it"""
    # If the module is already imported, eagerly run init
    assert "." not in name, f"Expected a root module name, but got {name}"
    if name in sys.modules:
        init_func()

    # Module is not yet imported, delay processing until needed
    assert name not in _lazy_module_init
    _lazy_module_init[name].append(init_func)


def _maybe_init_lazy_module(obj: object) -> None:
    module = getattr(obj, "__module__", None)
    if module is None:
        return

    base_module = module.split(".")[0]
    init_funcs = _lazy_module_init.pop(base_module, None)
    if init_funcs is not None:
        for fn in init_funcs:
            fn()


def is_allowed(obj) -> bool:
    """Is this safe to trace like torch.add ?"""
    _maybe_init_lazy_module(obj)

    if id(obj) in _disallowed_function_ids:
        return False

    if id(obj) in _allowed_function_ids:
        return True

    # torch.ops is populated lazily so we don't necessarily have them in
    # _allowed_function_ids.  Figure it out by testing the type instead
    # in those cases
    return isinstance(
        obj,
        (torch._ops.OpOverloadPacket, torch._ops.OpOverload, torch._ops._OpNamespace),
    )


def is_user_defined_allowed(obj) -> bool:
    _maybe_init_lazy_module(obj)
    return id(obj) in _allowed_user_defined_function_ids


def is_forbidden(obj) -> bool:
    _maybe_init_lazy_module(obj)
    return getattr(obj, "_dynamo_forbidden", False)


def torch_get_name(obj, default) -> str:
    """Convert a torch.* function to a string"""
    return _allowed_function_ids.get_name(id(obj), default)


def is_builtin_callable(obj) -> bool:
    return id(obj) in _builtin_function_ids


def is_builtin_constant(obj) -> bool:
    return id(obj) in _builtin_constant_ids


def is_numpy(obj) -> bool:
    if np is None:
        return False
    return isinstance(obj, (np.ndarray, np.generic)) or id(obj) in _numpy_function_ids
