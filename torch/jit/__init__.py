import torch._C
from typing import List, Dict, Optional, Tuple, Union
import typing
from sys import version_info

from torch.utils import set_module

# These are imported so users can access them from the `torch.jit` module
from torch._jit_internal import (
    Final,
    Future,
    _overload,
    _overload_method,
    ignore,
    is_scripting,
    export,
    unused,
)
from torch.jit._script import (
    script,
    Attribute,
    ScriptModule,
    script_method,
    RecursiveScriptModule,
    ScriptWarning,
    interface,
    CompilationUnit,
    ScriptFunction,
    _unwrap_optional,
)
from torch.jit._trace import (
    trace,
    trace_module,
    TracedModule,
    TracerWarning,
    TracingCheckError,
    is_tracing,
    ONNXTracedModule,
    TopLevelTracedModule,
    _unique_state_dict,
    _flatten,
    _script_if_tracing,
    _get_trace_graph,
)
from torch.jit._async import fork, wait
from torch.jit._serialization import save, load
from torch.jit._fuser import optimized_execution, fuser, last_executed_optimized_graph

from torch.jit._freeze import freeze

# For backwards compatibility
_fork = fork
_wait = wait


def export_opnames(m):
    r"""
        Returns a list of operator names of a script module and its submodules
    """
    return torch._C._export_opnames(m._c)


# torch.jit.Error
Error = torch._C.JITException
set_module(Error, "torch.jit")
# This is not perfect but works in common cases
Error.__name__ = "Error"
Error.__qualname__ = "Error"

# for use in python if using annotate
def annotate(the_type, the_value):
    # noop in python
    return the_value


def get_origin(the_type):
    return getattr(the_type, "__origin__", None)


def get_args(the_type):
    return getattr(the_type, "__args__", None)


def generics_checker(the_obj, the_type):
    origin_type = get_origin(the_type)
    if origin_type is None:
        pass
    elif origin_type is list or origin_type is typing.List:
        if isinstance(the_obj, list):
            for el in the_obj:
                # check if nested generics, ex: List[List[str]]
                arg_type = get_args(the_type)[0]
                arg_origin = get_origin(arg_type)
                if arg_origin:  # processes nested generics, ex: List[List[str]]
                    if not generics_checker(el, arg_type):
                        return False
                elif not isinstance(el, arg_type):
                    return False
        else:
            return False
    elif origin_type is dict or origin_type is typing.Dict:
        if isinstance(the_obj, dict):
            key_type = get_args(the_type)[0]
            val_type = get_args(the_type)[1]
            for key, val in the_obj.items():
                # check if keys are of right type
                if not isinstance(key, key_type):
                    return False
                val_origin = get_origin(val_type)
                if val_origin:
                    if not generics_checker(val, val_type):
                        return False
                elif not isinstance(val, val_type):
                    return False
        else:
            return False
    elif origin_type is Union:  # TODO actually handles Optional Case
        if the_obj is None:  # check before recursion because None is always fine
            return True
        optional_type = get_args(the_type)[0]
        optional_origin = get_origin(optional_type)
        if optional_origin:
            return generics_checker(the_obj, optional_type)
        elif isinstance(the_obj, optional_type):
            return True
        else:
            return False
    elif origin_type is tuple or typing.Tuple:
        if isinstance(the_obj, tuple):
            arg_types = get_args(the_type)
            if len(the_obj) != len(arg_types):
                return False  # TODO actually figure out what should happen here
            for el, el_type in zip(the_obj, arg_types):
                el_origin = get_origin(el_type)
                if el_origin:
                    if not generics_checker(el, el_type):
                        return False
                elif not isinstance(el, el_type):
                    return False
        else:
            return False 
    return True


def isinstance2(the_obj, the_type) -> bool:
    origin_type = get_origin(the_type)
    if origin_type:
        return generics_checker(the_obj, the_type)
    # handle non-generics
    return isinstance(the_obj, the_type)


if not torch._C._jit_init():
    raise RuntimeError("JIT initialization failed")
