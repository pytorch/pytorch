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


def python_3_8_generics_checker(the_obj, the_type):
    origin_type = typing.get_origin(the_type)
    if origin_type is None:
        pass
    elif origin_type is list:
        if isinstance(the_obj, list):
            for el in the_obj:
                # check if nested generics, ex: List[List[str]]
                arg_type = typing.get_args(the_type)[0]  # list has one arg
                arg_origin = typing.get_origin(arg_type)
                if arg_origin:  # processes nested generics, ex: List[List[str]]
                    if not python_3_8_generics_checker(el, arg_type):
                        return False
                elif not isinstance(el, arg_type):
                    return False
        else:
            return False  # the_obj contains wrong inner type for list
    elif origin_type is dict:
        if isinstance(the_obj, dict):
            key_type = typing.get_args(the_type)[0]
            val_type = typing.get_args(the_type)[1]
            for key, val in the_obj.items():
                # check if keys are of right type
                if not isinstance(key, key_type):
                    return False
                val_origin = typing.get_origin(val_type)
                if val_origin:
                    if not python_3_8_generics_checker(val, val_type):
                        return False
                elif not isinstance(val, val_type):
                    return False
        else:
            return False  # the_obj contains wrong inner type for dict
    elif origin_type is Union:  # TODO actually handles Optional Case
        if the_obj is None:  # check before recursion because None is always fine
            return True
        optional_type = typing.get_args(the_type)[0]
        optional_origin = typing.get_origin(optional_type)
        if optional_origin:
            return python_3_8_generics_checker(the_obj, optional_type)
        elif isinstance(the_obj, optional_type):
            return True
        else:
            return False  # the_obj had wrong data type
    elif origin_type is tuple:
        if isinstance(the_obj, tuple):
            arg_types = typing.get_args(the_type)
            if len(the_obj) != len(arg_types):
                return False  # TODO actually figure out what should happen here
            for el, el_type in zip(the_obj, arg_types):
                el_origin = typing.get_origin(el_type)
                if el_origin:
                    if not python_3_8_generics_checker(el, el_type):
                        return False
                elif not isinstance(el, el_type):
                    return False
        else:
            return False  # the_obj had wrong type inside
    return True


def isinstance2(the_obj, the_type) -> bool:

    python_3_version = version_info[1]

    origin_type = typing.get_origin(the_type)
    if origin_type:
        return python_3_8_generics_checker(the_obj, the_type)

    return isinstance(the_obj, the_type)


if not torch._C._jit_init():
    raise RuntimeError("JIT initialization failed")
