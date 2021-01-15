import torch._C

from torch.utils import set_module

# These are imported so users can access them from the `torch.jit` module
from torch._jit_internal import (
    Final,
    Future,
    _overload,
    _overload_method,
    ignore,
    _isinstance,
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

from torch.jit.cuda import stream
from torch.jit._freeze import freeze, optimize_frozen_module

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


def script_if_tracing(fn):
    """
    Compiles ``fn`` when it is first called during tracing. ``torch.jit.script``
    has a non-negligible start up time when it is first called due to
    lazy-initializations of many compiler builtins. Therefore you should not use
    it in library code. However, you may want to have parts of your library work
    in tracing even if they use control flow. In these cases, you should use
    ``@torch.jit.script_if_tracing`` to substitute for
    ``torch.jit.script``.

    Args:
        fn: A function to compile.

    Returns:
        If called during tracing, a :class:`ScriptFunction` created by `torch.jit.script` is returned.
        Otherwise, the original function `fn` is returned.
    """

    return _script_if_tracing(fn)


# for torch.jit.isinstance
def isinstance(obj, target_type):
    """
    This function provides for conatiner type refinement in TorchScript. It can refine
    parameterized containers of the List, Dict, Tuple, and Optional types. E.g. ``List[str]``,
    ``Dict[str, List[torch.Tensor]]``, ``Optional[Tuple[int,str,int]]``. It can also
    refine basic types such as bools and ints that are available in TorchScript.

    Args:
        obj: object to refine the type of
        target_type: type to try to refine obj to
    Returns:
        ``bool``: True if obj was successfully refined to the type of target_type,
            False otherwise with no new type refinement


    Example (using ``torch.jit.isinstance`` for type refinement):
    .. testcode::

        import torch
        from typing import Any, Dict, List

        class MyModule(torch.nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()

            def forward(self, input: Any): # note the Any type
                if torch.jit.isinstance(input, List[torch.Tensor]):
                    for t in input:
                        y = t.clamp(0, 0.5)
                elif torch.jit.isinstance(input, Dict[str, str]):
                    for val in input.values():
                        print(val)

        m = torch.jit.script(MyModule())
        x = [torch.rand(3,3), torch.rand(4,3)]
        m(x)
        y = {"key1":"val1","key2":"val2"}
        m(y)
    """
    return _isinstance(obj, target_type)


if not torch._C._jit_init():
    raise RuntimeError("JIT initialization failed")
