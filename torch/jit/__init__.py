import torch._C

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


if not torch._C._jit_init():
    raise RuntimeError("JIT initialization failed")
