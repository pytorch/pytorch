import torch._C
import torch._jit_internal as _jit_internal

from torch.jit._builtins import _find_builtin, _get_builtin_table, _register_builtin  # noqa
from torch._jit_internal import Future
from torch.nn import Module
from torch.utils import set_module
from torch.autograd.grad_mode import _DecoratorContextManager
from typing import Optional, List

import collections
import contextlib
import functools
import os
import pathlib

# These are imported so users can access them from the `torch.jit` module
from torch._jit_internal import Final, _overload, _overload_method
from torch._jit_internal import ignore, export, unused
from torch.jit._script import script, Attribute, ScriptModule, is_scripting, script_method, \
    RecursiveScriptModule, ScriptWarning, interface
from torch.jit._trace import trace, trace_module, TracedModule, TracerWarning, TracingCheckError, \
    is_tracing, ONNXTracedModule, _unique_state_dict, _flatten, TopLevelTracedModule
from torch.jit._async import fork, wait
from torch.jit._serialization import save, load

set_module(Future, "torch.jit")

# For backwards compatibility
_fork = fork
_wait = wait

@contextlib.contextmanager
def optimized_execution(should_optimize):
    """
    A context manager that controls whether the JIT's executor will run
    optimizations before executing a function.
    """
    stored_flag = torch._C._get_graph_executor_optimize()
    torch._C._set_graph_executor_optimize(should_optimize)
    try:
        yield
    finally:
        torch._C._set_graph_executor_optimize(stored_flag)

@contextlib.contextmanager
def fuser(name):
    """
    A context manager that facilitates switching between
    backend fusers.

    Valid names:
    * ``fuser0`` - enables only legacy fuser
    * ``fuser1`` - enables only NNC
    * ``fuser2`` - enables only nvFuser
    """
    old_cpu_fuse = torch._C._jit_can_fuse_on_cpu()
    old_gpu_fuse = torch._C._jit_can_fuse_on_gpu()
    old_texpr_fuser_state = torch._C._jit_texpr_fuser_enabled()
    old_nvfuser_state = torch._C._jit_nvfuser_enabled()
    if name == 'fuser0':  # legacy fuser
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(False)
    elif name == 'fuser1':  # NNC
        old_profiling_executor = torch._C._jit_set_profiling_executor(True)
        old_profiling_mode = torch._C._jit_set_profiling_mode(True)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(True)
        torch._C._jit_set_nvfuser_enabled(False)
    elif name == 'fuser2':  # nvFuser
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(True)
    else:
        raise Exception("unrecognized fuser option")
    try:
        yield
    finally:
        if name == 'fuser1':  # NNC
            torch._C._jit_set_profiling_executor(old_profiling_executor)
            torch._C._jit_set_profiling_mode(old_profiling_mode)
        # recover the previous values
        torch._C._jit_override_can_fuse_on_cpu(old_cpu_fuse)
        torch._C._jit_override_can_fuse_on_gpu(old_gpu_fuse)
        torch._C._jit_set_texpr_fuser_enabled(old_texpr_fuser_state)
        torch._C._jit_set_nvfuser_enabled(old_nvfuser_state)

def export_opnames(m):
    r"""
        Returns a list of operator names of a script module and its submodules
    """
    return torch._C._export_opnames(m._c)

def _get_trace_graph(f, args=(), kwargs=None, strict=True, _force_outplace=False,
                     return_inputs=False, _return_inputs_states=False):
    """
    .. warning::
        This function is internal-only and should only be used by the ONNX
        exporter. If you are trying to get a graph through tracing, please go
        through the public API instead::

            trace = torch.jit.trace(nn.LSTMCell(), (input, hidden))
            trace_graph = trace.graph

    Trace a function or model, returning a tuple consisting of the both the
    *trace* of an execution, as well as the original return value. If return_inputs,
    also returns the trace inputs as part of the tuple

    Tracing is guaranteed not to change the semantics of the function/module
    that is traced.

    Arguments:
        f (torch.nn.Module or function): the function or module
            to be traced.
        args (tuple or Tensor): the positional arguments to pass to the
            function/module to be traced.  A non-tuple is assumed to
            be a single positional argument to be passed to the model.
        kwargs (dict): the keyword arguments to pass to the function/module
            to be traced.

    Example (trace a cell):

    .. testcode::

        trace = torch.jit.trace(nn.LSTMCell(), (input, hidden))
    """
    if kwargs is None:
        kwargs = {}
    if not isinstance(args, tuple):
        args = (args,)
    outs = ONNXTracedModule(f, strict, _force_outplace, return_inputs, _return_inputs_states)(*args, **kwargs)
    return outs


def freeze(mod, preserved_attrs : Optional[List[str]] = None):
    r"""
    Freezing a :class:`ScriptModule` will clone it and attempt to inline the cloned
    module's submodules, parameters, and attributes as constants in the TorchScript IR Graph.
    By default, `forward` will be preserved, as well as attributes & methods specified in
    `preserved_attrs`. Additionally, any attribute that is modified within a preserved
    method will be preserved.

    Freezing currently only accepts ScriptModules that are in eval mode.

    Arguments:
        mod (:class:`ScriptModule`): a module to be frozen

        preserved_attrs (Optional[List[str]]): a list of attributes to preserve in addition to the forward method.
        Attributes modified in preserved methods will also be preserved.

    Returns:
        Frozen :class:`ScriptModule`.

    Example (Freezing a simple module with a Parameter):

    .. testcode::
        import torch
        class MyModule(torch.nn.Module):
            def __init__(self, N, M):
                super(MyModule, self).__init__()
                self.weight = torch.nn.Parameter(torch.rand(N, M))
                self.linear = torch.nn.Linear(N, M)

            def forward(self, input):
                output = self.weight.mm(input)
                output = self.linear(output)
                return output

        scripted_module = torch.jit.script(MyModule(2, 3).eval())
        frozen_module = torch.jit.freeze(scripted_module)
        # parameters have been removed and inlined into the Graph as constants
        assert len(list(frozen_module.named_parameters())) == 0
        # See the compiled graph as Python code
        print(frozen_module.code)

    Example (Freezing a module with preserved attributes)

    .. testcode::
        import torch
        class MyModule2(torch.nn.Module):
            def __init__(self):
                super(MyModule2, self).__init__()
                self.modified_tensor = torch.tensor(10.)
                self.version = 1

            def forward(self, input):
                self.modified_tensor += 1
                return input + self.modified_tensor

        scripted_module = torch.jit.script(MyModule2().eval())
        frozen_module = torch.jit.freeze(scripted_module, preserved_attrs=["version"])
        # we've manually preserved `version`, so it still exists on the frozen module and can be modified
        assert frozen_module.version == 1
        frozen_module.version = 2
        # `modified_tensor` is detected as being mutated in the forward, so freezing preserves
        # it to retain model semantics
        assert frozen_module(torch.tensor(1)) == torch.tensor(12)
        # now that we've run it once, the next result will be incremented by one
        assert frozen_module(torch.tensor(1)) == torch.tensor(13)

    Note:
        If you're not sure why an attribute is not being inlined as a constant, you can run
        `dump_alias_db` on frozen_module.forward.graph to see if freezing has detected the
        attribute is being modified.
    """
    if not isinstance(mod, ScriptModule):
        raise RuntimeError("Freezing expects a ScriptModule as input. "
                           "Please use torch.jit.script or torch.jit.trace to script your 'nn.Module'.")

    if mod.training:
        raise RuntimeError("Freezing is currently only implemented for modules in eval mode. "
                           "Please call .eval() on your module before freezing.")

    preserved_attrs = preserved_attrs if preserved_attrs is not None else []

    out = RecursiveScriptModule(torch._C._freeze_module(mod._c, preserved_attrs))
    RecursiveScriptModule._finalize_scriptmodule(out)

    return out


class CompilationUnit(object):
    def __init__(self, lang=None, _frames_up=0):
        self._c = torch._C.CompilationUnit()
        if lang is not None:
            self.define(lang, _frames_up=_frames_up + 1)

    def define(self, lang, rcb=None, _frames_up=0):
        if not rcb:
            rcb = _jit_internal.createResolutionCallbackFromFrame(_frames_up + 1)
        self._c.define(lang, rcb)

    def __getattr__(self, attr):
        r = self._c.find_function(attr)
        if r is None:
            raise AttributeError("'CompilationUnit' has no attribute '{}'".format(attr))
        return r


def _try_get_dispatched_fn(fn):
    if not callable(fn):
        return None
    return _jit_internal.boolean_dispatched.get(fn)


def _try_get_overloaded_fn(mod, field):
    return mod._overloads.get(field, None) if isinstance(mod, ScriptModule) else None


@contextlib.contextmanager
def _disable_emit_hooks():
    hooks = torch._C._jit_get_emit_hooks()
    torch._C._jit_set_emit_hooks(None, None)
    yield
    torch._C._jit_set_emit_hooks(hooks[0], hooks[1])


def _disable_emit_hooks_decorator(_DecoratorContextManager):  # noqa: F811
    def __enter__(self):
        self.hooks = torch._C._jit_get_emit_hooks()
        torch._C._jit_set_emit_hooks(None, None)

    def __exit__(self, *args):
        torch._C._jit_set_emit_hooks(self.hooks[0], self.hooks[1])


def _script_if_tracing(fn):
    """
    Compiles ``fn`` when it is first called during tracing. ``torch.jit.script``
    has a non-negligible start up time when it is first called due to
    lazy-initializations of many compiler builtins. Therefore you should not use
    it in library code. However, you may want to have parts of your library work
    in tracing even if they use control flow. In these cases, you should use
    ``@torch.jit._script_if_tracing`` to substitute for
    ``torch.jit.script``.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not is_tracing():
            # Not tracing, don't do anything
            return fn(*args, **kwargs)

        compiled_fn = script(wrapper.__original_fn)
        return compiled_fn(*args, **kwargs)

    wrapper.__original_fn = fn
    wrapper.__script_if_tracing_wrapper = True

    return wrapper

def _unwrap_optional(x):
    assert x is not None, "Unwrapping null optional"
    return x

_register_builtin(_unwrap_optional, 'aten::_unwrap_optional')
_register_builtin(_wait, 'aten::wait')
_register_builtin(wait, 'aten::wait')
_register_builtin(is_scripting, 'aten::is_scripting')


# torch.jit.Error
Error = torch._C.JITException
set_module(Error, "torch.jit")
# This is not perfect but works in common cases
Error.__name__ = "Error"
Error.__qualname__ = "Error"

def _get_named_tuple_properties(obj):
    assert issubclass(obj, tuple) and hasattr(obj, '_fields')
    fields = list(obj._fields)
    annotations = []
    has_annotations = hasattr(obj, '__annotations__')
    for field in fields:
        if has_annotations and field in obj.__annotations__:
            the_type = torch.jit.annotations.ann_to_type(obj.__annotations__[field], _jit_internal.fake_range())
            annotations.append(the_type)
        else:
            annotations.append(torch._C.TensorType.get())
    return type(obj).__name__, fields, annotations

def _create_named_tuple(t, unqual_name, field_names):
    TupleType = collections.namedtuple(unqual_name, field_names)
    return TupleType(*t)

class _disable_tracing(object):
    def __enter__(self):
        self.state = torch._C._get_tracing_state()
        torch._C._set_tracing_state(None)

    def __exit__(self, *args):
        torch._C._set_tracing_state(self.state)
        self.state = None


# for use in python if using annotate
def annotate(the_type, the_value):
    # noop in python
    return the_value

last_executed_optimized_graph = torch._C._last_executed_optimized_graph


def _graph_for(self, *args, **kwargs):
    self(*args, **kwargs)
    return last_executed_optimized_graph()

torch._C.ScriptMethod.graph_for = _graph_for
torch._C.ScriptFunction.graph_for = _graph_for
ScriptFunction = torch._C.ScriptFunction
ScriptFunction.__doc__ = """
Functionally equivalent to a :class:`ScriptModule`, but represents a single
function and does not have any attributes or Parameters.
"""
set_module(ScriptFunction, "torch.jit")

if not torch._C._jit_init():
    raise RuntimeError("JIT initialization failed")
