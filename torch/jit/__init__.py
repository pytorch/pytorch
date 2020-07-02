import torch._C
import torch._jit_internal as _jit_internal
import torch.jit.annotations
import torch.testing
import torch.jit._recursive

from torch.jit._recursive import ScriptMethodStub, wrap_cpp_module
from torch.jit._builtins import _find_builtin, _get_builtin_table, _register_builtin  # noqa
from torch._jit_internal import Future, _qualified_name
from torch.autograd import Variable, function
from torch.jit.frontend import get_jit_class_def, get_jit_def, get_default_args
from torch.nn import Module
from torch.serialization import validate_cuda_device
from torch._six import PY37, with_metaclass, string_classes, get_function_from_type
from torch.utils import set_module
from torch.autograd.grad_mode import _DecoratorContextManager
from typing import Optional, List

import collections
import contextlib
import copy
import functools
import inspect
import os
import pathlib
import pickle
import re
import sys
import textwrap
import warnings
import weakref

# These are imported so users can access them from the `torch.jit` module
from torch._jit_internal import Final, _overload, _overload_method
from torch._jit_internal import ignore, export, unused
from torch.jit._script import Attribute, ScriptModule
from torch.jit._trace import trace, trace_module, TracedModule, TracerWarning, TracingCheckError, \
    is_tracing, ONNXTracedModule, _unique_state_dict, _flatten, TopLevelTracedModule
from torch.jit._state import _python_cu, _enabled

_jit_script_class_compile = torch._C._jit_script_class_compile

set_module(Future, "torch.jit")
_fork = torch._C.fork
_wait = torch._C.wait

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

DEFAULT_EXTRA_FILES_MAP = torch._C.ExtraFilesMap()


def save(m, f, _extra_files=DEFAULT_EXTRA_FILES_MAP):
    r"""
    Save an offline version of this module for use in a separate process. The
    saved module serializes all of the methods, submodules, parameters, and
    attributes of this module. It can be loaded into the C++ API using
    ``torch::jit::load(filename)`` or into the Python API with
    :func:`torch.jit.load <torch.jit.load>`.

    To be able to save a module, it must not make any calls to native Python
    functions.  This means that all submodules must be subclasses of
    :class:`ScriptModule` as well.

    .. DANGER::
        All modules, no matter their device, are always loaded onto the CPU
        during loading.  This is different from :func:`torch.load`'s semantics
        and may change in the future.

    Arguments:
        m: A :class:`ScriptModule` to save.
        f: A file-like object (has to implement write and flush) or a string
           containing a file name.
        _extra_files: Map from filename to contents which will be stored as part of `f`.

    .. note::
        torch.jit.save attempts to preserve the behavior of some operators
        across versions. For example, dividing two integer tensors in
        PyTorch 1.5 performed floor division, and if the module
        containing that code is saved in PyTorch 1.5 and loaded in PyTorch 1.6
        its division behavior will be preserved. The same module saved in
        PyTorch 1.6 will fail to load in PyTorch 1.5, however, since the
        behavior of division changed in 1.6, and 1.5 does not know how to
        replicate the 1.6 behavior.

    Example:

    .. testcode::

        import torch
        import io

        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x + 10

        m = torch.jit.script(MyModule())

        # Save to file
        torch.jit.save(m, 'scriptmodule.pt')
        # This line is equivalent to the previous
        m.save("scriptmodule.pt")

        # Save to io.BytesIO buffer
        buffer = io.BytesIO()
        torch.jit.save(m, buffer)

        # Save with extra files
        extra_files = torch._C.ExtraFilesMap()
        extra_files['foo.txt'] = 'bar'
        torch.jit.save(m, 'scriptmodule.pt', _extra_files=extra_files)
    """
    if isinstance(f, str) or isinstance(f, pathlib.Path):
        m.save(f, _extra_files=_extra_files)
    else:
        ret = m.save_to_buffer(_extra_files=_extra_files)
        f.write(ret)

def load(f, map_location=None, _extra_files=DEFAULT_EXTRA_FILES_MAP):
    r"""
    Load a :class:`ScriptModule` or :class:`ScriptFunction` previously
    saved with :func:`torch.jit.save <torch.jit.save>`

    All previously saved modules, no matter their device, are first loaded onto CPU,
    and then are moved to the devices they were saved from. If this fails (e.g.
    because the run time system doesn't have certain devices), an exception is
    raised.

    Arguments:
        f: a file-like object (has to implement read, readline, tell, and seek),
            or a string containing a file name
        map_location (string or torch.device): A simplified version of
            ``map_location`` in `torch.jit.save` used to dynamically remap
            storages to an alternative set of devices.
        _extra_files (dictionary of filename to content): The extra
            filenames given in the map would be loaded and their content
            would be stored in the provided map.

    Returns:
        A :class:`ScriptModule` object.

    Example:

    .. testcode::

        import torch
        import io

        torch.jit.load('scriptmodule.pt')

        # Load ScriptModule from io.BytesIO object
        with open('scriptmodule.pt', 'rb') as f:
            buffer = io.BytesIO(f.read())

        # Load all tensors to the original device
        torch.jit.load(buffer)

        # Load all tensors onto CPU, using a device
        buffer.seek(0)
        torch.jit.load(buffer, map_location=torch.device('cpu'))

        # Load all tensors onto CPU, using a string
        buffer.seek(0)
        torch.jit.load(buffer, map_location='cpu')

        # Load with extra files.
        extra_files = torch._C.ExtraFilesMap()
        extra_files['foo.txt'] = 'bar'
        torch.jit.load('scriptmodule.pt', _extra_files=extra_files)
        print(extra_files['foo.txt'])

    .. testoutput::
        :hide:

        ...

    .. testcleanup::

        import os
        os.remove("scriptmodule.pt")
    """
    if isinstance(f, string_classes):
        if not os.path.exists(f):
            raise ValueError("The provided filename {} does not exist".format(f))
        if os.path.isdir(f):
            raise ValueError("The provided filename {} is a directory".format(f))

    map_location = validate_map_location(map_location)

    cu = torch._C.CompilationUnit()
    if isinstance(f, str) or isinstance(f, pathlib.Path):
        cpp_module = torch._C.import_ir_module(cu, f, map_location, _extra_files)
    else:
        cpp_module = torch._C.import_ir_module_from_buffer(cu, f.read(), map_location, _extra_files)

    # TODO: Pretty sure this approach loses ConstSequential status and such
    return torch.jit._recursive.wrap_cpp_module(cpp_module)

def validate_map_location(map_location=None):
    if isinstance(map_location, str):
        map_location = torch.device(map_location)
    elif not (map_location is None or
              isinstance(map_location, torch.device)):
        raise ValueError("map_location should be either None, string or torch.device, "
                         "but got type: " + str(type(map_location)))

    if (str(map_location).startswith('cuda')):
        validate_cuda_device(map_location)

    return map_location

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


class ConstMap:
    def __init__(self, const_mapping):
        self.const_mapping = const_mapping

    def __getattr__(self, attr):
        return self.const_mapping[attr]

def fork(func, *args, **kwargs):
    """
    Creates an asynchronous task executing `func` and a reference to the value
    of the result of this execution. `fork` will return immediately,
    so the return value of `func` may not have been computed yet. To force completion
    of the task and access the return value invoke `torch.jit.wait` on the Future. `fork` invoked
    with a `func` which returns `T` is typed as `torch.jit.Future[T]`. `fork` calls can be arbitrarily
    nested, and may be invoked with positional and keyword arguments.
    Asynchronous execution will only occur when run in TorchScript. If run in pure python,
    `fork` will not execute in parallel. `fork` will also not execute in parallel when invoked
    while tracing, however the `fork` and `wait` calls will be captured in the exported IR Graph.
    Warning:
        `fork` tasks will execute non-deterministicly. We recommend only spawning
        parallel fork tasks for pure functions that do not modify their inputs,
        module attributes, or global state.
    Arguments:
        func (callable or torch.nn.Module):  A Python function or `torch.nn.Module`
            that will be invoked. If executed in TorchScript, it will execute asynchronously,
            otherwise it will not. Traced invocations of fork will be captured in the IR.
        *args, **kwargs: arguments to invoke `func` with.
    Returns:
        `torch.jit.Future[T]`: a reference to the execution of `func`. The value `T`
        can only be accessed by forcing completion of `func` through `torch.jit.wait`.
    Example (fork a free function):
    .. testcode::
        import torch
        from torch import Tensor
        def foo(a : Tensor, b : int) -> Tensor:
            return a + b
        def bar(a):
            fut : torch.jit.Future[Tensor] = torch.jit.fork(foo, a, b=2)
            return torch.jit.wait(fut)
        script_bar = torch.jit.script(bar)
        input = torch.tensor(2)
        # only the scripted version executes asynchronously
        assert script_bar(input) == bar(input)
        # trace is not run asynchronously, but fork is captured in IR
        graph = torch.jit.trace(bar, (input,)).graph
        assert "fork" in str(graph)
    Example (fork a module method):
    .. testcode::
        import torch
        from torch import Tensor
        class SubMod(torch.nn.Module):
            def forward(self, a: Tensor, b : int):
                return a + b
        class Mod(torch.nn.Module):
            def __init__(self):
                super(self).__init__()
                self.mod = SubMod()
            def forward(self, input):
                fut = torch.jit.fork(self.mod, a, b=2)
                return torch.jit.wait(fut)
        input = torch.tensor(2)
        mod = Mod()
        assert mod(input) == torch.jit.script(mod).forward(input)
    """
    return torch._C.fork(func, *args, **kwargs)


def wait(future):
    """
    Forces completion of a `torch.jit.Future[T]` asynchronous task, returning the
    result of the task. See :func:`~fork` for docs and examples.
    Arguments:
        func (torch.jit.Future[T]): an asynchronous task reference, created through `torch.jit.fork`
    Returns:
        `T`: the return value of the the completed task
    """
    return torch._C.wait(future)


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


class ScriptWarning(Warning):
    pass

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


# ScriptClasses must be new-style classes because we construct them using their
# __new__ method.
def _is_new_style_class(cls):
    if hasattr(cls, '__class__'):
        return ('__dict__' in dir(cls) or hasattr(cls, '__slots__'))


def whichmodule(obj):
    """Find the module an object belong to."""
    module_name = getattr(obj, '__module__', None)
    # Protect the iteration by using a list copy of sys.modules against dynamic
    # modules that trigger imports of other modules upon calls to getattr.
    for name, module in list(sys.modules.items()):
        if name == '__main__' or module is None:
            continue
        try:
            if _getattribute(module, name)[0] is obj:
                return module_name
        except AttributeError:
            pass
    return '__main__'

def _recursive_compile_class(obj, loc):
    _qual_name = _qualified_name(obj)
    # We're starting a new compilation, so update the error call stack in
    # case it fails
    error_stack = torch._C.CallStack(_qual_name, loc)
    rcb = _jit_internal.createResolutionCallbackForClassMethods(obj)
    _compile_and_register_class(obj, rcb, _qual_name)

def _compile_and_register_class(obj, rcb, qualified_name):
    ast = get_jit_class_def(obj, obj.__name__)
    _jit_script_class_compile(qualified_name, ast, rcb)
    _add_script_class(obj, qualified_name)

def script(obj, optimize=None, _frames_up=0, _rcb=None):
    r"""
    Scripting a function or ``nn.Module`` will inspect the source code, compile
    it as TorchScript code using the TorchScript compiler, and return a :class:`ScriptModule` or
    :class:`ScriptFunction`. TorchScript itself is a subset of the Python language, so not all
    features in Python work, but we provide enough functionality to compute on
    tensors and do control-dependent operations. For a complete guide, see the
    :ref:`language-reference`.

    ``torch.jit.script`` can be used as a function for modules and functions, and as a decorator
    ``@torch.jit.script`` for :ref:`torchscript-classes` and functions.

    Arguments:
        obj (callable, class, or ``nn.Module``):  The ``nn.Module``, function, or class type to
                                                  compile.

    Returns:
        If ``obj`` is ``nn.Module``, ``script`` returns
        a :class:`ScriptModule` object. The returned :class:`ScriptModule` will
        have the same set of sub-modules and parameters as the
        original ``nn.Module``. If ``obj`` is a standalone function,
        a :class:`ScriptFunction` will be returned.

    **Scripting a function**
        The ``@torch.jit.script`` decorator will construct a :class:`ScriptFunction`
        by compiling the body of the function.

        Example (scripting a function):

        .. testcode::

            import torch

            @torch.jit.script
            def foo(x, y):
                if x.max() > y.max():
                    r = x
                else:
                    r = y
                return r

            print(type(foo))  # torch.jit.ScriptFuncion

            # See the compiled graph as Python code
            print(foo.code)

            # Call the function using the TorchScript interpreter
            foo(torch.ones(2, 2), torch.ones(2, 2))

        .. testoutput::
            :hide:

            ...

    **Scripting an nn.Module**
        Scripting an ``nn.Module`` by default will compile the ``forward`` method and recursively
        compile any methods, submodules, and functions called by ``forward``. If a ``nn.Module`` only uses
        features supported in TorchScript, no changes to the original module code should be necessary. ``script``
        will construct :class:`ScriptModule` that has copies of the attributes, parameters, and methods of
        the original module.

        Example (scripting a simple module with a Parameter):

        .. testcode::

            import torch

            class MyModule(torch.nn.Module):
                def __init__(self, N, M):
                    super(MyModule, self).__init__()
                    # This parameter will be copied to the new ScriptModule
                    self.weight = torch.nn.Parameter(torch.rand(N, M))

                    # When this submodule is used, it will be compiled
                    self.linear = torch.nn.Linear(N, M)

                def forward(self, input):
                    output = self.weight.mv(input)

                    # This calls the `forward` method of the `nn.Linear` module, which will
                    # cause the `self.linear` submodule to be compiled to a `ScriptModule` here
                    output = self.linear(output)
                    return output

            scripted_module = torch.jit.script(MyModule(2, 3))

        Example (scripting a module with traced submodules):

        .. testcode::

            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            class MyModule(nn.Module):
                def __init__(self):
                    super(MyModule, self).__init__()
                    # torch.jit.trace produces a ScriptModule's conv1 and conv2
                    self.conv1 = torch.jit.trace(nn.Conv2d(1, 20, 5), torch.rand(1, 1, 16, 16))
                    self.conv2 = torch.jit.trace(nn.Conv2d(20, 20, 5), torch.rand(1, 20, 16, 16))

                def forward(self, input):
                  input = F.relu(self.conv1(input))
                  input = F.relu(self.conv2(input))
                  return input

            scripted_module = torch.jit.script(MyModule())

        To compile a method other than ``forward`` (and recursively compile anything it calls), add
        the :func:`@torch.jit.export <torch.jit.export>` decorator to the method. To opt out of compilation
        use :func:`@torch.jit.ignore <torch.jit.ignore>` or :func:`@torch.jit.unused <torch.jit.unused>`.

        Example (an exported and ignored method in a module)::

            import torch
            import torch.nn as nn

            class MyModule(nn.Module):
                def __init__(self):
                    super(MyModule, self).__init__()

                @torch.jit.export
                def some_entry_point(self, input):
                    return input + 10

                @torch.jit.ignore
                def python_only_fn(self, input):
                    # This function won't be compiled, so any
                    # Python APIs can be used
                    import pdb
                    pdb.set_trace()

                def forward(self, input):
                    if self.training:
                        self.python_only_fn(input)
                    return input * 99

            scripted_module = torch.jit.script(MyModule())
            print(scripted_module.some_entry_point(torch.randn(2, 2)))
            print(scripted_module(torch.randn(2, 2)))
    """
    if not _enabled:
        return obj

    if optimize is not None:
        warnings.warn("`optimize` is deprecated and has no effect. Use `with torch.jit.optimized_execution() instead")
    if isinstance(obj, ScriptModule):
        return obj

    if isinstance(obj, torch.nn.Module):
        return torch.jit._recursive.create_script_module(obj, torch.jit._recursive.infer_methods_to_compile)

    qualified_name = _qualified_name(obj)
    if inspect.isclass(obj):
        # If this type is a `nn.Module` subclass, they probably meant to pass
        # an instance instead of a Module
        if issubclass(obj, torch.nn.Module):
            raise RuntimeError("Type '{}' cannot be compiled since it inherits"
                               " from nn.Module,"
                               " pass an instance instead".format(obj))

        if not _is_new_style_class(obj):
            raise RuntimeError("TorchScript classes must be new-style classes. "
                               "Please inherit from 'object'.")
        if len(obj.mro()) > 2:
            raise RuntimeError("TorchScript classes does not support inheritance yet. "
                               "Please directly inherit from 'object'.")
        if _rcb is None:
            _rcb = _jit_internal.createResolutionCallbackFromFrame(_frames_up + 1)
        _compile_and_register_class(obj, _rcb, qualified_name)
        return obj
    else:
        # this is a decorated fn, and we need to the underlying fn and its rcb
        if hasattr(obj, "__script_if_tracing_wrapper"):
            obj = obj.__original_fn
            _rcb = _jit_internal.createResolutionCallbackFromClosure(obj)

        _check_directly_compile_overloaded(obj)
        maybe_already_compiled_fn = _try_get_jit_cached_function(obj)
        if maybe_already_compiled_fn:
            return maybe_already_compiled_fn
        ast = get_jit_def(obj, obj.__name__)
        if _rcb is None:
            _rcb = _jit_internal.createResolutionCallbackFromClosure(obj)
        fn = torch._C._jit_script_compile(qualified_name, ast, _rcb, get_default_args(obj))
        # Forward docstrings
        fn.__doc__ = obj.__doc__
        _set_jit_function_cache(obj, fn)
        return fn

def interface(obj):
    if not inspect.isclass(obj):
        raise RuntimeError("interface must be applied to a class")
    if not _is_new_style_class(obj):
        raise RuntimeError("TorchScript interfaces must inherit from 'object'")

    # Expected MRO is:
    #   User module
    #   torch.nn.modules.module.Module
    #   object
    is_module_interface = issubclass(obj, torch.nn.Module) and len(obj.mro()) == 3

    if not is_module_interface and len(obj.mro()) > 2:
        raise RuntimeError("TorchScript interface does not support inheritance yet. "
                           "Please directly inherit from 'object' or 'nn.Module'.")

    qualified_name = _qualified_name(obj)
    rcb = _jit_internal.createResolutionCallbackFromFrame(1)
    # if this type is a `nn.Module` subclass, generate an module interface type
    # instead of a class interface type, an module interface type only compile
    # the user provided methods as part of the interface
    ast = get_jit_class_def(obj, obj.__name__)
    torch._C._jit_script_interface_compile(qualified_name, ast, rcb, is_module_interface)
    obj.__torch_script_interface__ = True
    return obj


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


def script_method(fn):
    if not _enabled:
        return fn
    # NOTE: we need to traverse two frames here because the meta-class frame
    # for ScriptModule will be present, as opposed to invoking @script on a
    # a function or invoking define() on a CompilationUnit.
    # The stack will look like:
    #
    # 0. createResolutionCallback()
    # 1. script_method()
    # 2. ScriptModule metaclass frame
    # 3. Surrounding scope
    #
    # createResolutionCallback internally adds 1 to get us to the scope of this
    # function (the calling function). Adding 2 gets us to the proper surrounding scope.
    _rcb = _jit_internal.createResolutionCallbackFromFrame(frames_up=2)
    ast = get_jit_def(fn, fn.__name__, self_name="ScriptModule")
    return ScriptMethodStub(_rcb, ast, fn)



# These OrderedDictWrapper classes replace the actual OrderedDicts in
# module with versions that get/set properties inside of Module.
# This allows us to reuse most of nn.Module while still storing the
# data in C++.
# Each OrderedDict needs to support:
#  x not in view
#  x in view
#  view[name] = ...
#  view.values()
#  del view[name]
#  view.items()
#  view.keys()
#  len(view)

class OrderedDictWrapper(object):
    def __init__(self, _c):
        self._c = _c

    def keys(self):
        return [k for k, v in self.items()]

    def values(self):
        return [v for k, v in self.items()]

    def __len__(self):
        return len(self.values())

    def __delitem__(self, k):
        raise RuntimeError("cannot delete methods or parameters of a script module")

    def items(self):
        return self._c.items()

    def __setitem__(self, k, v):
        if k not in self:
            raise RuntimeError("Can't add a new parameter after ScriptModule construction."
                               " Tried to add '{}".format(k))
        self._c.setattr(k, v)

    def __contains__(self, k):
        return self._c.contains(k)

    def __getitem__(self, k):
        if k not in self:
            raise KeyError(k)
        return self._c.getattr(k)


class OrderedModuleDict(OrderedDictWrapper):
    def __init__(self, module, python_dict):
        super(OrderedModuleDict, self).__init__(torch._C.ModuleDict(module))
        # contains _both_ script modules and non-script python-only modules

        # because script modules are subclassed in python and the
        # C++ Module class will not hold references to them,
        # to ensure that you always get the same python value here
        # we store it in the python dict as well
        self._python_modules = python_dict

    def items(self):
        r = self._python_modules.items()
        return r

    def __contains__(self, k):
        return k in self._python_modules

    def __setitem__(self, k, v):
        # Cases where sub-module can be re-assigned after ScriptModule construction
        # 1. If the attr is an module interface type, it's guaranteed that the module is
        #    not inlined in the graph, so it's safe to swap a new ScriptModule in.
        # 2. if the new value if a ScriptModule with the same JIT type, IR won't change
        #    and it's legit to swap a new module in.
        # In these two cases we allow swapping a new scripted module and update the
        # corresponding python module dict to keep sync.
        # Note: the value to be swapped in has to be ScriptModule instead of nn.Module,
        # otherwise it's illegal and we throw error.
        if isinstance(v, ScriptModule):
            self._c.setattr(k, v)
            self._python_modules[k] = v
        else:
            raise RuntimeError("Cannot re-assign modules in a ScriptModule with non-scripted "
                               "module, tried to replace existing module '{}': {}".format(k, v))


    def __getitem__(self, k):
        return self._python_modules[k]


if _enabled:
    class RecursiveScriptModule(ScriptModule):
        # XXX: RecursiveScriptModule inherits from ScriptModule for the sole
        # reason that it retains the existing isinstance(ScriptModule)
        # behavior.
        r"""
        The core data structure in TorchScript is the ``ScriptModule``. It is an
        analogue of torch's ``nn.Module`` and represents an entire model as a tree of
        submodules. Like normal modules, each individual module in a ``ScriptModule`` can
        have submodules, parameters, and methods. In ``nn.Module``\s methods are implemented
        as Python functions, but in ``ScriptModule``\s methods are implemented as
        TorchScript functions,  a statically-typed subset of Python that contains all
        of PyTorch's built-in Tensor operations. This difference allows your
        ``ScriptModule``\s code to run without the need for a Python interpreter.

        ``ScriptModule``\s should not be created manually, instead use
        either :func:`tracing <torch.jit.trace>` or :func:`scripting <torch.jit.script>`.
        Tracing and scripting can be applied incrementally and :ref:`composed as necessary <Types>`.

        * Tracing records the tensor operations as executed with a set of example inputs and uses these
          operations to construct a computation graph. You can use the full dynamic behavior of Python with tracing,
          but values other than Tensors and control flow aren't captured in the graph.

        * Scripting inspects the Python code of the model
          and compiles it to TorchScript. Scripting allows the use of many `types`_ of values and supports dynamic control flow.
          Many, but not all features of Python are supported by the compiler, so changes to the source code may be necessary.
        """
        _disable_script_meta = True

        def __init__(self, cpp_module):
            self.__dict__['_initializing'] = True
            self._c = cpp_module
            super(RecursiveScriptModule, self).__init__()
            # Delete the 'training' attribute set up by `Module.__init__`. It
            # will get set on the underlying cpp module, so we delete it here
            # to avoid this version shadowing the cpp module version.
            delattr(self, 'training')

        @staticmethod
        def _construct(cpp_module, init_fn):
            """
            Construct a RecursiveScriptModule that's ready for use. PyTorch
            code should use this to construct a RecursiveScriptModule instead
            of instead of calling `__init__` directly, as it makes sure the
            object is properly finalized (and in the future we may take
            control of how the RecursiveScriptModule instance is created).

            Arguments:
                cpp_module:  The C++ Module that will hold the actual state of
                             this RecursiveScriptModule instance.
                init_fn:  Lambda that initializes the RecursiveScriptModule passed to it.
            """
            script_module = RecursiveScriptModule(cpp_module)
            init_fn(script_module)

            # Finalize the ScriptModule: replace the nn.Module state with our
            # custom implementations and flip the _initializing bit.
            RecursiveScriptModule._finalize_scriptmodule(script_module)
            return script_module

        @staticmethod
        def _finalize_scriptmodule(script_module):
            script_module._parameters = OrderedDictWrapper(torch._C.ParameterDict(script_module._c))
            script_module._buffers = OrderedDictWrapper(torch._C.BufferDict(script_module._c))
            script_module._modules = OrderedModuleDict(script_module._c, script_module._modules)
            script_module._initializing = False

        def _reconstruct(self, cpp_module):
            """
            Re-construct an instance of RecursiveScriptModule using an instance of a C++ module.

            Arguments:
                cpp_module: The C++ module that this RecursiveScriptModule will be rebuilt around.
            """
            self.__init__(cpp_module)

            # Copy the concrete type from the C++ module to this ScriptModule.
            self._concrete_type = torch._C.ConcreteModuleType.from_jit_type(self._c._type())

            # Copy submodules from the C++ module to this ScriptModule.
            modules = {}
            for name, cpp_module in torch._C.ModuleDict(self._c).items():
                modules[name] = wrap_cpp_module(cpp_module)
            self._modules = OrderedModuleDict(self._c, modules)

            # Copy parameters and buffers.
            self._parameters = OrderedDictWrapper(torch._C.ParameterDict(self._c))
            self._buffers = OrderedDictWrapper(torch._C.BufferDict(self._c))

            # Get rid of the functions from the old C++ module.
            self.__dict__ = {k: v for k, v in self.__dict__.items() if not isinstance(v, torch._C.ScriptMethod)}
            self.__dict__['_initializing'] = False

        @property
        def graph(self):
            r"""
            Returns a string representation of the internal graph for the
            ``forward`` method. See `Interpreting Graphs`_ for details.
            """
            return self.forward.graph

        @property
        def inlined_graph(self):
            r"""
            Returns a string representation of the internal graph for the
            ``forward`` method. This graph will be preprocessed to inline all function and method calls.
            See `Interpreting Graphs`_ for details.
            """
            return self.forward.inlined_graph

        @property
        def code(self):
            r"""
            Returns a pretty-printed representation (as valid Python syntax) of
            the internal graph for the ``forward`` method. See `Inspecting Code`_
            for details.
            """
            return self.forward.code

        @property
        def code_with_constants(self):
            r"""
            Returns a tuple of:

            [0] a pretty-printed representation (as valid Python syntax) of
            the internal graph for the ``forward`` method. See `code`.
            [1] a ConstMap following the CONSTANT.cN format of the output in [0].
            The indices in the [0] output are keys to the underlying constant's values.

            See `Inspecting Code`_ for details.
            """
            r = self.forward.code_with_constants
            return (r[0], ConstMap(r[1]))

        def save(self, *args, **kwargs):
            r"""
            save(f, _extra_files=ExtraFilesMap{})

            See :func:`torch.jit.save <torch.jit.save>` for details.
            """
            return self._c.save(*args, **kwargs)

        def _save_for_lite_interpreter(self, *args, **kwargs):
            r"""
            _save_for_lite_interpreter(f)

            Add (or update) the bytecode session to the script model. The updated model is used
            in lite interpreter for mobile applications.

            Arguments:
                f: a string containing a file name.
                _extra_files: Map from filename to contents which will be stored as part of 'f'.

            """
            return self._c._save_for_mobile(*args, **kwargs)

        def _save_to_buffer_for_lite_interpreter(self, *args, **kwargs):
            return self._c._save_to_buffer_for_mobile(*args, **kwargs)

        def save_to_buffer(self, *args, **kwargs):
            return self._c.save_to_buffer(*args, **kwargs)

        def get_debug_state(self, *args, **kwargs):
            return self._c.get_debug_state()

        def extra_repr(self):
            return 'original_name={}'.format(self.original_name)

        def graph_for(self, *args, **kwargs):
            return self.forward.graph_for(*args, **kwargs)

        @property
        def original_name(self):
            if type(self) == str(self._c._type().name()):
                return ''
            return str(self._c._type().name())

        def define(self, src):
            # We use frames_up=1 to get to the proper surrounding scope. The stack
            # will look like:
            # 0. createResolutionCallback
            # 1. define()
            # 2. surrounding scope.
            #
            # createResolutionCallback internally adds 1 to get us to our frame, then
            # we add 1 to get to the proper surrounding scope.
            rcb = _jit_internal.createResolutionCallbackFromFrame(frames_up=1)
            self._c._define(self._concrete_type, src, rcb)

        def __getattr__(self, attr):
            if '_initializing' not in self.__dict__:
                raise RuntimeError("ScriptModule has not been initialized, did you forget to call super's init?")

            if self._initializing:
                return super(RecursiveScriptModule, self).__getattr__(attr)

            # _modules check is before hasattr since modules are included as attributes in _c,
            # but we want to get the python wrapper from _modules instead of the raw _c object.
            if attr in self._modules:
                return self._modules[attr]
            elif self._c.hasattr(attr):
                return self._c.getattr(attr)
            elif self._c._has_method(attr):
                script_method = self._c._get_method(attr)
                # cache method so future calls do not go through __getattr__
                # to improve invocation performance
                self.__dict__[attr] = script_method
                return script_method

            return super(RecursiveScriptModule, self).__getattr__(attr)

        def __setattr__(self, attr, value):
            if self._initializing:
                return super(RecursiveScriptModule, self).__setattr__(attr, value)

            if attr in self._modules:
                self._modules[attr] = value
            elif self._c.hasattr(attr):
                self._c.setattr(attr, value)
            elif hasattr(self, "_concrete_type") and attr in self._concrete_type.get_constants().keys():
                # TODO: we don't have _concrete_type set after load(), and in general we lose constant information.
                # We should encode constants as class type attributes (or something) so it persists across save/load.
                raise AttributeError("Cannot mutate TorchScript constant value: '{}'. Value: '{}'".format(attr, value))
            else:
                # We allow setting Python attributes on the ScriptModule, for
                # when people want to stash some convenience info on it.
                # TODO: it's possible that the following is confusing:
                #   s = torch.jit.script(...)
                #   s.python_attr = ...
                #   s.save()   <--- this doesn't have `python_attr`
                # It's fairly trivial to save enough info to warn in this case.
                return super(RecursiveScriptModule, self).__setattr__(attr, value)

        def __getstate__(self):
            raise pickle.PickleError(
                "ScriptModules cannot be deepcopied using copy.deepcopy or saved using torch.save. " +
                "Mixed serialization of script and non-script modules is not supported. " +
                "For purely script modules use my_script_module.save(<filename>) instead.")

        def __copy__(self):
            return torch.jit._recursive.wrap_cpp_module(copy.copy(self._c))

        def __deepcopy__(self, memo):
            return torch.jit._recursive.wrap_cpp_module(copy.deepcopy(self._c, memo))

        # Python magic methods do method lookups on an object's class type, instead of looking up
        # the method defines on the class instance. In order to continue to expose the magic methods
        # of builtin-containers (ModuleList, Sequential, ModuleDict) to python we
        # define magic methods here as a shim to the correct attribute.
        def forward_magic_method(self, method_name, *args, **kwargs):
            self_method = getattr(self, method_name)
            if getattr(self_method, "__func__", None) == getattr(RecursiveScriptModule, method_name):
                raise NotImplementedError()
            return self_method(*args, **kwargs)

        def __iter__(self):
            return self.forward_magic_method("__iter__")

        def __getitem__(self, idx):
            return self.forward_magic_method("__getitem__", idx)

        def __len__(self):
            return self.forward_magic_method("__len__")

        def __contains__(self, key):
            return self.forward_magic_method("__contains__", key)

        # dir is defined by the base nn.Module, so instead of throwing if
        # it is not overriden, we call into the nn.Module __dir__ method
        def __dir__(self):
            self_method = self.__dir__
            if self_method.__func__ == get_function_from_type(RecursiveScriptModule, "__dir__"):
                return super(RecursiveScriptModule, self).__dir__()
            return self_method()

        # to resolve bool(value), python looks if __bool__ is defined then __iter__
        # is defined then returns true for classes. because __iter__() on this
        # class throws if it isn't overriden, we define __bool__ to preserve default behavior
        def __bool__(self):
            self_method = self.__bool__
            if self_method.__func__ == get_function_from_type(RecursiveScriptModule, "__bool__"):
                return True
            return self_method()

        def _replicate_for_data_parallel(self):
            # we have to initialize ScriptModule properly so that
            # it works with pybind11
            def init_fn(script_module):
                # Don't do anything here, we'll initialize the ScriptModule below
                return
            return RecursiveScriptModule._construct(self._c._replicate_for_data_parallel(), init_fn)

    # Need to copy all RecursiveScriptModule methods to ScriptModule.
    #
    # This is because `super(MyScriptModule, self).foo()` does not use
    # `__getattr__` to look up `foo`. So we need to make each method available on
    # the ScriptModule manually.
    for name, item in RecursiveScriptModule.__dict__.items():
        if not callable(item) and not isinstance(item, property):
            continue
        if name.startswith('__') or hasattr(ScriptModule, name):
            continue
        # We can copy over the implementation wholesale because besides the
        # `super()` thing above, ScriptModule behaves exactly like
        # RecursiveScriptModule
        setattr(ScriptModule, name, item)

    def _get_methods(cls):
        import inspect
        # In Python 3 unbound methods are functions, but in Python 2 they are methods
        return inspect.getmembers(cls, predicate=lambda x: inspect.isfunction(x) or inspect.ismethod(x))


    _compiled_methods_whitelist = {
        'forward', 'register_buffer', 'register_parameter', 'add_module',
        '_apply', 'apply', 'cuda', 'cpu', 'to', 'type', 'float', 'double', 'half',
        'state_dict', '_save_to_state_dict', 'load_state_dict',
        '_load_from_state_dict', '_named_members', 'parameters', 'named_parameters',
        'buffers', 'named_buffers', 'children', 'named_children', 'modules',
        'named_modules', 'zero_grad', 'share_memory', '_get_name', 'extra_repr',
        '_slow_forward', '_tracing_name', 'eval', 'train',
    }


    def _make_fail(name):
        def fail(self, *args, **kwargs):
            raise RuntimeError(name + " is not supported on ScriptModules")
        return fail

    for name, method in _get_methods(torch.nn.Module):
        if name.startswith('__'):
            continue
        if name not in RecursiveScriptModule.__dict__ and name not in _compiled_methods_whitelist:
            setattr(RecursiveScriptModule, method.__name__, _make_fail(name))

def is_scripting():
    r"""
    Function that returns True when in compilation and False otherwise. This
    is useful especially with the @unused decorator to leave code in your
    model that is not yet TorchScript compatible.
    .. testcode::

        import torch

        @torch.jit.unused
        def unsupported_linear_op(x):
            return x

        def linear(x):
           if not torch.jit.is_scripting():
              return torch.linear(x)
           else:
              return unsupported_linear_op(x)
    """
    return False


def _unwrap_optional(x):
    assert x is not None, "Unwrapping null optional"
    return x

_register_builtin(_unwrap_optional, 'aten::_unwrap_optional')
_register_builtin(_wait, 'aten::wait')
_register_builtin(wait, 'aten::wait')
_register_builtin(is_scripting, 'aten::is_scripting')


# Caching: we currently cache compilation of free functions and overloaded functions.
# To cache free functions we hold a weak ref to the function object and
# map to the compiled fn's qualified name.
# To cache overloaded functions we hold a weak ref to the function obj and
# map to all of its overloaded compiled fns.
# In the future we could consider caching more types of objects so that
# aliasing is preserved across separate compilations of the same object.

_jit_caching_layer = weakref.WeakKeyDictionary()
_jit_function_overload_caching = weakref.WeakKeyDictionary()

def _try_get_jit_cached_overloads(key):
    qual_names = _jit_function_overload_caching.get(key, None)
    if qual_names:
        return [_python_cu.find_function(qual_name) for qual_name in qual_names]
    else:
        return None

def _set_jit_overload_cache(key, compiled_fns):
    _jit_function_overload_caching[key] = [fn.qualified_name for fn in compiled_fns]

def _try_get_jit_cached_function(key):
    if getattr(key, "__disable_jit_function_caching__", False) is True:
        return None
    qual_name = _jit_caching_layer.get(key, None)
    if qual_name:
        return _python_cu.find_function(qual_name)
    else:
        return None

def _set_jit_function_cache(key, value):
    # only free functions currently supported
    assert isinstance(value, torch.jit.ScriptFunction)
    _jit_caching_layer[key] = value.qualified_name


# qualified_name => ScriptClass mapping
_script_classes = {}


def _add_script_class(cls, name):
    cls.__torch_script_class__ = True
    global _script_classes
    _script_classes[name] = cls


def _get_script_class(name):
    global _script_classes
    if name not in _script_classes:
        return None
    return _script_classes[name]

# overloads are registered in _jit_internal and compiled here so that _overload
# can be used in nn/functional.py without an import cycle

def _check_overload_defaults(impl_defaults, overload_defaults, loc):
    for name, overload_value in overload_defaults.items():
        if name not in impl_defaults or impl_defaults[name] != overload_value:
            raise torch.jit.frontend.FrontendError(
                loc, "Default parameters on overloads do not affect the runtime so they "
                "must equal to the default parameter on the implementation function. Found on "
                "parameter {name}".format(name=name))

def _compile_function_with_overload(overload_fn, qual_name, impl_fn):
    overload_decl = torch.jit.get_jit_def(overload_fn, overload_fn.__name__).decl()
    overload_signature = torch.jit.annotations.get_signature(overload_fn, None, None, inspect.ismethod(overload_fn))
    impl_ast = torch.jit.get_jit_def(impl_fn, impl_fn.__name__)
    overload_defaults = get_default_args(overload_fn)
    implementation_defaults = get_default_args(impl_fn)
    _rcb = _jit_internal.createResolutionCallbackFromClosure(impl_fn)
    _check_overload_defaults(implementation_defaults, overload_defaults, overload_decl.range())
    fn = torch._C._jit_script_compile_overload(qual_name, overload_decl, impl_ast, _rcb,
                                               implementation_defaults, overload_signature)
    return fn

def _get_overloads(obj):
    # check for cached compiled fns
    existing_compiled_fns = _try_get_jit_cached_overloads(obj)
    qual_name = _qualified_name(obj)
    uncompiled_overloads = _jit_internal._get_fn_overloads(qual_name)
    if uncompiled_overloads is None:
        return existing_compiled_fns

    compiled_fns = []
    for overload_fn in uncompiled_overloads:
        compiled_fns.append(_compile_function_with_overload(overload_fn, qual_name, obj))

    if existing_compiled_fns:
        compiled_fns = existing_compiled_fns + compiled_fns

    # cache compilation, remove information stored to do compilation
    _set_jit_overload_cache(obj, compiled_fns)
    _jit_internal._clear_fn_overloads(qual_name)
    return compiled_fns

def _check_directly_compile_overloaded(obj):
    qual_name = _qualified_name(obj)
    if _jit_internal._get_fn_overloads(qual_name) or _try_get_jit_cached_overloads(obj):
        raise RuntimeError("Function {} cannot be directly compiled because it"
                           " is overloaded. It must be used in a context of a function"
                           " where its inputs can determine which overload to call.".format(qual_name))

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
