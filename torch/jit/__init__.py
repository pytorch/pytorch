import torch._C
from torch import Tensor
from torch.autograd import Variable, function
from torch.nn import Module, ModuleList, ParameterList, Parameter, Sequential
from torch.jit.frontend import get_jit_ast
import torch.jit.annotations
from torch._six import raise_from, with_metaclass
from collections import defaultdict, OrderedDict, namedtuple
import sys
import warnings
import itertools
import weakref
import types
import contextlib
import os
import functools
import inspect
import copy
import numbers
import collections
import re

_flatten = torch._C._jit_flatten
_unflatten = torch._C._jit_unflatten
_jit_script_compile = torch._C._jit_script_compile
BatchTensor = torch._C._jit.BatchTensor


@contextlib.contextmanager
def scope(scope_name):
    tracing_state = torch._C._get_tracing_state()
    if tracing_state:
        tracing_state.push_scope(scope_name)
    try:
        yield
    finally:
        if tracing_state:
            tracing_state.pop_scope()


def load(filename):
    m = ScriptModule()
    m._load(filename)
    return m


def get_trace_graph(f, args=(), kwargs=None):
    """
    Trace a function or model, returning a tuple consisting of the both the
    *trace* of an execution, as well as the original return value.

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

    Example: Trace a cell.

        >>> trace, out = jit.trace(nn.LSTMCell(), (input, hidden))
        >>> print(trace)
    """
    if kwargs is None:
        kwargs = {}
    if not isinstance(args, tuple):
        args = (args,)
    return LegacyTracedModule(f)(*args, **kwargs)


def _unique_state_dict(module, keep_vars=False):
    state_dict = module.state_dict(keep_vars=keep_vars)
    filtered_dict = type(state_dict)()
    seen_ids = set()
    for k, v in state_dict.items():
        if id(v) in seen_ids:
            continue
        seen_ids.add(id(v))
        filtered_dict[k] = v
    return filtered_dict


class LegacyTracedModule(Module):
    def __init__(self, inner):
        super(LegacyTracedModule, self).__init__()
        # inner may be a Module, or it may be an arbitrary callable
        # If it's a Module, we get its parameters automatically, which lets
        # us avoid a special casing functions versus modules.
        self.inner = inner

    def forward(self, *args):
        in_vars, in_desc = _flatten(args)
        # NOTE: use full state, because we need it for BatchNorm export
        # This differs from the compiler path, which doesn't support it at the moment.
        module_state = list(_unique_state_dict(self, keep_vars=True).values())
        trace, all_trace_inputs = torch._C._tracer_enter(in_vars + module_state)
        try:
            trace_inputs = _unflatten(all_trace_inputs[:len(in_vars)], in_desc)
            out = self.inner(*trace_inputs)
            out_vars, _ = _flatten(out)
            torch._C._tracer_exit(out_vars)
        except Exception:
            torch._C._tracer_abandon()
            raise
        return trace, out


def _clone_inputs(args):
    def clone_input(a):
        if a is None:
            return None
        elif isinstance(a, torch.Tensor):
            # TODO: figure out one liner to .clone() and set requires_grad
            v = Variable(a.data.clone(), requires_grad=a.requires_grad)
            if a.grad is not None:
                v.grad = clone_input(v.grad)
            return v
        else:
            return a.clone()
    return function._nested_map(lambda x: isinstance(x, torch.Tensor),
                                clone_input, condition_msg="tensors")(args)


# This is purely for developer debugging.  We are not going to advertise it.
_JIT_DUMP = os.environ.get('PYTORCH_JIT_DUMP', False)
_JIT_TIME = os.environ.get('PYTORCH_JIT_TIME', False)  # CUDA-only timing
_JIT_DISABLE = os.environ.get('PYTORCH_JIT_DISABLE', False)
_JIT_STATS = os.environ.get('PYTORCH_JIT_STATS', False)


def _dump_trace(trace_name, pass_name, input_key, trace):
    if not _JIT_DUMP:
        return

    import torch.contrib._graph_vis as graph_vis

    filename = "{}_{}".format(trace_name, pass_name)
    # TODO: Also paste out the backtrace when the trace was compiled
    # (and maybe also when it was run?)
    with open(filename + ".ir", "w") as f:
        f.write("Input key: {}\n\n{}".format(input_key, str(trace)))
    graph_vis.write(trace.graph(), filename + ".html")


@contextlib.contextmanager
def _time(trace_name, name, time=True):
    if (not _JIT_TIME and not time) or not torch.cuda.is_available():
        yield
        return
    stream = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    stream.record_event(start)
    try:
        yield
    finally:
        stream.record_event(end)
        end.synchronize()
        print("{} {} time: {} ms".format(trace_name, name, start.elapsed_time(end)))


def verify(model, args, loss_fn=torch.sum, devices=None):
    """
    Verify that a JIT compiled model has the same behavior as its uncompiled
    version along with its backwards pass.  If your model returns multiple
    outputs, you must also specify a `loss_fn` to produce a loss for which
    the backwards will be computed.

    This function has side-effects (e.g., it executes your model / saves and loads
    parameters), so don't expect the model to come out exactly the same as what
    you passed in.

    Arguments:
        model (compiled torch.nn.Module or function): the module/function to be
            verified.  The module/function definition MUST have been decorated with
            `@torch.jit.compile`.
        args (tuple or Tensor): the positional arguments to pass to the
            compiled function/module to be verified.  A non-tuple is assumed to
            be a single positional argument to be passed to the model.
        loss_fn (function, optional): the loss function to be applied to
            the output of the model, before backwards is invoked.  By default,
            we assume that a model returns a single result, and we :func:`torch.sum`
            before calling backwards; if this is inappropriate, you can pass your
            own loss function.  Note that if a model returns a tuple of results,
            these are passed as separate positional arguments to `loss_fn`.
        devices (iterable of device IDs, optional): the GPU devices which the
            compiled module will be run on.  This determines the RNG state we
            must save when running both compiled and uncompiled versions of the model.
    """
    # TODO: In principle, we track device information in our trace, so it
    # should be possible to check if our execution actually obeyed the 'devices'
    # the user provided.

    # TODO: Consider adding a utility function to torch.jit to test
    # for this case
    if not isinstance(model, torch._C.CompiledFunction):
        raise TypeError("Cannot verify an uncompiled module.  Add @torch.jit.compile to compile it")
    is_module = isinstance(model, Module)

    if not isinstance(args, tuple):
        args = (args,)

    saved_args = _clone_inputs(args)
    if is_module:
        saved_state = copy.deepcopy(model.state_dict())

    def run_fwd_bwd(args, force_trace=False, assert_compiled=False):
        params = list(model.parameters()) if is_module else []
        in_vars, _ = _flatten((args, params))
        # We use a special API to reset the trace and compile it from scratch.
        compiled_fn = model
        if force_trace:
            compiled_fn.clear_cache()
        if assert_compiled:
            hits = compiled_fn.hits
        out = model(*args)
        if assert_compiled and compiled_fn.hits == hits:
            raise RuntimeError("failed to use the compiled function")
        if not isinstance(out, tuple):
            out = (out, )
        if loss_fn == torch.sum and len(out) != 1:
            raise ValueError(("Model returns {} outputs, but default loss function "
                              "(torch.sum) can only handle a single output").format(len(out)))
        out_vars, _ = _flatten(out)
        saved_outs = [v.data.clone() for v in out_vars]
        loss = loss_fn(*out)
        grads = torch.autograd.grad([loss], in_vars)
        # TODO: I'm not sure if the clone here is necessary but it is safer
        saved_grads = [v.data.clone() for v in grads]
        return (saved_outs, saved_grads)

    with torch.random.fork_rng(devices, _caller="torch.jit.verify"):
        uncompiled_outs, uncompiled_grads = run_fwd_bwd(args, force_trace=True)
        assert model.has_trace_for(*args)

    if is_module:
        model.load_state_dict(saved_state)
    compiled_outs, compiled_grads = run_fwd_bwd(args, assert_compiled=True)

    _verify_equal(uncompiled_outs, compiled_outs)
    _verify_equal(uncompiled_grads, compiled_grads)


def _verify_equal(xs, ys):
    for x, y in zip(xs, ys):
        if x.sub(y).abs().max() > 1e-6:
            raise RuntimeError("JIT and real computation mismatch")


def trace(*args, **kwargs):
    """
    Trace a function and return an executable trace that will be optimized
    using just-in-time compilation.

    .. warning::

        Just-in-time compilation currently only works for functions/modules
        which are not data dependent (e.g., have conditionals on data in
        tensors) and do not have any untracked external dependencies (e.g.,
        perform input/output or access global variables). If you trace such
        models, you will silently get incorrect results on subsequent
        invocations of the model.

    Arg:
        *args - a list of example tensors that will be passed to the function
                as inputs while tracing. The resulting trace can be run with
                inputs of different types and shapes assuming the traced operations
                support those types and shapes.

    Keyword arguments:
        optimize (bool, optional): whether or not to apply optimizations.  Default: ``True``.

        >>> @jit.trace(torch.rand(1))
        ... def f(x):
        ...     return x * 2
    """
    def wrapper(func):
        executor_options = {'optimize': True}
        for name in executor_options:
            executor_options[name] = kwargs.pop(name, executor_options[name])
        if len(kwargs) != 0:
            raise TypeError("got unexpected keyword arguments: {}".format(", ".join(kwargs.keys())))

        module = TopLevelTracedModule(func, **executor_options)
        module._create_method_from_trace('forward', func, args)
        return module

    return wrapper


def createResolutionCallback(frames_up=0):
    """
    Creates a function which, given a string variable name,
    returns the value of the variable in the scope of the caller of
    the function which called createResolutionCallback (by default).
    For example, the following program prints 2::

        def bar():
            cb = createResolutionCallback()
            print(x("foo"))

        def baz():
            foo = 2
            bar()

        baz()

    This is used to enable access in-scope Python variables inside
    TorchScript fragments.

    frames_up is
    """
    frame = inspect.stack()[1 + frames_up][0]

    def env(key):
        if key in frame.f_locals:
            return frame.f_locals[key]
        elif key in frame.f_globals:
            return frame.f_globals[key]
        else:
            return None

    return env


class CompilationUnit(object):
    def __init__(self, lang=None, optimize=True, _frames_up=0):
        self.module = torch._C.ScriptModule()
        self.module._set_optimized(optimize)
        if lang is not None:
            self.define(lang, _frames_up=_frames_up + 1)
        self.optimize = optimize

    def define(self, lang, rcb=None, _frames_up=0):
        if not rcb:
            rcb = createResolutionCallback(_frames_up + 1)
        self.module._define(lang, rcb, False)

    def __getattr__(self, attr):
        return self.module._get_method(attr)


def script(fn, optimize=True, _frames_up=0):
    rcb = createResolutionCallback(_frames_up + 1)
    ast = get_jit_ast(fn, is_method=False)
    graph = _jit_script_compile(ast, rcb)
    mod = ScriptModule()
    mod._create_method_from_graph('forward', graph)
    # TODO: refactor everything so we're not 1) creating a ScriptModule
    # 2) Throwing everything away except for the graph 3) Creating a new
    # ScriptModule and dumping that graph in 4) Re-populating the schema
    # because it was lost doing the previous
    mod.__getattr__('forward').forward_schema(ast, False)
    # Forward docstrings
    mod.__doc__ = fn.__doc__
    return mod


ScriptMethodStub = namedtuple('ScriptMethodStub', ('resolution_callback', 'def_', 'original_method'))


def script_method(fn):
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
    rcb = createResolutionCallback(frames_up=2)
    ast = get_jit_ast(fn, is_method=True)
    return ScriptMethodStub(rcb, ast, fn)


def batch(batch_size=1, optimize=True, _frames_up=0):
    def decorator(fn):
        import torch.jit.batchop
        mod = script(fn, optimize, _frames_up)
        res_graph = torch.to_batch_graph(mod.graph)
        res_mod = ScriptModule()
        res_mod._create_method_from_graph('forward', res_graph)

        def wrapper(*args):
            new_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    arg = BatchTensor(arg, batch_size)
                if isinstance(arg, BatchTensor):
                    new_args.extend([arg.get_data(), arg.get_mask(), arg.get_dims()])
                else:
                    new_args.append(arg)
            res = res_mod(*new_args)
            assert len(res) % 3 == 0
            if len(res) % 3 != 0:
                raise "non-batched-tensor output is not supported yet"
            result = [BatchTensor(*res[i * 3: i * 3 + 3]) for i in range(len(res) // 3)]
            if len(result) == 1:
                return result[0]
            return result
        wrapper.__doc__ = fn.__doc__
        return wrapper
    return decorator


# These OrderedDictWrapper classes replace the actual OrderedDicts in
# module with versions that get/set properties inside of script::Module.
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
    def __init__(self, module):
        self.module_ref = weakref.ref(module)

    @property
    def module(self):
        r = self.module_ref()
        if r is None:
            raise RuntimeError("_parameters or _modules alive after module is dead")
        return r

    def keys(self):
        return [k for k, v in self.items()]

    def values(self):
        return [v for k, v in self.items()]

    def __delitem__(self, k):
        raise RuntimeError("cannot delete methods or parameters of a script module")

    def items(self):
        raise NotImplementedError

    def __contains__(self, k):
        raise NotImplementedError

    def __getitem__(self, k):
        raise NotImplementedError

    def __setitem__(self, k, v):
        raise NotImplementedError


class OrderedModuleDict(OrderedDictWrapper):
    def __init__(self, module):
        super(OrderedModuleDict, self).__init__(module)
        # contains _both_ script modules and non-script python-only modules

        # because script modules are subclassed in python and the
        # C++ script::Module class will not hold references to them,
        # to ensure that you always get the same python value here
        # we store it in the python dict as well
        self._python_modules = OrderedDict()

    def items(self):
        r = self._python_modules.items()
        return r

    def __contains__(self, k):
        return k in self._python_modules

    def __setitem__(self, k, v):
        if k in self._python_modules:
            raise RuntimeError("cannot re-assign modules in a ScriptModule")
        if isinstance(v, ScriptModule):
            self.module._register_module(k, v)

        self._python_modules[k] = v

    def __getitem__(self, k):
        return self._python_modules[k]


class OrderedParameterDict(OrderedDictWrapper):
    def __init__(self, module):
        super(OrderedParameterDict, self).__init__(module)

    def items(self):
        return [(name, param) for name, param, is_buffer
                in self.module._get_parameters()
                if not is_buffer]

    def __setitem__(self, k, v):
        self.module._register_parameter(k, v, False)

    def __contains__(self, k):
        return self.module._has_parameter(k)

    def __getitem__(self, k):
        if k not in self:
            raise KeyError(k)
        return self.module._get_parameter(k)


class OrderedBufferDict(OrderedDictWrapper):
    def __init__(self, module):
        super(OrderedBufferDict, self).__init__(module)

    def items(self):
        return [(name, param) for name, param, is_buffer
                in self.module._get_parameters()
                if is_buffer]

    def __setitem__(self, k, v):
        self.module._register_parameter(k, v, True)

    def __contains__(self, k):
        return self.module._has_buffer(k)

    def __getitem__(self, k):
        if k not in self:
            raise KeyError(k)
        return self.module._get_parameter(k)

# base types that can be constants
# in addition, tuples and lists of these base types are also considered constants
# If you edit this list, then you also need to edit the handlers in
# ConstantValue in jit/script/init.cpp
_constant_types = (bool, float, int, types.FunctionType, torch.device, torch.layout, torch.dtype)


def _get_valid_constant(v):
    if isinstance(v, _constant_types):
        return v
    elif isinstance(v, tuple) or isinstance(v, list):
        return tuple(_get_valid_constant(x) for x in v)
    constants = ", ".join(typ.__name__ for typ in _constant_types)
    raise TypeError(
        "'{}' object is not a valid constant.\n".format(type(v).__name__) +
        "Valid constants are:\n" +
        "  1. a nn.ModuleList\n" +
        "  2. a value of type {{{}}}\n".format(constants) +
        "  3. a list or tuple of (2)\n")

# For each user-defined class that subclasses ScriptModule this meta-class,
# (1) finds all the methods annotated with @script_method
# in a ScriptModule and removes them from the class attributes, and
# (2) puts a wrapper around the class's __init__ method to register
# all of the script_methods with the module after the original __init__
# has run. This has to occur after the user-defined __init__ so that
# submodules and parameters are initialized _before_ the script compiler
# resolve references to `self.param` or `self.module`.


class ScriptMeta(type(torch._C.ScriptModule)):
    # this has to inherit from pybind11's metaclass otherwise we get
    # issues because ScriptModule inherits from torch._C.ScriptModule,
    # a pybind11 type
    def __init__(cls, name, bases, attrs):
        # find all the script methods
        cls._original_methods = {}
        methods = []
        for k, v in sorted(attrs.items()):
            if isinstance(v, ScriptMethodStub):
                delattr(cls, k)
                methods.append(v)
                cls._original_methods[v.original_method.__name__] = v.original_method
        # after the user's __init__ register all the script methods
        # with the module
        original_init = getattr(cls, '__init__', lambda self: None)
        super_constants = getattr(super(cls), '_constants_set', set())
        cls._constants_set = set(getattr(cls, '__constants__', ())).union(super_constants)

        def init_then_register(self, *args, **kwargs):
            # ensure even if the user forgets to call super that
            # the pybind object is initialized so it will not segfault
            # run this once, before the most-derived __init__ is called
            if cls is type(self):
                torch._C.ScriptModule.__init__(self)
            original_init(self, *args, **kwargs)
            defs = [m.def_ for m in methods]
            rcbs = [m.resolution_callback for m in methods]
            self._create_methods(defs, rcbs)

        cls.__init__ = init_then_register
        return super(ScriptMeta, cls).__init__(name, bases, attrs)


class ScriptModule(with_metaclass(ScriptMeta, torch._C.ScriptModule, Module)):
    def __init__(self, optimize=True):
        # must be before Module.init since the field is used in __getattr__
        Module.__init__(self)
        self._set_optimized(optimize)
        self._parameters = OrderedParameterDict(self)
        self._buffers = OrderedBufferDict(self)
        self._modules = OrderedModuleDict(self)

    def __getattr__(self, attr):
        if self._has_method(attr):
            if attr in self.__class__._original_methods:
                original_method = self.__class__._original_methods[attr]
                script_method = self._get_method(attr)
                return functools.wraps(original_method)(script_method)
            else:
                return self._get_method(attr)
        if attr == 'graph' and self._has_method('forward'):
            return self.__getattr__('forward').graph
        return Module.__getattr__(self, attr)

    def __setattr__(self, attr, value):
        if attr not in self._constants_set:
            return super(ScriptModule, self).__setattr__(attr, value)
        if hasattr(self, attr):
            raise RuntimeError("attempting to re-assign constant '{}'".format(attr))
        if isinstance(value, ModuleList):
            # special case for list of modules. Modules need to be registered with their
            # parent module. To do this, we create a ConstModuleList, which is itself a module, that
            # contains each of these modules as submodules. The ConstModuleList then
            # is set as an attribute of the parent module.
            super(ScriptModule, self).__setattr__(attr, _ConstModuleList(value))
        elif isinstance(value, Sequential):
            super(ScriptModule, self).__setattr__(attr, _ConstSequential(value))
        else:
            super(ScriptModule, self).__setattr__(attr, _get_valid_constant(value))

    def __dir__(self):
        return sorted(Module.__dir__(self) + self._method_names())

    def define(self, lang):
        # We use frames_up=1 to get to the proper surrounding scope. The stack
        # will look like:
        # 0. createResolutionCallback
        # 1. define()
        # 2. surrounding scope.
        #
        # createResolutionCallback internally adds 1 to get us to our frame, then
        # we add 1 to get to the proper surrounding scope.
        rcb = createResolutionCallback(frames_up=1)
        self._define(lang, rcb, True)


def _get_methods(cls):
    import inspect
    # In Python 3 unbound methods are functions, but in Python 2 they are methods
    return inspect.getmembers(cls, predicate=lambda x: inspect.isfunction(x) or inspect.ismethod(x))


_compiled_methods_whitelist = {
    'forward', 'register_buffer', 'register_parameter', 'add_module',
    '_apply', 'apply', 'cuda', 'cpu', 'type', 'float', 'double', 'half',
    'state_dict', 'load_state_dict', '_load_from_state_dict',
    '_named_members', 'parameters', 'named_parameters',
    'buffers', 'named_buffers', 'children', 'named_children', 'modules',
    'named_modules', 'zero_grad', 'share_memory', '_get_name', 'extra_repr',
    '_slow_forward', '_tracing_name'
}


def _make_fail(name):
    def fail(self, *args, **kwargs):
        raise RuntimeError(name + " is not supported on TracedModules")
    return fail


for name, method in _get_methods(torch.nn.Module):
    if name.startswith('__'):
        continue
    if name not in ScriptModule.__dict__ and name not in _compiled_methods_whitelist:
        setattr(ScriptModule, method.__name__, _make_fail(name))


class TracedModule(ScriptModule):
    __frozen = False

    def __init__(self, orig, id_set=None, optimize=True):
        # XXX: orig can be a nn.Module or a function!
        super(TracedModule, self).__init__(optimize=optimize)
        if id_set is None:
            id_set = set()

        if not isinstance(orig, torch.nn.Module):
            self._name = orig.__name__
            orig = torch.nn.Module()
        else:
            self._name = 'TracedModule[' + type(orig).__name__ + ']'

        def check_unique(param):
            if param in id_set:
                raise ValueError("TracedModules don't support parameter sharing between modules")
            id_set.add(param)

        self.training = orig.training

        for name, param in orig._parameters.items():
            if param is not None:
                self._parameters[name] = param
                check_unique(param)
        for name, buf in orig._buffers.items():
            if buf is not None:
                self._buffers[name] = buf
                check_unique(buf)

        if orig._backward_hooks or orig._forward_hooks or orig._forward_pre_hooks:
            raise ValueError("Modules that have hooks assigned can't be compiled")

        for name, submodule in orig._modules.items():
            self._modules[name] = TracedModule(submodule, id_set, optimize=optimize)

        self._freeze()

    def forward(self, *args, **kwargs):
        raise RuntimeError('Trace submodules cannot be called.')

    def _freeze(self):
        self.__frozen = True

    def _get_name(self):
        return self._name

    def __setattr__(self, attr, value):
        if not self.__frozen or hasattr(self, attr):
            return super(TracedModule, self).__setattr__(attr, value)
        raise RuntimeError("Cannot set new properties on a traced module.")


class TopLevelTracedModule(TracedModule):
    def forward(self, *args, **kwargs):
        return self._get_method('forward')(*args, **kwargs)


class _ConstModuleList(ScriptModule):
    def __init__(self, modules):
        super(_ConstModuleList, self).__init__()
        for i, module in enumerate(modules):
            self.add_module(str(i), module)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return _ConstModuleList(list(self._modules.values())[idx])
        else:
            if not (-len(self) <= idx < len(self)):
                raise IndexError('index {} is out of range'.format(idx))
            if idx < 0:
                idx += len(self)
            return self._modules[str(idx)]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def __dir__(self):
        keys = super(_ConstModuleList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys


class _ConstSequential(_ConstModuleList):
    __constants__ = ['mods']

    def __init__(self, mods):
        super(_ConstSequential, self).__init__(mods._modules.values())

        # we define the forward method via self.define rather than
        # making it a direct class member (with a @script) annotation
        # because, in optimized runtime environments where only .pyc files
        # are shipped, we cant retrieve the source code.
        # TODO: find a workaround for this and remove this hack
        self.define("""
        def forward(self, input):
            for m in self:
                input = m(input)
            return input
        """)

if not torch._C._jit_init():
    raise RuntimeError("JIT initialization failed")
