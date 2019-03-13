"""
The weak_script annotation needs to be here instead of inside torch/jit/ so it
can be used in other places in torch/ (namely torch.nn) without running into
circular dependency problems
"""

import torch._C
import weakref
import inspect
import textwrap
import os
import types
import functools
from torch._six import builtins, with_metaclass, get_function_from_type
from torch.nn.modules import Module, Container, Sequential, ModuleList, \
    ModuleDict, Parameter, ParameterList, ParameterDict

from torch.frontend import get_jit_class_def, get_jit_def, get_default_args
from collections import defaultdict, OrderedDict, namedtuple

_jit_script_compile = torch._C._jit_script_compile
_jit_script_class_compile = torch._C._jit_script_class_compile

# Tracks standalone weak script functions
compiled_weak_fns = weakref.WeakKeyDictionary()  # noqa: T484

# Tracks which methods should be converted to strong methods
weak_script_methods = weakref.WeakKeyDictionary()  # noqa: T484

# Converted modules and their corresponding WeakScriptModuleProxy objects
weak_modules = weakref.WeakKeyDictionary()  # noqa: T484

# Types that have been declared as weak modules
weak_types = weakref.WeakKeyDictionary()  # noqa: T484

# Wrapper functions that can call either of 2 functions depending on a boolean
# argument
boolean_dispatched = weakref.WeakKeyDictionary()  # noqa: T484

# Python Op functions that should be ignored by the compiler. These will be replaced
# with an operator that always throws an error
ignored_fns = weakref.WeakSet()  # noqa: T484

COMPILATION_PENDING = object()
COMPILED = object()


def createResolutionCallback(frames_up=0):
    """
    Creates a function which, given a string variable name,
    returns the value of the variable in the scope of the caller of
    the function which called createResolutionCallback (by default).

    This is used to enable access in-scope Python variables inside
    TorchScript fragments.

    frames_up is number of additional frames to go up on the stack.
    The default value is 0, which correspond to the frame of the caller
    of createResolutionCallback. Also for example, if frames_up is set
    to 1, then the frame of the caller's caller of createResolutionCallback
    will be taken.

    For example, the following program prints 2::

        def bar():
            cb = createResolutionCallback(1)
            print(cb("foo"))

        def baz():
            foo = 2
            bar()

        baz()
    """
    frame = inspect.currentframe()
    i = 0
    while i < frames_up + 1:
        frame = frame.f_back
        i += 1

    f_locals = frame.f_locals
    f_globals = frame.f_globals

    def env(key):
        if key in f_locals:
            return f_locals[key]
        elif key in f_globals:
            return f_globals[key]
        elif hasattr(builtins, key):
            return getattr(builtins, key)
        else:
            return None

    return env


def weak_script(fn, _frames_up=0):
    """
    Marks a function as a weak script function. When used in a script function
    or ScriptModule, the weak script function will be lazily compiled and
    inlined in the graph. When not used in a script function, the weak script
    annotation has no effect.
    """
    compiled_weak_fns[fn] = {
        "status": COMPILATION_PENDING,
        "compiled_fn": None,
        "rcb": createResolutionCallback(_frames_up + 1)
    }
    return fn


def weak_module(cls):
    weak_types[cls] = {
        "method_stubs": None
    }
    return cls


def weak_script_method(fn):
    weak_script_methods[fn] = {
        "rcb": createResolutionCallback(frames_up=2),
        "original_method": fn
    }
    return fn


def boolean_dispatch(arg_name, arg_index, default, if_true, if_false, module_name, func_name):
    """
    Dispatches to either of 2 weak script functions based on a boolean argument.
    In TorchScript, the boolean argument must be constant so that the correct
    function to use can be determined at compile time.
    """
    if compiled_weak_fns.get(if_true) is None or compiled_weak_fns.get(if_false) is None:
        raise RuntimeError("both functions must be weak script")

    def fn(*args, **kwargs):
        dispatch_flag = False
        if arg_name in kwargs:
            dispatch_flag = kwargs[arg_name]
        elif arg_index < len(args):
            dispatch_flag = args[arg_index]

        if dispatch_flag:
            return if_true(*args, **kwargs)
        else:
            return if_false(*args, **kwargs)

    if if_true.__doc__ is None and if_false.__doc__ is not None:
        doc = if_false.__doc__
        if_true.__doc__ = doc
    elif if_false.__doc__ is None and if_true.__doc__ is not None:
        doc = if_true.__doc__
        if_false.__doc__ = doc
    elif if_false.__doc__ is None and if_true.__doc__ is None:
        # neither function has a docstring
        doc = None
    else:
        raise RuntimeError("only one function can have a docstring")
    fn.__doc__ = doc

    if module_name is not None:
        fn.__module__ = module_name
    if func_name is not None:
        fn.__name__ = func_name

    boolean_dispatched[fn] = {
        "if_true": if_true,
        "if_false": if_false,
        "index": arg_index,
        "default": default,
        "arg_name": arg_name
    }
    return fn


def ignore(fn):
    ignored_fns.add(fn)
    return fn


def _parameter_list(fn):
    """
    Decorator to denote that a function returns a list of all the parameters
    in a module
    """
    fn._is_parameter_list = True
    return fn


try:
    import typing
    from typing import Tuple, List, Dict

    def is_tuple(ann):
        # For some reason Python 3.7 violates the Type[A, B].__origin__ == Type rule
        return ann.__module__ == 'typing' and \
            (getattr(ann, '__origin__', None) is typing.Tuple or
             getattr(ann, '__origin__', None) is tuple)

    def is_list(ann):
        return ann.__module__ == 'typing' and \
            (getattr(ann, '__origin__', None) is typing.List or
             getattr(ann, '__origin__', None) is list)

    def is_dict(ann):
        return ann.__module__ == 'typing' and \
            (getattr(ann, '__origin__', None) is typing.Dict or
             getattr(ann, '__origin__', None) is dict)
except ImportError:
    # A minimal polyfill for versions of Python that don't have typing.
    # Note that this means that they also don't support the fancy annotation syntax, so
    # those instances will only be used in our tiny `type: ` comment interpreter.

    # The __getitem__ in typing is implemented using metaclasses, but I'm too lazy for that.
    class TupleCls(object):
        def __getitem__(self, types):
            return TupleInstance(types)

    class TupleInstance(object):
        def __init__(self, types):
            setattr(self, '__args__', types)

    class ListInstance(object):
        def __init__(self, types):
            setattr(self, '__args__', types)

    class ListCls(object):
        def __getitem__(self, types):
            return TupleInstance(types)

    class DictInstance(object):
        def __init__(self, types):
            setattr(self, '__args__', types)

    class DictCls(object):
        def __getitem__(self, types):
            return DictInstance(types)

    Tuple = TupleCls()  # noqa: T484
    List = ListCls()  # noqa: T484
    Dict = DictCls()  # noqa: T484

    def is_tuple(ann):
        return isinstance(ann, TupleInstance)

    def is_list(ann):
        return isinstance(ann, ListInstance)

    def is_dict(ann):
        return isinstance(ann, DictInstance)


# allows BroadcastingList instance to be subscriptable
class BroadcastingListCls(object):
    def __getitem__(self, types):
        return

# mypy doesn't support parameters on types, so we have to explicitly type each
# list size
BroadcastingList1 = BroadcastingListCls()
for i in range(2, 7):
    globals()["BroadcastingList{}".format(i)] = BroadcastingList1


def _parse_env(name, default, true_message, false_message):
    value = os.environ.get(name)
    if value is None:
        return default
    if value.lower() in {'1', 'true', 'yes'}:
        return True
    elif value.lower() in {'0', 'false', 'no'}:
        return False
    if value == '1v':
        print(true_message)
        return True
    elif value == '0v':
        print(false_message)
        return False
    raise ValueError('Unknown setting of {}. Try using 0 or 1.'.format(name))


_enabled = _parse_env('PYTORCH_JIT', True, "> Using PyTorch JIT", "> PyTorch JIT DISABLED")


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


def _try_get_dispatched_fn(fn):
    if not callable(fn):
        return None
    return boolean_dispatched.get(fn)


def _try_get_overloaded_fn(fn):
    if not hasattr(fn, '__self__') or not isinstance(fn.__self__, ScriptModule):
        # Only allow overloads for bound methods
        return None
    overloads = fn.__self__._overloads.get(fn.__name__, None)
    if overloads is None:
        return None
    return [getattr(fn.__self__, overload) for overload in overloads]


def _try_compile_weak_script(fn):
    entry = compiled_weak_fns.get(fn)
    if entry is None:
        return None
    if entry["status"] == COMPILATION_PENDING:
        compiled_fn = torch.jit.script(fn, True, 0, entry["rcb"])
        del entry["rcb"]
        compiled_weak_fns[fn]["compiled_fn"] = compiled_fn
        entry["status"] = COMPILED
        return compiled_fn
    else:
        return entry["compiled_fn"]


def script(obj, optimize=True, _frames_up=0, _rcb=None):
    if not _enabled:
        return obj
    if _rcb is None:
        _rcb = createResolutionCallback(_frames_up + 1)
    if inspect.isclass(obj):
        mod = ScriptClass(obj.__name__)
        ast = get_jit_class_def(obj)
        _jit_script_class_compile(mod, ast, _rcb)
    else:
        mod = ScriptModule()
        ast = get_jit_def(obj)
        _jit_script_compile(mod, ast, _rcb, get_default_args(obj))
    # Forward docstrings
    mod.__doc__ = obj.__doc__
    return mod


ScriptMethodStub = namedtuple('ScriptMethodStub', ('resolution_callback', 'def_', 'original_method'))


def script_method(fn, _rcb=None):
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
    if _rcb is None:
        _rcb = createResolutionCallback(frames_up=2)
    ast = get_jit_def(fn, self_name="ScriptModule")
    return ScriptMethodStub(_rcb, ast, fn)


def _try_get_weak_module(mod):
    """
    Get the WeakScriptModuleProxy corresponding to mod if it exists
    """
    if not isinstance(mod, Module):
        return None
    return weak_modules.get(mod)


def _try_get_ignored_op(fn):
    if not callable(fn):
        return False
    if hasattr(fn, '__func__'):
        fn = fn.__func__
    return fn in ignored_fns


def _is_weak_type(cls):
    """
    Check if a type has been annotated with `weak_module`
    """
    return cls in weak_types

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
        return [(name, param) for name, param in self.module._get_parameters()]

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
        return [(name, param) for name, _, param in
                self.module._get_attributes() if isinstance(param, torch.Tensor)]

    def __setitem__(self, k, v):
        self.module._register_buffer(k, v)

    def __contains__(self, k):
        return self.module._has_buffer(k)

    def __getitem__(self, k):
        if k not in self:
            raise KeyError(k)
        return self.module._get_buffer(k)


# base types that can be constants
# in addition, tuples and lists of these base types are also considered constants
# If you edit this list, then you also need to edit the handlers in
# ConstantValue in jit/script/init.cpp
_constant_types = (bool, float, int, str, type(None), types.FunctionType, torch.device, torch.layout, torch.dtype)


def _get_valid_constant(attr, v):
    if isinstance(v, _constant_types):
        return v
    elif isinstance(v, tuple) or isinstance(v, list):
        return tuple(_get_valid_constant(attr, x) for x in v)
    constants = ", ".join(typ.__name__ for typ in _constant_types)
    raise TypeError(textwrap.dedent("""
        '{}' object for attribute '{}' is not a valid constant.
        Valid constants are:
          1. a nn.ModuleList
          2. a value of type {{{}}}
          3. a list or tuple of (2)
        """.format(type(v).__name__, attr, constants)))


def _create_methods_from_stubs(self, stubs):
    defs = [m.def_ for m in stubs]
    rcbs = [m.resolution_callback for m in stubs]
    defaults = [get_default_args(m.original_method) for m in stubs]
    self._create_methods(defs, rcbs, defaults)

# For each user-defined class that subclasses ScriptModule this meta-class,
# (1) finds all the methods annotated with @script_method
# in a ScriptModule and removes them from the class attributes, and
# (2) puts a wrapper around the class's __init__ method to register
# all of the script_methods with the module after the original __init__
# has run. This has to occur after the user-defined __init__ so that
# submodules and parameters are initialized _before_ the script compiler
# resolve references to `self.param` or `self.module`.


class ScriptMeta(type(torch._C.ScriptModule)):  # noqa T484
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
        cls._overloads = dict(getattr(cls, '__overloads__', {}))

        @functools.wraps(original_init)
        def init_then_register(self, *args, **kwargs):
            # ensure even if the user forgets to call super that
            # the pybind object is initialized so it will not segfault
            # run this once, before the most-derived __init__ is called
            if cls is type(self):
                torch._C.ScriptModule.__init__(self)
            original_init(self, *args, **kwargs)
            _create_methods_from_stubs(self, methods)

        cls.__init__ = init_then_register
        return super(ScriptMeta, cls).__init__(name, bases, attrs)


if _enabled:
    class ScriptModule(with_metaclass(ScriptMeta, torch._C.ScriptModule, Module)):  # noqa T484
        r"""
        The core data structure in TorchScript is the ``ScriptModule``. It is an
        analogue of torch's nn.Module and represents an entire model as a tree of
        submodules. Like normal modules, each individual module in a ScriptModule can
        have submodules, parameters, and methods. In nn.Modules methods are implemented
        as Python functions, but in ScriptModules methods typically implemented as
        *TorchScript* functions,  a statically-typed subset of Python that contains all
        of PyTorch's built-in Tensor operations. This difference allows your
        ScriptModules code to run without the need for a Python interpreter.

        ScriptModules and the TorchScript functions inside of them can be created in
        two ways:

        **Tracing:**

            Using ``torch.jit.trace``, you can take an existing module or python
            function, provide example inputs, and we run the function, recording the
            operations performed on all the tensors. We turn the resulting recording
            into a TorchScript method that is installed as the ``forward`` method of a
            ScriptModule. This module also contains any parameters that the original
            module had as well.

            Example::

                import torch
                def foo(x, y):
                    return 2*x + y
                traced_foo = torch.jit.trace(foo, (torch.rand(3), torch.rand(3)))

            .. note::
                Tracing a *function* will produce a ``ScriptModule`` with a single
                ``forward`` method that implements that function, and that contains
                no parameters.

            Example::

                import torch
                import torchvision
                traced_net = torch.jit.trace(torchvision.models.resnet18(),
                                             torch.rand(1, 3, 224, 224))

            .. note::

                Tracing only records operations done when the given function is run on the given
                tensors. Therefore, the returned ``ScriptModule`` will always run the same traced
                graph on any input. This has some important implications when your module is
                expected to run different sets of operations, depending on the input and/or the
                module state. For example,

                    + Tracing will not record any control-flow like if statements or loops. When
                      this control-flow is constant across your module, this is fine and it often
                      just inlines configuration decisions. But sometimes the control-flow is
                      actually part of the model itself. For instance, a recurrent network is
                      a loop over the (possibly dynamic) length of an input sequence.

                    + In the returned ``ScriptModule``, operations that have different behaviors
                      in ``training`` and ``eval`` modes will always behave as if it is in the
                      mode it was in during tracing, no matter which mode the ``ScriptModule``
                      is in.

                In cases like these, tracing would not be appropriate and scripting is a better
                choice.

        **Scripting:**

            You can write TorchScript code directly using Python syntax. You do this
            using the ``torch.jit.script`` annotation (for functions) or
            ``torch.jit.script_method`` annotation (for methods) on subclasses of
            ScriptModule. With this annotation the body of the annotated function is
            directly translated into TorchScript. TorchScript itself is a subset of
            the Python language, so not all features in python work, but we provide
            enough functionality to compute on tensors and do control-dependent
            operations.

            Example::

                import torch
                @torch.jit.script
                def foo(x, y):
                    if x.max() > y.max():
                        r = x
                    else:
                        r = y
                    return r

            .. note::
                A script *function* annotation will construct a ScriptModule
                with a single ``forward`` method that implements that function,
                and that contains no parameters.

            Example::

              import torch
              class MyModule(torch.jit.ScriptModule):
                  def __init__(self, N, M):
                      super(MyModule, self).__init__()
                      self.weight = torch.nn.Parameter(torch.rand(N, M))

                  @torch.jit.script_method
                  def forward(self, input):
                      return self.weight.mv(input)

            Example::

                import torch
                import torch.nn as nn
                import torch.nn.functional as F
                from torch.jit import ScriptModule, script_method, trace

                class MyScriptModule(ScriptModule):
                    def __init__(self):
                        super(MyScriptModule, self).__init__()
                        # trace produces a ScriptModule's conv1 and conv2
                        self.conv1 = trace(nn.Conv2d(1, 20, 5), torch.rand(1, 1, 16, 16))
                        self.conv2 = trace(nn.Conv2d(20, 20, 5), torch.rand(1, 20, 16, 16))

                    @script_method
                    def forward(self, input):
                      input = F.relu(self.conv1(input))
                      input = F.relu(self.conv2(input))
                      return input
        """

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
                if isinstance(value, Module) and _is_weak_type(type(value)):
                    # Compile weak script module
                    value = _make_strong(value)
                if attr == 'training':
                    if self._has_buffer('training'):
                        self.__dict__['training'] = value
                        self._get_buffer('training').fill_(int(value))
                        return
                if isinstance(value, Attribute):
                    the_type = torch.jit.annotations.ann_to_type(value.type)
                    self._register_attribute(attr, the_type, value.value)
                    return
                return super(ScriptModule, self).__setattr__(attr, value)

            if hasattr(self, attr):
                raise RuntimeError("attempting to re-assign constant '{}'".format(attr))

            def conv_module_to_const(module_value):
                if not isinstance(module_value, (ModuleList, Sequential)):
                    return module_value
                for i in range(len(module_value)):
                    module_value[i] = conv_module_to_const(module_value[i])
                if isinstance(module_value, Sequential):
                    return _ConstSequential(module_value)
                else:
                    return _ConstModuleList(module_value)

            if isinstance(value, (ModuleList, Sequential)):
                # special case for list of modules. Modules need to be registered with their
                # parent module. To do this, we create a ConstModuleList, which is itself a module, that
                # contains each of these modules as submodules. The ConstModuleList then
                # is set as an attribute of the parent module.
                super(ScriptModule, self).__setattr__(attr, conv_module_to_const(value))
            else:
                super(ScriptModule, self).__setattr__(attr, _get_valid_constant(attr, value))

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

        def copy(self):
            m = ScriptModule()

            def module_lookup(names):
                curr = m
                for name in names:
                    if not hasattr(curr, name):
                        setattr(curr, name, ScriptModule())
                    curr = getattr(curr, name)
                return curr
            self._copy_into(module_lookup, {}, [])
            return m

        def __getstate__(self):
            raise pickle.PickleError(
                "ScriptModules cannot be saved using torch.save. " +
                "Mixed serialization of script and non-script modules is not supported. " +
                "For purely script modules use my_script_module.save(<filename>) instead.")

    class WeakScriptModuleProxy(ScriptModule):
        def __init__(self, original, stubs):
            # Guards behavior of __setattr__ and __getattr__ so ScriptModule
            # __init__ can run correctly
            self.__dict__['_initialized'] = False
            super(WeakScriptModuleProxy, self).__init__()

            self.__dict__["_original"] = weakref.ref(original)

            # Copy Parameters / Modules / Buffers
            for name in dir(original):
                item = getattr(original, name)
                if item is None and name in original._parameters:
                    # XXX: treat None value simply as module attributes instead of adding them to the parameter list
                    # TODO: need to handle this more generally when non-tensor attributes added to module
                    object.__setattr__(self, name, item)
                elif isinstance(item, Parameter) or (isinstance(item, Module) and item is not self):
                    ScriptModule.__setattr__(self, name, item)
            for name in original._buffers:
                if original._buffers[name] is None:
                    object.__setattr__(self, name, None)
                else:
                    self.register_buffer(name, original._buffers[name])

            # Copy constants
            self.__dict__["_constants_set"] = set(getattr(original, "__constants__", []))

            # Copy overloads
            self.__dict__["_overloads"] = dict(getattr(original, "__overloads__", {}))

            self.__dict__["_initialized"] = True
            _create_methods_from_stubs(self, stubs)

        def __getattr__(self, attr):
            # Try to get the attribute directly, if that fails, fall back to the
            # weak module itself
            try:
                return ScriptModule.__getattr__(self, attr)
            except AttributeError:
                if self.__dict__["_initialized"]:
                    return getattr(self.__dict__["_original"](), attr)
                else:
                    # Only fall back to original once __init__() is done
                    raise AttributeError("Weak module has no attribute '{}'"
                                         .format(attr))

        def __setattr__(self, attr, value):
            # Once constructed, no new properties can be set

            if not self.__dict__["_initialized"]:
                # If constructing, don't fall back to original module
                return ScriptModule.__setattr__(self, attr, value)

            if hasattr(self, attr):
                return ScriptModule.__setattr__(self, attr, value)
            else:
                raise AttributeError("Cannot set new attribute '{}' on "
                                     "weak script module once it has been "
                                     "created".format(attr))

    class ScriptClass(ScriptModule):
        def __init__(self, name):
            super(ScriptClass, self).__init__()
            self._name = name

else:
    class ScriptModule(torch.nn.Module):  # noqa T484
        def __init__(self, optimize=True):
            super(ScriptModule, self).__init__()

    class ScriptClass(ScriptModule):  # noqa T484
        def __init__(self, name):
            super(ScriptClass, self).__init__()


def _get_weak_stubs(cls):
    """
    Calls script_method for each method on the type of the object passed in and
    returns the generated ScriptMethodStubs
    """
    stubs = []
    for name in dir(cls):
        func = get_function_from_type(cls, name)
        if func in weak_script_methods:
            entry = weak_script_methods[func]
            stub = script_method(entry["original_method"], entry["rcb"])
            stubs.append(stub)
    return stubs


def _make_strong(mod):
    """
    Converts a weak module into a subclass of ScriptModule
    """
    if mod in weak_modules:
        return weak_modules[mod]

    stubs = weak_types.get(type(mod))["method_stubs"]

    if stubs is None:
        # Generate stubs and and store on weak_types in case this type is
        # used again
        stubs = _get_weak_stubs(type(mod))
        weak_types[type(mod)]["method_stubs"] = stubs

    # Create proxy with stubs
    proxy = WeakScriptModuleProxy(mod, stubs)

    weak_modules[mod] = proxy

    return proxy


class _ConstModuleList(ScriptModule):
    def __init__(self, modules):
        super(_ConstModuleList, self).__init__()
        for i, module in enumerate(modules):
            if _is_weak_type(type(module)):
                module = _make_strong(module)
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


class Attribute(object):
    def __init__(self, value, the_type):
        self.value = value
        self.type = the_type
