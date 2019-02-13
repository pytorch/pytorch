"""
The weak_script annotation needs to be here instead of inside torch/jit/ so it
can be used in other places in torch/ (namely torch.nn) without running into
circular dependency problems
"""

import weakref
import inspect
from torch._six import builtins

# Tracks standalone weak script functions
compiled_weak_fns = weakref.WeakKeyDictionary()

# Tracks which methods should be converted to strong methods
weak_script_methods = weakref.WeakKeyDictionary()

# Converted modules and their corresponding WeakScriptModuleProxy objects
weak_modules = weakref.WeakKeyDictionary()

# Types that have been declared as weak modules
weak_types = weakref.WeakKeyDictionary()

# Wrapper functions that can call either of 2 functions depending on a boolean
# argument
boolean_dispatched = weakref.WeakKeyDictionary()

# Python Op functions that should be ignored by the compiler. These will be replaced
# with an operator that always throws an error
ignored_fns = weakref.WeakSet()

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

    Tuple = TupleCls()
    List = ListCls()
    Dict = DictCls()

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
