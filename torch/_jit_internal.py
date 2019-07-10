"""
The weak_script annotation needs to be here instead of inside torch/jit/ so it
can be used in other places in torch/ (namely torch.nn) without running into
circular dependency problems
"""

import inspect
import weakref
from torch._six import builtins

# Wrapper functions that can call either of 2 functions depending on a boolean
# argument
boolean_dispatched = weakref.WeakKeyDictionary()  # noqa: T484


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

    return env


def createResolutionCallbackFromClosure(fn):
    """
    Create a resolutionCallback by introspecting the function instead of
    looking up the stack for the enclosing scope
    """
    var_names = fn.__code__.co_freevars

    # map of captured name -> value
    free_vars = {}

    for index, name in enumerate(var_names):
        free_vars[name] = fn.__closure__[index].cell_contents
    f_globals = fn.__globals__

    def env(key):
        if key in free_vars:
            return free_vars[key]
        elif hasattr(builtins, key):
            return getattr(builtins, key)
        else:
            return f_globals.get(key)

    return env


def boolean_dispatch(arg_name, arg_index, default, if_true, if_false, module_name, func_name):
    """
    Dispatches to either of 2 script functions based on a boolean argument.
    In TorchScript, the boolean argument must be constant so that the correct
    function to use can be determined at compile time.
    """
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



class FunctionModifiers(object):
    """
    Used to denote the behavior of a function in TorchScript. See export() and
    ignore() for details.
    """
    IGNORE_AND_DROP = "ignore (leave as a call to Python, replace with a 'raise' on torch.jit.save)"
    IGNORE = "ignore (leave as a call to Python, cannot be torch.jit.save'd)"
    EXPORT = "export (compile this function even if nothing calls it)"
    DEFAULT = "default (compile if called from a exported function / forward)"


def export(fn):
    """
    This decorator indicates that a method is used as an entry point into a
    ScriptModule. `forward` implicitly is used as an entry point, so it does
    not need this decorator.

    Methods are added to a ScriptModule as they are called in Python. If a
    method is never called, it will not be included in the ScriptModule when
    saving. This decorator explicitly marks that a method should be included
    even if it is not called from Python.
    """
    fn._torchscript_modifier = FunctionModifiers.EXPORT
    return fn


def ignore(drop_on_export=False):
    """
    This decorator indicates to the compiler that a function or method should
    be ignored and left as a Python function.

    With `drop_on_export=False` (the default), calls to this function will
    prevent saving a TorchScript model.

    With `drop_on_export=True`, any calls to this function from other
    TorchScript code will be replaced with a `raise`. This allows you to leave
    code in your TorchScript model that is only ever run when the Python
    interpreter is present.
    """
    if callable(drop_on_export):
        # used without any args, so drop_on_export is actually a function
        #   @torch.jit.ignore
        #   def fn(...):
        fn = drop_on_export
        fn._torchscript_modifier = FunctionModifiers.IGNORE
        return fn

    if isinstance(drop_on_export, bool):
        def decorator(fn):
            if drop_on_export:
                fn._torchscript_modifier = FunctionModifiers.IGNORE_AND_DROP
            else:
                fn._torchscript_modifier = FunctionModifiers.IGNORE
            return fn
        return decorator
    raise RuntimeError("Argument to @torch.jit.ignore must be a bool or "
                       "a function but got {}".format(drop_on_export))


def should_drop_on_export(fn):
    attr = get_torchscript_modifier(fn)
    if attr is None:
        return False
    return attr is FunctionModifiers.IGNORE_AND_DROP


def is_ignored_fn(fn):
    mod = get_torchscript_modifier(fn)
    return mod is FunctionModifiers.IGNORE_AND_DROP or mod is FunctionModifiers.IGNORE


def get_torchscript_modifier(fn):
    if not callable(fn):
        return None
    if hasattr(fn, '__func__'):
        fn = fn.__func__
    return getattr(fn, '_torchscript_modifier', FunctionModifiers.DEFAULT)


def _parameter_list(parameter_names_fn):
    """
    Decorator to denote that a function returns a list of all the parameters
    in a module
    """
    def decorator(fn):
        fn._parameter_names_fn = parameter_names_fn
        return fn

    return decorator


try:
    import typing
    from typing import Tuple, List, Dict, Optional

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

    def is_optional(ann):
        # Optional[T] is just shorthand for Union[T, None], so check for both
        union_optional = False
        if ann.__module__ == 'typing' and \
           (getattr(ann, '__origin__', None) is typing.Union):
            args = getattr(ann, '__args__', ())
            if len(args) == 2:
                union_optional = (issubclass(args[1], type(None)) and not issubclass(args[0], type(None))) \
                    or (issubclass(args[0], type(None)) and not issubclass(args[1], type(None)))

        optional = ann.__module__ == 'typing' and \
            (getattr(ann, '__origin__', None) is typing.Optional)

        return optional or union_optional

except ImportError:
    # A minimal polyfill for versions of Python that don't have typing.
    # Note that this means that they also don't support the fancy annotation syntax, so
    # those instances will only be used in our tiny `type: ` comment interpreter.

    # The __getitem__ in typing is implemented using metaclasses, but I'm too lazy for that.
    class TupleCls(object):
        def __getitem__(self, types):
            return TupleInstance(types)

    class TupleInstance(object):
        __slots__ = ['__args__']

        def __init__(self, types):
            self.__args__ = types

    class ListInstance(object):
        __slots__ = ['__args__']

        def __init__(self, types):
            self.__args__ = types

    class ListCls(object):
        def __getitem__(self, types):
            return TupleInstance(types)

    class DictInstance(object):
        __slots__ = ['__args__']

        def __init__(self, types):
            self.__args__ = types

    class DictCls(object):
        def __getitem__(self, types):
            return DictInstance(types)

    class OptionalInstance(object):
        __slots__ = ['__args__']

        def __init__(self, types):
            self.__args__ = types

    class OptionalCls(object):
        def __getitem__(self, types):
            return OptionalInstance(types)

    Tuple = TupleCls()  # noqa: T484
    List = ListCls()  # noqa: T484
    Dict = DictCls()  # noqa: T484
    Optional = DictCls()  # noqa: T484

    def is_tuple(ann):
        return isinstance(ann, TupleInstance)

    def is_list(ann):
        return isinstance(ann, ListInstance)

    def is_dict(ann):
        return isinstance(ann, DictInstance)

    def is_optional(ann):
        return isinstance(ann, OptionalInstance)


try:
    import typing_extensions
    from typing_extensions import Final

    def is_final(ann):
        return ann.__module__ == 'typing_extensions' and \
            (getattr(ann, '__origin__', None) is typing_extensions.Final)
except ImportError:
    # Same as above, this polyfill is only for `typing_extensions`
    class FinalInstance(object):
        __slots__ = ['__args__']

        def __init__(self, types):
            self.__args__ = types

    class FinalCls(object):
        def __getitem__(self, types):
            return FinalInstance(types)

    Final = FinalCls()  # noqa: T484

    def is_final(ann):
        return isinstance(ann, FinalInstance)


# allows BroadcastingList instance to be subscriptable
class BroadcastingListCls(object):
    def __getitem__(self, types):
        return

# mypy doesn't support parameters on types, so we have to explicitly type each
# list size
BroadcastingList1 = BroadcastingListCls()
for i in range(2, 7):
    globals()["BroadcastingList{}".format(i)] = BroadcastingList1
