"""
The weak_script annotation needs to be here instead of inside torch/jit/ so it
can be used in other places in torch/ (namely torch.nn) without running into
circular dependency problems
"""

import inspect
import weakref
import warnings
import torch._C
from torch._six import builtins
from torch._utils_internal import get_source_lines_and_file

# Wrapper functions that can call either of 2 functions depending on a boolean
# argument
boolean_dispatched = weakref.WeakKeyDictionary()  # noqa: T484


def createResolutionCallbackFromFrame(frames_up=0):
    """
    Creates a function which, given a string variable name,
    returns the value of the variable in the scope of the caller of
    the function which called createResolutionCallbackFromFrame (by default).

    This is used to enable access in-scope Python variables inside
    TorchScript fragments.

    frames_up is number of additional frames to go up on the stack.
    The default value is 0, which correspond to the frame of the caller
    of createResolutionCallbackFromFrame. Also for example, if frames_up is set
    to 1, then the frame of the caller's caller of createResolutionCallbackFromFrame
    will be taken.

    For example, the following program prints 2::

        def bar():
            cb = createResolutionCallbackFromFrame(1)
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


def get_closure(fn):
    """
    Get a dictionary of closed over variables from a function
    """
    captures = {}
    captures.update(fn.__globals__)

    for index, captured_name in enumerate(fn.__code__.co_freevars):
        captures[captured_name] = fn.__closure__[index].cell_contents

    return captures

# [local resolution in python]
# Depending on where a variable is defined, and where it is used, we may
# or may not be able to recover its value when recursively compiling a
# script function. Remember in the general case, a module or function is
# first defined and then later scripted. This means we do not have a
# chance to capture the active frames when the function is defined. Hence any
# name resolution has to happen later on the created closure. The way
# python captures type annotations restricts what we can recover. The
# follow example illustrates the different cases:
#
#         class MyGlobalClass:
#         ...
#         def my_local_scope():
#             @torch.jit.script
#             class MyClass:
#                 ...
#             @torch.jit.script
#             class MyClassUsedAsVar:
#                 ...
#             def eg(x: MyClass, y: MyGlobalClass):
#                 a_local_capture : Foo
#                 return MyClassUsedAsVar(x)
#
# MyGlobalClass is defined in the __globals__ dictionary of function
# 'eg', so it is always recoverable. my_local_scope introduces a new local
# variable scope in the function. Classes defined here are only visible as
# local variables. For the case of MyClassUsedAsVar, it is captured
# because it is used as a variable inside the body of the function, and we
# can resolve it using the captures returned from `get_closure`. However,
# the type annotations are not captured by the closure. In Python
# 3.0--3.9, the _value_ of MyClass and MyGlobalClass will be availiable as
# annotations on `eg``, but starting in Python 4.0, they will represented as
# strings and no longer present. Furthermore, since the body of `eg` does
# not reference those names, they do not appear in the list of closed over
# variables. In Python 2.x, type annotations are in comments, leading to a
# similar situation where their definitions are not available. We anticipate
# that most users will not run into this issue because their modules and
# functions will be defined at a global scope like MyGlobalClass. In cases
# where they are not, it is possible to work around issues by declaring the
# values global in the function.



def createResolutionCallbackFromClosure(fn):
    """
    Create a resolutionCallback by introspecting the function instead of
    looking up the stack for the enclosing scope
    """
    closure = get_closure(fn)

    def env(key):
        if key in closure:
            return closure[key]
        elif hasattr(builtins, key):
            return getattr(builtins, key)
        return None

    return env


def can_compile_class(cls):
    # If any of the functions on a type don't have a code object, this type can't
    # be compiled and is probably a builtin / bound from C
    if is_ignored_fn(cls):
        return False
    fns = [getattr(cls, name) for name in cls.__dict__ if inspect.isroutine(getattr(cls, name))]
    has_code = [hasattr(fn, '__code__') for fn in fns]
    return all(has_code)


def createResolutionCallbackForClassMethods(cls):
    """
    This looks at all the methods defined in a class and pulls their closed-over
    variables into a dictionary and uses that to resolve variables.
    """
    # cls is a type here, so `ismethod` is false since the methods on the type
    # aren't bound to anything, so Python treats them as regular functions
    fns = [getattr(cls, name) for name in cls.__dict__ if inspect.isroutine(getattr(cls, name))]
    captures = {}

    for fn in fns:
        captures.update(get_closure(fn))

    return lambda key: captures.get(key, None)


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
    UNUSED = "unused (ignored and replaced with raising of an exception)"
    IGNORE = "ignore (leave as a call to Python, cannot be torch.jit.save'd)"
    EXPORT = "export (compile this function even if nothing calls it)"
    DEFAULT = "default (compile if called from a exported function / forward)"
    COPY_TO_SCRIPT_WRAPPER = \
        "if this method is not scripted, copy the python method onto the scripted model"


def export(fn):
    """
    This decorator indicates that a method on an ``nn.Module`` is used as an entry point into a
    :class:`ScriptModule` and should be compiled.

    ``forward`` implicitly is assumed to be an entry point, so it does not need this decorator.
    Functions and methods called from ``forward`` are compiled as they are seen
    by the compiler, so they do not need this decorator either.

    Example (using ``@torch.jit.export`` on a method):

    .. testcode::

        import torch
        import torch.nn as nn

        class MyModule(nn.Module):
            def implicitly_compiled_method(self, x):
                return x + 99

            # `forward` is implicitly decorated with `@torch.jit.export`,
            # so adding it here would have no effect
            def forward(self, x):
                return x + 10

            @torch.jit.export
            def another_forward(self, x):
                # When the compiler sees this call, it will compile
                # `implicitly_compiled_method`
                return self.implicitly_compiled_method(x)

            def unused_method(self, x):
                return x - 20

        # `m` will contain compiled methods:
        #     `forward`
        #     `another_forward`
        #     `implicitly_compiled_method`
        # `unused_method` will not be compiled since it was not called from
        # any compiled methods and wasn't decorated with `@torch.jit.export`
        m = torch.jit.script(MyModule())
    """
    fn._torchscript_modifier = FunctionModifiers.EXPORT
    return fn


def unused(fn):
    """
    This decorator indicates to the compiler that a function or method should
    be ignored and replaced with the raising of an exception. This allows you
    to leave code in your model that is not yet TorchScript compatible and still
    export your model.

        Example (using ``@torch.jit.unused`` on a method)::

            import torch
            import torch.nn as nn

            class MyModule(nn.Module):
                def __init__(self, use_memory_efficent):
                    super(MyModule, self).__init__()
                    self.use_memory_efficent = use_memory_efficent

                @torch.jit.unused
                def memory_efficient(self, x):
                    import pdb
                    pdb.set_trace()
                    return x + 10

                def forward(self, x):
                    # Use not-yet-scriptable memory efficient mode
                    if self.use_memory_efficient:
                        return self.memory_efficient(x)
                    else:
                        return x + 10

            m = torch.jit.script(MyModule(use_memory_efficent=False))
            m.save("m.pt")

            m = torch.jit.script(MyModule(use_memory_efficient=True))
            # exception raised
            m(torch.rand(100))
    """
    fn._torchscript_modifier = FunctionModifiers.UNUSED
    return fn

def ignore(drop=False, **kwargs):
    """
    This decorator indicates to the compiler that a function or method should
    be ignored and left as a Python function. This allows you to leave code in
    your model that is not yet TorchScript compatible. Models with ignored
    functions cannot be exported; use torch.jit.unused instead.

    Example (using ``@torch.jit.ignore`` on a method)::

        import torch
        import torch.nn as nn

        class MyModule(nn.Module):
            @torch.jit.ignore
            def debugger(self, x):
                import pdb
                pdb.set_trace()

            def forward(self, x):
                x += 10
                # The compiler would normally try to compile `debugger`,
                # but since it is `@ignore`d, it will be left as a call
                # to Python
                self.debugger(x)
                return x

        m = torch.jit.script(MyModule())

        # Error! The call `debugger` cannot be saved since it calls into Python
        m.save("m.pt")

    Example (using ``@torch.jit.ignore(drop=True)`` on a method):

    .. testcode::

        import torch
        import torch.nn as nn

        class MyModule(nn.Module):
            @torch.jit.ignore(drop=True)
            def training_method(self, x):
                import pdb
                pdb.set_trace()

            def forward(self, x):
                if self.training:
                    self.training_method(x)
                return x

        m = torch.jit.script(MyModule())

        # This is OK since `training_method` is not saved, the call is replaced
        # with a `raise`.
        m.save("m.pt")

    .. testcleanup::

        import os
        os.remove('m.pt')
    """

    if callable(drop):
        # used without any args, so drop is actually a function
        #   @torch.jit.ignore
        #   def fn(...):
        fn = drop
        fn._torchscript_modifier = FunctionModifiers.IGNORE
        return fn

    if not isinstance(drop, bool):
        raise RuntimeError("Argument to @torch.jit.ignore must be a bool or "
                           "a function but got {}".format(drop))

    # for backwards compat
    drop_on_export = kwargs.pop("drop_on_export", None)
    if drop_on_export:
        warnings.warn("ignore(drop_on_export=True) has been deprecated. TorchScript will now drop the function "
                      "call on compilation. Use torch.jit.unused now. {}", category=DeprecationWarning)

        drop = drop_on_export
    elif drop:
        warnings.warn("ignore(True) has been deprecated. TorchScript will now drop the function "
                      "call on compilation. Use torch.jit.unused now. {}", category=DeprecationWarning)

    def decorator(fn):
        if drop:
            fn._torchscript_modifier = FunctionModifiers.UNUSED
        else:
            fn._torchscript_modifier = FunctionModifiers.IGNORE
        return fn
    return decorator


def _copy_to_script_wrapper(fn):
    fn._torchscript_modifier = FunctionModifiers.COPY_TO_SCRIPT_WRAPPER
    return fn

def module_has_exports(mod):
    for name in dir(mod):
        item = getattr(mod, name)
        if callable(item):
            if get_torchscript_modifier(item) is FunctionModifiers.EXPORT:
                return True
    return False

def should_drop(fn):
    attr = get_torchscript_modifier(fn)
    if attr is None:
        return False
    return attr is FunctionModifiers.UNUSED


def is_ignored_fn(fn):
    mod = get_torchscript_modifier(fn)
    return mod is FunctionModifiers.UNUSED or mod is FunctionModifiers.IGNORE

def get_torchscript_modifier(fn):
    if not callable(fn):
        return None
    if hasattr(fn, '__func__'):
        fn = fn.__func__
    return getattr(fn, '_torchscript_modifier', FunctionModifiers.DEFAULT)

def copy_torchscript_modifier(orig, new):
    attr = get_torchscript_modifier(orig)
    if attr is None:
        return
    new._torchscript_modifier = attr

# overloading registration
# overloads get registered in this file, and compiled in torch/jit/__init__.py
# so that they can be imported in nn/functional.py without an import cycle

# qualified_name => list[overload_functions]
_overloaded_fns = {}  # noqa: T484

def _overload(func):
    qual_name = _qualified_name(func)
    global _overloaded_fns
    fn_overload_list = _overloaded_fns.get(qual_name)
    if fn_overload_list is None:
        fn_overload_list = []
        _overloaded_fns[qual_name] = fn_overload_list
    fn_overload_list.append(func)
    return func

def _get_fn_overloads(qual_name):
    return _overloaded_fns.get(qual_name)

def _clear_fn_overloads(qual_name):
    del _overloaded_fns[qual_name]

def get_class_name_lineno(method):
    current_frame = inspect.currentframe()

    # one for the get_class_name call, one for _overload_method call
    for i in range(2):
        current_frame = current_frame.f_back
    class_name = current_frame.f_code.co_name
    line_no = current_frame.f_code.co_firstlineno
    return class_name, line_no

# At the the point the decorator is applied to class methods the method
# has no reference to its owning class. _qualified_name would not include
# the class it is defined in, so any methods with the same name in the same file
# would have the same _qualified_name, even if they were defined in different
# classes. This problem only exists in python 2.
# We get around this problem by looking at the stack frame and identifying
# the class name, and throwing an error whenever overloads are used
# when modules of the same name are in the same file

# qualified_name => class name => list[overload_functions]
_overloaded_methods = {}  # noqa: T484


# (qualified_name, class name) => class_fileno
_overloaded_method_class_fileno = {}

def _overload_method(func):
    qual_name = _qualified_name(func)
    global _overloaded_methods
    class_name_map = _overloaded_methods.get(qual_name, None)
    if class_name_map is None:
        class_name_map = {}
        _overloaded_methods[qual_name] = class_name_map

    class_name, line_no = get_class_name_lineno(func)
    method_overloads = class_name_map.get(class_name, None)
    if method_overloads is None:
        method_overloads = []
        class_name_map[class_name] = method_overloads
        _overloaded_method_class_fileno[(qual_name, class_name)] = line_no
    else:
        existing_lineno = _overloaded_method_class_fileno[(qual_name, class_name)]
        if existing_lineno != line_no:
            raise RuntimeError("Cannot currently overload the same method name in two different"
                               " classes with the same name in the same module")

    method_overloads.append(func)
    return func

def _get_overloaded_methods(method, mod_class):
    # TODO: __name__ not set for submodules in recursive script
    if not hasattr(method, "__name__"):
        return None
    qual_name = _qualified_name(method)
    class_name_map = _overloaded_methods.get(qual_name, None)
    if class_name_map is None:
        return None
    overloads = class_name_map.get(mod_class.__name__, None)
    if overloads is None:
        return None

    method_line_no = get_source_lines_and_file(method)[1]
    mod_class_fileno = get_source_lines_and_file(mod_class)[1]
    mod_end_fileno = mod_class_fileno + len(get_source_lines_and_file(mod_class)[0])
    if not (method_line_no >= mod_class_fileno and method_line_no <= mod_end_fileno):
        raise Exception("Overloads are not useable when a module is redeclared within the same file: " + str(method))
    return overloads

try:
    import typing
    from typing import Tuple, List, Dict, Optional, Any

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
        def safe_is_subclass(the_type, super_type):
            # Don't throw if `the_type` isn't a class type (e.g. if it is
            # another type annotation instance)
            if not inspect.isclass(the_type):
                return False
            return issubclass(the_type, super_type)

        union_optional = False
        if ann.__module__ == 'typing' and \
           (getattr(ann, '__origin__', None) is typing.Union):
            args = getattr(ann, '__args__', ())
            if len(args) == 2:
                union_optional = (safe_is_subclass(args[1], type(None)) and not safe_is_subclass(args[0], type(None))) \
                    or (safe_is_subclass(args[0], type(None)) and not safe_is_subclass(args[1], type(None)))

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

    class AnyCls(object):
        pass

    Tuple = TupleCls()  # noqa: T484
    List = ListCls()  # noqa: T484
    Dict = DictCls()  # noqa: T484
    Optional = DictCls()  # noqa: T484
    Any = AnyCls()  # noqa: T484

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

# Retrieves a fully-qualified name (module hierarchy + classname) for a given obj.
def _qualified_name(obj):
    # short-circuit in cases where the object already has a known qualified name
    if isinstance(obj, torch.jit.ScriptFunction):
        return obj.qualified_name

    name = obj.__name__
    if name == '<lambda>':
        name = '_lambda'  # make name a valid identifier

    module_name = obj.__module__

    # If the module is actually a torchbind module, then we should short circuit
    if module_name == "torch._classes":
        return obj.qualified_name

    # The Python docs are very clear that `__module__` can be None, but I can't
    # figure out when it actually would be.
    if module_name is None:
        raise RuntimeError("Could not get qualified name for class '{}': "
                           "__module__ can't be None.".format(name))

    # if getattr(sys.modules[module_name], name) is not obj:
    #     raise RuntimeError("Could not get qualified name for class '{}': "
    #                        "the attr {} on module {} is not the the class".format(name, name, module_name))

    # __main__ is a builtin module, so rewrite it to "__torch__".
    if module_name == "__main__":
        module_name = "__torch__"
    else:
        # Everything else gets a "__torch__" prefix to avoid name collisions
        # with the names of user values.
        module_name = "__torch__." + module_name

    if "." in name:
        raise RuntimeError("Could not get qualified name for class '{}': "
                           "'{}' is not a valid identifier".format(name, name))

    return module_name + "." + name
