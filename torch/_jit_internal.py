"""
The weak_script annotation needs to be here instead of inside torch/jit/ so it
can be used in other places in torch/ (namely torch.nn) without running into
circular dependency problems
"""

import ast
import builtins
import collections
import contextlib
import enum
import inspect
import io
import pickle
import sys
import threading
import typing
import warnings
import weakref
from textwrap import dedent
from typing import (  # noqa: F401
    Any,
    Callable,
    Dict,
    Final,
    ForwardRef,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import torch

# This is needed. `torch._jit_internal` is imported before `torch.distributed.__init__`.
# Explicitly ask to import `torch.distributed.__init__` first.
# Otherwise, "AttributeError: module 'torch' has no attribute 'distributed'" is raised.
import torch.distributed.rpc
import torch.package._mangling as package_mangling
from torch._awaits import _Await
from torch._C import _Await as CAwait, Future as CFuture
from torch._sources import fake_range, get_source_lines_and_file, parse_def
from torch.futures import Future

LockType: Type
try:
    import _thread

    LockType = _thread.LockType
except ImportError:
    import _dummy_thread

    LockType = _dummy_thread.LockType

# Wrapper functions that can call either of 2 functions depending on a boolean
# argument
boolean_dispatched: "weakref.WeakKeyDictionary[Callable, Dict[str, Callable]]" = (
    weakref.WeakKeyDictionary()
)  # noqa: T484


FAKE_FILENAME_PREFIX = "__torch_jit_dataclass"


class SourceLoader:
    def __init__(self):
        self.content = {}

    def cache(self, fn, source):
        self.content[fn] = source

    def get_source(self, fn):
        return self.content.get(fn)


loader = SourceLoader()


IS_PY39_PLUS = sys.version_info >= (3, 9)


def createResolutionCallbackFromEnv(lookup_base):
    """
    Creates a resolution callback that will look up qualified names in an
    environment, starting with `lookup_base` for the base of any qualified
    names, then proceeding down the lookup chain with the resolved object.

    You should not use this directly, it should only be used from the other
    createResolutionCallbackFrom* functions.
    """

    def lookupInModule(qualified_name, module):
        if "." in qualified_name:
            parts = qualified_name.split(".")
            base = parts[0]
            remaining_pieces = ".".join(parts[1:])
            module_value = getattr(module, base)
            return lookupInModule(remaining_pieces, module_value)
        else:
            return getattr(module, qualified_name)

    def parseNestedExpr(expr, module) -> Tuple[Any, int]:
        i = 0
        while i < len(expr) and expr[i] not in (",", "[", "]"):
            i += 1

        # Special case logic for the empty Tuple as a subscript (used
        # in the type annotation `Tuple[()]`)
        if expr[:i] == "()":
            return (), i

        base = lookupInModule(expr[:i].strip(), module)
        assert base is not None, f"Unresolvable type {expr[:i]}"
        if i == len(expr) or expr[i] != "[":
            return base, i

        assert expr[i] == "["
        parts = []
        while expr[i] != "]":
            part_len = 0
            i += 1
            part, part_len = parseNestedExpr(expr[i:], module)
            parts.append(part)
            i += part_len
        if len(parts) > 1:
            return base[tuple(parts)], i + 1
        else:
            return base[parts[0]], i + 1

    def parseExpr(expr, module):
        try:
            value, len_parsed = parseNestedExpr(expr, module)
            assert len_parsed == len(
                expr
            ), "whole expression was not parsed, falling back to c++ parser"
            return value
        except Exception:
            """
            The python resolver fails in several cases in known unit tests, and is intended
            to fall back gracefully to the c++ resolver in general.  For example, python 2 style
            annotations which are frequent in our unit tests often fail with types e.g. int not
            resolvable from the calling frame.
            """
            return None

    return lambda expr: parseExpr(expr, lookup_base)


def createResolutionCallbackFromFrame(frames_up: int = 0):
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
        assert frame is not None
        frame = frame.f_back
        i += 1

    assert frame is not None
    f_locals = frame.f_locals
    f_globals = frame.f_globals

    class env:
        def __getattr__(self, key):
            if key in f_locals:
                return f_locals[key]
            elif key in f_globals:
                return f_globals[key]
            elif key in dir(builtins):
                return getattr(builtins, key)

    return createResolutionCallbackFromEnv(env())


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
# 3.0--3.9, the _value_ of MyClass and MyGlobalClass will be available as
# annotations on `eg``, but starting in Python 4.0, they will represented as
# strings and no longer present. Furthermore, since the body of `eg` does
# not reference those names, they do not appear in the list of closed over
# variables. In Python 2.x, type annotations are in comments, leading to a
# similar situation where their definitions are not available. We anticipate
# that most users will not run into this issue because their modules and
# functions will be defined at a global scope like MyGlobalClass. In cases
# where they are not, it is possible to work around issues by declaring the
# values global in the function.
# In Python 3.9 declaring class as global will make it invisible to
# `inspect.getsource`, see https://bugs.python.org/issue42666 .
# This could be worked around by manualy adding it to `global()` dictionary.


def createResolutionCallbackFromClosure(fn):
    """
    Create a resolutionCallback by introspecting the function instead of
    looking up the stack for the enclosing scope
    """
    closure = get_closure(fn)

    class closure_lookup:
        # This is a class since `closure` is a dict and it's easier in
        # `env_helper` if everything just works with `getattr` calls
        def __getattr__(self, key):
            if key in closure:
                return closure[key]
            elif hasattr(typing, key):
                return getattr(typing, key)
            elif hasattr(builtins, key):
                return getattr(builtins, key)
            return None

    return createResolutionCallbackFromEnv(closure_lookup())


def can_compile_class(cls) -> bool:
    # If any of the functions on a type don't have a code object, this type can't
    # be compiled and is probably a builtin / bound from C
    if is_ignored_fn(cls):
        return False

    # Ignore the following list of built-in classes.
    ignored_builtin_classes = (torch.nn.Module, tuple, list, Exception)
    if issubclass(cls, ignored_builtin_classes):
        return False

    names = cls.__dict__
    fns = [
        getattr(cls, name)
        for name in names
        if inspect.isroutine(getattr(cls, name, None))
    ]
    has_code = [hasattr(fn, "__code__") for fn in fns]
    return all(has_code)


def get_callable_argument_names(fn) -> List[str]:
    """
    Gets names of all POSITIONAL_OR_KEYWORD arguments for callable `fn`.
    Returns an empty list when other types of arguments are present.

    This is used by `torch.jit.trace` to assign meaningful argument names to
    traced functions and modules.

    Args:
        fn: A callable.
    Returns:
        Argument names: List[str]
    """
    # inspect.signature may fail, give up in that case.
    try:
        callable_signature = inspect.signature(fn)
    except Exception:
        return []

    argument_names = []
    for name, param in callable_signature.parameters.items():
        # All four other types of arguments do not map to individual values
        # with a keyword as name.
        if not param.kind == param.POSITIONAL_OR_KEYWORD:
            continue

        argument_names.append(name)

    return argument_names


def get_annotation_str(annotation):
    """
    Convert an AST node containing a type annotation to the string present in the source
    that represents the same annotation.
    """
    if isinstance(annotation, ast.Name):
        return annotation.id
    elif isinstance(annotation, ast.Attribute):
        return ".".join([get_annotation_str(annotation.value), annotation.attr])
    elif isinstance(annotation, ast.Subscript):
        # In Python3.9+ subscript indicies are not wrapped in ast.Index
        subscript_slice = annotation.slice if IS_PY39_PLUS else annotation.slice.value  # type: ignore[attr-defined]
        return f"{get_annotation_str(annotation.value)}[{get_annotation_str(subscript_slice)}]"
    elif isinstance(annotation, ast.Tuple):
        return ",".join([get_annotation_str(elt) for elt in annotation.elts])
    elif isinstance(annotation, (ast.Constant, ast.NameConstant)):
        return f"{annotation.value}"

    # If an AST node is not handled here, it's probably handled in ScriptTypeParser.
    return None


def get_type_hint_captures(fn):
    """
    Get a dictionary containing type resolution mappings necessary to resolve types
    for the literal annotations on 'fn'. These are not considered to be closed-over by fn
    and must be obtained separately (e.g. using this function).

    Args:
        fn: A callable.
    Returns:
        A Dict[str, Any] containing a mapping from the literal annotations used on
        fn to the Python objects they refer to.
    """
    # First, try to get the source of the function. We'll need to parse it to find the actual string names
    # that were used to annotate the types, since inspect.signature() will only return the class object that
    # the annotation refers to, not the string name. If we can't get the source, simply return an empty dict.
    # This may happen in cases where the function is synthesized dynamically at runtime.
    src = loader.get_source(fn)
    if src is None:
        src = inspect.getsource(fn)

    # Gather a dictionary of parameter name -> type, skipping any parameters whose annotated
    # types are strings. These are only understood by TorchScript in the context of a type annotation
    # that refers to a class in its own definition, but trying to include a mapping for this in the result
    # function would cause infinite recursion because the class is currently being compiled.
    # In addition, there is logic in ScriptTypeParser to handle this.
    signature = inspect.signature(fn)
    name_to_type = {
        name: parameter.annotation
        for name, parameter in signature.parameters.items()
        if parameter.annotation is not inspect.Parameter.empty
        and not isinstance(parameter.annotation, str)
    }

    # Then, get the literal type annotations from the function declaration
    # by source inspection. This accounts for the case in which aliases are used
    # to annotate the arguments (e.g device_t = torch.device, and then d: device_t).
    # frontend.py cannot be used here because it includes _jit_internal, so use ast instead.
    a = ast.parse(dedent(src))
    if len(a.body) != 1 or not isinstance(a.body[0], ast.FunctionDef):
        raise RuntimeError(f"Expected {fn} to be a function")
    f = a.body[0]

    # Prepare a dictionary of source annotation -> type, which will be the final result of this function,
    # by using the parsed AST (f) to reconstruct source annotations as strings for each parameter and mapping
    # them to the type object corresponding to the annotation via name_to_type using the parameter name.
    annotation_to_type = {}

    for arg in f.args.args:
        # Get the source type annotation string for this argument if possible.
        arg_annotation_str = (
            get_annotation_str(arg.annotation) if arg.annotation else None
        )

        # If the argument has no annotation or get_annotation_str cannot convert it to a string,
        # arg_annotation_str will be None. Skip this arg; ScriptTypeParser will probably handle
        # this in the latter case.
        if arg_annotation_str is None:
            continue

        # Insert {arg_annotation_str: type} into annotation_to_type if possible. One reason arg_name may not
        # be present in name_to_type is that the annotation itself is a string and not a type object
        # (common for self-refential annotations in classes). Once again, let ScriptTypeParser handle this.
        arg_name = arg.arg
        if arg_name in name_to_type:
            annotation_to_type[arg_annotation_str] = name_to_type[arg_name]

    # If there is a valid return annotation, include it in annotation_to_type. As with argument annotations,
    # the literal annotation has to be convertible to a string by get_annotation_str, and the actual type
    # of the annotation cannot be a string.
    literal_return_annotation = get_annotation_str(f.returns)
    valid_literal_annotation = literal_return_annotation is not None
    return_annotation = signature.return_annotation
    valid_return_annotation_type = (
        return_annotation is not inspect.Parameter.empty
        and not isinstance(return_annotation, str)
    )
    if valid_literal_annotation and valid_return_annotation_type:
        annotation_to_type[literal_return_annotation] = return_annotation

    return annotation_to_type


def createResolutionCallbackForClassMethods(cls):
    """
    This looks at all the methods defined in a class and pulls their closed-over
    variables into a dictionary and uses that to resolve variables.
    """
    # cls is a type here, so `ismethod` is false since the methods on the type
    # aren't bound to anything, so Python treats them as regular functions
    fns = [
        getattr(cls, name)
        for name in cls.__dict__
        if inspect.isroutine(getattr(cls, name))
    ]
    # Skip built-ins, as they do not have global scope nor type hints
    # Needed to support `enum.Enum` derived classes in Python-3.11
    # That adds `_new_member_` property which is an alias to `__new__`
    fns = [fn for fn in fns if not inspect.isbuiltin(fn) and hasattr(fn, "__globals__")]
    captures = {}

    for fn in fns:
        captures.update(get_closure(fn))
        captures.update(get_type_hint_captures(fn))

    def lookup_in_class(key):
        if key in captures:
            return captures[key]
        else:
            return getattr(builtins, key, None)

    return lookup_in_class


def boolean_dispatch(
    arg_name, arg_index, default, if_true, if_false, module_name, func_name
):
    """
    Dispatches to either of 2 script functions based on a boolean argument.
    In TorchScript, the boolean argument must be constant so that the correct
    function to use can be determined at compile time.
    """

    def fn(*args, **kwargs):
        dispatch_flag = default
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
        "arg_name": arg_name,
    }
    return fn


class FunctionModifiers:
    """
    Used to denote the behavior of a function in TorchScript. See export() and
    ignore() for details.
    """

    UNUSED = "unused (ignored and replaced with raising of an exception)"
    IGNORE = "ignore (leave as a call to Python, cannot be torch.jit.save'd)"
    EXPORT = "export (compile this function even if nothing calls it)"
    DEFAULT = "default (compile if called from a exported function / forward)"
    COPY_TO_SCRIPT_WRAPPER = (
        "if this method is not scripted, copy the python method onto the scripted model"
    )
    _DROP = "_drop (function is fully ignored, declaration can be unscriptable)"


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
                def __init__(self, use_memory_efficient):
                    super().__init__()
                    self.use_memory_efficient = use_memory_efficient

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

            m = torch.jit.script(MyModule(use_memory_efficient=False))
            m.save("m.pt")

            m = torch.jit.script(MyModule(use_memory_efficient=True))
            # exception raised
            m(torch.rand(100))
    """
    if isinstance(fn, property):
        prop = fn
        setattr(  # noqa: B010
            prop.fget, "_torchscript_modifier", FunctionModifiers.UNUSED
        )

        if prop.fset:
            setattr(  # noqa: B010
                prop.fset, "_torchscript_modifier", FunctionModifiers.UNUSED
            )

        return prop

    fn._torchscript_modifier = FunctionModifiers.UNUSED
    return fn


# No op context manager from python side
class _IgnoreContextManager(contextlib.AbstractContextManager):
    def __init__(self, **kwargs):
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        pass


def ignore(drop=False, **kwargs):
    """
    This decorator indicates to the compiler that a function or method should
    be ignored and left as a Python function. This allows you to leave code in
    your model that is not yet TorchScript compatible. If called from TorchScript,
    ignored functions will dispatch the call to the Python interpreter. Models with ignored
    functions cannot be exported; use :func:`@torch.jit.unused <torch.jit.unused>` instead.

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
        raise RuntimeError(
            "Argument to @torch.jit.ignore must be a bool or "
            f"a function but got {drop}"
        )

    # for backwards compat
    drop_on_export = kwargs.pop("drop_on_export", None)
    if drop_on_export:
        warnings.warn(
            "ignore(drop_on_export=True) has been deprecated. TorchScript will now drop the function "
            "call on compilation. Use torch.jit.unused now. {}",
            category=FutureWarning,
        )

        drop = drop_on_export
    elif drop:
        warnings.warn(
            "ignore(True) has been deprecated. TorchScript will now drop the function "
            "call on compilation. Use torch.jit.unused now. {}",
            category=FutureWarning,
        )

    def decorator(fn):
        if drop:
            fn._torchscript_modifier = FunctionModifiers.UNUSED
        else:
            fn._torchscript_modifier = FunctionModifiers.IGNORE
        return fn

    return decorator


def _drop(fn):
    fn._torchscript_modifier = FunctionModifiers._DROP
    return fn


def _copy_to_script_wrapper(fn):
    fn._torchscript_modifier = FunctionModifiers.COPY_TO_SCRIPT_WRAPPER
    return fn


def module_has_exports(mod):
    for name in dir(mod):
        if hasattr(mod, name):
            item = getattr(mod, name)
            if callable(item):
                if get_torchscript_modifier(item) is FunctionModifiers.EXPORT:
                    return True
    return False


# WARNING: should_drop is currently being used by our JIT code coverage plug-in to mark JIT'd code as covered. If you
# rename this function, please update references in tools/coverage_plugins_package/src/coverage_plugins/jit_plugin.py to
# allow JIT'd code to still be covered.
def should_drop(fn) -> bool:
    attr = get_torchscript_modifier(fn)
    if attr is None:
        return False
    return attr is FunctionModifiers.UNUSED or attr is FunctionModifiers._DROP


def is_ignored_fn(fn) -> bool:
    mod = get_torchscript_modifier(fn)
    return (
        mod is FunctionModifiers.UNUSED
        or mod is FunctionModifiers.IGNORE
        or mod is FunctionModifiers._DROP
    )


def _is_drop_fn(fn) -> bool:
    mod = get_torchscript_modifier(fn)
    return mod is FunctionModifiers._DROP


def is_static_fn(cls, fn) -> bool:
    return isinstance(inspect.getattr_static(cls, fn, default=None), staticmethod)


def get_static_fn(cls, fn):
    return inspect.getattr_static(cls, fn).__func__


def get_torchscript_modifier(fn):
    if not callable(fn):
        return None
    if hasattr(fn, "__func__"):
        fn = fn.__func__
    return getattr(fn, "_torchscript_modifier", FunctionModifiers.DEFAULT)


def copy_torchscript_modifier(orig, new) -> None:
    attr = get_torchscript_modifier(orig)
    if attr is None:
        return
    new._torchscript_modifier = attr


# overloading registration
# overloads get registered in this file, and compiled in torch/jit/__init__.py
# so that they can be imported in nn/functional.py without an import cycle

# qualified_name => list[overload_functions]
_overloaded_fns: Dict[str, List[Callable]] = {}  # noqa: T484


_OVERLOAD_EXAMPLE = """
Example usage of overload function:
@torch.jit._overload
def my_function(x: type0) -> type0: # decl 1
    pass

@torch.jit._overload
def my_function(x: type1) -> type1: # decl 2
    pass

def my_function(x):                 # implementation
    if isinstance(x, type0):
        return x
    elif isinstance(x, type1):
        return x
"""


def get_overload_no_implementation_error_message(kind, obj):
    sourcelines, file_lineno, filename = get_source_lines_and_file(obj)
    return (
        f'Implementation for the {kind} "{_qualified_name(obj)}" is missing. Please make '
        f"sure a definition is provided and defined after all overload declarations.\n"
        f'File "{filename}", line {file_lineno}:\n'
        + "".join(sourcelines)
        + "\n"
        + _OVERLOAD_EXAMPLE
    )


def _check_overload_body(func):
    try:
        parsed_def = parse_def(func)
    except OSError as e:
        # Parsing the function definition can raise an OSError if source is unavailable.
        # Since this is just an initial check, just raise a warning if this is the case.
        warnings.warn(
            f"Unable to retrieve source for @torch.jit._overload function: {func}."
        )
        return

    body = parsed_def.ast.body[0].body

    def is_pass(x):
        return isinstance(x, ast.Pass)

    def is_ellipsis(x):
        return isinstance(x, ast.Expr) and isinstance(x.value, ast.Ellipsis)

    if len(body) != 1 or not (is_pass(body[0]) or is_ellipsis(body[0])):
        msg = (
            "Only `pass` statement or `...` can be the body of overload declaration:\n"
        )
        msg += "\n".join(parsed_def.source.split("\n")[:3])
        msg += " <- Expecting `pass` or `...` here!\n" + _OVERLOAD_EXAMPLE
        raise RuntimeError(msg)


def _overload(func):
    _check_overload_body(func)
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


def _clear_fn_overloads(qual_name) -> None:
    del _overloaded_fns[qual_name]


def get_class_name_lineno(method) -> Tuple[str, int]:
    current_frame = inspect.currentframe()

    # one for the get_class_name call, one for _overload_method call
    for i in range(2):
        assert (
            current_frame is not None
        )  # assert current frame is not an Optional[FrameType]
        current_frame = current_frame.f_back

    assert current_frame is not None  # same here
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
_overloaded_methods: Dict[str, Dict[str, List[Callable]]] = {}  # noqa: T484


# (qualified_name, class name) => class_fileno
_overloaded_method_class_fileno = {}


def _overload_method(func):
    _check_overload_body(func)
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
            raise RuntimeError(
                "Cannot currently overload the same method name in two different"
                " classes with the same name in the same module"
            )

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
        raise Exception(
            "Overloads are not useable when a module is redeclared within the same file: "
            + str(method)
        )
    return overloads


def is_tuple(ann) -> bool:
    if ann is Tuple:
        raise_error_container_parameter_missing("Tuple")

    # For some reason Python 3.7 violates the Type[A, B].__origin__ == Type rule
    if not hasattr(ann, "__module__"):
        return False

    ann_origin = getattr(ann, "__origin__", None)
    if IS_PY39_PLUS and ann.__module__ == "builtins" and ann_origin is tuple:
        return True
    return ann.__module__ == "typing" and (ann_origin is Tuple or ann_origin is tuple)


def is_list(ann) -> bool:
    if ann is List:
        raise_error_container_parameter_missing("List")

    if not hasattr(ann, "__module__"):
        return False

    ann_origin = getattr(ann, "__origin__", None)
    if IS_PY39_PLUS and ann.__module__ == "builtins" and ann_origin is list:
        return True
    return ann.__module__ == "typing" and (ann_origin is List or ann_origin is list)


def is_dict(ann) -> bool:
    if ann is Dict:
        raise_error_container_parameter_missing("Dict")

    if not hasattr(ann, "__module__"):
        return False

    ann_origin = getattr(ann, "__origin__", None)
    if IS_PY39_PLUS and ann.__module__ == "builtins" and ann_origin is dict:
        return True
    return ann.__module__ == "typing" and (ann_origin is Dict or ann_origin is dict)


def is_union(ann):
    if ann is Union:
        raise_error_container_parameter_missing("Union")

    return (
        hasattr(ann, "__module__")
        and ann.__module__ == "typing"
        and (getattr(ann, "__origin__", None) is Union)
    )


def is_optional(ann):
    if ann is Optional:
        raise_error_container_parameter_missing("Optional")

    def is_optional_as_optional(ann):
        return (
            hasattr(ann, "__module__")
            and ann.__module__ == "typing"
            and (getattr(ann, "__origin__", None) is Optional)
        )

    def is_union_as_optional(ann):
        ann_args = ann.__args__
        return len(ann_args) == 2 and (None in ann_args or type(None) in ann_args)

    return is_optional_as_optional(ann) or (is_union(ann) and is_union_as_optional(ann))


def is_future(ann) -> bool:
    if ann is Future:
        raise RuntimeError(
            "Attempted to use Future without a "
            "contained type. Please add a contained type, e.g. "
            "Future[int]"
        )
    return getattr(ann, "__origin__", None) is Future


def is_await(ann) -> bool:
    if ann is _Await:
        return True
    return getattr(ann, "__origin__", None) is _Await


if torch.distributed.rpc.is_available():
    from torch._C._distributed_rpc import PyRRef
    from torch.distributed.rpc import RRef

    def is_rref(ann) -> bool:
        if ann is RRef:
            raise RuntimeError(
                "Attempted to use RRef without a "
                "contained type. Please add a contained type, e.g. "
                "RRef[int]"
            )
        return getattr(ann, "__origin__", None) is RRef

    def is_rref_instance(obj) -> bool:
        return isinstance(obj, PyRRef)

else:

    def is_rref_instance(obj) -> bool:
        # If the RPC module doesn't exist then RRefs don't exist either.
        return False


def is_final(ann) -> bool:
    return ann.__module__ in {"typing", "typing_extensions"} and (
        getattr(ann, "__origin__", None) is Final or isinstance(ann, type(Final))
    )


# allows BroadcastingList instance to be subscriptable
class BroadcastingListCls:
    def __getitem__(self, types):
        return


# mypy doesn't support parameters on types, so we have to explicitly type each
# list size
BroadcastingList1 = BroadcastingListCls()
for i in range(2, 7):
    globals()[f"BroadcastingList{i}"] = BroadcastingList1


def is_scripting() -> bool:
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
           if torch.jit.is_scripting():
              return torch.linear(x)
           else:
              return unsupported_linear_op(x)
    """
    return False


# Retrieves a fully-qualified name (module hierarchy + classname) for a given obj.
def _qualified_name(obj, mangle_name=True) -> str:
    # This special case allows us to override the qualified name on a type.
    # It's currently used in conjunction with tracing, where we create a
    # fake module to filter only supported attributes. However, since this
    # new type is defined as a local class, we need a mechanism to override
    # its qualname so it appears correctly in the TorchScript system. This,
    # we set '_jit_override_qualname' with the original traced module's
    # qualified name, which is picked up here
    if hasattr(obj, "_jit_override_qualname"):
        return obj._jit_override_qualname
    # short-circuit in cases where the object already has a known qualified name
    if isinstance(obj, torch._C.ScriptFunction):
        return obj.qualified_name

    if getattr(obj, "__name__", None):
        name = obj.__name__
    # Enum classes do not have `__name__` attr, instead they have `name`.
    elif isinstance(obj, enum.Enum):
        name = obj.name
    else:
        raise RuntimeError("Could not get name of python class object")

    if name == "<lambda>":
        name = "_lambda"  # make name a valid identifier

    module_name = obj.__module__

    # If the module is actually a torchbind module, then we should short circuit
    if module_name == "torch._classes":
        return obj.qualified_name

    # The Python docs are very clear that `__module__` can be None, but I can't
    # figure out when it actually would be.
    if module_name is None:
        raise RuntimeError(
            f"Could not get qualified name for class '{name}': "
            "__module__ can't be None."
        )

    # if getattr(sys.modules[module_name], name) is not obj:
    #     raise RuntimeError(f"Could not get qualified name for class '{name}': "
    #                        f"the attr {name} on module {module_name} is not the the class")

    # torch.package and TorchScript have separate mangling schemes to avoid
    # name collisions from multiple packages. To avoid them interfering with
    # each other, normalize the package manging here.
    if package_mangling.is_mangled(module_name):
        module_name = module_name.replace("<", "_")
        module_name = module_name.replace(">", "_")

    # The PythonExceptionValue C++ class in torch/csrc/jit/python/python_sugared_value.h
    # does not need mangle the python class name.
    if mangle_name:
        # __main__ is a builtin module, so rewrite it to "__torch__".
        if module_name == "__main__":
            module_name = "__torch__"
        else:
            # Everything else gets a "__torch__" prefix to avoid name collisions
            # with the names of user values.
            module_name = "__torch__." + module_name

    if "." in name:
        raise RuntimeError(
            f"Could not get qualified name for class '{name}': "
            f"'{name}' is not a valid identifier"
        )

    return module_name + "." + name


def _try_get_dispatched_fn(fn):
    if not callable(fn):
        return None
    return boolean_dispatched.get(fn)


def _get_named_tuple_properties(
    obj, loc: Optional[torch._C._jit_tree_views.SourceRange] = None, rcb=None
):
    if loc is None:
        loc = fake_range()

    assert issubclass(obj, tuple) and hasattr(obj, "_fields")
    if hasattr(obj, "_field_defaults"):
        defaults = [
            obj._field_defaults[field]
            for field in obj._fields
            if field in obj._field_defaults
        ]
    else:
        defaults = []
    # In 3.10 recommended way to get annotations is to call `inspect.get_annotations` function
    # Also, annotations from base class are not inherited so they need to be queried explicitly
    if sys.version_info[:2] < (3, 10):
        obj_annotations = getattr(obj, "__annotations__", {})
    else:
        obj_annotations = inspect.get_annotations(obj)
        if len(obj_annotations) == 0 and hasattr(obj, "__base__"):
            obj_annotations = inspect.get_annotations(obj.__base__)

    annotations = []
    for field in obj._fields:
        if field in obj_annotations:
            field_type = obj_annotations[field]
            # [Note: ForwardRef annotations in NamedTuple attributes]
            # NamedTuple types are slightly different from normal types.
            #
            # Normally, annotations are evaluted like this (during jit.script):
            # 1. Load strings of python code into c++ and parse.
            # 2. Get annotations as strings
            # 3. Use the PythonResolver's resolution callback (rcb) to convert
            #    the string into a python object
            # 4. We call into annotations.py:ann_to_type to convert python obj
            #    from step 3 into a type that torchscript understands.
            #
            # NamedTuples are more complicated, because it has sub-types.
            # Normally, once we have the NamedTuple type object from #3,
            # we can just look at the annotation literal values and use
            # ann_to_type directly on them.
            #
            # But sometimes, users will annotate with string literals, e.g.
            #    x: 'int'
            # This also happens with PEP563 (from __forward__ import annotations)
            #
            # These annotations appear in the annotation dict as ForwardRef('int').
            #
            # Then, we need to convert the string into a python object. This
            # requires having local context for custom objects or imported types.
            # rcb() is what gives us this. So, we plumb rcb through the stack so
            # it can be used in this context for the if block below.
            #
            # FAQ:
            # - Why do we need this special handling for NamedTuple but string
            #   annotations work fine for normal types? Normally, we parse the
            #   string directly and then call rcb() directly from C++.
            # - Why not use ForwardRef._evaluate? For that, we need globals()
            #   and locals() for the local context where the NamedTuple was defined.
            #   rcb is what lets us look up into these. So, basically rcb does the
            #   hard work for us.
            if isinstance(field_type, ForwardRef) and rcb is not None:
                rcb_type = rcb(field_type.__forward_arg__)
                # rcb returns None if it can't find anything.
                if rcb_type is None:
                    raise ValueError(
                        f"Unknown type annotation: '{field_type}' in NamedTuple {obj.__name__}."
                        f" Likely due to partial support for ForwardRef parameters in NamedTuples, see #95858."
                        f" Issue occurred at {loc.highlight()}"
                    )
                field_type = rcb_type
            the_type = torch.jit.annotations.ann_to_type(field_type, loc, rcb)
            annotations.append(the_type)
        else:
            annotations.append(torch._C.TensorType.getInferred())
    return type(obj).__name__, obj._fields, annotations, defaults


def _create_named_tuple(
    t, unqual_name: str, field_names: List[str], defaults: Tuple[Any, ...]
):
    TupleType = collections.namedtuple(unqual_name, field_names, defaults=defaults)  # type: ignore[call-arg, no-redef, misc]
    return TupleType(*t)


@contextlib.contextmanager
def _disable_emit_hooks():
    hooks = torch._C._jit_get_emit_hooks()
    torch._C._jit_set_emit_hooks(None, None)
    try:
        yield
    finally:
        torch._C._jit_set_emit_hooks(hooks[0], hooks[1])


def _disable_emit_hooks_decorator(_DecoratorContextManager) -> None:  # noqa: F811
    def __enter__(self) -> None:
        self.hooks = torch._C._jit_get_emit_hooks()
        torch._C._jit_set_emit_hooks(None, None)

    def __exit__(self, *args) -> None:
        torch._C._jit_set_emit_hooks(self.hooks[0], self.hooks[1])


def _is_exception(obj) -> bool:
    if not inspect.isclass(obj):
        return False
    return issubclass(obj, Exception)


def raise_error_container_parameter_missing(target_type) -> None:
    if target_type == "Dict":
        raise RuntimeError(
            "Attempted to use Dict without "
            "contained types. Please add contained type, e.g. "
            "Dict[int, int]"
        )
    raise RuntimeError(
        f"Attempted to use {target_type} without a "
        "contained type. Please add a contained type, e.g. "
        f"{target_type}[int]"
    )


def get_origin(target_type):
    return getattr(target_type, "__origin__", None)


def get_args(target_type):
    return getattr(target_type, "__args__", None)


def check_args_exist(target_type) -> None:
    if target_type is List or target_type is list:
        raise_error_container_parameter_missing("List")
    elif target_type is Tuple or target_type is tuple:
        raise_error_container_parameter_missing("Tuple")
    elif target_type is Dict or target_type is dict:
        raise_error_container_parameter_missing("Dict")
    elif target_type is None or target_type is Optional:
        raise_error_container_parameter_missing("Optional")


def check_empty_containers(obj) -> None:
    if obj == [] or obj == {} or obj == ():
        warnings.warn(
            "The inner type of a container is lost when "
            "calling torch.jit.isinstance in eager mode. For "
            "example, List[int] would become list and "
            "therefore falsely return True for List[float] or"
            " List[str]."
        )


# supports List/Dict/Tuple and Optional types
# TODO support future
def container_checker(obj, target_type) -> bool:
    origin_type = get_origin(target_type)
    check_args_exist(target_type)
    if origin_type is list or origin_type is List:
        check_empty_containers(obj)
        if not isinstance(obj, list):
            return False
        arg_type = get_args(target_type)[0]
        arg_origin = get_origin(arg_type)
        for el in obj:
            # check if nested container, ex: List[List[str]]
            if arg_origin:  # processes nested container, ex: List[List[str]]
                if not container_checker(el, arg_type):
                    return False
            elif not isinstance(el, arg_type):
                return False
        return True
    elif origin_type is Dict or origin_type is dict:
        check_empty_containers(obj)
        if not isinstance(obj, dict):
            return False
        key_type = get_args(target_type)[0]
        val_type = get_args(target_type)[1]
        for key, val in obj.items():
            # check if keys are of right type
            if not isinstance(key, key_type):
                return False
            val_origin = get_origin(val_type)
            if val_origin:
                if not container_checker(val, val_type):
                    return False
            elif not isinstance(val, val_type):
                return False
        return True
    elif origin_type is Tuple or origin_type is tuple:
        check_empty_containers(obj)
        if not isinstance(obj, tuple):
            return False
        arg_types = get_args(target_type)
        if len(obj) != len(arg_types):
            return False
        for el, el_type in zip(obj, arg_types):
            el_origin = get_origin(el_type)
            if el_origin:
                if not container_checker(el, el_type):
                    return False
            elif not isinstance(el, el_type):
                return False
        return True
    elif origin_type is Union:  # also handles Optional
        if obj is None:  # check before recursion because None is always fine
            return True
        inner_types = get_args(target_type)
        for t in inner_types:
            t_origin = get_origin(t)
            if t_origin:
                return container_checker(obj, t)
            elif isinstance(obj, t):
                return True
    return False


def _isinstance(obj, target_type) -> bool:
    if isinstance(target_type, collections.abc.Container):
        if not isinstance(target_type, tuple):
            raise RuntimeError(
                "The second argument to "
                "`torch.jit.isinstance` must be a type "
                "or a tuple of types"
            )
        for t_type in target_type:
            if _isinstance(obj, t_type):
                return True
        return False

    origin_type = get_origin(target_type)
    if origin_type:
        return container_checker(obj, target_type)

    # Check to handle non-typed optional origin returns as none instead
    #    of as optional in 3.7-3.8
    check_args_exist(target_type)

    # handle non-containers
    return isinstance(obj, target_type)


class _TensorExtractor(pickle.Pickler):
    def __init__(self, *args, tensors: List[torch.Tensor], **kwargs):
        super().__init__(*args, **kwargs)
        self.tensors = tensors

    def persistent_id(self, obj):
        if isinstance(obj, torch.Tensor):
            self.tensors.append(obj)
            return ""
        # Since we just want to extract tensors, we don't mind if an object is
        # unpicklable if it doesn't contain tensors, as we can just ignore/skip
        # it. To play it safe, we only do so for common objects that we're sure
        # don't contain tensors. Feel free to add new types here. Note also that
        # even if a type isn't listed here this won't block users, since thet
        # can just add a __getstate__ or __reduce__ method to their class.
        if isinstance(obj, LockType):
            return ""
        # Futures and RRefs don't technically contain a value, they just offer
        # the means to access a value.
        if isinstance(obj, CFuture) or is_rref_instance(obj):
            return ""
        if isinstance(obj, CAwait):
            return ""
        if isinstance(obj, torch.cuda.Event):
            return ""
        if isinstance(obj, threading.Thread):
            return ""
        return None


def _extract_tensors(obj):
    r"""
    This function is exclusively called from C++.
    See ``torch/csrc/jit/python/python_ivalue.h``.

    It extracts the tensors contained in the given object, through pickling.
    """
    tensors: List[torch.Tensor] = []
    extractor = _TensorExtractor(io.BytesIO(), protocol=-1, tensors=tensors)
    extractor.dump(obj)
    return tensors


# In Python-3.11+ typed enums (i.e. IntEnum for example) retain number of base class methods in subclass
# that were previously dropped. To preserve the behavior, explicitly drop them there

if sys.version_info > (3, 10):
    _drop(enum.Enum.__new__)
    _drop(enum.Enum.__format__)
    _drop(enum.Enum.__repr__)
    _drop(enum.Enum.__str__)
