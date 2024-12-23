# mypy: allow-untyped-defs
"""Python version compatibility code."""

from __future__ import annotations

import dataclasses
import enum
import functools
import inspect
from inspect import Parameter
from inspect import signature
import os
from pathlib import Path
import sys
from typing import Any
from typing import Callable
from typing import Final
from typing import NoReturn

import py


#: constant to prepare valuing pylib path replacements/lazy proxies later on
#  intended for removal in pytest 8.0 or 9.0

# fmt: off
# intentional space to create a fake difference for the verification
LEGACY_PATH = py.path. local
# fmt: on


def legacy_path(path: str | os.PathLike[str]) -> LEGACY_PATH:
    """Internal wrapper to prepare lazy proxies for legacy_path instances"""
    return LEGACY_PATH(path)


# fmt: off
# Singleton type for NOTSET, as described in:
# https://www.python.org/dev/peps/pep-0484/#support-for-singleton-types-in-unions
class NotSetType(enum.Enum):
    token = 0
NOTSET: Final = NotSetType.token
# fmt: on


def is_generator(func: object) -> bool:
    genfunc = inspect.isgeneratorfunction(func)
    return genfunc and not iscoroutinefunction(func)


def iscoroutinefunction(func: object) -> bool:
    """Return True if func is a coroutine function (a function defined with async
    def syntax, and doesn't contain yield), or a function decorated with
    @asyncio.coroutine.

    Note: copied and modified from Python 3.5's builtin coroutines.py to avoid
    importing asyncio directly, which in turns also initializes the "logging"
    module as a side-effect (see issue #8).
    """
    return inspect.iscoroutinefunction(func) or getattr(func, "_is_coroutine", False)


def is_async_function(func: object) -> bool:
    """Return True if the given function seems to be an async function or
    an async generator."""
    return iscoroutinefunction(func) or inspect.isasyncgenfunction(func)


def getlocation(function, curdir: str | os.PathLike[str] | None = None) -> str:
    function = get_real_func(function)
    fn = Path(inspect.getfile(function))
    lineno = function.__code__.co_firstlineno
    if curdir is not None:
        try:
            relfn = fn.relative_to(curdir)
        except ValueError:
            pass
        else:
            return "%s:%d" % (relfn, lineno + 1)
    return "%s:%d" % (fn, lineno + 1)


def num_mock_patch_args(function) -> int:
    """Return number of arguments used up by mock arguments (if any)."""
    patchings = getattr(function, "patchings", None)
    if not patchings:
        return 0

    mock_sentinel = getattr(sys.modules.get("mock"), "DEFAULT", object())
    ut_mock_sentinel = getattr(sys.modules.get("unittest.mock"), "DEFAULT", object())

    return len(
        [
            p
            for p in patchings
            if not p.attribute_name
            and (p.new is mock_sentinel or p.new is ut_mock_sentinel)
        ]
    )


def getfuncargnames(
    function: Callable[..., object],
    *,
    name: str = "",
    cls: type | None = None,
) -> tuple[str, ...]:
    """Return the names of a function's mandatory arguments.

    Should return the names of all function arguments that:
    * Aren't bound to an instance or type as in instance or class methods.
    * Don't have default values.
    * Aren't bound with functools.partial.
    * Aren't replaced with mocks.

    The cls arguments indicate that the function should be treated as a bound
    method even though it's not unless the function is a static method.

    The name parameter should be the original name in which the function was collected.
    """
    # TODO(RonnyPfannschmidt): This function should be refactored when we
    # revisit fixtures. The fixture mechanism should ask the node for
    # the fixture names, and not try to obtain directly from the
    # function object well after collection has occurred.

    # The parameters attribute of a Signature object contains an
    # ordered mapping of parameter names to Parameter instances.  This
    # creates a tuple of the names of the parameters that don't have
    # defaults.
    try:
        parameters = signature(function).parameters
    except (ValueError, TypeError) as e:
        from _pytest.outcomes import fail

        fail(
            f"Could not determine arguments of {function!r}: {e}",
            pytrace=False,
        )

    arg_names = tuple(
        p.name
        for p in parameters.values()
        if (
            p.kind is Parameter.POSITIONAL_OR_KEYWORD
            or p.kind is Parameter.KEYWORD_ONLY
        )
        and p.default is Parameter.empty
    )
    if not name:
        name = function.__name__

    # If this function should be treated as a bound method even though
    # it's passed as an unbound method or function, remove the first
    # parameter name.
    if (
        # Not using `getattr` because we don't want to resolve the staticmethod.
        # Not using `cls.__dict__` because we want to check the entire MRO.
        cls
        and not isinstance(
            inspect.getattr_static(cls, name, default=None), staticmethod
        )
    ):
        arg_names = arg_names[1:]
    # Remove any names that will be replaced with mocks.
    if hasattr(function, "__wrapped__"):
        arg_names = arg_names[num_mock_patch_args(function) :]
    return arg_names


def get_default_arg_names(function: Callable[..., Any]) -> tuple[str, ...]:
    # Note: this code intentionally mirrors the code at the beginning of
    # getfuncargnames, to get the arguments which were excluded from its result
    # because they had default values.
    return tuple(
        p.name
        for p in signature(function).parameters.values()
        if p.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
        and p.default is not Parameter.empty
    )


_non_printable_ascii_translate_table = {
    i: f"\\x{i:02x}" for i in range(128) if i not in range(32, 127)
}
_non_printable_ascii_translate_table.update(
    {ord("\t"): "\\t", ord("\r"): "\\r", ord("\n"): "\\n"}
)


def ascii_escaped(val: bytes | str) -> str:
    r"""If val is pure ASCII, return it as an str, otherwise, escape
    bytes objects into a sequence of escaped bytes:

    b'\xc3\xb4\xc5\xd6' -> r'\xc3\xb4\xc5\xd6'

    and escapes strings into a sequence of escaped unicode ids, e.g.:

    r'4\nV\U00043efa\x0eMXWB\x1e\u3028\u15fd\xcd\U0007d944'

    Note:
       The obvious "v.decode('unicode-escape')" will return
       valid UTF-8 unicode if it finds them in bytes, but we
       want to return escaped bytes for any byte, even if they match
       a UTF-8 string.
    """
    if isinstance(val, bytes):
        ret = val.decode("ascii", "backslashreplace")
    else:
        ret = val.encode("unicode_escape").decode("ascii")
    return ret.translate(_non_printable_ascii_translate_table)


@dataclasses.dataclass
class _PytestWrapper:
    """Dummy wrapper around a function object for internal use only.

    Used to correctly unwrap the underlying function object when we are
    creating fixtures, because we wrap the function object ourselves with a
    decorator to issue warnings when the fixture function is called directly.
    """

    obj: Any


def get_real_func(obj):
    """Get the real function object of the (possibly) wrapped object by
    functools.wraps or functools.partial."""
    start_obj = obj
    for i in range(100):
        # __pytest_wrapped__ is set by @pytest.fixture when wrapping the fixture function
        # to trigger a warning if it gets called directly instead of by pytest: we don't
        # want to unwrap further than this otherwise we lose useful wrappings like @mock.patch (#3774)
        new_obj = getattr(obj, "__pytest_wrapped__", None)
        if isinstance(new_obj, _PytestWrapper):
            obj = new_obj.obj
            break
        new_obj = getattr(obj, "__wrapped__", None)
        if new_obj is None:
            break
        obj = new_obj
    else:
        from _pytest._io.saferepr import saferepr

        raise ValueError(
            f"could not find real function of {saferepr(start_obj)}\nstopped at {saferepr(obj)}"
        )
    if isinstance(obj, functools.partial):
        obj = obj.func
    return obj


def get_real_method(obj, holder):
    """Attempt to obtain the real function object that might be wrapping
    ``obj``, while at the same time returning a bound method to ``holder`` if
    the original object was a bound method."""
    try:
        is_method = hasattr(obj, "__func__")
        obj = get_real_func(obj)
    except Exception:  # pragma: no cover
        return obj
    if is_method and hasattr(obj, "__get__") and callable(obj.__get__):
        obj = obj.__get__(holder)
    return obj


def getimfunc(func):
    try:
        return func.__func__
    except AttributeError:
        return func


def safe_getattr(object: Any, name: str, default: Any) -> Any:
    """Like getattr but return default upon any Exception or any OutcomeException.

    Attribute access can potentially fail for 'evil' Python objects.
    See issue #214.
    It catches OutcomeException because of #2490 (issue #580), new outcomes
    are derived from BaseException instead of Exception (for more details
    check #2707).
    """
    from _pytest.outcomes import TEST_OUTCOME

    try:
        return getattr(object, name, default)
    except TEST_OUTCOME:
        return default


def safe_isclass(obj: object) -> bool:
    """Ignore any exception via isinstance on Python 3."""
    try:
        return inspect.isclass(obj)
    except Exception:
        return False


def get_user_id() -> int | None:
    """Return the current process's real user id or None if it could not be
    determined.

    :return: The user id or None if it could not be determined.
    """
    # mypy follows the version and platform checking expectation of PEP 484:
    # https://mypy.readthedocs.io/en/stable/common_issues.html?highlight=platform#python-version-and-system-platform-checks
    # Containment checks are too complex for mypy v1.5.0 and cause failure.
    if sys.platform == "win32" or sys.platform == "emscripten":
        # win32 does not have a getuid() function.
        # Emscripten has a return 0 stub.
        return None
    else:
        # On other platforms, a return value of -1 is assumed to indicate that
        # the current process's real user id could not be determined.
        ERROR = -1
        uid = os.getuid()
        return uid if uid != ERROR else None


# Perform exhaustiveness checking.
#
# Consider this example:
#
#     MyUnion = Union[int, str]
#
#     def handle(x: MyUnion) -> int {
#         if isinstance(x, int):
#             return 1
#         elif isinstance(x, str):
#             return 2
#         else:
#             raise Exception('unreachable')
#
# Now suppose we add a new variant:
#
#     MyUnion = Union[int, str, bytes]
#
# After doing this, we must remember ourselves to go and update the handle
# function to handle the new variant.
#
# With `assert_never` we can do better:
#
#     // raise Exception('unreachable')
#     return assert_never(x)
#
# Now, if we forget to handle the new variant, the type-checker will emit a
# compile-time error, instead of the runtime error we would have gotten
# previously.
#
# This also work for Enums (if you use `is` to compare) and Literals.
def assert_never(value: NoReturn) -> NoReturn:
    assert False, f"Unhandled value: {value} ({type(value).__name__})"
