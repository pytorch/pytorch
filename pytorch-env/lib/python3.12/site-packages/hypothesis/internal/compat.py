# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import codecs
import copy
import dataclasses
import inspect
import platform
import sys
import sysconfig
import typing
from functools import partial
from typing import Any, ForwardRef, Optional, TypedDict as TypedDict, get_args

try:
    BaseExceptionGroup = BaseExceptionGroup
    ExceptionGroup = ExceptionGroup  # pragma: no cover
except NameError:
    from exceptiongroup import (
        BaseExceptionGroup as BaseExceptionGroup,
        ExceptionGroup as ExceptionGroup,
    )
if typing.TYPE_CHECKING:  # pragma: no cover
    from typing_extensions import (
        Concatenate as Concatenate,
        NotRequired as NotRequired,
        ParamSpec as ParamSpec,
        TypeAlias as TypeAlias,
        TypedDict as TypedDict,
        override as override,
    )
else:
    # In order to use NotRequired, we need the version of TypedDict included in Python 3.11+.
    if sys.version_info[:2] >= (3, 11):
        from typing import NotRequired as NotRequired, TypedDict as TypedDict
    else:
        try:
            from typing_extensions import (
                NotRequired as NotRequired,
                TypedDict as TypedDict,
            )
        except ImportError:
            # We can use the old TypedDict from Python 3.8+ at runtime.
            class NotRequired:
                """A runtime placeholder for the NotRequired type, which is not available in Python <3.11."""

                def __class_getitem__(cls, item):
                    return cls

    try:
        from typing import (
            Concatenate as Concatenate,
            ParamSpec as ParamSpec,
            TypeAlias as TypeAlias,
            override as override,
        )
    except ImportError:
        try:
            from typing_extensions import (
                Concatenate as Concatenate,
                ParamSpec as ParamSpec,
                TypeAlias as TypeAlias,
                override as override,
            )
        except ImportError:
            Concatenate, ParamSpec = None, None
            TypeAlias = None
            override = lambda f: f


PYPY = platform.python_implementation() == "PyPy"
GRAALPY = platform.python_implementation() == "GraalVM"
WINDOWS = platform.system() == "Windows"
# First defined in CPython 3.13, defaults to False
FREE_THREADED_CPYTHON = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def add_note(exc, note):
    try:
        exc.add_note(note)
    except AttributeError:
        if not hasattr(exc, "__notes__"):
            try:
                exc.__notes__ = []
            except AttributeError:
                return  # give up, might be e.g. a frozen dataclass
        exc.__notes__.append(note)


def escape_unicode_characters(s: str) -> str:
    return codecs.encode(s, "unicode_escape").decode("ascii")


def int_from_bytes(data: typing.Union[bytes, bytearray]) -> int:
    return int.from_bytes(data, "big")


def int_to_bytes(i: int, size: int) -> bytes:
    return i.to_bytes(size, "big")


def int_to_byte(i: int) -> bytes:
    return bytes([i])


def is_typed_named_tuple(cls: type) -> bool:
    """Return True if cls is probably a subtype of `typing.NamedTuple`.

    Unfortunately types created with `class T(NamedTuple):` actually
    subclass `tuple` directly rather than NamedTuple.  This is annoying,
    and means we just have to hope that nobody defines a different tuple
    subclass with similar attributes.
    """
    return (
        issubclass(cls, tuple)
        and hasattr(cls, "_fields")
        and (hasattr(cls, "_field_types") or hasattr(cls, "__annotations__"))
    )


def _hint_and_args(x):
    return (x, *get_args(x))


def get_type_hints(thing):
    """Like the typing version, but tries harder and never errors.

    Tries harder: if the thing to inspect is a class but typing.get_type_hints
    raises an error or returns no hints, then this function will try calling it
    on the __init__ method. This second step often helps with user-defined
    classes on older versions of Python. The third step we take is trying
    to fetch types from the __signature__ property.
    They override any other ones we found earlier.

    Never errors: instead of raising TypeError for uninspectable objects, or
    NameError for unresolvable forward references, just return an empty dict.
    """
    if isinstance(thing, partial):
        from hypothesis.internal.reflection import get_signature

        bound = set(get_signature(thing.func).parameters).difference(
            get_signature(thing).parameters
        )
        return {k: v for k, v in get_type_hints(thing.func).items() if k not in bound}

    try:
        hints = typing.get_type_hints(thing, include_extras=True)
    except (AttributeError, TypeError, NameError):  # pragma: no cover
        hints = {}

    if inspect.isclass(thing):
        try:
            hints.update(typing.get_type_hints(thing.__init__, include_extras=True))
        except (TypeError, NameError, AttributeError):
            pass

    try:
        if hasattr(thing, "__signature__"):
            # It is possible for the signature and annotations attributes to
            # differ on an object due to renamed arguments.
            from hypothesis.internal.reflection import get_signature
            from hypothesis.strategies._internal.types import is_a_type

            vkinds = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            for p in get_signature(thing).parameters.values():
                if (
                    p.kind not in vkinds
                    and is_a_type(p.annotation)
                    and p.annotation is not p.empty
                ):
                    p_hint = p.annotation

                    # Defer to `get_type_hints` if signature annotation is, or
                    # contains, a forward reference that is otherwise resolved.
                    if any(
                        isinstance(sig_hint, ForwardRef)
                        and not isinstance(hint, ForwardRef)
                        for sig_hint, hint in zip(
                            _hint_and_args(p.annotation),
                            _hint_and_args(hints.get(p.name, Any)),
                        )
                    ):
                        p_hint = hints[p.name]
                    if p.default is None:
                        hints[p.name] = typing.Optional[p_hint]
                    else:
                        hints[p.name] = p_hint
    except (AttributeError, TypeError, NameError):  # pragma: no cover
        pass

    return hints


# Under Python 2, math.floor and math.ceil returned floats, which cannot
# represent large integers - eg `float(2**53) == float(2**53 + 1)`.
# We therefore implement them entirely in (long) integer operations.
# We still use the same trick on Python 3, because Numpy values and other
# custom __floor__ or __ceil__ methods may convert via floats.
# See issue #1667, Numpy issue 9068.
def floor(x):
    y = int(x)
    if y != x and x < 0:
        return y - 1
    return y


def ceil(x):
    y = int(x)
    if y != x and x > 0:
        return y + 1
    return y


def extract_bits(x: int, /, width: Optional[int] = None) -> list[int]:
    assert x >= 0
    result = []
    while x:
        result.append(x & 1)
        x >>= 1
    if width is not None:
        result = (result + [0] * width)[:width]
    result.reverse()
    return result


# int.bit_count was added in python 3.10
try:
    bit_count = int.bit_count
except AttributeError:  # pragma: no cover
    bit_count = lambda self: sum(extract_bits(abs(self)))


def bad_django_TestCase(runner):
    if runner is None or "django.test" not in sys.modules:
        return False
    else:  # pragma: no cover
        if not isinstance(runner, sys.modules["django.test"].TransactionTestCase):
            return False

        from hypothesis.extra.django._impl import HypothesisTestCase

        return not isinstance(runner, HypothesisTestCase)


# see issue #3812
if sys.version_info[:2] < (3, 12):

    def dataclass_asdict(obj, *, dict_factory=dict):
        """
        A vendored variant of dataclasses.asdict. Includes the bugfix for
        defaultdicts (cpython/32056) for all versions. See also issues/3812.

        This should be removed whenever we drop support for 3.11. We can use the
        standard dataclasses.asdict after that point.
        """
        if not dataclasses._is_dataclass_instance(obj):  # pragma: no cover
            raise TypeError("asdict() should be called on dataclass instances")
        return _asdict_inner(obj, dict_factory)

else:  # pragma: no cover
    dataclass_asdict = dataclasses.asdict


def _asdict_inner(obj, dict_factory):
    if dataclasses._is_dataclass_instance(obj):
        return dict_factory(
            (f.name, _asdict_inner(getattr(obj, f.name), dict_factory))
            for f in dataclasses.fields(obj)
        )
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return type(obj)(*[_asdict_inner(v, dict_factory) for v in obj])
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_asdict_inner(v, dict_factory) for v in obj)
    elif isinstance(obj, dict):
        if hasattr(type(obj), "default_factory"):
            result = type(obj)(obj.default_factory)
            for k, v in obj.items():
                result[_asdict_inner(k, dict_factory)] = _asdict_inner(v, dict_factory)
            return result
        return type(obj)(
            (_asdict_inner(k, dict_factory), _asdict_inner(v, dict_factory))
            for k, v in obj.items()
        )
    else:
        return copy.deepcopy(obj)
