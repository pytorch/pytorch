# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import contextlib
import os
import sys
import textwrap
import traceback
from functools import partial
from inspect import getframeinfo
from pathlib import Path
from typing import NamedTuple, Optional

import hypothesis
from hypothesis.errors import _Trimmable
from hypothesis.internal.compat import BaseExceptionGroup
from hypothesis.utils.dynamicvariables import DynamicVariable


def belongs_to(package):
    if not hasattr(package, "__file__"):  # pragma: no cover
        return lambda filepath: False

    root = Path(package.__file__).resolve().parent
    cache = {str: {}, bytes: {}}

    def accept(filepath):
        ftype = type(filepath)
        try:
            return cache[ftype][filepath]
        except KeyError:
            pass
        try:
            Path(filepath).resolve().relative_to(root)
            result = True
        except Exception:
            result = False
        cache[ftype][filepath] = result
        return result

    accept.__name__ = f"is_{package.__name__}_file"
    return accept


FILE_CACHE: dict[bytes, bool] = {}


is_hypothesis_file = belongs_to(hypothesis)


def get_trimmed_traceback(exception=None):
    """Return the current traceback, minus any frames added by Hypothesis."""
    if exception is None:
        _, exception, tb = sys.exc_info()
    else:
        tb = exception.__traceback__
    # Avoid trimming the traceback if we're in verbose mode, or the error
    # was raised inside Hypothesis. Additionally, the environment variable
    # HYPOTHESIS_NO_TRACEBACK_TRIM is respected if nonempty, because verbose
    # mode is prohibitively slow when debugging strategy recursion errors.
    if (
        tb is None
        or os.environ.get("HYPOTHESIS_NO_TRACEBACK_TRIM", None)
        or hypothesis.settings.default.verbosity >= hypothesis.Verbosity.debug
        or is_hypothesis_file(traceback.extract_tb(tb)[-1][0])
        and not isinstance(exception, _Trimmable)
    ):
        return tb
    while tb.tb_next is not None and (
        # If the frame is from one of our files, it's been added by Hypothesis.
        is_hypothesis_file(getframeinfo(tb.tb_frame).filename)
        # But our `@proxies` decorator overrides the source location,
        # so we check for an attribute it injects into the frame too.
        or tb.tb_frame.f_globals.get("__hypothesistracebackhide__") is True
    ):
        tb = tb.tb_next
    return tb


class InterestingOrigin(NamedTuple):
    # The `interesting_origin` is how Hypothesis distinguishes between multiple
    # failures, for reporting and also to replay from the example database (even
    # if report_multiple_bugs=False).  We traditionally use the exception type and
    # location, but have extracted this logic in order to see through `except ...:`
    # blocks and understand the __cause__ (`raise x from y`) or __context__ that
    # first raised an exception as well as PEP-654 exception groups.
    exc_type: type[BaseException]
    filename: Optional[str]
    lineno: Optional[int]
    context: "InterestingOrigin | tuple[()]"
    group_elems: "tuple[InterestingOrigin, ...]"

    def __str__(self) -> str:
        ctx = ""
        if self.context:
            ctx = textwrap.indent(f"\ncontext: {self.context}", prefix="    ")
        group = ""
        if self.group_elems:
            chunks = "\n  ".join(str(x) for x in self.group_elems)
            group = textwrap.indent(f"\nchild exceptions:\n  {chunks}", prefix="    ")
        return f"{self.exc_type.__name__} at {self.filename}:{self.lineno}{ctx}{group}"

    @classmethod
    def from_exception(
        cls, exception: BaseException, /, seen: tuple[BaseException, ...] = ()
    ) -> "InterestingOrigin":
        filename, lineno = None, None
        if tb := get_trimmed_traceback(exception):
            filename, lineno, *_ = traceback.extract_tb(tb)[-1]
        seen = (*seen, exception)
        make = partial(cls.from_exception, seen=seen)
        context: "InterestingOrigin | tuple[()]" = ()
        if exception.__context__ is not None and exception.__context__ not in seen:
            context = make(exception.__context__)
        return cls(
            type(exception),
            filename,
            lineno,
            # Note that if __cause__ is set it is always equal to __context__, explicitly
            # to support introspection when debugging, so we can use that unconditionally.
            context,
            # We distinguish exception groups by the inner exceptions, as for __context__
            (
                tuple(make(exc) for exc in exception.exceptions if exc not in seen)
                if isinstance(exception, BaseExceptionGroup)
                else ()
            ),
        )


current_pytest_item = DynamicVariable(None)


def _get_exceptioninfo():
    # ExceptionInfo was moved to the top-level namespace in Pytest 7.0
    if "pytest" in sys.modules:
        with contextlib.suppress(Exception):
            # From Pytest 7, __init__ warns on direct calls.
            return sys.modules["pytest"].ExceptionInfo.from_exc_info
    if "_pytest._code" in sys.modules:  # old versions only
        with contextlib.suppress(Exception):
            return sys.modules["_pytest._code"].ExceptionInfo
    return None  # pragma: no cover  # coverage tests always use pytest


def format_exception(err, tb):
    # Try using Pytest to match the currently configured traceback style
    ExceptionInfo = _get_exceptioninfo()
    if current_pytest_item.value is not None and ExceptionInfo is not None:
        item = current_pytest_item.value
        return str(item.repr_failure(ExceptionInfo((type(err), err, tb)))) + "\n"

    # Or use better_exceptions, if that's installed and enabled
    if "better_exceptions" in sys.modules:
        better_exceptions = sys.modules["better_exceptions"]
        if sys.excepthook is better_exceptions.excepthook:
            return "".join(better_exceptions.format_exception(type(err), err, tb))

    # If all else fails, use the standard-library formatting tools
    return "".join(traceback.format_exception(type(err), err, tb))
