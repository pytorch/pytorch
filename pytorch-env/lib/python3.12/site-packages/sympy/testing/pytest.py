"""py.test hacks to support XFAIL/XPASS"""

import sys
import re
import functools
import os
import contextlib
import warnings
import inspect
import pathlib
from typing import Any, Callable

from sympy.utilities.exceptions import SymPyDeprecationWarning
# Imported here for backwards compatibility. Note: do not import this from
# here in library code (importing sympy.pytest in library code will break the
# pytest integration).
from sympy.utilities.exceptions import ignore_warnings # noqa:F401

ON_CI = os.getenv('CI', None) == "true"

try:
    import pytest
    USE_PYTEST = getattr(sys, '_running_pytest', False)
except ImportError:
    USE_PYTEST = False


raises: Callable[[Any, Any], Any]
XFAIL: Callable[[Any], Any]
skip: Callable[[Any], Any]
SKIP: Callable[[Any], Any]
slow: Callable[[Any], Any]
tooslow: Callable[[Any], Any]
nocache_fail: Callable[[Any], Any]


if USE_PYTEST:
    raises = pytest.raises
    skip = pytest.skip
    XFAIL = pytest.mark.xfail
    SKIP = pytest.mark.skip
    slow = pytest.mark.slow
    tooslow = pytest.mark.tooslow
    nocache_fail = pytest.mark.nocache_fail
    from _pytest.outcomes import Failed

else:
    # Not using pytest so define the things that would have been imported from
    # there.

    # _pytest._code.code.ExceptionInfo
    class ExceptionInfo:
        def __init__(self, value):
            self.value = value

        def __repr__(self):
            return "<ExceptionInfo {!r}>".format(self.value)


    def raises(expectedException, code=None):
        """
        Tests that ``code`` raises the exception ``expectedException``.

        ``code`` may be a callable, such as a lambda expression or function
        name.

        If ``code`` is not given or None, ``raises`` will return a context
        manager for use in ``with`` statements; the code to execute then
        comes from the scope of the ``with``.

        ``raises()`` does nothing if the callable raises the expected exception,
        otherwise it raises an AssertionError.

        Examples
        ========

        >>> from sympy.testing.pytest import raises

        >>> raises(ZeroDivisionError, lambda: 1/0)
        <ExceptionInfo ZeroDivisionError(...)>
        >>> raises(ZeroDivisionError, lambda: 1/2)
        Traceback (most recent call last):
        ...
        Failed: DID NOT RAISE

        >>> with raises(ZeroDivisionError):
        ...     n = 1/0
        >>> with raises(ZeroDivisionError):
        ...     n = 1/2
        Traceback (most recent call last):
        ...
        Failed: DID NOT RAISE

        Note that you cannot test multiple statements via
        ``with raises``:

        >>> with raises(ZeroDivisionError):
        ...     n = 1/0    # will execute and raise, aborting the ``with``
        ...     n = 9999/0 # never executed

        This is just what ``with`` is supposed to do: abort the
        contained statement sequence at the first exception and let
        the context manager deal with the exception.

        To test multiple statements, you'll need a separate ``with``
        for each:

        >>> with raises(ZeroDivisionError):
        ...     n = 1/0    # will execute and raise
        >>> with raises(ZeroDivisionError):
        ...     n = 9999/0 # will also execute and raise

        """
        if code is None:
            return RaisesContext(expectedException)
        elif callable(code):
            try:
                code()
            except expectedException as e:
                return ExceptionInfo(e)
            raise Failed("DID NOT RAISE")
        elif isinstance(code, str):
            raise TypeError(
                '\'raises(xxx, "code")\' has been phased out; '
                'change \'raises(xxx, "expression")\' '
                'to \'raises(xxx, lambda: expression)\', '
                '\'raises(xxx, "statement")\' '
                'to \'with raises(xxx): statement\'')
        else:
            raise TypeError(
                'raises() expects a callable for the 2nd argument.')

    class RaisesContext:
        def __init__(self, expectedException):
            self.expectedException = expectedException

        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_value, traceback):
            if exc_type is None:
                raise Failed("DID NOT RAISE")
            return issubclass(exc_type, self.expectedException)

    class XFail(Exception):
        pass

    class XPass(Exception):
        pass

    class Skipped(Exception):
        pass

    class Failed(Exception):  # type: ignore
        pass

    def XFAIL(func):
        def wrapper():
            try:
                func()
            except Exception as e:
                message = str(e)
                if message != "Timeout":
                    raise XFail(func.__name__)
                else:
                    raise Skipped("Timeout")
            raise XPass(func.__name__)

        wrapper = functools.update_wrapper(wrapper, func)
        return wrapper

    def skip(str):
        raise Skipped(str)

    def SKIP(reason):
        """Similar to ``skip()``, but this is a decorator. """
        def wrapper(func):
            def func_wrapper():
                raise Skipped(reason)

            func_wrapper = functools.update_wrapper(func_wrapper, func)
            return func_wrapper

        return wrapper

    def slow(func):
        func._slow = True

        def func_wrapper():
            func()

        func_wrapper = functools.update_wrapper(func_wrapper, func)
        func_wrapper.__wrapped__ = func
        return func_wrapper

    def tooslow(func):
        func._slow = True
        func._tooslow = True

        def func_wrapper():
            skip("Too slow")

        func_wrapper = functools.update_wrapper(func_wrapper, func)
        func_wrapper.__wrapped__ = func
        return func_wrapper

    def nocache_fail(func):
        "Dummy decorator for marking tests that fail when cache is disabled"
        return func

@contextlib.contextmanager
def warns(warningcls, *, match='', test_stacklevel=True):
    '''
    Like raises but tests that warnings are emitted.

    >>> from sympy.testing.pytest import warns
    >>> import warnings

    >>> with warns(UserWarning):
    ...     warnings.warn('deprecated', UserWarning, stacklevel=2)

    >>> with warns(UserWarning):
    ...     pass
    Traceback (most recent call last):
    ...
    Failed: DID NOT WARN. No warnings of type UserWarning\
    was emitted. The list of emitted warnings is: [].

    ``test_stacklevel`` makes it check that the ``stacklevel`` parameter to
    ``warn()`` is set so that the warning shows the user line of code (the
    code under the warns() context manager). Set this to False if this is
    ambiguous or if the context manager does not test the direct user code
    that emits the warning.

    If the warning is a ``SymPyDeprecationWarning``, this additionally tests
    that the ``active_deprecations_target`` is a real target in the
    ``active-deprecations.md`` file.

    '''
    # Absorbs all warnings in warnrec
    with warnings.catch_warnings(record=True) as warnrec:
        # Any warning other than the one we are looking for is an error
        warnings.simplefilter("error")
        warnings.filterwarnings("always", category=warningcls)
        # Now run the test
        yield warnrec

    # Raise if expected warning not found
    if not any(issubclass(w.category, warningcls) for w in warnrec):
        msg = ('Failed: DID NOT WARN.'
               ' No warnings of type %s was emitted.'
               ' The list of emitted warnings is: %s.'
               ) % (warningcls, [w.message for w in warnrec])
        raise Failed(msg)

    # We don't include the match in the filter above because it would then
    # fall to the error filter, so we instead manually check that it matches
    # here
    for w in warnrec:
        # Should always be true due to the filters above
        assert issubclass(w.category, warningcls)
        if not re.compile(match, re.I).match(str(w.message)):
            raise Failed(f"Failed: WRONG MESSAGE. A warning with of the correct category ({warningcls.__name__}) was issued, but it did not match the given match regex ({match!r})")

    if test_stacklevel:
        for f in inspect.stack():
            thisfile = f.filename
            file = os.path.split(thisfile)[1]
            if file.startswith('test_'):
                break
            elif file == 'doctest.py':
                # skip the stacklevel testing in the doctests of this
                # function
                return
        else:
            raise RuntimeError("Could not find the file for the given warning to test the stacklevel")
        for w in warnrec:
            if w.filename != thisfile:
                msg = f'''\
Failed: Warning has the wrong stacklevel. The warning stacklevel needs to be
set so that the line of code shown in the warning message is user code that
calls the deprecated code (the current stacklevel is showing code from
{w.filename} (line {w.lineno}), expected {thisfile})'''.replace('\n', ' ')
                raise Failed(msg)

    if warningcls == SymPyDeprecationWarning:
        this_file = pathlib.Path(__file__)
        active_deprecations_file = (this_file.parent.parent.parent / 'doc' /
                                    'src' / 'explanation' /
                                    'active-deprecations.md')
        if not active_deprecations_file.exists():
            # We can only test that the active_deprecations_target works if we are
            # in the git repo.
            return
        targets = []
        for w in warnrec:
            targets.append(w.message.active_deprecations_target)
        with open(active_deprecations_file, encoding="utf-8") as f:
            text = f.read()
        for target in targets:
            if f'({target})=' not in text:
                raise Failed(f"The active deprecations target {target!r} does not appear to be a valid target in the active-deprecations.md file ({active_deprecations_file}).")

def _both_exp_pow(func):
    """
    Decorator used to run the test twice: the first time `e^x` is represented
    as ``Pow(E, x)``, the second time as ``exp(x)`` (exponential object is not
    a power).

    This is a temporary trick helping to manage the elimination of the class
    ``exp`` in favor of a replacement by ``Pow(E, ...)``.
    """
    from sympy.core.parameters import _exp_is_pow

    def func_wrap():
        with _exp_is_pow(True):
            func()
        with _exp_is_pow(False):
            func()

    wrapper = functools.update_wrapper(func_wrap, func)
    return wrapper


@contextlib.contextmanager
def warns_deprecated_sympy():
    '''
    Shorthand for ``warns(SymPyDeprecationWarning)``

    This is the recommended way to test that ``SymPyDeprecationWarning`` is
    emitted for deprecated features in SymPy. To test for other warnings use
    ``warns``. To suppress warnings without asserting that they are emitted
    use ``ignore_warnings``.

    .. note::

       ``warns_deprecated_sympy()`` is only intended for internal use in the
       SymPy test suite to test that a deprecation warning triggers properly.
       All other code in the SymPy codebase, including documentation examples,
       should not use deprecated behavior.

       If you are a user of SymPy and you want to disable
       SymPyDeprecationWarnings, use ``warnings`` filters (see
       :ref:`silencing-sympy-deprecation-warnings`).

    >>> from sympy.testing.pytest import warns_deprecated_sympy
    >>> from sympy.utilities.exceptions import sympy_deprecation_warning
    >>> with warns_deprecated_sympy():
    ...     sympy_deprecation_warning("Don't use",
    ...        deprecated_since_version="1.0",
    ...        active_deprecations_target="active-deprecations")

    >>> with warns_deprecated_sympy():
    ...     pass
    Traceback (most recent call last):
    ...
    Failed: DID NOT WARN. No warnings of type \
    SymPyDeprecationWarning was emitted. The list of emitted warnings is: [].

    .. note::

       Sometimes the stacklevel test will fail because the same warning is
       emitted multiple times. In this case, you can use
       :func:`sympy.utilities.exceptions.ignore_warnings` in the code to
       prevent the ``SymPyDeprecationWarning`` from being emitted again
       recursively. In rare cases it is impossible to have a consistent
       ``stacklevel`` for deprecation warnings because different ways of
       calling a function will produce different call stacks.. In those cases,
       use ``warns(SymPyDeprecationWarning)`` instead.

    See Also
    ========
    sympy.utilities.exceptions.SymPyDeprecationWarning
    sympy.utilities.exceptions.sympy_deprecation_warning
    sympy.utilities.decorator.deprecated

    '''
    with warns(SymPyDeprecationWarning):
        yield


def _running_under_pyodide():
    """Test if running under pyodide."""
    try:
        import pyodide_js  # type: ignore  # noqa
    except ImportError:
        return False
    else:
        return True


def skip_under_pyodide(message):
    """Decorator to skip a test if running under pyodide."""
    def decorator(test_func):
        @functools.wraps(test_func)
        def test_wrapper():
            if _running_under_pyodide():
                skip(message)
            return test_func()
        return test_wrapper
    return decorator
