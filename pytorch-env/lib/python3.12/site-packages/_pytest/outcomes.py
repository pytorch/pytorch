"""Exception classes and constants handling test outcomes as well as
functions creating them."""

from __future__ import annotations

import sys
from typing import Any
from typing import Callable
from typing import cast
from typing import NoReturn
from typing import Protocol
from typing import Type
from typing import TypeVar

from .warning_types import PytestDeprecationWarning


class OutcomeException(BaseException):
    """OutcomeException and its subclass instances indicate and contain info
    about test and collection outcomes."""

    def __init__(self, msg: str | None = None, pytrace: bool = True) -> None:
        if msg is not None and not isinstance(msg, str):
            error_msg = (  # type: ignore[unreachable]
                "{} expected string as 'msg' parameter, got '{}' instead.\n"
                "Perhaps you meant to use a mark?"
            )
            raise TypeError(error_msg.format(type(self).__name__, type(msg).__name__))
        super().__init__(msg)
        self.msg = msg
        self.pytrace = pytrace

    def __repr__(self) -> str:
        if self.msg is not None:
            return self.msg
        return f"<{self.__class__.__name__} instance>"

    __str__ = __repr__


TEST_OUTCOME = (OutcomeException, Exception)


class Skipped(OutcomeException):
    # XXX hackish: on 3k we fake to live in the builtins
    # in order to have Skipped exception printing shorter/nicer
    __module__ = "builtins"

    def __init__(
        self,
        msg: str | None = None,
        pytrace: bool = True,
        allow_module_level: bool = False,
        *,
        _use_item_location: bool = False,
    ) -> None:
        super().__init__(msg=msg, pytrace=pytrace)
        self.allow_module_level = allow_module_level
        # If true, the skip location is reported as the item's location,
        # instead of the place that raises the exception/calls skip().
        self._use_item_location = _use_item_location


class Failed(OutcomeException):
    """Raised from an explicit call to pytest.fail()."""

    __module__ = "builtins"


class Exit(Exception):
    """Raised for immediate program exits (no tracebacks/summaries)."""

    def __init__(
        self, msg: str = "unknown reason", returncode: int | None = None
    ) -> None:
        self.msg = msg
        self.returncode = returncode
        super().__init__(msg)


# Elaborate hack to work around https://github.com/python/mypy/issues/2087.
# Ideally would just be `exit.Exception = Exit` etc.

_F = TypeVar("_F", bound=Callable[..., object])
_ET = TypeVar("_ET", bound=Type[BaseException])


class _WithException(Protocol[_F, _ET]):
    Exception: _ET
    __call__: _F


def _with_exception(exception_type: _ET) -> Callable[[_F], _WithException[_F, _ET]]:
    def decorate(func: _F) -> _WithException[_F, _ET]:
        func_with_exception = cast(_WithException[_F, _ET], func)
        func_with_exception.Exception = exception_type
        return func_with_exception

    return decorate


# Exposed helper methods.


@_with_exception(Exit)
def exit(
    reason: str = "",
    returncode: int | None = None,
) -> NoReturn:
    """Exit testing process.

    :param reason:
        The message to show as the reason for exiting pytest.  reason has a default value
        only because `msg` is deprecated.

    :param returncode:
        Return code to be used when exiting pytest. None means the same as ``0`` (no error), same as :func:`sys.exit`.

    :raises pytest.exit.Exception:
        The exception that is raised.
    """
    __tracebackhide__ = True
    raise Exit(reason, returncode)


@_with_exception(Skipped)
def skip(
    reason: str = "",
    *,
    allow_module_level: bool = False,
) -> NoReturn:
    """Skip an executing test with the given message.

    This function should be called only during testing (setup, call or teardown) or
    during collection by using the ``allow_module_level`` flag.  This function can
    be called in doctests as well.

    :param reason:
        The message to show the user as reason for the skip.

    :param allow_module_level:
        Allows this function to be called at module level.
        Raising the skip exception at module level will stop
        the execution of the module and prevent the collection of all tests in the module,
        even those defined before the `skip` call.

        Defaults to False.

    :raises pytest.skip.Exception:
        The exception that is raised.

    .. note::
        It is better to use the :ref:`pytest.mark.skipif ref` marker when
        possible to declare a test to be skipped under certain conditions
        like mismatching platforms or dependencies.
        Similarly, use the ``# doctest: +SKIP`` directive (see :py:data:`doctest.SKIP`)
        to skip a doctest statically.
    """
    __tracebackhide__ = True
    raise Skipped(msg=reason, allow_module_level=allow_module_level)


@_with_exception(Failed)
def fail(reason: str = "", pytrace: bool = True) -> NoReturn:
    """Explicitly fail an executing test with the given message.

    :param reason:
        The message to show the user as reason for the failure.

    :param pytrace:
        If False, msg represents the full failure information and no
        python traceback will be reported.

    :raises pytest.fail.Exception:
        The exception that is raised.
    """
    __tracebackhide__ = True
    raise Failed(msg=reason, pytrace=pytrace)


class XFailed(Failed):
    """Raised from an explicit call to pytest.xfail()."""


@_with_exception(XFailed)
def xfail(reason: str = "") -> NoReturn:
    """Imperatively xfail an executing test or setup function with the given reason.

    This function should be called only during testing (setup, call or teardown).

    No other code is executed after using ``xfail()`` (it is implemented
    internally by raising an exception).

    :param reason:
        The message to show the user as reason for the xfail.

    .. note::
        It is better to use the :ref:`pytest.mark.xfail ref` marker when
        possible to declare a test to be xfailed under certain conditions
        like known bugs or missing features.

    :raises pytest.xfail.Exception:
        The exception that is raised.
    """
    __tracebackhide__ = True
    raise XFailed(reason)


def importorskip(
    modname: str,
    minversion: str | None = None,
    reason: str | None = None,
    *,
    exc_type: type[ImportError] | None = None,
) -> Any:
    """Import and return the requested module ``modname``, or skip the
    current test if the module cannot be imported.

    :param modname:
        The name of the module to import.
    :param minversion:
        If given, the imported module's ``__version__`` attribute must be at
        least this minimal version, otherwise the test is still skipped.
    :param reason:
        If given, this reason is shown as the message when the module cannot
        be imported.
    :param exc_type:
        The exception that should be captured in order to skip modules.
        Must be :py:class:`ImportError` or a subclass.

        If the module can be imported but raises :class:`ImportError`, pytest will
        issue a warning to the user, as often users expect the module not to be
        found (which would raise :class:`ModuleNotFoundError` instead).

        This warning can be suppressed by passing ``exc_type=ImportError`` explicitly.

        See :ref:`import-or-skip-import-error` for details.


    :returns:
        The imported module. This should be assigned to its canonical name.

    :raises pytest.skip.Exception:
        If the module cannot be imported.

    Example::

        docutils = pytest.importorskip("docutils")

    .. versionadded:: 8.2

        The ``exc_type`` parameter.
    """
    import warnings

    __tracebackhide__ = True
    compile(modname, "", "eval")  # to catch syntaxerrors

    # Until pytest 9.1, we will warn the user if we catch ImportError (instead of ModuleNotFoundError),
    # as this might be hiding an installation/environment problem, which is not usually what is intended
    # when using importorskip() (#11523).
    # In 9.1, to keep the function signature compatible, we just change the code below to:
    # 1. Use `exc_type = ModuleNotFoundError` if `exc_type` is not given.
    # 2. Remove `warn_on_import` and the warning handling.
    if exc_type is None:
        exc_type = ImportError
        warn_on_import_error = True
    else:
        warn_on_import_error = False

    skipped: Skipped | None = None
    warning: Warning | None = None

    with warnings.catch_warnings():
        # Make sure to ignore ImportWarnings that might happen because
        # of existing directories with the same name we're trying to
        # import but without a __init__.py file.
        warnings.simplefilter("ignore")

        try:
            __import__(modname)
        except exc_type as exc:
            # Do not raise or issue warnings inside the catch_warnings() block.
            if reason is None:
                reason = f"could not import {modname!r}: {exc}"
            skipped = Skipped(reason, allow_module_level=True)

            if warn_on_import_error and not isinstance(exc, ModuleNotFoundError):
                lines = [
                    "",
                    f"Module '{modname}' was found, but when imported by pytest it raised:",
                    f"    {exc!r}",
                    "In pytest 9.1 this warning will become an error by default.",
                    "You can fix the underlying problem, or alternatively overwrite this behavior and silence this "
                    "warning by passing exc_type=ImportError explicitly.",
                    "See https://docs.pytest.org/en/stable/deprecations.html#pytest-importorskip-default-behavior-regarding-importerror",
                ]
                warning = PytestDeprecationWarning("\n".join(lines))

    if warning:
        warnings.warn(warning, stacklevel=2)
    if skipped:
        raise skipped

    mod = sys.modules[modname]
    if minversion is None:
        return mod
    verattr = getattr(mod, "__version__", None)
    if minversion is not None:
        # Imported lazily to improve start-up time.
        from packaging.version import Version

        if verattr is None or Version(verattr) < Version(minversion):
            raise Skipped(
                f"module {modname!r} has __version__ {verattr!r}, required is: {minversion!r}",
                allow_module_level=True,
            )
    return mod
