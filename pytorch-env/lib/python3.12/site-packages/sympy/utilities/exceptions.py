"""
General SymPy exceptions and warnings.
"""

import warnings
import contextlib

from textwrap import dedent


class SymPyDeprecationWarning(DeprecationWarning):
    r"""
    A warning for deprecated features of SymPy.

    See the :ref:`deprecation-policy` document for details on when and how
    things should be deprecated in SymPy.

    Note that simply constructing this class will not cause a warning to be
    issued. To do that, you must call the :func`sympy_deprecation_warning`
    function. For this reason, it is not recommended to ever construct this
    class directly.

    Explanation
    ===========

    The ``SymPyDeprecationWarning`` class is a subclass of
    ``DeprecationWarning`` that is used for all deprecations in SymPy. A
    special subclass is used so that we can automatically augment the warning
    message with additional metadata about the version the deprecation was
    introduced in and a link to the documentation. This also allows users to
    explicitly filter deprecation warnings from SymPy using ``warnings``
    filters (see :ref:`silencing-sympy-deprecation-warnings`).

    Additionally, ``SymPyDeprecationWarning`` is enabled to be shown by
    default, unlike normal ``DeprecationWarning``\s, which are only shown by
    default in interactive sessions. This ensures that deprecation warnings in
    SymPy will actually be seen by users.

    See the documentation of :func:`sympy_deprecation_warning` for a
    description of the parameters to this function.

    To mark a function as deprecated, you can use the :func:`@deprecated
    <sympy.utilities.decorator.deprecated>` decorator.

    See Also
    ========
    sympy.utilities.exceptions.sympy_deprecation_warning
    sympy.utilities.exceptions.ignore_warnings
    sympy.utilities.decorator.deprecated
    sympy.testing.pytest.warns_deprecated_sympy

    """
    def __init__(self, message, *, deprecated_since_version, active_deprecations_target):

        super().__init__(message, deprecated_since_version,
                     active_deprecations_target)
        self.message = message
        if not isinstance(deprecated_since_version, str):
            raise TypeError(f"'deprecated_since_version' should be a string, got {deprecated_since_version!r}")
        self.deprecated_since_version = deprecated_since_version
        self.active_deprecations_target = active_deprecations_target
        if any(i in active_deprecations_target for i in '()='):
            raise ValueError("active_deprecations_target be the part inside of the '(...)='")

        self.full_message = f"""

{dedent(message).strip()}

See https://docs.sympy.org/latest/explanation/active-deprecations.html#{active_deprecations_target}
for details.

This has been deprecated since SymPy version {deprecated_since_version}. It
will be removed in a future version of SymPy.
"""

    def __str__(self):
        return self.full_message

    def __repr__(self):
        return f"{self.__class__.__name__}({self.message!r}, deprecated_since_version={self.deprecated_since_version!r}, active_deprecations_target={self.active_deprecations_target!r})"

    def __eq__(self, other):
        return isinstance(other, SymPyDeprecationWarning) and self.args == other.args

    # Make pickling work. The by default, it tries to recreate the expression
    # from its args, but this doesn't work because of our keyword-only
    # arguments.
    @classmethod
    def _new(cls, message, deprecated_since_version,
              active_deprecations_target):
        return cls(message, deprecated_since_version=deprecated_since_version, active_deprecations_target=active_deprecations_target)

    def __reduce__(self):
        return (self._new, (self.message, self.deprecated_since_version, self.active_deprecations_target))

# Python by default hides DeprecationWarnings, which we do not want.
warnings.simplefilter("once", SymPyDeprecationWarning)

def sympy_deprecation_warning(message, *, deprecated_since_version,
                              active_deprecations_target, stacklevel=3):
    r'''
    Warn that a feature is deprecated in SymPy.

    See the :ref:`deprecation-policy` document for details on when and how
    things should be deprecated in SymPy.

    To mark an entire function or class as deprecated, you can use the
    :func:`@deprecated <sympy.utilities.decorator.deprecated>` decorator.

    Parameters
    ==========

    message : str
         The deprecation message. This may span multiple lines and contain
         code examples. Messages should be wrapped to 80 characters. The
         message is automatically dedented and leading and trailing whitespace
         stripped. Messages may include dynamic content based on the user
         input, but avoid using ``str(expression)`` if an expression can be
         arbitrary, as it might be huge and make the warning message
         unreadable.

    deprecated_since_version : str
         The version of SymPy the feature has been deprecated since. For new
         deprecations, this should be the version in `sympy/release.py
         <https://github.com/sympy/sympy/blob/master/sympy/release.py>`_
         without the ``.dev``. If the next SymPy version ends up being
         different from this, the release manager will need to update any
         ``SymPyDeprecationWarning``\s using the incorrect version. This
         argument is required and must be passed as a keyword argument.
         (example:  ``deprecated_since_version="1.10"``).

    active_deprecations_target : str
        The Sphinx target corresponding to the section for the deprecation in
        the :ref:`active-deprecations` document (see
        ``doc/src/explanation/active-deprecations.md``). This is used to
        automatically generate a URL to the page in the warning message. This
        argument is required and must be passed as a keyword argument.
        (example: ``active_deprecations_target="deprecated-feature-abc"``)

    stacklevel : int, default: 3
        The ``stacklevel`` parameter that is passed to ``warnings.warn``. If
        you create a wrapper that calls this function, this should be
        increased so that the warning message shows the user line of code that
        produced the warning. Note that in some cases there will be multiple
        possible different user code paths that could result in the warning.
        In that case, just choose the smallest common stacklevel.

    Examples
    ========

    >>> from sympy.utilities.exceptions import sympy_deprecation_warning
    >>> def is_this_zero(x, y=0):
    ...     """
    ...     Determine if x = 0.
    ...
    ...     Parameters
    ...     ==========
    ...
    ...     x : Expr
    ...       The expression to check.
    ...
    ...     y : Expr, optional
    ...       If provided, check if x = y.
    ...
    ...       .. deprecated:: 1.1
    ...
    ...          The ``y`` argument to ``is_this_zero`` is deprecated. Use
    ...          ``is_this_zero(x - y)`` instead.
    ...
    ...     """
    ...     from sympy import simplify
    ...
    ...     if y != 0:
    ...         sympy_deprecation_warning("""
    ...     The y argument to is_zero() is deprecated. Use is_zero(x - y) instead.""",
    ...             deprecated_since_version="1.1",
    ...             active_deprecations_target='is-this-zero-y-deprecation')
    ...     return simplify(x - y) == 0
    >>> is_this_zero(0)
    True
    >>> is_this_zero(1, 1) # doctest: +SKIP
    <stdin>:1: SymPyDeprecationWarning:
    <BLANKLINE>
    The y argument to is_zero() is deprecated. Use is_zero(x - y) instead.
    <BLANKLINE>
    See https://docs.sympy.org/latest/explanation/active-deprecations.html#is-this-zero-y-deprecation
    for details.
    <BLANKLINE>
    This has been deprecated since SymPy version 1.1. It
    will be removed in a future version of SymPy.
    <BLANKLINE>
      is_this_zero(1, 1)
    True

    See Also
    ========

    sympy.utilities.exceptions.SymPyDeprecationWarning
    sympy.utilities.exceptions.ignore_warnings
    sympy.utilities.decorator.deprecated
    sympy.testing.pytest.warns_deprecated_sympy

    '''
    w = SymPyDeprecationWarning(message,
                            deprecated_since_version=deprecated_since_version,
                                active_deprecations_target=active_deprecations_target)
    warnings.warn(w, stacklevel=stacklevel)


@contextlib.contextmanager
def ignore_warnings(warningcls):
    '''
    Context manager to suppress warnings during tests.

    .. note::

       Do not use this with SymPyDeprecationWarning in the tests.
       warns_deprecated_sympy() should be used instead.

    This function is useful for suppressing warnings during tests. The warns
    function should be used to assert that a warning is raised. The
    ignore_warnings function is useful in situation when the warning is not
    guaranteed to be raised (e.g. on importing a module) or if the warning
    comes from third-party code.

    This function is also useful to prevent the same or similar warnings from
    being issue twice due to recursive calls.

    When the warning is coming (reliably) from SymPy the warns function should
    be preferred to ignore_warnings.

    >>> from sympy.utilities.exceptions import ignore_warnings
    >>> import warnings

    Here's a warning:

    >>> with warnings.catch_warnings():  # reset warnings in doctest
    ...     warnings.simplefilter('error')
    ...     warnings.warn('deprecated', UserWarning)
    Traceback (most recent call last):
      ...
    UserWarning: deprecated

    Let's suppress it with ignore_warnings:

    >>> with warnings.catch_warnings():  # reset warnings in doctest
    ...     warnings.simplefilter('error')
    ...     with ignore_warnings(UserWarning):
    ...         warnings.warn('deprecated', UserWarning)

    (No warning emitted)

    See Also
    ========
    sympy.utilities.exceptions.SymPyDeprecationWarning
    sympy.utilities.exceptions.sympy_deprecation_warning
    sympy.utilities.decorator.deprecated
    sympy.testing.pytest.warns_deprecated_sympy

    '''
    # Absorbs all warnings in warnrec
    with warnings.catch_warnings(record=True) as warnrec:
        # Make sure our warning doesn't get filtered
        warnings.simplefilter("always", warningcls)
        # Now run the test
        yield

    # Reissue any warnings that we aren't testing for
    for w in warnrec:
        if not issubclass(w.category, warningcls):
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
