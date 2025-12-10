import os
USE_SYMENGINE = os.getenv('USE_SYMENGINE', '0')
USE_SYMENGINE = USE_SYMENGINE.lower() in ('1', 't', 'true')  # type: ignore

if USE_SYMENGINE:
    from symengine import (Symbol, Integer, sympify as sympify_symengine, S,
        SympifyError, exp, log, gamma, sqrt, I, E, pi, Matrix,
        sin, cos, tan, cot, csc, sec, asin, acos, atan, acot, acsc, asec,
        sinh, cosh, tanh, coth, asinh, acosh, atanh, acoth,
        lambdify, symarray, diff, zeros, eye, diag, ones,
        expand, Function, symbols, var, Add, Mul, Derivative,
        ImmutableMatrix, MatrixBase, Rational, Basic)
    from symengine.lib.symengine_wrapper import gcd as igcd
    from symengine import AppliedUndef

    def sympify(a, *, strict=False):
        """
        Notes
        =====

        SymEngine's ``sympify`` does not accept keyword arguments and is
        therefore not compatible with SymPy's ``sympify`` with ``strict=True``
        (which ensures that only the types for which an explicit conversion has
        been defined are converted). This wrapper adds an additional parameter
        ``strict`` (with default ``False``) that will raise a ``SympifyError``
        if ``strict=True`` and the argument passed to the parameter ``a`` is a
        string.

        See Also
        ========

        sympify: Converts an arbitrary expression to a type that can be used
            inside SymPy.

        """
        # The parameter ``a`` is used for this function to keep compatibility
        # with the SymEngine docstring.
        if strict and isinstance(a, str):
            raise SympifyError(a)
        return sympify_symengine(a)

    # Keep the SymEngine docstring and append the additional "Notes" and "See
    # Also" sections. Replacement of spaces is required to correctly format the
    # indentation of the combined docstring.
    sympify.__doc__ = (
        sympify_symengine.__doc__
        + sympify.__doc__.replace('        ', '    ')  # type: ignore
    )
else:
    from sympy.core.add import Add
    from sympy.core.basic import Basic
    from sympy.core.function import (diff, Function, AppliedUndef,
        expand, Derivative)
    from sympy.core.mul import Mul
    from sympy.core.intfunc import igcd
    from sympy.core.numbers import pi, I, Integer, Rational, E
    from sympy.core.singleton import S
    from sympy.core.symbol import Symbol, var, symbols
    from sympy.core.sympify import SympifyError, sympify
    from sympy.functions.elementary.exponential import log, exp
    from sympy.functions.elementary.hyperbolic import (coth, sinh,
        acosh, acoth, tanh, asinh, atanh, cosh)
    from sympy.functions.elementary.miscellaneous import sqrt
    from sympy.functions.elementary.trigonometric import (csc,
        asec, cos, atan, sec, acot, asin, tan, sin, cot, acsc, acos)
    from sympy.functions.special.gamma_functions import gamma
    from sympy.matrices.dense import (eye, zeros, diag, Matrix,
        ones, symarray)
    from sympy.matrices.immutable import ImmutableMatrix
    from sympy.matrices.matrixbase import MatrixBase
    from sympy.utilities.lambdify import lambdify


#
# XXX: Handling of immutable and mutable matrices in SymEngine is inconsistent
# with SymPy's matrix classes in at least SymEngine version 0.7.0. Until that
# is fixed the function below is needed for consistent behaviour when
# attempting to simplify a matrix.
#
# Expected behaviour of a SymPy mutable/immutable matrix .simplify() method:
#
#   Matrix.simplify() : works in place, returns None
#   ImmutableMatrix.simplify() : returns a simplified copy
#
# In SymEngine both mutable and immutable matrices simplify in place and return
# None. This is inconsistent with the matrix being "immutable" and also the
# returned None leads to problems in the mechanics module.
#
# The simplify function should not be used because simplify(M) sympifies the
# matrix M and the SymEngine matrices all sympify to SymPy matrices. If we want
# to work with SymEngine matrices then we need to use their .simplify() method
# but that method does not work correctly with immutable matrices.
#
# The _simplify_matrix function can be removed when the SymEngine bug is fixed.
# Since this should be a temporary problem we do not make this function part of
# the public API.
#
#   SymEngine issue: https://github.com/symengine/symengine.py/issues/363
#

def _simplify_matrix(M):
    """Return a simplified copy of the matrix M"""
    if not isinstance(M, (Matrix, ImmutableMatrix)):
        raise TypeError("The matrix M must be an instance of Matrix or ImmutableMatrix")
    Mnew = M.as_mutable() # makes a copy if mutable
    Mnew.simplify()
    if isinstance(M, ImmutableMatrix):
        Mnew = Mnew.as_immutable()
    return Mnew


__all__ = [
    'Symbol', 'Integer', 'sympify', 'S', 'SympifyError', 'exp', 'log',
    'gamma', 'sqrt', 'I', 'E', 'pi', 'Matrix', 'sin', 'cos', 'tan', 'cot',
    'csc', 'sec', 'asin', 'acos', 'atan', 'acot', 'acsc', 'asec', 'sinh',
    'cosh', 'tanh', 'coth', 'asinh', 'acosh', 'atanh', 'acoth', 'lambdify',
    'symarray', 'diff', 'zeros', 'eye', 'diag', 'ones', 'expand', 'Function',
    'symbols', 'var', 'Add', 'Mul', 'Derivative', 'ImmutableMatrix',
    'MatrixBase', 'Rational', 'Basic', 'igcd', 'AppliedUndef',
]
