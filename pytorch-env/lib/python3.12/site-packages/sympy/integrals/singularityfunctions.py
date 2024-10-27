from sympy.functions import SingularityFunction, DiracDelta
from sympy.integrals import integrate


def singularityintegrate(f, x):
    """
    This function handles the indefinite integrations of Singularity functions.
    The ``integrate`` function calls this function internally whenever an
    instance of SingularityFunction is passed as argument.

    Explanation
    ===========

    The idea for integration is the following:

    - If we are dealing with a SingularityFunction expression,
      i.e. ``SingularityFunction(x, a, n)``, we just return
      ``SingularityFunction(x, a, n + 1)/(n + 1)`` if ``n >= 0`` and
      ``SingularityFunction(x, a, n + 1)`` if ``n < 0``.

    - If the node is a multiplication or power node having a
      SingularityFunction term we rewrite the whole expression in terms of
      Heaviside and DiracDelta and then integrate the output. Lastly, we
      rewrite the output of integration back in terms of SingularityFunction.

    - If none of the above case arises, we return None.

    Examples
    ========

    >>> from sympy.integrals.singularityfunctions import singularityintegrate
    >>> from sympy import SingularityFunction, symbols, Function
    >>> x, a, n, y = symbols('x a n y')
    >>> f = Function('f')
    >>> singularityintegrate(SingularityFunction(x, a, 3), x)
    SingularityFunction(x, a, 4)/4
    >>> singularityintegrate(5*SingularityFunction(x, 5, -2), x)
    5*SingularityFunction(x, 5, -1)
    >>> singularityintegrate(6*SingularityFunction(x, 5, -1), x)
    6*SingularityFunction(x, 5, 0)
    >>> singularityintegrate(x*SingularityFunction(x, 0, -1), x)
    0
    >>> singularityintegrate(SingularityFunction(x, 1, -1) * f(x), x)
    f(1)*SingularityFunction(x, 1, 0)

    """

    if not f.has(SingularityFunction):
        return None

    if isinstance(f, SingularityFunction):
        x, a, n = f.args
        if n.is_positive or n.is_zero:
            return SingularityFunction(x, a, n + 1)/(n + 1)
        elif n in (-1, -2, -3, -4):
            return SingularityFunction(x, a, n + 1)

    if f.is_Mul or f.is_Pow:

        expr = f.rewrite(DiracDelta)
        expr = integrate(expr, x)
        return expr.rewrite(SingularityFunction)
    return None
