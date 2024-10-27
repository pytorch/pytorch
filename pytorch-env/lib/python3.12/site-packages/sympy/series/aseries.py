from sympy.core.sympify import sympify


def aseries(expr, x=None, n=6, bound=0, hir=False):
    """
    See the docstring of Expr.aseries() for complete details of this wrapper.

    """
    expr = sympify(expr)
    return expr.aseries(x, n, bound, hir)
