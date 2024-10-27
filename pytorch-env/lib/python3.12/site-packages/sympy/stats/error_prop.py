"""Tools for arithmetic error propagation."""

from itertools import repeat, combinations

from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.simplify.simplify import simplify
from sympy.stats.symbolic_probability import RandomSymbol, Variance, Covariance
from sympy.stats.rv import is_random

_arg0_or_var = lambda var: var.args[0] if len(var.args) > 0 else var


def variance_prop(expr, consts=(), include_covar=False):
    r"""Symbolically propagates variance (`\sigma^2`) for expressions.
    This is computed as as seen in [1]_.

    Parameters
    ==========

    expr : Expr
        A SymPy expression to compute the variance for.
    consts : sequence of Symbols, optional
        Represents symbols that are known constants in the expr,
        and thus have zero variance. All symbols not in consts are
        assumed to be variant.
    include_covar : bool, optional
        Flag for whether or not to include covariances, default=False.

    Returns
    =======

    var_expr : Expr
        An expression for the total variance of the expr.
        The variance for the original symbols (e.g. x) are represented
        via instance of the Variance symbol (e.g. Variance(x)).

    Examples
    ========

    >>> from sympy import symbols, exp
    >>> from sympy.stats.error_prop import variance_prop
    >>> x, y = symbols('x y')

    >>> variance_prop(x + y)
    Variance(x) + Variance(y)

    >>> variance_prop(x * y)
    x**2*Variance(y) + y**2*Variance(x)

    >>> variance_prop(exp(2*x))
    4*exp(4*x)*Variance(x)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Propagation_of_uncertainty

    """
    args = expr.args
    if len(args) == 0:
        if expr in consts:
            return S.Zero
        elif is_random(expr):
            return Variance(expr).doit()
        elif isinstance(expr, Symbol):
            return Variance(RandomSymbol(expr)).doit()
        else:
            return S.Zero
    nargs = len(args)
    var_args = list(map(variance_prop, args, repeat(consts, nargs),
                        repeat(include_covar, nargs)))
    if isinstance(expr, Add):
        var_expr = Add(*var_args)
        if include_covar:
            terms = [2 * Covariance(_arg0_or_var(x), _arg0_or_var(y)).expand() \
                     for x, y in combinations(var_args, 2)]
            var_expr += Add(*terms)
    elif isinstance(expr, Mul):
        terms = [v/a**2 for a, v in zip(args, var_args)]
        var_expr = simplify(expr**2 * Add(*terms))
        if include_covar:
            terms = [2*Covariance(_arg0_or_var(x), _arg0_or_var(y)).expand()/(a*b) \
                     for (a, b), (x, y) in zip(combinations(args, 2),
                                               combinations(var_args, 2))]
            var_expr += Add(*terms)
    elif isinstance(expr, Pow):
        b = args[1]
        v = var_args[0] * (expr * b / args[0])**2
        var_expr = simplify(v)
    elif isinstance(expr, exp):
        var_expr = simplify(var_args[0] * expr**2)
    else:
        # unknown how to proceed, return variance of whole expr.
        var_expr = Variance(expr)
    return var_expr
