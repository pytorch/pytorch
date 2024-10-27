""" Functions to support rewriting of SymPy expressions """

from sympy.core.expr import Expr
from sympy.assumptions import ask
from sympy.strategies.tools import subs
from sympy.unify.usympy import rebuild, unify

def rewriterule(source, target, variables=(), condition=None, assume=None):
    """ Rewrite rule.

    Transform expressions that match source into expressions that match target
    treating all ``variables`` as wilds.

    Examples
    ========

    >>> from sympy.abc import w, x, y, z
    >>> from sympy.unify.rewrite import rewriterule
    >>> from sympy import default_sort_key
    >>> rl = rewriterule(x + y, x**y, [x, y])
    >>> sorted(rl(z + 3), key=default_sort_key)
    [3**z, z**3]

    Use ``condition`` to specify additional requirements.  Inputs are taken in
    the same order as is found in variables.

    >>> rl = rewriterule(x + y, x**y, [x, y], lambda x, y: x.is_integer)
    >>> list(rl(z + 3))
    [3**z]

    Use ``assume`` to specify additional requirements using new assumptions.

    >>> from sympy.assumptions import Q
    >>> rl = rewriterule(x + y, x**y, [x, y], assume=Q.integer(x))
    >>> list(rl(z + 3))
    [3**z]

    Assumptions for the local context are provided at rule runtime

    >>> list(rl(w + z, Q.integer(z)))
    [z**w]
    """

    def rewrite_rl(expr, assumptions=True):
        for match in unify(source, expr, {}, variables=variables):
            if (condition and
                not condition(*[match.get(var, var) for var in variables])):
                continue
            if (assume and not ask(assume.xreplace(match), assumptions)):
                continue
            expr2 = subs(match)(target)
            if isinstance(expr2, Expr):
                expr2 = rebuild(expr2)
            yield expr2
    return rewrite_rl
