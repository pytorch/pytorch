""" Generic Rules for SymPy

This file assumes knowledge of Basic and little else.
"""
from sympy.utilities.iterables import sift
from .util import new


# Functions that create rules
def rm_id(isid, new=new):
    """ Create a rule to remove identities.

    isid - fn :: x -> Bool  --- whether or not this element is an identity.

    Examples
    ========

    >>> from sympy.strategies import rm_id
    >>> from sympy import Basic, S
    >>> remove_zeros = rm_id(lambda x: x==0)
    >>> remove_zeros(Basic(S(1), S(0), S(2)))
    Basic(1, 2)
    >>> remove_zeros(Basic(S(0), S(0))) # If only identites then we keep one
    Basic(0)

    See Also:
        unpack
    """
    def ident_remove(expr):
        """ Remove identities """
        ids = list(map(isid, expr.args))
        if sum(ids) == 0:           # No identities. Common case
            return expr
        elif sum(ids) != len(ids):  # there is at least one non-identity
            return new(expr.__class__,
                       *[arg for arg, x in zip(expr.args, ids) if not x])
        else:
            return new(expr.__class__, expr.args[0])

    return ident_remove


def glom(key, count, combine):
    """ Create a rule to conglomerate identical args.

    Examples
    ========

    >>> from sympy.strategies import glom
    >>> from sympy import Add
    >>> from sympy.abc import x

    >>> key     = lambda x: x.as_coeff_Mul()[1]
    >>> count   = lambda x: x.as_coeff_Mul()[0]
    >>> combine = lambda cnt, arg: cnt * arg
    >>> rl = glom(key, count, combine)

    >>> rl(Add(x, -x, 3*x, 2, 3, evaluate=False))
    3*x + 5

    Wait, how are key, count and combine supposed to work?

    >>> key(2*x)
    x
    >>> count(2*x)
    2
    >>> combine(2, x)
    2*x
    """
    def conglomerate(expr):
        """ Conglomerate together identical args x + x -> 2x """
        groups = sift(expr.args, key)
        counts = {k: sum(map(count, args)) for k, args in groups.items()}
        newargs = [combine(cnt, mat) for mat, cnt in counts.items()]
        if set(newargs) != set(expr.args):
            return new(type(expr), *newargs)
        else:
            return expr

    return conglomerate


def sort(key, new=new):
    """ Create a rule to sort by a key function.

    Examples
    ========

    >>> from sympy.strategies import sort
    >>> from sympy import Basic, S
    >>> sort_rl = sort(str)
    >>> sort_rl(Basic(S(3), S(1), S(2)))
    Basic(1, 2, 3)
    """

    def sort_rl(expr):
        return new(expr.__class__, *sorted(expr.args, key=key))
    return sort_rl


def distribute(A, B):
    """ Turns an A containing Bs into a B of As

    where A, B are container types

    >>> from sympy.strategies import distribute
    >>> from sympy import Add, Mul, symbols
    >>> x, y = symbols('x,y')
    >>> dist = distribute(Mul, Add)
    >>> expr = Mul(2, x+y, evaluate=False)
    >>> expr
    2*(x + y)
    >>> dist(expr)
    2*x + 2*y
    """

    def distribute_rl(expr):
        for i, arg in enumerate(expr.args):
            if isinstance(arg, B):
                first, b, tail = expr.args[:i], expr.args[i], expr.args[i + 1:]
                return B(*[A(*(first + (arg,) + tail)) for arg in b.args])
        return expr
    return distribute_rl


def subs(a, b):
    """ Replace expressions exactly """
    def subs_rl(expr):
        if expr == a:
            return b
        else:
            return expr
    return subs_rl


# Functions that are rules
def unpack(expr):
    """ Rule to unpack singleton args

    >>> from sympy.strategies import unpack
    >>> from sympy import Basic, S
    >>> unpack(Basic(S(2)))
    2
    """
    if len(expr.args) == 1:
        return expr.args[0]
    else:
        return expr


def flatten(expr, new=new):
    """ Flatten T(a, b, T(c, d), T2(e)) to T(a, b, c, d, T2(e)) """
    cls = expr.__class__
    args = []
    for arg in expr.args:
        if arg.__class__ == cls:
            args.extend(arg.args)
        else:
            args.append(arg)
    return new(expr.__class__, *args)


def rebuild(expr):
    """ Rebuild a SymPy tree.

    Explanation
    ===========

    This function recursively calls constructors in the expression tree.
    This forces canonicalization and removes ugliness introduced by the use of
    Basic.__new__
    """
    if expr.is_Atom:
        return expr
    else:
        return expr.func(*list(map(rebuild, expr.args)))
