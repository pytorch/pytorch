"""Definitions of monomial orderings. """

from __future__ import annotations

__all__ = ["lex", "grlex", "grevlex", "ilex", "igrlex", "igrevlex"]

from sympy.core import Symbol
from sympy.utilities.iterables import iterable

class MonomialOrder:
    """Base class for monomial orderings. """

    alias: str | None = None
    is_global: bool | None = None
    is_default = False

    def __repr__(self):
        return self.__class__.__name__ + "()"

    def __str__(self):
        return self.alias

    def __call__(self, monomial):
        raise NotImplementedError

    def __eq__(self, other):
        return self.__class__ == other.__class__

    def __hash__(self):
        return hash(self.__class__)

    def __ne__(self, other):
        return not (self == other)

class LexOrder(MonomialOrder):
    """Lexicographic order of monomials. """

    alias = 'lex'
    is_global = True
    is_default = True

    def __call__(self, monomial):
        return monomial

class GradedLexOrder(MonomialOrder):
    """Graded lexicographic order of monomials. """

    alias = 'grlex'
    is_global = True

    def __call__(self, monomial):
        return (sum(monomial), monomial)

class ReversedGradedLexOrder(MonomialOrder):
    """Reversed graded lexicographic order of monomials. """

    alias = 'grevlex'
    is_global = True

    def __call__(self, monomial):
        return (sum(monomial), tuple(reversed([-m for m in monomial])))

class ProductOrder(MonomialOrder):
    """
    A product order built from other monomial orders.

    Given (not necessarily total) orders O1, O2, ..., On, their product order
    P is defined as M1 > M2 iff there exists i such that O1(M1) = O2(M2),
    ..., Oi(M1) = Oi(M2), O{i+1}(M1) > O{i+1}(M2).

    Product orders are typically built from monomial orders on different sets
    of variables.

    ProductOrder is constructed by passing a list of pairs
    [(O1, L1), (O2, L2), ...] where Oi are MonomialOrders and Li are callables.
    Upon comparison, the Li are passed the total monomial, and should filter
    out the part of the monomial to pass to Oi.

    Examples
    ========

    We can use a lexicographic order on x_1, x_2 and also on
    y_1, y_2, y_3, and their product on {x_i, y_i} as follows:

    >>> from sympy.polys.orderings import lex, grlex, ProductOrder
    >>> P = ProductOrder(
    ...     (lex, lambda m: m[:2]), # lex order on x_1 and x_2 of monomial
    ...     (grlex, lambda m: m[2:]) # grlex on y_1, y_2, y_3
    ... )
    >>> P((2, 1, 1, 0, 0)) > P((1, 10, 0, 2, 0))
    True

    Here the exponent `2` of `x_1` in the first monomial
    (`x_1^2 x_2 y_1`) is bigger than the exponent `1` of `x_1` in the
    second monomial (`x_1 x_2^10 y_2^2`), so the first monomial is greater
    in the product ordering.

    >>> P((2, 1, 1, 0, 0)) < P((2, 1, 0, 2, 0))
    True

    Here the exponents of `x_1` and `x_2` agree, so the grlex order on
    `y_1, y_2, y_3` is used to decide the ordering. In this case the monomial
    `y_2^2` is ordered larger than `y_1`, since for the grlex order the degree
    of the monomial is most important.
    """

    def __init__(self, *args):
        self.args = args

    def __call__(self, monomial):
        return tuple(O(lamda(monomial)) for (O, lamda) in self.args)

    def __repr__(self):
        contents = [repr(x[0]) for x in self.args]
        return self.__class__.__name__ + '(' + ", ".join(contents) + ')'

    def __str__(self):
        contents = [str(x[0]) for x in self.args]
        return self.__class__.__name__ + '(' + ", ".join(contents) + ')'

    def __eq__(self, other):
        if not isinstance(other, ProductOrder):
            return False
        return self.args == other.args

    def __hash__(self):
        return hash((self.__class__, self.args))

    @property
    def is_global(self):
        if all(o.is_global is True for o, _ in self.args):
            return True
        if all(o.is_global is False for o, _ in self.args):
            return False
        return None

class InverseOrder(MonomialOrder):
    """
    The "inverse" of another monomial order.

    If O is any monomial order, we can construct another monomial order iO
    such that `A >_{iO} B` if and only if `B >_O A`. This is useful for
    constructing local orders.

    Note that many algorithms only work with *global* orders.

    For example, in the inverse lexicographic order on a single variable `x`,
    high powers of `x` count as small:

    >>> from sympy.polys.orderings import lex, InverseOrder
    >>> ilex = InverseOrder(lex)
    >>> ilex((5,)) < ilex((0,))
    True
    """

    def __init__(self, O):
        self.O = O

    def __str__(self):
        return "i" + str(self.O)

    def __call__(self, monomial):
        def inv(l):
            if iterable(l):
                return tuple(inv(x) for x in l)
            return -l
        return inv(self.O(monomial))

    @property
    def is_global(self):
        if self.O.is_global is True:
            return False
        if self.O.is_global is False:
            return True
        return None

    def __eq__(self, other):
        return isinstance(other, InverseOrder) and other.O == self.O

    def __hash__(self):
        return hash((self.__class__, self.O))

lex = LexOrder()
grlex = GradedLexOrder()
grevlex = ReversedGradedLexOrder()
ilex = InverseOrder(lex)
igrlex = InverseOrder(grlex)
igrevlex = InverseOrder(grevlex)

_monomial_key = {
    'lex': lex,
    'grlex': grlex,
    'grevlex': grevlex,
    'ilex': ilex,
    'igrlex': igrlex,
    'igrevlex': igrevlex
}

def monomial_key(order=None, gens=None):
    """
    Return a function defining admissible order on monomials.

    The result of a call to :func:`monomial_key` is a function which should
    be used as a key to :func:`sorted` built-in function, to provide order
    in a set of monomials of the same length.

    Currently supported monomial orderings are:

    1. lex       - lexicographic order (default)
    2. grlex     - graded lexicographic order
    3. grevlex   - reversed graded lexicographic order
    4. ilex, igrlex, igrevlex - the corresponding inverse orders

    If the ``order`` input argument is not a string but has ``__call__``
    attribute, then it will pass through with an assumption that the
    callable object defines an admissible order on monomials.

    If the ``gens`` input argument contains a list of generators, the
    resulting key function can be used to sort SymPy ``Expr`` objects.

    """
    if order is None:
        order = lex

    if isinstance(order, Symbol):
        order = str(order)

    if isinstance(order, str):
        try:
            order = _monomial_key[order]
        except KeyError:
            raise ValueError("supported monomial orderings are 'lex', 'grlex' and 'grevlex', got %r" % order)
    if hasattr(order, '__call__'):
        if gens is not None:
            def _order(expr):
                return order(expr.as_poly(*gens).degree_list())
            return _order
        return order
    else:
        raise ValueError("monomial ordering specification must be a string or a callable, got %s" % order)

class _ItemGetter:
    """Helper class to return a subsequence of values."""

    def __init__(self, seq):
        self.seq = tuple(seq)

    def __call__(self, m):
        return tuple(m[idx] for idx in self.seq)

    def __eq__(self, other):
        if not isinstance(other, _ItemGetter):
            return False
        return self.seq == other.seq

def build_product_order(arg, gens):
    """
    Build a monomial order on ``gens``.

    ``arg`` should be a tuple of iterables. The first element of each iterable
    should be a string or monomial order (will be passed to monomial_key),
    the others should be subsets of the generators. This function will build
    the corresponding product order.

    For example, build a product of two grlex orders:

    >>> from sympy.polys.orderings import build_product_order
    >>> from sympy.abc import x, y, z, t

    >>> O = build_product_order((("grlex", x, y), ("grlex", z, t)), [x, y, z, t])
    >>> O((1, 2, 3, 4))
    ((3, (1, 2)), (7, (3, 4)))

    """
    gens2idx = {}
    for i, g in enumerate(gens):
        gens2idx[g] = i
    order = []
    for expr in arg:
        name = expr[0]
        var = expr[1:]

        def makelambda(var):
            return _ItemGetter(gens2idx[g] for g in var)
        order.append((monomial_key(name), makelambda(var)))
    return ProductOrder(*order)
