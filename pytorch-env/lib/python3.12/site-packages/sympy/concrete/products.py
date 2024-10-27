from typing import Tuple as tTuple

from .expr_with_intlimits import ExprWithIntLimits
from .summations import Sum, summation, _dummy_with_inherited_properties_concrete
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.function import Derivative
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.combinatorial.factorials import RisingFactorial
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.polys import quo, roots


class Product(ExprWithIntLimits):
    r"""
    Represents unevaluated products.

    Explanation
    ===========

    ``Product`` represents a finite or infinite product, with the first
    argument being the general form of terms in the series, and the second
    argument being ``(dummy_variable, start, end)``, with ``dummy_variable``
    taking all integer values from ``start`` through ``end``. In accordance
    with long-standing mathematical convention, the end term is included in
    the product.

    Finite products
    ===============

    For finite products (and products with symbolic limits assumed to be finite)
    we follow the analogue of the summation convention described by Karr [1],
    especially definition 3 of section 1.4. The product:

    .. math::

        \prod_{m \leq i < n} f(i)

    has *the obvious meaning* for `m < n`, namely:

    .. math::

        \prod_{m \leq i < n} f(i) = f(m) f(m+1) \cdot \ldots \cdot f(n-2) f(n-1)

    with the upper limit value `f(n)` excluded. The product over an empty set is
    one if and only if `m = n`:

    .. math::

        \prod_{m \leq i < n} f(i) = 1  \quad \mathrm{for} \quad  m = n

    Finally, for all other products over empty sets we assume the following
    definition:

    .. math::

        \prod_{m \leq i < n} f(i) = \frac{1}{\prod_{n \leq i < m} f(i)}  \quad \mathrm{for} \quad  m > n

    It is important to note that above we define all products with the upper
    limit being exclusive. This is in contrast to the usual mathematical notation,
    but does not affect the product convention. Indeed we have:

    .. math::

        \prod_{m \leq i < n} f(i) = \prod_{i = m}^{n - 1} f(i)

    where the difference in notation is intentional to emphasize the meaning,
    with limits typeset on the top being inclusive.

    Examples
    ========

    >>> from sympy.abc import a, b, i, k, m, n, x
    >>> from sympy import Product, oo
    >>> Product(k, (k, 1, m))
    Product(k, (k, 1, m))
    >>> Product(k, (k, 1, m)).doit()
    factorial(m)
    >>> Product(k**2,(k, 1, m))
    Product(k**2, (k, 1, m))
    >>> Product(k**2,(k, 1, m)).doit()
    factorial(m)**2

    Wallis' product for pi:

    >>> W = Product(2*i/(2*i-1) * 2*i/(2*i+1), (i, 1, oo))
    >>> W
    Product(4*i**2/((2*i - 1)*(2*i + 1)), (i, 1, oo))

    Direct computation currently fails:

    >>> W.doit()
    Product(4*i**2/((2*i - 1)*(2*i + 1)), (i, 1, oo))

    But we can approach the infinite product by a limit of finite products:

    >>> from sympy import limit
    >>> W2 = Product(2*i/(2*i-1)*2*i/(2*i+1), (i, 1, n))
    >>> W2
    Product(4*i**2/((2*i - 1)*(2*i + 1)), (i, 1, n))
    >>> W2e = W2.doit()
    >>> W2e
    4**n*factorial(n)**2/(2**(2*n)*RisingFactorial(1/2, n)*RisingFactorial(3/2, n))
    >>> limit(W2e, n, oo)
    pi/2

    By the same formula we can compute sin(pi/2):

    >>> from sympy import combsimp, pi, gamma, simplify
    >>> P = pi * x * Product(1 - x**2/k**2, (k, 1, n))
    >>> P = P.subs(x, pi/2)
    >>> P
    pi**2*Product(1 - pi**2/(4*k**2), (k, 1, n))/2
    >>> Pe = P.doit()
    >>> Pe
    pi**2*RisingFactorial(1 - pi/2, n)*RisingFactorial(1 + pi/2, n)/(2*factorial(n)**2)
    >>> limit(Pe, n, oo).gammasimp()
    sin(pi**2/2)
    >>> Pe.rewrite(gamma)
    (-1)**n*pi**2*gamma(pi/2)*gamma(n + 1 + pi/2)/(2*gamma(1 + pi/2)*gamma(-n + pi/2)*gamma(n + 1)**2)

    Products with the lower limit being larger than the upper one:

    >>> Product(1/i, (i, 6, 1)).doit()
    120
    >>> Product(i, (i, 2, 5)).doit()
    120

    The empty product:

    >>> Product(i, (i, n, n-1)).doit()
    1

    An example showing that the symbolic result of a product is still
    valid for seemingly nonsensical values of the limits. Then the Karr
    convention allows us to give a perfectly valid interpretation to
    those products by interchanging the limits according to the above rules:

    >>> P = Product(2, (i, 10, n)).doit()
    >>> P
    2**(n - 9)
    >>> P.subs(n, 5)
    1/16
    >>> Product(2, (i, 10, 5)).doit()
    1/16
    >>> 1/Product(2, (i, 6, 9)).doit()
    1/16

    An explicit example of the Karr summation convention applied to products:

    >>> P1 = Product(x, (i, a, b)).doit()
    >>> P1
    x**(-a + b + 1)
    >>> P2 = Product(x, (i, b+1, a-1)).doit()
    >>> P2
    x**(a - b - 1)
    >>> simplify(P1 * P2)
    1

    And another one:

    >>> P1 = Product(i, (i, b, a)).doit()
    >>> P1
    RisingFactorial(b, a - b + 1)
    >>> P2 = Product(i, (i, a+1, b-1)).doit()
    >>> P2
    RisingFactorial(a + 1, -a + b - 1)
    >>> P1 * P2
    RisingFactorial(b, a - b + 1)*RisingFactorial(a + 1, -a + b - 1)
    >>> combsimp(P1 * P2)
    1

    See Also
    ========

    Sum, summation
    product

    References
    ==========

    .. [1] Michael Karr, "Summation in Finite Terms", Journal of the ACM,
           Volume 28 Issue 2, April 1981, Pages 305-350
           https://dl.acm.org/doi/10.1145/322248.322255
    .. [2] https://en.wikipedia.org/wiki/Multiplication#Capital_Pi_notation
    .. [3] https://en.wikipedia.org/wiki/Empty_product
    """

    __slots__ = ()

    limits: tTuple[tTuple[Symbol, Expr, Expr]]

    def __new__(cls, function, *symbols, **assumptions):
        obj = ExprWithIntLimits.__new__(cls, function, *symbols, **assumptions)
        return obj

    def _eval_rewrite_as_Sum(self, *args, **kwargs):
        return exp(Sum(log(self.function), *self.limits))

    @property
    def term(self):
        return self._args[0]
    function = term

    def _eval_is_zero(self):
        if self.has_empty_sequence:
            return False

        z = self.term.is_zero
        if z is True:
            return True
        if self.has_finite_limits:
            # A Product is zero only if its term is zero assuming finite limits.
            return z

    def _eval_is_extended_real(self):
        if self.has_empty_sequence:
            return True

        return self.function.is_extended_real

    def _eval_is_positive(self):
        if self.has_empty_sequence:
            return True
        if self.function.is_positive and self.has_finite_limits:
            return True

    def _eval_is_nonnegative(self):
        if self.has_empty_sequence:
            return True
        if self.function.is_nonnegative and self.has_finite_limits:
            return True

    def _eval_is_extended_nonnegative(self):
        if self.has_empty_sequence:
            return True
        if self.function.is_extended_nonnegative:
            return True

    def _eval_is_extended_nonpositive(self):
        if self.has_empty_sequence:
            return True

    def _eval_is_finite(self):
        if self.has_finite_limits and self.function.is_finite:
            return True

    def doit(self, **hints):
        # first make sure any definite limits have product
        # variables with matching assumptions
        reps = {}
        for xab in self.limits:
            d = _dummy_with_inherited_properties_concrete(xab)
            if d:
                reps[xab[0]] = d
        if reps:
            undo = {v: k for k, v in reps.items()}
            did = self.xreplace(reps).doit(**hints)
            if isinstance(did, tuple):  # when separate=True
                did = tuple([i.xreplace(undo) for i in did])
            else:
                did = did.xreplace(undo)
            return did

        from sympy.simplify.powsimp import powsimp
        f = self.function
        for index, limit in enumerate(self.limits):
            i, a, b = limit
            dif = b - a
            if dif.is_integer and dif.is_negative:
                a, b = b + 1, a - 1
                f = 1 / f

            g = self._eval_product(f, (i, a, b))
            if g in (None, S.NaN):
                return self.func(powsimp(f), *self.limits[index:])
            else:
                f = g

        if hints.get('deep', True):
            return f.doit(**hints)
        else:
            return powsimp(f)

    def _eval_adjoint(self):
        if self.is_commutative:
            return self.func(self.function.adjoint(), *self.limits)
        return None

    def _eval_conjugate(self):
        return self.func(self.function.conjugate(), *self.limits)

    def _eval_product(self, term, limits):

        (k, a, n) = limits

        if k not in term.free_symbols:
            if (term - 1).is_zero:
                return S.One
            return term**(n - a + 1)

        if a == n:
            return term.subs(k, a)

        from .delta import deltaproduct, _has_simple_delta
        if term.has(KroneckerDelta) and _has_simple_delta(term, limits[0]):
            return deltaproduct(term, limits)

        dif = n - a
        definite = dif.is_Integer
        if definite and (dif < 100):
            return self._eval_product_direct(term, limits)

        elif term.is_polynomial(k):
            poly = term.as_poly(k)

            A = B = Q = S.One

            all_roots = roots(poly)

            M = 0
            for r, m in all_roots.items():
                M += m
                A *= RisingFactorial(a - r, n - a + 1)**m
                Q *= (n - r)**m

            if M < poly.degree():
                arg = quo(poly, Q.as_poly(k))
                B = self.func(arg, (k, a, n)).doit()

            return poly.LC()**(n - a + 1) * A * B

        elif term.is_Add:
            factored = factor_terms(term, fraction=True)
            if factored.is_Mul:
                return self._eval_product(factored, (k, a, n))

        elif term.is_Mul:
            # Factor in part without the summation variable and part with
            without_k, with_k = term.as_coeff_mul(k)

            if len(with_k) >= 2:
                # More than one term including k, so still a multiplication
                exclude, include = [], []
                for t in with_k:
                    p = self._eval_product(t, (k, a, n))

                    if p is not None:
                        exclude.append(p)
                    else:
                        include.append(t)

                if not exclude:
                    return None
                else:
                    arg = term._new_rawargs(*include)
                    A = Mul(*exclude)
                    B = self.func(arg, (k, a, n)).doit()
                    return without_k**(n - a + 1)*A * B
            else:
                # Just a single term
                p = self._eval_product(with_k[0], (k, a, n))
                if p is None:
                    p = self.func(with_k[0], (k, a, n)).doit()
                return without_k**(n - a + 1)*p


        elif term.is_Pow:
            if not term.base.has(k):
                s = summation(term.exp, (k, a, n))

                return term.base**s
            elif not term.exp.has(k):
                p = self._eval_product(term.base, (k, a, n))

                if p is not None:
                    return p**term.exp

        elif isinstance(term, Product):
            evaluated = term.doit()
            f = self._eval_product(evaluated, limits)
            if f is None:
                return self.func(evaluated, limits)
            else:
                return f

        if definite:
            return self._eval_product_direct(term, limits)

    def _eval_simplify(self, **kwargs):
        from sympy.simplify.simplify import product_simplify
        rv = product_simplify(self, **kwargs)
        return rv.doit() if kwargs['doit'] else rv

    def _eval_transpose(self):
        if self.is_commutative:
            return self.func(self.function.transpose(), *self.limits)
        return None

    def _eval_product_direct(self, term, limits):
        (k, a, n) = limits
        return Mul(*[term.subs(k, a + i) for i in range(n - a + 1)])

    def _eval_derivative(self, x):
        if isinstance(x, Symbol) and x not in self.free_symbols:
            return S.Zero
        f, limits = self.function, list(self.limits)
        limit = limits.pop(-1)
        if limits:
            f = self.func(f, *limits)
        i, a, b = limit
        if x in a.free_symbols or x in b.free_symbols:
            return None
        h = Dummy()
        rv = Sum( Product(f, (i, a, h - 1)) * Product(f, (i, h + 1, b)) * Derivative(f, x, evaluate=True).subs(i, h), (h, a, b))
        return rv

    def is_convergent(self):
        r"""
        See docs of :obj:`.Sum.is_convergent()` for explanation of convergence
        in SymPy.

        Explanation
        ===========

        The infinite product:

        .. math::

            \prod_{1 \leq i < \infty} f(i)

        is defined by the sequence of partial products:

        .. math::

            \prod_{i=1}^{n} f(i) = f(1) f(2) \cdots f(n)

        as n increases without bound. The product converges to a non-zero
        value if and only if the sum:

        .. math::

            \sum_{1 \leq i < \infty} \log{f(n)}

        converges.

        Examples
        ========

        >>> from sympy import Product, Symbol, cos, pi, exp, oo
        >>> n = Symbol('n', integer=True)
        >>> Product(n/(n + 1), (n, 1, oo)).is_convergent()
        False
        >>> Product(1/n**2, (n, 1, oo)).is_convergent()
        False
        >>> Product(cos(pi/n), (n, 1, oo)).is_convergent()
        True
        >>> Product(exp(-n**2), (n, 1, oo)).is_convergent()
        False

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Infinite_product
        """
        sequence_term = self.function
        log_sum = log(sequence_term)
        lim = self.limits
        try:
            is_conv = Sum(log_sum, *lim).is_convergent()
        except NotImplementedError:
            if Sum(sequence_term - 1, *lim).is_absolutely_convergent() is S.true:
                return S.true
            raise NotImplementedError("The algorithm to find the product convergence of %s "
                                        "is not yet implemented" % (sequence_term))
        return is_conv

    def reverse_order(expr, *indices):
        """
        Reverse the order of a limit in a Product.

        Explanation
        ===========

        ``reverse_order(expr, *indices)`` reverses some limits in the expression
        ``expr`` which can be either a ``Sum`` or a ``Product``. The selectors in
        the argument ``indices`` specify some indices whose limits get reversed.
        These selectors are either variable names or numerical indices counted
        starting from the inner-most limit tuple.

        Examples
        ========

        >>> from sympy import gamma, Product, simplify, Sum
        >>> from sympy.abc import x, y, a, b, c, d
        >>> P = Product(x, (x, a, b))
        >>> Pr = P.reverse_order(x)
        >>> Pr
        Product(1/x, (x, b + 1, a - 1))
        >>> Pr = Pr.doit()
        >>> Pr
        1/RisingFactorial(b + 1, a - b - 1)
        >>> simplify(Pr.rewrite(gamma))
        Piecewise((gamma(b + 1)/gamma(a), b > -1), ((-1)**(-a + b + 1)*gamma(1 - a)/gamma(-b), True))
        >>> P = P.doit()
        >>> P
        RisingFactorial(a, -a + b + 1)
        >>> simplify(P.rewrite(gamma))
        Piecewise((gamma(b + 1)/gamma(a), a > 0), ((-1)**(-a + b + 1)*gamma(1 - a)/gamma(-b), True))

        While one should prefer variable names when specifying which limits
        to reverse, the index counting notation comes in handy in case there
        are several symbols with the same name.

        >>> S = Sum(x*y, (x, a, b), (y, c, d))
        >>> S
        Sum(x*y, (x, a, b), (y, c, d))
        >>> S0 = S.reverse_order(0)
        >>> S0
        Sum(-x*y, (x, b + 1, a - 1), (y, c, d))
        >>> S1 = S0.reverse_order(1)
        >>> S1
        Sum(x*y, (x, b + 1, a - 1), (y, d + 1, c - 1))

        Of course we can mix both notations:

        >>> Sum(x*y, (x, a, b), (y, 2, 5)).reverse_order(x, 1)
        Sum(x*y, (x, b + 1, a - 1), (y, 6, 1))
        >>> Sum(x*y, (x, a, b), (y, 2, 5)).reverse_order(y, x)
        Sum(x*y, (x, b + 1, a - 1), (y, 6, 1))

        See Also
        ========

        sympy.concrete.expr_with_intlimits.ExprWithIntLimits.index,
        reorder_limit,
        sympy.concrete.expr_with_intlimits.ExprWithIntLimits.reorder

        References
        ==========

        .. [1] Michael Karr, "Summation in Finite Terms", Journal of the ACM,
               Volume 28 Issue 2, April 1981, Pages 305-350
               https://dl.acm.org/doi/10.1145/322248.322255

        """
        l_indices = list(indices)

        for i, indx in enumerate(l_indices):
            if not isinstance(indx, int):
                l_indices[i] = expr.index(indx)

        e = 1
        limits = []
        for i, limit in enumerate(expr.limits):
            l = limit
            if i in l_indices:
                e = -e
                l = (limit[0], limit[2] + 1, limit[1] - 1)
            limits.append(l)

        return Product(expr.function ** e, *limits)


def product(*args, **kwargs):
    r"""
    Compute the product.

    Explanation
    ===========

    The notation for symbols is similar to the notation used in Sum or
    Integral. product(f, (i, a, b)) computes the product of f with
    respect to i from a to b, i.e.,

    ::

                                     b
                                   _____
        product(f(n), (i, a, b)) = |   | f(n)
                                   |   |
                                   i = a

    If it cannot compute the product, it returns an unevaluated Product object.
    Repeated products can be computed by introducing additional symbols tuples::

    Examples
    ========

    >>> from sympy import product, symbols
    >>> i, n, m, k = symbols('i n m k', integer=True)

    >>> product(i, (i, 1, k))
    factorial(k)
    >>> product(m, (i, 1, k))
    m**k
    >>> product(i, (i, 1, k), (k, 1, n))
    Product(factorial(k), (k, 1, n))

    """

    prod = Product(*args, **kwargs)

    if isinstance(prod, Product):
        return prod.doit(deep=False)
    else:
        return prod
