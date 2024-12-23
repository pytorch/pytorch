import math
from sympy.sets.sets import Interval
from sympy.calculus.singularities import is_increasing, is_decreasing
from sympy.codegen.rewriting import Optimization
from sympy.core.function import UndefinedFunction

"""
This module collects classes useful for approimate rewriting of expressions.
This can be beneficial when generating numeric code for which performance is
of greater importance than precision (e.g. for preconditioners used in iterative
methods).
"""

class SumApprox(Optimization):
    """
    Approximates sum by neglecting small terms.

    Explanation
    ===========

    If terms are expressions which can be determined to be monotonic, then
    bounds for those expressions are added.

    Parameters
    ==========

    bounds : dict
        Mapping expressions to length 2 tuple of bounds (low, high).
    reltol : number
        Threshold for when to ignore a term. Taken relative to the largest
        lower bound among bounds.

    Examples
    ========

    >>> from sympy import exp
    >>> from sympy.abc import x, y, z
    >>> from sympy.codegen.rewriting import optimize
    >>> from sympy.codegen.approximations import SumApprox
    >>> bounds = {x: (-1, 1), y: (1000, 2000), z: (-10, 3)}
    >>> sum_approx3 = SumApprox(bounds, reltol=1e-3)
    >>> sum_approx2 = SumApprox(bounds, reltol=1e-2)
    >>> sum_approx1 = SumApprox(bounds, reltol=1e-1)
    >>> expr = 3*(x + y + exp(z))
    >>> optimize(expr, [sum_approx3])
    3*(x + y + exp(z))
    >>> optimize(expr, [sum_approx2])
    3*y + 3*exp(z)
    >>> optimize(expr, [sum_approx1])
    3*y

    """

    def __init__(self, bounds, reltol, **kwargs):
        super().__init__(**kwargs)
        self.bounds = bounds
        self.reltol = reltol

    def __call__(self, expr):
        return expr.factor().replace(self.query, lambda arg: self.value(arg))

    def query(self, expr):
        return expr.is_Add

    def value(self, add):
        for term in add.args:
            if term.is_number or term in self.bounds or len(term.free_symbols) != 1:
                continue
            fs, = term.free_symbols
            if fs not in self.bounds:
                continue
            intrvl = Interval(*self.bounds[fs])
            if is_increasing(term, intrvl, fs):
                self.bounds[term] = (
                    term.subs({fs: self.bounds[fs][0]}),
                    term.subs({fs: self.bounds[fs][1]})
                )
            elif is_decreasing(term, intrvl, fs):
                self.bounds[term] = (
                    term.subs({fs: self.bounds[fs][1]}),
                    term.subs({fs: self.bounds[fs][0]})
                )
            else:
                return add

        if all(term.is_number or term in self.bounds for term in add.args):
            bounds = [(term, term) if term.is_number else self.bounds[term] for term in add.args]
            largest_abs_guarantee = 0
            for lo, hi in bounds:
                if lo <= 0 <= hi:
                    continue
                largest_abs_guarantee = max(largest_abs_guarantee,
                                            min(abs(lo), abs(hi)))
            new_terms = []
            for term, (lo, hi) in zip(add.args, bounds):
                if max(abs(lo), abs(hi)) >= largest_abs_guarantee*self.reltol:
                    new_terms.append(term)
            return add.func(*new_terms)
        else:
            return add


class SeriesApprox(Optimization):
    """ Approximates functions by expanding them as a series.

    Parameters
    ==========

    bounds : dict
        Mapping expressions to length 2 tuple of bounds (low, high).
    reltol : number
        Threshold for when to ignore a term. Taken relative to the largest
        lower bound among bounds.
    max_order : int
        Largest order to include in series expansion
    n_point_checks : int (even)
        The validity of an expansion (with respect to reltol) is checked at
        discrete points (linearly spaced over the bounds of the variable). The
        number of points used in this numerical check is given by this number.

    Examples
    ========

    >>> from sympy import sin, pi
    >>> from sympy.abc import x, y
    >>> from sympy.codegen.rewriting import optimize
    >>> from sympy.codegen.approximations import SeriesApprox
    >>> bounds = {x: (-.1, .1), y: (pi-1, pi+1)}
    >>> series_approx2 = SeriesApprox(bounds, reltol=1e-2)
    >>> series_approx3 = SeriesApprox(bounds, reltol=1e-3)
    >>> series_approx8 = SeriesApprox(bounds, reltol=1e-8)
    >>> expr = sin(x)*sin(y)
    >>> optimize(expr, [series_approx2])
    x*(-y + (y - pi)**3/6 + pi)
    >>> optimize(expr, [series_approx3])
    (-x**3/6 + x)*sin(y)
    >>> optimize(expr, [series_approx8])
    sin(x)*sin(y)

    """
    def __init__(self, bounds, reltol, max_order=4, n_point_checks=4, **kwargs):
        super().__init__(**kwargs)
        self.bounds = bounds
        self.reltol = reltol
        self.max_order = max_order
        if n_point_checks % 2 == 1:
            raise ValueError("Checking the solution at expansion point is not helpful")
        self.n_point_checks = n_point_checks
        self._prec = math.ceil(-math.log10(self.reltol))

    def __call__(self, expr):
        return expr.factor().replace(self.query, lambda arg: self.value(arg))

    def query(self, expr):
        return (expr.is_Function and not isinstance(expr, UndefinedFunction)
                and len(expr.args) == 1)

    def value(self, fexpr):
        free_symbols = fexpr.free_symbols
        if len(free_symbols) != 1:
            return fexpr
        symb, = free_symbols
        if symb not in self.bounds:
            return fexpr
        lo, hi = self.bounds[symb]
        x0 = (lo + hi)/2
        cheapest = None
        for n in range(self.max_order+1, 0, -1):
            fseri = fexpr.series(symb, x0=x0, n=n).removeO()
            n_ok = True
            for idx in range(self.n_point_checks):
                x = lo + idx*(hi - lo)/(self.n_point_checks - 1)
                val = fseri.xreplace({symb: x})
                ref = fexpr.xreplace({symb: x})
                if abs((1 - val/ref).evalf(self._prec)) > self.reltol:
                    n_ok = False
                    break

            if n_ok:
                cheapest = fseri
            else:
                break

        if cheapest is None:
            return fexpr
        else:
            return cheapest
