#
# This is the module for ODE solver classes for single ODEs.
#

from __future__ import annotations
from typing import ClassVar, Iterator

from .riccati import match_riccati, solve_riccati
from sympy.core import Add, S, Pow, Rational
from sympy.core.cache import cached_property
from sympy.core.exprtools import factor_terms
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef, Derivative, diff, Function, expand, Subs, _mexpand
from sympy.core.numbers import zoo
from sympy.core.relational import Equality, Eq
from sympy.core.symbol import Symbol, Dummy, Wild
from sympy.core.mul import Mul
from sympy.functions import exp, tan, log, sqrt, besselj, bessely, cbrt, airyai, airybi
from sympy.integrals import Integral
from sympy.polys import Poly
from sympy.polys.polytools import cancel, factor, degree
from sympy.simplify import collect, simplify, separatevars, logcombine, posify # type: ignore
from sympy.simplify.radsimp import fraction
from sympy.utilities import numbered_symbols
from sympy.solvers.solvers import solve
from sympy.solvers.deutils import ode_order, _preprocess
from sympy.polys.matrices.linsolve import _lin_eq2dict
from sympy.polys.solvers import PolyNonlinearError
from .hypergeometric import equivalence_hypergeometric, match_2nd_2F1_hypergeometric, \
    get_sol_2F1_hypergeometric, match_2nd_hypergeometric
from .nonhomogeneous import _get_euler_characteristic_eq_sols, _get_const_characteristic_eq_sols, \
    _solve_undetermined_coefficients, _solve_variation_of_parameters, _test_term, _undetermined_coefficients_match, \
        _get_simplified_sol
from .lie_group import _ode_lie_group


class ODEMatchError(NotImplementedError):
    """Raised if a SingleODESolver is asked to solve an ODE it does not match"""
    pass


class SingleODEProblem:
    """Represents an ordinary differential equation (ODE)

    This class is used internally in the by dsolve and related
    functions/classes so that properties of an ODE can be computed
    efficiently.

    Examples
    ========

    This class is used internally by dsolve. To instantiate an instance
    directly first define an ODE problem:

    >>> from sympy import Function, Symbol
    >>> x = Symbol('x')
    >>> f = Function('f')
    >>> eq = f(x).diff(x, 2)

    Now you can create a SingleODEProblem instance and query its properties:

    >>> from sympy.solvers.ode.single import SingleODEProblem
    >>> problem = SingleODEProblem(f(x).diff(x), f(x), x)
    >>> problem.eq
    Derivative(f(x), x)
    >>> problem.func
    f(x)
    >>> problem.sym
    x
    """

    # Instance attributes:
    eq: Expr
    func: AppliedUndef
    sym: Symbol
    _order: int
    _eq_expanded: Expr
    _eq_preprocessed: Expr
    _eq_high_order_free = None

    def __init__(self, eq, func, sym, prep=True, **kwargs):
        assert isinstance(eq, Expr)
        assert isinstance(func, AppliedUndef)
        assert isinstance(sym, Symbol)
        assert isinstance(prep, bool)
        self.eq = eq
        self.func = func
        self.sym = sym
        self.prep = prep
        self.params = kwargs

    @cached_property
    def order(self) -> int:
        return ode_order(self.eq, self.func)

    @cached_property
    def eq_preprocessed(self) -> Expr:
        return self._get_eq_preprocessed()

    @cached_property
    def eq_high_order_free(self) -> Expr:
        a = Wild('a', exclude=[self.func])
        c1 = Wild('c1', exclude=[self.sym])
        # Precondition to try remove f(x) from highest order derivative
        reduced_eq = None
        if self.eq.is_Add:
            deriv_coef = self.eq.coeff(self.func.diff(self.sym, self.order))
            if deriv_coef not in (1, 0):
                r = deriv_coef.match(a*self.func**c1)
                if r and r[c1]:
                    den = self.func**r[c1]
                    reduced_eq = Add(*[arg/den for arg in self.eq.args])
        if reduced_eq is None:
            reduced_eq = expand(self.eq)
        return reduced_eq

    @cached_property
    def eq_expanded(self) -> Expr:
        return expand(self.eq_preprocessed)

    def _get_eq_preprocessed(self) -> Expr:
        if self.prep:
            process_eq, process_func = _preprocess(self.eq, self.func)
            if process_func != self.func:
                raise ValueError
        else:
            process_eq = self.eq
        return process_eq

    def get_numbered_constants(self, num=1, start=1, prefix='C') -> list[Symbol]:
        """
        Returns a list of constants that do not occur
        in eq already.
        """
        ncs = self.iter_numbered_constants(start, prefix)
        Cs = [next(ncs) for i in range(num)]
        return Cs

    def iter_numbered_constants(self, start=1, prefix='C') -> Iterator[Symbol]:
        """
        Returns an iterator of constants that do not occur
        in eq already.
        """
        atom_set = self.eq.free_symbols
        func_set = self.eq.atoms(Function)
        if func_set:
            atom_set |= {Symbol(str(f.func)) for f in func_set}
        return numbered_symbols(start=start, prefix=prefix, exclude=atom_set)

    @cached_property
    def is_autonomous(self):
        u = Dummy('u')
        x = self.sym
        syms = self.eq.subs(self.func, u).free_symbols
        return x not in syms

    def get_linear_coefficients(self, eq, func, order):
        r"""
        Matches a differential equation to the linear form:

        .. math:: a_n(x) y^{(n)} + \cdots + a_1(x)y' + a_0(x) y + B(x) = 0

        Returns a dict of order:coeff terms, where order is the order of the
        derivative on each term, and coeff is the coefficient of that derivative.
        The key ``-1`` holds the function `B(x)`. Returns ``None`` if the ODE is
        not linear.  This function assumes that ``func`` has already been checked
        to be good.

        Examples
        ========

        >>> from sympy import Function, cos, sin
        >>> from sympy.abc import x
        >>> from sympy.solvers.ode.single import SingleODEProblem
        >>> f = Function('f')
        >>> eq = f(x).diff(x, 3) + 2*f(x).diff(x) + \
        ... x*f(x).diff(x, 2) + cos(x)*f(x).diff(x) + x - f(x) - \
        ... sin(x)
        >>> obj = SingleODEProblem(eq, f(x), x)
        >>> obj.get_linear_coefficients(eq, f(x), 3)
        {-1: x - sin(x), 0: -1, 1: cos(x) + 2, 2: x, 3: 1}
        >>> eq = f(x).diff(x, 3) + 2*f(x).diff(x) + \
        ... x*f(x).diff(x, 2) + cos(x)*f(x).diff(x) + x - f(x) - \
        ... sin(f(x))
        >>> obj = SingleODEProblem(eq, f(x), x)
        >>> obj.get_linear_coefficients(eq, f(x), 3) == None
        True

        """
        f = func.func
        x = func.args[0]
        symset = {Derivative(f(x), x, i) for i in range(order+1)}
        try:
            rhs, lhs_terms = _lin_eq2dict(eq, symset)
        except PolyNonlinearError:
            return None

        if rhs.has(func) or any(c.has(func) for c in lhs_terms.values()):
            return None
        terms = {i: lhs_terms.get(f(x).diff(x, i), S.Zero) for i in range(order+1)}
        terms[-1] = rhs
        return terms

    # TODO: Add methods that can be used by many ODE solvers:
    # order
    # is_linear()
    # get_linear_coefficients()
    # eq_prepared (the ODE in prepared form)


class SingleODESolver:
    """
    Base class for Single ODE solvers.

    Subclasses should implement the _matches and _get_general_solution
    methods. This class is not intended to be instantiated directly but its
    subclasses are as part of dsolve.

    Examples
    ========

    You can use a subclass of SingleODEProblem to solve a particular type of
    ODE. We first define a particular ODE problem:

    >>> from sympy import Function, Symbol
    >>> x = Symbol('x')
    >>> f = Function('f')
    >>> eq = f(x).diff(x, 2)

    Now we solve this problem using the NthAlgebraic solver which is a
    subclass of SingleODESolver:

    >>> from sympy.solvers.ode.single import NthAlgebraic, SingleODEProblem
    >>> problem = SingleODEProblem(eq, f(x), x)
    >>> solver = NthAlgebraic(problem)
    >>> solver.get_general_solution()
    [Eq(f(x), _C*x + _C)]

    The normal way to solve an ODE is to use dsolve (which would use
    NthAlgebraic and other solvers internally). When using dsolve a number of
    other things are done such as evaluating integrals, simplifying the
    solution and renumbering the constants:

    >>> from sympy import dsolve
    >>> dsolve(eq, hint='nth_algebraic')
    Eq(f(x), C1 + C2*x)
    """

    # Subclasses should store the hint name (the argument to dsolve) in this
    # attribute
    hint: ClassVar[str]

    # Subclasses should define this to indicate if they support an _Integral
    # hint.
    has_integral: ClassVar[bool]

    # The ODE to be solved
    ode_problem: SingleODEProblem

    # Cache whether or not the equation has matched the method
    _matched: bool | None = None

    # Subclasses should store in this attribute the list of order(s) of ODE
    # that subclass can solve or leave it to None if not specific to any order
    order: list | None = None

    def __init__(self, ode_problem):
        self.ode_problem = ode_problem

    def matches(self) -> bool:
        if self.order is not None and self.ode_problem.order not in self.order:
            self._matched = False
            return self._matched

        if self._matched is None:
            self._matched = self._matches()
        return self._matched

    def get_general_solution(self, *, simplify: bool = True) -> list[Equality]:
        if not self.matches():
            msg = "%s solver cannot solve:\n%s"
            raise ODEMatchError(msg % (self.hint, self.ode_problem.eq))
        return self._get_general_solution(simplify_flag=simplify)

    def _matches(self) -> bool:
        msg = "Subclasses of SingleODESolver should implement matches."
        raise NotImplementedError(msg)

    def _get_general_solution(self, *, simplify_flag: bool = True) -> list[Equality]:
        msg = "Subclasses of SingleODESolver should implement get_general_solution."
        raise NotImplementedError(msg)


class SinglePatternODESolver(SingleODESolver):
    '''Superclass for ODE solvers based on pattern matching'''

    def wilds(self):
        prob = self.ode_problem
        f = prob.func.func
        x = prob.sym
        order = prob.order
        return self._wilds(f, x, order)

    def wilds_match(self):
        match = self._wilds_match
        return [match.get(w, S.Zero) for w in self.wilds()]

    def _matches(self):
        eq = self.ode_problem.eq_expanded
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        order = self.ode_problem.order
        df = f(x).diff(x, order)

        if order not in [1, 2]:
            return False

        pattern = self._equation(f(x), x, order)

        if not pattern.coeff(df).has(Wild):
            eq = expand(eq / eq.coeff(df))
        eq = eq.collect([f(x).diff(x), f(x)], func = cancel)

        self._wilds_match = match = eq.match(pattern)
        if match is not None:
            return self._verify(f(x))
        return False

    def _verify(self, fx) -> bool:
        return True

    def _wilds(self, f, x, order):
        msg = "Subclasses of SingleODESolver should implement _wilds"
        raise NotImplementedError(msg)

    def _equation(self, fx, x, order):
        msg = "Subclasses of SingleODESolver should implement _equation"
        raise NotImplementedError(msg)


class NthAlgebraic(SingleODESolver):
    r"""
    Solves an `n`\th order ordinary differential equation using algebra and
    integrals.

    There is no general form for the kind of equation that this can solve. The
    the equation is solved algebraically treating differentiation as an
    invertible algebraic function.

    Examples
    ========

    >>> from sympy import Function, dsolve, Eq
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> eq = Eq(f(x) * (f(x).diff(x)**2 - 1), 0)
    >>> dsolve(eq, f(x), hint='nth_algebraic')
    [Eq(f(x), 0), Eq(f(x), C1 - x), Eq(f(x), C1 + x)]

    Note that this solver can return algebraic solutions that do not have any
    integration constants (f(x) = 0 in the above example).
    """

    hint = 'nth_algebraic'
    has_integral = True  # nth_algebraic_Integral hint

    def _matches(self):
        r"""
        Matches any differential equation that nth_algebraic can solve. Uses
        `sympy.solve` but teaches it how to integrate derivatives.

        This involves calling `sympy.solve` and does most of the work of finding a
        solution (apart from evaluating the integrals).
        """
        eq = self.ode_problem.eq
        func = self.ode_problem.func
        var = self.ode_problem.sym

        # Derivative that solve can handle:
        diffx = self._get_diffx(var)

        # Replace derivatives wrt the independent variable with diffx
        def replace(eq, var):
            def expand_diffx(*args):
                differand, diffs = args[0], args[1:]
                toreplace = differand
                for v, n in diffs:
                    for _ in range(n):
                        if v == var:
                            toreplace = diffx(toreplace)
                        else:
                            toreplace = Derivative(toreplace, v)
                return toreplace
            return eq.replace(Derivative, expand_diffx)

        # Restore derivatives in solution afterwards
        def unreplace(eq, var):
            return eq.replace(diffx, lambda e: Derivative(e, var))

        subs_eqn = replace(eq, var)
        try:
            # turn off simplification to protect Integrals that have
            # _t instead of fx in them and would otherwise factor
            # as t_*Integral(1, x)
            solns = solve(subs_eqn, func, simplify=False)
        except NotImplementedError:
            solns = []

        solns = [simplify(unreplace(soln, var)) for soln in solns]
        solns = [Equality(func, soln) for soln in solns]

        self.solutions = solns
        return len(solns) != 0

    def _get_general_solution(self, *, simplify_flag: bool = True):
        return self.solutions

    # This needs to produce an invertible function but the inverse depends
    # which variable we are integrating with respect to. Since the class can
    # be stored in cached results we need to ensure that we always get the
    # same class back for each particular integration variable so we store these
    # classes in a global dict:
    _diffx_stored: dict[Symbol, type[Function]] = {}

    @staticmethod
    def _get_diffx(var):
        diffcls = NthAlgebraic._diffx_stored.get(var, None)

        if diffcls is None:
            # A class that behaves like Derivative wrt var but is "invertible".
            class diffx(Function):
                def inverse(self):
                    # don't use integrate here because fx has been replaced by _t
                    # in the equation; integrals will not be correct while solve
                    # is at work.
                    return lambda expr: Integral(expr, var) + Dummy('C')

            diffcls = NthAlgebraic._diffx_stored.setdefault(var, diffx)

        return diffcls


class FirstExact(SinglePatternODESolver):
    r"""
    Solves 1st order exact ordinary differential equations.

    A 1st order differential equation is called exact if it is the total
    differential of a function. That is, the differential equation

    .. math:: P(x, y) \,\partial{}x + Q(x, y) \,\partial{}y = 0

    is exact if there is some function `F(x, y)` such that `P(x, y) =
    \partial{}F/\partial{}x` and `Q(x, y) = \partial{}F/\partial{}y`.  It can
    be shown that a necessary and sufficient condition for a first order ODE
    to be exact is that `\partial{}P/\partial{}y = \partial{}Q/\partial{}x`.
    Then, the solution will be as given below::

        >>> from sympy import Function, Eq, Integral, symbols, pprint
        >>> x, y, t, x0, y0, C1= symbols('x,y,t,x0,y0,C1')
        >>> P, Q, F= map(Function, ['P', 'Q', 'F'])
        >>> pprint(Eq(Eq(F(x, y), Integral(P(t, y), (t, x0, x)) +
        ... Integral(Q(x0, t), (t, y0, y))), C1))
                    x                y
                    /                /
                   |                |
        F(x, y) =  |  P(t, y) dt +  |  Q(x0, t) dt = C1
                   |                |
                  /                /
                  x0               y0

    Where the first partials of `P` and `Q` exist and are continuous in a
    simply connected region.

    A note: SymPy currently has no way to represent inert substitution on an
    expression, so the hint ``1st_exact_Integral`` will return an integral
    with `dy`.  This is supposed to represent the function that you are
    solving for.

    Examples
    ========

    >>> from sympy import Function, dsolve, cos, sin
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> dsolve(cos(f(x)) - (x*sin(f(x)) - f(x)**2)*f(x).diff(x),
    ... f(x), hint='1st_exact')
    Eq(x*cos(f(x)) + f(x)**3/3, C1)

    References
    ==========

    - https://en.wikipedia.org/wiki/Exact_differential_equation
    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 73

    # indirect doctest

    """
    hint = "1st_exact"
    has_integral = True
    order = [1]

    def _wilds(self, f, x, order):
        P = Wild('P', exclude=[f(x).diff(x)])
        Q = Wild('Q', exclude=[f(x).diff(x)])
        return P, Q

    def _equation(self, fx, x, order):
        P, Q = self.wilds()
        return P + Q*fx.diff(x)

    def _verify(self, fx) -> bool:
        P, Q = self.wilds()
        x = self.ode_problem.sym
        y = Dummy('y')

        m, n = self.wilds_match()

        m = m.subs(fx, y)
        n = n.subs(fx, y)
        numerator = cancel(m.diff(y) - n.diff(x))

        if numerator.is_zero:
            # Is exact
            return True
        else:
            # The following few conditions try to convert a non-exact
            # differential equation into an exact one.
            # References:
            # 1. Differential equations with applications
            # and historical notes - George E. Simmons
            # 2. https://math.okstate.edu/people/binegar/2233-S99/2233-l12.pdf

            factor_n = cancel(numerator/n)
            factor_m = cancel(-numerator/m)
            if y not in factor_n.free_symbols:
                # If (dP/dy - dQ/dx) / Q = f(x)
                # then exp(integral(f(x))*equation becomes exact
                factor = factor_n
                integration_variable = x
            elif x not in factor_m.free_symbols:
                # If (dP/dy - dQ/dx) / -P = f(y)
                # then exp(integral(f(y))*equation becomes exact
                factor = factor_m
                integration_variable = y
            else:
                # Couldn't convert to exact
                return False

            factor = exp(Integral(factor, integration_variable))
            m *= factor
            n *= factor
            self._wilds_match[P] = m.subs(y, fx)
            self._wilds_match[Q] = n.subs(y, fx)
            return True

    def _get_general_solution(self, *, simplify_flag: bool = True):
        m, n = self.wilds_match()
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        (C1,) = self.ode_problem.get_numbered_constants(num=1)
        y = Dummy('y')

        m = m.subs(fx, y)
        n = n.subs(fx, y)

        gen_sol = Eq(Subs(Integral(m, x)
                          + Integral(n - Integral(m, x).diff(y), y), y, fx), C1)
        return [gen_sol]


class FirstLinear(SinglePatternODESolver):
    r"""
    Solves 1st order linear differential equations.

    These are differential equations of the form

    .. math:: dy/dx + P(x) y = Q(x)\text{.}

    These kinds of differential equations can be solved in a general way.  The
    integrating factor `e^{\int P(x) \,dx}` will turn the equation into a
    separable equation.  The general solution is::

        >>> from sympy import Function, dsolve, Eq, pprint, diff, sin
        >>> from sympy.abc import x
        >>> f, P, Q = map(Function, ['f', 'P', 'Q'])
        >>> genform = Eq(f(x).diff(x) + P(x)*f(x), Q(x))
        >>> pprint(genform)
                    d
        P(x)*f(x) + --(f(x)) = Q(x)
                    dx
        >>> pprint(dsolve(genform, f(x), hint='1st_linear_Integral'))
                /       /                   \
                |      |                    |
                |      |         /          |     /
                |      |        |           |    |
                |      |        | P(x) dx   |  - | P(x) dx
                |      |        |           |    |
                |      |       /            |   /
        f(x) = |C1 +  | Q(x)*e           dx|*e
                |      |                    |
                \     /                     /


    Examples
    ========

    >>> f = Function('f')
    >>> pprint(dsolve(Eq(x*diff(f(x), x) - f(x), x**2*sin(x)),
    ... f(x), '1st_linear'))
    f(x) = x*(C1 - cos(x))

    References
    ==========

    - https://en.wikipedia.org/wiki/Linear_differential_equation#First-order_equation_with_variable_coefficients
    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 92

    # indirect doctest

    """
    hint = '1st_linear'
    has_integral = True
    order = [1]

    def _wilds(self, f, x, order):
        P = Wild('P', exclude=[f(x)])
        Q = Wild('Q', exclude=[f(x), f(x).diff(x)])
        return P, Q

    def _equation(self, fx, x, order):
        P, Q = self.wilds()
        return fx.diff(x) + P*fx - Q

    def _get_general_solution(self, *, simplify_flag: bool = True):
        P, Q = self.wilds_match()
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        (C1,)  = self.ode_problem.get_numbered_constants(num=1)
        gensol = Eq(fx, ((C1 + Integral(Q*exp(Integral(P, x)), x))
            * exp(-Integral(P, x))))
        return [gensol]


class AlmostLinear(SinglePatternODESolver):
    r"""
    Solves an almost-linear differential equation.

    The general form of an almost linear differential equation is

    .. math:: a(x) g'(f(x)) f'(x) + b(x) g(f(x)) + c(x)

    Here `f(x)` is the function to be solved for (the dependent variable).
    The substitution `g(f(x)) = u(x)` leads to a linear differential equation
    for `u(x)` of the form `a(x) u' + b(x) u + c(x) = 0`. This can be solved
    for `u(x)` by the `first_linear` hint and then `f(x)` is found by solving
    `g(f(x)) = u(x)`.

    See Also
    ========
    :obj:`sympy.solvers.ode.single.FirstLinear`

    Examples
    ========

    >>> from sympy import dsolve, Function, pprint, sin, cos
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> d = f(x).diff(x)
    >>> eq = x*d + x*f(x) + 1
    >>> dsolve(eq, f(x), hint='almost_linear')
    Eq(f(x), (C1 - Ei(x))*exp(-x))
    >>> pprint(dsolve(eq, f(x), hint='almost_linear'))
                        -x
    f(x) = (C1 - Ei(x))*e
    >>> example = cos(f(x))*f(x).diff(x) + sin(f(x)) + 1
    >>> pprint(example)
                        d
    sin(f(x)) + cos(f(x))*--(f(x)) + 1
                        dx
    >>> pprint(dsolve(example, f(x), hint='almost_linear'))
                    /    -x    \             /    -x    \
    [f(x) = pi - asin\C1*e   - 1/, f(x) = asin\C1*e   - 1/]


    References
    ==========

    - Joel Moses, "Symbolic Integration - The Stormy Decade", Communications
      of the ACM, Volume 14, Number 8, August 1971, pp. 558
    """
    hint = "almost_linear"
    has_integral = True
    order = [1]

    def _wilds(self, f, x, order):
        P = Wild('P', exclude=[f(x).diff(x)])
        Q = Wild('Q', exclude=[f(x).diff(x)])
        return P, Q

    def _equation(self, fx, x, order):
        P, Q = self.wilds()
        return P*fx.diff(x) + Q

    def _verify(self, fx):
        a, b = self.wilds_match()
        c, b = b.as_independent(fx) if b.is_Add else (S.Zero, b)
        # a, b and c are the function a(x), b(x) and c(x) respectively.
        # c(x) is obtained by separating out b as terms with and without fx i.e, l(y)
        # The following conditions checks if the given equation is an almost-linear differential equation using the fact that
        # a(x)*(l(y))' / l(y)' is independent of l(y)

        if b.diff(fx) != 0 and not simplify(b.diff(fx)/a).has(fx):
            self.ly = factor_terms(b).as_independent(fx, as_Add=False)[1] # Gives the term containing fx i.e., l(y)
            self.ax = a / self.ly.diff(fx)
            self.cx = -c  # cx is taken as -c(x) to simplify expression in the solution integral
            self.bx = factor_terms(b) / self.ly
            return True

        return False

    def _get_general_solution(self, *, simplify_flag: bool = True):
        x = self.ode_problem.sym
        (C1,)  = self.ode_problem.get_numbered_constants(num=1)
        gensol = Eq(self.ly, ((C1 + Integral((self.cx/self.ax)*exp(Integral(self.bx/self.ax, x)), x))
                * exp(-Integral(self.bx/self.ax, x))))

        return [gensol]


class Bernoulli(SinglePatternODESolver):
    r"""
    Solves Bernoulli differential equations.

    These are equations of the form

    .. math:: dy/dx + P(x) y = Q(x) y^n\text{, }n \ne 1`\text{.}

    The substitution `w = 1/y^{1-n}` will transform an equation of this form
    into one that is linear (see the docstring of
    :obj:`~sympy.solvers.ode.single.FirstLinear`).  The general solution is::

        >>> from sympy import Function, dsolve, Eq, pprint
        >>> from sympy.abc import x, n
        >>> f, P, Q = map(Function, ['f', 'P', 'Q'])
        >>> genform = Eq(f(x).diff(x) + P(x)*f(x), Q(x)*f(x)**n)
        >>> pprint(genform)
                    d                n
        P(x)*f(x) + --(f(x)) = Q(x)*f (x)
                    dx
        >>> pprint(dsolve(genform, f(x), hint='Bernoulli_Integral'), num_columns=110)
                                                                                                                -1
                                                                                                               -----
                                                                                                               n - 1
               //         /                                 /                            \                    \
               ||        |                                 |                             |                    |
               ||        |                  /              |                  /          |            /       |
               ||        |                 |               |                 |           |           |        |
               ||        |       -(n - 1)* | P(x) dx       |       -(n - 1)* | P(x) dx   |  (n - 1)* | P(x) dx|
               ||        |                 |               |                 |           |           |        |
               ||        |                /                |                /            |          /         |
        f(x) = ||C1 - n* | Q(x)*e                    dx +  | Q(x)*e                    dx|*e                  |
               ||        |                                 |                             |                    |
               \\       /                                 /                              /                    /


    Note that the equation is separable when `n = 1` (see the docstring of
    :obj:`~sympy.solvers.ode.single.Separable`).

    >>> pprint(dsolve(Eq(f(x).diff(x) + P(x)*f(x), Q(x)*f(x)), f(x),
    ... hint='separable_Integral'))
    f(x)
        /
    |                /
    |  1            |
    |  - dy = C1 +  | (-P(x) + Q(x)) dx
    |  y            |
    |              /
    /


    Examples
    ========

    >>> from sympy import Function, dsolve, Eq, pprint, log
    >>> from sympy.abc import x
    >>> f = Function('f')

    >>> pprint(dsolve(Eq(x*f(x).diff(x) + f(x), log(x)*f(x)**2),
    ... f(x), hint='Bernoulli'))
                    1
    f(x) =  -----------------
            C1*x + log(x) + 1

    References
    ==========

    - https://en.wikipedia.org/wiki/Bernoulli_differential_equation

    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 95

    # indirect doctest

    """
    hint = "Bernoulli"
    has_integral = True
    order = [1]

    def _wilds(self, f, x, order):
        P = Wild('P', exclude=[f(x)])
        Q = Wild('Q', exclude=[f(x)])
        n = Wild('n', exclude=[x, f(x), f(x).diff(x)])
        return P, Q, n

    def _equation(self, fx, x, order):
        P, Q, n = self.wilds()
        return fx.diff(x) + P*fx - Q*fx**n

    def _get_general_solution(self, *, simplify_flag: bool = True):
        P, Q, n = self.wilds_match()
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        (C1,) = self.ode_problem.get_numbered_constants(num=1)
        if n==1:
            gensol = Eq(log(fx), (
            C1 + Integral((-P + Q), x)
        ))
        else:
            gensol = Eq(fx**(1-n), (
                (C1 - (n - 1) * Integral(Q*exp(-n*Integral(P, x))
                            * exp(Integral(P, x)), x)
                ) * exp(-(1 - n)*Integral(P, x)))
            )
        return [gensol]


class Factorable(SingleODESolver):
    r"""
        Solves equations having a solvable factor.

        This function is used to solve the equation having factors. Factors may be of type algebraic or ode. It
        will try to solve each factor independently. Factors will be solved by calling dsolve. We will return the
        list of solutions.

        Examples
        ========

        >>> from sympy import Function, dsolve, pprint
        >>> from sympy.abc import x
        >>> f = Function('f')
        >>> eq = (f(x)**2-4)*(f(x).diff(x)+f(x))
        >>> pprint(dsolve(eq, f(x)))
                                        -x
        [f(x) = 2, f(x) = -2, f(x) = C1*e  ]


        """
    hint = "factorable"
    has_integral = False

    def _matches(self):
        eq_orig = self.ode_problem.eq
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        df = f(x).diff(x)
        self.eqs = []
        eq = eq_orig.collect(f(x), func = cancel)
        eq = fraction(factor(eq))[0]
        factors = Mul.make_args(factor(eq))
        roots = [fac.as_base_exp() for fac in factors if len(fac.args)!=0]
        if len(roots)>1 or roots[0][1]>1:
            for base, expo in roots:
                if base.has(f(x)):
                    self.eqs.append(base)
            if len(self.eqs)>0:
                return True
        roots = solve(eq, df)
        if len(roots)>0:
            self.eqs = [(df - root) for root in roots]
            # Avoid infinite recursion
            matches = self.eqs != [eq_orig]
            return matches
        for i in factors:
            if i.has(f(x)):
                self.eqs.append(i)
        return len(self.eqs)>0 and len(factors)>1

    def _get_general_solution(self, *, simplify_flag: bool = True):
        func = self.ode_problem.func.func
        x = self.ode_problem.sym
        eqns = self.eqs
        sols = []
        for eq in eqns:
            try:
                sol = dsolve(eq, func(x))
            except NotImplementedError:
                continue
            else:
                if isinstance(sol, list):
                    sols.extend(sol)
                else:
                    sols.append(sol)

        if sols == []:
            raise NotImplementedError("The given ODE " + str(eq) + " cannot be solved by"
                + " the factorable group method")
        return sols


class RiccatiSpecial(SinglePatternODESolver):
    r"""
    The general Riccati equation has the form

    .. math:: dy/dx = f(x) y^2 + g(x) y + h(x)\text{.}

    While it does not have a general solution [1], the "special" form, `dy/dx
    = a y^2 - b x^c`, does have solutions in many cases [2].  This routine
    returns a solution for `a(dy/dx) = b y^2 + c y/x + d/x^2` that is obtained
    by using a suitable change of variables to reduce it to the special form
    and is valid when neither `a` nor `b` are zero and either `c` or `d` is
    zero.

    >>> from sympy.abc import x, a, b, c, d
    >>> from sympy import dsolve, checkodesol, pprint, Function
    >>> f = Function('f')
    >>> y = f(x)
    >>> genform = a*y.diff(x) - (b*y**2 + c*y/x + d/x**2)
    >>> sol = dsolve(genform, y, hint="Riccati_special_minus2")
    >>> pprint(sol, wrap_line=False)
            /                                 /        __________________       \\
            |           __________________    |       /                2        ||
            |          /                2     |     \/  4*b*d - (a + c)  *log(x)||
           -|a + c - \/  4*b*d - (a + c)  *tan|C1 + ----------------------------||
            \                                 \                 2*a             //
    f(x) = ------------------------------------------------------------------------
                                            2*b*x

    >>> checkodesol(genform, sol, order=1)[0]
    True

    References
    ==========

    - https://www.maplesoft.com/support/help/Maple/view.aspx?path=odeadvisor/Riccati
    - https://eqworld.ipmnet.ru/en/solutions/ode/ode0106.pdf -
      https://eqworld.ipmnet.ru/en/solutions/ode/ode0123.pdf
    """
    hint = "Riccati_special_minus2"
    has_integral = False
    order = [1]

    def _wilds(self, f, x, order):
        a = Wild('a', exclude=[x, f(x), f(x).diff(x), 0])
        b = Wild('b', exclude=[x, f(x), f(x).diff(x), 0])
        c = Wild('c', exclude=[x, f(x), f(x).diff(x)])
        d = Wild('d', exclude=[x, f(x), f(x).diff(x)])
        return a, b, c, d

    def _equation(self, fx, x, order):
        a, b, c, d = self.wilds()
        return a*fx.diff(x) + b*fx**2 + c*fx/x + d/x**2

    def _get_general_solution(self, *, simplify_flag: bool = True):
        a, b, c, d = self.wilds_match()
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        (C1,) = self.ode_problem.get_numbered_constants(num=1)
        mu = sqrt(4*d*b - (a - c)**2)

        gensol = Eq(fx, (a - c - mu*tan(mu/(2*a)*log(x) + C1))/(2*b*x))
        return [gensol]


class RationalRiccati(SinglePatternODESolver):
    r"""
    Gives general solutions to the first order Riccati differential
    equations that have atleast one rational particular solution.

    .. math :: y' = b_0(x) + b_1(x) y + b_2(x) y^2

    where `b_0`, `b_1` and `b_2` are rational functions of `x`
    with `b_2 \ne 0` (`b_2 = 0` would make it a Bernoulli equation).

    Examples
    ========

    >>> from sympy import Symbol, Function, dsolve, checkodesol
    >>> f = Function('f')
    >>> x = Symbol('x')

    >>> eq = -x**4*f(x)**2 + x**3*f(x).diff(x) + x**2*f(x) + 20
    >>> sol = dsolve(eq, hint="1st_rational_riccati")
    >>> sol
    Eq(f(x), (4*C1 - 5*x**9 - 4)/(x**2*(C1 + x**9 - 1)))
    >>> checkodesol(eq, sol)
    (True, 0)

    References
    ==========

    - Riccati ODE:  https://en.wikipedia.org/wiki/Riccati_equation
    - N. Thieu Vo - Rational and Algebraic Solutions of First-Order Algebraic ODEs:
      Algorithm 11, pp. 78 - https://www3.risc.jku.at/publications/download/risc_5387/PhDThesisThieu.pdf
    """
    has_integral = False
    hint = "1st_rational_riccati"
    order = [1]

    def _wilds(self, f, x, order):
        b0 = Wild('b0', exclude=[f(x), f(x).diff(x)])
        b1 = Wild('b1', exclude=[f(x), f(x).diff(x)])
        b2 = Wild('b2', exclude=[f(x), f(x).diff(x)])
        return (b0, b1, b2)

    def _equation(self, fx, x, order):
        b0, b1, b2 = self.wilds()
        return fx.diff(x) - b0 - b1*fx - b2*fx**2

    def _matches(self):
        eq = self.ode_problem.eq_expanded
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        order = self.ode_problem.order

        if order != 1:
            return False

        match, funcs = match_riccati(eq, f, x)
        if not match:
            return False
        _b0, _b1, _b2 = funcs
        b0, b1, b2 = self.wilds()
        self._wilds_match = match = {b0: _b0, b1: _b1, b2: _b2}
        return True

    def _get_general_solution(self, *, simplify_flag: bool = True):
        # Match the equation
        b0, b1, b2 = self.wilds_match()
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        return solve_riccati(fx, x, b0, b1, b2, gensol=True)


class SecondNonlinearAutonomousConserved(SinglePatternODESolver):
    r"""
    Gives solution for the autonomous second order nonlinear
    differential equation of the form

    .. math :: f''(x) = g(f(x))

    The solution for this differential equation can be computed
    by multiplying by `f'(x)` and integrating on both sides,
    converting it into a first order differential equation.

    Examples
    ========

    >>> from sympy import Function, symbols, dsolve
    >>> f, g = symbols('f g', cls=Function)
    >>> x = symbols('x')

    >>> eq = f(x).diff(x, 2) - g(f(x))
    >>> dsolve(eq, simplify=False)
    [Eq(Integral(1/sqrt(C1 + 2*Integral(g(_u), _u)), (_u, f(x))), C2 + x),
    Eq(Integral(1/sqrt(C1 + 2*Integral(g(_u), _u)), (_u, f(x))), C2 - x)]

    >>> from sympy import exp, log
    >>> eq = f(x).diff(x, 2) - exp(f(x)) + log(f(x))
    >>> dsolve(eq, simplify=False)
    [Eq(Integral(1/sqrt(-2*_u*log(_u) + 2*_u + C1 + 2*exp(_u)), (_u, f(x))), C2 + x),
    Eq(Integral(1/sqrt(-2*_u*log(_u) + 2*_u + C1 + 2*exp(_u)), (_u, f(x))), C2 - x)]

    References
    ==========

    - https://eqworld.ipmnet.ru/en/solutions/ode/ode0301.pdf
    """
    hint = "2nd_nonlinear_autonomous_conserved"
    has_integral = True
    order = [2]

    def _wilds(self, f, x, order):
        fy = Wild('fy', exclude=[0, f(x).diff(x), f(x).diff(x, 2)])
        return (fy, )

    def _equation(self, fx, x, order):
        fy = self.wilds()[0]
        return fx.diff(x, 2) + fy

    def _verify(self, fx):
        return self.ode_problem.is_autonomous

    def _get_general_solution(self, *, simplify_flag: bool = True):
        g = self.wilds_match()[0]
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        u = Dummy('u')
        g = g.subs(fx, u)
        C1, C2 = self.ode_problem.get_numbered_constants(num=2)
        inside = -2*Integral(g, u) + C1
        lhs = Integral(1/sqrt(inside), (u, fx))
        return [Eq(lhs, C2 + x), Eq(lhs, C2 - x)]


class Liouville(SinglePatternODESolver):
    r"""
    Solves 2nd order Liouville differential equations.

    The general form of a Liouville ODE is

    .. math:: \frac{d^2 y}{dx^2} + g(y) \left(\!
                \frac{dy}{dx}\!\right)^2 + h(x)
                \frac{dy}{dx}\text{.}

    The general solution is:

        >>> from sympy import Function, dsolve, Eq, pprint, diff
        >>> from sympy.abc import x
        >>> f, g, h = map(Function, ['f', 'g', 'h'])
        >>> genform = Eq(diff(f(x),x,x) + g(f(x))*diff(f(x),x)**2 +
        ... h(x)*diff(f(x),x), 0)
        >>> pprint(genform)
                          2                    2
                /d       \         d          d
        g(f(x))*|--(f(x))|  + h(x)*--(f(x)) + ---(f(x)) = 0
                \dx      /         dx           2
                                              dx
        >>> pprint(dsolve(genform, f(x), hint='Liouville_Integral'))
                                          f(x)
                  /                     /
                 |                     |
                 |     /               |     /
                 |    |                |    |
                 |  - | h(x) dx        |    | g(y) dy
                 |    |                |    |
                 |   /                 |   /
        C1 + C2* | e            dx +   |  e           dy = 0
                 |                     |
                /                     /

    Examples
    ========

    >>> from sympy import Function, dsolve, Eq, pprint
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(diff(f(x), x, x) + diff(f(x), x)**2/f(x) +
    ... diff(f(x), x)/x, f(x), hint='Liouville'))
               ________________           ________________
    [f(x) = -\/ C1 + C2*log(x) , f(x) = \/ C1 + C2*log(x) ]

    References
    ==========

    - Goldstein and Braun, "Advanced Methods for the Solution of Differential
      Equations", pp. 98
    - https://www.maplesoft.com/support/help/Maple/view.aspx?path=odeadvisor/Liouville

    # indirect doctest

    """
    hint = "Liouville"
    has_integral = True
    order = [2]

    def _wilds(self, f, x, order):
        d = Wild('d', exclude=[f(x).diff(x), f(x).diff(x, 2)])
        e = Wild('e', exclude=[f(x).diff(x)])
        k = Wild('k', exclude=[f(x).diff(x)])
        return d, e, k

    def _equation(self, fx, x, order):
        # Liouville ODE in the form
        # f(x).diff(x, 2) + g(f(x))*(f(x).diff(x))**2 + h(x)*f(x).diff(x)
        # See Goldstein and Braun, "Advanced Methods for the Solution of
        # Differential Equations", pg. 98
        d, e, k = self.wilds()
        return d*fx.diff(x, 2) + e*fx.diff(x)**2 + k*fx.diff(x)

    def _verify(self, fx):
        d, e, k = self.wilds_match()
        self.y = Dummy('y')
        x = self.ode_problem.sym
        self.g = simplify(e/d).subs(fx, self.y)
        self.h = simplify(k/d).subs(fx, self.y)
        if self.y in self.h.free_symbols or x in self.g.free_symbols:
            return False
        return True

    def _get_general_solution(self, *, simplify_flag: bool = True):
        d, e, k = self.wilds_match()
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        C1, C2 = self.ode_problem.get_numbered_constants(num=2)
        int = Integral(exp(Integral(self.g, self.y)), (self.y, None, fx))
        gen_sol = Eq(int + C1*Integral(exp(-Integral(self.h, x)), x) + C2, 0)

        return [gen_sol]


class Separable(SinglePatternODESolver):
    r"""
    Solves separable 1st order differential equations.

    This is any differential equation that can be written as `P(y)
    \tfrac{dy}{dx} = Q(x)`.  The solution can then just be found by
    rearranging terms and integrating: `\int P(y) \,dy = \int Q(x) \,dx`.
    This hint uses :py:meth:`sympy.simplify.simplify.separatevars` as its back
    end, so if a separable equation is not caught by this solver, it is most
    likely the fault of that function.
    :py:meth:`~sympy.simplify.simplify.separatevars` is
    smart enough to do most expansion and factoring necessary to convert a
    separable equation `F(x, y)` into the proper form `P(x)\cdot{}Q(y)`.  The
    general solution is::

        >>> from sympy import Function, dsolve, Eq, pprint
        >>> from sympy.abc import x
        >>> a, b, c, d, f = map(Function, ['a', 'b', 'c', 'd', 'f'])
        >>> genform = Eq(a(x)*b(f(x))*f(x).diff(x), c(x)*d(f(x)))
        >>> pprint(genform)
                     d
        a(x)*b(f(x))*--(f(x)) = c(x)*d(f(x))
                     dx
        >>> pprint(dsolve(genform, f(x), hint='separable_Integral'))
             f(x)
           /                  /
          |                  |
          |  b(y)            | c(x)
          |  ---- dy = C1 +  | ---- dx
          |  d(y)            | a(x)
          |                  |
         /                  /

    Examples
    ========

    >>> from sympy import Function, dsolve, Eq
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(Eq(f(x)*f(x).diff(x) + x, 3*x*f(x)**2), f(x),
    ... hint='separable', simplify=False))
       /   2       \         2
    log\3*f (x) - 1/        x
    ---------------- = C1 + --
           6                2

    References
    ==========

    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 52

    # indirect doctest

    """
    hint = "separable"
    has_integral = True
    order = [1]

    def _wilds(self, f, x, order):
        d = Wild('d', exclude=[f(x).diff(x), f(x).diff(x, 2)])
        e = Wild('e', exclude=[f(x).diff(x)])
        return d, e

    def _equation(self, fx, x, order):
        d, e = self.wilds()
        return d + e*fx.diff(x)

    def _verify(self, fx):
        d, e = self.wilds_match()
        self.y = Dummy('y')
        x = self.ode_problem.sym
        d = separatevars(d.subs(fx, self.y))
        e = separatevars(e.subs(fx, self.y))
        # m1[coeff]*m1[x]*m1[y] + m2[coeff]*m2[x]*m2[y]*y'
        self.m1 = separatevars(d, dict=True, symbols=(x, self.y))
        self.m2 = separatevars(e, dict=True, symbols=(x, self.y))
        return bool(self.m1 and self.m2)

    def _get_match_object(self):
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        return self.m1, self.m2, x, fx

    def _get_general_solution(self, *, simplify_flag: bool = True):
        m1, m2, x, fx = self._get_match_object()
        (C1,) = self.ode_problem.get_numbered_constants(num=1)
        int = Integral(m2['coeff']*m2[self.y]/m1[self.y],
        (self.y, None, fx))
        gen_sol = Eq(int, Integral(-m1['coeff']*m1[x]/
        m2[x], x) + C1)
        return [gen_sol]


class SeparableReduced(Separable):
    r"""
    Solves a differential equation that can be reduced to the separable form.

    The general form of this equation is

    .. math:: y' + (y/x) H(x^n y) = 0\text{}.

    This can be solved by substituting `u(y) = x^n y`.  The equation then
    reduces to the separable form `\frac{u'}{u (\mathrm{power} - H(u))} -
    \frac{1}{x} = 0`.

    The general solution is:

        >>> from sympy import Function, dsolve, pprint
        >>> from sympy.abc import x, n
        >>> f, g = map(Function, ['f', 'g'])
        >>> genform = f(x).diff(x) + (f(x)/x)*g(x**n*f(x))
        >>> pprint(genform)
                         / n     \
        d          f(x)*g\x *f(x)/
        --(f(x)) + ---------------
        dx                x
        >>> pprint(dsolve(genform, hint='separable_reduced'))
         n
        x *f(x)
          /
         |
         |         1
         |    ------------ dy = C1 + log(x)
         |    y*(n - g(y))
         |
         /

    See Also
    ========
    :obj:`sympy.solvers.ode.single.Separable`

    Examples
    ========

    >>> from sympy import dsolve, Function, pprint
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> d = f(x).diff(x)
    >>> eq = (x - x**2*f(x))*d - f(x)
    >>> dsolve(eq, hint='separable_reduced')
    [Eq(f(x), (1 - sqrt(C1*x**2 + 1))/x), Eq(f(x), (sqrt(C1*x**2 + 1) + 1)/x)]
    >>> pprint(dsolve(eq, hint='separable_reduced'))
                   ___________            ___________
                  /     2                /     2
            1 - \/  C1*x  + 1          \/  C1*x  + 1  + 1
    [f(x) = ------------------, f(x) = ------------------]
                    x                          x

    References
    ==========

    - Joel Moses, "Symbolic Integration - The Stormy Decade", Communications
      of the ACM, Volume 14, Number 8, August 1971, pp. 558
    """
    hint = "separable_reduced"
    has_integral = True
    order = [1]

    def _degree(self, expr, x):
        # Made this function to calculate the degree of
        # x in an expression. If expr will be of form
        # x**p*y, (wheare p can be variables/rationals) then it
        # will return p.
        for val in expr:
            if val.has(x):
                if isinstance(val, Pow) and val.as_base_exp()[0] == x:
                    return (val.as_base_exp()[1])
                elif val == x:
                    return (val.as_base_exp()[1])
                else:
                    return self._degree(val.args, x)
        return 0

    def _powers(self, expr):
        # this function will return all the different relative power of x w.r.t f(x).
        # expr = x**p * f(x)**q then it will return {p/q}.
        pows = set()
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        self.y = Dummy('y')
        if isinstance(expr, Add):
            exprs = expr.atoms(Add)
        elif isinstance(expr, Mul):
            exprs = expr.atoms(Mul)
        elif isinstance(expr, Pow):
            exprs = expr.atoms(Pow)
        else:
            exprs = {expr}

        for arg in exprs:
            if arg.has(x):
                _, u = arg.as_independent(x, fx)
                pow = self._degree((u.subs(fx, self.y), ), x)/self._degree((u.subs(fx, self.y), ), self.y)
                pows.add(pow)
        return pows

    def _verify(self, fx):
        num, den = self.wilds_match()
        x = self.ode_problem.sym
        factor = simplify(x/fx*num/den)
        # Try representing factor in terms of x^n*y
        # where n is lowest power of x in factor;
        # first remove terms like sqrt(2)*3 from factor.atoms(Mul)
        num, dem = factor.as_numer_denom()
        num = expand(num)
        dem = expand(dem)
        pows = self._powers(num)
        pows.update(self._powers(dem))
        pows = list(pows)
        if(len(pows)==1) and pows[0]!=zoo:
            self.t = Dummy('t')
            self.r2 = {'t': self.t}
            num = num.subs(x**pows[0]*fx, self.t)
            dem = dem.subs(x**pows[0]*fx, self.t)
            test = num/dem
            free = test.free_symbols
            if len(free) == 1 and free.pop() == self.t:
                self.r2.update({'power' : pows[0], 'u' : test})
                return True
            return False
        return False

    def _get_match_object(self):
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        u = self.r2['u'].subs(self.r2['t'], self.y)
        ycoeff = 1/(self.y*(self.r2['power'] - u))
        m1 = {self.y: 1, x: -1/x, 'coeff': 1}
        m2 = {self.y: ycoeff, x: 1, 'coeff': 1}
        return m1, m2, x, x**self.r2['power']*fx


class HomogeneousCoeffSubsDepDivIndep(SinglePatternODESolver):
    r"""
    Solves a 1st order differential equation with homogeneous coefficients
    using the substitution `u_1 = \frac{\text{<dependent
    variable>}}{\text{<independent variable>}}`.

    This is a differential equation

    .. math:: P(x, y) + Q(x, y) dy/dx = 0

    such that `P` and `Q` are homogeneous and of the same order.  A function
    `F(x, y)` is homogeneous of order `n` if `F(x t, y t) = t^n F(x, y)`.
    Equivalently, `F(x, y)` can be rewritten as `G(y/x)` or `H(x/y)`.  See
    also the docstring of :py:meth:`~sympy.solvers.ode.homogeneous_order`.

    If the coefficients `P` and `Q` in the differential equation above are
    homogeneous functions of the same order, then it can be shown that the
    substitution `y = u_1 x` (i.e. `u_1 = y/x`) will turn the differential
    equation into an equation separable in the variables `x` and `u`.  If
    `h(u_1)` is the function that results from making the substitution `u_1 =
    f(x)/x` on `P(x, f(x))` and `g(u_2)` is the function that results from the
    substitution on `Q(x, f(x))` in the differential equation `P(x, f(x)) +
    Q(x, f(x)) f'(x) = 0`, then the general solution is::

        >>> from sympy import Function, dsolve, pprint
        >>> from sympy.abc import x
        >>> f, g, h = map(Function, ['f', 'g', 'h'])
        >>> genform = g(f(x)/x) + h(f(x)/x)*f(x).diff(x)
        >>> pprint(genform)
         /f(x)\    /f(x)\ d
        g|----| + h|----|*--(f(x))
         \ x  /    \ x  / dx
        >>> pprint(dsolve(genform, f(x),
        ... hint='1st_homogeneous_coeff_subs_dep_div_indep_Integral'))
                       f(x)
                       ----
                        x
                         /
                        |
                        |       -h(u1)
        log(x) = C1 +   |  ---------------- d(u1)
                        |  u1*h(u1) + g(u1)
                        |
                       /

    Where `u_1 h(u_1) + g(u_1) \ne 0` and `x \ne 0`.

    See also the docstrings of
    :obj:`~sympy.solvers.ode.single.HomogeneousCoeffBest` and
    :obj:`~sympy.solvers.ode.single.HomogeneousCoeffSubsIndepDivDep`.

    Examples
    ========

    >>> from sympy import Function, dsolve
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(2*x*f(x) + (x**2 + f(x)**2)*f(x).diff(x), f(x),
    ... hint='1st_homogeneous_coeff_subs_dep_div_indep', simplify=False))
                          /          3   \
                          |3*f(x)   f (x)|
                       log|------ + -----|
                          |  x         3 |
                          \           x  /
    log(x) = log(C1) - -------------------
                                3

    References
    ==========

    - https://en.wikipedia.org/wiki/Homogeneous_differential_equation
    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 59

    # indirect doctest

    """
    hint = "1st_homogeneous_coeff_subs_dep_div_indep"
    has_integral = True
    order = [1]

    def _wilds(self, f, x, order):
        d = Wild('d', exclude=[f(x).diff(x), f(x).diff(x, 2)])
        e = Wild('e', exclude=[f(x).diff(x)])
        return d, e

    def _equation(self, fx, x, order):
        d, e = self.wilds()
        return d + e*fx.diff(x)

    def _verify(self, fx):
        self.d, self.e = self.wilds_match()
        self.y = Dummy('y')
        x = self.ode_problem.sym
        self.d = separatevars(self.d.subs(fx, self.y))
        self.e = separatevars(self.e.subs(fx, self.y))
        ordera = homogeneous_order(self.d, x, self.y)
        orderb = homogeneous_order(self.e, x, self.y)
        if ordera == orderb and ordera is not None:
            self.u = Dummy('u')
            if simplify((self.d + self.u*self.e).subs({x: 1, self.y: self.u})) != 0:
                return True
            return False
        return False

    def _get_match_object(self):
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        self.u1 = Dummy('u1')
        xarg = 0
        yarg = 0
        return [self.d, self.e, fx, x, self.u, self.u1, self.y, xarg, yarg]

    def _get_general_solution(self, *, simplify_flag: bool = True):
        d, e, fx, x, u, u1, y, xarg, yarg = self._get_match_object()
        (C1,) = self.ode_problem.get_numbered_constants(num=1)
        int = Integral(
            (-e/(d + u1*e)).subs({x: 1, y: u1}),
            (u1, None, fx/x))
        sol = logcombine(Eq(log(x), int + log(C1)), force=True)
        gen_sol = sol.subs(fx, u).subs(((u, u - yarg), (x, x - xarg), (u, fx)))
        return [gen_sol]


class HomogeneousCoeffSubsIndepDivDep(SinglePatternODESolver):
    r"""
    Solves a 1st order differential equation with homogeneous coefficients
    using the substitution `u_2 = \frac{\text{<independent
    variable>}}{\text{<dependent variable>}}`.

    This is a differential equation

    .. math:: P(x, y) + Q(x, y) dy/dx = 0

    such that `P` and `Q` are homogeneous and of the same order.  A function
    `F(x, y)` is homogeneous of order `n` if `F(x t, y t) = t^n F(x, y)`.
    Equivalently, `F(x, y)` can be rewritten as `G(y/x)` or `H(x/y)`.  See
    also the docstring of :py:meth:`~sympy.solvers.ode.homogeneous_order`.

    If the coefficients `P` and `Q` in the differential equation above are
    homogeneous functions of the same order, then it can be shown that the
    substitution `x = u_2 y` (i.e. `u_2 = x/y`) will turn the differential
    equation into an equation separable in the variables `y` and `u_2`.  If
    `h(u_2)` is the function that results from making the substitution `u_2 =
    x/f(x)` on `P(x, f(x))` and `g(u_2)` is the function that results from the
    substitution on `Q(x, f(x))` in the differential equation `P(x, f(x)) +
    Q(x, f(x)) f'(x) = 0`, then the general solution is:

    >>> from sympy import Function, dsolve, pprint
    >>> from sympy.abc import x
    >>> f, g, h = map(Function, ['f', 'g', 'h'])
    >>> genform = g(x/f(x)) + h(x/f(x))*f(x).diff(x)
    >>> pprint(genform)
     / x  \    / x  \ d
    g|----| + h|----|*--(f(x))
     \f(x)/    \f(x)/ dx
    >>> pprint(dsolve(genform, f(x),
    ... hint='1st_homogeneous_coeff_subs_indep_div_dep_Integral'))
                 x
                ----
                f(x)
                  /
                 |
                 |       -g(u1)
                 |  ---------------- d(u1)
                 |  u1*g(u1) + h(u1)
                 |
                /
    <BLANKLINE>
    f(x) = C1*e

    Where `u_1 g(u_1) + h(u_1) \ne 0` and `f(x) \ne 0`.

    See also the docstrings of
    :obj:`~sympy.solvers.ode.single.HomogeneousCoeffBest` and
    :obj:`~sympy.solvers.ode.single.HomogeneousCoeffSubsDepDivIndep`.

    Examples
    ========

    >>> from sympy import Function, pprint, dsolve
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(2*x*f(x) + (x**2 + f(x)**2)*f(x).diff(x), f(x),
    ... hint='1st_homogeneous_coeff_subs_indep_div_dep',
    ... simplify=False))
                             /   2     \
                             |3*x      |
                          log|----- + 1|
                             | 2       |
                             \f (x)    /
    log(f(x)) = log(C1) - --------------
                                3

    References
    ==========

    - https://en.wikipedia.org/wiki/Homogeneous_differential_equation
    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 59

    # indirect doctest

    """
    hint = "1st_homogeneous_coeff_subs_indep_div_dep"
    has_integral = True
    order = [1]

    def _wilds(self, f, x, order):
        d = Wild('d', exclude=[f(x).diff(x), f(x).diff(x, 2)])
        e = Wild('e', exclude=[f(x).diff(x)])
        return d, e

    def _equation(self, fx, x, order):
        d, e = self.wilds()
        return d + e*fx.diff(x)

    def _verify(self, fx):
        self.d, self.e = self.wilds_match()
        self.y = Dummy('y')
        x = self.ode_problem.sym
        self.d = separatevars(self.d.subs(fx, self.y))
        self.e = separatevars(self.e.subs(fx, self.y))
        ordera = homogeneous_order(self.d, x, self.y)
        orderb = homogeneous_order(self.e, x, self.y)
        if ordera == orderb and ordera is not None:
            self.u = Dummy('u')
            if simplify((self.e + self.u*self.d).subs({x: self.u, self.y: 1})) != 0:
                return True
            return False
        return False

    def _get_match_object(self):
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        self.u1 = Dummy('u1')
        xarg = 0
        yarg = 0
        return [self.d, self.e, fx, x, self.u, self.u1, self.y, xarg, yarg]

    def _get_general_solution(self, *, simplify_flag: bool = True):
        d, e, fx, x, u, u1, y, xarg, yarg = self._get_match_object()
        (C1,) = self.ode_problem.get_numbered_constants(num=1)
        int = Integral(simplify((-d/(e + u1*d)).subs({x: u1, y: 1})), (u1, None, x/fx)) # type: ignore
        sol = logcombine(Eq(log(fx), int + log(C1)), force=True)
        gen_sol = sol.subs(fx, u).subs(((u, u - yarg), (x, x - xarg), (u, fx)))
        return [gen_sol]


class HomogeneousCoeffBest(HomogeneousCoeffSubsIndepDivDep, HomogeneousCoeffSubsDepDivIndep):
    r"""
    Returns the best solution to an ODE from the two hints
    ``1st_homogeneous_coeff_subs_dep_div_indep`` and
    ``1st_homogeneous_coeff_subs_indep_div_dep``.

    This is as determined by :py:meth:`~sympy.solvers.ode.ode.ode_sol_simplicity`.

    See the
    :obj:`~sympy.solvers.ode.single.HomogeneousCoeffSubsIndepDivDep`
    and
    :obj:`~sympy.solvers.ode.single.HomogeneousCoeffSubsDepDivIndep`
    docstrings for more information on these hints.  Note that there is no
    ``ode_1st_homogeneous_coeff_best_Integral`` hint.

    Examples
    ========

    >>> from sympy import Function, dsolve, pprint
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(2*x*f(x) + (x**2 + f(x)**2)*f(x).diff(x), f(x),
    ... hint='1st_homogeneous_coeff_best', simplify=False))
                             /   2     \
                             |3*x      |
                          log|----- + 1|
                             | 2       |
                             \f (x)    /
    log(f(x)) = log(C1) - --------------
                                3

    References
    ==========

    - https://en.wikipedia.org/wiki/Homogeneous_differential_equation
    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 59

    # indirect doctest

    """
    hint = "1st_homogeneous_coeff_best"
    has_integral = False
    order = [1]

    def _verify(self, fx):
        return HomogeneousCoeffSubsIndepDivDep._verify(self, fx) and \
               HomogeneousCoeffSubsDepDivIndep._verify(self, fx)

    def _get_general_solution(self, *, simplify_flag: bool = True):
        # There are two substitutions that solve the equation, u1=y/x and u2=x/y
        # # They produce different integrals, so try them both and see which
        # # one is easier
        sol1 = HomogeneousCoeffSubsIndepDivDep._get_general_solution(self)
        sol2 = HomogeneousCoeffSubsDepDivIndep._get_general_solution(self)
        fx = self.ode_problem.func
        if simplify_flag:
            sol1 = odesimp(self.ode_problem.eq, *sol1, fx, "1st_homogeneous_coeff_subs_indep_div_dep")
            sol2 = odesimp(self.ode_problem.eq, *sol2, fx, "1st_homogeneous_coeff_subs_dep_div_indep")
        # XXX: not simplify should be not simplify_flag. mypy correctly complains
        return min([sol1, sol2], key=lambda x: ode_sol_simplicity(x, fx, trysolving=not simplify)) # type: ignore


class LinearCoefficients(HomogeneousCoeffBest):
    r"""
    Solves a differential equation with linear coefficients.

    The general form of a differential equation with linear coefficients is

    .. math:: y' + F\left(\!\frac{a_1 x + b_1 y + c_1}{a_2 x + b_2 y +
                c_2}\!\right) = 0\text{,}

    where `a_1`, `b_1`, `c_1`, `a_2`, `b_2`, `c_2` are constants and `a_1 b_2
    - a_2 b_1 \ne 0`.

    This can be solved by substituting:

    .. math:: x = x' + \frac{b_2 c_1 - b_1 c_2}{a_2 b_1 - a_1 b_2}

              y = y' + \frac{a_1 c_2 - a_2 c_1}{a_2 b_1 - a_1
                  b_2}\text{.}

    This substitution reduces the equation to a homogeneous differential
    equation.

    See Also
    ========
    :obj:`sympy.solvers.ode.single.HomogeneousCoeffBest`
    :obj:`sympy.solvers.ode.single.HomogeneousCoeffSubsIndepDivDep`
    :obj:`sympy.solvers.ode.single.HomogeneousCoeffSubsDepDivIndep`

    Examples
    ========

    >>> from sympy import dsolve, Function, pprint
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> df = f(x).diff(x)
    >>> eq = (x + f(x) + 1)*df + (f(x) - 6*x + 1)
    >>> dsolve(eq, hint='linear_coefficients')
    [Eq(f(x), -x - sqrt(C1 + 7*x**2) - 1), Eq(f(x), -x + sqrt(C1 + 7*x**2) - 1)]
    >>> pprint(dsolve(eq, hint='linear_coefficients'))
                      ___________                     ___________
                   /         2                     /         2
    [f(x) = -x - \/  C1 + 7*x   - 1, f(x) = -x + \/  C1 + 7*x   - 1]


    References
    ==========

    - Joel Moses, "Symbolic Integration - The Stormy Decade", Communications
      of the ACM, Volume 14, Number 8, August 1971, pp. 558
    """
    hint = "linear_coefficients"
    has_integral = True
    order = [1]

    def _wilds(self, f, x, order):
        d = Wild('d', exclude=[f(x).diff(x), f(x).diff(x, 2)])
        e = Wild('e', exclude=[f(x).diff(x)])
        return d, e

    def _equation(self, fx, x, order):
        d, e = self.wilds()
        return d + e*fx.diff(x)

    def _verify(self, fx):
        self.d, self.e = self.wilds_match()
        a, b = self.wilds()
        F = self.d/self.e
        x = self.ode_problem.sym
        params = self._linear_coeff_match(F, fx)
        if params:
            self.xarg, self.yarg = params
            u = Dummy('u')
            t = Dummy('t')
            self.y = Dummy('y')
            # Dummy substitution for df and f(x).
            dummy_eq = self.ode_problem.eq.subs(((fx.diff(x), t), (fx, u)))
            reps = ((x, x + self.xarg), (u, u + self.yarg), (t, fx.diff(x)), (u, fx))
            dummy_eq = simplify(dummy_eq.subs(reps))
            # get the re-cast values for e and d
            r2 = collect(expand(dummy_eq), [fx.diff(x), fx]).match(a*fx.diff(x) + b)
            if r2:
                self.d, self.e = r2[b], r2[a]
                orderd = homogeneous_order(self.d, x, fx)
                ordere = homogeneous_order(self.e, x, fx)
                if orderd == ordere and orderd is not None:
                    self.d = self.d.subs(fx, self.y)
                    self.e = self.e.subs(fx, self.y)
                    return True
                return False
            return False

    def _linear_coeff_match(self, expr, func):
        r"""
        Helper function to match hint ``linear_coefficients``.

        Matches the expression to the form `(a_1 x + b_1 f(x) + c_1)/(a_2 x + b_2
        f(x) + c_2)` where the following conditions hold:

        1. `a_1`, `b_1`, `c_1`, `a_2`, `b_2`, `c_2` are Rationals;
        2. `c_1` or `c_2` are not equal to zero;
        3. `a_2 b_1 - a_1 b_2` is not equal to zero.

        Return ``xarg``, ``yarg`` where

        1. ``xarg`` = `(b_2 c_1 - b_1 c_2)/(a_2 b_1 - a_1 b_2)`
        2. ``yarg`` = `(a_1 c_2 - a_2 c_1)/(a_2 b_1 - a_1 b_2)`


        Examples
        ========

        >>> from sympy import Function, sin
        >>> from sympy.abc import x
        >>> from sympy.solvers.ode.single import LinearCoefficients
        >>> f = Function('f')
        >>> eq = (-25*f(x) - 8*x + 62)/(4*f(x) + 11*x - 11)
        >>> obj = LinearCoefficients(eq)
        >>> obj._linear_coeff_match(eq, f(x))
        (1/9, 22/9)
        >>> eq = sin((-5*f(x) - 8*x + 6)/(4*f(x) + x - 1))
        >>> obj = LinearCoefficients(eq)
        >>> obj._linear_coeff_match(eq, f(x))
        (19/27, 2/27)
        >>> eq = sin(f(x)/x)
        >>> obj = LinearCoefficients(eq)
        >>> obj._linear_coeff_match(eq, f(x))

        """
        f = func.func
        x = func.args[0]
        def abc(eq):
            r'''
            Internal function of _linear_coeff_match
            that returns Rationals a, b, c
            if eq is a*x + b*f(x) + c, else None.
            '''
            eq = _mexpand(eq)
            c = eq.as_independent(x, f(x), as_Add=True)[0]
            if not c.is_Rational:
                return
            a = eq.coeff(x)
            if not a.is_Rational:
                return
            b = eq.coeff(f(x))
            if not b.is_Rational:
                return
            if eq == a*x + b*f(x) + c:
                return a, b, c

        def match(arg):
            r'''
            Internal function of _linear_coeff_match that returns Rationals a1,
            b1, c1, a2, b2, c2 and a2*b1 - a1*b2 of the expression (a1*x + b1*f(x)
            + c1)/(a2*x + b2*f(x) + c2) if one of c1 or c2 and a2*b1 - a1*b2 is
            non-zero, else None.
            '''
            n, d = arg.together().as_numer_denom()
            m = abc(n)
            if m is not None:
                a1, b1, c1 = m
                m = abc(d)
                if m is not None:
                    a2, b2, c2 = m
                    d = a2*b1 - a1*b2
                    if (c1 or c2) and d:
                        return a1, b1, c1, a2, b2, c2, d

        m = [fi.args[0] for fi in expr.atoms(Function) if fi.func != f and
            len(fi.args) == 1 and not fi.args[0].is_Function] or {expr}
        m1 = match(m.pop())
        if m1 and all(match(mi) == m1 for mi in m):
            a1, b1, c1, a2, b2, c2, denom = m1
            return (b2*c1 - b1*c2)/denom, (a1*c2 - a2*c1)/denom

    def _get_match_object(self):
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        self.u1 = Dummy('u1')
        u = Dummy('u')
        return [self.d, self.e, fx, x, u, self.u1, self.y, self.xarg, self.yarg]


class NthOrderReducible(SingleODESolver):
    r"""
    Solves ODEs that only involve derivatives of the dependent variable using
    a substitution of the form `f^n(x) = g(x)`.

    For example any second order ODE of the form `f''(x) = h(f'(x), x)` can be
    transformed into a pair of 1st order ODEs `g'(x) = h(g(x), x)` and
    `f'(x) = g(x)`. Usually the 1st order ODE for `g` is easier to solve. If
    that gives an explicit solution for `g` then `f` is found simply by
    integration.


    Examples
    ========

    >>> from sympy import Function, dsolve, Eq
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> eq = Eq(x*f(x).diff(x)**2 + f(x).diff(x, 2), 0)
    >>> dsolve(eq, f(x), hint='nth_order_reducible')
    ... # doctest: +NORMALIZE_WHITESPACE
    Eq(f(x), C1 - sqrt(-1/C2)*log(-C2*sqrt(-1/C2) + x) + sqrt(-1/C2)*log(C2*sqrt(-1/C2) + x))

    """
    hint = "nth_order_reducible"
    has_integral = False

    def _matches(self):
        # Any ODE that can be solved with a substitution and
        # repeated integration e.g.:
        # `d^2/dx^2(y) + x*d/dx(y) = constant
        #f'(x) must be finite for this to work
        eq = self.ode_problem.eq_preprocessed
        func = self.ode_problem.func
        x = self.ode_problem.sym
        r"""
        Matches any differential equation that can be rewritten with a smaller
        order. Only derivatives of ``func`` alone, wrt a single variable,
        are considered, and only in them should ``func`` appear.
        """
        # ODE only handles functions of 1 variable so this affirms that state
        assert len(func.args) == 1
        vc = [d.variable_count[0] for d in eq.atoms(Derivative)
            if d.expr == func and len(d.variable_count) == 1]
        ords = [c for v, c in vc if v == x]
        if len(ords) < 2:
            return False
        self.smallest = min(ords)
        # make sure func does not appear outside of derivatives
        D = Dummy()
        if eq.subs(func.diff(x, self.smallest), D).has(func):
            return False
        return True

    def _get_general_solution(self, *, simplify_flag: bool = True):
        eq = self.ode_problem.eq
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        n = self.smallest
        # get a unique function name for g
        names = [a.name for a in eq.atoms(AppliedUndef)]
        while True:
            name = Dummy().name
            if name not in names:
                g = Function(name)
                break
        w = f(x).diff(x, n)
        geq = eq.subs(w, g(x))
        gsol = dsolve(geq, g(x))

        if not isinstance(gsol, list):
            gsol = [gsol]

        # Might be multiple solutions to the reduced ODE:
        fsol = []
        for gsoli in gsol:
            fsoli = dsolve(gsoli.subs(g(x), w), f(x))  # or do integration n times
            fsol.append(fsoli)

        return fsol


class SecondHypergeometric(SingleODESolver):
    r"""
    Solves 2nd order linear differential equations.

    It computes special function solutions which can be expressed using the
    2F1, 1F1 or 0F1 hypergeometric functions.

    .. math:: y'' + A(x) y' + B(x) y = 0\text{,}

    where `A` and `B` are rational functions.

    These kinds of differential equations have solution of non-Liouvillian form.

    Given linear ODE can be obtained from 2F1 given by

    .. math:: (x^2 - x) y'' + ((a + b + 1) x - c) y' + b a y = 0\text{,}

    where {a, b, c} are arbitrary constants.

    Notes
    =====

    The algorithm should find any solution of the form

    .. math:: y = P(x) _pF_q(..; ..;\frac{\alpha x^k + \beta}{\gamma x^k + \delta})\text{,}

    where pFq is any of 2F1, 1F1 or 0F1 and `P` is an "arbitrary function".
    Currently only the 2F1 case is implemented in SymPy but the other cases are
    described in the paper and could be implemented in future (contributions
    welcome!).


    Examples
    ========

    >>> from sympy import Function, dsolve, pprint
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> eq = (x*x - x)*f(x).diff(x,2) + (5*x - 1)*f(x).diff(x) + 4*f(x)
    >>> pprint(dsolve(eq, f(x), '2nd_hypergeometric'))
                                        _
           /        /           4  \\  |_  /-1, -1 |  \
           |C1 + C2*|log(x) + -----||* |   |       | x|
           \        \         x + 1// 2  1 \  1    |  /
    f(x) = --------------------------------------------
                                    3
                             (x - 1)


    References
    ==========

    - "Non-Liouvillian solutions for second order linear ODEs" by L. Chan, E.S. Cheb-Terrab

    """
    hint = "2nd_hypergeometric"
    has_integral = True

    def _matches(self):
        eq = self.ode_problem.eq_preprocessed
        func = self.ode_problem.func
        r = match_2nd_hypergeometric(eq, func)
        self.match_object = None
        if r:
            A, B = r
            d = equivalence_hypergeometric(A, B, func)
            if d:
                if d['type'] == "2F1":
                    self.match_object = match_2nd_2F1_hypergeometric(d['I0'], d['k'], d['sing_point'], func)
                    if self.match_object is not None:
                        self.match_object.update({'A':A, 'B':B})
            # We can extend it for 1F1 and 0F1 type also.
        return self.match_object is not None

    def _get_general_solution(self, *, simplify_flag: bool = True):
        eq = self.ode_problem.eq
        func = self.ode_problem.func
        if self.match_object['type'] == "2F1":
            sol = get_sol_2F1_hypergeometric(eq, func, self.match_object)
            if sol is None:
                raise NotImplementedError("The given ODE " + str(eq) + " cannot be solved by"
                    + " the hypergeometric method")

        return [sol]


class NthLinearConstantCoeffHomogeneous(SingleODESolver):
    r"""
    Solves an `n`\th order linear homogeneous differential equation with
    constant coefficients.

    This is an equation of the form

    .. math:: a_n f^{(n)}(x) + a_{n-1} f^{(n-1)}(x) + \cdots + a_1 f'(x)
                + a_0 f(x) = 0\text{.}

    These equations can be solved in a general manner, by taking the roots of
    the characteristic equation `a_n m^n + a_{n-1} m^{n-1} + \cdots + a_1 m +
    a_0 = 0`.  The solution will then be the sum of `C_n x^i e^{r x}` terms,
    for each where `C_n` is an arbitrary constant, `r` is a root of the
    characteristic equation and `i` is one of each from 0 to the multiplicity
    of the root - 1 (for example, a root 3 of multiplicity 2 would create the
    terms `C_1 e^{3 x} + C_2 x e^{3 x}`).  The exponential is usually expanded
    for complex roots using Euler's equation `e^{I x} = \cos(x) + I \sin(x)`.
    Complex roots always come in conjugate pairs in polynomials with real
    coefficients, so the two roots will be represented (after simplifying the
    constants) as `e^{a x} \left(C_1 \cos(b x) + C_2 \sin(b x)\right)`.

    If SymPy cannot find exact roots to the characteristic equation, a
    :py:class:`~sympy.polys.rootoftools.ComplexRootOf` instance will be return
    instead.

    >>> from sympy import Function, dsolve
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> dsolve(f(x).diff(x, 5) + 10*f(x).diff(x) - 2*f(x), f(x),
    ... hint='nth_linear_constant_coeff_homogeneous')
    ... # doctest: +NORMALIZE_WHITESPACE
    Eq(f(x), C5*exp(x*CRootOf(_x**5 + 10*_x - 2, 0))
    + (C1*sin(x*im(CRootOf(_x**5 + 10*_x - 2, 1)))
    + C2*cos(x*im(CRootOf(_x**5 + 10*_x - 2, 1))))*exp(x*re(CRootOf(_x**5 + 10*_x - 2, 1)))
    + (C3*sin(x*im(CRootOf(_x**5 + 10*_x - 2, 3)))
    + C4*cos(x*im(CRootOf(_x**5 + 10*_x - 2, 3))))*exp(x*re(CRootOf(_x**5 + 10*_x - 2, 3))))

    Note that because this method does not involve integration, there is no
    ``nth_linear_constant_coeff_homogeneous_Integral`` hint.

    Examples
    ========

    >>> from sympy import Function, dsolve, pprint
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(f(x).diff(x, 4) + 2*f(x).diff(x, 3) -
    ... 2*f(x).diff(x, 2) - 6*f(x).diff(x) + 5*f(x), f(x),
    ... hint='nth_linear_constant_coeff_homogeneous'))
                        x                            -2*x
    f(x) = (C1 + C2*x)*e  + (C3*sin(x) + C4*cos(x))*e

    References
    ==========

    - https://en.wikipedia.org/wiki/Linear_differential_equation section:
      Nonhomogeneous_equation_with_constant_coefficients
    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 211

    # indirect doctest

    """
    hint = "nth_linear_constant_coeff_homogeneous"
    has_integral = False

    def _matches(self):
        eq = self.ode_problem.eq_high_order_free
        func = self.ode_problem.func
        order = self.ode_problem.order
        x = self.ode_problem.sym
        self.r = self.ode_problem.get_linear_coefficients(eq, func, order)
        if order and self.r and not any(self.r[i].has(x) for i in self.r if i >= 0):
            if not self.r[-1]:
                return True
            else:
                return False
        return False

    def _get_general_solution(self, *, simplify_flag: bool = True):
        fx = self.ode_problem.func
        order = self.ode_problem.order
        roots, collectterms = _get_const_characteristic_eq_sols(self.r, fx, order)
        # A generator of constants
        constants = self.ode_problem.get_numbered_constants(num=len(roots))
        gsol_rhs = Add(*[i*j for (i, j) in zip(constants, roots)])
        gsol = Eq(fx, gsol_rhs)
        if simplify_flag:
            gsol = _get_simplified_sol([gsol], fx, collectterms)

        return [gsol]


class NthLinearConstantCoeffVariationOfParameters(SingleODESolver):
    r"""
    Solves an `n`\th order linear differential equation with constant
    coefficients using the method of variation of parameters.

    This method works on any differential equations of the form

    .. math:: f^{(n)}(x) + a_{n-1} f^{(n-1)}(x) + \cdots + a_1 f'(x) + a_0
                f(x) = P(x)\text{.}

    This method works by assuming that the particular solution takes the form

    .. math:: \sum_{x=1}^{n} c_i(x) y_i(x)\text{,}

    where `y_i` is the `i`\th solution to the homogeneous equation.  The
    solution is then solved using Wronskian's and Cramer's Rule.  The
    particular solution is given by

    .. math:: \sum_{x=1}^n \left( \int \frac{W_i(x)}{W(x)} \,dx
                \right) y_i(x) \text{,}

    where `W(x)` is the Wronskian of the fundamental system (the system of `n`
    linearly independent solutions to the homogeneous equation), and `W_i(x)`
    is the Wronskian of the fundamental system with the `i`\th column replaced
    with `[0, 0, \cdots, 0, P(x)]`.

    This method is general enough to solve any `n`\th order inhomogeneous
    linear differential equation with constant coefficients, but sometimes
    SymPy cannot simplify the Wronskian well enough to integrate it.  If this
    method hangs, try using the
    ``nth_linear_constant_coeff_variation_of_parameters_Integral`` hint and
    simplifying the integrals manually.  Also, prefer using
    ``nth_linear_constant_coeff_undetermined_coefficients`` when it
    applies, because it does not use integration, making it faster and more
    reliable.

    Warning, using simplify=False with
    'nth_linear_constant_coeff_variation_of_parameters' in
    :py:meth:`~sympy.solvers.ode.dsolve` may cause it to hang, because it will
    not attempt to simplify the Wronskian before integrating.  It is
    recommended that you only use simplify=False with
    'nth_linear_constant_coeff_variation_of_parameters_Integral' for this
    method, especially if the solution to the homogeneous equation has
    trigonometric functions in it.

    Examples
    ========

    >>> from sympy import Function, dsolve, pprint, exp, log
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(f(x).diff(x, 3) - 3*f(x).diff(x, 2) +
    ... 3*f(x).diff(x) - f(x) - exp(x)*log(x), f(x),
    ... hint='nth_linear_constant_coeff_variation_of_parameters'))
           /       /       /     x*log(x)   11*x\\\  x
    f(x) = |C1 + x*|C2 + x*|C3 + -------- - ----|||*e
           \       \       \        6        36 ///

    References
    ==========

    - https://en.wikipedia.org/wiki/Variation_of_parameters
    - https://planetmath.org/VariationOfParameters
    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 233

    # indirect doctest

    """
    hint = "nth_linear_constant_coeff_variation_of_parameters"
    has_integral = True

    def _matches(self):
        eq = self.ode_problem.eq_high_order_free
        func = self.ode_problem.func
        order = self.ode_problem.order
        x = self.ode_problem.sym
        self.r = self.ode_problem.get_linear_coefficients(eq, func, order)

        if order and self.r and not any(self.r[i].has(x) for i in self.r if i >= 0):
            if self.r[-1]:
                return True
            else:
                return False
        return False

    def _get_general_solution(self, *, simplify_flag: bool = True):
        eq = self.ode_problem.eq_high_order_free
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        order = self.ode_problem.order
        roots, collectterms = _get_const_characteristic_eq_sols(self.r, f(x), order)
        # A generator of constants
        constants = self.ode_problem.get_numbered_constants(num=len(roots))
        homogen_sol_rhs = Add(*[i*j for (i, j) in zip(constants, roots)])
        homogen_sol = Eq(f(x), homogen_sol_rhs)
        homogen_sol = _solve_variation_of_parameters(eq, f(x), roots, homogen_sol, order, self.r, simplify_flag)
        if simplify_flag:
            homogen_sol = _get_simplified_sol([homogen_sol], f(x), collectterms)
        return [homogen_sol]


class NthLinearConstantCoeffUndeterminedCoefficients(SingleODESolver):
    r"""
    Solves an `n`\th order linear differential equation with constant
    coefficients using the method of undetermined coefficients.

    This method works on differential equations of the form

    .. math:: a_n f^{(n)}(x) + a_{n-1} f^{(n-1)}(x) + \cdots + a_1 f'(x)
                + a_0 f(x) = P(x)\text{,}

    where `P(x)` is a function that has a finite number of linearly
    independent derivatives.

    Functions that fit this requirement are finite sums functions of the form
    `a x^i e^{b x} \sin(c x + d)` or `a x^i e^{b x} \cos(c x + d)`, where `i`
    is a non-negative integer and `a`, `b`, `c`, and `d` are constants.  For
    example any polynomial in `x`, functions like `x^2 e^{2 x}`, `x \sin(x)`,
    and `e^x \cos(x)` can all be used.  Products of `\sin`'s and `\cos`'s have
    a finite number of derivatives, because they can be expanded into `\sin(a
    x)` and `\cos(b x)` terms.  However, SymPy currently cannot do that
    expansion, so you will need to manually rewrite the expression in terms of
    the above to use this method.  So, for example, you will need to manually
    convert `\sin^2(x)` into `(1 + \cos(2 x))/2` to properly apply the method
    of undetermined coefficients on it.

    This method works by creating a trial function from the expression and all
    of its linear independent derivatives and substituting them into the
    original ODE.  The coefficients for each term will be a system of linear
    equations, which are be solved for and substituted, giving the solution.
    If any of the trial functions are linearly dependent on the solution to
    the homogeneous equation, they are multiplied by sufficient `x` to make
    them linearly independent.

    Examples
    ========

    >>> from sympy import Function, dsolve, pprint, exp, cos
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(f(x).diff(x, 2) + 2*f(x).diff(x) + f(x) -
    ... 4*exp(-x)*x**2 + cos(2*x), f(x),
    ... hint='nth_linear_constant_coeff_undetermined_coefficients'))
           /       /      3\\
           |       |     x ||  -x   4*sin(2*x)   3*cos(2*x)
    f(x) = |C1 + x*|C2 + --||*e   - ---------- + ----------
           \       \     3 //           25           25

    References
    ==========

    - https://en.wikipedia.org/wiki/Method_of_undetermined_coefficients
    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 221

    # indirect doctest

    """
    hint = "nth_linear_constant_coeff_undetermined_coefficients"
    has_integral = False

    def _matches(self):
        eq = self.ode_problem.eq_high_order_free
        func = self.ode_problem.func
        order = self.ode_problem.order
        x = self.ode_problem.sym
        self.r = self.ode_problem.get_linear_coefficients(eq, func, order)
        does_match = False
        if order and self.r and not any(self.r[i].has(x) for i in self.r if i >= 0):
            if self.r[-1]:
                eq_homogeneous = Add(eq, -self.r[-1])
                undetcoeff = _undetermined_coefficients_match(self.r[-1], x, func, eq_homogeneous)
                if undetcoeff['test']:
                    self.trialset = undetcoeff['trialset']
                    does_match = True
        return does_match

    def _get_general_solution(self, *, simplify_flag: bool = True):
        eq = self.ode_problem.eq
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        order = self.ode_problem.order
        roots, collectterms = _get_const_characteristic_eq_sols(self.r, f(x), order)
        # A generator of constants
        constants = self.ode_problem.get_numbered_constants(num=len(roots))
        homogen_sol_rhs = Add(*[i*j for (i, j) in zip(constants, roots)])
        homogen_sol = Eq(f(x), homogen_sol_rhs)
        self.r.update({'list': roots, 'sol': homogen_sol, 'simpliy_flag': simplify_flag})
        gsol = _solve_undetermined_coefficients(eq, f(x), order, self.r, self.trialset)
        if simplify_flag:
            gsol = _get_simplified_sol([gsol], f(x), collectterms)
        return [gsol]


class NthLinearEulerEqHomogeneous(SingleODESolver):
    r"""
    Solves an `n`\th order linear homogeneous variable-coefficient
    Cauchy-Euler equidimensional ordinary differential equation.

    This is an equation with form `0 = a_0 f(x) + a_1 x f'(x) + a_2 x^2 f''(x)
    \cdots`.

    These equations can be solved in a general manner, by substituting
    solutions of the form `f(x) = x^r`, and deriving a characteristic equation
    for `r`.  When there are repeated roots, we include extra terms of the
    form `C_{r k} \ln^k(x) x^r`, where `C_{r k}` is an arbitrary integration
    constant, `r` is a root of the characteristic equation, and `k` ranges
    over the multiplicity of `r`.  In the cases where the roots are complex,
    solutions of the form `C_1 x^a \sin(b \log(x)) + C_2 x^a \cos(b \log(x))`
    are returned, based on expansions with Euler's formula.  The general
    solution is the sum of the terms found.  If SymPy cannot find exact roots
    to the characteristic equation, a
    :py:obj:`~.ComplexRootOf` instance will be returned
    instead.

    >>> from sympy import Function, dsolve
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> dsolve(4*x**2*f(x).diff(x, 2) + f(x), f(x),
    ... hint='nth_linear_euler_eq_homogeneous')
    ... # doctest: +NORMALIZE_WHITESPACE
    Eq(f(x), sqrt(x)*(C1 + C2*log(x)))

    Note that because this method does not involve integration, there is no
    ``nth_linear_euler_eq_homogeneous_Integral`` hint.

    The following is for internal use:

    - ``returns = 'sol'`` returns the solution to the ODE.
    - ``returns = 'list'`` returns a list of linearly independent solutions,
      corresponding to the fundamental solution set, for use with non
      homogeneous solution methods like variation of parameters and
      undetermined coefficients.  Note that, though the solutions should be
      linearly independent, this function does not explicitly check that.  You
      can do ``assert simplify(wronskian(sollist)) != 0`` to check for linear
      independence.  Also, ``assert len(sollist) == order`` will need to pass.
    - ``returns = 'both'``, return a dictionary ``{'sol': <solution to ODE>,
      'list': <list of linearly independent solutions>}``.

    Examples
    ========

    >>> from sympy import Function, dsolve, pprint
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> eq = f(x).diff(x, 2)*x**2 - 4*f(x).diff(x)*x + 6*f(x)
    >>> pprint(dsolve(eq, f(x),
    ... hint='nth_linear_euler_eq_homogeneous'))
            2
    f(x) = x *(C1 + C2*x)

    References
    ==========

    - https://en.wikipedia.org/wiki/Cauchy%E2%80%93Euler_equation
    - C. Bender & S. Orszag, "Advanced Mathematical Methods for Scientists and
      Engineers", Springer 1999, pp. 12

    # indirect doctest

    """
    hint = "nth_linear_euler_eq_homogeneous"
    has_integral = False

    def _matches(self):
        eq = self.ode_problem.eq_preprocessed
        f = self.ode_problem.func.func
        order = self.ode_problem.order
        x = self.ode_problem.sym
        match = self.ode_problem.get_linear_coefficients(eq, f(x), order)
        self.r = None
        does_match = False

        if order and match:
            coeff = match[order]
            factor = x**order / coeff
            self.r = {i: factor*match[i] for i in match}
        if self.r and all(_test_term(self.r[i], f(x), i) for i in
                          self.r if i >= 0):
            if not self.r[-1]:
                does_match = True
        return does_match

    def _get_general_solution(self, *, simplify_flag: bool = True):
        fx = self.ode_problem.func
        eq = self.ode_problem.eq
        homogen_sol = _get_euler_characteristic_eq_sols(eq, fx, self.r)[0]
        return [homogen_sol]


class NthLinearEulerEqNonhomogeneousVariationOfParameters(SingleODESolver):
    r"""
    Solves an `n`\th order linear non homogeneous Cauchy-Euler equidimensional
    ordinary differential equation using variation of parameters.

    This is an equation with form `g(x) = a_0 f(x) + a_1 x f'(x) + a_2 x^2 f''(x)
    \cdots`.

    This method works by assuming that the particular solution takes the form

    .. math:: \sum_{x=1}^{n} c_i(x) y_i(x) {a_n} {x^n} \text{, }

    where `y_i` is the `i`\th solution to the homogeneous equation.  The
    solution is then solved using Wronskian's and Cramer's Rule.  The
    particular solution is given by multiplying eq given below with `a_n x^{n}`

    .. math:: \sum_{x=1}^n \left( \int \frac{W_i(x)}{W(x)} \, dx
                \right) y_i(x) \text{, }

    where `W(x)` is the Wronskian of the fundamental system (the system of `n`
    linearly independent solutions to the homogeneous equation), and `W_i(x)`
    is the Wronskian of the fundamental system with the `i`\th column replaced
    with `[0, 0, \cdots, 0, \frac{x^{- n}}{a_n} g{\left(x \right)}]`.

    This method is general enough to solve any `n`\th order inhomogeneous
    linear differential equation, but sometimes SymPy cannot simplify the
    Wronskian well enough to integrate it.  If this method hangs, try using the
    ``nth_linear_constant_coeff_variation_of_parameters_Integral`` hint and
    simplifying the integrals manually.  Also, prefer using
    ``nth_linear_constant_coeff_undetermined_coefficients`` when it
    applies, because it does not use integration, making it faster and more
    reliable.

    Warning, using simplify=False with
    'nth_linear_constant_coeff_variation_of_parameters' in
    :py:meth:`~sympy.solvers.ode.dsolve` may cause it to hang, because it will
    not attempt to simplify the Wronskian before integrating.  It is
    recommended that you only use simplify=False with
    'nth_linear_constant_coeff_variation_of_parameters_Integral' for this
    method, especially if the solution to the homogeneous equation has
    trigonometric functions in it.

    Examples
    ========

    >>> from sympy import Function, dsolve, Derivative
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> eq = x**2*Derivative(f(x), x, x) - 2*x*Derivative(f(x), x) + 2*f(x) - x**4
    >>> dsolve(eq, f(x),
    ... hint='nth_linear_euler_eq_nonhomogeneous_variation_of_parameters').expand()
    Eq(f(x), C1*x + C2*x**2 + x**4/6)

    """
    hint = "nth_linear_euler_eq_nonhomogeneous_variation_of_parameters"
    has_integral = True

    def _matches(self):
        eq = self.ode_problem.eq_preprocessed
        f = self.ode_problem.func.func
        order = self.ode_problem.order
        x = self.ode_problem.sym
        match = self.ode_problem.get_linear_coefficients(eq, f(x), order)
        self.r = None
        does_match = False

        if order and match:
            coeff = match[order]
            factor = x**order / coeff
            self.r = {i: factor*match[i] for i in match}
        if self.r and all(_test_term(self.r[i], f(x), i) for i in
                          self.r if i >= 0):
            if self.r[-1]:
                does_match = True

        return does_match

    def _get_general_solution(self, *, simplify_flag: bool = True):
        eq = self.ode_problem.eq
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        order = self.ode_problem.order
        homogen_sol, roots = _get_euler_characteristic_eq_sols(eq, f(x), self.r)
        self.r[-1] = self.r[-1]/self.r[order]
        sol = _solve_variation_of_parameters(eq, f(x), roots, homogen_sol, order, self.r, simplify_flag)

        return [Eq(f(x), homogen_sol.rhs + (sol.rhs - homogen_sol.rhs)*self.r[order])]


class NthLinearEulerEqNonhomogeneousUndeterminedCoefficients(SingleODESolver):
    r"""
    Solves an `n`\th order linear non homogeneous Cauchy-Euler equidimensional
    ordinary differential equation using undetermined coefficients.

    This is an equation with form `g(x) = a_0 f(x) + a_1 x f'(x) + a_2 x^2 f''(x)
    \cdots`.

    These equations can be solved in a general manner, by substituting
    solutions of the form `x = exp(t)`, and deriving a characteristic equation
    of form `g(exp(t)) = b_0 f(t) + b_1 f'(t) + b_2 f''(t) \cdots` which can
    be then solved by nth_linear_constant_coeff_undetermined_coefficients if
    g(exp(t)) has finite number of linearly independent derivatives.

    Functions that fit this requirement are finite sums functions of the form
    `a x^i e^{b x} \sin(c x + d)` or `a x^i e^{b x} \cos(c x + d)`, where `i`
    is a non-negative integer and `a`, `b`, `c`, and `d` are constants.  For
    example any polynomial in `x`, functions like `x^2 e^{2 x}`, `x \sin(x)`,
    and `e^x \cos(x)` can all be used.  Products of `\sin`'s and `\cos`'s have
    a finite number of derivatives, because they can be expanded into `\sin(a
    x)` and `\cos(b x)` terms.  However, SymPy currently cannot do that
    expansion, so you will need to manually rewrite the expression in terms of
    the above to use this method.  So, for example, you will need to manually
    convert `\sin^2(x)` into `(1 + \cos(2 x))/2` to properly apply the method
    of undetermined coefficients on it.

    After replacement of x by exp(t), this method works by creating a trial function
    from the expression and all of its linear independent derivatives and
    substituting them into the original ODE.  The coefficients for each term
    will be a system of linear equations, which are be solved for and
    substituted, giving the solution. If any of the trial functions are linearly
    dependent on the solution to the homogeneous equation, they are multiplied
    by sufficient `x` to make them linearly independent.

    Examples
    ========

    >>> from sympy import dsolve, Function, Derivative, log
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> eq = x**2*Derivative(f(x), x, x) - 2*x*Derivative(f(x), x) + 2*f(x) - log(x)
    >>> dsolve(eq, f(x),
    ... hint='nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients').expand()
    Eq(f(x), C1*x + C2*x**2 + log(x)/2 + 3/4)

    """
    hint = "nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients"
    has_integral = False

    def _matches(self):
        eq = self.ode_problem.eq_high_order_free
        f = self.ode_problem.func.func
        order = self.ode_problem.order
        x = self.ode_problem.sym
        match = self.ode_problem.get_linear_coefficients(eq, f(x), order)
        self.r = None
        does_match = False

        if order and match:
            coeff = match[order]
            factor = x**order / coeff
            self.r = {i: factor*match[i] for i in match}
        if self.r and all(_test_term(self.r[i], f(x), i) for i in
                          self.r if i >= 0):
            if self.r[-1]:
                e, re = posify(self.r[-1].subs(x, exp(x)))
                undetcoeff = _undetermined_coefficients_match(e.subs(re), x)
                if undetcoeff['test']:
                    does_match = True
        return does_match

    def _get_general_solution(self, *, simplify_flag: bool = True):
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        chareq, eq, symbol = S.Zero, S.Zero, Dummy('x')
        for i in self.r.keys():
            if i >= 0:
                chareq += (self.r[i]*diff(x**symbol, x, i)*x**-symbol).expand()

        for i in range(1, degree(Poly(chareq, symbol))+1):
            eq += chareq.coeff(symbol**i)*diff(f(x), x, i)

        if chareq.as_coeff_add(symbol)[0]:
            eq += chareq.as_coeff_add(symbol)[0]*f(x)
        e, re = posify(self.r[-1].subs(x, exp(x)))
        eq += e.subs(re)

        self.const_undet_instance = NthLinearConstantCoeffUndeterminedCoefficients(SingleODEProblem(eq, f(x), x))
        sol = self.const_undet_instance.get_general_solution(simplify = simplify_flag)[0]
        sol = sol.subs(x, log(x)) # type: ignore
        sol = sol.subs(f(log(x)), f(x)).expand() # type: ignore

        return [sol]


class SecondLinearBessel(SingleODESolver):
    r"""
    Gives solution of the Bessel differential equation

    .. math :: x^2 \frac{d^2y}{dx^2} + x \frac{dy}{dx} y(x) + (x^2-n^2) y(x)

    if `n` is integer then the solution is of the form ``Eq(f(x), C0 besselj(n,x)
    + C1 bessely(n,x))`` as both the solutions are linearly independent else if
    `n` is a fraction then the solution is of the form ``Eq(f(x), C0 besselj(n,x)
    + C1 besselj(-n,x))`` which can also transform into ``Eq(f(x), C0 besselj(n,x)
    + C1 bessely(n,x))``.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy import Symbol
    >>> v = Symbol('v', positive=True)
    >>> from sympy import dsolve, Function
    >>> f = Function('f')
    >>> y = f(x)
    >>> genform = x**2*y.diff(x, 2) + x*y.diff(x) + (x**2 - v**2)*y
    >>> dsolve(genform)
    Eq(f(x), C1*besselj(v, x) + C2*bessely(v, x))

    References
    ==========

    https://math24.net/bessel-differential-equation.html

    """
    hint = "2nd_linear_bessel"
    has_integral = False

    def _matches(self):
        eq = self.ode_problem.eq_high_order_free
        f = self.ode_problem.func
        order = self.ode_problem.order
        x = self.ode_problem.sym
        df = f.diff(x)
        a = Wild('a', exclude=[f,df])
        b = Wild('b', exclude=[x, f,df])
        a4 = Wild('a4', exclude=[x,f,df])
        b4 = Wild('b4', exclude=[x,f,df])
        c4 = Wild('c4', exclude=[x,f,df])
        d4 = Wild('d4', exclude=[x,f,df])
        a3 = Wild('a3', exclude=[f, df, f.diff(x, 2)])
        b3 = Wild('b3', exclude=[f, df, f.diff(x, 2)])
        c3 = Wild('c3', exclude=[f, df, f.diff(x, 2)])
        deq = a3*(f.diff(x, 2)) + b3*df + c3*f
        r = collect(eq,
            [f.diff(x, 2), df, f]).match(deq)
        if order == 2 and r:
            if not all(r[key].is_polynomial() for key in r):
                n, d = eq.as_numer_denom()
                eq = expand(n)
                r = collect(eq,
                    [f.diff(x, 2), df, f]).match(deq)

        if r and r[a3] != 0:
            # leading coeff of f(x).diff(x, 2)
            coeff = factor(r[a3]).match(a4*(x-b)**b4)

            if coeff:
            # if coeff[b4] = 0 means constant coefficient
                if coeff[b4] == 0:
                    return False
                point = coeff[b]
            else:
                return False

            if point:
                r[a3] = simplify(r[a3].subs(x, x+point))
                r[b3] = simplify(r[b3].subs(x, x+point))
                r[c3] = simplify(r[c3].subs(x, x+point))

            # making a3 in the form of x**2
            r[a3] = cancel(r[a3]/(coeff[a4]*(x)**(-2+coeff[b4])))
            r[b3] = cancel(r[b3]/(coeff[a4]*(x)**(-2+coeff[b4])))
            r[c3] = cancel(r[c3]/(coeff[a4]*(x)**(-2+coeff[b4])))
            # checking if b3 is of form c*(x-b)
            coeff1 = factor(r[b3]).match(a4*(x))
            if coeff1 is None:
                return False
            # c3 maybe of very complex form so I am simply checking (a - b) form
            # if yes later I will match with the standard form of bessel in a and b
            # a, b are wild variable defined above.
            _coeff2 = expand(r[c3]).match(a - b)
            if _coeff2 is None:
                return False
            # matching with standard form for c3
            coeff2 = factor(_coeff2[a]).match(c4**2*(x)**(2*a4))
            if coeff2 is None:
                return False

            if _coeff2[b] == 0:
                coeff2[d4] = 0
            else:
                coeff2[d4] = factor(_coeff2[b]).match(d4**2)[d4]

            self.rn = {'n':coeff2[d4], 'a4':coeff2[c4], 'd4':coeff2[a4]}
            self.rn['c4'] = coeff1[a4]
            self.rn['b4'] = point
            return True
        return False

    def _get_general_solution(self, *, simplify_flag: bool = True):
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        n = self.rn['n']
        a4 = self.rn['a4']
        c4 = self.rn['c4']
        d4 = self.rn['d4']
        b4 = self.rn['b4']
        n = sqrt(n**2 + Rational(1, 4)*(c4 - 1)**2)
        (C1, C2) = self.ode_problem.get_numbered_constants(num=2)
        return [Eq(f(x), ((x**(Rational(1-c4,2)))*(C1*besselj(n/d4,a4*x**d4/d4)
            + C2*bessely(n/d4,a4*x**d4/d4))).subs(x, x-b4))]


class SecondLinearAiry(SingleODESolver):
    r"""
    Gives solution of the Airy differential equation

    .. math :: \frac{d^2y}{dx^2} + (a + b x) y(x) = 0

    in terms of Airy special functions airyai and airybi.

    Examples
    ========

    >>> from sympy import dsolve, Function
    >>> from sympy.abc import x
    >>> f = Function("f")
    >>> eq = f(x).diff(x, 2) - x*f(x)
    >>> dsolve(eq)
    Eq(f(x), C1*airyai(x) + C2*airybi(x))
    """
    hint = "2nd_linear_airy"
    has_integral = False

    def _matches(self):
        eq = self.ode_problem.eq_high_order_free
        f = self.ode_problem.func
        order = self.ode_problem.order
        x = self.ode_problem.sym
        df = f.diff(x)
        a4 = Wild('a4', exclude=[x,f,df])
        b4 = Wild('b4', exclude=[x,f,df])
        match = self.ode_problem.get_linear_coefficients(eq, f, order)
        does_match = False
        if order == 2 and match and match[2] != 0:
            if match[1].is_zero:
                self.rn = cancel(match[0]/match[2]).match(a4+b4*x)
                if self.rn and self.rn[b4] != 0:
                    self.rn = {'b':self.rn[a4],'m':self.rn[b4]}
                    does_match = True
        return does_match

    def _get_general_solution(self, *, simplify_flag: bool = True):
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        (C1, C2) = self.ode_problem.get_numbered_constants(num=2)
        b = self.rn['b']
        m = self.rn['m']
        if m.is_positive:
            arg = - b/cbrt(m)**2 - cbrt(m)*x
        elif m.is_negative:
            arg = - b/cbrt(-m)**2 + cbrt(-m)*x
        else:
            arg = - b/cbrt(-m)**2 + cbrt(-m)*x

        return [Eq(f(x), C1*airyai(arg) + C2*airybi(arg))]


class LieGroup(SingleODESolver):
    r"""
    This hint implements the Lie group method of solving first order differential
    equations. The aim is to convert the given differential equation from the
    given coordinate system into another coordinate system where it becomes
    invariant under the one-parameter Lie group of translations. The converted
    ODE can be easily solved by quadrature. It makes use of the
    :py:meth:`sympy.solvers.ode.infinitesimals` function which returns the
    infinitesimals of the transformation.

    The coordinates `r` and `s` can be found by solving the following Partial
    Differential Equations.

    .. math :: \xi\frac{\partial r}{\partial x} + \eta\frac{\partial r}{\partial y}
                  = 0

    .. math :: \xi\frac{\partial s}{\partial x} + \eta\frac{\partial s}{\partial y}
                  = 1

    The differential equation becomes separable in the new coordinate system

    .. math :: \frac{ds}{dr} = \frac{\frac{\partial s}{\partial x} +
                 h(x, y)\frac{\partial s}{\partial y}}{
                 \frac{\partial r}{\partial x} + h(x, y)\frac{\partial r}{\partial y}}

    After finding the solution by integration, it is then converted back to the original
    coordinate system by substituting `r` and `s` in terms of `x` and `y` again.

    Examples
    ========

    >>> from sympy import Function, dsolve, exp, pprint
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(f(x).diff(x) + 2*x*f(x) - x*exp(-x**2), f(x),
    ... hint='lie_group'))
           /      2\    2
           |     x |  -x
    f(x) = |C1 + --|*e
           \     2 /


    References
    ==========

    - Solving differential equations by Symmetry Groups,
      John Starrett, pp. 1 - pp. 14

    """
    hint = "lie_group"
    has_integral = False

    def _has_additional_params(self):
        return 'xi' in self.ode_problem.params and 'eta' in self.ode_problem.params

    def _matches(self):
        eq = self.ode_problem.eq
        f = self.ode_problem.func.func
        order = self.ode_problem.order
        x = self.ode_problem.sym
        df = f(x).diff(x)
        y = Dummy('y')
        d = Wild('d', exclude=[df, f(x).diff(x, 2)])
        e = Wild('e', exclude=[df])
        does_match = False
        if self._has_additional_params() and order == 1:
            xi = self.ode_problem.params['xi']
            eta = self.ode_problem.params['eta']
            self.r3 = {'xi': xi, 'eta': eta}
            r = collect(eq, df, exact=True).match(d + e * df)
            if r:
                r['d'] = d
                r['e'] = e
                r['y'] = y
                r[d] = r[d].subs(f(x), y)
                r[e] = r[e].subs(f(x), y)
                self.r3.update(r)
            does_match = True
        return does_match

    def _get_general_solution(self, *, simplify_flag: bool = True):
        eq = self.ode_problem.eq
        x = self.ode_problem.sym
        func = self.ode_problem.func
        order = self.ode_problem.order
        df = func.diff(x)

        try:
            eqsol = solve(eq, df)
        except NotImplementedError:
            eqsol = []

        desols = []
        for s in eqsol:
            sol = _ode_lie_group(s, func, order, match=self.r3)
            if sol:
                desols.extend(sol)

        if desols == []:
            raise NotImplementedError("The given ODE " + str(eq) + " cannot be solved by"
                + " the lie group method")
        return desols


solver_map = {
    'factorable': Factorable,
    'nth_linear_constant_coeff_homogeneous': NthLinearConstantCoeffHomogeneous,
    'nth_linear_euler_eq_homogeneous': NthLinearEulerEqHomogeneous,
    'nth_linear_constant_coeff_undetermined_coefficients': NthLinearConstantCoeffUndeterminedCoefficients,
    'nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients': NthLinearEulerEqNonhomogeneousUndeterminedCoefficients,
    'separable': Separable,
    '1st_exact': FirstExact,
    '1st_linear': FirstLinear,
    'Bernoulli': Bernoulli,
    'Riccati_special_minus2': RiccatiSpecial,
    '1st_rational_riccati': RationalRiccati,
    '1st_homogeneous_coeff_best': HomogeneousCoeffBest,
    '1st_homogeneous_coeff_subs_indep_div_dep': HomogeneousCoeffSubsIndepDivDep,
    '1st_homogeneous_coeff_subs_dep_div_indep': HomogeneousCoeffSubsDepDivIndep,
    'almost_linear': AlmostLinear,
    'linear_coefficients': LinearCoefficients,
    'separable_reduced': SeparableReduced,
    'nth_linear_constant_coeff_variation_of_parameters': NthLinearConstantCoeffVariationOfParameters,
    'nth_linear_euler_eq_nonhomogeneous_variation_of_parameters': NthLinearEulerEqNonhomogeneousVariationOfParameters,
    'Liouville': Liouville,
    '2nd_linear_airy': SecondLinearAiry,
    '2nd_linear_bessel': SecondLinearBessel,
    '2nd_hypergeometric': SecondHypergeometric,
    'nth_order_reducible': NthOrderReducible,
    '2nd_nonlinear_autonomous_conserved': SecondNonlinearAutonomousConserved,
    'nth_algebraic': NthAlgebraic,
    'lie_group': LieGroup,
    }

# Avoid circular import:
from .ode import dsolve, ode_sol_simplicity, odesimp, homogeneous_order
