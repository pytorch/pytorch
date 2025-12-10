"""Solvers of systems of polynomial equations. """

from __future__ import annotations

from typing import Any
from collections.abc import Sequence, Iterable

import itertools

from sympy import Dummy
from sympy.core import S
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.sorting import default_sort_key
from sympy.logic.boolalg import Boolean
from sympy.polys import Poly, groebner, roots
from sympy.polys.domains import ZZ
from sympy.polys.polyoptions import build_options
from sympy.polys.polytools import parallel_poly_from_expr, sqf_part
from sympy.polys.polyerrors import (
    ComputationFailed,
    PolificationFailed,
    CoercionFailed,
    GeneratorsNeeded,
    DomainError
)
from sympy.simplify import rcollect
from sympy.utilities import postfixes
from sympy.utilities.iterables import cartes
from sympy.utilities.misc import filldedent
from sympy.logic.boolalg import Or, And
from sympy.core.relational import Eq


class SolveFailed(Exception):
    """Raised when solver's conditions were not met. """


def solve_poly_system(seq, *gens, strict=False, **args):
    """
    Return a list of solutions for the system of polynomial equations
    or else None.

    Parameters
    ==========

    seq: a list/tuple/set
        Listing all the equations that are needed to be solved
    gens: generators
        generators of the equations in seq for which we want the
        solutions
    strict: a boolean (default is False)
        if strict is True, NotImplementedError will be raised if
        the solution is known to be incomplete (which can occur if
        not all solutions are expressible in radicals)
    args: Keyword arguments
        Special options for solving the equations.


    Returns
    =======

    List[Tuple]
        a list of tuples with elements being solutions for the
        symbols in the order they were passed as gens
    None
        None is returned when the computed basis contains only the ground.

    Examples
    ========

    >>> from sympy import solve_poly_system
    >>> from sympy.abc import x, y

    >>> solve_poly_system([x*y - 2*y, 2*y**2 - x**2], x, y)
    [(0, 0), (2, -sqrt(2)), (2, sqrt(2))]

    >>> solve_poly_system([x**5 - x + y**3, y**2 - 1], x, y, strict=True)
    Traceback (most recent call last):
    ...
    UnsolvableFactorError

    """
    try:
        polys, opt = parallel_poly_from_expr(seq, *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('solve_poly_system', len(seq), exc)

    if len(polys) == len(opt.gens) == 2:
        f, g = polys

        if all(i <= 2 for i in f.degree_list() + g.degree_list()):
            try:
                return solve_biquadratic(f, g, opt)
            except SolveFailed:
                pass

    return solve_generic(polys, opt, strict=strict)


def solve_biquadratic(f, g, opt):
    """Solve a system of two bivariate quadratic polynomial equations.

    Parameters
    ==========

    f: a single Expr or Poly
        First equation
    g: a single Expr or Poly
        Second Equation
    opt: an Options object
        For specifying keyword arguments and generators

    Returns
    =======

    List[Tuple]
        a list of tuples with elements being solutions for the
        symbols in the order they were passed as gens
    None
        None is returned when the computed basis contains only the ground.

    Examples
    ========

    >>> from sympy import Options, Poly
    >>> from sympy.abc import x, y
    >>> from sympy.solvers.polysys import solve_biquadratic
    >>> NewOption = Options((x, y), {'domain': 'ZZ'})

    >>> a = Poly(y**2 - 4 + x, y, x, domain='ZZ')
    >>> b = Poly(y*2 + 3*x - 7, y, x, domain='ZZ')
    >>> solve_biquadratic(a, b, NewOption)
    [(1/3, 3), (41/27, 11/9)]

    >>> a = Poly(y + x**2 - 3, y, x, domain='ZZ')
    >>> b = Poly(-y + x - 4, y, x, domain='ZZ')
    >>> solve_biquadratic(a, b, NewOption)
    [(7/2 - sqrt(29)/2, -sqrt(29)/2 - 1/2), (sqrt(29)/2 + 7/2, -1/2 + \
      sqrt(29)/2)]
    """
    G = groebner([f, g])

    if len(G) == 1 and G[0].is_ground:
        return None

    if len(G) != 2:
        raise SolveFailed

    x, y = opt.gens
    p, q = G
    if not p.gcd(q).is_ground:
        # not 0-dimensional
        raise SolveFailed

    p = Poly(p, x, expand=False)
    p_roots = [rcollect(expr, y) for expr in roots(p).keys()]

    q = q.ltrim(-1)
    q_roots = list(roots(q).keys())

    solutions = [(p_root.subs(y, q_root), q_root) for q_root, p_root in
                 itertools.product(q_roots, p_roots)]

    return sorted(solutions, key=default_sort_key)


def solve_generic(polys, opt, strict=False):
    """
    Solve a generic system of polynomial equations.

    Returns all possible solutions over C[x_1, x_2, ..., x_m] of a
    set F = { f_1, f_2, ..., f_n } of polynomial equations, using
    Groebner basis approach. For now only zero-dimensional systems
    are supported, which means F can have at most a finite number
    of solutions. If the basis contains only the ground, None is
    returned.

    The algorithm works by the fact that, supposing G is the basis
    of F with respect to an elimination order (here lexicographic
    order is used), G and F generate the same ideal, they have the
    same set of solutions. By the elimination property, if G is a
    reduced, zero-dimensional Groebner basis, then there exists an
    univariate polynomial in G (in its last variable). This can be
    solved by computing its roots. Substituting all computed roots
    for the last (eliminated) variable in other elements of G, new
    polynomial system is generated. Applying the above procedure
    recursively, a finite number of solutions can be found.

    The ability of finding all solutions by this procedure depends
    on the root finding algorithms. If no solutions were found, it
    means only that roots() failed, but the system is solvable. To
    overcome this difficulty use numerical algorithms instead.

    Parameters
    ==========

    polys: a list/tuple/set
        Listing all the polynomial equations that are needed to be solved
    opt: an Options object
        For specifying keyword arguments and generators
    strict: a boolean
        If strict is True, NotImplementedError will be raised if the solution
        is known to be incomplete

    Returns
    =======

    List[Tuple]
        a list of tuples with elements being solutions for the
        symbols in the order they were passed as gens
    None
        None is returned when the computed basis contains only the ground.

    References
    ==========

    .. [Buchberger01] B. Buchberger, Groebner Bases: A Short
    Introduction for Systems Theorists, In: R. Moreno-Diaz,
    B. Buchberger, J.L. Freire, Proceedings of EUROCAST'01,
    February, 2001

    .. [Cox97] D. Cox, J. Little, D. O'Shea, Ideals, Varieties
    and Algorithms, Springer, Second Edition, 1997, pp. 112

    Raises
    ========

    NotImplementedError
        If the system is not zero-dimensional (does not have a finite
        number of solutions)

    UnsolvableFactorError
        If ``strict`` is True and not all solution components are
        expressible in radicals

    Examples
    ========

    >>> from sympy import Poly, Options
    >>> from sympy.solvers.polysys import solve_generic
    >>> from sympy.abc import x, y
    >>> NewOption = Options((x, y), {'domain': 'ZZ'})

    >>> a = Poly(x - y + 5, x, y, domain='ZZ')
    >>> b = Poly(x + y - 3, x, y, domain='ZZ')
    >>> solve_generic([a, b], NewOption)
    [(-1, 4)]

    >>> a = Poly(x - 2*y + 5, x, y, domain='ZZ')
    >>> b = Poly(2*x - y - 3, x, y, domain='ZZ')
    >>> solve_generic([a, b], NewOption)
    [(11/3, 13/3)]

    >>> a = Poly(x**2 + y, x, y, domain='ZZ')
    >>> b = Poly(x + y*4, x, y, domain='ZZ')
    >>> solve_generic([a, b], NewOption)
    [(0, 0), (1/4, -1/16)]

    >>> a = Poly(x**5 - x + y**3, x, y, domain='ZZ')
    >>> b = Poly(y**2 - 1, x, y, domain='ZZ')
    >>> solve_generic([a, b], NewOption, strict=True)
    Traceback (most recent call last):
    ...
    UnsolvableFactorError

    """
    def _is_univariate(f):
        """Returns True if 'f' is univariate in its last variable. """
        for monom in f.monoms():
            if any(monom[:-1]):
                return False

        return True

    def _subs_root(f, gen, zero):
        """Replace generator with a root so that the result is nice. """
        p = f.as_expr({gen: zero})

        if f.degree(gen) >= 2:
            p = p.expand(deep=False)

        return p

    def _solve_reduced_system(system, gens, entry=False):
        """Recursively solves reduced polynomial systems. """
        if len(system) == len(gens) == 1:
            # the below line will produce UnsolvableFactorError if
            # strict=True and the solution from `roots` is incomplete
            zeros = list(roots(system[0], gens[-1], strict=strict).keys())
            return [(zero,) for zero in zeros]

        basis = groebner(system, gens, polys=True)

        if len(basis) == 1 and basis[0].is_ground:
            if not entry:
                return []
            else:
                return None

        univariate = list(filter(_is_univariate, basis))

        if len(basis) < len(gens):
            raise NotImplementedError(filldedent('''
                only zero-dimensional systems supported
                (finite number of solutions)
                '''))

        if len(univariate) == 1:
            f = univariate.pop()
        else:
            raise NotImplementedError(filldedent('''
                only zero-dimensional systems supported
                (finite number of solutions)
                '''))

        gens = f.gens
        gen = gens[-1]

        # the below line will produce UnsolvableFactorError if
        # strict=True and the solution from `roots` is incomplete
        zeros = list(roots(f.ltrim(gen), strict=strict).keys())

        if not zeros:
            return []

        if len(basis) == 1:
            return [(zero,) for zero in zeros]

        solutions = []

        for zero in zeros:
            new_system = []
            new_gens = gens[:-1]

            for b in basis[:-1]:
                eq = _subs_root(b, gen, zero)

                if eq is not S.Zero:
                    new_system.append(eq)

            for solution in _solve_reduced_system(new_system, new_gens):
                solutions.append(solution + (zero,))

        if solutions and len(solutions[0]) != len(gens):
            raise NotImplementedError(filldedent('''
                only zero-dimensional systems supported
                (finite number of solutions)
                '''))
        return solutions

    try:
        result = _solve_reduced_system(polys, opt.gens, entry=True)
    except CoercionFailed:
        raise NotImplementedError

    if result is not None:
        return sorted(result, key=default_sort_key)


def solve_triangulated(polys, *gens, **args):
    """
    Solve a polynomial system using Gianni-Kalkbrenner algorithm.

    The algorithm proceeds by computing one Groebner basis in the ground
    domain and then by iteratively computing polynomial factorizations in
    appropriately constructed algebraic extensions of the ground domain.

    Parameters
    ==========

    polys: a list/tuple/set
        Listing all the equations that are needed to be solved
    gens: generators
        generators of the equations in polys for which we want the
        solutions
    args: Keyword arguments
        Special options for solving the equations

    Returns
    =======

    List[Tuple]
        A List of tuples. Solutions for symbols that satisfy the
        equations listed in polys

    Examples
    ========

    >>> from sympy import solve_triangulated
    >>> from sympy.abc import x, y, z

    >>> F = [x**2 + y + z - 1, x + y**2 + z - 1, x + y + z**2 - 1]

    >>> solve_triangulated(F, x, y, z)
    [(0, 0, 1), (0, 1, 0), (1, 0, 0)]

    Using extension for algebraic solutions.

    >>> solve_triangulated(F, x, y, z, extension=True) #doctest: +NORMALIZE_WHITESPACE
    [(0, 0, 1), (0, 1, 0), (1, 0, 0),
     (CRootOf(x**2 + 2*x - 1, 0), CRootOf(x**2 + 2*x - 1, 0), CRootOf(x**2 + 2*x - 1, 0)),
     (CRootOf(x**2 + 2*x - 1, 1), CRootOf(x**2 + 2*x - 1, 1), CRootOf(x**2 + 2*x - 1, 1))]

    References
    ==========

    1. Patrizia Gianni, Teo Mora, Algebraic Solution of System of
    Polynomial Equations using Groebner Bases, AAECC-5 on Applied Algebra,
    Algebraic Algorithms and Error-Correcting Codes, LNCS 356 247--257, 1989

    """
    opt = build_options(gens, args)

    G = groebner(polys, gens, polys=True)
    G = list(reversed(G))

    extension = opt.get('extension', False)
    if extension:
        def _solve_univariate(f):
            return [r for r, _ in f.all_roots(multiple=False, radicals=False)]
    else:
        domain = opt.get('domain')

        if domain is not None:
            for i, g in enumerate(G):
                G[i] = g.set_domain(domain)

        def _solve_univariate(f):
            return list(f.ground_roots().keys())

    f, G = G[0].ltrim(-1), G[1:]
    dom = f.get_domain()

    zeros = _solve_univariate(f)

    if extension:
        solutions = {((zero,), dom.algebraic_field(zero)) for zero in zeros}
    else:
        solutions = {((zero,), dom) for zero in zeros}

    var_seq = reversed(gens[:-1])
    vars_seq = postfixes(gens[1:])

    for var, vars in zip(var_seq, vars_seq):
        _solutions = set()

        for values, dom in solutions:
            H, mapping = [], list(zip(vars, values))

            for g in G:
                _vars = (var,) + vars

                if g.has_only_gens(*_vars) and g.degree(var) != 0:
                    if extension:
                        g = g.set_domain(g.domain.unify(dom))
                    h = g.ltrim(var).eval(dict(mapping))

                    if g.degree(var) == h.degree():
                        H.append(h)

            p = min(H, key=lambda h: h.degree())
            zeros = _solve_univariate(p)

            for zero in zeros:
                if not (zero in dom):
                    dom_zero = dom.algebraic_field(zero)
                else:
                    dom_zero = dom

                _solutions.add(((zero,) + values, dom_zero))

        solutions = _solutions
    return sorted((s for s, _ in solutions), key=default_sort_key)


def factor_system(eqs: Sequence[Expr | complex], gens: Sequence[Expr] = (), **kwargs: Any) -> list[list[Expr]]:
    """
    Factorizes a system of polynomial equations into
    irreducible subsystems.

    Parameters
    ==========

    eqs : list
        List of expressions to be factored.
        Each expression is assumed to be equal to zero.

    gens : list, optional
        Generator(s) of the polynomial ring.
        If not provided, all free symbols will be used.

    **kwargs : dict, optional
        Same optional arguments taken by ``factor``

    Returns
    =======

    list[list[Expr]]
        A list of lists of expressions, where each sublist represents
        an irreducible subsystem. When solved, each subsystem gives
        one component of the solution. Only generic solutions are
        returned (cases not requiring parameters to be zero).

    Examples
    ========

    >>> from sympy.solvers.polysys import factor_system, factor_system_cond
    >>> from sympy.abc import x, y, a, b, c

    A simple system with multiple solutions:

    >>> factor_system([x**2 - 1, y - 1])
    [[x + 1, y - 1], [x - 1, y - 1]]

    A system with no solution:

    >>> factor_system([x, 1])
    []

    A system where any value of the symbol(s) is a solution:

    >>> factor_system([x - x, (x + 1)**2 - (x**2 + 2*x + 1)])
    [[]]

    A system with no generic solution:

    >>> factor_system([a*x*(x-1), b*y, c], [x, y])
    []

    If c is added to the unknowns then the system has a generic solution:

    >>> factor_system([a*x*(x-1), b*y, c], [x, y, c])
    [[x - 1, y, c], [x, y, c]]

    Alternatively :func:`factor_system_cond` can be used to get degenerate
    cases as well:

    >>> factor_system_cond([a*x*(x-1), b*y, c], [x, y])
    [[x - 1, y, c], [x, y, c], [x - 1, b, c], [x, b, c], [y, a, c], [a, b, c]]

    Each of the above cases is only satisfiable in the degenerate case `c = 0`.

    The solution set of the original system represented
    by eqs is the union of the solution sets of the
    factorized systems.

    An empty list [] means no generic solution exists.
    A list containing an empty list [[]] means any value of
    the symbol(s) is a solution.

    See Also
    ========

    factor_system_cond : Returns both generic and degenerate solutions
    factor_system_bool : Returns a Boolean combination representing all solutions
    sympy.polys.polytools.factor : Factors a polynomial into irreducible factors
                                   over the rational numbers
    """

    systems = _factor_system_poly_from_expr(eqs, gens, **kwargs)
    systems_generic = [sys for sys in systems if not _is_degenerate(sys)]
    systems_expr = [[p.as_expr() for p in system] for system in systems_generic]
    return systems_expr


def _is_degenerate(system: list[Poly]) -> bool:
    """Helper function to check if a system is degenerate"""
    return any(p.is_ground for p in system)


def factor_system_bool(eqs: Sequence[Expr | complex], gens: Sequence[Expr] = (), **kwargs: Any) -> Boolean:
    """
    Factorizes a system of polynomial equations into irreducible DNF.

    The system of expressions(eqs) is taken and a Boolean combination
    of equations is returned that represents the same solution set.
    The result is in disjunctive normal form (OR of ANDs).

    Parameters
    ==========

    eqs : list
       List of expressions to be factored.
       Each expression is assumed to be equal to zero.

    gens : list, optional
       Generator(s) of the polynomial ring.
       If not provided, all free symbols will be used.

    **kwargs : dict, optional
       Optional keyword arguments


    Returns
    =======

    Boolean:
       A Boolean combination of equations. The result is typically in
       the form of a conjunction (AND) of a disjunctive normal form
       with additional conditions.

    Examples
    ========

    >>> from sympy.solvers.polysys import factor_system_bool
    >>> from sympy.abc import x, y, a, b, c
    >>> factor_system_bool([x**2 - 1])
    Eq(x - 1, 0) | Eq(x + 1, 0)

    >>> factor_system_bool([x**2 - 1, y - 1])
    (Eq(x - 1, 0) & Eq(y - 1, 0)) | (Eq(x + 1, 0) & Eq(y - 1, 0))

    >>> eqs = [a * (x - 1), b]
    >>> factor_system_bool([a*(x - 1), b])
    (Eq(a, 0) & Eq(b, 0)) | (Eq(b, 0) & Eq(x - 1, 0))

    >>> factor_system_bool([a*x**2 - a, b*(x + 1), c], [x])
    (Eq(c, 0) & Eq(x + 1, 0)) | (Eq(a, 0) & Eq(b, 0) & Eq(c, 0)) | (Eq(b, 0) & Eq(c, 0) & Eq(x - 1, 0))

    >>> factor_system_bool([x**2 + 2*x + 1 - (x + 1)**2])
    True

    The result is logically equivalent to the system of equations
    i.e. eqs. The function returns ``True`` when all values of
    the symbol(s) is a solution and ``False`` when the system
    cannot be solved.

    See Also
    ========

    factor_system : Returns factors and solvability condition separately
    factor_system_cond : Returns both factors and conditions

    """

    systems = factor_system_cond(eqs, gens, **kwargs)
    return Or(*[And(*[Eq(eq, 0) for eq in sys]) for sys in systems])


def factor_system_cond(eqs: Sequence[Expr | complex], gens: Sequence[Expr] = (), **kwargs: Any) -> list[list[Expr]]:
    """
    Factorizes a polynomial system into irreducible components and returns
    both generic and degenerate solutions.

    Parameters
    ==========

    eqs : list
        List of expressions to be factored.
        Each expression is assumed to be equal to zero.

    gens : list, optional
        Generator(s) of the polynomial ring.
        If not provided, all free symbols will be used.

    **kwargs : dict, optional
        Optional keyword arguments.

    Returns
    =======

    list[list[Expr]]
        A list of lists of expressions, where each sublist represents
        an irreducible subsystem. Includes both generic solutions and
        degenerate cases requiring equality conditions on parameters.

    Examples
    ========

    >>> from sympy.solvers.polysys import factor_system_cond
    >>> from sympy.abc import x, y, a, b, c

    >>> factor_system_cond([x**2 - 4, a*y, b], [x, y])
    [[x + 2, y, b], [x - 2, y, b], [x + 2, a, b], [x - 2, a, b]]

    >>> factor_system_cond([a*x*(x-1), b*y, c], [x, y])
    [[x - 1, y, c], [x, y, c], [x - 1, b, c], [x, b, c], [y, a, c], [a, b, c]]

    An empty list [] means no solution exists.
    A list containing an empty list [[]] means any value of
    the symbol(s) is a solution.

    See Also
    ========

    factor_system : Returns only generic solutions
    factor_system_bool : Returns a Boolean combination representing all solutions
    sympy.polys.polytools.factor : Factors a polynomial into irreducible factors
                                   over the rational numbers
    """
    systems_poly = _factor_system_poly_from_expr(eqs, gens, **kwargs)
    systems = [[p.as_expr() for p in system] for system in systems_poly]
    return systems


def _factor_system_poly_from_expr(
        eqs: Sequence[Expr | complex], gens: Sequence[Expr], **kwargs: Any
) -> list[list[Poly]]:
    """
    Convert expressions to polynomials and factor the system.

    Takes a sequence of expressions, converts them to
    polynomials, and factors the resulting system. Handles both regular
    polynomial systems and purely numerical cases.
    """
    try:
        polys, opts = parallel_poly_from_expr(eqs, *gens, **kwargs)
        only_numbers = False
    except (GeneratorsNeeded, PolificationFailed):
        _u = Dummy('u')
        polys, opts = parallel_poly_from_expr(eqs, [_u], **kwargs)
        assert opts['domain'].is_Numerical
        only_numbers = True

    if only_numbers:
        return [[]] if all(p == 0 for p in polys) else []

    return factor_system_poly(polys)


def factor_system_poly(polys: list[Poly]) -> list[list[Poly]]:
    """
    Factors a system of polynomial equations into irreducible subsystems

    Core implementation that works directly with Poly instances.

    Parameters
    ==========

    polys : list[Poly]
        A list of Poly instances to be factored.

    Returns
    =======

    list[list[Poly]]
        A list of lists of polynomials, where each sublist represents
        an irreducible component of the solution. Includes both
        generic and degenerate cases.

    Examples
    ========

    >>> from sympy import symbols, Poly, ZZ
    >>> from sympy.solvers.polysys import factor_system_poly
    >>> a, b, c, x = symbols('a b c x')
    >>> p1 = Poly((a - 1)*(x - 2), x, domain=ZZ[a,b,c])
    >>> p2 = Poly((b - 3)*(x - 2), x, domain=ZZ[a,b,c])
    >>> p3 = Poly(c, x, domain=ZZ[a,b,c])

    The equation to be solved for x is ``x - 2 = 0`` provided either
    of the two conditions on the parameters ``a`` and ``b`` is nonzero
    and the constant parameter ``c`` should be zero.

    >>> sys1, sys2 = factor_system_poly([p1, p2, p3])
    >>> sys1
    [Poly(x - 2, x, domain='ZZ[a,b,c]'),
     Poly(c, x, domain='ZZ[a,b,c]')]
    >>> sys2
    [Poly(a - 1, x, domain='ZZ[a,b,c]'),
     Poly(b - 3, x, domain='ZZ[a,b,c]'),
     Poly(c, x, domain='ZZ[a,b,c]')]

     An empty list [] when returned means no solution exists.
     Whereas a list containing an empty list [[]] means any value is a solution.

    See Also
    ========

    factor_system : Returns only generic solutions
    factor_system_bool : Returns a Boolean combination representing the solutions
    factor_system_cond : Returns both generic and degenerate solutions
    sympy.polys.polytools.factor : Factors a polynomial into irreducible factors
                                   over the rational numbers
    """
    if not all(isinstance(poly, Poly) for poly in polys):
        raise TypeError("polys should be a list of Poly instances")
    if not polys:
        return [[]]

    base_domain = polys[0].domain
    base_gens = polys[0].gens
    if not all(poly.domain == base_domain and poly.gens == base_gens for poly in polys[1:]):
        raise DomainError("All polynomials must have the same domain and generators")

    factor_sets = []
    for poly in polys:
        constant, factors_mult = poly.factor_list()

        if constant.is_zero is True:
            continue
        elif constant.is_zero is False:
            if not factors_mult:
                return []
            factor_sets.append([f for f, _ in factors_mult])
        else:
            constant = sqf_part(factor_terms(constant).as_coeff_Mul()[1])
            constp = Poly(constant, base_gens, domain=base_domain)
            factors = [f for f, _ in factors_mult]
            factors.append(constp)
            factor_sets.append(factors)

    if not factor_sets:
        return [[]]

    result = _factor_sets(factor_sets)
    return _sort_systems(result)


def _factor_sets_slow(eqs: list[list]) -> set[frozenset]:
    """
    Helper to find the minimal set of factorised subsystems that is
    equivalent to the original system.

    The result is in DNF.
    """
    if not eqs:
        return {frozenset()}
    systems_set = {frozenset(sys) for sys in cartes(*eqs)}
    return {s1 for s1 in systems_set if not any(s1 > s2 for s2 in systems_set)}


def _factor_sets(eqs: list[list]) -> set[frozenset]:
    """
    Helper that builds factor combinations.
    """
    if not eqs:
        return {frozenset()}

    current_set = min(eqs, key=len)
    other_sets = [s for s in eqs if s is not current_set]

    stack = [(factor, [s for s in other_sets if factor not in s], {factor})
             for factor in current_set]

    result = set()

    while stack:
        factor, remaining_sets, current_solution = stack.pop()

        if not remaining_sets:
            result.add(frozenset(current_solution))
            continue

        next_set = min(remaining_sets, key=len)
        next_remaining = [s for s in remaining_sets if s is not next_set]

        for next_factor in next_set:
            valid_remaining = [s for s in next_remaining if next_factor not in s]
            new_solution = current_solution | {next_factor}
            stack.append((next_factor, valid_remaining, new_solution))

    return {s1 for s1 in result if not any(s1 > s2 for s2 in result)}


def _sort_systems(systems: Iterable[Iterable[Poly]]) -> list[list[Poly]]:
    """Sorts a list of lists of polynomials"""
    systems_list = [sorted(s, key=_poly_sort_key, reverse=True) for s in systems]
    return sorted(systems_list, key=_sys_sort_key, reverse=True)


def _poly_sort_key(poly):
    """Sort key for polynomials"""
    if poly.domain.is_FF:
        poly = poly.set_domain(ZZ)
    return poly.degree_list(), poly.rep.to_list()


def _sys_sort_key(sys):
    """Sort key for lists of polynomials"""
    return list(zip(*map(_poly_sort_key, sys)))
