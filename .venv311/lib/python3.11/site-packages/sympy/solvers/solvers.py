"""
This module contain solvers for all kinds of equations:

    - algebraic or transcendental, use solve()

    - recurrence, use rsolve()

    - differential, use dsolve()

    - nonlinear (numerically), use nsolve()
      (you will need a good starting point)

"""
from __future__ import annotations

from sympy.core import (S, Add, Symbol, Dummy, Expr, Mul)
from sympy.core.assumptions import check_assumptions
from sympy.core.exprtools import factor_terms
from sympy.core.function import (expand_mul, expand_log, Derivative,
                                 AppliedUndef, UndefinedFunction, nfloat,
                                 Function, expand_power_exp, _mexpand, expand,
                                 expand_func)
from sympy.core.logic import fuzzy_not, fuzzy_and
from sympy.core.numbers import Float, Rational, _illegal
from sympy.core.intfunc import integer_log, ilcm
from sympy.core.power import Pow
from sympy.core.relational import Eq, Ne
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.sympify import sympify, _sympify
from sympy.core.traversal import preorder_traversal
from sympy.logic.boolalg import And, BooleanAtom

from sympy.functions import (log, exp, LambertW, cos, sin, tan, acos, asin, atan,
                             Abs, re, im, arg, sqrt, atan2)
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.piecewise import piecewise_fold, Piecewise
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.integrals.integrals import Integral
from sympy.ntheory.factor_ import divisors
from sympy.simplify import (simplify, collect, powsimp, posify,  # type: ignore
    powdenest, nsimplify, denom, logcombine, sqrtdenest, fraction,
    separatevars)
from sympy.simplify.sqrtdenest import sqrt_depth
from sympy.simplify.fu import TR1, TR2i, TR10, TR11
from sympy.strategies.rl import rebuild
from sympy.matrices.exceptions import NonInvertibleMatrixError
from sympy.matrices import Matrix, zeros
from sympy.polys import roots, cancel, factor, Poly
from sympy.polys.solvers import sympy_eqs_to_ring, solve_lin_sys
from sympy.polys.polyerrors import GeneratorsNeeded, PolynomialError
from sympy.polys.polytools import gcd
from sympy.utilities.lambdify import lambdify
from sympy.utilities.misc import filldedent, debugf
from sympy.utilities.iterables import (connected_components,
    generate_bell, uniq, iterable, is_sequence, subsets, flatten, sift)
from sympy.utilities.decorator import conserve_mpmath_dps

from mpmath import findroot

from sympy.solvers.polysys import solve_poly_system

from types import GeneratorType
from collections import defaultdict
from itertools import combinations, product

import warnings


def recast_to_symbols(eqs, symbols):
    """
    Return (e, s, d) where e and s are versions of *eqs* and
    *symbols* in which any non-Symbol objects in *symbols* have
    been replaced with generic Dummy symbols and d is a dictionary
    that can be used to restore the original expressions.

    Examples
    ========

    >>> from sympy.solvers.solvers import recast_to_symbols
    >>> from sympy import symbols, Function
    >>> x, y = symbols('x y')
    >>> fx = Function('f')(x)
    >>> eqs, syms = [fx + 1, x, y], [fx, y]
    >>> e, s, d = recast_to_symbols(eqs, syms); (e, s, d)
    ([_X0 + 1, x, y], [_X0, y], {_X0: f(x)})

    The original equations and symbols can be restored using d:

    >>> assert [i.xreplace(d) for i in eqs] == eqs
    >>> assert [d.get(i, i) for i in s] == syms

    """
    if not iterable(eqs) and iterable(symbols):
        raise ValueError('Both eqs and symbols must be iterable')
    orig = list(symbols)
    symbols = list(ordered(symbols))
    swap_sym = {}
    i = 0
    for s in symbols:
        if not isinstance(s, Symbol) and s not in swap_sym:
            swap_sym[s] = Dummy('X%d' % i)
            i += 1
    new_f = []
    for i in eqs:
        isubs = getattr(i, 'subs', None)
        if isubs is not None:
            new_f.append(isubs(swap_sym))
        else:
            new_f.append(i)
    restore = {v: k for k, v in swap_sym.items()}
    return new_f, [swap_sym.get(i, i) for i in orig], restore


def _ispow(e):
    """Return True if e is a Pow or is exp."""
    return isinstance(e, Expr) and (e.is_Pow or isinstance(e, exp))


def _simple_dens(f, symbols):
    # when checking if a denominator is zero, we can just check the
    # base of powers with nonzero exponents since if the base is zero
    # the power will be zero, too. To keep it simple and fast, we
    # limit simplification to exponents that are Numbers
    dens = set()
    for d in denoms(f, symbols):
        if d.is_Pow and d.exp.is_Number:
            if d.exp.is_zero:
                continue  # foo**0 is never 0
            d = d.base
        dens.add(d)
    return dens


def denoms(eq, *symbols):
    """
    Return (recursively) set of all denominators that appear in *eq*
    that contain any symbol in *symbols*; if *symbols* are not
    provided then all denominators will be returned.

    Examples
    ========

    >>> from sympy.solvers.solvers import denoms
    >>> from sympy.abc import x, y, z

    >>> denoms(x/y)
    {y}

    >>> denoms(x/(y*z))
    {y, z}

    >>> denoms(3/x + y/z)
    {x, z}

    >>> denoms(x/2 + y/z)
    {2, z}

    If *symbols* are provided then only denominators containing
    those symbols will be returned:

    >>> denoms(1/x + 1/y + 1/z, y, z)
    {y, z}

    """

    pot = preorder_traversal(eq)
    dens = set()
    for p in pot:
        # Here p might be Tuple or Relational
        # Expr subtrees (e.g. lhs and rhs) will be traversed after by pot
        if not isinstance(p, Expr):
            continue
        den = denom(p)
        if den is S.One:
            continue
        dens.update(Mul.make_args(den))
    if not symbols:
        return dens
    elif len(symbols) == 1:
        if iterable(symbols[0]):
            symbols = symbols[0]
    return {d for d in dens if any(s in d.free_symbols for s in symbols)}


def checksol(f, symbol, sol=None, **flags):
    """
    Checks whether sol is a solution of equation f == 0.

    Explanation
    ===========

    Input can be either a single symbol and corresponding value
    or a dictionary of symbols and values. When given as a dictionary
    and flag ``simplify=True``, the values in the dictionary will be
    simplified. *f* can be a single equation or an iterable of equations.
    A solution must satisfy all equations in *f* to be considered valid;
    if a solution does not satisfy any equation, False is returned; if one or
    more checks are inconclusive (and none are False) then None is returned.

    Examples
    ========

    >>> from sympy import checksol, symbols
    >>> x, y = symbols('x,y')
    >>> checksol(x**4 - 1, x, 1)
    True
    >>> checksol(x**4 - 1, x, 0)
    False
    >>> checksol(x**2 + y**2 - 5**2, {x: 3, y: 4})
    True

    To check if an expression is zero using ``checksol()``, pass it
    as *f* and send an empty dictionary for *symbol*:

    >>> checksol(x**2 + x - x*(x + 1), {})
    True

    None is returned if ``checksol()`` could not conclude.

    flags:
        'numerical=True (default)'
           do a fast numerical check if ``f`` has only one symbol.
        'minimal=True (default is False)'
           a very fast, minimal testing.
        'warn=True (default is False)'
           show a warning if checksol() could not conclude.
        'simplify=True (default)'
           simplify solution before substituting into function and
           simplify the function before trying specific simplifications
        'force=True (default is False)'
           make positive all symbols without assumptions regarding sign.

    """
    from sympy.physics.units import Unit

    minimal = flags.get('minimal', False)

    if sol is not None:
        sol = {symbol: sol}
    elif isinstance(symbol, dict):
        sol = symbol
    else:
        msg = 'Expecting (sym, val) or ({sym: val}, None) but got (%s, %s)'
        raise ValueError(msg % (symbol, sol))

    if iterable(f):
        if not f:
            raise ValueError('no functions to check')
        return fuzzy_and(checksol(fi, sol, **flags) for fi in f)

    f = _sympify(f)

    if f.is_number:
        return f.is_zero

    if isinstance(f, Poly):
        f = f.as_expr()
    elif isinstance(f, (Eq, Ne)):
        if f.rhs in (S.true, S.false):
            f = f.reversed
        B, E = f.args
        if isinstance(B, BooleanAtom):
            f = f.subs(sol)
            if not f.is_Boolean:
                return
        elif isinstance(f, Eq):
            f = Add(f.lhs, -f.rhs, evaluate=False)

    if isinstance(f, BooleanAtom):
        return bool(f)
    elif not f.is_Relational and not f:
        return True

    illegal = set(_illegal)
    if any(sympify(v).atoms() & illegal for k, v in sol.items()):
        return False

    attempt = -1
    numerical = flags.get('numerical', True)
    while 1:
        attempt += 1
        if attempt == 0:
            val = f.subs(sol)
            if isinstance(val, Mul):
                val = val.as_independent(Unit)[0]
            if val.atoms() & illegal:
                return False
        elif attempt == 1:
            if not val.is_number:
                if not val.is_constant(*list(sol.keys()), simplify=not minimal):
                    return False
                # there are free symbols -- simple expansion might work
                _, val = val.as_content_primitive()
                val = _mexpand(val.as_numer_denom()[0], recursive=True)
        elif attempt == 2:
            if minimal:
                return
            if flags.get('simplify', True):
                for k in sol:
                    sol[k] = simplify(sol[k])
            # start over without the failed expanded form, possibly
            # with a simplified solution
            val = simplify(f.subs(sol))
            if flags.get('force', True):
                val, reps = posify(val)
                # expansion may work now, so try again and check
                exval = _mexpand(val, recursive=True)
                if exval.is_number:
                    # we can decide now
                    val = exval
        else:
            # if there are no radicals and no functions then this can't be
            # zero anymore -- can it?
            pot = preorder_traversal(expand_mul(val))
            seen = set()
            saw_pow_func = False
            for p in pot:
                if p in seen:
                    continue
                seen.add(p)
                if p.is_Pow and not p.exp.is_Integer:
                    saw_pow_func = True
                elif p.is_Function:
                    saw_pow_func = True
                elif isinstance(p, UndefinedFunction):
                    saw_pow_func = True
                if saw_pow_func:
                    break
            if saw_pow_func is False:
                return False
            if flags.get('force', True):
                # don't do a zero check with the positive assumptions in place
                val = val.subs(reps)
            nz = fuzzy_not(val.is_zero)
            if nz is not None:
                # issue 5673: nz may be True even when False
                # so these are just hacks to keep a false positive
                # from being returned

                # HACK 1: LambertW (issue 5673)
                if val.is_number and val.has(LambertW):
                    # don't eval this to verify solution since if we got here,
                    # numerical must be False
                    return None

                # add other HACKs here if necessary, otherwise we assume
                # the nz value is correct
                return not nz
            break
        if val.is_Rational:
            return val == 0
        if numerical and val.is_number:
            return (abs(val.n(18).n(12, chop=True)) < 1e-9) is S.true

    if flags.get('warn', False):
        warnings.warn("\n\tWarning: could not verify solution %s." % sol)
    # returns None if it can't conclude
    # TODO: improve solution testing


def solve(f, *symbols, **flags):
    r"""
    Algebraically solves equations and systems of equations.

    Explanation
    ===========

    Currently supported:
        - polynomial
        - transcendental
        - piecewise combinations of the above
        - systems of linear and polynomial equations
        - systems containing relational expressions
        - systems implied by undetermined coefficients

    Examples
    ========

    The default output varies according to the input and might
    be a list (possibly empty), a dictionary, a list of
    dictionaries or tuples, or an expression involving relationals.
    For specifics regarding different forms of output that may appear, see :ref:`solve_output`.
    Let it suffice here to say that to obtain a uniform output from
    `solve` use ``dict=True`` or ``set=True`` (see below).

        >>> from sympy import solve, Poly, Eq, Matrix, Symbol
        >>> from sympy.abc import x, y, z, a, b

    The expressions that are passed can be Expr, Equality, or Poly
    classes (or lists of the same); a Matrix is considered to be a
    list of all the elements of the matrix:

        >>> solve(x - 3, x)
        [3]
        >>> solve(Eq(x, 3), x)
        [3]
        >>> solve(Poly(x - 3), x)
        [3]
        >>> solve(Matrix([[x, x + y]]), x, y) == solve([x, x + y], x, y)
        True

    If no symbols are indicated to be of interest and the equation is
    univariate, a list of values is returned; otherwise, the keys in
    a dictionary will indicate which (of all the variables used in
    the expression(s)) variables and solutions were found:

        >>> solve(x**2 - 4)
        [-2, 2]
        >>> solve((x - a)*(y - b))
        [{a: x}, {b: y}]
        >>> solve([x - 3, y - 1])
        {x: 3, y: 1}
        >>> solve([x - 3, y**2 - 1])
        [{x: 3, y: -1}, {x: 3, y: 1}]

    If you pass symbols for which solutions are sought, the output will vary
    depending on the number of symbols you passed, whether you are passing
    a list of expressions or not, and whether a linear system was solved.
    Uniform output is attained by using ``dict=True`` or ``set=True``.

        >>> #### *** feel free to skip to the stars below *** ####
        >>> from sympy import TableForm
        >>> h = [None, ';|;'.join(['e', 's', 'solve(e, s)', 'solve(e, s, dict=True)',
        ... 'solve(e, s, set=True)']).split(';')]
        >>> t = []
        >>> for e, s in [
        ...         (x - y, y),
        ...         (x - y, [x, y]),
        ...         (x**2 - y, [x, y]),
        ...         ([x - 3, y -1], [x, y]),
        ...         ]:
        ...     how = [{}, dict(dict=True), dict(set=True)]
        ...     res = [solve(e, s, **f) for f in how]
        ...     t.append([e, '|', s, '|'] + [res[0], '|', res[1], '|', res[2]])
        ...
        >>> # ******************************************************* #
        >>> TableForm(t, headings=h, alignments="<")
        e              | s      | solve(e, s)  | solve(e, s, dict=True) | solve(e, s, set=True)
        ---------------------------------------------------------------------------------------
        x - y          | y      | [x]          | [{y: x}]               | ([y], {(x,)})
        x - y          | [x, y] | [(y, y)]     | [{x: y}]               | ([x, y], {(y, y)})
        x**2 - y       | [x, y] | [(x, x**2)]  | [{y: x**2}]            | ([x, y], {(x, x**2)})
        [x - 3, y - 1] | [x, y] | {x: 3, y: 1} | [{x: 3, y: 1}]         | ([x, y], {(3, 1)})

        * If any equation does not depend on the symbol(s) given, it will be
          eliminated from the equation set and an answer may be given
          implicitly in terms of variables that were not of interest:

            >>> solve([x - y, y - 3], x)
            {x: y}

    When you pass all but one of the free symbols, an attempt
    is made to find a single solution based on the method of
    undetermined coefficients. If it succeeds, a dictionary of values
    is returned. If you want an algebraic solutions for one
    or more of the symbols, pass the expression to be solved in a list:

        >>> e = a*x + b - 2*x - 3
        >>> solve(e, [a, b])
        {a: 2, b: 3}
        >>> solve([e], [a, b])
        {a: -b/x + (2*x + 3)/x}

    When there is no solution for any given symbol which will make all
    expressions zero, the empty list is returned (or an empty set in
    the tuple when ``set=True``):

        >>> from sympy import sqrt
        >>> solve(3, x)
        []
        >>> solve(x - 3, y)
        []
        >>> solve(sqrt(x) + 1, x, set=True)
        ([x], set())

    When an object other than a Symbol is given as a symbol, it is
    isolated algebraically and an implicit solution may be obtained.
    This is mostly provided as a convenience to save you from replacing
    the object with a Symbol and solving for that Symbol. It will only
    work if the specified object can be replaced with a Symbol using the
    subs method:

        >>> from sympy import exp, Function
        >>> f = Function('f')

        >>> solve(f(x) - x, f(x))
        [x]
        >>> solve(f(x).diff(x) - f(x) - x, f(x).diff(x))
        [x + f(x)]
        >>> solve(f(x).diff(x) - f(x) - x, f(x))
        [-x + Derivative(f(x), x)]
        >>> solve(x + exp(x)**2, exp(x), set=True)
        ([exp(x)], {(-sqrt(-x),), (sqrt(-x),)})

        >>> from sympy import Indexed, IndexedBase, Tuple
        >>> A = IndexedBase('A')
        >>> eqs = Tuple(A[1] + A[2] - 3, A[1] - A[2] + 1)
        >>> solve(eqs, eqs.atoms(Indexed))
        {A[1]: 1, A[2]: 2}

        * To solve for a function within a derivative, use :func:`~.dsolve`.

    To solve for a symbol implicitly, use implicit=True:

        >>> solve(x + exp(x), x)
        [-LambertW(1)]
        >>> solve(x + exp(x), x, implicit=True)
        [-exp(x)]

    It is possible to solve for anything in an expression that can be
    replaced with a symbol using :obj:`~sympy.core.basic.Basic.subs`:

        >>> solve(x + 2 + sqrt(3), x + 2)
        [-sqrt(3)]
        >>> solve((x + 2 + sqrt(3), x + 4 + y), y, x + 2)
        {y: -2 + sqrt(3), x + 2: -sqrt(3)}

        * Nothing heroic is done in this implicit solving so you may end up
          with a symbol still in the solution:

            >>> eqs = (x*y + 3*y + sqrt(3), x + 4 + y)
            >>> solve(eqs, y, x + 2)
            {y: -sqrt(3)/(x + 3), x + 2: -2*x/(x + 3) - 6/(x + 3) + sqrt(3)/(x + 3)}
            >>> solve(eqs, y*x, x)
            {x: -y - 4, x*y: -3*y - sqrt(3)}

        * If you attempt to solve for a number, remember that the number
          you have obtained does not necessarily mean that the value is
          equivalent to the expression obtained:

            >>> solve(sqrt(2) - 1, 1)
            [sqrt(2)]
            >>> solve(x - y + 1, 1)  # /!\ -1 is targeted, too
            [x/(y - 1)]
            >>> [_.subs(z, -1) for _ in solve((x - y + 1).subs(-1, z), 1)]
            [-x + y]

    **Additional Examples**

    ``solve()`` with check=True (default) will run through the symbol tags to
    eliminate unwanted solutions. If no assumptions are included, all possible
    solutions will be returned:

        >>> x = Symbol("x")
        >>> solve(x**2 - 1)
        [-1, 1]

    By setting the ``positive`` flag, only one solution will be returned:

        >>> pos = Symbol("pos", positive=True)
        >>> solve(pos**2 - 1)
        [1]

    When the solutions are checked, those that make any denominator zero
    are automatically excluded. If you do not want to exclude such solutions,
    then use the check=False option:

        >>> from sympy import sin, limit
        >>> solve(sin(x)/x)  # 0 is excluded
        [pi]

    If ``check=False``, then a solution to the numerator being zero is found
    but the value of $x = 0$ is a spurious solution since $\sin(x)/x$ has the well
    known limit (without discontinuity) of 1 at $x = 0$:

        >>> solve(sin(x)/x, check=False)
        [0, pi]

    In the following case, however, the limit exists and is equal to the
    value of $x = 0$ that is excluded when check=True:

        >>> eq = x**2*(1/x - z**2/x)
        >>> solve(eq, x)
        []
        >>> solve(eq, x, check=False)
        [0]
        >>> limit(eq, x, 0, '-')
        0
        >>> limit(eq, x, 0, '+')
        0

    **Solving Relationships**

    When one or more expressions passed to ``solve`` is a relational,
    a relational result is returned (and the ``dict`` and ``set`` flags
    are ignored):

        >>> solve(x < 3)
        (-oo < x) & (x < 3)
        >>> solve([x < 3, x**2 > 4], x)
        ((-oo < x) & (x < -2)) | ((2 < x) & (x < 3))
        >>> solve([x + y - 3, x > 3], x)
        (3 < x) & (x < oo) & Eq(x, 3 - y)

    Although checking of assumptions on symbols in relationals
    is not done, setting assumptions will affect how certain
    relationals might automatically simplify:

        >>> solve(x**2 > 4)
        ((-oo < x) & (x < -2)) | ((2 < x) & (x < oo))

        >>> r = Symbol('r', real=True)
        >>> solve(r**2 > 4)
        (2 < r) | (r < -2)

    There is currently no algorithm in SymPy that allows you to use
    relationships to resolve more than one variable. So the following
    does not determine that ``q < 0`` (and trying to solve for ``r``
    and ``q`` will raise an error):

        >>> from sympy import symbols
        >>> r, q = symbols('r, q', real=True)
        >>> solve([r + q - 3, r > 3], r)
        (3 < r) & Eq(r, 3 - q)

    You can directly call the routine that ``solve`` calls
    when it encounters a relational: :func:`~.reduce_inequalities`.
    It treats Expr like Equality.

        >>> from sympy import reduce_inequalities
        >>> reduce_inequalities([x**2 - 4])
        Eq(x, -2) | Eq(x, 2)

    If each relationship contains only one symbol of interest,
    the expressions can be processed for multiple symbols:

        >>> reduce_inequalities([0 <= x  - 1, y < 3], [x, y])
        (-oo < y) & (1 <= x) & (x < oo) & (y < 3)

    But an error is raised if any relationship has more than one
    symbol of interest:

        >>> reduce_inequalities([0 <= x*y  - 1, y < 3], [x, y])
        Traceback (most recent call last):
        ...
        NotImplementedError:
        inequality has more than one symbol of interest.

    **Disabling High-Order Explicit Solutions**

    When solving polynomial expressions, you might not want explicit solutions
    (which can be quite long). If the expression is univariate, ``CRootOf``
    instances will be returned instead:

        >>> solve(x**3 - x + 1)
        [-1/((-1/2 - sqrt(3)*I/2)*(3*sqrt(69)/2 + 27/2)**(1/3)) -
        (-1/2 - sqrt(3)*I/2)*(3*sqrt(69)/2 + 27/2)**(1/3)/3,
        -(-1/2 + sqrt(3)*I/2)*(3*sqrt(69)/2 + 27/2)**(1/3)/3 -
        1/((-1/2 + sqrt(3)*I/2)*(3*sqrt(69)/2 + 27/2)**(1/3)),
        -(3*sqrt(69)/2 + 27/2)**(1/3)/3 -
        1/(3*sqrt(69)/2 + 27/2)**(1/3)]
        >>> solve(x**3 - x + 1, cubics=False)
        [CRootOf(x**3 - x + 1, 0),
         CRootOf(x**3 - x + 1, 1),
         CRootOf(x**3 - x + 1, 2)]

    If the expression is multivariate, no solution might be returned:

        >>> solve(x**3 - x + a, x, cubics=False)
        []

    Sometimes solutions will be obtained even when a flag is False because the
    expression could be factored. In the following example, the equation can
    be factored as the product of a linear and a quadratic factor so explicit
    solutions (which did not require solving a cubic expression) are obtained:

        >>> eq = x**3 + 3*x**2 + x - 1
        >>> solve(eq, cubics=False)
        [-1, -1 + sqrt(2), -sqrt(2) - 1]

    **Solving Equations Involving Radicals**

    Because of SymPy's use of the principle root, some solutions
    to radical equations will be missed unless check=False:

        >>> from sympy import root
        >>> eq = root(x**3 - 3*x**2, 3) + 1 - x
        >>> solve(eq)
        []
        >>> solve(eq, check=False)
        [1/3]

    In the above example, there is only a single solution to the
    equation. Other expressions will yield spurious roots which
    must be checked manually; roots which give a negative argument
    to odd-powered radicals will also need special checking:

        >>> from sympy import real_root, S
        >>> eq = root(x, 3) - root(x, 5) + S(1)/7
        >>> solve(eq)  # this gives 2 solutions but misses a 3rd
        [CRootOf(7*x**5 - 7*x**3 + 1, 1)**15,
        CRootOf(7*x**5 - 7*x**3 + 1, 2)**15]
        >>> sol = solve(eq, check=False)
        >>> [abs(eq.subs(x,i).n(2)) for i in sol]
        [0.48, 0.e-110, 0.e-110, 0.052, 0.052]

    The first solution is negative so ``real_root`` must be used to see that it
    satisfies the expression:

        >>> abs(real_root(eq.subs(x, sol[0])).n(2))
        0.e-110

    If the roots of the equation are not real then more care will be
    necessary to find the roots, especially for higher order equations.
    Consider the following expression:

        >>> expr = root(x, 3) - root(x, 5)

    We will construct a known value for this expression at x = 3 by selecting
    the 1-th root for each radical:

        >>> expr1 = root(x, 3, 1) - root(x, 5, 1)
        >>> v = expr1.subs(x, -3)

    The ``solve`` function is unable to find any exact roots to this equation:

        >>> eq = Eq(expr, v); eq1 = Eq(expr1, v)
        >>> solve(eq, check=False), solve(eq1, check=False)
        ([], [])

    The function ``unrad``, however, can be used to get a form of the equation
    for which numerical roots can be found:

        >>> from sympy.solvers.solvers import unrad
        >>> from sympy import nroots
        >>> e, (p, cov) = unrad(eq)
        >>> pvals = nroots(e)
        >>> inversion = solve(cov, x)[0]
        >>> xvals = [inversion.subs(p, i) for i in pvals]

    Although ``eq`` or ``eq1`` could have been used to find ``xvals``, the
    solution can only be verified with ``expr1``:

        >>> z = expr - v
        >>> [xi.n(chop=1e-9) for xi in xvals if abs(z.subs(x, xi).n()) < 1e-9]
        []
        >>> z1 = expr1 - v
        >>> [xi.n(chop=1e-9) for xi in xvals if abs(z1.subs(x, xi).n()) < 1e-9]
        [-3.0]

    Parameters
    ==========

    f :
        - a single Expr or Poly that must be zero
        - an Equality
        - a Relational expression
        - a Boolean
        - iterable of one or more of the above

    symbols : (object(s) to solve for) specified as
        - none given (other non-numeric objects will be used)
        - single symbol
        - denested list of symbols
          (e.g., ``solve(f, x, y)``)
        - ordered iterable of symbols
          (e.g., ``solve(f, [x, y])``)

    flags :
        dict=True (default is False)
            Return list (perhaps empty) of solution mappings.
        set=True (default is False)
            Return list of symbols and set of tuple(s) of solution(s).
        exclude=[] (default)
            Do not try to solve for any of the free symbols in exclude;
            if expressions are given, the free symbols in them will
            be extracted automatically.
        check=True (default)
            If False, do not do any testing of solutions. This can be
            useful if you want to include solutions that make any
            denominator zero.
        numerical=True (default)
            Do a fast numerical check if *f* has only one symbol.
        minimal=True (default is False)
            A very fast, minimal testing.
        warn=True (default is False)
            Show a warning if ``checksol()`` could not conclude.
        simplify=True (default)
            Simplify all but polynomials of order 3 or greater before
            returning them and (if check is not False) use the
            general simplify function on the solutions and the
            expression obtained when they are substituted into the
            function which should be zero.
        force=True (default is False)
            Make positive all symbols without assumptions regarding sign.
        rational=True (default)
            Recast Floats as Rational; if this option is not used, the
            system containing Floats may fail to solve because of issues
            with polys. If rational=None, Floats will be recast as
            rationals but the answer will be recast as Floats. If the
            flag is False then nothing will be done to the Floats.
        manual=True (default is False)
            Do not use the polys/matrix method to solve a system of
            equations, solve them one at a time as you might "manually."
        implicit=True (default is False)
            Allows ``solve`` to return a solution for a pattern in terms of
            other functions that contain that pattern; this is only
            needed if the pattern is inside of some invertible function
            like cos, exp, etc.
        particular=True (default is False)
            Instructs ``solve`` to try to find a particular solution to
            a linear system with as many zeros as possible; this is very
            expensive.
        quick=True (default is False; ``particular`` must be True)
            Selects a fast heuristic to find a solution with many zeros
            whereas a value of False uses the very slow method guaranteed
            to find the largest number of zeros possible.
        cubics=True (default)
            Return explicit solutions when cubic expressions are encountered.
            When False, quartics and quintics are disabled, too.
        quartics=True (default)
            Return explicit solutions when quartic expressions are encountered.
            When False, quintics are disabled, too.
        quintics=True (default)
            Return explicit solutions (if possible) when quintic expressions
            are encountered.

    See Also
    ========

    rsolve: For solving recurrence relationships
    sympy.solvers.ode.dsolve: For solving differential equations

    """
    from .inequalities import reduce_inequalities

    # checking/recording flags
    ###########################################################################

    # set solver types explicitly; as soon as one is False
    # all the rest will be False
    hints = ('cubics', 'quartics', 'quintics')
    default = True
    for k in hints:
        default = flags.setdefault(k, bool(flags.get(k, default)))

    # allow solution to contain symbol if True:
    implicit = flags.get('implicit', False)

    # record desire to see warnings
    warn = flags.get('warn', False)

    # this flag will be needed for quick exits below, so record
    # now -- but don't record `dict` yet since it might change
    as_set = flags.get('set', False)

    # keeping track of how f was passed
    bare_f = not iterable(f)

    # check flag usage for particular/quick which should only be used
    # with systems of equations
    if flags.get('quick', None) is not None:
        if not flags.get('particular', None):
            raise ValueError('when using `quick`, `particular` should be True')
    if flags.get('particular', False) and bare_f:
        raise ValueError(filldedent("""
            The 'particular/quick' flag is usually used with systems of
            equations. Either pass your equation in a list or
            consider using a solver like `diophantine` if you are
            looking for a solution in integers."""))

    # sympify everything, creating list of expressions and list of symbols
    ###########################################################################

    def _sympified_list(w):
        return list(map(sympify, w if iterable(w) else [w]))
    f, symbols = (_sympified_list(w) for w in [f, symbols])

    # preprocess symbol(s)
    ###########################################################################

    ordered_symbols = None  # were the symbols in a well defined order?
    if not symbols:
        # get symbols from equations
        symbols = set().union(*[fi.free_symbols for fi in f])
        if len(symbols) < len(f):
            for fi in f:
                pot = preorder_traversal(fi)
                for p in pot:
                    if isinstance(p, AppliedUndef):
                        if not as_set:
                            flags['dict'] = True  # better show symbols
                        symbols.add(p)
                        pot.skip()  # don't go any deeper
        ordered_symbols = False
        symbols = list(ordered(symbols))  # to make it canonical
    else:
        if len(symbols) == 1 and iterable(symbols[0]):
            symbols = symbols[0]
        ordered_symbols = symbols and is_sequence(symbols,
                        include=GeneratorType)
        _symbols = list(uniq(symbols))
        if len(_symbols) != len(symbols):
            ordered_symbols = False
            symbols = list(ordered(symbols))
        else:
            symbols = _symbols

    # check for duplicates
    if len(symbols) != len(set(symbols)):
        raise ValueError('duplicate symbols given')
    # remove those not of interest
    exclude = flags.pop('exclude', set())
    if exclude:
        if isinstance(exclude, Expr):
            exclude = [exclude]
        exclude = set().union(*[e.free_symbols for e in sympify(exclude)])
        symbols = [s for s in symbols if s not in exclude]

    # preprocess equation(s)
    ###########################################################################

    # automatically ignore True values
    if isinstance(f, list):
        f = [s for s in f if s is not S.true]

    # handle canonicalization of equation types
    for i, fi in enumerate(f):
        if isinstance(fi, (Eq, Ne)):
            if 'ImmutableDenseMatrix' in [type(a).__name__ for a in fi.args]:
                fi = fi.lhs - fi.rhs
            else:
                L, R = fi.args
                if isinstance(R, BooleanAtom):
                    L, R = R, L
                if isinstance(L, BooleanAtom):
                    if isinstance(fi, Ne):
                        L = ~L
                    if R.is_Relational:
                        fi = ~R if L is S.false else R
                    elif R.is_Symbol:
                        return L
                    elif R.is_Boolean and (~R).is_Symbol:
                        return ~L
                    else:
                        raise NotImplementedError(filldedent('''
                            Unanticipated argument of Eq when other arg
                            is True or False.
                        '''))
                elif isinstance(fi, Eq):
                    fi = Add(fi.lhs, -fi.rhs, evaluate=False)
            f[i] = fi

        # *** dispatch and handle as a system of relationals
        # **************************************************
        if fi.is_Relational:
            if len(symbols) != 1:
                raise ValueError("can only solve for one symbol at a time")
            if warn and symbols[0].assumptions0:
                warnings.warn(filldedent("""
                    \tWarning: assumptions about variable '%s' are
                    not handled currently.""" % symbols[0]))
            return reduce_inequalities(f, symbols=symbols)

        # convert Poly to expression
        if isinstance(fi, Poly):
            f[i] = fi.as_expr()

        # rewrite hyperbolics in terms of exp if they have symbols of
        # interest
        f[i] = f[i].replace(lambda w: isinstance(w, HyperbolicFunction) and \
            w.has_free(*symbols), lambda w: w.rewrite(exp))

        # if we have a Matrix, we need to iterate over its elements again
        if f[i].is_Matrix:
            try:
                f[i] = f[i].as_explicit()
            except ValueError:
                raise ValueError(
                    "solve cannot handle matrices with symbolic shape."
                )
            bare_f = False
            f.extend(list(f[i]))
            f[i] = S.Zero

        # if we can split it into real and imaginary parts then do so
        freei = f[i].free_symbols
        if freei and all(s.is_extended_real or s.is_imaginary for s in freei):
            fr, fi = f[i].as_real_imag()
            # accept as long as new re, im, arg or atan2 are not introduced
            had = f[i].atoms(re, im, arg, atan2)
            if fr and fi and fr != fi and not any(
                    i.atoms(re, im, arg, atan2) - had for i in (fr, fi)):
                if bare_f:
                    bare_f = False
                f[i: i + 1] = [fr, fi]

    # real/imag handling -----------------------------
    if any(isinstance(fi, (bool, BooleanAtom)) for fi in f):
        if as_set:
            return [], set()
        return []

    for i, fi in enumerate(f):
        # Abs
        while True:
            was = fi
            fi = fi.replace(Abs, lambda arg:
                separatevars(Abs(arg)).rewrite(Piecewise) if arg.has(*symbols)
                else Abs(arg))
            if was == fi:
                break

        for e in fi.find(Abs):
            if e.has(*symbols):
                raise NotImplementedError('solving %s when the argument '
                    'is not real or imaginary.' % e)

        # arg
        fi = fi.replace(arg, lambda a: arg(a).rewrite(atan2).rewrite(atan))

        # save changes
        f[i] = fi

    # see if re(s) or im(s) appear
    freim = [fi for fi in f if fi.has(re, im)]
    if freim:
        irf = []
        for s in symbols:
            if s.is_real or s.is_imaginary:
                continue  # neither re(x) nor im(x) will appear
            # if re(s) or im(s) appear, the auxiliary equation must be present
            if any(fi.has(re(s), im(s)) for fi in freim):
                irf.append((s, re(s) + S.ImaginaryUnit*im(s)))
        if irf:
            for s, rhs in irf:
                f = [fi.xreplace({s: rhs}) for fi in f] + [s - rhs]
                symbols.extend([re(s), im(s)])
            if bare_f:
                bare_f = False
            flags['dict'] = True
    # end of real/imag handling  -----------------------------

    # we can solve for non-symbol entities by replacing them with Dummy symbols
    f, symbols, swap_sym = recast_to_symbols(f, symbols)
    # this set of symbols (perhaps recast) is needed below
    symset = set(symbols)

    # get rid of equations that have no symbols of interest; we don't
    # try to solve them because the user didn't ask and they might be
    # hard to solve; this means that solutions may be given in terms
    # of the eliminated equations e.g. solve((x-y, y-3), x) -> {x: y}
    newf = []
    for fi in f:
        # let the solver handle equations that..
        # - have no symbols but are expressions
        # - have symbols of interest
        # - have no symbols of interest but are constant
        # but when an expression is not constant and has no symbols of
        # interest, it can't change what we obtain for a solution from
        # the remaining equations so we don't include it; and if it's
        # zero it can be removed and if it's not zero, there is no
        # solution for the equation set as a whole
        #
        # The reason for doing this filtering is to allow an answer
        # to be obtained to queries like solve((x - y, y), x); without
        # this mod the return value is []
        ok = False
        if fi.free_symbols & symset:
            ok = True
        else:
            if fi.is_number:
                if fi.is_Number:
                    if fi.is_zero:
                        continue
                    return []
                ok = True
            else:
                if fi.is_constant():
                    ok = True
        if ok:
            newf.append(fi)
    if not newf:
        if as_set:
            return symbols, set()
        return []
    f = newf
    del newf

    # mask off any Object that we aren't going to invert: Derivative,
    # Integral, etc... so that solving for anything that they contain will
    # give an implicit solution
    seen = set()
    non_inverts = set()
    for fi in f:
        pot = preorder_traversal(fi)
        for p in pot:
            if not isinstance(p, Expr) or isinstance(p, Piecewise):
                pass
            elif (isinstance(p, bool) or
                    not p.args or
                    p in symset or
                    p.is_Add or p.is_Mul or
                    p.is_Pow and not implicit or
                    p.is_Function and not implicit) and p.func not in (re, im):
                continue
            elif p not in seen:
                seen.add(p)
                if p.free_symbols & symset:
                    non_inverts.add(p)
                else:
                    continue
            pot.skip()
    del seen
    non_inverts = dict(list(zip(non_inverts, [Dummy() for _ in non_inverts])))
    f = [fi.subs(non_inverts) for fi in f]

    # Both xreplace and subs are needed below: xreplace to force substitution
    # inside Derivative, subs to handle non-straightforward substitutions
    non_inverts = [(v, k.xreplace(swap_sym).subs(swap_sym)) for k, v in non_inverts.items()]

    # rationalize Floats
    floats = False
    if flags.get('rational', True) is not False:
        for i, fi in enumerate(f):
            if fi.has(Float):
                floats = True
                f[i] = nsimplify(fi, rational=True)

    # capture any denominators before rewriting since
    # they may disappear after the rewrite, e.g. issue 14779
    flags['_denominators'] = _simple_dens(f[0], symbols)

    # Any embedded piecewise functions need to be brought out to the
    # top level so that the appropriate strategy gets selected.
    # However, this is necessary only if one of the piecewise
    # functions depends on one of the symbols we are solving for.
    def _has_piecewise(e):
        if e.is_Piecewise:
            return e.has(*symbols)
        return any(_has_piecewise(a) for a in e.args)
    for i, fi in enumerate(f):
        if _has_piecewise(fi):
            f[i] = piecewise_fold(fi)

    # expand angles of sums; in general, expand_trig will allow
    # more roots to be found but this is not a great solultion
    # to not returning a parametric solution, otherwise
    # many values can be returned that have a simple
    # relationship between values
    targs = {t for fi in f for t in fi.atoms(TrigonometricFunction)}
    if len(targs) > 1:
        add, other = sift(targs, lambda x: x.args[0].is_Add, binary=True)
        add, other = [[i for i in l if i.has_free(*symbols)] for l in (add, other)]
        trep = {}
        for t in add:
            a = t.args[0]
            ind, dep = a.as_independent(*symbols)
            if dep in symbols or -dep in symbols:
                # don't let expansion expand wrt anything in ind
                n = Dummy() if not ind.is_Number else ind
                trep[t] = TR10(t.func(dep + n)).xreplace({n: ind})
        if other and len(other) <= 2:
            base = gcd(*[i.args[0] for i in other]) if len(other) > 1 else other[0].args[0]
            for i in other:
                trep[i] = TR11(i, base)
        f = [fi.xreplace(trep) for fi in f]

    #
    # try to get a solution
    ###########################################################################
    if bare_f:
        solution = None
        if len(symbols) != 1:
            solution = _solve_undetermined(f[0], symbols, flags)
        if not solution:
            solution = _solve(f[0], *symbols, **flags)
    else:
        linear, solution = _solve_system(f, symbols, **flags)
    assert type(solution) is list
    assert not solution or type(solution[0]) is dict, solution
    #
    # postprocessing
    ###########################################################################
    # capture as_dict flag now (as_set already captured)
    as_dict = flags.get('dict', False)

    # define how solution will get unpacked
    tuple_format = lambda s: [tuple([i.get(x, x) for x in symbols]) for i in s]
    if as_dict or as_set:
        unpack = None
    elif bare_f:
        if len(symbols) == 1:
            unpack = lambda s: [i[symbols[0]] for i in s]
        elif len(solution) == 1 and len(solution[0]) == len(symbols):
            # undetermined linear coeffs solution
            unpack = lambda s: s[0]
        elif ordered_symbols:
            unpack = tuple_format
        else:
            unpack = lambda s: s
    else:
        if solution:
            if linear and len(solution) == 1:
                # if you want the tuple solution for the linear
                # case, use `set=True`
                unpack = lambda s: s[0]
            elif ordered_symbols:
                unpack = tuple_format
            else:
                unpack = lambda s: s
        else:
            unpack = None

    # Restore masked-off objects
    if non_inverts and type(solution) is list:
        solution = [{k: v.subs(non_inverts) for k, v in s.items()}
            for s in solution]

    # Restore original "symbols" if a dictionary is returned.
    # This is not necessary for
    #   - the single univariate equation case
    #     since the symbol will have been removed from the solution;
    #   - the nonlinear poly_system since that only supports zero-dimensional
    #     systems and those results come back as a list
    #
    # ** unless there were Derivatives with the symbols, but those were handled
    #    above.
    if swap_sym:
        symbols = [swap_sym.get(k, k) for k in symbols]
        for i, sol in enumerate(solution):
            solution[i] = {swap_sym.get(k, k): v.subs(swap_sym)
                      for k, v in sol.items()}

    # Get assumptions about symbols, to filter solutions.
    # Note that if assumptions about a solution can't be verified, it is still
    # returned.
    check = flags.get('check', True)

    # restore floats
    if floats and solution and flags.get('rational', None) is None:
        solution = nfloat(solution, exponent=False)
        # nfloat might reveal more duplicates
        solution = _remove_duplicate_solutions(solution)

    if check and solution:  # assumption checking
        warn = flags.get('warn', False)
        got_None = []  # solutions for which one or more symbols gave None
        no_False = []  # solutions for which no symbols gave False
        for sol in solution:
            v = fuzzy_and(check_assumptions(val, **symb.assumptions0)
                          for symb, val in sol.items())
            if v is False:
                continue
            no_False.append(sol)
            if v is None:
                got_None.append(sol)

        solution = no_False
        if warn and got_None:
            warnings.warn(filldedent("""
                \tWarning: assumptions concerning following solution(s)
                cannot be checked:""" + '\n\t' +
                ', '.join(str(s) for s in got_None)))

    #
    # done
    ###########################################################################

    if not solution:
        if as_set:
            return symbols, set()
        return []

    # make orderings canonical for list of dictionaries
    if not as_set:  # for set, no point in ordering
        solution = [{k: s[k] for k in ordered(s)} for s in solution]
        solution.sort(key=default_sort_key)

    if not (as_set or as_dict):
        return unpack(solution)

    if as_dict:
        return solution

    # set output: (symbols, {t1, t2, ...}) from list of dictionaries;
    # include all symbols for those that like a verbose solution
    # and to resolve any differences in dictionary keys.
    #
    # The set results can easily be used to make a verbose dict as
    #   k, v = solve(eqs, syms, set=True)
    #   sol = [dict(zip(k,i)) for i in v]
    #
    if ordered_symbols:
        k = symbols  # keep preferred order
    else:
        # just unify the symbols for which solutions were found
        k = list(ordered(set(flatten(tuple(i.keys()) for i in solution))))
    return k, {tuple([s.get(ki, ki) for ki in k]) for s in solution}


def _solve_undetermined(g, symbols, flags):
    """solve helper to return a list with one dict (solution) else None

    A direct call to solve_undetermined_coeffs is more flexible and
    can return both multiple solutions and handle more than one independent
    variable. Here, we have to be more cautious to keep from solving
    something that does not look like an undetermined coeffs system --
    to minimize the surprise factor since singularities that cancel are not
    prohibited in solve_undetermined_coeffs.
    """
    if g.free_symbols - set(symbols):
        sol = solve_undetermined_coeffs(g, symbols, **dict(flags, dict=True, set=None))
        if len(sol) == 1:
            return sol


def _solve(f, *symbols, **flags):
    """Return a checked solution for *f* in terms of one or more of the
    symbols in the form of a list of dictionaries.

    If no method is implemented to solve the equation, a NotImplementedError
    will be raised. In the case that conversion of an expression to a Poly
    gives None a ValueError will be raised.
    """

    not_impl_msg = "No algorithms are implemented to solve equation %s"

    if len(symbols) != 1:
        # look for solutions for desired symbols that are independent
        # of symbols already solved for, e.g. if we solve for x = y
        # then no symbol having x in its solution will be returned.

        # First solve for linear symbols (since that is easier and limits
        # solution size) and then proceed with symbols appearing
        # in a non-linear fashion. Ideally, if one is solving a single
        # expression for several symbols, they would have to be
        # appear in factors of an expression, but we do not here
        # attempt factorization.  XXX perhaps handling a Mul
        # should come first in this routine whether there is
        # one or several symbols.
        nonlin_s = []
        got_s = set()
        rhs_s = set()
        result = []
        for s in symbols:
            xi, v = solve_linear(f, symbols=[s])
            if xi == s:
                # no need to check but we should simplify if desired
                if flags.get('simplify', True):
                    v = simplify(v)
                vfree = v.free_symbols
                if vfree & got_s:
                    # was linear, but has redundant relationship
                    # e.g. x - y = 0 has y == x is redundant for x == y
                    # so ignore
                    continue
                rhs_s |= vfree
                got_s.add(xi)
                result.append({xi: v})
            elif xi:  # there might be a non-linear solution if xi is not 0
                nonlin_s.append(s)
        if not nonlin_s:
            return result
        for s in nonlin_s:
            try:
                soln = _solve(f, s, **flags)
                for sol in soln:
                    if sol[s].free_symbols & got_s:
                        # depends on previously solved symbols: ignore
                        continue
                    got_s.add(s)
                    result.append(sol)
            except NotImplementedError:
                continue
        if got_s:
            return result
        else:
            raise NotImplementedError(not_impl_msg % f)

    # solve f for a single variable

    symbol = symbols[0]

    # expand binomials only if it has the unknown symbol
    f = f.replace(lambda e: isinstance(e, binomial) and e.has(symbol),
        lambda e: expand_func(e))

    # checking will be done unless it is turned off before making a
    # recursive call; the variables `checkdens` and `check` are
    # captured here (for reference below) in case flag value changes
    flags['check'] = checkdens = check = flags.pop('check', True)

    # build up solutions if f is a Mul
    if f.is_Mul:
        result = set()
        for m in f.args:
            if m in {S.NegativeInfinity, S.ComplexInfinity, S.Infinity}:
                result = set()
                break
            soln = _vsolve(m, symbol, **flags)
            result.update(set(soln))
        result = [{symbol: v} for v in result]
        if check:
            # all solutions have been checked but now we must
            # check that the solutions do not set denominators
            # in any factor to zero
            dens = flags.get('_denominators', _simple_dens(f, symbols))
            result = [s for s in result if
                not any(checksol(den, s, **flags) for den in
                        dens)]
        # set flags for quick exit at end; solutions for each
        # factor were already checked and simplified
        check = False
        flags['simplify'] = False

    elif f.is_Piecewise:
        result = set()
        if any(e.is_zero for e, c in f.args):
            f = f.simplify()  # failure imminent w/o help

        cond = neg = True
        for expr, cnd in f.args:
            # the explicit condition for this expr is the current cond
            # and none of the previous conditions
            cond = And(neg, cnd)
            neg = And(neg, ~cond)

            if expr.is_zero and cond.simplify() != False:
                raise NotImplementedError(filldedent('''
                    An expression is already zero when %s.
                    This means that in this *region* the solution
                    is zero but solve can only represent discrete,
                    not interval, solutions. If this is a spurious
                    interval it might be resolved with simplification
                    of the Piecewise conditions.''' % cond))
            candidates = _vsolve(expr, symbol, **flags)

            for candidate in candidates:
                if candidate in result:
                    # an unconditional value was already there
                    continue
                try:
                    v = cond.subs(symbol, candidate)
                    _eval_simplify = getattr(v, '_eval_simplify', None)
                    if _eval_simplify is not None:
                        # unconditionally take the simplification of v
                        v = _eval_simplify(ratio=2, measure=lambda x: 1)
                except TypeError:
                    # incompatible type with condition(s)
                    continue
                if v == False:
                    continue
                if v == True:
                    result.add(candidate)
                else:
                    result.add(Piecewise(
                        (candidate, v),
                        (S.NaN, True)))
        # solutions already checked and simplified
        # ****************************************
        return [{symbol: r} for r in result]
    else:
        # first see if it really depends on symbol and whether there
        # is only a linear solution
        f_num, sol = solve_linear(f, symbols=symbols)
        if f_num.is_zero or sol is S.NaN:
            return []
        elif f_num.is_Symbol:
            # no need to check but simplify if desired
            if flags.get('simplify', True):
                sol = simplify(sol)
            return [{f_num: sol}]

        poly = None
        # check for a single Add generator
        if not f_num.is_Add:
            add_args = [i for i in f_num.atoms(Add)
                if symbol in i.free_symbols]
            if len(add_args) == 1:
                gen = add_args[0]
                spart = gen.as_independent(symbol)[1].as_base_exp()[0]
                if spart == symbol:
                    try:
                        poly = Poly(f_num, spart)
                    except PolynomialError:
                        pass

        result = False  # no solution was obtained
        msg = ''  # there is no failure message

        # Poly is generally robust enough to convert anything to
        # a polynomial and tell us the different generators that it
        # contains, so we will inspect the generators identified by
        # polys to figure out what to do.

        # try to identify a single generator that will allow us to solve this
        # as a polynomial, followed (perhaps) by a change of variables if the
        # generator is not a symbol

        try:
            if poly is None:
                poly = Poly(f_num)
            if poly is None:
                raise ValueError('could not convert %s to Poly' % f_num)
        except GeneratorsNeeded:
            simplified_f = simplify(f_num)
            if simplified_f != f_num:
                return _solve(simplified_f, symbol, **flags)
            raise ValueError('expression appears to be a constant')

        gens = [g for g in poly.gens if g.has(symbol)]

        def _as_base_q(x):
            """Return (b**e, q) for x = b**(p*e/q) where p/q is the leading
            Rational of the exponent of x, e.g. exp(-2*x/3) -> (exp(x), 3)
            """
            b, e = x.as_base_exp()
            if e.is_Rational:
                return b, e.q
            if not e.is_Mul:
                return x, 1
            c, ee = e.as_coeff_Mul()
            if c.is_Rational and c is not S.One:  # c could be a Float
                return b**ee, c.q
            return x, 1

        if len(gens) > 1:
            # If there is more than one generator, it could be that the
            # generators have the same base but different powers, e.g.
            #   >>> Poly(exp(x) + 1/exp(x))
            #   Poly(exp(-x) + exp(x), exp(-x), exp(x), domain='ZZ')
            #
            # If unrad was not disabled then there should be no rational
            # exponents appearing as in
            #   >>> Poly(sqrt(x) + sqrt(sqrt(x)))
            #   Poly(sqrt(x) + x**(1/4), sqrt(x), x**(1/4), domain='ZZ')

            bases, qs = list(zip(*[_as_base_q(g) for g in gens]))
            bases = set(bases)

            if len(bases) > 1 or not all(q == 1 for q in qs):
                funcs = {b for b in bases if b.is_Function}

                trig = {_ for _ in funcs if
                    isinstance(_, TrigonometricFunction)}
                other = funcs - trig
                if not other and len(funcs.intersection(trig)) > 1:
                    newf = None
                    if f_num.is_Add and len(f_num.args) == 2:
                        # check for sin(x)**p = cos(x)**p
                        _args = f_num.args
                        t = a, b = [i.atoms(Function).intersection(
                            trig) for i in _args]
                        if all(len(i) == 1 for i in t):
                            a, b = [i.pop() for i in t]
                            if isinstance(a, cos):
                                a, b = b, a
                                _args = _args[::-1]
                            if isinstance(a, sin) and isinstance(b, cos
                                    ) and a.args[0] == b.args[0]:
                                # sin(x) + cos(x) = 0 -> tan(x) + 1 = 0
                                newf, _d = (TR2i(_args[0]/_args[1]) + 1
                                    ).as_numer_denom()
                                if not _d.is_Number:
                                    newf = None
                    if newf is None:
                        newf = TR1(f_num).rewrite(tan)
                    if newf != f_num:
                        # don't check the rewritten form --check
                        # solutions in the un-rewritten form below
                        flags['check'] = False
                        result = _solve(newf, symbol, **flags)
                        flags['check'] = check

                # just a simple case - see if replacement of single function
                # clears all symbol-dependent functions, e.g.
                # log(x) - log(log(x) - 1) - 3 can be solved even though it has
                # two generators.

                if result is False and funcs:
                    funcs = list(ordered(funcs))  # put shallowest function first
                    f1 = funcs[0]
                    t = Dummy('t')
                    # perform the substitution
                    ftry = f_num.subs(f1, t)

                    # if no Functions left, we can proceed with usual solve
                    if not ftry.has(symbol):
                        cv_sols = _solve(ftry, t, **flags)
                        cv_inv = list(ordered(_vsolve(t - f1, symbol, **flags)))[0]
                        result = [{symbol: cv_inv.subs(sol)} for sol in cv_sols]

                if result is False:
                    msg = 'multiple generators %s' % gens

            else:
                # e.g. case where gens are exp(x), exp(-x)
                u = bases.pop()
                t = Dummy('t')
                inv = _vsolve(u - t, symbol, **flags)
                if isinstance(u, (Pow, exp)):
                    # this will be resolved by factor in _tsolve but we might
                    # as well try a simple expansion here to get things in
                    # order so something like the following will work now without
                    # having to factor:
                    #
                    # >>> eq = (exp(I*(-x-2))+exp(I*(x+2)))
                    # >>> eq.subs(exp(x),y)  # fails
                    # exp(I*(-x - 2)) + exp(I*(x + 2))
                    # >>> eq.expand().subs(exp(x),y)  # works
                    # y**I*exp(2*I) + y**(-I)*exp(-2*I)
                    def _expand(p):
                        b, e = p.as_base_exp()
                        e = expand_mul(e)
                        return expand_power_exp(b**e)
                    ftry = f_num.replace(
                        lambda w: w.is_Pow or isinstance(w, exp),
                        _expand).subs(u, t)
                    if not ftry.has(symbol):
                        soln = _solve(ftry, t, **flags)
                        result = [{symbol: i.subs(s)} for i in inv for s in soln]

        elif len(gens) == 1:

            # There is only one generator that we are interested in, but
            # there may have been more than one generator identified by
            # polys (e.g. for symbols other than the one we are interested
            # in) so recast the poly in terms of our generator of interest.
            # Also use composite=True with f_num since Poly won't update
            # poly as documented in issue 8810.

            poly = Poly(f_num, gens[0], composite=True)

            # if we aren't on the tsolve-pass, use roots
            if not flags.pop('tsolve', False):
                soln = None
                deg = poly.degree()
                flags['tsolve'] = True
                hints = ('cubics', 'quartics', 'quintics')
                solvers = {h: flags.get(h) for h in hints}
                soln = roots(poly, **solvers)
                if sum(soln.values()) < deg:
                    # e.g. roots(32*x**5 + 400*x**4 + 2032*x**3 +
                    #            5000*x**2 + 6250*x + 3189) -> {}
                    # so all_roots is used and RootOf instances are
                    # returned *unless* the system is multivariate
                    # or high-order EX domain.
                    try:
                        soln = poly.all_roots()
                    except NotImplementedError:
                        if not flags.get('incomplete', True):
                                raise NotImplementedError(
                                filldedent('''
    Neither high-order multivariate polynomials
    nor sorting of EX-domain polynomials is supported.
    If you want to see any results, pass keyword incomplete=True to
    solve; to see numerical values of roots
    for univariate expressions, use nroots.
    '''))
                        else:
                            pass
                else:
                    soln = list(soln.keys())

                if soln is not None:
                    u = poly.gen
                    if u != symbol:
                        try:
                            t = Dummy('t')
                            inv = _vsolve(u - t, symbol, **flags)
                            soln = {i.subs(t, s) for i in inv for s in soln}
                        except NotImplementedError:
                            # perhaps _tsolve can handle f_num
                            soln = None
                    else:
                        check = False  # only dens need to be checked
                    if soln is not None:
                        if len(soln) > 2:
                            # if the flag wasn't set then unset it since high-order
                            # results are quite long. Perhaps one could base this
                            # decision on a certain critical length of the
                            # roots. In addition, wester test M2 has an expression
                            # whose roots can be shown to be real with the
                            # unsimplified form of the solution whereas only one of
                            # the simplified forms appears to be real.
                            flags['simplify'] = flags.get('simplify', False)
                if soln is not None:
                    result = [{symbol: v} for v in soln]

    # fallback if above fails
    # -----------------------
    if result is False:
        # try unrad
        if flags.pop('_unrad', True):
            try:
                u = unrad(f_num, symbol)
            except (ValueError, NotImplementedError):
                u = False
            if u:
                eq, cov = u
                if cov:
                    isym, ieq = cov
                    inv = _vsolve(ieq, symbol, **flags)[0]
                    rv = {inv.subs(xi) for xi in _solve(eq, isym, **flags)}
                else:
                    try:
                        rv = set(_vsolve(eq, symbol, **flags))
                    except NotImplementedError:
                        rv = None
                if rv is not None:
                    result = [{symbol: v} for v in rv]
                    # if the flag wasn't set then unset it since unrad results
                    # can be quite long or of very high order
                    flags['simplify'] = flags.get('simplify', False)
            else:
                pass  # for coverage

    # try _tsolve
    if result is False:
        flags.pop('tsolve', None)  # allow tsolve to be used on next pass
        try:
            soln = _tsolve(f_num, symbol, **flags)
            if soln is not None:
                result = [{symbol: v} for v in soln]
        except PolynomialError:
            pass
    # ----------- end of fallback ----------------------------

    if result is False:
        raise NotImplementedError('\n'.join([msg, not_impl_msg % f]))

    result = _remove_duplicate_solutions(result)

    if flags.get('simplify', True):
        result = [{k: d[k].simplify() for k in d} for d in result]
        # Simplification might reveal more duplicates
        result = _remove_duplicate_solutions(result)
        # we just simplified the solution so we now set the flag to
        # False so the simplification doesn't happen again in checksol()
        flags['simplify'] = False

    if checkdens:
        # reject any result that makes any denom. affirmatively 0;
        # if in doubt, keep it
        dens = _simple_dens(f, symbols)
        result = [r for r in result if
                  not any(checksol(d, r, **flags)
                          for d in dens)]
    if check:
        # keep only results if the check is not False
        result = [r for r in result if
                  checksol(f_num, r, **flags) is not False]
    return result


def _remove_duplicate_solutions(solutions: list[dict[Expr, Expr]]
                                ) -> list[dict[Expr, Expr]]:
    """Remove duplicates from a list of dicts"""
    solutions_set = set()
    solutions_new = []

    for sol in solutions:
        solset = frozenset(sol.items())
        if solset not in solutions_set:
            solutions_new.append(sol)
            solutions_set.add(solset)

    return solutions_new


def _solve_system(exprs, symbols, **flags):
    """return ``(linear, solution)`` where ``linear`` is True
    if the system was linear, else False; ``solution``
    is a list of dictionaries giving solutions for the symbols
    """
    if not exprs:
        return False, []

    if flags.pop('_split', True):
        # Split the system into connected components
        V = exprs
        symsset = set(symbols)
        exprsyms = {e: e.free_symbols & symsset for e in exprs}
        E = []
        sym_indices = {sym: i for i, sym in enumerate(symbols)}
        for n, e1 in enumerate(exprs):
            for e2 in exprs[:n]:
                # Equations are connected if they share a symbol
                if exprsyms[e1] & exprsyms[e2]:
                    E.append((e1, e2))
        G = V, E
        subexprs = connected_components(G)
        if len(subexprs) > 1:
            subsols = []
            linear = True
            for subexpr in subexprs:
                subsyms = set()
                for e in subexpr:
                    subsyms |= exprsyms[e]
                subsyms = sorted(subsyms, key = lambda x: sym_indices[x])
                flags['_split'] = False  # skip split step
                _linear, subsol = _solve_system(subexpr, subsyms, **flags)
                if linear:
                    linear = linear and _linear
                if not isinstance(subsol, list):
                    subsol = [subsol]
                subsols.append(subsol)
            # Full solution is cartesian product of subsystems
            sols = []
            for soldicts in product(*subsols):
                sols.append(dict(item for sd in soldicts
                    for item in sd.items()))
            return linear, sols

    polys = []
    dens = set()
    failed = []
    result = []
    solved_syms = []
    linear = True
    manual = flags.get('manual', False)
    checkdens = check = flags.get('check', True)

    for j, g in enumerate(exprs):
        dens.update(_simple_dens(g, symbols))
        i, d = _invert(g, *symbols)
        if d in symbols:
            if linear:
                linear = solve_linear(g, 0, [d])[0] == d
        g = d - i
        g = g.as_numer_denom()[0]
        if manual:
            failed.append(g)
            continue

        poly = g.as_poly(*symbols, extension=True)

        if poly is not None:
            polys.append(poly)
        else:
            failed.append(g)

    if polys:
        if all(p.is_linear for p in polys):
            n, m = len(polys), len(symbols)
            matrix = zeros(n, m + 1)

            for i, poly in enumerate(polys):
                for monom, coeff in poly.terms():
                    try:
                        j = monom.index(1)
                        matrix[i, j] = coeff
                    except ValueError:
                        matrix[i, m] = -coeff

            # returns a dictionary ({symbols: values}) or None
            if flags.pop('particular', False):
                result = minsolve_linear_system(matrix, *symbols, **flags)
            else:
                result = solve_linear_system(matrix, *symbols, **flags)
            result = [result] if result else []
            if failed:
                if result:
                    solved_syms = list(result[0].keys())  # there is only one result dict
                else:
                    solved_syms = []
            # linear doesn't change
        else:
            linear = False
            if len(symbols) > len(polys):

                free = set().union(*[p.free_symbols for p in polys])
                free = list(ordered(free.intersection(symbols)))
                got_s = set()
                result = []
                for syms in subsets(free, min(len(free), len(polys))):
                    try:
                        # returns [], None or list of tuples
                        res = solve_poly_system(polys, *syms)
                        if res:
                            for r in set(res):
                                skip = False
                                for r1 in r:
                                    if got_s and any(ss in r1.free_symbols
                                           for ss in got_s):
                                        # sol depends on previously
                                        # solved symbols: discard it
                                        skip = True
                                if not skip:
                                    got_s.update(syms)
                                    result.append(dict(list(zip(syms, r))))
                    except NotImplementedError:
                        pass
                if got_s:
                    solved_syms = list(got_s)
                else:
                    failed.extend([g.as_expr() for g in polys])
            else:
                try:
                    result = solve_poly_system(polys, *symbols)
                    if result:
                        solved_syms = symbols
                        result = [dict(list(zip(solved_syms, r))) for r in set(result)]
                except NotImplementedError:
                    failed.extend([g.as_expr() for g in polys])
                    solved_syms = []

    # convert None or [] to [{}]
    result = result or [{}]

    if failed:
        linear = False
        # For each failed equation, see if we can solve for one of the
        # remaining symbols from that equation. If so, we update the
        # solution set and continue with the next failed equation,
        # repeating until we are done or we get an equation that can't
        # be solved.
        def _ok_syms(e, sort=False):
            rv = e.free_symbols & legal

            # Solve first for symbols that have lower degree in the equation.
            # Ideally we want to solve firstly for symbols that appear linearly
            # with rational coefficients e.g. if e = x*y + z then we should
            # solve for z first.
            def key(sym):
                ep = e.as_poly(sym)
                if ep is None:
                    complexity = (S.Infinity, S.Infinity, S.Infinity)
                else:
                    coeff_syms = ep.LC().free_symbols
                    complexity = (ep.degree(), len(coeff_syms & rv), len(coeff_syms))
                return complexity + (default_sort_key(sym),)

            if sort:
                rv = sorted(rv, key=key)
            return rv

        legal = set(symbols)  # what we are interested in
        # sort so equation with the fewest potential symbols is first
        u = Dummy()  # used in solution checking
        for eq in ordered(failed, lambda _: len(_ok_syms(_))):
            newresult = []
            bad_results = []
            hit = False
            for r in result:
                got_s = set()
                # update eq with everything that is known so far
                eq2 = eq.subs(r)
                # if check is True then we see if it satisfies this
                # equation, otherwise we just accept it
                if check and r:
                    b = checksol(u, u, eq2, minimal=True)
                    if b is not None:
                        # this solution is sufficient to know whether
                        # it is valid or not so we either accept or
                        # reject it, then continue
                        if b:
                            newresult.append(r)
                        else:
                            bad_results.append(r)
                        continue
                # search for a symbol amongst those available that
                # can be solved for
                ok_syms = _ok_syms(eq2, sort=True)
                if not ok_syms:
                    if r:
                        newresult.append(r)
                    break  # skip as it's independent of desired symbols
                for s in ok_syms:
                    try:
                        soln = _vsolve(eq2, s, **flags)
                    except NotImplementedError:
                        continue
                    # put each solution in r and append the now-expanded
                    # result in the new result list; use copy since the
                    # solution for s is being added in-place
                    for sol in soln:
                        if got_s and any(ss in sol.free_symbols for ss in got_s):
                            # sol depends on previously solved symbols: discard it
                            continue
                        rnew = r.copy()
                        for k, v in r.items():
                            rnew[k] = v.subs(s, sol)
                        # and add this new solution
                        rnew[s] = sol
                        # check that it is independent of previous solutions
                        iset = set(rnew.items())
                        for i in newresult:
                            if len(i) < len(iset):
                                # update i with what is known
                                i_items_updated = {(k, v.xreplace(rnew)) for k, v in i.items()}
                                if not i_items_updated - iset:
                                    # this is a superset of a known solution that
                                    # is smaller
                                    break
                        else:
                            # keep it
                            newresult.append(rnew)
                    hit = True
                    got_s.add(s)
                if not hit:
                    raise NotImplementedError('could not solve %s' % eq2)
            else:
                result = newresult
                for b in bad_results:
                    if b in result:
                        result.remove(b)

    if not result:
        return False, []

    # rely on linear/polynomial system solvers to simplify
    # XXX the following tests show that the expressions
    # returned are not the same as they would be if simplify
    # were applied to this:
    #   sympy/solvers/ode/tests/test_systems/test__classify_linear_system
    #   sympy/solvers/tests/test_solvers/test_issue_4886
    # so the docs should be updated to reflect that or else
    # the following should be `bool(failed) or not linear`
    default_simplify = bool(failed)
    if flags.get('simplify', default_simplify):
        for r in result:
            for k in r:
                r[k] = simplify(r[k])
        flags['simplify'] = False  # don't need to do so in checksol now

    if checkdens:
        result = [r for r in result
            if not any(checksol(d, r, **flags) for d in dens)]

    if check and not linear:
        result = [r for r in result
            if not any(checksol(e, r, **flags) is False for e in exprs)]

    result = [r for r in result if r]
    return linear, result


def solve_linear(lhs, rhs=0, symbols=[], exclude=[]):
    r"""
    Return a tuple derived from ``f = lhs - rhs`` that is one of
    the following: ``(0, 1)``, ``(0, 0)``, ``(symbol, solution)``, ``(n, d)``.

    Explanation
    ===========

    ``(0, 1)`` meaning that ``f`` is independent of the symbols in *symbols*
    that are not in *exclude*.

    ``(0, 0)`` meaning that there is no solution to the equation amongst the
    symbols given. If the first element of the tuple is not zero, then the
    function is guaranteed to be dependent on a symbol in *symbols*.

    ``(symbol, solution)`` where symbol appears linearly in the numerator of
    ``f``, is in *symbols* (if given), and is not in *exclude* (if given). No
    simplification is done to ``f`` other than a ``mul=True`` expansion, so the
    solution will correspond strictly to a unique solution.

    ``(n, d)`` where ``n`` and ``d`` are the numerator and denominator of ``f``
    when the numerator was not linear in any symbol of interest; ``n`` will
    never be a symbol unless a solution for that symbol was found (in which case
    the second element is the solution, not the denominator).

    Examples
    ========

    >>> from sympy import cancel, Pow

    ``f`` is independent of the symbols in *symbols* that are not in
    *exclude*:

    >>> from sympy import cos, sin, solve_linear
    >>> from sympy.abc import x, y, z
    >>> eq = y*cos(x)**2 + y*sin(x)**2 - y  # = y*(1 - 1) = 0
    >>> solve_linear(eq)
    (0, 1)
    >>> eq = cos(x)**2 + sin(x)**2  # = 1
    >>> solve_linear(eq)
    (0, 1)
    >>> solve_linear(x, exclude=[x])
    (0, 1)

    The variable ``x`` appears as a linear variable in each of the
    following:

    >>> solve_linear(x + y**2)
    (x, -y**2)
    >>> solve_linear(1/x - y**2)
    (x, y**(-2))

    When not linear in ``x`` or ``y`` then the numerator and denominator are
    returned:

    >>> solve_linear(x**2/y**2 - 3)
    (x**2 - 3*y**2, y**2)

    If the numerator of the expression is a symbol, then ``(0, 0)`` is
    returned if the solution for that symbol would have set any
    denominator to 0:

    >>> eq = 1/(1/x - 2)
    >>> eq.as_numer_denom()
    (x, 1 - 2*x)
    >>> solve_linear(eq)
    (0, 0)

    But automatic rewriting may cause a symbol in the denominator to
    appear in the numerator so a solution will be returned:

    >>> (1/x)**-1
    x
    >>> solve_linear((1/x)**-1)
    (x, 0)

    Use an unevaluated expression to avoid this:

    >>> solve_linear(Pow(1/x, -1, evaluate=False))
    (0, 0)

    If ``x`` is allowed to cancel in the following expression, then it
    appears to be linear in ``x``, but this sort of cancellation is not
    done by ``solve_linear`` so the solution will always satisfy the
    original expression without causing a division by zero error.

    >>> eq = x**2*(1/x - z**2/x)
    >>> solve_linear(cancel(eq))
    (x, 0)
    >>> solve_linear(eq)
    (x**2*(1 - z**2), x)

    A list of symbols for which a solution is desired may be given:

    >>> solve_linear(x + y + z, symbols=[y])
    (y, -x - z)

    A list of symbols to ignore may also be given:

    >>> solve_linear(x + y + z, exclude=[x])
    (y, -x - z)

    (A solution for ``y`` is obtained because it is the first variable
    from the canonically sorted list of symbols that had a linear
    solution.)

    """
    if isinstance(lhs, Eq):
        if rhs:
            raise ValueError(filldedent('''
            If lhs is an Equality, rhs must be 0 but was %s''' % rhs))
        rhs = lhs.rhs
        lhs = lhs.lhs
    dens = None
    eq = lhs - rhs
    n, d = eq.as_numer_denom()
    if not n:
        return S.Zero, S.One

    free = n.free_symbols
    if not symbols:
        symbols = free
    else:
        bad = [s for s in symbols if not s.is_Symbol]
        if bad:
            if len(bad) == 1:
                bad = bad[0]
            if len(symbols) == 1:
                eg = 'solve(%s, %s)' % (eq, symbols[0])
            else:
                eg = 'solve(%s, *%s)' % (eq, list(symbols))
            raise ValueError(filldedent('''
                solve_linear only handles symbols, not %s. To isolate
                non-symbols use solve, e.g. >>> %s <<<.
                             ''' % (bad, eg)))
        symbols = free.intersection(symbols)
    symbols = symbols.difference(exclude)
    if not symbols:
        return S.Zero, S.One

    # derivatives are easy to do but tricky to analyze to see if they
    # are going to disallow a linear solution, so for simplicity we
    # just evaluate the ones that have the symbols of interest
    derivs = defaultdict(list)
    for der in n.atoms(Derivative):
        csym = der.free_symbols & symbols
        for c in csym:
            derivs[c].append(der)

    all_zero = True
    for xi in sorted(symbols, key=default_sort_key):  # canonical order
        # if there are derivatives in this var, calculate them now
        if isinstance(derivs[xi], list):
            derivs[xi] = {der: der.doit() for der in derivs[xi]}
        newn = n.subs(derivs[xi])
        dnewn_dxi = newn.diff(xi)
        # dnewn_dxi can be nonzero if it survives differentation by any
        # of its free symbols
        free = dnewn_dxi.free_symbols
        if dnewn_dxi and (not free or any(dnewn_dxi.diff(s) for s in free) or free == symbols):
            all_zero = False
            if dnewn_dxi is S.NaN:
                break
            if xi not in dnewn_dxi.free_symbols:
                vi = -1/dnewn_dxi*(newn.subs(xi, 0))
                if dens is None:
                    dens = _simple_dens(eq, symbols)
                if not any(checksol(di, {xi: vi}, minimal=True) is True
                          for di in dens):
                    # simplify any trivial integral
                    irep = [(i, i.doit()) for i in vi.atoms(Integral) if
                            i.function.is_number]
                    # do a slight bit of simplification
                    vi = expand_mul(vi.subs(irep))
                    return xi, vi
    if all_zero:
        return S.Zero, S.One
    if n.is_Symbol: # no solution for this symbol was found
        return S.Zero, S.Zero
    return n, d


def minsolve_linear_system(system, *symbols, **flags):
    r"""
    Find a particular solution to a linear system.

    Explanation
    ===========

    In particular, try to find a solution with the minimal possible number
    of non-zero variables using a naive algorithm with exponential complexity.
    If ``quick=True``, a heuristic is used.

    """
    quick = flags.get('quick', False)
    # Check if there are any non-zero solutions at all
    s0 = solve_linear_system(system, *symbols, **flags)
    if not s0 or all(v == 0 for v in s0.values()):
        return s0
    if quick:
        # We just solve the system and try to heuristically find a nice
        # solution.
        s = solve_linear_system(system, *symbols)
        def update(determined, solution):
            delete = []
            for k, v in solution.items():
                solution[k] = v.subs(determined)
                if not solution[k].free_symbols:
                    delete.append(k)
                    determined[k] = solution[k]
            for k in delete:
                del solution[k]
        determined = {}
        update(determined, s)
        while s:
            # NOTE sort by default_sort_key to get deterministic result
            k = max((k for k in s.values()),
                    key=lambda x: (len(x.free_symbols), default_sort_key(x)))
            kfree = k.free_symbols
            x = next(reversed(list(ordered(kfree))))
            if len(kfree) != 1:
                determined[x] = S.Zero
            else:
                val = _vsolve(k, x, check=False)[0]
                if not val and not any(v.subs(x, val) for v in s.values()):
                    determined[x] = S.One
                else:
                    determined[x] = val
            update(determined, s)
        return determined
    else:
        # We try to select n variables which we want to be non-zero.
        # All others will be assumed zero. We try to solve the modified system.
        # If there is a non-trivial solution, just set the free variables to
        # one. If we do this for increasing n, trying all combinations of
        # variables, we will find an optimal solution.
        # We speed up slightly by starting at one less than the number of
        # variables the quick method manages.
        N = len(symbols)
        bestsol = minsolve_linear_system(system, *symbols, quick=True)
        n0 = len([x for x in bestsol.values() if x != 0])
        for n in range(n0 - 1, 1, -1):
            debugf('minsolve: %s', n)
            thissol = None
            for nonzeros in combinations(range(N), n):
                subm = Matrix([system.col(i).T for i in nonzeros] + [system.col(-1).T]).T
                s = solve_linear_system(subm, *[symbols[i] for i in nonzeros])
                if s and not all(v == 0 for v in s.values()):
                    subs = [(symbols[v], S.One) for v in nonzeros]
                    for k, v in s.items():
                        s[k] = v.subs(subs)
                    for sym in symbols:
                        if sym not in s:
                            if symbols.index(sym) in nonzeros:
                                s[sym] = S.One
                            else:
                                s[sym] = S.Zero
                    thissol = s
                    break
            if thissol is None:
                break
            bestsol = thissol
        return bestsol


def solve_linear_system(system, *symbols, **flags):
    r"""
    Solve system of $N$ linear equations with $M$ variables, which means
    both under- and overdetermined systems are supported.

    Explanation
    ===========

    The possible number of solutions is zero, one, or infinite. Respectively,
    this procedure will return None or a dictionary with solutions. In the
    case of underdetermined systems, all arbitrary parameters are skipped.
    This may cause a situation in which an empty dictionary is returned.
    In that case, all symbols can be assigned arbitrary values.

    Input to this function is a $N\times M + 1$ matrix, which means it has
    to be in augmented form. If you prefer to enter $N$ equations and $M$
    unknowns then use ``solve(Neqs, *Msymbols)`` instead. Note: a local
    copy of the matrix is made by this routine so the matrix that is
    passed will not be modified.

    The algorithm used here is fraction-free Gaussian elimination,
    which results, after elimination, in an upper-triangular matrix.
    Then solutions are found using back-substitution. This approach
    is more efficient and compact than the Gauss-Jordan method.

    Examples
    ========

    >>> from sympy import Matrix, solve_linear_system
    >>> from sympy.abc import x, y

    Solve the following system::

           x + 4 y ==  2
        -2 x +   y == 14

    >>> system = Matrix(( (1, 4, 2), (-2, 1, 14)))
    >>> solve_linear_system(system, x, y)
    {x: -6, y: 2}

    A degenerate system returns an empty dictionary:

    >>> system = Matrix(( (0,0,0), (0,0,0) ))
    >>> solve_linear_system(system, x, y)
    {}

    """
    assert system.shape[1] == len(symbols) + 1

    # This is just a wrapper for solve_lin_sys
    eqs = list(system * Matrix(symbols + (-1,)))
    eqs, ring = sympy_eqs_to_ring(eqs, symbols)
    sol = solve_lin_sys(eqs, ring, _raw=False)
    if sol is not None:
        sol = {sym:val for sym, val in sol.items() if sym != val}
    return sol


def solve_undetermined_coeffs(equ, coeffs, *syms, **flags):
    r"""
    Solve a system of equations in $k$ parameters that is formed by
    matching coefficients in variables ``coeffs`` that are on
    factors dependent on the remaining variables (or those given
    explicitly by ``syms``.

    Explanation
    ===========

    The result of this function is a dictionary with symbolic values of those
    parameters with respect to coefficients in $q$ -- empty if there
    is no solution or coefficients do not appear in the equation -- else
    None (if the system was not recognized). If there is more than one
    solution, the solutions are passed as a list. The output can be modified using
    the same semantics as for `solve` since the flags that are passed are sent
    directly to `solve` so, for example the flag ``dict=True`` will always return a list
    of solutions as dictionaries.

    This function accepts both Equality and Expr class instances.
    The solving process is most efficient when symbols are specified
    in addition to parameters to be determined,  but an attempt to
    determine them (if absent) will be made. If an expected solution is not
    obtained (and symbols were not specified) try specifying them.

    Examples
    ========

    >>> from sympy import Eq, solve_undetermined_coeffs
    >>> from sympy.abc import a, b, c, h, p, k, x, y

    >>> solve_undetermined_coeffs(Eq(a*x + a + b, x/2), [a, b], x)
    {a: 1/2, b: -1/2}
    >>> solve_undetermined_coeffs(a - 2, [a])
    {a: 2}

    The equation can be nonlinear in the symbols:

    >>> X, Y, Z = y, x**y, y*x**y
    >>> eq = a*X + b*Y + c*Z - X - 2*Y - 3*Z
    >>> coeffs = a, b, c
    >>> syms = x, y
    >>> solve_undetermined_coeffs(eq, coeffs, syms)
    {a: 1, b: 2, c: 3}

    And the system can be nonlinear in coefficients, too, but if
    there is only a single solution, it will be returned as a
    dictionary:

    >>> eq = a*x**2 + b*x + c - ((x - h)**2 + 4*p*k)/4/p
    >>> solve_undetermined_coeffs(eq, (h, p, k), x)
    {h: -b/(2*a), k: (4*a*c - b**2)/(4*a), p: 1/(4*a)}

    Multiple solutions are always returned in a list:

    >>> solve_undetermined_coeffs(a**2*x + b - x, [a, b], x)
    [{a: -1, b: 0}, {a: 1, b: 0}]

    Using flag ``dict=True`` (in keeping with semantics in :func:`~.solve`)
    will force the result to always be a list with any solutions
    as elements in that list.

    >>> solve_undetermined_coeffs(a*x - 2*x, [a], dict=True)
    [{a: 2}]
    """
    if not (coeffs and all(i.is_Symbol for i in coeffs)):
        raise ValueError('must provide symbols for coeffs')

    if isinstance(equ, Eq):
        eq = equ.lhs - equ.rhs
    else:
        eq = equ

    ceq = cancel(eq)
    xeq = _mexpand(ceq.as_numer_denom()[0], recursive=True)

    free = xeq.free_symbols
    coeffs = free & set(coeffs)
    if not coeffs:
        return ([], {}) if flags.get('set', None) else []  # solve(0, x) -> []

    if not syms:
        # e.g. A*exp(x) + B - (exp(x) + y) separated into parts that
        # don't/do depend on coeffs gives
        # -(exp(x) + y), A*exp(x) + B
        # then see what symbols are common to both
        # {x} = {x, A, B} - {x, y}
        ind, dep = xeq.as_independent(*coeffs, as_Add=True)
        dfree = dep.free_symbols
        syms = dfree & ind.free_symbols
        if not syms:
            # but if the system looks like (a + b)*x + b - c
            # then {} = {a, b, x} - c
            # so calculate {x} = {a, b, x} - {a, b}
            syms = dfree - set(coeffs)
        if not syms:
            syms = [Dummy()]
    else:
        if len(syms) == 1 and iterable(syms[0]):
            syms = syms[0]
        e, s, _ = recast_to_symbols([xeq], syms)
        xeq = e[0]
        syms = s

    # find the functional forms in which symbols appear

    gens = set(xeq.as_coefficients_dict(*syms).keys()) - {1}
    cset = set(coeffs)
    if any(g.has_xfree(cset) for g in gens):
        return  # a generator contained a coefficient symbol

    # make sure we are working with symbols for generators

    e, gens, _ = recast_to_symbols([xeq], list(gens))
    xeq = e[0]

    # collect coefficients in front of generators

    system = list(collect(xeq, gens, evaluate=False).values())

    # get a solution

    soln = solve(system, coeffs, **flags)

    # unpack unless told otherwise if length is 1

    settings = flags.get('dict', None) or flags.get('set', None)
    if type(soln) is dict or settings or len(soln) != 1:
        return soln
    return soln[0]


def solve_linear_system_LU(matrix, syms):
    """
    Solves the augmented matrix system using ``LUsolve`` and returns a
    dictionary in which solutions are keyed to the symbols of *syms* as ordered.

    Explanation
    ===========

    The matrix must be invertible.

    Examples
    ========

    >>> from sympy import Matrix, solve_linear_system_LU
    >>> from sympy.abc import x, y, z

    >>> solve_linear_system_LU(Matrix([
    ... [1, 2, 0, 1],
    ... [3, 2, 2, 1],
    ... [2, 0, 0, 1]]), [x, y, z])
    {x: 1/2, y: 1/4, z: -1/2}

    See Also
    ========

    LUsolve

    """
    if matrix.rows != matrix.cols - 1:
        raise ValueError("Rows should be equal to columns - 1")
    A = matrix[:matrix.rows, :matrix.rows]
    b = matrix[:, matrix.cols - 1:]
    soln = A.LUsolve(b)
    solutions = {}
    for i in range(soln.rows):
        solutions[syms[i]] = soln[i, 0]
    return solutions


def det_perm(M):
    """
    Return the determinant of *M* by using permutations to select factors.

    Explanation
    ===========

    For sizes larger than 8 the number of permutations becomes prohibitively
    large, or if there are no symbols in the matrix, it is better to use the
    standard determinant routines (e.g., ``M.det()``.)

    See Also
    ========

    det_minor
    det_quick

    """
    args = []
    s = True
    n = M.rows
    list_ = M.flat()
    for perm in generate_bell(n):
        fac = []
        idx = 0
        for j in perm:
            fac.append(list_[idx + j])
            idx += n
        term = Mul(*fac) # disaster with unevaluated Mul -- takes forever for n=7
        args.append(term if s else -term)
        s = not s
    return Add(*args)


def det_minor(M):
    """
    Return the ``det(M)`` computed from minors without
    introducing new nesting in products.

    See Also
    ========

    det_perm
    det_quick

    """
    n = M.rows
    if n == 2:
        return M[0, 0]*M[1, 1] - M[1, 0]*M[0, 1]
    else:
        return sum((1, -1)[i % 2]*Add(*[M[0, i]*d for d in
            Add.make_args(det_minor(M.minor_submatrix(0, i)))])
            if M[0, i] else S.Zero for i in range(n))


def det_quick(M, method=None):
    """
    Return ``det(M)`` assuming that either
    there are lots of zeros or the size of the matrix
    is small. If this assumption is not met, then the normal
    Matrix.det function will be used with method = ``method``.

    See Also
    ========

    det_minor
    det_perm

    """
    if any(i.has(Symbol) for i in M):
        if M.rows < 8 and all(i.has(Symbol) for i in M):
            return det_perm(M)
        return det_minor(M)
    else:
        return M.det(method=method) if method else M.det()


def inv_quick(M):
    """Return the inverse of ``M``, assuming that either
    there are lots of zeros or the size of the matrix
    is small.
    """
    if not all(i.is_Number for i in M):
        if not any(i.is_Number for i in M):
            det = lambda _: det_perm(_)
        else:
            det = lambda _: det_minor(_)
    else:
        return M.inv()
    n = M.rows
    d = det(M)
    if d == S.Zero:
        raise NonInvertibleMatrixError("Matrix det == 0; not invertible")
    ret = zeros(n)
    s1 = -1
    for i in range(n):
        s = s1 = -s1
        for j in range(n):
            di = det(M.minor_submatrix(i, j))
            ret[j, i] = s*di/d
            s = -s
    return ret


# these are functions that have multiple inverse values per period
multi_inverses = {
    sin: lambda x: (asin(x), S.Pi - asin(x)),
    cos: lambda x: (acos(x), 2*S.Pi - acos(x)),
}


def _vsolve(e, s, **flags):
    """return list of scalar values for the solution of e for symbol s"""
    return [i[s] for i in _solve(e, s, **flags)]


def _tsolve(eq, sym, **flags):
    """
    Helper for ``_solve`` that solves a transcendental equation with respect
    to the given symbol. Various equations containing powers and logarithms,
    can be solved.

    There is currently no guarantee that all solutions will be returned or
    that a real solution will be favored over a complex one.

    Either a list of potential solutions will be returned or None will be
    returned (in the case that no method was known to get a solution
    for the equation). All other errors (like the inability to cast an
    expression as a Poly) are unhandled.

    Examples
    ========

    >>> from sympy import log, ordered
    >>> from sympy.solvers.solvers import _tsolve as tsolve
    >>> from sympy.abc import x

    >>> list(ordered(tsolve(3**(2*x + 5) - 4, x)))
    [-5/2 + log(2)/log(3), (-5*log(3)/2 + log(2) + I*pi)/log(3)]

    >>> tsolve(log(x) + 2*x, x)
    [LambertW(2)/2]

    """
    if 'tsolve_saw' not in flags:
        flags['tsolve_saw'] = []
    if eq in flags['tsolve_saw']:
        return None
    else:
        flags['tsolve_saw'].append(eq)

    rhs, lhs = _invert(eq, sym)

    if lhs == sym:
        return [rhs]
    try:
        if lhs.is_Add:
            # it's time to try factoring; powdenest is used
            # to try get powers in standard form for better factoring
            f = factor(powdenest(lhs - rhs))
            if f.is_Mul:
                return _vsolve(f, sym, **flags)
            if rhs:
                f = logcombine(lhs, force=flags.get('force', True))
                if f.count(log) != lhs.count(log):
                    if isinstance(f, log):
                        return _vsolve(f.args[0] - exp(rhs), sym, **flags)
                    return _tsolve(f - rhs, sym, **flags)

        elif lhs.is_Pow:
            if lhs.exp.is_Integer:
                if lhs - rhs != eq:
                    return _vsolve(lhs - rhs, sym, **flags)

            if sym not in lhs.exp.free_symbols:
                return _vsolve(lhs.base - rhs**(1/lhs.exp), sym, **flags)

            # _tsolve calls this with Dummy before passing the actual number in.
            if any(t.is_Dummy for t in rhs.free_symbols):
                raise NotImplementedError # _tsolve will call here again...

            # a ** g(x) == 0
            if not rhs:
                # f(x)**g(x) only has solutions where f(x) == 0 and g(x) != 0 at
                # the same place
                sol_base = _vsolve(lhs.base, sym, **flags)
                return [s for s in sol_base if lhs.exp.subs(sym, s) != 0]  # XXX use checksol here?

            # a ** g(x) == b
            if not lhs.base.has(sym):
                if lhs.base == 0:
                    return _vsolve(lhs.exp, sym, **flags) if rhs != 0 else []

                # Gets most solutions...
                if lhs.base == rhs.as_base_exp()[0]:
                    # handles case when bases are equal
                    sol = _vsolve(lhs.exp - rhs.as_base_exp()[1], sym, **flags)
                else:
                    # handles cases when bases are not equal and exp
                    # may or may not be equal
                    f = exp(log(lhs.base)*lhs.exp) - exp(log(rhs))
                    sol = _vsolve(f, sym, **flags)

                # Check for duplicate solutions
                def equal(expr1, expr2):
                    _ = Dummy()
                    eq = checksol(expr1 - _, _, expr2)
                    if eq is None:
                        if nsimplify(expr1) != nsimplify(expr2):
                            return False
                        # they might be coincidentally the same
                        # so check more rigorously
                        eq = expr1.equals(expr2)  # XXX expensive but necessary?
                    return eq

                # Guess a rational exponent
                e_rat = nsimplify(log(abs(rhs))/log(abs(lhs.base)))
                e_rat = simplify(posify(e_rat)[0])
                n, d = fraction(e_rat)
                if expand(lhs.base**n - rhs**d) == 0:
                    sol = [s for s in sol if not equal(lhs.exp.subs(sym, s), e_rat)]
                    sol.extend(_vsolve(lhs.exp - e_rat, sym, **flags))

                return list(set(sol))

            # f(x) ** g(x) == c
            else:
                sol = []
                logform = lhs.exp*log(lhs.base) - log(rhs)
                if logform != lhs - rhs:
                    try:
                        sol.extend(_vsolve(logform, sym, **flags))
                    except NotImplementedError:
                        pass

                # Collect possible solutions and check with substitution later.
                check = []
                if rhs == 1:
                    # f(x) ** g(x) = 1 -- g(x)=0 or f(x)=+-1
                    check.extend(_vsolve(lhs.exp, sym, **flags))
                    check.extend(_vsolve(lhs.base - 1, sym, **flags))
                    check.extend(_vsolve(lhs.base + 1, sym, **flags))
                elif rhs.is_Rational:
                    for d in (i for i in divisors(abs(rhs.p)) if i != 1):
                        e, t = integer_log(rhs.p, d)
                        if not t:
                            continue  # rhs.p != d**b
                        for s in divisors(abs(rhs.q)):
                            if s**e== rhs.q:
                                r = Rational(d, s)
                                check.extend(_vsolve(lhs.base - r, sym, **flags))
                                check.extend(_vsolve(lhs.base + r, sym, **flags))
                                check.extend(_vsolve(lhs.exp - e, sym, **flags))
                elif rhs.is_irrational:
                    b_l, e_l = lhs.base.as_base_exp()
                    n, d = (e_l*lhs.exp).as_numer_denom()
                    b, e = sqrtdenest(rhs).as_base_exp()
                    check = [sqrtdenest(i) for i in (_vsolve(lhs.base - b, sym, **flags))]
                    check.extend([sqrtdenest(i) for i in (_vsolve(lhs.exp - e, sym, **flags))])
                    if e_l*d != 1:
                        check.extend(_vsolve(b_l**n - rhs**(e_l*d), sym, **flags))
                for s in check:
                    ok = checksol(eq, sym, s)
                    if ok is None:
                        ok = eq.subs(sym, s).equals(0)
                    if ok:
                        sol.append(s)
                return list(set(sol))

        elif lhs.is_Function and len(lhs.args) == 1:
            if lhs.func in multi_inverses:
                # sin(x) = 1/3 -> x - asin(1/3) & x - (pi - asin(1/3))
                soln = []
                for i in multi_inverses[type(lhs)](rhs):
                    soln.extend(_vsolve(lhs.args[0] - i, sym, **flags))
                return list(set(soln))
            elif lhs.func == LambertW:
                return _vsolve(lhs.args[0] - rhs*exp(rhs), sym, **flags)

        rewrite = lhs.rewrite(exp)
        rewrite = rebuild(rewrite) # avoid rewrites involving evaluate=False
        if rewrite != lhs:
            return _vsolve(rewrite - rhs, sym, **flags)
    except NotImplementedError:
        pass

    # maybe it is a lambert pattern
    if flags.pop('bivariate', True):
        # lambert forms may need some help being recognized, e.g. changing
        # 2**(3*x) + x**3*log(2)**3 + 3*x**2*log(2)**2 + 3*x*log(2) + 1
        # to 2**(3*x) + (x*log(2) + 1)**3

        # make generator in log have exponent of 1
        logs = eq.atoms(log)
        spow = min(
            {i.exp for j in logs for i in j.atoms(Pow)
             if i.base == sym} or {1})
        if spow != 1:
            p = sym**spow
            u = Dummy('bivariate-cov')
            ueq = eq.subs(p, u)
            if not ueq.has_free(sym):
                sol = _vsolve(ueq, u, **flags)
                inv = _vsolve(p - u, sym)
                return [i.subs(u, s) for i in inv for s in sol]

        g = _filtered_gens(eq.as_poly(), sym)
        up_or_log = set()
        for gi in g:
            if isinstance(gi, (exp, log)) or (gi.is_Pow and gi.base == S.Exp1):
                up_or_log.add(gi)
            elif gi.is_Pow:
                gisimp = powdenest(expand_power_exp(gi))
                if gisimp.is_Pow and sym in gisimp.exp.free_symbols:
                    up_or_log.add(gi)
        eq_down = expand_log(expand_power_exp(eq)).subs(
            dict(list(zip(up_or_log, [0]*len(up_or_log)))))
        eq = expand_power_exp(factor(eq_down, deep=True) + (eq - eq_down))
        rhs, lhs = _invert(eq, sym)
        if lhs.has(sym):
            try:
                poly = lhs.as_poly()
                g = _filtered_gens(poly, sym)
                _eq = lhs - rhs
                sols = _solve_lambert(_eq, sym, g)
                # use a simplified form if it satisfies eq
                # and has fewer operations
                for n, s in enumerate(sols):
                    ns = nsimplify(s)
                    if ns != s and ns.count_ops() <= s.count_ops():
                        ok = checksol(_eq, sym, ns)
                        if ok is None:
                            ok = _eq.subs(sym, ns).equals(0)
                        if ok:
                            sols[n] = ns
                return sols
            except NotImplementedError:
                # maybe it's a convoluted function
                if len(g) == 2:
                    try:
                        gpu = bivariate_type(lhs - rhs, *g)
                        if gpu is None:
                            raise NotImplementedError
                        g, p, u = gpu
                        flags['bivariate'] = False
                        inversion = _tsolve(g - u, sym, **flags)
                        if inversion:
                            sol = _vsolve(p, u, **flags)
                            return list({i.subs(u, s)
                                for i in inversion for s in sol})
                    except NotImplementedError:
                        pass
                else:
                    pass

    if flags.pop('force', True):
        flags['force'] = False
        pos, reps = posify(lhs - rhs)
        if rhs == S.ComplexInfinity:
            return []
        for u, s in reps.items():
            if s == sym:
                break
        else:
            u = sym
        if pos.has(u):
            try:
                soln = _vsolve(pos, u, **flags)
                return [s.subs(reps) for s in soln]
            except NotImplementedError:
                pass
        else:
            pass  # here for coverage

    return  # here for coverage


# TODO: option for calculating J numerically

@conserve_mpmath_dps
def nsolve(*args, dict=False, **kwargs):
    r"""
    Solve a nonlinear equation system numerically: ``nsolve(f, [args,] x0,
    modules=['mpmath'], **kwargs)``.

    Explanation
    ===========

    ``f`` is a vector function of symbolic expressions representing the system.
    *args* are the variables. If there is only one variable, this argument can
    be omitted. ``x0`` is a starting vector close to a solution.

    Use the modules keyword to specify which modules should be used to
    evaluate the function and the Jacobian matrix. Make sure to use a module
    that supports matrices. For more information on the syntax, please see the
    docstring of ``lambdify``.

    If the keyword arguments contain ``dict=True`` (default is False) ``nsolve``
    will return a list (perhaps empty) of solution mappings. This might be
    especially useful if you want to use ``nsolve`` as a fallback to solve since
    using the dict argument for both methods produces return values of
    consistent type structure. Please note: to keep this consistent with
    ``solve``, the solution will be returned in a list even though ``nsolve``
    (currently at least) only finds one solution at a time.

    Overdetermined systems are supported.

    Examples
    ========

    >>> from sympy import Symbol, nsolve
    >>> import mpmath
    >>> mpmath.mp.dps = 15
    >>> x1 = Symbol('x1')
    >>> x2 = Symbol('x2')
    >>> f1 = 3 * x1**2 - 2 * x2**2 - 1
    >>> f2 = x1**2 - 2 * x1 + x2**2 + 2 * x2 - 8
    >>> print(nsolve((f1, f2), (x1, x2), (-1, 1)))
    Matrix([[-1.19287309935246], [1.27844411169911]])

    For one-dimensional functions the syntax is simplified:

    >>> from sympy import sin, nsolve
    >>> from sympy.abc import x
    >>> nsolve(sin(x), x, 2)
    3.14159265358979
    >>> nsolve(sin(x), 2)
    3.14159265358979

    To solve with higher precision than the default, use the prec argument:

    >>> from sympy import cos
    >>> nsolve(cos(x) - x, 1)
    0.739085133215161
    >>> nsolve(cos(x) - x, 1, prec=50)
    0.73908513321516064165531208767387340401341175890076
    >>> cos(_)
    0.73908513321516064165531208767387340401341175890076

    To solve for complex roots of real functions, a nonreal initial point
    must be specified:

    >>> from sympy import I
    >>> nsolve(x**2 + 2, I)
    1.4142135623731*I

    ``mpmath.findroot`` is used and you can find their more extensive
    documentation, especially concerning keyword parameters and
    available solvers. Note, however, that functions which are very
    steep near the root, the verification of the solution may fail. In
    this case you should use the flag ``verify=False`` and
    independently verify the solution.

    >>> from sympy import cos, cosh
    >>> f = cos(x)*cosh(x) - 1
    >>> nsolve(f, 3.14*100)
    Traceback (most recent call last):
    ...
    ValueError: Could not find root within given tolerance. (1.39267e+230 > 2.1684e-19)
    >>> ans = nsolve(f, 3.14*100, verify=False); ans
    312.588469032184
    >>> f.subs(x, ans).n(2)
    2.1e+121
    >>> (f/f.diff(x)).subs(x, ans).n(2)
    7.4e-15

    One might safely skip the verification if bounds of the root are known
    and a bisection method is used:

    >>> bounds = lambda i: (3.14*i, 3.14*(i + 1))
    >>> nsolve(f, bounds(100), solver='bisect', verify=False)
    315.730061685774

    Alternatively, a function may be better behaved when the
    denominator is ignored. Since this is not always the case, however,
    the decision of what function to use is left to the discretion of
    the user.

    >>> eq = x**2/(1 - x)/(1 - 2*x)**2 - 100
    >>> nsolve(eq, 0.46)
    Traceback (most recent call last):
    ...
    ValueError: Could not find root within given tolerance. (10000 > 2.1684e-19)
    Try another starting point or tweak arguments.
    >>> nsolve(eq.as_numer_denom()[0], 0.46)
    0.46792545969349058

    """
    # there are several other SymPy functions that use method= so
    # guard against that here
    if 'method' in kwargs:
        raise ValueError(filldedent('''
            Keyword "method" should not be used in this context.  When using
            some mpmath solvers directly, the keyword "method" is
            used, but when using nsolve (and findroot) the keyword to use is
            "solver".'''))

    if 'prec' in kwargs:
        import mpmath
        mpmath.mp.dps = kwargs.pop('prec')

    # keyword argument to return result as a dictionary
    as_dict = dict
    from builtins import dict  # to unhide the builtin

    # interpret arguments
    if len(args) == 3:
        f = args[0]
        fargs = args[1]
        x0 = args[2]
        if iterable(fargs) and iterable(x0):
            if len(x0) != len(fargs):
                raise TypeError('nsolve expected exactly %i guess vectors, got %i'
                                % (len(fargs), len(x0)))
    elif len(args) == 2:
        f = args[0]
        fargs = None
        x0 = args[1]
        if iterable(f):
            raise TypeError('nsolve expected 3 arguments, got 2')
    elif len(args) < 2:
        raise TypeError('nsolve expected at least 2 arguments, got %i'
                        % len(args))
    else:
        raise TypeError('nsolve expected at most 3 arguments, got %i'
                        % len(args))
    modules = kwargs.get('modules', ['mpmath'])
    if iterable(f):
        f = list(f)
        for i, fi in enumerate(f):
            if isinstance(fi, Eq):
                f[i] = fi.lhs - fi.rhs
        f = Matrix(f).T
    if iterable(x0):
        x0 = list(x0)
    if not isinstance(f, Matrix):
        # assume it's a SymPy expression
        if isinstance(f, Eq):
            f = f.lhs - f.rhs
        elif f.is_Relational:
            raise TypeError('nsolve cannot accept inequalities')
        syms = f.free_symbols
        if fargs is None:
            fargs = syms.copy().pop()
        if not (len(syms) == 1 and (fargs in syms or fargs[0] in syms)):
            raise ValueError(filldedent('''
                expected a one-dimensional and numerical function'''))

        # the function is much better behaved if there is no denominator
        # but sending the numerator is left to the user since sometimes
        # the function is better behaved when the denominator is present
        # e.g., issue 11768

        f = lambdify(fargs, f, modules)
        x = sympify(findroot(f, x0, **kwargs))
        if as_dict:
            return [{fargs: x}]
        return x

    if len(fargs) > f.cols:
        raise NotImplementedError(filldedent('''
            need at least as many equations as variables'''))
    verbose = kwargs.get('verbose', False)
    if verbose:
        print('f(x):')
        print(f)
    # derive Jacobian
    J = f.jacobian(fargs)
    if verbose:
        print('J(x):')
        print(J)
    # create functions
    f = lambdify(fargs, f.T, modules)
    J = lambdify(fargs, J, modules)
    # solve the system numerically
    x = findroot(f, x0, J=J, **kwargs)
    if as_dict:
        return [dict(zip(fargs, [sympify(xi) for xi in x]))]
    return Matrix(x)


def _invert(eq, *symbols, **kwargs):
    """
    Return tuple (i, d) where ``i`` is independent of *symbols* and ``d``
    contains symbols.

    Explanation
    ===========

    ``i`` and ``d`` are obtained after recursively using algebraic inversion
    until an uninvertible ``d`` remains. If there are no free symbols then
    ``d`` will be zero. Some (but not necessarily all) solutions to the
    expression ``i - d`` will be related to the solutions of the original
    expression.

    Examples
    ========

    >>> from sympy.solvers.solvers import _invert as invert
    >>> from sympy import sqrt, cos
    >>> from sympy.abc import x, y
    >>> invert(x - 3)
    (3, x)
    >>> invert(3)
    (3, 0)
    >>> invert(2*cos(x) - 1)
    (1/2, cos(x))
    >>> invert(sqrt(x) - 3)
    (3, sqrt(x))
    >>> invert(sqrt(x) + y, x)
    (-y, sqrt(x))
    >>> invert(sqrt(x) + y, y)
    (-sqrt(x), y)
    >>> invert(sqrt(x) + y, x, y)
    (0, sqrt(x) + y)

    If there is more than one symbol in a power's base and the exponent
    is not an Integer, then the principal root will be used for the
    inversion:

    >>> invert(sqrt(x + y) - 2)
    (4, x + y)
    >>> invert(sqrt(x + y) + 2)  # note +2 instead of -2
    (4, x + y)

    If the exponent is an Integer, setting ``integer_power`` to True
    will force the principal root to be selected:

    >>> invert(x**2 - 4, integer_power=True)
    (2, x)

    """
    eq = sympify(eq)
    if eq.args:
        # make sure we are working with flat eq
        eq = eq.func(*eq.args)
    free = eq.free_symbols
    if not symbols:
        symbols = free
    if not free & set(symbols):
        return eq, S.Zero

    dointpow = bool(kwargs.get('integer_power', False))

    lhs = eq
    rhs = S.Zero
    while True:
        was = lhs
        while True:
            indep, dep = lhs.as_independent(*symbols)

            # dep + indep == rhs
            if lhs.is_Add:
                # this indicates we have done it all
                if indep.is_zero:
                    break

                lhs = dep
                rhs -= indep

            # dep * indep == rhs
            else:
                # this indicates we have done it all
                if indep is S.One:
                    break

                lhs = dep
                rhs /= indep

        # collect like-terms in symbols
        if lhs.is_Add:
            terms = {}
            for a in lhs.args:
                i, d = a.as_independent(*symbols)
                terms.setdefault(d, []).append(i)
            if any(len(v) > 1 for v in terms.values()):
                args = []
                for d, i in terms.items():
                    if len(i) > 1:
                        args.append(Add(*i)*d)
                    else:
                        args.append(i[0]*d)
                lhs = Add(*args)

        # if it's a two-term Add with rhs = 0 and two powers we can get the
        # dependent terms together, e.g. 3*f(x) + 2*g(x) -> f(x)/g(x) = -2/3
        if lhs.is_Add and not rhs and len(lhs.args) == 2 and \
                not lhs.is_polynomial(*symbols):
            a, b = ordered(lhs.args)
            ai, ad = a.as_independent(*symbols)
            bi, bd = b.as_independent(*symbols)
            if any(_ispow(i) for i in (ad, bd)):
                a_base, a_exp = ad.as_base_exp()
                b_base, b_exp = bd.as_base_exp()
                if a_base == b_base and a_exp.extract_additively(b_exp) is None:
                    # a = -b and exponents do not have canceling terms/factors
                    # e.g. if exponents were 3*x and x then the ratio would have
                    # an exponent of 2*x: one of the roots would be lost
                    rat = powsimp(powdenest(ad/bd))
                    lhs = rat
                    rhs = -bi/ai
                else:
                    rat = ad/bd
                    _lhs = powsimp(ad/bd)
                    if _lhs != rat:
                        lhs = _lhs
                        rhs = -bi/ai
            elif ai == -bi:
                if isinstance(ad, Function) and ad.func == bd.func:
                    if len(ad.args) == len(bd.args) == 1:
                        lhs = ad.args[0] - bd.args[0]
                    elif len(ad.args) == len(bd.args):
                        # should be able to solve
                        # f(x, y) - f(2 - x, 0) == 0 -> x == 1
                        raise NotImplementedError(
                            'equal function with more than 1 argument')
                    else:
                        raise ValueError(
                            'function with different numbers of args')

        elif lhs.is_Mul and any(_ispow(a) for a in lhs.args):
            lhs = powsimp(powdenest(lhs))

        if lhs.is_Function:
            if hasattr(lhs, 'inverse') and lhs.inverse() is not None and len(lhs.args) == 1:
                #                    -1
                # f(x) = g  ->  x = f  (g)
                #
                # /!\ inverse should not be defined if there are multiple values
                # for the function -- these are handled in _tsolve
                #
                rhs = lhs.inverse()(rhs)
                lhs = lhs.args[0]
            elif isinstance(lhs, atan2):
                y, x = lhs.args
                lhs = 2*atan(y/(sqrt(x**2 + y**2) + x))
            elif lhs.func == rhs.func:
                if len(lhs.args) == len(rhs.args) == 1:
                    lhs = lhs.args[0]
                    rhs = rhs.args[0]
                elif len(lhs.args) == len(rhs.args):
                    # should be able to solve
                    # f(x, y) == f(2, 3) -> x == 2
                    # f(x, x + y) == f(2, 3) -> x == 2
                    raise NotImplementedError(
                        'equal function with more than 1 argument')
                else:
                    raise ValueError(
                        'function with different numbers of args')


        if rhs and lhs.is_Pow and lhs.exp.is_Integer and lhs.exp < 0:
            lhs = 1/lhs
            rhs = 1/rhs

        # base**a = b -> base = b**(1/a) if
        #    a is an Integer and dointpow=True (this gives real branch of root)
        #    a is not an Integer and the equation is multivariate and the
        #      base has more than 1 symbol in it
        # The rationale for this is that right now the multi-system solvers
        # doesn't try to resolve generators to see, for example, if the whole
        # system is written in terms of sqrt(x + y) so it will just fail, so we
        # do that step here.
        if lhs.is_Pow and (
            lhs.exp.is_Integer and dointpow or not lhs.exp.is_Integer and
                len(symbols) > 1 and len(lhs.base.free_symbols & set(symbols)) > 1):
            rhs = rhs**(1/lhs.exp)
            lhs = lhs.base

        if lhs == was:
            break
    return rhs, lhs


def unrad(eq, *syms, **flags):
    """
    Remove radicals with symbolic arguments and return (eq, cov),
    None, or raise an error.

    Explanation
    ===========

    None is returned if there are no radicals to remove.

    NotImplementedError is raised if there are radicals and they cannot be
    removed or if the relationship between the original symbols and the
    change of variable needed to rewrite the system as a polynomial cannot
    be solved.

    Otherwise the tuple, ``(eq, cov)``, is returned where:

    *eq*, ``cov``
        *eq* is an equation without radicals (in the symbol(s) of
        interest) whose solutions are a superset of the solutions to the
        original expression. *eq* might be rewritten in terms of a new
        variable; the relationship to the original variables is given by
        ``cov`` which is a list containing ``v`` and ``v**p - b`` where
        ``p`` is the power needed to clear the radical and ``b`` is the
        radical now expressed as a polynomial in the symbols of interest.
        For example, for sqrt(2 - x) the tuple would be
        ``(c, c**2 - 2 + x)``. The solutions of *eq* will contain
        solutions to the original equation (if there are any).

    *syms*
        An iterable of symbols which, if provided, will limit the focus of
        radical removal: only radicals with one or more of the symbols of
        interest will be cleared. All free symbols are used if *syms* is not
        set.

    *flags* are used internally for communication during recursive calls.
    Two options are also recognized:

        ``take``, when defined, is interpreted as a single-argument function
        that returns True if a given Pow should be handled.

    Radicals can be removed from an expression if:

        *   All bases of the radicals are the same; a change of variables is
            done in this case.
        *   If all radicals appear in one term of the expression.
        *   There are only four terms with sqrt() factors or there are less than
            four terms having sqrt() factors.
        *   There are only two terms with radicals.

    Examples
    ========

    >>> from sympy.solvers.solvers import unrad
    >>> from sympy.abc import x
    >>> from sympy import sqrt, Rational, root

    >>> unrad(sqrt(x)*x**Rational(1, 3) + 2)
    (x**5 - 64, [])
    >>> unrad(sqrt(x) + root(x + 1, 3))
    (-x**3 + x**2 + 2*x + 1, [])
    >>> eq = sqrt(x) + root(x, 3) - 2
    >>> unrad(eq)
    (_p**3 + _p**2 - 2, [_p, _p**6 - x])

    """

    uflags = {"check": False, "simplify": False}

    def _cov(p, e):
        if cov:
            # XXX - uncovered
            oldp, olde = cov
            if Poly(e, p).degree(p) in (1, 2):
                cov[:] = [p, olde.subs(oldp, _vsolve(e, p, **uflags)[0])]
            else:
                raise NotImplementedError
        else:
            cov[:] = [p, e]

    def _canonical(eq, cov):
        if cov:
            # change symbol to vanilla so no solutions are eliminated
            p, e = cov
            rep = {p: Dummy(p.name)}
            eq = eq.xreplace(rep)
            cov = [p.xreplace(rep), e.xreplace(rep)]

        # remove constants and powers of factors since these don't change
        # the location of the root; XXX should factor or factor_terms be used?
        eq = factor_terms(_mexpand(eq.as_numer_denom()[0], recursive=True), clear=True)
        if eq.is_Mul:
            args = []
            for f in eq.args:
                if f.is_number:
                    continue
                if f.is_Pow:
                    args.append(f.base)
                else:
                    args.append(f)
            eq = Mul(*args)  # leave as Mul for more efficient solving

        # make the sign canonical
        margs = list(Mul.make_args(eq))
        changed = False
        for i, m in enumerate(margs):
            if m.could_extract_minus_sign():
                margs[i] = -m
                changed = True
        if changed:
            eq = Mul(*margs, evaluate=False)

        return eq, cov

    def _Q(pow):
        # return leading Rational of denominator of Pow's exponent
        c = pow.as_base_exp()[1].as_coeff_Mul()[0]
        if not c.is_Rational:
            return S.One
        return c.q

    # define the _take method that will determine whether a term is of interest
    def _take(d):
        # return True if coefficient of any factor's exponent's den is not 1
        for pow in Mul.make_args(d):
            if not pow.is_Pow:
                continue
            if _Q(pow) == 1:
                continue
            if pow.free_symbols & syms:
                return True
        return False
    _take = flags.setdefault('_take', _take)

    if isinstance(eq, Eq):
        eq = eq.lhs - eq.rhs  # XXX legacy Eq as Eqn support
    elif not isinstance(eq, Expr):
        return

    cov, nwas, rpt = [flags.setdefault(k, v) for k, v in
        sorted({"cov": [], "n": None, "rpt": 0}.items())]

    # preconditioning
    eq = powdenest(factor_terms(eq, radical=True, clear=True))
    eq = eq.as_numer_denom()[0]
    eq = _mexpand(eq, recursive=True)
    if eq.is_number:
        return

    # see if there are radicals in symbols of interest
    syms = set(syms) or eq.free_symbols  # _take uses this
    poly = eq.as_poly()
    gens = [g for g in poly.gens if _take(g)]
    if not gens:
        return

    # recast poly in terms of eigen-gens
    poly = eq.as_poly(*gens)

    # not a polynomial e.g. 1 + sqrt(x)*exp(sqrt(x)) with gen sqrt(x)
    if poly is None:
        return

    # - an exponent has a symbol of interest (don't handle)
    if any(g.exp.has(*syms) for g in gens):
        return

    def _rads_bases_lcm(poly):
        # if all the bases are the same or all the radicals are in one
        # term, `lcm` will be the lcm of the denominators of the
        # exponents of the radicals
        lcm = 1
        rads = set()
        bases = set()
        for g in poly.gens:
            q = _Q(g)
            if q != 1:
                rads.add(g)
                lcm = ilcm(lcm, q)
                bases.add(g.base)
        return rads, bases, lcm
    rads, bases, lcm = _rads_bases_lcm(poly)

    covsym = Dummy('p', nonnegative=True)

    # only keep in syms symbols that actually appear in radicals;
    # and update gens
    newsyms = set()
    for r in rads:
        newsyms.update(syms & r.free_symbols)
    if newsyms != syms:
        syms = newsyms
    # get terms together that have common generators
    drad = dict(zip(rads, range(len(rads))))
    rterms = {(): []}
    args = Add.make_args(poly.as_expr())
    for t in args:
        if _take(t):
            common = set(t.as_poly().gens).intersection(rads)
            key = tuple(sorted([drad[i] for i in common]))
        else:
            key = ()
        rterms.setdefault(key, []).append(t)
    others = Add(*rterms.pop(()))
    rterms = [Add(*rterms[k]) for k in rterms.keys()]

    # the output will depend on the order terms are processed, so
    # make it canonical quickly
    rterms = list(reversed(list(ordered(rterms))))

    ok = False  # we don't have a solution yet
    depth = sqrt_depth(eq)

    if len(rterms) == 1 and not (rterms[0].is_Add and lcm > 2):
        eq = rterms[0]**lcm - ((-others)**lcm)
        ok = True
    else:
        if len(rterms) == 1 and rterms[0].is_Add:
            rterms = list(rterms[0].args)
        if len(bases) == 1:
            b = bases.pop()
            if len(syms) > 1:
                x = b.free_symbols
            else:
                x = syms
            x = list(ordered(x))[0]
            try:
                inv = _vsolve(covsym**lcm - b, x, **uflags)
                if not inv:
                    raise NotImplementedError
                eq = poly.as_expr().subs(b, covsym**lcm).subs(x, inv[0])
                _cov(covsym, covsym**lcm - b)
                return _canonical(eq, cov)
            except NotImplementedError:
                pass

        if len(rterms) == 2:
            if not others:
                eq = rterms[0]**lcm - (-rterms[1])**lcm
                ok = True
            elif not log(lcm, 2).is_Integer:
                # the lcm-is-power-of-two case is handled below
                r0, r1 = rterms
                if flags.get('_reverse', False):
                    r1, r0 = r0, r1
                i0 = _rads0, _bases0, lcm0 = _rads_bases_lcm(r0.as_poly())
                i1 = _rads1, _bases1, lcm1 = _rads_bases_lcm(r1.as_poly())
                for reverse in range(2):
                    if reverse:
                        i0, i1 = i1, i0
                        r0, r1 = r1, r0
                    _rads1, _, lcm1 = i1
                    _rads1 = Mul(*_rads1)
                    t1 = _rads1**lcm1
                    c = covsym**lcm1 - t1
                    for x in syms:
                        try:
                            sol = _vsolve(c, x, **uflags)
                            if not sol:
                                raise NotImplementedError
                            neweq = r0.subs(x, sol[0]) + covsym*r1/_rads1 + \
                                others
                            tmp = unrad(neweq, covsym)
                            if tmp:
                                eq, newcov = tmp
                                if newcov:
                                    newp, newc = newcov
                                    _cov(newp, c.subs(covsym,
                                        _vsolve(newc, covsym, **uflags)[0]))
                                else:
                                    _cov(covsym, c)
                            else:
                                eq = neweq
                                _cov(covsym, c)
                            ok = True
                            break
                        except NotImplementedError:
                            if reverse:
                                raise NotImplementedError(
                                    'no successful change of variable found')
                            else:
                                pass
                    if ok:
                        break
        elif len(rterms) == 3:
            # two cube roots and another with order less than 5
            # (so an analytical solution can be found) or a base
            # that matches one of the cube root bases
            info = [_rads_bases_lcm(i.as_poly()) for i in rterms]
            RAD = 0
            BASES = 1
            LCM = 2
            if info[0][LCM] != 3:
                info.append(info.pop(0))
                rterms.append(rterms.pop(0))
            elif info[1][LCM] != 3:
                info.append(info.pop(1))
                rterms.append(rterms.pop(1))
            if info[0][LCM] == info[1][LCM] == 3:
                if info[1][BASES] != info[2][BASES]:
                    info[0], info[1] = info[1], info[0]
                    rterms[0], rterms[1] = rterms[1], rterms[0]
                if info[1][BASES] == info[2][BASES]:
                    eq = rterms[0]**3 + (rterms[1] + rterms[2] + others)**3
                    ok = True
                elif info[2][LCM] < 5:
                    # a*root(A, 3) + b*root(B, 3) + others = c
                    a, b, c, d, A, B = [Dummy(i) for i in 'abcdAB']
                    # zz represents the unraded expression into which the
                    # specifics for this case are substituted
                    zz = (c - d)*(A**3*a**9 + 3*A**2*B*a**6*b**3 -
                        3*A**2*a**6*c**3 + 9*A**2*a**6*c**2*d - 9*A**2*a**6*c*d**2 +
                        3*A**2*a**6*d**3 + 3*A*B**2*a**3*b**6 + 21*A*B*a**3*b**3*c**3 -
                        63*A*B*a**3*b**3*c**2*d + 63*A*B*a**3*b**3*c*d**2 -
                        21*A*B*a**3*b**3*d**3 + 3*A*a**3*c**6 - 18*A*a**3*c**5*d +
                        45*A*a**3*c**4*d**2 - 60*A*a**3*c**3*d**3 + 45*A*a**3*c**2*d**4 -
                        18*A*a**3*c*d**5 + 3*A*a**3*d**6 + B**3*b**9 - 3*B**2*b**6*c**3 +
                        9*B**2*b**6*c**2*d - 9*B**2*b**6*c*d**2 + 3*B**2*b**6*d**3 +
                        3*B*b**3*c**6 - 18*B*b**3*c**5*d + 45*B*b**3*c**4*d**2 -
                        60*B*b**3*c**3*d**3 + 45*B*b**3*c**2*d**4 - 18*B*b**3*c*d**5 +
                        3*B*b**3*d**6 - c**9 + 9*c**8*d - 36*c**7*d**2 + 84*c**6*d**3 -
                        126*c**5*d**4 + 126*c**4*d**5 - 84*c**3*d**6 + 36*c**2*d**7 -
                        9*c*d**8 + d**9)
                    def _t(i):
                        b = Mul(*info[i][RAD])
                        return cancel(rterms[i]/b), Mul(*info[i][BASES])
                    aa, AA = _t(0)
                    bb, BB = _t(1)
                    cc = -rterms[2]
                    dd = others
                    eq = zz.xreplace(dict(zip(
                        (a, A, b, B, c, d),
                        (aa, AA, bb, BB, cc, dd))))
                    ok = True
        # handle power-of-2 cases
        if not ok:
            if log(lcm, 2).is_Integer and (not others and
                    len(rterms) == 4 or len(rterms) < 4):
                def _norm2(a, b):
                    return a**2 + b**2 + 2*a*b

                if len(rterms) == 4:
                    # (r0+r1)**2 - (r2+r3)**2
                    r0, r1, r2, r3 = rterms
                    eq = _norm2(r0, r1) - _norm2(r2, r3)
                    ok = True
                elif len(rterms) == 3:
                    # (r1+r2)**2 - (r0+others)**2
                    r0, r1, r2 = rterms
                    eq = _norm2(r1, r2) - _norm2(r0, others)
                    ok = True
                elif len(rterms) == 2:
                    # r0**2 - (r1+others)**2
                    r0, r1 = rterms
                    eq = r0**2 - _norm2(r1, others)
                    ok = True

    new_depth = sqrt_depth(eq) if ok else depth
    rpt += 1  # XXX how many repeats with others unchanging is enough?
    if not ok or (
                nwas is not None and len(rterms) == nwas and
                new_depth is not None and new_depth == depth and
                rpt > 3):
        raise NotImplementedError('Cannot remove all radicals')

    flags.update({"cov": cov, "n": len(rterms), "rpt": rpt})
    neq = unrad(eq, *syms, **flags)
    if neq:
        eq, cov = neq
    eq, cov = _canonical(eq, cov)
    return eq, cov


# delayed imports
from sympy.solvers.bivariate import (
    bivariate_type, _solve_lambert, _filtered_gens)
