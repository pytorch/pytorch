"""Tools for optimizing a linear function for a given simplex.

For the linear objective function ``f`` with linear constraints
expressed using `Le`, `Ge` or `Eq` can be found with ``lpmin`` or
``lpmax``. The symbols are **unbounded** unless specifically
constrained.

As an alternative, the matrices describing the objective and the
constraints, and an optional list of bounds can be passed to
``linprog`` which will solve for the minimization of ``C*x``
under constraints ``A*x <= b`` and/or ``Aeq*x = beq``, and
individual bounds for variables given as ``(lo, hi)``. The values
returned are **nonnegative** unless bounds are provided that
indicate otherwise.

Errors that might be raised are UnboundedLPError when there is no
finite solution for the system or InfeasibleLPError when the
constraints represent impossible conditions (i.e. a non-existent
 simplex).

Here is a simple 1-D system: minimize `x` given that ``x >= 1``.

    >>> from sympy.solvers.simplex import lpmin, linprog
    >>> from sympy.abc import x

    The function and a list with the constraint is passed directly
    to `lpmin`:

    >>> lpmin(x, [x >= 1])
    (1, {x: 1})

    For `linprog` the matrix for the objective is `[1]` and the
    uivariate constraint can be passed as a bound with None acting
    as infinity:

    >>> linprog([1], bounds=(1, None))
    (1, [1])

    Or the matrices, corresponding to ``x >= 1`` expressed as
    ``-x <= -1`` as required by the routine, can be passed:

    >>> linprog([1], [-1], [-1])
    (1, [1])

    If there is no limit for the objective, an error is raised.
    In this case there is a valid region of interest (simplex)
    but no limit to how small ``x`` can be:

    >>> lpmin(x, [])
    Traceback (most recent call last):
    ...
    sympy.solvers.simplex.UnboundedLPError:
    Objective function can assume arbitrarily large values!

    An error is raised if there is no possible solution:

    >>> lpmin(x,[x<=1,x>=2])
    Traceback (most recent call last):
    ...
    sympy.solvers.simplex.InfeasibleLPError:
    Inconsistent/False constraint
"""

from sympy.core import sympify
from sympy.core.exprtools import factor_terms
from sympy.core.relational import Le, Ge, Eq
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sorting import ordered
from sympy.functions.elementary.complexes import sign
from sympy.matrices.dense import Matrix, zeros
from sympy.solvers.solveset import linear_eq_to_matrix
from sympy.utilities.iterables import numbered_symbols
from sympy.utilities.misc import filldedent


class UnboundedLPError(Exception):
    """
    A linear programming problem is said to be unbounded if its objective
    function can assume arbitrarily large values.

    Example
    =======

    Suppose you want to maximize
        2x
    subject to
        x >= 0

    There's no upper limit that 2x can take.
    """

    pass


class InfeasibleLPError(Exception):
    """
    A linear programming problem is considered infeasible if its
    constraint set is empty. That is, if the set of all vectors
    satisfying the constraints is empty, then the problem is infeasible.

    Example
    =======

    Suppose you want to maximize
        x
    subject to
        x >= 10
        x <= 9

    No x can satisfy those constraints.
    """

    pass


def _pivot(M, i, j):
    """
    The pivot element `M[i, j]` is inverted and the rest of the matrix
    modified and returned as a new matrix; original is left unmodified.

    Example
    =======

    >>> from sympy.matrices.dense import Matrix
    >>> from sympy.solvers.simplex import _pivot
    >>> from sympy import var
    >>> Matrix(3, 3, var('a:i'))
    Matrix([
    [a, b, c],
    [d, e, f],
    [g, h, i]])
    >>> _pivot(_, 1, 0)
    Matrix([
    [-a/d, -a*e/d + b, -a*f/d + c],
    [ 1/d,        e/d,        f/d],
    [-g/d,  h - e*g/d,  i - f*g/d]])
    """
    Mi, Mj, Mij = M[i, :], M[:, j], M[i, j]
    if Mij == 0:
        raise ZeroDivisionError(
            "Tried to pivot about zero-valued entry.")
    A = M - Mj * (Mi / Mij)
    A[i, :] = Mi / Mij
    A[:, j] = -Mj / Mij
    A[i, j] = 1 / Mij
    return A


def _choose_pivot_row(A, B, candidate_rows, pivot_col, Y):
    # Choose row with smallest ratio
    # If there are ties, pick using Bland's rule
    return min(candidate_rows, key=lambda i: (B[i] / A[i, pivot_col], Y[i]))


def _simplex(A, B, C, D=None, dual=False):
    """Return ``(o, x, y)`` obtained from the two-phase simplex method
    using Bland's rule: ``o`` is the minimum value of primal,
    ``Cx - D``, under constraints ``Ax <= B`` (with ``x >= 0``) and
    the maximum of the dual, ``y^{T}B - D``, under constraints
    ``A^{T}*y >= C^{T}`` (with ``y >= 0``). To compute the dual of
    the system, pass `dual=True` and ``(o, y, x)`` will be returned.

    Note: the nonnegative constraints for ``x`` and ``y`` supercede
    any values of ``A`` and ``B`` that are inconsistent with that
    assumption, so if a constraint of ``x >= -1`` is represented
    in ``A`` and ``B``, no value will be obtained that is negative; if
    a constraint of ``x <= -1`` is represented, an error will be
    raised since no solution is possible.

    This routine relies on the ability of determining whether an
    expression is 0 or not. This is guaranteed if the input contains
    only Float or Rational entries. It will raise a TypeError if
    a relationship does not evaluate to True or False.

    Examples
    ========

    >>> from sympy.solvers.simplex import _simplex
    >>> from sympy import Matrix

    Consider the simple minimization of ``f = x + y + 1`` under the
    constraint that ``y + 2*x >= 4``. This is the "standard form" of
    a minimization.

    In the nonnegative quadrant, this inequality describes a area above
    a triangle with vertices at (0, 4), (0, 0) and (2, 0). The minimum
    of ``f`` occurs at (2, 0). Define A, B, C, D for the standard
    minimization:

    >>> A = Matrix([[2, 1]])
    >>> B = Matrix([4])
    >>> C = Matrix([[1, 1]])
    >>> D = Matrix([-1])

    Confirm that this is the system of interest:

    >>> from sympy.abc import x, y
    >>> X = Matrix([x, y])
    >>> (C*X - D)[0]
    x + y + 1
    >>> [i >= j for i, j in zip(A*X, B)]
    [2*x + y >= 4]

    Since `_simplex` will do a minimization for constraints given as
    ``A*x <= B``, the signs of ``A`` and ``B`` must be negated since
    the currently correspond to a greater-than inequality:

    >>> _simplex(-A, -B, C, D)
    (3, [2, 0], [1/2])

    The dual of minimizing ``f`` is maximizing ``F = c*y - d`` for
    ``a*y <= b`` where ``a``, ``b``, ``c``, ``d`` are derived from the
    transpose of the matrix representation of the standard minimization:

    >>> tr = lambda a, b, c, d: [i.T for i in (a, c, b, d)]
    >>> a, b, c, d = tr(A, B, C, D)

    This time ``a*x <= b`` is the expected inequality for the `_simplex`
    method, but to maximize ``F``, the sign of ``c`` and ``d`` must be
    changed (so that minimizing the negative will give the negative of
    the maximum of ``F``):

    >>> _simplex(a, b, -c, -d)
    (-3, [1/2], [2, 0])

    The negative of ``F`` and the min of ``f`` are the same. The dual
    point `[1/2]` is the value of ``y`` that minimized ``F = c*y - d``
    under constraints a*x <= b``:

    >>> y = Matrix(['y'])
    >>> (c*y - d)[0]
    4*y + 1
    >>> [i <= j for i, j in zip(a*y,b)]
    [2*y <= 1, y <= 1]

    In this 1-dimensional dual system, the more restrictive constraint is
    the first which limits ``y`` between 0 and 1/2 and the maximum of
    ``F`` is attained at the nonzero value, hence is ``4*(1/2) + 1 = 3``.

    In this case the values for ``x`` and ``y`` were the same when the
    dual representation was solved. This is not always the case (though
    the value of the function will be the same).

    >>> l = [[1, 1], [-1, 1], [0, 1], [-1, 0]], [5, 1, 2, -1], [[1, 1]], [-1]
    >>> A, B, C, D = [Matrix(i) for i in l]
    >>> _simplex(A, B, -C, -D)
    (-6, [3, 2], [1, 0, 0, 0])
    >>> _simplex(A, B, -C, -D, dual=True)  # [5, 0] != [3, 2]
    (-6, [1, 0, 0, 0], [5, 0])

    In both cases the function has the same value:

    >>> Matrix(C)*Matrix([3, 2]) == Matrix(C)*Matrix([5, 0])
    True

    See Also
    ========
    _lp - poses min/max problem in form compatible with _simplex
    lpmin - minimization which calls _lp
    lpmax - maximimzation which calls _lp

    References
    ==========

    .. [1] Thomas S. Ferguson, LINEAR PROGRAMMING: A Concise Introduction
           web.tecnico.ulisboa.pt/mcasquilho/acad/or/ftp/FergusonUCLA_lp.pdf

    """
    A, B, C, D = [Matrix(i) for i in (A, B, C, D or [0])]
    if dual:
        _o, d, p = _simplex(-A.T, C.T, B.T, -D)
        return -_o, d, p

    if A and B:
        M = Matrix([[A, B], [C, D]])
    else:
        if A or B:
            raise ValueError("must give A and B")
        # no constraints given
        M = Matrix([[C, D]])
    n = M.cols - 1
    m = M.rows - 1

    if not all(i.is_Float or i.is_Rational for i in M):
        # with literal Float and Rational we are guaranteed the
        # ability of determining whether an expression is 0 or not
        raise TypeError(filldedent("""
            Only rationals and floats are allowed.
            """
            )
        )

    # x variables have priority over y variables during Bland's rule
    # since False < True
    X = [(False, j) for j in range(n)]
    Y = [(True, i) for i in range(m)]

    # Phase 1: find a feasible solution or determine none exist

    ## keep track of last pivot row and column
    last = None

    while True:
        B = M[:-1, -1]
        A = M[:-1, :-1]
        if all(B[i] >= 0 for i in range(B.rows)):
            # We have found a feasible solution
            break

        # Find k: first row with a negative rightmost entry
        for k in range(B.rows):
            if B[k] < 0:
                break  # use current value of k below
        else:
            pass  # error will raise below

        # Choose pivot column, c
        piv_cols = [_ for _ in range(A.cols) if A[k, _] < 0]
        if not piv_cols:
            raise InfeasibleLPError(filldedent("""
                The constraint set is empty!"""))
        _, c = min((X[i], i) for i in piv_cols) # Bland's rule

        # Choose pivot row, r
        piv_rows = [_ for _ in range(A.rows) if A[_, c] > 0 and B[_] > 0]
        piv_rows.append(k)
        r = _choose_pivot_row(A, B, piv_rows, c, Y)

        # check for oscillation
        if (r, c) == last:
            # Not sure what to do here; it looks like there will be
            # oscillations; see o1 test added at this commit to
            # see a system with no solution and the o2 for one
            # with a solution. In the case of o2, the solution
            # from linprog is the same as the one from lpmin, but
            # the matrices created in the lpmin case are different
            # than those created without replacements in linprog and
            # the matrices in the linprog case lead to oscillations.
            # If the matrices could be re-written in linprog like
            # lpmin does, this behavior could be avoided and then
            # perhaps the oscillating case would only occur when
            # there is no solution. For now, the output is checked
            # before exit if oscillations were detected and an
            # error is raised there if the solution was invalid.
            #
            # cf section 6 of Ferguson for a non-cycling modification
            last = True
            break
        last = r, c

        M = _pivot(M, r, c)
        X[c], Y[r] = Y[r], X[c]

    # Phase 2: from a feasible solution, pivot to optimal
    while True:
        B = M[:-1, -1]
        A = M[:-1, :-1]
        C = M[-1, :-1]

        # Choose a pivot column, c
        piv_cols = [_ for _ in range(n) if C[_] < 0]
        if not piv_cols:
            break
        _, c = min((X[i], i) for i in piv_cols)  # Bland's rule

        # Choose a pivot row, r
        piv_rows = [_ for _ in range(m) if A[_, c] > 0]
        if not piv_rows:
            raise UnboundedLPError(filldedent("""
                Objective function can assume
                arbitrarily large values!"""))
        r = _choose_pivot_row(A, B, piv_rows, c, Y)

        M = _pivot(M, r, c)
        X[c], Y[r] = Y[r], X[c]

    argmax = [None] * n
    argmin_dual = [None] * m

    for i, (v, n) in enumerate(X):
        if v == False:
            argmax[n] = 0
        else:
            argmin_dual[n] = M[-1, i]

    for i, (v, n) in enumerate(Y):
        if v == True:
            argmin_dual[n] = 0
        else:
            argmax[n] = M[i, -1]

    if last and not all(i >= 0 for i in argmax + argmin_dual):
        raise InfeasibleLPError(filldedent("""
            Oscillating system led to invalid solution.
            If you believe there was a valid solution, please
            report this as a bug."""))
    return -M[-1, -1], argmax, argmin_dual


## routines that use _simplex or support those that do


def _abcd(M, list=False):
    """return parts of M as matrices or lists

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.solvers.simplex import _abcd

    >>> m = Matrix(3, 3, range(9)); m
    Matrix([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]])
    >>> a, b, c, d = _abcd(m)
    >>> a
    Matrix([
    [0, 1],
    [3, 4]])
    >>> b
    Matrix([
    [2],
    [5]])
    >>> c
    Matrix([[6, 7]])
    >>> d
    Matrix([[8]])

    The matrices can be returned as compact lists, too:

    >>> L = a, b, c, d = _abcd(m, list=True); L
    ([[0, 1], [3, 4]], [2, 5], [[6, 7]], [8])
    """

    def aslist(i):
        l = i.tolist()
        if len(l[0]) == 1:  # col vector
            return [i[0] for i in l]
        return l

    m = M[:-1, :-1], M[:-1, -1], M[-1, :-1], M[-1:, -1:]
    if not list:
        return m
    return tuple([aslist(i) for i in m])


def _m(a, b, c, d=None):
    """return Matrix([[a, b], [c, d]]) from matrices
    in Matrix or list form.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.solvers.simplex import _abcd, _m
    >>> m = Matrix(3, 3, range(9))
    >>> L = _abcd(m, list=True); L
    ([[0, 1], [3, 4]], [2, 5], [[6, 7]], [8])
    >>> _abcd(m)
    (Matrix([
    [0, 1],
    [3, 4]]), Matrix([
    [2],
    [5]]), Matrix([[6, 7]]), Matrix([[8]]))
    >>> assert m == _m(*L) == _m(*_)
    """
    a, b, c, d = [Matrix(i) for i in (a, b, c, d or [0])]
    return Matrix([[a, b], [c, d]])


def _primal_dual(M, factor=True):
    """return primal and dual function and constraints
    assuming that ``M = Matrix([[A, b], [c, d]])`` and the
    function ``c*x - d`` is being minimized with ``Ax >= b``
    for nonnegative values of ``x``. The dual and its
    constraints will be for maximizing `b.T*y - d` subject
    to ``A.T*y <= c.T``.

    Examples
    ========

    >>> from sympy.solvers.simplex import _primal_dual, lpmin, lpmax
    >>> from sympy import Matrix

    The following matrix represents the primal task of
    minimizing x + y + 7 for y >= x + 1 and y >= -2*x + 3.
    The dual task seeks to maximize x + 3*y + 7 with
    2*y - x <= 1 and and x + y <= 1:

    >>> M = Matrix([
    ...     [-1, 1,  1],
    ...     [ 2, 1,  3],
    ...     [ 1, 1, -7]])
    >>> p, d = _primal_dual(M)

    The minimum of the primal and maximum of the dual are the same
    (though they occur at different points):

    >>> lpmin(*p)
    (28/3, {x1: 2/3, x2: 5/3})
    >>> lpmax(*d)
    (28/3, {y1: 1/3, y2: 2/3})

    If the equivalent (but canonical) inequalities are
    desired, leave `factor=True`, otherwise the unmodified
    inequalities for M will be returned.

    >>> m = Matrix([
    ... [-3, -2,  4, -2],
    ... [ 2,  0,  0, -2],
    ... [ 0,  1, -3,  0]])

    >>> _primal_dual(m, False)  # last condition is 2*x1 >= -2
    ((x2 - 3*x3,
        [-3*x1 - 2*x2 + 4*x3 >= -2, 2*x1 >= -2]),
    (-2*y1 - 2*y2,
        [-3*y1 + 2*y2 <= 0, -2*y1 <= 1, 4*y1 <= -3]))

    >>> _primal_dual(m)  # condition now x1 >= -1
    ((x2 - 3*x3,
        [-3*x1 - 2*x2 + 4*x3 >= -2, x1 >= -1]),
    (-2*y1 - 2*y2,
        [-3*y1 + 2*y2 <= 0, -2*y1 <= 1, 4*y1 <= -3]))

    If you pass the transpose of the matrix, the primal will be
    identified as the standard minimization problem and the
    dual as the standard maximization:

    >>> _primal_dual(m.T)
    ((-2*x1 - 2*x2,
        [-3*x1 + 2*x2 >= 0, -2*x1 >= 1, 4*x1 >= -3]),
    (y2 - 3*y3,
        [-3*y1 - 2*y2 + 4*y3 <= -2, y1 <= -1]))

    A matrix must have some size or else None will be returned for
    the functions:

    >>> _primal_dual(Matrix([[1, 2]]))
    ((x1 - 2, []), (-2, []))

    >>> _primal_dual(Matrix([]))
    ((None, []), (None, []))

    References
    ==========

    .. [1] David Galvin, Relations between Primal and Dual
           www3.nd.edu/~dgalvin1/30210/30210_F07/presentations/dual_opt.pdf
    """
    if not M:
        return (None, []), (None, [])
    if not hasattr(M, "shape"):
        if len(M) not in (3, 4):
            raise ValueError("expecting Matrix or 3 or 4 lists")
        M = _m(*M)
    m, n = [i - 1 for i in M.shape]
    A, b, c, d = _abcd(M)
    d = d[0]
    _ = lambda x: numbered_symbols(x, start=1)
    x = Matrix([i for i, j in zip(_("x"), range(n))])
    yT = Matrix([i for i, j in zip(_("y"), range(m))]).T

    def ineq(L, r, op):
        rv = []
        for r in (op(i, j) for i, j in zip(L, r)):
            if r == True:
                continue
            elif r == False:
                return [False]
            if factor:
                f = factor_terms(r)
                if f.lhs.is_Mul and f.rhs % f.lhs.args[0] == 0:
                    assert len(f.lhs.args) == 2, f.lhs
                    k = f.lhs.args[0]
                    r = r.func(sign(k) * f.lhs.args[1], f.rhs // abs(k))
            rv.append(r)
        return rv

    eq = lambda x, d: x[0] - d if x else -d
    F = eq(c * x, d)
    f = eq(yT * b, d)
    return (F, ineq(A * x, b, Ge)), (f, ineq(yT * A, c, Le))


def _rel_as_nonpos(constr, syms):
    """return `(np, d, aux)` where `np` is a list of nonpositive
    expressions that represent the given constraints (possibly
    rewritten in terms of auxilliary variables) expressible with
    nonnegative symbols, and `d` is a dictionary mapping a given
    symbols to an expression with an auxilliary variable. In some
    cases a symbol will be used as part of the change of variables,
    e.g. x: x - z1 instead of x: z1 - z2.

    If any constraint is False/empty, return None. All variables in
    ``constr`` are assumed to be unbounded unless explicitly indicated
    otherwise with a univariate constraint, e.g. ``x >= 0`` will
    restrict ``x`` to nonnegative values.

    The ``syms`` must be included so all symbols can be given an
    unbounded assumption if they are not otherwise bound with
    univariate conditions like ``x <= 3``.

    Examples
    ========

    >>> from sympy.solvers.simplex import _rel_as_nonpos
    >>> from sympy.abc import x, y
    >>> _rel_as_nonpos([x >= y, x >= 0, y >= 0], (x, y))
    ([-x + y], {}, [])
    >>> _rel_as_nonpos([x >= 3, x <= 5], [x])
    ([_z1 - 2], {x: _z1 + 3}, [_z1])
    >>> _rel_as_nonpos([x <= 5], [x])
    ([], {x: 5 - _z1}, [_z1])
    >>> _rel_as_nonpos([x >= 1], [x])
    ([], {x: _z1 + 1}, [_z1])
    """
    r = {}  # replacements to handle change of variables
    np = []  # nonpositive expressions
    aux = []  # auxilliary symbols added
    ui = numbered_symbols("z", start=1, cls=Dummy)  # auxilliary symbols
    univariate = {}  # {x: interval} for univariate constraints
    unbound = []  # symbols designated as unbound
    syms = set(syms)  # the expected syms of the system

    # separate out univariates
    for i in constr:
        if i == True:
            continue  # ignore
        if i == False:
            return  # no solution
        if i.has(S.Infinity, S.NegativeInfinity):
            raise ValueError("only finite bounds are permitted")
        if isinstance(i, (Le, Ge)):
            i = i.lts - i.gts
            freei = i.free_symbols
            if freei - syms:
                raise ValueError(
                    "unexpected symbol(s) in constraint: %s" % (freei - syms)
                )
            if len(freei) > 1:
                np.append(i)
            elif freei:
                x = freei.pop()
                if x in unbound:
                    continue  # will handle later
                ivl = Le(i, 0, evaluate=False).as_set()
                if x not in univariate:
                    univariate[x] = ivl
                else:
                    univariate[x] &= ivl
            elif i:
                return False
        else:
            raise TypeError(filldedent("""
                only equalities like Eq(x, y) or non-strict
                inequalities like x >= y are allowed in lp, not %s""" % i))

    # introduce auxilliary variables as needed for univariate
    # inequalities
    for x in syms:
        i = univariate.get(x, True)
        if not i:
            return None  # no solution possible
        if i == True:
            unbound.append(x)
            continue
        a, b = i.inf, i.sup
        if a.is_infinite:
            u = next(ui)
            r[x] = b - u
            aux.append(u)
        elif b.is_infinite:
            if a:
                u = next(ui)
                r[x] = a + u
                aux.append(u)
            else:
                # standard nonnegative relationship
                pass
        else:
            u = next(ui)
            aux.append(u)
            # shift so u = x - a => x = u + a
            r[x] = u + a
            # add constraint for u <= b - a
            # since when u = b-a then x = u + a = b - a + a = b:
            # the upper limit for x
            np.append(u - (b - a))

    # make change of variables for unbound variables
    for x in unbound:
        u = next(ui)
        r[x] = u - x  # reusing x
        aux.append(u)

    return np, r, aux


def _lp_matrices(objective, constraints):
    """return A, B, C, D, r, x+X, X for maximizing
    objective = Cx - D with constraints Ax <= B, introducing
    introducing auxilliary variables, X, as necessary to make
    replacements of symbols as given in r, {xi: expression with Xj},
    so all variables in x+X will take on nonnegative values.

    Every univariate condition creates a semi-infinite
    condition, e.g. a single ``x <= 3`` creates the
    interval ``[-oo, 3]`` while ``x <= 3`` and ``x >= 2``
    create an interval ``[2, 3]``. Variables not in a univariate
    expression will take on nonnegative values.
    """

    # sympify input and collect free symbols
    F = sympify(objective)
    np = [sympify(i) for i in constraints]
    syms = set.union(*[i.free_symbols for i in [F] + np], set())

    # change Eq(x, y) to x - y <= 0 and y - x <= 0
    for i in range(len(np)):
        if isinstance(np[i], Eq):
            np[i] = np[i].lhs - np[i].rhs <= 0
            np.append(-np[i].lhs <= 0)

    # convert constraints to nonpositive expressions
    _ = _rel_as_nonpos(np, syms)
    if _ is None:
        raise InfeasibleLPError(filldedent("""
            Inconsistent/False constraint"""))
    np, r, aux = _

    # do change of variables
    F = F.xreplace(r)
    np = [i.xreplace(r) for i in np]

    # convert to matrices
    xx = list(ordered(syms)) + aux
    A, B = linear_eq_to_matrix(np, xx)
    C, D = linear_eq_to_matrix([F], xx)
    return A, B, C, D, r, xx, aux


def _lp(min_max, f, constr):
    """Return the optimization (min or max) of ``f`` with the given
    constraints. All variables are unbounded unless constrained.

    If `min_max` is 'max' then the results corresponding to the
    maximization of ``f`` will be returned, else the minimization.
    The constraints can be given as Le, Ge or Eq expressions.

    Examples
    ========

    >>> from sympy.solvers.simplex import _lp as lp
    >>> from sympy import Eq
    >>> from sympy.abc import x, y, z
    >>> f = x + y - 2*z
    >>> c = [7*x + 4*y - 7*z <= 3, 3*x - y + 10*z <= 6]
    >>> c += [i >= 0 for i in (x, y, z)]
    >>> lp(min, f, c)
    (-6/5, {x: 0, y: 0, z: 3/5})

    By passing max, the maximum value for f under the constraints
    is returned (if possible):

    >>> lp(max, f, c)
    (3/4, {x: 0, y: 3/4, z: 0})

    Constraints that are equalities will require that the solution
    also satisfy them:

    >>> lp(max, f, c + [Eq(y - 9*x, 1)])
    (5/7, {x: 0, y: 1, z: 1/7})

    All symbols are reported, even if they are not in the objective
    function:

    >>> lp(min, x, [y + x >= 3, x >= 0])
    (0, {x: 0, y: 3})
    """
    # get the matrix components for the system expressed
    # in terms of only nonnegative variables
    A, B, C, D, r, xx, aux = _lp_matrices(f, constr)

    how = str(min_max).lower()
    if "max" in how:
        # _simplex minimizes for Ax <= B so we
        # have to change the sign of the function
        # and negate the optimal value returned
        _o, p, d = _simplex(A, B, -C, -D)
        o = -_o
    elif "min" in how:
        o, p, d = _simplex(A, B, C, D)
    else:
        raise ValueError("expecting min or max")

    # restore original variables and remove aux from p
    p = dict(zip(xx, p))
    if r:  # p has original symbols and auxilliary symbols
        # if r has x: x - z1 use values from p to update
        r = {k: v.xreplace(p) for k, v in r.items()}
        # then use the actual value of x (= x - z1) in p
        p.update(r)
        # don't show aux
        p = {k: p[k] for k in ordered(p) if k not in aux}

    # not returning dual since there may be extra constraints
    # when a variable has finite bounds
    return o, p


def lpmin(f, constr):
    """return minimum of linear equation ``f`` under
    linear constraints expressed using Ge, Le or Eq.

    All variables are unbounded unless constrained.

    Examples
    ========

    >>> from sympy.solvers.simplex import lpmin
    >>> from sympy import Eq
    >>> from sympy.abc import x, y
    >>> lpmin(x, [2*x - 3*y >= -1, Eq(x + 3*y, 2), x <= 2*y])
    (1/3, {x: 1/3, y: 5/9})

    Negative values for variables are permitted unless explicitly
    excluding, so minimizing ``x`` for ``x <= 3`` is an
    unbounded problem while the following has a bounded solution:

    >>> lpmin(x, [x >= 0, x <= 3])
    (0, {x: 0})

    Without indicating that ``x`` is nonnegative, there
    is no minimum for this objective:

    >>> lpmin(x, [x <= 3])
    Traceback (most recent call last):
    ...
    sympy.solvers.simplex.UnboundedLPError:
    Objective function can assume arbitrarily large values!

    See Also
    ========
    linprog, lpmax
    """
    return _lp(min, f, constr)


def lpmax(f, constr):
    """return maximum of linear equation ``f`` under
    linear constraints expressed using Ge, Le or Eq.

    All variables are unbounded unless constrained.

    Examples
    ========

    >>> from sympy.solvers.simplex import lpmax
    >>> from sympy import Eq
    >>> from sympy.abc import x, y
    >>> lpmax(x, [2*x - 3*y >= -1, Eq(x+ 3*y,2), x <= 2*y])
    (4/5, {x: 4/5, y: 2/5})

    Negative values for variables are permitted unless explicitly
    excluding:

    >>> lpmax(x, [x <= -1])
    (-1, {x: -1})

    If a non-negative constraint is added for x, there is no
    possible solution:

    >>> lpmax(x, [x <= -1, x >= 0])
    Traceback (most recent call last):
    ...
    sympy.solvers.simplex.InfeasibleLPError: inconsistent/False constraint

    See Also
    ========
    linprog, lpmin
    """
    return _lp(max, f, constr)


def _handle_bounds(bounds):
    # introduce auxiliary variables as needed for univariate
    # inequalities

    def _make_list(length: int, index_value_pairs):
        li = [0] * length
        for idx, val in index_value_pairs:
            li[idx] = val
        return li

    unbound = []
    row = []
    row2 = []
    b_len = len(bounds)
    for x, (a, b) in enumerate(bounds):
        if a is None and b is None:
            unbound.append(x)
        elif a is None:
            # r[x] = b - u
            b_len += 1
            row.append(_make_list(b_len, [(x, 1), (-1, 1)]))
            row.append(_make_list(b_len, [(x, -1), (-1, -1)]))
            row2.extend([[b], [-b]])
        elif b is None:
            if a:
                # r[x] = a + u
                b_len += 1
                row.append(_make_list(b_len, [(x, 1), (-1, -1)]))
                row.append(_make_list(b_len, [(x, -1), (-1, 1)]))
                row2.extend([[a], [-a]])
            else:
                # standard nonnegative relationship
                pass
        else:
            # r[x] = u + a
            b_len += 1
            row.append(_make_list(b_len, [(x, 1), (-1, -1)]))
            row.append(_make_list(b_len, [(x, -1), (-1, 1)]))
            # u <= b - a
            row.append(_make_list(b_len, [(-1, 1)]))
            row2.extend([[a], [-a], [b - a]])

    # make change of variables for unbound variables
    for x in unbound:
        # r[x] = u - v
        b_len += 2
        row.append(_make_list(b_len, [(x, 1), (-1, 1), (-2, -1)]))
        row.append(_make_list(b_len, [(x, -1), (-1, -1), (-2, 1)]))
        row2.extend([[0], [0]])

    return Matrix([r + [0]*(b_len - len(r)) for r in row]), Matrix(row2)


def linprog(c, A=None, b=None, A_eq=None, b_eq=None, bounds=None):
    """Return the minimization of ``c*x`` with the given
    constraints ``A*x <= b`` and ``A_eq*x = b_eq``. Unless bounds
    are given, variables will have nonnegative values in the solution.

    If ``A`` is not given, then the dimension of the system will
    be determined by the length of ``C``.

    By default, all variables will be nonnegative. If ``bounds``
    is given as a single tuple, ``(lo, hi)``, then all variables
    will be constrained to be between ``lo`` and ``hi``. Use
    None for a ``lo`` or ``hi`` if it is unconstrained in the
    negative or positive direction, respectively, e.g.
    ``(None, 0)`` indicates nonpositive values. To set
    individual ranges, pass a list with length equal to the
    number of columns in ``A``, each element being a tuple; if
    only a few variables take on non-default values they can be
    passed as a dictionary with keys giving the corresponding
    column to which the variable is assigned, e.g. ``bounds={2:
    (1, 4)}`` would limit the 3rd variable to have a value in
    range ``[1, 4]``.

    Examples
    ========

    >>> from sympy.solvers.simplex import linprog
    >>> from sympy import symbols, Eq, linear_eq_to_matrix as M, Matrix
    >>> x = x1, x2, x3, x4 = symbols('x1:5')
    >>> X = Matrix(x)
    >>> c, d = M(5*x2 + x3 + 4*x4 - x1, x)
    >>> a, b = M([5*x2 + 2*x3 + 5*x4 - (x1 + 5)], x)
    >>> aeq, beq = M([Eq(3*x2 + x4, 2), Eq(-x1 + x3 + 2*x4, 1)], x)
    >>> constr = [i <= j for i,j in zip(a*X, b)]
    >>> constr += [Eq(i, j) for i,j in zip(aeq*X, beq)]
    >>> linprog(c, a, b, aeq, beq)
    (9/2, [0, 1/2, 0, 1/2])
    >>> assert all(i.subs(dict(zip(x, _[1]))) for i in constr)

    See Also
    ========
    lpmin, lpmax
    """

    ## the objective
    C = Matrix(c)
    if C.rows != 1 and C.cols == 1:
        C = C.T
    if C.rows != 1:
        raise ValueError("C must be a single row.")

    ## the inequalities
    if not A:
        if b:
            raise ValueError("A and b must both be given")
        # the governing equations will be simple constraints
        # on variables
        A, b = zeros(0, C.cols), zeros(C.cols, 1)
    else:
        A, b = [Matrix(i) for i in (A, b)]

    if A.cols != C.cols:
        raise ValueError("number of columns in A and C must match")

    ## the equalities
    if A_eq is None:
        if not b_eq is None:
            raise ValueError("A_eq and b_eq must both be given")
    else:
        A_eq, b_eq = [Matrix(i) for i in (A_eq, b_eq)]
        # if x == y then x <= y and x >= y (-x <= -y)
        A = A.col_join(A_eq)
        A = A.col_join(-A_eq)
        b = b.col_join(b_eq)
        b = b.col_join(-b_eq)

    if not (bounds is None or bounds == {} or bounds == (0, None)):
        ## the bounds are interpreted
        if type(bounds) is tuple and len(bounds) == 2:
            bounds = [bounds] * A.cols
        elif len(bounds) == A.cols and all(
                type(i) is tuple and len(i) == 2 for i in bounds):
            pass # individual bounds
        elif type(bounds) is dict and all(
                type(i) is tuple and len(i) == 2
                for i in bounds.values()):
            # sparse bounds
            db = bounds
            bounds = [(0, None)] * A.cols
            while db:
                i, j = db.popitem()
                bounds[i] = j  # IndexError if out-of-bounds indices
        else:
            raise ValueError("unexpected bounds %s" % bounds)
        A_, b_ = _handle_bounds(bounds)
        aux = A_.cols - A.cols
        if A:
            A = Matrix([[A, zeros(A.rows, aux)], [A_]])
            b = b.col_join(b_)
        else:
            A = A_
            b = b_
        C = C.row_join(zeros(1, aux))
    else:
        aux = -A.cols  # set so -aux will give all cols below

    o, p, d = _simplex(A, b, C)
    return o, p[:-aux]  # don't include aux values

def show_linprog(c, A=None, b=None, A_eq=None, b_eq=None, bounds=None):
    from sympy import symbols
    ## the objective
    C = Matrix(c)
    if C.rows != 1 and C.cols == 1:
        C = C.T
    if C.rows != 1:
        raise ValueError("C must be a single row.")

    ## the inequalities
    if not A:
        if b:
            raise ValueError("A and b must both be given")
        # the governing equations will be simple constraints
        # on variables
        A, b = zeros(0, C.cols), zeros(C.cols, 1)
    else:
        A, b = [Matrix(i) for i in (A, b)]

    if A.cols != C.cols:
        raise ValueError("number of columns in A and C must match")

    ## the equalities
    if A_eq is None:
        if not b_eq is None:
            raise ValueError("A_eq and b_eq must both be given")
    else:
        A_eq, b_eq = [Matrix(i) for i in (A_eq, b_eq)]

    if not (bounds is None or bounds == {} or bounds == (0, None)):
        ## the bounds are interpreted
        if type(bounds) is tuple and len(bounds) == 2:
            bounds = [bounds] * A.cols
        elif len(bounds) == A.cols and all(
                type(i) is tuple and len(i) == 2 for i in bounds):
            pass # individual bounds
        elif type(bounds) is dict and all(
                type(i) is tuple and len(i) == 2
                for i in bounds.values()):
            # sparse bounds
            db = bounds
            bounds = [(0, None)] * A.cols
            while db:
                i, j = db.popitem()
                bounds[i] = j  # IndexError if out-of-bounds indices
        else:
            raise ValueError("unexpected bounds %s" % bounds)

    x = Matrix(symbols('x1:%s' % (A.cols+1)))
    f,c = (C*x)[0], [i<=j for i,j in zip(A*x, b)] + [Eq(i,j) for i,j in zip(A_eq*x,b_eq)]
    for i, (lo, hi) in enumerate(bounds):
        if lo is not None:
            c.append(x[i]>=lo)
        if hi is not None:
            c.append(x[i]<=hi)
    return f,c
