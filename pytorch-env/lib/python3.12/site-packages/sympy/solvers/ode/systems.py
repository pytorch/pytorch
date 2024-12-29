from sympy.core import Add, Mul, S
from sympy.core.containers import Tuple
from sympy.core.exprtools import factor_terms
from sympy.core.numbers import I
from sympy.core.relational import Eq, Equality
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Dummy, Symbol
from sympy.core.function import (expand_mul, expand, Derivative,
                                 AppliedUndef, Function, Subs)
from sympy.functions import (exp, im, cos, sin, re, Piecewise,
                             piecewise_fold, sqrt, log)
from sympy.functions.combinatorial.factorials import factorial
from sympy.matrices import zeros, Matrix, NonSquareMatrixError, MatrixBase, eye
from sympy.polys import Poly, together
from sympy.simplify import collect, radsimp, signsimp # type: ignore
from sympy.simplify.powsimp import powdenest, powsimp
from sympy.simplify.ratsimp import ratsimp
from sympy.simplify.simplify import simplify
from sympy.sets.sets import FiniteSet
from sympy.solvers.deutils import ode_order
from sympy.solvers.solveset import NonlinearError, solveset
from sympy.utilities.iterables import (connected_components, iterable,
                                       strongly_connected_components)
from sympy.utilities.misc import filldedent
from sympy.integrals.integrals import Integral, integrate


def _get_func_order(eqs, funcs):
    return {func: max(ode_order(eq, func) for eq in eqs) for func in funcs}


class ODEOrderError(ValueError):
    """Raised by linear_ode_to_matrix if the system has the wrong order"""
    pass


class ODENonlinearError(NonlinearError):
    """Raised by linear_ode_to_matrix if the system is nonlinear"""
    pass


def _simpsol(soleq):
    lhs = soleq.lhs
    sol = soleq.rhs
    sol = powsimp(sol)
    gens = list(sol.atoms(exp))
    p = Poly(sol, *gens, expand=False)
    gens = [factor_terms(g) for g in gens]
    if not gens:
        gens = p.gens
    syms = [Symbol('C1'), Symbol('C2')]
    terms = []
    for coeff, monom in zip(p.coeffs(), p.monoms()):
        coeff = piecewise_fold(coeff)
        if isinstance(coeff, Piecewise):
            coeff = Piecewise(*((ratsimp(coef).collect(syms), cond) for coef, cond in coeff.args))
        else:
            coeff = ratsimp(coeff).collect(syms)
        monom = Mul(*(g ** i for g, i in zip(gens, monom)))
        terms.append(coeff * monom)
    return Eq(lhs, Add(*terms))


def _solsimp(e, t):
    no_t, has_t = powsimp(expand_mul(e)).as_independent(t)

    no_t = ratsimp(no_t)
    has_t = has_t.replace(exp, lambda a: exp(factor_terms(a)))

    return no_t + has_t


def simpsol(sol, wrt1, wrt2, doit=True):
    """Simplify solutions from dsolve_system."""

    # The parameter sol is the solution as returned by dsolve (list of Eq).
    #
    # The parameters wrt1 and wrt2 are lists of symbols to be collected for
    # with those in wrt1 being collected for first. This allows for collecting
    # on any factors involving the independent variable before collecting on
    # the integration constants or vice versa using e.g.:
    #
    #     sol = simpsol(sol, [t], [C1, C2])  # t first, constants after
    #     sol = simpsol(sol, [C1, C2], [t])  # constants first, t after
    #
    # If doit=True (default) then simpsol will begin by evaluating any
    # unevaluated integrals. Since many integrals will appear multiple times
    # in the solutions this is done intelligently by computing each integral
    # only once.
    #
    # The strategy is to first perform simple cancellation with factor_terms
    # and then multiply out all brackets with expand_mul. This gives an Add
    # with many terms.
    #
    # We split each term into two multiplicative factors dep and coeff where
    # all factors that involve wrt1 are in dep and any constant factors are in
    # coeff e.g.
    #         sqrt(2)*C1*exp(t) -> ( exp(t), sqrt(2)*C1 )
    #
    # The dep factors are simplified using powsimp to combine expanded
    # exponential factors e.g.
    #              exp(a*t)*exp(b*t) -> exp(t*(a+b))
    #
    # We then collect coefficients for all terms having the same (simplified)
    # dep. The coefficients are then simplified using together and ratsimp and
    # lastly by recursively applying the same transformation to the
    # coefficients to collect on wrt2.
    #
    # Finally the result is recombined into an Add and signsimp is used to
    # normalise any minus signs.

    def simprhs(rhs, rep, wrt1, wrt2):
        """Simplify the rhs of an ODE solution"""
        if rep:
            rhs = rhs.subs(rep)
        rhs = factor_terms(rhs)
        rhs = simp_coeff_dep(rhs, wrt1, wrt2)
        rhs = signsimp(rhs)
        return rhs

    def simp_coeff_dep(expr, wrt1, wrt2=None):
        """Split rhs into terms, split terms into dep and coeff and collect on dep"""
        add_dep_terms = lambda e: e.is_Add and e.has(*wrt1)
        expandable = lambda e: e.is_Mul and any(map(add_dep_terms, e.args))
        expand_func = lambda e: expand_mul(e, deep=False)
        expand_mul_mod = lambda e: e.replace(expandable, expand_func)
        terms = Add.make_args(expand_mul_mod(expr))
        dc = {}
        for term in terms:
            coeff, dep = term.as_independent(*wrt1, as_Add=False)
            # Collect together the coefficients for terms that have the same
            # dependence on wrt1 (after dep is normalised using simpdep).
            dep = simpdep(dep, wrt1)

            # See if the dependence on t cancels out...
            if dep is not S.One:
                dep2 = factor_terms(dep)
                if not dep2.has(*wrt1):
                    coeff *= dep2
                    dep = S.One

            if dep not in dc:
                dc[dep] = coeff
            else:
                dc[dep] += coeff
        # Apply the method recursively to the coefficients but this time
        # collecting on wrt2 rather than wrt2.
        termpairs = ((simpcoeff(c, wrt2), d) for d, c in dc.items())
        if wrt2 is not None:
            termpairs = ((simp_coeff_dep(c, wrt2), d) for c, d in termpairs)
        return Add(*(c * d for c, d in termpairs))

    def simpdep(term, wrt1):
        """Normalise factors involving t with powsimp and recombine exp"""
        def canonicalise(a):
            # Using factor_terms here isn't quite right because it leads to things
            # like exp(t*(1+t)) that we don't want. We do want to cancel factors
            # and pull out a common denominator but ideally the numerator would be
            # expressed as a standard form polynomial in t so we expand_mul
            # and collect afterwards.
            a = factor_terms(a)
            num, den = a.as_numer_denom()
            num = expand_mul(num)
            num = collect(num, wrt1)
            return num / den

        term = powsimp(term)
        rep = {e: exp(canonicalise(e.args[0])) for e in term.atoms(exp)}
        term = term.subs(rep)
        return term

    def simpcoeff(coeff, wrt2):
        """Bring to a common fraction and cancel with ratsimp"""
        coeff = together(coeff)
        if coeff.is_polynomial():
            # Calling ratsimp can be expensive. The main reason is to simplify
            # sums of terms with irrational denominators so we limit ourselves
            # to the case where the expression is polynomial in any symbols.
            # Maybe there's a better approach...
            coeff = ratsimp(radsimp(coeff))
        # collect on secondary variables first and any remaining symbols after
        if wrt2 is not None:
            syms = list(wrt2) + list(ordered(coeff.free_symbols - set(wrt2)))
        else:
            syms = list(ordered(coeff.free_symbols))
        coeff = collect(coeff, syms)
        coeff = together(coeff)
        return coeff

    # There are often repeated integrals. Collect unique integrals and
    # evaluate each once and then substitute into the final result to replace
    # all occurrences in each of the solution equations.
    if doit:
        integrals = set().union(*(s.atoms(Integral) for s in sol))
        rep = {i: factor_terms(i).doit() for i in integrals}
    else:
        rep = {}

    sol = [Eq(s.lhs, simprhs(s.rhs, rep, wrt1, wrt2)) for s in sol]
    return sol


def linodesolve_type(A, t, b=None):
    r"""
    Helper function that determines the type of the system of ODEs for solving with :obj:`sympy.solvers.ode.systems.linodesolve()`

    Explanation
    ===========

    This function takes in the coefficient matrix and/or the non-homogeneous term
    and returns the type of the equation that can be solved by :obj:`sympy.solvers.ode.systems.linodesolve()`.

    If the system is constant coefficient homogeneous, then "type1" is returned

    If the system is constant coefficient non-homogeneous, then "type2" is returned

    If the system is non-constant coefficient homogeneous, then "type3" is returned

    If the system is non-constant coefficient non-homogeneous, then "type4" is returned

    If the system has a non-constant coefficient matrix which can be factorized into constant
    coefficient matrix, then "type5" or "type6" is returned for when the system is homogeneous or
    non-homogeneous respectively.

    Note that, if the system of ODEs is of "type3" or "type4", then along with the type,
    the commutative antiderivative of the coefficient matrix is also returned.

    If the system cannot be solved by :obj:`sympy.solvers.ode.systems.linodesolve()`, then
    NotImplementedError is raised.

    Parameters
    ==========

    A : Matrix
        Coefficient matrix of the system of ODEs
    b : Matrix or None
        Non-homogeneous term of the system. The default value is None.
        If this argument is None, then the system is assumed to be homogeneous.

    Examples
    ========

    >>> from sympy import symbols, Matrix
    >>> from sympy.solvers.ode.systems import linodesolve_type
    >>> t = symbols("t")
    >>> A = Matrix([[1, 1], [2, 3]])
    >>> b = Matrix([t, 1])

    >>> linodesolve_type(A, t)
    {'antiderivative': None, 'type_of_equation': 'type1'}

    >>> linodesolve_type(A, t, b=b)
    {'antiderivative': None, 'type_of_equation': 'type2'}

    >>> A_t = Matrix([[1, t], [-t, 1]])

    >>> linodesolve_type(A_t, t)
    {'antiderivative': Matrix([
    [      t, t**2/2],
    [-t**2/2,      t]]), 'type_of_equation': 'type3'}

    >>> linodesolve_type(A_t, t, b=b)
    {'antiderivative': Matrix([
    [      t, t**2/2],
    [-t**2/2,      t]]), 'type_of_equation': 'type4'}

    >>> A_non_commutative = Matrix([[1, t], [t, -1]])
    >>> linodesolve_type(A_non_commutative, t)
    Traceback (most recent call last):
    ...
    NotImplementedError:
    The system does not have a commutative antiderivative, it cannot be
    solved by linodesolve.

    Returns
    =======

    Dict

    Raises
    ======

    NotImplementedError
        When the coefficient matrix does not have a commutative antiderivative

    See Also
    ========

    linodesolve: Function for which linodesolve_type gets the information

    """

    match = {}
    is_non_constant = not _matrix_is_constant(A, t)
    is_non_homogeneous = not (b is None or b.is_zero_matrix)
    type = "type{}".format(int("{}{}".format(int(is_non_constant), int(is_non_homogeneous)), 2) + 1)

    B = None
    match.update({"type_of_equation": type, "antiderivative": B})

    if is_non_constant:
        B, is_commuting = _is_commutative_anti_derivative(A, t)
        if not is_commuting:
            raise NotImplementedError(filldedent('''
                The system does not have a commutative antiderivative, it cannot be solved
                by linodesolve.
            '''))

        match['antiderivative'] = B
        match.update(_first_order_type5_6_subs(A, t, b=b))

    return match


def _first_order_type5_6_subs(A, t, b=None):
    match = {}

    factor_terms = _factor_matrix(A, t)
    is_homogeneous = b is None or b.is_zero_matrix

    if factor_terms is not None:
        t_ = Symbol("{}_".format(t))
        F_t = integrate(factor_terms[0], t)
        inverse = solveset(Eq(t_, F_t), t)

        # Note: A simple way to check if a function is invertible
        # or not.
        if isinstance(inverse, FiniteSet) and not inverse.has(Piecewise)\
            and len(inverse) == 1:

            A = factor_terms[1]
            if not is_homogeneous:
                b = b / factor_terms[0]
                b = b.subs(t, list(inverse)[0])
            type = "type{}".format(5 + (not is_homogeneous))
            match.update({'func_coeff': A, 'tau': F_t,
                          't_': t_, 'type_of_equation': type, 'rhs': b})

    return match


def linear_ode_to_matrix(eqs, funcs, t, order):
    r"""
    Convert a linear system of ODEs to matrix form

    Explanation
    ===========

    Express a system of linear ordinary differential equations as a single
    matrix differential equation [1]. For example the system $x' = x + y + 1$
    and $y' = x - y$ can be represented as

    .. math:: A_1 X' = A_0 X + b

    where $A_1$ and $A_0$ are $2 \times 2$ matrices and $b$, $X$ and $X'$ are
    $2 \times 1$ matrices with $X = [x, y]^T$.

    Higher-order systems are represented with additional matrices e.g. a
    second-order system would look like

    .. math:: A_2 X'' =  A_1 X' + A_0 X  + b

    Examples
    ========

    >>> from sympy import Function, Symbol, Matrix, Eq
    >>> from sympy.solvers.ode.systems import linear_ode_to_matrix
    >>> t = Symbol('t')
    >>> x = Function('x')
    >>> y = Function('y')

    We can create a system of linear ODEs like

    >>> eqs = [
    ...     Eq(x(t).diff(t), x(t) + y(t) + 1),
    ...     Eq(y(t).diff(t), x(t) - y(t)),
    ... ]
    >>> funcs = [x(t), y(t)]
    >>> order = 1 # 1st order system

    Now ``linear_ode_to_matrix`` can represent this as a matrix
    differential equation.

    >>> (A1, A0), b = linear_ode_to_matrix(eqs, funcs, t, order)
    >>> A1
    Matrix([
    [1, 0],
    [0, 1]])
    >>> A0
    Matrix([
    [1, 1],
    [1,  -1]])
    >>> b
    Matrix([
    [1],
    [0]])

    The original equations can be recovered from these matrices:

    >>> eqs_mat = Matrix([eq.lhs - eq.rhs for eq in eqs])
    >>> X = Matrix(funcs)
    >>> A1 * X.diff(t) - A0 * X - b == eqs_mat
    True

    If the system of equations has a maximum order greater than the
    order of the system specified, a ODEOrderError exception is raised.

    >>> eqs = [Eq(x(t).diff(t, 2), x(t).diff(t) + x(t)), Eq(y(t).diff(t), y(t) + x(t))]
    >>> linear_ode_to_matrix(eqs, funcs, t, 1)
    Traceback (most recent call last):
    ...
    ODEOrderError: Cannot represent system in 1-order form

    If the system of equations is nonlinear, then ODENonlinearError is
    raised.

    >>> eqs = [Eq(x(t).diff(t), x(t) + y(t)), Eq(y(t).diff(t), y(t)**2 + x(t))]
    >>> linear_ode_to_matrix(eqs, funcs, t, 1)
    Traceback (most recent call last):
    ...
    ODENonlinearError: The system of ODEs is nonlinear.

    Parameters
    ==========

    eqs : list of SymPy expressions or equalities
        The equations as expressions (assumed equal to zero).
    funcs : list of applied functions
        The dependent variables of the system of ODEs.
    t : symbol
        The independent variable.
    order : int
        The order of the system of ODEs.

    Returns
    =======

    The tuple ``(As, b)`` where ``As`` is a tuple of matrices and ``b`` is the
    the matrix representing the rhs of the matrix equation.

    Raises
    ======

    ODEOrderError
        When the system of ODEs have an order greater than what was specified
    ODENonlinearError
        When the system of ODEs is nonlinear

    See Also
    ========

    linear_eq_to_matrix: for systems of linear algebraic equations.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Matrix_differential_equation

    """
    from sympy.solvers.solveset import linear_eq_to_matrix

    if any(ode_order(eq, func) > order for eq in eqs for func in funcs):
        msg = "Cannot represent system in {}-order form"
        raise ODEOrderError(msg.format(order))

    As = []

    for o in range(order, -1, -1):
        # Work from the highest derivative down
        syms = [func.diff(t, o) for func in funcs]

        # Ai is the matrix for X(t).diff(t, o)
        # eqs is minus the remainder of the equations.
        try:
            Ai, b = linear_eq_to_matrix(eqs, syms)
        except NonlinearError:
            raise ODENonlinearError("The system of ODEs is nonlinear.")

        Ai = Ai.applyfunc(expand_mul)

        As.append(Ai if o == order else -Ai)

        if o:
            eqs = [-eq for eq in b]
        else:
            rhs = b

    return As, rhs


def matrix_exp(A, t):
    r"""
    Matrix exponential $\exp(A*t)$ for the matrix ``A`` and scalar ``t``.

    Explanation
    ===========

    This functions returns the $\exp(A*t)$ by doing a simple
    matrix multiplication:

    .. math:: \exp(A*t) = P * expJ * P^{-1}

    where $expJ$ is $\exp(J*t)$. $J$ is the Jordan normal
    form of $A$ and $P$ is matrix such that:

    .. math:: A = P * J * P^{-1}

    The matrix exponential $\exp(A*t)$ appears in the solution of linear
    differential equations. For example if $x$ is a vector and $A$ is a matrix
    then the initial value problem

    .. math:: \frac{dx(t)}{dt} = A \times x(t),   x(0) = x0

    has the unique solution

    .. math:: x(t) = \exp(A t) x0

    Examples
    ========

    >>> from sympy import Symbol, Matrix, pprint
    >>> from sympy.solvers.ode.systems import matrix_exp
    >>> t = Symbol('t')

    We will consider a 2x2 matrix for comupting the exponential

    >>> A = Matrix([[2, -5], [2, -4]])
    >>> pprint(A)
    [2  -5]
    [     ]
    [2  -4]

    Now, exp(A*t) is given as follows:

    >>> pprint(matrix_exp(A, t))
    [   -t           -t                    -t              ]
    [3*e  *sin(t) + e  *cos(t)         -5*e  *sin(t)       ]
    [                                                      ]
    [         -t                     -t           -t       ]
    [      2*e  *sin(t)         - 3*e  *sin(t) + e  *cos(t)]

    Parameters
    ==========

    A : Matrix
        The matrix $A$ in the expression $\exp(A*t)$
    t : Symbol
        The independent variable

    See Also
    ========

    matrix_exp_jordan_form: For exponential of Jordan normal form

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Jordan_normal_form
    .. [2] https://en.wikipedia.org/wiki/Matrix_exponential

    """
    P, expJ = matrix_exp_jordan_form(A, t)
    return P * expJ * P.inv()


def matrix_exp_jordan_form(A, t):
    r"""
    Matrix exponential $\exp(A*t)$ for the matrix *A* and scalar *t*.

    Explanation
    ===========

    Returns the Jordan form of the $\exp(A*t)$ along with the matrix $P$ such that:

    .. math::
        \exp(A*t) = P * expJ * P^{-1}

    Examples
    ========

    >>> from sympy import Matrix, Symbol
    >>> from sympy.solvers.ode.systems import matrix_exp, matrix_exp_jordan_form
    >>> t = Symbol('t')

    We will consider a 2x2 defective matrix. This shows that our method
    works even for defective matrices.

    >>> A = Matrix([[1, 1], [0, 1]])

    It can be observed that this function gives us the Jordan normal form
    and the required invertible matrix P.

    >>> P, expJ = matrix_exp_jordan_form(A, t)

    Here, it is shown that P and expJ returned by this function is correct
    as they satisfy the formula: P * expJ * P_inverse = exp(A*t).

    >>> P * expJ * P.inv() == matrix_exp(A, t)
    True

    Parameters
    ==========

    A : Matrix
        The matrix $A$ in the expression $\exp(A*t)$
    t : Symbol
        The independent variable

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Defective_matrix
    .. [2] https://en.wikipedia.org/wiki/Jordan_matrix
    .. [3] https://en.wikipedia.org/wiki/Jordan_normal_form

    """

    N, M = A.shape
    if N != M:
        raise ValueError('Needed square matrix but got shape (%s, %s)' % (N, M))
    elif A.has(t):
        raise ValueError('Matrix A should not depend on t')

    def jordan_chains(A):
        '''Chains from Jordan normal form analogous to M.eigenvects().
        Returns a dict with eignevalues as keys like:
            {e1: [[v111,v112,...], [v121, v122,...]], e2:...}
        where vijk is the kth vector in the jth chain for eigenvalue i.
        '''
        P, blocks = A.jordan_cells()
        basis = [P[:,i] for i in range(P.shape[1])]
        n = 0
        chains = {}
        for b in blocks:
            eigval = b[0, 0]
            size = b.shape[0]
            if eigval not in chains:
                chains[eigval] = []
            chains[eigval].append(basis[n:n+size])
            n += size
        return chains

    eigenchains = jordan_chains(A)

    # Needed for consistency across Python versions
    eigenchains_iter = sorted(eigenchains.items(), key=default_sort_key)
    isreal = not A.has(I)

    blocks = []
    vectors = []
    seen_conjugate = set()
    for e, chains in eigenchains_iter:
        for chain in chains:
            n = len(chain)
            if isreal and e != e.conjugate() and e.conjugate() in eigenchains:
                if e in seen_conjugate:
                    continue
                seen_conjugate.add(e.conjugate())
                exprt = exp(re(e) * t)
                imrt = im(e) * t
                imblock = Matrix([[cos(imrt), sin(imrt)],
                                  [-sin(imrt), cos(imrt)]])
                expJblock2 = Matrix(n, n, lambda i,j:
                        imblock * t**(j-i) / factorial(j-i) if j >= i
                        else zeros(2, 2))
                expJblock = Matrix(2*n, 2*n, lambda i,j: expJblock2[i//2,j//2][i%2,j%2])

                blocks.append(exprt * expJblock)
                for i in range(n):
                    vectors.append(re(chain[i]))
                    vectors.append(im(chain[i]))
            else:
                vectors.extend(chain)
                fun = lambda i,j: t**(j-i)/factorial(j-i) if j >= i else 0
                expJblock = Matrix(n, n, fun)
                blocks.append(exp(e * t) * expJblock)

    expJ = Matrix.diag(*blocks)
    P = Matrix(N, N, lambda i,j: vectors[j][i])

    return P, expJ


# Note: To add a docstring example with tau
def linodesolve(A, t, b=None, B=None, type="auto", doit=False,
                tau=None):
    r"""
    System of n equations linear first-order differential equations

    Explanation
    ===========

    This solver solves the system of ODEs of the following form:

    .. math::
        X'(t) = A(t) X(t) +  b(t)

    Here, $A(t)$ is the coefficient matrix, $X(t)$ is the vector of n independent variables,
    $b(t)$ is the non-homogeneous term and $X'(t)$ is the derivative of $X(t)$

    Depending on the properties of $A(t)$ and $b(t)$, this solver evaluates the solution
    differently.

    When $A(t)$ is constant coefficient matrix and $b(t)$ is zero vector i.e. system is homogeneous,
    the system is "type1". The solution is:

    .. math::
        X(t) = \exp(A t) C

    Here, $C$ is a vector of constants and $A$ is the constant coefficient matrix.

    When $A(t)$ is constant coefficient matrix and $b(t)$ is non-zero i.e. system is non-homogeneous,
    the system is "type2". The solution is:

    .. math::
        X(t) = e^{A t} ( \int e^{- A t} b \,dt + C)

    When $A(t)$ is coefficient matrix such that its commutative with its antiderivative $B(t)$ and
    $b(t)$ is a zero vector i.e. system is homogeneous, the system is "type3". The solution is:

    .. math::
        X(t) = \exp(B(t)) C

    When $A(t)$ is commutative with its antiderivative $B(t)$ and $b(t)$ is non-zero i.e. system is
    non-homogeneous, the system is "type4". The solution is:

    .. math::
        X(t) =  e^{B(t)} ( \int e^{-B(t)} b(t) \,dt + C)

    When $A(t)$ is a coefficient matrix such that it can be factorized into a scalar and a constant
    coefficient matrix:

    .. math::
        A(t) = f(t) * A

    Where $f(t)$ is a scalar expression in the independent variable $t$ and $A$ is a constant matrix,
    then we can do the following substitutions:

    .. math::
        tau = \int f(t) dt, X(t) = Y(tau), b(t) = b(f^{-1}(tau))

    Here, the substitution for the non-homogeneous term is done only when its non-zero.
    Using these substitutions, our original system becomes:

    .. math::
        Y'(tau) = A * Y(tau) + b(tau)/f(tau)

    The above system can be easily solved using the solution for "type1" or "type2" depending
    on the homogeneity of the system. After we get the solution for $Y(tau)$, we substitute the
    solution for $tau$ as $t$ to get back $X(t)$

    .. math::
        X(t) = Y(tau)

    Systems of "type5" and "type6" have a commutative antiderivative but we use this solution
    because its faster to compute.

    The final solution is the general solution for all the four equations since a constant coefficient
    matrix is always commutative with its antidervative.

    An additional feature of this function is, if someone wants to substitute for value of the independent
    variable, they can pass the substitution `tau` and the solution will have the independent variable
    substituted with the passed expression(`tau`).

    Parameters
    ==========

    A : Matrix
        Coefficient matrix of the system of linear first order ODEs.
    t : Symbol
        Independent variable in the system of ODEs.
    b : Matrix or None
        Non-homogeneous term in the system of ODEs. If None is passed,
        a homogeneous system of ODEs is assumed.
    B : Matrix or None
        Antiderivative of the coefficient matrix. If the antiderivative
        is not passed and the solution requires the term, then the solver
        would compute it internally.
    type : String
        Type of the system of ODEs passed. Depending on the type, the
        solution is evaluated. The type values allowed and the corresponding
        system it solves are: "type1" for constant coefficient homogeneous
        "type2" for constant coefficient non-homogeneous, "type3" for non-constant
        coefficient homogeneous, "type4" for non-constant coefficient non-homogeneous,
        "type5" and "type6" for non-constant coefficient homogeneous and non-homogeneous
        systems respectively where the coefficient matrix can be factorized to a constant
        coefficient matrix.
        The default value is "auto" which will let the solver decide the correct type of
        the system passed.
    doit : Boolean
        Evaluate the solution if True, default value is False
    tau: Expression
        Used to substitute for the value of `t` after we get the solution of the system.

    Examples
    ========

    To solve the system of ODEs using this function directly, several things must be
    done in the right order. Wrong inputs to the function will lead to incorrect results.

    >>> from sympy import symbols, Function, Eq
    >>> from sympy.solvers.ode.systems import canonical_odes, linear_ode_to_matrix, linodesolve, linodesolve_type
    >>> from sympy.solvers.ode.subscheck import checkodesol
    >>> f, g = symbols("f, g", cls=Function)
    >>> x, a = symbols("x, a")
    >>> funcs = [f(x), g(x)]
    >>> eqs = [Eq(f(x).diff(x) - f(x), a*g(x) + 1), Eq(g(x).diff(x) + g(x), a*f(x))]

    Here, it is important to note that before we derive the coefficient matrix, it is
    important to get the system of ODEs into the desired form. For that we will use
    :obj:`sympy.solvers.ode.systems.canonical_odes()`.

    >>> eqs = canonical_odes(eqs, funcs, x)
    >>> eqs
    [[Eq(Derivative(f(x), x), a*g(x) + f(x) + 1), Eq(Derivative(g(x), x), a*f(x) - g(x))]]

    Now, we will use :obj:`sympy.solvers.ode.systems.linear_ode_to_matrix()` to get the coefficient matrix and the
    non-homogeneous term if it is there.

    >>> eqs = eqs[0]
    >>> (A1, A0), b = linear_ode_to_matrix(eqs, funcs, x, 1)
    >>> A = A0

    We have the coefficient matrices and the non-homogeneous term ready. Now, we can use
    :obj:`sympy.solvers.ode.systems.linodesolve_type()` to get the information for the system of ODEs
    to finally pass it to the solver.

    >>> system_info = linodesolve_type(A, x, b=b)
    >>> sol_vector = linodesolve(A, x, b=b, B=system_info['antiderivative'], type=system_info['type_of_equation'])

    Now, we can prove if the solution is correct or not by using :obj:`sympy.solvers.ode.checkodesol()`

    >>> sol = [Eq(f, s) for f, s in zip(funcs, sol_vector)]
    >>> checkodesol(eqs, sol)
    (True, [0, 0])

    We can also use the doit method to evaluate the solutions passed by the function.

    >>> sol_vector_evaluated = linodesolve(A, x, b=b, type="type2", doit=True)

    Now, we will look at a system of ODEs which is non-constant.

    >>> eqs = [Eq(f(x).diff(x), f(x) + x*g(x)), Eq(g(x).diff(x), -x*f(x) + g(x))]

    The system defined above is already in the desired form, so we do not have to convert it.

    >>> (A1, A0), b = linear_ode_to_matrix(eqs, funcs, x, 1)
    >>> A = A0

    A user can also pass the commutative antiderivative required for type3 and type4 system of ODEs.
    Passing an incorrect one will lead to incorrect results. If the coefficient matrix is not commutative
    with its antiderivative, then :obj:`sympy.solvers.ode.systems.linodesolve_type()` raises a NotImplementedError.
    If it does have a commutative antiderivative, then the function just returns the information about the system.

    >>> system_info = linodesolve_type(A, x, b=b)

    Now, we can pass the antiderivative as an argument to get the solution. If the system information is not
    passed, then the solver will compute the required arguments internally.

    >>> sol_vector = linodesolve(A, x, b=b)

    Once again, we can verify the solution obtained.

    >>> sol = [Eq(f, s) for f, s in zip(funcs, sol_vector)]
    >>> checkodesol(eqs, sol)
    (True, [0, 0])

    Returns
    =======

    List

    Raises
    ======

    ValueError
        This error is raised when the coefficient matrix, non-homogeneous term
        or the antiderivative, if passed, are not a matrix or
        do not have correct dimensions
    NonSquareMatrixError
        When the coefficient matrix or its antiderivative, if passed is not a
        square matrix
    NotImplementedError
        If the coefficient matrix does not have a commutative antiderivative

    See Also
    ========

    linear_ode_to_matrix: Coefficient matrix computation function
    canonical_odes: System of ODEs representation change
    linodesolve_type: Getting information about systems of ODEs to pass in this solver

    """

    if not isinstance(A, MatrixBase):
        raise ValueError(filldedent('''\
            The coefficients of the system of ODEs should be of type Matrix
        '''))

    if not A.is_square:
        raise NonSquareMatrixError(filldedent('''\
            The coefficient matrix must be a square
        '''))

    if b is not None:
        if not isinstance(b, MatrixBase):
            raise ValueError(filldedent('''\
                The non-homogeneous terms of the system of ODEs should be of type Matrix
            '''))

        if A.rows != b.rows:
            raise ValueError(filldedent('''\
                The system of ODEs should have the same number of non-homogeneous terms and the number of
                equations
            '''))

    if B is not None:
        if not isinstance(B, MatrixBase):
            raise ValueError(filldedent('''\
                The antiderivative of coefficients of the system of ODEs should be of type Matrix
            '''))

        if not B.is_square:
            raise NonSquareMatrixError(filldedent('''\
                The antiderivative of the coefficient matrix must be a square
            '''))

        if A.rows != B.rows:
            raise ValueError(filldedent('''\
                        The coefficient matrix and its antiderivative should have same dimensions
                    '''))

    if not any(type == "type{}".format(i) for i in range(1, 7)) and not type == "auto":
        raise ValueError(filldedent('''\
                    The input type should be a valid one
                '''))

    n = A.rows

    # constants = numbered_symbols(prefix='C', cls=Dummy, start=const_idx+1)
    Cvect = Matrix([Dummy() for _ in range(n)])

    if b is None and any(type == typ for typ in ["type2", "type4", "type6"]):
        b = zeros(n, 1)

    is_transformed = tau is not None
    passed_type = type

    if type == "auto":
        system_info = linodesolve_type(A, t, b=b)
        type = system_info["type_of_equation"]
        B = system_info["antiderivative"]

    if type in ("type5", "type6"):
        is_transformed = True
        if passed_type != "auto":
            if tau is None:
                system_info = _first_order_type5_6_subs(A, t, b=b)
                if not system_info:
                    raise ValueError(filldedent('''
                        The system passed isn't {}.
                    '''.format(type)))

                tau = system_info['tau']
                t = system_info['t_']
                A = system_info['A']
                b = system_info['b']

    intx_wrtt = lambda x: Integral(x, t) if x else 0
    if type in ("type1", "type2", "type5", "type6"):
        P, J = matrix_exp_jordan_form(A, t)
        P = simplify(P)

        if type in ("type1", "type5"):
            sol_vector = P * (J * Cvect)
        else:
            Jinv = J.subs(t, -t)
            sol_vector = P * J * ((Jinv * P.inv() * b).applyfunc(intx_wrtt) + Cvect)
    else:
        if B is None:
            B, _ = _is_commutative_anti_derivative(A, t)

        if type == "type3":
            sol_vector = B.exp() * Cvect
        else:
            sol_vector = B.exp() * (((-B).exp() * b).applyfunc(intx_wrtt) + Cvect)

    if is_transformed:
        sol_vector = sol_vector.subs(t, tau)

    gens = sol_vector.atoms(exp)

    if type != "type1":
        sol_vector = [expand_mul(s) for s in sol_vector]

    sol_vector = [collect(s, ordered(gens), exact=True) for s in sol_vector]

    if doit:
        sol_vector = [s.doit() for s in sol_vector]

    return sol_vector


def _matrix_is_constant(M, t):
    """Checks if the matrix M is independent of t or not."""
    return all(coef.as_independent(t, as_Add=True)[1] == 0 for coef in M)


def canonical_odes(eqs, funcs, t):
    r"""
    Function that solves for highest order derivatives in a system

    Explanation
    ===========

    This function inputs a system of ODEs and based on the system,
    the dependent variables and their highest order, returns the system
    in the following form:

    .. math::
        X'(t) = A(t) X(t) + b(t)

    Here, $X(t)$ is the vector of dependent variables of lower order, $A(t)$ is
    the coefficient matrix, $b(t)$ is the non-homogeneous term and $X'(t)$ is the
    vector of dependent variables in their respective highest order. We use the term
    canonical form to imply the system of ODEs which is of the above form.

    If the system passed has a non-linear term with multiple solutions, then a list of
    systems is returned in its canonical form.

    Parameters
    ==========

    eqs : List
        List of the ODEs
    funcs : List
        List of dependent variables
    t : Symbol
        Independent variable

    Examples
    ========

    >>> from sympy import symbols, Function, Eq, Derivative
    >>> from sympy.solvers.ode.systems import canonical_odes
    >>> f, g = symbols("f g", cls=Function)
    >>> x, y = symbols("x y")
    >>> funcs = [f(x), g(x)]
    >>> eqs = [Eq(f(x).diff(x) - 7*f(x), 12*g(x)), Eq(g(x).diff(x) + g(x), 20*f(x))]

    >>> canonical_eqs = canonical_odes(eqs, funcs, x)
    >>> canonical_eqs
    [[Eq(Derivative(f(x), x), 7*f(x) + 12*g(x)), Eq(Derivative(g(x), x), 20*f(x) - g(x))]]

    >>> system = [Eq(Derivative(f(x), x)**2 - 2*Derivative(f(x), x) + 1, 4), Eq(-y*f(x) + Derivative(g(x), x), 0)]

    >>> canonical_system = canonical_odes(system, funcs, x)
    >>> canonical_system
    [[Eq(Derivative(f(x), x), -1), Eq(Derivative(g(x), x), y*f(x))], [Eq(Derivative(f(x), x), 3), Eq(Derivative(g(x), x), y*f(x))]]

    Returns
    =======

    List

    """
    from sympy.solvers.solvers import solve

    order = _get_func_order(eqs, funcs)

    canon_eqs = solve(eqs, *[func.diff(t, order[func]) for func in funcs], dict=True)

    systems = []
    for eq in canon_eqs:
        system = [Eq(func.diff(t, order[func]), eq[func.diff(t, order[func])]) for func in funcs]
        systems.append(system)

    return systems


def _is_commutative_anti_derivative(A, t):
    r"""
    Helper function for determining if the Matrix passed is commutative with its antiderivative

    Explanation
    ===========

    This function checks if the Matrix $A$ passed is commutative with its antiderivative with respect
    to the independent variable $t$.

    .. math::
        B(t) = \int A(t) dt

    The function outputs two values, first one being the antiderivative $B(t)$, second one being a
    boolean value, if True, then the matrix $A(t)$ passed is commutative with $B(t)$, else the matrix
    passed isn't commutative with $B(t)$.

    Parameters
    ==========

    A : Matrix
        The matrix which has to be checked
    t : Symbol
        Independent variable

    Examples
    ========

    >>> from sympy import symbols, Matrix
    >>> from sympy.solvers.ode.systems import _is_commutative_anti_derivative
    >>> t = symbols("t")
    >>> A = Matrix([[1, t], [-t, 1]])

    >>> B, is_commuting = _is_commutative_anti_derivative(A, t)
    >>> is_commuting
    True

    Returns
    =======

    Matrix, Boolean

    """
    B = integrate(A, t)
    is_commuting = (B*A - A*B).applyfunc(expand).applyfunc(factor_terms).is_zero_matrix

    is_commuting = False if is_commuting is None else is_commuting

    return B, is_commuting


def _factor_matrix(A, t):
    term = None
    for element in A:
        temp_term = element.as_independent(t)[1]
        if temp_term.has(t):
            term = temp_term
            break

    if term is not None:
        A_factored = (A/term).applyfunc(ratsimp)
        can_factor = _matrix_is_constant(A_factored, t)
        term = (term, A_factored) if can_factor else None

    return term


def _is_second_order_type2(A, t):
    term = _factor_matrix(A, t)
    is_type2 = False

    if term is not None:
        term = 1/term[0]
        is_type2 = term.is_polynomial()

    if is_type2:
        poly = Poly(term.expand(), t)
        monoms = poly.monoms()

        if monoms[0][0] in (2, 4):
            cs = _get_poly_coeffs(poly, 4)
            a, b, c, d, e = cs

            a1 = powdenest(sqrt(a), force=True)
            c1 = powdenest(sqrt(e), force=True)
            b1 = powdenest(sqrt(c - 2*a1*c1), force=True)

            is_type2 = (b == 2*a1*b1) and (d == 2*b1*c1)
            term = a1*t**2 + b1*t + c1

        else:
            is_type2 = False

    return is_type2, term


def _get_poly_coeffs(poly, order):
    cs = [0 for _ in range(order+1)]
    for c, m in zip(poly.coeffs(), poly.monoms()):
        cs[-1-m[0]] = c
    return cs


def _match_second_order_type(A1, A0, t, b=None):
    r"""
    Works only for second order system in its canonical form.

    Type 0: Constant coefficient matrix, can be simply solved by
            introducing dummy variables.
    Type 1: When the substitution: $U = t*X' - X$ works for reducing
            the second order system to first order system.
    Type 2: When the system is of the form: $poly * X'' = A*X$ where
            $poly$ is square of a quadratic polynomial with respect to
            *t* and $A$ is a constant coefficient matrix.

    """
    match = {"type_of_equation": "type0"}
    n = A1.shape[0]

    if _matrix_is_constant(A1, t) and _matrix_is_constant(A0, t):
        return match

    if (A1 + A0*t).applyfunc(expand_mul).is_zero_matrix:
        match.update({"type_of_equation": "type1", "A1": A1})

    elif A1.is_zero_matrix and (b is None or b.is_zero_matrix):
        is_type2, term = _is_second_order_type2(A0, t)
        if is_type2:
            a, b, c = _get_poly_coeffs(Poly(term, t), 2)
            A = (A0*(term**2).expand()).applyfunc(ratsimp) + (b**2/4 - a*c)*eye(n, n)
            tau = integrate(1/term, t)
            t_ = Symbol("{}_".format(t))
            match.update({"type_of_equation": "type2", "A0": A,
                          "g(t)": sqrt(term), "tau": tau, "is_transformed": True,
                          "t_": t_})

    return match


def _second_order_subs_type1(A, b, funcs, t):
    r"""
    For a linear, second order system of ODEs, a particular substitution.

    A system of the below form can be reduced to a linear first order system of
    ODEs:
    .. math::
        X'' = A(t) * (t*X' - X) + b(t)

    By substituting:
    .. math::  U = t*X' - X

    To get the system:
    .. math::  U' = t*(A(t)*U + b(t))

    Where $U$ is the vector of dependent variables, $X$ is the vector of dependent
    variables in `funcs` and $X'$ is the first order derivative of $X$ with respect to
    $t$. It may or may not reduce the system into linear first order system of ODEs.

    Then a check is made to determine if the system passed can be reduced or not, if
    this substitution works, then the system is reduced and its solved for the new
    substitution. After we get the solution for $U$:

    .. math::  U = a(t)

    We substitute and return the reduced system:

    .. math::
        a(t) = t*X' - X

    Parameters
    ==========

    A: Matrix
        Coefficient matrix($A(t)*t$) of the second order system of this form.
    b: Matrix
        Non-homogeneous term($b(t)$) of the system of ODEs.
    funcs: List
        List of dependent variables
    t: Symbol
        Independent variable of the system of ODEs.

    Returns
    =======

    List

    """

    U = Matrix([t*func.diff(t) - func for func in funcs])

    sol = linodesolve(A, t, t*b)
    reduced_eqs = [Eq(u, s) for s, u in zip(sol, U)]
    reduced_eqs = canonical_odes(reduced_eqs, funcs, t)[0]

    return reduced_eqs


def _second_order_subs_type2(A, funcs, t_):
    r"""
    Returns a second order system based on the coefficient matrix passed.

    Explanation
    ===========

    This function returns a system of second order ODE of the following form:

    .. math::
        X'' = A * X

    Here, $X$ is the vector of dependent variables, but a bit modified, $A$ is the
    coefficient matrix passed.

    Along with returning the second order system, this function also returns the new
    dependent variables with the new independent variable `t_` passed.

    Parameters
    ==========

    A: Matrix
        Coefficient matrix of the system
    funcs: List
        List of old dependent variables
    t_: Symbol
        New independent variable

    Returns
    =======

    List, List

    """
    func_names = [func.func.__name__ for func in funcs]
    new_funcs = [Function(Dummy("{}_".format(name)))(t_) for name in func_names]
    rhss = A * Matrix(new_funcs)
    new_eqs = [Eq(func.diff(t_, 2), rhs) for func, rhs in zip(new_funcs, rhss)]

    return new_eqs, new_funcs


def _is_euler_system(As, t):
    return all(_matrix_is_constant((A*t**i).applyfunc(ratsimp), t) for i, A in enumerate(As))


def _classify_linear_system(eqs, funcs, t, is_canon=False):
    r"""
    Returns a dictionary with details of the eqs if the system passed is linear
    and can be classified by this function else returns None

    Explanation
    ===========

    This function takes the eqs, converts it into a form Ax = b where x is a vector of terms
    containing dependent variables and their derivatives till their maximum order. If it is
    possible to convert eqs into Ax = b, then all the equations in eqs are linear otherwise
    they are non-linear.

    To check if the equations are constant coefficient, we need to check if all the terms in
    A obtained above are constant or not.

    To check if the equations are homogeneous or not, we need to check if b is a zero matrix
    or not.

    Parameters
    ==========

    eqs: List
        List of ODEs
    funcs: List
        List of dependent variables
    t: Symbol
        Independent variable of the equations in eqs
    is_canon: Boolean
        If True, then this function will not try to get the
        system in canonical form. Default value is False

    Returns
    =======

    match = {
        'no_of_equation': len(eqs),
        'eq': eqs,
        'func': funcs,
        'order': order,
        'is_linear': is_linear,
        'is_constant': is_constant,
        'is_homogeneous': is_homogeneous,
    }

    Dict or list of Dicts or None
        Dict with values for keys:
            1. no_of_equation: Number of equations
            2. eq: The set of equations
            3. func: List of dependent variables
            4. order: A dictionary that gives the order of the
                      dependent variable in eqs
            5. is_linear: Boolean value indicating if the set of
                          equations are linear or not.
            6. is_constant: Boolean value indicating if the set of
                          equations have constant coefficients or not.
            7. is_homogeneous: Boolean value indicating if the set of
                          equations are homogeneous or not.
            8. commutative_antiderivative: Antiderivative of the coefficient
                          matrix if the coefficient matrix is non-constant
                          and commutative with its antiderivative. This key
                          may or may not exist.
            9. is_general: Boolean value indicating if the system of ODEs is
                           solvable using one of the general case solvers or not.
            10. rhs: rhs of the non-homogeneous system of ODEs in Matrix form. This
                     key may or may not exist.
            11. is_higher_order: True if the system passed has an order greater than 1.
                                 This key may or may not exist.
            12. is_second_order: True if the system passed is a second order ODE. This
                                 key may or may not exist.
        This Dict is the answer returned if the eqs are linear and constant
        coefficient. Otherwise, None is returned.

    """

    # Error for i == 0 can be added but isn't for now

    # Check for len(funcs) == len(eqs)
    if len(funcs) != len(eqs):
        raise ValueError("Number of functions given is not equal to the number of equations %s" % funcs)

    # ValueError when functions have more than one arguments
    for func in funcs:
        if len(func.args) != 1:
            raise ValueError("dsolve() and classify_sysode() work with "
            "functions of one variable only, not %s" % func)

    # Getting the func_dict and order using the helper
    # function
    order = _get_func_order(eqs, funcs)
    system_order = max(order[func] for func in funcs)
    is_higher_order = system_order > 1
    is_second_order = system_order == 2 and all(order[func] == 2 for func in funcs)

    # Not adding the check if the len(func.args) for
    # every func in funcs is 1

    # Linearity check
    try:

        canon_eqs = canonical_odes(eqs, funcs, t) if not is_canon else [eqs]
        if len(canon_eqs) == 1:
            As, b = linear_ode_to_matrix(canon_eqs[0], funcs, t, system_order)
        else:

            match = {
                'is_implicit': True,
                'canon_eqs': canon_eqs
            }

            return match

    # When the system of ODEs is non-linear, an ODENonlinearError is raised.
    # This function catches the error and None is returned.
    except ODENonlinearError:
        return None

    is_linear = True

    # Homogeneous check
    is_homogeneous = True if b.is_zero_matrix else False

    # Is general key is used to identify if the system of ODEs can be solved by
    # one of the general case solvers or not.
    match = {
        'no_of_equation': len(eqs),
        'eq': eqs,
        'func': funcs,
        'order': order,
        'is_linear': is_linear,
        'is_homogeneous': is_homogeneous,
        'is_general': True
    }

    if not is_homogeneous:
        match['rhs'] = b

    is_constant = all(_matrix_is_constant(A_, t) for A_ in As)

    # The match['is_linear'] check will be added in the future when this
    # function becomes ready to deal with non-linear systems of ODEs

    if not is_higher_order:
        A = As[1]
        match['func_coeff'] = A

        # Constant coefficient check
        is_constant = _matrix_is_constant(A, t)
        match['is_constant'] = is_constant

        try:
            system_info = linodesolve_type(A, t, b=b)
        except NotImplementedError:
            return None

        match.update(system_info)
        antiderivative = match.pop("antiderivative")

        if not is_constant:
            match['commutative_antiderivative'] = antiderivative

        return match
    else:
        match['type_of_equation'] = "type0"

        if is_second_order:
            A1, A0 = As[1:]

            match_second_order = _match_second_order_type(A1, A0, t)
            match.update(match_second_order)

            match['is_second_order'] = True

        # If system is constant, then no need to check if its in euler
        # form or not. It will be easier and faster to directly proceed
        # to solve it.
        if match['type_of_equation'] == "type0" and not is_constant:
            is_euler = _is_euler_system(As, t)
            if is_euler:
                t_ = Symbol('{}_'.format(t))
                match.update({'is_transformed': True, 'type_of_equation': 'type1',
                              't_': t_})
            else:
                is_jordan = lambda M: M == Matrix.jordan_block(M.shape[0], M[0, 0])
                terms = _factor_matrix(As[-1], t)
                if all(A.is_zero_matrix for A in As[1:-1]) and terms is not None and not is_jordan(terms[1]):
                    P, J = terms[1].jordan_form()
                    match.update({'type_of_equation': 'type2', 'J': J,
                                  'f(t)': terms[0], 'P': P, 'is_transformed': True})

            if match['type_of_equation'] != 'type0' and is_second_order:
                match.pop('is_second_order', None)

        match['is_higher_order'] = is_higher_order

        return match

def _preprocess_eqs(eqs):
    processed_eqs = []
    for eq in eqs:
        processed_eqs.append(eq if isinstance(eq, Equality) else Eq(eq, 0))

    return processed_eqs


def _eqs2dict(eqs, funcs):
    eqsorig = {}
    eqsmap = {}
    funcset = set(funcs)
    for eq in eqs:
        f1, = eq.lhs.atoms(AppliedUndef)
        f2s = (eq.rhs.atoms(AppliedUndef) - {f1}) & funcset
        eqsmap[f1] = f2s
        eqsorig[f1] = eq
    return eqsmap, eqsorig


def _dict2graph(d):
    nodes = list(d)
    edges = [(f1, f2) for f1, f2s in d.items() for f2 in f2s]
    G = (nodes, edges)
    return G


def _is_type1(scc, t):
    eqs, funcs = scc

    try:
        (A1, A0), b = linear_ode_to_matrix(eqs, funcs, t, 1)
    except (ODENonlinearError, ODEOrderError):
        return False

    if _matrix_is_constant(A0, t) and b.is_zero_matrix:
        return True

    return False


def _combine_type1_subsystems(subsystem, funcs, t):
    indices = [i for i, sys in enumerate(zip(subsystem, funcs)) if _is_type1(sys, t)]
    remove = set()
    for ip, i in enumerate(indices):
        for j in indices[ip+1:]:
            if any(eq2.has(funcs[i]) for eq2 in subsystem[j]):
                subsystem[j] = subsystem[i] + subsystem[j]
                remove.add(i)
    subsystem = [sys for i, sys in enumerate(subsystem) if i not in remove]
    return subsystem


def _component_division(eqs, funcs, t):

    # Assuming that each eq in eqs is in canonical form,
    # that is, [f(x).diff(x) = .., g(x).diff(x) = .., etc]
    # and that the system passed is in its first order
    eqsmap, eqsorig = _eqs2dict(eqs, funcs)

    subsystems = []
    for cc in connected_components(_dict2graph(eqsmap)):
        eqsmap_c = {f: eqsmap[f] for f in cc}
        sccs = strongly_connected_components(_dict2graph(eqsmap_c))
        subsystem = [[eqsorig[f] for f in scc] for scc in sccs]
        subsystem = _combine_type1_subsystems(subsystem, sccs, t)
        subsystems.append(subsystem)

    return subsystems


# Returns: List of equations
def _linear_ode_solver(match):
    t = match['t']
    funcs = match['func']

    rhs = match.get('rhs', None)
    tau = match.get('tau', None)
    t = match['t_'] if 't_' in match else t
    A = match['func_coeff']

    # Note: To make B None when the matrix has constant
    # coefficient
    B = match.get('commutative_antiderivative', None)
    type = match['type_of_equation']

    sol_vector = linodesolve(A, t, b=rhs, B=B,
                             type=type, tau=tau)

    sol = [Eq(f, s) for f, s in zip(funcs, sol_vector)]

    return sol


def _select_equations(eqs, funcs, key=lambda x: x):
    eq_dict = {e.lhs: e.rhs for e in eqs}
    return [Eq(f, eq_dict[key(f)]) for f in funcs]


def _higher_order_ode_solver(match):
    eqs = match["eq"]
    funcs = match["func"]
    t = match["t"]
    sysorder = match['order']
    type = match.get('type_of_equation', "type0")

    is_second_order = match.get('is_second_order', False)
    is_transformed = match.get('is_transformed', False)
    is_euler = is_transformed and type == "type1"
    is_higher_order_type2 = is_transformed and type == "type2" and 'P' in match

    if is_second_order:
        new_eqs, new_funcs = _second_order_to_first_order(eqs, funcs, t,
                                                          A1=match.get("A1", None), A0=match.get("A0", None),
                                                          b=match.get("rhs", None), type=type,
                                                          t_=match.get("t_", None))
    else:
        new_eqs, new_funcs = _higher_order_to_first_order(eqs, sysorder, t, funcs=funcs,
                                                          type=type, J=match.get('J', None),
                                                          f_t=match.get('f(t)', None),
                                                          P=match.get('P', None), b=match.get('rhs', None))

    if is_transformed:
        t = match.get('t_', t)

    if not is_higher_order_type2:
        new_eqs = _select_equations(new_eqs, [f.diff(t) for f in new_funcs])

    sol = None

    # NotImplementedError may be raised when the system may be actually
    # solvable if it can be just divided into sub-systems
    try:
        if not is_higher_order_type2:
            sol = _strong_component_solver(new_eqs, new_funcs, t)
    except NotImplementedError:
        sol = None

    # Dividing the system only when it becomes essential
    if sol is None:
        try:
            sol = _component_solver(new_eqs, new_funcs, t)
        except NotImplementedError:
            sol = None

    if sol is None:
        return sol

    is_second_order_type2 = is_second_order and type == "type2"

    underscores = '__' if is_transformed else '_'

    sol = _select_equations(sol, funcs,
                            key=lambda x: Function(Dummy('{}{}0'.format(x.func.__name__, underscores)))(t))

    if match.get("is_transformed", False):
        if is_second_order_type2:
            g_t = match["g(t)"]
            tau = match["tau"]
            sol = [Eq(s.lhs, s.rhs.subs(t, tau) * g_t) for s in sol]
        elif is_euler:
            t = match['t']
            tau = match['t_']
            sol = [s.subs(tau, log(t)) for s in sol]
        elif is_higher_order_type2:
            P = match['P']
            sol_vector = P * Matrix([s.rhs for s in sol])
            sol = [Eq(f, s) for f, s in zip(funcs, sol_vector)]

    return sol


# Returns: List of equations or None
# If None is returned by this solver, then the system
# of ODEs cannot be solved directly by dsolve_system.
def _strong_component_solver(eqs, funcs, t):
    from sympy.solvers.ode.ode import dsolve, constant_renumber

    match = _classify_linear_system(eqs, funcs, t, is_canon=True)
    sol = None

    # Assuming that we can't get an implicit system
    # since we are already canonical equations from
    # dsolve_system
    if match:
        match['t'] = t

        if match.get('is_higher_order', False):
            sol = _higher_order_ode_solver(match)

        elif match.get('is_linear', False):
            sol = _linear_ode_solver(match)

        # Note: For now, only linear systems are handled by this function
        # hence, the match condition is added. This can be removed later.
        if sol is None and len(eqs) == 1:
            sol = dsolve(eqs[0], func=funcs[0])
            variables = Tuple(eqs[0]).free_symbols
            new_constants = [Dummy() for _ in range(ode_order(eqs[0], funcs[0]))]
            sol = constant_renumber(sol, variables=variables, newconstants=new_constants)
            sol = [sol]

        # To add non-linear case here in future

    return sol


def _get_funcs_from_canon(eqs):
    return [eq.lhs.args[0] for eq in eqs]


# Returns: List of Equations(a solution)
def _weak_component_solver(wcc, t):

    # We will divide the systems into sccs
    # only when the wcc cannot be solved as
    # a whole
    eqs = []
    for scc in wcc:
        eqs += scc
    funcs = _get_funcs_from_canon(eqs)

    sol = _strong_component_solver(eqs, funcs, t)
    if sol:
        return sol

    sol = []

    for scc in wcc:
        eqs = scc
        funcs = _get_funcs_from_canon(eqs)

        # Substituting solutions for the dependent
        # variables solved in previous SCC, if any solved.
        comp_eqs = [eq.subs({s.lhs: s.rhs for s in sol}) for eq in eqs]
        scc_sol = _strong_component_solver(comp_eqs, funcs, t)

        if scc_sol is None:
            raise NotImplementedError(filldedent('''
                The system of ODEs passed cannot be solved by dsolve_system.
            '''))

        # scc_sol: List of equations
        # scc_sol is a solution
        sol += scc_sol

    return sol


# Returns: List of Equations(a solution)
def _component_solver(eqs, funcs, t):
    components = _component_division(eqs, funcs, t)
    sol = []

    for wcc in components:

        # wcc_sol: List of Equations
        sol += _weak_component_solver(wcc, t)

    # sol: List of Equations
    return sol


def _second_order_to_first_order(eqs, funcs, t, type="auto", A1=None,
                                 A0=None, b=None, t_=None):
    r"""
    Expects the system to be in second order and in canonical form

    Explanation
    ===========

    Reduces a second order system into a first order one depending on the type of second
    order system.
    1. "type0": If this is passed, then the system will be reduced to first order by
                introducing dummy variables.
    2. "type1": If this is passed, then a particular substitution will be used to reduce the
                the system into first order.
    3. "type2": If this is passed, then the system will be transformed with new dependent
                variables and independent variables. This transformation is a part of solving
                the corresponding system of ODEs.

    `A1` and `A0` are the coefficient matrices from the system and it is assumed that the
    second order system has the form given below:

    .. math::
        A2 * X'' = A1 * X' + A0 * X + b

    Here, $A2$ is the coefficient matrix for the vector $X''$ and $b$ is the non-homogeneous
    term.

    Default value for `b` is None but if `A1` and `A0` are passed and `b` is not passed, then the
    system will be assumed homogeneous.

    """
    is_a1 = A1 is None
    is_a0 = A0 is None

    if (type == "type1" and is_a1) or (type == "type2" and is_a0)\
        or (type == "auto" and (is_a1 or is_a0)):
        (A2, A1, A0), b = linear_ode_to_matrix(eqs, funcs, t, 2)

        if not A2.is_Identity:
            raise ValueError(filldedent('''
                The system must be in its canonical form.
            '''))

    if type == "auto":
        match = _match_second_order_type(A1, A0, t)
        type = match["type_of_equation"]
        A1 = match.get("A1", None)
        A0 = match.get("A0", None)

    sys_order = dict.fromkeys(funcs, 2)

    if type == "type1":
        if b is None:
            b = zeros(len(eqs))
        eqs = _second_order_subs_type1(A1, b, funcs, t)
        sys_order = dict.fromkeys(funcs, 1)

    if type == "type2":
        if t_ is None:
            t_ = Symbol("{}_".format(t))
        t = t_
        eqs, funcs = _second_order_subs_type2(A0, funcs, t_)
        sys_order = dict.fromkeys(funcs, 2)

    return _higher_order_to_first_order(eqs, sys_order, t, funcs=funcs)


def _higher_order_type2_to_sub_systems(J, f_t, funcs, t, max_order, b=None, P=None):

    # Note: To add a test for this ValueError
    if J is None or f_t is None or not _matrix_is_constant(J, t):
        raise ValueError(filldedent('''
            Correctly input for args 'A' and 'f_t' for Linear, Higher Order,
            Type 2
        '''))

    if P is None and b is not None and not b.is_zero_matrix:
        raise ValueError(filldedent('''
            Provide the keyword 'P' for matrix P in A = P * J * P-1.
        '''))

    new_funcs = Matrix([Function(Dummy('{}__0'.format(f.func.__name__)))(t) for f in funcs])
    new_eqs = new_funcs.diff(t, max_order) - f_t * J * new_funcs

    if b is not None and not b.is_zero_matrix:
        new_eqs -= P.inv() * b

    new_eqs = canonical_odes(new_eqs, new_funcs, t)[0]

    return new_eqs, new_funcs


def _higher_order_to_first_order(eqs, sys_order, t, funcs=None, type="type0", **kwargs):
    if funcs is None:
        funcs = sys_order.keys()

    # Standard Cauchy Euler system
    if type == "type1":
        t_ = Symbol('{}_'.format(t))
        new_funcs = [Function(Dummy('{}_'.format(f.func.__name__)))(t_) for f in funcs]
        max_order = max(sys_order[func] for func in funcs)
        subs_dict = dict(zip(funcs, new_funcs))
        subs_dict[t] = exp(t_)

        free_function = Function(Dummy())

        def _get_coeffs_from_subs_expression(expr):
            if isinstance(expr, Subs):
                free_symbol = expr.args[1][0]
                term = expr.args[0]
                return {ode_order(term, free_symbol): 1}

            if isinstance(expr, Mul):
                coeff = expr.args[0]
                order = list(_get_coeffs_from_subs_expression(expr.args[1]).keys())[0]
                return {order: coeff}

            if isinstance(expr, Add):
                coeffs = {}
                for arg in expr.args:

                    if isinstance(arg, Mul):
                        coeffs.update(_get_coeffs_from_subs_expression(arg))

                    else:
                        order = list(_get_coeffs_from_subs_expression(arg).keys())[0]
                        coeffs[order] = 1

                return coeffs

        for o in range(1, max_order + 1):
            expr = free_function(log(t_)).diff(t_, o)*t_**o
            coeff_dict = _get_coeffs_from_subs_expression(expr)
            coeffs = [coeff_dict[order] if order in coeff_dict else 0 for order in range(o + 1)]
            expr_to_subs = sum(free_function(t_).diff(t_, i) * c for i, c in
                        enumerate(coeffs)) / t**o
            subs_dict.update({f.diff(t, o): expr_to_subs.subs(free_function(t_), nf)
                              for f, nf in zip(funcs, new_funcs)})

        new_eqs = [eq.subs(subs_dict) for eq in eqs]
        new_sys_order = {nf: sys_order[f] for f, nf in zip(funcs, new_funcs)}

        new_eqs = canonical_odes(new_eqs, new_funcs, t_)[0]

        return _higher_order_to_first_order(new_eqs, new_sys_order, t_, funcs=new_funcs)

    # Systems of the form: X(n)(t) = f(t)*A*X + b
    # where X(n)(t) is the nth derivative of the vector of dependent variables
    # with respect to the independent variable and A is a constant matrix.
    if type == "type2":
        J = kwargs.get('J', None)
        f_t = kwargs.get('f_t', None)
        b = kwargs.get('b', None)
        P = kwargs.get('P', None)
        max_order = max(sys_order[func] for func in funcs)

        return _higher_order_type2_to_sub_systems(J, f_t, funcs, t, max_order, P=P, b=b)

        # Note: To be changed to this after doit option is disabled for default cases
        # new_sysorder = _get_func_order(new_eqs, new_funcs)
        #
        # return _higher_order_to_first_order(new_eqs, new_sysorder, t, funcs=new_funcs)

    new_funcs = []

    for prev_func in funcs:
        func_name = prev_func.func.__name__
        func = Function(Dummy('{}_0'.format(func_name)))(t)
        new_funcs.append(func)
        subs_dict = {prev_func: func}
        new_eqs = []

        for i in range(1, sys_order[prev_func]):
            new_func = Function(Dummy('{}_{}'.format(func_name, i)))(t)
            subs_dict[prev_func.diff(t, i)] = new_func
            new_funcs.append(new_func)

            prev_f = subs_dict[prev_func.diff(t, i-1)]
            new_eq = Eq(prev_f.diff(t), new_func)
            new_eqs.append(new_eq)

        eqs = [eq.subs(subs_dict) for eq in eqs] + new_eqs

    return eqs, new_funcs


def dsolve_system(eqs, funcs=None, t=None, ics=None, doit=False, simplify=True):
    r"""
    Solves any(supported) system of Ordinary Differential Equations

    Explanation
    ===========

    This function takes a system of ODEs as an input, determines if the
    it is solvable by this function, and returns the solution if found any.

    This function can handle:
    1. Linear, First Order, Constant coefficient homogeneous system of ODEs
    2. Linear, First Order, Constant coefficient non-homogeneous system of ODEs
    3. Linear, First Order, non-constant coefficient homogeneous system of ODEs
    4. Linear, First Order, non-constant coefficient non-homogeneous system of ODEs
    5. Any implicit system which can be divided into system of ODEs which is of the above 4 forms
    6. Any higher order linear system of ODEs that can be reduced to one of the 5 forms of systems described above.

    The types of systems described above are not limited by the number of equations, i.e. this
    function can solve the above types irrespective of the number of equations in the system passed.
    But, the bigger the system, the more time it will take to solve the system.

    This function returns a list of solutions. Each solution is a list of equations where LHS is
    the dependent variable and RHS is an expression in terms of the independent variable.

    Among the non constant coefficient types, not all the systems are solvable by this function. Only
    those which have either a coefficient matrix with a commutative antiderivative or those systems which
    may be divided further so that the divided systems may have coefficient matrix with commutative antiderivative.

    Parameters
    ==========

    eqs : List
        system of ODEs to be solved
    funcs : List or None
        List of dependent variables that make up the system of ODEs
    t : Symbol or None
        Independent variable in the system of ODEs
    ics : Dict or None
        Set of initial boundary/conditions for the system of ODEs
    doit : Boolean
        Evaluate the solutions if True. Default value is True. Can be
        set to false if the integral evaluation takes too much time and/or
        is not required.
    simplify: Boolean
        Simplify the solutions for the systems. Default value is True.
        Can be set to false if simplification takes too much time and/or
        is not required.

    Examples
    ========

    >>> from sympy import symbols, Eq, Function
    >>> from sympy.solvers.ode.systems import dsolve_system
    >>> f, g = symbols("f g", cls=Function)
    >>> x = symbols("x")

    >>> eqs = [Eq(f(x).diff(x), g(x)), Eq(g(x).diff(x), f(x))]
    >>> dsolve_system(eqs)
    [[Eq(f(x), -C1*exp(-x) + C2*exp(x)), Eq(g(x), C1*exp(-x) + C2*exp(x))]]

    You can also pass the initial conditions for the system of ODEs:

    >>> dsolve_system(eqs, ics={f(0): 1, g(0): 0})
    [[Eq(f(x), exp(x)/2 + exp(-x)/2), Eq(g(x), exp(x)/2 - exp(-x)/2)]]

    Optionally, you can pass the dependent variables and the independent
    variable for which the system is to be solved:

    >>> funcs = [f(x), g(x)]
    >>> dsolve_system(eqs, funcs=funcs, t=x)
    [[Eq(f(x), -C1*exp(-x) + C2*exp(x)), Eq(g(x), C1*exp(-x) + C2*exp(x))]]

    Lets look at an implicit system of ODEs:

    >>> eqs = [Eq(f(x).diff(x)**2, g(x)**2), Eq(g(x).diff(x), g(x))]
    >>> dsolve_system(eqs)
    [[Eq(f(x), C1 - C2*exp(x)), Eq(g(x), C2*exp(x))], [Eq(f(x), C1 + C2*exp(x)), Eq(g(x), C2*exp(x))]]

    Returns
    =======

    List of List of Equations

    Raises
    ======

    NotImplementedError
        When the system of ODEs is not solvable by this function.
    ValueError
        When the parameters passed are not in the required form.

    """
    from sympy.solvers.ode.ode import solve_ics, _extract_funcs, constant_renumber

    if not iterable(eqs):
        raise ValueError(filldedent('''
            List of equations should be passed. The input is not valid.
        '''))

    eqs = _preprocess_eqs(eqs)

    if funcs is not None and not isinstance(funcs, list):
        raise ValueError(filldedent('''
            Input to the funcs should be a list of functions.
        '''))

    if funcs is None:
        funcs = _extract_funcs(eqs)

    if any(len(func.args) != 1 for func in funcs):
        raise ValueError(filldedent('''
            dsolve_system can solve a system of ODEs with only one independent
            variable.
        '''))

    if len(eqs) != len(funcs):
        raise ValueError(filldedent('''
            Number of equations and number of functions do not match
        '''))

    if t is not None and not isinstance(t, Symbol):
        raise ValueError(filldedent('''
            The independent variable must be of type Symbol
        '''))

    if t is None:
        t = list(list(eqs[0].atoms(Derivative))[0].atoms(Symbol))[0]

    sols = []
    canon_eqs = canonical_odes(eqs, funcs, t)

    for canon_eq in canon_eqs:
        try:
            sol = _strong_component_solver(canon_eq, funcs, t)
        except NotImplementedError:
            sol = None

        if sol is None:
            sol = _component_solver(canon_eq, funcs, t)

        sols.append(sol)

    if sols:
        final_sols = []
        variables = Tuple(*eqs).free_symbols

        for sol in sols:

            sol = _select_equations(sol, funcs)
            sol = constant_renumber(sol, variables=variables)

            if ics:
                constants = Tuple(*sol).free_symbols - variables
                solved_constants = solve_ics(sol, funcs, constants, ics)
                sol = [s.subs(solved_constants) for s in sol]

            if simplify:
                constants = Tuple(*sol).free_symbols - variables
                sol = simpsol(sol, [t], constants, doit=doit)

            final_sols.append(sol)

        sols = final_sols

    return sols
