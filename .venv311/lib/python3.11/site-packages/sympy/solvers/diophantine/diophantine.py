from __future__ import annotations

from sympy.core.add import Add
from sympy.core.assumptions import check_assumptions
from sympy.core.containers import Tuple
from sympy.core.exprtools import factor_terms
from sympy.core.function import _mexpand
from sympy.core.mul import Mul
from sympy.core.numbers import Rational, int_valued
from sympy.core.intfunc import igcdex, ilcm, igcd, integer_nthroot, isqrt
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Symbol, symbols
from sympy.core.sympify import _sympify
from sympy.external.gmpy import jacobi, remove, invert, iroot
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.ntheory.factor_ import divisors, factorint, perfect_power
from sympy.ntheory.generate import nextprime
from sympy.ntheory.primetest import is_square, isprime
from sympy.ntheory.modular import symmetric_residue
from sympy.ntheory.residue_ntheory import sqrt_mod, sqrt_mod_iter
from sympy.polys.polyerrors import GeneratorsNeeded
from sympy.polys.polytools import Poly, factor_list
from sympy.simplify.simplify import signsimp
from sympy.solvers.solveset import solveset_real
from sympy.utilities import numbered_symbols
from sympy.utilities.misc import as_int, filldedent
from sympy.utilities.iterables import (is_sequence, subsets, permute_signs,
                                       signed_permutations, ordered_partitions)


# these are imported with 'from sympy.solvers.diophantine import *
__all__ = ['diophantine', 'classify_diop']


class DiophantineSolutionSet(set):
    """
    Container for a set of solutions to a particular diophantine equation.

    The base representation is a set of tuples representing each of the solutions.

    Parameters
    ==========

    symbols : list
        List of free symbols in the original equation.
    parameters: list
        List of parameters to be used in the solution.

    Examples
    ========

    Adding solutions:

        >>> from sympy.solvers.diophantine.diophantine import DiophantineSolutionSet
        >>> from sympy.abc import x, y, t, u
        >>> s1 = DiophantineSolutionSet([x, y], [t, u])
        >>> s1
        set()
        >>> s1.add((2, 3))
        >>> s1.add((-1, u))
        >>> s1
        {(-1, u), (2, 3)}
        >>> s2 = DiophantineSolutionSet([x, y], [t, u])
        >>> s2.add((3, 4))
        >>> s1.update(*s2)
        >>> s1
        {(-1, u), (2, 3), (3, 4)}

    Conversion of solutions into dicts:

        >>> list(s1.dict_iterator())
        [{x: -1, y: u}, {x: 2, y: 3}, {x: 3, y: 4}]

    Substituting values:

        >>> s3 = DiophantineSolutionSet([x, y], [t, u])
        >>> s3.add((t**2, t + u))
        >>> s3
        {(t**2, t + u)}
        >>> s3.subs({t: 2, u: 3})
        {(4, 5)}
        >>> s3.subs(t, -1)
        {(1, u - 1)}
        >>> s3.subs(t, 3)
        {(9, u + 3)}

    Evaluation at specific values. Positional arguments are given in the same order as the parameters:

        >>> s3(-2, 3)
        {(4, 1)}
        >>> s3(5)
        {(25, u + 5)}
        >>> s3(None, 2)
        {(t**2, t + 2)}
    """

    def __init__(self, symbols_seq, parameters):
        super().__init__()

        if not is_sequence(symbols_seq):
            raise ValueError("Symbols must be given as a sequence.")

        if not is_sequence(parameters):
            raise ValueError("Parameters must be given as a sequence.")

        self.symbols = tuple(symbols_seq)
        self.parameters = tuple(parameters)

    def add(self, solution):
        if len(solution) != len(self.symbols):
            raise ValueError("Solution should have a length of %s, not %s" % (len(self.symbols), len(solution)))
        # make solution canonical wrt sign (i.e. no -x unless x is also present as an arg)
        args = set(solution)
        for i in range(len(solution)):
            x = solution[i]
            if not type(x) is int and (-x).is_Symbol and -x not in args:
                solution = [_.subs(-x, x) for _ in solution]
        super().add(Tuple(*solution))

    def update(self, *solutions):
        for solution in solutions:
            self.add(solution)

    def dict_iterator(self):
        for solution in ordered(self):
            yield dict(zip(self.symbols, solution))

    def subs(self, *args, **kwargs):
        result = DiophantineSolutionSet(self.symbols, self.parameters)
        for solution in self:
            result.add(solution.subs(*args, **kwargs))
        return result

    def __call__(self, *args):
        if len(args) > len(self.parameters):
            raise ValueError("Evaluation should have at most %s values, not %s" % (len(self.parameters), len(args)))
        rep = {p: v for p, v in zip(self.parameters, args) if v is not None}
        return self.subs(rep)


class DiophantineEquationType:
    """
    Internal representation of a particular diophantine equation type.

    Parameters
    ==========

    equation :
        The diophantine equation that is being solved.
    free_symbols : list (optional)
        The symbols being solved for.

    Attributes
    ==========

    total_degree :
        The maximum of the degrees of all terms in the equation
    homogeneous :
        Does the equation contain a term of degree 0
    homogeneous_order :
        Does the equation contain any coefficient that is in the symbols being solved for
    dimension :
        The number of symbols being solved for
    """
    name: str

    def __init__(self, equation, free_symbols=None):
        self.equation = _sympify(equation).expand(force=True)

        if free_symbols is not None:
            self.free_symbols = free_symbols
        else:
            self.free_symbols = list(self.equation.free_symbols)
            self.free_symbols.sort(key=default_sort_key)

        if not self.free_symbols:
            raise ValueError('equation should have 1 or more free symbols')

        self.coeff = self.equation.as_coefficients_dict()
        if not all(int_valued(c) for c in self.coeff.values()):
            raise TypeError("Coefficients should be Integers")

        self.total_degree = Poly(self.equation).total_degree()
        self.homogeneous = 1 not in self.coeff
        self.homogeneous_order = not (set(self.coeff) & set(self.free_symbols))
        self.dimension = len(self.free_symbols)
        self._parameters = None

    def matches(self):
        """
        Determine whether the given equation can be matched to the particular equation type.
        """
        return False

    @property
    def n_parameters(self):
        return self.dimension

    @property
    def parameters(self):
        if self._parameters is None:
            self._parameters = symbols('t_:%i' % (self.n_parameters,), integer=True)
        return self._parameters

    def solve(self, parameters=None, limit=None) -> DiophantineSolutionSet:
        raise NotImplementedError('No solver has been written for %s.' % self.name)

    def pre_solve(self, parameters=None):
        if not self.matches():
            raise ValueError("This equation does not match the %s equation type." % self.name)

        if parameters is not None:
            if len(parameters) != self.n_parameters:
                raise ValueError("Expected %s parameter(s) but got %s" % (self.n_parameters, len(parameters)))

        self._parameters = parameters


class Univariate(DiophantineEquationType):
    """
    Representation of a univariate diophantine equation.

    A univariate diophantine equation is an equation of the form
    `a_{0} + a_{1}x + a_{2}x^2 + .. + a_{n}x^n = 0` where `a_{1}, a_{2}, ..a_{n}` are
    integer constants and `x` is an integer variable.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import Univariate
    >>> from sympy.abc import x
    >>> Univariate((x - 2)*(x - 3)**2).solve() # solves equation (x - 2)*(x - 3)**2 == 0
    {(2,), (3,)}

    """

    name = 'univariate'

    def matches(self):
        return self.dimension == 1

    def solve(self, parameters=None, limit=None):
        self.pre_solve(parameters)

        result = DiophantineSolutionSet(self.free_symbols, parameters=self.parameters)
        for i in solveset_real(self.equation, self.free_symbols[0]).intersect(S.Integers):
            result.add((i,))
        return result


class Linear(DiophantineEquationType):
    """
    Representation of a linear diophantine equation.

    A linear diophantine equation is an equation of the form `a_{1}x_{1} +
    a_{2}x_{2} + .. + a_{n}x_{n} = 0` where `a_{1}, a_{2}, ..a_{n}` are
    integer constants and `x_{1}, x_{2}, ..x_{n}` are integer variables.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import Linear
    >>> from sympy.abc import x, y, z
    >>> l1 = Linear(2*x - 3*y - 5)
    >>> l1.matches() # is this equation linear
    True
    >>> l1.solve() # solves equation 2*x - 3*y - 5 == 0
    {(3*t_0 - 5, 2*t_0 - 5)}

    Here x = -3*t_0 - 5 and y = -2*t_0 - 5

    >>> Linear(2*x - 3*y - 4*z -3).solve()
    {(t_0, 2*t_0 + 4*t_1 + 3, -t_0 - 3*t_1 - 3)}

    """

    name = 'linear'

    def matches(self):
        return self.total_degree == 1

    def solve(self, parameters=None, limit=None):
        self.pre_solve(parameters)

        coeff = self.coeff
        var = self.free_symbols

        if 1 in coeff:
            # negate coeff[] because input is of the form: ax + by + c ==  0
            #                              but is used as: ax + by     == -c
            c = -coeff[1]
        else:
            c = 0

        result = DiophantineSolutionSet(var, parameters=self.parameters)
        params = result.parameters

        if len(var) == 1:
            q, r = divmod(c, coeff[var[0]])
            if not r:
                result.add((q,))
            return result

        '''
        base_solution_linear() can solve diophantine equations of the form:

        a*x + b*y == c

        We break down multivariate linear diophantine equations into a
        series of bivariate linear diophantine equations which can then
        be solved individually by base_solution_linear().

        Consider the following:

        a_0*x_0 + a_1*x_1 + a_2*x_2 == c

        which can be re-written as:

        a_0*x_0 + g_0*y_0 == c

        where

        g_0 == gcd(a_1, a_2)

        and

        y == (a_1*x_1)/g_0 + (a_2*x_2)/g_0

        This leaves us with two binary linear diophantine equations.
        For the first equation:

        a == a_0
        b == g_0
        c == c

        For the second:

        a == a_1/g_0
        b == a_2/g_0
        c == the solution we find for y_0 in the first equation.

        The arrays A and B are the arrays of integers used for
        'a' and 'b' in each of the n-1 bivariate equations we solve.
        '''

        A = [coeff[v] for v in var]
        B = []
        if len(var) > 2:
            B.append(igcd(A[-2], A[-1]))
            A[-2] = A[-2] // B[0]
            A[-1] = A[-1] // B[0]
            for i in range(len(A) - 3, 0, -1):
                gcd = igcd(B[0], A[i])
                B[0] = B[0] // gcd
                A[i] = A[i] // gcd
                B.insert(0, gcd)
        B.append(A[-1])

        '''
        Consider the trivariate linear equation:

        4*x_0 + 6*x_1 + 3*x_2 == 2

        This can be re-written as:

        4*x_0 + 3*y_0 == 2

        where

        y_0 == 2*x_1 + x_2
        (Note that gcd(3, 6) == 3)

        The complete integral solution to this equation is:

        x_0 ==  2 + 3*t_0
        y_0 == -2 - 4*t_0

        where 't_0' is any integer.

        Now that we have a solution for 'x_0', find 'x_1' and 'x_2':

        2*x_1 + x_2 == -2 - 4*t_0

        We can then solve for '-2' and '-4' independently,
        and combine the results:

        2*x_1a + x_2a == -2
        x_1a == 0 + t_0
        x_2a == -2 - 2*t_0

        2*x_1b + x_2b == -4*t_0
        x_1b == 0*t_0 + t_1
        x_2b == -4*t_0 - 2*t_1

        ==>

        x_1 == t_0 + t_1
        x_2 == -2 - 6*t_0 - 2*t_1

        where 't_0' and 't_1' are any integers.

        Note that:

        4*(2 + 3*t_0) + 6*(t_0 + t_1) + 3*(-2 - 6*t_0 - 2*t_1) == 2

        for any integral values of 't_0', 't_1'; as required.

        This method is generalised for many variables, below.

        '''
        solutions = []
        for Ai, Bi in zip(A, B):
            tot_x, tot_y = [], []

            for arg in Add.make_args(c):
                if arg.is_Integer:
                    # example: 5 -> k = 5
                    k, p = arg, S.One
                    pnew = params[0]
                else:  # arg is a Mul or Symbol
                    # example: 3*t_1 -> k = 3
                    # example: t_0 -> k = 1
                    k, p = arg.as_coeff_Mul()
                    pnew = params[params.index(p) + 1]

                sol = sol_x, sol_y = base_solution_linear(k, Ai, Bi, pnew)

                if p is S.One:
                    if None in sol:
                        return result
                else:
                    # convert a + b*pnew -> a*p + b*pnew
                    if isinstance(sol_x, Add):
                        sol_x = sol_x.args[0]*p + sol_x.args[1]
                    if isinstance(sol_y, Add):
                        sol_y = sol_y.args[0]*p + sol_y.args[1]

                tot_x.append(sol_x)
                tot_y.append(sol_y)

            solutions.append(Add(*tot_x))
            c = Add(*tot_y)

        solutions.append(c)
        result.add(solutions)
        return result


class BinaryQuadratic(DiophantineEquationType):
    """
    Representation of a binary quadratic diophantine equation.

    A binary quadratic diophantine equation is an equation of the
    form `Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0`, where `A, B, C, D, E,
    F` are integer constants and `x` and `y` are integer variables.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.solvers.diophantine.diophantine import BinaryQuadratic
    >>> b1 = BinaryQuadratic(x**3 + y**2 + 1)
    >>> b1.matches()
    False
    >>> b2 = BinaryQuadratic(x**2 + y**2 + 2*x + 2*y + 2)
    >>> b2.matches()
    True
    >>> b2.solve()
    {(-1, -1)}

    References
    ==========

    .. [1] Methods to solve Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0, [online],
          Available: https://www.alpertron.com.ar/METHODS.HTM
    .. [2] Solving the equation ax^2+ bxy + cy^2 + dx + ey + f= 0, [online],
          Available: https://web.archive.org/web/20160323033111/http://www.jpr2718.org/ax2p.pdf

    """

    name = 'binary_quadratic'

    def matches(self):
        return self.total_degree == 2 and self.dimension == 2

    def solve(self, parameters=None, limit=None) -> DiophantineSolutionSet:
        self.pre_solve(parameters)

        var = self.free_symbols
        coeff = self.coeff

        x, y = var

        A = coeff[x**2]
        B = coeff[x*y]
        C = coeff[y**2]
        D = coeff[x]
        E = coeff[y]
        F = coeff[S.One]

        A, B, C, D, E, F = [as_int(i) for i in _remove_gcd(A, B, C, D, E, F)]

        # (1) Simple-Hyperbolic case: A = C = 0, B != 0
        # In this case equation can be converted to (Bx + E)(By + D) = DE - BF
        # We consider two cases; DE - BF = 0 and DE - BF != 0
        # More details, https://www.alpertron.com.ar/METHODS.HTM#SHyperb

        result = DiophantineSolutionSet(var, self.parameters)
        t, u = result.parameters

        discr = B**2 - 4*A*C
        if A == 0 and C == 0 and B != 0:

            if D*E - B*F == 0:
                q, r = divmod(E, B)
                if not r:
                    result.add((-q, t))
                q, r = divmod(D, B)
                if not r:
                    result.add((t, -q))
            else:
                div = divisors(D*E - B*F)
                div = div + [-term for term in div]
                for d in div:
                    x0, r = divmod(d - E, B)
                    if not r:
                        q, r = divmod(D*E - B*F, d)
                        if not r:
                            y0, r = divmod(q - D, B)
                            if not r:
                                result.add((x0, y0))

        # (2) Parabolic case: B**2 - 4*A*C = 0
        # There are two subcases to be considered in this case.
        # sqrt(c)D - sqrt(a)E = 0 and sqrt(c)D - sqrt(a)E != 0
        # More Details, https://www.alpertron.com.ar/METHODS.HTM#Parabol

        elif discr == 0:

            if A == 0:
                s = BinaryQuadratic(self.equation, free_symbols=[y, x]).solve(parameters=[t, u])
                for soln in s:
                    result.add((soln[1], soln[0]))

            else:
                g = sign(A)*igcd(A, C)
                a = A // g
                c = C // g
                e = sign(B / A)

                sqa = isqrt(a)
                sqc = isqrt(c)
                _c = e*sqc*D - sqa*E
                if not _c:
                    z = Symbol("z", real=True)
                    eq = sqa*g*z**2 + D*z + sqa*F
                    roots = solveset_real(eq, z).intersect(S.Integers)
                    for root in roots:
                        ans = diop_solve(sqa*x + e*sqc*y - root)
                        result.add((ans[0], ans[1]))

                elif int_valued(c):
                    solve_x = lambda u: -e*sqc*g*_c*t**2 - (E + 2*e*sqc*g*u)*t \
                                        - (e*sqc*g*u**2 + E*u + e*sqc*F) // _c

                    solve_y = lambda u: sqa*g*_c*t**2 + (D + 2*sqa*g*u)*t \
                                        + (sqa*g*u**2 + D*u + sqa*F) // _c

                    for z0 in range(0, abs(_c)):
                        # Check if the coefficients of y and x obtained are integers or not
                        if (divisible(sqa*g*z0**2 + D*z0 + sqa*F, _c) and
                            divisible(e*sqc*g*z0**2 + E*z0 + e*sqc*F, _c)):
                            result.add((solve_x(z0), solve_y(z0)))

        # (3) Method used when B**2 - 4*A*C is a square, is described in p. 6 of the below paper
        # by John P. Robertson.
        # https://web.archive.org/web/20160323033111/http://www.jpr2718.org/ax2p.pdf

        elif is_square(discr):
            if A != 0:
                r = sqrt(discr)
                u, v = symbols("u, v", integer=True)
                eq = _mexpand(
                    4*A*r*u*v + 4*A*D*(B*v + r*u + r*v - B*u) +
                    2*A*4*A*E*(u - v) + 4*A*r*4*A*F)

                solution = diop_solve(eq, t)

                for s0, t0 in solution:

                    num = B*t0 + r*s0 + r*t0 - B*s0
                    x_0 = S(num) / (4*A*r)
                    y_0 = S(s0 - t0) / (2*r)
                    if isinstance(s0, Symbol) or isinstance(t0, Symbol):
                        if len(check_param(x_0, y_0, 4*A*r, parameters)) > 0:
                            ans = check_param(x_0, y_0, 4*A*r, parameters)
                            result.update(*ans)
                    elif x_0.is_Integer and y_0.is_Integer:
                        if is_solution_quad(var, coeff, x_0, y_0):
                            result.add((x_0, y_0))

            else:
                s = BinaryQuadratic(self.equation, free_symbols=var[::-1]).solve(parameters=[t, u])  # Interchange x and y
                while s:
                    result.add(s.pop()[::-1])  # and solution <--------+

        # (4) B**2 - 4*A*C > 0 and B**2 - 4*A*C not a square or B**2 - 4*A*C < 0

        else:

            P, Q = _transformation_to_DN(var, coeff)
            D, N = _find_DN(var, coeff)
            solns_pell = diop_DN(D, N)

            if D < 0:
                for x0, y0 in solns_pell:
                    for x in [-x0, x0]:
                        for y in [-y0, y0]:
                            s = P*Matrix([x, y]) + Q
                            try:
                                result.add([as_int(_) for _ in s])
                            except ValueError:
                                pass
            else:
                # In this case equation can be transformed into a Pell equation

                solns_pell = set(solns_pell)
                solns_pell.update((-X, -Y) for X, Y in list(solns_pell))

                a = diop_DN(D, 1)
                T = a[0][0]
                U = a[0][1]

                if all(int_valued(_) for _ in P[:4] + Q[:2]):
                    for r, s in solns_pell:
                        _a = (r + s*sqrt(D))*(T + U*sqrt(D))**t
                        _b = (r - s*sqrt(D))*(T - U*sqrt(D))**t
                        x_n = _mexpand(S(_a + _b) / 2)
                        y_n = _mexpand(S(_a - _b) / (2*sqrt(D)))
                        s = P*Matrix([x_n, y_n]) + Q
                        result.add(s)

                else:
                    L = ilcm(*[_.q for _ in P[:4] + Q[:2]])

                    k = 1

                    T_k = T
                    U_k = U

                    while (T_k - 1) % L != 0 or U_k % L != 0:
                        T_k, U_k = T_k*T + D*U_k*U, T_k*U + U_k*T
                        k += 1

                    for X, Y in solns_pell:

                        for i in range(k):
                            if all(int_valued(_) for _ in P*Matrix([X, Y]) + Q):
                                _a = (X + sqrt(D)*Y)*(T_k + sqrt(D)*U_k)**t
                                _b = (X - sqrt(D)*Y)*(T_k - sqrt(D)*U_k)**t
                                Xt = S(_a + _b) / 2
                                Yt = S(_a - _b) / (2*sqrt(D))
                                s = P*Matrix([Xt, Yt]) + Q
                                result.add(s)

                            X, Y = X*T + D*U*Y, X*U + Y*T

        return result


class InhomogeneousTernaryQuadratic(DiophantineEquationType):
    """

    Representation of an inhomogeneous ternary quadratic.

    No solver is currently implemented for this equation type.

    """

    name = 'inhomogeneous_ternary_quadratic'

    def matches(self):
        if not (self.total_degree == 2 and self.dimension == 3):
            return False
        if not self.homogeneous:
            return False
        return not self.homogeneous_order


class HomogeneousTernaryQuadraticNormal(DiophantineEquationType):
    """
    Representation of a homogeneous ternary quadratic normal diophantine equation.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.solvers.diophantine.diophantine import HomogeneousTernaryQuadraticNormal
    >>> HomogeneousTernaryQuadraticNormal(4*x**2 - 5*y**2 + z**2).solve()
    {(1, 2, 4)}

    """

    name = 'homogeneous_ternary_quadratic_normal'

    def matches(self):
        if not (self.total_degree == 2 and self.dimension == 3):
            return False
        if not self.homogeneous:
            return False
        if not self.homogeneous_order:
            return False

        nonzero = [k for k in self.coeff if self.coeff[k]]
        return len(nonzero) == 3 and all(i**2 in nonzero for i in self.free_symbols)

    def solve(self, parameters=None, limit=None) -> DiophantineSolutionSet:
        self.pre_solve(parameters)

        var = self.free_symbols
        coeff = self.coeff

        x, y, z = var

        a = coeff[x**2]
        b = coeff[y**2]
        c = coeff[z**2]

        (sqf_of_a, sqf_of_b, sqf_of_c), (a_1, b_1, c_1), (a_2, b_2, c_2) = \
            sqf_normal(a, b, c, steps=True)

        A = -a_2*c_2
        B = -b_2*c_2

        result = DiophantineSolutionSet(var, parameters=self.parameters)

        # If following two conditions are satisfied then there are no solutions
        if A < 0 and B < 0:
            return result

        if (
            sqrt_mod(-b_2*c_2, a_2) is None or
            sqrt_mod(-c_2*a_2, b_2) is None or
            sqrt_mod(-a_2*b_2, c_2) is None):
            return result

        z_0, x_0, y_0 = descent(A, B)

        z_0, q = _rational_pq(z_0, abs(c_2))
        x_0 *= q
        y_0 *= q

        x_0, y_0, z_0 = _remove_gcd(x_0, y_0, z_0)

        # Holzer reduction
        if sign(a) == sign(b):
            x_0, y_0, z_0 = holzer(x_0, y_0, z_0, abs(a_2), abs(b_2), abs(c_2))
        elif sign(a) == sign(c):
            x_0, z_0, y_0 = holzer(x_0, z_0, y_0, abs(a_2), abs(c_2), abs(b_2))
        else:
            y_0, z_0, x_0 = holzer(y_0, z_0, x_0, abs(b_2), abs(c_2), abs(a_2))

        x_0 = reconstruct(b_1, c_1, x_0)
        y_0 = reconstruct(a_1, c_1, y_0)
        z_0 = reconstruct(a_1, b_1, z_0)

        sq_lcm = ilcm(sqf_of_a, sqf_of_b, sqf_of_c)

        x_0 = abs(x_0*sq_lcm // sqf_of_a)
        y_0 = abs(y_0*sq_lcm // sqf_of_b)
        z_0 = abs(z_0*sq_lcm // sqf_of_c)

        result.add(_remove_gcd(x_0, y_0, z_0))
        return result


class HomogeneousTernaryQuadratic(DiophantineEquationType):
    """
    Representation of a homogeneous ternary quadratic diophantine equation.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.solvers.diophantine.diophantine import HomogeneousTernaryQuadratic
    >>> HomogeneousTernaryQuadratic(x**2 + y**2 - 3*z**2 + x*y).solve()
    {(-1, 2, 1)}
    >>> HomogeneousTernaryQuadratic(3*x**2 + y**2 - 3*z**2 + 5*x*y + y*z).solve()
    {(3, 12, 13)}

    """

    name = 'homogeneous_ternary_quadratic'

    def matches(self):
        if not (self.total_degree == 2 and self.dimension == 3):
            return False
        if not self.homogeneous:
            return False
        if not self.homogeneous_order:
            return False

        nonzero = [k for k in self.coeff if self.coeff[k]]
        return not (len(nonzero) == 3 and all(i**2 in nonzero for i in self.free_symbols))

    def solve(self, parameters=None, limit=None):
        self.pre_solve(parameters)

        _var = self.free_symbols
        coeff = self.coeff

        x, y, z = _var
        var = [x, y, z]

        # Equations of the form B*x*y + C*z*x + E*y*z = 0 and At least two of the
        # coefficients A, B, C are non-zero.
        # There are infinitely many solutions for the equation.
        # Ex: (0, 0, t), (0, t, 0), (t, 0, 0)
        # Equation can be re-written as y*(B*x + E*z) = -C*x*z and we can find rather
        # unobvious solutions. Set y = -C and B*x + E*z = x*z. The latter can be solved by
        # using methods for binary quadratic diophantine equations. Let's select the
        # solution which minimizes |x| + |z|

        result = DiophantineSolutionSet(var, parameters=self.parameters)

        def unpack_sol(sol):
            if len(sol) > 0:
                return list(sol)[0]
            return None, None, None

        if not any(coeff[i**2] for i in var):
            if coeff[x*z]:
                sols = diophantine(coeff[x*y]*x + coeff[y*z]*z - x*z)
                s = min(sols, key=lambda r: abs(r[0]) + abs(r[1]))
                result.add(_remove_gcd(s[0], -coeff[x*z], s[1]))
                return result

            var[0], var[1] = _var[1], _var[0]
            y_0, x_0, z_0 = unpack_sol(_diop_ternary_quadratic(var, coeff))
            if x_0 is not None:
                result.add((x_0, y_0, z_0))
            return result

        if coeff[x**2] == 0:
            # If the coefficient of x is zero change the variables
            if coeff[y**2] == 0:
                var[0], var[2] = _var[2], _var[0]
                z_0, y_0, x_0 = unpack_sol(_diop_ternary_quadratic(var, coeff))

            else:
                var[0], var[1] = _var[1], _var[0]
                y_0, x_0, z_0 = unpack_sol(_diop_ternary_quadratic(var, coeff))

        else:
            if coeff[x*y] or coeff[x*z]:
                # Apply the transformation x --> X - (B*y + C*z)/(2*A)
                A = coeff[x**2]
                B = coeff[x*y]
                C = coeff[x*z]
                D = coeff[y**2]
                E = coeff[y*z]
                F = coeff[z**2]

                _coeff = {}

                _coeff[x**2] = 4*A**2
                _coeff[y**2] = 4*A*D - B**2
                _coeff[z**2] = 4*A*F - C**2
                _coeff[y*z] = 4*A*E - 2*B*C
                _coeff[x*y] = 0
                _coeff[x*z] = 0

                x_0, y_0, z_0 = unpack_sol(_diop_ternary_quadratic(var, _coeff))

                if x_0 is None:
                    return result

                p, q = _rational_pq(B*y_0 + C*z_0, 2*A)
                x_0, y_0, z_0 = x_0*q - p, y_0*q, z_0*q

            elif coeff[z*y] != 0:
                if coeff[y**2] == 0:
                    if coeff[z**2] == 0:
                        # Equations of the form A*x**2 + E*yz = 0.
                        A = coeff[x**2]
                        E = coeff[y*z]

                        b, a = _rational_pq(-E, A)

                        x_0, y_0, z_0 = b, a, b

                    else:
                        # Ax**2 + E*y*z + F*z**2  = 0
                        var[0], var[2] = _var[2], _var[0]
                        z_0, y_0, x_0 = unpack_sol(_diop_ternary_quadratic(var, coeff))

                else:
                    # A*x**2 + D*y**2 + E*y*z + F*z**2 = 0, C may be zero
                    var[0], var[1] = _var[1], _var[0]
                    y_0, x_0, z_0 = unpack_sol(_diop_ternary_quadratic(var, coeff))

            else:
                # Ax**2 + D*y**2 + F*z**2 = 0, C may be zero
                x_0, y_0, z_0 = unpack_sol(_diop_ternary_quadratic_normal(var, coeff))

        if x_0 is None:
            return result

        result.add(_remove_gcd(x_0, y_0, z_0))
        return result


class InhomogeneousGeneralQuadratic(DiophantineEquationType):
    """

    Representation of an inhomogeneous general quadratic.

    No solver is currently implemented for this equation type.

    """

    name = 'inhomogeneous_general_quadratic'

    def matches(self):
        if not (self.total_degree == 2 and self.dimension >= 3):
            return False
        if not self.homogeneous_order:
            return True
        # there may be Pow keys like x**2 or Mul keys like x*y
        return any(k.is_Mul for k in self.coeff) and not self.homogeneous


class HomogeneousGeneralQuadratic(DiophantineEquationType):
    """

    Representation of a homogeneous general quadratic.

    No solver is currently implemented for this equation type.

    """

    name = 'homogeneous_general_quadratic'

    def matches(self):
        if not (self.total_degree == 2 and self.dimension >= 3):
            return False
        if not self.homogeneous_order:
            return False
        # there may be Pow keys like x**2 or Mul keys like x*y
        return any(k.is_Mul for k in self.coeff) and self.homogeneous


class GeneralSumOfSquares(DiophantineEquationType):
    r"""
    Representation of the diophantine equation

    `x_{1}^2 + x_{2}^2 + . . . + x_{n}^2 - k = 0`.

    Details
    =======

    When `n = 3` if `k = 4^a(8m + 7)` for some `a, m \in Z` then there will be
    no solutions. Refer [1]_ for more details.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import GeneralSumOfSquares
    >>> from sympy.abc import a, b, c, d, e
    >>> GeneralSumOfSquares(a**2 + b**2 + c**2 + d**2 + e**2 - 2345).solve()
    {(15, 22, 22, 24, 24)}

    By default only 1 solution is returned. Use the `limit` keyword for more:

    >>> sorted(GeneralSumOfSquares(a**2 + b**2 + c**2 + d**2 + e**2 - 2345).solve(limit=3))
    [(15, 22, 22, 24, 24), (16, 19, 24, 24, 24), (16, 20, 22, 23, 26)]

    References
    ==========

    .. [1] Representing an integer as a sum of three squares, [online],
        Available:
        https://proofwiki.org/wiki/Integer_as_Sum_of_Three_Squares
    """

    name = 'general_sum_of_squares'

    def matches(self):
        if not (self.total_degree == 2 and self.dimension >= 3):
            return False
        if not self.homogeneous_order:
            return False
        if any(k.is_Mul for k in self.coeff):
            return False
        return all(self.coeff[k] == 1 for k in self.coeff if k != 1)

    def solve(self, parameters=None, limit=1):
        self.pre_solve(parameters)

        var = self.free_symbols
        k = -int(self.coeff[1])
        n = self.dimension

        result = DiophantineSolutionSet(var, parameters=self.parameters)

        if k < 0 or limit < 1:
            return result

        signs = [-1 if x.is_nonpositive else 1 for x in var]
        negs = signs.count(-1) != 0

        took = 0
        for t in sum_of_squares(k, n, zeros=True):
            if negs:
                result.add([signs[i]*j for i, j in enumerate(t)])
            else:
                result.add(t)
            took += 1
            if took == limit:
                break
        return result


class GeneralPythagorean(DiophantineEquationType):
    """
    Representation of the general pythagorean equation,
    `a_{1}^2x_{1}^2 + a_{2}^2x_{2}^2 + . . . + a_{n}^2x_{n}^2 - a_{n + 1}^2x_{n + 1}^2 = 0`.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import GeneralPythagorean
    >>> from sympy.abc import a, b, c, d, e, x, y, z, t
    >>> GeneralPythagorean(a**2 + b**2 + c**2 - d**2).solve()
    {(t_0**2 + t_1**2 - t_2**2, 2*t_0*t_2, 2*t_1*t_2, t_0**2 + t_1**2 + t_2**2)}
    >>> GeneralPythagorean(9*a**2 - 4*b**2 + 16*c**2 + 25*d**2 + e**2).solve(parameters=[x, y, z, t])
    {(-10*t**2 + 10*x**2 + 10*y**2 + 10*z**2, 15*t**2 + 15*x**2 + 15*y**2 + 15*z**2, 15*t*x, 12*t*y, 60*t*z)}
    """

    name = 'general_pythagorean'

    def matches(self):
        if not (self.total_degree == 2 and self.dimension >= 3):
            return False
        if not self.homogeneous_order:
            return False
        if any(k.is_Mul for k in self.coeff):
            return False
        if all(self.coeff[k] == 1 for k in self.coeff if k != 1):
            return False
        if not all(is_square(abs(self.coeff[k])) for k in self.coeff):
            return False
        # all but one has the same sign
        # e.g. 4*x**2 + y**2 - 4*z**2
        return abs(sum(sign(self.coeff[k]) for k in self.coeff)) == self.dimension - 2

    @property
    def n_parameters(self):
        return self.dimension - 1

    def solve(self, parameters=None, limit=1):
        self.pre_solve(parameters)

        coeff = self.coeff
        var = self.free_symbols
        n = self.dimension

        if sign(coeff[var[0] ** 2]) + sign(coeff[var[1] ** 2]) + sign(coeff[var[2] ** 2]) < 0:
            for key in coeff.keys():
                coeff[key] = -coeff[key]

        result = DiophantineSolutionSet(var, parameters=self.parameters)

        index = 0

        for i, v in enumerate(var):
            if sign(coeff[v ** 2]) == -1:
                index = i

        m = result.parameters

        ith = sum(m_i ** 2 for m_i in m)
        L = [ith - 2 * m[n - 2] ** 2]
        L.extend([2 * m[i] * m[n - 2] for i in range(n - 2)])
        sol = L[:index] + [ith] + L[index:]

        lcm = 1
        for i, v in enumerate(var):
            if i == index or (index > 0 and i == 0) or (index == 0 and i == 1):
                lcm = ilcm(lcm, sqrt(abs(coeff[v ** 2])))
            else:
                s = sqrt(coeff[v ** 2])
                lcm = ilcm(lcm, s if _odd(s) else s // 2)

        for i, v in enumerate(var):
            sol[i] = (lcm * sol[i]) / sqrt(abs(coeff[v ** 2]))

        result.add(sol)
        return result


class CubicThue(DiophantineEquationType):
    """
    Representation of a cubic Thue diophantine equation.

    A cubic Thue diophantine equation is a polynomial of the form
    `f(x, y) = r` of degree 3, where `x` and `y` are integers
    and `r` is a rational number.

    No solver is currently implemented for this equation type.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.solvers.diophantine.diophantine import CubicThue
    >>> c1 = CubicThue(x**3 + y**2 + 1)
    >>> c1.matches()
    True

    """

    name = 'cubic_thue'

    def matches(self):
        return self.total_degree == 3 and self.dimension == 2


class GeneralSumOfEvenPowers(DiophantineEquationType):
    """
    Representation of the diophantine equation

    `x_{1}^e + x_{2}^e + . . . + x_{n}^e - k = 0`

    where `e` is an even, integer power.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import GeneralSumOfEvenPowers
    >>> from sympy.abc import a, b
    >>> GeneralSumOfEvenPowers(a**4 + b**4 - (2**4 + 3**4)).solve()
    {(2, 3)}

    """

    name = 'general_sum_of_even_powers'

    def matches(self):
        if not self.total_degree > 3:
            return False
        if self.total_degree % 2 != 0:
            return False
        if not all(k.is_Pow and k.exp == self.total_degree for k in self.coeff if k != 1):
            return False
        return all(self.coeff[k] == 1 for k in self.coeff if k != 1)

    def solve(self, parameters=None, limit=1):
        self.pre_solve(parameters)

        var = self.free_symbols
        coeff = self.coeff

        p = None
        for q in coeff.keys():
            if q.is_Pow and coeff[q]:
                p = q.exp

        k = len(var)
        n = -coeff[1]

        result = DiophantineSolutionSet(var, parameters=self.parameters)

        if n < 0 or limit < 1:
            return result

        sign = [-1 if x.is_nonpositive else 1 for x in var]
        negs = sign.count(-1) != 0

        took = 0
        for t in power_representation(n, p, k):
            if negs:
                result.add([sign[i]*j for i, j in enumerate(t)])
            else:
                result.add(t)
            took += 1
            if took == limit:
                break
        return result

# these types are known (but not necessarily handled)
# note that order is important here (in the current solver state)
all_diop_classes = [
    Linear,
    Univariate,
    BinaryQuadratic,
    InhomogeneousTernaryQuadratic,
    HomogeneousTernaryQuadraticNormal,
    HomogeneousTernaryQuadratic,
    InhomogeneousGeneralQuadratic,
    HomogeneousGeneralQuadratic,
    GeneralSumOfSquares,
    GeneralPythagorean,
    CubicThue,
    GeneralSumOfEvenPowers,
]

diop_known = {diop_class.name for diop_class in all_diop_classes}


def _remove_gcd(*x):
    try:
        g = igcd(*x)
    except ValueError:
        fx = list(filter(None, x))
        if len(fx) < 2:
            return x
        g = igcd(*[i.as_content_primitive()[0] for i in fx])
    except TypeError:
        raise TypeError('_remove_gcd(a,b,c) or _remove_gcd(*container)')
    if g == 1:
        return x
    return tuple([i//g for i in x])


def _rational_pq(a, b):
    # return `(numer, denom)` for a/b; sign in numer and gcd removed
    return _remove_gcd(sign(b)*a, abs(b))


def _nint_or_floor(p, q):
    # return nearest int to p/q; in case of tie return floor(p/q)
    w, r = divmod(p, q)
    if abs(r) <= abs(q)//2:
        return w
    return w + 1


def _odd(i):
    return i % 2 != 0


def _even(i):
    return i % 2 == 0


def diophantine(eq, param=symbols("t", integer=True), syms=None,
                permute=False):
    """
    Simplify the solution procedure of diophantine equation ``eq`` by
    converting it into a product of terms which should equal zero.

    Explanation
    ===========

    For example, when solving, `x^2 - y^2 = 0` this is treated as
    `(x + y)(x - y) = 0` and `x + y = 0` and `x - y = 0` are solved
    independently and combined. Each term is solved by calling
    ``diop_solve()``. (Although it is possible to call ``diop_solve()``
    directly, one must be careful to pass an equation in the correct
    form and to interpret the output correctly; ``diophantine()`` is
    the public-facing function to use in general.)

    Output of ``diophantine()`` is a set of tuples. The elements of the
    tuple are the solutions for each variable in the equation and
    are arranged according to the alphabetic ordering of the variables.
    e.g. For an equation with two variables, `a` and `b`, the first
    element of the tuple is the solution for `a` and the second for `b`.

    Usage
    =====

    ``diophantine(eq, t, syms)``: Solve the diophantine
    equation ``eq``.
    ``t`` is the optional parameter to be used by ``diop_solve()``.
    ``syms`` is an optional list of symbols which determines the
    order of the elements in the returned tuple.

    By default, only the base solution is returned. If ``permute`` is set to
    True then permutations of the base solution and/or permutations of the
    signs of the values will be returned when applicable.

    Details
    =======

    ``eq`` should be an expression which is assumed to be zero.
    ``t`` is the parameter to be used in the solution.

    Examples
    ========

    >>> from sympy import diophantine
    >>> from sympy.abc import a, b
    >>> eq = a**4 + b**4 - (2**4 + 3**4)
    >>> diophantine(eq)
    {(2, 3)}
    >>> diophantine(eq, permute=True)
    {(-3, -2), (-3, 2), (-2, -3), (-2, 3), (2, -3), (2, 3), (3, -2), (3, 2)}

    >>> from sympy.abc import x, y, z
    >>> diophantine(x**2 - y**2)
    {(t_0, -t_0), (t_0, t_0)}

    >>> diophantine(x*(2*x + 3*y - z))
    {(0, n1, n2), (t_0, t_1, 2*t_0 + 3*t_1)}
    >>> diophantine(x**2 + 3*x*y + 4*x)
    {(0, n1), (-3*t_0 - 4, t_0)}

    See Also
    ========

    diop_solve
    sympy.utilities.iterables.permute_signs
    sympy.utilities.iterables.signed_permutations
    """

    eq = _sympify(eq)

    if isinstance(eq, Eq):
        eq = eq.lhs - eq.rhs

    try:
        var = list(eq.expand(force=True).free_symbols)
        var.sort(key=default_sort_key)
        if syms:
            if not is_sequence(syms):
                raise TypeError(
                    'syms should be given as a sequence, e.g. a list')
            syms = [i for i in syms if i in var]
            if syms != var:
                dict_sym_index = dict(zip(syms, range(len(syms))))
                return {tuple([t[dict_sym_index[i]] for i in var])
                            for t in diophantine(eq, param, permute=permute)}
        n, d = eq.as_numer_denom()
        if n.is_number:
            return set()
        if not d.is_number:
            dsol = diophantine(d)
            good = diophantine(n) - dsol
            return {s for s in good if _mexpand(d.subs(zip(var, s)))}
        eq = factor_terms(n)
        assert not eq.is_number
        eq = eq.as_independent(*var, as_Add=False)[1]
        p = Poly(eq)
        assert not any(g.is_number for g in p.gens)
        eq = p.as_expr()
        assert eq.is_polynomial()
    except (GeneratorsNeeded, AssertionError):
        raise TypeError(filldedent('''
    Equation should be a polynomial with Rational coefficients.'''))

    # permute only sign
    do_permute_signs = False
    # permute sign and values
    do_permute_signs_var = False
    # permute few signs
    permute_few_signs = False
    try:
        # if we know that factoring should not be attempted, skip
        # the factoring step
        v, c, t = classify_diop(eq)

        # check for permute sign
        if permute:
            len_var = len(v)
            permute_signs_for = [
                GeneralSumOfSquares.name,
                GeneralSumOfEvenPowers.name]
            permute_signs_check = [
                HomogeneousTernaryQuadratic.name,
                HomogeneousTernaryQuadraticNormal.name,
                BinaryQuadratic.name]
            if t in permute_signs_for:
                do_permute_signs_var = True
            elif t in permute_signs_check:
                # if all the variables in eq have even powers
                # then do_permute_sign = True
                if len_var == 3:
                    var_mul = list(subsets(v, 2))
                    # here var_mul is like [(x, y), (x, z), (y, z)]
                    xy_coeff = True
                    x_coeff = True
                    var1_mul_var2 = (a[0]*a[1] for a in var_mul)
                    # if coeff(y*z), coeff(y*x), coeff(x*z) is not 0 then
                    # `xy_coeff` => True and do_permute_sign => False.
                    # Means no permuted solution.
                    for v1_mul_v2 in var1_mul_var2:
                        try:
                            coeff = c[v1_mul_v2]
                        except KeyError:
                            coeff = 0
                        xy_coeff = bool(xy_coeff) and bool(coeff)
                    var_mul = list(subsets(v, 1))
                    # here var_mul is like [(x,), (y, )]
                    for v1 in var_mul:
                        try:
                            coeff = c[v1[0]]
                        except KeyError:
                            coeff = 0
                        x_coeff = bool(x_coeff) and bool(coeff)
                    if not any((xy_coeff, x_coeff)):
                        # means only x**2, y**2, z**2, const is present
                        do_permute_signs = True
                    elif not x_coeff:
                        permute_few_signs = True
                elif len_var == 2:
                    var_mul = list(subsets(v, 2))
                    # here var_mul is like [(x, y)]
                    xy_coeff = True
                    x_coeff = True
                    var1_mul_var2 = (x[0]*x[1] for x in var_mul)
                    for v1_mul_v2 in var1_mul_var2:
                        try:
                            coeff = c[v1_mul_v2]
                        except KeyError:
                            coeff = 0
                        xy_coeff = bool(xy_coeff) and bool(coeff)
                    var_mul = list(subsets(v, 1))
                    # here var_mul is like [(x,), (y, )]
                    for v1 in var_mul:
                        try:
                            coeff = c[v1[0]]
                        except KeyError:
                            coeff = 0
                        x_coeff = bool(x_coeff) and bool(coeff)
                    if not any((xy_coeff, x_coeff)):
                        # means only x**2, y**2 and const is present
                        # so we can get more soln by permuting this soln.
                        do_permute_signs = True
                    elif not x_coeff:
                        # when coeff(x), coeff(y) is not present then signs of
                        #  x, y can be permuted such that their sign are same
                        # as sign of x*y.
                        # e.g 1. (x_val,y_val)=> (x_val,y_val), (-x_val,-y_val)
                        # 2. (-x_vall, y_val)=> (-x_val,y_val), (x_val,-y_val)
                        permute_few_signs = True
        if t == 'general_sum_of_squares':
            # trying to factor such expressions will sometimes hang
            terms = [(eq, 1)]
        else:
            raise TypeError
    except (TypeError, NotImplementedError):
        fl = factor_list(eq)
        if fl[0].is_Rational and fl[0] != 1:
            return diophantine(eq/fl[0], param=param, syms=syms, permute=permute)
        terms = fl[1]

    sols = set()

    for term in terms:

        base, _ = term
        var_t, _, eq_type = classify_diop(base, _dict=False)
        _, base = signsimp(base, evaluate=False).as_coeff_Mul()
        solution = diop_solve(base, param)

        if eq_type in [
                Linear.name,
                HomogeneousTernaryQuadratic.name,
                HomogeneousTernaryQuadraticNormal.name,
                GeneralPythagorean.name]:
            sols.add(merge_solution(var, var_t, solution))

        elif eq_type in [
                BinaryQuadratic.name,
                GeneralSumOfSquares.name,
                GeneralSumOfEvenPowers.name,
                Univariate.name]:
            sols.update(merge_solution(var, var_t, sol) for sol in solution)

        else:
            raise NotImplementedError('unhandled type: %s' % eq_type)

    sols.discard(())
    null = tuple([0]*len(var))
    # if there is no solution, return trivial solution
    if not sols and eq.subs(zip(var, null)).is_zero:
        if all(check_assumptions(val, **s.assumptions0) is not False for val, s in zip(null, var)):
            sols.add(null)

    final_soln = set()
    for sol in sols:
        if all(int_valued(s) for s in sol):
            if do_permute_signs:
                permuted_sign = set(permute_signs(sol))
                final_soln.update(permuted_sign)
            elif permute_few_signs:
                lst = list(permute_signs(sol))
                lst = list(filter(lambda x: x[0]*x[1] == sol[1]*sol[0], lst))
                permuted_sign = set(lst)
                final_soln.update(permuted_sign)
            elif do_permute_signs_var:
                permuted_sign_var = set(signed_permutations(sol))
                final_soln.update(permuted_sign_var)
            else:
                final_soln.add(sol)
        else:
                final_soln.add(sol)
    return final_soln


def merge_solution(var, var_t, solution):
    """
    This is used to construct the full solution from the solutions of sub
    equations.

    Explanation
    ===========

    For example when solving the equation `(x - y)(x^2 + y^2 - z^2) = 0`,
    solutions for each of the equations `x - y = 0` and `x^2 + y^2 - z^2` are
    found independently. Solutions for `x - y = 0` are `(x, y) = (t, t)`. But
    we should introduce a value for z when we output the solution for the
    original equation. This function converts `(t, t)` into `(t, t, n_{1})`
    where `n_{1}` is an integer parameter.
    """
    sol = []

    if None in solution:
        return ()

    solution = iter(solution)
    params = numbered_symbols("n", integer=True, start=1)
    for v in var:
        if v in var_t:
            sol.append(next(solution))
        else:
            sol.append(next(params))

    for val, symb in zip(sol, var):
        if check_assumptions(val, **symb.assumptions0) is False:
            return ()

    return tuple(sol)


def _diop_solve(eq, params=None):
    for diop_type in all_diop_classes:
        if diop_type(eq).matches():
            return diop_type(eq).solve(parameters=params)


def diop_solve(eq, param=symbols("t", integer=True)):
    """
    Solves the diophantine equation ``eq``.

    Explanation
    ===========

    Unlike ``diophantine()``, factoring of ``eq`` is not attempted. Uses
    ``classify_diop()`` to determine the type of the equation and calls
    the appropriate solver function.

    Use of ``diophantine()`` is recommended over other helper functions.
    ``diop_solve()`` can return either a set or a tuple depending on the
    nature of the equation. All non-trivial solutions are returned: assumptions
    on symbols are ignored.

    Usage
    =====

    ``diop_solve(eq, t)``: Solve diophantine equation, ``eq`` using ``t``
    as a parameter if needed.

    Details
    =======

    ``eq`` should be an expression which is assumed to be zero.
    ``t`` is a parameter to be used in the solution.

    Examples
    ========

    >>> from sympy.solvers.diophantine import diop_solve
    >>> from sympy.abc import x, y, z, w
    >>> diop_solve(2*x + 3*y - 5)
    (3*t_0 - 5, 5 - 2*t_0)
    >>> diop_solve(4*x + 3*y - 4*z + 5)
    (t_0, 8*t_0 + 4*t_1 + 5, 7*t_0 + 3*t_1 + 5)
    >>> diop_solve(x + 3*y - 4*z + w - 6)
    (t_0, t_0 + t_1, 6*t_0 + 5*t_1 + 4*t_2 - 6, 5*t_0 + 4*t_1 + 3*t_2 - 6)
    >>> diop_solve(x**2 + y**2 - 5)
    {(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)}


    See Also
    ========

    diophantine()
    """
    var, coeff, eq_type = classify_diop(eq, _dict=False)

    if eq_type == Linear.name:
        return diop_linear(eq, param)

    elif eq_type == BinaryQuadratic.name:
        return diop_quadratic(eq, param)

    elif eq_type == HomogeneousTernaryQuadratic.name:
        return diop_ternary_quadratic(eq, parameterize=True)

    elif eq_type == HomogeneousTernaryQuadraticNormal.name:
        return diop_ternary_quadratic_normal(eq, parameterize=True)

    elif eq_type == GeneralPythagorean.name:
        return diop_general_pythagorean(eq, param)

    elif eq_type == Univariate.name:
        return diop_univariate(eq)

    elif eq_type == GeneralSumOfSquares.name:
        return diop_general_sum_of_squares(eq, limit=S.Infinity)

    elif eq_type == GeneralSumOfEvenPowers.name:
        return diop_general_sum_of_even_powers(eq, limit=S.Infinity)

    if eq_type is not None and eq_type not in diop_known:
            raise ValueError(filldedent('''
    Although this type of equation was identified, it is not yet
    handled. It should, however, be listed in `diop_known` at the
    top of this file. Developers should see comments at the end of
    `classify_diop`.
            '''))  # pragma: no cover
    else:
        raise NotImplementedError(
            'No solver has been written for %s.' % eq_type)


def classify_diop(eq, _dict=True):
    # docstring supplied externally

    matched = False
    diop_type = None
    for diop_class in all_diop_classes:
        diop_type = diop_class(eq)
        if diop_type.matches():
            matched = True
            break

    if matched:
        return diop_type.free_symbols, dict(diop_type.coeff) if _dict else diop_type.coeff, diop_type.name

    # new diop type instructions
    # --------------------------
    # if this error raises and the equation *can* be classified,
    #  * it should be identified in the if-block above
    #  * the type should be added to the diop_known
    # if a solver can be written for it,
    #  * a dedicated handler should be written (e.g. diop_linear)
    #  * it should be passed to that handler in diop_solve
    raise NotImplementedError(filldedent('''
        This equation is not yet recognized or else has not been
        simplified sufficiently to put it in a form recognized by
        diop_classify().'''))


classify_diop.func_doc = (  # type: ignore
    '''
    Helper routine used by diop_solve() to find information about ``eq``.

    Explanation
    ===========

    Returns a tuple containing the type of the diophantine equation
    along with the variables (free symbols) and their coefficients.
    Variables are returned as a list and coefficients are returned
    as a dict with the key being the respective term and the constant
    term is keyed to 1. The type is one of the following:

    * %s

    Usage
    =====

    ``classify_diop(eq)``: Return variables, coefficients and type of the
    ``eq``.

    Details
    =======

    ``eq`` should be an expression which is assumed to be zero.
    ``_dict`` is for internal use: when True (default) a dict is returned,
    otherwise a defaultdict which supplies 0 for missing keys is returned.

    Examples
    ========

    >>> from sympy.solvers.diophantine import classify_diop
    >>> from sympy.abc import x, y, z, w, t
    >>> classify_diop(4*x + 6*y - 4)
    ([x, y], {1: -4, x: 4, y: 6}, 'linear')
    >>> classify_diop(x + 3*y -4*z + 5)
    ([x, y, z], {1: 5, x: 1, y: 3, z: -4}, 'linear')
    >>> classify_diop(x**2 + y**2 - x*y + x + 5)
    ([x, y], {1: 5, x: 1, x**2: 1, y**2: 1, x*y: -1}, 'binary_quadratic')
    ''' % ('\n    * '.join(sorted(diop_known))))


def diop_linear(eq, param=symbols("t", integer=True)):
    """
    Solves linear diophantine equations.

    A linear diophantine equation is an equation of the form `a_{1}x_{1} +
    a_{2}x_{2} + .. + a_{n}x_{n} = 0` where `a_{1}, a_{2}, ..a_{n}` are
    integer constants and `x_{1}, x_{2}, ..x_{n}` are integer variables.

    Usage
    =====

    ``diop_linear(eq)``: Returns a tuple containing solutions to the
    diophantine equation ``eq``. Values in the tuple is arranged in the same
    order as the sorted variables.

    Details
    =======

    ``eq`` is a linear diophantine equation which is assumed to be zero.
    ``param`` is the parameter to be used in the solution.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import diop_linear
    >>> from sympy.abc import x, y, z
    >>> diop_linear(2*x - 3*y - 5) # solves equation 2*x - 3*y - 5 == 0
    (3*t_0 - 5, 2*t_0 - 5)

    Here x = -3*t_0 - 5 and y = -2*t_0 - 5

    >>> diop_linear(2*x - 3*y - 4*z -3)
    (t_0, 2*t_0 + 4*t_1 + 3, -t_0 - 3*t_1 - 3)

    See Also
    ========

    diop_quadratic(), diop_ternary_quadratic(), diop_general_pythagorean(),
    diop_general_sum_of_squares()
    """
    var, coeff, diop_type = classify_diop(eq, _dict=False)

    if diop_type == Linear.name:
        parameters = None
        if param is not None:
            parameters = symbols('%s_0:%i' % (param, len(var)), integer=True)

        result = Linear(eq).solve(parameters=parameters)

        if param is None:
            result = result(*[0]*len(result.parameters))

        if len(result) > 0:
            return list(result)[0]
        else:
            return tuple([None]*len(result.parameters))


def base_solution_linear(c, a, b, t=None):
    """
    Return the base solution for the linear equation, `ax + by = c`.

    Explanation
    ===========

    Used by ``diop_linear()`` to find the base solution of a linear
    Diophantine equation. If ``t`` is given then the parametrized solution is
    returned.

    Usage
    =====

    ``base_solution_linear(c, a, b, t)``: ``a``, ``b``, ``c`` are coefficients
    in `ax + by = c` and ``t`` is the parameter to be used in the solution.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import base_solution_linear
    >>> from sympy.abc import t
    >>> base_solution_linear(5, 2, 3) # equation 2*x + 3*y = 5
    (-5, 5)
    >>> base_solution_linear(0, 5, 7) # equation 5*x + 7*y = 0
    (0, 0)
    >>> base_solution_linear(5, 2, 3, t) # equation 2*x + 3*y = 5
    (3*t - 5, 5 - 2*t)
    >>> base_solution_linear(0, 5, 7, t) # equation 5*x + 7*y = 0
    (7*t, -5*t)
    """
    a, b, c = _remove_gcd(a, b, c)

    if c == 0:
        if t is None:
            return (0, 0)
        if b < 0:
            t = -t
        return (b*t, -a*t)

    x0, y0, d = igcdex(abs(a), abs(b))
    x0 *= sign(a)
    y0 *= sign(b)
    if c % d:
        return (None, None)
    if t is None:
        return (c*x0, c*y0)
    if b < 0:
        t = -t
    return (c*x0 + b*t, c*y0 - a*t)


def diop_univariate(eq):
    """
    Solves a univariate diophantine equations.

    Explanation
    ===========

    A univariate diophantine equation is an equation of the form
    `a_{0} + a_{1}x + a_{2}x^2 + .. + a_{n}x^n = 0` where `a_{1}, a_{2}, ..a_{n}` are
    integer constants and `x` is an integer variable.

    Usage
    =====

    ``diop_univariate(eq)``: Returns a set containing solutions to the
    diophantine equation ``eq``.

    Details
    =======

    ``eq`` is a univariate diophantine equation which is assumed to be zero.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import diop_univariate
    >>> from sympy.abc import x
    >>> diop_univariate((x - 2)*(x - 3)**2) # solves equation (x - 2)*(x - 3)**2 == 0
    {(2,), (3,)}

    """
    var, coeff, diop_type = classify_diop(eq, _dict=False)

    if diop_type == Univariate.name:
        return {(int(i),) for i in solveset_real(
            eq, var[0]).intersect(S.Integers)}


def divisible(a, b):
    """
    Returns `True` if ``a`` is divisible by ``b`` and `False` otherwise.
    """
    return not a % b


def diop_quadratic(eq, param=symbols("t", integer=True)):
    """
    Solves quadratic diophantine equations.

    i.e. equations of the form `Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0`. Returns a
    set containing the tuples `(x, y)` which contains the solutions. If there
    are no solutions then `(None, None)` is returned.

    Usage
    =====

    ``diop_quadratic(eq, param)``: ``eq`` is a quadratic binary diophantine
    equation. ``param`` is used to indicate the parameter to be used in the
    solution.

    Details
    =======

    ``eq`` should be an expression which is assumed to be zero.
    ``param`` is a parameter to be used in the solution.

    Examples
    ========

    >>> from sympy.abc import x, y, t
    >>> from sympy.solvers.diophantine.diophantine import diop_quadratic
    >>> diop_quadratic(x**2 + y**2 + 2*x + 2*y + 2, t)
    {(-1, -1)}

    References
    ==========

    .. [1] Methods to solve Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0, [online],
          Available: https://www.alpertron.com.ar/METHODS.HTM
    .. [2] Solving the equation ax^2+ bxy + cy^2 + dx + ey + f= 0, [online],
          Available: https://web.archive.org/web/20160323033111/http://www.jpr2718.org/ax2p.pdf

    See Also
    ========

    diop_linear(), diop_ternary_quadratic(), diop_general_sum_of_squares(),
    diop_general_pythagorean()
    """
    var, coeff, diop_type = classify_diop(eq, _dict=False)

    if diop_type == BinaryQuadratic.name:
        if param is not None:
            parameters = [param, Symbol("u", integer=True)]
        else:
            parameters = None
        return set(BinaryQuadratic(eq).solve(parameters=parameters))


def is_solution_quad(var, coeff, u, v):
    """
    Check whether `(u, v)` is solution to the quadratic binary diophantine
    equation with the variable list ``var`` and coefficient dictionary
    ``coeff``.

    Not intended for use by normal users.
    """
    reps = dict(zip(var, (u, v)))
    eq = Add(*[j*i.xreplace(reps) for i, j in coeff.items()])
    return _mexpand(eq) == 0


def diop_DN(D, N, t=symbols("t", integer=True)):
    """
    Solves the equation `x^2 - Dy^2 = N`.

    Explanation
    ===========

    Mainly concerned with the case `D > 0, D` is not a perfect square,
    which is the same as the generalized Pell equation. The LMM
    algorithm [1]_ is used to solve this equation.

    Returns one solution tuple, (`x, y)` for each class of the solutions.
    Other solutions of the class can be constructed according to the
    values of ``D`` and ``N``.

    Usage
    =====

    ``diop_DN(D, N, t)``: D and N are integers as in `x^2 - Dy^2 = N` and
    ``t`` is the parameter to be used in the solutions.

    Details
    =======

    ``D`` and ``N`` correspond to D and N in the equation.
    ``t`` is the parameter to be used in the solutions.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import diop_DN
    >>> diop_DN(13, -4) # Solves equation x**2 - 13*y**2 = -4
    [(3, 1), (393, 109), (36, 10)]

    The output can be interpreted as follows: There are three fundamental
    solutions to the equation `x^2 - 13y^2 = -4` given by (3, 1), (393, 109)
    and (36, 10). Each tuple is in the form (x, y), i.e. solution (3, 1) means
    that `x = 3` and `y = 1`.

    >>> diop_DN(986, 1) # Solves equation x**2 - 986*y**2 = 1
    [(49299, 1570)]

    See Also
    ========

    find_DN(), diop_bf_DN()

    References
    ==========

    .. [1] Solving the generalized Pell equation x**2 - D*y**2 = N, John P.
        Robertson, July 31, 2004, Pages 16 - 17. [online], Available:
        https://web.archive.org/web/20160323033128/http://www.jpr2718.org/pell.pdf
    """
    if D < 0:
        if N == 0:
            return [(0, 0)]
        if N < 0:
            return []
        # N > 0:
        sol = []
        for d in divisors(square_factor(N), generator=True):
            for x, y in cornacchia(1, int(-D), int(N // d**2)):
                sol.append((d*x, d*y))
                if D == -1:
                    sol.append((d*y, d*x))
        return sol

    if D == 0:
        if N < 0:
            return []
        if N == 0:
            return [(0, t)]
        sN, _exact = integer_nthroot(N, 2)
        if _exact:
            return [(sN, t)]
        return []

    # D > 0
    sD, _exact = integer_nthroot(D, 2)
    if _exact:
        if N == 0:
            return [(sD*t, t)]

        sol = []
        for y in range(floor(sign(N)*(N - 1)/(2*sD)) + 1):
            try:
                sq, _exact = integer_nthroot(D*y**2 + N, 2)
            except ValueError:
                _exact = False
            if _exact:
                sol.append((sq, y))
        return sol

    if 1 < N**2 < D:
        # It is much faster to call `_special_diop_DN`.
        return _special_diop_DN(D, N)

    if N == 0:
        return [(0, 0)]

    sol = []
    if abs(N) == 1:
        pqa = PQa(0, 1, D)
        *_, prev_B, prev_G = next(pqa)
        for j, (*_, a, _, _B, _G) in enumerate(pqa):
            if a == 2*sD:
                break
            prev_B, prev_G = _B, _G
        if j % 2:
            if N == 1:
                sol.append((prev_G, prev_B))
            return sol
        if N == -1:
            return [(prev_G, prev_B)]
        for _ in range(j):
            *_, _B, _G = next(pqa)
        return [(_G, _B)]

    for f in divisors(square_factor(N), generator=True):
        m = N // f**2
        am = abs(m)
        for sqm in sqrt_mod(D, am, all_roots=True):
            z = symmetric_residue(sqm, am)
            pqa = PQa(z, am, D)
            *_, prev_B, prev_G = next(pqa)
            for _ in range(length(z, am, D) - 1):
                _, q, *_, _B, _G = next(pqa)
                if abs(q) == 1:
                    if prev_G**2 - D*prev_B**2 == m:
                        sol.append((f*prev_G, f*prev_B))
                    elif a := diop_DN(D, -1):
                        sol.append((f*(prev_G*a[0][0] + prev_B*D*a[0][1]),
                                    f*(prev_G*a[0][1] + prev_B*a[0][0])))
                    break
                prev_B, prev_G = _B, _G
    return sol


def _special_diop_DN(D, N):
    """
    Solves the equation `x^2 - Dy^2 = N` for the special case where
    `1 < N**2 < D` and `D` is not a perfect square.
    It is better to call `diop_DN` rather than this function, as
    the former checks the condition `1 < N**2 < D`, and calls the latter only
    if appropriate.

    Usage
    =====

    WARNING: Internal method. Do not call directly!

    ``_special_diop_DN(D, N)``: D and N are integers as in `x^2 - Dy^2 = N`.

    Details
    =======

    ``D`` and ``N`` correspond to D and N in the equation.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import _special_diop_DN
    >>> _special_diop_DN(13, -3) # Solves equation x**2 - 13*y**2 = -3
    [(7, 2), (137, 38)]

    The output can be interpreted as follows: There are two fundamental
    solutions to the equation `x^2 - 13y^2 = -3` given by (7, 2) and
    (137, 38). Each tuple is in the form (x, y), i.e. solution (7, 2) means
    that `x = 7` and `y = 2`.

    >>> _special_diop_DN(2445, -20) # Solves equation x**2 - 2445*y**2 = -20
    [(445, 9), (17625560, 356454), (698095554475, 14118073569)]

    See Also
    ========

    diop_DN()

    References
    ==========

    .. [1] Section 4.4.4 of the following book:
        Quadratic Diophantine Equations, T. Andreescu and D. Andrica,
        Springer, 2015.
    """

    # The following assertion was removed for efficiency, with the understanding
    #     that this method is not called directly. The parent method, `diop_DN`
    #     is responsible for performing the appropriate checks.
    #
    # assert (1 < N**2 < D) and (not integer_nthroot(D, 2)[1])

    sqrt_D = isqrt(D)
    F = {N // f**2: f for f in divisors(square_factor(abs(N)), generator=True)}
    P = 0
    Q = 1
    G0, G1 = 0, 1
    B0, B1 = 1, 0

    solutions = []
    while True:
        for _ in range(2):
            a = (P + sqrt_D) // Q
            P = a*Q - P
            Q = (D - P**2) // Q
            G0, G1 = G1, a*G1 + G0
            B0, B1 = B1, a*B1 + B0
            if (s := G1**2 - D*B1**2) in F:
                f = F[s]
                solutions.append((f*G1, f*B1))
        if Q == 1:
            break
    return solutions


def cornacchia(a:int, b:int, m:int) -> set[tuple[int, int]]:
    r"""
    Solves `ax^2 + by^2 = m` where `\gcd(a, b) = 1 = gcd(a, m)` and `a, b > 0`.

    Explanation
    ===========

    Uses the algorithm due to Cornacchia. The method only finds primitive
    solutions, i.e. ones with `\gcd(x, y) = 1`. So this method cannot be used to
    find the solutions of `x^2 + y^2 = 20` since the only solution to former is
    `(x, y) = (4, 2)` and it is not primitive. When `a = b`, only the
    solutions with `x \leq y` are found. For more details, see the References.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import cornacchia
    >>> cornacchia(2, 3, 35) # equation 2x**2 + 3y**2 = 35
    {(2, 3), (4, 1)}
    >>> cornacchia(1, 1, 25) # equation x**2 + y**2 = 25
    {(4, 3)}

    References
    ===========

    .. [1] A. Nitaj, "L'algorithme de Cornacchia"
    .. [2] Solving the diophantine equation ax**2 + by**2 = m by Cornacchia's
        method, [online], Available:
        http://www.numbertheory.org/php/cornacchia.html

    See Also
    ========

    sympy.utilities.iterables.signed_permutations
    """
    # Assume gcd(a, b) = gcd(a, m) = 1 and a, b > 0 but no error checking
    sols = set()

    if a + b > m:
        # xy = 0 must hold if there exists a solution
        if a == 1:
            # y = 0
            s, _exact = iroot(m // a, 2)
            if _exact:
                sols.add((int(s), 0))
            if a == b:
                # only keep one solution
                return sols
        if m % b == 0:
            # x = 0
            s, _exact = iroot(m // b, 2)
            if _exact:
                sols.add((0, int(s)))
        return sols

    # the original cornacchia
    for t in sqrt_mod_iter(-b*invert(a, m), m):
        if t < m // 2:
            continue
        u, r = m, t
        while (m1 := m - a*r**2) <= 0:
            u, r = r, u % r
        m1, _r = divmod(m1, b)
        if _r:
            continue
        s, _exact = iroot(m1, 2)
        if _exact:
            if a == b and r < s:
                r, s = s, r
            sols.add((int(r), int(s)))
    return sols


def PQa(P_0, Q_0, D):
    r"""
    Returns useful information needed to solve the Pell equation.

    Explanation
    ===========

    There are six sequences of integers defined related to the continued
    fraction representation of `\\frac{P + \sqrt{D}}{Q}`, namely {`P_{i}`},
    {`Q_{i}`}, {`a_{i}`},{`A_{i}`}, {`B_{i}`}, {`G_{i}`}. ``PQa()`` Returns
    these values as a 6-tuple in the same order as mentioned above. Refer [1]_
    for more detailed information.

    Usage
    =====

    ``PQa(P_0, Q_0, D)``: ``P_0``, ``Q_0`` and ``D`` are integers corresponding
    to `P_{0}`, `Q_{0}` and `D` in the continued fraction
    `\\frac{P_{0} + \sqrt{D}}{Q_{0}}`.
    Also it's assumed that `P_{0}^2 == D mod(|Q_{0}|)` and `D` is square free.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import PQa
    >>> pqa = PQa(13, 4, 5) # (13 + sqrt(5))/4
    >>> next(pqa) # (P_0, Q_0, a_0, A_0, B_0, G_0)
    (13, 4, 3, 3, 1, -1)
    >>> next(pqa) # (P_1, Q_1, a_1, A_1, B_1, G_1)
    (-1, 1, 1, 4, 1, 3)

    References
    ==========

    .. [1] Solving the generalized Pell equation x^2 - Dy^2 = N, John P.
        Robertson, July 31, 2004, Pages 4 - 8. https://web.archive.org/web/20160323033128/http://www.jpr2718.org/pell.pdf
    """
    sqD = isqrt(D)
    A2 = B1 = 0
    A1 = B2 = 1
    G1 = Q_0
    G2 = -P_0
    P_i = P_0
    Q_i = Q_0

    while True:
        a_i = (P_i + sqD) // Q_i
        A1, A2 = a_i*A1 + A2, A1
        B1, B2 = a_i*B1 + B2, B1
        G1, G2 = a_i*G1 + G2, G1
        yield P_i, Q_i, a_i, A1, B1, G1

        P_i = a_i*Q_i - P_i
        Q_i = (D - P_i**2) // Q_i


def diop_bf_DN(D, N, t=symbols("t", integer=True)):
    r"""
    Uses brute force to solve the equation, `x^2 - Dy^2 = N`.

    Explanation
    ===========

    Mainly concerned with the generalized Pell equation which is the case when
    `D > 0, D` is not a perfect square. For more information on the case refer
    [1]_. Let `(t, u)` be the minimal positive solution of the equation
    `x^2 - Dy^2 = 1`. Then this method requires
    `\sqrt{\\frac{\mid N \mid (t \pm 1)}{2D}}` to be small.

    Usage
    =====

    ``diop_bf_DN(D, N, t)``: ``D`` and ``N`` are coefficients in
    `x^2 - Dy^2 = N` and ``t`` is the parameter to be used in the solutions.

    Details
    =======

    ``D`` and ``N`` correspond to D and N in the equation.
    ``t`` is the parameter to be used in the solutions.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import diop_bf_DN
    >>> diop_bf_DN(13, -4)
    [(3, 1), (-3, 1), (36, 10)]
    >>> diop_bf_DN(986, 1)
    [(49299, 1570)]

    See Also
    ========

    diop_DN()

    References
    ==========

    .. [1] Solving the generalized Pell equation x**2 - D*y**2 = N, John P.
        Robertson, July 31, 2004, Page 15. https://web.archive.org/web/20160323033128/http://www.jpr2718.org/pell.pdf
    """
    D = as_int(D)
    N = as_int(N)

    sol = []
    a = diop_DN(D, 1)
    u = a[0][0]

    if N == 0:
        if D < 0:
            return [(0, 0)]
        if D == 0:
            return [(0, t)]
        sD, _exact = integer_nthroot(D, 2)
        if _exact:
            return [(sD*t, t), (-sD*t, t)]
        return [(0, 0)]

    if abs(N) == 1:
        return diop_DN(D, N)

    if N > 1:
        L1 = 0
        L2 = integer_nthroot(int(N*(u - 1)/(2*D)), 2)[0] + 1
    else: # N < -1
        L1, _exact = integer_nthroot(-int(N/D), 2)
        if not _exact:
            L1 += 1
        L2 = integer_nthroot(-int(N*(u + 1)/(2*D)), 2)[0] + 1

    for y in range(L1, L2):
        try:
            x, _exact = integer_nthroot(N + D*y**2, 2)
        except ValueError:
            _exact = False
        if _exact:
            sol.append((x, y))
            if not equivalent(x, y, -x, y, D, N):
                sol.append((-x, y))

    return sol


def equivalent(u, v, r, s, D, N):
    """
    Returns True if two solutions `(u, v)` and `(r, s)` of `x^2 - Dy^2 = N`
    belongs to the same equivalence class and False otherwise.

    Explanation
    ===========

    Two solutions `(u, v)` and `(r, s)` to the above equation fall to the same
    equivalence class iff both `(ur - Dvs)` and `(us - vr)` are divisible by
    `N`. See reference [1]_. No test is performed to test whether `(u, v)` and
    `(r, s)` are actually solutions to the equation. User should take care of
    this.

    Usage
    =====

    ``equivalent(u, v, r, s, D, N)``: `(u, v)` and `(r, s)` are two solutions
    of the equation `x^2 - Dy^2 = N` and all parameters involved are integers.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import equivalent
    >>> equivalent(18, 5, -18, -5, 13, -1)
    True
    >>> equivalent(3, 1, -18, 393, 109, -4)
    False

    References
    ==========

    .. [1] Solving the generalized Pell equation x**2 - D*y**2 = N, John P.
        Robertson, July 31, 2004, Page 12. https://web.archive.org/web/20160323033128/http://www.jpr2718.org/pell.pdf

    """
    return divisible(u*r - D*v*s, N) and divisible(u*s - v*r, N)


def length(P, Q, D):
    r"""
    Returns the (length of aperiodic part + length of periodic part) of
    continued fraction representation of `\\frac{P + \sqrt{D}}{Q}`.

    It is important to remember that this does NOT return the length of the
    periodic part but the sum of the lengths of the two parts as mentioned
    above.

    Usage
    =====

    ``length(P, Q, D)``: ``P``, ``Q`` and ``D`` are integers corresponding to
    the continued fraction `\\frac{P + \sqrt{D}}{Q}`.

    Details
    =======

    ``P``, ``D`` and ``Q`` corresponds to P, D and Q in the continued fraction,
    `\\frac{P + \sqrt{D}}{Q}`.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import length
    >>> length(-2, 4, 5) # (-2 + sqrt(5))/4
    3
    >>> length(-5, 4, 17) # (-5 + sqrt(17))/4
    4

    See Also
    ========
    sympy.ntheory.continued_fraction.continued_fraction_periodic
    """
    from sympy.ntheory.continued_fraction import continued_fraction_periodic
    v = continued_fraction_periodic(P, Q, D)
    if isinstance(v[-1], list):
        rpt = len(v[-1])
        nonrpt = len(v) - 1
    else:
        rpt = 0
        nonrpt = len(v)
    return rpt + nonrpt


def transformation_to_DN(eq):
    """
    This function transforms general quadratic,
    `ax^2 + bxy + cy^2 + dx + ey + f = 0`
    to more easy to deal with `X^2 - DY^2 = N` form.

    Explanation
    ===========

    This is used to solve the general quadratic equation by transforming it to
    the latter form. Refer to [1]_ for more detailed information on the
    transformation. This function returns a tuple (A, B) where A is a 2 X 2
    matrix and B is a 2 X 1 matrix such that,

    Transpose([x y]) =  A * Transpose([X Y]) + B

    Usage
    =====

    ``transformation_to_DN(eq)``: where ``eq`` is the quadratic to be
    transformed.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.solvers.diophantine.diophantine import transformation_to_DN
    >>> A, B = transformation_to_DN(x**2 - 3*x*y - y**2 - 2*y + 1)
    >>> A
    Matrix([
    [1/26, 3/26],
    [   0, 1/13]])
    >>> B
    Matrix([
    [-6/13],
    [-4/13]])

    A, B  returned are such that Transpose((x y)) =  A * Transpose((X Y)) + B.
    Substituting these values for `x` and `y` and a bit of simplifying work
    will give an equation of the form `x^2 - Dy^2 = N`.

    >>> from sympy.abc import X, Y
    >>> from sympy import Matrix, simplify
    >>> u = (A*Matrix([X, Y]) + B)[0] # Transformation for x
    >>> u
    X/26 + 3*Y/26 - 6/13
    >>> v = (A*Matrix([X, Y]) + B)[1] # Transformation for y
    >>> v
    Y/13 - 4/13

    Next we will substitute these formulas for `x` and `y` and do
    ``simplify()``.

    >>> eq = simplify((x**2 - 3*x*y - y**2 - 2*y + 1).subs(zip((x, y), (u, v))))
    >>> eq
    X**2/676 - Y**2/52 + 17/13

    By multiplying the denominator appropriately, we can get a Pell equation
    in the standard form.

    >>> eq * 676
    X**2 - 13*Y**2 + 884

    If only the final equation is needed, ``find_DN()`` can be used.

    See Also
    ========

    find_DN()

    References
    ==========

    .. [1] Solving the equation ax^2 + bxy + cy^2 + dx + ey + f = 0,
           John P.Robertson, May 8, 2003, Page 7 - 11.
           https://web.archive.org/web/20160323033111/http://www.jpr2718.org/ax2p.pdf
    """

    var, coeff, diop_type = classify_diop(eq, _dict=False)
    if diop_type == BinaryQuadratic.name:
        return _transformation_to_DN(var, coeff)


def _transformation_to_DN(var, coeff):

    x, y = var

    a = coeff[x**2]
    b = coeff[x*y]
    c = coeff[y**2]
    d = coeff[x]
    e = coeff[y]
    f = coeff[1]

    a, b, c, d, e, f = [as_int(i) for i in _remove_gcd(a, b, c, d, e, f)]

    X, Y = symbols("X, Y", integer=True)

    if b:
        B, C = _rational_pq(2*a, b)
        A, T = _rational_pq(a, B**2)

        # eq_1 = A*B*X**2 + B*(c*T - A*C**2)*Y**2 + d*T*X + (B*e*T - d*T*C)*Y + f*T*B
        coeff = {X**2: A*B, X*Y: 0, Y**2: B*(c*T - A*C**2), X: d*T, Y: B*e*T - d*T*C, 1: f*T*B}
        A_0, B_0 = _transformation_to_DN([X, Y], coeff)
        return Matrix(2, 2, [S.One/B, -S(C)/B, 0, 1])*A_0, Matrix(2, 2, [S.One/B, -S(C)/B, 0, 1])*B_0

    if d:
        B, C = _rational_pq(2*a, d)
        A, T = _rational_pq(a, B**2)

        # eq_2 = A*X**2 + c*T*Y**2 + e*T*Y + f*T - A*C**2
        coeff = {X**2: A, X*Y: 0, Y**2: c*T, X: 0, Y: e*T, 1: f*T - A*C**2}
        A_0, B_0 = _transformation_to_DN([X, Y], coeff)
        return Matrix(2, 2, [S.One/B, 0, 0, 1])*A_0, Matrix(2, 2, [S.One/B, 0, 0, 1])*B_0 + Matrix([-S(C)/B, 0])

    if e:
        B, C = _rational_pq(2*c, e)
        A, T = _rational_pq(c, B**2)

        # eq_3 = a*T*X**2 + A*Y**2 + f*T - A*C**2
        coeff = {X**2: a*T, X*Y: 0, Y**2: A, X: 0, Y: 0, 1: f*T - A*C**2}
        A_0, B_0 = _transformation_to_DN([X, Y], coeff)
        return Matrix(2, 2, [1, 0, 0, S.One/B])*A_0, Matrix(2, 2, [1, 0, 0, S.One/B])*B_0 + Matrix([0, -S(C)/B])

    # TODO: pre-simplification: Not necessary but may simplify
    # the equation.
    return Matrix(2, 2, [S.One/a, 0, 0, 1]), Matrix([0, 0])


def find_DN(eq):
    """
    This function returns a tuple, `(D, N)` of the simplified form,
    `x^2 - Dy^2 = N`, corresponding to the general quadratic,
    `ax^2 + bxy + cy^2 + dx + ey + f = 0`.

    Solving the general quadratic is then equivalent to solving the equation
    `X^2 - DY^2 = N` and transforming the solutions by using the transformation
    matrices returned by ``transformation_to_DN()``.

    Usage
    =====

    ``find_DN(eq)``: where ``eq`` is the quadratic to be transformed.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.solvers.diophantine.diophantine import find_DN
    >>> find_DN(x**2 - 3*x*y - y**2 - 2*y + 1)
    (13, -884)

    Interpretation of the output is that we get `X^2 -13Y^2 = -884` after
    transforming `x^2 - 3xy - y^2 - 2y + 1` using the transformation returned
    by ``transformation_to_DN()``.

    See Also
    ========

    transformation_to_DN()

    References
    ==========

    .. [1] Solving the equation ax^2 + bxy + cy^2 + dx + ey + f = 0,
           John P.Robertson, May 8, 2003, Page 7 - 11.
           https://web.archive.org/web/20160323033111/http://www.jpr2718.org/ax2p.pdf
    """
    var, coeff, diop_type = classify_diop(eq, _dict=False)
    if diop_type == BinaryQuadratic.name:
        return _find_DN(var, coeff)


def _find_DN(var, coeff):

    x, y = var
    X, Y = symbols("X, Y", integer=True)
    A, B = _transformation_to_DN(var, coeff)

    u = (A*Matrix([X, Y]) + B)[0]
    v = (A*Matrix([X, Y]) + B)[1]
    eq = x**2*coeff[x**2] + x*y*coeff[x*y] + y**2*coeff[y**2] + x*coeff[x] + y*coeff[y] + coeff[1]

    simplified = _mexpand(eq.subs(zip((x, y), (u, v))))

    coeff = simplified.as_coefficients_dict()

    return -coeff[Y**2]/coeff[X**2], -coeff[1]/coeff[X**2]


def check_param(x, y, a, params):
    """
    If there is a number modulo ``a`` such that ``x`` and ``y`` are both
    integers, then return a parametric representation for ``x`` and ``y``
    else return (None, None).

    Here ``x`` and ``y`` are functions of ``t``.
    """
    from sympy.simplify.simplify import clear_coefficients

    if x.is_number and not x.is_Integer:
        return DiophantineSolutionSet([x, y], parameters=params)

    if y.is_number and not y.is_Integer:
        return DiophantineSolutionSet([x, y], parameters=params)

    m, n = symbols("m, n", integer=True)
    c, p = (m*x + n*y).as_content_primitive()
    if a % c.q:
        return DiophantineSolutionSet([x, y], parameters=params)

    # clear_coefficients(mx + b, R)[1] -> (R - b)/m
    eq = clear_coefficients(x, m)[1] - clear_coefficients(y, n)[1]
    junk, eq = eq.as_content_primitive()

    return _diop_solve(eq, params=params)


def diop_ternary_quadratic(eq, parameterize=False):
    """
    Solves the general quadratic ternary form,
    `ax^2 + by^2 + cz^2 + fxy + gyz + hxz = 0`.

    Returns a tuple `(x, y, z)` which is a base solution for the above
    equation. If there are no solutions, `(None, None, None)` is returned.

    Usage
    =====

    ``diop_ternary_quadratic(eq)``: Return a tuple containing a basic solution
    to ``eq``.

    Details
    =======

    ``eq`` should be an homogeneous expression of degree two in three variables
    and it is assumed to be zero.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.solvers.diophantine.diophantine import diop_ternary_quadratic
    >>> diop_ternary_quadratic(x**2 + 3*y**2 - z**2)
    (1, 0, 1)
    >>> diop_ternary_quadratic(4*x**2 + 5*y**2 - z**2)
    (1, 0, 2)
    >>> diop_ternary_quadratic(45*x**2 - 7*y**2 - 8*x*y - z**2)
    (28, 45, 105)
    >>> diop_ternary_quadratic(x**2 - 49*y**2 - z**2 + 13*z*y -8*x*y)
    (9, 1, 5)
    """
    var, coeff, diop_type = classify_diop(eq, _dict=False)

    if diop_type in (
            HomogeneousTernaryQuadratic.name,
            HomogeneousTernaryQuadraticNormal.name):
        sol = _diop_ternary_quadratic(var, coeff)
        if len(sol) > 0:
            x_0, y_0, z_0 = list(sol)[0]
        else:
            x_0, y_0, z_0 = None, None, None

        if parameterize:
            return _parametrize_ternary_quadratic(
                (x_0, y_0, z_0), var, coeff)
        return x_0, y_0, z_0


def _diop_ternary_quadratic(_var, coeff):
    eq = sum(i*coeff[i] for i in coeff)
    if HomogeneousTernaryQuadratic(eq).matches():
        return HomogeneousTernaryQuadratic(eq, free_symbols=_var).solve()
    elif HomogeneousTernaryQuadraticNormal(eq).matches():
        return HomogeneousTernaryQuadraticNormal(eq, free_symbols=_var).solve()


def transformation_to_normal(eq):
    """
    Returns the transformation Matrix that converts a general ternary
    quadratic equation ``eq`` (`ax^2 + by^2 + cz^2 + dxy + eyz + fxz`)
    to a form without cross terms: `ax^2 + by^2 + cz^2 = 0`. This is
    not used in solving ternary quadratics; it is only implemented for
    the sake of completeness.
    """
    var, coeff, diop_type = classify_diop(eq, _dict=False)

    if diop_type in (
            "homogeneous_ternary_quadratic",
            "homogeneous_ternary_quadratic_normal"):
        return _transformation_to_normal(var, coeff)


def _transformation_to_normal(var, coeff):

    _var = list(var)  # copy
    x, y, z = var

    if not any(coeff[i**2] for i in var):
        # https://math.stackexchange.com/questions/448051/transform-quadratic-ternary-form-to-normal-form/448065#448065
        a = coeff[x*y]
        b = coeff[y*z]
        c = coeff[x*z]
        swap = False
        if not a:  # b can't be 0 or else there aren't 3 vars
            swap = True
            a, b = b, a
        T = Matrix(((1, 1, -b/a), (1, -1, -c/a), (0, 0, 1)))
        if swap:
            T.row_swap(0, 1)
            T.col_swap(0, 1)
        return T

    if coeff[x**2] == 0:
        # If the coefficient of x is zero change the variables
        if coeff[y**2] == 0:
            _var[0], _var[2] = var[2], var[0]
            T = _transformation_to_normal(_var, coeff)
            T.row_swap(0, 2)
            T.col_swap(0, 2)
            return T

        _var[0], _var[1] = var[1], var[0]
        T = _transformation_to_normal(_var, coeff)
        T.row_swap(0, 1)
        T.col_swap(0, 1)
        return T

    # Apply the transformation x --> X - (B*Y + C*Z)/(2*A)
    if coeff[x*y] != 0 or coeff[x*z] != 0:
        A = coeff[x**2]
        B = coeff[x*y]
        C = coeff[x*z]
        D = coeff[y**2]
        E = coeff[y*z]
        F = coeff[z**2]

        _coeff = {}

        _coeff[x**2] = 4*A**2
        _coeff[y**2] = 4*A*D - B**2
        _coeff[z**2] = 4*A*F - C**2
        _coeff[y*z] = 4*A*E - 2*B*C
        _coeff[x*y] = 0
        _coeff[x*z] = 0

        T_0 = _transformation_to_normal(_var, _coeff)
        return Matrix(3, 3, [1, S(-B)/(2*A), S(-C)/(2*A), 0, 1, 0, 0, 0, 1])*T_0

    elif coeff[y*z] != 0:
        if coeff[y**2] == 0:
            if coeff[z**2] == 0:
                # Equations of the form A*x**2 + E*yz = 0.
                # Apply transformation y -> Y + Z ans z -> Y - Z
                return Matrix(3, 3, [1, 0, 0, 0, 1, 1, 0, 1, -1])

            # Ax**2 + E*y*z + F*z**2  = 0
            _var[0], _var[2] = var[2], var[0]
            T = _transformation_to_normal(_var, coeff)
            T.row_swap(0, 2)
            T.col_swap(0, 2)
            return T

        # A*x**2 + D*y**2 + E*y*z + F*z**2 = 0, F may be zero
        _var[0], _var[1] = var[1], var[0]
        T = _transformation_to_normal(_var, coeff)
        T.row_swap(0, 1)
        T.col_swap(0, 1)
        return T

    return Matrix.eye(3)


def parametrize_ternary_quadratic(eq):
    """
    Returns the parametrized general solution for the ternary quadratic
    equation ``eq`` which has the form
    `ax^2 + by^2 + cz^2 + fxy + gyz + hxz = 0`.

    Examples
    ========

    >>> from sympy import Tuple, ordered
    >>> from sympy.abc import x, y, z
    >>> from sympy.solvers.diophantine.diophantine import parametrize_ternary_quadratic

    The parametrized solution may be returned with three parameters:

    >>> parametrize_ternary_quadratic(2*x**2 + y**2 - 2*z**2)
    (p**2 - 2*q**2, -2*p**2 + 4*p*q - 4*p*r - 4*q**2, p**2 - 4*p*q + 2*q**2 - 4*q*r)

    There might also be only two parameters:

    >>> parametrize_ternary_quadratic(4*x**2 + 2*y**2 - 3*z**2)
    (2*p**2 - 3*q**2, -4*p**2 + 12*p*q - 6*q**2, 4*p**2 - 8*p*q + 6*q**2)

    Notes
    =====

    Consider ``p`` and ``q`` in the previous 2-parameter
    solution and observe that more than one solution can be represented
    by a given pair of parameters. If `p` and ``q`` are not coprime, this is
    trivially true since the common factor will also be a common factor of the
    solution values. But it may also be true even when ``p`` and
    ``q`` are coprime:

    >>> sol = Tuple(*_)
    >>> p, q = ordered(sol.free_symbols)
    >>> sol.subs([(p, 3), (q, 2)])
    (6, 12, 12)
    >>> sol.subs([(q, 1), (p, 1)])
    (-1, 2, 2)
    >>> sol.subs([(q, 0), (p, 1)])
    (2, -4, 4)
    >>> sol.subs([(q, 1), (p, 0)])
    (-3, -6, 6)

    Except for sign and a common factor, these are equivalent to
    the solution of (1, 2, 2).

    References
    ==========

    .. [1] The algorithmic resolution of Diophantine equations, Nigel P. Smart,
           London Mathematical Society Student Texts 41, Cambridge University
           Press, Cambridge, 1998.

    """
    var, coeff, diop_type = classify_diop(eq, _dict=False)

    if diop_type in (
            "homogeneous_ternary_quadratic",
            "homogeneous_ternary_quadratic_normal"):
        x_0, y_0, z_0 = list(_diop_ternary_quadratic(var, coeff))[0]
        return _parametrize_ternary_quadratic(
            (x_0, y_0, z_0), var, coeff)


def _parametrize_ternary_quadratic(solution, _var, coeff):
    # called for a*x**2 + b*y**2 + c*z**2 + d*x*y + e*y*z + f*x*z = 0
    assert 1 not in coeff

    x_0, y_0, z_0 = solution

    v = list(_var)  # copy

    if x_0 is None:
        return (None, None, None)

    if solution.count(0) >= 2:
        # if there are 2 zeros the equation reduces
        # to k*X**2 == 0 where X is x, y, or z so X must
        # be zero, too. So there is only the trivial
        # solution.
        return (None, None, None)

    if x_0 == 0:
        v[0], v[1] = v[1], v[0]
        y_p, x_p, z_p = _parametrize_ternary_quadratic(
            (y_0, x_0, z_0), v, coeff)
        return x_p, y_p, z_p

    x, y, z = v
    r, p, q = symbols("r, p, q", integer=True)

    eq = sum(k*v for k, v in coeff.items())
    eq_1 = _mexpand(eq.subs(zip(
        (x, y, z), (r*x_0, r*y_0 + p, r*z_0 + q))))
    A, B = eq_1.as_independent(r, as_Add=True)


    x = A*x_0
    y = (A*y_0 - _mexpand(B/r*p))
    z = (A*z_0 - _mexpand(B/r*q))

    return _remove_gcd(x, y, z)


def diop_ternary_quadratic_normal(eq, parameterize=False):
    """
    Solves the quadratic ternary diophantine equation,
    `ax^2 + by^2 + cz^2 = 0`.

    Explanation
    ===========

    Here the coefficients `a`, `b`, and `c` should be non zero. Otherwise the
    equation will be a quadratic binary or univariate equation. If solvable,
    returns a tuple `(x, y, z)` that satisfies the given equation. If the
    equation does not have integer solutions, `(None, None, None)` is returned.

    Usage
    =====

    ``diop_ternary_quadratic_normal(eq)``: where ``eq`` is an equation of the form
    `ax^2 + by^2 + cz^2 = 0`.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.solvers.diophantine.diophantine import diop_ternary_quadratic_normal
    >>> diop_ternary_quadratic_normal(x**2 + 3*y**2 - z**2)
    (1, 0, 1)
    >>> diop_ternary_quadratic_normal(4*x**2 + 5*y**2 - z**2)
    (1, 0, 2)
    >>> diop_ternary_quadratic_normal(34*x**2 - 3*y**2 - 301*z**2)
    (4, 9, 1)
    """
    var, coeff, diop_type = classify_diop(eq, _dict=False)
    if diop_type == HomogeneousTernaryQuadraticNormal.name:
        sol = _diop_ternary_quadratic_normal(var, coeff)
        if len(sol) > 0:
            x_0, y_0, z_0 = list(sol)[0]
        else:
            x_0, y_0, z_0 = None, None, None
        if parameterize:
            return _parametrize_ternary_quadratic(
                (x_0, y_0, z_0), var, coeff)
        return x_0, y_0, z_0


def _diop_ternary_quadratic_normal(var, coeff):
    eq = sum(i * coeff[i] for i in coeff)
    return HomogeneousTernaryQuadraticNormal(eq, free_symbols=var).solve()


def sqf_normal(a, b, c, steps=False):
    """
    Return `a', b', c'`, the coefficients of the square-free normal
    form of `ax^2 + by^2 + cz^2 = 0`, where `a', b', c'` are pairwise
    prime.  If `steps` is True then also return three tuples:
    `sq`, `sqf`, and `(a', b', c')` where `sq` contains the square
    factors of `a`, `b` and `c` after removing the `gcd(a, b, c)`;
    `sqf` contains the values of `a`, `b` and `c` after removing
    both the `gcd(a, b, c)` and the square factors.

    The solutions for `ax^2 + by^2 + cz^2 = 0` can be
    recovered from the solutions of `a'x^2 + b'y^2 + c'z^2 = 0`.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import sqf_normal
    >>> sqf_normal(2 * 3**2 * 5, 2 * 5 * 11, 2 * 7**2 * 11)
    (11, 1, 5)
    >>> sqf_normal(2 * 3**2 * 5, 2 * 5 * 11, 2 * 7**2 * 11, True)
    ((3, 1, 7), (5, 55, 11), (11, 1, 5))

    References
    ==========

    .. [1] Legendre's Theorem, Legrange's Descent,
           https://public.csusm.edu/aitken_html/notes/legendre.pdf


    See Also
    ========

    reconstruct()
    """
    ABC = _remove_gcd(a, b, c)
    sq = tuple(square_factor(i) for i in ABC)
    sqf = A, B, C = tuple([i//j**2 for i,j in zip(ABC, sq)])
    pc = igcd(A, B)
    A /= pc
    B /= pc
    pa = igcd(B, C)
    B /= pa
    C /= pa
    pb = igcd(A, C)
    A /= pb
    B /= pb

    A *= pa
    B *= pb
    C *= pc

    if steps:
        return (sq, sqf, (A, B, C))
    else:
        return A, B, C


def square_factor(a):
    r"""
    Returns an integer `c` s.t. `a = c^2k, \ c,k \in Z`. Here `k` is square
    free. `a` can be given as an integer or a dictionary of factors.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import square_factor
    >>> square_factor(24)
    2
    >>> square_factor(-36*3)
    6
    >>> square_factor(1)
    1
    >>> square_factor({3: 2, 2: 1, -1: 1})  # -18
    3

    See Also
    ========
    sympy.ntheory.factor_.core
    """
    f = a if isinstance(a, dict) else factorint(a)
    return Mul(*[p**(e//2) for p, e in f.items()])


def reconstruct(A, B, z):
    """
    Reconstruct the `z` value of an equivalent solution of `ax^2 + by^2 + cz^2`
    from the `z` value of a solution of the square-free normal form of the
    equation, `a'*x^2 + b'*y^2 + c'*z^2`, where `a'`, `b'` and `c'` are square
    free and `gcd(a', b', c') == 1`.
    """
    f = factorint(igcd(A, B))
    for p, e in f.items():
        if e != 1:
            raise ValueError('a and b should be square-free')
        z *= p
    return z


def ldescent(A, B):
    """
    Return a non-trivial solution to `w^2 = Ax^2 + By^2` using
    Lagrange's method; return None if there is no such solution.

    Parameters
    ==========

    A : Integer
    B : Integer
        non-zero integer

    Returns
    =======

    (int, int, int) | None : a tuple `(w_0, x_0, y_0)` which is a solution to the above equation.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import ldescent
    >>> ldescent(1, 1) # w^2 = x^2 + y^2
    (1, 1, 0)
    >>> ldescent(4, -7) # w^2 = 4x^2 - 7y^2
    (2, -1, 0)

    This means that `x = -1, y = 0` and `w = 2` is a solution to the equation
    `w^2 = 4x^2 - 7y^2`

    >>> ldescent(5, -1) # w^2 = 5x^2 - y^2
    (2, 1, -1)

    References
    ==========

    .. [1] The algorithmic resolution of Diophantine equations, Nigel P. Smart,
           London Mathematical Society Student Texts 41, Cambridge University
           Press, Cambridge, 1998.
    .. [2] Cremona, J. E., Rusin, D. (2003). Efficient Solution of Rational Conics.
           Mathematics of Computation, 72(243), 1417-1441.
           https://doi.org/10.1090/S0025-5718-02-01480-1
    """
    if A == 0 or B == 0:
        raise ValueError("A and B must be non-zero integers")
    if abs(A) > abs(B):
        w, y, x = ldescent(B, A)
        return w, x, y
    if A == 1:
        return (1, 1, 0)
    if B == 1:
        return (1, 0, 1)
    if B == -1:  # and A == -1
        return

    r = sqrt_mod(A, B)
    if r is None:
        return
    Q = (r**2 - A) // B
    if Q == 0:
        return r, -1, 0
    for i in divisors(Q):
        d, _exact = integer_nthroot(abs(Q) // i, 2)
        if _exact:
            B_0 = sign(Q)*i
            W, X, Y = ldescent(A, B_0)
            return _remove_gcd(-A*X + r*W, r*X - W, Y*B_0*d)


def descent(A, B):
    """
    Returns a non-trivial solution, (x, y, z), to `x^2 = Ay^2 + Bz^2`
    using Lagrange's descent method with lattice-reduction. `A` and `B`
    are assumed to be valid for such a solution to exist.

    This is faster than the normal Lagrange's descent algorithm because
    the Gaussian reduction is used.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import descent
    >>> descent(3, 1) # x**2 = 3*y**2 + z**2
    (1, 0, 1)

    `(x, y, z) = (1, 0, 1)` is a solution to the above equation.

    >>> descent(41, -113)
    (-16, -3, 1)

    References
    ==========

    .. [1] Cremona, J. E., Rusin, D. (2003). Efficient Solution of Rational Conics.
           Mathematics of Computation, 72(243), 1417-1441.
           https://doi.org/10.1090/S0025-5718-02-01480-1
    """
    if abs(A) > abs(B):
        x, y, z = descent(B, A)
        return x, z, y

    if B == 1:
        return (1, 0, 1)
    if A == 1:
        return (1, 1, 0)
    if B == -A:
        return (0, 1, 1)
    if B == A:
        x, z, y = descent(-1, A)
        return (A*y, z, x)

    w = sqrt_mod(A, B)
    x_0, z_0 = gaussian_reduce(w, A, B)

    t = (x_0**2 - A*z_0**2) // B
    t_2 = square_factor(t)
    t_1 = t // t_2**2

    x_1, z_1, y_1 = descent(A, t_1)

    return _remove_gcd(x_0*x_1 + A*z_0*z_1, z_0*x_1 + x_0*z_1, t_1*t_2*y_1)


def gaussian_reduce(w:int, a:int, b:int) -> tuple[int, int]:
    r"""
    Returns a reduced solution `(x, z)` to the congruence
    `X^2 - aZ^2 \equiv 0 \pmod{b}` so that `x^2 + |a|z^2` is as small as possible.
    Here ``w`` is a solution of the congruence `x^2 \equiv a \pmod{b}`.

    This function is intended to be used only for ``descent()``.

    Explanation
    ===========

    The Gaussian reduction can find the shortest vector for any norm.
    So we define the special norm for the vectors `u = (u_1, u_2)` and `v = (v_1, v_2)` as follows.

    .. math ::
        u \cdot v := (wu_1 + bu_2)(wv_1 + bv_2) + |a|u_1v_1

    Note that, given the mapping `f: (u_1, u_2) \to (wu_1 + bu_2, u_1)`,
    `f((u_1,u_2))` is the solution to `X^2 - aZ^2 \equiv 0 \pmod{b}`.
    In other words, finding the shortest vector in this norm will yield a solution with smaller `X^2 + |a|Z^2`.
    The algorithm starts from basis vectors `(0, 1)` and `(1, 0)`
    (corresponding to solutions `(b, 0)` and `(w, 1)`, respectively) and finds the shortest vector.
    The shortest vector does not necessarily correspond to the smallest solution,
    but since ``descent()`` only wants the smallest possible solution, it is sufficient.

    Parameters
    ==========

    w : int
        ``w`` s.t. `w^2 \equiv a \pmod{b}`
    a : int
        square-free nonzero integer
    b : int
        square-free nonzero integer

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import gaussian_reduce
    >>> from sympy.ntheory.residue_ntheory import sqrt_mod
    >>> a, b = 19, 101
    >>> gaussian_reduce(sqrt_mod(a, b), a, b) # 1**2 - 19*(-4)**2 = -303
    (1, -4)
    >>> a, b = 11, 14
    >>> x, z = gaussian_reduce(sqrt_mod(a, b), a, b)
    >>> (x**2 - a*z**2) % b == 0
    True

    It does not always return the smallest solution.

    >>> a, b = 6, 95
    >>> min_x, min_z = 1, 4
    >>> x, z = gaussian_reduce(sqrt_mod(a, b), a, b)
    >>> (x**2 - a*z**2) % b == 0 and (min_x**2 - a*min_z**2) % b == 0
    True
    >>> min_x**2 + abs(a)*min_z**2 < x**2 + abs(a)*z**2
    True

    References
    ==========

    .. [1] Gaussian lattice Reduction [online]. Available:
           https://web.archive.org/web/20201021115213/http://home.ie.cuhk.edu.hk/~wkshum/wordpress/?p=404
    .. [2] Cremona, J. E., Rusin, D. (2003). Efficient Solution of Rational Conics.
           Mathematics of Computation, 72(243), 1417-1441.
           https://doi.org/10.1090/S0025-5718-02-01480-1
    """
    a = abs(a)
    def _dot(u, v):
        return u[0]*v[0] + a*u[1]*v[1]

    u = (b, 0)
    v = (w, 1) if b*w >= 0 else (-w, -1)
    # i.e., _dot(u, v) >= 0

    if b**2 < w**2 + a:
        u, v = v, u
    # i.e., norm(u) >= norm(v), where norm(u) := sqrt(_dot(u, u))

    while _dot(u, u) > (dv := _dot(v, v)):
        k = _dot(u, v) // dv
        u, v = v, (u[0] - k*v[0], u[1] - k*v[1])
    c = (v[0] - u[0], v[1] - u[1])
    if _dot(c, c) <= _dot(u, u) <= 2*_dot(u, v):
        return c
    return u


def holzer(x, y, z, a, b, c):
    r"""
    Simplify the solution `(x, y, z)` of the equation
    `ax^2 + by^2 = cz^2` with `a, b, c > 0` and `z^2 \geq \mid ab \mid` to
    a new reduced solution `(x', y', z')` such that `z'^2 \leq \mid ab \mid`.

    The algorithm is an interpretation of Mordell's reduction as described
    on page 8 of Cremona and Rusin's paper [1]_ and the work of Mordell in
    reference [2]_.

    References
    ==========

    .. [1] Cremona, J. E., Rusin, D. (2003). Efficient Solution of Rational Conics.
           Mathematics of Computation, 72(243), 1417-1441.
           https://doi.org/10.1090/S0025-5718-02-01480-1
    .. [2] Diophantine Equations, L. J. Mordell, page 48.

    """

    if _odd(c):
        k = 2*c
    else:
        k = c//2

    small = a*b*c
    step = 0
    while True:
        t1, t2, t3 = a*x**2, b*y**2, c*z**2
        # check that it's a solution
        if t1 + t2 != t3:
            if step == 0:
                raise ValueError('bad starting solution')
            break
        x_0, y_0, z_0 = x, y, z
        if max(t1, t2, t3) <= small:
            # Holzer condition
            break

        uv = u, v = base_solution_linear(k, y_0, -x_0)
        if None in uv:
            break

        p, q = -(a*u*x_0 + b*v*y_0), c*z_0
        r = Rational(p, q)
        if _even(c):
            w = _nint_or_floor(p, q)
            assert abs(w - r) <= S.Half
        else:
            w = p//q  # floor
            if _odd(a*u + b*v + c*w):
                w += 1
            assert abs(w - r) <= S.One

        A = (a*u**2 + b*v**2 + c*w**2)
        B = (a*u*x_0 + b*v*y_0 + c*w*z_0)
        x = Rational(x_0*A - 2*u*B, k)
        y = Rational(y_0*A - 2*v*B, k)
        z = Rational(z_0*A - 2*w*B, k)
        assert all(i.is_Integer for i in (x, y, z))
        step += 1

    return tuple([int(i) for i in (x_0, y_0, z_0)])


def diop_general_pythagorean(eq, param=symbols("m", integer=True)):
    """
    Solves the general pythagorean equation,
    `a_{1}^2x_{1}^2 + a_{2}^2x_{2}^2 + . . . + a_{n}^2x_{n}^2 - a_{n + 1}^2x_{n + 1}^2 = 0`.

    Returns a tuple which contains a parametrized solution to the equation,
    sorted in the same order as the input variables.

    Usage
    =====

    ``diop_general_pythagorean(eq, param)``: where ``eq`` is a general
    pythagorean equation which is assumed to be zero and ``param`` is the base
    parameter used to construct other parameters by subscripting.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import diop_general_pythagorean
    >>> from sympy.abc import a, b, c, d, e
    >>> diop_general_pythagorean(a**2 + b**2 + c**2 - d**2)
    (m1**2 + m2**2 - m3**2, 2*m1*m3, 2*m2*m3, m1**2 + m2**2 + m3**2)
    >>> diop_general_pythagorean(9*a**2 - 4*b**2 + 16*c**2 + 25*d**2 + e**2)
    (10*m1**2  + 10*m2**2  + 10*m3**2 - 10*m4**2, 15*m1**2  + 15*m2**2  + 15*m3**2  + 15*m4**2, 15*m1*m4, 12*m2*m4, 60*m3*m4)
    """
    var, coeff, diop_type  = classify_diop(eq, _dict=False)

    if diop_type == GeneralPythagorean.name:
        if param is None:
            params = None
        else:
            params = symbols('%s1:%i' % (param, len(var)), integer=True)
        return list(GeneralPythagorean(eq).solve(parameters=params))[0]


def diop_general_sum_of_squares(eq, limit=1):
    r"""
    Solves the equation `x_{1}^2 + x_{2}^2 + . . . + x_{n}^2 - k = 0`.

    Returns at most ``limit`` number of solutions.

    Usage
    =====

    ``general_sum_of_squares(eq, limit)`` : Here ``eq`` is an expression which
    is assumed to be zero. Also, ``eq`` should be in the form,
    `x_{1}^2 + x_{2}^2 + . . . + x_{n}^2 - k = 0`.

    Details
    =======

    When `n = 3` if `k = 4^a(8m + 7)` for some `a, m \in Z` then there will be
    no solutions. Refer to [1]_ for more details.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import diop_general_sum_of_squares
    >>> from sympy.abc import a, b, c, d, e
    >>> diop_general_sum_of_squares(a**2 + b**2 + c**2 + d**2 + e**2 - 2345)
    {(15, 22, 22, 24, 24)}

    Reference
    =========

    .. [1] Representing an integer as a sum of three squares, [online],
        Available:
        https://proofwiki.org/wiki/Integer_as_Sum_of_Three_Squares
    """
    var, coeff, diop_type = classify_diop(eq, _dict=False)

    if diop_type == GeneralSumOfSquares.name:
        return set(GeneralSumOfSquares(eq).solve(limit=limit))


def diop_general_sum_of_even_powers(eq, limit=1):
    """
    Solves the equation `x_{1}^e + x_{2}^e + . . . + x_{n}^e - k = 0`
    where `e` is an even, integer power.

    Returns at most ``limit`` number of solutions.

    Usage
    =====

    ``general_sum_of_even_powers(eq, limit)`` : Here ``eq`` is an expression which
    is assumed to be zero. Also, ``eq`` should be in the form,
    `x_{1}^e + x_{2}^e + . . . + x_{n}^e - k = 0`.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import diop_general_sum_of_even_powers
    >>> from sympy.abc import a, b
    >>> diop_general_sum_of_even_powers(a**4 + b**4 - (2**4 + 3**4))
    {(2, 3)}

    See Also
    ========

    power_representation
    """
    var, coeff, diop_type = classify_diop(eq, _dict=False)

    if diop_type == GeneralSumOfEvenPowers.name:
        return set(GeneralSumOfEvenPowers(eq).solve(limit=limit))


## Functions below this comment can be more suitably grouped under
## an Additive number theory module rather than the Diophantine
## equation module.


def partition(n, k=None, zeros=False):
    """
    Returns a generator that can be used to generate partitions of an integer
    `n`.

    Explanation
    ===========

    A partition of `n` is a set of positive integers which add up to `n`. For
    example, partitions of 3 are 3, 1 + 2, 1 + 1 + 1. A partition is returned
    as a tuple. If ``k`` equals None, then all possible partitions are returned
    irrespective of their size, otherwise only the partitions of size ``k`` are
    returned. If the ``zero`` parameter is set to True then a suitable
    number of zeros are added at the end of every partition of size less than
    ``k``.

    ``zero`` parameter is considered only if ``k`` is not None. When the
    partitions are over, the last `next()` call throws the ``StopIteration``
    exception, so this function should always be used inside a try - except
    block.

    Details
    =======

    ``partition(n, k)``: Here ``n`` is a positive integer and ``k`` is the size
    of the partition which is also positive integer.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import partition
    >>> f = partition(5)
    >>> next(f)
    (1, 1, 1, 1, 1)
    >>> next(f)
    (1, 1, 1, 2)
    >>> g = partition(5, 3)
    >>> next(g)
    (1, 1, 3)
    >>> next(g)
    (1, 2, 2)
    >>> g = partition(5, 3, zeros=True)
    >>> next(g)
    (0, 0, 5)

    """
    if not zeros or k is None:
        for i in ordered_partitions(n, k):
            yield tuple(i)
    else:
        for m in range(1, k + 1):
            for i in ordered_partitions(n, m):
                i = tuple(i)
                yield (0,)*(k - len(i)) + i


def prime_as_sum_of_two_squares(p):
    """
    Represent a prime `p` as a unique sum of two squares; this can
    only be done if the prime is congruent to 1 mod 4.

    Parameters
    ==========

    p : Integer
        A prime that is congruent to 1 mod 4

    Returns
    =======

    (int, int) | None : Pair of positive integers ``(x, y)`` satisfying ``x**2 + y**2 = p``.
                        None if ``p`` is not congruent to 1 mod 4.

    Raises
    ======

    ValueError
        If ``p`` is not prime number

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import prime_as_sum_of_two_squares
    >>> prime_as_sum_of_two_squares(7)  # can't be done
    >>> prime_as_sum_of_two_squares(5)
    (1, 2)

    Reference
    =========

    .. [1] Representing a number as a sum of four squares, [online],
           Available: https://schorn.ch/lagrange.html

    See Also
    ========

    sum_of_squares

    """
    p = as_int(p)
    if p % 4 != 1:
        return
    if not isprime(p):
        raise ValueError("p should be a prime number")

    if p % 8 == 5:
        # Legendre symbol (2/p) == -1 if p % 8 in [3, 5]
        b = 2
    elif p % 12 == 5:
        # Legendre symbol (3/p) == -1 if p % 12 in [5, 7]
        b = 3
    elif p % 5 in [2, 3]:
        # Legendre symbol (5/p) == -1 if p % 5 in [2, 3]
        b = 5
    else:
        b = 7
        while jacobi(b, p) == 1:
            b = nextprime(b)

    b = pow(b, p >> 2, p)
    a = p
    while b**2 > p:
        a, b = b, a % b
    return (int(a % b), int(b))  # convert from long


def sum_of_three_squares(n):
    r"""
    Returns a 3-tuple $(a, b, c)$ such that $a^2 + b^2 + c^2 = n$ and
    $a, b, c \geq 0$.

    Returns None if $n = 4^a(8m + 7)$ for some `a, m \in \mathbb{Z}`. See
    [1]_ for more details.

    Parameters
    ==========

    n : Integer
        non-negative integer

    Returns
    =======

    (int, int, int) | None : 3-tuple non-negative integers ``(a, b, c)`` satisfying ``a**2 + b**2 + c**2 = n``.
                             a,b,c are sorted in ascending order. ``None`` if no such ``(a,b,c)``.

    Raises
    ======

    ValueError
        If ``n`` is a negative integer

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import sum_of_three_squares
    >>> sum_of_three_squares(44542)
    (18, 37, 207)

    References
    ==========

    .. [1] Representing a number as a sum of three squares, [online],
        Available: https://schorn.ch/lagrange.html

    See Also
    ========

    power_representation :
        ``sum_of_three_squares(n)`` is one of the solutions output by ``power_representation(n, 2, 3, zeros=True)``

    """
    # https://math.stackexchange.com/questions/483101/rabin-and-shallit-algorithm/651425#651425
    # discusses these numbers (except for 1, 2, 3) as the exceptions of H&L's conjecture that
    # Every sufficiently large number n is either a square or the sum of a prime and a square.
    special = {1: (0, 0, 1), 2: (0, 1, 1), 3: (1, 1, 1), 10: (0, 1, 3), 34: (3, 3, 4),
               58: (0, 3, 7), 85: (0, 6, 7), 130: (0, 3, 11), 214: (3, 6, 13), 226: (8, 9, 9),
               370: (8, 9, 15), 526: (6, 7, 21), 706: (15, 15, 16), 730: (0, 1, 27),
               1414: (6, 17, 33), 1906: (13, 21, 36), 2986: (21, 32, 39), 9634: (56, 57, 57)}
    n = as_int(n)
    if n < 0:
        raise ValueError("n should be a non-negative integer")
    if n == 0:
        return (0, 0, 0)
    n, v = remove(n, 4)
    v = 1 << v
    if n % 8 == 7:
        return
    if n in special:
        return tuple([v*i for i in special[n]])

    s, _exact = integer_nthroot(n, 2)
    if _exact:
        return (0, 0, v*s)
    if n % 8 == 3:
        if not s % 2:
            s -= 1
        for x in range(s, -1, -2):
            N = (n - x**2) // 2
            if isprime(N):
                # n % 8 == 3 and x % 2 == 1 => N % 4 == 1
                y, z = prime_as_sum_of_two_squares(N)
                return tuple(sorted([v*x, v*(y + z), v*abs(y - z)]))
        # We will never reach this point because there must be a solution.
        assert False

    # assert n % 4 in [1, 2]
    if not((n % 2) ^ (s % 2)):
        s -= 1
    for x in range(s, -1, -2):
        N = n - x**2
        if isprime(N):
            # assert N % 4 == 1
            y, z = prime_as_sum_of_two_squares(N)
            return tuple(sorted([v*x, v*y, v*z]))
    # We will never reach this point because there must be a solution.
    assert False


def sum_of_four_squares(n):
    r"""
    Returns a 4-tuple `(a, b, c, d)` such that `a^2 + b^2 + c^2 + d^2 = n`.
    Here `a, b, c, d \geq 0`.

    Parameters
    ==========

    n : Integer
        non-negative integer

    Returns
    =======

    (int, int, int, int) : 4-tuple non-negative integers ``(a, b, c, d)`` satisfying ``a**2 + b**2 + c**2 + d**2 = n``.
                           a,b,c,d are sorted in ascending order.

    Raises
    ======

    ValueError
        If ``n`` is a negative integer

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import sum_of_four_squares
    >>> sum_of_four_squares(3456)
    (8, 8, 32, 48)
    >>> sum_of_four_squares(1294585930293)
    (0, 1234, 2161, 1137796)

    References
    ==========

    .. [1] Representing a number as a sum of four squares, [online],
        Available: https://schorn.ch/lagrange.html

    See Also
    ========

    power_representation :
        ``sum_of_four_squares(n)`` is one of the solutions output by ``power_representation(n, 2, 4, zeros=True)``

    """
    n = as_int(n)
    if n < 0:
        raise ValueError("n should be a non-negative integer")
    if n == 0:
        return (0, 0, 0, 0)
    # remove factors of 4 since a solution in terms of 3 squares is
    # going to be returned; this is also done in sum_of_three_squares,
    # but it needs to be done here to select d
    n, v = remove(n, 4)
    v = 1 << v
    if n % 8 == 7:
        d = 2
        n = n - 4
    elif n % 8 in (2, 6):
        d = 1
        n = n - 1
    else:
        d = 0
    x, y, z = sum_of_three_squares(n)  # sorted
    return tuple(sorted([v*d, v*x, v*y, v*z]))


def power_representation(n, p, k, zeros=False):
    r"""
    Returns a generator for finding k-tuples of integers,
    `(n_{1}, n_{2}, . . . n_{k})`, such that
    `n = n_{1}^p + n_{2}^p + . . . n_{k}^p`.

    Usage
    =====

    ``power_representation(n, p, k, zeros)``: Represent non-negative number
    ``n`` as a sum of ``k`` ``p``\ th powers. If ``zeros`` is true, then the
    solutions is allowed to contain zeros.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import power_representation

    Represent 1729 as a sum of two cubes:

    >>> f = power_representation(1729, 3, 2)
    >>> next(f)
    (9, 10)
    >>> next(f)
    (1, 12)

    If the flag `zeros` is True, the solution may contain tuples with
    zeros; any such solutions will be generated after the solutions
    without zeros:

    >>> list(power_representation(125, 2, 3, zeros=True))
    [(5, 6, 8), (3, 4, 10), (0, 5, 10), (0, 2, 11)]

    For even `p` the `permute_sign` function can be used to get all
    signed values:

    >>> from sympy.utilities.iterables import permute_signs
    >>> list(permute_signs((1, 12)))
    [(1, 12), (-1, 12), (1, -12), (-1, -12)]

    All possible signed permutations can also be obtained:

    >>> from sympy.utilities.iterables import signed_permutations
    >>> list(signed_permutations((1, 12)))
    [(1, 12), (-1, 12), (1, -12), (-1, -12), (12, 1), (-12, 1), (12, -1), (-12, -1)]
    """
    n, p, k = [as_int(i) for i in (n, p, k)]

    if n < 0:
        if p % 2:
            for t in power_representation(-n, p, k, zeros):
                yield tuple(-i for i in t)
        return

    if p < 1 or k < 1:
        raise ValueError(filldedent('''
    Expecting positive integers for `(p, k)`, but got `(%s, %s)`'''
    % (p, k)))

    if n == 0:
        if zeros:
            yield (0,)*k
        return

    if k == 1:
        if p == 1:
            yield (n,)
        elif n == 1:
            yield (1,)
        else:
            be = perfect_power(n)
            if be:
                b, e = be
                d, r = divmod(e, p)
                if not r:
                    yield (b**d,)
        return

    if p == 1:
        yield from partition(n, k, zeros=zeros)
        return

    if p == 2:
        if k == 3:
            n, v = remove(n, 4)
            if v:
                v = 1 << v
                for t in power_representation(n, p, k, zeros):
                    yield tuple(i*v for i in t)
                return
        feasible = _can_do_sum_of_squares(n, k)
        if not feasible:
            return
        if not zeros:
            if n > 33 and k >= 5 and k <= n and n - k in (
                13, 10, 7, 5, 4, 2, 1):
                '''Todd G. Will, "When Is n^2 a Sum of k Squares?", [online].
                Available: https://www.maa.org/sites/default/files/Will-MMz-201037918.pdf'''
                return
            # quick tests since feasibility includes the possibility of 0
            if k == 4 and (n in (1, 3, 5, 9, 11, 17, 29, 41) or remove(n, 4)[0] in (2, 6, 14)):
                # A000534
                return
            if k == 3 and n in (1, 2, 5, 10, 13, 25, 37, 58, 85, 130):  # or n = some number >= 5*10**10
                # A051952
                return
        if feasible is not True:  # it's prime and k == 2
            yield prime_as_sum_of_two_squares(n)
            return

    if k == 2 and p > 2:
        be = perfect_power(n)
        if be and be[1] % p == 0:
            return  # Fermat: a**n + b**n = c**n has no solution for n > 2

    if n >= k:
        a = integer_nthroot(n - (k - 1), p)[0]
        for t in pow_rep_recursive(a, k, n, [], p):
            yield tuple(reversed(t))

    if zeros:
        a = integer_nthroot(n, p)[0]
        for i in range(1, k):
            for t in pow_rep_recursive(a, i, n, [], p):
                yield tuple(reversed(t + (0,)*(k - i)))


sum_of_powers = power_representation


def pow_rep_recursive(n_i, k, n_remaining, terms, p):
    # Invalid arguments
    if n_i <= 0 or k <= 0:
        return

    # No solutions may exist
    if n_remaining < k:
        return
    if k * pow(n_i, p) < n_remaining:
        return

    if k == 0 and n_remaining == 0:
        yield tuple(terms)

    elif k == 1:
        # next_term^p must equal to n_remaining
        next_term, exact = integer_nthroot(n_remaining, p)
        if exact and next_term <= n_i:
            yield tuple(terms + [next_term])
        return

    else:
        # TODO: Fall back to diop_DN when k = 2
        if n_i >= 1 and k > 0:
            for next_term in range(1, n_i + 1):
                residual = n_remaining - pow(next_term, p)
                if residual < 0:
                    break
                yield from pow_rep_recursive(next_term, k - 1, residual, terms + [next_term], p)


def sum_of_squares(n, k, zeros=False):
    """Return a generator that yields the k-tuples of nonnegative
    values, the squares of which sum to n. If zeros is False (default)
    then the solution will not contain zeros. The nonnegative
    elements of a tuple are sorted.

    * If k == 1 and n is square, (n,) is returned.

    * If k == 2 then n can only be written as a sum of squares if
      every prime in the factorization of n that has the form
      4*k + 3 has an even multiplicity. If n is prime then
      it can only be written as a sum of two squares if it is
      in the form 4*k + 1.

    * if k == 3 then n can be written as a sum of squares if it does
      not have the form 4**m*(8*k + 7).

    * all integers can be written as the sum of 4 squares.

    * if k > 4 then n can be partitioned and each partition can
      be written as a sum of 4 squares; if n is not evenly divisible
      by 4 then n can be written as a sum of squares only if the
      an additional partition can be written as sum of squares.
      For example, if k = 6 then n is partitioned into two parts,
      the first being written as a sum of 4 squares and the second
      being written as a sum of 2 squares -- which can only be
      done if the condition above for k = 2 can be met, so this will
      automatically reject certain partitions of n.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import sum_of_squares
    >>> list(sum_of_squares(25, 2))
    [(3, 4)]
    >>> list(sum_of_squares(25, 2, True))
    [(3, 4), (0, 5)]
    >>> list(sum_of_squares(25, 4))
    [(1, 2, 2, 4)]

    See Also
    ========

    sympy.utilities.iterables.signed_permutations
    """
    yield from power_representation(n, 2, k, zeros)


def _can_do_sum_of_squares(n, k):
    """Return True if n can be written as the sum of k squares,
    False if it cannot, or 1 if ``k == 2`` and ``n`` is prime (in which
    case it *can* be written as a sum of two squares). A False
    is returned only if it cannot be written as ``k``-squares, even
    if 0s are allowed.
    """
    if k < 1:
        return False
    if n < 0:
        return False
    if n == 0:
        return True
    if k == 1:
        return is_square(n)
    if k == 2:
        if n in (1, 2):
            return True
        if isprime(n):
            if n % 4 == 1:
                return 1  # signal that it was prime
            return False
        # n is a composite number
        # we can proceed iff no prime factor in the form 4*k + 3
        # has an odd multiplicity
        return all(p % 4 !=3 or m % 2 == 0 for p, m in factorint(n).items())
    if k == 3:
        return remove(n, 4)[0] % 8 != 7
    # every number can be written as a sum of 4 squares; for k > 4 partitions
    # can be 0
    return True
