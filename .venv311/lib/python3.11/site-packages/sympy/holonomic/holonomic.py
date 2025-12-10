"""
This module implements Holonomic Functions and
various operations on them.
"""

from sympy.core import Add, Mul, Pow
from sympy.core.numbers import (NaN, Infinity, NegativeInfinity, Float, I, pi,
        equal_valued, int_valued)
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial, factorial, rf
from sympy.functions.elementary.exponential import exp_polar, exp, log
from sympy.functions.elementary.hyperbolic import (cosh, sinh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin, sinc)
from sympy.functions.special.error_functions import (Ci, Shi, Si, erf, erfc, erfi)
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper, meijerg
from sympy.integrals import meijerint
from sympy.matrices import Matrix
from sympy.polys.rings import PolyElement
from sympy.polys.fields import FracElement
from sympy.polys.domains import QQ, RR
from sympy.polys.polyclasses import DMF
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly
from sympy.polys.matrices import DomainMatrix
from sympy.printing import sstr
from sympy.series.limits import limit
from sympy.series.order import Order
from sympy.simplify.hyperexpand import hyperexpand
from sympy.simplify.simplify import nsimplify
from sympy.solvers.solvers import solve

from .recurrence import HolonomicSequence, RecurrenceOperator, RecurrenceOperators
from .holonomicerrors import (NotPowerSeriesError, NotHyperSeriesError,
    SingularityError, NotHolonomicError)


def _find_nonzero_solution(r, homosys):
    ones = lambda shape: DomainMatrix.ones(shape, r.domain)
    particular, nullspace = r._solve(homosys)
    nullity = nullspace.shape[0]
    nullpart = ones((1, nullity)) * nullspace
    sol = (particular + nullpart).transpose()
    return sol



def DifferentialOperators(base, generator):
    r"""
    This function is used to create annihilators using ``Dx``.

    Explanation
    ===========

    Returns an Algebra of Differential Operators also called Weyl Algebra
    and the operator for differentiation i.e. the ``Dx`` operator.

    Parameters
    ==========

    base:
        Base polynomial ring for the algebra.
        The base polynomial ring is the ring of polynomials in :math:`x` that
        will appear as coefficients in the operators.
    generator:
        Generator of the algebra which can
        be either a noncommutative ``Symbol`` or a string. e.g. "Dx" or "D".

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy.abc import x
    >>> from sympy.holonomic.holonomic import DifferentialOperators
    >>> R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    >>> R
    Univariate Differential Operator Algebra in intermediate Dx over the base ring ZZ[x]
    >>> Dx*x
    (1) + (x)*Dx
    """

    ring = DifferentialOperatorAlgebra(base, generator)
    return (ring, ring.derivative_operator)


class DifferentialOperatorAlgebra:
    r"""
    An Ore Algebra is a set of noncommutative polynomials in the
    intermediate ``Dx`` and coefficients in a base polynomial ring :math:`A`.
    It follows the commutation rule:

    .. math ::
       Dxa = \sigma(a)Dx + \delta(a)

    for :math:`a \subset A`.

    Where :math:`\sigma: A \Rightarrow A` is an endomorphism and :math:`\delta: A \rightarrow A`
    is a skew-derivation i.e. :math:`\delta(ab) = \delta(a) b + \sigma(a) \delta(b)`.

    If one takes the sigma as identity map and delta as the standard derivation
    then it becomes the algebra of Differential Operators also called
    a Weyl Algebra i.e. an algebra whose elements are Differential Operators.

    This class represents a Weyl Algebra and serves as the parent ring for
    Differential Operators.

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy import symbols
    >>> from sympy.holonomic.holonomic import DifferentialOperators
    >>> x = symbols('x')
    >>> R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    >>> R
    Univariate Differential Operator Algebra in intermediate Dx over the base ring
    ZZ[x]

    See Also
    ========

    DifferentialOperator
    """

    def __init__(self, base, generator):
        # the base polynomial ring for the algebra
        self.base = base
        # the operator representing differentiation i.e. `Dx`
        self.derivative_operator = DifferentialOperator(
            [base.zero, base.one], self)

        if generator is None:
            self.gen_symbol = Symbol('Dx', commutative=False)
        else:
            if isinstance(generator, str):
                self.gen_symbol = Symbol(generator, commutative=False)
            elif isinstance(generator, Symbol):
                self.gen_symbol = generator

    def __str__(self):
        string = 'Univariate Differential Operator Algebra in intermediate '\
            + sstr(self.gen_symbol) + ' over the base ring ' + \
            (self.base).__str__()

        return string

    __repr__ = __str__

    def __eq__(self, other):
        return self.base == other.base and \
               self.gen_symbol == other.gen_symbol


class DifferentialOperator:
    """
    Differential Operators are elements of Weyl Algebra. The Operators
    are defined by a list of polynomials in the base ring and the
    parent ring of the Operator i.e. the algebra it belongs to.

    Explanation
    ===========

    Takes a list of polynomials for each power of ``Dx`` and the
    parent ring which must be an instance of DifferentialOperatorAlgebra.

    A Differential Operator can be created easily using
    the operator ``Dx``. See examples below.

    Examples
    ========

    >>> from sympy.holonomic.holonomic import DifferentialOperator, DifferentialOperators
    >>> from sympy import ZZ
    >>> from sympy import symbols
    >>> x = symbols('x')
    >>> R, Dx = DifferentialOperators(ZZ.old_poly_ring(x),'Dx')

    >>> DifferentialOperator([0, 1, x**2], R)
    (1)*Dx + (x**2)*Dx**2

    >>> (x*Dx*x + 1 - Dx**2)**2
    (2*x**2 + 2*x + 1) + (4*x**3 + 2*x**2 - 4)*Dx + (x**4 - 6*x - 2)*Dx**2 + (-2*x**2)*Dx**3 + (1)*Dx**4

    See Also
    ========

    DifferentialOperatorAlgebra
    """

    _op_priority = 20

    def __init__(self, list_of_poly, parent):
        """
        Parameters
        ==========

        list_of_poly:
            List of polynomials belonging to the base ring of the algebra.
        parent:
            Parent algebra of the operator.
        """

        # the parent ring for this operator
        # must be an DifferentialOperatorAlgebra object
        self.parent = parent
        base = self.parent.base
        self.x = base.gens[0] if isinstance(base.gens[0], Symbol) else base.gens[0][0]
        # sequence of polynomials in x for each power of Dx
        # the list should not have trailing zeroes
        # represents the operator
        # convert the expressions into ring elements using from_sympy
        for i, j in enumerate(list_of_poly):
            if not isinstance(j, base.dtype):
                list_of_poly[i] = base.from_sympy(sympify(j))
            else:
                list_of_poly[i] = base.from_sympy(base.to_sympy(j))

        self.listofpoly = list_of_poly
        # highest power of `Dx`
        self.order = len(self.listofpoly) - 1

    def __mul__(self, other):
        """
        Multiplies two DifferentialOperator and returns another
        DifferentialOperator instance using the commutation rule
        Dx*a = a*Dx + a'
        """

        listofself = self.listofpoly
        if isinstance(other, DifferentialOperator):
            listofother = other.listofpoly
        elif isinstance(other, self.parent.base.dtype):
            listofother = [other]
        else:
            listofother = [self.parent.base.from_sympy(sympify(other))]

        # multiplies a polynomial `b` with a list of polynomials
        def _mul_dmp_diffop(b, listofother):
            if isinstance(listofother, list):
                return [i * b for i in listofother]
            return [b * listofother]

        sol = _mul_dmp_diffop(listofself[0], listofother)

        # compute Dx^i * b
        def _mul_Dxi_b(b):
            sol1 = [self.parent.base.zero]
            sol2 = []

            if isinstance(b, list):
                for i in b:
                    sol1.append(i)
                    sol2.append(i.diff())
            else:
                sol1.append(self.parent.base.from_sympy(b))
                sol2.append(self.parent.base.from_sympy(b).diff())

            return _add_lists(sol1, sol2)

        for i in range(1, len(listofself)):
            # find Dx^i * b in ith iteration
            listofother = _mul_Dxi_b(listofother)
            # solution = solution + listofself[i] * (Dx^i * b)
            sol = _add_lists(sol, _mul_dmp_diffop(listofself[i], listofother))

        return DifferentialOperator(sol, self.parent)

    def __rmul__(self, other):
        if not isinstance(other, DifferentialOperator):

            if not isinstance(other, self.parent.base.dtype):
                other = (self.parent.base).from_sympy(sympify(other))

            sol = [other * j for j in self.listofpoly]
            return DifferentialOperator(sol, self.parent)

    def __add__(self, other):
        if isinstance(other, DifferentialOperator):

            sol = _add_lists(self.listofpoly, other.listofpoly)
            return DifferentialOperator(sol, self.parent)

        list_self = self.listofpoly
        if not isinstance(other, self.parent.base.dtype):
            list_other = [((self.parent).base).from_sympy(sympify(other))]
        else:
            list_other = [other]
        sol = [list_self[0] + list_other[0]] + list_self[1:]
        return DifferentialOperator(sol, self.parent)

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-1) * other

    def __rsub__(self, other):
        return (-1) * self + other

    def __neg__(self):
        return -1 * self

    def __truediv__(self, other):
        return self * (S.One / other)

    def __pow__(self, n):
        if n == 1:
            return self
        result = DifferentialOperator([self.parent.base.one], self.parent)
        if n == 0:
            return result
        # if self is `Dx`
        if self.listofpoly == self.parent.derivative_operator.listofpoly:
            sol = [self.parent.base.zero]*n + [self.parent.base.one]
            return DifferentialOperator(sol, self.parent)
        x = self
        while True:
            if n % 2:
                result *= x
            n >>= 1
            if not n:
                break
            x *= x
        return result

    def __str__(self):
        listofpoly = self.listofpoly
        print_str = ''

        for i, j in enumerate(listofpoly):
            if j == self.parent.base.zero:
                continue

            j = self.parent.base.to_sympy(j)

            if i == 0:
                print_str += '(' + sstr(j) + ')'
                continue

            if print_str:
                print_str += ' + '

            if i == 1:
                print_str += '(' + sstr(j) + ')*%s' %(self.parent.gen_symbol)
                continue

            print_str += '(' + sstr(j) + ')' + '*%s**' %(self.parent.gen_symbol) + sstr(i)

        return print_str

    __repr__ = __str__

    def __eq__(self, other):
        if isinstance(other, DifferentialOperator):
            return self.listofpoly == other.listofpoly and \
                   self.parent == other.parent
        return self.listofpoly[0] == other and \
            all(i is self.parent.base.zero for i in self.listofpoly[1:])

    def is_singular(self, x0):
        """
        Checks if the differential equation is singular at x0.
        """

        base = self.parent.base
        return x0 in roots(base.to_sympy(self.listofpoly[-1]), self.x)


class HolonomicFunction:
    r"""
    A Holonomic Function is a solution to a linear homogeneous ordinary
    differential equation with polynomial coefficients. This differential
    equation can also be represented by an annihilator i.e. a Differential
    Operator ``L`` such that :math:`L.f = 0`. For uniqueness of these functions,
    initial conditions can also be provided along with the annihilator.

    Explanation
    ===========

    Holonomic functions have closure properties and thus forms a ring.
    Given two Holonomic Functions f and g, their sum, product,
    integral and derivative is also a Holonomic Function.

    For ordinary points initial condition should be a vector of values of
    the derivatives i.e. :math:`[y(x_0), y'(x_0), y''(x_0) ... ]`.

    For regular singular points initial conditions can also be provided in this
    format:
    :math:`{s0: [C_0, C_1, ...], s1: [C^1_0, C^1_1, ...], ...}`
    where s0, s1, ... are the roots of indicial equation and vectors
    :math:`[C_0, C_1, ...], [C^0_0, C^0_1, ...], ...` are the corresponding initial
    terms of the associated power series. See Examples below.

    Examples
    ========

    >>> from sympy.holonomic.holonomic import HolonomicFunction, DifferentialOperators
    >>> from sympy import QQ
    >>> from sympy import symbols, S
    >>> x = symbols('x')
    >>> R, Dx = DifferentialOperators(QQ.old_poly_ring(x),'Dx')

    >>> p = HolonomicFunction(Dx - 1, x, 0, [1])  # e^x
    >>> q = HolonomicFunction(Dx**2 + 1, x, 0, [0, 1])  # sin(x)

    >>> p + q  # annihilator of e^x + sin(x)
    HolonomicFunction((-1) + (1)*Dx + (-1)*Dx**2 + (1)*Dx**3, x, 0, [1, 2, 1])

    >>> p * q  # annihilator of e^x * sin(x)
    HolonomicFunction((2) + (-2)*Dx + (1)*Dx**2, x, 0, [0, 1])

    An example of initial conditions for regular singular points,
    the indicial equation has only one root `1/2`.

    >>> HolonomicFunction(-S(1)/2 + x*Dx, x, 0, {S(1)/2: [1]})
    HolonomicFunction((-1/2) + (x)*Dx, x, 0, {1/2: [1]})

    >>> HolonomicFunction(-S(1)/2 + x*Dx, x, 0, {S(1)/2: [1]}).to_expr()
    sqrt(x)

    To plot a Holonomic Function, one can use `.evalf()` for numerical
    computation. Here's an example on `sin(x)**2/x` using numpy and matplotlib.

    >>> import sympy.holonomic # doctest: +SKIP
    >>> from sympy import var, sin # doctest: +SKIP
    >>> import matplotlib.pyplot as plt # doctest: +SKIP
    >>> import numpy as np # doctest: +SKIP
    >>> var("x") # doctest: +SKIP
    >>> r = np.linspace(1, 5, 100) # doctest: +SKIP
    >>> y = sympy.holonomic.expr_to_holonomic(sin(x)**2/x, x0=1).evalf(r) # doctest: +SKIP
    >>> plt.plot(r, y, label="holonomic function") # doctest: +SKIP
    >>> plt.show() # doctest: +SKIP

    """

    _op_priority = 20

    def __init__(self, annihilator, x, x0=0, y0=None):
        """

        Parameters
        ==========

        annihilator:
            Annihilator of the Holonomic Function, represented by a
            `DifferentialOperator` object.
        x:
            Variable of the function.
        x0:
            The point at which initial conditions are stored.
            Generally an integer.
        y0:
            The initial condition. The proper format for the initial condition
            is described in class docstring. To make the function unique,
            length of the vector `y0` should be equal to or greater than the
            order of differential equation.
        """

        # initial condition
        self.y0 = y0
        # the point for initial conditions, default is zero.
        self.x0 = x0
        # differential operator L such that L.f = 0
        self.annihilator = annihilator
        self.x = x

    def __str__(self):
        if self._have_init_cond():
            str_sol = 'HolonomicFunction(%s, %s, %s, %s)' % (str(self.annihilator),\
                sstr(self.x), sstr(self.x0), sstr(self.y0))
        else:
            str_sol = 'HolonomicFunction(%s, %s)' % (str(self.annihilator),\
                sstr(self.x))

        return str_sol

    __repr__ = __str__

    def unify(self, other):
        """
        Unifies the base polynomial ring of a given two Holonomic
        Functions.
        """

        R1 = self.annihilator.parent.base
        R2 = other.annihilator.parent.base

        dom1 = R1.dom
        dom2 = R2.dom

        if R1 == R2:
            return (self, other)

        R = (dom1.unify(dom2)).old_poly_ring(self.x)

        newparent, _ = DifferentialOperators(R, str(self.annihilator.parent.gen_symbol))

        sol1 = [R1.to_sympy(i) for i in self.annihilator.listofpoly]
        sol2 = [R2.to_sympy(i) for i in other.annihilator.listofpoly]

        sol1 = DifferentialOperator(sol1, newparent)
        sol2 = DifferentialOperator(sol2, newparent)

        sol1 = HolonomicFunction(sol1, self.x, self.x0, self.y0)
        sol2 = HolonomicFunction(sol2, other.x, other.x0, other.y0)

        return (sol1, sol2)

    def is_singularics(self):
        """
        Returns True if the function have singular initial condition
        in the dictionary format.

        Returns False if the function have ordinary initial condition
        in the list format.

        Returns None for all other cases.
        """

        if isinstance(self.y0, dict):
            return True
        elif isinstance(self.y0, list):
            return False

    def _have_init_cond(self):
        """
        Checks if the function have initial condition.
        """
        return bool(self.y0)

    def _singularics_to_ord(self):
        """
        Converts a singular initial condition to ordinary if possible.
        """
        a = list(self.y0)[0]
        b = self.y0[a]

        if len(self.y0) == 1 and a == int(a) and a > 0:
            a = int(a)
            y0 = [S.Zero] * a
            y0 += [j * factorial(a + i) for i, j in enumerate(b)]

            return HolonomicFunction(self.annihilator, self.x, self.x0, y0)

    def __add__(self, other):
        # if the ground domains are different
        if self.annihilator.parent.base != other.annihilator.parent.base:
            a, b = self.unify(other)
            return a + b

        deg1 = self.annihilator.order
        deg2 = other.annihilator.order
        dim = max(deg1, deg2)
        R = self.annihilator.parent.base
        K = R.get_field()

        rowsself = [self.annihilator]
        rowsother = [other.annihilator]
        gen = self.annihilator.parent.derivative_operator

        # constructing annihilators up to order dim
        for i in range(dim - deg1):
            diff1 = (gen * rowsself[-1])
            rowsself.append(diff1)

        for i in range(dim - deg2):
            diff2 = (gen * rowsother[-1])
            rowsother.append(diff2)

        row = rowsself + rowsother

        # constructing the matrix of the ansatz
        r = []

        for expr in row:
            p = []
            for i in range(dim + 1):
                if i >= len(expr.listofpoly):
                    p.append(K.zero)
                else:
                    p.append(K.new(expr.listofpoly[i].to_list()))
            r.append(p)

        # solving the linear system using gauss jordan solver
        r = DomainMatrix(r, (len(row), dim+1), K).transpose()
        homosys = DomainMatrix.zeros((dim+1, 1), K)
        sol = _find_nonzero_solution(r, homosys)

        # if a solution is not obtained then increasing the order by 1 in each
        # iteration
        while sol.is_zero_matrix:
            dim += 1

            diff1 = (gen * rowsself[-1])
            rowsself.append(diff1)

            diff2 = (gen * rowsother[-1])
            rowsother.append(diff2)

            row = rowsself + rowsother
            r = []

            for expr in row:
                p = []
                for i in range(dim + 1):
                    if i >= len(expr.listofpoly):
                        p.append(K.zero)
                    else:
                        p.append(K.new(expr.listofpoly[i].to_list()))
                r.append(p)

            # solving the linear system using gauss jordan solver
            r = DomainMatrix(r, (len(row), dim+1), K).transpose()
            homosys = DomainMatrix.zeros((dim+1, 1), K)
            sol = _find_nonzero_solution(r, homosys)

        # taking only the coefficients needed to multiply with `self`
        # can be also be done the other way by taking R.H.S and multiplying with
        # `other`
        sol = sol.flat()[:dim + 1 - deg1]
        sol1 = _normalize(sol, self.annihilator.parent)
        # annihilator of the solution
        sol = sol1 * (self.annihilator)
        sol = _normalize(sol.listofpoly, self.annihilator.parent, negative=False)

        if not (self._have_init_cond() and other._have_init_cond()):
            return HolonomicFunction(sol, self.x)

        # both the functions have ordinary initial conditions
        if self.is_singularics() == False and other.is_singularics() == False:

            # directly add the corresponding value
            if self.x0 == other.x0:
                # try to extended the initial conditions
                # using the annihilator
                y1 = _extend_y0(self, sol.order)
                y2 = _extend_y0(other, sol.order)
                y0 = [a + b for a, b in zip(y1, y2)]
                return HolonomicFunction(sol, self.x, self.x0, y0)

            # change the initial conditions to a same point
            selfat0 = self.annihilator.is_singular(0)
            otherat0 = other.annihilator.is_singular(0)
            if self.x0 == 0 and not selfat0 and not otherat0:
                return self + other.change_ics(0)
            if other.x0 == 0 and not selfat0 and not otherat0:
                return self.change_ics(0) + other

            selfatx0 = self.annihilator.is_singular(self.x0)
            otheratx0 = other.annihilator.is_singular(self.x0)
            if not selfatx0 and not otheratx0:
                return self + other.change_ics(self.x0)
            return self.change_ics(other.x0) + other

        if self.x0 != other.x0:
            return HolonomicFunction(sol, self.x)

        # if the functions have singular_ics
        y1 = None
        y2 = None

        if self.is_singularics() == False and other.is_singularics() == True:
            # convert the ordinary initial condition to singular.
            _y0 = [j / factorial(i) for i, j in enumerate(self.y0)]
            y1 = {S.Zero: _y0}
            y2 = other.y0
        elif self.is_singularics() == True and other.is_singularics() == False:
            _y0 = [j / factorial(i) for i, j in enumerate(other.y0)]
            y1 = self.y0
            y2 = {S.Zero: _y0}
        elif self.is_singularics() == True and other.is_singularics() == True:
            y1 = self.y0
            y2 = other.y0

        # computing singular initial condition for the result
        # taking union of the series terms of both functions
        y0 = {}
        for i in y1:
            # add corresponding initial terms if the power
            # on `x` is same
            if i in y2:
                y0[i] = [a + b for a, b in zip(y1[i], y2[i])]
            else:
                y0[i] = y1[i]
        for i in y2:
            if i not in y1:
                y0[i] = y2[i]
        return HolonomicFunction(sol, self.x, self.x0, y0)

    def integrate(self, limits, initcond=False):
        """
        Integrates the given holonomic function.

        Examples
        ========

        >>> from sympy.holonomic.holonomic import HolonomicFunction, DifferentialOperators
        >>> from sympy import QQ
        >>> from sympy import symbols
        >>> x = symbols('x')
        >>> R, Dx = DifferentialOperators(QQ.old_poly_ring(x),'Dx')
        >>> HolonomicFunction(Dx - 1, x, 0, [1]).integrate((x, 0, x))  # e^x - 1
        HolonomicFunction((-1)*Dx + (1)*Dx**2, x, 0, [0, 1])
        >>> HolonomicFunction(Dx**2 + 1, x, 0, [1, 0]).integrate((x, 0, x))
        HolonomicFunction((1)*Dx + (1)*Dx**3, x, 0, [0, 1, 0])
        """

        # to get the annihilator, just multiply by Dx from right
        D = self.annihilator.parent.derivative_operator

        # if the function have initial conditions of the series format
        if self.is_singularics() == True:

            r = self._singularics_to_ord()
            if r:
                return r.integrate(limits, initcond=initcond)

            # computing singular initial condition for the function
            # produced after integration.
            y0 = {}
            for i in self.y0:
                c = self.y0[i]
                c2 = []
                for j, cj in enumerate(c):
                    if cj == 0:
                        c2.append(S.Zero)

                    # if power on `x` is -1, the integration becomes log(x)
                    # TODO: Implement this case
                    elif i + j + 1 == 0:
                        raise NotImplementedError("logarithmic terms in the series are not supported")
                    else:
                        c2.append(cj / S(i + j + 1))
                y0[i + 1] = c2

            if hasattr(limits, "__iter__"):
                raise NotImplementedError("Definite integration for singular initial conditions")

            return HolonomicFunction(self.annihilator * D, self.x, self.x0, y0)

        # if no initial conditions are available for the function
        if not self._have_init_cond():
            if initcond:
                return HolonomicFunction(self.annihilator * D, self.x, self.x0, [S.Zero])
            return HolonomicFunction(self.annihilator * D, self.x)

        # definite integral
        # initial conditions for the answer will be stored at point `a`,
        # where `a` is the lower limit of the integrand
        if hasattr(limits, "__iter__"):

            if len(limits) == 3 and limits[0] == self.x:
                x0 = self.x0
                a = limits[1]
                b = limits[2]
                definite = True

        else:
            definite = False

        y0 = [S.Zero]
        y0 += self.y0

        indefinite_integral = HolonomicFunction(self.annihilator * D, self.x, self.x0, y0)

        if not definite:
            return indefinite_integral

        # use evalf to get the values at `a`
        if x0 != a:
            try:
                indefinite_expr = indefinite_integral.to_expr()
            except (NotHyperSeriesError, NotPowerSeriesError):
                indefinite_expr = None

            if indefinite_expr:
                lower = indefinite_expr.subs(self.x, a)
                if isinstance(lower, NaN):
                    lower = indefinite_expr.limit(self.x, a)
            else:
                lower = indefinite_integral.evalf(a)

            if b == self.x:
                y0[0] = y0[0] - lower
                return HolonomicFunction(self.annihilator * D, self.x, x0, y0)

            elif S(b).is_Number:
                if indefinite_expr:
                    upper = indefinite_expr.subs(self.x, b)
                    if isinstance(upper, NaN):
                        upper = indefinite_expr.limit(self.x, b)
                else:
                    upper = indefinite_integral.evalf(b)

                return upper - lower


        # if the upper limit is `x`, the answer will be a function
        if b == self.x:
            return HolonomicFunction(self.annihilator * D, self.x, a, y0)

        # if the upper limits is a Number, a numerical value will be returned
        elif S(b).is_Number:
            try:
                s = HolonomicFunction(self.annihilator * D, self.x, a,\
                    y0).to_expr()
                indefinite = s.subs(self.x, b)
                if not isinstance(indefinite, NaN):
                    return indefinite
                else:
                    return s.limit(self.x, b)
            except (NotHyperSeriesError, NotPowerSeriesError):
                return HolonomicFunction(self.annihilator * D, self.x, a, y0).evalf(b)

        return HolonomicFunction(self.annihilator * D, self.x)

    def diff(self, *args, **kwargs):
        r"""
        Differentiation of the given Holonomic function.

        Examples
        ========

        >>> from sympy.holonomic.holonomic import HolonomicFunction, DifferentialOperators
        >>> from sympy import ZZ
        >>> from sympy import symbols
        >>> x = symbols('x')
        >>> R, Dx = DifferentialOperators(ZZ.old_poly_ring(x),'Dx')
        >>> HolonomicFunction(Dx**2 + 1, x, 0, [0, 1]).diff().to_expr()
        cos(x)
        >>> HolonomicFunction(Dx - 2, x, 0, [1]).diff().to_expr()
        2*exp(2*x)

        See Also
        ========

        integrate
        """
        kwargs.setdefault('evaluate', True)
        if args:
            if args[0] != self.x:
                return S.Zero
            elif len(args) == 2:
                sol = self
                for i in range(args[1]):
                    sol = sol.diff(args[0])
                return sol

        ann = self.annihilator

        # if the function is constant.
        if ann.listofpoly[0] == ann.parent.base.zero and ann.order == 1:
            return S.Zero

        # if the coefficient of y in the differential equation is zero.
        # a shifting is done to compute the answer in this case.
        elif ann.listofpoly[0] == ann.parent.base.zero:

            sol = DifferentialOperator(ann.listofpoly[1:], ann.parent)

            if self._have_init_cond():
                # if ordinary initial condition
                if self.is_singularics() == False:
                    return HolonomicFunction(sol, self.x, self.x0, self.y0[1:])
                # TODO: support for singular initial condition
                return HolonomicFunction(sol, self.x)
            else:
                return HolonomicFunction(sol, self.x)

        # the general algorithm
        R = ann.parent.base
        K = R.get_field()

        seq_dmf = [K.new(i.to_list()) for i in ann.listofpoly]

        # -y = a1*y'/a0 + a2*y''/a0 ... + an*y^n/a0
        rhs = [i / seq_dmf[0] for i in seq_dmf[1:]]
        rhs.insert(0, K.zero)

        # differentiate both lhs and rhs
        sol = _derivate_diff_eq(rhs, K)

        # add the term y' in lhs to rhs
        sol = _add_lists(sol, [K.zero, K.one])

        sol = _normalize(sol[1:], self.annihilator.parent, negative=False)

        if not self._have_init_cond() or self.is_singularics() == True:
            return HolonomicFunction(sol, self.x)

        y0 = _extend_y0(self, sol.order + 1)[1:]
        return HolonomicFunction(sol, self.x, self.x0, y0)

    def __eq__(self, other):
        if self.annihilator != other.annihilator or self.x != other.x:
            return False
        if self._have_init_cond() and other._have_init_cond():
            return self.x0 == other.x0 and self.y0 == other.y0
        return True

    def __mul__(self, other):
        ann_self = self.annihilator

        if not isinstance(other, HolonomicFunction):
            other = sympify(other)

            if other.has(self.x):
                raise NotImplementedError(" Can't multiply a HolonomicFunction and expressions/functions.")

            if not self._have_init_cond():
                return self
            y0 = _extend_y0(self, ann_self.order)
            y1 = [(Poly.new(j, self.x) * other).rep for j in y0]
            return HolonomicFunction(ann_self, self.x, self.x0, y1)

        if self.annihilator.parent.base != other.annihilator.parent.base:
            a, b = self.unify(other)
            return a * b

        ann_other = other.annihilator

        a = ann_self.order
        b = ann_other.order

        R = ann_self.parent.base
        K = R.get_field()

        list_self = [K.new(j.to_list()) for j in ann_self.listofpoly]
        list_other = [K.new(j.to_list()) for j in ann_other.listofpoly]

        # will be used to reduce the degree
        self_red = [-list_self[i] / list_self[a] for i in range(a)]

        other_red = [-list_other[i] / list_other[b] for i in range(b)]

        # coeff_mull[i][j] is the coefficient of Dx^i(f).Dx^j(g)
        coeff_mul = [[K.zero for i in range(b + 1)] for j in range(a + 1)]
        coeff_mul[0][0] = K.one

        # making the ansatz
        lin_sys_elements = [[coeff_mul[i][j] for i in range(a) for j in range(b)]]
        lin_sys = DomainMatrix(lin_sys_elements, (1, a*b), K).transpose()

        homo_sys = DomainMatrix.zeros((a*b, 1), K)

        sol = _find_nonzero_solution(lin_sys, homo_sys)

        # until a non trivial solution is found
        while sol.is_zero_matrix:

            # updating the coefficients Dx^i(f).Dx^j(g) for next degree
            for i in range(a - 1, -1, -1):
                for j in range(b - 1, -1, -1):
                    coeff_mul[i][j + 1] += coeff_mul[i][j]
                    coeff_mul[i + 1][j] += coeff_mul[i][j]
                    if isinstance(coeff_mul[i][j], K.dtype):
                        coeff_mul[i][j] = DMFdiff(coeff_mul[i][j], K)
                    else:
                        coeff_mul[i][j] = coeff_mul[i][j].diff(self.x)

            # reduce the terms to lower power using annihilators of f, g
            for i in range(a + 1):
                if coeff_mul[i][b].is_zero:
                    continue
                for j in range(b):
                    coeff_mul[i][j] += other_red[j] * coeff_mul[i][b]
                coeff_mul[i][b] = K.zero

            # not d2 + 1, as that is already covered in previous loop
            for j in range(b):
                if coeff_mul[a][j] == 0:
                    continue
                for i in range(a):
                    coeff_mul[i][j] += self_red[i] * coeff_mul[a][j]
                coeff_mul[a][j] = K.zero

            lin_sys_elements.append([coeff_mul[i][j] for i in range(a) for j in range(b)])
            lin_sys = DomainMatrix(lin_sys_elements, (len(lin_sys_elements), a*b), K).transpose()

            sol = _find_nonzero_solution(lin_sys, homo_sys)

        sol_ann = _normalize(sol.flat(), self.annihilator.parent, negative=False)

        if not (self._have_init_cond() and other._have_init_cond()):
            return HolonomicFunction(sol_ann, self.x)

        if self.is_singularics() == False and other.is_singularics() == False:

            # if both the conditions are at same point
            if self.x0 == other.x0:

                # try to find more initial conditions
                y0_self = _extend_y0(self, sol_ann.order)
                y0_other = _extend_y0(other, sol_ann.order)
                # h(x0) = f(x0) * g(x0)
                y0 = [y0_self[0] * y0_other[0]]

                # coefficient of Dx^j(f)*Dx^i(g) in Dx^i(fg)
                for i in range(1, min(len(y0_self), len(y0_other))):
                    coeff = [[0 for i in range(i + 1)] for j in range(i + 1)]
                    for j in range(i + 1):
                        for k in range(i + 1):
                            if j + k == i:
                                coeff[j][k] = binomial(i, j)

                    sol = 0
                    for j in range(i + 1):
                        for k in range(i + 1):
                            sol += coeff[j][k]* y0_self[j] * y0_other[k]

                    y0.append(sol)

                return HolonomicFunction(sol_ann, self.x, self.x0, y0)

            # if the points are different, consider one
            selfat0 = self.annihilator.is_singular(0)
            otherat0 = other.annihilator.is_singular(0)

            if self.x0 == 0 and not selfat0 and not otherat0:
                return self * other.change_ics(0)
            if other.x0 == 0 and not selfat0 and not otherat0:
                return self.change_ics(0) * other

            selfatx0 = self.annihilator.is_singular(self.x0)
            otheratx0 = other.annihilator.is_singular(self.x0)
            if not selfatx0 and not otheratx0:
                return self * other.change_ics(self.x0)
            return self.change_ics(other.x0) * other

        if self.x0 != other.x0:
            return HolonomicFunction(sol_ann, self.x)

        # if the functions have singular_ics
        y1 = None
        y2 = None

        if self.is_singularics() == False and other.is_singularics() == True:
            _y0 = [j / factorial(i) for i, j in enumerate(self.y0)]
            y1 = {S.Zero: _y0}
            y2 = other.y0
        elif self.is_singularics() == True and other.is_singularics() == False:
            _y0 = [j / factorial(i) for i, j in enumerate(other.y0)]
            y1 = self.y0
            y2 = {S.Zero: _y0}
        elif self.is_singularics() == True and other.is_singularics() == True:
            y1 = self.y0
            y2 = other.y0

        y0 = {}
        # multiply every possible pair of the series terms
        for i in y1:
            for j in y2:
                k = min(len(y1[i]), len(y2[j]))
                c = [sum((y1[i][b] * y2[j][a - b] for b in range(a + 1)),
                         start=S.Zero) for a in range(k)]
                if not i + j in y0:
                    y0[i + j] = c
                else:
                    y0[i + j] = [a + b for a, b in zip(c, y0[i + j])]
        return HolonomicFunction(sol_ann, self.x, self.x0, y0)

    __rmul__ = __mul__

    def __sub__(self, other):
        return self + other * -1

    def __rsub__(self, other):
        return self * -1 + other

    def __neg__(self):
        return -1 * self

    def __truediv__(self, other):
        return self * (S.One / other)

    def __pow__(self, n):
        if self.annihilator.order <= 1:
            ann = self.annihilator
            parent = ann.parent

            if self.y0 is None:
                y0 = None
            else:
                y0 = [list(self.y0)[0] ** n]

            p0 = ann.listofpoly[0]
            p1 = ann.listofpoly[1]

            p0 = (Poly.new(p0, self.x) * n).rep

            sol = [parent.base.to_sympy(i) for i in [p0, p1]]
            dd = DifferentialOperator(sol, parent)
            return HolonomicFunction(dd, self.x, self.x0, y0)
        if n < 0:
            raise NotHolonomicError("Negative Power on a Holonomic Function")
        Dx = self.annihilator.parent.derivative_operator
        result = HolonomicFunction(Dx, self.x, S.Zero, [S.One])
        if n == 0:
            return result
        x = self
        while True:
            if n % 2:
                result *= x
            n >>= 1
            if not n:
                break
            x *= x
        return result

    def degree(self):
        """
        Returns the highest power of `x` in the annihilator.
        """
        return max(i.degree() for i in self.annihilator.listofpoly)

    def composition(self, expr, *args, **kwargs):
        """
        Returns function after composition of a holonomic
        function with an algebraic function. The method cannot compute
        initial conditions for the result by itself, so they can be also be
        provided.

        Examples
        ========

        >>> from sympy.holonomic.holonomic import HolonomicFunction, DifferentialOperators
        >>> from sympy import QQ
        >>> from sympy import symbols
        >>> x = symbols('x')
        >>> R, Dx = DifferentialOperators(QQ.old_poly_ring(x),'Dx')
        >>> HolonomicFunction(Dx - 1, x).composition(x**2, 0, [1])  # e^(x**2)
        HolonomicFunction((-2*x) + (1)*Dx, x, 0, [1])
        >>> HolonomicFunction(Dx**2 + 1, x).composition(x**2 - 1, 1, [1, 0])
        HolonomicFunction((4*x**3) + (-1)*Dx + (x)*Dx**2, x, 1, [1, 0])

        See Also
        ========

        from_hyper
        """

        R = self.annihilator.parent
        a = self.annihilator.order
        diff = expr.diff(self.x)
        listofpoly = self.annihilator.listofpoly

        for i, j in enumerate(listofpoly):
            if isinstance(j, self.annihilator.parent.base.dtype):
                listofpoly[i] = self.annihilator.parent.base.to_sympy(j)

        r = listofpoly[a].subs({self.x:expr})
        subs = [-listofpoly[i].subs({self.x:expr}) / r for i in range (a)]
        coeffs = [S.Zero for i in range(a)]  # coeffs[i] == coeff of (D^i f)(a) in D^k (f(a))
        coeffs[0] = S.One
        system = [coeffs]
        homogeneous = Matrix([[S.Zero for i in range(a)]]).transpose()
        while True:
            coeffs_next = [p.diff(self.x) for p in coeffs]
            for i in range(a - 1):
                coeffs_next[i + 1] += (coeffs[i] * diff)
            for i in range(a):
                coeffs_next[i] += (coeffs[-1] * subs[i] * diff)
            coeffs = coeffs_next
            # check for linear relations
            system.append(coeffs)
            sol, taus = (Matrix(system).transpose()
                ).gauss_jordan_solve(homogeneous)
            if sol.is_zero_matrix is not True:
                break

        tau = list(taus)[0]
        sol = sol.subs(tau, 1)
        sol = _normalize(sol[0:], R, negative=False)

        # if initial conditions are given for the resulting function
        if args:
            return HolonomicFunction(sol, self.x, args[0], args[1])
        return HolonomicFunction(sol, self.x)

    def to_sequence(self, lb=True):
        r"""
        Finds recurrence relation for the coefficients in the series expansion
        of the function about :math:`x_0`, where :math:`x_0` is the point at
        which the initial condition is stored.

        Explanation
        ===========

        If the point :math:`x_0` is ordinary, solution of the form :math:`[(R, n_0)]`
        is returned. Where :math:`R` is the recurrence relation and :math:`n_0` is the
        smallest ``n`` for which the recurrence holds true.

        If the point :math:`x_0` is regular singular, a list of solutions in
        the format :math:`(R, p, n_0)` is returned, i.e. `[(R, p, n_0), ... ]`.
        Each tuple in this vector represents a recurrence relation :math:`R`
        associated with a root of the indicial equation ``p``. Conditions of
        a different format can also be provided in this case, see the
        docstring of HolonomicFunction class.

        If it's not possible to numerically compute a initial condition,
        it is returned as a symbol :math:`C_j`, denoting the coefficient of
        :math:`(x - x_0)^j` in the power series about :math:`x_0`.

        Examples
        ========

        >>> from sympy.holonomic.holonomic import HolonomicFunction, DifferentialOperators
        >>> from sympy import QQ
        >>> from sympy import symbols, S
        >>> x = symbols('x')
        >>> R, Dx = DifferentialOperators(QQ.old_poly_ring(x),'Dx')
        >>> HolonomicFunction(Dx - 1, x, 0, [1]).to_sequence()
        [(HolonomicSequence((-1) + (n + 1)Sn, n), u(0) = 1, 0)]
        >>> HolonomicFunction((1 + x)*Dx**2 + Dx, x, 0, [0, 1]).to_sequence()
        [(HolonomicSequence((n**2) + (n**2 + n)Sn, n), u(0) = 0, u(1) = 1, u(2) = -1/2, 2)]
        >>> HolonomicFunction(-S(1)/2 + x*Dx, x, 0, {S(1)/2: [1]}).to_sequence()
        [(HolonomicSequence((n), n), u(0) = 1, 1/2, 1)]

        See Also
        ========

        HolonomicFunction.series

        References
        ==========

        .. [1] https://hal.inria.fr/inria-00070025/document
        .. [2] https://www3.risc.jku.at/publications/download/risc_2244/DIPLFORM.pdf

        """

        if self.x0 != 0:
            return self.shift_x(self.x0).to_sequence()

        # check whether a power series exists if the point is singular
        if self.annihilator.is_singular(self.x0):
            return self._frobenius(lb=lb)

        dict1 = {}
        n = Symbol('n', integer=True)
        dom = self.annihilator.parent.base.dom
        R, _ = RecurrenceOperators(dom.old_poly_ring(n), 'Sn')

        # substituting each term of the form `x^k Dx^j` in the
        # annihilator, according to the formula below:
        # x^k Dx^j = Sum(rf(n + 1 - k, j) * a(n + j - k) * x^n, (n, k, oo))
        # for explanation see [2].
        for i, j in enumerate(self.annihilator.listofpoly):

            listofdmp = j.all_coeffs()
            degree = len(listofdmp) - 1

            for k in range(degree + 1):
                coeff = listofdmp[degree - k]

                if coeff == 0:
                    continue

                if (i - k, k) in dict1:
                    dict1[(i - k, k)] += (dom.to_sympy(coeff) * rf(n - k + 1, i))
                else:
                    dict1[(i - k, k)] = (dom.to_sympy(coeff) * rf(n - k + 1, i))


        sol = []
        keylist = [i[0] for i in dict1]
        lower = min(keylist)
        upper = max(keylist)
        degree = self.degree()

        # the recurrence relation holds for all values of
        # n greater than smallest_n, i.e. n >= smallest_n
        smallest_n = lower + degree
        dummys = {}
        eqs = []
        unknowns = []

        # an appropriate shift of the recurrence
        for j in range(lower, upper + 1):
            if j in keylist:
                temp = sum((v.subs(n, n - lower)
                           for k, v in dict1.items() if k[0] == j),
                           start=S.Zero)
                sol.append(temp)
            else:
                sol.append(S.Zero)

        # the recurrence relation
        sol = RecurrenceOperator(sol, R)

        # computing the initial conditions for recurrence
        order = sol.order
        all_roots = roots(R.base.to_sympy(sol.listofpoly[-1]), n, filter='Z')
        all_roots = all_roots.keys()

        if all_roots:
            max_root = max(all_roots) + 1
            smallest_n = max(max_root, smallest_n)
        order += smallest_n

        y0 = _extend_y0(self, order)
        # u(n) = y^n(0)/factorial(n)
        u0 = [j / factorial(i) for i, j in enumerate(y0)]

        # if sufficient conditions can't be computed then
        # try to use the series method i.e.
        # equate the coefficients of x^k in the equation formed by
        # substituting the series in differential equation, to zero.
        if len(u0) < order:

            for i in range(degree):
                eq = S.Zero

                for j in dict1:

                    if i + j[0] < 0:
                        dummys[i + j[0]] = S.Zero

                    elif i + j[0] < len(u0):
                        dummys[i + j[0]] = u0[i + j[0]]

                    elif not i + j[0] in dummys:
                        dummys[i + j[0]] = Symbol('C_%s' %(i + j[0]))
                        unknowns.append(dummys[i + j[0]])

                    if j[1] <= i:
                        eq += dict1[j].subs(n, i) * dummys[i + j[0]]

                eqs.append(eq)

            # solve the system of equations formed
            soleqs = solve(eqs, *unknowns)

            if isinstance(soleqs, dict):

                for i in range(len(u0), order):

                    if i not in dummys:
                        dummys[i] = Symbol('C_%s' %i)

                    if dummys[i] in soleqs:
                        u0.append(soleqs[dummys[i]])

                    else:
                        u0.append(dummys[i])

                if lb:
                    return [(HolonomicSequence(sol, u0), smallest_n)]
                return [HolonomicSequence(sol, u0)]

            for i in range(len(u0), order):

                if i not in dummys:
                    dummys[i] = Symbol('C_%s' %i)

                s = False
                for j in soleqs:
                    if dummys[i] in j:
                        u0.append(j[dummys[i]])
                        s = True
                if not s:
                    u0.append(dummys[i])

        if lb:
            return [(HolonomicSequence(sol, u0), smallest_n)]

        return [HolonomicSequence(sol, u0)]

    def _frobenius(self, lb=True):
        # compute the roots of indicial equation
        indicialroots = self._indicial()

        reals = []
        compl = []
        for i in ordered(indicialroots.keys()):
            if i.is_real:
                reals.extend([i] * indicialroots[i])
            else:
                a, b = i.as_real_imag()
                compl.extend([(i, a, b)] * indicialroots[i])

        # sort the roots for a fixed ordering of solution
        compl.sort(key=lambda x : x[1])
        compl.sort(key=lambda x : x[2])
        reals.sort()

        # grouping the roots, roots differ by an integer are put in the same group.
        grp = []

        for i in reals:
            if len(grp) == 0:
                grp.append([i])
                continue
            for j in grp:
                if int_valued(j[0] - i):
                    j.append(i)
                    break
            else:
                grp.append([i])

        # True if none of the roots differ by an integer i.e.
        # each element in group have only one member
        independent = all(len(i) == 1 for i in grp)

        allpos = all(i >= 0 for i in reals)
        allint = all(int_valued(i) for i in reals)

        # if initial conditions are provided
        # then use them.
        if self.is_singularics() == True:
            rootstoconsider = []
            for i in ordered(self.y0.keys()):
                for j in ordered(indicialroots.keys()):
                    if equal_valued(j, i):
                        rootstoconsider.append(i)

        elif allpos and allint:
            rootstoconsider = [min(reals)]

        elif independent:
            rootstoconsider = [i[0] for i in grp] + [j[0] for j in compl]

        elif not allint:
            rootstoconsider = [i for i in reals if not int(i) == i]

        elif not allpos:

            if not self._have_init_cond() or S(self.y0[0]).is_finite ==  False:
                rootstoconsider = [min(reals)]

            else:
                posroots = [i for i in reals if i >= 0]
                rootstoconsider = [min(posroots)]

        n = Symbol('n', integer=True)
        dom = self.annihilator.parent.base.dom
        R, _ = RecurrenceOperators(dom.old_poly_ring(n), 'Sn')

        finalsol = []
        char = ord('C')

        for p in rootstoconsider:
            dict1 = {}

            for i, j in enumerate(self.annihilator.listofpoly):

                listofdmp = j.all_coeffs()
                degree = len(listofdmp) - 1

                for k in range(degree + 1):
                    coeff = listofdmp[degree - k]

                    if coeff == 0:
                        continue

                    if (i - k, k - i) in dict1:
                        dict1[(i - k, k - i)] += (dom.to_sympy(coeff) * rf(n - k + 1 + p, i))
                    else:
                        dict1[(i - k, k - i)] = (dom.to_sympy(coeff) * rf(n - k + 1 + p, i))

            sol = []
            keylist = [i[0] for i in dict1]
            lower = min(keylist)
            upper = max(keylist)
            degree = max(i[1] for i in dict1)
            degree2 = min(i[1] for i in dict1)

            smallest_n = lower + degree
            dummys = {}
            eqs = []
            unknowns = []

            for j in range(lower, upper + 1):
                if j in keylist:
                    temp = sum((v.subs(n, n - lower)
                               for k, v in dict1.items() if k[0] == j),
                               start=S.Zero)
                    sol.append(temp)
                else:
                    sol.append(S.Zero)

            # the recurrence relation
            sol = RecurrenceOperator(sol, R)

            # computing the initial conditions for recurrence
            order = sol.order
            all_roots = roots(R.base.to_sympy(sol.listofpoly[-1]), n, filter='Z')
            all_roots = all_roots.keys()

            if all_roots:
                max_root = max(all_roots) + 1
                smallest_n = max(max_root, smallest_n)
            order += smallest_n

            u0 = []

            if self.is_singularics() == True:
                u0 = self.y0[p]

            elif self.is_singularics() == False and p >= 0 and int(p) == p and len(rootstoconsider) == 1:
                y0 = _extend_y0(self, order + int(p))
                # u(n) = y^n(0)/factorial(n)
                if len(y0) > int(p):
                    u0 = [y0[i] / factorial(i) for i in range(int(p), len(y0))]

            if len(u0) < order:

                for i in range(degree2, degree):
                    eq = S.Zero

                    for j in dict1:
                        if i + j[0] < 0:
                            dummys[i + j[0]] = S.Zero

                        elif i + j[0] < len(u0):
                            dummys[i + j[0]] = u0[i + j[0]]

                        elif not i + j[0] in dummys:
                            letter = chr(char) + '_%s' %(i + j[0])
                            dummys[i + j[0]] = Symbol(letter)
                            unknowns.append(dummys[i + j[0]])

                        if j[1] <= i:
                            eq += dict1[j].subs(n, i) * dummys[i + j[0]]

                    eqs.append(eq)

                # solve the system of equations formed
                soleqs = solve(eqs, *unknowns)

                if isinstance(soleqs, dict):

                    for i in range(len(u0), order):

                        if i not in dummys:
                            letter = chr(char) + '_%s' %i
                            dummys[i] = Symbol(letter)

                        if dummys[i] in soleqs:
                            u0.append(soleqs[dummys[i]])

                        else:
                            u0.append(dummys[i])

                    if lb:
                        finalsol.append((HolonomicSequence(sol, u0), p, smallest_n))
                        continue
                    else:
                        finalsol.append((HolonomicSequence(sol, u0), p))
                        continue

                for i in range(len(u0), order):

                    if i not in dummys:
                        letter = chr(char) + '_%s' %i
                        dummys[i] = Symbol(letter)

                    s = False
                    for j in soleqs:
                        if dummys[i] in j:
                            u0.append(j[dummys[i]])
                            s = True
                    if not s:
                        u0.append(dummys[i])
            if lb:
                finalsol.append((HolonomicSequence(sol, u0), p, smallest_n))

            else:
                finalsol.append((HolonomicSequence(sol, u0), p))
            char += 1
        return finalsol

    def series(self, n=6, coefficient=False, order=True, _recur=None):
        r"""
        Finds the power series expansion of given holonomic function about :math:`x_0`.

        Explanation
        ===========

        A list of series might be returned if :math:`x_0` is a regular point with
        multiple roots of the indicial equation.

        Examples
        ========

        >>> from sympy.holonomic.holonomic import HolonomicFunction, DifferentialOperators
        >>> from sympy import QQ
        >>> from sympy import symbols
        >>> x = symbols('x')
        >>> R, Dx = DifferentialOperators(QQ.old_poly_ring(x),'Dx')
        >>> HolonomicFunction(Dx - 1, x, 0, [1]).series()  # e^x
        1 + x + x**2/2 + x**3/6 + x**4/24 + x**5/120 + O(x**6)
        >>> HolonomicFunction(Dx**2 + 1, x, 0, [0, 1]).series(n=8)  # sin(x)
        x - x**3/6 + x**5/120 - x**7/5040 + O(x**8)

        See Also
        ========

        HolonomicFunction.to_sequence
        """

        if _recur is None:
            recurrence = self.to_sequence()
        else:
            recurrence = _recur

        if isinstance(recurrence, tuple) and len(recurrence) == 2:
            recurrence = recurrence[0]
            constantpower = 0
        elif isinstance(recurrence, tuple) and len(recurrence) == 3:
            constantpower = recurrence[1]
            recurrence = recurrence[0]

        elif len(recurrence) == 1 and len(recurrence[0]) == 2:
            recurrence = recurrence[0][0]
            constantpower = 0
        elif len(recurrence) == 1 and len(recurrence[0]) == 3:
            constantpower = recurrence[0][1]
            recurrence = recurrence[0][0]
        else:
            return [self.series(_recur=i) for i in recurrence]

        n = n - int(constantpower)
        l = len(recurrence.u0) - 1
        k = recurrence.recurrence.order
        x = self.x
        x0 = self.x0
        seq_dmp = recurrence.recurrence.listofpoly
        R = recurrence.recurrence.parent.base
        K = R.get_field()
        seq = [K.new(j.to_list()) for j in seq_dmp]
        sub = [-seq[i] / seq[k] for i in range(k)]
        sol = list(recurrence.u0)

        if l + 1 < n:
            # use the initial conditions to find the next term
            for i in range(l + 1 - k, n - k):
                coeff = sum((DMFsubs(sub[j], i) * sol[i + j]
                            for j in range(k) if i + j >= 0), start=S.Zero)
                sol.append(coeff)

        if coefficient:
            return sol

        ser = sum((x**(i + constantpower) * j for i, j in enumerate(sol)),
                  start=S.Zero)
        if order:
            ser += Order(x**(n + int(constantpower)), x)
        if x0 != 0:
            return ser.subs(x, x - x0)
        return ser

    def _indicial(self):
        """
        Computes roots of the Indicial equation.
        """

        if self.x0 != 0:
            return self.shift_x(self.x0)._indicial()

        list_coeff = self.annihilator.listofpoly
        R = self.annihilator.parent.base
        x = self.x
        s = R.zero
        y = R.one

        def _pole_degree(poly):
            root_all = roots(R.to_sympy(poly), x, filter='Z')
            if 0 in root_all.keys():
                return root_all[0]
            else:
                return 0

        degree = max(j.degree() for j in list_coeff)
        inf = 10 * (max(1, degree) + max(1, self.annihilator.order))

        deg = lambda q: inf if q.is_zero else _pole_degree(q)
        b = min(deg(q) - j for j, q in enumerate(list_coeff))

        for i, j in enumerate(list_coeff):
            listofdmp = j.all_coeffs()
            degree = len(listofdmp) - 1
            if 0 <= i + b <= degree:
                s = s + listofdmp[degree - i - b] * y
            y *= R.from_sympy(x - i)

        return roots(R.to_sympy(s), x)

    def evalf(self, points, method='RK4', h=0.05, derivatives=False):
        r"""
        Finds numerical value of a holonomic function using numerical methods.
        (RK4 by default). A set of points (real or complex) must be provided
        which will be the path for the numerical integration.

        Explanation
        ===========

        The path should be given as a list :math:`[x_1, x_2, \dots x_n]`. The numerical
        values will be computed at each point in this order
        :math:`x_1 \rightarrow x_2 \rightarrow x_3 \dots \rightarrow x_n`.

        Returns values of the function at :math:`x_1, x_2, \dots x_n` in a list.

        Examples
        ========

        >>> from sympy.holonomic.holonomic import HolonomicFunction, DifferentialOperators
        >>> from sympy import QQ
        >>> from sympy import symbols
        >>> x = symbols('x')
        >>> R, Dx = DifferentialOperators(QQ.old_poly_ring(x),'Dx')

        A straight line on the real axis from (0 to 1)

        >>> r = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

        Runge-Kutta 4th order on e^x from 0.1 to 1.
        Exact solution at 1 is 2.71828182845905

        >>> HolonomicFunction(Dx - 1, x, 0, [1]).evalf(r)
        [1.10517083333333, 1.22140257085069, 1.34985849706254, 1.49182424008069,
        1.64872063859684, 1.82211796209193, 2.01375162659678, 2.22553956329232,
        2.45960141378007, 2.71827974413517]

        Euler's method for the same

        >>> HolonomicFunction(Dx - 1, x, 0, [1]).evalf(r, method='Euler')
        [1.1, 1.21, 1.331, 1.4641, 1.61051, 1.771561, 1.9487171, 2.14358881,
        2.357947691, 2.5937424601]

        One can also observe that the value obtained using Runge-Kutta 4th order
        is much more accurate than Euler's method.
        """

        from sympy.holonomic.numerical import _evalf
        lp = False

        # if a point `b` is given instead of a mesh
        if not hasattr(points, "__iter__"):
            lp = True
            b = S(points)
            if self.x0 == b:
                return _evalf(self, [b], method=method, derivatives=derivatives)[-1]

            if not b.is_Number:
                raise NotImplementedError

            a = self.x0
            if a > b:
                h = -h
            n = int((b - a) / h)
            points = [a + h]
            for i in range(n - 1):
                points.append(points[-1] + h)

        for i in roots(self.annihilator.parent.base.to_sympy(self.annihilator.listofpoly[-1]), self.x):
            if i == self.x0 or i in points:
                raise SingularityError(self, i)

        if lp:
            return _evalf(self, points, method=method, derivatives=derivatives)[-1]
        return _evalf(self, points, method=method, derivatives=derivatives)

    def change_x(self, z):
        """
        Changes only the variable of Holonomic Function, for internal
        purposes. For composition use HolonomicFunction.composition()
        """

        dom = self.annihilator.parent.base.dom
        R = dom.old_poly_ring(z)
        parent, _ = DifferentialOperators(R, 'Dx')
        sol = [R(j.to_list()) for j in self.annihilator.listofpoly]
        sol =  DifferentialOperator(sol, parent)
        return HolonomicFunction(sol, z, self.x0, self.y0)

    def shift_x(self, a):
        """
        Substitute `x + a` for `x`.
        """

        x = self.x
        listaftershift = self.annihilator.listofpoly
        base = self.annihilator.parent.base

        sol = [base.from_sympy(base.to_sympy(i).subs(x, x + a)) for i in listaftershift]
        sol = DifferentialOperator(sol, self.annihilator.parent)
        x0 = self.x0 - a
        if not self._have_init_cond():
            return HolonomicFunction(sol, x)
        return HolonomicFunction(sol, x, x0, self.y0)

    def to_hyper(self, as_list=False, _recur=None):
        r"""
        Returns a hypergeometric function (or linear combination of them)
        representing the given holonomic function.

        Explanation
        ===========

        Returns an answer of the form:
        `a_1 \cdot x^{b_1} \cdot{hyper()} + a_2 \cdot x^{b_2} \cdot{hyper()} \dots`

        This is very useful as one can now use ``hyperexpand`` to find the
        symbolic expressions/functions.

        Examples
        ========

        >>> from sympy.holonomic.holonomic import HolonomicFunction, DifferentialOperators
        >>> from sympy import ZZ
        >>> from sympy import symbols
        >>> x = symbols('x')
        >>> R, Dx = DifferentialOperators(ZZ.old_poly_ring(x),'Dx')
        >>> # sin(x)
        >>> HolonomicFunction(Dx**2 + 1, x, 0, [0, 1]).to_hyper()
        x*hyper((), (3/2,), -x**2/4)
        >>> # exp(x)
        >>> HolonomicFunction(Dx - 1, x, 0, [1]).to_hyper()
        hyper((), (), x)

        See Also
        ========

        from_hyper, from_meijerg
        """

        if _recur is None:
            recurrence = self.to_sequence()
        else:
            recurrence = _recur

        if isinstance(recurrence, tuple) and len(recurrence) == 2:
            smallest_n = recurrence[1]
            recurrence = recurrence[0]
            constantpower = 0
        elif isinstance(recurrence, tuple) and len(recurrence) == 3:
            smallest_n = recurrence[2]
            constantpower = recurrence[1]
            recurrence = recurrence[0]
        elif len(recurrence) == 1 and len(recurrence[0]) == 2:
            smallest_n = recurrence[0][1]
            recurrence = recurrence[0][0]
            constantpower = 0
        elif len(recurrence) == 1 and len(recurrence[0]) == 3:
            smallest_n = recurrence[0][2]
            constantpower = recurrence[0][1]
            recurrence = recurrence[0][0]
        else:
            sol = self.to_hyper(as_list=as_list, _recur=recurrence[0])
            for i in recurrence[1:]:
                sol += self.to_hyper(as_list=as_list, _recur=i)
            return sol

        u0 = recurrence.u0
        r = recurrence.recurrence
        x = self.x
        x0 = self.x0

        # order of the recurrence relation
        m = r.order

        # when no recurrence exists, and the power series have finite terms
        if m == 0:
            nonzeroterms = roots(r.parent.base.to_sympy(r.listofpoly[0]), recurrence.n, filter='R')

            sol = S.Zero
            for j, i in enumerate(nonzeroterms):

                if i < 0 or not int_valued(i):
                    continue

                i = int(i)
                if i < len(u0):
                    if isinstance(u0[i], (PolyElement, FracElement)):
                        u0[i] = u0[i].as_expr()
                    sol += u0[i] * x**i

                else:
                    sol += Symbol('C_%s' %j) * x**i

            if isinstance(sol, (PolyElement, FracElement)):
                sol = sol.as_expr() * x**constantpower
            else:
                sol = sol * x**constantpower
            if as_list:
                if x0 != 0:
                    return [(sol.subs(x, x - x0), )]
                return [(sol, )]
            if x0 != 0:
                return sol.subs(x, x - x0)
            return sol

        if smallest_n + m > len(u0):
            raise NotImplementedError("Can't compute sufficient Initial Conditions")

        # check if the recurrence represents a hypergeometric series
        if any(i != r.parent.base.zero for i in r.listofpoly[1:-1]):
            raise NotHyperSeriesError(self, self.x0)

        a = r.listofpoly[0]
        b = r.listofpoly[-1]

        # the constant multiple of argument of hypergeometric function
        if isinstance(a.LC(), (PolyElement, FracElement)):
            c = - (S(a.LC().as_expr()) * m**(a.degree())) / (S(b.LC().as_expr()) * m**(b.degree()))
        else:
            c = - (S(a.LC()) * m**(a.degree())) / (S(b.LC()) * m**(b.degree()))

        sol = 0

        arg1 = roots(r.parent.base.to_sympy(a), recurrence.n)
        arg2 = roots(r.parent.base.to_sympy(b), recurrence.n)

        # iterate through the initial conditions to find
        # the hypergeometric representation of the given
        # function.
        # The answer will be a linear combination
        # of different hypergeometric series which satisfies
        # the recurrence.
        if as_list:
            listofsol = []
        for i in range(smallest_n + m):

            # if the recurrence relation doesn't hold for `n = i`,
            # then a Hypergeometric representation doesn't exist.
            # add the algebraic term a * x**i to the solution,
            # where a is u0[i]
            if i < smallest_n:
                if as_list:
                    listofsol.append(((S(u0[i]) * x**(i+constantpower)).subs(x, x-x0), ))
                else:
                    sol += S(u0[i]) * x**i
                continue

            # if the coefficient u0[i] is zero, then the
            # independent hypergeomtric series starting with
            # x**i is not a part of the answer.
            if S(u0[i]) == 0:
                continue

            ap = []
            bq = []

            # substitute m * n + i for n
            for k in ordered(arg1.keys()):
                ap.extend([nsimplify((i - k) / m)] * arg1[k])

            for k in ordered(arg2.keys()):
                bq.extend([nsimplify((i - k) / m)] * arg2[k])

            # convention of (k + 1) in the denominator
            if 1 in bq:
                bq.remove(1)
            else:
                ap.append(1)
            if as_list:
                listofsol.append(((S(u0[i])*x**(i+constantpower)).subs(x, x-x0), (hyper(ap, bq, c*x**m)).subs(x, x-x0)))
            else:
                sol += S(u0[i]) * hyper(ap, bq, c * x**m) * x**i
        if as_list:
            return listofsol
        sol = sol * x**constantpower
        if x0 != 0:
            return sol.subs(x, x - x0)

        return sol

    def to_expr(self):
        """
        Converts a Holonomic Function back to elementary functions.

        Examples
        ========

        >>> from sympy.holonomic.holonomic import HolonomicFunction, DifferentialOperators
        >>> from sympy import ZZ
        >>> from sympy import symbols, S
        >>> x = symbols('x')
        >>> R, Dx = DifferentialOperators(ZZ.old_poly_ring(x),'Dx')
        >>> HolonomicFunction(x**2*Dx**2 + x*Dx + (x**2 - 1), x, 0, [0, S(1)/2]).to_expr()
        besselj(1, x)
        >>> HolonomicFunction((1 + x)*Dx**3 + Dx**2, x, 0, [1, 1, 1]).to_expr()
        x*log(x + 1) + log(x + 1) + 1

        """

        return hyperexpand(self.to_hyper()).simplify()

    def change_ics(self, b, lenics=None):
        """
        Changes the point `x0` to ``b`` for initial conditions.

        Examples
        ========

        >>> from sympy.holonomic import expr_to_holonomic
        >>> from sympy import symbols, sin, exp
        >>> x = symbols('x')

        >>> expr_to_holonomic(sin(x)).change_ics(1)
        HolonomicFunction((1) + (1)*Dx**2, x, 1, [sin(1), cos(1)])

        >>> expr_to_holonomic(exp(x)).change_ics(2)
        HolonomicFunction((-1) + (1)*Dx, x, 2, [exp(2)])
        """

        symbolic = True

        if lenics is None and len(self.y0) > self.annihilator.order:
            lenics = len(self.y0)
        dom = self.annihilator.parent.base.domain

        try:
            sol = expr_to_holonomic(self.to_expr(), x=self.x, x0=b, lenics=lenics, domain=dom)
        except (NotPowerSeriesError, NotHyperSeriesError):
            symbolic = False

        if symbolic and sol.x0 == b:
            return sol

        y0 = self.evalf(b, derivatives=True)
        return HolonomicFunction(self.annihilator, self.x, b, y0)

    def to_meijerg(self):
        """
        Returns a linear combination of Meijer G-functions.

        Examples
        ========

        >>> from sympy.holonomic import expr_to_holonomic
        >>> from sympy import sin, cos, hyperexpand, log, symbols
        >>> x = symbols('x')
        >>> hyperexpand(expr_to_holonomic(cos(x) + sin(x)).to_meijerg())
        sin(x) + cos(x)
        >>> hyperexpand(expr_to_holonomic(log(x)).to_meijerg()).simplify()
        log(x)

        See Also
        ========

        to_hyper
        """

        # convert to hypergeometric first
        rep = self.to_hyper(as_list=True)
        sol = S.Zero

        for i in rep:
            if len(i) == 1:
                sol += i[0]

            elif len(i) == 2:
                sol += i[0] * _hyper_to_meijerg(i[1])

        return sol


def from_hyper(func, x0=0, evalf=False):
    r"""
    Converts a hypergeometric function to holonomic.
    ``func`` is the Hypergeometric Function and ``x0`` is the point at
    which initial conditions are required.

    Examples
    ========

    >>> from sympy.holonomic.holonomic import from_hyper
    >>> from sympy import symbols, hyper, S
    >>> x = symbols('x')
    >>> from_hyper(hyper([], [S(3)/2], x**2/4))
    HolonomicFunction((-x) + (2)*Dx + (x)*Dx**2, x, 1, [sinh(1), -sinh(1) + cosh(1)])
    """

    a = func.ap
    b = func.bq
    z = func.args[2]
    x = z.atoms(Symbol).pop()
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')

    # generalized hypergeometric differential equation
    xDx = x*Dx
    r1 = 1
    for ai in a:  # XXX gives sympify error if Mul is used with list of all factors
        r1 *= xDx + ai
    xDx_1 = xDx - 1
    # r2 = Mul(*([Dx] + [xDx_1 + bi for bi in b]))  # XXX gives sympify error
    r2 = Dx
    for bi in b:
        r2 *= xDx_1 + bi
    sol = r1 - r2

    simp = hyperexpand(func)

    if simp in (Infinity, NegativeInfinity):
        return HolonomicFunction(sol, x).composition(z)

    # if the function is known symbolically
    if not isinstance(simp, hyper):
        y0 = _find_conditions(simp, x, x0, sol.order, use_limit=False)
        while not y0:
            # if values don't exist at 0, then try to find initial
            # conditions at 1. If it doesn't exist at 1 too then
            # try 2 and so on.
            x0 += 1
            y0 = _find_conditions(simp, x, x0, sol.order, use_limit=False)

        return HolonomicFunction(sol, x).composition(z, x0, y0)

    if isinstance(simp, hyper):
        x0 = 1
        # use evalf if the function can't be simplified
        y0 = _find_conditions(simp, x, x0, sol.order, evalf, use_limit=False)
        while not y0:
            x0 += 1
            y0 = _find_conditions(simp, x, x0, sol.order, evalf, use_limit=False)
        return HolonomicFunction(sol, x).composition(z, x0, y0)

    return HolonomicFunction(sol, x).composition(z)


def from_meijerg(func, x0=0, evalf=False, initcond=True, domain=QQ):
    """
    Converts a Meijer G-function to Holonomic.
    ``func`` is the G-Function and ``x0`` is the point at
    which initial conditions are required.

    Examples
    ========

    >>> from sympy.holonomic.holonomic import from_meijerg
    >>> from sympy import symbols, meijerg, S
    >>> x = symbols('x')
    >>> from_meijerg(meijerg(([], []), ([S(1)/2], [0]), x**2/4))
    HolonomicFunction((1) + (1)*Dx**2, x, 0, [0, 1/sqrt(pi)])
    """

    a = func.ap
    b = func.bq
    n = len(func.an)
    m = len(func.bm)
    p = len(a)
    z = func.args[2]
    x = z.atoms(Symbol).pop()
    R, Dx = DifferentialOperators(domain.old_poly_ring(x), 'Dx')

    # compute the differential equation satisfied by the
    # Meijer G-function.
    xDx = x*Dx
    xDx1 = xDx + 1
    r1 = x*(-1)**(m + n - p)
    for ai in a:  # XXX gives sympify error if args given in list
        r1 *= xDx1 - ai
    # r2 = Mul(*[xDx - bi for bi in b])  # gives sympify error
    r2 = 1
    for bi in b:
        r2 *= xDx - bi
    sol = r1 - r2

    if not initcond:
        return HolonomicFunction(sol, x).composition(z)

    simp = hyperexpand(func)

    if simp in (Infinity, NegativeInfinity):
        return HolonomicFunction(sol, x).composition(z)

    # computing initial conditions
    if not isinstance(simp, meijerg):
        y0 = _find_conditions(simp, x, x0, sol.order, use_limit=False)
        while not y0:
            x0 += 1
            y0 = _find_conditions(simp, x, x0, sol.order, use_limit=False)

        return HolonomicFunction(sol, x).composition(z, x0, y0)

    if isinstance(simp, meijerg):
        x0 = 1
        y0 = _find_conditions(simp, x, x0, sol.order, evalf, use_limit=False)
        while not y0:
            x0 += 1
            y0 = _find_conditions(simp, x, x0, sol.order, evalf, use_limit=False)

        return HolonomicFunction(sol, x).composition(z, x0, y0)

    return HolonomicFunction(sol, x).composition(z)


x_1 = Dummy('x_1')
_lookup_table = None
domain_for_table = None
from sympy.integrals.meijerint import _mytype


def expr_to_holonomic(func, x=None, x0=0, y0=None, lenics=None, domain=None, initcond=True):
    """
    Converts a function or an expression to a holonomic function.

    Parameters
    ==========

    func:
        The expression to be converted.
    x:
        variable for the function.
    x0:
        point at which initial condition must be computed.
    y0:
        One can optionally provide initial condition if the method
        is not able to do it automatically.
    lenics:
        Number of terms in the initial condition. By default it is
        equal to the order of the annihilator.
    domain:
        Ground domain for the polynomials in ``x`` appearing as coefficients
        in the annihilator.
    initcond:
        Set it false if you do not want the initial conditions to be computed.

    Examples
    ========

    >>> from sympy.holonomic.holonomic import expr_to_holonomic
    >>> from sympy import sin, exp, symbols
    >>> x = symbols('x')
    >>> expr_to_holonomic(sin(x))
    HolonomicFunction((1) + (1)*Dx**2, x, 0, [0, 1])
    >>> expr_to_holonomic(exp(x))
    HolonomicFunction((-1) + (1)*Dx, x, 0, [1])

    See Also
    ========

    sympy.integrals.meijerint._rewrite1, _convert_poly_rat_alg, _create_table
    """
    func = sympify(func)
    syms = func.free_symbols

    if not x:
        if len(syms) == 1:
            x= syms.pop()
        else:
            raise ValueError("Specify the variable for the function")
    elif x in syms:
        syms.remove(x)

    extra_syms = list(syms)

    if domain is None:
        if func.has(Float):
            domain = RR
        else:
            domain = QQ
        if len(extra_syms) != 0:
            domain = domain[extra_syms].get_field()

    # try to convert if the function is polynomial or rational
    solpoly = _convert_poly_rat_alg(func, x, x0=x0, y0=y0, lenics=lenics, domain=domain, initcond=initcond)
    if solpoly:
        return solpoly

    # create the lookup table
    global _lookup_table, domain_for_table
    if not _lookup_table:
        domain_for_table = domain
        _lookup_table = {}
        _create_table(_lookup_table, domain=domain)
    elif domain != domain_for_table:
        domain_for_table = domain
        _lookup_table = {}
        _create_table(_lookup_table, domain=domain)

    # use the table directly to convert to Holonomic
    if func.is_Function:
        f = func.subs(x, x_1)
        t = _mytype(f, x_1)
        if t in _lookup_table:
            l = _lookup_table[t]
            sol = l[0][1].change_x(x)
        else:
            sol = _convert_meijerint(func, x, initcond=False, domain=domain)
            if not sol:
                raise NotImplementedError
            if y0:
                sol.y0 = y0
            if y0 or not initcond:
                sol.x0 = x0
                return sol
            if not lenics:
                lenics = sol.annihilator.order
            _y0 = _find_conditions(func, x, x0, lenics)
            while not _y0:
                x0 += 1
                _y0 = _find_conditions(func, x, x0, lenics)
            return HolonomicFunction(sol.annihilator, x, x0, _y0)

        if y0 or not initcond:
            sol = sol.composition(func.args[0])
            if y0:
                sol.y0 = y0
            sol.x0 = x0
            return sol
        if not lenics:
            lenics = sol.annihilator.order

        _y0 = _find_conditions(func, x, x0, lenics)
        while not _y0:
            x0 += 1
            _y0 = _find_conditions(func, x, x0, lenics)
        return sol.composition(func.args[0], x0, _y0)

    # iterate through the expression recursively
    args = func.args
    f = func.func
    sol = expr_to_holonomic(args[0], x=x, initcond=False, domain=domain)

    if f is Add:
        for i in range(1, len(args)):
            sol += expr_to_holonomic(args[i], x=x, initcond=False, domain=domain)

    elif f is Mul:
        for i in range(1, len(args)):
            sol *= expr_to_holonomic(args[i], x=x, initcond=False, domain=domain)

    elif f is Pow:
        sol = sol**args[1]
    sol.x0 = x0
    if not sol:
        raise NotImplementedError
    if y0:
        sol.y0 = y0
    if y0 or not initcond:
        return sol
    if sol.y0:
        return sol
    if not lenics:
        lenics = sol.annihilator.order
    if sol.annihilator.is_singular(x0):
        r = sol._indicial()
        l = list(r)
        if len(r) == 1 and r[l[0]] == S.One:
            r = l[0]
            g = func / (x - x0)**r
            singular_ics = _find_conditions(g, x, x0, lenics)
            singular_ics = [j / factorial(i) for i, j in enumerate(singular_ics)]
            y0 = {r:singular_ics}
            return HolonomicFunction(sol.annihilator, x, x0, y0)

    _y0 = _find_conditions(func, x, x0, lenics)
    while not _y0:
        x0 += 1
        _y0 = _find_conditions(func, x, x0, lenics)

    return HolonomicFunction(sol.annihilator, x, x0, _y0)


## Some helper functions ##

def _normalize(list_of, parent, negative=True):
    """
    Normalize a given annihilator
    """

    num = []
    denom = []
    base = parent.base
    K = base.get_field()
    lcm_denom = base.from_sympy(S.One)
    list_of_coeff = []

    # convert polynomials to the elements of associated
    # fraction field
    for i, j in enumerate(list_of):
        if isinstance(j, base.dtype):
            list_of_coeff.append(K.new(j.to_list()))
        elif not isinstance(j, K.dtype):
            list_of_coeff.append(K.from_sympy(sympify(j)))
        else:
            list_of_coeff.append(j)

        # corresponding numerators of the sequence of polynomials
        num.append(list_of_coeff[i].numer())

        # corresponding denominators
        denom.append(list_of_coeff[i].denom())

    # lcm of denominators in the coefficients
    for i in denom:
        lcm_denom = i.lcm(lcm_denom)

    if negative:
        lcm_denom = -lcm_denom

    lcm_denom = K.new(lcm_denom.to_list())

    # multiply the coefficients with lcm
    for i, j in enumerate(list_of_coeff):
        list_of_coeff[i] = j * lcm_denom

    gcd_numer = base((list_of_coeff[-1].numer() / list_of_coeff[-1].denom()).to_list())

    # gcd of numerators in the coefficients
    for i in num:
        gcd_numer = i.gcd(gcd_numer)

    gcd_numer = K.new(gcd_numer.to_list())

    # divide all the coefficients by the gcd
    for i, j in enumerate(list_of_coeff):
        frac_ans = j / gcd_numer
        list_of_coeff[i] = base((frac_ans.numer() / frac_ans.denom()).to_list())

    return DifferentialOperator(list_of_coeff, parent)


def _derivate_diff_eq(listofpoly, K):
    """
    Let a differential equation a0(x)y(x) + a1(x)y'(x) + ... = 0
    where a0, a1,... are polynomials or rational functions. The function
    returns b0, b1, b2... such that the differential equation
    b0(x)y(x) + b1(x)y'(x) +... = 0 is formed after differentiating the
    former equation.
    """

    sol = []
    a = len(listofpoly) - 1
    sol.append(DMFdiff(listofpoly[0], K))

    for i, j in enumerate(listofpoly[1:]):
        sol.append(DMFdiff(j, K) + listofpoly[i])

    sol.append(listofpoly[a])
    return sol


def _hyper_to_meijerg(func):
    """
    Converts a `hyper` to meijerg.
    """
    ap = func.ap
    bq = func.bq

    if any(i <= 0 and int(i) == i for i in ap):
        return hyperexpand(func)

    z = func.args[2]

    # parameters of the `meijerg` function.
    an = (1 - i for i in ap)
    anp = ()
    bm = (S.Zero, )
    bmq = (1 - i for i in bq)

    k = S.One

    for i in bq:
        k = k * gamma(i)

    for i in ap:
        k = k / gamma(i)

    return k * meijerg(an, anp, bm, bmq, -z)


def _add_lists(list1, list2):
    """Takes polynomial sequences of two annihilators a and b and returns
    the list of polynomials of sum of a and b.
    """
    if len(list1) <= len(list2):
        sol = [a + b for a, b in zip(list1, list2)] + list2[len(list1):]
    else:
        sol = [a + b for a, b in zip(list1, list2)] + list1[len(list2):]
    return sol


def _extend_y0(Holonomic, n):
    """
    Tries to find more initial conditions by substituting the initial
    value point in the differential equation.
    """

    if Holonomic.annihilator.is_singular(Holonomic.x0) or Holonomic.is_singularics() == True:
        return Holonomic.y0

    annihilator = Holonomic.annihilator
    a = annihilator.order

    listofpoly = []

    y0 = Holonomic.y0
    R = annihilator.parent.base
    K = R.get_field()

    for j in annihilator.listofpoly:
        if isinstance(j, annihilator.parent.base.dtype):
            listofpoly.append(K.new(j.to_list()))

    if len(y0) < a or n <= len(y0):
        return y0
    list_red = [-listofpoly[i] / listofpoly[a]
                for i in range(a)]
    y1 = y0[:min(len(y0), a)]
    for _ in range(n - a):
        sol = 0
        for a, b in zip(y1, list_red):
            r = DMFsubs(b, Holonomic.x0)
            if not getattr(r, 'is_finite', True):
                return y0
            if isinstance(r, (PolyElement, FracElement)):
                r = r.as_expr()
            sol += a * r
        y1.append(sol)
        list_red = _derivate_diff_eq(list_red, K)
    return y0 + y1[len(y0):]


def DMFdiff(frac, K):
    # differentiate a DMF object represented as p/q
    if not isinstance(frac, DMF):
        return frac.diff()

    p = K.numer(frac)
    q = K.denom(frac)
    sol_num = - p * q.diff() + q * p.diff()
    sol_denom = q**2
    return K((sol_num.to_list(), sol_denom.to_list()))


def DMFsubs(frac, x0, mpm=False):
    # substitute the point x0 in DMF object of the form p/q
    if not isinstance(frac, DMF):
        return frac

    p = frac.num
    q = frac.den
    sol_p = S.Zero
    sol_q = S.Zero

    if mpm:
        from mpmath import mp

    for i, j in enumerate(reversed(p)):
        if mpm:
            j = sympify(j)._to_mpmath(mp.prec)
        sol_p += j * x0**i

    for i, j in enumerate(reversed(q)):
        if mpm:
            j = sympify(j)._to_mpmath(mp.prec)
        sol_q += j * x0**i

    if isinstance(sol_p, (PolyElement, FracElement)):
        sol_p = sol_p.as_expr()
    if isinstance(sol_q, (PolyElement, FracElement)):
        sol_q = sol_q.as_expr()

    return sol_p / sol_q


def _convert_poly_rat_alg(func, x, x0=0, y0=None, lenics=None, domain=QQ, initcond=True):
    """
    Converts polynomials, rationals and algebraic functions to holonomic.
    """

    ispoly = func.is_polynomial()
    if not ispoly:
        israt = func.is_rational_function()
    else:
        israt = True

    if not (ispoly or israt):
        basepoly, ratexp = func.as_base_exp()
        if basepoly.is_polynomial() and ratexp.is_Number:
            if isinstance(ratexp, Float):
                ratexp = nsimplify(ratexp)
            m, n = ratexp.p, ratexp.q
            is_alg = True
        else:
            is_alg = False
    else:
        is_alg = True

    if not (ispoly or israt or is_alg):
        return None

    R = domain.old_poly_ring(x)
    _, Dx = DifferentialOperators(R, 'Dx')

    # if the function is constant
    if not func.has(x):
        return HolonomicFunction(Dx, x, 0, [func])

    if ispoly:
        # differential equation satisfied by polynomial
        sol = func * Dx - func.diff(x)
        sol = _normalize(sol.listofpoly, sol.parent, negative=False)
        is_singular = sol.is_singular(x0)

        # try to compute the conditions for singular points
        if y0 is None and x0 == 0 and is_singular:
            rep = R.from_sympy(func).to_list()
            for i, j in enumerate(reversed(rep)):
                if j == 0:
                    continue
                coeff = list(reversed(rep))[i:]
                indicial = i
                break
            for i, j in enumerate(coeff):
                if isinstance(j, (PolyElement, FracElement)):
                    coeff[i] = j.as_expr()
            y0 = {indicial: S(coeff)}

    elif israt:
        p, q = func.as_numer_denom()
        # differential equation satisfied by rational
        sol = p * q * Dx + p * q.diff(x) - q * p.diff(x)
        sol = _normalize(sol.listofpoly, sol.parent, negative=False)

    elif is_alg:
        sol = n * (x / m) * Dx - 1
        sol = HolonomicFunction(sol, x).composition(basepoly).annihilator
        is_singular = sol.is_singular(x0)

        # try to compute the conditions for singular points
        if y0 is None and x0 == 0 and is_singular and \
            (lenics is None or lenics <= 1):
            rep = R.from_sympy(basepoly).to_list()
            for i, j in enumerate(reversed(rep)):
                if j == 0:
                    continue
                if isinstance(j, (PolyElement, FracElement)):
                    j = j.as_expr()

                coeff = S(j)**ratexp
                indicial = S(i) * ratexp
                break
            if isinstance(coeff, (PolyElement, FracElement)):
                coeff = coeff.as_expr()
            y0 = {indicial: S([coeff])}

    if y0 or not initcond:
        return HolonomicFunction(sol, x, x0, y0)

    if not lenics:
        lenics = sol.order

    if sol.is_singular(x0):
        r = HolonomicFunction(sol, x, x0)._indicial()
        l = list(r)
        if len(r) == 1 and r[l[0]] == S.One:
            r = l[0]
            g = func / (x - x0)**r
            singular_ics = _find_conditions(g, x, x0, lenics)
            singular_ics = [j / factorial(i) for i, j in enumerate(singular_ics)]
            y0 = {r:singular_ics}
            return HolonomicFunction(sol, x, x0, y0)

    y0 = _find_conditions(func, x, x0, lenics)
    while not y0:
        x0 += 1
        y0 = _find_conditions(func, x, x0, lenics)

    return HolonomicFunction(sol, x, x0, y0)


def _convert_meijerint(func, x, initcond=True, domain=QQ):
    args = meijerint._rewrite1(func, x)

    if args:
        fac, po, g, _ = args
    else:
        return None

    # lists for sum of meijerg functions
    fac_list = [fac * i[0] for i in g]
    t = po.as_base_exp()
    s = t[1] if t[0] == x else S.Zero
    po_list = [s + i[1] for i in g]
    G_list = [i[2] for i in g]

    # finds meijerg representation of x**s * meijerg(a1 ... ap, b1 ... bq, z)
    def _shift(func, s):
        z = func.args[-1]
        if z.has(I):
            z = z.subs(exp_polar, exp)

        d = z.collect(x, evaluate=False)
        b = list(d)[0]
        a = d[b]

        t = b.as_base_exp()
        b = t[1] if t[0] == x else S.Zero
        r = s / b
        an = (i + r for i in func.args[0][0])
        ap = (i + r for i in func.args[0][1])
        bm = (i + r for i in func.args[1][0])
        bq = (i + r for i in func.args[1][1])

        return a**-r, meijerg((an, ap), (bm, bq), z)

    coeff, m = _shift(G_list[0], po_list[0])
    sol = fac_list[0] * coeff * from_meijerg(m, initcond=initcond, domain=domain)

    # add all the meijerg functions after converting to holonomic
    for i in range(1, len(G_list)):
        coeff, m = _shift(G_list[i], po_list[i])
        sol += fac_list[i] * coeff * from_meijerg(m, initcond=initcond, domain=domain)

    return sol


def _create_table(table, domain=QQ):
    """
    Creates the look-up table. For a similar implementation
    see meijerint._create_lookup_table.
    """

    def add(formula, annihilator, arg, x0=0, y0=()):
        """
        Adds a formula in the dictionary
        """
        table.setdefault(_mytype(formula, x_1), []).append((formula,
            HolonomicFunction(annihilator, arg, x0, y0)))

    R = domain.old_poly_ring(x_1)
    _, Dx = DifferentialOperators(R, 'Dx')

    # add some basic functions
    add(sin(x_1), Dx**2 + 1, x_1, 0, [0, 1])
    add(cos(x_1), Dx**2 + 1, x_1, 0, [1, 0])
    add(exp(x_1), Dx - 1, x_1, 0, 1)
    add(log(x_1), Dx + x_1*Dx**2, x_1, 1, [0, 1])

    add(erf(x_1), 2*x_1*Dx + Dx**2, x_1, 0, [0, 2/sqrt(pi)])
    add(erfc(x_1), 2*x_1*Dx + Dx**2, x_1, 0, [1, -2/sqrt(pi)])
    add(erfi(x_1), -2*x_1*Dx + Dx**2, x_1, 0, [0, 2/sqrt(pi)])

    add(sinh(x_1), Dx**2 - 1, x_1, 0, [0, 1])
    add(cosh(x_1), Dx**2 - 1, x_1, 0, [1, 0])

    add(sinc(x_1), x_1 + 2*Dx + x_1*Dx**2, x_1)

    add(Si(x_1), x_1*Dx + 2*Dx**2 + x_1*Dx**3, x_1)
    add(Ci(x_1), x_1*Dx + 2*Dx**2 + x_1*Dx**3, x_1)

    add(Shi(x_1), -x_1*Dx + 2*Dx**2 + x_1*Dx**3, x_1)


def _find_conditions(func, x, x0, order, evalf=False, use_limit=True):
    y0 = []
    for _ in range(order):
        val = func.subs(x, x0)
        if evalf:
            val = val.evalf()
        if use_limit and isinstance(val, NaN):
            val = limit(func, x, x0)
        if val.is_finite is False or isinstance(val, NaN):
            return None
        y0.append(val)
        func = func.diff(x)
    return y0
