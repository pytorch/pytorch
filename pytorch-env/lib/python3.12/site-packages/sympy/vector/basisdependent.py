from __future__ import annotations
from typing import TYPE_CHECKING

from sympy.simplify import simplify as simp, trigsimp as tsimp  # type: ignore
from sympy.core.decorators import call_highest_priority, _sympifyit
from sympy.core.assumptions import StdFactKB
from sympy.core.function import diff as df
from sympy.integrals.integrals import Integral
from sympy.polys.polytools import factor as fctr
from sympy.core import S, Add, Mul
from sympy.core.expr import Expr

if TYPE_CHECKING:
    from sympy.vector.vector import BaseVector


class BasisDependent(Expr):
    """
    Super class containing functionality common to vectors and
    dyadics.
    Named so because the representation of these quantities in
    sympy.vector is dependent on the basis they are expressed in.
    """

    zero: BasisDependentZero

    @call_highest_priority('__radd__')
    def __add__(self, other):
        return self._add_func(self, other)

    @call_highest_priority('__add__')
    def __radd__(self, other):
        return self._add_func(other, self)

    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        return self._add_func(self, -other)

    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        return self._add_func(other, -self)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        return self._mul_func(self, other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return self._mul_func(other, self)

    def __neg__(self):
        return self._mul_func(S.NegativeOne, self)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rtruediv__')
    def __truediv__(self, other):
        return self._div_helper(other)

    @call_highest_priority('__truediv__')
    def __rtruediv__(self, other):
        return TypeError("Invalid divisor for division")

    def evalf(self, n=15, subs=None, maxn=100, chop=False, strict=False, quad=None, verbose=False):
        """
        Implements the SymPy evalf routine for this quantity.

        evalf's documentation
        =====================

        """
        options = {'subs':subs, 'maxn':maxn, 'chop':chop, 'strict':strict,
                'quad':quad, 'verbose':verbose}
        vec = self.zero
        for k, v in self.components.items():
            vec += v.evalf(n, **options) * k
        return vec

    evalf.__doc__ += Expr.evalf.__doc__  # type: ignore

    n = evalf

    def simplify(self, **kwargs):
        """
        Implements the SymPy simplify routine for this quantity.

        simplify's documentation
        ========================

        """
        simp_components = [simp(v, **kwargs) * k for
                           k, v in self.components.items()]
        return self._add_func(*simp_components)

    simplify.__doc__ += simp.__doc__  # type: ignore

    def trigsimp(self, **opts):
        """
        Implements the SymPy trigsimp routine, for this quantity.

        trigsimp's documentation
        ========================

        """
        trig_components = [tsimp(v, **opts) * k for
                           k, v in self.components.items()]
        return self._add_func(*trig_components)

    trigsimp.__doc__ += tsimp.__doc__  # type: ignore

    def _eval_simplify(self, **kwargs):
        return self.simplify(**kwargs)

    def _eval_trigsimp(self, **opts):
        return self.trigsimp(**opts)

    def _eval_derivative(self, wrt):
        return self.diff(wrt)

    def _eval_Integral(self, *symbols, **assumptions):
        integral_components = [Integral(v, *symbols, **assumptions) * k
                               for k, v in self.components.items()]
        return self._add_func(*integral_components)

    def as_numer_denom(self):
        """
        Returns the expression as a tuple wrt the following
        transformation -

        expression -> a/b -> a, b

        """
        return self, S.One

    def factor(self, *args, **kwargs):
        """
        Implements the SymPy factor routine, on the scalar parts
        of a basis-dependent expression.

        factor's documentation
        ========================

        """
        fctr_components = [fctr(v, *args, **kwargs) * k for
                           k, v in self.components.items()]
        return self._add_func(*fctr_components)

    factor.__doc__ += fctr.__doc__  # type: ignore

    def as_coeff_Mul(self, rational=False):
        """Efficiently extract the coefficient of a product."""
        return (S.One, self)

    def as_coeff_add(self, *deps):
        """Efficiently extract the coefficient of a summation."""
        l = [x * self.components[x] for x in self.components]
        return 0, tuple(l)

    def diff(self, *args, **kwargs):
        """
        Implements the SymPy diff routine, for vectors.

        diff's documentation
        ========================

        """
        for x in args:
            if isinstance(x, BasisDependent):
                raise TypeError("Invalid arg for differentiation")
        diff_components = [df(v, *args, **kwargs) * k for
                           k, v in self.components.items()]
        return self._add_func(*diff_components)

    diff.__doc__ += df.__doc__  # type: ignore

    def doit(self, **hints):
        """Calls .doit() on each term in the Dyadic"""
        doit_components = [self.components[x].doit(**hints) * x
                           for x in self.components]
        return self._add_func(*doit_components)


class BasisDependentAdd(BasisDependent, Add):
    """
    Denotes sum of basis dependent quantities such that they cannot
    be expressed as base or Mul instances.
    """

    def __new__(cls, *args, **options):
        components = {}

        # Check each arg and simultaneously learn the components
        for arg in args:
            if not isinstance(arg, cls._expr_type):
                if isinstance(arg, Mul):
                    arg = cls._mul_func(*(arg.args))
                elif isinstance(arg, Add):
                    arg = cls._add_func(*(arg.args))
                else:
                    raise TypeError(str(arg) +
                                    " cannot be interpreted correctly")
            # If argument is zero, ignore
            if arg == cls.zero:
                continue
            # Else, update components accordingly
            if hasattr(arg, "components"):
                for x in arg.components:
                    components[x] = components.get(x, 0) + arg.components[x]

        temp = list(components.keys())
        for x in temp:
            if components[x] == 0:
                del components[x]

        # Handle case of zero vector
        if len(components) == 0:
            return cls.zero

        # Build object
        newargs = [x * components[x] for x in components]
        obj = super().__new__(cls, *newargs, **options)
        if isinstance(obj, Mul):
            return cls._mul_func(*obj.args)
        assumptions = {'commutative': True}
        obj._assumptions = StdFactKB(assumptions)
        obj._components = components
        obj._sys = (list(components.keys()))[0]._sys

        return obj


class BasisDependentMul(BasisDependent, Mul):
    """
    Denotes product of base- basis dependent quantity with a scalar.
    """

    def __new__(cls, *args, **options):
        from sympy.vector import Cross, Dot, Curl, Gradient
        count = 0
        measure_number = S.One
        zeroflag = False
        extra_args = []

        # Determine the component and check arguments
        # Also keep a count to ensure two vectors aren't
        # being multiplied
        for arg in args:
            if isinstance(arg, cls._zero_func):
                count += 1
                zeroflag = True
            elif arg == S.Zero:
                zeroflag = True
            elif isinstance(arg, (cls._base_func, cls._mul_func)):
                count += 1
                expr = arg._base_instance
                measure_number *= arg._measure_number
            elif isinstance(arg, cls._add_func):
                count += 1
                expr = arg
            elif isinstance(arg, (Cross, Dot, Curl, Gradient)):
                extra_args.append(arg)
            else:
                measure_number *= arg
        # Make sure incompatible types weren't multiplied
        if count > 1:
            raise ValueError("Invalid multiplication")
        elif count == 0:
            return Mul(*args, **options)
        # Handle zero vector case
        if zeroflag:
            return cls.zero

        # If one of the args was a VectorAdd, return an
        # appropriate VectorAdd instance
        if isinstance(expr, cls._add_func):
            newargs = [cls._mul_func(measure_number, x) for
                       x in expr.args]
            return cls._add_func(*newargs)

        obj = super().__new__(cls, measure_number,
                              expr._base_instance,
                              *extra_args,
                              **options)
        if isinstance(obj, Add):
            return cls._add_func(*obj.args)
        obj._base_instance = expr._base_instance
        obj._measure_number = measure_number
        assumptions = {'commutative': True}
        obj._assumptions = StdFactKB(assumptions)
        obj._components = {expr._base_instance: measure_number}
        obj._sys = expr._base_instance._sys

        return obj

    def _sympystr(self, printer):
        measure_str = printer._print(self._measure_number)
        if ('(' in measure_str or '-' in measure_str or
                '+' in measure_str):
            measure_str = '(' + measure_str + ')'
        return measure_str + '*' + printer._print(self._base_instance)


class BasisDependentZero(BasisDependent):
    """
    Class to denote a zero basis dependent instance.
    """
    components: dict['BaseVector', Expr] = {}
    _latex_form: str

    def __new__(cls):
        obj = super().__new__(cls)
        # Pre-compute a specific hash value for the zero vector
        # Use the same one always
        obj._hash = (S.Zero, cls).__hash__()
        return obj

    def __hash__(self):
        return self._hash

    @call_highest_priority('__req__')
    def __eq__(self, other):
        return isinstance(other, self._zero_func)

    __req__ = __eq__

    @call_highest_priority('__radd__')
    def __add__(self, other):
        if isinstance(other, self._expr_type):
            return other
        else:
            raise TypeError("Invalid argument types for addition")

    @call_highest_priority('__add__')
    def __radd__(self, other):
        if isinstance(other, self._expr_type):
            return other
        else:
            raise TypeError("Invalid argument types for addition")

    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        if isinstance(other, self._expr_type):
            return -other
        else:
            raise TypeError("Invalid argument types for subtraction")

    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        if isinstance(other, self._expr_type):
            return other
        else:
            raise TypeError("Invalid argument types for subtraction")

    def __neg__(self):
        return self

    def normalize(self):
        """
        Returns the normalized version of this vector.
        """
        return self

    def _sympystr(self, printer):
        return '0'
