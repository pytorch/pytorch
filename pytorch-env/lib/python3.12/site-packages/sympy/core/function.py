"""
There are three types of functions implemented in SymPy:

    1) defined functions (in the sense that they can be evaluated) like
       exp or sin; they have a name and a body:
           f = exp
    2) undefined function which have a name but no body. Undefined
       functions can be defined using a Function class as follows:
           f = Function('f')
       (the result will be a Function instance)
    3) anonymous function (or lambda function) which have a body (defined
       with dummy variables) but have no name:
           f = Lambda(x, exp(x)*x)
           f = Lambda((x, y), exp(x)*y)
    The fourth type of functions are composites, like (sin + cos)(x); these work in
    SymPy core, but are not yet part of SymPy.

    Examples
    ========

    >>> import sympy
    >>> f = sympy.Function("f")
    >>> from sympy.abc import x
    >>> f(x)
    f(x)
    >>> print(sympy.srepr(f(x).func))
    Function('f')
    >>> f(x).args
    (x,)

"""

from __future__ import annotations
from typing import Any
from collections.abc import Iterable

from .add import Add
from .basic import Basic, _atomic
from .cache import cacheit
from .containers import Tuple, Dict
from .decorators import _sympifyit
from .evalf import pure_complex
from .expr import Expr, AtomicExpr
from .logic import fuzzy_and, fuzzy_or, fuzzy_not, FuzzyBool
from .mul import Mul
from .numbers import Rational, Float, Integer
from .operations import LatticeOp
from .parameters import global_parameters
from .rules import Transform
from .singleton import S
from .sympify import sympify, _sympify

from .sorting import default_sort_key, ordered
from sympy.utilities.exceptions import (sympy_deprecation_warning,
                                        SymPyDeprecationWarning, ignore_warnings)
from sympy.utilities.iterables import (has_dups, sift, iterable,
    is_sequence, uniq, topological_sort)
from sympy.utilities.lambdify import MPMATH_TRANSLATIONS
from sympy.utilities.misc import as_int, filldedent, func_name

import mpmath
from mpmath.libmp.libmpf import prec_to_dps

import inspect
from collections import Counter

def _coeff_isneg(a):
    """Return True if the leading Number is negative.

    Examples
    ========

    >>> from sympy.core.function import _coeff_isneg
    >>> from sympy import S, Symbol, oo, pi
    >>> _coeff_isneg(-3*pi)
    True
    >>> _coeff_isneg(S(3))
    False
    >>> _coeff_isneg(-oo)
    True
    >>> _coeff_isneg(Symbol('n', negative=True)) # coeff is 1
    False

    For matrix expressions:

    >>> from sympy import MatrixSymbol, sqrt
    >>> A = MatrixSymbol("A", 3, 3)
    >>> _coeff_isneg(-sqrt(2)*A)
    True
    >>> _coeff_isneg(sqrt(2)*A)
    False
    """

    if a.is_MatMul:
        a = a.args[0]
    if a.is_Mul:
        a = a.args[0]
    return a.is_Number and a.is_extended_negative


class PoleError(Exception):
    pass


class ArgumentIndexError(ValueError):
    def __str__(self):
        return ("Invalid operation with argument number %s for Function %s" %
               (self.args[1], self.args[0]))


class BadSignatureError(TypeError):
    '''Raised when a Lambda is created with an invalid signature'''
    pass


class BadArgumentsError(TypeError):
    '''Raised when a Lambda is called with an incorrect number of arguments'''
    pass


# Python 3 version that does not raise a Deprecation warning
def arity(cls):
    """Return the arity of the function if it is known, else None.

    Explanation
    ===========

    When default values are specified for some arguments, they are
    optional and the arity is reported as a tuple of possible values.

    Examples
    ========

    >>> from sympy import arity, log
    >>> arity(lambda x: x)
    1
    >>> arity(log)
    (1, 2)
    >>> arity(lambda *x: sum(x)) is None
    True
    """
    eval_ = getattr(cls, 'eval', cls)

    parameters = inspect.signature(eval_).parameters.items()
    if [p for _, p in parameters if p.kind == p.VAR_POSITIONAL]:
        return
    p_or_k = [p for _, p in parameters if p.kind == p.POSITIONAL_OR_KEYWORD]
    # how many have no default and how many have a default value
    no, yes = map(len, sift(p_or_k,
        lambda p:p.default == p.empty, binary=True))
    return no if not yes else tuple(range(no, no + yes + 1))

class FunctionClass(type):
    """
    Base class for function classes. FunctionClass is a subclass of type.

    Use Function('<function name>' [ , signature ]) to create
    undefined function classes.
    """
    _new = type.__new__

    def __init__(cls, *args, **kwargs):
        # honor kwarg value or class-defined value before using
        # the number of arguments in the eval function (if present)
        nargs = kwargs.pop('nargs', cls.__dict__.get('nargs', arity(cls)))
        if nargs is None and 'nargs' not in cls.__dict__:
            for supcls in cls.__mro__:
                if hasattr(supcls, '_nargs'):
                    nargs = supcls._nargs
                    break
                else:
                    continue

        # Canonicalize nargs here; change to set in nargs.
        if is_sequence(nargs):
            if not nargs:
                raise ValueError(filldedent('''
                    Incorrectly specified nargs as %s:
                    if there are no arguments, it should be
                    `nargs = 0`;
                    if there are any number of arguments,
                    it should be
                    `nargs = None`''' % str(nargs)))
            nargs = tuple(ordered(set(nargs)))
        elif nargs is not None:
            nargs = (as_int(nargs),)
        cls._nargs = nargs

        # When __init__ is called from UndefinedFunction it is called with
        # just one arg but when it is called from subclassing Function it is
        # called with the usual (name, bases, namespace) type() signature.
        if len(args) == 3:
            namespace = args[2]
            if 'eval' in namespace and not isinstance(namespace['eval'], classmethod):
                raise TypeError("eval on Function subclasses should be a class method (defined with @classmethod)")

    @property
    def __signature__(self):
        """
        Allow Python 3's inspect.signature to give a useful signature for
        Function subclasses.
        """
        # Python 3 only, but backports (like the one in IPython) still might
        # call this.
        try:
            from inspect import signature
        except ImportError:
            return None

        # TODO: Look at nargs
        return signature(self.eval)

    @property
    def free_symbols(self):
        return set()

    @property
    def xreplace(self):
        # Function needs args so we define a property that returns
        # a function that takes args...and then use that function
        # to return the right value
        return lambda rule, **_: rule.get(self, self)

    @property
    def nargs(self):
        """Return a set of the allowed number of arguments for the function.

        Examples
        ========

        >>> from sympy import Function
        >>> f = Function('f')

        If the function can take any number of arguments, the set of whole
        numbers is returned:

        >>> Function('f').nargs
        Naturals0

        If the function was initialized to accept one or more arguments, a
        corresponding set will be returned:

        >>> Function('f', nargs=1).nargs
        {1}
        >>> Function('f', nargs=(2, 1)).nargs
        {1, 2}

        The undefined function, after application, also has the nargs
        attribute; the actual number of arguments is always available by
        checking the ``args`` attribute:

        >>> f = Function('f')
        >>> f(1).nargs
        Naturals0
        >>> len(f(1).args)
        1
        """
        from sympy.sets.sets import FiniteSet
        # XXX it would be nice to handle this in __init__ but there are import
        # problems with trying to import FiniteSet there
        return FiniteSet(*self._nargs) if self._nargs else S.Naturals0

    def _valid_nargs(self, n : int) -> bool:
        """ Return True if the specified integer is a valid number of arguments

        The number of arguments n is guaranteed to be an integer and positive

        """
        if self._nargs:
            return n in self._nargs

        nargs = self.nargs
        return nargs is S.Naturals0 or n in nargs

    def __repr__(cls):
        return cls.__name__


class Application(Basic, metaclass=FunctionClass):
    """
    Base class for applied functions.

    Explanation
    ===========

    Instances of Application represent the result of applying an application of
    any type to any object.
    """

    is_Function = True

    @cacheit
    def __new__(cls, *args, **options):
        from sympy.sets.fancysets import Naturals0
        from sympy.sets.sets import FiniteSet

        args = list(map(sympify, args))
        evaluate = options.pop('evaluate', global_parameters.evaluate)
        # WildFunction (and anything else like it) may have nargs defined
        # and we throw that value away here
        options.pop('nargs', None)

        if options:
            raise ValueError("Unknown options: %s" % options)

        if evaluate:
            evaluated = cls.eval(*args)
            if evaluated is not None:
                return evaluated

        obj = super().__new__(cls, *args, **options)

        # make nargs uniform here
        sentinel = object()
        objnargs = getattr(obj, "nargs", sentinel)
        if objnargs is not sentinel:
            # things passing through here:
            #  - functions subclassed from Function (e.g. myfunc(1).nargs)
            #  - functions like cos(1).nargs
            #  - AppliedUndef with given nargs like Function('f', nargs=1)(1).nargs
            # Canonicalize nargs here
            if is_sequence(objnargs):
                nargs = tuple(ordered(set(objnargs)))
            elif objnargs is not None:
                nargs = (as_int(objnargs),)
            else:
                nargs = None
        else:
            # things passing through here:
            #  - WildFunction('f').nargs
            #  - AppliedUndef with no nargs like Function('f')(1).nargs
            nargs = obj._nargs  # note the underscore here
        # convert to FiniteSet
        obj.nargs = FiniteSet(*nargs) if nargs else Naturals0()
        return obj

    @classmethod
    def eval(cls, *args):
        """
        Returns a canonical form of cls applied to arguments args.

        Explanation
        ===========

        The ``eval()`` method is called when the class ``cls`` is about to be
        instantiated and it should return either some simplified instance
        (possible of some other class), or if the class ``cls`` should be
        unmodified, return None.

        Examples of ``eval()`` for the function "sign"

        .. code-block:: python

            @classmethod
            def eval(cls, arg):
                if arg is S.NaN:
                    return S.NaN
                if arg.is_zero: return S.Zero
                if arg.is_positive: return S.One
                if arg.is_negative: return S.NegativeOne
                if isinstance(arg, Mul):
                    coeff, terms = arg.as_coeff_Mul(rational=True)
                    if coeff is not S.One:
                        return cls(coeff) * cls(terms)

        """
        return

    @property
    def func(self):
        return self.__class__

    def _eval_subs(self, old, new):
        if (old.is_Function and new.is_Function and
            callable(old) and callable(new) and
            old == self.func and len(self.args) in new.nargs):
            return new(*[i._subs(old, new) for i in self.args])


class Function(Application, Expr):
    r"""
    Base class for applied mathematical functions.

    It also serves as a constructor for undefined function classes.

    See the :ref:`custom-functions` guide for details on how to subclass
    ``Function`` and what methods can be defined.

    Examples
    ========

    **Undefined Functions**

    To create an undefined function, pass a string of the function name to
    ``Function``.

    >>> from sympy import Function, Symbol
    >>> x = Symbol('x')
    >>> f = Function('f')
    >>> g = Function('g')(x)
    >>> f
    f
    >>> f(x)
    f(x)
    >>> g
    g(x)
    >>> f(x).diff(x)
    Derivative(f(x), x)
    >>> g.diff(x)
    Derivative(g(x), x)

    Assumptions can be passed to ``Function`` the same as with a
    :class:`~.Symbol`. Alternatively, you can use a ``Symbol`` with
    assumptions for the function name and the function will inherit the name
    and assumptions associated with the ``Symbol``:

    >>> f_real = Function('f', real=True)
    >>> f_real(x).is_real
    True
    >>> f_real_inherit = Function(Symbol('f', real=True))
    >>> f_real_inherit(x).is_real
    True

    Note that assumptions on a function are unrelated to the assumptions on
    the variables it is called on. If you want to add a relationship, subclass
    ``Function`` and define custom assumptions handler methods. See the
    :ref:`custom-functions-assumptions` section of the :ref:`custom-functions`
    guide for more details.

    **Custom Function Subclasses**

    The :ref:`custom-functions` guide has several
    :ref:`custom-functions-complete-examples` of how to subclass ``Function``
    to create a custom function.

    """

    @property
    def _diff_wrt(self):
        return False

    @cacheit
    def __new__(cls, *args, **options):
        # Handle calls like Function('f')
        if cls is Function:
            return UndefinedFunction(*args, **options)

        n = len(args)

        if not cls._valid_nargs(n):
            # XXX: exception message must be in exactly this format to
            # make it work with NumPy's functions like vectorize(). See,
            # for example, https://github.com/numpy/numpy/issues/1697.
            # The ideal solution would be just to attach metadata to
            # the exception and change NumPy to take advantage of this.
            temp = ('%(name)s takes %(qual)s %(args)s '
                   'argument%(plural)s (%(given)s given)')
            raise TypeError(temp % {
                'name': cls,
                'qual': 'exactly' if len(cls.nargs) == 1 else 'at least',
                'args': min(cls.nargs),
                'plural': 's'*(min(cls.nargs) != 1),
                'given': n})

        evaluate = options.get('evaluate', global_parameters.evaluate)
        result = super().__new__(cls, *args, **options)
        if evaluate and isinstance(result, cls) and result.args:
            _should_evalf = [cls._should_evalf(a) for a in result.args]
            pr2 = min(_should_evalf)
            if pr2 > 0:
                pr = max(_should_evalf)
                result = result.evalf(prec_to_dps(pr))

        return _sympify(result)

    @classmethod
    def _should_evalf(cls, arg):
        """
        Decide if the function should automatically evalf().

        Explanation
        ===========

        By default (in this implementation), this happens if (and only if) the
        ARG is a floating point number (including complex numbers).
        This function is used by __new__.

        Returns the precision to evalf to, or -1 if it should not evalf.
        """
        if arg.is_Float:
            return arg._prec
        if not arg.is_Add:
            return -1
        m = pure_complex(arg)
        if m is None:
            return -1
        # the elements of m are of type Number, so have a _prec
        return max(m[0]._prec, m[1]._prec)

    @classmethod
    def class_key(cls):
        from sympy.sets.fancysets import Naturals0
        funcs = {
            'exp': 10,
            'log': 11,
            'sin': 20,
            'cos': 21,
            'tan': 22,
            'cot': 23,
            'sinh': 30,
            'cosh': 31,
            'tanh': 32,
            'coth': 33,
            'conjugate': 40,
            're': 41,
            'im': 42,
            'arg': 43,
        }
        name = cls.__name__

        try:
            i = funcs[name]
        except KeyError:
            i = 0 if isinstance(cls.nargs, Naturals0) else 10000

        return 4, i, name

    def _eval_evalf(self, prec):

        def _get_mpmath_func(fname):
            """Lookup mpmath function based on name"""
            if isinstance(self, AppliedUndef):
                # Shouldn't lookup in mpmath but might have ._imp_
                return None

            if not hasattr(mpmath, fname):
                fname = MPMATH_TRANSLATIONS.get(fname, None)
                if fname is None:
                    return None
            return getattr(mpmath, fname)

        _eval_mpmath = getattr(self, '_eval_mpmath', None)
        if _eval_mpmath is None:
            func = _get_mpmath_func(self.func.__name__)
            args = self.args
        else:
            func, args = _eval_mpmath()

        # Fall-back evaluation
        if func is None:
            imp = getattr(self, '_imp_', None)
            if imp is None:
                return None
            try:
                return Float(imp(*[i.evalf(prec) for i in self.args]), prec)
            except (TypeError, ValueError):
                return None

        # Convert all args to mpf or mpc
        # Convert the arguments to *higher* precision than requested for the
        # final result.
        # XXX + 5 is a guess, it is similar to what is used in evalf.py. Should
        #     we be more intelligent about it?
        try:
            args = [arg._to_mpmath(prec + 5) for arg in args]
            def bad(m):
                from mpmath import mpf, mpc
                # the precision of an mpf value is the last element
                # if that is 1 (and m[1] is not 1 which would indicate a
                # power of 2), then the eval failed; so check that none of
                # the arguments failed to compute to a finite precision.
                # Note: An mpc value has two parts, the re and imag tuple;
                # check each of those parts, too. Anything else is allowed to
                # pass
                if isinstance(m, mpf):
                    m = m._mpf_
                    return m[1] !=1 and m[-1] == 1
                elif isinstance(m, mpc):
                    m, n = m._mpc_
                    return m[1] !=1 and m[-1] == 1 and \
                        n[1] !=1 and n[-1] == 1
                else:
                    return False
            if any(bad(a) for a in args):
                raise ValueError  # one or more args failed to compute with significance
        except ValueError:
            return

        with mpmath.workprec(prec):
            v = func(*args)

        return Expr._from_mpmath(v, prec)

    def _eval_derivative(self, s):
        # f(x).diff(s) -> x.diff(s) * f.fdiff(1)(s)
        i = 0
        l = []
        for a in self.args:
            i += 1
            da = a.diff(s)
            if da.is_zero:
                continue
            try:
                df = self.fdiff(i)
            except ArgumentIndexError:
                df = Function.fdiff(self, i)
            l.append(df * da)
        return Add(*l)

    def _eval_is_commutative(self):
        return fuzzy_and(a.is_commutative for a in self.args)

    def _eval_is_meromorphic(self, x, a):
        if not self.args:
            return True
        if any(arg.has(x) for arg in self.args[1:]):
            return False

        arg = self.args[0]
        if not arg._eval_is_meromorphic(x, a):
            return None

        return fuzzy_not(type(self).is_singular(arg.subs(x, a)))

    _singularities: FuzzyBool | tuple[Expr, ...] = None

    @classmethod
    def is_singular(cls, a):
        """
        Tests whether the argument is an essential singularity
        or a branch point, or the functions is non-holomorphic.
        """
        ss = cls._singularities
        if ss in (True, None, False):
            return ss

        return fuzzy_or(a.is_infinite if s is S.ComplexInfinity
                        else (a - s).is_zero for s in ss)

    def as_base_exp(self):
        """
        Returns the method as the 2-tuple (base, exponent).
        """
        return self, S.One

    def _eval_aseries(self, n, args0, x, logx):
        """
        Compute an asymptotic expansion around args0, in terms of self.args.
        This function is only used internally by _eval_nseries and should not
        be called directly; derived classes can overwrite this to implement
        asymptotic expansions.
        """
        raise PoleError(filldedent('''
            Asymptotic expansion of %s around %s is
            not implemented.''' % (type(self), args0)))

    def _eval_nseries(self, x, n, logx, cdir=0):
        """
        This function does compute series for multivariate functions,
        but the expansion is always in terms of *one* variable.

        Examples
        ========

        >>> from sympy import atan2
        >>> from sympy.abc import x, y
        >>> atan2(x, y).series(x, n=2)
        atan2(0, y) + x/y + O(x**2)
        >>> atan2(x, y).series(y, n=2)
        -y/x + atan2(x, 0) + O(y**2)

        This function also computes asymptotic expansions, if necessary
        and possible:

        >>> from sympy import loggamma
        >>> loggamma(1/x)._eval_nseries(x,0,None)
        -1/x - log(x)/x + log(x)/2 + O(1)

        """
        from .symbol import uniquely_named_symbol
        from sympy.series.order import Order
        from sympy.sets.sets import FiniteSet
        args = self.args
        args0 = [t.limit(x, 0) for t in args]
        if any(t.is_finite is False for t in args0):
            from .numbers import oo, zoo, nan
            a = [t.as_leading_term(x, logx=logx) for t in args]
            a0 = [t.limit(x, 0) for t in a]
            if any(t.has(oo, -oo, zoo, nan) for t in a0):
                return self._eval_aseries(n, args0, x, logx)
            # Careful: the argument goes to oo, but only logarithmically so. We
            # are supposed to do a power series expansion "around the
            # logarithmic term". e.g.
            #      f(1+x+log(x))
            #     -> f(1+logx) + x*f'(1+logx) + O(x**2)
            # where 'logx' is given in the argument
            a = [t._eval_nseries(x, n, logx) for t in args]
            z = [r - r0 for (r, r0) in zip(a, a0)]
            p = [Dummy() for _ in z]
            q = []
            v = None
            for ai, zi, pi in zip(a0, z, p):
                if zi.has(x):
                    if v is not None:
                        raise NotImplementedError
                    q.append(ai + pi)
                    v = pi
                else:
                    q.append(ai)
            e1 = self.func(*q)
            if v is None:
                return e1
            s = e1._eval_nseries(v, n, logx)
            o = s.getO()
            s = s.removeO()
            s = s.subs(v, zi).expand() + Order(o.expr.subs(v, zi), x)
            return s
        if (self.func.nargs is S.Naturals0
                or (self.func.nargs == FiniteSet(1) and args0[0])
                or any(c > 1 for c in self.func.nargs)):
            e = self
            e1 = e.expand()
            if e == e1:
                #for example when e = sin(x+1) or e = sin(cos(x))
                #let's try the general algorithm
                if len(e.args) == 1:
                    # issue 14411
                    e = e.func(e.args[0].cancel())
                term = e.subs(x, S.Zero)
                if term.is_finite is False or term is S.NaN:
                    raise PoleError("Cannot expand %s around 0" % (self))
                series = term
                fact = S.One

                _x = uniquely_named_symbol('xi', self)
                e = e.subs(x, _x)
                for i in range(1, n):
                    fact *= Rational(i)
                    e = e.diff(_x)
                    subs = e.subs(_x, S.Zero)
                    if subs is S.NaN:
                        # try to evaluate a limit if we have to
                        subs = e.limit(_x, S.Zero)
                    if subs.is_finite is False:
                        raise PoleError("Cannot expand %s around 0" % (self))
                    term = subs*(x**i)/fact
                    term = term.expand()
                    series += term
                return series + Order(x**n, x)
            return e1.nseries(x, n=n, logx=logx)
        arg = self.args[0]
        l = []
        g = None
        # try to predict a number of terms needed
        nterms = n + 2
        cf = Order(arg.as_leading_term(x), x).getn()
        if cf != 0:
            nterms = (n/cf).ceiling()
        for i in range(nterms):
            g = self.taylor_term(i, arg, g)
            g = g.nseries(x, n=n, logx=logx)
            l.append(g)
        return Add(*l) + Order(x**n, x)

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of the function.
        """
        if not (1 <= argindex <= len(self.args)):
            raise ArgumentIndexError(self, argindex)
        ix = argindex - 1
        A = self.args[ix]
        if A._diff_wrt:
            if len(self.args) == 1 or not A.is_Symbol:
                return _derivative_dispatch(self, A)
            for i, v in enumerate(self.args):
                if i != ix and A in v.free_symbols:
                    # it can't be in any other argument's free symbols
                    # issue 8510
                    break
            else:
                return _derivative_dispatch(self, A)

        # See issue 4624 and issue 4719, 5600 and 8510
        D = Dummy('xi_%i' % argindex, dummy_index=hash(A))
        args = self.args[:ix] + (D,) + self.args[ix + 1:]
        return Subs(Derivative(self.func(*args), D), D, A)

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        """Stub that should be overridden by new Functions to return
        the first non-zero term in a series if ever an x-dependent
        argument whose leading term vanishes as x -> 0 might be encountered.
        See, for example, cos._eval_as_leading_term.
        """
        from sympy.series.order import Order
        args = [a.as_leading_term(x, logx=logx) for a in self.args]
        o = Order(1, x)
        if any(x in a.free_symbols and o.contains(a) for a in args):
            # Whereas x and any finite number are contained in O(1, x),
            # expressions like 1/x are not. If any arg simplified to a
            # vanishing expression as x -> 0 (like x or x**2, but not
            # 3, 1/x, etc...) then the _eval_as_leading_term is needed
            # to supply the first non-zero term of the series,
            #
            # e.g. expression    leading term
            #      ----------    ------------
            #      cos(1/x)      cos(1/x)
            #      cos(cos(x))   cos(1)
            #      cos(x)        1        <- _eval_as_leading_term needed
            #      sin(x)        x        <- _eval_as_leading_term needed
            #
            raise NotImplementedError(
                '%s has no _eval_as_leading_term routine' % self.func)
        else:
            return self


class AppliedUndef(Function):
    """
    Base class for expressions resulting from the application of an undefined
    function.
    """

    is_number = False

    def __new__(cls, *args, **options):
        args = list(map(sympify, args))
        u = [a.name for a in args if isinstance(a, UndefinedFunction)]
        if u:
            raise TypeError('Invalid argument: expecting an expression, not UndefinedFunction%s: %s' % (
                's'*(len(u) > 1), ', '.join(u)))
        obj = super().__new__(cls, *args, **options)
        return obj

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        return self

    @property
    def _diff_wrt(self):
        """
        Allow derivatives wrt to undefined functions.

        Examples
        ========

        >>> from sympy import Function, Symbol
        >>> f = Function('f')
        >>> x = Symbol('x')
        >>> f(x)._diff_wrt
        True
        >>> f(x).diff(x)
        Derivative(f(x), x)
        """
        return True


class UndefSageHelper:
    """
    Helper to facilitate Sage conversion.
    """
    def __get__(self, ins, typ):
        import sage.all as sage
        if ins is None:
            return lambda: sage.function(typ.__name__)
        else:
            args = [arg._sage_() for arg in ins.args]
            return lambda : sage.function(ins.__class__.__name__)(*args)

_undef_sage_helper = UndefSageHelper()

class UndefinedFunction(FunctionClass):
    """
    The (meta)class of undefined functions.
    """
    def __new__(mcl, name, bases=(AppliedUndef,), __dict__=None, **kwargs):
        from .symbol import _filter_assumptions
        # Allow Function('f', real=True)
        # and/or Function(Symbol('f', real=True))
        assumptions, kwargs = _filter_assumptions(kwargs)
        if isinstance(name, Symbol):
            assumptions = name._merge(assumptions)
            name = name.name
        elif not isinstance(name, str):
            raise TypeError('expecting string or Symbol for name')
        else:
            commutative = assumptions.get('commutative', None)
            assumptions = Symbol(name, **assumptions).assumptions0
            if commutative is None:
                assumptions.pop('commutative')
        __dict__ = __dict__ or {}
        # put the `is_*` for into __dict__
        __dict__.update({'is_%s' % k: v for k, v in assumptions.items()})
        # You can add other attributes, although they do have to be hashable
        # (but seriously, if you want to add anything other than assumptions,
        # just subclass Function)
        __dict__.update(kwargs)
        # add back the sanitized assumptions without the is_ prefix
        kwargs.update(assumptions)
        # Save these for __eq__
        __dict__.update({'_kwargs': kwargs})
        # do this for pickling
        __dict__['__module__'] = None
        obj = super().__new__(mcl, name, bases, __dict__)
        obj.name = name
        obj._sage_ = _undef_sage_helper
        return obj

    def __instancecheck__(cls, instance):
        return cls in type(instance).__mro__

    _kwargs: dict[str, bool | None] = {}

    def __hash__(self):
        return hash((self.class_key(), frozenset(self._kwargs.items())))

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
            self.class_key() == other.class_key() and
            self._kwargs == other._kwargs)

    def __ne__(self, other):
        return not self == other

    @property
    def _diff_wrt(self):
        return False


# XXX: The type: ignore on WildFunction is because mypy complains:
#
# sympy/core/function.py:939: error: Cannot determine type of 'sort_key' in
# base class 'Expr'
#
# Somehow this is because of the @cacheit decorator but it is not clear how to
# fix it.


class WildFunction(Function, AtomicExpr):  # type: ignore
    """
    A WildFunction function matches any function (with its arguments).

    Examples
    ========

    >>> from sympy import WildFunction, Function, cos
    >>> from sympy.abc import x, y
    >>> F = WildFunction('F')
    >>> f = Function('f')
    >>> F.nargs
    Naturals0
    >>> x.match(F)
    >>> F.match(F)
    {F_: F_}
    >>> f(x).match(F)
    {F_: f(x)}
    >>> cos(x).match(F)
    {F_: cos(x)}
    >>> f(x, y).match(F)
    {F_: f(x, y)}

    To match functions with a given number of arguments, set ``nargs`` to the
    desired value at instantiation:

    >>> F = WildFunction('F', nargs=2)
    >>> F.nargs
    {2}
    >>> f(x).match(F)
    >>> f(x, y).match(F)
    {F_: f(x, y)}

    To match functions with a range of arguments, set ``nargs`` to a tuple
    containing the desired number of arguments, e.g. if ``nargs = (1, 2)``
    then functions with 1 or 2 arguments will be matched.

    >>> F = WildFunction('F', nargs=(1, 2))
    >>> F.nargs
    {1, 2}
    >>> f(x).match(F)
    {F_: f(x)}
    >>> f(x, y).match(F)
    {F_: f(x, y)}
    >>> f(x, y, 1).match(F)

    """

    # XXX: What is this class attribute used for?
    include: set[Any] = set()

    def __init__(cls, name, **assumptions):
        from sympy.sets.sets import Set, FiniteSet
        cls.name = name
        nargs = assumptions.pop('nargs', S.Naturals0)
        if not isinstance(nargs, Set):
            # Canonicalize nargs here.  See also FunctionClass.
            if is_sequence(nargs):
                nargs = tuple(ordered(set(nargs)))
            elif nargs is not None:
                nargs = (as_int(nargs),)
            nargs = FiniteSet(*nargs)
        cls.nargs = nargs

    def matches(self, expr, repl_dict=None, old=False):
        if not isinstance(expr, (AppliedUndef, Function)):
            return None
        if len(expr.args) not in self.nargs:
            return None

        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()

        repl_dict[self] = expr
        return repl_dict


class Derivative(Expr):
    """
    Carries out differentiation of the given expression with respect to symbols.

    Examples
    ========

    >>> from sympy import Derivative, Function, symbols, Subs
    >>> from sympy.abc import x, y
    >>> f, g = symbols('f g', cls=Function)

    >>> Derivative(x**2, x, evaluate=True)
    2*x

    Denesting of derivatives retains the ordering of variables:

        >>> Derivative(Derivative(f(x, y), y), x)
        Derivative(f(x, y), y, x)

    Contiguously identical symbols are merged into a tuple giving
    the symbol and the count:

        >>> Derivative(f(x), x, x, y, x)
        Derivative(f(x), (x, 2), y, x)

    If the derivative cannot be performed, and evaluate is True, the
    order of the variables of differentiation will be made canonical:

        >>> Derivative(f(x, y), y, x, evaluate=True)
        Derivative(f(x, y), x, y)

    Derivatives with respect to undefined functions can be calculated:

        >>> Derivative(f(x)**2, f(x), evaluate=True)
        2*f(x)

    Such derivatives will show up when the chain rule is used to
    evalulate a derivative:

        >>> f(g(x)).diff(x)
        Derivative(f(g(x)), g(x))*Derivative(g(x), x)

    Substitution is used to represent derivatives of functions with
    arguments that are not symbols or functions:

        >>> f(2*x + 3).diff(x) == 2*Subs(f(y).diff(y), y, 2*x + 3)
        True

    Notes
    =====

    Simplification of high-order derivatives:

    Because there can be a significant amount of simplification that can be
    done when multiple differentiations are performed, results will be
    automatically simplified in a fairly conservative fashion unless the
    keyword ``simplify`` is set to False.

        >>> from sympy import sqrt, diff, Function, symbols
        >>> from sympy.abc import x, y, z
        >>> f, g = symbols('f,g', cls=Function)

        >>> e = sqrt((x + 1)**2 + x)
        >>> diff(e, (x, 5), simplify=False).count_ops()
        136
        >>> diff(e, (x, 5)).count_ops()
        30

    Ordering of variables:

    If evaluate is set to True and the expression cannot be evaluated, the
    list of differentiation symbols will be sorted, that is, the expression is
    assumed to have continuous derivatives up to the order asked.

    Derivative wrt non-Symbols:

    For the most part, one may not differentiate wrt non-symbols.
    For example, we do not allow differentiation wrt `x*y` because
    there are multiple ways of structurally defining where x*y appears
    in an expression: a very strict definition would make
    (x*y*z).diff(x*y) == 0. Derivatives wrt defined functions (like
    cos(x)) are not allowed, either:

        >>> (x*y*z).diff(x*y)
        Traceback (most recent call last):
        ...
        ValueError: Can't calculate derivative wrt x*y.

    To make it easier to work with variational calculus, however,
    derivatives wrt AppliedUndef and Derivatives are allowed.
    For example, in the Euler-Lagrange method one may write
    F(t, u, v) where u = f(t) and v = f'(t). These variables can be
    written explicitly as functions of time::

        >>> from sympy.abc import t
        >>> F = Function('F')
        >>> U = f(t)
        >>> V = U.diff(t)

    The derivative wrt f(t) can be obtained directly:

        >>> direct = F(t, U, V).diff(U)

    When differentiation wrt a non-Symbol is attempted, the non-Symbol
    is temporarily converted to a Symbol while the differentiation
    is performed and the same answer is obtained:

        >>> indirect = F(t, U, V).subs(U, x).diff(x).subs(x, U)
        >>> assert direct == indirect

    The implication of this non-symbol replacement is that all
    functions are treated as independent of other functions and the
    symbols are independent of the functions that contain them::

        >>> x.diff(f(x))
        0
        >>> g(x).diff(f(x))
        0

    It also means that derivatives are assumed to depend only
    on the variables of differentiation, not on anything contained
    within the expression being differentiated::

        >>> F = f(x)
        >>> Fx = F.diff(x)
        >>> Fx.diff(F)  # derivative depends on x, not F
        0
        >>> Fxx = Fx.diff(x)
        >>> Fxx.diff(Fx)  # derivative depends on x, not Fx
        0

    The last example can be made explicit by showing the replacement
    of Fx in Fxx with y:

        >>> Fxx.subs(Fx, y)
        Derivative(y, x)

        Since that in itself will evaluate to zero, differentiating
        wrt Fx will also be zero:

        >>> _.doit()
        0

    Replacing undefined functions with concrete expressions

    One must be careful to replace undefined functions with expressions
    that contain variables consistent with the function definition and
    the variables of differentiation or else insconsistent result will
    be obtained. Consider the following example:

    >>> eq = f(x)*g(y)
    >>> eq.subs(f(x), x*y).diff(x, y).doit()
    y*Derivative(g(y), y) + g(y)
    >>> eq.diff(x, y).subs(f(x), x*y).doit()
    y*Derivative(g(y), y)

    The results differ because `f(x)` was replaced with an expression
    that involved both variables of differentiation. In the abstract
    case, differentiation of `f(x)` by `y` is 0; in the concrete case,
    the presence of `y` made that derivative nonvanishing and produced
    the extra `g(y)` term.

    Defining differentiation for an object

    An object must define ._eval_derivative(symbol) method that returns
    the differentiation result. This function only needs to consider the
    non-trivial case where expr contains symbol and it should call the diff()
    method internally (not _eval_derivative); Derivative should be the only
    one to call _eval_derivative.

    Any class can allow derivatives to be taken with respect to
    itself (while indicating its scalar nature). See the
    docstring of Expr._diff_wrt.

    See Also
    ========
    _sort_variable_count
    """

    is_Derivative = True

    @property
    def _diff_wrt(self):
        """An expression may be differentiated wrt a Derivative if
        it is in elementary form.

        Examples
        ========

        >>> from sympy import Function, Derivative, cos
        >>> from sympy.abc import x
        >>> f = Function('f')

        >>> Derivative(f(x), x)._diff_wrt
        True
        >>> Derivative(cos(x), x)._diff_wrt
        False
        >>> Derivative(x + 1, x)._diff_wrt
        False

        A Derivative might be an unevaluated form of what will not be
        a valid variable of differentiation if evaluated. For example,

        >>> Derivative(f(f(x)), x).doit()
        Derivative(f(x), x)*Derivative(f(f(x)), f(x))

        Such an expression will present the same ambiguities as arise
        when dealing with any other product, like ``2*x``, so ``_diff_wrt``
        is False:

        >>> Derivative(f(f(x)), x)._diff_wrt
        False
        """
        return self.expr._diff_wrt and isinstance(self.doit(), Derivative)

    def __new__(cls, expr, *variables, **kwargs):
        expr = sympify(expr)
        symbols_or_none = getattr(expr, "free_symbols", None)
        has_symbol_set = isinstance(symbols_or_none, set)

        if not has_symbol_set:
            raise ValueError(filldedent('''
                Since there are no variables in the expression %s,
                it cannot be differentiated.''' % expr))

        # determine value for variables if it wasn't given
        if not variables:
            variables = expr.free_symbols
            if len(variables) != 1:
                if expr.is_number:
                    return S.Zero
                if len(variables) == 0:
                    raise ValueError(filldedent('''
                        Since there are no variables in the expression,
                        the variable(s) of differentiation must be supplied
                        to differentiate %s''' % expr))
                else:
                    raise ValueError(filldedent('''
                        Since there is more than one variable in the
                        expression, the variable(s) of differentiation
                        must be supplied to differentiate %s''' % expr))

        # Split the list of variables into a list of the variables we are diff
        # wrt, where each element of the list has the form (s, count) where
        # s is the entity to diff wrt and count is the order of the
        # derivative.
        variable_count = []
        array_likes = (tuple, list, Tuple)

        from sympy.tensor.array import Array, NDimArray

        for i, v in enumerate(variables):
            if isinstance(v, UndefinedFunction):
                raise TypeError(
                    "cannot differentiate wrt "
                    "UndefinedFunction: %s" % v)

            if isinstance(v, array_likes):
                if len(v) == 0:
                    # Ignore empty tuples: Derivative(expr, ... , (), ... )
                    continue
                if isinstance(v[0], array_likes):
                    # Derive by array: Derivative(expr, ... , [[x, y, z]], ... )
                    if len(v) == 1:
                        v = Array(v[0])
                        count = 1
                    else:
                        v, count = v
                        v = Array(v)
                else:
                    v, count = v
                if count == 0:
                    continue
                variable_count.append(Tuple(v, count))
                continue

            v = sympify(v)
            if isinstance(v, Integer):
                if i == 0:
                    raise ValueError("First variable cannot be a number: %i" % v)
                count = v
                prev, prevcount = variable_count[-1]
                if prevcount != 1:
                    raise TypeError("tuple {} followed by number {}".format((prev, prevcount), v))
                if count == 0:
                    variable_count.pop()
                else:
                    variable_count[-1] = Tuple(prev, count)
            else:
                count = 1
                variable_count.append(Tuple(v, count))

        # light evaluation of contiguous, identical
        # items: (x, 1), (x, 1) -> (x, 2)
        merged = []
        for t in variable_count:
            v, c = t
            if c.is_negative:
                raise ValueError(
                    'order of differentiation must be nonnegative')
            if merged and merged[-1][0] == v:
                c += merged[-1][1]
                if not c:
                    merged.pop()
                else:
                    merged[-1] = Tuple(v, c)
            else:
                merged.append(t)
        variable_count = merged

        # sanity check of variables of differentation; we waited
        # until the counts were computed since some variables may
        # have been removed because the count was 0
        for v, c in variable_count:
            # v must have _diff_wrt True
            if not v._diff_wrt:
                __ = ''  # filler to make error message neater
                raise ValueError(filldedent('''
                    Can't calculate derivative wrt %s.%s''' % (v,
                    __)))

        # We make a special case for 0th derivative, because there is no
        # good way to unambiguously print this.
        if len(variable_count) == 0:
            return expr

        evaluate = kwargs.get('evaluate', False)

        if evaluate:
            if isinstance(expr, Derivative):
                expr = expr.canonical
            variable_count = [
                (v.canonical if isinstance(v, Derivative) else v, c)
                for v, c in variable_count]

            # Look for a quick exit if there are symbols that don't appear in
            # expression at all. Note, this cannot check non-symbols like
            # Derivatives as those can be created by intermediate
            # derivatives.
            zero = False
            free = expr.free_symbols
            from sympy.matrices.expressions.matexpr import MatrixExpr

            for v, c in variable_count:
                vfree = v.free_symbols
                if c.is_positive and vfree:
                    if isinstance(v, AppliedUndef):
                        # these match exactly since
                        # x.diff(f(x)) == g(x).diff(f(x)) == 0
                        # and are not created by differentiation
                        D = Dummy()
                        if not expr.xreplace({v: D}).has(D):
                            zero = True
                            break
                    elif isinstance(v, MatrixExpr):
                        zero = False
                        break
                    elif isinstance(v, Symbol) and v not in free:
                        zero = True
                        break
                    else:
                        if not free & vfree:
                            # e.g. v is IndexedBase or Matrix
                            zero = True
                            break
            if zero:
                return cls._get_zero_with_shape_like(expr)

            # make the order of symbols canonical
            #TODO: check if assumption of discontinuous derivatives exist
            variable_count = cls._sort_variable_count(variable_count)

        # denest
        if isinstance(expr, Derivative):
            variable_count = list(expr.variable_count) + variable_count
            expr = expr.expr
            return _derivative_dispatch(expr, *variable_count, **kwargs)

        # we return here if evaluate is False or if there is no
        # _eval_derivative method
        if not evaluate or not hasattr(expr, '_eval_derivative'):
            # return an unevaluated Derivative
            if evaluate and variable_count == [(expr, 1)] and expr.is_scalar:
                # special hack providing evaluation for classes
                # that have defined is_scalar=True but have no
                # _eval_derivative defined
                return S.One
            return Expr.__new__(cls, expr, *variable_count)

        # evaluate the derivative by calling _eval_derivative method
        # of expr for each variable
        # -------------------------------------------------------------
        nderivs = 0  # how many derivatives were performed
        unhandled = []
        from sympy.matrices.matrixbase import MatrixBase
        for i, (v, count) in enumerate(variable_count):

            old_expr = expr
            old_v = None

            is_symbol = v.is_symbol or isinstance(v,
                (Iterable, Tuple, MatrixBase, NDimArray))

            if not is_symbol:
                old_v = v
                v = Dummy('xi')
                expr = expr.xreplace({old_v: v})
                # Derivatives and UndefinedFunctions are independent
                # of all others
                clashing = not (isinstance(old_v, (Derivative, AppliedUndef)))
                if v not in expr.free_symbols and not clashing:
                    return expr.diff(v)  # expr's version of 0
                if not old_v.is_scalar and not hasattr(
                        old_v, '_eval_derivative'):
                    # special hack providing evaluation for classes
                    # that have defined is_scalar=True but have no
                    # _eval_derivative defined
                    expr *= old_v.diff(old_v)

            obj = cls._dispatch_eval_derivative_n_times(expr, v, count)
            if obj is not None and obj.is_zero:
                return obj

            nderivs += count

            if old_v is not None:
                if obj is not None:
                    # remove the dummy that was used
                    obj = obj.subs(v, old_v)
                # restore expr
                expr = old_expr

            if obj is None:
                # we've already checked for quick-exit conditions
                # that give 0 so the remaining variables
                # are contained in the expression but the expression
                # did not compute a derivative so we stop taking
                # derivatives
                unhandled = variable_count[i:]
                break

            expr = obj

        # what we have so far can be made canonical
        expr = expr.replace(
            lambda x: isinstance(x, Derivative),
            lambda x: x.canonical)

        if unhandled:
            if isinstance(expr, Derivative):
                unhandled = list(expr.variable_count) + unhandled
                expr = expr.expr
            expr = Expr.__new__(cls, expr, *unhandled)

        if (nderivs > 1) == True and kwargs.get('simplify', True):
            from .exprtools import factor_terms
            from sympy.simplify.simplify import signsimp
            expr = factor_terms(signsimp(expr))
        return expr

    @property
    def canonical(cls):
        return cls.func(cls.expr,
            *Derivative._sort_variable_count(cls.variable_count))

    @classmethod
    def _sort_variable_count(cls, vc):
        """
        Sort (variable, count) pairs into canonical order while
        retaining order of variables that do not commute during
        differentiation:

        * symbols and functions commute with each other
        * derivatives commute with each other
        * a derivative does not commute with anything it contains
        * any other object is not allowed to commute if it has
          free symbols in common with another object

        Examples
        ========

        >>> from sympy import Derivative, Function, symbols
        >>> vsort = Derivative._sort_variable_count
        >>> x, y, z = symbols('x y z')
        >>> f, g, h = symbols('f g h', cls=Function)

        Contiguous items are collapsed into one pair:

        >>> vsort([(x, 1), (x, 1)])
        [(x, 2)]
        >>> vsort([(y, 1), (f(x), 1), (y, 1), (f(x), 1)])
        [(y, 2), (f(x), 2)]

        Ordering is canonical.

        >>> def vsort0(*v):
        ...     # docstring helper to
        ...     # change vi -> (vi, 0), sort, and return vi vals
        ...     return [i[0] for i in vsort([(i, 0) for i in v])]

        >>> vsort0(y, x)
        [x, y]
        >>> vsort0(g(y), g(x), f(y))
        [f(y), g(x), g(y)]

        Symbols are sorted as far to the left as possible but never
        move to the left of a derivative having the same symbol in
        its variables; the same applies to AppliedUndef which are
        always sorted after Symbols:

        >>> dfx = f(x).diff(x)
        >>> assert vsort0(dfx, y) == [y, dfx]
        >>> assert vsort0(dfx, x) == [dfx, x]
        """
        if not vc:
            return []
        vc = list(vc)
        if len(vc) == 1:
            return [Tuple(*vc[0])]
        V = list(range(len(vc)))
        E = []
        v = lambda i: vc[i][0]
        D = Dummy()
        def _block(d, v, wrt=False):
            # return True if v should not come before d else False
            if d == v:
                return wrt
            if d.is_Symbol:
                return False
            if isinstance(d, Derivative):
                # a derivative blocks if any of it's variables contain
                # v; the wrt flag will return True for an exact match
                # and will cause an AppliedUndef to block if v is in
                # the arguments
                if any(_block(k, v, wrt=True)
                        for k in d._wrt_variables):
                    return True
                return False
            if not wrt and isinstance(d, AppliedUndef):
                return False
            if v.is_Symbol:
                return v in d.free_symbols
            if isinstance(v, AppliedUndef):
                return _block(d.xreplace({v: D}), D)
            return d.free_symbols & v.free_symbols
        for i in range(len(vc)):
            for j in range(i):
                if _block(v(j), v(i)):
                    E.append((j,i))
        # this is the default ordering to use in case of ties
        O = dict(zip(ordered(uniq([i for i, c in vc])), range(len(vc))))
        ix = topological_sort((V, E), key=lambda i: O[v(i)])
        # merge counts of contiguously identical items
        merged = []
        for v, c in [vc[i] for i in ix]:
            if merged and merged[-1][0] == v:
                merged[-1][1] += c
            else:
                merged.append([v, c])
        return [Tuple(*i) for i in merged]

    def _eval_is_commutative(self):
        return self.expr.is_commutative

    def _eval_derivative(self, v):
        # If v (the variable of differentiation) is not in
        # self.variables, we might be able to take the derivative.
        if v not in self._wrt_variables:
            dedv = self.expr.diff(v)
            if isinstance(dedv, Derivative):
                return dedv.func(dedv.expr, *(self.variable_count + dedv.variable_count))
            # dedv (d(self.expr)/dv) could have simplified things such that the
            # derivative wrt things in self.variables can now be done. Thus,
            # we set evaluate=True to see if there are any other derivatives
            # that can be done. The most common case is when dedv is a simple
            # number so that the derivative wrt anything else will vanish.
            return self.func(dedv, *self.variables, evaluate=True)
        # In this case v was in self.variables so the derivative wrt v has
        # already been attempted and was not computed, either because it
        # couldn't be or evaluate=False originally.
        variable_count = list(self.variable_count)
        variable_count.append((v, 1))
        return self.func(self.expr, *variable_count, evaluate=False)

    def doit(self, **hints):
        expr = self.expr
        if hints.get('deep', True):
            expr = expr.doit(**hints)
        hints['evaluate'] = True
        rv = self.func(expr, *self.variable_count, **hints)
        if rv!= self and rv.has(Derivative):
            rv =  rv.doit(**hints)
        return rv

    @_sympifyit('z0', NotImplementedError)
    def doit_numerically(self, z0):
        """
        Evaluate the derivative at z numerically.

        When we can represent derivatives at a point, this should be folded
        into the normal evalf. For now, we need a special method.
        """
        if len(self.free_symbols) != 1 or len(self.variables) != 1:
            raise NotImplementedError('partials and higher order derivatives')
        z = list(self.free_symbols)[0]

        def eval(x):
            f0 = self.expr.subs(z, Expr._from_mpmath(x, prec=mpmath.mp.prec))
            f0 = f0.evalf(prec_to_dps(mpmath.mp.prec))
            return f0._to_mpmath(mpmath.mp.prec)
        return Expr._from_mpmath(mpmath.diff(eval,
                                             z0._to_mpmath(mpmath.mp.prec)),
                                 mpmath.mp.prec)

    @property
    def expr(self):
        return self._args[0]

    @property
    def _wrt_variables(self):
        # return the variables of differentiation without
        # respect to the type of count (int or symbolic)
        return [i[0] for i in self.variable_count]

    @property
    def variables(self):
        # TODO: deprecate?  YES, make this 'enumerated_variables' and
        #       name _wrt_variables as variables
        # TODO: support for `d^n`?
        rv = []
        for v, count in self.variable_count:
            if not count.is_Integer:
                raise TypeError(filldedent('''
                Cannot give expansion for symbolic count. If you just
                want a list of all variables of differentiation, use
                _wrt_variables.'''))
            rv.extend([v]*count)
        return tuple(rv)

    @property
    def variable_count(self):
        return self._args[1:]

    @property
    def derivative_count(self):
        return sum([count for _, count in self.variable_count], 0)

    @property
    def free_symbols(self):
        ret = self.expr.free_symbols
        # Add symbolic counts to free_symbols
        for _, count in self.variable_count:
            ret.update(count.free_symbols)
        return ret

    @property
    def kind(self):
        return self.args[0].kind

    def _eval_subs(self, old, new):
        # The substitution (old, new) cannot be done inside
        # Derivative(expr, vars) for a variety of reasons
        # as handled below.
        if old in self._wrt_variables:
            # first handle the counts
            expr = self.func(self.expr, *[(v, c.subs(old, new))
                for v, c in self.variable_count])
            if expr != self:
                return expr._eval_subs(old, new)
            # quick exit case
            if not getattr(new, '_diff_wrt', False):
                # case (0): new is not a valid variable of
                # differentiation
                if isinstance(old, Symbol):
                    # don't introduce a new symbol if the old will do
                    return Subs(self, old, new)
                else:
                    xi = Dummy('xi')
                    return Subs(self.xreplace({old: xi}), xi, new)

        # If both are Derivatives with the same expr, check if old is
        # equivalent to self or if old is a subderivative of self.
        if old.is_Derivative and old.expr == self.expr:
            if self.canonical == old.canonical:
                return new

            # collections.Counter doesn't have __le__
            def _subset(a, b):
                return all((a[i] <= b[i]) == True for i in a)

            old_vars = Counter(dict(reversed(old.variable_count)))
            self_vars = Counter(dict(reversed(self.variable_count)))
            if _subset(old_vars, self_vars):
                return _derivative_dispatch(new, *(self_vars - old_vars).items()).canonical

        args = list(self.args)
        newargs = [x._subs(old, new) for x in args]
        if args[0] == old:
            # complete replacement of self.expr
            # we already checked that the new is valid so we know
            # it won't be a problem should it appear in variables
            return _derivative_dispatch(*newargs)

        if newargs[0] != args[0]:
            # case (1) can't change expr by introducing something that is in
            # the _wrt_variables if it was already in the expr
            # e.g.
            # for Derivative(f(x, g(y)), y), x cannot be replaced with
            # anything that has y in it; for f(g(x), g(y)).diff(g(y))
            # g(x) cannot be replaced with anything that has g(y)
            syms = {vi: Dummy() for vi in self._wrt_variables
                if not vi.is_Symbol}
            wrt = {syms.get(vi, vi) for vi in self._wrt_variables}
            forbidden = args[0].xreplace(syms).free_symbols & wrt
            nfree = new.xreplace(syms).free_symbols
            ofree = old.xreplace(syms).free_symbols
            if (nfree - ofree) & forbidden:
                return Subs(self, old, new)

        viter = ((i, j) for ((i, _), (j, _)) in zip(newargs[1:], args[1:]))
        if any(i != j for i, j in viter):  # a wrt-variable change
            # case (2) can't change vars by introducing a variable
            # that is contained in expr, e.g.
            # for Derivative(f(z, g(h(x), y)), y), y cannot be changed to
            # x, h(x), or g(h(x), y)
            for a in _atomic(self.expr, recursive=True):
                for i in range(1, len(newargs)):
                    vi, _ = newargs[i]
                    if a == vi and vi != args[i][0]:
                        return Subs(self, old, new)
            # more arg-wise checks
            vc = newargs[1:]
            oldv = self._wrt_variables
            newe = self.expr
            subs = []
            for i, (vi, ci) in enumerate(vc):
                if not vi._diff_wrt:
                    # case (3) invalid differentiation expression so
                    # create a replacement dummy
                    xi = Dummy('xi_%i' % i)
                    # replace the old valid variable with the dummy
                    # in the expression
                    newe = newe.xreplace({oldv[i]: xi})
                    # and replace the bad variable with the dummy
                    vc[i] = (xi, ci)
                    # and record the dummy with the new (invalid)
                    # differentiation expression
                    subs.append((xi, vi))

            if subs:
                # handle any residual substitution in the expression
                newe = newe._subs(old, new)
                # return the Subs-wrapped derivative
                return Subs(Derivative(newe, *vc), *zip(*subs))

        # everything was ok
        return _derivative_dispatch(*newargs)

    def _eval_lseries(self, x, logx, cdir=0):
        dx = self.variables
        for term in self.expr.lseries(x, logx=logx, cdir=cdir):
            yield self.func(term, *dx)

    def _eval_nseries(self, x, n, logx, cdir=0):
        arg = self.expr.nseries(x, n=n, logx=logx)
        o = arg.getO()
        dx = self.variables
        rv = [self.func(a, *dx) for a in Add.make_args(arg.removeO())]
        if o:
            rv.append(o/x)
        return Add(*rv)

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        series_gen = self.expr.lseries(x)
        d = S.Zero
        for leading_term in series_gen:
            d = diff(leading_term, *self.variables)
            if d != 0:
                break
        return d

    def as_finite_difference(self, points=1, x0=None, wrt=None):
        """ Expresses a Derivative instance as a finite difference.

        Parameters
        ==========

        points : sequence or coefficient, optional
            If sequence: discrete values (length >= order+1) of the
            independent variable used for generating the finite
            difference weights.
            If it is a coefficient, it will be used as the step-size
            for generating an equidistant sequence of length order+1
            centered around ``x0``. Default: 1 (step-size 1)

        x0 : number or Symbol, optional
            the value of the independent variable (``wrt``) at which the
            derivative is to be approximated. Default: same as ``wrt``.

        wrt : Symbol, optional
            "with respect to" the variable for which the (partial)
            derivative is to be approximated for. If not provided it
            is required that the derivative is ordinary. Default: ``None``.


        Examples
        ========

        >>> from sympy import symbols, Function, exp, sqrt, Symbol
        >>> x, h = symbols('x h')
        >>> f = Function('f')
        >>> f(x).diff(x).as_finite_difference()
        -f(x - 1/2) + f(x + 1/2)

        The default step size and number of points are 1 and
        ``order + 1`` respectively. We can change the step size by
        passing a symbol as a parameter:

        >>> f(x).diff(x).as_finite_difference(h)
        -f(-h/2 + x)/h + f(h/2 + x)/h

        We can also specify the discretized values to be used in a
        sequence:

        >>> f(x).diff(x).as_finite_difference([x, x+h, x+2*h])
        -3*f(x)/(2*h) + 2*f(h + x)/h - f(2*h + x)/(2*h)

        The algorithm is not restricted to use equidistant spacing, nor
        do we need to make the approximation around ``x0``, but we can get
        an expression estimating the derivative at an offset:

        >>> e, sq2 = exp(1), sqrt(2)
        >>> xl = [x-h, x+h, x+e*h]
        >>> f(x).diff(x, 1).as_finite_difference(xl, x+h*sq2)  # doctest: +ELLIPSIS
        2*h*((h + sqrt(2)*h)/(2*h) - (-sqrt(2)*h + h)/(2*h))*f(E*h + x)/...

        To approximate ``Derivative`` around ``x0`` using a non-equidistant
        spacing step, the algorithm supports assignment of undefined
        functions to ``points``:

        >>> dx = Function('dx')
        >>> f(x).diff(x).as_finite_difference(points=dx(x), x0=x-h)
        -f(-h + x - dx(-h + x)/2)/dx(-h + x) + f(-h + x + dx(-h + x)/2)/dx(-h + x)

        Partial derivatives are also supported:

        >>> y = Symbol('y')
        >>> d2fdxdy=f(x,y).diff(x,y)
        >>> d2fdxdy.as_finite_difference(wrt=x)
        -Derivative(f(x - 1/2, y), y) + Derivative(f(x + 1/2, y), y)

        We can apply ``as_finite_difference`` to ``Derivative`` instances in
        compound expressions using ``replace``:

        >>> (1 + 42**f(x).diff(x)).replace(lambda arg: arg.is_Derivative,
        ...     lambda arg: arg.as_finite_difference())
        42**(-f(x - 1/2) + f(x + 1/2)) + 1


        See also
        ========

        sympy.calculus.finite_diff.apply_finite_diff
        sympy.calculus.finite_diff.differentiate_finite
        sympy.calculus.finite_diff.finite_diff_weights

        """
        from sympy.calculus.finite_diff import _as_finite_diff
        return _as_finite_diff(self, points, x0, wrt)

    @classmethod
    def _get_zero_with_shape_like(cls, expr):
        return S.Zero

    @classmethod
    def _dispatch_eval_derivative_n_times(cls, expr, v, count):
        # Evaluate the derivative `n` times.  If
        # `_eval_derivative_n_times` is not overridden by the current
        # object, the default in `Basic` will call a loop over
        # `_eval_derivative`:
        return expr._eval_derivative_n_times(v, count)


def _derivative_dispatch(expr, *variables, **kwargs):
    from sympy.matrices.matrixbase import MatrixBase
    from sympy.matrices.expressions.matexpr import MatrixExpr
    from sympy.tensor.array import NDimArray
    array_types = (MatrixBase, MatrixExpr, NDimArray, list, tuple, Tuple)
    if isinstance(expr, array_types) or any(isinstance(i[0], array_types) if isinstance(i, (tuple, list, Tuple)) else isinstance(i, array_types) for i in variables):
        from sympy.tensor.array.array_derivatives import ArrayDerivative
        return ArrayDerivative(expr, *variables, **kwargs)
    return Derivative(expr, *variables, **kwargs)


class Lambda(Expr):
    """
    Lambda(x, expr) represents a lambda function similar to Python's
    'lambda x: expr'. A function of several variables is written as
    Lambda((x, y, ...), expr).

    Examples
    ========

    A simple example:

    >>> from sympy import Lambda
    >>> from sympy.abc import x
    >>> f = Lambda(x, x**2)
    >>> f(4)
    16

    For multivariate functions, use:

    >>> from sympy.abc import y, z, t
    >>> f2 = Lambda((x, y, z, t), x + y**z + t**z)
    >>> f2(1, 2, 3, 4)
    73

    It is also possible to unpack tuple arguments:

    >>> f = Lambda(((x, y), z), x + y + z)
    >>> f((1, 2), 3)
    6

    A handy shortcut for lots of arguments:

    >>> p = x, y, z
    >>> f = Lambda(p, x + y*z)
    >>> f(*p)
    x + y*z

    """
    is_Function = True

    def __new__(cls, signature, expr):
        if iterable(signature) and not isinstance(signature, (tuple, Tuple)):
            sympy_deprecation_warning(
                """
                Using a non-tuple iterable as the first argument to Lambda
                is deprecated. Use Lambda(tuple(args), expr) instead.
                """,
                deprecated_since_version="1.5",
                active_deprecations_target="deprecated-non-tuple-lambda",
            )
            signature = tuple(signature)
        sig = signature if iterable(signature) else (signature,)
        sig = sympify(sig)
        cls._check_signature(sig)

        if len(sig) == 1 and sig[0] == expr:
            return S.IdentityFunction

        return Expr.__new__(cls, sig, sympify(expr))

    @classmethod
    def _check_signature(cls, sig):
        syms = set()

        def rcheck(args):
            for a in args:
                if a.is_symbol:
                    if a in syms:
                        raise BadSignatureError("Duplicate symbol %s" % a)
                    syms.add(a)
                elif isinstance(a, Tuple):
                    rcheck(a)
                else:
                    raise BadSignatureError("Lambda signature should be only tuples"
                        " and symbols, not %s" % a)

        if not isinstance(sig, Tuple):
            raise BadSignatureError("Lambda signature should be a tuple not %s" % sig)
        # Recurse through the signature:
        rcheck(sig)

    @property
    def signature(self):
        """The expected form of the arguments to be unpacked into variables"""
        return self._args[0]

    @property
    def expr(self):
        """The return value of the function"""
        return self._args[1]

    @property
    def variables(self):
        """The variables used in the internal representation of the function"""
        def _variables(args):
            if isinstance(args, Tuple):
                for arg in args:
                    yield from _variables(arg)
            else:
                yield args
        return tuple(_variables(self.signature))

    @property
    def nargs(self):
        from sympy.sets.sets import FiniteSet
        return FiniteSet(len(self.signature))

    bound_symbols = variables

    @property
    def free_symbols(self):
        return self.expr.free_symbols - set(self.variables)

    def __call__(self, *args):
        n = len(args)
        if n not in self.nargs:  # Lambda only ever has 1 value in nargs
            # XXX: exception message must be in exactly this format to
            # make it work with NumPy's functions like vectorize(). See,
            # for example, https://github.com/numpy/numpy/issues/1697.
            # The ideal solution would be just to attach metadata to
            # the exception and change NumPy to take advantage of this.
            ## XXX does this apply to Lambda? If not, remove this comment.
            temp = ('%(name)s takes exactly %(args)s '
                   'argument%(plural)s (%(given)s given)')
            raise BadArgumentsError(temp % {
                'name': self,
                'args': list(self.nargs)[0],
                'plural': 's'*(list(self.nargs)[0] != 1),
                'given': n})

        d = self._match_signature(self.signature, args)

        return self.expr.xreplace(d)

    def _match_signature(self, sig, args):

        symargmap = {}

        def rmatch(pars, args):
            for par, arg in zip(pars, args):
                if par.is_symbol:
                    symargmap[par] = arg
                elif isinstance(par, Tuple):
                    if not isinstance(arg, (tuple, Tuple)) or len(args) != len(pars):
                        raise BadArgumentsError("Can't match %s and %s" % (args, pars))
                    rmatch(par, arg)

        rmatch(sig, args)

        return symargmap

    @property
    def is_identity(self):
        """Return ``True`` if this ``Lambda`` is an identity function. """
        return self.signature == self.expr

    def _eval_evalf(self, prec):
        return self.func(self.args[0], self.args[1].evalf(n=prec_to_dps(prec)))


class Subs(Expr):
    """
    Represents unevaluated substitutions of an expression.

    ``Subs(expr, x, x0)`` represents the expression resulting
    from substituting x with x0 in expr.

    Parameters
    ==========

    expr : Expr
        An expression.

    x : tuple, variable
        A variable or list of distinct variables.

    x0 : tuple or list of tuples
        A point or list of evaluation points
        corresponding to those variables.

    Examples
    ========

    >>> from sympy import Subs, Function, sin, cos
    >>> from sympy.abc import x, y, z
    >>> f = Function('f')

    Subs are created when a particular substitution cannot be made. The
    x in the derivative cannot be replaced with 0 because 0 is not a
    valid variables of differentiation:

    >>> f(x).diff(x).subs(x, 0)
    Subs(Derivative(f(x), x), x, 0)

    Once f is known, the derivative and evaluation at 0 can be done:

    >>> _.subs(f, sin).doit() == sin(x).diff(x).subs(x, 0) == cos(0)
    True

    Subs can also be created directly with one or more variables:

    >>> Subs(f(x)*sin(y) + z, (x, y), (0, 1))
    Subs(z + f(x)*sin(y), (x, y), (0, 1))
    >>> _.doit()
    z + f(0)*sin(1)

    Notes
    =====

    ``Subs`` objects are generally useful to represent unevaluated derivatives
    calculated at a point.

    The variables may be expressions, but they are subjected to the limitations
    of subs(), so it is usually a good practice to use only symbols for
    variables, since in that case there can be no ambiguity.

    There's no automatic expansion - use the method .doit() to effect all
    possible substitutions of the object and also of objects inside the
    expression.

    When evaluating derivatives at a point that is not a symbol, a Subs object
    is returned. One is also able to calculate derivatives of Subs objects - in
    this case the expression is always expanded (for the unevaluated form, use
    Derivative()).

    In order to allow expressions to combine before doit is done, a
    representation of the Subs expression is used internally to make
    expressions that are superficially different compare the same:

    >>> a, b = Subs(x, x, 0), Subs(y, y, 0)
    >>> a + b
    2*Subs(x, x, 0)

    This can lead to unexpected consequences when using methods
    like `has` that are cached:

    >>> s = Subs(x, x, 0)
    >>> s.has(x), s.has(y)
    (True, False)
    >>> ss = s.subs(x, y)
    >>> ss.has(x), ss.has(y)
    (True, False)
    >>> s, ss
    (Subs(x, x, 0), Subs(y, y, 0))
    """
    def __new__(cls, expr, variables, point, **assumptions):
        if not is_sequence(variables, Tuple):
            variables = [variables]
        variables = Tuple(*variables)

        if has_dups(variables):
            repeated = [str(v) for v, i in Counter(variables).items() if i > 1]
            __ = ', '.join(repeated)
            raise ValueError(filldedent('''
                The following expressions appear more than once: %s
                ''' % __))

        point = Tuple(*(point if is_sequence(point, Tuple) else [point]))

        if len(point) != len(variables):
            raise ValueError('Number of point values must be the same as '
                             'the number of variables.')

        if not point:
            return sympify(expr)

        # denest
        if isinstance(expr, Subs):
            variables = expr.variables + variables
            point = expr.point + point
            expr = expr.expr
        else:
            expr = sympify(expr)

        # use symbols with names equal to the point value (with prepended _)
        # to give a variable-independent expression
        pre = "_"
        pts = sorted(set(point), key=default_sort_key)
        from sympy.printing.str import StrPrinter
        class CustomStrPrinter(StrPrinter):
            def _print_Dummy(self, expr):
                return str(expr) + str(expr.dummy_index)
        def mystr(expr, **settings):
            p = CustomStrPrinter(settings)
            return p.doprint(expr)
        while 1:
            s_pts = {p: Symbol(pre + mystr(p)) for p in pts}
            reps = [(v, s_pts[p])
                for v, p in zip(variables, point)]
            # if any underscore-prepended symbol is already a free symbol
            # and is a variable with a different point value, then there
            # is a clash, e.g. _0 clashes in Subs(_0 + _1, (_0, _1), (1, 0))
            # because the new symbol that would be created is _1 but _1
            # is already mapped to 0 so __0 and __1 are used for the new
            # symbols
            if any(r in expr.free_symbols and
                   r in variables and
                   Symbol(pre + mystr(point[variables.index(r)])) != r
                   for _, r in reps):
                pre += "_"
                continue
            break

        obj = Expr.__new__(cls, expr, Tuple(*variables), point)
        obj._expr = expr.xreplace(dict(reps))
        return obj

    def _eval_is_commutative(self):
        return self.expr.is_commutative

    def doit(self, **hints):
        e, v, p = self.args

        # remove self mappings
        for i, (vi, pi) in enumerate(zip(v, p)):
            if vi == pi:
                v = v[:i] + v[i + 1:]
                p = p[:i] + p[i + 1:]
        if not v:
            return self.expr

        if isinstance(e, Derivative):
            # apply functions first, e.g. f -> cos
            undone = []
            for i, vi in enumerate(v):
                if isinstance(vi, FunctionClass):
                    e = e.subs(vi, p[i])
                else:
                    undone.append((vi, p[i]))
            if not isinstance(e, Derivative):
                e = e.doit()
            if isinstance(e, Derivative):
                # do Subs that aren't related to differentiation
                undone2 = []
                D = Dummy()
                arg = e.args[0]
                for vi, pi in undone:
                    if D not in e.xreplace({vi: D}).free_symbols:
                        if arg.has(vi):
                            e = e.subs(vi, pi)
                    else:
                        undone2.append((vi, pi))
                undone = undone2
                # differentiate wrt variables that are present
                wrt = []
                D = Dummy()
                expr = e.expr
                free = expr.free_symbols
                for vi, ci in e.variable_count:
                    if isinstance(vi, Symbol) and vi in free:
                        expr = expr.diff((vi, ci))
                    elif D in expr.subs(vi, D).free_symbols:
                        expr = expr.diff((vi, ci))
                    else:
                        wrt.append((vi, ci))
                # inject remaining subs
                rv = expr.subs(undone)
                # do remaining differentiation *in order given*
                for vc in wrt:
                    rv = rv.diff(vc)
            else:
                # inject remaining subs
                rv = e.subs(undone)
        else:
            rv = e.doit(**hints).subs(list(zip(v, p)))

        if hints.get('deep', True) and rv != self:
            rv = rv.doit(**hints)
        return rv

    def evalf(self, prec=None, **options):
        return self.doit().evalf(prec, **options)

    n = evalf  # type:ignore

    @property
    def variables(self):
        """The variables to be evaluated"""
        return self._args[1]

    bound_symbols = variables

    @property
    def expr(self):
        """The expression on which the substitution operates"""
        return self._args[0]

    @property
    def point(self):
        """The values for which the variables are to be substituted"""
        return self._args[2]

    @property
    def free_symbols(self):
        return (self.expr.free_symbols - set(self.variables) |
            set(self.point.free_symbols))

    @property
    def expr_free_symbols(self):
        sympy_deprecation_warning("""
        The expr_free_symbols property is deprecated. Use free_symbols to get
        the free symbols of an expression.
        """,
            deprecated_since_version="1.9",
            active_deprecations_target="deprecated-expr-free-symbols")
        # Don't show the warning twice from the recursive call
        with ignore_warnings(SymPyDeprecationWarning):
            return (self.expr.expr_free_symbols - set(self.variables) |
                    set(self.point.expr_free_symbols))

    def __eq__(self, other):
        if not isinstance(other, Subs):
            return False
        return self._hashable_content() == other._hashable_content()

    def __ne__(self, other):
        return not(self == other)

    def __hash__(self):
        return super().__hash__()

    def _hashable_content(self):
        return (self._expr.xreplace(self.canonical_variables),
            ) + tuple(ordered([(v, p) for v, p in
            zip(self.variables, self.point) if not self.expr.has(v)]))

    def _eval_subs(self, old, new):
        # Subs doit will do the variables in order; the semantics
        # of subs for Subs is have the following invariant for
        # Subs object foo:
        #    foo.doit().subs(reps) == foo.subs(reps).doit()
        pt = list(self.point)
        if old in self.variables:
            if _atomic(new) == {new} and not any(
                    i.has(new) for i in self.args):
                # the substitution is neutral
                return self.xreplace({old: new})
            # any occurrence of old before this point will get
            # handled by replacements from here on
            i = self.variables.index(old)
            for j in range(i, len(self.variables)):
                pt[j] = pt[j]._subs(old, new)
            return self.func(self.expr, self.variables, pt)
        v = [i._subs(old, new) for i in self.variables]
        if v != list(self.variables):
            return self.func(self.expr, self.variables + (old,), pt + [new])
        expr = self.expr._subs(old, new)
        pt = [i._subs(old, new) for i in self.point]
        return self.func(expr, v, pt)

    def _eval_derivative(self, s):
        # Apply the chain rule of the derivative on the substitution variables:
        f = self.expr
        vp = V, P = self.variables, self.point
        val = Add.fromiter(p.diff(s)*Subs(f.diff(v), *vp).doit()
            for v, p in zip(V, P))

        # these are all the free symbols in the expr
        efree = f.free_symbols
        # some symbols like IndexedBase include themselves and args
        # as free symbols
        compound = {i for i in efree if len(i.free_symbols) > 1}
        # hide them and see what independent free symbols remain
        dums = {Dummy() for i in compound}
        masked = f.xreplace(dict(zip(compound, dums)))
        ifree = masked.free_symbols - dums
        # include the compound symbols
        free = ifree | compound
        # remove the variables already handled
        free -= set(V)
        # add back any free symbols of remaining compound symbols
        free |= {i for j in free & compound for i in j.free_symbols}
        # if symbols of s are in free then there is more to do
        if free & s.free_symbols:
            val += Subs(f.diff(s), self.variables, self.point).doit()
        return val

    def _eval_nseries(self, x, n, logx, cdir=0):
        if x in self.point:
            # x is the variable being substituted into
            apos = self.point.index(x)
            other = self.variables[apos]
        else:
            other = x
        arg = self.expr.nseries(other, n=n, logx=logx)
        o = arg.getO()
        terms = Add.make_args(arg.removeO())
        rv = Add(*[self.func(a, *self.args[1:]) for a in terms])
        if o:
            rv += o.subs(other, x)
        return rv

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        if x in self.point:
            ipos = self.point.index(x)
            xvar = self.variables[ipos]
            return self.expr.as_leading_term(xvar)
        if x in self.variables:
            # if `x` is a dummy variable, it means it won't exist after the
            # substitution has been performed:
            return self
        # The variable is independent of the substitution:
        return self.expr.as_leading_term(x)


def diff(f, *symbols, **kwargs):
    """
    Differentiate f with respect to symbols.

    Explanation
    ===========

    This is just a wrapper to unify .diff() and the Derivative class; its
    interface is similar to that of integrate().  You can use the same
    shortcuts for multiple variables as with Derivative.  For example,
    diff(f(x), x, x, x) and diff(f(x), x, 3) both return the third derivative
    of f(x).

    You can pass evaluate=False to get an unevaluated Derivative class.  Note
    that if there are 0 symbols (such as diff(f(x), x, 0), then the result will
    be the function (the zeroth derivative), even if evaluate=False.

    Examples
    ========

    >>> from sympy import sin, cos, Function, diff
    >>> from sympy.abc import x, y
    >>> f = Function('f')

    >>> diff(sin(x), x)
    cos(x)
    >>> diff(f(x), x, x, x)
    Derivative(f(x), (x, 3))
    >>> diff(f(x), x, 3)
    Derivative(f(x), (x, 3))
    >>> diff(sin(x)*cos(y), x, 2, y, 2)
    sin(x)*cos(y)

    >>> type(diff(sin(x), x))
    cos
    >>> type(diff(sin(x), x, evaluate=False))
    <class 'sympy.core.function.Derivative'>
    >>> type(diff(sin(x), x, 0))
    sin
    >>> type(diff(sin(x), x, 0, evaluate=False))
    sin

    >>> diff(sin(x))
    cos(x)
    >>> diff(sin(x*y))
    Traceback (most recent call last):
    ...
    ValueError: specify differentiation variables to differentiate sin(x*y)

    Note that ``diff(sin(x))`` syntax is meant only for convenience
    in interactive sessions and should be avoided in library code.

    References
    ==========

    .. [1] https://reference.wolfram.com/legacy/v5_2/Built-inFunctions/AlgebraicComputation/Calculus/D.html

    See Also
    ========

    Derivative
    idiff: computes the derivative implicitly

    """
    if hasattr(f, 'diff'):
        return f.diff(*symbols, **kwargs)
    kwargs.setdefault('evaluate', True)
    return _derivative_dispatch(f, *symbols, **kwargs)


def expand(e, deep=True, modulus=None, power_base=True, power_exp=True,
        mul=True, log=True, multinomial=True, basic=True, **hints):
    r"""
    Expand an expression using methods given as hints.

    Explanation
    ===========

    Hints evaluated unless explicitly set to False are:  ``basic``, ``log``,
    ``multinomial``, ``mul``, ``power_base``, and ``power_exp`` The following
    hints are supported but not applied unless set to True:  ``complex``,
    ``func``, and ``trig``.  In addition, the following meta-hints are
    supported by some or all of the other hints:  ``frac``, ``numer``,
    ``denom``, ``modulus``, and ``force``.  ``deep`` is supported by all
    hints.  Additionally, subclasses of Expr may define their own hints or
    meta-hints.

    The ``basic`` hint is used for any special rewriting of an object that
    should be done automatically (along with the other hints like ``mul``)
    when expand is called. This is a catch-all hint to handle any sort of
    expansion that may not be described by the existing hint names. To use
    this hint an object should override the ``_eval_expand_basic`` method.
    Objects may also define their own expand methods, which are not run by
    default.  See the API section below.

    If ``deep`` is set to ``True`` (the default), things like arguments of
    functions are recursively expanded.  Use ``deep=False`` to only expand on
    the top level.

    If the ``force`` hint is used, assumptions about variables will be ignored
    in making the expansion.

    Hints
    =====

    These hints are run by default

    mul
    ---

    Distributes multiplication over addition:

    >>> from sympy import cos, exp, sin
    >>> from sympy.abc import x, y, z
    >>> (y*(x + z)).expand(mul=True)
    x*y + y*z

    multinomial
    -----------

    Expand (x + y + ...)**n where n is a positive integer.

    >>> ((x + y + z)**2).expand(multinomial=True)
    x**2 + 2*x*y + 2*x*z + y**2 + 2*y*z + z**2

    power_exp
    ---------

    Expand addition in exponents into multiplied bases.

    >>> exp(x + y).expand(power_exp=True)
    exp(x)*exp(y)
    >>> (2**(x + y)).expand(power_exp=True)
    2**x*2**y

    power_base
    ----------

    Split powers of multiplied bases.

    This only happens by default if assumptions allow, or if the
    ``force`` meta-hint is used:

    >>> ((x*y)**z).expand(power_base=True)
    (x*y)**z
    >>> ((x*y)**z).expand(power_base=True, force=True)
    x**z*y**z
    >>> ((2*y)**z).expand(power_base=True)
    2**z*y**z

    Note that in some cases where this expansion always holds, SymPy performs
    it automatically:

    >>> (x*y)**2
    x**2*y**2

    log
    ---

    Pull out power of an argument as a coefficient and split logs products
    into sums of logs.

    Note that these only work if the arguments of the log function have the
    proper assumptions--the arguments must be positive and the exponents must
    be real--or else the ``force`` hint must be True:

    >>> from sympy import log, symbols
    >>> log(x**2*y).expand(log=True)
    log(x**2*y)
    >>> log(x**2*y).expand(log=True, force=True)
    2*log(x) + log(y)
    >>> x, y = symbols('x,y', positive=True)
    >>> log(x**2*y).expand(log=True)
    2*log(x) + log(y)

    basic
    -----

    This hint is intended primarily as a way for custom subclasses to enable
    expansion by default.

    These hints are not run by default:

    complex
    -------

    Split an expression into real and imaginary parts.

    >>> x, y = symbols('x,y')
    >>> (x + y).expand(complex=True)
    re(x) + re(y) + I*im(x) + I*im(y)
    >>> cos(x).expand(complex=True)
    -I*sin(re(x))*sinh(im(x)) + cos(re(x))*cosh(im(x))

    Note that this is just a wrapper around ``as_real_imag()``.  Most objects
    that wish to redefine ``_eval_expand_complex()`` should consider
    redefining ``as_real_imag()`` instead.

    func
    ----

    Expand other functions.

    >>> from sympy import gamma
    >>> gamma(x + 1).expand(func=True)
    x*gamma(x)

    trig
    ----

    Do trigonometric expansions.

    >>> cos(x + y).expand(trig=True)
    -sin(x)*sin(y) + cos(x)*cos(y)
    >>> sin(2*x).expand(trig=True)
    2*sin(x)*cos(x)

    Note that the forms of ``sin(n*x)`` and ``cos(n*x)`` in terms of ``sin(x)``
    and ``cos(x)`` are not unique, due to the identity `\sin^2(x) + \cos^2(x)
    = 1`.  The current implementation uses the form obtained from Chebyshev
    polynomials, but this may change.  See `this MathWorld article
    <https://mathworld.wolfram.com/Multiple-AngleFormulas.html>`_ for more
    information.

    Notes
    =====

    - You can shut off unwanted methods::

        >>> (exp(x + y)*(x + y)).expand()
        x*exp(x)*exp(y) + y*exp(x)*exp(y)
        >>> (exp(x + y)*(x + y)).expand(power_exp=False)
        x*exp(x + y) + y*exp(x + y)
        >>> (exp(x + y)*(x + y)).expand(mul=False)
        (x + y)*exp(x)*exp(y)

    - Use deep=False to only expand on the top level::

        >>> exp(x + exp(x + y)).expand()
        exp(x)*exp(exp(x)*exp(y))
        >>> exp(x + exp(x + y)).expand(deep=False)
        exp(x)*exp(exp(x + y))

    - Hints are applied in an arbitrary, but consistent order (in the current
      implementation, they are applied in alphabetical order, except
      multinomial comes before mul, but this may change).  Because of this,
      some hints may prevent expansion by other hints if they are applied
      first. For example, ``mul`` may distribute multiplications and prevent
      ``log`` and ``power_base`` from expanding them. Also, if ``mul`` is
      applied before ``multinomial`, the expression might not be fully
      distributed. The solution is to use the various ``expand_hint`` helper
      functions or to use ``hint=False`` to this function to finely control
      which hints are applied. Here are some examples::

        >>> from sympy import expand, expand_mul, expand_power_base
        >>> x, y, z = symbols('x,y,z', positive=True)

        >>> expand(log(x*(y + z)))
        log(x) + log(y + z)

      Here, we see that ``log`` was applied before ``mul``.  To get the mul
      expanded form, either of the following will work::

        >>> expand_mul(log(x*(y + z)))
        log(x*y + x*z)
        >>> expand(log(x*(y + z)), log=False)
        log(x*y + x*z)

      A similar thing can happen with the ``power_base`` hint::

        >>> expand((x*(y + z))**x)
        (x*y + x*z)**x

      To get the ``power_base`` expanded form, either of the following will
      work::

        >>> expand((x*(y + z))**x, mul=False)
        x**x*(y + z)**x
        >>> expand_power_base((x*(y + z))**x)
        x**x*(y + z)**x

        >>> expand((x + y)*y/x)
        y + y**2/x

      The parts of a rational expression can be targeted::

        >>> expand((x + y)*y/x/(x + 1), frac=True)
        (x*y + y**2)/(x**2 + x)
        >>> expand((x + y)*y/x/(x + 1), numer=True)
        (x*y + y**2)/(x*(x + 1))
        >>> expand((x + y)*y/x/(x + 1), denom=True)
        y*(x + y)/(x**2 + x)

    - The ``modulus`` meta-hint can be used to reduce the coefficients of an
      expression post-expansion::

        >>> expand((3*x + 1)**2)
        9*x**2 + 6*x + 1
        >>> expand((3*x + 1)**2, modulus=5)
        4*x**2 + x + 1

    - Either ``expand()`` the function or ``.expand()`` the method can be
      used.  Both are equivalent::

        >>> expand((x + 1)**2)
        x**2 + 2*x + 1
        >>> ((x + 1)**2).expand()
        x**2 + 2*x + 1

    API
    ===

    Objects can define their own expand hints by defining
    ``_eval_expand_hint()``.  The function should take the form::

        def _eval_expand_hint(self, **hints):
            # Only apply the method to the top-level expression
            ...

    See also the example below.  Objects should define ``_eval_expand_hint()``
    methods only if ``hint`` applies to that specific object.  The generic
    ``_eval_expand_hint()`` method defined in Expr will handle the no-op case.

    Each hint should be responsible for expanding that hint only.
    Furthermore, the expansion should be applied to the top-level expression
    only.  ``expand()`` takes care of the recursion that happens when
    ``deep=True``.

    You should only call ``_eval_expand_hint()`` methods directly if you are
    100% sure that the object has the method, as otherwise you are liable to
    get unexpected ``AttributeError``s.  Note, again, that you do not need to
    recursively apply the hint to args of your object: this is handled
    automatically by ``expand()``.  ``_eval_expand_hint()`` should
    generally not be used at all outside of an ``_eval_expand_hint()`` method.
    If you want to apply a specific expansion from within another method, use
    the public ``expand()`` function, method, or ``expand_hint()`` functions.

    In order for expand to work, objects must be rebuildable by their args,
    i.e., ``obj.func(*obj.args) == obj`` must hold.

    Expand methods are passed ``**hints`` so that expand hints may use
    'metahints'--hints that control how different expand methods are applied.
    For example, the ``force=True`` hint described above that causes
    ``expand(log=True)`` to ignore assumptions is such a metahint.  The
    ``deep`` meta-hint is handled exclusively by ``expand()`` and is not
    passed to ``_eval_expand_hint()`` methods.

    Note that expansion hints should generally be methods that perform some
    kind of 'expansion'.  For hints that simply rewrite an expression, use the
    .rewrite() API.

    Examples
    ========

    >>> from sympy import Expr, sympify
    >>> class MyClass(Expr):
    ...     def __new__(cls, *args):
    ...         args = sympify(args)
    ...         return Expr.__new__(cls, *args)
    ...
    ...     def _eval_expand_double(self, *, force=False, **hints):
    ...         '''
    ...         Doubles the args of MyClass.
    ...
    ...         If there more than four args, doubling is not performed,
    ...         unless force=True is also used (False by default).
    ...         '''
    ...         if not force and len(self.args) > 4:
    ...             return self
    ...         return self.func(*(self.args + self.args))
    ...
    >>> a = MyClass(1, 2, MyClass(3, 4))
    >>> a
    MyClass(1, 2, MyClass(3, 4))
    >>> a.expand(double=True)
    MyClass(1, 2, MyClass(3, 4, 3, 4), 1, 2, MyClass(3, 4, 3, 4))
    >>> a.expand(double=True, deep=False)
    MyClass(1, 2, MyClass(3, 4), 1, 2, MyClass(3, 4))

    >>> b = MyClass(1, 2, 3, 4, 5)
    >>> b.expand(double=True)
    MyClass(1, 2, 3, 4, 5)
    >>> b.expand(double=True, force=True)
    MyClass(1, 2, 3, 4, 5, 1, 2, 3, 4, 5)

    See Also
    ========

    expand_log, expand_mul, expand_multinomial, expand_complex, expand_trig,
    expand_power_base, expand_power_exp, expand_func, sympy.simplify.hyperexpand.hyperexpand

    """
    # don't modify this; modify the Expr.expand method
    hints['power_base'] = power_base
    hints['power_exp'] = power_exp
    hints['mul'] = mul
    hints['log'] = log
    hints['multinomial'] = multinomial
    hints['basic'] = basic
    return sympify(e).expand(deep=deep, modulus=modulus, **hints)

# This is a special application of two hints

def _mexpand(expr, recursive=False):
    # expand multinomials and then expand products; this may not always
    # be sufficient to give a fully expanded expression (see
    # test_issue_8247_8354 in test_arit)
    if expr is None:
        return
    was = None
    while was != expr:
        was, expr = expr, expand_mul(expand_multinomial(expr))
        if not recursive:
            break
    return expr


# These are simple wrappers around single hints.


def expand_mul(expr, deep=True):
    """
    Wrapper around expand that only uses the mul hint.  See the expand
    docstring for more information.

    Examples
    ========

    >>> from sympy import symbols, expand_mul, exp, log
    >>> x, y = symbols('x,y', positive=True)
    >>> expand_mul(exp(x+y)*(x+y)*log(x*y**2))
    x*exp(x + y)*log(x*y**2) + y*exp(x + y)*log(x*y**2)

    """
    return sympify(expr).expand(deep=deep, mul=True, power_exp=False,
    power_base=False, basic=False, multinomial=False, log=False)


def expand_multinomial(expr, deep=True):
    """
    Wrapper around expand that only uses the multinomial hint.  See the expand
    docstring for more information.

    Examples
    ========

    >>> from sympy import symbols, expand_multinomial, exp
    >>> x, y = symbols('x y', positive=True)
    >>> expand_multinomial((x + exp(x + 1))**2)
    x**2 + 2*x*exp(x + 1) + exp(2*x + 2)

    """
    return sympify(expr).expand(deep=deep, mul=False, power_exp=False,
    power_base=False, basic=False, multinomial=True, log=False)


def expand_log(expr, deep=True, force=False, factor=False):
    """
    Wrapper around expand that only uses the log hint.  See the expand
    docstring for more information.

    Examples
    ========

    >>> from sympy import symbols, expand_log, exp, log
    >>> x, y = symbols('x,y', positive=True)
    >>> expand_log(exp(x+y)*(x+y)*log(x*y**2))
    (x + y)*(log(x) + 2*log(y))*exp(x + y)

    """
    from sympy.functions.elementary.exponential import log
    from sympy.simplify.radsimp import fraction
    if factor is False:
        def _handleMul(x):
            # look for the simple case of expanded log(b**a)/log(b) -> a in args
            n, d = fraction(x)
            n = [i for i in n.atoms(log) if i.args[0].is_Integer]
            d = [i for i in d.atoms(log) if i.args[0].is_Integer]
            if len(n) == 1 and len(d) == 1:
                n = n[0]
                d = d[0]
                from sympy import multiplicity
                m = multiplicity(d.args[0], n.args[0])
                if m:
                    r = m + log(n.args[0]//d.args[0]**m)/d
                    x = x.subs(n, d*r)
            x1 = expand_mul(expand_log(x, deep=deep, force=force, factor=True))
            if x1.count(log) <= x.count(log):
                return x1
            return x

        expr = expr.replace(
        lambda x: x.is_Mul and all(any(isinstance(i, log) and i.args[0].is_Rational
        for i in Mul.make_args(j)) for j in x.as_numer_denom()),
        _handleMul)

    return sympify(expr).expand(deep=deep, log=True, mul=False,
        power_exp=False, power_base=False, multinomial=False,
        basic=False, force=force, factor=factor)


def expand_func(expr, deep=True):
    """
    Wrapper around expand that only uses the func hint.  See the expand
    docstring for more information.

    Examples
    ========

    >>> from sympy import expand_func, gamma
    >>> from sympy.abc import x
    >>> expand_func(gamma(x + 2))
    x*(x + 1)*gamma(x)

    """
    return sympify(expr).expand(deep=deep, func=True, basic=False,
    log=False, mul=False, power_exp=False, power_base=False, multinomial=False)


def expand_trig(expr, deep=True):
    """
    Wrapper around expand that only uses the trig hint.  See the expand
    docstring for more information.

    Examples
    ========

    >>> from sympy import expand_trig, sin
    >>> from sympy.abc import x, y
    >>> expand_trig(sin(x+y)*(x+y))
    (x + y)*(sin(x)*cos(y) + sin(y)*cos(x))

    """
    return sympify(expr).expand(deep=deep, trig=True, basic=False,
    log=False, mul=False, power_exp=False, power_base=False, multinomial=False)


def expand_complex(expr, deep=True):
    """
    Wrapper around expand that only uses the complex hint.  See the expand
    docstring for more information.

    Examples
    ========

    >>> from sympy import expand_complex, exp, sqrt, I
    >>> from sympy.abc import z
    >>> expand_complex(exp(z))
    I*exp(re(z))*sin(im(z)) + exp(re(z))*cos(im(z))
    >>> expand_complex(sqrt(I))
    sqrt(2)/2 + sqrt(2)*I/2

    See Also
    ========

    sympy.core.expr.Expr.as_real_imag
    """
    return sympify(expr).expand(deep=deep, complex=True, basic=False,
    log=False, mul=False, power_exp=False, power_base=False, multinomial=False)


def expand_power_base(expr, deep=True, force=False):
    """
    Wrapper around expand that only uses the power_base hint.

    A wrapper to expand(power_base=True) which separates a power with a base
    that is a Mul into a product of powers, without performing any other
    expansions, provided that assumptions about the power's base and exponent
    allow.

    deep=False (default is True) will only apply to the top-level expression.

    force=True (default is False) will cause the expansion to ignore
    assumptions about the base and exponent. When False, the expansion will
    only happen if the base is non-negative or the exponent is an integer.

    >>> from sympy.abc import x, y, z
    >>> from sympy import expand_power_base, sin, cos, exp, Symbol

    >>> (x*y)**2
    x**2*y**2

    >>> (2*x)**y
    (2*x)**y
    >>> expand_power_base(_)
    2**y*x**y

    >>> expand_power_base((x*y)**z)
    (x*y)**z
    >>> expand_power_base((x*y)**z, force=True)
    x**z*y**z
    >>> expand_power_base(sin((x*y)**z), deep=False)
    sin((x*y)**z)
    >>> expand_power_base(sin((x*y)**z), force=True)
    sin(x**z*y**z)

    >>> expand_power_base((2*sin(x))**y + (2*cos(x))**y)
    2**y*sin(x)**y + 2**y*cos(x)**y

    >>> expand_power_base((2*exp(y))**x)
    2**x*exp(y)**x

    >>> expand_power_base((2*cos(x))**y)
    2**y*cos(x)**y

    Notice that sums are left untouched. If this is not the desired behavior,
    apply full ``expand()`` to the expression:

    >>> expand_power_base(((x+y)*z)**2)
    z**2*(x + y)**2
    >>> (((x+y)*z)**2).expand()
    x**2*z**2 + 2*x*y*z**2 + y**2*z**2

    >>> expand_power_base((2*y)**(1+z))
    2**(z + 1)*y**(z + 1)
    >>> ((2*y)**(1+z)).expand()
    2*2**z*y**(z + 1)

    The power that is unexpanded can be expanded safely when
    ``y != 0``, otherwise different values might be obtained for the expression:

    >>> prev = _

    If we indicate that ``y`` is positive but then replace it with
    a value of 0 after expansion, the expression becomes 0:

    >>> p = Symbol('p', positive=True)
    >>> prev.subs(y, p).expand().subs(p, 0)
    0

    But if ``z = -1`` the expression would not be zero:

    >>> prev.subs(y, 0).subs(z, -1)
    1

    See Also
    ========

    expand

    """
    return sympify(expr).expand(deep=deep, log=False, mul=False,
        power_exp=False, power_base=True, multinomial=False,
        basic=False, force=force)


def expand_power_exp(expr, deep=True):
    """
    Wrapper around expand that only uses the power_exp hint.

    See the expand docstring for more information.

    Examples
    ========

    >>> from sympy import expand_power_exp, Symbol
    >>> from sympy.abc import x, y
    >>> expand_power_exp(3**(y + 2))
    9*3**y
    >>> expand_power_exp(x**(y + 2))
    x**(y + 2)

    If ``x = 0`` the value of the expression depends on the
    value of ``y``; if the expression were expanded the result
    would be 0. So expansion is only done if ``x != 0``:

    >>> expand_power_exp(Symbol('x', zero=False)**(y + 2))
    x**2*x**y
    """
    return sympify(expr).expand(deep=deep, complex=False, basic=False,
    log=False, mul=False, power_exp=True, power_base=False, multinomial=False)


def count_ops(expr, visual=False):
    """
    Return a representation (integer or expression) of the operations in expr.

    Parameters
    ==========

    expr : Expr
        If expr is an iterable, the sum of the op counts of the
        items will be returned.

    visual : bool, optional
        If ``False`` (default) then the sum of the coefficients of the
        visual expression will be returned.
        If ``True`` then the number of each type of operation is shown
        with the core class types (or their virtual equivalent) multiplied by the
        number of times they occur.

    Examples
    ========

    >>> from sympy.abc import a, b, x, y
    >>> from sympy import sin, count_ops

    Although there is not a SUB object, minus signs are interpreted as
    either negations or subtractions:

    >>> (x - y).count_ops(visual=True)
    SUB
    >>> (-x).count_ops(visual=True)
    NEG

    Here, there are two Adds and a Pow:

    >>> (1 + a + b**2).count_ops(visual=True)
    2*ADD + POW

    In the following, an Add, Mul, Pow and two functions:

    >>> (sin(x)*x + sin(x)**2).count_ops(visual=True)
    ADD + MUL + POW + 2*SIN

    for a total of 5:

    >>> (sin(x)*x + sin(x)**2).count_ops(visual=False)
    5

    Note that "what you type" is not always what you get. The expression
    1/x/y is translated by sympy into 1/(x*y) so it gives a DIV and MUL rather
    than two DIVs:

    >>> (1/x/y).count_ops(visual=True)
    DIV + MUL

    The visual option can be used to demonstrate the difference in
    operations for expressions in different forms. Here, the Horner
    representation is compared with the expanded form of a polynomial:

    >>> eq=x*(1 + x*(2 + x*(3 + x)))
    >>> count_ops(eq.expand(), visual=True) - count_ops(eq, visual=True)
    -MUL + 3*POW

    The count_ops function also handles iterables:

    >>> count_ops([x, sin(x), None, True, x + 2], visual=False)
    2
    >>> count_ops([x, sin(x), None, True, x + 2], visual=True)
    ADD + SIN
    >>> count_ops({x: sin(x), x + 2: y + 1}, visual=True)
    2*ADD + SIN

    """
    from .relational import Relational
    from sympy.concrete.summations import Sum
    from sympy.integrals.integrals import Integral
    from sympy.logic.boolalg import BooleanFunction
    from sympy.simplify.radsimp import fraction

    expr = sympify(expr)
    if isinstance(expr, Expr) and not expr.is_Relational:

        ops = []
        args = [expr]
        NEG = Symbol('NEG')
        DIV = Symbol('DIV')
        SUB = Symbol('SUB')
        ADD = Symbol('ADD')
        EXP = Symbol('EXP')
        while args:
            a = args.pop()

            # if the following fails because the object is
            # not Basic type, then the object should be fixed
            # since it is the intention that all args of Basic
            # should themselves be Basic
            if a.is_Rational:
                #-1/3 = NEG + DIV
                if a is not S.One:
                    if a.p < 0:
                        ops.append(NEG)
                    if a.q != 1:
                        ops.append(DIV)
                    continue
            elif a.is_Mul or a.is_MatMul:
                if _coeff_isneg(a):
                    ops.append(NEG)
                    if a.args[0] is S.NegativeOne:
                        a = a.as_two_terms()[1]
                    else:
                        a = -a
                n, d = fraction(a)
                if n.is_Integer:
                    ops.append(DIV)
                    if n < 0:
                        ops.append(NEG)
                    args.append(d)
                    continue  # won't be -Mul but could be Add
                elif d is not S.One:
                    if not d.is_Integer:
                        args.append(d)
                    ops.append(DIV)
                    args.append(n)
                    continue  # could be -Mul
            elif a.is_Add or a.is_MatAdd:
                aargs = list(a.args)
                negs = 0
                for i, ai in enumerate(aargs):
                    if _coeff_isneg(ai):
                        negs += 1
                        args.append(-ai)
                        if i > 0:
                            ops.append(SUB)
                    else:
                        args.append(ai)
                        if i > 0:
                            ops.append(ADD)
                if negs == len(aargs):  # -x - y = NEG + SUB
                    ops.append(NEG)
                elif _coeff_isneg(aargs[0]):  # -x + y = SUB, but already recorded ADD
                    ops.append(SUB - ADD)
                continue
            if a.is_Pow and a.exp is S.NegativeOne:
                ops.append(DIV)
                args.append(a.base)  # won't be -Mul but could be Add
                continue
            if a == S.Exp1:
                ops.append(EXP)
                continue
            if a.is_Pow and a.base == S.Exp1:
                ops.append(EXP)
                args.append(a.exp)
                continue
            if a.is_Mul or isinstance(a, LatticeOp):
                o = Symbol(a.func.__name__.upper())
                # count the args
                ops.append(o*(len(a.args) - 1))
            elif a.args and (
                    a.is_Pow or a.is_Function or isinstance(a, (Derivative, Integral, Sum))):
                # if it's not in the list above we don't
                # consider a.func something to count, e.g.
                # Tuple, MatrixSymbol, etc...
                if isinstance(a.func, UndefinedFunction):
                    o = Symbol("FUNC_" + a.func.__name__.upper())
                else:
                    o = Symbol(a.func.__name__.upper())
                ops.append(o)

            if not a.is_Symbol:
                args.extend(a.args)

    elif isinstance(expr, Dict):
        ops = [count_ops(k, visual=visual) +
               count_ops(v, visual=visual) for k, v in expr.items()]
    elif iterable(expr):
        ops = [count_ops(i, visual=visual) for i in expr]
    elif isinstance(expr, (Relational, BooleanFunction)):
        ops = []
        for arg in expr.args:
            ops.append(count_ops(arg, visual=True))
        o = Symbol(func_name(expr, short=True).upper())
        ops.append(o)
    elif not isinstance(expr, Basic):
        ops = []
    else:  # it's Basic not isinstance(expr, Expr):
        if not isinstance(expr, Basic):
            raise TypeError("Invalid type of expr")
        else:
            ops = []
            args = [expr]
            while args:
                a = args.pop()

                if a.args:
                    o = Symbol(type(a).__name__.upper())
                    if a.is_Boolean:
                        ops.append(o*(len(a.args)-1))
                    else:
                        ops.append(o)
                    args.extend(a.args)

    if not ops:
        if visual:
            return S.Zero
        return 0

    ops = Add(*ops)

    if visual:
        return ops

    if ops.is_Number:
        return int(ops)

    return sum(int((a.args or [1])[0]) for a in Add.make_args(ops))


def nfloat(expr, n=15, exponent=False, dkeys=False):
    """Make all Rationals in expr Floats except those in exponents
    (unless the exponents flag is set to True) and those in undefined
    functions. When processing dictionaries, do not modify the keys
    unless ``dkeys=True``.

    Examples
    ========

    >>> from sympy import nfloat, cos, pi, sqrt
    >>> from sympy.abc import x, y
    >>> nfloat(x**4 + x/2 + cos(pi/3) + 1 + sqrt(y))
    x**4 + 0.5*x + sqrt(y) + 1.5
    >>> nfloat(x**4 + sqrt(y), exponent=True)
    x**4.0 + y**0.5

    Container types are not modified:

    >>> type(nfloat((1, 2))) is tuple
    True
    """
    from sympy.matrices.matrixbase import MatrixBase

    kw = {"n": n, "exponent": exponent, "dkeys": dkeys}

    if isinstance(expr, MatrixBase):
        return expr.applyfunc(lambda e: nfloat(e, **kw))

    # handling of iterable containers
    if iterable(expr, exclude=str):
        if isinstance(expr, (dict, Dict)):
            if dkeys:
                args = [tuple((nfloat(i, **kw) for i in a))
                    for a in expr.items()]
            else:
                args = [(k, nfloat(v, **kw)) for k, v in expr.items()]
            if isinstance(expr, dict):
                return type(expr)(args)
            else:
                return expr.func(*args)
        elif isinstance(expr, Basic):
            return expr.func(*[nfloat(a, **kw) for a in expr.args])
        return type(expr)([nfloat(a, **kw) for a in expr])

    rv = sympify(expr)

    if rv.is_Number:
        return Float(rv, n)
    elif rv.is_number:
        # evalf doesn't always set the precision
        rv = rv.n(n)
        if rv.is_Number:
            rv = Float(rv.n(n), n)
        else:
            pass  # pure_complex(rv) is likely True
        return rv
    elif rv.is_Atom:
        return rv
    elif rv.is_Relational:
        args_nfloat = (nfloat(arg, **kw) for arg in rv.args)
        return rv.func(*args_nfloat)


    # watch out for RootOf instances that don't like to have
    # their exponents replaced with Dummies and also sometimes have
    # problems with evaluating at low precision (issue 6393)
    from sympy.polys.rootoftools import RootOf
    rv = rv.xreplace({ro: ro.n(n) for ro in rv.atoms(RootOf)})

    from .power import Pow
    if not exponent:
        reps = [(p, Pow(p.base, Dummy())) for p in rv.atoms(Pow)]
        rv = rv.xreplace(dict(reps))
    rv = rv.n(n)
    if not exponent:
        rv = rv.xreplace({d.exp: p.exp for p, d in reps})
    else:
        # Pow._eval_evalf special cases Integer exponents so if
        # exponent is suppose to be handled we have to do so here
        rv = rv.xreplace(Transform(
            lambda x: Pow(x.base, Float(x.exp, n)),
            lambda x: x.is_Pow and x.exp.is_Integer))

    return rv.xreplace(Transform(
        lambda x: x.func(*nfloat(x.args, n, exponent)),
        lambda x: isinstance(x, Function) and not isinstance(x, AppliedUndef)))


from .symbol import Dummy, Symbol
