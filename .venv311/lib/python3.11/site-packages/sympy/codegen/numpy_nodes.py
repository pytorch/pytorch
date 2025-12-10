from sympy.core.function import Add, ArgumentIndexError, Function
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import sympify
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import Max, Min
from .ast import Token, none


def _logaddexp(x1, x2, *, evaluate=True):
    return log(Add(exp(x1, evaluate=evaluate), exp(x2, evaluate=evaluate), evaluate=evaluate))


_two = S.One*2
_ln2 = log(_two)


def _lb(x, *, evaluate=True):
    return log(x, evaluate=evaluate)/_ln2


def _exp2(x, *, evaluate=True):
    return Pow(_two, x, evaluate=evaluate)


def _logaddexp2(x1, x2, *, evaluate=True):
    return _lb(Add(_exp2(x1, evaluate=evaluate),
                   _exp2(x2, evaluate=evaluate), evaluate=evaluate))


class logaddexp(Function):
    """ Logarithm of the sum of exponentiations of the inputs.

    Helper class for use with e.g. numpy.logaddexp

    See Also
    ========

    https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html
    """
    nargs = 2

    def __new__(cls, *args):
        return Function.__new__(cls, *sorted(args, key=default_sort_key))

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            wrt, other = self.args
        elif argindex == 2:
            other, wrt = self.args
        else:
            raise ArgumentIndexError(self, argindex)
        return S.One/(S.One + exp(other-wrt))

    def _eval_rewrite_as_log(self, x1, x2, **kwargs):
        return _logaddexp(x1, x2)

    def _eval_evalf(self, *args, **kwargs):
        return self.rewrite(log).evalf(*args, **kwargs)

    def _eval_simplify(self, *args, **kwargs):
        a, b = (x.simplify(**kwargs) for x in self.args)
        candidate = _logaddexp(a, b)
        if candidate != _logaddexp(a, b, evaluate=False):
            return candidate
        else:
            return logaddexp(a, b)


class logaddexp2(Function):
    """ Logarithm of the sum of exponentiations of the inputs in base-2.

    Helper class for use with e.g. numpy.logaddexp2

    See Also
    ========

    https://numpy.org/doc/stable/reference/generated/numpy.logaddexp2.html
    """
    nargs = 2

    def __new__(cls, *args):
        return Function.__new__(cls, *sorted(args, key=default_sort_key))

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            wrt, other = self.args
        elif argindex == 2:
            other, wrt = self.args
        else:
            raise ArgumentIndexError(self, argindex)
        return S.One/(S.One + _exp2(other-wrt))

    def _eval_rewrite_as_log(self, x1, x2, **kwargs):
        return _logaddexp2(x1, x2)

    def _eval_evalf(self, *args, **kwargs):
        return self.rewrite(log).evalf(*args, **kwargs)

    def _eval_simplify(self, *args, **kwargs):
        a, b = (x.simplify(**kwargs).factor() for x in self.args)
        candidate = _logaddexp2(a, b)
        if candidate != _logaddexp2(a, b, evaluate=False):
            return candidate
        else:
            return logaddexp2(a, b)


class amin(Token):
    """ Minimum value along an axis.

    Helper class for use with e.g. numpy.amin


    See Also
    ========

    https://numpy.org/doc/stable/reference/generated/numpy.amin.html
    """
    __slots__ = _fields = ('array', 'axis')
    defaults = {'axis': none}
    _construct_axis = staticmethod(sympify)


class amax(Token):
    """ Maximum value along an axis.

    Helper class for use with e.g. numpy.amax


    See Also
    ========

    https://numpy.org/doc/stable/reference/generated/numpy.amax.html
    """
    __slots__ = _fields = ('array', 'axis')
    defaults = {'axis': none}
    _construct_axis = staticmethod(sympify)


class maximum(Function):
    """ Element-wise maximum of array elements.

    Helper class for use with e.g. numpy.maximum


    See Also
    ========

    https://numpy.org/doc/stable/reference/generated/numpy.maximum.html
    """

    def _eval_rewrite_as_Max(self, *args):
        return Max(*self.args)


class minimum(Function):
    """ Element-wise minimum of array elements.

    Helper class for use with e.g. numpy.minimum


    See Also
    ========

    https://numpy.org/doc/stable/reference/generated/numpy.minimum.html
    """

    def _eval_rewrite_as_Min(self, *args):
        return Min(*self.args)
