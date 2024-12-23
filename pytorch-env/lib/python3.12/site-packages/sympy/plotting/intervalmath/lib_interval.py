""" The module contains implemented functions for interval arithmetic."""
from functools import reduce

from sympy.plotting.intervalmath import interval
from sympy.external import import_module


def Abs(x):
    if isinstance(x, (int, float)):
        return interval(abs(x))
    elif isinstance(x, interval):
        if x.start < 0 and x.end > 0:
            return interval(0, max(abs(x.start), abs(x.end)), is_valid=x.is_valid)
        else:
            return interval(abs(x.start), abs(x.end))
    else:
        raise NotImplementedError

#Monotonic


def exp(x):
    """evaluates the exponential of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.exp(x), np.exp(x))
    elif isinstance(x, interval):
        return interval(np.exp(x.start), np.exp(x.end), is_valid=x.is_valid)
    else:
        raise NotImplementedError


#Monotonic
def log(x):
    """evaluates the natural logarithm of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        if x <= 0:
            return interval(-np.inf, np.inf, is_valid=False)
        else:
            return interval(np.log(x))
    elif isinstance(x, interval):
        if not x.is_valid:
            return interval(-np.inf, np.inf, is_valid=x.is_valid)
        elif x.end <= 0:
            return interval(-np.inf, np.inf, is_valid=False)
        elif x.start <= 0:
            return interval(-np.inf, np.inf, is_valid=None)

        return interval(np.log(x.start), np.log(x.end))
    else:
        raise NotImplementedError


#Monotonic
def log10(x):
    """evaluates the logarithm to the base 10 of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        if x <= 0:
            return interval(-np.inf, np.inf, is_valid=False)
        else:
            return interval(np.log10(x))
    elif isinstance(x, interval):
        if not x.is_valid:
            return interval(-np.inf, np.inf, is_valid=x.is_valid)
        elif x.end <= 0:
            return interval(-np.inf, np.inf, is_valid=False)
        elif x.start <= 0:
            return interval(-np.inf, np.inf, is_valid=None)
        return interval(np.log10(x.start), np.log10(x.end))
    else:
        raise NotImplementedError


#Monotonic
def atan(x):
    """evaluates the tan inverse of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.arctan(x))
    elif isinstance(x, interval):
        start = np.arctan(x.start)
        end = np.arctan(x.end)
        return interval(start, end, is_valid=x.is_valid)
    else:
        raise NotImplementedError


#periodic
def sin(x):
    """evaluates the sine of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.sin(x))
    elif isinstance(x, interval):
        if not x.is_valid:
            return interval(-1, 1, is_valid=x.is_valid)
        na, __ = divmod(x.start, np.pi / 2.0)
        nb, __ = divmod(x.end, np.pi / 2.0)
        start = min(np.sin(x.start), np.sin(x.end))
        end = max(np.sin(x.start), np.sin(x.end))
        if nb - na > 4:
            return interval(-1, 1, is_valid=x.is_valid)
        elif na == nb:
            return interval(start, end, is_valid=x.is_valid)
        else:
            if (na - 1) // 4 != (nb - 1) // 4:
                #sin has max
                end = 1
            if (na - 3) // 4 != (nb - 3) // 4:
                #sin has min
                start = -1
            return interval(start, end)
    else:
        raise NotImplementedError


#periodic
def cos(x):
    """Evaluates the cos of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.sin(x))
    elif isinstance(x, interval):
        if not (np.isfinite(x.start) and np.isfinite(x.end)):
            return interval(-1, 1, is_valid=x.is_valid)
        na, __ = divmod(x.start, np.pi / 2.0)
        nb, __ = divmod(x.end, np.pi / 2.0)
        start = min(np.cos(x.start), np.cos(x.end))
        end = max(np.cos(x.start), np.cos(x.end))
        if nb - na > 4:
            #differ more than 2*pi
            return interval(-1, 1, is_valid=x.is_valid)
        elif na == nb:
            #in the same quadarant
            return interval(start, end, is_valid=x.is_valid)
        else:
            if (na) // 4 != (nb) // 4:
                #cos has max
                end = 1
            if (na - 2) // 4 != (nb - 2) // 4:
                #cos has min
                start = -1
            return interval(start, end, is_valid=x.is_valid)
    else:
        raise NotImplementedError


def tan(x):
    """Evaluates the tan of an interval"""
    return sin(x) / cos(x)


#Monotonic
def sqrt(x):
    """Evaluates the square root of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        if x > 0:
            return interval(np.sqrt(x))
        else:
            return interval(-np.inf, np.inf, is_valid=False)
    elif isinstance(x, interval):
        #Outside the domain
        if x.end < 0:
            return interval(-np.inf, np.inf, is_valid=False)
        #Partially outside the domain
        elif x.start < 0:
            return interval(-np.inf, np.inf, is_valid=None)
        else:
            return interval(np.sqrt(x.start), np.sqrt(x.end),
                    is_valid=x.is_valid)
    else:
        raise NotImplementedError


def imin(*args):
    """Evaluates the minimum of a list of intervals"""
    np = import_module('numpy')
    if not all(isinstance(arg, (int, float, interval)) for arg in args):
        return NotImplementedError
    else:
        new_args = [a for a in args if isinstance(a, (int, float))
                    or a.is_valid]
        if len(new_args) == 0:
            if all(a.is_valid is False for a in args):
                return interval(-np.inf, np.inf, is_valid=False)
            else:
                return interval(-np.inf, np.inf, is_valid=None)
        start_array = [a if isinstance(a, (int, float)) else a.start
                       for a in new_args]

        end_array = [a if isinstance(a, (int, float)) else a.end
                     for a in new_args]
        return interval(min(start_array), min(end_array))


def imax(*args):
    """Evaluates the maximum of a list of intervals"""
    np = import_module('numpy')
    if not all(isinstance(arg, (int, float, interval)) for arg in args):
        return NotImplementedError
    else:
        new_args = [a for a in args if isinstance(a, (int, float))
                    or a.is_valid]
        if len(new_args) == 0:
            if all(a.is_valid is False for a in args):
                return interval(-np.inf, np.inf, is_valid=False)
            else:
                return interval(-np.inf, np.inf, is_valid=None)
        start_array = [a if isinstance(a, (int, float)) else a.start
                       for a in new_args]

        end_array = [a if isinstance(a, (int, float)) else a.end
                     for a in new_args]

        return interval(max(start_array), max(end_array))


#Monotonic
def sinh(x):
    """Evaluates the hyperbolic sine of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.sinh(x), np.sinh(x))
    elif isinstance(x, interval):
        return interval(np.sinh(x.start), np.sinh(x.end), is_valid=x.is_valid)
    else:
        raise NotImplementedError


def cosh(x):
    """Evaluates the hyperbolic cos of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.cosh(x), np.cosh(x))
    elif isinstance(x, interval):
        #both signs
        if x.start < 0 and x.end > 0:
            end = max(np.cosh(x.start), np.cosh(x.end))
            return interval(1, end, is_valid=x.is_valid)
        else:
            #Monotonic
            start = np.cosh(x.start)
            end = np.cosh(x.end)
            return interval(start, end, is_valid=x.is_valid)
    else:
        raise NotImplementedError


#Monotonic
def tanh(x):
    """Evaluates the hyperbolic tan of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.tanh(x), np.tanh(x))
    elif isinstance(x, interval):
        return interval(np.tanh(x.start), np.tanh(x.end), is_valid=x.is_valid)
    else:
        raise NotImplementedError


def asin(x):
    """Evaluates the inverse sine of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        #Outside the domain
        if abs(x) > 1:
            return interval(-np.inf, np.inf, is_valid=False)
        else:
            return interval(np.arcsin(x), np.arcsin(x))
    elif isinstance(x, interval):
        #Outside the domain
        if x.is_valid is False or x.start > 1 or x.end < -1:
            return interval(-np.inf, np.inf, is_valid=False)
        #Partially outside the domain
        elif x.start < -1 or x.end > 1:
            return interval(-np.inf, np.inf, is_valid=None)
        else:
            start = np.arcsin(x.start)
            end = np.arcsin(x.end)
            return interval(start, end, is_valid=x.is_valid)


def acos(x):
    """Evaluates the inverse cos of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        if abs(x) > 1:
            #Outside the domain
            return interval(-np.inf, np.inf, is_valid=False)
        else:
            return interval(np.arccos(x), np.arccos(x))
    elif isinstance(x, interval):
        #Outside the domain
        if x.is_valid is False or x.start > 1 or x.end < -1:
            return interval(-np.inf, np.inf, is_valid=False)
        #Partially outside the domain
        elif x.start < -1 or x.end > 1:
            return interval(-np.inf, np.inf, is_valid=None)
        else:
            start = np.arccos(x.start)
            end = np.arccos(x.end)
            return interval(start, end, is_valid=x.is_valid)


def ceil(x):
    """Evaluates the ceiling of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.ceil(x))
    elif isinstance(x, interval):
        if x.is_valid is False:
            return interval(-np.inf, np.inf, is_valid=False)
        else:
            start = np.ceil(x.start)
            end = np.ceil(x.end)
            #Continuous over the interval
            if start == end:
                return interval(start, end, is_valid=x.is_valid)
            else:
                #Not continuous over the interval
                return interval(start, end, is_valid=None)
    else:
        return NotImplementedError


def floor(x):
    """Evaluates the floor of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.floor(x))
    elif isinstance(x, interval):
        if x.is_valid is False:
            return interval(-np.inf, np.inf, is_valid=False)
        else:
            start = np.floor(x.start)
            end = np.floor(x.end)
            #continuous over the argument
            if start == end:
                return interval(start, end, is_valid=x.is_valid)
            else:
                #not continuous over the interval
                return interval(start, end, is_valid=None)
    else:
        return NotImplementedError


def acosh(x):
    """Evaluates the inverse hyperbolic cosine of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        #Outside the domain
        if x < 1:
            return interval(-np.inf, np.inf, is_valid=False)
        else:
            return interval(np.arccosh(x))
    elif isinstance(x, interval):
        #Outside the domain
        if x.end < 1:
            return interval(-np.inf, np.inf, is_valid=False)
        #Partly outside the domain
        elif x.start < 1:
            return interval(-np.inf, np.inf, is_valid=None)
        else:
            start = np.arccosh(x.start)
            end = np.arccosh(x.end)
            return interval(start, end, is_valid=x.is_valid)
    else:
        return NotImplementedError


#Monotonic
def asinh(x):
    """Evaluates the inverse hyperbolic sine of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.arcsinh(x))
    elif isinstance(x, interval):
        start = np.arcsinh(x.start)
        end = np.arcsinh(x.end)
        return interval(start, end, is_valid=x.is_valid)
    else:
        return NotImplementedError


def atanh(x):
    """Evaluates the inverse hyperbolic tangent of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        #Outside the domain
        if abs(x) >= 1:
            return interval(-np.inf, np.inf, is_valid=False)
        else:
            return interval(np.arctanh(x))
    elif isinstance(x, interval):
        #outside the domain
        if x.is_valid is False or x.start >= 1 or x.end <= -1:
            return interval(-np.inf, np.inf, is_valid=False)
        #partly outside the domain
        elif x.start <= -1 or x.end >= 1:
            return interval(-np.inf, np.inf, is_valid=None)
        else:
            start = np.arctanh(x.start)
            end = np.arctanh(x.end)
            return interval(start, end, is_valid=x.is_valid)
    else:
        return NotImplementedError


#Three valued logic for interval plotting.

def And(*args):
    """Defines the three valued ``And`` behaviour for a 2-tuple of
     three valued logic values"""
    def reduce_and(cmp_intervala, cmp_intervalb):
        if cmp_intervala[0] is False or cmp_intervalb[0] is False:
            first = False
        elif cmp_intervala[0] is None or cmp_intervalb[0] is None:
            first = None
        else:
            first = True
        if cmp_intervala[1] is False or cmp_intervalb[1] is False:
            second = False
        elif cmp_intervala[1] is None or cmp_intervalb[1] is None:
            second = None
        else:
            second = True
        return (first, second)
    return reduce(reduce_and, args)


def Or(*args):
    """Defines the three valued ``Or`` behaviour for a 2-tuple of
     three valued logic values"""
    def reduce_or(cmp_intervala, cmp_intervalb):
        if cmp_intervala[0] is True or cmp_intervalb[0] is True:
            first = True
        elif cmp_intervala[0] is None or cmp_intervalb[0] is None:
            first = None
        else:
            first = False

        if cmp_intervala[1] is True or cmp_intervalb[1] is True:
            second = True
        elif cmp_intervala[1] is None or cmp_intervalb[1] is None:
            second = None
        else:
            second = False
        return (first, second)
    return reduce(reduce_or, args)
