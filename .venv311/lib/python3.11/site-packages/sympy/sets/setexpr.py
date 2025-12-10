from sympy.core import Expr
from sympy.core.decorators import call_highest_priority, _sympifyit
from .fancysets import ImageSet
from .sets import set_add, set_sub, set_mul, set_div, set_pow, set_function


class SetExpr(Expr):
    """An expression that can take on values of a set.

    Examples
    ========

    >>> from sympy import Interval, FiniteSet
    >>> from sympy.sets.setexpr import SetExpr

    >>> a = SetExpr(Interval(0, 5))
    >>> b = SetExpr(FiniteSet(1, 10))
    >>> (a + b).set
    Union(Interval(1, 6), Interval(10, 15))
    >>> (2*a + b).set
    Interval(1, 20)
    """
    _op_priority = 11.0

    def __new__(cls, setarg):
        return Expr.__new__(cls, setarg)

    set = property(lambda self: self.args[0])

    def _latex(self, printer):
        return r"SetExpr\left({}\right)".format(printer._print(self.set))

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__radd__')
    def __add__(self, other):
        return _setexpr_apply_operation(set_add, self, other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__add__')
    def __radd__(self, other):
        return _setexpr_apply_operation(set_add, other, self)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        return _setexpr_apply_operation(set_mul, self, other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return _setexpr_apply_operation(set_mul, other, self)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        return _setexpr_apply_operation(set_sub, self, other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        return _setexpr_apply_operation(set_sub, other, self)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rpow__')
    def __pow__(self, other):
        return _setexpr_apply_operation(set_pow, self, other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__pow__')
    def __rpow__(self, other):
        return _setexpr_apply_operation(set_pow, other, self)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rtruediv__')
    def __truediv__(self, other):
        return _setexpr_apply_operation(set_div, self, other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__truediv__')
    def __rtruediv__(self, other):
        return _setexpr_apply_operation(set_div, other, self)

    def _eval_func(self, func):
        # TODO: this could be implemented straight into `imageset`:
        res = set_function(func, self.set)
        if res is None:
            return SetExpr(ImageSet(func, self.set))
        return SetExpr(res)


def _setexpr_apply_operation(op, x, y):
    if isinstance(x, SetExpr):
        x = x.set
    if isinstance(y, SetExpr):
        y = y.set
    out = op(x, y)
    return SetExpr(out)
