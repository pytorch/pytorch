from collections.abc import Iterable
from functools import singledispatch

from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.core.parameters import global_parameters


class TensorProduct(Expr):
    """
    Generic class for tensor products.
    """
    is_number = False

    def __new__(cls, *args, **kwargs):
        from sympy.tensor.array import NDimArray, tensorproduct, Array
        from sympy.matrices.expressions.matexpr import MatrixExpr
        from sympy.matrices.matrixbase import MatrixBase
        from sympy.strategies import flatten

        args = [sympify(arg) for arg in args]
        evaluate = kwargs.get("evaluate", global_parameters.evaluate)

        if not evaluate:
            obj = Expr.__new__(cls, *args)
            return obj

        arrays = []
        other = []
        scalar = S.One
        for arg in args:
            if isinstance(arg, (Iterable, MatrixBase, NDimArray)):
                arrays.append(Array(arg))
            elif isinstance(arg, (MatrixExpr,)):
                other.append(arg)
            else:
                scalar *= arg

        coeff = scalar*tensorproduct(*arrays)
        if len(other) == 0:
            return coeff
        if coeff != 1:
            newargs = [coeff] + other
        else:
            newargs = other
        obj = Expr.__new__(cls, *newargs, **kwargs)
        return flatten(obj)

    def rank(self):
        return len(self.shape)

    def _get_args_shapes(self):
        from sympy.tensor.array import Array
        return [i.shape if hasattr(i, "shape") else Array(i).shape for i in self.args]

    @property
    def shape(self):
        shape_list = self._get_args_shapes()
        return sum(shape_list, ())

    def __getitem__(self, index):
        index = iter(index)
        return Mul.fromiter(
            arg.__getitem__(tuple(next(index) for i in shp))
            for arg, shp in zip(self.args, self._get_args_shapes())
        )


@singledispatch
def shape(expr):
    """
    Return the shape of the *expr* as a tuple. *expr* should represent
    suitable object such as matrix or array.

    Parameters
    ==========

    expr : SymPy object having ``MatrixKind`` or ``ArrayKind``.

    Raises
    ======

    NoShapeError : Raised when object with wrong kind is passed.

    Examples
    ========

    This function returns the shape of any object representing matrix or array.

    >>> from sympy import shape, Array, ImmutableDenseMatrix, Integral
    >>> from sympy.abc import x
    >>> A = Array([1, 2])
    >>> shape(A)
    (2,)
    >>> shape(Integral(A, x))
    (2,)
    >>> M = ImmutableDenseMatrix([1, 2])
    >>> shape(M)
    (2, 1)
    >>> shape(Integral(M, x))
    (2, 1)

    You can support new type by dispatching.

    >>> from sympy import Expr
    >>> class NewExpr(Expr):
    ...     pass
    >>> @shape.register(NewExpr)
    ... def _(expr):
    ...     return shape(expr.args[0])
    >>> shape(NewExpr(M))
    (2, 1)

    If unsuitable expression is passed, ``NoShapeError()`` will be raised.

    >>> shape(Integral(x, x))
    Traceback (most recent call last):
      ...
    sympy.tensor.functions.NoShapeError: shape() called on non-array object: Integral(x, x)

    Notes
    =====

    Array-like classes (such as ``Matrix`` or ``NDimArray``) has ``shape``
    property which returns its shape, but it cannot be used for non-array
    classes containing array. This function returns the shape of any
    registered object representing array.

    """
    if hasattr(expr, "shape"):
        return expr.shape
    raise NoShapeError(
        "%s does not have shape, or its type is not registered to shape()." % expr)


class NoShapeError(Exception):
    """
    Raised when ``shape()`` is called on non-array object.

    This error can be imported from ``sympy.tensor.functions``.

    Examples
    ========

    >>> from sympy import shape
    >>> from sympy.abc import x
    >>> shape(x)
    Traceback (most recent call last):
      ...
    sympy.tensor.functions.NoShapeError: shape() called on non-array object: x
    """
    pass
