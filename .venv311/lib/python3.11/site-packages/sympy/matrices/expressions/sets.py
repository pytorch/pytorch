from sympy.core.assumptions import check_assumptions
from sympy.core.logic import fuzzy_and
from sympy.core.sympify import _sympify
from sympy.matrices.kind import MatrixKind
from sympy.sets.sets import Set, SetKind
from sympy.core.kind import NumberKind
from .matexpr import MatrixExpr


class MatrixSet(Set):
    """
    MatrixSet represents the set of matrices with ``shape = (n, m)`` over the
    given set.

    Examples
    ========

    >>> from sympy.matrices import MatrixSet
    >>> from sympy import S, I, Matrix
    >>> M = MatrixSet(2, 2, set=S.Reals)
    >>> X = Matrix([[1, 2], [3, 4]])
    >>> X in M
    True
    >>> X = Matrix([[1, 2], [I, 4]])
    >>> X in M
    False

    """
    is_empty = False

    def __new__(cls, n, m, set):
        n, m, set = _sympify(n), _sympify(m), _sympify(set)
        cls._check_dim(n)
        cls._check_dim(m)
        if not isinstance(set, Set):
            raise TypeError("{} should be an instance of Set.".format(set))
        return Set.__new__(cls, n, m, set)

    @property
    def shape(self):
        return self.args[:2]

    @property
    def set(self):
        return self.args[2]

    def _contains(self, other):
        if not isinstance(other, MatrixExpr):
            raise TypeError("{} should be an instance of MatrixExpr.".format(other))
        if other.shape != self.shape:
            are_symbolic = any(_sympify(x).is_Symbol for x in other.shape + self.shape)
            if are_symbolic:
                return None
            return False
        return fuzzy_and(self.set.contains(x) for x in other)

    @classmethod
    def _check_dim(cls, dim):
        """Helper function to check invalid matrix dimensions"""
        ok = not dim.is_Float and check_assumptions(
            dim, integer=True, nonnegative=True)
        if ok is False:
            raise ValueError(
                "The dimension specification {} should be "
                "a nonnegative integer.".format(dim))

    def _kind(self):
        return SetKind(MatrixKind(NumberKind))
