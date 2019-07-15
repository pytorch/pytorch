r"""
The following constraints are implemented:

- ``constraints.boolean``
- ``constraints.cat``
- ``constraints.dependent``
- ``constraints.greater_than(lower_bound)``
- ``constraints.integer_interval(lower_bound, upper_bound)``
- ``constraints.interval(lower_bound, upper_bound)``
- ``constraints.lower_cholesky``
- ``constraints.lower_triangular``
- ``constraints.nonnegative_integer``
- ``constraints.positive``
- ``constraints.positive_definite``
- ``constraints.positive_integer``
- ``constraints.real``
- ``constraints.real_vector``
- ``constraints.simplex``
- ``constraints.stack``
- ``constraints.unit_interval``
"""

import torch

__all__ = [
    'Constraint',
    'boolean',
    'cat',
    'dependent',
    'dependent_property',
    'greater_than',
    'greater_than_eq',
    'integer_interval',
    'interval',
    'half_open_interval',
    'is_dependent',
    'less_than',
    'lower_cholesky',
    'lower_triangular',
    'nonnegative_integer',
    'positive',
    'positive_definite',
    'positive_integer',
    'real',
    'real_vector',
    'simplex',
    'stack',
    'unit_interval',
]


class Constraint(object):
    """
    Abstract base class for constraints.

    A constraint object represents a region over which a variable is valid,
    e.g. within which a variable can be optimized.
    """
    def check(self, value):
        """
        Returns a byte tensor of `sample_shape + batch_shape` indicating
        whether each event in value satisfies this constraint.
        """
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__[1:] + '()'


class _Dependent(Constraint):
    """
    Placeholder for variables whose support depends on other variables.
    These variables obey no simple coordinate-wise constraints.
    """
    def check(self, x):
        raise ValueError('Cannot determine validity of dependent constraint')


def is_dependent(constraint):
    return isinstance(constraint, _Dependent)


class _DependentProperty(property, _Dependent):
    """
    Decorator that extends @property to act like a `Dependent` constraint when
    called on a class and act like a property when called on an object.

    Example::

        class Uniform(Distribution):
            def __init__(self, low, high):
                self.low = low
                self.high = high
            @constraints.dependent_property
            def support(self):
                return constraints.interval(self.low, self.high)
    """
    pass


class _Boolean(Constraint):
    """
    Constrain to the two values `{0, 1}`.
    """
    def check(self, value):
        return (value == 0) | (value == 1)


class _IntegerInterval(Constraint):
    """
    Constrain to an integer interval `[lower_bound, upper_bound]`.
    """
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def check(self, value):
        return (value % 1 == 0) & (self.lower_bound <= value) & (value <= self.upper_bound)

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += '(lower_bound={}, upper_bound={})'.format(self.lower_bound, self.upper_bound)
        return fmt_string


class _IntegerLessThan(Constraint):
    """
    Constrain to an integer interval `(-inf, upper_bound]`.
    """
    def __init__(self, upper_bound):
        self.upper_bound = upper_bound

    def check(self, value):
        return (value % 1 == 0) & (value <= self.upper_bound)

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += '(upper_bound={})'.format(self.upper_bound)
        return fmt_string


class _IntegerGreaterThan(Constraint):
    """
    Constrain to an integer interval `[lower_bound, inf)`.
    """
    def __init__(self, lower_bound):
        self.lower_bound = lower_bound

    def check(self, value):
        return (value % 1 == 0) & (value >= self.lower_bound)

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += '(lower_bound={})'.format(self.lower_bound)
        return fmt_string


class _Real(Constraint):
    """
    Trivially constrain to the extended real line `[-inf, inf]`.
    """
    def check(self, value):
        return value == value  # False for NANs.


class _GreaterThan(Constraint):
    """
    Constrain to a real half line `(lower_bound, inf]`.
    """
    def __init__(self, lower_bound):
        self.lower_bound = lower_bound

    def check(self, value):
        return self.lower_bound < value

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += '(lower_bound={})'.format(self.lower_bound)
        return fmt_string


class _GreaterThanEq(Constraint):
    """
    Constrain to a real half line `[lower_bound, inf)`.
    """
    def __init__(self, lower_bound):
        self.lower_bound = lower_bound

    def check(self, value):
        return self.lower_bound <= value

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += '(lower_bound={})'.format(self.lower_bound)
        return fmt_string


class _LessThan(Constraint):
    """
    Constrain to a real half line `[-inf, upper_bound)`.
    """
    def __init__(self, upper_bound):
        self.upper_bound = upper_bound

    def check(self, value):
        return value < self.upper_bound

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += '(upper_bound={})'.format(self.upper_bound)
        return fmt_string


class _Interval(Constraint):
    """
    Constrain to a real interval `[lower_bound, upper_bound]`.
    """
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def check(self, value):
        return (self.lower_bound <= value) & (value <= self.upper_bound)

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += '(lower_bound={}, upper_bound={})'.format(self.lower_bound, self.upper_bound)
        return fmt_string


class _HalfOpenInterval(Constraint):
    """
    Constrain to a real interval `[lower_bound, upper_bound)`.
    """
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def check(self, value):
        return (self.lower_bound <= value) & (value < self.upper_bound)

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += '(lower_bound={}, upper_bound={})'.format(self.lower_bound, self.upper_bound)
        return fmt_string


class _Simplex(Constraint):
    """
    Constrain to the unit simplex in the innermost (rightmost) dimension.
    Specifically: `x >= 0` and `x.sum(-1) == 1`.
    """
    def check(self, value):
        return torch.all(value >= 0, dim=-1) & ((value.sum(-1) - 1).abs() < 1e-6)


class _LowerTriangular(Constraint):
    """
    Constrain to lower-triangular square matrices.
    """
    def check(self, value):
        value_tril = value.tril()
        return (value_tril == value).view(value.shape[:-2] + (-1,)).min(-1)[0]


class _LowerCholesky(Constraint):
    """
    Constrain to lower-triangular square matrices with positive diagonals.
    """
    def check(self, value):
        value_tril = value.tril()
        lower_triangular = (value_tril == value).view(value.shape[:-2] + (-1,)).min(-1)[0]

        positive_diagonal = (value.diagonal(dim1=-2, dim2=-1) > 0).min(-1)[0]
        return lower_triangular & positive_diagonal


class _PositiveDefinite(Constraint):
    """
    Constrain to positive-definite matrices.
    """
    def check(self, value):
        matrix_shape = value.shape[-2:]
        batch_shape = value.unsqueeze(0).shape[:-2]
        # TODO: replace with batched linear algebra routine when one becomes available
        # note that `symeig()` returns eigenvalues in ascending order
        flattened_value = value.reshape((-1,) + matrix_shape)
        return torch.stack([v.symeig(eigenvectors=False)[0][:1] > 0.0
                            for v in flattened_value]).view(batch_shape)


class _RealVector(Constraint):
    """
    Constrain to real-valued vectors. This is the same as `constraints.real`,
    but additionally reduces across the `event_shape` dimension.
    """
    def check(self, value):
        return torch.all(value == value, dim=-1)  # False for NANs.


class _Cat(Constraint):
    """
    Constraint functor that applies a sequence of constraints
    `cseq` at the submatrices at dimension `dim`,
    each of size `lengths[dim]`, in a way compatible with :func:`torch.cat`.
    """
    def __init__(self, cseq, dim=0, lengths=None):
        assert all(isinstance(c, Constraint) for c in cseq)
        self.cseq = list(cseq)
        if lengths is None:
            lengths = [1] * len(self.cseq)
        self.lengths = list(lengths)
        assert len(self.lengths) == len(self.cseq)
        self.dim = dim

    def check(self, value):
        assert -value.dim() <= self.dim < value.dim()
        checks = []
        start = 0
        for constr, length in zip(self.cseq, self.lengths):
            v = value.narrow(self.dim, start, length)
            checks.append(constr.check(v))
            start = start + length  # avoid += for jit compat
        return torch.cat(checks, self.dim)


class _Stack(Constraint):
    """
    Constraint functor that applies a sequence of constraints
    `cseq` at the submatrices at dimension `dim`,
    in a way compatible with :func:`torch.stack`.
    """
    def __init__(self, cseq, dim=0):
        assert all(isinstance(c, Constraint) for c in cseq)
        self.cseq = list(cseq)
        self.dim = dim

    def check(self, value):
        assert -value.dim() <= self.dim < value.dim()
        vs = [value.select(self.dim, i) for i in range(value.size(self.dim))]
        return torch.stack([constr.check(v)
                            for v, constr in zip(vs, self.cseq)], self.dim)

# Public interface.
dependent = _Dependent()
dependent_property = _DependentProperty
boolean = _Boolean()
nonnegative_integer = _IntegerGreaterThan(0)
positive_integer = _IntegerGreaterThan(1)
integer_interval = _IntegerInterval
real = _Real()
real_vector = _RealVector()
positive = _GreaterThan(0.)
greater_than = _GreaterThan
greater_than_eq = _GreaterThanEq
less_than = _LessThan
unit_interval = _Interval(0., 1.)
interval = _Interval
half_open_interval = _HalfOpenInterval
simplex = _Simplex()
lower_triangular = _LowerTriangular()
lower_cholesky = _LowerCholesky()
positive_definite = _PositiveDefinite()
cat = _Cat
stack = _Stack
