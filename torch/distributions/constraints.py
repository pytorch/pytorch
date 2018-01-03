import torch
from torch.nn.functional import sigmoid, softmax


class Constraint(object):
    """
    Abstract base class for constraints.

    A constraint object represents a region over which a continuous variable is
    valid, e.g. within which a variable can be optimized.
    """
    def check(self, value):
        """
        Returns a byte tensor of sample_shape + batch_shape indicating whether
        each value satisfies this constraint.
        """
        raise NotImplementedError


class Dependent(Constraint):
    """
    Placeholder for variables whose support depends on other variables.
    These variables obey no simple coordinate-wise constraints.
    """
    def check(self, x):
        raise ValueError('Cannot determine validity of dependent constraint')


class Boolean(Constraint):
    """
    Constrain to the two values `{0, 1}`.
    """
    def check(self, value):
        return (value == 0) | (value == 1)


class NonnegativeInteger(Constraint):
    """
    Constrain to non-negative integers `{0, 1, 2, ...}`.
    """
    def check(self, value):
        return (value % 1 == 0) & (value >= 0)


class IntegerInterval(Constraint):
    """
    Constrain to an integer interval `[lower_bound, upper_bound]`.
    """
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def check(self, value):
        return (value % 1 == 0) & (self.lower_bound <= value) & (value <= self.upper_bound)


class Real(Constraint):
    """
    Trivially constrain to the extended real line `[-inf, inf]`.
    """
    def check(self, value):
        return value == value  # False for NANs.


class Positive(Constraint):
    """
    Constrain to the positive half line `[0, inf]`.
    """
    def check(self, value):
        return value >= 0


class Interval(Constraint):
    """
    Constrain to a real interval `[lower_bound, upper_bound]`.
    """
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def check(self, value):
        return (self.lower_bound <= value) & (value <= self.upper_bound)


class Simplex(Constraint):
    """
    Constrain to the unit simplex in the innermost (rightmost) dimension.
    Specifically: `x >= 0` and `x.sum(-1) == 1`.
    """
    def check(self, value):
        return (value >= 0) & ((value.sum(-1, True) - 1).abs() < 1e-6)


class LowerTriangular(Constraint):
    """
    Constrain to lower-triangular square matrices.
    """
    def check(self, value):
        return (torch.tril(value) == value).min(-1).min(-1)


# Functions and constants are the recommended interface.
dependent = Dependent()
boolean = Boolean()
nonnegative_integer = NonnegativeInteger()
integer_interval = IntegerInterval
real = Real()
positive = Positive()
unit_interval = Interval(0, 1)
interval = Interval
simplex = Simplex()
lower_triangular = LowerTriangular()
