""" Common Exceptions for `holonomic` module. """

class BaseHolonomicError(Exception):

    def new(self, *args):
        raise NotImplementedError("abstract base class")

class NotPowerSeriesError(BaseHolonomicError):

    def __init__(self, holonomic, x0):
        self.holonomic = holonomic
        self.x0 = x0

    def __str__(self):
        s = 'A Power Series does not exists for '
        s += str(self.holonomic)
        s += ' about %s.' %self.x0
        return s

class NotHolonomicError(BaseHolonomicError):

    def __init__(self, m):
        self.m = m

    def __str__(self):
        return self.m

class SingularityError(BaseHolonomicError):

    def __init__(self, holonomic, x0):
        self.holonomic = holonomic
        self.x0 = x0

    def __str__(self):
        s = str(self.holonomic)
        s += ' has a singularity at %s.' %self.x0
        return s

class NotHyperSeriesError(BaseHolonomicError):

    def __init__(self, holonomic, x0):
        self.holonomic = holonomic
        self.x0 = x0

    def __str__(self):
        s = 'Power series expansion of '
        s += str(self.holonomic)
        s += ' about %s is not hypergeometric' %self.x0
        return s
