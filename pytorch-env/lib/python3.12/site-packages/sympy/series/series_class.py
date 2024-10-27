"""
Contains the base class for series
Made using sequences in mind
"""

from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.core.cache import cacheit


class SeriesBase(Expr):
    """Base Class for series"""

    @property
    def interval(self):
        """The interval on which the series is defined"""
        raise NotImplementedError("(%s).interval" % self)

    @property
    def start(self):
        """The starting point of the series. This point is included"""
        raise NotImplementedError("(%s).start" % self)

    @property
    def stop(self):
        """The ending point of the series. This point is included"""
        raise NotImplementedError("(%s).stop" % self)

    @property
    def length(self):
        """Length of the series expansion"""
        raise NotImplementedError("(%s).length" % self)

    @property
    def variables(self):
        """Returns a tuple of variables that are bounded"""
        return ()

    @property
    def free_symbols(self):
        """
        This method returns the symbols in the object, excluding those
        that take on a specific value (i.e. the dummy symbols).
        """
        return ({j for i in self.args for j in i.free_symbols}
                .difference(self.variables))

    @cacheit
    def term(self, pt):
        """Term at point pt of a series"""
        if pt < self.start or pt > self.stop:
            raise IndexError("Index %s out of bounds %s" % (pt, self.interval))
        return self._eval_term(pt)

    def _eval_term(self, pt):
        raise NotImplementedError("The _eval_term method should be added to"
                                  "%s to return series term so it is available"
                                  "when 'term' calls it."
                                  % self.func)

    def _ith_point(self, i):
        """
        Returns the i'th point of a series
        If start point is negative infinity, point is returned from the end.
        Assumes the first point to be indexed zero.

        Examples
        ========

        TODO
        """
        if self.start is S.NegativeInfinity:
            initial = self.stop
            step = -1
        else:
            initial = self.start
            step = 1

        return initial + i*step

    def __iter__(self):
        i = 0
        while i < self.length:
            pt = self._ith_point(i)
            yield self.term(pt)
            i += 1

    def __getitem__(self, index):
        if isinstance(index, int):
            index = self._ith_point(index)
            return self.term(index)
        elif isinstance(index, slice):
            start, stop = index.start, index.stop
            if start is None:
                start = 0
            if stop is None:
                stop = self.length
            return [self.term(self._ith_point(i)) for i in
                    range(start, stop, index.step or 1)]
