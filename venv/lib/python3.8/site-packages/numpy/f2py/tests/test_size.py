import os
import pytest

from numpy.testing import assert_equal
from . import util


def _path(*a):
    return os.path.join(*((os.path.dirname(__file__),) + a))


class TestSizeSumExample(util.F2PyTest):
    sources = [_path('src', 'size', 'foo.f90')]

    @pytest.mark.slow
    def test_all(self):
        r = self.module.foo([[]])
        assert_equal(r, [0], repr(r))

        r = self.module.foo([[1, 2]])
        assert_equal(r, [3], repr(r))

        r = self.module.foo([[1, 2], [3, 4]])
        assert_equal(r, [3, 7], repr(r))

        r = self.module.foo([[1, 2], [3, 4], [5, 6]])
        assert_equal(r, [3, 7, 11], repr(r))

    @pytest.mark.slow
    def test_transpose(self):
        r = self.module.trans([[]])
        assert_equal(r.T, [[]], repr(r))

        r = self.module.trans([[1, 2]])
        assert_equal(r, [[1], [2]], repr(r))

        r = self.module.trans([[1, 2, 3], [4, 5, 6]])
        assert_equal(r, [[1, 4], [2, 5], [3, 6]], repr(r))

    @pytest.mark.slow
    def test_flatten(self):
        r = self.module.flatten([[]])
        assert_equal(r, [], repr(r))

        r = self.module.flatten([[1, 2]])
        assert_equal(r, [1, 2], repr(r))

        r = self.module.flatten([[1, 2, 3], [4, 5, 6]])
        assert_equal(r, [1, 2, 3, 4, 5, 6], repr(r))
