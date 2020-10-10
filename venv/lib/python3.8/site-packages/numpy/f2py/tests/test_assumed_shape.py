import os
import pytest
import tempfile

from numpy.testing import assert_
from . import util


def _path(*a):
    return os.path.join(*((os.path.dirname(__file__),) + a))


class TestAssumedShapeSumExample(util.F2PyTest):
    sources = [_path('src', 'assumed_shape', 'foo_free.f90'),
               _path('src', 'assumed_shape', 'foo_use.f90'),
               _path('src', 'assumed_shape', 'precision.f90'),
               _path('src', 'assumed_shape', 'foo_mod.f90'),
               _path('src', 'assumed_shape', '.f2py_f2cmap'),
               ]

    @pytest.mark.slow
    def test_all(self):
        r = self.module.fsum([1, 2])
        assert_(r == 3, repr(r))
        r = self.module.sum([1, 2])
        assert_(r == 3, repr(r))
        r = self.module.sum_with_use([1, 2])
        assert_(r == 3, repr(r))

        r = self.module.mod.sum([1, 2])
        assert_(r == 3, repr(r))
        r = self.module.mod.fsum([1, 2])
        assert_(r == 3, repr(r))


class TestF2cmapOption(TestAssumedShapeSumExample):
    def setup(self):
        # Use a custom file name for .f2py_f2cmap
        self.sources = list(self.sources)
        f2cmap_src = self.sources.pop(-1)

        self.f2cmap_file = tempfile.NamedTemporaryFile(delete=False)
        with open(f2cmap_src, 'rb') as f:
            self.f2cmap_file.write(f.read())
        self.f2cmap_file.close()

        self.sources.append(self.f2cmap_file.name)
        self.options = ["--f2cmap", self.f2cmap_file.name]

        super(TestF2cmapOption, self).setup()

    def teardown(self):
        os.unlink(self.f2cmap_file.name)
