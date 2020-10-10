import os
import textwrap
import pytest

from numpy.testing import assert_, assert_equal, IS_PYPY
from . import util


def _path(*a):
    return os.path.join(*((os.path.dirname(__file__),) + a))


class TestMixed(util.F2PyTest):
    sources = [_path('src', 'mixed', 'foo.f'),
               _path('src', 'mixed', 'foo_fixed.f90'),
               _path('src', 'mixed', 'foo_free.f90')]

    def test_all(self):
        assert_(self.module.bar11() == 11)
        assert_(self.module.foo_fixed.bar12() == 12)
        assert_(self.module.foo_free.bar13() == 13)

    @pytest.mark.xfail(IS_PYPY,
                       reason="PyPy cannot modify tp_doc after PyType_Ready")
    def test_docstring(self):
        expected = textwrap.dedent("""\
        a = bar11()

        Wrapper for ``bar11``.

        Returns
        -------
        a : int
        """)
        assert_equal(self.module.bar11.__doc__, expected)
