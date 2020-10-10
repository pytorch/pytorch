import math
import textwrap
import sys
import pytest

import numpy as np
from numpy.testing import assert_, assert_equal, IS_PYPY
from . import util


class TestF77Callback(util.F2PyTest):
    code = """
       subroutine t(fun,a)
       integer a
cf2py  intent(out) a
       external fun
       call fun(a)
       end

       subroutine func(a)
cf2py  intent(in,out) a
       integer a
       a = a + 11
       end

       subroutine func0(a)
cf2py  intent(out) a
       integer a
       a = 11
       end

       subroutine t2(a)
cf2py  intent(callback) fun
       integer a
cf2py  intent(out) a
       external fun
       call fun(a)
       end

       subroutine string_callback(callback, a)
       external callback
       double precision callback
       double precision a
       character*1 r
cf2py  intent(out) a
       r = 'r'
       a = callback(r)
       end

       subroutine string_callback_array(callback, cu, lencu, a)
       external callback
       integer callback
       integer lencu
       character*8 cu(lencu)
       integer a
cf2py  intent(out) a

       a = callback(cu, lencu)
       end
    """

    @pytest.mark.parametrize('name', 't,t2'.split(','))
    def test_all(self, name):
        self.check_function(name)

    @pytest.mark.xfail(IS_PYPY,
                       reason="PyPy cannot modify tp_doc after PyType_Ready")
    def test_docstring(self):
        expected = textwrap.dedent("""\
        a = t(fun,[fun_extra_args])

        Wrapper for ``t``.

        Parameters
        ----------
        fun : call-back function

        Other Parameters
        ----------------
        fun_extra_args : input tuple, optional
            Default: ()

        Returns
        -------
        a : int

        Notes
        -----
        Call-back functions::

          def fun(): return a
          Return objects:
            a : int
        """)
        assert_equal(self.module.t.__doc__, expected)

    def check_function(self, name):
        t = getattr(self.module, name)
        r = t(lambda: 4)
        assert_(r == 4, repr(r))
        r = t(lambda a: 5, fun_extra_args=(6,))
        assert_(r == 5, repr(r))
        r = t(lambda a: a, fun_extra_args=(6,))
        assert_(r == 6, repr(r))
        r = t(lambda a: 5 + a, fun_extra_args=(7,))
        assert_(r == 12, repr(r))
        r = t(lambda a: math.degrees(a), fun_extra_args=(math.pi,))
        assert_(r == 180, repr(r))
        r = t(math.degrees, fun_extra_args=(math.pi,))
        assert_(r == 180, repr(r))

        r = t(self.module.func, fun_extra_args=(6,))
        assert_(r == 17, repr(r))
        r = t(self.module.func0)
        assert_(r == 11, repr(r))
        r = t(self.module.func0._cpointer)
        assert_(r == 11, repr(r))

        class A:

            def __call__(self):
                return 7

            def mth(self):
                return 9
        a = A()
        r = t(a)
        assert_(r == 7, repr(r))
        r = t(a.mth)
        assert_(r == 9, repr(r))

    @pytest.mark.skipif(sys.platform=='win32',
                        reason='Fails with MinGW64 Gfortran (Issue #9673)')
    def test_string_callback(self):

        def callback(code):
            if code == 'r':
                return 0
            else:
                return 1

        f = getattr(self.module, 'string_callback')
        r = f(callback)
        assert_(r == 0, repr(r))

    @pytest.mark.skipif(sys.platform=='win32',
                        reason='Fails with MinGW64 Gfortran (Issue #9673)')
    def test_string_callback_array(self):
        # See gh-10027
        cu = np.zeros((1, 8), 'S1')

        def callback(cu, lencu):
            if cu.shape != (lencu, 8):
                return 1
            if cu.dtype != 'S1':
                return 2
            if not np.all(cu == b''):
                return 3
            return 0

        f = getattr(self.module, 'string_callback_array')
        res = f(callback, cu, len(cu))
        assert_(res == 0, repr(res))
