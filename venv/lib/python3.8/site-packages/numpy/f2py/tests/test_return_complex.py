import pytest

from numpy import array
from numpy.testing import assert_, assert_raises
from . import util


class TestReturnComplex(util.F2PyTest):

    def check_function(self, t, tname):
        if tname in ['t0', 't8', 's0', 's8']:
            err = 1e-5
        else:
            err = 0.0
        assert_(abs(t(234j) - 234.0j) <= err)
        assert_(abs(t(234.6) - 234.6) <= err)
        assert_(abs(t(234) - 234.0) <= err)
        assert_(abs(t(234.6 + 3j) - (234.6 + 3j)) <= err)
        #assert_( abs(t('234')-234.)<=err)
        #assert_( abs(t('234.6')-234.6)<=err)
        assert_(abs(t(-234) + 234.) <= err)
        assert_(abs(t([234]) - 234.) <= err)
        assert_(abs(t((234,)) - 234.) <= err)
        assert_(abs(t(array(234)) - 234.) <= err)
        assert_(abs(t(array(23 + 4j, 'F')) - (23 + 4j)) <= err)
        assert_(abs(t(array([234])) - 234.) <= err)
        assert_(abs(t(array([[234]])) - 234.) <= err)
        assert_(abs(t(array([234], 'b')) + 22.) <= err)
        assert_(abs(t(array([234], 'h')) - 234.) <= err)
        assert_(abs(t(array([234], 'i')) - 234.) <= err)
        assert_(abs(t(array([234], 'l')) - 234.) <= err)
        assert_(abs(t(array([234], 'q')) - 234.) <= err)
        assert_(abs(t(array([234], 'f')) - 234.) <= err)
        assert_(abs(t(array([234], 'd')) - 234.) <= err)
        assert_(abs(t(array([234 + 3j], 'F')) - (234 + 3j)) <= err)
        assert_(abs(t(array([234], 'D')) - 234.) <= err)

        #assert_raises(TypeError, t, array([234], 'a1'))
        assert_raises(TypeError, t, 'abc')

        assert_raises(IndexError, t, [])
        assert_raises(IndexError, t, ())

        assert_raises(TypeError, t, t)
        assert_raises(TypeError, t, {})

        try:
            r = t(10 ** 400)
            assert_(repr(r) in ['(inf+0j)', '(Infinity+0j)'], repr(r))
        except OverflowError:
            pass


class TestF77ReturnComplex(TestReturnComplex):
    code = """
       function t0(value)
         complex value
         complex t0
         t0 = value
       end
       function t8(value)
         complex*8 value
         complex*8 t8
         t8 = value
       end
       function t16(value)
         complex*16 value
         complex*16 t16
         t16 = value
       end
       function td(value)
         double complex value
         double complex td
         td = value
       end

       subroutine s0(t0,value)
         complex value
         complex t0
cf2py    intent(out) t0
         t0 = value
       end
       subroutine s8(t8,value)
         complex*8 value
         complex*8 t8
cf2py    intent(out) t8
         t8 = value
       end
       subroutine s16(t16,value)
         complex*16 value
         complex*16 t16
cf2py    intent(out) t16
         t16 = value
       end
       subroutine sd(td,value)
         double complex value
         double complex td
cf2py    intent(out) td
         td = value
       end
    """

    @pytest.mark.parametrize('name', 't0,t8,t16,td,s0,s8,s16,sd'.split(','))
    def test_all(self, name):
        self.check_function(getattr(self.module, name), name)


class TestF90ReturnComplex(TestReturnComplex):
    suffix = ".f90"
    code = """
module f90_return_complex
  contains
       function t0(value)
         complex :: value
         complex :: t0
         t0 = value
       end function t0
       function t8(value)
         complex(kind=4) :: value
         complex(kind=4) :: t8
         t8 = value
       end function t8
       function t16(value)
         complex(kind=8) :: value
         complex(kind=8) :: t16
         t16 = value
       end function t16
       function td(value)
         double complex :: value
         double complex :: td
         td = value
       end function td

       subroutine s0(t0,value)
         complex :: value
         complex :: t0
!f2py    intent(out) t0
         t0 = value
       end subroutine s0
       subroutine s8(t8,value)
         complex(kind=4) :: value
         complex(kind=4) :: t8
!f2py    intent(out) t8
         t8 = value
       end subroutine s8
       subroutine s16(t16,value)
         complex(kind=8) :: value
         complex(kind=8) :: t16
!f2py    intent(out) t16
         t16 = value
       end subroutine s16
       subroutine sd(td,value)
         double complex :: value
         double complex :: td
!f2py    intent(out) td
         td = value
       end subroutine sd
end module f90_return_complex
    """

    @pytest.mark.parametrize('name', 't0,t8,t16,td,s0,s8,s16,sd'.split(','))
    def test_all(self, name):
        self.check_function(getattr(self.module.f90_return_complex, name), name)
