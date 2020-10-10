import platform
import pytest

from numpy import array
from numpy.testing import assert_, assert_raises
from . import util


class TestReturnReal(util.F2PyTest):

    def check_function(self, t, tname):
        if tname in ['t0', 't4', 's0', 's4']:
            err = 1e-5
        else:
            err = 0.0
        assert_(abs(t(234) - 234.0) <= err)
        assert_(abs(t(234.6) - 234.6) <= err)
        assert_(abs(t('234') - 234) <= err)
        assert_(abs(t('234.6') - 234.6) <= err)
        assert_(abs(t(-234) + 234) <= err)
        assert_(abs(t([234]) - 234) <= err)
        assert_(abs(t((234,)) - 234.) <= err)
        assert_(abs(t(array(234)) - 234.) <= err)
        assert_(abs(t(array([234])) - 234.) <= err)
        assert_(abs(t(array([[234]])) - 234.) <= err)
        assert_(abs(t(array([234], 'b')) + 22) <= err)
        assert_(abs(t(array([234], 'h')) - 234.) <= err)
        assert_(abs(t(array([234], 'i')) - 234.) <= err)
        assert_(abs(t(array([234], 'l')) - 234.) <= err)
        assert_(abs(t(array([234], 'B')) - 234.) <= err)
        assert_(abs(t(array([234], 'f')) - 234.) <= err)
        assert_(abs(t(array([234], 'd')) - 234.) <= err)
        if tname in ['t0', 't4', 's0', 's4']:
            assert_(t(1e200) == t(1e300))  # inf

        #assert_raises(ValueError, t, array([234], 'S1'))
        assert_raises(ValueError, t, 'abc')

        assert_raises(IndexError, t, [])
        assert_raises(IndexError, t, ())

        assert_raises(Exception, t, t)
        assert_raises(Exception, t, {})

        try:
            r = t(10 ** 400)
            assert_(repr(r) in ['inf', 'Infinity'], repr(r))
        except OverflowError:
            pass



@pytest.mark.skipif(
    platform.system() == 'Darwin',
    reason="Prone to error when run with numpy/f2py/tests on mac os, "
           "but not when run in isolation")
class TestCReturnReal(TestReturnReal):
    suffix = ".pyf"
    module_name = "c_ext_return_real"
    code = """
python module c_ext_return_real
usercode \'\'\'
float t4(float value) { return value; }
void s4(float *t4, float value) { *t4 = value; }
double t8(double value) { return value; }
void s8(double *t8, double value) { *t8 = value; }
\'\'\'
interface
  function t4(value)
    real*4 intent(c) :: t4,value
  end
  function t8(value)
    real*8 intent(c) :: t8,value
  end
  subroutine s4(t4,value)
    intent(c) s4
    real*4 intent(out) :: t4
    real*4 intent(c) :: value
  end
  subroutine s8(t8,value)
    intent(c) s8
    real*8 intent(out) :: t8
    real*8 intent(c) :: value
  end
end interface
end python module c_ext_return_real
    """

    @pytest.mark.parametrize('name', 't4,t8,s4,s8'.split(','))
    def test_all(self, name):
        self.check_function(getattr(self.module, name), name)


class TestF77ReturnReal(TestReturnReal):
    code = """
       function t0(value)
         real value
         real t0
         t0 = value
       end
       function t4(value)
         real*4 value
         real*4 t4
         t4 = value
       end
       function t8(value)
         real*8 value
         real*8 t8
         t8 = value
       end
       function td(value)
         double precision value
         double precision td
         td = value
       end

       subroutine s0(t0,value)
         real value
         real t0
cf2py    intent(out) t0
         t0 = value
       end
       subroutine s4(t4,value)
         real*4 value
         real*4 t4
cf2py    intent(out) t4
         t4 = value
       end
       subroutine s8(t8,value)
         real*8 value
         real*8 t8
cf2py    intent(out) t8
         t8 = value
       end
       subroutine sd(td,value)
         double precision value
         double precision td
cf2py    intent(out) td
         td = value
       end
    """

    @pytest.mark.parametrize('name', 't0,t4,t8,td,s0,s4,s8,sd'.split(','))
    def test_all(self, name):
        self.check_function(getattr(self.module, name), name)


class TestF90ReturnReal(TestReturnReal):
    suffix = ".f90"
    code = """
module f90_return_real
  contains
       function t0(value)
         real :: value
         real :: t0
         t0 = value
       end function t0
       function t4(value)
         real(kind=4) :: value
         real(kind=4) :: t4
         t4 = value
       end function t4
       function t8(value)
         real(kind=8) :: value
         real(kind=8) :: t8
         t8 = value
       end function t8
       function td(value)
         double precision :: value
         double precision :: td
         td = value
       end function td

       subroutine s0(t0,value)
         real :: value
         real :: t0
!f2py    intent(out) t0
         t0 = value
       end subroutine s0
       subroutine s4(t4,value)
         real(kind=4) :: value
         real(kind=4) :: t4
!f2py    intent(out) t4
         t4 = value
       end subroutine s4
       subroutine s8(t8,value)
         real(kind=8) :: value
         real(kind=8) :: t8
!f2py    intent(out) t8
         t8 = value
       end subroutine s8
       subroutine sd(td,value)
         double precision :: value
         double precision :: td
!f2py    intent(out) td
         td = value
       end subroutine sd
end module f90_return_real
    """

    @pytest.mark.parametrize('name', 't0,t4,t8,td,s0,s4,s8,sd'.split(','))
    def test_all(self, name):
        self.check_function(getattr(self.module.f90_return_real, name), name)
