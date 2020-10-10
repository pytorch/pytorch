import pytest

from numpy import array
from numpy.testing import assert_, assert_raises
from . import util


class TestReturnLogical(util.F2PyTest):

    def check_function(self, t):
        assert_(t(True) == 1, repr(t(True)))
        assert_(t(False) == 0, repr(t(False)))
        assert_(t(0) == 0)
        assert_(t(None) == 0)
        assert_(t(0.0) == 0)
        assert_(t(0j) == 0)
        assert_(t(1j) == 1)
        assert_(t(234) == 1)
        assert_(t(234.6) == 1)
        assert_(t(234.6 + 3j) == 1)
        assert_(t('234') == 1)
        assert_(t('aaa') == 1)
        assert_(t('') == 0)
        assert_(t([]) == 0)
        assert_(t(()) == 0)
        assert_(t({}) == 0)
        assert_(t(t) == 1)
        assert_(t(-234) == 1)
        assert_(t(10 ** 100) == 1)
        assert_(t([234]) == 1)
        assert_(t((234,)) == 1)
        assert_(t(array(234)) == 1)
        assert_(t(array([234])) == 1)
        assert_(t(array([[234]])) == 1)
        assert_(t(array([234], 'b')) == 1)
        assert_(t(array([234], 'h')) == 1)
        assert_(t(array([234], 'i')) == 1)
        assert_(t(array([234], 'l')) == 1)
        assert_(t(array([234], 'f')) == 1)
        assert_(t(array([234], 'd')) == 1)
        assert_(t(array([234 + 3j], 'F')) == 1)
        assert_(t(array([234], 'D')) == 1)
        assert_(t(array(0)) == 0)
        assert_(t(array([0])) == 0)
        assert_(t(array([[0]])) == 0)
        assert_(t(array([0j])) == 0)
        assert_(t(array([1])) == 1)
        assert_raises(ValueError, t, array([0, 0]))


class TestF77ReturnLogical(TestReturnLogical):
    code = """
       function t0(value)
         logical value
         logical t0
         t0 = value
       end
       function t1(value)
         logical*1 value
         logical*1 t1
         t1 = value
       end
       function t2(value)
         logical*2 value
         logical*2 t2
         t2 = value
       end
       function t4(value)
         logical*4 value
         logical*4 t4
         t4 = value
       end
c       function t8(value)
c         logical*8 value
c         logical*8 t8
c         t8 = value
c       end

       subroutine s0(t0,value)
         logical value
         logical t0
cf2py    intent(out) t0
         t0 = value
       end
       subroutine s1(t1,value)
         logical*1 value
         logical*1 t1
cf2py    intent(out) t1
         t1 = value
       end
       subroutine s2(t2,value)
         logical*2 value
         logical*2 t2
cf2py    intent(out) t2
         t2 = value
       end
       subroutine s4(t4,value)
         logical*4 value
         logical*4 t4
cf2py    intent(out) t4
         t4 = value
       end
c       subroutine s8(t8,value)
c         logical*8 value
c         logical*8 t8
cf2py    intent(out) t8
c         t8 = value
c       end
    """

    @pytest.mark.slow
    @pytest.mark.parametrize('name', 't0,t1,t2,t4,s0,s1,s2,s4'.split(','))
    def test_all(self, name):
        self.check_function(getattr(self.module, name))


class TestF90ReturnLogical(TestReturnLogical):
    suffix = ".f90"
    code = """
module f90_return_logical
  contains
       function t0(value)
         logical :: value
         logical :: t0
         t0 = value
       end function t0
       function t1(value)
         logical(kind=1) :: value
         logical(kind=1) :: t1
         t1 = value
       end function t1
       function t2(value)
         logical(kind=2) :: value
         logical(kind=2) :: t2
         t2 = value
       end function t2
       function t4(value)
         logical(kind=4) :: value
         logical(kind=4) :: t4
         t4 = value
       end function t4
       function t8(value)
         logical(kind=8) :: value
         logical(kind=8) :: t8
         t8 = value
       end function t8

       subroutine s0(t0,value)
         logical :: value
         logical :: t0
!f2py    intent(out) t0
         t0 = value
       end subroutine s0
       subroutine s1(t1,value)
         logical(kind=1) :: value
         logical(kind=1) :: t1
!f2py    intent(out) t1
         t1 = value
       end subroutine s1
       subroutine s2(t2,value)
         logical(kind=2) :: value
         logical(kind=2) :: t2
!f2py    intent(out) t2
         t2 = value
       end subroutine s2
       subroutine s4(t4,value)
         logical(kind=4) :: value
         logical(kind=4) :: t4
!f2py    intent(out) t4
         t4 = value
       end subroutine s4
       subroutine s8(t8,value)
         logical(kind=8) :: value
         logical(kind=8) :: t8
!f2py    intent(out) t8
         t8 = value
       end subroutine s8
end module f90_return_logical
    """

    @pytest.mark.slow
    @pytest.mark.parametrize('name',
                             't0,t1,t2,t4,t8,s0,s1,s2,s4,s8'.split(','))
    def test_all(self, name):
        self.check_function(getattr(self.module.f90_return_logical, name))
