import pytest

from numpy import array
from numpy.testing import assert_, assert_raises
from . import util


class TestReturnInteger(util.F2PyTest):

    def check_function(self, t, tname):
        assert_(t(123) == 123, repr(t(123)))
        assert_(t(123.6) == 123)
        assert_(t('123') == 123)
        assert_(t(-123) == -123)
        assert_(t([123]) == 123)
        assert_(t((123,)) == 123)
        assert_(t(array(123)) == 123)
        assert_(t(array([123])) == 123)
        assert_(t(array([[123]])) == 123)
        assert_(t(array([123], 'b')) == 123)
        assert_(t(array([123], 'h')) == 123)
        assert_(t(array([123], 'i')) == 123)
        assert_(t(array([123], 'l')) == 123)
        assert_(t(array([123], 'B')) == 123)
        assert_(t(array([123], 'f')) == 123)
        assert_(t(array([123], 'd')) == 123)

        #assert_raises(ValueError, t, array([123],'S3'))
        assert_raises(ValueError, t, 'abc')

        assert_raises(IndexError, t, [])
        assert_raises(IndexError, t, ())

        assert_raises(Exception, t, t)
        assert_raises(Exception, t, {})

        if tname in ['t8', 's8']:
            assert_raises(OverflowError, t, 100000000000000000000000)
            assert_raises(OverflowError, t, 10000000011111111111111.23)


class TestF77ReturnInteger(TestReturnInteger):
    code = """
       function t0(value)
         integer value
         integer t0
         t0 = value
       end
       function t1(value)
         integer*1 value
         integer*1 t1
         t1 = value
       end
       function t2(value)
         integer*2 value
         integer*2 t2
         t2 = value
       end
       function t4(value)
         integer*4 value
         integer*4 t4
         t4 = value
       end
       function t8(value)
         integer*8 value
         integer*8 t8
         t8 = value
       end

       subroutine s0(t0,value)
         integer value
         integer t0
cf2py    intent(out) t0
         t0 = value
       end
       subroutine s1(t1,value)
         integer*1 value
         integer*1 t1
cf2py    intent(out) t1
         t1 = value
       end
       subroutine s2(t2,value)
         integer*2 value
         integer*2 t2
cf2py    intent(out) t2
         t2 = value
       end
       subroutine s4(t4,value)
         integer*4 value
         integer*4 t4
cf2py    intent(out) t4
         t4 = value
       end
       subroutine s8(t8,value)
         integer*8 value
         integer*8 t8
cf2py    intent(out) t8
         t8 = value
       end
    """

    @pytest.mark.parametrize('name',
                             't0,t1,t2,t4,t8,s0,s1,s2,s4,s8'.split(','))
    def test_all(self, name):
        self.check_function(getattr(self.module, name), name)


class TestF90ReturnInteger(TestReturnInteger):
    suffix = ".f90"
    code = """
module f90_return_integer
  contains
       function t0(value)
         integer :: value
         integer :: t0
         t0 = value
       end function t0
       function t1(value)
         integer(kind=1) :: value
         integer(kind=1) :: t1
         t1 = value
       end function t1
       function t2(value)
         integer(kind=2) :: value
         integer(kind=2) :: t2
         t2 = value
       end function t2
       function t4(value)
         integer(kind=4) :: value
         integer(kind=4) :: t4
         t4 = value
       end function t4
       function t8(value)
         integer(kind=8) :: value
         integer(kind=8) :: t8
         t8 = value
       end function t8

       subroutine s0(t0,value)
         integer :: value
         integer :: t0
!f2py    intent(out) t0
         t0 = value
       end subroutine s0
       subroutine s1(t1,value)
         integer(kind=1) :: value
         integer(kind=1) :: t1
!f2py    intent(out) t1
         t1 = value
       end subroutine s1
       subroutine s2(t2,value)
         integer(kind=2) :: value
         integer(kind=2) :: t2
!f2py    intent(out) t2
         t2 = value
       end subroutine s2
       subroutine s4(t4,value)
         integer(kind=4) :: value
         integer(kind=4) :: t4
!f2py    intent(out) t4
         t4 = value
       end subroutine s4
       subroutine s8(t8,value)
         integer(kind=8) :: value
         integer(kind=8) :: t8
!f2py    intent(out) t8
         t8 = value
       end subroutine s8
end module f90_return_integer
    """

    @pytest.mark.parametrize('name',
                             't0,t1,t2,t4,t8,s0,s1,s2,s4,s8'.split(','))
    def test_all(self, name):
        self.check_function(getattr(self.module.f90_return_integer, name), name)
