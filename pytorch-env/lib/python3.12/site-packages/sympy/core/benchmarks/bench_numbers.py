from sympy.core.numbers import Integer, Rational, pi, oo
from sympy.core.intfunc import integer_nthroot, igcd
from sympy.core.singleton import S

i3 = Integer(3)
i4 = Integer(4)
r34 = Rational(3, 4)
q45 = Rational(4, 5)


def timeit_Integer_create():
    Integer(2)


def timeit_Integer_int():
    int(i3)


def timeit_neg_one():
    -S.One


def timeit_Integer_neg():
    -i3


def timeit_Integer_abs():
    abs(i3)


def timeit_Integer_sub():
    i3 - i3


def timeit_abs_pi():
    abs(pi)


def timeit_neg_oo():
    -oo


def timeit_Integer_add_i1():
    i3 + 1


def timeit_Integer_add_ij():
    i3 + i4


def timeit_Integer_add_Rational():
    i3 + r34


def timeit_Integer_mul_i4():
    i3*4


def timeit_Integer_mul_ij():
    i3*i4


def timeit_Integer_mul_Rational():
    i3*r34


def timeit_Integer_eq_i3():
    i3 == 3


def timeit_Integer_ed_Rational():
    i3 == r34


def timeit_integer_nthroot():
    integer_nthroot(100, 2)


def timeit_number_igcd_23_17():
    igcd(23, 17)


def timeit_number_igcd_60_3600():
    igcd(60, 3600)


def timeit_Rational_add_r1():
    r34 + 1


def timeit_Rational_add_rq():
    r34 + q45
