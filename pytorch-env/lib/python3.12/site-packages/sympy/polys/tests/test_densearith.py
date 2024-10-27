"""Tests for dense recursive polynomials' arithmetics. """

from sympy.external.gmpy import GROUND_TYPES

from sympy.polys.densebasic import (
    dup_normal, dmp_normal,
)

from sympy.polys.densearith import (
    dup_add_term, dmp_add_term,
    dup_sub_term, dmp_sub_term,
    dup_mul_term, dmp_mul_term,
    dup_add_ground, dmp_add_ground,
    dup_sub_ground, dmp_sub_ground,
    dup_mul_ground, dmp_mul_ground,
    dup_quo_ground, dmp_quo_ground,
    dup_exquo_ground, dmp_exquo_ground,
    dup_lshift, dup_rshift,
    dup_abs, dmp_abs,
    dup_neg, dmp_neg,
    dup_add, dmp_add,
    dup_sub, dmp_sub,
    dup_mul, dmp_mul,
    dup_sqr, dmp_sqr,
    dup_pow, dmp_pow,
    dup_add_mul, dmp_add_mul,
    dup_sub_mul, dmp_sub_mul,
    dup_pdiv, dup_prem, dup_pquo, dup_pexquo,
    dmp_pdiv, dmp_prem, dmp_pquo, dmp_pexquo,
    dup_rr_div, dmp_rr_div,
    dup_ff_div, dmp_ff_div,
    dup_div, dup_rem, dup_quo, dup_exquo,
    dmp_div, dmp_rem, dmp_quo, dmp_exquo,
    dup_max_norm, dmp_max_norm,
    dup_l1_norm, dmp_l1_norm,
    dup_l2_norm_squared, dmp_l2_norm_squared,
    dup_expand, dmp_expand,
)

from sympy.polys.polyerrors import (
    ExactQuotientFailed,
)

from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import FF, ZZ, QQ

from sympy.testing.pytest import raises

f_0, f_1, f_2, f_3, f_4, f_5, f_6 = [ f.to_dense() for f in f_polys() ]
F_0 = dmp_mul_ground(dmp_normal(f_0, 2, QQ), QQ(1, 7), 2, QQ)

def test_dup_add_term():
    f = dup_normal([], ZZ)

    assert dup_add_term(f, ZZ(0), 0, ZZ) == dup_normal([], ZZ)

    assert dup_add_term(f, ZZ(1), 0, ZZ) == dup_normal([1], ZZ)
    assert dup_add_term(f, ZZ(1), 1, ZZ) == dup_normal([1, 0], ZZ)
    assert dup_add_term(f, ZZ(1), 2, ZZ) == dup_normal([1, 0, 0], ZZ)

    f = dup_normal([1, 1, 1], ZZ)

    assert dup_add_term(f, ZZ(1), 0, ZZ) == dup_normal([1, 1, 2], ZZ)
    assert dup_add_term(f, ZZ(1), 1, ZZ) == dup_normal([1, 2, 1], ZZ)
    assert dup_add_term(f, ZZ(1), 2, ZZ) == dup_normal([2, 1, 1], ZZ)

    assert dup_add_term(f, ZZ(1), 3, ZZ) == dup_normal([1, 1, 1, 1], ZZ)
    assert dup_add_term(f, ZZ(1), 4, ZZ) == dup_normal([1, 0, 1, 1, 1], ZZ)
    assert dup_add_term(f, ZZ(1), 5, ZZ) == dup_normal([1, 0, 0, 1, 1, 1], ZZ)
    assert dup_add_term(
        f, ZZ(1), 6, ZZ) == dup_normal([1, 0, 0, 0, 1, 1, 1], ZZ)

    assert dup_add_term(f, ZZ(-1), 2, ZZ) == dup_normal([1, 1], ZZ)


def test_dmp_add_term():
    assert dmp_add_term([ZZ(1), ZZ(1), ZZ(1)], ZZ(1), 2, 0, ZZ) == \
        dup_add_term([ZZ(1), ZZ(1), ZZ(1)], ZZ(1), 2, ZZ)
    assert dmp_add_term(f_0, [[]], 3, 2, ZZ) == f_0
    assert dmp_add_term(F_0, [[]], 3, 2, QQ) == F_0


def test_dup_sub_term():
    f = dup_normal([], ZZ)

    assert dup_sub_term(f, ZZ(0), 0, ZZ) == dup_normal([], ZZ)

    assert dup_sub_term(f, ZZ(1), 0, ZZ) == dup_normal([-1], ZZ)
    assert dup_sub_term(f, ZZ(1), 1, ZZ) == dup_normal([-1, 0], ZZ)
    assert dup_sub_term(f, ZZ(1), 2, ZZ) == dup_normal([-1, 0, 0], ZZ)

    f = dup_normal([1, 1, 1], ZZ)

    assert dup_sub_term(f, ZZ(2), 0, ZZ) == dup_normal([ 1, 1, -1], ZZ)
    assert dup_sub_term(f, ZZ(2), 1, ZZ) == dup_normal([ 1, -1, 1], ZZ)
    assert dup_sub_term(f, ZZ(2), 2, ZZ) == dup_normal([-1, 1, 1], ZZ)

    assert dup_sub_term(f, ZZ(1), 3, ZZ) == dup_normal([-1, 1, 1, 1], ZZ)
    assert dup_sub_term(f, ZZ(1), 4, ZZ) == dup_normal([-1, 0, 1, 1, 1], ZZ)
    assert dup_sub_term(f, ZZ(1), 5, ZZ) == dup_normal([-1, 0, 0, 1, 1, 1], ZZ)
    assert dup_sub_term(
        f, ZZ(1), 6, ZZ) == dup_normal([-1, 0, 0, 0, 1, 1, 1], ZZ)

    assert dup_sub_term(f, ZZ(1), 2, ZZ) == dup_normal([1, 1], ZZ)


def test_dmp_sub_term():
    assert dmp_sub_term([ZZ(1), ZZ(1), ZZ(1)], ZZ(1), 2, 0, ZZ) == \
        dup_sub_term([ZZ(1), ZZ(1), ZZ(1)], ZZ(1), 2, ZZ)
    assert dmp_sub_term(f_0, [[]], 3, 2, ZZ) == f_0
    assert dmp_sub_term(F_0, [[]], 3, 2, QQ) == F_0


def test_dup_mul_term():
    f = dup_normal([], ZZ)

    assert dup_mul_term(f, ZZ(2), 3, ZZ) == dup_normal([], ZZ)

    f = dup_normal([1, 1], ZZ)

    assert dup_mul_term(f, ZZ(0), 3, ZZ) == dup_normal([], ZZ)

    f = dup_normal([1, 2, 3], ZZ)

    assert dup_mul_term(f, ZZ(2), 0, ZZ) == dup_normal([2, 4, 6], ZZ)
    assert dup_mul_term(f, ZZ(2), 1, ZZ) == dup_normal([2, 4, 6, 0], ZZ)
    assert dup_mul_term(f, ZZ(2), 2, ZZ) == dup_normal([2, 4, 6, 0, 0], ZZ)
    assert dup_mul_term(f, ZZ(2), 3, ZZ) == dup_normal([2, 4, 6, 0, 0, 0], ZZ)


def test_dmp_mul_term():
    assert dmp_mul_term([ZZ(1), ZZ(2), ZZ(3)], ZZ(2), 1, 0, ZZ) == \
        dup_mul_term([ZZ(1), ZZ(2), ZZ(3)], ZZ(2), 1, ZZ)

    assert dmp_mul_term([[]], [ZZ(2)], 3, 1, ZZ) == [[]]
    assert dmp_mul_term([[ZZ(1)]], [], 3, 1, ZZ) == [[]]

    assert dmp_mul_term([[ZZ(1), ZZ(2)], [ZZ(3)]], [ZZ(2)], 2, 1, ZZ) == \
        [[ZZ(2), ZZ(4)], [ZZ(6)], [], []]

    assert dmp_mul_term([[]], [QQ(2, 3)], 3, 1, QQ) == [[]]
    assert dmp_mul_term([[QQ(1, 2)]], [], 3, 1, QQ) == [[]]

    assert dmp_mul_term([[QQ(1, 5), QQ(2, 5)], [QQ(3, 5)]], [QQ(2, 3)], 2, 1, QQ) == \
        [[QQ(2, 15), QQ(4, 15)], [QQ(6, 15)], [], []]


def test_dup_add_ground():
    f = ZZ.map([1, 2, 3, 4])
    g = ZZ.map([1, 2, 3, 8])

    assert dup_add_ground(f, ZZ(4), ZZ) == g


def test_dmp_add_ground():
    f = ZZ.map([[1], [2], [3], [4]])
    g = ZZ.map([[1], [2], [3], [8]])

    assert dmp_add_ground(f, ZZ(4), 1, ZZ) == g


def test_dup_sub_ground():
    f = ZZ.map([1, 2, 3, 4])
    g = ZZ.map([1, 2, 3, 0])

    assert dup_sub_ground(f, ZZ(4), ZZ) == g


def test_dmp_sub_ground():
    f = ZZ.map([[1], [2], [3], [4]])
    g = ZZ.map([[1], [2], [3], []])

    assert dmp_sub_ground(f, ZZ(4), 1, ZZ) == g


def test_dup_mul_ground():
    f = dup_normal([], ZZ)

    assert dup_mul_ground(f, ZZ(2), ZZ) == dup_normal([], ZZ)

    f = dup_normal([1, 2, 3], ZZ)

    assert dup_mul_ground(f, ZZ(0), ZZ) == dup_normal([], ZZ)
    assert dup_mul_ground(f, ZZ(2), ZZ) == dup_normal([2, 4, 6], ZZ)


def test_dmp_mul_ground():
    assert dmp_mul_ground(f_0, ZZ(2), 2, ZZ) == [
        [[ZZ(2), ZZ(4), ZZ(6)], [ZZ(4)]],
        [[ZZ(6)]],
        [[ZZ(8), ZZ(10), ZZ(12)], [ZZ(2), ZZ(4), ZZ(2)], [ZZ(2)]]
    ]

    assert dmp_mul_ground(F_0, QQ(1, 2), 2, QQ) == [
        [[QQ(1, 14), QQ(2, 14), QQ(3, 14)], [QQ(2, 14)]],
        [[QQ(3, 14)]],
        [[QQ(4, 14), QQ(5, 14), QQ(6, 14)], [QQ(1, 14), QQ(2, 14),
             QQ(1, 14)], [QQ(1, 14)]]
    ]


def test_dup_quo_ground():
    raises(ZeroDivisionError, lambda: dup_quo_ground(dup_normal([1, 2,
           3], ZZ), ZZ(0), ZZ))

    f = dup_normal([], ZZ)

    assert dup_quo_ground(f, ZZ(3), ZZ) == dup_normal([], ZZ)

    f = dup_normal([6, 2, 8], ZZ)

    assert dup_quo_ground(f, ZZ(1), ZZ) == f
    assert dup_quo_ground(f, ZZ(2), ZZ) == dup_normal([3, 1, 4], ZZ)

    assert dup_quo_ground(f, ZZ(3), ZZ) == dup_normal([2, 0, 2], ZZ)

    f = dup_normal([6, 2, 8], QQ)

    assert dup_quo_ground(f, QQ(1), QQ) == f
    assert dup_quo_ground(f, QQ(2), QQ) == [QQ(3), QQ(1), QQ(4)]
    assert dup_quo_ground(f, QQ(7), QQ) == [QQ(6, 7), QQ(2, 7), QQ(8, 7)]


def test_dup_exquo_ground():
    raises(ZeroDivisionError, lambda: dup_exquo_ground(dup_normal([1,
           2, 3], ZZ), ZZ(0), ZZ))
    raises(ExactQuotientFailed, lambda: dup_exquo_ground(dup_normal([1,
           2, 3], ZZ), ZZ(3), ZZ))

    f = dup_normal([], ZZ)

    assert dup_exquo_ground(f, ZZ(3), ZZ) == dup_normal([], ZZ)

    f = dup_normal([6, 2, 8], ZZ)

    assert dup_exquo_ground(f, ZZ(1), ZZ) == f
    assert dup_exquo_ground(f, ZZ(2), ZZ) == dup_normal([3, 1, 4], ZZ)

    f = dup_normal([6, 2, 8], QQ)

    assert dup_exquo_ground(f, QQ(1), QQ) == f
    assert dup_exquo_ground(f, QQ(2), QQ) == [QQ(3), QQ(1), QQ(4)]
    assert dup_exquo_ground(f, QQ(7), QQ) == [QQ(6, 7), QQ(2, 7), QQ(8, 7)]


def test_dmp_quo_ground():
    f = dmp_normal([[6], [2], [8]], 1, ZZ)

    assert dmp_quo_ground(f, ZZ(1), 1, ZZ) == f
    assert dmp_quo_ground(
        f, ZZ(2), 1, ZZ) == dmp_normal([[3], [1], [4]], 1, ZZ)

    assert dmp_normal(dmp_quo_ground(
        f, ZZ(3), 1, ZZ), 1, ZZ) == dmp_normal([[2], [], [2]], 1, ZZ)


def test_dmp_exquo_ground():
    f = dmp_normal([[6], [2], [8]], 1, ZZ)

    assert dmp_exquo_ground(f, ZZ(1), 1, ZZ) == f
    assert dmp_exquo_ground(
        f, ZZ(2), 1, ZZ) == dmp_normal([[3], [1], [4]], 1, ZZ)


def test_dup_lshift():
    assert dup_lshift([], 3, ZZ) == []
    assert dup_lshift([1], 3, ZZ) == [1, 0, 0, 0]


def test_dup_rshift():
    assert dup_rshift([], 3, ZZ) == []
    assert dup_rshift([1, 0, 0, 0], 3, ZZ) == [1]


def test_dup_abs():
    assert dup_abs([], ZZ) == []
    assert dup_abs([ZZ( 1)], ZZ) == [ZZ(1)]
    assert dup_abs([ZZ(-7)], ZZ) == [ZZ(7)]
    assert dup_abs([ZZ(-1), ZZ(2), ZZ(3)], ZZ) == [ZZ(1), ZZ(2), ZZ(3)]

    assert dup_abs([], QQ) == []
    assert dup_abs([QQ( 1, 2)], QQ) == [QQ(1, 2)]
    assert dup_abs([QQ(-7, 3)], QQ) == [QQ(7, 3)]
    assert dup_abs(
        [QQ(-1, 7), QQ(2, 7), QQ(3, 7)], QQ) == [QQ(1, 7), QQ(2, 7), QQ(3, 7)]


def test_dmp_abs():
    assert dmp_abs([ZZ(-1)], 0, ZZ) == [ZZ(1)]
    assert dmp_abs([QQ(-1, 2)], 0, QQ) == [QQ(1, 2)]

    assert dmp_abs([[[]]], 2, ZZ) == [[[]]]
    assert dmp_abs([[[ZZ(1)]]], 2, ZZ) == [[[ZZ(1)]]]
    assert dmp_abs([[[ZZ(-7)]]], 2, ZZ) == [[[ZZ(7)]]]

    assert dmp_abs([[[]]], 2, QQ) == [[[]]]
    assert dmp_abs([[[QQ(1, 2)]]], 2, QQ) == [[[QQ(1, 2)]]]
    assert dmp_abs([[[QQ(-7, 9)]]], 2, QQ) == [[[QQ(7, 9)]]]


def test_dup_neg():
    assert dup_neg([], ZZ) == []
    assert dup_neg([ZZ(1)], ZZ) == [ZZ(-1)]
    assert dup_neg([ZZ(-7)], ZZ) == [ZZ(7)]
    assert dup_neg([ZZ(-1), ZZ(2), ZZ(3)], ZZ) == [ZZ(1), ZZ(-2), ZZ(-3)]

    assert dup_neg([], QQ) == []
    assert dup_neg([QQ(1, 2)], QQ) == [QQ(-1, 2)]
    assert dup_neg([QQ(-7, 9)], QQ) == [QQ(7, 9)]
    assert dup_neg([QQ(
        -1, 7), QQ(2, 7), QQ(3, 7)], QQ) == [QQ(1, 7), QQ(-2, 7), QQ(-3, 7)]


def test_dmp_neg():
    assert dmp_neg([ZZ(-1)], 0, ZZ) == [ZZ(1)]
    assert dmp_neg([QQ(-1, 2)], 0, QQ) == [QQ(1, 2)]

    assert dmp_neg([[[]]], 2, ZZ) == [[[]]]
    assert dmp_neg([[[ZZ(1)]]], 2, ZZ) == [[[ZZ(-1)]]]
    assert dmp_neg([[[ZZ(-7)]]], 2, ZZ) == [[[ZZ(7)]]]

    assert dmp_neg([[[]]], 2, QQ) == [[[]]]
    assert dmp_neg([[[QQ(1, 9)]]], 2, QQ) == [[[QQ(-1, 9)]]]
    assert dmp_neg([[[QQ(-7, 9)]]], 2, QQ) == [[[QQ(7, 9)]]]


def test_dup_add():
    assert dup_add([], [], ZZ) == []
    assert dup_add([ZZ(1)], [], ZZ) == [ZZ(1)]
    assert dup_add([], [ZZ(1)], ZZ) == [ZZ(1)]
    assert dup_add([ZZ(1)], [ZZ(1)], ZZ) == [ZZ(2)]
    assert dup_add([ZZ(1)], [ZZ(2)], ZZ) == [ZZ(3)]

    assert dup_add([ZZ(1), ZZ(2)], [ZZ(1)], ZZ) == [ZZ(1), ZZ(3)]
    assert dup_add([ZZ(1)], [ZZ(1), ZZ(2)], ZZ) == [ZZ(1), ZZ(3)]

    assert dup_add([ZZ(1), ZZ(
        2), ZZ(3)], [ZZ(8), ZZ(9), ZZ(10)], ZZ) == [ZZ(9), ZZ(11), ZZ(13)]

    assert dup_add([], [], QQ) == []
    assert dup_add([QQ(1, 2)], [], QQ) == [QQ(1, 2)]
    assert dup_add([], [QQ(1, 2)], QQ) == [QQ(1, 2)]
    assert dup_add([QQ(1, 4)], [QQ(1, 4)], QQ) == [QQ(1, 2)]
    assert dup_add([QQ(1, 4)], [QQ(1, 2)], QQ) == [QQ(3, 4)]

    assert dup_add([QQ(1, 2), QQ(2, 3)], [QQ(1)], QQ) == [QQ(1, 2), QQ(5, 3)]
    assert dup_add([QQ(1)], [QQ(1, 2), QQ(2, 3)], QQ) == [QQ(1, 2), QQ(5, 3)]

    assert dup_add([QQ(1, 7), QQ(2, 7), QQ(3, 7)], [QQ(
        8, 7), QQ(9, 7), QQ(10, 7)], QQ) == [QQ(9, 7), QQ(11, 7), QQ(13, 7)]


def test_dmp_add():
    assert dmp_add([ZZ(1), ZZ(2)], [ZZ(1)], 0, ZZ) == \
        dup_add([ZZ(1), ZZ(2)], [ZZ(1)], ZZ)
    assert dmp_add([QQ(1, 2), QQ(2, 3)], [QQ(1)], 0, QQ) == \
        dup_add([QQ(1, 2), QQ(2, 3)], [QQ(1)], QQ)

    assert dmp_add([[[]]], [[[]]], 2, ZZ) == [[[]]]
    assert dmp_add([[[ZZ(1)]]], [[[]]], 2, ZZ) == [[[ZZ(1)]]]
    assert dmp_add([[[]]], [[[ZZ(1)]]], 2, ZZ) == [[[ZZ(1)]]]
    assert dmp_add([[[ZZ(2)]]], [[[ZZ(1)]]], 2, ZZ) == [[[ZZ(3)]]]
    assert dmp_add([[[ZZ(1)]]], [[[ZZ(2)]]], 2, ZZ) == [[[ZZ(3)]]]

    assert dmp_add([[[]]], [[[]]], 2, QQ) == [[[]]]
    assert dmp_add([[[QQ(1, 2)]]], [[[]]], 2, QQ) == [[[QQ(1, 2)]]]
    assert dmp_add([[[]]], [[[QQ(1, 2)]]], 2, QQ) == [[[QQ(1, 2)]]]
    assert dmp_add([[[QQ(2, 7)]]], [[[QQ(1, 7)]]], 2, QQ) == [[[QQ(3, 7)]]]
    assert dmp_add([[[QQ(1, 7)]]], [[[QQ(2, 7)]]], 2, QQ) == [[[QQ(3, 7)]]]


def test_dup_sub():
    assert dup_sub([], [], ZZ) == []
    assert dup_sub([ZZ(1)], [], ZZ) == [ZZ(1)]
    assert dup_sub([], [ZZ(1)], ZZ) == [ZZ(-1)]
    assert dup_sub([ZZ(1)], [ZZ(1)], ZZ) == []
    assert dup_sub([ZZ(1)], [ZZ(2)], ZZ) == [ZZ(-1)]

    assert dup_sub([ZZ(1), ZZ(2)], [ZZ(1)], ZZ) == [ZZ(1), ZZ(1)]
    assert dup_sub([ZZ(1)], [ZZ(1), ZZ(2)], ZZ) == [ZZ(-1), ZZ(-1)]

    assert dup_sub([ZZ(3), ZZ(
        2), ZZ(1)], [ZZ(8), ZZ(9), ZZ(10)], ZZ) == [ZZ(-5), ZZ(-7), ZZ(-9)]

    assert dup_sub([], [], QQ) == []
    assert dup_sub([QQ(1, 2)], [], QQ) == [QQ(1, 2)]
    assert dup_sub([], [QQ(1, 2)], QQ) == [QQ(-1, 2)]
    assert dup_sub([QQ(1, 3)], [QQ(1, 3)], QQ) == []
    assert dup_sub([QQ(1, 3)], [QQ(2, 3)], QQ) == [QQ(-1, 3)]

    assert dup_sub([QQ(1, 7), QQ(2, 7)], [QQ(1)], QQ) == [QQ(1, 7), QQ(-5, 7)]
    assert dup_sub([QQ(1)], [QQ(1, 7), QQ(2, 7)], QQ) == [QQ(-1, 7), QQ(5, 7)]

    assert dup_sub([QQ(3, 7), QQ(2, 7), QQ(1, 7)], [QQ(
        8, 7), QQ(9, 7), QQ(10, 7)], QQ) == [QQ(-5, 7), QQ(-7, 7), QQ(-9, 7)]


def test_dmp_sub():
    assert dmp_sub([ZZ(1), ZZ(2)], [ZZ(1)], 0, ZZ) == \
        dup_sub([ZZ(1), ZZ(2)], [ZZ(1)], ZZ)
    assert dmp_sub([QQ(1, 2), QQ(2, 3)], [QQ(1)], 0, QQ) == \
        dup_sub([QQ(1, 2), QQ(2, 3)], [QQ(1)], QQ)

    assert dmp_sub([[[]]], [[[]]], 2, ZZ) == [[[]]]
    assert dmp_sub([[[ZZ(1)]]], [[[]]], 2, ZZ) == [[[ZZ(1)]]]
    assert dmp_sub([[[]]], [[[ZZ(1)]]], 2, ZZ) == [[[ZZ(-1)]]]
    assert dmp_sub([[[ZZ(2)]]], [[[ZZ(1)]]], 2, ZZ) == [[[ZZ(1)]]]
    assert dmp_sub([[[ZZ(1)]]], [[[ZZ(2)]]], 2, ZZ) == [[[ZZ(-1)]]]

    assert dmp_sub([[[]]], [[[]]], 2, QQ) == [[[]]]
    assert dmp_sub([[[QQ(1, 2)]]], [[[]]], 2, QQ) == [[[QQ(1, 2)]]]
    assert dmp_sub([[[]]], [[[QQ(1, 2)]]], 2, QQ) == [[[QQ(-1, 2)]]]
    assert dmp_sub([[[QQ(2, 7)]]], [[[QQ(1, 7)]]], 2, QQ) == [[[QQ(1, 7)]]]
    assert dmp_sub([[[QQ(1, 7)]]], [[[QQ(2, 7)]]], 2, QQ) == [[[QQ(-1, 7)]]]


def test_dup_add_mul():
    assert dup_add_mul([ZZ(1), ZZ(2), ZZ(3)], [ZZ(3), ZZ(2), ZZ(1)],
               [ZZ(1), ZZ(2)], ZZ) == [ZZ(3), ZZ(9), ZZ(7), ZZ(5)]
    assert dmp_add_mul([[ZZ(1), ZZ(2)], [ZZ(3)]], [[ZZ(3)], [ZZ(2), ZZ(1)]],
               [[ZZ(1)], [ZZ(2)]], 1, ZZ) == [[ZZ(3)], [ZZ(3), ZZ(9)], [ZZ(4), ZZ(5)]]


def test_dup_sub_mul():
    assert dup_sub_mul([ZZ(1), ZZ(2), ZZ(3)], [ZZ(3), ZZ(2), ZZ(1)],
               [ZZ(1), ZZ(2)], ZZ) == [ZZ(-3), ZZ(-7), ZZ(-3), ZZ(1)]
    assert dmp_sub_mul([[ZZ(1), ZZ(2)], [ZZ(3)]], [[ZZ(3)], [ZZ(2), ZZ(1)]],
               [[ZZ(1)], [ZZ(2)]], 1, ZZ) == [[ZZ(-3)], [ZZ(-1), ZZ(-5)], [ZZ(-4), ZZ(1)]]


def test_dup_mul():
    assert dup_mul([], [], ZZ) == []
    assert dup_mul([], [ZZ(1)], ZZ) == []
    assert dup_mul([ZZ(1)], [], ZZ) == []
    assert dup_mul([ZZ(1)], [ZZ(1)], ZZ) == [ZZ(1)]
    assert dup_mul([ZZ(5)], [ZZ(7)], ZZ) == [ZZ(35)]

    assert dup_mul([], [], QQ) == []
    assert dup_mul([], [QQ(1, 2)], QQ) == []
    assert dup_mul([QQ(1, 2)], [], QQ) == []
    assert dup_mul([QQ(1, 2)], [QQ(4, 7)], QQ) == [QQ(2, 7)]
    assert dup_mul([QQ(5, 7)], [QQ(3, 7)], QQ) == [QQ(15, 49)]

    f = dup_normal([3, 0, 0, 6, 1, 2], ZZ)
    g = dup_normal([4, 0, 1, 0], ZZ)
    h = dup_normal([12, 0, 3, 24, 4, 14, 1, 2, 0], ZZ)

    assert dup_mul(f, g, ZZ) == h
    assert dup_mul(g, f, ZZ) == h

    f = dup_normal([2, 0, 0, 1, 7], ZZ)
    h = dup_normal([4, 0, 0, 4, 28, 0, 1, 14, 49], ZZ)

    assert dup_mul(f, f, ZZ) == h

    K = FF(6)

    assert dup_mul([K(2), K(1)], [K(3), K(4)], K) == [K(5), K(4)]

    p1 = dup_normal([79, -1, 78, -94, -10, 11, 32, -19, 78, 2, -89, 30, 73, 42,
        85, 77, 83, -30, -34, -2, 95, -81, 37, -49, -46, -58, -16, 37, 35, -11,
        -57, -15, -31, 67, -20, 27, 76, 2, 70, 67, -65, 65, -26, -93, -44, -12,
        -92, 57, -90, -57, -11, -67, -98, -69, 97, -41, 89, 33, 89, -50, 81,
        -31, 60, -27, 43, 29, -77, 44, 21, -91, 32, -57, 33, 3, 53, -51, -38,
        -99, -84, 23, -50, 66, -100, 1, -75, -25, 27, -60, 98, -51, -87, 6, 8,
        78, -28, -95, -88, 12, -35, 26, -9, 16, -92, 55, -7, -86, 68, -39, -46,
        84, 94, 45, 60, 92, 68, -75, -74, -19, 8, 75, 78, 91, 57, 34, 14, -3,
        -49, 65, 78, -18, 6, -29, -80, -98, 17, 13, 58, 21, 20, 9, 37, 7, -30,
        -53, -20, 34, 67, -42, 89, -22, 73, 43, -6, 5, 51, -8, -15, -52, -22,
        -58, -72, -3, 43, -92, 82, 83, -2, -13, -23, -60, 16, -94, -8, -28,
        -95, -72, 63, -90, 76, 6, -43, -100, -59, 76, 3, 3, 46, -85, 75, 62,
        -71, -76, 88, 97, -72, -1, 30, -64, 72, -48, 14, -78, 58, 63, -91, 24,
        -87, -27, -80, -100, -44, 98, 70, 100, -29, -38, 11, 77, 100, 52, 86,
        65, -5, -42, -81, -38, -42, 43, -2, -70, -63, -52], ZZ)
    p2 = dup_normal([65, -19, -47, 1, 90, 81, -15, -34, 25, -75, 9, -83, 50, -5,
        -44, 31, 1, 70, -7, 78, 74, 80, 85, 65, 21, 41, 66, 19, -40, 63, -21,
        -27, 32, 69, 83, 34, -35, 14, 81, 57, -75, 32, -67, -89, -100, -61, 46,
        84, -78, -29, -50, -94, -24, -32, -68, -16, 100, -7, -72, -89, 35, 82,
        58, 81, -92, 62, 5, -47, -39, -58, -72, -13, 84, 44, 55, -25, 48, -54,
        -31, -56, -11, -50, -84, 10, 67, 17, 13, -14, 61, 76, -64, -44, -40,
        -96, 11, -11, -94, 2, 6, 27, -6, 68, -54, 66, -74, -14, -1, -24, -73,
        96, 89, -11, -89, 56, -53, 72, -43, 96, 25, 63, -31, 29, 68, 83, 91,
        -93, -19, -38, -40, 40, -12, -19, -79, 44, 100, -66, -29, -77, 62, 39,
        -8, 11, -97, 14, 87, 64, 21, -18, 13, 15, -59, -75, -99, -88, 57, 54,
        56, -67, 6, -63, -59, -14, 28, 87, -20, -39, 84, -91, -2, 49, -75, 11,
        -24, -95, 36, 66, 5, 25, -72, -40, 86, 90, 37, -33, 57, -35, 29, -18,
        4, -79, 64, -17, -27, 21, 29, -5, -44, -87, -24, 52, 78, 11, -23, -53,
        36, 42, 21, -68, 94, -91, -51, -21, 51, -76, 72, 31, 24, -48, -80, -9,
        37, -47, -6, -8, -63, -91, 79, -79, -100, 38, -20, 38, 100, 83, -90,
        87, 63, -36, 82, -19, 18, -98, -38, 26, 98, -70, 79, 92, 12, 12, 70,
        74, 36, 48, -13, 31, 31, -47, -71, -12, -64, 36, -42, 32, -86, 60, 83,
        70, 55, 0, 1, 29, -35, 8, -82, 8, -73, -46, -50, 43, 48, -5, -86, -72,
        44, -90, 19, 19, 5, -20, 97, -13, -66, -5, 5, -69, 64, -30, 41, 51, 36,
        13, -99, -61, 94, -12, 74, 98, 68, 24, 46, -97, -87, -6, -27, 82, 62,
        -11, -77, 86, 66, -47, -49, -50, 13, 18, 89, -89, 46, -80, 13, 98, -35,
        -36, -25, 12, 20, 26, -52, 79, 27, 79, 100, 8, 62, -58, -28, 37], ZZ)
    res = dup_normal([5135, -1566, 1376, -7466, 4579, 11710, 8001, -7183,
        -3737, -7439, 345, -10084, 24522, -1201, 1070, -10245, 9582, 9264,
        1903, 23312, 18953, 10037, -15268, -5450, 6442, -6243, -3777, 5110,
        10936, -16649, -6022, 16255, 31300, 24818, 31922, 32760, 7854, 27080,
        15766, 29596, 7139, 31945, -19810, 465, -38026, -3971, 9641, 465,
        -19375, 5524, -30112, -11960, -12813, 13535, 30670, 5925, -43725,
        -14089, 11503, -22782, 6371, 43881, 37465, -33529, -33590, -39798,
        -37854, -18466, -7908, -35825, -26020, -36923, -11332, -5699, 25166,
        -3147, 19885, 12962, -20659, -1642, 27723, -56331, -24580, -11010,
        -20206, 20087, -23772, -16038, 38580, 20901, -50731, 32037, -4299,
        26508, 18038, -28357, 31846, -7405, -20172, -15894, 2096, 25110,
        -45786, 45918, -55333, -31928, -49428, -29824, -58796, -24609, -15408,
        69, -35415, -18439, 10123, -20360, -65949, 33356, -20333, 26476,
        -32073, 33621, 930, 28803, -42791, 44716, 38164, 12302, -1739, 11421,
        73385, -7613, 14297, 38155, -414, 77587, 24338, -21415, 29367, 42639,
        13901, -288, 51027, -11827, 91260, 43407, 88521, -15186, 70572, -12049,
        5090, -12208, -56374, 15520, -623, -7742, 50825, 11199, -14894, 40892,
        59591, -31356, -28696, -57842, -87751, -33744, -28436, -28945, -40287,
        37957, -35638, 33401, -61534, 14870, 40292, 70366, -10803, 102290,
        -71719, -85251, 7902, -22409, 75009, 99927, 35298, -1175, -762, -34744,
        -10587, -47574, -62629, -19581, -43659, -54369, -32250, -39545, 15225,
        -24454, 11241, -67308, -30148, 39929, 37639, 14383, -73475, -77636,
        -81048, -35992, 41601, -90143, 76937, -8112, 56588, 9124, -40094,
        -32340, 13253, 10898, -51639, 36390, 12086, -1885, 100714, -28561,
        -23784, -18735, 18916, 16286, 10742, -87360, -13697, 10689, -19477,
        -29770, 5060, 20189, -8297, 112407, 47071, 47743, 45519, -4109, 17468,
        -68831, 78325, -6481, -21641, -19459, 30919, 96115, 8607, 53341, 32105,
        -16211, 23538, 57259, -76272, -40583, 62093, 38511, -34255, -40665,
        -40604, -37606, -15274, 33156, -13885, 103636, 118678, -14101, -92682,
        -100791, 2634, 63791, 98266, 19286, -34590, -21067, -71130, 25380,
        -40839, -27614, -26060, 52358, -15537, 27138, -6749, 36269, -33306,
        13207, -91084, -5540, -57116, 69548, 44169, -57742, -41234, -103327,
        -62904, -8566, 41149, -12866, 71188, 23980, 1838, 58230, 73950, 5594,
        43113, -8159, -15925, 6911, 85598, -75016, -16214, -62726, -39016,
        8618, -63882, -4299, 23182, 49959, 49342, -3238, -24913, -37138, 78361,
        32451, 6337, -11438, -36241, -37737, 8169, -3077, -24829, 57953, 53016,
        -31511, -91168, 12599, -41849, 41576, 55275, -62539, 47814, -62319,
        12300, -32076, -55137, -84881, -27546, 4312, -3433, -54382, 113288,
        -30157, 74469, 18219, 79880, -2124, 98911, 17655, -33499, -32861,
        47242, -37393, 99765, 14831, -44483, 10800, -31617, -52710, 37406,
        22105, 29704, -20050, 13778, 43683, 36628, 8494, 60964, -22644, 31550,
        -17693, 33805, -124879, -12302, 19343, 20400, -30937, -21574, -34037,
        -33380, 56539, -24993, -75513, -1527, 53563, 65407, -101, 53577, 37991,
        18717, -23795, -8090, -47987, -94717, 41967, 5170, -14815, -94311,
        17896, -17734, -57718, -774, -38410, 24830, 29682, 76480, 58802,
        -46416, -20348, -61353, -68225, -68306, 23822, -31598, 42972, 36327,
        28968, -65638, -21638, 24354, -8356, 26777, 52982, -11783, -44051,
        -26467, -44721, -28435, -53265, -25574, -2669, 44155, 22946, -18454,
        -30718, -11252, 58420, 8711, 67447, 4425, 41749, 67543, 43162, 11793,
        -41907, 20477, -13080, 6559, -6104, -13244, 42853, 42935, 29793, 36730,
        -28087, 28657, 17946, 7503, 7204, 21491, -27450, -24241, -98156,
        -18082, -42613, -24928, 10775, -14842, -44127, 55910, 14777, 31151, -2194,
        39206, -2100, -4211, 11827, -8918, -19471, 72567, 36447, -65590, -34861,
        -17147, -45303, 9025, -7333, -35473, 11101, 11638, 3441, 6626, -41800,
        9416, 13679, 33508, 40502, -60542, 16358, 8392, -43242, -35864, -34127,
        -48721, 35878, 30598, 28630, 20279, -19983, -14638, -24455, -1851, -11344,
        45150, 42051, 26034, -28889, -32382, -3527, -14532, 22564, -22346, 477,
        11706, 28338, -25972, -9185, -22867, -12522, 32120, -4424, 11339, -33913,
        -7184, 5101, -23552, -17115, -31401, -6104, 21906, 25708, 8406, 6317,
        -7525, 5014, 20750, 20179, 22724, 11692, 13297, 2493, -253, -16841, -17339,
        -6753, -4808, 2976, -10881, -10228, -13816, -12686, 1385, 2316, 2190, -875,
        -1924], ZZ)

    assert dup_mul(p1, p2, ZZ) == res

    p1 = dup_normal([83, -61, -86, -24, 12, 43, -88, -9, 42, 55, -66, 74, 95,
        -25, -12, 68, -99, 4, 45, 6, -15, -19, 78, 65, -55, 47, -13, 17, 86,
        81, -58, -27, 50, -40, -24, 39, -41, -92, 75, 90, -1, 40, -15, -27,
        -35, 68, 70, -64, -40, 78, -88, -58, -39, 69, 46, 12, 28, -94, -37,
        -50, -80, -96, -61, 25, 1, 71, 4, 12, 48, 4, 34, -47, -75, 5, 48, 82,
        88, 23, 98, 35, 17, -10, 48, -61, -95, 47, 65, -19, -66, -57, -6, -51,
        -42, -89, 66, -13, 18, 37, 90, -23, 72, 96, -53, 0, 40, -73, -52, -68,
        32, -25, -53, 79, -52, 18, 44, 73, -81, 31, -90, 70, 3, 36, 48, 76,
        -24, -44, 23, 98, -4, 73, 69, 88, -70, 14, -68, 94, -78, -15, -64, -97,
        -70, -35, 65, 88, 49, -53, -7, 12, -45, -7, 59, -94, 99, -2, 67, -60,
        -71, 29, -62, -77, 1, 51, 17, 80, -20, -47, -19, 24, -9, 39, -23, 21,
        -84, 10, 84, 56, -17, -21, -66, 85, 70, 46, -51, -22, -95, 78, -60,
        -96, -97, -45, 72, 35, 30, -61, -92, -93, -60, -61, 4, -4, -81, -73,
        46, 53, -11, 26, 94, 45, 14, -78, 55, 84, -68, 98, 60, 23, 100, -63,
        68, 96, -16, 3, 56, 21, -58, 62, -67, 66, 85, 41, -79, -22, 97, -67,
        82, 82, -96, -20, -7, 48, -67, 48, -9, -39, 78], ZZ)
    p2 = dup_normal([52, 88, 76, 66, 9, -64, 46, -20, -28, 69, 60, 96, -36,
        -92, -30, -11, -35, 35, 55, 63, -92, -7, 25, -58, 74, 55, -6, 4, 47,
        -92, -65, 67, -45, 74, -76, 59, -6, 69, 39, 24, -71, -7, 39, -45, 60,
        -68, 98, 97, -79, 17, 4, 94, -64, 68, -100, -96, -2, 3, 22, 96, 54,
        -77, -86, 67, 6, 57, 37, 40, 89, -78, 64, -94, -45, -92, 57, 87, -26,
        36, 19, 97, 25, 77, -87, 24, 43, -5, 35, 57, 83, 71, 35, 63, 61, 96,
        -22, 8, -1, 96, 43, 45, 94, -93, 36, 71, -41, -99, 85, -48, 59, 52,
        -17, 5, 87, -16, -68, -54, 76, -18, 100, 91, -42, -70, -66, -88, -12,
        1, 95, -82, 52, 43, -29, 3, 12, 72, -99, -43, -32, -93, -51, 16, -20,
        -12, -11, 5, 33, -38, 93, -5, -74, 25, 74, -58, 93, 59, -63, -86, 63,
        -20, -4, -74, -73, -95, 29, -28, 93, -91, -2, -38, -62, 77, -58, -85,
        -28, 95, 38, 19, -69, 86, 94, 25, -2, -4, 47, 34, -59, 35, -48, 29,
        -63, -53, 34, 29, 66, 73, 6, 92, -84, 89, 15, 81, 93, 97, 51, -72, -78,
        25, 60, 90, -45, 39, 67, -84, -62, 57, 26, -32, -56, -14, -83, 76, 5,
        -2, 99, -100, 28, 46, 94, -7, 53, -25, 16, -23, -36, 89, -78, -63, 31,
        1, 84, -99, -52, 76, 48, 90, -76, 44, -19, 54, -36, -9, -73, -100, -69,
        31, 42, 25, -39, 76, -26, -8, -14, 51, 3, 37, 45, 2, -54, 13, -34, -92,
        17, -25, -65, 53, -63, 30, 4, -70, -67, 90, 52, 51, 18, -3, 31, -45,
        -9, 59, 63, -87, 22, -32, 29, -38, 21, 36, -82, 27, -11], ZZ)
    res = dup_normal([4316, 4132, -3532, -7974, -11303, -10069, 5484, -3330,
        -5874, 7734, 4673, 11327, -9884, -8031, 17343, 21035, -10570, -9285,
        15893, 3780, -14083, 8819, 17592, 10159, 7174, -11587, 8598, -16479,
        3602, 25596, 9781, 12163, 150, 18749, -21782, -12307, 27578, -2757,
        -12573, 12565, 6345, -18956, 19503, -15617, 1443, -16778, 36851, 23588,
        -28474, 5749, 40695, -7521, -53669, -2497, -18530, 6770, 57038, 3926,
        -6927, -15399, 1848, -64649, -27728, 3644, 49608, 15187, -8902, -9480,
        -7398, -40425, 4824, 23767, -7594, -6905, 33089, 18786, 12192, 24670,
        31114, 35334, -4501, -14676, 7107, -59018, -21352, 20777, 19661, 20653,
        33754, -885, -43758, 6269, 51897, -28719, -97488, -9527, 13746, 11644,
        17644, -21720, 23782, -10481, 47867, 20752, 33810, -1875, 39918, -7710,
        -40840, 19808, -47075, 23066, 46616, 25201, 9287, 35436, -1602, 9645,
        -11978, 13273, 15544, 33465, 20063, 44539, 11687, 27314, -6538, -37467,
        14031, 32970, -27086, 41323, 29551, 65910, -39027, -37800, -22232,
        8212, 46316, -28981, -55282, 50417, -44929, -44062, 73879, 37573,
        -2596, -10877, -21893, -133218, -33707, -25753, -9531, 17530, 61126,
        2748, -56235, 43874, -10872, -90459, -30387, 115267, -7264, -44452,
        122626, 14839, -599, 10337, 57166, -67467, -54957, 63669, 1202, 18488,
        52594, 7205, -97822, 612, 78069, -5403, -63562, 47236, 36873, -154827,
        -26188, 82427, -39521, 5628, 7416, 5276, -53095, 47050, 26121, -42207,
        79021, -13035, 2499, -66943, 29040, -72355, -23480, 23416, -12885,
        -44225, -42688, -4224, 19858, 55299, 15735, 11465, 101876, -39169,
        51786, 14723, 43280, -68697, 16410, 92295, 56767, 7183, 111850, 4550,
        115451, -38443, -19642, -35058, 10230, 93829, 8925, 63047, 3146, 29250,
        8530, 5255, -98117, -115517, -76817, -8724, 41044, 1312, -35974, 79333,
        -28567, 7547, -10580, -24559, -16238, 10794, -3867, 24848, 57770,
        -51536, -35040, 71033, 29853, 62029, -7125, -125585, -32169, -47907,
        156811, -65176, -58006, -15757, -57861, 11963, 30225, -41901, -41681,
        31310, 27982, 18613, 61760, 60746, -59096, 33499, 30097, -17997, 24032,
        56442, -83042, 23747, -20931, -21978, -158752, -9883, -73598, -7987,
        -7333, -125403, -116329, 30585, 53281, 51018, -29193, 88575, 8264,
        -40147, -16289, 113088, 12810, -6508, 101552, -13037, 34440, -41840,
        101643, 24263, 80532, 61748, 65574, 6423, -20672, 6591, -10834, -71716,
        86919, -92626, 39161, 28490, 81319, 46676, 106720, 43530, 26998, 57456,
        -8862, 60989, 13982, 3119, -2224, 14743, 55415, -49093, -29303, 28999,
        1789, 55953, -84043, -7780, -65013, 57129, -47251, 61484, 61994,
        -78361, -82778, 22487, -26894, 9756, -74637, -15519, -4360, 30115,
        42433, 35475, 15286, 69768, 21509, -20214, 78675, -21163, 13596, 11443,
        -10698, -53621, -53867, -24155, 64500, -42784, -33077, -16500, 873,
        -52788, 14546, -38011, 36974, -39849, -34029, -94311, 83068, -50437,
        -26169, -46746, 59185, 42259, -101379, -12943, 30089, -59086, 36271,
        22723, -30253, -52472, -70826, -23289, 3331, -31687, 14183, -857,
        -28627, 35246, -51284, 5636, -6933, 66539, 36654, 50927, 24783, 3457,
        33276, 45281, 45650, -4938, -9968, -22590, 47995, 69229, 5214, -58365,
        -17907, -14651, 18668, 18009, 12649, -11851, -13387, 20339, 52472,
        -1087, -21458, -68647, 52295, 15849, 40608, 15323, 25164, -29368,
        10352, -7055, 7159, 21695, -5373, -54849, 101103, -24963, -10511,
        33227, 7659, 41042, -69588, 26718, -20515, 6441, 38135, -63, 24088,
        -35364, -12785, -18709, 47843, 48533, -48575, 17251, -19394, 32878,
        -9010, -9050, 504, -12407, 28076, -3429, 25324, -4210, -26119, 752,
        -29203, 28251, -11324, -32140, -3366, -25135, 18702, -31588, -7047,
        -24267, 49987, -14975, -33169, 37744, -7720, -9035, 16964, -2807, -421,
        14114, -17097, -13662, 40628, -12139, -9427, 5369, 17551, -13232, -16211,
        9804, -7422, 2677, 28635, -8280, -4906, 2908, -22558, 5604, 12459, 8756,
        -3980, -4745, -18525, 7913, 5970, -16457, 20230, -6247, -13812, 2505,
        11899, 1409, -15094, 22540, -18863, 137, 11123, -4516, 2290, -8594, 12150,
        -10380, 3005, 5235, -7350, 2535, -858], ZZ)

    assert dup_mul(p1, p2, ZZ) == res


def test_dmp_mul():
    assert dmp_mul([ZZ(5)], [ZZ(7)], 0, ZZ) == \
        dup_mul([ZZ(5)], [ZZ(7)], ZZ)
    assert dmp_mul([QQ(5, 7)], [QQ(3, 7)], 0, QQ) == \
        dup_mul([QQ(5, 7)], [QQ(3, 7)], QQ)

    assert dmp_mul([[[]]], [[[]]], 2, ZZ) == [[[]]]
    assert dmp_mul([[[ZZ(1)]]], [[[]]], 2, ZZ) == [[[]]]
    assert dmp_mul([[[]]], [[[ZZ(1)]]], 2, ZZ) == [[[]]]
    assert dmp_mul([[[ZZ(2)]]], [[[ZZ(1)]]], 2, ZZ) == [[[ZZ(2)]]]
    assert dmp_mul([[[ZZ(1)]]], [[[ZZ(2)]]], 2, ZZ) == [[[ZZ(2)]]]

    assert dmp_mul([[[]]], [[[]]], 2, QQ) == [[[]]]
    assert dmp_mul([[[QQ(1, 2)]]], [[[]]], 2, QQ) == [[[]]]
    assert dmp_mul([[[]]], [[[QQ(1, 2)]]], 2, QQ) == [[[]]]
    assert dmp_mul([[[QQ(2, 7)]]], [[[QQ(1, 3)]]], 2, QQ) == [[[QQ(2, 21)]]]
    assert dmp_mul([[[QQ(1, 7)]]], [[[QQ(2, 3)]]], 2, QQ) == [[[QQ(2, 21)]]]

    K = FF(6)

    assert dmp_mul(
        [[K(2)], [K(1)]], [[K(3)], [K(4)]], 1, K) == [[K(5)], [K(4)]]


def test_dup_sqr():
    assert dup_sqr([], ZZ) == []
    assert dup_sqr([ZZ(2)], ZZ) == [ZZ(4)]
    assert dup_sqr([ZZ(1), ZZ(2)], ZZ) == [ZZ(1), ZZ(4), ZZ(4)]

    assert dup_sqr([], QQ) == []
    assert dup_sqr([QQ(2, 3)], QQ) == [QQ(4, 9)]
    assert dup_sqr([QQ(1, 3), QQ(2, 3)], QQ) == [QQ(1, 9), QQ(4, 9), QQ(4, 9)]

    f = dup_normal([2, 0, 0, 1, 7], ZZ)

    assert dup_sqr(f, ZZ) == dup_normal([4, 0, 0, 4, 28, 0, 1, 14, 49], ZZ)

    K = FF(9)

    assert dup_sqr([K(3), K(4)], K) == [K(6), K(7)]


def test_dmp_sqr():
    assert dmp_sqr([ZZ(1), ZZ(2)], 0, ZZ) == \
        dup_sqr([ZZ(1), ZZ(2)], ZZ)

    assert dmp_sqr([[[]]], 2, ZZ) == [[[]]]
    assert dmp_sqr([[[ZZ(2)]]], 2, ZZ) == [[[ZZ(4)]]]

    assert dmp_sqr([[[]]], 2, QQ) == [[[]]]
    assert dmp_sqr([[[QQ(2, 3)]]], 2, QQ) == [[[QQ(4, 9)]]]

    K = FF(9)

    assert dmp_sqr([[K(3)], [K(4)]], 1, K) == [[K(6)], [K(7)]]


def test_dup_pow():
    assert dup_pow([], 0, ZZ) == [ZZ(1)]
    assert dup_pow([], 0, QQ) == [QQ(1)]

    assert dup_pow([], 1, ZZ) == []
    assert dup_pow([], 7, ZZ) == []

    assert dup_pow([ZZ(1)], 0, ZZ) == [ZZ(1)]
    assert dup_pow([ZZ(1)], 1, ZZ) == [ZZ(1)]
    assert dup_pow([ZZ(1)], 7, ZZ) == [ZZ(1)]

    assert dup_pow([ZZ(3)], 0, ZZ) == [ZZ(1)]
    assert dup_pow([ZZ(3)], 1, ZZ) == [ZZ(3)]
    assert dup_pow([ZZ(3)], 7, ZZ) == [ZZ(2187)]

    assert dup_pow([QQ(1, 1)], 0, QQ) == [QQ(1, 1)]
    assert dup_pow([QQ(1, 1)], 1, QQ) == [QQ(1, 1)]
    assert dup_pow([QQ(1, 1)], 7, QQ) == [QQ(1, 1)]

    assert dup_pow([QQ(3, 7)], 0, QQ) == [QQ(1, 1)]
    assert dup_pow([QQ(3, 7)], 1, QQ) == [QQ(3, 7)]
    assert dup_pow([QQ(3, 7)], 7, QQ) == [QQ(2187, 823543)]

    f = dup_normal([2, 0, 0, 1, 7], ZZ)

    assert dup_pow(f, 0, ZZ) == dup_normal([1], ZZ)
    assert dup_pow(f, 1, ZZ) == dup_normal([2, 0, 0, 1, 7], ZZ)
    assert dup_pow(f, 2, ZZ) == dup_normal([4, 0, 0, 4, 28, 0, 1, 14, 49], ZZ)
    assert dup_pow(f, 3, ZZ) == dup_normal(
        [8, 0, 0, 12, 84, 0, 6, 84, 294, 1, 21, 147, 343], ZZ)


def test_dmp_pow():
    assert dmp_pow([[]], 0, 1, ZZ) == [[ZZ(1)]]
    assert dmp_pow([[]], 0, 1, QQ) == [[QQ(1)]]

    assert dmp_pow([[]], 1, 1, ZZ) == [[]]
    assert dmp_pow([[]], 7, 1, ZZ) == [[]]

    assert dmp_pow([[ZZ(1)]], 0, 1, ZZ) == [[ZZ(1)]]
    assert dmp_pow([[ZZ(1)]], 1, 1, ZZ) == [[ZZ(1)]]
    assert dmp_pow([[ZZ(1)]], 7, 1, ZZ) == [[ZZ(1)]]

    assert dmp_pow([[QQ(3, 7)]], 0, 1, QQ) == [[QQ(1, 1)]]
    assert dmp_pow([[QQ(3, 7)]], 1, 1, QQ) == [[QQ(3, 7)]]
    assert dmp_pow([[QQ(3, 7)]], 7, 1, QQ) == [[QQ(2187, 823543)]]

    f = dup_normal([2, 0, 0, 1, 7], ZZ)

    assert dmp_pow(f, 2, 0, ZZ) == dup_pow(f, 2, ZZ)


def test_dup_pdiv():
    f = dup_normal([3, 1, 1, 5], ZZ)
    g = dup_normal([5, -3, 1], ZZ)

    q = dup_normal([15, 14], ZZ)
    r = dup_normal([52, 111], ZZ)

    assert dup_pdiv(f, g, ZZ) == (q, r)
    assert dup_pquo(f, g, ZZ) == q
    assert dup_prem(f, g, ZZ) == r

    raises(ExactQuotientFailed, lambda: dup_pexquo(f, g, ZZ))

    f = dup_normal([3, 1, 1, 5], QQ)
    g = dup_normal([5, -3, 1], QQ)

    q = dup_normal([15, 14], QQ)
    r = dup_normal([52, 111], QQ)

    assert dup_pdiv(f, g, QQ) == (q, r)
    assert dup_pquo(f, g, QQ) == q
    assert dup_prem(f, g, QQ) == r

    raises(ExactQuotientFailed, lambda: dup_pexquo(f, g, QQ))


def test_dmp_pdiv():
    f = dmp_normal([[1], [], [1, 0, 0]], 1, ZZ)
    g = dmp_normal([[1], [-1, 0]], 1, ZZ)

    q = dmp_normal([[1], [1, 0]], 1, ZZ)
    r = dmp_normal([[2, 0, 0]], 1, ZZ)

    assert dmp_pdiv(f, g, 1, ZZ) == (q, r)
    assert dmp_pquo(f, g, 1, ZZ) == q
    assert dmp_prem(f, g, 1, ZZ) == r

    raises(ExactQuotientFailed, lambda: dmp_pexquo(f, g, 1, ZZ))

    f = dmp_normal([[1], [], [1, 0, 0]], 1, ZZ)
    g = dmp_normal([[2], [-2, 0]], 1, ZZ)

    q = dmp_normal([[2], [2, 0]], 1, ZZ)
    r = dmp_normal([[8, 0, 0]], 1, ZZ)

    assert dmp_pdiv(f, g, 1, ZZ) == (q, r)
    assert dmp_pquo(f, g, 1, ZZ) == q
    assert dmp_prem(f, g, 1, ZZ) == r

    raises(ExactQuotientFailed, lambda: dmp_pexquo(f, g, 1, ZZ))


def test_dup_rr_div():
    raises(ZeroDivisionError, lambda: dup_rr_div([1, 2, 3], [], ZZ))

    f = dup_normal([3, 1, 1, 5], ZZ)
    g = dup_normal([5, -3, 1], ZZ)

    q, r = [], f

    assert dup_rr_div(f, g, ZZ) == (q, r)


def test_dmp_rr_div():
    raises(ZeroDivisionError, lambda: dmp_rr_div([[1, 2], [3]], [[]], 1, ZZ))

    f = dmp_normal([[1], [], [1, 0, 0]], 1, ZZ)
    g = dmp_normal([[1], [-1, 0]], 1, ZZ)

    q = dmp_normal([[1], [1, 0]], 1, ZZ)
    r = dmp_normal([[2, 0, 0]], 1, ZZ)

    assert dmp_rr_div(f, g, 1, ZZ) == (q, r)

    f = dmp_normal([[1], [], [1, 0, 0]], 1, ZZ)
    g = dmp_normal([[-1], [1, 0]], 1, ZZ)

    q = dmp_normal([[-1], [-1, 0]], 1, ZZ)
    r = dmp_normal([[2, 0, 0]], 1, ZZ)

    assert dmp_rr_div(f, g, 1, ZZ) == (q, r)

    f = dmp_normal([[1], [], [1, 0, 0]], 1, ZZ)
    g = dmp_normal([[2], [-2, 0]], 1, ZZ)

    q, r = [[]], f

    assert dmp_rr_div(f, g, 1, ZZ) == (q, r)


def test_dup_ff_div():
    raises(ZeroDivisionError, lambda: dup_ff_div([1, 2, 3], [], QQ))

    f = dup_normal([3, 1, 1, 5], QQ)
    g = dup_normal([5, -3, 1], QQ)

    q = [QQ(3, 5), QQ(14, 25)]
    r = [QQ(52, 25), QQ(111, 25)]

    assert dup_ff_div(f, g, QQ) == (q, r)

def test_dup_ff_div_gmpy2():
    if GROUND_TYPES != 'gmpy2':
        return

    from gmpy2 import mpq
    from sympy.polys.domains import GMPYRationalField
    K = GMPYRationalField()

    f = [mpq(1,3), mpq(3,2)]
    g = [mpq(2,1)]
    assert dmp_ff_div(f, g, 0, K) == ([mpq(1,6), mpq(3,4)], [])

    f = [mpq(1,2), mpq(1,3), mpq(1,4), mpq(1,5)]
    g = [mpq(-1,1), mpq(1,1), mpq(-1,1)]
    assert dmp_ff_div(f, g, 0, K) == ([mpq(-1,2), mpq(-5,6)], [mpq(7,12), mpq(-19,30)])

def test_dmp_ff_div():
    raises(ZeroDivisionError, lambda: dmp_ff_div([[1, 2], [3]], [[]], 1, QQ))

    f = dmp_normal([[1], [], [1, 0, 0]], 1, QQ)
    g = dmp_normal([[1], [-1, 0]], 1, QQ)

    q = [[QQ(1, 1)], [QQ(1, 1), QQ(0, 1)]]
    r = [[QQ(2, 1), QQ(0, 1), QQ(0, 1)]]

    assert dmp_ff_div(f, g, 1, QQ) == (q, r)

    f = dmp_normal([[1], [], [1, 0, 0]], 1, QQ)
    g = dmp_normal([[-1], [1, 0]], 1, QQ)

    q = [[QQ(-1, 1)], [QQ(-1, 1), QQ(0, 1)]]
    r = [[QQ(2, 1), QQ(0, 1), QQ(0, 1)]]

    assert dmp_ff_div(f, g, 1, QQ) == (q, r)

    f = dmp_normal([[1], [], [1, 0, 0]], 1, QQ)
    g = dmp_normal([[2], [-2, 0]], 1, QQ)

    q = [[QQ(1, 2)], [QQ(1, 2), QQ(0, 1)]]
    r = [[QQ(2, 1), QQ(0, 1), QQ(0, 1)]]

    assert dmp_ff_div(f, g, 1, QQ) == (q, r)


def test_dup_div():
    f, g, q, r = [5, 4, 3, 2, 1], [1, 2, 3], [5, -6, 0], [20, 1]

    assert dup_div(f, g, ZZ) == (q, r)
    assert dup_quo(f, g, ZZ) == q
    assert dup_rem(f, g, ZZ) == r

    raises(ExactQuotientFailed, lambda: dup_exquo(f, g, ZZ))

    f, g, q, r = [5, 4, 3, 2, 1, 0], [1, 2, 0, 0, 9], [5, -6], [15, 2, -44, 54]

    assert dup_div(f, g, ZZ) == (q, r)
    assert dup_quo(f, g, ZZ) == q
    assert dup_rem(f, g, ZZ) == r

    raises(ExactQuotientFailed, lambda: dup_exquo(f, g, ZZ))


def test_dmp_div():
    f, g, q, r = [5, 4, 3, 2, 1], [1, 2, 3], [5, -6, 0], [20, 1]

    assert dmp_div(f, g, 0, ZZ) == (q, r)
    assert dmp_quo(f, g, 0, ZZ) == q
    assert dmp_rem(f, g, 0, ZZ) == r

    raises(ExactQuotientFailed, lambda: dmp_exquo(f, g, 0, ZZ))

    f, g, q, r = [[[1]]], [[[2]], [1]], [[[]]], [[[1]]]

    assert dmp_div(f, g, 2, ZZ) == (q, r)
    assert dmp_quo(f, g, 2, ZZ) == q
    assert dmp_rem(f, g, 2, ZZ) == r

    raises(ExactQuotientFailed, lambda: dmp_exquo(f, g, 2, ZZ))


def test_dup_max_norm():
    assert dup_max_norm([], ZZ) == 0
    assert dup_max_norm([1], ZZ) == 1

    assert dup_max_norm([1, 4, 2, 3], ZZ) == 4


def test_dmp_max_norm():
    assert dmp_max_norm([[[]]], 2, ZZ) == 0
    assert dmp_max_norm([[[1]]], 2, ZZ) == 1

    assert dmp_max_norm(f_0, 2, ZZ) == 6


def test_dup_l1_norm():
    assert dup_l1_norm([], ZZ) == 0
    assert dup_l1_norm([1], ZZ) == 1
    assert dup_l1_norm([1, 4, 2, 3], ZZ) == 10


def test_dmp_l1_norm():
    assert dmp_l1_norm([[[]]], 2, ZZ) == 0
    assert dmp_l1_norm([[[1]]], 2, ZZ) == 1

    assert dmp_l1_norm(f_0, 2, ZZ) == 31


def test_dup_l2_norm_squared():
    assert dup_l2_norm_squared([], ZZ) == 0
    assert dup_l2_norm_squared([1], ZZ) == 1
    assert dup_l2_norm_squared([1, 4, 2, 3], ZZ) == 30


def test_dmp_l2_norm_squared():
    assert dmp_l2_norm_squared([[[]]], 2, ZZ) == 0
    assert dmp_l2_norm_squared([[[1]]], 2, ZZ) == 1
    assert dmp_l2_norm_squared(f_0, 2, ZZ) == 111


def test_dup_expand():
    assert dup_expand((), ZZ) == [1]
    assert dup_expand(([1, 2, 3], [1, 2], [7, 5, 4, 3]), ZZ) == \
        dup_mul([1, 2, 3], dup_mul([1, 2], [7, 5, 4, 3], ZZ), ZZ)


def test_dmp_expand():
    assert dmp_expand((), 1, ZZ) == [[1]]
    assert dmp_expand(([[1], [2], [3]], [[1], [2]], [[7], [5], [4], [3]]), 1, ZZ) == \
        dmp_mul([[1], [2], [3]], dmp_mul([[1], [2]], [[7], [5], [
                4], [3]], 1, ZZ), 1, ZZ)
