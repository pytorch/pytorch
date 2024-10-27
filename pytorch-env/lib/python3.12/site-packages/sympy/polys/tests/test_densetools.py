"""Tests for dense recursive polynomials' tools. """

from sympy.polys.densebasic import (
    dup_normal, dmp_normal,
    dup_from_raw_dict,
    dmp_convert, dmp_swap,
)

from sympy.polys.densearith import dmp_mul_ground

from sympy.polys.densetools import (
    dup_clear_denoms, dmp_clear_denoms,
    dup_integrate, dmp_integrate, dmp_integrate_in,
    dup_diff, dmp_diff, dmp_diff_in,
    dup_eval, dmp_eval, dmp_eval_in,
    dmp_eval_tail, dmp_diff_eval_in,
    dup_trunc, dmp_trunc, dmp_ground_trunc,
    dup_monic, dmp_ground_monic,
    dup_content, dmp_ground_content,
    dup_primitive, dmp_ground_primitive,
    dup_extract, dmp_ground_extract,
    dup_real_imag,
    dup_mirror, dup_scale, dup_shift, dmp_shift,
    dup_transform,
    dup_compose, dmp_compose,
    dup_decompose,
    dmp_lift,
    dup_sign_variations,
    dup_revert, dmp_revert,
)
from sympy.polys.polyclasses import ANP

from sympy.polys.polyerrors import (
    MultivariatePolynomialError,
    ExactQuotientFailed,
    NotReversible,
    DomainError,
)

from sympy.polys.specialpolys import f_polys

from sympy.polys.domains import FF, ZZ, QQ, ZZ_I, QQ_I, EX, RR
from sympy.polys.rings import ring

from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.functions.elementary.trigonometric import sin

from sympy.abc import x
from sympy.testing.pytest import raises

f_0, f_1, f_2, f_3, f_4, f_5, f_6 = [ f.to_dense() for f in f_polys() ]

def test_dup_integrate():
    assert dup_integrate([], 1, QQ) == []
    assert dup_integrate([], 2, QQ) == []

    assert dup_integrate([QQ(1)], 1, QQ) == [QQ(1), QQ(0)]
    assert dup_integrate([QQ(1)], 2, QQ) == [QQ(1, 2), QQ(0), QQ(0)]

    assert dup_integrate([QQ(1), QQ(2), QQ(3)], 0, QQ) == \
        [QQ(1), QQ(2), QQ(3)]
    assert dup_integrate([QQ(1), QQ(2), QQ(3)], 1, QQ) == \
        [QQ(1, 3), QQ(1), QQ(3), QQ(0)]
    assert dup_integrate([QQ(1), QQ(2), QQ(3)], 2, QQ) == \
        [QQ(1, 12), QQ(1, 3), QQ(3, 2), QQ(0), QQ(0)]
    assert dup_integrate([QQ(1), QQ(2), QQ(3)], 3, QQ) == \
        [QQ(1, 60), QQ(1, 12), QQ(1, 2), QQ(0), QQ(0), QQ(0)]

    assert dup_integrate(dup_from_raw_dict({29: QQ(17)}, QQ), 3, QQ) == \
        dup_from_raw_dict({32: QQ(17, 29760)}, QQ)

    assert dup_integrate(dup_from_raw_dict({29: QQ(17), 5: QQ(1, 2)}, QQ), 3, QQ) == \
        dup_from_raw_dict({32: QQ(17, 29760), 8: QQ(1, 672)}, QQ)


def test_dmp_integrate():
    assert dmp_integrate([QQ(1)], 2, 0, QQ) == [QQ(1, 2), QQ(0), QQ(0)]

    assert dmp_integrate([[[]]], 1, 2, QQ) == [[[]]]
    assert dmp_integrate([[[]]], 2, 2, QQ) == [[[]]]

    assert dmp_integrate([[[QQ(1)]]], 1, 2, QQ) == [[[QQ(1)]], [[]]]
    assert dmp_integrate([[[QQ(1)]]], 2, 2, QQ) == [[[QQ(1, 2)]], [[]], [[]]]

    assert dmp_integrate([[QQ(1)], [QQ(2)], [QQ(3)]], 0, 1, QQ) == \
        [[QQ(1)], [QQ(2)], [QQ(3)]]
    assert dmp_integrate([[QQ(1)], [QQ(2)], [QQ(3)]], 1, 1, QQ) == \
        [[QQ(1, 3)], [QQ(1)], [QQ(3)], []]
    assert dmp_integrate([[QQ(1)], [QQ(2)], [QQ(3)]], 2, 1, QQ) == \
        [[QQ(1, 12)], [QQ(1, 3)], [QQ(3, 2)], [], []]
    assert dmp_integrate([[QQ(1)], [QQ(2)], [QQ(3)]], 3, 1, QQ) == \
        [[QQ(1, 60)], [QQ(1, 12)], [QQ(1, 2)], [], [], []]


def test_dmp_integrate_in():
    f = dmp_convert(f_6, 3, ZZ, QQ)

    assert dmp_integrate_in(f, 2, 1, 3, QQ) == \
        dmp_swap(
            dmp_integrate(dmp_swap(f, 0, 1, 3, QQ), 2, 3, QQ), 0, 1, 3, QQ)
    assert dmp_integrate_in(f, 3, 1, 3, QQ) == \
        dmp_swap(
            dmp_integrate(dmp_swap(f, 0, 1, 3, QQ), 3, 3, QQ), 0, 1, 3, QQ)
    assert dmp_integrate_in(f, 2, 2, 3, QQ) == \
        dmp_swap(
            dmp_integrate(dmp_swap(f, 0, 2, 3, QQ), 2, 3, QQ), 0, 2, 3, QQ)
    assert dmp_integrate_in(f, 3, 2, 3, QQ) == \
        dmp_swap(
            dmp_integrate(dmp_swap(f, 0, 2, 3, QQ), 3, 3, QQ), 0, 2, 3, QQ)

    raises(IndexError, lambda: dmp_integrate_in(f, 1, -1, 3, QQ))
    raises(IndexError, lambda: dmp_integrate_in(f, 1, 4, 3, QQ))


def test_dup_diff():
    assert dup_diff([], 1, ZZ) == []
    assert dup_diff([7], 1, ZZ) == []
    assert dup_diff([2, 7], 1, ZZ) == [2]
    assert dup_diff([1, 2, 1], 1, ZZ) == [2, 2]
    assert dup_diff([1, 2, 3, 4], 1, ZZ) == [3, 4, 3]
    assert dup_diff([1, -1, 0, 0, 2], 1, ZZ) == [4, -3, 0, 0]

    f = dup_normal([17, 34, 56, -345, 23, 76, 0, 0, 12, 3, 7], ZZ)

    assert dup_diff(f, 0, ZZ) == f
    assert dup_diff(f, 1, ZZ) == [170, 306, 448, -2415, 138, 380, 0, 0, 24, 3]
    assert dup_diff(f, 2, ZZ) == dup_diff(dup_diff(f, 1, ZZ), 1, ZZ)
    assert dup_diff(
        f, 3, ZZ) == dup_diff(dup_diff(dup_diff(f, 1, ZZ), 1, ZZ), 1, ZZ)

    K = FF(3)
    f = dup_normal([17, 34, 56, -345, 23, 76, 0, 0, 12, 3, 7], K)

    assert dup_diff(f, 1, K) == dup_normal([2, 0, 1, 0, 0, 2, 0, 0, 0, 0], K)
    assert dup_diff(f, 2, K) == dup_normal([1, 0, 0, 2, 0, 0, 0], K)
    assert dup_diff(f, 3, K) == dup_normal([], K)

    assert dup_diff(f, 0, K) == f
    assert dup_diff(f, 2, K) == dup_diff(dup_diff(f, 1, K), 1, K)
    assert dup_diff(
        f, 3, K) == dup_diff(dup_diff(dup_diff(f, 1, K), 1, K), 1, K)


def test_dmp_diff():
    assert dmp_diff([], 1, 0, ZZ) == []
    assert dmp_diff([[]], 1, 1, ZZ) == [[]]
    assert dmp_diff([[[]]], 1, 2, ZZ) == [[[]]]

    assert dmp_diff([[[1], [2]]], 1, 2, ZZ) == [[[]]]

    assert dmp_diff([[[1]], [[]]], 1, 2, ZZ) == [[[1]]]
    assert dmp_diff([[[3]], [[1]], [[]]], 1, 2, ZZ) == [[[6]], [[1]]]

    assert dmp_diff([1, -1, 0, 0, 2], 1, 0, ZZ) == \
        dup_diff([1, -1, 0, 0, 2], 1, ZZ)

    assert dmp_diff(f_6, 0, 3, ZZ) == f_6
    assert dmp_diff(f_6, 1, 3, ZZ) == [[[[8460]], [[]]],
                       [[[135, 0, 0], [], [], [-135, 0, 0]]],
                       [[[]]],
                       [[[-423]], [[-47]], [[]], [[141], [], [94, 0], []], [[]]]]
    assert dmp_diff(
        f_6, 2, 3, ZZ) == dmp_diff(dmp_diff(f_6, 1, 3, ZZ), 1, 3, ZZ)
    assert dmp_diff(f_6, 3, 3, ZZ) == dmp_diff(
        dmp_diff(dmp_diff(f_6, 1, 3, ZZ), 1, 3, ZZ), 1, 3, ZZ)

    K = FF(23)
    F_6 = dmp_normal(f_6, 3, K)

    assert dmp_diff(F_6, 0, 3, K) == F_6
    assert dmp_diff(F_6, 1, 3, K) == dmp_diff(F_6, 1, 3, K)
    assert dmp_diff(F_6, 2, 3, K) == dmp_diff(dmp_diff(F_6, 1, 3, K), 1, 3, K)
    assert dmp_diff(F_6, 3, 3, K) == dmp_diff(
        dmp_diff(dmp_diff(F_6, 1, 3, K), 1, 3, K), 1, 3, K)


def test_dmp_diff_in():
    assert dmp_diff_in(f_6, 2, 1, 3, ZZ) == \
        dmp_swap(dmp_diff(dmp_swap(f_6, 0, 1, 3, ZZ), 2, 3, ZZ), 0, 1, 3, ZZ)
    assert dmp_diff_in(f_6, 3, 1, 3, ZZ) == \
        dmp_swap(dmp_diff(dmp_swap(f_6, 0, 1, 3, ZZ), 3, 3, ZZ), 0, 1, 3, ZZ)
    assert dmp_diff_in(f_6, 2, 2, 3, ZZ) == \
        dmp_swap(dmp_diff(dmp_swap(f_6, 0, 2, 3, ZZ), 2, 3, ZZ), 0, 2, 3, ZZ)
    assert dmp_diff_in(f_6, 3, 2, 3, ZZ) == \
        dmp_swap(dmp_diff(dmp_swap(f_6, 0, 2, 3, ZZ), 3, 3, ZZ), 0, 2, 3, ZZ)

    raises(IndexError, lambda: dmp_diff_in(f_6, 1, -1, 3, ZZ))
    raises(IndexError, lambda: dmp_diff_in(f_6, 1, 4, 3, ZZ))

def test_dup_eval():
    assert dup_eval([], 7, ZZ) == 0
    assert dup_eval([1, 2], 0, ZZ) == 2
    assert dup_eval([1, 2, 3], 7, ZZ) == 66


def test_dmp_eval():
    assert dmp_eval([], 3, 0, ZZ) == 0

    assert dmp_eval([[]], 3, 1, ZZ) == []
    assert dmp_eval([[[]]], 3, 2, ZZ) == [[]]

    assert dmp_eval([[1, 2]], 0, 1, ZZ) == [1, 2]

    assert dmp_eval([[[1]]], 3, 2, ZZ) == [[1]]
    assert dmp_eval([[[1, 2]]], 3, 2, ZZ) == [[1, 2]]

    assert dmp_eval([[3, 2], [1, 2]], 3, 1, ZZ) == [10, 8]
    assert dmp_eval([[[3, 2]], [[1, 2]]], 3, 2, ZZ) == [[10, 8]]


def test_dmp_eval_in():
    assert dmp_eval_in(
        f_6, -2, 1, 3, ZZ) == dmp_eval(dmp_swap(f_6, 0, 1, 3, ZZ), -2, 3, ZZ)
    assert dmp_eval_in(
        f_6, 7, 1, 3, ZZ) == dmp_eval(dmp_swap(f_6, 0, 1, 3, ZZ), 7, 3, ZZ)
    assert dmp_eval_in(f_6, -2, 2, 3, ZZ) == dmp_swap(
        dmp_eval(dmp_swap(f_6, 0, 2, 3, ZZ), -2, 3, ZZ), 0, 1, 2, ZZ)
    assert dmp_eval_in(f_6, 7, 2, 3, ZZ) == dmp_swap(
        dmp_eval(dmp_swap(f_6, 0, 2, 3, ZZ), 7, 3, ZZ), 0, 1, 2, ZZ)

    f = [[[int(45)]], [[]], [[]], [[int(-9)], [-1], [], [int(3), int(0), int(10), int(0)]]]

    assert dmp_eval_in(f, -2, 2, 2, ZZ) == \
        [[45], [], [], [-9, -1, 0, -44]]

    raises(IndexError, lambda: dmp_eval_in(f_6, ZZ(1), -1, 3, ZZ))
    raises(IndexError, lambda: dmp_eval_in(f_6, ZZ(1), 4, 3, ZZ))


def test_dmp_eval_tail():
    assert dmp_eval_tail([[]], [1], 1, ZZ) == []
    assert dmp_eval_tail([[[]]], [1], 2, ZZ) == [[]]
    assert dmp_eval_tail([[[]]], [1, 2], 2, ZZ) == []

    assert dmp_eval_tail(f_0, [], 2, ZZ) == f_0

    assert dmp_eval_tail(f_0, [1, -17, 8], 2, ZZ) == 84496
    assert dmp_eval_tail(f_0, [-17, 8], 2, ZZ) == [-1409, 3, 85902]
    assert dmp_eval_tail(f_0, [8], 2, ZZ) == [[83, 2], [3], [302, 81, 1]]

    assert dmp_eval_tail(f_1, [-17, 8], 2, ZZ) == [-136, 15699, 9166, -27144]

    assert dmp_eval_tail(
        f_2, [-12, 3], 2, ZZ) == [-1377, 0, -702, -1224, 0, -624]
    assert dmp_eval_tail(
        f_3, [-12, 3], 2, ZZ) == [144, 82, -5181, -28872, -14868, -540]

    assert dmp_eval_tail(
        f_4, [25, -1], 2, ZZ) == [152587890625, 9765625, -59605407714843750,
        -3839159765625, -1562475, 9536712644531250, 610349546750, -4, 24414375000, 1562520]
    assert dmp_eval_tail(f_5, [25, -1], 2, ZZ) == [-1, -78, -2028, -17576]

    assert dmp_eval_tail(f_6, [0, 2, 4], 3, ZZ) == [5040, 0, 0, 4480]


def test_dmp_diff_eval_in():
    assert dmp_diff_eval_in(f_6, 2, 7, 1, 3, ZZ) == \
        dmp_eval(dmp_diff(dmp_swap(f_6, 0, 1, 3, ZZ), 2, 3, ZZ), 7, 3, ZZ)

    assert dmp_diff_eval_in(f_6, 2, 7, 0, 3, ZZ) == \
        dmp_eval(dmp_diff(f_6, 2, 3, ZZ), 7, 3, ZZ)

    raises(IndexError, lambda: dmp_diff_eval_in(f_6, 1, ZZ(1), 4, 3, ZZ))


def test_dup_revert():
    f = [-QQ(1, 720), QQ(0), QQ(1, 24), QQ(0), -QQ(1, 2), QQ(0), QQ(1)]
    g = [QQ(61, 720), QQ(0), QQ(5, 24), QQ(0), QQ(1, 2), QQ(0), QQ(1)]

    assert dup_revert(f, 8, QQ) == g

    raises(NotReversible, lambda: dup_revert([QQ(1), QQ(0)], 3, QQ))


def test_dmp_revert():
    f = [-QQ(1, 720), QQ(0), QQ(1, 24), QQ(0), -QQ(1, 2), QQ(0), QQ(1)]
    g = [QQ(61, 720), QQ(0), QQ(5, 24), QQ(0), QQ(1, 2), QQ(0), QQ(1)]

    assert dmp_revert(f, 8, 0, QQ) == g

    raises(MultivariatePolynomialError, lambda: dmp_revert([[1]], 2, 1, QQ))


def test_dup_trunc():
    assert dup_trunc([1, 2, 3, 4, 5, 6], ZZ(3), ZZ) == [1, -1, 0, 1, -1, 0]
    assert dup_trunc([6, 5, 4, 3, 2, 1], ZZ(3), ZZ) == [-1, 1, 0, -1, 1]

    R = ZZ_I
    assert dup_trunc([R(3), R(4), R(5)], R(3), R) == [R(1), R(-1)]

    K = FF(5)
    assert dup_trunc([K(3), K(4), K(5)], K(3), K) == [K(1), K(0)]


def test_dmp_trunc():
    assert dmp_trunc([[]], [1, 2], 2, ZZ) == [[]]
    assert dmp_trunc([[1, 2], [1, 4, 1], [1]], [1, 2], 1, ZZ) == [[-3], [1]]


def test_dmp_ground_trunc():
    assert dmp_ground_trunc(f_0, ZZ(3), 2, ZZ) == \
        dmp_normal(
            [[[1, -1, 0], [-1]], [[]], [[1, -1, 0], [1, -1, 1], [1]]], 2, ZZ)


def test_dup_monic():
    assert dup_monic([3, 6, 9], ZZ) == [1, 2, 3]

    raises(ExactQuotientFailed, lambda: dup_monic([3, 4, 5], ZZ))

    assert dup_monic([], QQ) == []
    assert dup_monic([QQ(1)], QQ) == [QQ(1)]
    assert dup_monic([QQ(7), QQ(1), QQ(21)], QQ) == [QQ(1), QQ(1, 7), QQ(3)]


def test_dmp_ground_monic():
    assert dmp_ground_monic([3, 6, 9], 0, ZZ) == [1, 2, 3]

    assert dmp_ground_monic([[3], [6], [9]], 1, ZZ) == [[1], [2], [3]]

    raises(
        ExactQuotientFailed, lambda: dmp_ground_monic([[3], [4], [5]], 1, ZZ))

    assert dmp_ground_monic([[]], 1, QQ) == [[]]
    assert dmp_ground_monic([[QQ(1)]], 1, QQ) == [[QQ(1)]]
    assert dmp_ground_monic(
        [[QQ(7)], [QQ(1)], [QQ(21)]], 1, QQ) == [[QQ(1)], [QQ(1, 7)], [QQ(3)]]


def test_dup_content():
    assert dup_content([], ZZ) == ZZ(0)
    assert dup_content([1], ZZ) == ZZ(1)
    assert dup_content([-1], ZZ) == ZZ(1)
    assert dup_content([1, 1], ZZ) == ZZ(1)
    assert dup_content([2, 2], ZZ) == ZZ(2)
    assert dup_content([1, 2, 1], ZZ) == ZZ(1)
    assert dup_content([2, 4, 2], ZZ) == ZZ(2)

    assert dup_content([QQ(2, 3), QQ(4, 9)], QQ) == QQ(2, 9)
    assert dup_content([QQ(2, 3), QQ(4, 5)], QQ) == QQ(2, 15)


def test_dmp_ground_content():
    assert dmp_ground_content([[]], 1, ZZ) == ZZ(0)
    assert dmp_ground_content([[]], 1, QQ) == QQ(0)
    assert dmp_ground_content([[1]], 1, ZZ) == ZZ(1)
    assert dmp_ground_content([[-1]], 1, ZZ) == ZZ(1)
    assert dmp_ground_content([[1], [1]], 1, ZZ) == ZZ(1)
    assert dmp_ground_content([[2], [2]], 1, ZZ) == ZZ(2)
    assert dmp_ground_content([[1], [2], [1]], 1, ZZ) == ZZ(1)
    assert dmp_ground_content([[2], [4], [2]], 1, ZZ) == ZZ(2)

    assert dmp_ground_content([[QQ(2, 3)], [QQ(4, 9)]], 1, QQ) == QQ(2, 9)
    assert dmp_ground_content([[QQ(2, 3)], [QQ(4, 5)]], 1, QQ) == QQ(2, 15)

    assert dmp_ground_content(f_0, 2, ZZ) == ZZ(1)
    assert dmp_ground_content(
        dmp_mul_ground(f_0, ZZ(2), 2, ZZ), 2, ZZ) == ZZ(2)

    assert dmp_ground_content(f_1, 2, ZZ) == ZZ(1)
    assert dmp_ground_content(
        dmp_mul_ground(f_1, ZZ(3), 2, ZZ), 2, ZZ) == ZZ(3)

    assert dmp_ground_content(f_2, 2, ZZ) == ZZ(1)
    assert dmp_ground_content(
        dmp_mul_ground(f_2, ZZ(4), 2, ZZ), 2, ZZ) == ZZ(4)

    assert dmp_ground_content(f_3, 2, ZZ) == ZZ(1)
    assert dmp_ground_content(
        dmp_mul_ground(f_3, ZZ(5), 2, ZZ), 2, ZZ) == ZZ(5)

    assert dmp_ground_content(f_4, 2, ZZ) == ZZ(1)
    assert dmp_ground_content(
        dmp_mul_ground(f_4, ZZ(6), 2, ZZ), 2, ZZ) == ZZ(6)

    assert dmp_ground_content(f_5, 2, ZZ) == ZZ(1)
    assert dmp_ground_content(
        dmp_mul_ground(f_5, ZZ(7), 2, ZZ), 2, ZZ) == ZZ(7)

    assert dmp_ground_content(f_6, 3, ZZ) == ZZ(1)
    assert dmp_ground_content(
        dmp_mul_ground(f_6, ZZ(8), 3, ZZ), 3, ZZ) == ZZ(8)


def test_dup_primitive():
    assert dup_primitive([], ZZ) == (ZZ(0), [])
    assert dup_primitive([ZZ(1)], ZZ) == (ZZ(1), [ZZ(1)])
    assert dup_primitive([ZZ(1), ZZ(1)], ZZ) == (ZZ(1), [ZZ(1), ZZ(1)])
    assert dup_primitive([ZZ(2), ZZ(2)], ZZ) == (ZZ(2), [ZZ(1), ZZ(1)])
    assert dup_primitive(
        [ZZ(1), ZZ(2), ZZ(1)], ZZ) == (ZZ(1), [ZZ(1), ZZ(2), ZZ(1)])
    assert dup_primitive(
        [ZZ(2), ZZ(4), ZZ(2)], ZZ) == (ZZ(2), [ZZ(1), ZZ(2), ZZ(1)])

    assert dup_primitive([], QQ) == (QQ(0), [])
    assert dup_primitive([QQ(1)], QQ) == (QQ(1), [QQ(1)])
    assert dup_primitive([QQ(1), QQ(1)], QQ) == (QQ(1), [QQ(1), QQ(1)])
    assert dup_primitive([QQ(2), QQ(2)], QQ) == (QQ(2), [QQ(1), QQ(1)])
    assert dup_primitive(
        [QQ(1), QQ(2), QQ(1)], QQ) == (QQ(1), [QQ(1), QQ(2), QQ(1)])
    assert dup_primitive(
        [QQ(2), QQ(4), QQ(2)], QQ) == (QQ(2), [QQ(1), QQ(2), QQ(1)])

    assert dup_primitive(
        [QQ(2, 3), QQ(4, 9)], QQ) == (QQ(2, 9), [QQ(3), QQ(2)])
    assert dup_primitive(
        [QQ(2, 3), QQ(4, 5)], QQ) == (QQ(2, 15), [QQ(5), QQ(6)])


def test_dmp_ground_primitive():
    assert dmp_ground_primitive([ZZ(1)], 0, ZZ) == (ZZ(1), [ZZ(1)])

    assert dmp_ground_primitive([[]], 1, ZZ) == (ZZ(0), [[]])

    assert dmp_ground_primitive(f_0, 2, ZZ) == (ZZ(1), f_0)
    assert dmp_ground_primitive(
        dmp_mul_ground(f_0, ZZ(2), 2, ZZ), 2, ZZ) == (ZZ(2), f_0)

    assert dmp_ground_primitive(f_1, 2, ZZ) == (ZZ(1), f_1)
    assert dmp_ground_primitive(
        dmp_mul_ground(f_1, ZZ(3), 2, ZZ), 2, ZZ) == (ZZ(3), f_1)

    assert dmp_ground_primitive(f_2, 2, ZZ) == (ZZ(1), f_2)
    assert dmp_ground_primitive(
        dmp_mul_ground(f_2, ZZ(4), 2, ZZ), 2, ZZ) == (ZZ(4), f_2)

    assert dmp_ground_primitive(f_3, 2, ZZ) == (ZZ(1), f_3)
    assert dmp_ground_primitive(
        dmp_mul_ground(f_3, ZZ(5), 2, ZZ), 2, ZZ) == (ZZ(5), f_3)

    assert dmp_ground_primitive(f_4, 2, ZZ) == (ZZ(1), f_4)
    assert dmp_ground_primitive(
        dmp_mul_ground(f_4, ZZ(6), 2, ZZ), 2, ZZ) == (ZZ(6), f_4)

    assert dmp_ground_primitive(f_5, 2, ZZ) == (ZZ(1), f_5)
    assert dmp_ground_primitive(
        dmp_mul_ground(f_5, ZZ(7), 2, ZZ), 2, ZZ) == (ZZ(7), f_5)

    assert dmp_ground_primitive(f_6, 3, ZZ) == (ZZ(1), f_6)
    assert dmp_ground_primitive(
        dmp_mul_ground(f_6, ZZ(8), 3, ZZ), 3, ZZ) == (ZZ(8), f_6)

    assert dmp_ground_primitive([[ZZ(2)]], 1, ZZ) == (ZZ(2), [[ZZ(1)]])
    assert dmp_ground_primitive([[QQ(2)]], 1, QQ) == (QQ(2), [[QQ(1)]])

    assert dmp_ground_primitive(
        [[QQ(2, 3)], [QQ(4, 9)]], 1, QQ) == (QQ(2, 9), [[QQ(3)], [QQ(2)]])
    assert dmp_ground_primitive(
        [[QQ(2, 3)], [QQ(4, 5)]], 1, QQ) == (QQ(2, 15), [[QQ(5)], [QQ(6)]])


def test_dup_extract():
    f = dup_normal([2930944, 0, 2198208, 0, 549552, 0, 45796], ZZ)
    g = dup_normal([17585664, 0, 8792832, 0, 1099104, 0], ZZ)

    F = dup_normal([64, 0, 48, 0, 12, 0, 1], ZZ)
    G = dup_normal([384, 0, 192, 0, 24, 0], ZZ)

    assert dup_extract(f, g, ZZ) == (45796, F, G)


def test_dmp_ground_extract():
    f = dmp_normal(
        [[2930944], [], [2198208], [], [549552], [], [45796]], 1, ZZ)
    g = dmp_normal([[17585664], [], [8792832], [], [1099104], []], 1, ZZ)

    F = dmp_normal([[64], [], [48], [], [12], [], [1]], 1, ZZ)
    G = dmp_normal([[384], [], [192], [], [24], []], 1, ZZ)

    assert dmp_ground_extract(f, g, 1, ZZ) == (45796, F, G)


def test_dup_real_imag():
    assert dup_real_imag([], ZZ) == ([[]], [[]])
    assert dup_real_imag([1], ZZ) == ([[1]], [[]])

    assert dup_real_imag([1, 1], ZZ) == ([[1], [1]], [[1, 0]])
    assert dup_real_imag([1, 2], ZZ) == ([[1], [2]], [[1, 0]])

    assert dup_real_imag(
        [1, 2, 3], ZZ) == ([[1], [2], [-1, 0, 3]], [[2, 0], [2, 0]])

    assert dup_real_imag([ZZ(1), ZZ(0), ZZ(1), ZZ(3)], ZZ) == (
        [[ZZ(1)], [], [ZZ(-3), ZZ(0), ZZ(1)], [ZZ(3)]],
        [[ZZ(3), ZZ(0)], [], [ZZ(-1), ZZ(0), ZZ(1), ZZ(0)]]
    )

    raises(DomainError, lambda: dup_real_imag([EX(1), EX(2)], EX))



def test_dup_mirror():
    assert dup_mirror([], ZZ) == []
    assert dup_mirror([1], ZZ) == [1]

    assert dup_mirror([1, 2, 3, 4, 5], ZZ) == [1, -2, 3, -4, 5]
    assert dup_mirror([1, 2, 3, 4, 5, 6], ZZ) == [-1, 2, -3, 4, -5, 6]


def test_dup_scale():
    assert dup_scale([], -1, ZZ) == []
    assert dup_scale([1], -1, ZZ) == [1]

    assert dup_scale([1, 2, 3, 4, 5], -1, ZZ) == [1, -2, 3, -4, 5]
    assert dup_scale([1, 2, 3, 4, 5], -7, ZZ) == [2401, -686, 147, -28, 5]


def test_dup_shift():
    assert dup_shift([], 1, ZZ) == []
    assert dup_shift([1], 1, ZZ) == [1]

    assert dup_shift([1, 2, 3, 4, 5], 1, ZZ) == [1, 6, 15, 20, 15]
    assert dup_shift([1, 2, 3, 4, 5], 7, ZZ) == [1, 30, 339, 1712, 3267]


def test_dmp_shift():
    assert dmp_shift([ZZ(1), ZZ(2)], [ZZ(1)], 0, ZZ) == [ZZ(1), ZZ(3)]

    assert dmp_shift([[]], [ZZ(1), ZZ(2)], 1, ZZ) == [[]]

    xy = [[ZZ(1), ZZ(0)], []]               # x*y
    x1y2 = [[ZZ(1), ZZ(2)], [ZZ(1), ZZ(2)]] # (x+1)*(y+2)
    assert dmp_shift(xy, [ZZ(1), ZZ(2)], 1, ZZ) == x1y2


def test_dup_transform():
    assert dup_transform([], [], [1, 1], ZZ) == []
    assert dup_transform([], [1], [1, 1], ZZ) == []
    assert dup_transform([], [1, 2], [1, 1], ZZ) == []

    assert dup_transform([6, -5, 4, -3, 17], [1, -3, 4], [2, -3], ZZ) == \
        [6, -82, 541, -2205, 6277, -12723, 17191, -13603, 4773]


def test_dup_compose():
    assert dup_compose([], [], ZZ) == []
    assert dup_compose([], [1], ZZ) == []
    assert dup_compose([], [1, 2], ZZ) == []

    assert dup_compose([1], [], ZZ) == [1]

    assert dup_compose([1, 2, 0], [], ZZ) == []
    assert dup_compose([1, 2, 1], [], ZZ) == [1]

    assert dup_compose([1, 2, 1], [1], ZZ) == [4]
    assert dup_compose([1, 2, 1], [7], ZZ) == [64]

    assert dup_compose([1, 2, 1], [1, -1], ZZ) == [1, 0, 0]
    assert dup_compose([1, 2, 1], [1, 1], ZZ) == [1, 4, 4]
    assert dup_compose([1, 2, 1], [1, 2, 1], ZZ) == [1, 4, 8, 8, 4]


def test_dmp_compose():
    assert dmp_compose([1, 2, 1], [1, 2, 1], 0, ZZ) == [1, 4, 8, 8, 4]

    assert dmp_compose([[[]]], [[[]]], 2, ZZ) == [[[]]]
    assert dmp_compose([[[]]], [[[1]]], 2, ZZ) == [[[]]]
    assert dmp_compose([[[]]], [[[1]], [[2]]], 2, ZZ) == [[[]]]

    assert dmp_compose([[[1]]], [], 2, ZZ) == [[[1]]]

    assert dmp_compose([[1], [2], [ ]], [[]], 1, ZZ) == [[]]
    assert dmp_compose([[1], [2], [1]], [[]], 1, ZZ) == [[1]]

    assert dmp_compose([[1], [2], [1]], [[1]], 1, ZZ) == [[4]]
    assert dmp_compose([[1], [2], [1]], [[7]], 1, ZZ) == [[64]]

    assert dmp_compose([[1], [2], [1]], [[1], [-1]], 1, ZZ) == [[1], [ ], [ ]]
    assert dmp_compose([[1], [2], [1]], [[1], [ 1]], 1, ZZ) == [[1], [4], [4]]

    assert dmp_compose(
        [[1], [2], [1]], [[1], [2], [1]], 1, ZZ) == [[1], [4], [8], [8], [4]]


def test_dup_decompose():
    assert dup_decompose([1], ZZ) == [[1]]

    assert dup_decompose([1, 0], ZZ) == [[1, 0]]
    assert dup_decompose([1, 0, 0, 0], ZZ) == [[1, 0, 0, 0]]

    assert dup_decompose([1, 0, 0, 0, 0], ZZ) == [[1, 0, 0], [1, 0, 0]]
    assert dup_decompose(
        [1, 0, 0, 0, 0, 0, 0], ZZ) == [[1, 0, 0, 0], [1, 0, 0]]

    assert dup_decompose([7, 0, 0, 0, 1], ZZ) == [[7, 0, 1], [1, 0, 0]]
    assert dup_decompose([4, 0, 3, 0, 2], ZZ) == [[4, 3, 2], [1, 0, 0]]

    f = [1, 0, 20, 0, 150, 0, 500, 0, 625, -2, 0, -10, 9]

    assert dup_decompose(f, ZZ) == [[1, 0, 0, -2, 9], [1, 0, 5, 0]]

    f = [2, 0, 40, 0, 300, 0, 1000, 0, 1250, -4, 0, -20, 18]

    assert dup_decompose(f, ZZ) == [[2, 0, 0, -4, 18], [1, 0, 5, 0]]

    f = [1, 0, 20, -8, 150, -120, 524, -600, 865, -1034, 600, -170, 29]

    assert dup_decompose(f, ZZ) == [[1, -8, 24, -34, 29], [1, 0, 5, 0]]

    R, t = ring("t", ZZ)
    f = [6*t**2 - 42,
         48*t**2 + 96,
         144*t**2 + 648*t + 288,
         624*t**2 + 864*t + 384,
         108*t**3 + 312*t**2 + 432*t + 192]

    assert dup_decompose(f, R.to_domain()) == [f]


def test_dmp_lift():
    q = [QQ(1, 1), QQ(0, 1), QQ(1, 1)]

    f_a = [ANP([QQ(1, 1)], q, QQ), ANP([], q, QQ), ANP([], q, QQ),
         ANP([QQ(1, 1), QQ(0, 1)], q, QQ), ANP([QQ(17, 1), QQ(0, 1)], q, QQ)]

    f_lift = [QQ(1), QQ(0), QQ(0), QQ(0), QQ(0), QQ(0), QQ(2), QQ(0), QQ(578),
              QQ(0), QQ(0), QQ(0), QQ(1), QQ(0), QQ(-578), QQ(0), QQ(83521)]

    assert dmp_lift(f_a, 0, QQ.algebraic_field(I)) == f_lift

    f_g = [QQ_I(1), QQ_I(0), QQ_I(0), QQ_I(0, 1), QQ_I(0, 17)]

    assert dmp_lift(f_g, 0, QQ_I) == f_lift

    raises(DomainError, lambda: dmp_lift([EX(1), EX(2)], 0, EX))


def test_dup_sign_variations():
    assert dup_sign_variations([], ZZ) == 0
    assert dup_sign_variations([1, 0], ZZ) == 0
    assert dup_sign_variations([1, 0, 2], ZZ) == 0
    assert dup_sign_variations([1, 0, 3, 0], ZZ) == 0
    assert dup_sign_variations([1, 0, 4, 0, 5], ZZ) == 0

    assert dup_sign_variations([-1, 0, 2], ZZ) == 1
    assert dup_sign_variations([-1, 0, 3, 0], ZZ) == 1
    assert dup_sign_variations([-1, 0, 4, 0, 5], ZZ) == 1

    assert dup_sign_variations([-1, -4, -5], ZZ) == 0
    assert dup_sign_variations([ 1, -4, -5], ZZ) == 1
    assert dup_sign_variations([ 1, 4, -5], ZZ) == 1
    assert dup_sign_variations([ 1, -4, 5], ZZ) == 2
    assert dup_sign_variations([-1, 4, -5], ZZ) == 2
    assert dup_sign_variations([-1, 4, 5], ZZ) == 1
    assert dup_sign_variations([-1, -4, 5], ZZ) == 1
    assert dup_sign_variations([ 1, 4, 5], ZZ) == 0

    assert dup_sign_variations([-1, 0, -4, 0, -5], ZZ) == 0
    assert dup_sign_variations([ 1, 0, -4, 0, -5], ZZ) == 1
    assert dup_sign_variations([ 1, 0, 4, 0, -5], ZZ) == 1
    assert dup_sign_variations([ 1, 0, -4, 0, 5], ZZ) == 2
    assert dup_sign_variations([-1, 0, 4, 0, -5], ZZ) == 2
    assert dup_sign_variations([-1, 0, 4, 0, 5], ZZ) == 1
    assert dup_sign_variations([-1, 0, -4, 0, 5], ZZ) == 1
    assert dup_sign_variations([ 1, 0, 4, 0, 5], ZZ) == 0


def test_dup_clear_denoms():
    assert dup_clear_denoms([], QQ, ZZ) == (ZZ(1), [])

    assert dup_clear_denoms([QQ(1)], QQ, ZZ) == (ZZ(1), [QQ(1)])
    assert dup_clear_denoms([QQ(7)], QQ, ZZ) == (ZZ(1), [QQ(7)])

    assert dup_clear_denoms([QQ(7, 3)], QQ) == (ZZ(3), [QQ(7)])
    assert dup_clear_denoms([QQ(7, 3)], QQ, ZZ) == (ZZ(3), [QQ(7)])

    assert dup_clear_denoms(
        [QQ(3), QQ(1), QQ(0)], QQ, ZZ) == (ZZ(1), [QQ(3), QQ(1), QQ(0)])
    assert dup_clear_denoms(
        [QQ(1), QQ(1, 2), QQ(0)], QQ, ZZ) == (ZZ(2), [QQ(2), QQ(1), QQ(0)])

    assert dup_clear_denoms([QQ(3), QQ(
        1), QQ(0)], QQ, ZZ, convert=True) == (ZZ(1), [ZZ(3), ZZ(1), ZZ(0)])
    assert dup_clear_denoms([QQ(1), QQ(
        1, 2), QQ(0)], QQ, ZZ, convert=True) == (ZZ(2), [ZZ(2), ZZ(1), ZZ(0)])

    assert dup_clear_denoms(
        [EX(S(3)/2), EX(S(9)/4)], EX) == (EX(4), [EX(6), EX(9)])

    assert dup_clear_denoms([EX(7)], EX) == (EX(1), [EX(7)])
    assert dup_clear_denoms([EX(sin(x)/x), EX(0)], EX) == (EX(x), [EX(sin(x)), EX(0)])

    F = RR.frac_field(x)
    result = dup_clear_denoms([F(8.48717/(8.0089*x + 2.83)), F(0.0)], F)
    assert str(result) == "(x + 0.353356890459364, [1.05971731448763, 0.0])"

def test_dmp_clear_denoms():
    assert dmp_clear_denoms([[]], 1, QQ, ZZ) == (ZZ(1), [[]])

    assert dmp_clear_denoms([[QQ(1)]], 1, QQ, ZZ) == (ZZ(1), [[QQ(1)]])
    assert dmp_clear_denoms([[QQ(7)]], 1, QQ, ZZ) == (ZZ(1), [[QQ(7)]])

    assert dmp_clear_denoms([[QQ(7, 3)]], 1, QQ) == (ZZ(3), [[QQ(7)]])
    assert dmp_clear_denoms([[QQ(7, 3)]], 1, QQ, ZZ) == (ZZ(3), [[QQ(7)]])

    assert dmp_clear_denoms(
        [[QQ(3)], [QQ(1)], []], 1, QQ, ZZ) == (ZZ(1), [[QQ(3)], [QQ(1)], []])
    assert dmp_clear_denoms([[QQ(
        1)], [QQ(1, 2)], []], 1, QQ, ZZ) == (ZZ(2), [[QQ(2)], [QQ(1)], []])

    assert dmp_clear_denoms([QQ(3), QQ(
        1), QQ(0)], 0, QQ, ZZ, convert=True) == (ZZ(1), [ZZ(3), ZZ(1), ZZ(0)])
    assert dmp_clear_denoms([QQ(1), QQ(1, 2), QQ(
        0)], 0, QQ, ZZ, convert=True) == (ZZ(2), [ZZ(2), ZZ(1), ZZ(0)])

    assert dmp_clear_denoms([[QQ(3)], [QQ(
        1)], []], 1, QQ, ZZ, convert=True) == (ZZ(1), [[QQ(3)], [QQ(1)], []])
    assert dmp_clear_denoms([[QQ(1)], [QQ(1, 2)], []], 1, QQ, ZZ,
                            convert=True) == (ZZ(2), [[QQ(2)], [QQ(1)], []])

    assert dmp_clear_denoms(
        [[EX(S(3)/2)], [EX(S(9)/4)]], 1, EX) == (EX(4), [[EX(6)], [EX(9)]])
    assert dmp_clear_denoms([[EX(7)]], 1, EX) == (EX(1), [[EX(7)]])
    assert dmp_clear_denoms([[EX(sin(x)/x), EX(0)]], 1, EX) == (EX(x), [[EX(sin(x)), EX(0)]])
