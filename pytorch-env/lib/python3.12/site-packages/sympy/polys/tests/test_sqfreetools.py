"""Tests for square-free decomposition algorithms and related tools. """

from sympy.polys.rings import ring
from sympy.polys.domains import FF, ZZ, QQ
from sympy.polys.specialpolys import f_polys

from sympy.testing.pytest import raises
from sympy.external.gmpy import MPQ

f_0, f_1, f_2, f_3, f_4, f_5, f_6 = f_polys()

def test_dup_sqf():
    R, x = ring("x", ZZ)

    assert R.dup_sqf_part(0) == 0
    assert R.dup_sqf_p(0) is True

    assert R.dup_sqf_part(7) == 1
    assert R.dup_sqf_p(7) is True

    assert R.dup_sqf_part(2*x + 2) == x + 1
    assert R.dup_sqf_p(2*x + 2) is True

    assert R.dup_sqf_part(x**3 + x + 1) == x**3 + x + 1
    assert R.dup_sqf_p(x**3 + x + 1) is True

    assert R.dup_sqf_part(-x**3 + x + 1) == x**3 - x - 1
    assert R.dup_sqf_p(-x**3 + x + 1) is True

    assert R.dup_sqf_part(2*x**3 + 3*x**2) == 2*x**2 + 3*x
    assert R.dup_sqf_p(2*x**3 + 3*x**2) is False

    assert R.dup_sqf_part(-2*x**3 + 3*x**2) == 2*x**2 - 3*x
    assert R.dup_sqf_p(-2*x**3 + 3*x**2) is False

    assert R.dup_sqf_list(0) == (0, [])
    assert R.dup_sqf_list(1) == (1, [])

    assert R.dup_sqf_list(x) == (1, [(x, 1)])
    assert R.dup_sqf_list(2*x**2) == (2, [(x, 2)])
    assert R.dup_sqf_list(3*x**3) == (3, [(x, 3)])

    assert R.dup_sqf_list(-x**5 + x**4 + x - 1) == \
        (-1, [(x**3 + x**2 + x + 1, 1), (x - 1, 2)])
    assert R.dup_sqf_list(x**8 + 6*x**6 + 12*x**4 + 8*x**2) == \
        ( 1, [(x, 2), (x**2 + 2, 3)])

    assert R.dup_sqf_list(2*x**2 + 4*x + 2) == (2, [(x + 1, 2)])

    R, x = ring("x", QQ)
    assert R.dup_sqf_list(2*x**2 + 4*x + 2) == (2, [(x + 1, 2)])

    R, x = ring("x", FF(2))
    assert R.dup_sqf_list(x**2 + 1) == (1, [(x + 1, 2)])

    R, x = ring("x", FF(3))
    assert R.dup_sqf_list(x**10 + 2*x**7 + 2*x**4 + x) == \
        (1, [(x, 1),
             (x + 1, 3),
             (x + 2, 6)])

    R1, x = ring("x", ZZ)
    R2, y = ring("y", FF(3))

    f = x**3 + 1
    g = y**3 + 1

    assert R1.dup_sqf_part(f) == f
    assert R2.dup_sqf_part(g) == y + 1

    assert R1.dup_sqf_p(f) is True
    assert R2.dup_sqf_p(g) is False

    R, x, y = ring("x,y", ZZ)

    A = x**4 - 3*x**2 + 6
    D = x**6 - 5*x**4 + 5*x**2 + 4

    f, g = D, R.dmp_sub(A, R.dmp_mul(R.dmp_diff(D, 1), y))
    res = R.dmp_resultant(f, g)
    h = (4*y**2 + 1).drop(x)

    assert R.drop(x).dup_sqf_list(res) == (45796, [(h, 3)])

    Rt, t = ring("t", ZZ)
    R, x = ring("x", Rt)
    assert R.dup_sqf_list_include(t**3*x**2) == [(t**3, 1), (x, 2)]


def test_dmp_sqf():
    R, x, y = ring("x,y", ZZ)
    assert R.dmp_sqf_part(0) == 0
    assert R.dmp_sqf_p(0) is True

    assert R.dmp_sqf_part(7) == 1
    assert R.dmp_sqf_p(7) is True

    assert R.dmp_sqf_list(3) == (3, [])
    assert R.dmp_sqf_list_include(3) == [(3, 1)]

    R, x, y, z = ring("x,y,z", ZZ)
    assert R.dmp_sqf_p(f_0) is True
    assert R.dmp_sqf_p(f_0**2) is False
    assert R.dmp_sqf_p(f_1) is True
    assert R.dmp_sqf_p(f_1**2) is False
    assert R.dmp_sqf_p(f_2) is True
    assert R.dmp_sqf_p(f_2**2) is False
    assert R.dmp_sqf_p(f_3) is True
    assert R.dmp_sqf_p(f_3**2) is False
    assert R.dmp_sqf_p(f_5) is False
    assert R.dmp_sqf_p(f_5**2) is False

    assert R.dmp_sqf_p(f_4) is True
    assert R.dmp_sqf_part(f_4) == -f_4

    assert R.dmp_sqf_part(f_5) == x + y - z

    R, x, y, z, t = ring("x,y,z,t", ZZ)
    assert R.dmp_sqf_p(f_6) is True
    assert R.dmp_sqf_part(f_6) == f_6

    R, x = ring("x", ZZ)
    f = -x**5 + x**4 + x - 1

    assert R.dmp_sqf_list(f) == (-1, [(x**3 + x**2 + x + 1, 1), (x - 1, 2)])
    assert R.dmp_sqf_list_include(f) == [(-x**3 - x**2 - x - 1, 1), (x - 1, 2)]

    R, x, y = ring("x,y", ZZ)
    f = -x**5 + x**4 + x - 1

    assert R.dmp_sqf_list(f) == (-1, [(x**3 + x**2 + x + 1, 1), (x - 1, 2)])
    assert R.dmp_sqf_list_include(f) == [(-x**3 - x**2 - x - 1, 1), (x - 1, 2)]

    f = -x**2 + 2*x - 1
    assert R.dmp_sqf_list_include(f) == [(-1, 1), (x - 1, 2)]

    f = (y**2 + 1)**2*(x**2 + 2*x + 2)
    assert R.dmp_sqf_p(f) is False
    assert R.dmp_sqf_list(f) == (1, [(x**2 + 2*x + 2, 1), (y**2 + 1, 2)])

    R, x, y = ring("x,y", FF(2))
    raises(NotImplementedError, lambda: R.dmp_sqf_list(y**2 + 1))


def test_dup_gff_list():
    R, x = ring("x", ZZ)

    f = x**5 + 2*x**4 - x**3 - 2*x**2
    assert R.dup_gff_list(f) == [(x, 1), (x + 2, 4)]

    g = x**9 - 20*x**8 + 166*x**7 - 744*x**6 + 1965*x**5 - 3132*x**4 + 2948*x**3 - 1504*x**2 + 320*x
    assert R.dup_gff_list(g) == [(x**2 - 5*x + 4, 1), (x**2 - 5*x + 4, 2), (x, 3)]

    raises(ValueError, lambda: R.dup_gff_list(0))

def test_issue_26178():
    R, x, y, z = ring(['x', 'y', 'z'], QQ)
    assert (x**2 - 2*y**2 + 1).sqf_list() == (MPQ(1,1), [(x**2 - 2*y**2 + 1, 1)])
    assert (x**2 - 2*z**2 + 1).sqf_list() == (MPQ(1,1), [(x**2 - 2*z**2 + 1, 1)])
    assert (y**2 - 2*z**2 + 1).sqf_list() == (MPQ(1,1), [(y**2 - 2*z**2 + 1, 1)])
