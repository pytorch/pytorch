"""Test ideals.py code."""

from sympy.polys import QQ, ilex
from sympy.abc import x, y, z
from sympy.testing.pytest import raises


def test_ideal_operations():
    R = QQ.old_poly_ring(x, y)
    I = R.ideal(x)
    J = R.ideal(y)
    S = R.ideal(x*y)
    T = R.ideal(x, y)

    assert not (I == J)
    assert I == I

    assert I.union(J) == T
    assert I + J == T
    assert I + T == T

    assert not I.subset(T)
    assert T.subset(I)

    assert I.product(J) == S
    assert I*J == S
    assert x*J == S
    assert I*y == S
    assert R.convert(x)*J == S
    assert I*R.convert(y) == S

    assert not I.is_zero()
    assert not J.is_whole_ring()

    assert R.ideal(x**2 + 1, x).is_whole_ring()
    assert R.ideal() == R.ideal(0)
    assert R.ideal().is_zero()

    assert T.contains(x*y)
    assert T.subset([x, y])

    assert T.in_terms_of_generators(x) == [R(1), R(0)]

    assert T**0 == R.ideal(1)
    assert T**1 == T
    assert T**2 == R.ideal(x**2, y**2, x*y)
    assert I**5 == R.ideal(x**5)


def test_exceptions():
    I = QQ.old_poly_ring(x).ideal(x)
    J = QQ.old_poly_ring(y).ideal(1)
    raises(ValueError, lambda: I.union(x))
    raises(ValueError, lambda: I + J)
    raises(ValueError, lambda: I * J)
    raises(ValueError, lambda: I.union(J))
    assert (I == J) is False
    assert I != J


def test_nontriv_global():
    R = QQ.old_poly_ring(x, y, z)

    def contains(I, f):
        return R.ideal(*I).contains(f)

    assert contains([x, y], x)
    assert contains([x, y], x + y)
    assert not contains([x, y], 1)
    assert not contains([x, y], z)
    assert contains([x**2 + y, x**2 + x], x - y)
    assert not contains([x + y + z, x*y + x*z + y*z, x*y*z], x**2)
    assert contains([x + y + z, x*y + x*z + y*z, x*y*z], x**3)
    assert contains([x + y + z, x*y + x*z + y*z, x*y*z], x**4)
    assert not contains([x + y + z, x*y + x*z + y*z, x*y*z], x*y**2)
    assert contains([x + y + z, x*y + x*z + y*z, x*y*z], x**4 + y**3 + 2*z*y*x)
    assert contains([x + y + z, x*y + x*z + y*z, x*y*z], x*y*z)
    assert contains([x, 1 + x + y, 5 - 7*y], 1)
    assert contains(
        [x**3 + y**3, y**3 + z**3, z**3 + x**3, x**2*y + x**2*z + y**2*z],
        x**3)
    assert not contains(
        [x**3 + y**3, y**3 + z**3, z**3 + x**3, x**2*y + x**2*z + y**2*z],
        x**2 + y**2)

    # compare local order
    assert not contains([x*(1 + x + y), y*(1 + z)], x)
    assert not contains([x*(1 + x + y), y*(1 + z)], x + y)


def test_nontriv_local():
    R = QQ.old_poly_ring(x, y, z, order=ilex)

    def contains(I, f):
        return R.ideal(*I).contains(f)

    assert contains([x, y], x)
    assert contains([x, y], x + y)
    assert not contains([x, y], 1)
    assert not contains([x, y], z)
    assert contains([x**2 + y, x**2 + x], x - y)
    assert not contains([x + y + z, x*y + x*z + y*z, x*y*z], x**2)
    assert contains([x*(1 + x + y), y*(1 + z)], x)
    assert contains([x*(1 + x + y), y*(1 + z)], x + y)


def test_intersection():
    R = QQ.old_poly_ring(x, y, z)
    # SCA, example 1.8.11
    assert R.ideal(x, y).intersect(R.ideal(y**2, z)) == R.ideal(y**2, y*z, x*z)

    assert R.ideal(x, y).intersect(R.ideal()).is_zero()

    R = QQ.old_poly_ring(x, y, z, order="ilex")
    assert R.ideal(x, y).intersect(R.ideal(y**2 + y**2*z, z + z*x**3*y)) == \
        R.ideal(y**2, y*z, x*z)


def test_quotient():
    # SCA, example 1.8.13
    R = QQ.old_poly_ring(x, y, z)
    assert R.ideal(x, y).quotient(R.ideal(y**2, z)) == R.ideal(x, y)


def test_reduction():
    from sympy.polys.distributedmodules import sdm_nf_buchberger_reduced
    R = QQ.old_poly_ring(x, y)
    I = R.ideal(x**5, y)
    e = R.convert(x**3 + y**2)
    assert I.reduce_element(e) == e
    assert I.reduce_element(e, NF=sdm_nf_buchberger_reduced) == R.convert(x**3)
