"""Tests for efficient functions for generating Appell sequences."""
from sympy.core.numbers import Rational as Q
from sympy.polys.polytools import Poly
from sympy.testing.pytest import raises
from sympy.polys.appellseqs import (bernoulli_poly, bernoulli_c_poly,
    euler_poly, genocchi_poly, andre_poly)
from sympy.abc import x

def test_bernoulli_poly():
    raises(ValueError, lambda: bernoulli_poly(-1, x))
    assert bernoulli_poly(1, x, polys=True) == Poly(x - Q(1,2))

    assert bernoulli_poly(0, x) == 1
    assert bernoulli_poly(1, x) == x - Q(1,2)
    assert bernoulli_poly(2, x) == x**2 - x + Q(1,6)
    assert bernoulli_poly(3, x) == x**3 - Q(3,2)*x**2 + Q(1,2)*x
    assert bernoulli_poly(4, x) == x**4 - 2*x**3 + x**2 - Q(1,30)
    assert bernoulli_poly(5, x) == x**5 - Q(5,2)*x**4 + Q(5,3)*x**3 - Q(1,6)*x
    assert bernoulli_poly(6, x) == x**6 - 3*x**5 + Q(5,2)*x**4 - Q(1,2)*x**2 + Q(1,42)

    assert bernoulli_poly(1).dummy_eq(x - Q(1,2))
    assert bernoulli_poly(1, polys=True) == Poly(x - Q(1,2))

def test_bernoulli_c_poly():
    raises(ValueError, lambda: bernoulli_c_poly(-1, x))
    assert bernoulli_c_poly(1, x, polys=True) == Poly(x, domain='QQ')

    assert bernoulli_c_poly(0, x) == 1
    assert bernoulli_c_poly(1, x) == x
    assert bernoulli_c_poly(2, x) == x**2 - Q(1,3)
    assert bernoulli_c_poly(3, x) == x**3 - x
    assert bernoulli_c_poly(4, x) == x**4 - 2*x**2 + Q(7,15)
    assert bernoulli_c_poly(5, x) == x**5 - Q(10,3)*x**3 + Q(7,3)*x
    assert bernoulli_c_poly(6, x) == x**6 - 5*x**4 + 7*x**2 - Q(31,21)

    assert bernoulli_c_poly(1).dummy_eq(x)
    assert bernoulli_c_poly(1, polys=True) == Poly(x, domain='QQ')

    assert 2**8 * bernoulli_poly(8, (x+1)/2).expand() == bernoulli_c_poly(8, x)
    assert 2**9 * bernoulli_poly(9, (x+1)/2).expand() == bernoulli_c_poly(9, x)

def test_genocchi_poly():
    raises(ValueError, lambda: genocchi_poly(-1, x))
    assert genocchi_poly(2, x, polys=True) == Poly(-2*x + 1)

    assert genocchi_poly(0, x) == 0
    assert genocchi_poly(1, x) == -1
    assert genocchi_poly(2, x) == 1 - 2*x
    assert genocchi_poly(3, x) == 3*x - 3*x**2
    assert genocchi_poly(4, x) == -1 + 6*x**2 - 4*x**3
    assert genocchi_poly(5, x) == -5*x + 10*x**3 - 5*x**4
    assert genocchi_poly(6, x) == 3 - 15*x**2 + 15*x**4 - 6*x**5

    assert genocchi_poly(2).dummy_eq(-2*x + 1)
    assert genocchi_poly(2, polys=True) == Poly(-2*x + 1)

    assert 2 * (bernoulli_poly(8, x) - bernoulli_c_poly(8, x)) == genocchi_poly(8, x)
    assert 2 * (bernoulli_poly(9, x) - bernoulli_c_poly(9, x)) == genocchi_poly(9, x)

def test_euler_poly():
    raises(ValueError, lambda: euler_poly(-1, x))
    assert euler_poly(1, x, polys=True) == Poly(x - Q(1,2))

    assert euler_poly(0, x) == 1
    assert euler_poly(1, x) == x - Q(1,2)
    assert euler_poly(2, x) == x**2 - x
    assert euler_poly(3, x) == x**3 - Q(3,2)*x**2 + Q(1,4)
    assert euler_poly(4, x) == x**4 - 2*x**3 + x
    assert euler_poly(5, x) == x**5 - Q(5,2)*x**4 + Q(5,2)*x**2 - Q(1,2)
    assert euler_poly(6, x) == x**6 - 3*x**5 + 5*x**3 - 3*x

    assert euler_poly(1).dummy_eq(x - Q(1,2))
    assert euler_poly(1, polys=True) == Poly(x - Q(1,2))

    assert genocchi_poly(9, x) == euler_poly(8, x) * -9
    assert genocchi_poly(10, x) == euler_poly(9, x) * -10

def test_andre_poly():
    raises(ValueError, lambda: andre_poly(-1, x))
    assert andre_poly(1, x, polys=True) == Poly(x)

    assert andre_poly(0, x) == 1
    assert andre_poly(1, x) == x
    assert andre_poly(2, x) == x**2 - 1
    assert andre_poly(3, x) == x**3 - 3*x
    assert andre_poly(4, x) == x**4 - 6*x**2 + 5
    assert andre_poly(5, x) == x**5 - 10*x**3 + 25*x
    assert andre_poly(6, x) == x**6 - 15*x**4 + 75*x**2 - 61

    assert andre_poly(1).dummy_eq(x)
    assert andre_poly(1, polys=True) == Poly(x)
