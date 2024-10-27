"""Tests for noncommutative symbols and expressions."""

from sympy.core.function import expand
from sympy.core.numbers import I
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import (adjoint, conjugate, transpose)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.polys.polytools import (cancel, factor)
from sympy.simplify.combsimp import combsimp
from sympy.simplify.gammasimp import gammasimp
from sympy.simplify.radsimp import (collect, radsimp, rcollect)
from sympy.simplify.ratsimp import ratsimp
from sympy.simplify.simplify import (posify, simplify)
from sympy.simplify.trigsimp import trigsimp
from sympy.abc import x, y, z
from sympy.testing.pytest import XFAIL

A, B, C = symbols("A B C", commutative=False)
X = symbols("X", commutative=False, hermitian=True)
Y = symbols("Y", commutative=False, antihermitian=True)


def test_adjoint():
    assert adjoint(A).is_commutative is False
    assert adjoint(A*A) == adjoint(A)**2
    assert adjoint(A*B) == adjoint(B)*adjoint(A)
    assert adjoint(A*B**2) == adjoint(B)**2*adjoint(A)
    assert adjoint(A*B - B*A) == adjoint(B)*adjoint(A) - adjoint(A)*adjoint(B)
    assert adjoint(A + I*B) == adjoint(A) - I*adjoint(B)

    assert adjoint(X) == X
    assert adjoint(-I*X) == I*X
    assert adjoint(Y) == -Y
    assert adjoint(-I*Y) == -I*Y

    assert adjoint(X) == conjugate(transpose(X))
    assert adjoint(Y) == conjugate(transpose(Y))
    assert adjoint(X) == transpose(conjugate(X))
    assert adjoint(Y) == transpose(conjugate(Y))


def test_cancel():
    assert cancel(A*B - B*A) == A*B - B*A
    assert cancel(A*B*(x - 1)) == A*B*(x - 1)
    assert cancel(A*B*(x**2 - 1)/(x + 1)) == A*B*(x - 1)
    assert cancel(A*B*(x**2 - 1)/(x + 1) - B*A*(x - 1)) == A*B*(x - 1) + (1 - x)*B*A


@XFAIL
def test_collect():
    assert collect(A*B - B*A, A) == A*B - B*A
    assert collect(A*B - B*A, B) == A*B - B*A
    assert collect(A*B - B*A, x) == A*B - B*A


def test_combsimp():
    assert combsimp(A*B - B*A) == A*B - B*A


def test_gammasimp():
    assert gammasimp(A*B - B*A) == A*B - B*A


def test_conjugate():
    assert conjugate(A).is_commutative is False
    assert (A*A).conjugate() == conjugate(A)**2
    assert (A*B).conjugate() == conjugate(A)*conjugate(B)
    assert (A*B**2).conjugate() == conjugate(A)*conjugate(B)**2
    assert (A*B - B*A).conjugate() == \
        conjugate(A)*conjugate(B) - conjugate(B)*conjugate(A)
    assert (A*B).conjugate() - (B*A).conjugate() == \
        conjugate(A)*conjugate(B) - conjugate(B)*conjugate(A)
    assert (A + I*B).conjugate() == conjugate(A) - I*conjugate(B)


def test_expand():
    assert expand((A*B)**2) == A*B*A*B
    assert expand(A*B - B*A) == A*B - B*A
    assert expand((A*B/A)**2) == A*B*B/A
    assert expand(B*A*(A + B)*B) == B*A**2*B + B*A*B**2
    assert expand(B*A*(A + C)*B) == B*A**2*B + B*A*C*B


def test_factor():
    assert factor(A*B - B*A) == A*B - B*A


def test_posify():
    assert posify(A)[0].is_commutative is False
    for q in (A*B/A, (A*B/A)**2, (A*B)**2, A*B - B*A):
        p = posify(q)
        assert p[0].subs(p[1]) == q


def test_radsimp():
    assert radsimp(A*B - B*A) == A*B - B*A


@XFAIL
def test_ratsimp():
    assert ratsimp(A*B - B*A) == A*B - B*A


@XFAIL
def test_rcollect():
    assert rcollect(A*B - B*A, A) == A*B - B*A
    assert rcollect(A*B - B*A, B) == A*B - B*A
    assert rcollect(A*B - B*A, x) == A*B - B*A


def test_simplify():
    assert simplify(A*B - B*A) == A*B - B*A


def test_subs():
    assert (x*y*A).subs(x*y, z) == A*z
    assert (x*A*B).subs(x*A, C) == C*B
    assert (x*A*x*x).subs(x**2*A, C) == x*C
    assert (x*A*x*B).subs(x**2*A, C) == C*B
    assert (A**2*B**2).subs(A*B**2, C) == A*C
    assert (A*A*A + A*B*A).subs(A*A*A, C) == C + A*B*A


def test_transpose():
    assert transpose(A).is_commutative is False
    assert transpose(A*A) == transpose(A)**2
    assert transpose(A*B) == transpose(B)*transpose(A)
    assert transpose(A*B**2) == transpose(B)**2*transpose(A)
    assert transpose(A*B - B*A) == \
        transpose(B)*transpose(A) - transpose(A)*transpose(B)
    assert transpose(A + I*B) == transpose(A) + I*transpose(B)

    assert transpose(X) == conjugate(X)
    assert transpose(-I*X) == -I*conjugate(X)
    assert transpose(Y) == -conjugate(Y)
    assert transpose(-I*Y) == I*conjugate(Y)


def test_trigsimp():
    assert trigsimp(A*sin(x)**2 + A*cos(x)**2) == A
