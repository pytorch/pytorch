"""Tests for homomorphisms."""

from sympy.core.singleton import S
from sympy.polys.domains.rationalfield import QQ
from sympy.abc import x, y
from sympy.polys.agca import homomorphism
from sympy.testing.pytest import raises


def test_printing():
    R = QQ.old_poly_ring(x)

    assert str(homomorphism(R.free_module(1), R.free_module(1), [0])) == \
        'Matrix([[0]]) : QQ[x]**1 -> QQ[x]**1'
    assert str(homomorphism(R.free_module(2), R.free_module(2), [0, 0])) == \
        'Matrix([                       \n[0, 0], : QQ[x]**2 -> QQ[x]**2\n[0, 0]])                       '
    assert str(homomorphism(R.free_module(1), R.free_module(1) / [[x]], [0])) == \
        'Matrix([[0]]) : QQ[x]**1 -> QQ[x]**1/<[x]>'
    assert str(R.free_module(0).identity_hom()) == 'Matrix(0, 0, []) : QQ[x]**0 -> QQ[x]**0'

def test_operations():
    F = QQ.old_poly_ring(x).free_module(2)
    G = QQ.old_poly_ring(x).free_module(3)
    f = F.identity_hom()
    g = homomorphism(F, F, [0, [1, x]])
    h = homomorphism(F, F, [[1, 0], 0])
    i = homomorphism(F, G, [[1, 0, 0], [0, 1, 0]])

    assert f == f
    assert f != g
    assert f != i
    assert (f != F.identity_hom()) is False
    assert 2*f == f*2 == homomorphism(F, F, [[2, 0], [0, 2]])
    assert f/2 == homomorphism(F, F, [[S.Half, 0], [0, S.Half]])
    assert f + g == homomorphism(F, F, [[1, 0], [1, x + 1]])
    assert f - g == homomorphism(F, F, [[1, 0], [-1, 1 - x]])
    assert f*g == g == g*f
    assert h*g == homomorphism(F, F, [0, [1, 0]])
    assert g*h == homomorphism(F, F, [0, 0])
    assert i*f == i
    assert f([1, 2]) == [1, 2]
    assert g([1, 2]) == [2, 2*x]

    assert i.restrict_domain(F.submodule([x, x]))([x, x]) == i([x, x])
    h1 = h.quotient_domain(F.submodule([0, 1]))
    assert h1([1, 0]) == h([1, 0])
    assert h1.restrict_domain(h1.domain.submodule([x, 0]))([x, 0]) == h([x, 0])

    raises(TypeError, lambda: f/g)
    raises(TypeError, lambda: f + 1)
    raises(TypeError, lambda: f + i)
    raises(TypeError, lambda: f - 1)
    raises(TypeError, lambda: f*i)


def test_creation():
    F = QQ.old_poly_ring(x).free_module(3)
    G = QQ.old_poly_ring(x).free_module(2)
    SM = F.submodule([1, 1, 1])
    Q = F / SM
    SQ = Q.submodule([1, 0, 0])

    matrix = [[1, 0], [0, 1], [-1, -1]]
    h = homomorphism(F, G, matrix)
    h2 = homomorphism(Q, G, matrix)
    assert h.quotient_domain(SM) == h2
    raises(ValueError, lambda: h.quotient_domain(F.submodule([1, 0, 0])))
    assert h2.restrict_domain(SQ) == homomorphism(SQ, G, matrix)
    raises(ValueError, lambda: h.restrict_domain(G))
    raises(ValueError, lambda: h.restrict_codomain(G.submodule([1, 0])))
    raises(ValueError, lambda: h.quotient_codomain(F))

    im = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for M in [F, SM, Q, SQ]:
        assert M.identity_hom() == homomorphism(M, M, im)
    assert SM.inclusion_hom() == homomorphism(SM, F, im)
    assert SQ.inclusion_hom() == homomorphism(SQ, Q, im)
    assert Q.quotient_hom() == homomorphism(F, Q, im)
    assert SQ.quotient_hom() == homomorphism(SQ.base, SQ, im)

    class conv:
        def convert(x, y=None):
            return x

    class dummy:
        container = conv()

        def submodule(*args):
            return None
    raises(TypeError, lambda: homomorphism(dummy(), G, matrix))
    raises(TypeError, lambda: homomorphism(F, dummy(), matrix))
    raises(
        ValueError, lambda: homomorphism(QQ.old_poly_ring(x, y).free_module(3), G, matrix))
    raises(ValueError, lambda: homomorphism(F, G, [0, 0]))


def test_properties():
    R = QQ.old_poly_ring(x, y)
    F = R.free_module(2)
    h = homomorphism(F, F, [[x, 0], [y, 0]])
    assert h.kernel() == F.submodule([-y, x])
    assert h.image() == F.submodule([x, 0], [y, 0])
    assert not h.is_injective()
    assert not h.is_surjective()
    assert h.restrict_codomain(h.image()).is_surjective()
    assert h.restrict_domain(F.submodule([1, 0])).is_injective()
    assert h.quotient_domain(
        h.kernel()).restrict_codomain(h.image()).is_isomorphism()

    R2 = QQ.old_poly_ring(x, y, order=(("lex", x), ("ilex", y))) / [x**2 + 1]
    F = R2.free_module(2)
    h = homomorphism(F, F, [[x, 0], [y, y + 1]])
    assert h.is_isomorphism()
