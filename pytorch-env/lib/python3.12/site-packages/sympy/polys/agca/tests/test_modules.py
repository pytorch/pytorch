"""Test modules.py code."""

from sympy.polys.agca.modules import FreeModule, ModuleOrder, FreeModulePolyRing
from sympy.polys import CoercionFailed, QQ, lex, grlex, ilex, ZZ
from sympy.abc import x, y, z
from sympy.testing.pytest import raises
from sympy.core.numbers import Rational


def test_FreeModuleElement():
    M = QQ.old_poly_ring(x).free_module(3)
    e = M.convert([1, x, x**2])
    f = [QQ.old_poly_ring(x).convert(1), QQ.old_poly_ring(x).convert(x), QQ.old_poly_ring(x).convert(x**2)]
    assert list(e) == f
    assert f[0] == e[0]
    assert f[1] == e[1]
    assert f[2] == e[2]
    raises(IndexError, lambda: e[3])

    g = M.convert([x, 0, 0])
    assert e + g == M.convert([x + 1, x, x**2])
    assert f + g == M.convert([x + 1, x, x**2])
    assert -e == M.convert([-1, -x, -x**2])
    assert e - g == M.convert([1 - x, x, x**2])
    assert e != g

    assert M.convert([x, x, x]) / QQ.old_poly_ring(x).convert(x) == [1, 1, 1]
    R = QQ.old_poly_ring(x, order="ilex")
    assert R.free_module(1).convert([x]) / R.convert(x) == [1]


def test_FreeModule():
    M1 = FreeModule(QQ.old_poly_ring(x), 2)
    assert M1 == FreeModule(QQ.old_poly_ring(x), 2)
    assert M1 != FreeModule(QQ.old_poly_ring(y), 2)
    assert M1 != FreeModule(QQ.old_poly_ring(x), 3)
    M2 = FreeModule(QQ.old_poly_ring(x, order="ilex"), 2)

    assert [x, 1] in M1
    assert [x] not in M1
    assert [2, y] not in M1
    assert [1/(x + 1), 2] not in M1

    e = M1.convert([x, x**2 + 1])
    X = QQ.old_poly_ring(x).convert(x)
    assert e == [X, X**2 + 1]
    assert e == [x, x**2 + 1]
    assert 2*e == [2*x, 2*x**2 + 2]
    assert e*2 == [2*x, 2*x**2 + 2]
    assert e/2 == [x/2, (x**2 + 1)/2]
    assert x*e == [x**2, x**3 + x]
    assert e*x == [x**2, x**3 + x]
    assert X*e == [x**2, x**3 + x]
    assert e*X == [x**2, x**3 + x]

    assert [x, 1] in M2
    assert [x] not in M2
    assert [2, y] not in M2
    assert [1/(x + 1), 2] in M2

    e = M2.convert([x, x**2 + 1])
    X = QQ.old_poly_ring(x, order="ilex").convert(x)
    assert e == [X, X**2 + 1]
    assert e == [x, x**2 + 1]
    assert 2*e == [2*x, 2*x**2 + 2]
    assert e*2 == [2*x, 2*x**2 + 2]
    assert e/2 == [x/2, (x**2 + 1)/2]
    assert x*e == [x**2, x**3 + x]
    assert e*x == [x**2, x**3 + x]
    assert e/(1 + x) == [x/(1 + x), (x**2 + 1)/(1 + x)]
    assert X*e == [x**2, x**3 + x]
    assert e*X == [x**2, x**3 + x]

    M3 = FreeModule(QQ.old_poly_ring(x, y), 2)
    assert M3.convert(e) == M3.convert([x, x**2 + 1])

    assert not M3.is_submodule(0)
    assert not M3.is_zero()

    raises(NotImplementedError, lambda: ZZ.old_poly_ring(x).free_module(2))
    raises(NotImplementedError, lambda: FreeModulePolyRing(ZZ, 2))
    raises(CoercionFailed, lambda: M1.convert(QQ.old_poly_ring(x).free_module(3)
           .convert([1, 2, 3])))
    raises(CoercionFailed, lambda: M3.convert(1))


def test_ModuleOrder():
    o1 = ModuleOrder(lex, grlex, False)
    o2 = ModuleOrder(ilex, lex, False)

    assert o1 == ModuleOrder(lex, grlex, False)
    assert (o1 != ModuleOrder(lex, grlex, False)) is False
    assert o1 != o2

    assert o1((1, 2, 3)) == (1, (5, (2, 3)))
    assert o2((1, 2, 3)) == (-1, (2, 3))


def test_SubModulePolyRing_global():
    R = QQ.old_poly_ring(x, y)
    F = R.free_module(3)
    Fd = F.submodule([1, 0, 0], [1, 2, 0], [1, 2, 3])
    M = F.submodule([x**2 + y**2, 1, 0], [x, y, 1])

    assert F == Fd
    assert Fd == F
    assert F != M
    assert M != F
    assert Fd != M
    assert M != Fd
    assert Fd == F.submodule(*F.basis())

    assert Fd.is_full_module()
    assert not M.is_full_module()
    assert not Fd.is_zero()
    assert not M.is_zero()
    assert Fd.submodule().is_zero()

    assert M.contains([x**2 + y**2 + x, 1 + y, 1])
    assert not M.contains([x**2 + y**2 + x, 1 + y, 2])
    assert M.contains([y**2, 1 - x*y, -x])

    assert not F.submodule([1 + x, 0, 0]) == F.submodule([1, 0, 0])
    assert F.submodule([1, 0, 0], [0, 1, 0]).union(F.submodule([0, 0, 1])) == F
    assert not M.is_submodule(0)

    m = F.convert([x**2 + y**2, 1, 0])
    n = M.convert(m)
    assert m.module is F
    assert n.module is M

    raises(ValueError, lambda: M.submodule([1, 0, 0]))
    raises(TypeError, lambda: M.union(1))
    raises(ValueError, lambda: M.union(R.free_module(1).submodule([x])))

    assert F.submodule([x, x, x]) != F.submodule([x, x, x], order="ilex")


def test_SubModulePolyRing_local():
    R = QQ.old_poly_ring(x, y, order=ilex)
    F = R.free_module(3)
    Fd = F.submodule([1 + x, 0, 0], [1 + y, 2 + 2*y, 0], [1, 2, 3])
    M = F.submodule([x**2 + y**2, 1, 0], [x, y, 1])

    assert F == Fd
    assert Fd == F
    assert F != M
    assert M != F
    assert Fd != M
    assert M != Fd
    assert Fd == F.submodule(*F.basis())

    assert Fd.is_full_module()
    assert not M.is_full_module()
    assert not Fd.is_zero()
    assert not M.is_zero()
    assert Fd.submodule().is_zero()

    assert M.contains([x**2 + y**2 + x, 1 + y, 1])
    assert not M.contains([x**2 + y**2 + x, 1 + y, 2])
    assert M.contains([y**2, 1 - x*y, -x])

    assert F.submodule([1 + x, 0, 0]) == F.submodule([1, 0, 0])
    assert F.submodule(
        [1, 0, 0], [0, 1, 0]).union(F.submodule([0, 0, 1 + x*y])) == F

    raises(ValueError, lambda: M.submodule([1, 0, 0]))


def test_SubModulePolyRing_nontriv_global():
    R = QQ.old_poly_ring(x, y, z)
    F = R.free_module(1)

    def contains(I, f):
        return F.submodule(*[[g] for g in I]).contains([f])

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


def test_SubModulePolyRing_nontriv_local():
    R = QQ.old_poly_ring(x, y, z, order=ilex)
    F = R.free_module(1)

    def contains(I, f):
        return F.submodule(*[[g] for g in I]).contains([f])

    assert contains([x, y], x)
    assert contains([x, y], x + y)
    assert not contains([x, y], 1)
    assert not contains([x, y], z)
    assert contains([x**2 + y, x**2 + x], x - y)
    assert not contains([x + y + z, x*y + x*z + y*z, x*y*z], x**2)
    assert contains([x*(1 + x + y), y*(1 + z)], x)
    assert contains([x*(1 + x + y), y*(1 + z)], x + y)


def test_syzygy():
    R = QQ.old_poly_ring(x, y, z)
    M = R.free_module(1).submodule([x*y], [y*z], [x*z])
    S = R.free_module(3).submodule([0, x, -y], [z, -x, 0])
    assert M.syzygy_module() == S

    M2 = M / ([x*y*z],)
    S2 = R.free_module(3).submodule([z, 0, 0], [0, x, 0], [0, 0, y])
    assert M2.syzygy_module() == S2

    F = R.free_module(3)
    assert F.submodule(*F.basis()).syzygy_module() == F.submodule()

    R2 = QQ.old_poly_ring(x, y, z) / [x*y*z]
    M3 = R2.free_module(1).submodule([x*y], [y*z], [x*z])
    S3 = R2.free_module(3).submodule([z, 0, 0], [0, x, 0], [0, 0, y])
    assert M3.syzygy_module() == S3


def test_in_terms_of_generators():
    R = QQ.old_poly_ring(x, order="ilex")
    M = R.free_module(2).submodule([2*x, 0], [1, 2])
    assert M.in_terms_of_generators(
        [x, x]) == [R.convert(Rational(1, 4)), R.convert(x/2)]
    raises(ValueError, lambda: M.in_terms_of_generators([1, 0]))

    M = R.free_module(2) / ([x, 0], [1, 1])
    SM = M.submodule([1, x])
    assert SM.in_terms_of_generators([2, 0]) == [R.convert(-2/(x - 1))]

    R = QQ.old_poly_ring(x, y) / [x**2 - y**2]
    M = R.free_module(2)
    SM = M.submodule([x, 0], [0, y])
    assert SM.in_terms_of_generators(
        [x**2, x**2]) == [R.convert(x), R.convert(y)]


def test_QuotientModuleElement():
    R = QQ.old_poly_ring(x)
    F = R.free_module(3)
    N = F.submodule([1, x, x**2])
    M = F/N
    e = M.convert([x**2, 2, 0])

    assert M.convert([x + 1, x**2 + x, x**3 + x**2]) == 0
    assert e == [x**2, 2, 0] + N == F.convert([x**2, 2, 0]) + N == \
        M.convert(F.convert([x**2, 2, 0]))

    assert M.convert([x**2 + 1, 2*x + 2, x**2]) == e + [0, x, 0] == \
        e + M.convert([0, x, 0]) == e + F.convert([0, x, 0])
    assert M.convert([x**2 + 1, 2, x**2]) == e - [0, x, 0] == \
        e - M.convert([0, x, 0]) == e - F.convert([0, x, 0])
    assert M.convert([0, 2, 0]) == M.convert([x**2, 4, 0]) - e == \
        [x**2, 4, 0] - e == F.convert([x**2, 4, 0]) - e
    assert M.convert([x**3 + x**2, 2*x + 2, 0]) == (1 + x)*e == \
        R.convert(1 + x)*e == e*(1 + x) == e*R.convert(1 + x)
    assert -e == [-x**2, -2, 0]

    f = [x, x, 0] + N
    assert M.convert([1, 1, 0]) == f / x == f / R.convert(x)

    M2 = F/[(2, 2*x, 2*x**2), (0, 0, 1)]
    G = R.free_module(2)
    M3 = G/[[1, x]]
    M4 = F.submodule([1, x, x**2], [1, 0, 0]) / N
    raises(CoercionFailed, lambda: M.convert(G.convert([1, x])))
    raises(CoercionFailed, lambda: M.convert(M3.convert([1, x])))
    raises(CoercionFailed, lambda: M.convert(M2.convert([1, x, x])))
    assert M2.convert(M.convert([2, x, x**2])) == [2, x, 0]
    assert M.convert(M4.convert([2, 0, 0])) == [2, 0, 0]


def test_QuotientModule():
    R = QQ.old_poly_ring(x)
    F = R.free_module(3)
    N = F.submodule([1, x, x**2])
    M = F/N

    assert M != F
    assert M != N
    assert M == F / [(1, x, x**2)]
    assert not M.is_zero()
    assert (F / F.basis()).is_zero()

    SQ = F.submodule([1, x, x**2], [2, 0, 0]) / N
    assert SQ == M.submodule([2, x, x**2])
    assert SQ != M.submodule([2, 1, 0])
    assert SQ != M
    assert M.is_submodule(SQ)
    assert not SQ.is_full_module()

    raises(ValueError, lambda: N/F)
    raises(ValueError, lambda: F.submodule([2, 0, 0]) / N)
    raises(ValueError, lambda: R.free_module(2)/F)
    raises(CoercionFailed, lambda: F.convert(M.convert([1, x, x**2])))

    M1 = F / [[1, 1, 1]]
    M2 = M1.submodule([1, 0, 0], [0, 1, 0])
    assert M1 == M2


def test_ModulesQuotientRing():
    R = QQ.old_poly_ring(x, y, order=(("lex", x), ("ilex", y))) / [x**2 + 1]
    M1 = R.free_module(2)
    assert M1 == R.free_module(2)
    assert M1 != QQ.old_poly_ring(x).free_module(2)
    assert M1 != R.free_module(3)

    assert [x, 1] in M1
    assert [x] not in M1
    assert [1/(R.convert(x) + 1), 2] in M1
    assert [1, 2/(1 + y)] in M1
    assert [1, 2/y] not in M1

    assert M1.convert([x**2, y]) == [-1, y]

    F = R.free_module(3)
    Fd = F.submodule([x**2, 0, 0], [1, 2, 0], [1, 2, 3])
    M = F.submodule([x**2 + y**2, 1, 0], [x, y, 1])

    assert F == Fd
    assert Fd == F
    assert F != M
    assert M != F
    assert Fd != M
    assert M != Fd
    assert Fd == F.submodule(*F.basis())

    assert Fd.is_full_module()
    assert not M.is_full_module()
    assert not Fd.is_zero()
    assert not M.is_zero()
    assert Fd.submodule().is_zero()

    assert M.contains([x**2 + y**2 + x, -x**2 + y, 1])
    assert not M.contains([x**2 + y**2 + x, 1 + y, 2])
    assert M.contains([y**2, 1 - x*y, -x])

    assert F.submodule([x, 0, 0]) == F.submodule([1, 0, 0])
    assert not F.submodule([y, 0, 0]) == F.submodule([1, 0, 0])
    assert F.submodule([1, 0, 0], [0, 1, 0]).union(F.submodule([0, 0, 1])) == F
    assert not M.is_submodule(0)


def test_module_mul():
    R = QQ.old_poly_ring(x)
    M = R.free_module(2)
    S1 = M.submodule([x, 0], [0, x])
    S2 = M.submodule([x**2, 0], [0, x**2])
    I = R.ideal(x)

    assert I*M == M*I == S1 == x*M == M*x
    assert I*S1 == S2 == x*S1


def test_intersection():
    # SCA, example 2.8.5
    F = QQ.old_poly_ring(x, y).free_module(2)
    M1 = F.submodule([x, y], [y, 1])
    M2 = F.submodule([0, y - 1], [x, 1], [y, x])
    I = F.submodule([x, y], [y**2 - y, y - 1], [x*y + y, x + 1])
    I1, rel1, rel2 = M1.intersect(M2, relations=True)
    assert I1 == M2.intersect(M1) == I
    for i, g in enumerate(I1.gens):
        assert g == sum(c*x for c, x in zip(rel1[i], M1.gens)) \
                 == sum(d*y for d, y in zip(rel2[i], M2.gens))

    assert F.submodule([x, y]).intersect(F.submodule([y, x])).is_zero()


def test_quotient():
    # SCA, example 2.8.6
    R = QQ.old_poly_ring(x, y, z)
    F = R.free_module(2)
    assert F.submodule([x*y, x*z], [y*z, x*y]).module_quotient(
        F.submodule([y, z], [z, y])) == QQ.old_poly_ring(x, y, z).ideal(x**2*y**2 - x*y*z**2)
    assert F.submodule([x, y]).module_quotient(F.submodule()).is_whole_ring()

    M = F.submodule([x**2, x**2], [y**2, y**2])
    N = F.submodule([x + y, x + y])
    q, rel = M.module_quotient(N, relations=True)
    assert q == R.ideal(y**2, x - y)
    for i, g in enumerate(q.gens):
        assert g*N.gens[0] == sum(c*x for c, x in zip(rel[i], M.gens))


def test_groebner_extendend():
    M = QQ.old_poly_ring(x, y, z).free_module(3).submodule([x + 1, y, 1], [x*y, z, z**2])
    G, R = M._groebner_vec(extended=True)
    for i, g in enumerate(G):
        assert g == sum(c*gen for c, gen in zip(R[i], M.gens))
