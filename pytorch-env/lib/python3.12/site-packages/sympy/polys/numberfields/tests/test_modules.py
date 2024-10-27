from sympy.abc import x, zeta
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.numberfields.exceptions import (
    ClosureFailure, MissingUnityError, StructureError
)
from sympy.polys.numberfields.modules import (
    Module, ModuleElement, ModuleEndomorphism, PowerBasis, PowerBasisElement,
    find_min_poly, is_sq_maxrank_HNF, make_mod_elt, to_col,
)
from sympy.polys.numberfields.utilities import is_int
from sympy.polys.polyerrors import UnificationFailed
from sympy.testing.pytest import raises


def test_to_col():
    c = [1, 2, 3, 4]
    m = to_col(c)
    assert m.domain.is_ZZ
    assert m.shape == (4, 1)
    assert m.flat() == c


def test_Module_NotImplemented():
    M = Module()
    raises(NotImplementedError, lambda: M.n)
    raises(NotImplementedError, lambda: M.mult_tab())
    raises(NotImplementedError, lambda: M.represent(None))
    raises(NotImplementedError, lambda: M.starts_with_unity())
    raises(NotImplementedError, lambda: M.element_from_rational(QQ(2, 3)))


def test_Module_ancestors():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    C = B.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    D = B.submodule_from_matrix(5 * DomainMatrix.eye(4, ZZ))
    assert C.ancestors(include_self=True) == [A, B, C]
    assert D.ancestors(include_self=True) == [A, B, D]
    assert C.power_basis_ancestor() == A
    assert C.nearest_common_ancestor(D) == B
    M = Module()
    assert M.power_basis_ancestor() is None


def test_Module_compat_col():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    col = to_col([1, 2, 3, 4])
    row = col.transpose()
    assert A.is_compat_col(col) is True
    assert A.is_compat_col(row) is False
    assert A.is_compat_col(1) is False
    assert A.is_compat_col(DomainMatrix.eye(3, ZZ)[:, 0]) is False
    assert A.is_compat_col(DomainMatrix.eye(4, QQ)[:, 0]) is False
    assert A.is_compat_col(DomainMatrix.eye(4, ZZ)[:, 0]) is True


def test_Module_call():
    T = Poly(cyclotomic_poly(5, x))
    B = PowerBasis(T)
    assert B(0).col.flat() == [1, 0, 0, 0]
    assert B(1).col.flat() == [0, 1, 0, 0]
    col = DomainMatrix.eye(4, ZZ)[:, 2]
    assert B(col).col == col
    raises(ValueError, lambda: B(-1))


def test_Module_starts_with_unity():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    assert A.starts_with_unity() is True
    assert B.starts_with_unity() is False


def test_Module_basis_elements():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    basis = B.basis_elements()
    bp = B.basis_element_pullbacks()
    for i, (e, p) in enumerate(zip(basis, bp)):
        c = [0] * 4
        assert e.module == B
        assert p.module == A
        c[i] = 1
        assert e == B(to_col(c))
        c[i] = 2
        assert p == A(to_col(c))


def test_Module_zero():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    assert A.zero().col.flat() == [0, 0, 0, 0]
    assert A.zero().module == A
    assert B.zero().col.flat() == [0, 0, 0, 0]
    assert B.zero().module == B


def test_Module_one():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    assert A.one().col.flat() == [1, 0, 0, 0]
    assert A.one().module == A
    assert B.one().col.flat() == [1, 0, 0, 0]
    assert B.one().module == A


def test_Module_element_from_rational():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    rA = A.element_from_rational(QQ(22, 7))
    rB = B.element_from_rational(QQ(22, 7))
    assert rA.coeffs == [22, 0, 0, 0]
    assert rA.denom == 7
    assert rA.module == A
    assert rB.coeffs == [22, 0, 0, 0]
    assert rB.denom == 7
    assert rB.module == A


def test_Module_submodule_from_gens():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    gens = [2*A(0), 2*A(1), 6*A(0), 6*A(1)]
    B = A.submodule_from_gens(gens)
    # Because the 3rd and 4th generators do not add anything new, we expect
    # the cols of the matrix of B to just reproduce the first two gens:
    M = gens[0].column().hstack(gens[1].column())
    assert B.matrix == M
    # At least one generator must be provided:
    raises(ValueError, lambda: A.submodule_from_gens([]))
    # All generators must belong to A:
    raises(ValueError, lambda: A.submodule_from_gens([3*A(0), B(0)]))


def test_Module_submodule_from_matrix():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    e = B(to_col([1, 2, 3, 4]))
    f = e.to_parent()
    assert f.col.flat() == [2, 4, 6, 8]
    # Matrix must be over ZZ:
    raises(ValueError, lambda: A.submodule_from_matrix(DomainMatrix.eye(4, QQ)))
    # Number of rows of matrix must equal number of generators of module A:
    raises(ValueError, lambda: A.submodule_from_matrix(2 * DomainMatrix.eye(5, ZZ)))


def test_Module_whole_submodule():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.whole_submodule()
    e = B(to_col([1, 2, 3, 4]))
    f = e.to_parent()
    assert f.col.flat() == [1, 2, 3, 4]
    e0, e1, e2, e3 = B(0), B(1), B(2), B(3)
    assert e2 * e3 == e0
    assert e3 ** 2 == e1


def test_PowerBasis_repr():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    assert repr(A) == 'PowerBasis(x**4 + x**3 + x**2 + x + 1)'


def test_PowerBasis_eq():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = PowerBasis(T)
    assert A == B


def test_PowerBasis_mult_tab():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    M = A.mult_tab()
    exp = {0: {0: [1, 0, 0, 0], 1: [0, 1, 0, 0], 2: [0, 0, 1, 0], 3: [0, 0, 0, 1]},
           1: {1: [0, 0, 1, 0], 2: [0, 0, 0, 1], 3: [-1, -1, -1, -1]},
           2: {2: [-1, -1, -1, -1], 3: [1, 0, 0, 0]},
           3: {3: [0, 1, 0, 0]}}
    # We get the table we expect:
    assert M == exp
    # And all entries are of expected type:
    assert all(is_int(c) for u in M for v in M[u] for c in M[u][v])


def test_PowerBasis_represent():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    col = to_col([1, 2, 3, 4])
    a = A(col)
    assert A.represent(a) == col
    b = A(col, denom=2)
    raises(ClosureFailure, lambda: A.represent(b))


def test_PowerBasis_element_from_poly():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    f = Poly(1 + 2*x)
    g = Poly(x**4)
    h = Poly(0, x)
    assert A.element_from_poly(f).coeffs == [1, 2, 0, 0]
    assert A.element_from_poly(g).coeffs == [-1, -1, -1, -1]
    assert A.element_from_poly(h).coeffs == [0, 0, 0, 0]


def test_PowerBasis_element__conversions():
    k = QQ.cyclotomic_field(5)
    L = QQ.cyclotomic_field(7)
    B = PowerBasis(k)

    # ANP --> PowerBasisElement
    a = k([QQ(1, 2), QQ(1, 3), 5, 7])
    e = B.element_from_ANP(a)
    assert e.coeffs == [42, 30, 2, 3]
    assert e.denom == 6

    # PowerBasisElement --> ANP
    assert e.to_ANP() == a

    # Cannot convert ANP from different field
    d = L([QQ(1, 2), QQ(1, 3), 5, 7])
    raises(UnificationFailed, lambda: B.element_from_ANP(d))

    # AlgebraicNumber --> PowerBasisElement
    alpha = k.to_alg_num(a)
    eps = B.element_from_alg_num(alpha)
    assert eps.coeffs == [42, 30, 2, 3]
    assert eps.denom == 6

    # PowerBasisElement --> AlgebraicNumber
    assert eps.to_alg_num() == alpha

    # Cannot convert AlgebraicNumber from different field
    delta = L.to_alg_num(d)
    raises(UnificationFailed, lambda: B.element_from_alg_num(delta))

    # When we don't know the field:
    C = PowerBasis(k.ext.minpoly)
    # Can convert from AlgebraicNumber:
    eps = C.element_from_alg_num(alpha)
    assert eps.coeffs == [42, 30, 2, 3]
    assert eps.denom == 6
    # But can't convert back:
    raises(StructureError, lambda: eps.to_alg_num())


def test_Submodule_repr():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ), denom=3)
    assert repr(B) == 'Submodule[[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]]/3'


def test_Submodule_reduced():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    C = A.submodule_from_matrix(6 * DomainMatrix.eye(4, ZZ), denom=3)
    D = C.reduced()
    assert D.denom == 1 and D == C == B


def test_Submodule_discard_before():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    B.compute_mult_tab()
    C = B.discard_before(2)
    assert C.parent == B.parent
    assert B.is_sq_maxrank_HNF() and not C.is_sq_maxrank_HNF()
    assert C.matrix == B.matrix[:, 2:]
    assert C.mult_tab() == {0: {0: [-2, -2], 1: [0, 0]}, 1: {1: [0, 0]}}


def test_Submodule_QQ_matrix():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    C = A.submodule_from_matrix(6 * DomainMatrix.eye(4, ZZ), denom=3)
    assert C.QQ_matrix == B.QQ_matrix


def test_Submodule_represent():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    C = B.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    a0 = A(to_col([6, 12, 18, 24]))
    a1 = A(to_col([2, 4, 6, 8]))
    a2 = A(to_col([1, 3, 5, 7]))

    b1 = B.represent(a1)
    assert b1.flat() == [1, 2, 3, 4]

    c0 = C.represent(a0)
    assert c0.flat() == [1, 2, 3, 4]

    Y = A.submodule_from_matrix(DomainMatrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ], (3, 4), ZZ).transpose())

    U = Poly(cyclotomic_poly(7, x))
    Z = PowerBasis(U)
    z0 = Z(to_col([1, 2, 3, 4, 5, 6]))

    raises(ClosureFailure, lambda: Y.represent(A(3)))
    raises(ClosureFailure, lambda: B.represent(a2))
    raises(ClosureFailure, lambda: B.represent(z0))


def test_Submodule_is_compat_submodule():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    C = A.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    D = C.submodule_from_matrix(5 * DomainMatrix.eye(4, ZZ))
    assert B.is_compat_submodule(C) is True
    assert B.is_compat_submodule(A) is False
    assert B.is_compat_submodule(D) is False


def test_Submodule_eq():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    C = A.submodule_from_matrix(6 * DomainMatrix.eye(4, ZZ), denom=3)
    assert C == B


def test_Submodule_add():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(DomainMatrix([
        [4, 0, 0, 0],
        [0, 4, 0, 0],
    ], (2, 4), ZZ).transpose(), denom=6)
    C = A.submodule_from_matrix(DomainMatrix([
        [0, 10, 0, 0],
        [0,  0, 7, 0],
    ], (2, 4), ZZ).transpose(), denom=15)
    D = A.submodule_from_matrix(DomainMatrix([
        [20,  0,  0, 0],
        [ 0, 20,  0, 0],
        [ 0,  0, 14, 0],
    ], (3, 4), ZZ).transpose(), denom=30)
    assert B + C == D

    U = Poly(cyclotomic_poly(7, x))
    Z = PowerBasis(U)
    Y = Z.submodule_from_gens([Z(0), Z(1)])
    raises(TypeError, lambda: B + Y)


def test_Submodule_mul():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    C = A.submodule_from_matrix(DomainMatrix([
        [0, 10, 0, 0],
        [0, 0, 7, 0],
    ], (2, 4), ZZ).transpose(), denom=15)
    C1 = A.submodule_from_matrix(DomainMatrix([
        [0, 20, 0, 0],
        [0, 0, 14, 0],
    ], (2, 4), ZZ).transpose(), denom=3)
    C2 = A.submodule_from_matrix(DomainMatrix([
        [0, 0, 10, 0],
        [0, 0,  0, 7],
    ], (2, 4), ZZ).transpose(), denom=15)
    C3_unred = A.submodule_from_matrix(DomainMatrix([
        [0, 0, 100, 0],
        [0, 0, 0, 70],
        [0, 0, 0, 70],
        [-49, -49, -49, -49]
    ], (4, 4), ZZ).transpose(), denom=225)
    C3 = A.submodule_from_matrix(DomainMatrix([
        [4900, 4900, 0, 0],
        [4410, 4410, 10, 0],
        [2107, 2107, 7, 7]
    ], (3, 4), ZZ).transpose(), denom=225)
    assert C * 1 == C
    assert C ** 1 == C
    assert C * 10 == C1
    assert C * A(1) == C2
    assert C.mul(C, hnf=False) == C3_unred
    assert C * C == C3
    assert C ** 2 == C3


def test_Submodule_reduce_element():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.whole_submodule()
    b = B(to_col([90, 84, 80, 75]), denom=120)

    C = B.submodule_from_matrix(DomainMatrix.eye(4, ZZ), denom=2)
    b_bar_expected = B(to_col([30, 24, 20, 15]), denom=120)
    b_bar = C.reduce_element(b)
    assert b_bar == b_bar_expected

    C = B.submodule_from_matrix(DomainMatrix.eye(4, ZZ), denom=4)
    b_bar_expected = B(to_col([0, 24, 20, 15]), denom=120)
    b_bar = C.reduce_element(b)
    assert b_bar == b_bar_expected

    C = B.submodule_from_matrix(DomainMatrix.eye(4, ZZ), denom=8)
    b_bar_expected = B(to_col([0, 9, 5, 0]), denom=120)
    b_bar = C.reduce_element(b)
    assert b_bar == b_bar_expected

    a = A(to_col([1, 2, 3, 4]))
    raises(NotImplementedError, lambda: C.reduce_element(a))

    C = B.submodule_from_matrix(DomainMatrix([
        [5, 4, 3, 2],
        [0, 8, 7, 6],
        [0, 0,11,12],
        [0, 0, 0, 1]
    ], (4, 4), ZZ).transpose())
    raises(StructureError, lambda: C.reduce_element(b))


def test_is_HNF():
    M = DM([
        [3, 2, 1],
        [0, 2, 1],
        [0, 0, 1]
    ], ZZ)
    M1 = DM([
        [3, 2, 1],
        [0, -2, 1],
        [0, 0, 1]
    ], ZZ)
    M2 = DM([
        [3, 2, 3],
        [0, 2, 1],
        [0, 0, 1]
    ], ZZ)
    assert is_sq_maxrank_HNF(M) is True
    assert is_sq_maxrank_HNF(M1) is False
    assert is_sq_maxrank_HNF(M2) is False


def test_make_mod_elt():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    col = to_col([1, 2, 3, 4])
    eA = make_mod_elt(A, col)
    eB = make_mod_elt(B, col)
    assert isinstance(eA, PowerBasisElement)
    assert not isinstance(eB, PowerBasisElement)


def test_ModuleElement_repr():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    e = A(to_col([1, 2, 3, 4]), denom=2)
    assert repr(e) == '[1, 2, 3, 4]/2'


def test_ModuleElement_reduced():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    e = A(to_col([2, 4, 6, 8]), denom=2)
    f = e.reduced()
    assert f.denom == 1 and f == e


def test_ModuleElement_reduced_mod_p():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    e = A(to_col([20, 40, 60, 80]))
    f = e.reduced_mod_p(7)
    assert f.coeffs == [-1, -2, -3, 3]


def test_ModuleElement_from_int_list():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    c = [1, 2, 3, 4]
    assert ModuleElement.from_int_list(A, c).coeffs == c


def test_ModuleElement_len():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    e = A(0)
    assert len(e) == 4


def test_ModuleElement_column():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    e = A(0)
    col1 = e.column()
    assert col1 == e.col and col1 is not e.col
    col2 = e.column(domain=FF(5))
    assert col2.domain.is_FF


def test_ModuleElement_QQ_col():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    e = A(to_col([1, 2, 3, 4]), denom=1)
    f = A(to_col([3, 6, 9, 12]), denom=3)
    assert e.QQ_col == f.QQ_col


def test_ModuleElement_to_ancestors():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    C = B.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    D = C.submodule_from_matrix(5 * DomainMatrix.eye(4, ZZ))
    eD = D(0)
    eC = eD.to_parent()
    eB = eD.to_ancestor(B)
    eA = eD.over_power_basis()
    assert eC.module is C and eC.coeffs == [5, 0, 0, 0]
    assert eB.module is B and eB.coeffs == [15, 0, 0, 0]
    assert eA.module is A and eA.coeffs == [30, 0, 0, 0]

    a = A(0)
    raises(ValueError, lambda: a.to_parent())


def test_ModuleElement_compatibility():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    C = B.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    D = B.submodule_from_matrix(5 * DomainMatrix.eye(4, ZZ))
    assert C(0).is_compat(C(1)) is True
    assert C(0).is_compat(D(0)) is False
    u, v = C(0).unify(D(0))
    assert u.module is B and v.module is B
    assert C(C.represent(u)) == C(0) and D(D.represent(v)) == D(0)

    u, v = C(0).unify(C(1))
    assert u == C(0) and v == C(1)

    U = Poly(cyclotomic_poly(7, x))
    Z = PowerBasis(U)
    raises(UnificationFailed, lambda: C(0).unify(Z(1)))


def test_ModuleElement_eq():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    e = A(to_col([1, 2, 3, 4]), denom=1)
    f = A(to_col([3, 6, 9, 12]), denom=3)
    assert e == f

    U = Poly(cyclotomic_poly(7, x))
    Z = PowerBasis(U)
    assert e != Z(0)
    assert e != 3.14


def test_ModuleElement_equiv():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    e = A(to_col([1, 2, 3, 4]), denom=1)
    f = A(to_col([3, 6, 9, 12]), denom=3)
    assert e.equiv(f)

    C = A.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    g = C(to_col([1, 2, 3, 4]), denom=1)
    h = A(to_col([3, 6, 9, 12]), denom=1)
    assert g.equiv(h)
    assert C(to_col([5, 0, 0, 0]), denom=7).equiv(QQ(15, 7))

    U = Poly(cyclotomic_poly(7, x))
    Z = PowerBasis(U)
    raises(UnificationFailed, lambda: e.equiv(Z(0)))

    assert e.equiv(3.14) is False


def test_ModuleElement_add():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    C = A.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    e = A(to_col([1, 2, 3, 4]), denom=6)
    f = A(to_col([5, 6, 7, 8]), denom=10)
    g = C(to_col([1, 1, 1, 1]), denom=2)
    assert e + f == A(to_col([10, 14, 18, 22]), denom=15)
    assert e - f == A(to_col([-5, -4, -3, -2]), denom=15)
    assert e + g == A(to_col([10, 11, 12, 13]), denom=6)
    assert e + QQ(7, 10) == A(to_col([26, 10, 15, 20]), denom=30)
    assert g + QQ(7, 10) == A(to_col([22, 15, 15, 15]), denom=10)

    U = Poly(cyclotomic_poly(7, x))
    Z = PowerBasis(U)
    raises(TypeError, lambda: e + Z(0))
    raises(TypeError, lambda: e + 3.14)


def test_ModuleElement_mul():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    C = A.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    e = A(to_col([0, 2, 0, 0]), denom=3)
    f = A(to_col([0, 0, 0, 7]), denom=5)
    g = C(to_col([0, 0, 0, 1]), denom=2)
    h = A(to_col([0, 0, 3, 1]), denom=7)
    assert e * f == A(to_col([-14, -14, -14, -14]), denom=15)
    assert e * g == A(to_col([-1, -1, -1, -1]))
    assert e * h == A(to_col([-2, -2, -2, 4]), denom=21)
    assert e * QQ(6, 5) == A(to_col([0, 4, 0, 0]), denom=5)
    assert (g * QQ(10, 21)).equiv(A(to_col([0, 0, 0, 5]), denom=7))
    assert e // QQ(6, 5) == A(to_col([0, 5, 0, 0]), denom=9)

    U = Poly(cyclotomic_poly(7, x))
    Z = PowerBasis(U)
    raises(TypeError, lambda: e * Z(0))
    raises(TypeError, lambda: e * 3.14)
    raises(TypeError, lambda: e // 3.14)
    raises(ZeroDivisionError, lambda: e // 0)


def test_ModuleElement_div():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    C = A.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    e = A(to_col([0, 2, 0, 0]), denom=3)
    f = A(to_col([0, 0, 0, 7]), denom=5)
    g = C(to_col([1, 1, 1, 1]))
    assert e // f == 10*A(3)//21
    assert e // g == -2*A(2)//9
    assert 3 // g == -A(1)


def test_ModuleElement_pow():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    C = A.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    e = A(to_col([0, 2, 0, 0]), denom=3)
    g = C(to_col([0, 0, 0, 1]), denom=2)
    assert e ** 3 == A(to_col([0, 0, 0, 8]), denom=27)
    assert g ** 2 == C(to_col([0, 3, 0, 0]), denom=4)
    assert e ** 0 == A(to_col([1, 0, 0, 0]))
    assert g ** 0 == A(to_col([1, 0, 0, 0]))
    assert e ** 1 == e
    assert g ** 1 == g


def test_ModuleElement_mod():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    e = A(to_col([1, 15, 8, 0]), denom=2)
    assert e % 7 == A(to_col([1, 1, 8, 0]), denom=2)
    assert e % QQ(1, 2) == A.zero()
    assert e % QQ(1, 3) == A(to_col([1, 1, 0, 0]), denom=6)

    B = A.submodule_from_gens([A(0), 5*A(1), 3*A(2), A(3)])
    assert e % B == A(to_col([1, 5, 2, 0]), denom=2)

    C = B.whole_submodule()
    raises(TypeError, lambda: e % C)


def test_PowerBasisElement_polys():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    e = A(to_col([1, 15, 8, 0]), denom=2)
    assert e.numerator(x=zeta) == Poly(8 * zeta ** 2 + 15 * zeta + 1, domain=ZZ)
    assert e.poly(x=zeta) == Poly(4 * zeta ** 2 + QQ(15, 2) * zeta + QQ(1, 2), domain=QQ)


def test_PowerBasisElement_norm():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    lam = A(to_col([1, -1, 0, 0]))
    assert lam.norm() == 5


def test_PowerBasisElement_inverse():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    e = A(to_col([1, 1, 1, 1]))
    assert 2 // e == -2*A(1)
    assert e ** -3 == -A(3)


def test_ModuleHomomorphism_matrix():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    phi = ModuleEndomorphism(A, lambda a: a ** 2)
    M = phi.matrix()
    assert M == DomainMatrix([
        [1, 0, -1, 0],
        [0, 0, -1, 1],
        [0, 1, -1, 0],
        [0, 0, -1, 0]
    ], (4, 4), ZZ)


def test_ModuleHomomorphism_kernel():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    phi = ModuleEndomorphism(A, lambda a: a ** 5)
    N = phi.kernel()
    assert N.n == 3


def test_EndomorphismRing_represent():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    R = A.endomorphism_ring()
    phi = R.inner_endomorphism(A(1))
    col = R.represent(phi)
    assert col.transpose() == DomainMatrix([
        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -1, -1, -1, -1]
    ], (1, 16), ZZ)

    B = A.submodule_from_matrix(DomainMatrix.zeros((4, 0), ZZ))
    S = B.endomorphism_ring()
    psi = S.inner_endomorphism(A(1))
    col = S.represent(psi)
    assert col == DomainMatrix([], (0, 0), ZZ)

    raises(NotImplementedError, lambda: R.represent(3.14))


def test_find_min_poly():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    powers = []
    m = find_min_poly(A(1), QQ, x=x, powers=powers)
    assert m == Poly(T, domain=QQ)
    assert len(powers) == 5

    # powers list need not be passed
    m = find_min_poly(A(1), QQ, x=x)
    assert m == Poly(T, domain=QQ)

    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    raises(MissingUnityError, lambda: find_min_poly(B(1), QQ))
