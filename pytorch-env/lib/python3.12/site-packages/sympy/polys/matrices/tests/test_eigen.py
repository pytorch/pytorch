"""
Tests for the sympy.polys.matrices.eigen module
"""

from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix

from sympy.polys.agca.extensions import FiniteExtension
from sympy.polys.domains import QQ
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import CRootOf
from sympy.polys.matrices.domainmatrix import DomainMatrix

from sympy.polys.matrices.eigen import dom_eigenvects, dom_eigenvects_to_sympy


def test_dom_eigenvects_rational():
    # Rational eigenvalues
    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(1), QQ(2)]], (2, 2), QQ)
    rational_eigenvects = [
        (QQ, QQ(3), 1, DomainMatrix([[QQ(1), QQ(1)]], (1, 2), QQ)),
        (QQ, QQ(0), 1, DomainMatrix([[QQ(-2), QQ(1)]], (1, 2), QQ)),
    ]
    assert dom_eigenvects(A) == (rational_eigenvects, [])

    # Test converting to Expr:
    sympy_eigenvects = [
        (S(3), 1, [Matrix([1, 1])]),
        (S(0), 1, [Matrix([-2, 1])]),
    ]
    assert dom_eigenvects_to_sympy(rational_eigenvects, [], Matrix) == sympy_eigenvects


def test_dom_eigenvects_algebraic():
    # Algebraic eigenvalues
    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    Avects = dom_eigenvects(A)

    # Extract the dummy to build the expected result:
    lamda = Avects[1][0][1].gens[0]
    irreducible = Poly(lamda**2 - 5*lamda - 2, lamda, domain=QQ)
    K = FiniteExtension(irreducible)
    KK = K.from_sympy
    algebraic_eigenvects = [
        (K, irreducible, 1, DomainMatrix([[KK((lamda-4)/3), KK(1)]], (1, 2), K)),
    ]
    assert Avects == ([], algebraic_eigenvects)

    # Test converting to Expr:
    sympy_eigenvects = [
        (S(5)/2 - sqrt(33)/2, 1, [Matrix([[-sqrt(33)/6 - S(1)/2], [1]])]),
        (S(5)/2 + sqrt(33)/2, 1, [Matrix([[-S(1)/2 + sqrt(33)/6], [1]])]),
    ]
    assert dom_eigenvects_to_sympy([], algebraic_eigenvects, Matrix) == sympy_eigenvects


def test_dom_eigenvects_rootof():
    # Algebraic eigenvalues
    A = DomainMatrix([
        [0, 0, 0, 0, -1],
        [1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0]], (5, 5), QQ)
    Avects = dom_eigenvects(A)

    # Extract the dummy to build the expected result:
    lamda = Avects[1][0][1].gens[0]
    irreducible = Poly(lamda**5 - lamda + 1, lamda, domain=QQ)
    K = FiniteExtension(irreducible)
    KK = K.from_sympy
    algebraic_eigenvects = [
        (K, irreducible, 1,
            DomainMatrix([
                [KK(lamda**4-1), KK(lamda**3), KK(lamda**2), KK(lamda), KK(1)]
            ], (1, 5), K)),
    ]
    assert Avects == ([], algebraic_eigenvects)

    # Test converting to Expr (slow):
    l0, l1, l2, l3, l4 = [CRootOf(lamda**5 - lamda + 1, i) for i in range(5)]
    sympy_eigenvects = [
        (l0, 1, [Matrix([-1 + l0**4, l0**3, l0**2, l0, 1])]),
        (l1, 1, [Matrix([-1 + l1**4, l1**3, l1**2, l1, 1])]),
        (l2, 1, [Matrix([-1 + l2**4, l2**3, l2**2, l2, 1])]),
        (l3, 1, [Matrix([-1 + l3**4, l3**3, l3**2, l3, 1])]),
        (l4, 1, [Matrix([-1 + l4**4, l4**3, l4**2, l4, 1])]),
    ]
    assert dom_eigenvects_to_sympy([], algebraic_eigenvects, Matrix) == sympy_eigenvects
