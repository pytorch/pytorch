from sympy.combinatorics import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.homomorphisms import homomorphism, group_isomorphism, is_isomorphic
from sympy.combinatorics.free_groups import free_group
from sympy.combinatorics.fp_groups import FpGroup
from sympy.combinatorics.named_groups import AlternatingGroup, DihedralGroup, CyclicGroup
from sympy.testing.pytest import raises

def test_homomorphism():
    # FpGroup -> PermutationGroup
    F, a, b = free_group("a, b")
    G = FpGroup(F, [a**3, b**3, (a*b)**2])

    c = Permutation(3)(0, 1, 2)
    d = Permutation(3)(1, 2, 3)
    A = AlternatingGroup(4)
    T = homomorphism(G, A, [a, b], [c, d])
    assert T(a*b**2*a**-1) == c*d**2*c**-1
    assert T.is_isomorphism()
    assert T(T.invert(Permutation(3)(0, 2, 3))) == Permutation(3)(0, 2, 3)

    T = homomorphism(G, AlternatingGroup(4), G.generators)
    assert T.is_trivial()
    assert T.kernel().order() == G.order()

    E, e = free_group("e")
    G = FpGroup(E, [e**8])
    P = PermutationGroup([Permutation(0, 1, 2, 3), Permutation(0, 2)])
    T = homomorphism(G, P, [e], [Permutation(0, 1, 2, 3)])
    assert T.image().order() == 4
    assert T(T.invert(Permutation(0, 2)(1, 3))) == Permutation(0, 2)(1, 3)

    T = homomorphism(E, AlternatingGroup(4), E.generators, [c])
    assert T.invert(c**2) == e**-1 #order(c) == 3 so c**2 == c**-1

    # FreeGroup -> FreeGroup
    T = homomorphism(F, E, [a], [e])
    assert T(a**-2*b**4*a**2).is_identity

    # FreeGroup -> FpGroup
    G = FpGroup(F, [a*b*a**-1*b**-1])
    T = homomorphism(F, G, F.generators, G.generators)
    assert T.invert(a**-1*b**-1*a**2) == a*b**-1

    # PermutationGroup -> PermutationGroup
    D = DihedralGroup(8)
    p = Permutation(0, 1, 2, 3, 4, 5, 6, 7)
    P = PermutationGroup(p)
    T = homomorphism(P, D, [p], [p])
    assert T.is_injective()
    assert not T.is_isomorphism()
    assert T.invert(p**3) == p**3

    T2 = homomorphism(F, P, [F.generators[0]], P.generators)
    T = T.compose(T2)
    assert T.domain == F
    assert T.codomain == D
    assert T(a*b) == p

    D3 = DihedralGroup(3)
    T = homomorphism(D3, D3, D3.generators, D3.generators)
    assert T.is_isomorphism()


def test_isomorphisms():

    F, a, b = free_group("a, b")
    E, c, d = free_group("c, d")
    # Infinite groups with differently ordered relators.
    G = FpGroup(F, [a**2, b**3])
    H = FpGroup(F, [b**3, a**2])
    assert is_isomorphic(G, H)

    # Trivial Case
    # FpGroup -> FpGroup
    H = FpGroup(F, [a**3, b**3, (a*b)**2])
    F, c, d = free_group("c, d")
    G = FpGroup(F, [c**3, d**3, (c*d)**2])
    check, T =  group_isomorphism(G, H)
    assert check
    assert T(c**3*d**2) == a**3*b**2

    # FpGroup -> PermutationGroup
    # FpGroup is converted to the equivalent isomorphic group.
    F, a, b = free_group("a, b")
    G = FpGroup(F, [a**3, b**3, (a*b)**2])
    H = AlternatingGroup(4)
    check, T = group_isomorphism(G, H)
    assert check
    assert T(b*a*b**-1*a**-1*b**-1) == Permutation(0, 2, 3)
    assert T(b*a*b*a**-1*b**-1) == Permutation(0, 3, 2)

    # PermutationGroup -> PermutationGroup
    D = DihedralGroup(8)
    p = Permutation(0, 1, 2, 3, 4, 5, 6, 7)
    P = PermutationGroup(p)
    assert not is_isomorphic(D, P)

    A = CyclicGroup(5)
    B = CyclicGroup(7)
    assert not is_isomorphic(A, B)

    # Two groups of the same prime order are isomorphic to each other.
    G = FpGroup(F, [a, b**5])
    H = CyclicGroup(5)
    assert G.order() == H.order()
    assert is_isomorphic(G, H)


def test_check_homomorphism():
    a = Permutation(1,2,3,4)
    b = Permutation(1,3)
    G = PermutationGroup([a, b])
    raises(ValueError, lambda: homomorphism(G, G, [a], [a]))
