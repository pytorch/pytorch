from sympy.core.containers import Tuple
from sympy.combinatorics.generators import rubik_cube_generators
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup,\
    DihedralGroup, AlternatingGroup, AbelianGroup, RubikGroup
from sympy.combinatorics.perm_groups import (PermutationGroup,
    _orbit_transversal, Coset, SymmetricPermutationGroup)
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.polyhedron import tetrahedron as Tetra, cube
from sympy.combinatorics.testutil import _verify_bsgs, _verify_centralizer,\
    _verify_normal_closure
from sympy.testing.pytest import skip, XFAIL, slow

rmul = Permutation.rmul


def test_has():
    a = Permutation([1, 0])
    G = PermutationGroup([a])
    assert G.is_abelian
    a = Permutation([2, 0, 1])
    b = Permutation([2, 1, 0])
    G = PermutationGroup([a, b])
    assert not G.is_abelian

    G = PermutationGroup([a])
    assert G.has(a)
    assert not G.has(b)

    a = Permutation([2, 0, 1, 3, 4, 5])
    b = Permutation([0, 2, 1, 3, 4])
    assert PermutationGroup(a, b).degree == \
        PermutationGroup(a, b).degree == 6

    g = PermutationGroup(Permutation(0, 2, 1))
    assert Tuple(1, g).has(g)


def test_generate():
    a = Permutation([1, 0])
    g = list(PermutationGroup([a]).generate())
    assert g == [Permutation([0, 1]), Permutation([1, 0])]
    assert len(list(PermutationGroup(Permutation((0, 1))).generate())) == 1
    g = PermutationGroup([a]).generate(method='dimino')
    assert list(g) == [Permutation([0, 1]), Permutation([1, 0])]
    a = Permutation([2, 0, 1])
    b = Permutation([2, 1, 0])
    G = PermutationGroup([a, b])
    g = G.generate()
    v1 = [p.array_form for p in list(g)]
    v1.sort()
    assert v1 == [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0,
        1], [2, 1, 0]]
    v2 = list(G.generate(method='dimino', af=True))
    assert v1 == sorted(v2)
    a = Permutation([2, 0, 1, 3, 4, 5])
    b = Permutation([2, 1, 3, 4, 5, 0])
    g = PermutationGroup([a, b]).generate(af=True)
    assert len(list(g)) == 360


def test_order():
    a = Permutation([2, 0, 1, 3, 4, 5, 6, 7, 8, 9])
    b = Permutation([2, 1, 3, 4, 5, 6, 7, 8, 9, 0])
    g = PermutationGroup([a, b])
    assert g.order() == 1814400
    assert PermutationGroup().order() == 1


def test_equality():
    p_1 = Permutation(0, 1, 3)
    p_2 = Permutation(0, 2, 3)
    p_3 = Permutation(0, 1, 2)
    p_4 = Permutation(0, 1, 3)
    g_1 = PermutationGroup(p_1, p_2)
    g_2 = PermutationGroup(p_3, p_4)
    g_3 = PermutationGroup(p_2, p_1)
    g_4 = PermutationGroup(p_1, p_2)

    assert g_1 != g_2
    assert g_1.generators != g_2.generators
    assert g_1.equals(g_2)
    assert g_1 != g_3
    assert g_1.equals(g_3)
    assert g_1 == g_4


def test_stabilizer():
    S = SymmetricGroup(2)
    H = S.stabilizer(0)
    assert H.generators == [Permutation(1)]
    a = Permutation([2, 0, 1, 3, 4, 5])
    b = Permutation([2, 1, 3, 4, 5, 0])
    G = PermutationGroup([a, b])
    G0 = G.stabilizer(0)
    assert G0.order() == 60

    gens_cube = [[1, 3, 5, 7, 0, 2, 4, 6], [1, 3, 0, 2, 5, 7, 4, 6]]
    gens = [Permutation(p) for p in gens_cube]
    G = PermutationGroup(gens)
    G2 = G.stabilizer(2)
    assert G2.order() == 6
    G2_1 = G2.stabilizer(1)
    v = list(G2_1.generate(af=True))
    assert v == [[0, 1, 2, 3, 4, 5, 6, 7], [3, 1, 2, 0, 7, 5, 6, 4]]

    gens = (
        (1, 2, 0, 4, 5, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
        (0, 1, 2, 3, 4, 5, 19, 6, 8, 9, 10, 11, 12, 13, 14,
         15, 16, 7, 17, 18),
        (0, 1, 2, 3, 4, 5, 6, 7, 9, 18, 16, 11, 12, 13, 14, 15, 8, 17, 10, 19))
    gens = [Permutation(p) for p in gens]
    G = PermutationGroup(gens)
    G2 = G.stabilizer(2)
    assert G2.order() == 181440
    S = SymmetricGroup(3)
    assert [G.order() for G in S.basic_stabilizers] == [6, 2]


def test_center():
    # the center of the dihedral group D_n is of order 2 for even n
    for i in (4, 6, 10):
        D = DihedralGroup(i)
        assert (D.center()).order() == 2
    # the center of the dihedral group D_n is of order 1 for odd n>2
    for i in (3, 5, 7):
        D = DihedralGroup(i)
        assert (D.center()).order() == 1
    # the center of an abelian group is the group itself
    for i in (2, 3, 5):
        for j in (1, 5, 7):
            for k in (1, 1, 11):
                G = AbelianGroup(i, j, k)
                assert G.center().is_subgroup(G)
    # the center of a nonabelian simple group is trivial
    for i in(1, 5, 9):
        A = AlternatingGroup(i)
        assert (A.center()).order() == 1
    # brute-force verifications
    D = DihedralGroup(5)
    A = AlternatingGroup(3)
    C = CyclicGroup(4)
    G.is_subgroup(D*A*C)
    assert _verify_centralizer(G, G)


def test_centralizer():
    # the centralizer of the trivial group is the entire group
    S = SymmetricGroup(2)
    assert S.centralizer(Permutation(list(range(2)))).is_subgroup(S)
    A = AlternatingGroup(5)
    assert A.centralizer(Permutation(list(range(5)))).is_subgroup(A)
    # a centralizer in the trivial group is the trivial group itself
    triv = PermutationGroup([Permutation([0, 1, 2, 3])])
    D = DihedralGroup(4)
    assert triv.centralizer(D).is_subgroup(triv)
    # brute-force verifications for centralizers of groups
    for i in (4, 5, 6):
        S = SymmetricGroup(i)
        A = AlternatingGroup(i)
        C = CyclicGroup(i)
        D = DihedralGroup(i)
        for gp in (S, A, C, D):
            for gp2 in (S, A, C, D):
                if not gp2.is_subgroup(gp):
                    assert _verify_centralizer(gp, gp2)
    # verify the centralizer for all elements of several groups
    S = SymmetricGroup(5)
    elements = list(S.generate_dimino())
    for element in elements:
        assert _verify_centralizer(S, element)
    A = AlternatingGroup(5)
    elements = list(A.generate_dimino())
    for element in elements:
        assert _verify_centralizer(A, element)
    D = DihedralGroup(7)
    elements = list(D.generate_dimino())
    for element in elements:
        assert _verify_centralizer(D, element)
    # verify centralizers of small groups within small groups
    small = []
    for i in (1, 2, 3):
        small.append(SymmetricGroup(i))
        small.append(AlternatingGroup(i))
        small.append(DihedralGroup(i))
        small.append(CyclicGroup(i))
    for gp in small:
        for gp2 in small:
            if gp.degree == gp2.degree:
                assert _verify_centralizer(gp, gp2)


def test_coset_rank():
    gens_cube = [[1, 3, 5, 7, 0, 2, 4, 6], [1, 3, 0, 2, 5, 7, 4, 6]]
    gens = [Permutation(p) for p in gens_cube]
    G = PermutationGroup(gens)
    i = 0
    for h in G.generate(af=True):
        rk = G.coset_rank(h)
        assert rk == i
        h1 = G.coset_unrank(rk, af=True)
        assert h == h1
        i += 1
    assert G.coset_unrank(48) is None
    assert G.coset_unrank(G.coset_rank(gens[0])) == gens[0]


def test_coset_factor():
    a = Permutation([0, 2, 1])
    G = PermutationGroup([a])
    c = Permutation([2, 1, 0])
    assert not G.coset_factor(c)
    assert G.coset_rank(c) is None

    a = Permutation([2, 0, 1, 3, 4, 5])
    b = Permutation([2, 1, 3, 4, 5, 0])
    g = PermutationGroup([a, b])
    assert g.order() == 360
    d = Permutation([1, 0, 2, 3, 4, 5])
    assert not g.coset_factor(d.array_form)
    assert not g.contains(d)
    assert Permutation(2) in G
    c = Permutation([1, 0, 2, 3, 5, 4])
    v = g.coset_factor(c, True)
    tr = g.basic_transversals
    p = Permutation.rmul(*[tr[i][v[i]] for i in range(len(g.base))])
    assert p == c
    v = g.coset_factor(c)
    p = Permutation.rmul(*v)
    assert p == c
    assert g.contains(c)
    G = PermutationGroup([Permutation([2, 1, 0])])
    p = Permutation([1, 0, 2])
    assert G.coset_factor(p) == []


def test_orbits():
    a = Permutation([2, 0, 1])
    b = Permutation([2, 1, 0])
    g = PermutationGroup([a, b])
    assert g.orbit(0) == {0, 1, 2}
    assert g.orbits() == [{0, 1, 2}]
    assert g.is_transitive() and g.is_transitive(strict=False)
    assert g.orbit_transversal(0) == \
        [Permutation(
            [0, 1, 2]), Permutation([2, 0, 1]), Permutation([1, 2, 0])]
    assert g.orbit_transversal(0, True) == \
        [(0, Permutation([0, 1, 2])), (2, Permutation([2, 0, 1])),
        (1, Permutation([1, 2, 0]))]

    G = DihedralGroup(6)
    transversal, slps = _orbit_transversal(G.degree, G.generators, 0, True, slp=True)
    for i, t in transversal:
        slp = slps[i]
        w = G.identity
        for s in slp:
            w = G.generators[s]*w
        assert w == t

    a = Permutation(list(range(1, 100)) + [0])
    G = PermutationGroup([a])
    assert [min(o) for o in G.orbits()] == [0]
    G = PermutationGroup(rubik_cube_generators())
    assert [min(o) for o in G.orbits()] == [0, 1]
    assert not G.is_transitive() and not G.is_transitive(strict=False)
    G = PermutationGroup([Permutation(0, 1, 3), Permutation(3)(0, 1)])
    assert not G.is_transitive() and G.is_transitive(strict=False)
    assert PermutationGroup(
        Permutation(3)).is_transitive(strict=False) is False


def test_is_normal():
    gens_s5 = [Permutation(p) for p in [[1, 2, 3, 4, 0], [2, 1, 4, 0, 3]]]
    G1 = PermutationGroup(gens_s5)
    assert G1.order() == 120
    gens_a5 = [Permutation(p) for p in [[1, 0, 3, 2, 4], [2, 1, 4, 3, 0]]]
    G2 = PermutationGroup(gens_a5)
    assert G2.order() == 60
    assert G2.is_normal(G1)
    gens3 = [Permutation(p) for p in [[2, 1, 3, 0, 4], [1, 2, 0, 3, 4]]]
    G3 = PermutationGroup(gens3)
    assert not G3.is_normal(G1)
    assert G3.order() == 12
    G4 = G1.normal_closure(G3.generators)
    assert G4.order() == 60
    gens5 = [Permutation(p) for p in [[1, 2, 3, 0, 4], [1, 2, 0, 3, 4]]]
    G5 = PermutationGroup(gens5)
    assert G5.order() == 24
    G6 = G1.normal_closure(G5.generators)
    assert G6.order() == 120
    assert G1.is_subgroup(G6)
    assert not G1.is_subgroup(G4)
    assert G2.is_subgroup(G4)
    I5 = PermutationGroup(Permutation(4))
    assert I5.is_normal(G5)
    assert I5.is_normal(G6, strict=False)
    p1 = Permutation([1, 0, 2, 3, 4])
    p2 = Permutation([0, 1, 2, 4, 3])
    p3 = Permutation([3, 4, 2, 1, 0])
    id_ = Permutation([0, 1, 2, 3, 4])
    H = PermutationGroup([p1, p3])
    H_n1 = PermutationGroup([p1, p2])
    H_n2_1 = PermutationGroup(p1)
    H_n2_2 = PermutationGroup(p2)
    H_id = PermutationGroup(id_)
    assert H_n1.is_normal(H)
    assert H_n2_1.is_normal(H_n1)
    assert H_n2_2.is_normal(H_n1)
    assert H_id.is_normal(H_n2_1)
    assert H_id.is_normal(H_n1)
    assert H_id.is_normal(H)
    assert not H_n2_1.is_normal(H)
    assert not H_n2_2.is_normal(H)


def test_eq():
    a = [[1, 2, 0, 3, 4, 5], [1, 0, 2, 3, 4, 5], [2, 1, 0, 3, 4, 5], [
        1, 2, 0, 3, 4, 5]]
    a = [Permutation(p) for p in a + [[1, 2, 3, 4, 5, 0]]]
    g = Permutation([1, 2, 3, 4, 5, 0])
    G1, G2, G3 = [PermutationGroup(x) for x in [a[:2], a[2:4], [g, g**2]]]
    assert G1.order() == G2.order() == G3.order() == 6
    assert G1.is_subgroup(G2)
    assert not G1.is_subgroup(G3)
    G4 = PermutationGroup([Permutation([0, 1])])
    assert not G1.is_subgroup(G4)
    assert G4.is_subgroup(G1, 0)
    assert PermutationGroup(g, g).is_subgroup(PermutationGroup(g))
    assert SymmetricGroup(3).is_subgroup(SymmetricGroup(4), 0)
    assert SymmetricGroup(3).is_subgroup(SymmetricGroup(3)*CyclicGroup(5), 0)
    assert not CyclicGroup(5).is_subgroup(SymmetricGroup(3)*CyclicGroup(5), 0)
    assert CyclicGroup(3).is_subgroup(SymmetricGroup(3)*CyclicGroup(5), 0)


def test_derived_subgroup():
    a = Permutation([1, 0, 2, 4, 3])
    b = Permutation([0, 1, 3, 2, 4])
    G = PermutationGroup([a, b])
    C = G.derived_subgroup()
    assert C.order() == 3
    assert C.is_normal(G)
    assert C.is_subgroup(G, 0)
    assert not G.is_subgroup(C, 0)
    gens_cube = [[1, 3, 5, 7, 0, 2, 4, 6], [1, 3, 0, 2, 5, 7, 4, 6]]
    gens = [Permutation(p) for p in gens_cube]
    G = PermutationGroup(gens)
    C = G.derived_subgroup()
    assert C.order() == 12


def test_is_solvable():
    a = Permutation([1, 2, 0])
    b = Permutation([1, 0, 2])
    G = PermutationGroup([a, b])
    assert G.is_solvable
    G = PermutationGroup([a])
    assert G.is_solvable
    a = Permutation([1, 2, 3, 4, 0])
    b = Permutation([1, 0, 2, 3, 4])
    G = PermutationGroup([a, b])
    assert not G.is_solvable
    P = SymmetricGroup(10)
    S = P.sylow_subgroup(3)
    assert S.is_solvable

def test_rubik1():
    gens = rubik_cube_generators()
    gens1 = [gens[-1]] + [p**2 for p in gens[1:]]
    G1 = PermutationGroup(gens1)
    assert G1.order() == 19508428800
    gens2 = [p**2 for p in gens]
    G2 = PermutationGroup(gens2)
    assert G2.order() == 663552
    assert G2.is_subgroup(G1, 0)
    C1 = G1.derived_subgroup()
    assert C1.order() == 4877107200
    assert C1.is_subgroup(G1, 0)
    assert not G2.is_subgroup(C1, 0)

    G = RubikGroup(2)
    assert G.order() == 3674160


@XFAIL
def test_rubik():
    skip('takes too much time')
    G = PermutationGroup(rubik_cube_generators())
    assert G.order() == 43252003274489856000
    G1 = PermutationGroup(G[:3])
    assert G1.order() == 170659735142400
    assert not G1.is_normal(G)
    G2 = G.normal_closure(G1.generators)
    assert G2.is_subgroup(G)


def test_direct_product():
    C = CyclicGroup(4)
    D = DihedralGroup(4)
    G = C*C*C
    assert G.order() == 64
    assert G.degree == 12
    assert len(G.orbits()) == 3
    assert G.is_abelian is True
    H = D*C
    assert H.order() == 32
    assert H.is_abelian is False


def test_orbit_rep():
    G = DihedralGroup(6)
    assert G.orbit_rep(1, 3) in [Permutation([2, 3, 4, 5, 0, 1]),
    Permutation([4, 3, 2, 1, 0, 5])]
    H = CyclicGroup(4)*G
    assert H.orbit_rep(1, 5) is False


def test_schreier_vector():
    G = CyclicGroup(50)
    v = [0]*50
    v[23] = -1
    assert G.schreier_vector(23) == v
    H = DihedralGroup(8)
    assert H.schreier_vector(2) == [0, 1, -1, 0, 0, 1, 0, 0]
    L = SymmetricGroup(4)
    assert L.schreier_vector(1) == [1, -1, 0, 0]


def test_random_pr():
    D = DihedralGroup(6)
    r = 11
    n = 3
    _random_prec_n = {}
    _random_prec_n[0] = {'s': 7, 't': 3, 'x': 2, 'e': -1}
    _random_prec_n[1] = {'s': 5, 't': 5, 'x': 1, 'e': -1}
    _random_prec_n[2] = {'s': 3, 't': 4, 'x': 2, 'e': 1}
    D._random_pr_init(r, n, _random_prec_n=_random_prec_n)
    assert D._random_gens[11] == [0, 1, 2, 3, 4, 5]
    _random_prec = {'s': 2, 't': 9, 'x': 1, 'e': -1}
    assert D.random_pr(_random_prec=_random_prec) == \
        Permutation([0, 5, 4, 3, 2, 1])


def test_is_alt_sym():
    G = DihedralGroup(10)
    assert G.is_alt_sym() is False
    assert G._eval_is_alt_sym_naive() is False
    assert G._eval_is_alt_sym_naive(only_alt=True) is False
    assert G._eval_is_alt_sym_naive(only_sym=True) is False

    S = SymmetricGroup(10)
    assert S._eval_is_alt_sym_naive() is True
    assert S._eval_is_alt_sym_naive(only_alt=True) is False
    assert S._eval_is_alt_sym_naive(only_sym=True) is True

    N_eps = 10
    _random_prec = {'N_eps': N_eps,
        0: Permutation([[2], [1, 4], [0, 6, 7, 8, 9, 3, 5]]),
        1: Permutation([[1, 8, 7, 6, 3, 5, 2, 9], [0, 4]]),
        2: Permutation([[5, 8], [4, 7], [0, 1, 2, 3, 6, 9]]),
        3: Permutation([[3], [0, 8, 2, 7, 4, 1, 6, 9, 5]]),
        4: Permutation([[8], [4, 7, 9], [3, 6], [0, 5, 1, 2]]),
        5: Permutation([[6], [0, 2, 4, 5, 1, 8, 3, 9, 7]]),
        6: Permutation([[6, 9, 8], [4, 5], [1, 3, 7], [0, 2]]),
        7: Permutation([[4], [0, 2, 9, 1, 3, 8, 6, 5, 7]]),
        8: Permutation([[1, 5, 6, 3], [0, 2, 7, 8, 4, 9]]),
        9: Permutation([[8], [6, 7], [2, 3, 4, 5], [0, 1, 9]])}
    assert S.is_alt_sym(_random_prec=_random_prec) is True

    A = AlternatingGroup(10)
    assert A._eval_is_alt_sym_naive() is True
    assert A._eval_is_alt_sym_naive(only_alt=True) is True
    assert A._eval_is_alt_sym_naive(only_sym=True) is False

    _random_prec = {'N_eps': N_eps,
        0: Permutation([[1, 6, 4, 2, 7, 8, 5, 9, 3], [0]]),
        1: Permutation([[1], [0, 5, 8, 4, 9, 2, 3, 6, 7]]),
        2: Permutation([[1, 9, 8, 3, 2, 5], [0, 6, 7, 4]]),
        3: Permutation([[6, 8, 9], [4, 5], [1, 3, 7, 2], [0]]),
        4: Permutation([[8], [5], [4], [2, 6, 9, 3], [1], [0, 7]]),
        5: Permutation([[3, 6], [0, 8, 1, 7, 5, 9, 4, 2]]),
        6: Permutation([[5], [2, 9], [1, 8, 3], [0, 4, 7, 6]]),
        7: Permutation([[1, 8, 4, 7, 2, 3], [0, 6, 9, 5]]),
        8: Permutation([[5, 8, 7], [3], [1, 4, 2, 6], [0, 9]]),
        9: Permutation([[4, 9, 6], [3, 8], [1, 2], [0, 5, 7]])}
    assert A.is_alt_sym(_random_prec=_random_prec) is False

    G = PermutationGroup(
        Permutation(1, 3, size=8)(0, 2, 4, 6),
        Permutation(5, 7, size=8)(0, 2, 4, 6))
    assert G.is_alt_sym() is False

    # Tests for monte-carlo c_n parameter setting, and which guarantees
    # to give False.
    G = DihedralGroup(10)
    assert G._eval_is_alt_sym_monte_carlo() is False
    G = DihedralGroup(20)
    assert G._eval_is_alt_sym_monte_carlo() is False

    # A dry-running test to check if it looks up for the updated cache.
    G = DihedralGroup(6)
    G.is_alt_sym()
    assert G.is_alt_sym() is False


def test_minimal_block():
    D = DihedralGroup(6)
    block_system = D.minimal_block([0, 3])
    for i in range(3):
        assert block_system[i] == block_system[i + 3]
    S = SymmetricGroup(6)
    assert S.minimal_block([0, 1]) == [0, 0, 0, 0, 0, 0]

    assert Tetra.pgroup.minimal_block([0, 1]) == [0, 0, 0, 0]

    P1 = PermutationGroup(Permutation(1, 5)(2, 4), Permutation(0, 1, 2, 3, 4, 5))
    P2 = PermutationGroup(Permutation(0, 1, 2, 3, 4, 5), Permutation(1, 5)(2, 4))
    assert P1.minimal_block([0, 2]) == [0, 1, 0, 1, 0, 1]
    assert P2.minimal_block([0, 2]) == [0, 1, 0, 1, 0, 1]


def test_minimal_blocks():
    P = PermutationGroup(Permutation(1, 5)(2, 4), Permutation(0, 1, 2, 3, 4, 5))
    assert P.minimal_blocks() == [[0, 1, 0, 1, 0, 1], [0, 1, 2, 0, 1, 2]]

    P = SymmetricGroup(5)
    assert P.minimal_blocks() == [[0]*5]

    P = PermutationGroup(Permutation(0, 3))
    assert P.minimal_blocks() is False


def test_max_div():
    S = SymmetricGroup(10)
    assert S.max_div == 5


def test_is_primitive():
    S = SymmetricGroup(5)
    assert S.is_primitive() is True
    C = CyclicGroup(7)
    assert C.is_primitive() is True

    a = Permutation(0, 1, 2, size=6)
    b = Permutation(3, 4, 5, size=6)
    G = PermutationGroup(a, b)
    assert G.is_primitive() is False


def test_random_stab():
    S = SymmetricGroup(5)
    _random_el = Permutation([1, 3, 2, 0, 4])
    _random_prec = {'rand': _random_el}
    g = S.random_stab(2, _random_prec=_random_prec)
    assert g == Permutation([1, 3, 2, 0, 4])
    h = S.random_stab(1)
    assert h(1) == 1


def test_transitivity_degree():
    perm = Permutation([1, 2, 0])
    C = PermutationGroup([perm])
    assert C.transitivity_degree == 1
    gen1 = Permutation([1, 2, 0, 3, 4])
    gen2 = Permutation([1, 2, 3, 4, 0])
    # alternating group of degree 5
    Alt = PermutationGroup([gen1, gen2])
    assert Alt.transitivity_degree == 3


def test_schreier_sims_random():
    assert sorted(Tetra.pgroup.base) == [0, 1]

    S = SymmetricGroup(3)
    base = [0, 1]
    strong_gens = [Permutation([1, 2, 0]), Permutation([1, 0, 2]),
                  Permutation([0, 2, 1])]
    assert S.schreier_sims_random(base, strong_gens, 5) == (base, strong_gens)
    D = DihedralGroup(3)
    _random_prec = {'g': [Permutation([2, 0, 1]), Permutation([1, 2, 0]),
                         Permutation([1, 0, 2])]}
    base = [0, 1]
    strong_gens = [Permutation([1, 2, 0]), Permutation([2, 1, 0]),
                  Permutation([0, 2, 1])]
    assert D.schreier_sims_random([], D.generators, 2,
           _random_prec=_random_prec) == (base, strong_gens)


def test_baseswap():
    S = SymmetricGroup(4)
    S.schreier_sims()
    base = S.base
    strong_gens = S.strong_gens
    assert base == [0, 1, 2]
    deterministic = S.baseswap(base, strong_gens, 1, randomized=False)
    randomized = S.baseswap(base, strong_gens, 1)
    assert deterministic[0] == [0, 2, 1]
    assert _verify_bsgs(S, deterministic[0], deterministic[1]) is True
    assert randomized[0] == [0, 2, 1]
    assert _verify_bsgs(S, randomized[0], randomized[1]) is True


def test_schreier_sims_incremental():
    identity = Permutation([0, 1, 2, 3, 4])
    TrivialGroup = PermutationGroup([identity])
    base, strong_gens = TrivialGroup.schreier_sims_incremental(base=[0, 1, 2])
    assert _verify_bsgs(TrivialGroup, base, strong_gens) is True
    S = SymmetricGroup(5)
    base, strong_gens = S.schreier_sims_incremental(base=[0, 1, 2])
    assert _verify_bsgs(S, base, strong_gens) is True
    D = DihedralGroup(2)
    base, strong_gens = D.schreier_sims_incremental(base=[1])
    assert _verify_bsgs(D, base, strong_gens) is True
    A = AlternatingGroup(7)
    gens = A.generators[:]
    gen0 = gens[0]
    gen1 = gens[1]
    gen1 = rmul(gen1, ~gen0)
    gen0 = rmul(gen0, gen1)
    gen1 = rmul(gen0, gen1)
    base, strong_gens = A.schreier_sims_incremental(base=[0, 1], gens=gens)
    assert _verify_bsgs(A, base, strong_gens) is True
    C = CyclicGroup(11)
    gen = C.generators[0]
    base, strong_gens = C.schreier_sims_incremental(gens=[gen**3])
    assert _verify_bsgs(C, base, strong_gens) is True


def _subgroup_search(i, j, k):
    prop_true = lambda x: True
    prop_fix_points = lambda x: [x(point) for point in points] == points
    prop_comm_g = lambda x: rmul(x, g) == rmul(g, x)
    prop_even = lambda x: x.is_even
    for i in range(i, j, k):
        S = SymmetricGroup(i)
        A = AlternatingGroup(i)
        C = CyclicGroup(i)
        Sym = S.subgroup_search(prop_true)
        assert Sym.is_subgroup(S)
        Alt = S.subgroup_search(prop_even)
        assert Alt.is_subgroup(A)
        Sym = S.subgroup_search(prop_true, init_subgroup=C)
        assert Sym.is_subgroup(S)
        points = [7]
        assert S.stabilizer(7).is_subgroup(S.subgroup_search(prop_fix_points))
        points = [3, 4]
        assert S.stabilizer(3).stabilizer(4).is_subgroup(
            S.subgroup_search(prop_fix_points))
        points = [3, 5]
        fix35 = A.subgroup_search(prop_fix_points)
        points = [5]
        fix5 = A.subgroup_search(prop_fix_points)
        assert A.subgroup_search(prop_fix_points, init_subgroup=fix35
            ).is_subgroup(fix5)
        base, strong_gens = A.schreier_sims_incremental()
        g = A.generators[0]
        comm_g = \
            A.subgroup_search(prop_comm_g, base=base, strong_gens=strong_gens)
        assert _verify_bsgs(comm_g, base, comm_g.generators) is True
        assert [prop_comm_g(gen) is True for gen in comm_g.generators]


def test_subgroup_search():
    _subgroup_search(10, 15, 2)


@XFAIL
def test_subgroup_search2():
    skip('takes too much time')
    _subgroup_search(16, 17, 1)


def test_normal_closure():
    # the normal closure of the trivial group is trivial
    S = SymmetricGroup(3)
    identity = Permutation([0, 1, 2])
    closure = S.normal_closure(identity)
    assert closure.is_trivial
    # the normal closure of the entire group is the entire group
    A = AlternatingGroup(4)
    assert A.normal_closure(A).is_subgroup(A)
    # brute-force verifications for subgroups
    for i in (3, 4, 5):
        S = SymmetricGroup(i)
        A = AlternatingGroup(i)
        D = DihedralGroup(i)
        C = CyclicGroup(i)
        for gp in (A, D, C):
            assert _verify_normal_closure(S, gp)
    # brute-force verifications for all elements of a group
    S = SymmetricGroup(5)
    elements = list(S.generate_dimino())
    for element in elements:
        assert _verify_normal_closure(S, element)
    # small groups
    small = []
    for i in (1, 2, 3):
        small.append(SymmetricGroup(i))
        small.append(AlternatingGroup(i))
        small.append(DihedralGroup(i))
        small.append(CyclicGroup(i))
    for gp in small:
        for gp2 in small:
            if gp2.is_subgroup(gp, 0) and gp2.degree == gp.degree:
                assert _verify_normal_closure(gp, gp2)


def test_derived_series():
    # the derived series of the trivial group consists only of the trivial group
    triv = PermutationGroup([Permutation([0, 1, 2])])
    assert triv.derived_series()[0].is_subgroup(triv)
    # the derived series for a simple group consists only of the group itself
    for i in (5, 6, 7):
        A = AlternatingGroup(i)
        assert A.derived_series()[0].is_subgroup(A)
    # the derived series for S_4 is S_4 > A_4 > K_4 > triv
    S = SymmetricGroup(4)
    series = S.derived_series()
    assert series[1].is_subgroup(AlternatingGroup(4))
    assert series[2].is_subgroup(DihedralGroup(2))
    assert series[3].is_trivial


def test_lower_central_series():
    # the lower central series of the trivial group consists of the trivial
    # group
    triv = PermutationGroup([Permutation([0, 1, 2])])
    assert triv.lower_central_series()[0].is_subgroup(triv)
    # the lower central series of a simple group consists of the group itself
    for i in (5, 6, 7):
        A = AlternatingGroup(i)
        assert A.lower_central_series()[0].is_subgroup(A)
    # GAP-verified example
    S = SymmetricGroup(6)
    series = S.lower_central_series()
    assert len(series) == 2
    assert series[1].is_subgroup(AlternatingGroup(6))


def test_commutator():
    # the commutator of the trivial group and the trivial group is trivial
    S = SymmetricGroup(3)
    triv = PermutationGroup([Permutation([0, 1, 2])])
    assert S.commutator(triv, triv).is_subgroup(triv)
    # the commutator of the trivial group and any other group is again trivial
    A = AlternatingGroup(3)
    assert S.commutator(triv, A).is_subgroup(triv)
    # the commutator is commutative
    for i in (3, 4, 5):
        S = SymmetricGroup(i)
        A = AlternatingGroup(i)
        D = DihedralGroup(i)
        assert S.commutator(A, D).is_subgroup(S.commutator(D, A))
    # the commutator of an abelian group is trivial
    S = SymmetricGroup(7)
    A1 = AbelianGroup(2, 5)
    A2 = AbelianGroup(3, 4)
    triv = PermutationGroup([Permutation([0, 1, 2, 3, 4, 5, 6])])
    assert S.commutator(A1, A1).is_subgroup(triv)
    assert S.commutator(A2, A2).is_subgroup(triv)
    # examples calculated by hand
    S = SymmetricGroup(3)
    A = AlternatingGroup(3)
    assert S.commutator(A, S).is_subgroup(A)


def test_is_nilpotent():
    # every abelian group is nilpotent
    for i in (1, 2, 3):
        C = CyclicGroup(i)
        Ab = AbelianGroup(i, i + 2)
        assert C.is_nilpotent
        assert Ab.is_nilpotent
    Ab = AbelianGroup(5, 7, 10)
    assert Ab.is_nilpotent
    # A_5 is not solvable and thus not nilpotent
    assert AlternatingGroup(5).is_nilpotent is False


def test_is_trivial():
    for i in range(5):
        triv = PermutationGroup([Permutation(list(range(i)))])
        assert triv.is_trivial


def test_pointwise_stabilizer():
    S = SymmetricGroup(2)
    stab = S.pointwise_stabilizer([0])
    assert stab.generators == [Permutation(1)]
    S = SymmetricGroup(5)
    points = []
    stab = S
    for point in (2, 0, 3, 4, 1):
        stab = stab.stabilizer(point)
        points.append(point)
        assert S.pointwise_stabilizer(points).is_subgroup(stab)


def test_make_perm():
    assert cube.pgroup.make_perm(5, seed=list(range(5))) == \
        Permutation([4, 7, 6, 5, 0, 3, 2, 1])
    assert cube.pgroup.make_perm(7, seed=list(range(7))) == \
        Permutation([6, 7, 3, 2, 5, 4, 0, 1])


def test_elements():
    from sympy.sets.sets import FiniteSet

    p = Permutation(2, 3)
    assert set(PermutationGroup(p).elements) == {Permutation(3), Permutation(2, 3)}
    assert FiniteSet(*PermutationGroup(p).elements) \
        == FiniteSet(Permutation(2, 3), Permutation(3))


def test_is_group():
    assert PermutationGroup(Permutation(1,2), Permutation(2,4)).is_group is True
    assert SymmetricGroup(4).is_group is True


def test_PermutationGroup():
    assert PermutationGroup() == PermutationGroup(Permutation())
    assert (PermutationGroup() == 0) is False


def test_coset_transvesal():
    G = AlternatingGroup(5)
    H = PermutationGroup(Permutation(0,1,2),Permutation(1,2)(3,4))
    assert G.coset_transversal(H) == \
        [Permutation(4), Permutation(2, 3, 4), Permutation(2, 4, 3),
         Permutation(1, 2, 4), Permutation(4)(1, 2, 3), Permutation(1, 3)(2, 4),
         Permutation(0, 1, 2, 3, 4), Permutation(0, 1, 2, 4, 3),
         Permutation(0, 1, 3, 2, 4), Permutation(0, 2, 4, 1, 3)]


def test_coset_table():
    G = PermutationGroup(Permutation(0,1,2,3), Permutation(0,1,2),
         Permutation(0,4,2,7), Permutation(5,6), Permutation(0,7))
    H = PermutationGroup(Permutation(0,1,2,3), Permutation(0,7))
    assert G.coset_table(H) == \
        [[0, 0, 0, 0, 1, 2, 3, 3, 0, 0], [4, 5, 2, 5, 6, 0, 7, 7, 1, 1],
         [5, 4, 5, 1, 0, 6, 8, 8, 6, 6], [3, 3, 3, 3, 7, 8, 0, 0, 3, 3],
         [2, 1, 4, 4, 4, 4, 9, 9, 4, 4], [1, 2, 1, 2, 5, 5, 10, 10, 5, 5],
         [6, 6, 6, 6, 2, 1, 11, 11, 2, 2], [9, 10, 8, 10, 11, 3, 1, 1, 7, 7],
         [10, 9, 10, 7, 3, 11, 2, 2, 11, 11], [8, 7, 9, 9, 9, 9, 4, 4, 9, 9],
         [7, 8, 7, 8, 10, 10, 5, 5, 10, 10], [11, 11, 11, 11, 8, 7, 6, 6, 8, 8]]


def test_subgroup():
    G = PermutationGroup(Permutation(0,1,2), Permutation(0,2,3))
    H = G.subgroup([Permutation(0,1,3)])
    assert H.is_subgroup(G)


def test_generator_product():
    G = SymmetricGroup(5)
    p = Permutation(0, 2, 3)(1, 4)
    gens = G.generator_product(p)
    assert all(g in G.strong_gens for g in gens)
    w = G.identity
    for g in gens:
        w = g*w
    assert w == p


def test_sylow_subgroup():
    P = PermutationGroup(Permutation(1, 5)(2, 4), Permutation(0, 1, 2, 3, 4, 5))
    S = P.sylow_subgroup(2)
    assert S.order() == 4

    P = DihedralGroup(12)
    S = P.sylow_subgroup(3)
    assert S.order() == 3

    P = PermutationGroup(
        Permutation(1, 5)(2, 4), Permutation(0, 1, 2, 3, 4, 5), Permutation(0, 2))
    S = P.sylow_subgroup(3)
    assert S.order() == 9
    S = P.sylow_subgroup(2)
    assert S.order() == 8

    P = SymmetricGroup(10)
    S = P.sylow_subgroup(2)
    assert S.order() == 256
    S = P.sylow_subgroup(3)
    assert S.order() == 81
    S = P.sylow_subgroup(5)
    assert S.order() == 25

    # the length of the lower central series
    # of a p-Sylow subgroup of Sym(n) grows with
    # the highest exponent exp of p such
    # that n >= p**exp
    exp = 1
    length = 0
    for i in range(2, 9):
        P = SymmetricGroup(i)
        S = P.sylow_subgroup(2)
        ls = S.lower_central_series()
        if i // 2**exp > 0:
            # length increases with exponent
            assert len(ls) > length
            length = len(ls)
            exp += 1
        else:
            assert len(ls) == length

    G = SymmetricGroup(100)
    S = G.sylow_subgroup(3)
    assert G.order() % S.order() == 0
    assert G.order()/S.order() % 3 > 0

    G = AlternatingGroup(100)
    S = G.sylow_subgroup(2)
    assert G.order() % S.order() == 0
    assert G.order()/S.order() % 2 > 0

    G = DihedralGroup(18)
    S = G.sylow_subgroup(p=2)
    assert S.order() == 4

    G = DihedralGroup(50)
    S = G.sylow_subgroup(p=2)
    assert S.order() == 4


@slow
def test_presentation():
    def _test(P):
        G = P.presentation()
        return G.order() == P.order()

    def _strong_test(P):
        G = P.strong_presentation()
        chk = len(G.generators) == len(P.strong_gens)
        return chk and G.order() == P.order()

    P = PermutationGroup(Permutation(0,1,5,2)(3,7,4,6), Permutation(0,3,5,4)(1,6,2,7))
    assert _test(P)

    P = AlternatingGroup(5)
    assert _test(P)

    P = SymmetricGroup(5)
    assert _test(P)

    P = PermutationGroup(
        [Permutation(0,3,1,2), Permutation(3)(0,1), Permutation(0,1)(2,3)])
    assert _strong_test(P)

    P = DihedralGroup(6)
    assert _strong_test(P)

    a = Permutation(0,1)(2,3)
    b = Permutation(0,2)(3,1)
    c = Permutation(4,5)
    P = PermutationGroup(c, a, b)
    assert _strong_test(P)


def test_polycyclic():
    a = Permutation([0, 1, 2])
    b = Permutation([2, 1, 0])
    G = PermutationGroup([a, b])
    assert G.is_polycyclic is True

    a = Permutation([1, 2, 3, 4, 0])
    b = Permutation([1, 0, 2, 3, 4])
    G = PermutationGroup([a, b])
    assert G.is_polycyclic is False


def test_elementary():
    a = Permutation([1, 5, 2, 0, 3, 6, 4])
    G = PermutationGroup([a])
    assert G.is_elementary(7) is False

    a = Permutation(0, 1)(2, 3)
    b = Permutation(0, 2)(3, 1)
    G = PermutationGroup([a, b])
    assert G.is_elementary(2) is True
    c = Permutation(4, 5, 6)
    G = PermutationGroup([a, b, c])
    assert G.is_elementary(2) is False

    G = SymmetricGroup(4).sylow_subgroup(2)
    assert G.is_elementary(2) is False
    H = AlternatingGroup(4).sylow_subgroup(2)
    assert H.is_elementary(2) is True


def test_perfect():
    G = AlternatingGroup(3)
    assert G.is_perfect is False
    G = AlternatingGroup(5)
    assert G.is_perfect is True


def test_index():
    G = PermutationGroup(Permutation(0,1,2), Permutation(0,2,3))
    H = G.subgroup([Permutation(0,1,3)])
    assert G.index(H) == 4


def test_cyclic():
    G = SymmetricGroup(2)
    assert G.is_cyclic
    G = AbelianGroup(3, 7)
    assert G.is_cyclic
    G = AbelianGroup(7, 7)
    assert not G.is_cyclic
    G = AlternatingGroup(3)
    assert G.is_cyclic
    G = AlternatingGroup(4)
    assert not G.is_cyclic

    # Order less than 6
    G = PermutationGroup(Permutation(0, 1, 2), Permutation(0, 2, 1))
    assert G.is_cyclic
    G = PermutationGroup(
        Permutation(0, 1, 2, 3),
        Permutation(0, 2)(1, 3)
    )
    assert G.is_cyclic
    G = PermutationGroup(
        Permutation(3),
        Permutation(0, 1)(2, 3),
        Permutation(0, 2)(1, 3),
        Permutation(0, 3)(1, 2)
    )
    assert G.is_cyclic is False

    # Order 15
    G = PermutationGroup(
        Permutation(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
        Permutation(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13)
    )
    assert G.is_cyclic

    # Distinct prime orders
    assert PermutationGroup._distinct_primes_lemma([3, 5]) is True
    assert PermutationGroup._distinct_primes_lemma([5, 7]) is True
    assert PermutationGroup._distinct_primes_lemma([2, 3]) is None
    assert PermutationGroup._distinct_primes_lemma([3, 5, 7]) is None
    assert PermutationGroup._distinct_primes_lemma([5, 7, 13]) is True

    G = PermutationGroup(
        Permutation(0, 1, 2, 3),
        Permutation(0, 2)(1, 3))
    assert G.is_cyclic
    assert G._is_abelian

    # Non-abelian and therefore not cyclic
    G = PermutationGroup(*SymmetricGroup(3).generators)
    assert G.is_cyclic is False

    # Abelian and cyclic
    G = PermutationGroup(
        Permutation(0, 1, 2, 3),
        Permutation(4, 5, 6)
    )
    assert G.is_cyclic

    # Abelian but not cyclic
    G = PermutationGroup(
        Permutation(0, 1),
        Permutation(2, 3),
        Permutation(4, 5, 6)
    )
    assert G.is_cyclic is False


def test_dihedral():
    G = SymmetricGroup(2)
    assert G.is_dihedral
    G = SymmetricGroup(3)
    assert G.is_dihedral

    G = AbelianGroup(2, 2)
    assert G.is_dihedral
    G = CyclicGroup(4)
    assert not G.is_dihedral

    G = AbelianGroup(3, 5)
    assert not G.is_dihedral
    G = AbelianGroup(2)
    assert G.is_dihedral
    G = AbelianGroup(6)
    assert not G.is_dihedral

    # D6, generated by two adjacent flips
    G = PermutationGroup(
        Permutation(1, 5)(2, 4),
        Permutation(0, 1)(3, 4)(2, 5))
    assert G.is_dihedral

    # D7, generated by a flip and a rotation
    G = PermutationGroup(
        Permutation(1, 6)(2, 5)(3, 4),
        Permutation(0, 1, 2, 3, 4, 5, 6))
    assert G.is_dihedral

    # S4, presented by three generators, fails due to having exactly 9
    # elements of order 2:
    G = PermutationGroup(
        Permutation(0, 1), Permutation(0, 2),
        Permutation(0, 3))
    assert not G.is_dihedral

    # D7, given by three generators
    G = PermutationGroup(
        Permutation(1, 6)(2, 5)(3, 4),
        Permutation(2, 0)(3, 6)(4, 5),
        Permutation(0, 1, 2, 3, 4, 5, 6))
    assert G.is_dihedral


def test_abelian_invariants():
    G = AbelianGroup(2, 3, 4)
    assert G.abelian_invariants() == [2, 3, 4]
    G=PermutationGroup([Permutation(1, 2, 3, 4), Permutation(1, 2), Permutation(5, 6)])
    assert G.abelian_invariants() == [2, 2]
    G = AlternatingGroup(7)
    assert G.abelian_invariants() == []
    G = AlternatingGroup(4)
    assert G.abelian_invariants() == [3]
    G = DihedralGroup(4)
    assert G.abelian_invariants() == [2, 2]

    G = PermutationGroup([Permutation(1, 2, 3, 4, 5, 6, 7)])
    assert G.abelian_invariants() == [7]
    G = DihedralGroup(12)
    S = G.sylow_subgroup(3)
    assert S.abelian_invariants() == [3]
    G = PermutationGroup(Permutation(0, 1, 2), Permutation(0, 2, 3))
    assert G.abelian_invariants() == [3]
    G = PermutationGroup([Permutation(0, 1), Permutation(0, 2, 4, 6)(1, 3, 5, 7)])
    assert G.abelian_invariants() == [2, 4]
    G = SymmetricGroup(30)
    S = G.sylow_subgroup(2)
    assert S.abelian_invariants() == [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    S = G.sylow_subgroup(3)
    assert S.abelian_invariants() == [3, 3, 3, 3]
    S = G.sylow_subgroup(5)
    assert S.abelian_invariants() == [5, 5, 5]


def test_composition_series():
    a = Permutation(1, 2, 3)
    b = Permutation(1, 2)
    G = PermutationGroup([a, b])
    comp_series = G.composition_series()
    assert comp_series == G.derived_series()
    # The first group in the composition series is always the group itself and
    # the last group in the series is the trivial group.
    S = SymmetricGroup(4)
    assert S.composition_series()[0] == S
    assert len(S.composition_series()) == 5
    A = AlternatingGroup(4)
    assert A.composition_series()[0] == A
    assert len(A.composition_series()) == 4

    # the composition series for C_8 is C_8 > C_4 > C_2 > triv
    G = CyclicGroup(8)
    series = G.composition_series()
    assert is_isomorphic(series[1], CyclicGroup(4))
    assert is_isomorphic(series[2], CyclicGroup(2))
    assert series[3].is_trivial


def test_is_symmetric():
    a = Permutation(0, 1, 2)
    b = Permutation(0, 1, size=3)
    assert PermutationGroup(a, b).is_symmetric is True

    a = Permutation(0, 2, 1)
    b = Permutation(1, 2, size=3)
    assert PermutationGroup(a, b).is_symmetric is True

    a = Permutation(0, 1, 2, 3)
    b = Permutation(0, 3)(1, 2)
    assert PermutationGroup(a, b).is_symmetric is False

def test_conjugacy_class():
    S = SymmetricGroup(4)
    x = Permutation(1, 2, 3)
    C = {Permutation(0, 1, 2, size = 4), Permutation(0, 1, 3),
             Permutation(0, 2, 1, size = 4), Permutation(0, 2, 3),
             Permutation(0, 3, 1), Permutation(0, 3, 2),
             Permutation(1, 2, 3), Permutation(1, 3, 2)}
    assert S.conjugacy_class(x) == C

def test_conjugacy_classes():
    S = SymmetricGroup(3)
    expected = [{Permutation(size = 3)},
         {Permutation(0, 1, size = 3), Permutation(0, 2), Permutation(1, 2)},
         {Permutation(0, 1, 2), Permutation(0, 2, 1)}]
    computed = S.conjugacy_classes()

    assert len(expected) == len(computed)
    assert all(e in computed for e in expected)

def test_coset_class():
    a = Permutation(1, 2)
    b = Permutation(0, 1)
    G = PermutationGroup([a, b])
    #Creating right coset
    rht_coset = G*a
    #Checking whether it is left coset or right coset
    assert rht_coset.is_right_coset
    assert not rht_coset.is_left_coset
    #Creating list representation of coset
    list_repr = rht_coset.as_list()
    expected = [Permutation(0, 2), Permutation(0, 2, 1), Permutation(1, 2),
                Permutation(2), Permutation(2)(0, 1), Permutation(0, 1, 2)]
    for ele in list_repr:
        assert ele in expected
    #Creating left coset
    left_coset = a*G
    #Checking whether it is left coset or right coset
    assert not left_coset.is_right_coset
    assert left_coset.is_left_coset
    #Creating list representation of Coset
    list_repr = left_coset.as_list()
    expected = [Permutation(2)(0, 1), Permutation(0, 1, 2), Permutation(1, 2),
    Permutation(2), Permutation(0, 2), Permutation(0, 2, 1)]
    for ele in list_repr:
        assert ele in expected

    G = PermutationGroup(Permutation(1, 2, 3, 4), Permutation(2, 3, 4))
    H = PermutationGroup(Permutation(1, 2, 3, 4))
    g = Permutation(1, 3)(2, 4)
    rht_coset = Coset(g, H, G, dir='+')
    assert rht_coset.is_right_coset
    list_repr = rht_coset.as_list()
    expected = [Permutation(1, 2, 3, 4), Permutation(4), Permutation(1, 3)(2, 4),
    Permutation(1, 4, 3, 2)]
    for ele in list_repr:
        assert ele in expected

def test_symmetricpermutationgroup():
    a = SymmetricPermutationGroup(5)
    assert a.degree == 5
    assert a.order() == 120
    assert a.identity() == Permutation(4)
