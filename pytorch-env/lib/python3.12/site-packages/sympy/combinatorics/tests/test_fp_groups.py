from sympy.core.singleton import S
from sympy.combinatorics.fp_groups import (FpGroup, low_index_subgroups,
                                   reidemeister_presentation, FpSubgroup,
                                           simplify_presentation)
from sympy.combinatorics.free_groups import (free_group, FreeGroup)

from sympy.testing.pytest import slow

"""
References
==========

[1] Holt, D., Eick, B., O'Brien, E.
"Handbook of Computational Group Theory"

[2] John J. Cannon; Lucien A. Dimino; George Havas; Jane M. Watson
Mathematics of Computation, Vol. 27, No. 123. (Jul., 1973), pp. 463-490.
"Implementation and Analysis of the Todd-Coxeter Algorithm"

[3] PROC. SECOND  INTERNAT. CONF. THEORY OF GROUPS, CANBERRA 1973,
pp. 347-356. "A Reidemeister-Schreier program" by George Havas.
http://staff.itee.uq.edu.au/havas/1973cdhw.pdf

"""

def test_low_index_subgroups():
    F, x, y = free_group("x, y")

    # Example 5.10 from [1] Pg. 194
    f = FpGroup(F, [x**2, y**3, (x*y)**4])
    L = low_index_subgroups(f, 4)
    t1 = [[[0, 0, 0, 0]],
          [[0, 0, 1, 2], [1, 1, 2, 0], [3, 3, 0, 1], [2, 2, 3, 3]],
          [[0, 0, 1, 2], [2, 2, 2, 0], [1, 1, 0, 1]],
          [[1, 1, 0, 0], [0, 0, 1, 1]]]
    for i in range(len(t1)):
        assert L[i].table == t1[i]

    f = FpGroup(F, [x**2, y**3, (x*y)**7])
    L = low_index_subgroups(f, 15)
    t2 = [[[0, 0, 0, 0]],
           [[0, 0, 1, 2], [1, 1, 2, 0], [3, 3, 0, 1], [2, 2, 4, 5],
            [4, 4, 5, 3], [6, 6, 3, 4], [5, 5, 6, 6]],
           [[0, 0, 1, 2], [1, 1, 2, 0], [3, 3, 0, 1], [2, 2, 4, 5],
            [6, 6, 5, 3], [5, 5, 3, 4], [4, 4, 6, 6]],
           [[0, 0, 1, 2], [1, 1, 2, 0], [3, 3, 0, 1], [2, 2, 4, 5],
            [6, 6, 5, 3], [7, 7, 3, 4], [4, 4, 8, 9], [5, 5, 10, 11],
            [11, 11, 9, 6], [9, 9, 6, 8], [12, 12, 11, 7], [8, 8, 7, 10],
            [10, 10, 13, 14], [14, 14, 14, 12], [13, 13, 12, 13]],
           [[0, 0, 1, 2], [1, 1, 2, 0], [3, 3, 0, 1], [2, 2, 4, 5],
            [6, 6, 5, 3], [7, 7, 3, 4], [4, 4, 8, 9], [5, 5, 10, 11],
            [11, 11, 9, 6], [12, 12, 6, 8], [10, 10, 11, 7], [8, 8, 7, 10],
            [9, 9, 13, 14], [14, 14, 14, 12], [13, 13, 12, 13]],
           [[0, 0, 1, 2], [1, 1, 2, 0], [3, 3, 0, 1], [2, 2, 4, 5],
            [6, 6, 5, 3], [7, 7, 3, 4], [4, 4, 8, 9], [5, 5, 10, 11],
            [11, 11, 9, 6], [12, 12, 6, 8], [13, 13, 11, 7], [8, 8, 7, 10],
            [9, 9, 12, 12], [10, 10, 13, 13]],
           [[0, 0, 1, 2], [3, 3, 2, 0], [4, 4, 0, 1], [1, 1, 3, 3], [2, 2, 5, 6]
            , [7, 7, 6, 4], [8, 8, 4, 5], [5, 5, 8, 9], [6, 6, 9, 7],
            [10, 10, 7, 8], [9, 9, 11, 12], [11, 11, 12, 10], [13, 13, 10, 11],
            [12, 12, 13, 13]],
           [[0, 0, 1, 2], [3, 3, 2, 0], [4, 4, 0, 1], [1, 1, 3, 3], [2, 2, 5, 6]
            , [7, 7, 6, 4], [8, 8, 4, 5], [5, 5, 8, 9], [6, 6, 9, 7],
            [10, 10, 7, 8], [9, 9, 11, 12], [13, 13, 12, 10], [12, 12, 10, 11],
            [11, 11, 13, 13]],
           [[0, 0, 1, 2], [3, 3, 2, 0], [4, 4, 0, 1], [1, 1, 5, 6], [2, 2, 4, 4]
            , [7, 7, 6, 3], [8, 8, 3, 5], [5, 5, 8, 9], [6, 6, 9, 7],
            [10, 10, 7, 8], [9, 9, 11, 12], [13, 13, 12, 10], [12, 12, 10, 11],
            [11, 11, 13, 13]],
           [[0, 0, 1, 2], [3, 3, 2, 0], [4, 4, 0, 1], [1, 1, 5, 6], [2, 2, 7, 8]
            , [5, 5, 6, 3], [9, 9, 3, 5], [10, 10, 8, 4], [8, 8, 4, 7],
            [6, 6, 10, 11], [7, 7, 11, 9], [12, 12, 9, 10], [11, 11, 13, 14],
            [14, 14, 14, 12], [13, 13, 12, 13]],
           [[0, 0, 1, 2], [3, 3, 2, 0], [4, 4, 0, 1], [1, 1, 5, 6], [2, 2, 7, 8]
            , [6, 6, 6, 3], [5, 5, 3, 5], [8, 8, 8, 4], [7, 7, 4, 7]],
           [[0, 0, 1, 2], [3, 3, 2, 0], [4, 4, 0, 1], [1, 1, 5, 6], [2, 2, 7, 8]
            , [9, 9, 6, 3], [6, 6, 3, 5], [10, 10, 8, 4], [11, 11, 4, 7],
            [5, 5, 10, 12], [7, 7, 12, 9], [8, 8, 11, 11], [13, 13, 9, 10],
            [12, 12, 13, 13]],
           [[0, 0, 1, 2], [3, 3, 2, 0], [4, 4, 0, 1], [1, 1, 5, 6], [2, 2, 7, 8]
            , [9, 9, 6, 3], [6, 6, 3, 5], [10, 10, 8, 4], [11, 11, 4, 7],
            [5, 5, 12, 11], [7, 7, 10, 10], [8, 8, 9, 12], [13, 13, 11, 9],
            [12, 12, 13, 13]],
           [[0, 0, 1, 2], [3, 3, 2, 0], [4, 4, 0, 1], [1, 1, 5, 6], [2, 2, 7, 8]
            , [9, 9, 6, 3], [10, 10, 3, 5], [7, 7, 8, 4], [11, 11, 4, 7],
            [5, 5, 9, 9], [6, 6, 11, 12], [8, 8, 12, 10], [13, 13, 10, 11],
            [12, 12, 13, 13]],
           [[0, 0, 1, 2], [3, 3, 2, 0], [4, 4, 0, 1], [1, 1, 5, 6], [2, 2, 7, 8]
            , [9, 9, 6, 3], [10, 10, 3, 5], [7, 7, 8, 4], [11, 11, 4, 7],
            [5, 5, 12, 11], [6, 6, 10, 10], [8, 8, 9, 12], [13, 13, 11, 9],
            [12, 12, 13, 13]],
           [[0, 0, 1, 2], [3, 3, 2, 0], [4, 4, 0, 1], [1, 1, 5, 6], [2, 2, 7, 8]
            , [9, 9, 6, 3], [10, 10, 3, 5], [11, 11, 8, 4], [12, 12, 4, 7],
            [5, 5, 9, 9], [6, 6, 12, 13], [7, 7, 11, 11], [8, 8, 13, 10],
            [13, 13, 10, 12]],
           [[1, 1, 0, 0], [0, 0, 2, 3], [4, 4, 3, 1], [5, 5, 1, 2], [2, 2, 4, 4]
            , [3, 3, 6, 7], [7, 7, 7, 5], [6, 6, 5, 6]]]
    for i  in range(len(t2)):
        assert L[i].table == t2[i]

    f = FpGroup(F, [x**2, y**3, (x*y)**7])
    L = low_index_subgroups(f, 10, [x])
    t3 = [[[0, 0, 0, 0]],
          [[0, 0, 1, 2], [1, 1, 2, 0], [3, 3, 0, 1], [2, 2, 4, 5], [4, 4, 5, 3],
           [6, 6, 3, 4], [5, 5, 6, 6]],
          [[0, 0, 1, 2], [1, 1, 2, 0], [3, 3, 0, 1], [2, 2, 4, 5], [6, 6, 5, 3],
           [5, 5, 3, 4], [4, 4, 6, 6]],
          [[0, 0, 1, 2], [3, 3, 2, 0], [4, 4, 0, 1], [1, 1, 5, 6], [2, 2, 7, 8],
           [6, 6, 6, 3], [5, 5, 3, 5], [8, 8, 8, 4], [7, 7, 4, 7]]]
    for i in range(len(t3)):
        assert L[i].table == t3[i]


def test_subgroup_presentations():
    F, x, y = free_group("x, y")
    f = FpGroup(F, [x**3, y**5, (x*y)**2])
    H = [x*y, x**-1*y**-1*x*y*x]
    p1 = reidemeister_presentation(f, H)
    assert str(p1) == "((y_1, y_2), (y_1**2, y_2**3, y_2*y_1*y_2*y_1*y_2*y_1))"

    H = f.subgroup(H)
    assert (H.generators, H.relators) == p1

    f = FpGroup(F, [x**3, y**3, (x*y)**3])
    H = [x*y, x*y**-1]
    p2 = reidemeister_presentation(f, H)
    assert str(p2) == "((x_0, y_0), (x_0**3, y_0**3, x_0*y_0*x_0*y_0*x_0*y_0))"

    f = FpGroup(F, [x**2*y**2, y**-1*x*y*x**-3])
    H = [x]
    p3 = reidemeister_presentation(f, H)
    assert str(p3) == "((x_0,), (x_0**4,))"

    f = FpGroup(F, [x**3*y**-3, (x*y)**3, (x*y**-1)**2])
    H = [x]
    p4 = reidemeister_presentation(f, H)
    assert str(p4) == "((x_0,), (x_0**6,))"

    # this presentation can be improved, the most simplified form
    # of presentation is <a, b | a^11, b^2, (a*b)^3, (a^4*b*a^-5*b)^2>
    # See [2] Pg 474 group PSL_2(11)
    # This is the group PSL_2(11)
    F, a, b, c = free_group("a, b, c")
    f = FpGroup(F, [a**11, b**5, c**4, (b*c**2)**2, (a*b*c)**3, (a**4*c**2)**3, b**2*c**-1*b**-1*c, a**4*b**-1*a**-1*b])
    H = [a, b, c**2]
    gens, rels = reidemeister_presentation(f, H)
    assert str(gens) == "(b_1, c_3)"
    assert len(rels) == 18


@slow
def test_order():
    F, x, y = free_group("x, y")
    f = FpGroup(F, [x**4, y**2, x*y*x**-1*y])
    assert f.order() == 8

    f = FpGroup(F, [x*y*x**-1*y**-1, y**2])
    assert f.order() is S.Infinity

    F, a, b, c = free_group("a, b, c")
    f = FpGroup(F, [a**250, b**2, c*b*c**-1*b, c**4, c**-1*a**-1*c*a, a**-1*b**-1*a*b])
    assert f.order() == 2000

    F, x = free_group("x")
    f = FpGroup(F, [])
    assert f.order() is S.Infinity

    f = FpGroup(free_group('')[0], [])
    assert f.order() == 1

def test_fp_subgroup():
    def _test_subgroup(K, T, S):
        _gens = T(K.generators)
        assert all(elem in S for elem in _gens)
        assert T.is_injective()
        assert T.image().order() == S.order()
    F, x, y = free_group("x, y")
    f = FpGroup(F, [x**4, y**2, x*y*x**-1*y])
    S = FpSubgroup(f, [x*y])
    assert (x*y)**-3 in S
    K, T = f.subgroup([x*y], homomorphism=True)
    assert T(K.generators) == [y*x**-1]
    _test_subgroup(K, T, S)

    S = FpSubgroup(f, [x**-1*y*x])
    assert x**-1*y**4*x in S
    assert x**-1*y**4*x**2 not in S
    K, T = f.subgroup([x**-1*y*x], homomorphism=True)
    assert T(K.generators[0]**3) == y**3
    _test_subgroup(K, T, S)

    f = FpGroup(F, [x**3, y**5, (x*y)**2])
    H = [x*y, x**-1*y**-1*x*y*x]
    K, T = f.subgroup(H, homomorphism=True)
    S = FpSubgroup(f, H)
    _test_subgroup(K, T, S)

def test_permutation_methods():
    F, x, y = free_group("x, y")
    # DihedralGroup(8)
    G = FpGroup(F, [x**2, y**8, x*y*x**-1*y])
    T = G._to_perm_group()[1]
    assert T.is_isomorphism()
    assert G.center() == [y**4]

    # DiheadralGroup(4)
    G = FpGroup(F, [x**2, y**4, x*y*x**-1*y])
    S = FpSubgroup(G, G.normal_closure([x]))
    assert x in S
    assert y**-1*x*y in S

    # Z_5xZ_4
    G = FpGroup(F, [x*y*x**-1*y**-1, y**5, x**4])
    assert G.is_abelian
    assert G.is_solvable

    # AlternatingGroup(5)
    G = FpGroup(F, [x**3, y**2, (x*y)**5])
    assert not G.is_solvable

    # AlternatingGroup(4)
    G = FpGroup(F, [x**3, y**2, (x*y)**3])
    assert len(G.derived_series()) == 3
    S = FpSubgroup(G, G.derived_subgroup())
    assert S.order() == 4


def test_simplify_presentation():
    # ref #16083
    G = simplify_presentation(FpGroup(FreeGroup([]), []))
    assert not G.generators
    assert not G.relators

    # CyclicGroup(3)
    # The second generator in <x, y | x^2, x^5, y^3> is trivial due to relators {x^2, x^5}
    F, x, y = free_group("x, y")
    G = simplify_presentation(FpGroup(F, [x**2, x**5, y**3]))
    assert x in G.relators

def test_cyclic():
    F, x, y = free_group("x, y")
    f = FpGroup(F, [x*y, x**-1*y**-1*x*y*x])
    assert f.is_cyclic
    f = FpGroup(F, [x*y, x*y**-1])
    assert f.is_cyclic
    f = FpGroup(F, [x**4, y**2, x*y*x**-1*y])
    assert not f.is_cyclic


def test_abelian_invariants():
    F, x, y = free_group("x, y")
    f = FpGroup(F, [x*y, x**-1*y**-1*x*y*x])
    assert f.abelian_invariants() == []
    f = FpGroup(F, [x*y, x*y**-1])
    assert f.abelian_invariants() == [2]
    f = FpGroup(F, [x**4, y**2, x*y*x**-1*y])
    assert f.abelian_invariants() == [2, 4]
