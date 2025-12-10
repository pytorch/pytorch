from sympy.combinatorics.permutations import Permutation, Perm
from sympy.combinatorics.tensor_can import (perm_af_direct_product, dummy_sgs,
    riemann_bsgs, get_symmetric_group_sgs, canonicalize, bsgs_direct_product)
from sympy.combinatorics.testutil import canonicalize_naive, graph_certificate
from sympy.testing.pytest import skip, XFAIL

def test_perm_af_direct_product():
    gens1 = [[1,0,2,3], [0,1,3,2]]
    gens2 = [[1,0]]
    assert perm_af_direct_product(gens1, gens2, 0) == [[1, 0, 2, 3, 4, 5], [0, 1, 3, 2, 4, 5], [0, 1, 2, 3, 5, 4]]
    gens1 = [[1,0,2,3,5,4], [0,1,3,2,4,5]]
    gens2 = [[1,0,2,3]]
    assert [[1, 0, 2, 3, 4, 5, 7, 6], [0, 1, 3, 2, 4, 5, 6, 7], [0, 1, 2, 3, 5, 4, 6, 7]]

def test_dummy_sgs():
    a = dummy_sgs([1,2], 0, 4)
    assert a == [[0,2,1,3,4,5]]
    a = dummy_sgs([2,3,4,5], 0, 8)
    assert a == [x._array_form for x in [Perm(9)(2,3), Perm(9)(4,5),
        Perm(9)(2,4)(3,5)]]

    a = dummy_sgs([2,3,4,5], 1, 8)
    assert a == [x._array_form for x in [Perm(2,3)(8,9), Perm(4,5)(8,9),
        Perm(9)(2,4)(3,5)]]

def test_get_symmetric_group_sgs():
    assert get_symmetric_group_sgs(2) == ([0], [Permutation(3)(0,1)])
    assert get_symmetric_group_sgs(2, 1) == ([0], [Permutation(0,1)(2,3)])
    assert get_symmetric_group_sgs(3) == ([0,1], [Permutation(4)(0,1), Permutation(4)(1,2)])
    assert get_symmetric_group_sgs(3, 1) == ([0,1], [Permutation(0,1)(3,4), Permutation(1,2)(3,4)])
    assert get_symmetric_group_sgs(4) == ([0,1,2], [Permutation(5)(0,1), Permutation(5)(1,2), Permutation(5)(2,3)])
    assert get_symmetric_group_sgs(4, 1) == ([0,1,2], [Permutation(0,1)(4,5), Permutation(1,2)(4,5), Permutation(2,3)(4,5)])


def test_canonicalize_no_slot_sym():
    # cases in which there is no slot symmetry after fixing the
    # free indices; here and in the following if the symmetry of the
    # metric is not specified, it is assumed to be symmetric.
    # If it is not specified, tensors are commuting.

    # A_d0 * B^d0; g = [1,0, 2,3]; T_c = A^d0*B_d0; can = [0,1,2,3]
    base1, gens1 = get_symmetric_group_sgs(1)
    dummies = [0, 1]
    g = Permutation([1,0,2,3])
    can = canonicalize(g, dummies, 0, (base1,gens1,1,0), (base1,gens1,1,0))
    assert can == [0,1,2,3]
    # equivalently
    can = canonicalize(g, dummies, 0, (base1, gens1, 2, None))
    assert can == [0,1,2,3]

    # with antisymmetric metric; T_c = -A^d0*B_d0; can = [0,1,3,2]
    can = canonicalize(g, dummies, 1, (base1,gens1,1,0), (base1,gens1,1,0))
    assert can == [0,1,3,2]

    # A^a * B^b; ord = [a,b]; g = [0,1,2,3]; can = g
    g = Permutation([0,1,2,3])
    dummies = []
    t0 = t1 = (base1, gens1, 1, 0)
    can = canonicalize(g, dummies, 0, t0, t1)
    assert can == [0,1,2,3]
    # B^b * A^a
    g = Permutation([1,0,2,3])
    can = canonicalize(g, dummies, 0, t0, t1)
    assert can == [1,0,2,3]

    # A symmetric
    # A^{b}_{d0}*A^{d0, a} order a,b,d0,-d0; T_c = A^{a d0}*A{b}_{d0}
    # g = [1,3,2,0,4,5]; can = [0,2,1,3,4,5]
    base2, gens2 = get_symmetric_group_sgs(2)
    dummies = [2,3]
    g = Permutation([1,3,2,0,4,5])
    can = canonicalize(g, dummies, 0, (base2, gens2, 2, 0))
    assert can == [0, 2, 1, 3, 4, 5]
    # with antisymmetric metric
    can = canonicalize(g, dummies, 1, (base2, gens2, 2, 0))
    assert can == [0, 2, 1, 3, 4, 5]
    # A^{a}_{d0}*A^{d0, b}
    g = Permutation([0,3,2,1,4,5])
    can = canonicalize(g, dummies, 1, (base2, gens2, 2, 0))
    assert can == [0, 2, 1, 3, 5, 4]

    # A, B symmetric
    # A^b_d0*B^{d0,a}; g=[1,3,2,0,4,5]
    # T_c = A^{b,d0}*B_{a,d0}; can = [1,2,0,3,4,5]
    dummies = [2,3]
    g = Permutation([1,3,2,0,4,5])
    can = canonicalize(g, dummies, 0, (base2,gens2,1,0), (base2,gens2,1,0))
    assert can == [1,2,0,3,4,5]
    # same with antisymmetric metric
    can = canonicalize(g, dummies, 1, (base2,gens2,1,0), (base2,gens2,1,0))
    assert can == [1,2,0,3,5,4]

    # A^{d1}_{d0}*B^d0*C_d1 ord=[d0,-d0,d1,-d1]; g = [2,1,0,3,4,5]
    # T_c = A^{d0 d1}*B_d0*C_d1; can = [0,2,1,3,4,5]
    base1, gens1 = get_symmetric_group_sgs(1)
    base2, gens2 = get_symmetric_group_sgs(2)
    g = Permutation([2,1,0,3,4,5])
    dummies = [0,1,2,3]
    t0 = (base2, gens2, 1, 0)
    t1 = t2 = (base1, gens1, 1, 0)
    can = canonicalize(g, dummies, 0, t0, t1, t2)
    assert can == [0, 2, 1, 3, 4, 5]

    # A without symmetry
    # A^{d1}_{d0}*B^d0*C_d1 ord=[d0,-d0,d1,-d1]; g = [2,1,0,3,4,5]
    # T_c = A^{d0 d1}*B_d1*C_d0; can = [0,2,3,1,4,5]
    g = Permutation([2,1,0,3,4,5])
    dummies = [0,1,2,3]
    t0 = ([], [Permutation(list(range(4)))], 1, 0)
    can = canonicalize(g, dummies, 0, t0, t1, t2)
    assert can == [0,2,3,1,4,5]
    # A, B without symmetry
    # A^{d1}_{d0}*B_{d1}^{d0}; g = [2,1,3,0,4,5]
    # T_c = A^{d0 d1}*B_{d0 d1}; can = [0,2,1,3,4,5]
    t0 = t1 = ([], [Permutation(list(range(4)))], 1, 0)
    dummies = [0,1,2,3]
    g = Permutation([2,1,3,0,4,5])
    can = canonicalize(g, dummies, 0, t0, t1)
    assert can == [0, 2, 1, 3, 4, 5]
    # A_{d0}^{d1}*B_{d1}^{d0}; g = [1,2,3,0,4,5]
    # T_c = A^{d0 d1}*B_{d1 d0}; can = [0,2,3,1,4,5]
    g = Permutation([1,2,3,0,4,5])
    can = canonicalize(g, dummies, 0, t0, t1)
    assert can == [0,2,3,1,4,5]

    # A, B, C without symmetry
    # A^{d1 d0}*B_{a d0}*C_{d1 b} ord=[a,b,d0,-d0,d1,-d1]
    # g=[4,2,0,3,5,1,6,7]
    # T_c=A^{d0 d1}*B_{a d1}*C_{d0 b}; can = [2,4,0,5,3,1,6,7]
    t0 = t1 = t2 = ([], [Permutation(list(range(4)))], 1, 0)
    dummies = [2,3,4,5]
    g = Permutation([4,2,0,3,5,1,6,7])
    can = canonicalize(g, dummies, 0, t0, t1, t2)
    assert can == [2,4,0,5,3,1,6,7]

    # A symmetric, B and C without symmetry
    # A^{d1 d0}*B_{a d0}*C_{d1 b} ord=[a,b,d0,-d0,d1,-d1]
    # g=[4,2,0,3,5,1,6,7]
    # T_c = A^{d0 d1}*B_{a d0}*C_{d1 b}; can = [2,4,0,3,5,1,6,7]
    t0 = (base2,gens2,1,0)
    t1 = t2 = ([], [Permutation(list(range(4)))], 1, 0)
    dummies = [2,3,4,5]
    g = Permutation([4,2,0,3,5,1,6,7])
    can = canonicalize(g, dummies, 0, t0, t1, t2)
    assert can == [2,4,0,3,5,1,6,7]

    # A and C symmetric, B without symmetry
    # A^{d1 d0}*B_{a d0}*C_{d1 b} ord=[a,b,d0,-d0,d1,-d1]
    # g=[4,2,0,3,5,1,6,7]
    # T_c = A^{d0 d1}*B_{a d0}*C_{b d1}; can = [2,4,0,3,1,5,6,7]
    t0 = t2 = (base2,gens2,1,0)
    t1 = ([], [Permutation(list(range(4)))], 1, 0)
    dummies = [2,3,4,5]
    g = Permutation([4,2,0,3,5,1,6,7])
    can = canonicalize(g, dummies, 0, t0, t1, t2)
    assert can == [2,4,0,3,1,5,6,7]

    # A symmetric, B without symmetry, C antisymmetric
    # A^{d1 d0}*B_{a d0}*C_{d1 b} ord=[a,b,d0,-d0,d1,-d1]
    # g=[4,2,0,3,5,1,6,7]
    # T_c = -A^{d0 d1}*B_{a d0}*C_{b d1}; can = [2,4,0,3,1,5,7,6]
    t0 = (base2,gens2, 1, 0)
    t1 = ([], [Permutation(list(range(4)))], 1, 0)
    base2a, gens2a = get_symmetric_group_sgs(2, 1)
    t2 = (base2a, gens2a, 1, 0)
    dummies = [2,3,4,5]
    g = Permutation([4,2,0,3,5,1,6,7])
    can = canonicalize(g, dummies, 0, t0, t1, t2)
    assert can == [2,4,0,3,1,5,7,6]


def test_canonicalize_no_dummies():
    base1, gens1 = get_symmetric_group_sgs(1)
    base2, gens2 = get_symmetric_group_sgs(2)
    base2a, gens2a = get_symmetric_group_sgs(2, 1)

    # A commuting
    # A^c A^b A^a; ord = [a,b,c]; g = [2,1,0,3,4]
    # T_c = A^a A^b A^c; can = list(range(5))
    g = Permutation([2,1,0,3,4])
    can = canonicalize(g, [], 0, (base1, gens1, 3, 0))
    assert can == list(range(5))

    # A anticommuting
    # A^c A^b A^a; ord = [a,b,c]; g = [2,1,0,3,4]
    # T_c = -A^a A^b A^c; can = [0,1,2,4,3]
    g = Permutation([2,1,0,3,4])
    can = canonicalize(g, [], 0, (base1, gens1, 3, 1))
    assert can == [0,1,2,4,3]

    # A commuting and symmetric
    # A^{b,d}*A^{c,a}; ord = [a,b,c,d]; g = [1,3,2,0,4,5]
    # T_c = A^{a c}*A^{b d}; can = [0,2,1,3,4,5]
    g = Permutation([1,3,2,0,4,5])
    can = canonicalize(g, [], 0, (base2, gens2, 2, 0))
    assert can == [0,2,1,3,4,5]

    # A anticommuting and symmetric
    # A^{b,d}*A^{c,a}; ord = [a,b,c,d]; g = [1,3,2,0,4,5]
    # T_c = -A^{a c}*A^{b d}; can = [0,2,1,3,5,4]
    g = Permutation([1,3,2,0,4,5])
    can = canonicalize(g, [], 0, (base2, gens2, 2, 1))
    assert can == [0,2,1,3,5,4]
    # A^{c,a}*A^{b,d} ; g = [2,0,1,3,4,5]
    # T_c = A^{a c}*A^{b d}; can = [0,2,1,3,4,5]
    g = Permutation([2,0,1,3,4,5])
    can = canonicalize(g, [], 0, (base2, gens2, 2, 1))
    assert can == [0,2,1,3,4,5]

def test_no_metric_symmetry():
    # no metric symmetry
    # A^d1_d0 * A^d0_d1; ord = [d0,-d0,d1,-d1]; g= [2,1,0,3,4,5]
    # T_c = A^d0_d1 * A^d1_d0; can = [0,3,2,1,4,5]
    g = Permutation([2,1,0,3,4,5])
    can = canonicalize(g, list(range(4)), None, [[], [Permutation(list(range(4)))], 2, 0])
    assert can == [0,3,2,1,4,5]

    # A^d1_d2 * A^d0_d3 * A^d2_d1 * A^d3_d0
    # ord = [d0,-d0,d1,-d1,d2,-d2,d3,-d3]
    #        0    1  2  3  4   5   6   7
    # g = [2,5,0,7,4,3,6,1,8,9]
    # T_c = A^d0_d1 * A^d1_d0 * A^d2_d3 * A^d3_d2
    # can = [0,3,2,1,4,7,6,5,8,9]
    g = Permutation([2,5,0,7,4,3,6,1,8,9])
    #can = canonicalize(g, list(range(8)), 0, [[], [list(range(4))], 4, 0])
    #assert can == [0, 2, 3, 1, 4, 6, 7, 5, 8, 9]
    can = canonicalize(g, list(range(8)), None, [[], [Permutation(list(range(4)))], 4, 0])
    assert can == [0, 3, 2, 1, 4, 7, 6, 5, 8, 9]

    # A^d0_d2 * A^d1_d3 * A^d3_d0 * A^d2_d1
    # g = [0,5,2,7,6,1,4,3,8,9]
    # T_c = A^d0_d1 * A^d1_d2 * A^d2_d3 * A^d3_d0
    # can = [0,3,2,5,4,7,6,1,8,9]
    g = Permutation([0,5,2,7,6,1,4,3,8,9])
    can = canonicalize(g, list(range(8)), None, [[], [Permutation(list(range(4)))], 4, 0])
    assert can == [0,3,2,5,4,7,6,1,8,9]

    g = Permutation([12,7,10,3,14,13,4,11,6,1,2,9,0,15,8,5,16,17])
    can = canonicalize(g, list(range(16)), None, [[], [Permutation(list(range(4)))], 8, 0])
    assert can == [0,3,2,5,4,7,6,1,8,11,10,13,12,15,14,9,16,17]

def test_canonical_free():
    # t = A^{d0 a1}*A_d0^a0
    # ord = [a0,a1,d0,-d0];  g = [2,1,3,0,4,5]; dummies = [[2,3]]
    # t_c = A_d0^a0*A^{d0 a1}
    # can = [3,0, 2,1, 4,5]
    g = Permutation([2,1,3,0,4,5])
    dummies = [[2,3]]
    can = canonicalize(g, dummies, [None], ([], [Permutation(3)], 2, 0))
    assert can == [3,0, 2,1, 4,5]

def test_canonicalize1():
    base1, gens1 = get_symmetric_group_sgs(1)
    base1a, gens1a = get_symmetric_group_sgs(1, 1)
    base2, gens2 = get_symmetric_group_sgs(2)
    base3, gens3 = get_symmetric_group_sgs(3)
    base2a, gens2a = get_symmetric_group_sgs(2, 1)
    base3a, gens3a = get_symmetric_group_sgs(3, 1)

    # A_d0*A^d0; ord = [d0,-d0]; g = [1,0,2,3]
    # T_c = A^d0*A_d0; can = [0,1,2,3]
    g = Permutation([1,0,2,3])
    can = canonicalize(g, [0, 1], 0, (base1, gens1, 2, 0))
    assert can == list(range(4))

    # A commuting
    # A_d0*A_d1*A_d2*A^d2*A^d1*A^d0; ord=[d0,-d0,d1,-d1,d2,-d2]
    # g = [1,3,5,4,2,0,6,7]
    # T_c = A^d0*A_d0*A^d1*A_d1*A^d2*A_d2; can = list(range(8))
    g = Permutation([1,3,5,4,2,0,6,7])
    can = canonicalize(g, list(range(6)), 0, (base1, gens1, 6, 0))
    assert can == list(range(8))

    # A anticommuting
    # A_d0*A_d1*A_d2*A^d2*A^d1*A^d0; ord=[d0,-d0,d1,-d1,d2,-d2]
    # g = [1,3,5,4,2,0,6,7]
    # T_c 0;  can = 0
    g = Permutation([1,3,5,4,2,0,6,7])
    can = canonicalize(g, list(range(6)), 0, (base1, gens1, 6, 1))
    assert can == 0
    can1 = canonicalize_naive(g, list(range(6)), 0, (base1, gens1, 6, 1))
    assert can1 == 0

    # A commuting symmetric
    # A^{d0 b}*A^a_d1*A^d1_d0; ord=[a,b,d0,-d0,d1,-d1]
    # g = [2,1,0,5,4,3,6,7]
    # T_c = A^{a d0}*A^{b d1}*A_{d0 d1}; can = [0,2,1,4,3,5,6,7]
    g = Permutation([2,1,0,5,4,3,6,7])
    can = canonicalize(g, list(range(2,6)), 0, (base2, gens2, 3, 0))
    assert can == [0,2,1,4,3,5,6,7]

    # A, B commuting symmetric
    # A^{d0 b}*A^d1_d0*B^a_d1; ord=[a,b,d0,-d0,d1,-d1]
    # g = [2,1,4,3,0,5,6,7]
    # T_c = A^{b d0}*A_d0^d1*B^a_d1; can = [1,2,3,4,0,5,6,7]
    g = Permutation([2,1,4,3,0,5,6,7])
    can = canonicalize(g, list(range(2,6)), 0, (base2,gens2,2,0), (base2,gens2,1,0))
    assert can == [1,2,3,4,0,5,6,7]

    # A commuting symmetric
    # A^{d1 d0 b}*A^{a}_{d1 d0}; ord=[a,b, d0,-d0,d1,-d1]
    # g = [4,2,1,0,5,3,6,7]
    # T_c = A^{a d0 d1}*A^{b}_{d0 d1}; can = [0,2,4,1,3,5,6,7]
    g = Permutation([4,2,1,0,5,3,6,7])
    can = canonicalize(g, list(range(2,6)), 0, (base3, gens3, 2, 0))
    assert can == [0,2,4,1,3,5,6,7]


    # A^{d3 d0 d2}*A^a0_{d1 d2}*A^d1_d3^a1*A^{a2 a3}_d0
    # ord = [a0,a1,a2,a3,d0,-d0,d1,-d1,d2,-d2,d3,-d3]
    #        0   1  2  3  4  5  6   7  8   9  10  11
    # g = [10,4,8, 0,7,9, 6,11,1, 2,3,5, 12,13]
    # T_c = A^{a0 d0 d1}*A^a1_d0^d2*A^{a2 a3 d3}*A_{d1 d2 d3}
    # can = [0,4,6, 1,5,8, 2,3,10, 7,9,11, 12,13]
    g = Permutation([10,4,8, 0,7,9, 6,11,1, 2,3,5, 12,13])
    can = canonicalize(g, list(range(4,12)), 0, (base3, gens3, 4, 0))
    assert can == [0,4,6, 1,5,8, 2,3,10, 7,9,11, 12,13]

    # A commuting symmetric, B antisymmetric
    # A^{d0 d1 d2} * A_{d2 d3 d1} * B_d0^d3
    # ord = [d0,-d0,d1,-d1,d2,-d2,d3,-d3]
    # g = [0,2,4,5,7,3,1,6,8,9]
    # in this esxample and in the next three,
    # renaming dummy indices and using symmetry of A,
    # T = A^{d0 d1 d2} * A_{d0 d1 d3} * B_d2^d3
    # can = 0
    g = Permutation([0,2,4,5,7,3,1,6,8,9])
    can = canonicalize(g, list(range(8)), 0, (base3, gens3,2,0), (base2a,gens2a,1,0))
    assert can == 0
    # A anticommuting symmetric, B anticommuting
    # A^{d0 d1 d2} * A_{d2 d3 d1} * B_d0^d3
    # T_c = A^{d0 d1 d2} * A_{d0 d1}^d3 * B_{d2 d3}
    # can = [0,2,4, 1,3,6, 5,7, 8,9]
    can = canonicalize(g, list(range(8)), 0, (base3, gens3,2,1), (base2a,gens2a,1,0))
    assert can == [0,2,4, 1,3,6, 5,7, 8,9]
    # A anticommuting symmetric, B antisymmetric commuting, antisymmetric metric
    # A^{d0 d1 d2} * A_{d2 d3 d1} * B_d0^d3
    # T_c = -A^{d0 d1 d2} * A_{d0 d1}^d3 * B_{d2 d3}
    # can = [0,2,4, 1,3,6, 5,7, 9,8]
    can = canonicalize(g, list(range(8)), 1, (base3, gens3,2,1), (base2a,gens2a,1,0))
    assert can == [0,2,4, 1,3,6, 5,7, 9,8]

    # A anticommuting symmetric, B anticommuting anticommuting,
    # no metric symmetry
    # A^{d0 d1 d2} * A_{d2 d3 d1} * B_d0^d3
    # T_c = A^{d0 d1 d2} * A_{d0 d1 d3} * B_d2^d3
    # can = [0,2,4, 1,3,7, 5,6, 8,9]
    can = canonicalize(g, list(range(8)), None, (base3, gens3,2,1), (base2a,gens2a,1,0))
    assert can == [0,2,4,1,3,7,5,6,8,9]

    # Gamma anticommuting
    # Gamma_{mu nu} * gamma^rho * Gamma^{nu mu alpha}
    # ord = [alpha, rho, mu,-mu,nu,-nu]
    # g = [3,5,1,4,2,0,6,7]
    # T_c = -Gamma^{mu nu} * gamma^rho * Gamma_{alpha mu nu}
    # can = [2,4,1,0,3,5,7,6]]
    g = Permutation([3,5,1,4,2,0,6,7])
    t0 = (base2a, gens2a, 1, None)
    t1 = (base1, gens1, 1, None)
    t2 = (base3a, gens3a, 1, None)
    can = canonicalize(g, list(range(2, 6)), 0, t0, t1, t2)
    assert can == [2,4,1,0,3,5,7,6]

    # Gamma_{mu nu} * Gamma^{gamma beta} * gamma_rho * Gamma^{nu mu alpha}
    # ord = [alpha, beta, gamma, -rho, mu,-mu,nu,-nu]
    #         0      1      2     3    4   5   6  7
    # g = [5,7,2,1,3,6,4,0,8,9]
    # T_c = Gamma^{mu nu} * Gamma^{beta gamma} * gamma_rho * Gamma^alpha_{mu nu}    # can = [4,6,1,2,3,0,5,7,8,9]
    t0 = (base2a, gens2a, 2, None)
    g = Permutation([5,7,2,1,3,6,4,0,8,9])
    can = canonicalize(g, list(range(4, 8)), 0, t0, t1, t2)
    assert can == [4,6,1,2,3,0,5,7,8,9]

    # f^a_{b,c} antisymmetric in b,c; A_mu^a no symmetry
    # f^c_{d a} * f_{c e b} * A_mu^d * A_nu^a * A^{nu e} * A^{mu b}
    # ord = [mu,-mu,nu,-nu,a,-a,b,-b,c,-c,d,-d, e, -e]
    #         0  1  2   3  4  5 6  7 8  9 10 11 12 13
    # g = [8,11,5, 9,13,7, 1,10, 3,4, 2,12, 0,6, 14,15]
    # T_c = -f^{a b c} * f_a^{d e} * A^mu_b * A_{mu d} * A^nu_c * A_{nu e}
    # can = [4,6,8,       5,10,12,   0,7,     1,11,       2,9,    3,13, 15,14]
    g = Permutation([8,11,5, 9,13,7, 1,10, 3,4, 2,12, 0,6, 14,15])
    base_f, gens_f = bsgs_direct_product(base1, gens1, base2a, gens2a)
    base_A, gens_A = bsgs_direct_product(base1, gens1, base1, gens1)
    t0 = (base_f, gens_f, 2, 0)
    t1 = (base_A, gens_A, 4, 0)
    can = canonicalize(g, [list(range(4)), list(range(4, 14))], [0, 0], t0, t1)
    assert can == [4,6,8, 5,10,12, 0,7, 1,11, 2,9, 3,13, 15,14]


def test_riemann_invariants():
    baser, gensr = riemann_bsgs
    # R^{d0 d1}_{d1 d0}; ord = [d0,-d0,d1,-d1]; g = [0,2,3,1,4,5]
    # T_c = -R^{d0 d1}_{d0 d1}; can = [0,2,1,3,5,4]
    g = Permutation([0,2,3,1,4,5])
    can = canonicalize(g, list(range(2, 4)), 0, (baser, gensr, 1, 0))
    assert can == [0,2,1,3,5,4]
    # use a non minimal BSGS
    can = canonicalize(g, list(range(2, 4)), 0, ([2, 0], [Permutation([1,0,2,3,5,4]), Permutation([2,3,0,1,4,5])], 1, 0))
    assert can == [0,2,1,3,5,4]

    """
    The following tests in test_riemann_invariants and in
    test_riemann_invariants1 have been checked using xperm.c from XPerm in
    in [1] and with an older version contained in [2]

    [1] xperm.c part of xPerm written by J. M. Martin-Garcia
        http://www.xact.es/index.html
    [2] test_xperm.cc in cadabra by Kasper Peeters, http://cadabra.phi-sci.com/
    """
    # R_d11^d1_d0^d5 * R^{d6 d4 d0}_d5 * R_{d7 d2 d8 d9} *
    # R_{d10 d3 d6 d4} * R^{d2 d7 d11}_d1 * R^{d8 d9 d3 d10}
    # ord: contravariant d_k ->2*k, covariant d_k -> 2*k+1
    # T_c = R^{d0 d1 d2 d3} * R_{d0 d1}^{d4 d5} * R_{d2 d3}^{d6 d7} *
    # R_{d4 d5}^{d8 d9} * R_{d6 d7}^{d10 d11} * R_{d8 d9 d10 d11}
    g = Permutation([23,2,1,10,12,8,0,11,15,5,17,19,21,7,13,9,4,14,22,3,16,18,6,20,24,25])
    can = canonicalize(g, list(range(24)), 0, (baser, gensr, 6, 0))
    assert can == [0,2,4,6,1,3,8,10,5,7,12,14,9,11,16,18,13,15,20,22,17,19,21,23,24,25]

    # use a non minimal BSGS
    can = canonicalize(g, list(range(24)), 0, ([2, 0], [Permutation([1,0,2,3,5,4]), Permutation([2,3,0,1,4,5])], 6, 0))
    assert can == [0,2,4,6,1,3,8,10,5,7,12,14,9,11,16,18,13,15,20,22,17,19,21,23,24,25]

    g = Permutation([0,2,5,7,4,6,9,11,8,10,13,15,12,14,17,19,16,18,21,23,20,22,25,27,24,26,29,31,28,30,33,35,32,34,37,39,36,38,1,3,40,41])
    can = canonicalize(g, list(range(40)), 0, (baser, gensr, 10, 0))
    assert can == [0,2,4,6,1,3,8,10,5,7,12,14,9,11,16,18,13,15,20,22,17,19,24,26,21,23,28,30,25,27,32,34,29,31,36,38,33,35,37,39,40,41]


@XFAIL
def test_riemann_invariants1():
    skip('takes too much time')
    baser, gensr = riemann_bsgs
    g = Permutation([17, 44, 11, 3, 0, 19, 23, 15, 38, 4, 25, 27, 43, 36, 22, 14, 8, 30, 41, 20, 2, 10, 12, 28, 18, 1, 29, 13, 37, 42, 33, 7, 9, 31, 24, 26, 39, 5, 34, 47, 32, 6, 21, 40, 35, 46, 45, 16, 48, 49])
    can = canonicalize(g, list(range(48)), 0, (baser, gensr, 12, 0))
    assert can == [0, 2, 4, 6, 1, 3, 8, 10, 5, 7, 12, 14, 9, 11, 16, 18, 13, 15, 20, 22, 17, 19, 24, 26, 21, 23, 28, 30, 25, 27, 32, 34, 29, 31, 36, 38, 33, 35, 40, 42, 37, 39, 44, 46, 41, 43, 45, 47, 48, 49]

    g = Permutation([0,2,4,6, 7,8,10,12, 14,16,18,20, 19,22,24,26, 5,21,28,30, 32,34,36,38, 40,42,44,46, 13,48,50,52, 15,49,54,56, 17,33,41,58, 9,23,60,62, 29,35,63,64, 3,45,66,68, 25,37,47,57, 11,31,69,70, 27,39,53,72, 1,59,73,74, 55,61,67,76, 43,65,75,78, 51,71,77,79, 80,81])
    can = canonicalize(g, list(range(80)), 0, (baser, gensr, 20, 0))
    assert can == [0,2,4,6, 1,8,10,12, 3,14,16,18, 5,20,22,24, 7,26,28,30, 9,15,32,34, 11,36,23,38, 13,40,42,44, 17,39,29,46, 19,48,43,50, 21,45,52,54, 25,56,33,58, 27,60,53,62, 31,51,64,66, 35,65,47,68, 37,70,49,72, 41,74,57,76, 55,67,59,78, 61,69,71,75, 63,79,73,77, 80,81]


def test_riemann_products():
    baser, gensr = riemann_bsgs
    base1, gens1 = get_symmetric_group_sgs(1)
    base2, gens2 = get_symmetric_group_sgs(2)
    base2a, gens2a = get_symmetric_group_sgs(2, 1)

    # R^{a b d0}_d0 = 0
    g = Permutation([0,1,2,3,4,5])
    can = canonicalize(g, list(range(2,4)), 0, (baser, gensr, 1, 0))
    assert can == 0

    # R^{d0 b a}_d0 ; ord = [a,b,d0,-d0}; g = [2,1,0,3,4,5]
    # T_c = -R^{a d0 b}_d0;  can = [0,2,1,3,5,4]
    g = Permutation([2,1,0,3,4,5])
    can = canonicalize(g, list(range(2, 4)), 0, (baser, gensr, 1, 0))
    assert can == [0,2,1,3,5,4]

    # R^d1_d2^b_d0 * R^{d0 a}_d1^d2; ord=[a,b,d0,-d0,d1,-d1,d2,-d2]
    # g = [4,7,1,3,2,0,5,6,8,9]
    # T_c = -R^{a d0 d1 d2}* R^b_{d0 d1 d2}
    # can = [0,2,4,6,1,3,5,7,9,8]
    g = Permutation([4,7,1,3,2,0,5,6,8,9])
    can = canonicalize(g, list(range(2,8)), 0, (baser, gensr, 2, 0))
    assert can == [0,2,4,6,1,3,5,7,9,8]
    can1 = canonicalize_naive(g, list(range(2,8)), 0, (baser, gensr, 2, 0))
    assert can == can1

    # A symmetric commuting
    # R^{d6 d5}_d2^d1 * R^{d4 d0 d2 d3} * A_{d6 d0} A_{d3 d1} * A_{d4 d5}
    # g = [12,10,5,2, 8,0,4,6, 13,1, 7,3, 9,11,14,15]
    # T_c = -R^{d0 d1 d2 d3} * R_d0^{d4 d5 d6} * A_{d1 d4}*A_{d2 d5}*A_{d3 d6}

    g = Permutation([12,10,5,2,8,0,4,6,13,1,7,3,9,11,14,15])
    can = canonicalize(g, list(range(14)), 0, ((baser,gensr,2,0)), (base2,gens2,3,0))
    assert can == [0, 2, 4, 6, 1, 8, 10, 12, 3, 9, 5, 11, 7, 13, 15, 14]

    # R^{d2 a0 a2 d0} * R^d1_d2^{a1 a3} * R^{a4 a5}_{d0 d1}
    # ord = [a0,a1,a2,a3,a4,a5,d0,-d0,d1,-d1,d2,-d2]
    #         0  1  2  3 4  5  6   7  8   9  10  11
    # can = [0, 6, 2, 8, 1, 3, 7, 10, 4, 5, 9, 11, 12, 13]
    # T_c = R^{a0 d0 a2 d1}*R^{a1 a3}_d0^d2*R^{a4 a5}_{d1 d2}
    g = Permutation([10,0,2,6,8,11,1,3,4,5,7,9,12,13])
    can = canonicalize(g, list(range(6,12)), 0, (baser, gensr, 3, 0))
    assert can == [0, 6, 2, 8, 1, 3, 7, 10, 4, 5, 9, 11, 12, 13]
    #can1 = canonicalize_naive(g, list(range(6,12)), 0, (baser, gensr, 3, 0))
    #assert can == can1

    # A^n_{i, j} antisymmetric in i,j
    # A_m0^d0_a1 * A_m1^a0_d0; ord = [m0,m1,a0,a1,d0,-d0]
    # g = [0,4,3,1,2,5,6,7]
    # T_c = -A_{m a1}^d0 * A_m1^a0_d0
    # can = [0,3,4,1,2,5,7,6]
    base, gens = bsgs_direct_product(base1, gens1, base2a, gens2a)
    dummies = list(range(4, 6))
    g = Permutation([0,4,3,1,2,5,6,7])
    can = canonicalize(g, dummies, 0, (base, gens, 2, 0))
    assert can == [0, 3, 4, 1, 2, 5, 7, 6]


    # A^n_{i, j} symmetric in i,j
    # A^m0_a0^d2 * A^n0_d2^d1 * A^n1_d1^d0 * A_{m0 d0}^a1
    # ordering: first the free indices; then first n, then d
    # ord=[n0,n1,a0,a1, m0,-m0,d0,-d0,d1,-d1,d2,-d2]
    #      0  1   2  3   4  5  6   7  8   9  10  11]
    # g = [4,2,10, 0,11,8, 1,9,6, 5,7,3, 12,13]
    # if the dummy indices m_i and d_i were separated,
    # one gets
    # T_c = A^{n0 d0 d1} * A^n1_d0^d2 * A^m0^a0_d1 * A_m0^a1_d2
    # can = [0, 6, 8, 1, 7, 10, 4, 2, 9, 5, 3, 11, 12, 13]
    # If they are not, so can is
    # T_c = A^{n0 m0 d0} A^n1_m0^d1 A^{d2 a0}_d0 A_d2^a1_d1
    # can = [0, 4, 6, 1, 5, 8, 10, 2, 7, 11, 3, 9, 12, 13]
    # case with single type of indices

    base, gens = bsgs_direct_product(base1, gens1, base2, gens2)
    dummies = list(range(4, 12))
    g = Permutation([4,2,10, 0,11,8, 1,9,6, 5,7,3, 12,13])
    can = canonicalize(g, dummies, 0, (base, gens, 4, 0))
    assert can == [0, 4, 6, 1, 5, 8, 10, 2, 7, 11, 3, 9, 12, 13]
    # case with separated indices
    dummies = [list(range(4, 6)), list(range(6,12))]
    sym = [0, 0]
    can = canonicalize(g, dummies, sym, (base, gens, 4, 0))
    assert can == [0, 6, 8, 1, 7, 10, 4, 2, 9, 5, 3, 11, 12, 13]
    # case with separated indices with the second type of index
    # with antisymmetric metric: there is a sign change
    sym = [0, 1]
    can = canonicalize(g, dummies, sym, (base, gens, 4, 0))
    assert can == [0, 6, 8, 1, 7, 10, 4, 2, 9, 5, 3, 11, 13, 12]

def test_graph_certificate():
    # test tensor invariants constructed from random regular graphs;
    # checked graph isomorphism with networkx
    import random
    def randomize_graph(size, g):
        p = list(range(size))
        random.shuffle(p)
        g1a = {}
        for k, v in g1.items():
            g1a[p[k]] = [p[i] for i in v]
        return g1a

    g1 = {0: [2, 3, 7], 1: [4, 5, 7], 2: [0, 4, 6], 3: [0, 6, 7], 4: [1, 2, 5], 5: [1, 4, 6], 6: [2, 3, 5], 7: [0, 1, 3]}
    g2 = {0: [2, 3, 7], 1: [2, 4, 5], 2: [0, 1, 5], 3: [0, 6, 7], 4: [1, 5, 6], 5: [1, 2, 4], 6: [3, 4, 7], 7: [0, 3, 6]}

    c1 = graph_certificate(g1)
    c2 = graph_certificate(g2)
    assert c1 != c2
    g1a = randomize_graph(8, g1)
    c1a = graph_certificate(g1a)
    assert c1 == c1a

    g1 = {0: [8, 1, 9, 7], 1: [0, 9, 3, 4], 2: [3, 4, 6, 7], 3: [1, 2, 5, 6], 4: [8, 1, 2, 5], 5: [9, 3, 4, 7], 6: [8, 2, 3, 7], 7: [0, 2, 5, 6], 8: [0, 9, 4, 6], 9: [8, 0, 5, 1]}
    g2 = {0: [1, 2, 5, 6], 1: [0, 9, 5, 7], 2: [0, 4, 6, 7], 3: [8, 9, 6, 7], 4: [8, 2, 6, 7], 5: [0, 9, 8, 1], 6: [0, 2, 3, 4], 7: [1, 2, 3, 4], 8: [9, 3, 4, 5], 9: [8, 1, 3, 5]}
    c1 = graph_certificate(g1)
    c2 = graph_certificate(g2)
    assert c1 != c2
    g1a = randomize_graph(10, g1)
    c1a = graph_certificate(g1a)
    assert c1 == c1a
