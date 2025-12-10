from sympy.combinatorics.group_constructs import DirectProduct
from sympy.combinatorics.named_groups import CyclicGroup, DihedralGroup


def test_direct_product_n():
    C = CyclicGroup(4)
    D = DihedralGroup(4)
    G = DirectProduct(C, C, C)
    assert G.order() == 64
    assert G.degree == 12
    assert len(G.orbits()) == 3
    assert G.is_abelian is True
    H = DirectProduct(D, C)
    assert H.order() == 32
    assert H.is_abelian is False
