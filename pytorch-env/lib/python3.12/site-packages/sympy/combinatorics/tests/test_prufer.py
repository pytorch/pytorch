from sympy.combinatorics.prufer import Prufer
from sympy.testing.pytest import raises


def test_prufer():
    # number of nodes is optional
    assert Prufer([[0, 1], [0, 2], [0, 3], [0, 4]], 5).nodes == 5
    assert Prufer([[0, 1], [0, 2], [0, 3], [0, 4]]).nodes == 5

    a = Prufer([[0, 1], [0, 2], [0, 3], [0, 4]])
    assert a.rank == 0
    assert a.nodes == 5
    assert a.prufer_repr == [0, 0, 0]

    a = Prufer([[2, 4], [1, 4], [1, 3], [0, 5], [0, 4]])
    assert a.rank == 924
    assert a.nodes == 6
    assert a.tree_repr == [[2, 4], [1, 4], [1, 3], [0, 5], [0, 4]]
    assert a.prufer_repr == [4, 1, 4, 0]

    assert Prufer.edges([0, 1, 2, 3], [1, 4, 5], [1, 4, 6]) == \
        ([[0, 1], [1, 2], [1, 4], [2, 3], [4, 5], [4, 6]], 7)
    assert Prufer([0]*4).size == Prufer([6]*4).size == 1296

    # accept iterables but convert to list of lists
    tree = [(0, 1), (1, 5), (0, 3), (0, 2), (2, 6), (4, 7), (2, 4)]
    tree_lists = [list(t) for t in tree]
    assert Prufer(tree).tree_repr == tree_lists
    assert sorted(Prufer(set(tree)).tree_repr) == sorted(tree_lists)

    raises(ValueError, lambda: Prufer([[1, 2], [3, 4]]))  # 0 is missing
    raises(ValueError, lambda: Prufer([[2, 3], [3, 4]]))  # 0, 1 are missing
    assert Prufer(*Prufer.edges([1, 2], [3, 4])).prufer_repr == [1, 3]
    raises(ValueError, lambda: Prufer.edges(
        [1, 3], [3, 4]))  # a broken tree but edges doesn't care
    raises(ValueError, lambda: Prufer.edges([1, 2], [5, 6]))
    raises(ValueError, lambda: Prufer([[]]))

    a = Prufer([[0, 1], [0, 2], [0, 3]])
    b = a.next()
    assert b.tree_repr == [[0, 2], [0, 1], [1, 3]]
    assert b.rank == 1


def test_round_trip():
    def doit(t, b):
        e, n = Prufer.edges(*t)
        t = Prufer(e, n)
        a = sorted(t.tree_repr)
        b = [i - 1 for i in b]
        assert t.prufer_repr == b
        assert sorted(Prufer(b).tree_repr) == a
        assert Prufer.unrank(t.rank, n).prufer_repr == b

    doit([[1, 2]], [])
    doit([[2, 1, 3]], [1])
    doit([[1, 3, 2]], [3])
    doit([[1, 2, 3]], [2])
    doit([[2, 1, 4], [1, 3]], [1, 1])
    doit([[3, 2, 1, 4]], [2, 1])
    doit([[3, 2, 1], [2, 4]], [2, 2])
    doit([[1, 3, 2, 4]], [3, 2])
    doit([[1, 4, 2, 3]], [4, 2])
    doit([[3, 1, 4, 2]], [4, 1])
    doit([[4, 2, 1, 3]], [1, 2])
    doit([[1, 2, 4, 3]], [2, 4])
    doit([[1, 3, 4, 2]], [3, 4])
    doit([[2, 4, 1], [4, 3]], [4, 4])
    doit([[1, 2, 3, 4]], [2, 3])
    doit([[2, 3, 1], [3, 4]], [3, 3])
    doit([[1, 4, 3, 2]], [4, 3])
    doit([[2, 1, 4, 3]], [1, 4])
    doit([[2, 1, 3, 4]], [1, 3])
    doit([[6, 2, 1, 4], [1, 3, 5, 8], [3, 7]], [1, 2, 1, 3, 3, 5])
