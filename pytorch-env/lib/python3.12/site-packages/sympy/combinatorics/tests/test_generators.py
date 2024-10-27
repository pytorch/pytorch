from sympy.combinatorics.generators import symmetric, cyclic, alternating, \
    dihedral, rubik
from sympy.combinatorics.permutations import Permutation
from sympy.testing.pytest import raises

def test_generators():

    assert list(cyclic(6)) == [
        Permutation([0, 1, 2, 3, 4, 5]),
        Permutation([1, 2, 3, 4, 5, 0]),
        Permutation([2, 3, 4, 5, 0, 1]),
        Permutation([3, 4, 5, 0, 1, 2]),
        Permutation([4, 5, 0, 1, 2, 3]),
        Permutation([5, 0, 1, 2, 3, 4])]

    assert list(cyclic(10)) == [
        Permutation([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        Permutation([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
        Permutation([2, 3, 4, 5, 6, 7, 8, 9, 0, 1]),
        Permutation([3, 4, 5, 6, 7, 8, 9, 0, 1, 2]),
        Permutation([4, 5, 6, 7, 8, 9, 0, 1, 2, 3]),
        Permutation([5, 6, 7, 8, 9, 0, 1, 2, 3, 4]),
        Permutation([6, 7, 8, 9, 0, 1, 2, 3, 4, 5]),
        Permutation([7, 8, 9, 0, 1, 2, 3, 4, 5, 6]),
        Permutation([8, 9, 0, 1, 2, 3, 4, 5, 6, 7]),
        Permutation([9, 0, 1, 2, 3, 4, 5, 6, 7, 8])]

    assert list(alternating(4)) == [
        Permutation([0, 1, 2, 3]),
        Permutation([0, 2, 3, 1]),
        Permutation([0, 3, 1, 2]),
        Permutation([1, 0, 3, 2]),
        Permutation([1, 2, 0, 3]),
        Permutation([1, 3, 2, 0]),
        Permutation([2, 0, 1, 3]),
        Permutation([2, 1, 3, 0]),
        Permutation([2, 3, 0, 1]),
        Permutation([3, 0, 2, 1]),
        Permutation([3, 1, 0, 2]),
        Permutation([3, 2, 1, 0])]

    assert list(symmetric(3)) == [
        Permutation([0, 1, 2]),
        Permutation([0, 2, 1]),
        Permutation([1, 0, 2]),
        Permutation([1, 2, 0]),
        Permutation([2, 0, 1]),
        Permutation([2, 1, 0])]

    assert list(symmetric(4)) == [
        Permutation([0, 1, 2, 3]),
        Permutation([0, 1, 3, 2]),
        Permutation([0, 2, 1, 3]),
        Permutation([0, 2, 3, 1]),
        Permutation([0, 3, 1, 2]),
        Permutation([0, 3, 2, 1]),
        Permutation([1, 0, 2, 3]),
        Permutation([1, 0, 3, 2]),
        Permutation([1, 2, 0, 3]),
        Permutation([1, 2, 3, 0]),
        Permutation([1, 3, 0, 2]),
        Permutation([1, 3, 2, 0]),
        Permutation([2, 0, 1, 3]),
        Permutation([2, 0, 3, 1]),
        Permutation([2, 1, 0, 3]),
        Permutation([2, 1, 3, 0]),
        Permutation([2, 3, 0, 1]),
        Permutation([2, 3, 1, 0]),
        Permutation([3, 0, 1, 2]),
        Permutation([3, 0, 2, 1]),
        Permutation([3, 1, 0, 2]),
        Permutation([3, 1, 2, 0]),
        Permutation([3, 2, 0, 1]),
        Permutation([3, 2, 1, 0])]

    assert list(dihedral(1)) == [
        Permutation([0, 1]), Permutation([1, 0])]

    assert list(dihedral(2)) == [
        Permutation([0, 1, 2, 3]),
        Permutation([1, 0, 3, 2]),
        Permutation([2, 3, 0, 1]),
        Permutation([3, 2, 1, 0])]

    assert list(dihedral(3)) == [
        Permutation([0, 1, 2]),
        Permutation([2, 1, 0]),
        Permutation([1, 2, 0]),
        Permutation([0, 2, 1]),
        Permutation([2, 0, 1]),
        Permutation([1, 0, 2])]

    assert list(dihedral(5)) == [
        Permutation([0, 1, 2, 3, 4]),
        Permutation([4, 3, 2, 1, 0]),
        Permutation([1, 2, 3, 4, 0]),
        Permutation([0, 4, 3, 2, 1]),
        Permutation([2, 3, 4, 0, 1]),
        Permutation([1, 0, 4, 3, 2]),
        Permutation([3, 4, 0, 1, 2]),
        Permutation([2, 1, 0, 4, 3]),
        Permutation([4, 0, 1, 2, 3]),
        Permutation([3, 2, 1, 0, 4])]

    raises(ValueError, lambda: rubik(1))
