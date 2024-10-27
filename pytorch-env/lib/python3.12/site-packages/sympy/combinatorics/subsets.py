from itertools import combinations

from sympy.combinatorics.graycode import GrayCode


class Subset():
    """
    Represents a basic subset object.

    Explanation
    ===========

    We generate subsets using essentially two techniques,
    binary enumeration and lexicographic enumeration.
    The Subset class takes two arguments, the first one
    describes the initial subset to consider and the second
    describes the superset.

    Examples
    ========

    >>> from sympy.combinatorics import Subset
    >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
    >>> a.next_binary().subset
    ['b']
    >>> a.prev_binary().subset
    ['c']
    """

    _rank_binary = None
    _rank_lex = None
    _rank_graycode = None
    _subset = None
    _superset = None

    def __new__(cls, subset, superset):
        """
        Default constructor.

        It takes the ``subset`` and its ``superset`` as its parameters.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.subset
        ['c', 'd']
        >>> a.superset
        ['a', 'b', 'c', 'd']
        >>> a.size
        2
        """
        if len(subset) > len(superset):
            raise ValueError('Invalid arguments have been provided. The '
                             'superset must be larger than the subset.')
        for elem in subset:
            if elem not in superset:
                raise ValueError('The superset provided is invalid as it does '
                                 'not contain the element {}'.format(elem))
        obj = object.__new__(cls)
        obj._subset = subset
        obj._superset = superset
        return obj

    def __eq__(self, other):
        """Return a boolean indicating whether a == b on the basis of
        whether both objects are of the class Subset and if the values
        of the subset and superset attributes are the same.
        """
        if not isinstance(other, Subset):
            return NotImplemented
        return self.subset == other.subset and self.superset == other.superset

    def iterate_binary(self, k):
        """
        This is a helper function. It iterates over the
        binary subsets by ``k`` steps. This variable can be
        both positive or negative.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.iterate_binary(-2).subset
        ['d']
        >>> a = Subset(['a', 'b', 'c'], ['a', 'b', 'c', 'd'])
        >>> a.iterate_binary(2).subset
        []

        See Also
        ========

        next_binary, prev_binary
        """
        bin_list = Subset.bitlist_from_subset(self.subset, self.superset)
        n = (int(''.join(bin_list), 2) + k) % 2**self.superset_size
        bits = bin(n)[2:].rjust(self.superset_size, '0')
        return Subset.subset_from_bitlist(self.superset, bits)

    def next_binary(self):
        """
        Generates the next binary ordered subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.next_binary().subset
        ['b']
        >>> a = Subset(['a', 'b', 'c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.next_binary().subset
        []

        See Also
        ========

        prev_binary, iterate_binary
        """
        return self.iterate_binary(1)

    def prev_binary(self):
        """
        Generates the previous binary ordered subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset([], ['a', 'b', 'c', 'd'])
        >>> a.prev_binary().subset
        ['a', 'b', 'c', 'd']
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.prev_binary().subset
        ['c']

        See Also
        ========

        next_binary, iterate_binary
        """
        return self.iterate_binary(-1)

    def next_lexicographic(self):
        """
        Generates the next lexicographically ordered subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.next_lexicographic().subset
        ['d']
        >>> a = Subset(['d'], ['a', 'b', 'c', 'd'])
        >>> a.next_lexicographic().subset
        []

        See Also
        ========

        prev_lexicographic
        """
        i = self.superset_size - 1
        indices = Subset.subset_indices(self.subset, self.superset)

        if i in indices:
            if i - 1 in indices:
                indices.remove(i - 1)
            else:
                indices.remove(i)
                i = i - 1
                while i >= 0 and i not in indices:
                    i = i - 1
                if i >= 0:
                    indices.remove(i)
                    indices.append(i+1)
        else:
            while i not in indices and i >= 0:
                i = i - 1
            indices.append(i + 1)

        ret_set = []
        super_set = self.superset
        for i in indices:
            ret_set.append(super_set[i])
        return Subset(ret_set, super_set)

    def prev_lexicographic(self):
        """
        Generates the previous lexicographically ordered subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset([], ['a', 'b', 'c', 'd'])
        >>> a.prev_lexicographic().subset
        ['d']
        >>> a = Subset(['c','d'], ['a', 'b', 'c', 'd'])
        >>> a.prev_lexicographic().subset
        ['c']

        See Also
        ========

        next_lexicographic
        """
        i = self.superset_size - 1
        indices = Subset.subset_indices(self.subset, self.superset)

        while i >= 0 and i not in indices:
            i = i - 1

        if i == 0 or i - 1 in indices:
            indices.remove(i)
        else:
            if i >= 0:
                indices.remove(i)
                indices.append(i - 1)
            indices.append(self.superset_size - 1)

        ret_set = []
        super_set = self.superset
        for i in indices:
            ret_set.append(super_set[i])
        return Subset(ret_set, super_set)

    def iterate_graycode(self, k):
        """
        Helper function used for prev_gray and next_gray.
        It performs ``k`` step overs to get the respective Gray codes.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset([1, 2, 3], [1, 2, 3, 4])
        >>> a.iterate_graycode(3).subset
        [1, 4]
        >>> a.iterate_graycode(-2).subset
        [1, 2, 4]

        See Also
        ========

        next_gray, prev_gray
        """
        unranked_code = GrayCode.unrank(self.superset_size,
                                       (self.rank_gray + k) % self.cardinality)
        return Subset.subset_from_bitlist(self.superset,
                                          unranked_code)

    def next_gray(self):
        """
        Generates the next Gray code ordered subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset([1, 2, 3], [1, 2, 3, 4])
        >>> a.next_gray().subset
        [1, 3]

        See Also
        ========

        iterate_graycode, prev_gray
        """
        return self.iterate_graycode(1)

    def prev_gray(self):
        """
        Generates the previous Gray code ordered subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset([2, 3, 4], [1, 2, 3, 4, 5])
        >>> a.prev_gray().subset
        [2, 3, 4, 5]

        See Also
        ========

        iterate_graycode, next_gray
        """
        return self.iterate_graycode(-1)

    @property
    def rank_binary(self):
        """
        Computes the binary ordered rank.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset([], ['a','b','c','d'])
        >>> a.rank_binary
        0
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.rank_binary
        3

        See Also
        ========

        iterate_binary, unrank_binary
        """
        if self._rank_binary is None:
            self._rank_binary = int("".join(
                Subset.bitlist_from_subset(self.subset,
                                           self.superset)), 2)
        return self._rank_binary

    @property
    def rank_lexicographic(self):
        """
        Computes the lexicographic ranking of the subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.rank_lexicographic
        14
        >>> a = Subset([2, 4, 5], [1, 2, 3, 4, 5, 6])
        >>> a.rank_lexicographic
        43
        """
        if self._rank_lex is None:
            def _ranklex(self, subset_index, i, n):
                if subset_index == [] or i > n:
                    return 0
                if i in subset_index:
                    subset_index.remove(i)
                    return 1 + _ranklex(self, subset_index, i + 1, n)
                return 2**(n - i - 1) + _ranklex(self, subset_index, i + 1, n)
            indices = Subset.subset_indices(self.subset, self.superset)
            self._rank_lex = _ranklex(self, indices, 0, self.superset_size)
        return self._rank_lex

    @property
    def rank_gray(self):
        """
        Computes the Gray code ranking of the subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c','d'], ['a','b','c','d'])
        >>> a.rank_gray
        2
        >>> a = Subset([2, 4, 5], [1, 2, 3, 4, 5, 6])
        >>> a.rank_gray
        27

        See Also
        ========

        iterate_graycode, unrank_gray
        """
        if self._rank_graycode is None:
            bits = Subset.bitlist_from_subset(self.subset, self.superset)
            self._rank_graycode = GrayCode(len(bits), start=bits).rank
        return self._rank_graycode

    @property
    def subset(self):
        """
        Gets the subset represented by the current instance.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.subset
        ['c', 'd']

        See Also
        ========

        superset, size, superset_size, cardinality
        """
        return self._subset

    @property
    def size(self):
        """
        Gets the size of the subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.size
        2

        See Also
        ========

        subset, superset, superset_size, cardinality
        """
        return len(self.subset)

    @property
    def superset(self):
        """
        Gets the superset of the subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.superset
        ['a', 'b', 'c', 'd']

        See Also
        ========

        subset, size, superset_size, cardinality
        """
        return self._superset

    @property
    def superset_size(self):
        """
        Returns the size of the superset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.superset_size
        4

        See Also
        ========

        subset, superset, size, cardinality
        """
        return len(self.superset)

    @property
    def cardinality(self):
        """
        Returns the number of all possible subsets.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.cardinality
        16

        See Also
        ========

        subset, superset, size, superset_size
        """
        return 2**(self.superset_size)

    @classmethod
    def subset_from_bitlist(self, super_set, bitlist):
        """
        Gets the subset defined by the bitlist.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> Subset.subset_from_bitlist(['a', 'b', 'c', 'd'], '0011').subset
        ['c', 'd']

        See Also
        ========

        bitlist_from_subset
        """
        if len(super_set) != len(bitlist):
            raise ValueError("The sizes of the lists are not equal")
        ret_set = []
        for i in range(len(bitlist)):
            if bitlist[i] == '1':
                ret_set.append(super_set[i])
        return Subset(ret_set, super_set)

    @classmethod
    def bitlist_from_subset(self, subset, superset):
        """
        Gets the bitlist corresponding to a subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> Subset.bitlist_from_subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        '0011'

        See Also
        ========

        subset_from_bitlist
        """
        bitlist = ['0'] * len(superset)
        if isinstance(subset, Subset):
            subset = subset.subset
        for i in Subset.subset_indices(subset, superset):
            bitlist[i] = '1'
        return ''.join(bitlist)

    @classmethod
    def unrank_binary(self, rank, superset):
        """
        Gets the binary ordered subset of the specified rank.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> Subset.unrank_binary(4, ['a', 'b', 'c', 'd']).subset
        ['b']

        See Also
        ========

        iterate_binary, rank_binary
        """
        bits = bin(rank)[2:].rjust(len(superset), '0')
        return Subset.subset_from_bitlist(superset, bits)

    @classmethod
    def unrank_gray(self, rank, superset):
        """
        Gets the Gray code ordered subset of the specified rank.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> Subset.unrank_gray(4, ['a', 'b', 'c']).subset
        ['a', 'b']
        >>> Subset.unrank_gray(0, ['a', 'b', 'c']).subset
        []

        See Also
        ========

        iterate_graycode, rank_gray
        """
        graycode_bitlist = GrayCode.unrank(len(superset), rank)
        return Subset.subset_from_bitlist(superset, graycode_bitlist)

    @classmethod
    def subset_indices(self, subset, superset):
        """Return indices of subset in superset in a list; the list is empty
        if all elements of ``subset`` are not in ``superset``.

        Examples
        ========

            >>> from sympy.combinatorics import Subset
            >>> superset = [1, 3, 2, 5, 4]
            >>> Subset.subset_indices([3, 2, 1], superset)
            [1, 2, 0]
            >>> Subset.subset_indices([1, 6], superset)
            []
            >>> Subset.subset_indices([], superset)
            []

        """
        a, b = superset, subset
        sb = set(b)
        d = {}
        for i, ai in enumerate(a):
            if ai in sb:
                d[ai] = i
                sb.remove(ai)
                if not sb:
                    break
        else:
            return []
        return [d[bi] for bi in b]


def ksubsets(superset, k):
    """
    Finds the subsets of size ``k`` in lexicographic order.

    This uses the itertools generator.

    Examples
    ========

    >>> from sympy.combinatorics.subsets import ksubsets
    >>> list(ksubsets([1, 2, 3], 2))
    [(1, 2), (1, 3), (2, 3)]
    >>> list(ksubsets([1, 2, 3, 4, 5], 2))
    [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), \
    (2, 5), (3, 4), (3, 5), (4, 5)]

    See Also
    ========

    Subset
    """
    return combinations(superset, k)
