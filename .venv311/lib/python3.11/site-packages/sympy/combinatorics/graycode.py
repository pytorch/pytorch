from sympy.core import Basic, Integer

import random


class GrayCode(Basic):
    """
    A Gray code is essentially a Hamiltonian walk on
    a n-dimensional cube with edge length of one.
    The vertices of the cube are represented by vectors
    whose values are binary. The Hamilton walk visits
    each vertex exactly once. The Gray code for a 3d
    cube is ['000','100','110','010','011','111','101',
    '001'].

    A Gray code solves the problem of sequentially
    generating all possible subsets of n objects in such
    a way that each subset is obtained from the previous
    one by either deleting or adding a single object.
    In the above example, 1 indicates that the object is
    present, and 0 indicates that its absent.

    Gray codes have applications in statistics as well when
    we want to compute various statistics related to subsets
    in an efficient manner.

    Examples
    ========

    >>> from sympy.combinatorics import GrayCode
    >>> a = GrayCode(3)
    >>> list(a.generate_gray())
    ['000', '001', '011', '010', '110', '111', '101', '100']
    >>> a = GrayCode(4)
    >>> list(a.generate_gray())
    ['0000', '0001', '0011', '0010', '0110', '0111', '0101', '0100', \
    '1100', '1101', '1111', '1110', '1010', '1011', '1001', '1000']

    References
    ==========

    .. [1] Nijenhuis,A. and Wilf,H.S.(1978).
           Combinatorial Algorithms. Academic Press.
    .. [2] Knuth, D. (2011). The Art of Computer Programming, Vol 4
           Addison Wesley


    """

    _skip = False
    _current = 0
    _rank = None

    def __new__(cls, n, *args, **kw_args):
        """
        Default constructor.

        It takes a single argument ``n`` which gives the dimension of the Gray
        code. The starting Gray code string (``start``) or the starting ``rank``
        may also be given; the default is to start at rank = 0 ('0...0').

        Examples
        ========

        >>> from sympy.combinatorics import GrayCode
        >>> a = GrayCode(3)
        >>> a
        GrayCode(3)
        >>> a.n
        3

        >>> a = GrayCode(3, start='100')
        >>> a.current
        '100'

        >>> a = GrayCode(4, rank=4)
        >>> a.current
        '0110'
        >>> a.rank
        4

        """
        if n < 1 or int(n) != n:
            raise ValueError(
                'Gray code dimension must be a positive integer, not %i' % n)
        n = Integer(n)
        args = (n,) + args
        obj = Basic.__new__(cls, *args)
        if 'start' in kw_args:
            obj._current = kw_args["start"]
            if len(obj._current) > n:
                raise ValueError('Gray code start has length %i but '
                'should not be greater than %i' % (len(obj._current), n))
        elif 'rank' in kw_args:
            if int(kw_args["rank"]) != kw_args["rank"]:
                raise ValueError('Gray code rank must be a positive integer, '
                'not %i' % kw_args["rank"])
            obj._rank = int(kw_args["rank"]) % obj.selections
            obj._current = obj.unrank(n, obj._rank)
        return obj

    def next(self, delta=1):
        """
        Returns the Gray code a distance ``delta`` (default = 1) from the
        current value in canonical order.


        Examples
        ========

        >>> from sympy.combinatorics import GrayCode
        >>> a = GrayCode(3, start='110')
        >>> a.next().current
        '111'
        >>> a.next(-1).current
        '010'
        """
        return GrayCode(self.n, rank=(self.rank + delta) % self.selections)

    @property
    def selections(self):
        """
        Returns the number of bit vectors in the Gray code.

        Examples
        ========

        >>> from sympy.combinatorics import GrayCode
        >>> a = GrayCode(3)
        >>> a.selections
        8
        """
        return 2**self.n

    @property
    def n(self):
        """
        Returns the dimension of the Gray code.

        Examples
        ========

        >>> from sympy.combinatorics import GrayCode
        >>> a = GrayCode(5)
        >>> a.n
        5
        """
        return self.args[0]

    def generate_gray(self, **hints):
        """
        Generates the sequence of bit vectors of a Gray Code.

        Examples
        ========

        >>> from sympy.combinatorics import GrayCode
        >>> a = GrayCode(3)
        >>> list(a.generate_gray())
        ['000', '001', '011', '010', '110', '111', '101', '100']
        >>> list(a.generate_gray(start='011'))
        ['011', '010', '110', '111', '101', '100']
        >>> list(a.generate_gray(rank=4))
        ['110', '111', '101', '100']

        See Also
        ========

        skip

        References
        ==========

        .. [1] Knuth, D. (2011). The Art of Computer Programming,
               Vol 4, Addison Wesley

        """
        bits = self.n
        start = None
        if "start" in hints:
            start = hints["start"]
        elif "rank" in hints:
            start = GrayCode.unrank(self.n, hints["rank"])
        if start is not None:
            self._current = start
        current = self.current
        graycode_bin = gray_to_bin(current)
        if len(graycode_bin) > self.n:
            raise ValueError('Gray code start has length %i but should '
            'not be greater than %i' % (len(graycode_bin), bits))
        self._current = int(current, 2)
        graycode_int = int(''.join(graycode_bin), 2)
        for i in range(graycode_int, 1 << bits):
            if self._skip:
                self._skip = False
            else:
                yield self.current
            bbtc = (i ^ (i + 1))
            gbtc = (bbtc ^ (bbtc >> 1))
            self._current = (self._current ^ gbtc)
        self._current = 0

    def skip(self):
        """
        Skips the bit generation.

        Examples
        ========

        >>> from sympy.combinatorics import GrayCode
        >>> a = GrayCode(3)
        >>> for i in a.generate_gray():
        ...     if i == '010':
        ...         a.skip()
        ...     print(i)
        ...
        000
        001
        011
        010
        111
        101
        100

        See Also
        ========

        generate_gray
        """
        self._skip = True

    @property
    def rank(self):
        """
        Ranks the Gray code.

        A ranking algorithm determines the position (or rank)
        of a combinatorial object among all the objects w.r.t.
        a given order. For example, the 4 bit binary reflected
        Gray code (BRGC) '0101' has a rank of 6 as it appears in
        the 6th position in the canonical ordering of the family
        of 4 bit Gray codes.

        Examples
        ========

        >>> from sympy.combinatorics import GrayCode
        >>> a = GrayCode(3)
        >>> list(a.generate_gray())
        ['000', '001', '011', '010', '110', '111', '101', '100']
        >>> GrayCode(3, start='100').rank
        7
        >>> GrayCode(3, rank=7).current
        '100'

        See Also
        ========

        unrank

        References
        ==========

        .. [1] https://web.archive.org/web/20200224064753/http://statweb.stanford.edu/~susan/courses/s208/node12.html

        """
        if self._rank is None:
            self._rank = int(gray_to_bin(self.current), 2)
        return self._rank

    @property
    def current(self):
        """
        Returns the currently referenced Gray code as a bit string.

        Examples
        ========

        >>> from sympy.combinatorics import GrayCode
        >>> GrayCode(3, start='100').current
        '100'
        """
        rv = self._current or '0'
        if not isinstance(rv, str):
            rv = bin(rv)[2:]
        return rv.rjust(self.n, '0')

    @classmethod
    def unrank(self, n, rank):
        """
        Unranks an n-bit sized Gray code of rank k. This method exists
        so that a derivative GrayCode class can define its own code of
        a given rank.

        The string here is generated in reverse order to allow for tail-call
        optimization.

        Examples
        ========

        >>> from sympy.combinatorics import GrayCode
        >>> GrayCode(5, rank=3).current
        '00010'
        >>> GrayCode.unrank(5, 3)
        '00010'

        See Also
        ========

        rank
        """
        def _unrank(k, n):
            if n == 1:
                return str(k % 2)
            m = 2**(n - 1)
            if k < m:
                return '0' + _unrank(k, n - 1)
            return '1' + _unrank(m - (k % m) - 1, n - 1)
        return _unrank(rank, n)


def random_bitstring(n):
    """
    Generates a random bitlist of length n.

    Examples
    ========

    >>> from sympy.combinatorics.graycode import random_bitstring
    >>> random_bitstring(3) # doctest: +SKIP
    100
    """
    return ''.join([random.choice('01') for i in range(n)])


def gray_to_bin(bin_list):
    """
    Convert from Gray coding to binary coding.

    We assume big endian encoding.

    Examples
    ========

    >>> from sympy.combinatorics.graycode import gray_to_bin
    >>> gray_to_bin('100')
    '111'

    See Also
    ========

    bin_to_gray
    """
    b = [bin_list[0]]
    for i in range(1, len(bin_list)):
        b += str(int(b[i - 1] != bin_list[i]))
    return ''.join(b)


def bin_to_gray(bin_list):
    """
    Convert from binary coding to gray coding.

    We assume big endian encoding.

    Examples
    ========

    >>> from sympy.combinatorics.graycode import bin_to_gray
    >>> bin_to_gray('111')
    '100'

    See Also
    ========

    gray_to_bin
    """
    b = [bin_list[0]]
    for i in range(1, len(bin_list)):
        b += str(int(bin_list[i]) ^ int(bin_list[i - 1]))
    return ''.join(b)


def get_subset_from_bitstring(super_set, bitstring):
    """
    Gets the subset defined by the bitstring.

    Examples
    ========

    >>> from sympy.combinatorics.graycode import get_subset_from_bitstring
    >>> get_subset_from_bitstring(['a', 'b', 'c', 'd'], '0011')
    ['c', 'd']
    >>> get_subset_from_bitstring(['c', 'a', 'c', 'c'], '1100')
    ['c', 'a']

    See Also
    ========

    graycode_subsets
    """
    if len(super_set) != len(bitstring):
        raise ValueError("The sizes of the lists are not equal")
    return [super_set[i] for i, j in enumerate(bitstring)
            if bitstring[i] == '1']


def graycode_subsets(gray_code_set):
    """
    Generates the subsets as enumerated by a Gray code.

    Examples
    ========

    >>> from sympy.combinatorics.graycode import graycode_subsets
    >>> list(graycode_subsets(['a', 'b', 'c']))
    [[], ['c'], ['b', 'c'], ['b'], ['a', 'b'], ['a', 'b', 'c'], \
    ['a', 'c'], ['a']]
    >>> list(graycode_subsets(['a', 'b', 'c', 'c']))
    [[], ['c'], ['c', 'c'], ['c'], ['b', 'c'], ['b', 'c', 'c'], \
    ['b', 'c'], ['b'], ['a', 'b'], ['a', 'b', 'c'], ['a', 'b', 'c', 'c'], \
    ['a', 'b', 'c'], ['a', 'c'], ['a', 'c', 'c'], ['a', 'c'], ['a']]

    See Also
    ========

    get_subset_from_bitstring
    """
    for bitstring in list(GrayCode(len(gray_code_set)).generate_gray()):
        yield get_subset_from_bitstring(gray_code_set, bitstring)
