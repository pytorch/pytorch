from collections import Counter, defaultdict, OrderedDict
from itertools import (
    chain, combinations, combinations_with_replacement, cycle, islice,
    permutations, product, groupby
)
# For backwards compatibility
from itertools import product as cartes # noqa: F401
from operator import gt



# this is the logical location of these functions
from sympy.utilities.enumerative import (
    multiset_partitions_taocp, list_visitor, MultisetPartitionTraverser)

from sympy.utilities.misc import as_int
from sympy.utilities.decorator import deprecated


def is_palindromic(s, i=0, j=None):
    """
    Return True if the sequence is the same from left to right as it
    is from right to left in the whole sequence (default) or in the
    Python slice ``s[i: j]``; else False.

    Examples
    ========

    >>> from sympy.utilities.iterables import is_palindromic
    >>> is_palindromic([1, 0, 1])
    True
    >>> is_palindromic('abcbb')
    False
    >>> is_palindromic('abcbb', 1)
    False

    Normal Python slicing is performed in place so there is no need to
    create a slice of the sequence for testing:

    >>> is_palindromic('abcbb', 1, -1)
    True
    >>> is_palindromic('abcbb', -4, -1)
    True

    See Also
    ========

    sympy.ntheory.digits.is_palindromic: tests integers

    """
    i, j, _ = slice(i, j).indices(len(s))
    m = (j - i)//2
    # if length is odd, middle element will be ignored
    return all(s[i + k] == s[j - 1 - k] for k in range(m))


def flatten(iterable, levels=None, cls=None):  # noqa: F811
    """
    Recursively denest iterable containers.

    >>> from sympy import flatten

    >>> flatten([1, 2, 3])
    [1, 2, 3]
    >>> flatten([1, 2, [3]])
    [1, 2, 3]
    >>> flatten([1, [2, 3], [4, 5]])
    [1, 2, 3, 4, 5]
    >>> flatten([1.0, 2, (1, None)])
    [1.0, 2, 1, None]

    If you want to denest only a specified number of levels of
    nested containers, then set ``levels`` flag to the desired
    number of levels::

    >>> ls = [[(-2, -1), (1, 2)], [(0, 0)]]

    >>> flatten(ls, levels=1)
    [(-2, -1), (1, 2), (0, 0)]

    If cls argument is specified, it will only flatten instances of that
    class, for example:

    >>> from sympy import Basic, S
    >>> class MyOp(Basic):
    ...     pass
    ...
    >>> flatten([MyOp(S(1), MyOp(S(2), S(3)))], cls=MyOp)
    [1, 2, 3]

    adapted from https://kogs-www.informatik.uni-hamburg.de/~meine/python_tricks
    """
    from sympy.tensor.array import NDimArray
    if levels is not None:
        if not levels:
            return iterable
        elif levels > 0:
            levels -= 1
        else:
            raise ValueError(
                "expected non-negative number of levels, got %s" % levels)

    if cls is None:
        def reducible(x):
            return is_sequence(x, set)
    else:
        def reducible(x):
            return isinstance(x, cls)

    result = []

    for el in iterable:
        if reducible(el):
            if hasattr(el, 'args') and not isinstance(el, NDimArray):
                el = el.args
            result.extend(flatten(el, levels=levels, cls=cls))
        else:
            result.append(el)

    return result


def unflatten(iter, n=2):
    """Group ``iter`` into tuples of length ``n``. Raise an error if
    the length of ``iter`` is not a multiple of ``n``.
    """
    if n < 1 or len(iter) % n:
        raise ValueError('iter length is not a multiple of %i' % n)
    return list(zip(*(iter[i::n] for i in range(n))))


def reshape(seq, how):
    """Reshape the sequence according to the template in ``how``.

    Examples
    ========

    >>> from sympy.utilities import reshape
    >>> seq = list(range(1, 9))

    >>> reshape(seq, [4]) # lists of 4
    [[1, 2, 3, 4], [5, 6, 7, 8]]

    >>> reshape(seq, (4,)) # tuples of 4
    [(1, 2, 3, 4), (5, 6, 7, 8)]

    >>> reshape(seq, (2, 2)) # tuples of 4
    [(1, 2, 3, 4), (5, 6, 7, 8)]

    >>> reshape(seq, (2, [2])) # (i, i, [i, i])
    [(1, 2, [3, 4]), (5, 6, [7, 8])]

    >>> reshape(seq, ((2,), [2])) # etc....
    [((1, 2), [3, 4]), ((5, 6), [7, 8])]

    >>> reshape(seq, (1, [2], 1))
    [(1, [2, 3], 4), (5, [6, 7], 8)]

    >>> reshape(tuple(seq), ([[1], 1, (2,)],))
    (([[1], 2, (3, 4)],), ([[5], 6, (7, 8)],))

    >>> reshape(tuple(seq), ([1], 1, (2,)))
    (([1], 2, (3, 4)), ([5], 6, (7, 8)))

    >>> reshape(list(range(12)), [2, [3], {2}, (1, (3,), 1)])
    [[0, 1, [2, 3, 4], {5, 6}, (7, (8, 9, 10), 11)]]

    """
    m = sum(flatten(how))
    n, rem = divmod(len(seq), m)
    if m < 0 or rem:
        raise ValueError('template must sum to positive number '
        'that divides the length of the sequence')
    i = 0
    container = type(how)
    rv = [None]*n
    for k in range(len(rv)):
        _rv = []
        for hi in how:
            if isinstance(hi, int):
                _rv.extend(seq[i: i + hi])
                i += hi
            else:
                n = sum(flatten(hi))
                hi_type = type(hi)
                _rv.append(hi_type(reshape(seq[i: i + n], hi)[0]))
                i += n
        rv[k] = container(_rv)
    return type(seq)(rv)


def group(seq, multiple=True):
    """
    Splits a sequence into a list of lists of equal, adjacent elements.

    Examples
    ========

    >>> from sympy import group

    >>> group([1, 1, 1, 2, 2, 3])
    [[1, 1, 1], [2, 2], [3]]
    >>> group([1, 1, 1, 2, 2, 3], multiple=False)
    [(1, 3), (2, 2), (3, 1)]
    >>> group([1, 1, 3, 2, 2, 1], multiple=False)
    [(1, 2), (3, 1), (2, 2), (1, 1)]

    See Also
    ========

    multiset

    """
    if multiple:
        return [(list(g)) for _, g in groupby(seq)]
    return [(k, len(list(g))) for k, g in groupby(seq)]


def _iproduct2(iterable1, iterable2):
    '''Cartesian product of two possibly infinite iterables'''

    it1 = iter(iterable1)
    it2 = iter(iterable2)

    elems1 = []
    elems2 = []

    sentinel = object()
    def append(it, elems):
        e = next(it, sentinel)
        if e is not sentinel:
            elems.append(e)

    n = 0
    append(it1, elems1)
    append(it2, elems2)

    while n <= len(elems1) + len(elems2):
        for m in range(n-len(elems1)+1, len(elems2)):
            yield (elems1[n-m], elems2[m])
        n += 1
        append(it1, elems1)
        append(it2, elems2)


def iproduct(*iterables):
    '''
    Cartesian product of iterables.

    Generator of the Cartesian product of iterables. This is analogous to
    itertools.product except that it works with infinite iterables and will
    yield any item from the infinite product eventually.

    Examples
    ========

    >>> from sympy.utilities.iterables import iproduct
    >>> sorted(iproduct([1,2], [3,4]))
    [(1, 3), (1, 4), (2, 3), (2, 4)]

    With an infinite iterator:

    >>> from sympy import S
    >>> (3,) in iproduct(S.Integers)
    True
    >>> (3, 4) in iproduct(S.Integers, S.Integers)
    True

    .. seealso::

       `itertools.product
       <https://docs.python.org/3/library/itertools.html#itertools.product>`_
    '''
    if len(iterables) == 0:
        yield ()
        return
    elif len(iterables) == 1:
        for e in iterables[0]:
            yield (e,)
    elif len(iterables) == 2:
        yield from _iproduct2(*iterables)
    else:
        first, others = iterables[0], iterables[1:]
        for ef, eo in _iproduct2(first, iproduct(*others)):
            yield (ef,) + eo


def multiset(seq):
    """Return the hashable sequence in multiset form with values being the
    multiplicity of the item in the sequence.

    Examples
    ========

    >>> from sympy.utilities.iterables import multiset
    >>> multiset('mississippi')
    {'i': 4, 'm': 1, 'p': 2, 's': 4}

    See Also
    ========

    group

    """
    return dict(Counter(seq).items())




def ibin(n, bits=None, str=False):
    """Return a list of length ``bits`` corresponding to the binary value
    of ``n`` with small bits to the right (last). If bits is omitted, the
    length will be the number required to represent ``n``. If the bits are
    desired in reversed order, use the ``[::-1]`` slice of the returned list.

    If a sequence of all bits-length lists starting from ``[0, 0,..., 0]``
    through ``[1, 1, ..., 1]`` are desired, pass a non-integer for bits, e.g.
    ``'all'``.

    If the bit *string* is desired pass ``str=True``.

    Examples
    ========

    >>> from sympy.utilities.iterables import ibin
    >>> ibin(2)
    [1, 0]
    >>> ibin(2, 4)
    [0, 0, 1, 0]

    If all lists corresponding to 0 to 2**n - 1, pass a non-integer
    for bits:

    >>> bits = 2
    >>> for i in ibin(2, 'all'):
    ...     print(i)
    (0, 0)
    (0, 1)
    (1, 0)
    (1, 1)

    If a bit string is desired of a given length, use str=True:

    >>> n = 123
    >>> bits = 10
    >>> ibin(n, bits, str=True)
    '0001111011'
    >>> ibin(n, bits, str=True)[::-1]  # small bits left
    '1101111000'
    >>> list(ibin(3, 'all', str=True))
    ['000', '001', '010', '011', '100', '101', '110', '111']

    """
    if n < 0:
        raise ValueError("negative numbers are not allowed")
    n = as_int(n)

    if bits is None:
        bits = 0
    else:
        try:
            bits = as_int(bits)
        except ValueError:
            bits = -1
        else:
            if n.bit_length() > bits:
                raise ValueError(
                    "`bits` must be >= {}".format(n.bit_length()))

    if not str:
        if bits >= 0:
            return [1 if i == "1" else 0 for i in bin(n)[2:].rjust(bits, "0")]
        else:
            return variations(range(2), n, repetition=True)
    else:
        if bits >= 0:
            return bin(n)[2:].rjust(bits, "0")
        else:
            return (bin(i)[2:].rjust(n, "0") for i in range(2**n))


def variations(seq, n, repetition=False):
    r"""Returns an iterator over the n-sized variations of ``seq`` (size N).
    ``repetition`` controls whether items in ``seq`` can appear more than once;

    Examples
    ========

    ``variations(seq, n)`` will return `\frac{N!}{(N - n)!}` permutations without
    repetition of ``seq``'s elements:

        >>> from sympy import variations
        >>> list(variations([1, 2], 2))
        [(1, 2), (2, 1)]

    ``variations(seq, n, True)`` will return the `N^n` permutations obtained
    by allowing repetition of elements:

        >>> list(variations([1, 2], 2, repetition=True))
        [(1, 1), (1, 2), (2, 1), (2, 2)]

    If you ask for more items than are in the set you get the empty set unless
    you allow repetitions:

        >>> list(variations([0, 1], 3, repetition=False))
        []
        >>> list(variations([0, 1], 3, repetition=True))[:4]
        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]

    .. seealso::

       `itertools.permutations
       <https://docs.python.org/3/library/itertools.html#itertools.permutations>`_,
       `itertools.product
       <https://docs.python.org/3/library/itertools.html#itertools.product>`_
    """
    if not repetition:
        seq = tuple(seq)
        if len(seq) < n:
            return iter(())  # 0 length iterator
        return permutations(seq, n)
    else:
        if n == 0:
            return iter(((),))  # yields 1 empty tuple
        else:
            return product(seq, repeat=n)


def subsets(seq, k=None, repetition=False):
    r"""Generates all `k`-subsets (combinations) from an `n`-element set, ``seq``.

    A `k`-subset of an `n`-element set is any subset of length exactly `k`. The
    number of `k`-subsets of an `n`-element set is given by ``binomial(n, k)``,
    whereas there are `2^n` subsets all together. If `k` is ``None`` then all
    `2^n` subsets will be returned from shortest to longest.

    Examples
    ========

    >>> from sympy import subsets

    ``subsets(seq, k)`` will return the
    `\frac{n!}{k!(n - k)!}` `k`-subsets (combinations)
    without repetition, i.e. once an item has been removed, it can no
    longer be "taken":

        >>> list(subsets([1, 2], 2))
        [(1, 2)]
        >>> list(subsets([1, 2]))
        [(), (1,), (2,), (1, 2)]
        >>> list(subsets([1, 2, 3], 2))
        [(1, 2), (1, 3), (2, 3)]


    ``subsets(seq, k, repetition=True)`` will return the
    `\frac{(n - 1 + k)!}{k!(n - 1)!}`
    combinations *with* repetition:

        >>> list(subsets([1, 2], 2, repetition=True))
        [(1, 1), (1, 2), (2, 2)]

    If you ask for more items than are in the set you get the empty set unless
    you allow repetitions:

        >>> list(subsets([0, 1], 3, repetition=False))
        []
        >>> list(subsets([0, 1], 3, repetition=True))
        [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)]

    """
    if k is None:
        if not repetition:
            return chain.from_iterable((combinations(seq, k)
                                        for k in range(len(seq) + 1)))
        else:
            return chain.from_iterable((combinations_with_replacement(seq, k)
                                        for k in range(len(seq) + 1)))
    else:
        if not repetition:
            return combinations(seq, k)
        else:
            return combinations_with_replacement(seq, k)


def filter_symbols(iterator, exclude):
    """
    Only yield elements from `iterator` that do not occur in `exclude`.

    Parameters
    ==========

    iterator : iterable
        iterator to take elements from

    exclude : iterable
        elements to exclude

    Returns
    =======

    iterator : iterator
        filtered iterator
    """
    exclude = set(exclude)
    for s in iterator:
        if s not in exclude:
            yield s

def numbered_symbols(prefix='x', cls=None, start=0, exclude=(), *args, **assumptions):
    """
    Generate an infinite stream of Symbols consisting of a prefix and
    increasing subscripts provided that they do not occur in ``exclude``.

    Parameters
    ==========

    prefix : str, optional
        The prefix to use. By default, this function will generate symbols of
        the form "x0", "x1", etc.

    cls : class, optional
        The class to use. By default, it uses ``Symbol``, but you can also use ``Wild``
        or ``Dummy``.

    start : int, optional
        The start number.  By default, it is 0.

    exclude : list, tuple, set of cls, optional
        Symbols to be excluded.

    *args, **kwargs
        Additional positional and keyword arguments are passed to the *cls* class.

    Returns
    =======

    sym : Symbol
        The subscripted symbols.
    """
    exclude = set(exclude or [])
    if cls is None:
        # We can't just make the default cls=Symbol because it isn't
        # imported yet.
        from sympy.core import Symbol
        cls = Symbol

    while True:
        name = '%s%s' % (prefix, start)
        s = cls(name, *args, **assumptions)
        if s not in exclude:
            yield s
        start += 1


def capture(func):
    """Return the printed output of func().

    ``func`` should be a function without arguments that produces output with
    print statements.

    >>> from sympy.utilities.iterables import capture
    >>> from sympy import pprint
    >>> from sympy.abc import x
    >>> def foo():
    ...     print('hello world!')
    ...
    >>> 'hello' in capture(foo) # foo, not foo()
    True
    >>> capture(lambda: pprint(2/x))
    '2\\n-\\nx\\n'

    """
    from io import StringIO
    import sys

    stdout = sys.stdout
    sys.stdout = file = StringIO()
    try:
        func()
    finally:
        sys.stdout = stdout
    return file.getvalue()


def sift(seq, keyfunc, binary=False):
    """
    Sift the sequence, ``seq`` according to ``keyfunc``.

    Returns
    =======

    When ``binary`` is ``False`` (default), the output is a dictionary
    where elements of ``seq`` are stored in a list keyed to the value
    of keyfunc for that element. If ``binary`` is True then a tuple
    with lists ``T`` and ``F`` are returned where ``T`` is a list
    containing elements of seq for which ``keyfunc`` was ``True`` and
    ``F`` containing those elements for which ``keyfunc`` was ``False``;
    a ValueError is raised if the ``keyfunc`` is not binary.

    Examples
    ========

    >>> from sympy.utilities import sift
    >>> from sympy.abc import x, y
    >>> from sympy import sqrt, exp, pi, Tuple

    >>> sift(range(5), lambda x: x % 2)
    {0: [0, 2, 4], 1: [1, 3]}

    sift() returns a defaultdict() object, so any key that has no matches will
    give [].

    >>> sift([x], lambda x: x.is_commutative)
    {True: [x]}
    >>> _[False]
    []

    Sometimes you will not know how many keys you will get:

    >>> sift([sqrt(x), exp(x), (y**x)**2],
    ...      lambda x: x.as_base_exp()[0])
    {E: [exp(x)], x: [sqrt(x)], y: [y**(2*x)]}

    Sometimes you expect the results to be binary; the
    results can be unpacked by setting ``binary`` to True:

    >>> sift(range(4), lambda x: x % 2, binary=True)
    ([1, 3], [0, 2])
    >>> sift(Tuple(1, pi), lambda x: x.is_rational, binary=True)
    ([1], [pi])

    A ValueError is raised if the predicate was not actually binary
    (which is a good test for the logic where sifting is used and
    binary results were expected):

    >>> unknown = exp(1) - pi  # the rationality of this is unknown
    >>> args = Tuple(1, pi, unknown)
    >>> sift(args, lambda x: x.is_rational, binary=True)
    Traceback (most recent call last):
    ...
    ValueError: keyfunc gave non-binary output

    The non-binary sifting shows that there were 3 keys generated:

    >>> set(sift(args, lambda x: x.is_rational).keys())
    {None, False, True}

    If you need to sort the sifted items it might be better to use
    ``ordered`` which can economically apply multiple sort keys
    to a sequence while sorting.

    See Also
    ========

    ordered

    """
    if not binary:
        m = defaultdict(list)
        for i in seq:
            m[keyfunc(i)].append(i)
        return m
    sift = F, T = [], []
    for i in seq:
        try:
            sift[keyfunc(i)].append(i)
        except (IndexError, TypeError):
            raise ValueError('keyfunc gave non-binary output')
    return T, F


def take(iter, n):
    """Return ``n`` items from ``iter`` iterator. """
    return [ value for _, value in zip(range(n), iter) ]


def dict_merge(*dicts):
    """Merge dictionaries into a single dictionary. """
    merged = {}

    for dict in dicts:
        merged.update(dict)

    return merged


def common_prefix(*seqs):
    """Return the subsequence that is a common start of sequences in ``seqs``.

    >>> from sympy.utilities.iterables import common_prefix
    >>> common_prefix(list(range(3)))
    [0, 1, 2]
    >>> common_prefix(list(range(3)), list(range(4)))
    [0, 1, 2]
    >>> common_prefix([1, 2, 3], [1, 2, 5])
    [1, 2]
    >>> common_prefix([1, 2, 3], [1, 3, 5])
    [1]
    """
    if not all(seqs):
        return []
    elif len(seqs) == 1:
        return seqs[0]
    i = 0
    for i in range(min(len(s) for s in seqs)):
        if not all(seqs[j][i] == seqs[0][i] for j in range(len(seqs))):
            break
    else:
        i += 1
    return seqs[0][:i]


def common_suffix(*seqs):
    """Return the subsequence that is a common ending of sequences in ``seqs``.

    >>> from sympy.utilities.iterables import common_suffix
    >>> common_suffix(list(range(3)))
    [0, 1, 2]
    >>> common_suffix(list(range(3)), list(range(4)))
    []
    >>> common_suffix([1, 2, 3], [9, 2, 3])
    [2, 3]
    >>> common_suffix([1, 2, 3], [9, 7, 3])
    [3]
    """

    if not all(seqs):
        return []
    elif len(seqs) == 1:
        return seqs[0]
    i = 0
    for i in range(-1, -min(len(s) for s in seqs) - 1, -1):
        if not all(seqs[j][i] == seqs[0][i] for j in range(len(seqs))):
            break
    else:
        i -= 1
    if i == -1:
        return []
    else:
        return seqs[0][i + 1:]


def prefixes(seq):
    """
    Generate all prefixes of a sequence.

    Examples
    ========

    >>> from sympy.utilities.iterables import prefixes

    >>> list(prefixes([1,2,3,4]))
    [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]]

    """
    n = len(seq)

    for i in range(n):
        yield seq[:i + 1]


def postfixes(seq):
    """
    Generate all postfixes of a sequence.

    Examples
    ========

    >>> from sympy.utilities.iterables import postfixes

    >>> list(postfixes([1,2,3,4]))
    [[4], [3, 4], [2, 3, 4], [1, 2, 3, 4]]

    """
    n = len(seq)

    for i in range(n):
        yield seq[n - i - 1:]


def topological_sort(graph, key=None):
    r"""
    Topological sort of graph's vertices.

    Parameters
    ==========

    graph : tuple[list, list[tuple[T, T]]
        A tuple consisting of a list of vertices and a list of edges of
        a graph to be sorted topologically.

    key : callable[T] (optional)
        Ordering key for vertices on the same level. By default the natural
        (e.g. lexicographic) ordering is used (in this case the base type
        must implement ordering relations).

    Examples
    ========

    Consider a graph::

        +---+     +---+     +---+
        | 7 |\    | 5 |     | 3 |
        +---+ \   +---+     +---+
          |   _\___/ ____   _/ |
          |  /  \___/    \ /   |
          V  V           V V   |
         +----+         +---+  |
         | 11 |         | 8 |  |
         +----+         +---+  |
          | | \____   ___/ _   |
          | \      \ /    / \  |
          V  \     V V   /  V  V
        +---+ \   +---+ |  +----+
        | 2 |  |  | 9 | |  | 10 |
        +---+  |  +---+ |  +----+
               \________/

    where vertices are integers. This graph can be encoded using
    elementary Python's data structures as follows::

        >>> V = [2, 3, 5, 7, 8, 9, 10, 11]
        >>> E = [(7, 11), (7, 8), (5, 11), (3, 8), (3, 10),
        ...      (11, 2), (11, 9), (11, 10), (8, 9)]

    To compute a topological sort for graph ``(V, E)`` issue::

        >>> from sympy.utilities.iterables import topological_sort

        >>> topological_sort((V, E))
        [3, 5, 7, 8, 11, 2, 9, 10]

    If specific tie breaking approach is needed, use ``key`` parameter::

        >>> topological_sort((V, E), key=lambda v: -v)
        [7, 5, 11, 3, 10, 8, 9, 2]

    Only acyclic graphs can be sorted. If the input graph has a cycle,
    then ``ValueError`` will be raised::

        >>> topological_sort((V, E + [(10, 7)]))
        Traceback (most recent call last):
        ...
        ValueError: cycle detected

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Topological_sorting

    """
    V, E = graph

    L = []
    S = set(V)
    E = list(E)

    S.difference_update(u for v, u in E)

    if key is None:
        def key(value):
            return value

    S = sorted(S, key=key, reverse=True)

    while S:
        node = S.pop()
        L.append(node)

        for u, v in list(E):
            if u == node:
                E.remove((u, v))

                for _u, _v in E:
                    if v == _v:
                        break
                else:
                    kv = key(v)

                    for i, s in enumerate(S):
                        ks = key(s)

                        if kv > ks:
                            S.insert(i, v)
                            break
                    else:
                        S.append(v)

    if E:
        raise ValueError("cycle detected")
    else:
        return L


def strongly_connected_components(G):
    r"""
    Strongly connected components of a directed graph in reverse topological
    order.


    Parameters
    ==========

    G : tuple[list, list[tuple[T, T]]
        A tuple consisting of a list of vertices and a list of edges of
        a graph whose strongly connected components are to be found.


    Examples
    ========

    Consider a directed graph (in dot notation)::

        digraph {
            A -> B
            A -> C
            B -> C
            C -> B
            B -> D
        }

    .. graphviz::

        digraph {
            A -> B
            A -> C
            B -> C
            C -> B
            B -> D
        }

    where vertices are the letters A, B, C and D. This graph can be encoded
    using Python's elementary data structures as follows::

        >>> V = ['A', 'B', 'C', 'D']
        >>> E = [('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'B'), ('B', 'D')]

    The strongly connected components of this graph can be computed as

        >>> from sympy.utilities.iterables import strongly_connected_components

        >>> strongly_connected_components((V, E))
        [['D'], ['B', 'C'], ['A']]

    This also gives the components in reverse topological order.

    Since the subgraph containing B and C has a cycle they must be together in
    a strongly connected component. A and D are connected to the rest of the
    graph but not in a cyclic manner so they appear as their own strongly
    connected components.


    Notes
    =====

    The vertices of the graph must be hashable for the data structures used.
    If the vertices are unhashable replace them with integer indices.

    This function uses Tarjan's algorithm to compute the strongly connected
    components in `O(|V|+|E|)` (linear) time.


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Strongly_connected_component
    .. [2] https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm


    See Also
    ========

    sympy.utilities.iterables.connected_components

    """
    # Map from a vertex to its neighbours
    V, E = G
    Gmap = {vi: [] for vi in V}
    for v1, v2 in E:
        Gmap[v1].append(v2)
    return _strongly_connected_components(V, Gmap)


def _strongly_connected_components(V, Gmap):
    """More efficient internal routine for strongly_connected_components"""
    #
    # Here V is an iterable of vertices and Gmap is a dict mapping each vertex
    # to a list of neighbours e.g.:
    #
    #   V = [0, 1, 2, 3]
    #   Gmap = {0: [2, 3], 1: [0]}
    #
    # For a large graph these data structures can often be created more
    # efficiently then those expected by strongly_connected_components() which
    # in this case would be
    #
    #   V = [0, 1, 2, 3]
    #   Gmap = [(0, 2), (0, 3), (1, 0)]
    #
    # XXX: Maybe this should be the recommended function to use instead...
    #

    # Non-recursive Tarjan's algorithm:
    lowlink = {}
    indices = {}
    stack = OrderedDict()
    callstack = []
    components = []
    nomore = object()

    def start(v):
        index = len(stack)
        indices[v] = lowlink[v] = index
        stack[v] = None
        callstack.append((v, iter(Gmap[v])))

    def finish(v1):
        # Finished a component?
        if lowlink[v1] == indices[v1]:
            component = [stack.popitem()[0]]
            while component[-1] is not v1:
                component.append(stack.popitem()[0])
            components.append(component[::-1])
        v2, _ = callstack.pop()
        if callstack:
            v1, _ = callstack[-1]
            lowlink[v1] = min(lowlink[v1], lowlink[v2])

    for v in V:
        if v in indices:
            continue
        start(v)
        while callstack:
            v1, it1 = callstack[-1]
            v2 = next(it1, nomore)
            # Finished children of v1?
            if v2 is nomore:
                finish(v1)
            # Recurse on v2
            elif v2 not in indices:
                start(v2)
            elif v2 in stack:
                lowlink[v1] = min(lowlink[v1], indices[v2])

    # Reverse topological sort order:
    return components


def connected_components(G):
    r"""
    Connected components of an undirected graph or weakly connected components
    of a directed graph.


    Parameters
    ==========

    G : tuple[list, list[tuple[T, T]]
        A tuple consisting of a list of vertices and a list of edges of
        a graph whose connected components are to be found.


    Examples
    ========


    Given an undirected graph::

        graph {
            A -- B
            C -- D
        }

    .. graphviz::

        graph {
            A -- B
            C -- D
        }

    We can find the connected components using this function if we include
    each edge in both directions::

        >>> from sympy.utilities.iterables import connected_components

        >>> V = ['A', 'B', 'C', 'D']
        >>> E = [('A', 'B'), ('B', 'A'), ('C', 'D'), ('D', 'C')]
        >>> connected_components((V, E))
        [['A', 'B'], ['C', 'D']]

    The weakly connected components of a directed graph can found the same
    way.


    Notes
    =====

    The vertices of the graph must be hashable for the data structures used.
    If the vertices are unhashable replace them with integer indices.

    This function uses Tarjan's algorithm to compute the connected components
    in `O(|V|+|E|)` (linear) time.


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Component_%28graph_theory%29
    .. [2] https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm


    See Also
    ========

    sympy.utilities.iterables.strongly_connected_components

    """
    # Duplicate edges both ways so that the graph is effectively undirected
    # and return the strongly connected components:
    V, E = G
    E_undirected = []
    for v1, v2 in E:
        E_undirected.extend([(v1, v2), (v2, v1)])
    return strongly_connected_components((V, E_undirected))


def rotate_left(x, y):
    """
    Left rotates a list x by the number of steps specified
    in y.

    Examples
    ========

    >>> from sympy.utilities.iterables import rotate_left
    >>> a = [0, 1, 2]
    >>> rotate_left(a, 1)
    [1, 2, 0]
    """
    if len(x) == 0:
        return []
    y = y % len(x)
    return x[y:] + x[:y]


def rotate_right(x, y):
    """
    Right rotates a list x by the number of steps specified
    in y.

    Examples
    ========

    >>> from sympy.utilities.iterables import rotate_right
    >>> a = [0, 1, 2]
    >>> rotate_right(a, 1)
    [2, 0, 1]
    """
    if len(x) == 0:
        return []
    y = len(x) - y % len(x)
    return x[y:] + x[:y]


def least_rotation(x, key=None):
    '''
    Returns the number of steps of left rotation required to
    obtain lexicographically minimal string/list/tuple, etc.

    Examples
    ========

    >>> from sympy.utilities.iterables import least_rotation, rotate_left
    >>> a = [3, 1, 5, 1, 2]
    >>> least_rotation(a)
    3
    >>> rotate_left(a, _)
    [1, 2, 3, 1, 5]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Lexicographically_minimal_string_rotation

    '''
    from sympy.functions.elementary.miscellaneous import Id
    if key is None: key = Id
    S = x + x      # Concatenate string to it self to avoid modular arithmetic
    f = [-1] * len(S)     # Failure function
    k = 0       # Least rotation of string found so far
    for j in range(1,len(S)):
        sj = S[j]
        i = f[j-k-1]
        while i != -1 and sj != S[k+i+1]:
            if key(sj) < key(S[k+i+1]):
                k = j-i-1
            i = f[i]
        if sj != S[k+i+1]:
            if key(sj) < key(S[k]):
                k = j
            f[j-k] = -1
        else:
            f[j-k] = i+1
    return k


def multiset_combinations(m, n, g=None):
    """
    Return the unique combinations of size ``n`` from multiset ``m``.

    Examples
    ========

    >>> from sympy.utilities.iterables import multiset_combinations
    >>> from itertools import combinations
    >>> [''.join(i) for i in  multiset_combinations('baby', 3)]
    ['abb', 'aby', 'bby']

    >>> def count(f, s): return len(list(f(s, 3)))

    The number of combinations depends on the number of letters; the
    number of unique combinations depends on how the letters are
    repeated.

    >>> s1 = 'abracadabra'
    >>> s2 = 'banana tree'
    >>> count(combinations, s1), count(multiset_combinations, s1)
    (165, 23)
    >>> count(combinations, s2), count(multiset_combinations, s2)
    (165, 54)

    """
    from sympy.core.sorting import ordered
    if g is None:
        if isinstance(m, dict):
            if any(as_int(v) < 0 for v in m.values()):
                raise ValueError('counts cannot be negative')
            N = sum(m.values())
            if n > N:
                return
            g = [[k, m[k]] for k in ordered(m)]
        else:
            m = list(m)
            N = len(m)
            if n > N:
                return
            try:
                m = multiset(m)
                g = [(k, m[k]) for k in ordered(m)]
            except TypeError:
                m = list(ordered(m))
                g = [list(i) for i in group(m, multiple=False)]
        del m
    else:
        # not checking counts since g is intended for internal use
        N = sum(v for k, v in g)
    if n > N or not n:
        yield []
    else:
        for i, (k, v) in enumerate(g):
            if v >= n:
                yield [k]*n
                v = n - 1
            for v in range(min(n, v), 0, -1):
                for j in multiset_combinations(None, n - v, g[i + 1:]):
                    rv = [k]*v + j
                    if len(rv) == n:
                        yield rv

def multiset_permutations(m, size=None, g=None):
    """
    Return the unique permutations of multiset ``m``.

    Examples
    ========

    >>> from sympy.utilities.iterables import multiset_permutations
    >>> from sympy import factorial
    >>> [''.join(i) for i in multiset_permutations('aab')]
    ['aab', 'aba', 'baa']
    >>> factorial(len('banana'))
    720
    >>> len(list(multiset_permutations('banana')))
    60
    """
    from sympy.core.sorting import ordered
    if g is None:
        if isinstance(m, dict):
            if any(as_int(v) < 0 for v in m.values()):
                raise ValueError('counts cannot be negative')
            g = [[k, m[k]] for k in ordered(m)]
        else:
            m = list(ordered(m))
            g = [list(i) for i in group(m, multiple=False)]
        del m
    do = [gi for gi in g if gi[1] > 0]
    SUM = sum(gi[1] for gi in do)
    if not do or size is not None and (size > SUM or size < 1):
        if not do and size is None or size == 0:
            yield []
        return
    elif size == 1:
        for k, v in do:
            yield [k]
    elif len(do) == 1:
        k, v = do[0]
        v = v if size is None else (size if size <= v else 0)
        yield [k for i in range(v)]
    elif all(v == 1 for k, v in do):
        for p in permutations([k for k, v in do], size):
            yield list(p)
    else:
        size = size if size is not None else SUM
        for i, (k, v) in enumerate(do):
            do[i][1] -= 1
            for j in multiset_permutations(None, size - 1, do):
                if j:
                    yield [k] + j
            do[i][1] += 1


def _partition(seq, vector, m=None):
    """
    Return the partition of seq as specified by the partition vector.

    Examples
    ========

    >>> from sympy.utilities.iterables import _partition
    >>> _partition('abcde', [1, 0, 1, 2, 0])
    [['b', 'e'], ['a', 'c'], ['d']]

    Specifying the number of bins in the partition is optional:

    >>> _partition('abcde', [1, 0, 1, 2, 0], 3)
    [['b', 'e'], ['a', 'c'], ['d']]

    The output of _set_partitions can be passed as follows:

    >>> output = (3, [1, 0, 1, 2, 0])
    >>> _partition('abcde', *output)
    [['b', 'e'], ['a', 'c'], ['d']]

    See Also
    ========

    combinatorics.partitions.Partition.from_rgs

    """
    if m is None:
        m = max(vector) + 1
    elif isinstance(vector, int):  # entered as m, vector
        vector, m = m, vector
    p = [[] for i in range(m)]
    for i, v in enumerate(vector):
        p[v].append(seq[i])
    return p


def _set_partitions(n):
    """Cycle through all partitions of n elements, yielding the
    current number of partitions, ``m``, and a mutable list, ``q``
    such that ``element[i]`` is in part ``q[i]`` of the partition.

    NOTE: ``q`` is modified in place and generally should not be changed
    between function calls.

    Examples
    ========

    >>> from sympy.utilities.iterables import _set_partitions, _partition
    >>> for m, q in _set_partitions(3):
    ...     print('%s %s %s' % (m, q, _partition('abc', q, m)))
    1 [0, 0, 0] [['a', 'b', 'c']]
    2 [0, 0, 1] [['a', 'b'], ['c']]
    2 [0, 1, 0] [['a', 'c'], ['b']]
    2 [0, 1, 1] [['a'], ['b', 'c']]
    3 [0, 1, 2] [['a'], ['b'], ['c']]

    Notes
    =====

    This algorithm is similar to, and solves the same problem as,
    Algorithm 7.2.1.5H, from volume 4A of Knuth's The Art of Computer
    Programming.  Knuth uses the term "restricted growth string" where
    this code refers to a "partition vector". In each case, the meaning is
    the same: the value in the ith element of the vector specifies to
    which part the ith set element is to be assigned.

    At the lowest level, this code implements an n-digit big-endian
    counter (stored in the array q) which is incremented (with carries) to
    get the next partition in the sequence.  A special twist is that a
    digit is constrained to be at most one greater than the maximum of all
    the digits to the left of it.  The array p maintains this maximum, so
    that the code can efficiently decide when a digit can be incremented
    in place or whether it needs to be reset to 0 and trigger a carry to
    the next digit.  The enumeration starts with all the digits 0 (which
    corresponds to all the set elements being assigned to the same 0th
    part), and ends with 0123...n, which corresponds to each set element
    being assigned to a different, singleton, part.

    This routine was rewritten to use 0-based lists while trying to
    preserve the beauty and efficiency of the original algorithm.

    References
    ==========

    .. [1] Nijenhuis, Albert and Wilf, Herbert. (1978) Combinatorial Algorithms,
        2nd Ed, p 91, algorithm "nexequ". Available online from
        https://www.math.upenn.edu/~wilf/website/CombAlgDownld.html (viewed
        November 17, 2012).

    """
    p = [0]*n
    q = [0]*n
    nc = 1
    yield nc, q
    while nc != n:
        m = n
        while 1:
            m -= 1
            i = q[m]
            if p[i] != 1:
                break
            q[m] = 0
        i += 1
        q[m] = i
        m += 1
        nc += m - n
        p[0] += n - m
        if i == nc:
            p[nc] = 0
            nc += 1
        p[i - 1] -= 1
        p[i] += 1
        yield nc, q


def multiset_partitions(multiset, m=None):
    """
    Return unique partitions of the given multiset (in list form).
    If ``m`` is None, all multisets will be returned, otherwise only
    partitions with ``m`` parts will be returned.

    If ``multiset`` is an integer, a range [0, 1, ..., multiset - 1]
    will be supplied.

    Examples
    ========

    >>> from sympy.utilities.iterables import multiset_partitions
    >>> list(multiset_partitions([1, 2, 3, 4], 2))
    [[[1, 2, 3], [4]], [[1, 2, 4], [3]], [[1, 2], [3, 4]],
    [[1, 3, 4], [2]], [[1, 3], [2, 4]], [[1, 4], [2, 3]],
    [[1], [2, 3, 4]]]
    >>> list(multiset_partitions([1, 2, 3, 4], 1))
    [[[1, 2, 3, 4]]]

    Only unique partitions are returned and these will be returned in a
    canonical order regardless of the order of the input:

    >>> a = [1, 2, 2, 1]
    >>> ans = list(multiset_partitions(a, 2))
    >>> a.sort()
    >>> list(multiset_partitions(a, 2)) == ans
    True
    >>> a = range(3, 1, -1)
    >>> (list(multiset_partitions(a)) ==
    ...  list(multiset_partitions(sorted(a))))
    True

    If m is omitted then all partitions will be returned:

    >>> list(multiset_partitions([1, 1, 2]))
    [[[1, 1, 2]], [[1, 1], [2]], [[1, 2], [1]], [[1], [1], [2]]]
    >>> list(multiset_partitions([1]*3))
    [[[1, 1, 1]], [[1], [1, 1]], [[1], [1], [1]]]

    Counting
    ========

    The number of partitions of a set is given by the bell number:

    >>> from sympy import bell
    >>> len(list(multiset_partitions(5))) == bell(5) == 52
    True

    The number of partitions of length k from a set of size n is given by the
    Stirling Number of the 2nd kind:

    >>> from sympy.functions.combinatorial.numbers import stirling
    >>> stirling(5, 2) == len(list(multiset_partitions(5, 2))) == 15
    True

    These comments on counting apply to *sets*, not multisets.

    Notes
    =====

    When all the elements are the same in the multiset, the order
    of the returned partitions is determined by the ``partitions``
    routine. If one is counting partitions then it is better to use
    the ``nT`` function.

    See Also
    ========

    partitions
    sympy.combinatorics.partitions.Partition
    sympy.combinatorics.partitions.IntegerPartition
    sympy.functions.combinatorial.numbers.nT

    """
    # This function looks at the supplied input and dispatches to
    # several special-case routines as they apply.
    if isinstance(multiset, int):
        n = multiset
        if m and m > n:
            return
        multiset = list(range(n))
        if m == 1:
            yield [multiset[:]]
            return

        # If m is not None, it can sometimes be faster to use
        # MultisetPartitionTraverser.enum_range() even for inputs
        # which are sets.  Since the _set_partitions code is quite
        # fast, this is only advantageous when the overall set
        # partitions outnumber those with the desired number of parts
        # by a large factor.  (At least 60.)  Such a switch is not
        # currently implemented.
        for nc, q in _set_partitions(n):
            if m is None or nc == m:
                rv = [[] for i in range(nc)]
                for i in range(n):
                    rv[q[i]].append(multiset[i])
                yield rv
        return

    if len(multiset) == 1 and isinstance(multiset, str):
        multiset = [multiset]

    if not has_variety(multiset):
        # Only one component, repeated n times.  The resulting
        # partitions correspond to partitions of integer n.
        n = len(multiset)
        if m and m > n:
            return
        if m == 1:
            yield [multiset[:]]
            return
        x = multiset[:1]
        for size, p in partitions(n, m, size=True):
            if m is None or size == m:
                rv = []
                for k in sorted(p):
                    rv.extend([x*k]*p[k])
                yield rv
    else:
        from sympy.core.sorting import ordered
        multiset = list(ordered(multiset))
        n = len(multiset)
        if m and m > n:
            return
        if m == 1:
            yield [multiset[:]]
            return

        # Split the information of the multiset into two lists -
        # one of the elements themselves, and one (of the same length)
        # giving the number of repeats for the corresponding element.
        elements, multiplicities = zip(*group(multiset, False))

        if len(elements) < len(multiset):
            # General case - multiset with more than one distinct element
            # and at least one element repeated more than once.
            if m:
                mpt = MultisetPartitionTraverser()
                for state in mpt.enum_range(multiplicities, m-1, m):
                    yield list_visitor(state, elements)
            else:
                for state in multiset_partitions_taocp(multiplicities):
                    yield list_visitor(state, elements)
        else:
            # Set partitions case - no repeated elements. Pretty much
            # same as int argument case above, with same possible, but
            # currently unimplemented optimization for some cases when
            # m is not None
            for nc, q in _set_partitions(n):
                if m is None or nc == m:
                    rv = [[] for i in range(nc)]
                    for i in range(n):
                        rv[q[i]].append(i)
                    yield [[multiset[j] for j in i] for i in rv]


def partitions(n, m=None, k=None, size=False):
    """Generate all partitions of positive integer, n.

    Each partition is represented as a dictionary, mapping an integer
    to the number of copies of that integer in the partition.  For example,
    the first partition of 4 returned is {4: 1}, "4: one of them".

    Parameters
    ==========
    n : int
    m : int, optional
        limits number of parts in partition (mnemonic: m, maximum parts)
    k : int, optional
        limits the numbers that are kept in the partition (mnemonic: k, keys)
    size : bool, default: False
        If ``True``, (M, P) is returned where M is the sum of the
        multiplicities and P is the generated partition.
        If ``False``, only the generated partition is returned.

    Examples
    ========

    >>> from sympy.utilities.iterables import partitions

    The numbers appearing in the partition (the key of the returned dict)
    are limited with k:

    >>> for p in partitions(6, k=2):  # doctest: +SKIP
    ...     print(p)
    {2: 3}
    {1: 2, 2: 2}
    {1: 4, 2: 1}
    {1: 6}

    The maximum number of parts in the partition (the sum of the values in
    the returned dict) are limited with m (default value, None, gives
    partitions from 1 through n):

    >>> for p in partitions(6, m=2):  # doctest: +SKIP
    ...     print(p)
    ...
    {6: 1}
    {1: 1, 5: 1}
    {2: 1, 4: 1}
    {3: 2}

    References
    ==========

    .. [1] modified from Tim Peter's version to allow for k and m values:
           https://code.activestate.com/recipes/218332-generator-for-integer-partitions/

    See Also
    ========

    sympy.combinatorics.partitions.Partition
    sympy.combinatorics.partitions.IntegerPartition

    """
    if (n <= 0 or
        m is not None and m < 1 or
        k is not None and k < 1 or
        m and k and m*k < n):
        # the empty set is the only way to handle these inputs
        # and returning {} to represent it is consistent with
        # the counting convention, e.g. nT(0) == 1.
        if size:
            yield 0, {}
        else:
            yield {}
        return

    if m is None:
        m = n
    else:
        m = min(m, n)
    k = min(k or n, n)

    n, m, k = as_int(n), as_int(m), as_int(k)
    q, r = divmod(n, k)
    ms = {k: q}
    keys = [k]  # ms.keys(), from largest to smallest
    if r:
        ms[r] = 1
        keys.append(r)
    room = m - q - bool(r)
    if size:
        yield sum(ms.values()), ms.copy()
    else:
        yield ms.copy()

    while keys != [1]:
        # Reuse any 1's.
        if keys[-1] == 1:
            del keys[-1]
            reuse = ms.pop(1)
            room += reuse
        else:
            reuse = 0

        while 1:
            # Let i be the smallest key larger than 1.  Reuse one
            # instance of i.
            i = keys[-1]
            newcount = ms[i] = ms[i] - 1
            reuse += i
            if newcount == 0:
                del keys[-1], ms[i]
            room += 1

            # Break the remainder into pieces of size i-1.
            i -= 1
            q, r = divmod(reuse, i)
            need = q + bool(r)
            if need > room:
                if not keys:
                    return
                continue

            ms[i] = q
            keys.append(i)
            if r:
                ms[r] = 1
                keys.append(r)
            break
        room -= need
        if size:
            yield sum(ms.values()), ms.copy()
        else:
            yield ms.copy()


def ordered_partitions(n, m=None, sort=True):
    """Generates ordered partitions of integer *n*.

    Parameters
    ==========
    n : int
    m : int, optional
        The default value gives partitions of all sizes else only
        those with size m. In addition, if *m* is not None then
        partitions are generated *in place* (see examples).
    sort : bool, default: True
        Controls whether partitions are
        returned in sorted order when *m* is not None; when False,
        the partitions are returned as fast as possible with elements
        sorted, but when m|n the partitions will not be in
        ascending lexicographical order.

    Examples
    ========

    >>> from sympy.utilities.iterables import ordered_partitions

    All partitions of 5 in ascending lexicographical:

    >>> for p in ordered_partitions(5):
    ...     print(p)
    [1, 1, 1, 1, 1]
    [1, 1, 1, 2]
    [1, 1, 3]
    [1, 2, 2]
    [1, 4]
    [2, 3]
    [5]

    Only partitions of 5 with two parts:

    >>> for p in ordered_partitions(5, 2):
    ...     print(p)
    [1, 4]
    [2, 3]

    When ``m`` is given, a given list objects will be used more than
    once for speed reasons so you will not see the correct partitions
    unless you make a copy of each as it is generated:

    >>> [p for p in ordered_partitions(7, 3)]
    [[1, 1, 1], [1, 1, 1], [1, 1, 1], [2, 2, 2]]
    >>> [list(p) for p in ordered_partitions(7, 3)]
    [[1, 1, 5], [1, 2, 4], [1, 3, 3], [2, 2, 3]]

    When ``n`` is a multiple of ``m``, the elements are still sorted
    but the partitions themselves will be *unordered* if sort is False;
    the default is to return them in ascending lexicographical order.

    >>> for p in ordered_partitions(6, 2):
    ...     print(p)
    [1, 5]
    [2, 4]
    [3, 3]

    But if speed is more important than ordering, sort can be set to
    False:

    >>> for p in ordered_partitions(6, 2, sort=False):
    ...     print(p)
    [1, 5]
    [3, 3]
    [2, 4]

    References
    ==========

    .. [1] Generating Integer Partitions, [online],
        Available: https://jeromekelleher.net/generating-integer-partitions.html
    .. [2] Jerome Kelleher and Barry O'Sullivan, "Generating All
        Partitions: A Comparison Of Two Encodings", [online],
        Available: https://arxiv.org/pdf/0909.2331v2.pdf
    """
    if n < 1 or m is not None and m < 1:
        # the empty set is the only way to handle these inputs
        # and returning {} to represent it is consistent with
        # the counting convention, e.g. nT(0) == 1.
        yield []
        return

    if m is None:
        # The list `a`'s leading elements contain the partition in which
        # y is the biggest element and x is either the same as y or the
        # 2nd largest element; v and w are adjacent element indices
        # to which x and y are being assigned, respectively.
        a = [1]*n
        y = -1
        v = n
        while v > 0:
            v -= 1
            x = a[v] + 1
            while y >= 2 * x:
                a[v] = x
                y -= x
                v += 1
            w = v + 1
            while x <= y:
                a[v] = x
                a[w] = y
                yield a[:w + 1]
                x += 1
                y -= 1
            a[v] = x + y
            y = a[v] - 1
            yield a[:w]
    elif m == 1:
        yield [n]
    elif n == m:
        yield [1]*n
    else:
        # recursively generate partitions of size m
        for b in range(1, n//m + 1):
            a = [b]*m
            x = n - b*m
            if not x:
                if sort:
                    yield a
            elif not sort and x <= m:
                for ax in ordered_partitions(x, sort=False):
                    mi = len(ax)
                    a[-mi:] = [i + b for i in ax]
                    yield a
                    a[-mi:] = [b]*mi
            else:
                for mi in range(1, m):
                    for ax in ordered_partitions(x, mi, sort=True):
                        a[-mi:] = [i + b for i in ax]
                        yield a
                        a[-mi:] = [b]*mi


def binary_partitions(n):
    """
    Generates the binary partition of *n*.

    A binary partition consists only of numbers that are
    powers of two. Each step reduces a `2^{k+1}` to `2^k` and
    `2^k`. Thus 16 is converted to 8 and 8.

    Examples
    ========

    >>> from sympy.utilities.iterables import binary_partitions
    >>> for i in binary_partitions(5):
    ...     print(i)
    ...
    [4, 1]
    [2, 2, 1]
    [2, 1, 1, 1]
    [1, 1, 1, 1, 1]

    References
    ==========

    .. [1] TAOCP 4, section 7.2.1.5, problem 64

    """
    from math import ceil, log2
    power = int(2**(ceil(log2(n))))
    acc = 0
    partition = []
    while power:
        if acc + power <= n:
            partition.append(power)
            acc += power
        power >>= 1

    last_num = len(partition) - 1 - (n & 1)
    while last_num >= 0:
        yield partition
        if partition[last_num] == 2:
            partition[last_num] = 1
            partition.append(1)
            last_num -= 1
            continue
        partition.append(1)
        partition[last_num] >>= 1
        x = partition[last_num + 1] = partition[last_num]
        last_num += 1
        while x > 1:
            if x <= len(partition) - last_num - 1:
                del partition[-x + 1:]
                last_num += 1
                partition[last_num] = x
            else:
                x >>= 1
    yield [1]*n


def has_dups(seq):
    """Return True if there are any duplicate elements in ``seq``.

    Examples
    ========

    >>> from sympy import has_dups, Dict, Set
    >>> has_dups((1, 2, 1))
    True
    >>> has_dups(range(3))
    False
    >>> all(has_dups(c) is False for c in (set(), Set(), dict(), Dict()))
    True
    """
    from sympy.core.containers import Dict
    from sympy.sets.sets import Set
    if isinstance(seq, (dict, set, Dict, Set)):
        return False
    unique = set()
    try:
        return any(True for s in seq if s in unique or unique.add(s))
    except TypeError:
        return len(seq) != len(list(uniq(seq)))


def has_variety(seq):
    """Return True if there are any different elements in ``seq``.

    Examples
    ========

    >>> from sympy import has_variety

    >>> has_variety((1, 2, 1))
    True
    >>> has_variety((1, 1, 1))
    False
    """
    for i, s in enumerate(seq):
        if i == 0:
            sentinel = s
        else:
            if s != sentinel:
                return True
    return False


def uniq(seq, result=None):
    """
    Yield unique elements from ``seq`` as an iterator. The second
    parameter ``result``  is used internally; it is not necessary
    to pass anything for this.

    Note: changing the sequence during iteration will raise a
    RuntimeError if the size of the sequence is known; if you pass
    an iterator and advance the iterator you will change the
    output of this routine but there will be no warning.

    Examples
    ========

    >>> from sympy.utilities.iterables import uniq
    >>> dat = [1, 4, 1, 5, 4, 2, 1, 2]
    >>> type(uniq(dat)) in (list, tuple)
    False

    >>> list(uniq(dat))
    [1, 4, 5, 2]
    >>> list(uniq(x for x in dat))
    [1, 4, 5, 2]
    >>> list(uniq([[1], [2, 1], [1]]))
    [[1], [2, 1]]
    """
    try:
        n = len(seq)
    except TypeError:
        n = None
    def check():
        # check that size of seq did not change during iteration;
        # if n == None the object won't support size changing, e.g.
        # an iterator can't be changed
        if n is not None and len(seq) != n:
            raise RuntimeError('sequence changed size during iteration')
    try:
        seen = set()
        result = result or []
        for i, s in enumerate(seq):
            if not (s in seen or seen.add(s)):
                yield s
                check()
    except TypeError:
        if s not in result:
            yield s
            check()
            result.append(s)
        if hasattr(seq, '__getitem__'):
            yield from uniq(seq[i + 1:], result)
        else:
            yield from uniq(seq, result)


def generate_bell(n):
    """Return permutations of [0, 1, ..., n - 1] such that each permutation
    differs from the last by the exchange of a single pair of neighbors.
    The ``n!`` permutations are returned as an iterator. In order to obtain
    the next permutation from a random starting permutation, use the
    ``next_trotterjohnson`` method of the Permutation class (which generates
    the same sequence in a different manner).

    Examples
    ========

    >>> from itertools import permutations
    >>> from sympy.utilities.iterables import generate_bell
    >>> from sympy import zeros, Matrix

    This is the sort of permutation used in the ringing of physical bells,
    and does not produce permutations in lexicographical order. Rather, the
    permutations differ from each other by exactly one inversion, and the
    position at which the swapping occurs varies periodically in a simple
    fashion. Consider the first few permutations of 4 elements generated
    by ``permutations`` and ``generate_bell``:

    >>> list(permutations(range(4)))[:5]
    [(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3), (0, 2, 3, 1), (0, 3, 1, 2)]
    >>> list(generate_bell(4))[:5]
    [(0, 1, 2, 3), (0, 1, 3, 2), (0, 3, 1, 2), (3, 0, 1, 2), (3, 0, 2, 1)]

    Notice how the 2nd and 3rd lexicographical permutations have 3 elements
    out of place whereas each "bell" permutation always has only two
    elements out of place relative to the previous permutation (and so the
    signature (+/-1) of a permutation is opposite of the signature of the
    previous permutation).

    How the position of inversion varies across the elements can be seen
    by tracing out where the largest number appears in the permutations:

    >>> m = zeros(4, 24)
    >>> for i, p in enumerate(generate_bell(4)):
    ...     m[:, i] = Matrix([j - 3 for j in list(p)])  # make largest zero
    >>> m.print_nonzero('X')
    [XXX  XXXXXX  XXXXXX  XXX]
    [XX XX XXXX XX XXXX XX XX]
    [X XXXX XX XXXX XX XXXX X]
    [ XXXXXX  XXXXXX  XXXXXX ]

    See Also
    ========

    sympy.combinatorics.permutations.Permutation.next_trotterjohnson

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Method_ringing

    .. [2] https://stackoverflow.com/questions/4856615/recursive-permutation/4857018

    .. [3] https://web.archive.org/web/20160313023044/http://programminggeeks.com/bell-algorithm-for-permutation/

    .. [4] https://en.wikipedia.org/wiki/Steinhaus%E2%80%93Johnson%E2%80%93Trotter_algorithm

    .. [5] Generating involutions, derangements, and relatives by ECO
           Vincent Vajnovszki, DMTCS vol 1 issue 12, 2010

    """
    n = as_int(n)
    if n < 1:
        raise ValueError('n must be a positive integer')
    if n == 1:
        yield (0,)
    elif n == 2:
        yield (0, 1)
        yield (1, 0)
    elif n == 3:
        yield from [(0, 1, 2), (0, 2, 1), (2, 0, 1), (2, 1, 0), (1, 2, 0), (1, 0, 2)]
    else:
        m = n - 1
        op = [0] + [-1]*m
        l = list(range(n))
        while True:
            yield tuple(l)
            # find biggest element with op
            big = None, -1  # idx, value
            for i in range(n):
                if op[i] and l[i] > big[1]:
                    big = i, l[i]
            i, _ = big
            if i is None:
                break  # there are no ops left
            # swap it with neighbor in the indicated direction
            j = i + op[i]
            l[i], l[j] = l[j], l[i]
            op[i], op[j] = op[j], op[i]
            # if it landed at the end or if the neighbor in the same
            # direction is bigger then turn off op
            if j == 0 or j == m or l[j + op[j]] > l[j]:
                op[j] = 0
            # any element bigger to the left gets +1 op
            for i in range(j):
                if l[i] > l[j]:
                    op[i] = 1
            # any element bigger to the right gets -1 op
            for i in range(j + 1, n):
                if l[i] > l[j]:
                    op[i] = -1


def generate_involutions(n):
    """
    Generates involutions.

    An involution is a permutation that when multiplied
    by itself equals the identity permutation. In this
    implementation the involutions are generated using
    Fixed Points.

    Alternatively, an involution can be considered as
    a permutation that does not contain any cycles with
    a length that is greater than two.

    Examples
    ========

    >>> from sympy.utilities.iterables import generate_involutions
    >>> list(generate_involutions(3))
    [(0, 1, 2), (0, 2, 1), (1, 0, 2), (2, 1, 0)]
    >>> len(list(generate_involutions(4)))
    10

    References
    ==========

    .. [1] https://mathworld.wolfram.com/PermutationInvolution.html

    """
    idx = list(range(n))
    for p in permutations(idx):
        for i in idx:
            if p[p[i]] != i:
                break
        else:
            yield p


def multiset_derangements(s):
    """Generate derangements of the elements of s *in place*.

    Examples
    ========

    >>> from sympy.utilities.iterables import multiset_derangements, uniq

    Because the derangements of multisets (not sets) are generated
    in place, copies of the return value must be made if a collection
    of derangements is desired or else all values will be the same:

    >>> list(uniq([i for i in multiset_derangements('1233')]))
    [[None, None, None, None]]
    >>> [i.copy() for i in multiset_derangements('1233')]
    [['3', '3', '1', '2'], ['3', '3', '2', '1']]
    >>> [''.join(i) for i in multiset_derangements('1233')]
    ['3312', '3321']
    """
    from sympy.core.sorting import ordered
    # create multiset dictionary of hashable elements or else
    # remap elements to integers
    try:
        ms = multiset(s)
    except TypeError:
        # give each element a canonical integer value
        key = dict(enumerate(ordered(uniq(s))))
        h = []
        for si in s:
            for k in key:
                if key[k] == si:
                    h.append(k)
                    break
        for i in multiset_derangements(h):
            yield [key[j] for j in i]
        return

    mx = max(ms.values())  # max repetition of any element
    n = len(s)  # the number of elements

    ## special cases

    # 1) one element has more than half the total cardinality of s: no
    # derangements are possible.
    if mx*2 > n:
        return

    # 2) all elements appear once: singletons
    if len(ms) == n:
        yield from _set_derangements(s)
        return

    # find the first element that is repeated the most to place
    # in the following two special cases where the selection
    # is unambiguous: either there are two elements with multiplicity
    # of mx or else there is only one with multiplicity mx
    for M in ms:
        if ms[M] == mx:
            break

    inonM = [i for i in range(n) if s[i] != M]  # location of non-M
    iM = [i for i in range(n) if s[i] == M]  # locations of M
    rv = [None]*n

    # 3) half are the same
    if 2*mx == n:
        # M goes into non-M locations
        for i in inonM:
            rv[i] = M
        # permutations of non-M go to M locations
        for p in multiset_permutations([s[i] for i in inonM]):
            for i, pi in zip(iM, p):
                rv[i] = pi
            yield rv
        # clean-up (and encourages proper use of routine)
        rv[:] = [None]*n
        return

    # 4) single repeat covers all but 1 of the non-repeats:
    # if there is one repeat then the multiset of the values
    # of ms would be {mx: 1, 1: n - mx}, i.e. there would
    # be n - mx + 1 values with the condition that n - 2*mx = 1
    if n - 2*mx == 1 and len(ms.values()) == n - mx + 1:
        for i, i1 in enumerate(inonM):
            ifill = inonM[:i] + inonM[i+1:]
            for j in ifill:
                rv[j] = M
            for p in permutations([s[j] for j in ifill]):
                rv[i1] = s[i1]
                for j, pi in zip(iM, p):
                    rv[j] = pi
                k = i1
                for j in iM:
                    rv[j], rv[k] = rv[k], rv[j]
                    yield rv
                    k = j
        # clean-up (and encourages proper use of routine)
        rv[:] = [None]*n
        return

    ## general case is handled with 3 helpers:
    #    1) `finish_derangements` will place the last two elements
    #       which have arbitrary multiplicities, e.g. for multiset
    #       {c: 3, a: 2, b: 2}, the last two elements are a and b
    #    2) `iopen` will tell where a given element can be placed
    #    3) `do` will recursively place elements into subsets of
    #        valid locations

    def finish_derangements():
        """Place the last two elements into the partially completed
        derangement, and yield the results.
        """

        a = take[1][0]  # penultimate element
        a_ct = take[1][1]
        b = take[0][0]  # last element to be placed
        b_ct = take[0][1]

        # split the indexes of the not-already-assigned elements of rv into
        # three categories
        forced_a = []  # positions which must have an a
        forced_b = []  # positions which must have a b
        open_free = []  # positions which could take either
        for i in range(len(s)):
            if rv[i] is None:
                if s[i] == a:
                    forced_b.append(i)
                elif s[i] == b:
                    forced_a.append(i)
                else:
                    open_free.append(i)

        if len(forced_a) > a_ct or len(forced_b) > b_ct:
            # No derangement possible
            return

        for i in forced_a:
            rv[i] = a
        for i in forced_b:
            rv[i] = b
        for a_place in combinations(open_free, a_ct - len(forced_a)):
            for a_pos in a_place:
                rv[a_pos] = a
            for i in open_free:
                if rv[i] is None:  # anything not in the subset is set to b
                    rv[i] = b
            yield rv
            # Clean up/undo the final placements
            for i in open_free:
                rv[i] = None

        # additional cleanup - clear forced_a, forced_b
        for i in forced_a:
            rv[i] = None
        for i in forced_b:
            rv[i] = None

    def iopen(v):
        # return indices at which element v can be placed in rv:
        # locations which are not already occupied if that location
        # does not already contain v in the same location of s
        return [i for i in range(n) if rv[i] is None and s[i] != v]

    def do(j):
        if j == 1:
            # handle the last two elements (regardless of multiplicity)
            # with a special method
            yield from finish_derangements()
        else:
            # place the mx elements of M into a subset of places
            # into which it can be replaced
            M, mx = take[j]
            for i in combinations(iopen(M), mx):
                # place M
                for ii in i:
                    rv[ii] = M
                # recursively place the next element
                yield from do(j - 1)
                # mark positions where M was placed as once again
                # open for placement of other elements
                for ii in i:
                    rv[ii] = None

    # process elements in order of canonically decreasing multiplicity
    take = sorted(ms.items(), key=lambda x:(x[1], x[0]))
    yield from do(len(take) - 1)
    rv[:] = [None]*n


def random_derangement(t, choice=None, strict=True):
    """Return a list of elements in which none are in the same positions
    as they were originally. If an element fills more than half of the positions
    then an error will be raised since no derangement is possible. To obtain
    a derangement of as many items as possible--with some of the most numerous
    remaining in their original positions--pass `strict=False`. To produce a
    pseudorandom derangment, pass a pseudorandom selector like `choice` (see
    below).

    Examples
    ========

    >>> from sympy.utilities.iterables import random_derangement
    >>> t = 'SymPy: a CAS in pure Python'
    >>> d = random_derangement(t)
    >>> all(i != j for i, j in zip(d, t))
    True

    A predictable result can be obtained by using a pseudorandom
    generator for the choice:

    >>> from sympy.core.random import seed, choice as c
    >>> seed(1)
    >>> d = [''.join(random_derangement(t, c)) for i in range(5)]
    >>> assert len(set(d)) != 1  # we got different values

    By reseeding, the same sequence can be obtained:

    >>> seed(1)
    >>> d2 = [''.join(random_derangement(t, c)) for i in range(5)]
    >>> assert d == d2
    """
    if choice is None:
        import secrets
        choice = secrets.choice
    def shuffle(rv):
        '''Knuth shuffle'''
        for i in range(len(rv) - 1, 0, -1):
            x = choice(rv[:i + 1])
            j = rv.index(x)
            rv[i], rv[j] = rv[j], rv[i]
    def pick(rv, n):
        '''shuffle rv and return the first n values
        '''
        shuffle(rv)
        return rv[:n]
    ms = multiset(t)
    tot = len(t)
    ms = sorted(ms.items(), key=lambda x: x[1])
  # if there are not enough spaces for the most
  # plentiful element to move to then some of them
  # will have to stay in place
    M, mx = ms[-1]
    n = len(t)
    xs = 2*mx - tot
    if xs > 0:
        if strict:
            raise ValueError('no derangement possible')
        opts = [i for (i, c) in enumerate(t) if c == ms[-1][0]]
        pick(opts, xs)
        stay = sorted(opts[:xs])
        rv = list(t)
        for i in reversed(stay):
            rv.pop(i)
        rv = random_derangement(rv, choice)
        for i in stay:
            rv.insert(i, ms[-1][0])
        return ''.join(rv) if type(t) is str else rv
  # the normal derangement calculated from here
    if n == len(ms):
      # approx 1/3 will succeed
        rv = list(t)
        while True:
            shuffle(rv)
            if all(i != j for i,j in zip(rv, t)):
                break
    else:
      # general case
        rv = [None]*n
        while True:
            j = 0
            while j > -len(ms):  # do most numerous first
                j -= 1
                e, c = ms[j]
                opts = [i for i in range(n) if rv[i] is None and t[i] != e]
                if len(opts) < c:
                    for i in range(n):
                        rv[i] = None
                    break # try again
                pick(opts, c)
                for i in range(c):
                    rv[opts[i]] = e
            else:
                return rv
    return rv


def _set_derangements(s):
    """
    yield derangements of items in ``s`` which are assumed to contain
    no repeated elements
    """
    if len(s) < 2:
        return
    if len(s) == 2:
        yield [s[1], s[0]]
        return
    if len(s) == 3:
        yield [s[1], s[2], s[0]]
        yield [s[2], s[0], s[1]]
        return
    for p in permutations(s):
        if not any(i == j for i, j in zip(p, s)):
            yield list(p)


def generate_derangements(s):
    """
    Return unique derangements of the elements of iterable ``s``.

    Examples
    ========

    >>> from sympy.utilities.iterables import generate_derangements
    >>> list(generate_derangements([0, 1, 2]))
    [[1, 2, 0], [2, 0, 1]]
    >>> list(generate_derangements([0, 1, 2, 2]))
    [[2, 2, 0, 1], [2, 2, 1, 0]]
    >>> list(generate_derangements([0, 1, 1]))
    []

    See Also
    ========

    sympy.functions.combinatorial.factorials.subfactorial

    """
    if not has_dups(s):
        yield from _set_derangements(s)
    else:
        for p in multiset_derangements(s):
            yield list(p)


def necklaces(n, k, free=False):
    """
    A routine to generate necklaces that may (free=True) or may not
    (free=False) be turned over to be viewed. The "necklaces" returned
    are comprised of ``n`` integers (beads) with ``k`` different
    values (colors). Only unique necklaces are returned.

    Examples
    ========

    >>> from sympy.utilities.iterables import necklaces, bracelets
    >>> def show(s, i):
    ...     return ''.join(s[j] for j in i)

    The "unrestricted necklace" is sometimes also referred to as a
    "bracelet" (an object that can be turned over, a sequence that can
    be reversed) and the term "necklace" is used to imply a sequence
    that cannot be reversed. So ACB == ABC for a bracelet (rotate and
    reverse) while the two are different for a necklace since rotation
    alone cannot make the two sequences the same.

    (mnemonic: Bracelets can be viewed Backwards, but Not Necklaces.)

    >>> B = [show('ABC', i) for i in bracelets(3, 3)]
    >>> N = [show('ABC', i) for i in necklaces(3, 3)]
    >>> set(N) - set(B)
    {'ACB'}

    >>> list(necklaces(4, 2))
    [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 1),
     (0, 1, 0, 1), (0, 1, 1, 1), (1, 1, 1, 1)]

    >>> [show('.o', i) for i in bracelets(4, 2)]
    ['....', '...o', '..oo', '.o.o', '.ooo', 'oooo']

    References
    ==========

    .. [1] https://mathworld.wolfram.com/Necklace.html

    .. [2] Frank Ruskey, Carla Savage, and Terry Min Yih Wang,
        Generating necklaces, Journal of Algorithms 13 (1992), 414-430;
        https://doi.org/10.1016/0196-6774(92)90047-G

    """
    # The FKM algorithm
    if k == 0 and n > 0:
        return
    a = [0]*n
    yield tuple(a)
    if n == 0:
        return
    while True:
        i = n - 1
        while a[i] == k - 1:
            i -= 1
            if i == -1:
                return
        a[i] += 1
        for j in range(n - i - 1):
            a[j + i + 1] = a[j]
        if n % (i + 1) == 0 and (not free or all(a <= a[j::-1] + a[-1:j:-1] for j in range(n - 1))):
            # No need to test j = n - 1.
            yield tuple(a)


def bracelets(n, k):
    """Wrapper to necklaces to return a free (unrestricted) necklace."""
    return necklaces(n, k, free=True)


def generate_oriented_forest(n):
    """
    This algorithm generates oriented forests.

    An oriented graph is a directed graph having no symmetric pair of directed
    edges. A forest is an acyclic graph, i.e., it has no cycles. A forest can
    also be described as a disjoint union of trees, which are graphs in which
    any two vertices are connected by exactly one simple path.

    Examples
    ========

    >>> from sympy.utilities.iterables import generate_oriented_forest
    >>> list(generate_oriented_forest(4))
    [[0, 1, 2, 3], [0, 1, 2, 2], [0, 1, 2, 1], [0, 1, 2, 0], \
    [0, 1, 1, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 1, 0, 0], [0, 0, 0, 0]]

    References
    ==========

    .. [1] T. Beyer and S.M. Hedetniemi: constant time generation of
           rooted trees, SIAM J. Computing Vol. 9, No. 4, November 1980

    .. [2] https://stackoverflow.com/questions/1633833/oriented-forest-taocp-algorithm-in-python

    """
    P = list(range(-1, n))
    while True:
        yield P[1:]
        if P[n] > 0:
            P[n] = P[P[n]]
        else:
            for p in range(n - 1, 0, -1):
                if P[p] != 0:
                    target = P[p] - 1
                    for q in range(p - 1, 0, -1):
                        if P[q] == target:
                            break
                    offset = p - q
                    for i in range(p, n + 1):
                        P[i] = P[i - offset]
                    break
            else:
                break


def minlex(seq, directed=True, key=None):
    r"""
    Return the rotation of the sequence in which the lexically smallest
    elements appear first, e.g. `cba \rightarrow acb`.

    The sequence returned is a tuple, unless the input sequence is a string
    in which case a string is returned.

    If ``directed`` is False then the smaller of the sequence and the
    reversed sequence is returned, e.g. `cba \rightarrow abc`.

    If ``key`` is not None then it is used to extract a comparison key from each element in iterable.

    Examples
    ========

    >>> from sympy.combinatorics.polyhedron import minlex
    >>> minlex((1, 2, 0))
    (0, 1, 2)
    >>> minlex((1, 0, 2))
    (0, 2, 1)
    >>> minlex((1, 0, 2), directed=False)
    (0, 1, 2)

    >>> minlex('11010011000', directed=True)
    '00011010011'
    >>> minlex('11010011000', directed=False)
    '00011001011'

    >>> minlex(('bb', 'aaa', 'c', 'a'))
    ('a', 'bb', 'aaa', 'c')
    >>> minlex(('bb', 'aaa', 'c', 'a'), key=len)
    ('c', 'a', 'bb', 'aaa')

    """
    from sympy.functions.elementary.miscellaneous import Id
    if key is None: key = Id
    best = rotate_left(seq, least_rotation(seq, key=key))
    if not directed:
        rseq = seq[::-1]
        rbest = rotate_left(rseq, least_rotation(rseq, key=key))
        best = min(best, rbest, key=key)

    # Convert to tuple, unless we started with a string.
    return tuple(best) if not isinstance(seq, str) else best


def runs(seq, op=gt):
    """Group the sequence into lists in which successive elements
    all compare the same with the comparison operator, ``op``:
    op(seq[i + 1], seq[i]) is True from all elements in a run.

    Examples
    ========

    >>> from sympy.utilities.iterables import runs
    >>> from operator import ge
    >>> runs([0, 1, 2, 2, 1, 4, 3, 2, 2])
    [[0, 1, 2], [2], [1, 4], [3], [2], [2]]
    >>> runs([0, 1, 2, 2, 1, 4, 3, 2, 2], op=ge)
    [[0, 1, 2, 2], [1, 4], [3], [2, 2]]
    """
    cycles = []
    seq = iter(seq)
    try:
        run = [next(seq)]
    except StopIteration:
        return []
    while True:
        try:
            ei = next(seq)
        except StopIteration:
            break
        if op(ei, run[-1]):
            run.append(ei)
            continue
        else:
            cycles.append(run)
            run = [ei]
    if run:
        cycles.append(run)
    return cycles


def sequence_partitions(l, n, /):
    r"""Returns the partition of sequence $l$ into $n$ bins

    Explanation
    ===========

    Given the sequence $l_1 \cdots l_m \in V^+$ where
    $V^+$ is the Kleene plus of $V$

    The set of $n$ partitions of $l$ is defined as:

    .. math::
        \{(s_1, \cdots, s_n) | s_1 \in V^+, \cdots, s_n \in V^+,
        s_1 \cdots s_n = l_1 \cdots l_m\}

    Parameters
    ==========

    l : Sequence[T]
        A nonempty sequence of any Python objects

    n : int
        A positive integer

    Yields
    ======

    out : list[Sequence[T]]
        A list of sequences with concatenation equals $l$.
        This should conform with the type of $l$.

    Examples
    ========

    >>> from sympy.utilities.iterables import sequence_partitions
    >>> for out in sequence_partitions([1, 2, 3, 4], 2):
    ...     print(out)
    [[1], [2, 3, 4]]
    [[1, 2], [3, 4]]
    [[1, 2, 3], [4]]

    Notes
    =====

    This is modified version of EnricoGiampieri's partition generator
    from https://stackoverflow.com/questions/13131491/partition-n-items-into-k-bins-in-python-lazily

    See Also
    ========

    sequence_partitions_empty
    """
    # Asserting l is nonempty is done only for sanity check
    if n == 1 and l:
        yield [l]
        return
    for i in range(1, len(l)):
        for part in sequence_partitions(l[i:], n - 1):
            yield [l[:i]] + part


def sequence_partitions_empty(l, n, /):
    r"""Returns the partition of sequence $l$ into $n$ bins with
    empty sequence

    Explanation
    ===========

    Given the sequence $l_1 \cdots l_m \in V^*$ where
    $V^*$ is the Kleene star of $V$

    The set of $n$ partitions of $l$ is defined as:

    .. math::
        \{(s_1, \cdots, s_n) | s_1 \in V^*, \cdots, s_n \in V^*,
        s_1 \cdots s_n = l_1 \cdots l_m\}

    There are more combinations than :func:`sequence_partitions` because
    empty sequence can fill everywhere, so we try to provide different
    utility for this.

    Parameters
    ==========

    l : Sequence[T]
        A sequence of any Python objects (can be possibly empty)

    n : int
        A positive integer

    Yields
    ======

    out : list[Sequence[T]]
        A list of sequences with concatenation equals $l$.
        This should conform with the type of $l$.

    Examples
    ========

    >>> from sympy.utilities.iterables import sequence_partitions_empty
    >>> for out in sequence_partitions_empty([1, 2, 3, 4], 2):
    ...     print(out)
    [[], [1, 2, 3, 4]]
    [[1], [2, 3, 4]]
    [[1, 2], [3, 4]]
    [[1, 2, 3], [4]]
    [[1, 2, 3, 4], []]

    See Also
    ========

    sequence_partitions
    """
    if n < 1:
        return
    if n == 1:
        yield [l]
        return
    for i in range(0, len(l) + 1):
        for part in sequence_partitions_empty(l[i:], n - 1):
            yield [l[:i]] + part


def kbins(l, k, ordered=None):
    """
    Return sequence ``l`` partitioned into ``k`` bins.

    Examples
    ========

    The default is to give the items in the same order, but grouped
    into k partitions without any reordering:

    >>> from sympy.utilities.iterables import kbins
    >>> for p in kbins(list(range(5)), 2):
    ...     print(p)
    ...
    [[0], [1, 2, 3, 4]]
    [[0, 1], [2, 3, 4]]
    [[0, 1, 2], [3, 4]]
    [[0, 1, 2, 3], [4]]

    The ``ordered`` flag is either None (to give the simple partition
    of the elements) or is a 2 digit integer indicating whether the order of
    the bins and the order of the items in the bins matters. Given::

        A = [[0], [1, 2]]
        B = [[1, 2], [0]]
        C = [[2, 1], [0]]
        D = [[0], [2, 1]]

    the following values for ``ordered`` have the shown meanings::

        00 means A == B == C == D
        01 means A == B
        10 means A == D
        11 means A == A

    >>> for ordered_flag in [None, 0, 1, 10, 11]:
    ...     print('ordered = %s' % ordered_flag)
    ...     for p in kbins(list(range(3)), 2, ordered=ordered_flag):
    ...         print('     %s' % p)
    ...
    ordered = None
         [[0], [1, 2]]
         [[0, 1], [2]]
    ordered = 0
         [[0, 1], [2]]
         [[0, 2], [1]]
         [[0], [1, 2]]
    ordered = 1
         [[0], [1, 2]]
         [[0], [2, 1]]
         [[1], [0, 2]]
         [[1], [2, 0]]
         [[2], [0, 1]]
         [[2], [1, 0]]
    ordered = 10
         [[0, 1], [2]]
         [[2], [0, 1]]
         [[0, 2], [1]]
         [[1], [0, 2]]
         [[0], [1, 2]]
         [[1, 2], [0]]
    ordered = 11
         [[0], [1, 2]]
         [[0, 1], [2]]
         [[0], [2, 1]]
         [[0, 2], [1]]
         [[1], [0, 2]]
         [[1, 0], [2]]
         [[1], [2, 0]]
         [[1, 2], [0]]
         [[2], [0, 1]]
         [[2, 0], [1]]
         [[2], [1, 0]]
         [[2, 1], [0]]

    See Also
    ========

    partitions, multiset_partitions

    """
    if ordered is None:
        yield from sequence_partitions(l, k)
    elif ordered == 11:
        for pl in multiset_permutations(l):
            pl = list(pl)
            yield from sequence_partitions(pl, k)
    elif ordered == 00:
        yield from multiset_partitions(l, k)
    elif ordered == 10:
        for p in multiset_partitions(l, k):
            for perm in permutations(p):
                yield list(perm)
    elif ordered == 1:
        for kgot, p in partitions(len(l), k, size=True):
            if kgot != k:
                continue
            for li in multiset_permutations(l):
                rv = []
                i = j = 0
                li = list(li)
                for size, multiplicity in sorted(p.items()):
                    for m in range(multiplicity):
                        j = i + size
                        rv.append(li[i: j])
                        i = j
                yield rv
    else:
        raise ValueError(
            'ordered must be one of 00, 01, 10 or 11, not %s' % ordered)


def permute_signs(t):
    """Return iterator in which the signs of non-zero elements
    of t are permuted.

    Examples
    ========

    >>> from sympy.utilities.iterables import permute_signs
    >>> list(permute_signs((0, 1, 2)))
    [(0, 1, 2), (0, -1, 2), (0, 1, -2), (0, -1, -2)]
    """
    for signs in product(*[(1, -1)]*(len(t) - t.count(0))):
        signs = list(signs)
        yield type(t)([i*signs.pop() if i else i for i in t])


def signed_permutations(t):
    """Return iterator in which the signs of non-zero elements
    of t and the order of the elements are permuted and all
    returned values are unique.

    Examples
    ========

    >>> from sympy.utilities.iterables import signed_permutations
    >>> list(signed_permutations((0, 1, 2)))
    [(0, 1, 2), (0, -1, 2), (0, 1, -2), (0, -1, -2), (0, 2, 1),
    (0, -2, 1), (0, 2, -1), (0, -2, -1), (1, 0, 2), (-1, 0, 2),
    (1, 0, -2), (-1, 0, -2), (1, 2, 0), (-1, 2, 0), (1, -2, 0),
    (-1, -2, 0), (2, 0, 1), (-2, 0, 1), (2, 0, -1), (-2, 0, -1),
    (2, 1, 0), (-2, 1, 0), (2, -1, 0), (-2, -1, 0)]
    """
    return (type(t)(i) for j in multiset_permutations(t)
        for i in permute_signs(j))


def rotations(s, dir=1):
    """Return a generator giving the items in s as list where
    each subsequent list has the items rotated to the left (default)
    or right (``dir=-1``) relative to the previous list.

    Examples
    ========

    >>> from sympy import rotations
    >>> list(rotations([1,2,3]))
    [[1, 2, 3], [2, 3, 1], [3, 1, 2]]
    >>> list(rotations([1,2,3], -1))
    [[1, 2, 3], [3, 1, 2], [2, 3, 1]]
    """
    seq = list(s)
    for i in range(len(seq)):
        yield seq
        seq = rotate_left(seq, dir)


def roundrobin(*iterables):
    """roundrobin recipe taken from itertools documentation:
    https://docs.python.org/3/library/itertools.html#itertools-recipes

    roundrobin('ABC', 'D', 'EF') --> A D E B F C

    Recipe credited to George Sakkis
    """
    nexts = cycle(iter(it).__next__ for it in iterables)

    pending = len(iterables)
    while pending:
        try:
            for nxt in nexts:
                yield nxt()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))



class NotIterable:
    """
    Use this as mixin when creating a class which is not supposed to
    return true when iterable() is called on its instances because
    calling list() on the instance, for example, would result in
    an infinite loop.
    """
    pass


def iterable(i, exclude=(str, dict, NotIterable)):
    """
    Return a boolean indicating whether ``i`` is SymPy iterable.
    True also indicates that the iterator is finite, e.g. you can
    call list(...) on the instance.

    When SymPy is working with iterables, it is almost always assuming
    that the iterable is not a string or a mapping, so those are excluded
    by default. If you want a pure Python definition, make exclude=None. To
    exclude multiple items, pass them as a tuple.

    You can also set the _iterable attribute to True or False on your class,
    which will override the checks here, including the exclude test.

    As a rule of thumb, some SymPy functions use this to check if they should
    recursively map over an object. If an object is technically iterable in
    the Python sense but does not desire this behavior (e.g., because its
    iteration is not finite, or because iteration might induce an unwanted
    computation), it should disable it by setting the _iterable attribute to False.

    See also: is_sequence

    Examples
    ========

    >>> from sympy.utilities.iterables import iterable
    >>> from sympy import Tuple
    >>> things = [[1], (1,), set([1]), Tuple(1), (j for j in [1, 2]), {1:2}, '1', 1]
    >>> for i in things:
    ...     print('%s %s' % (iterable(i), type(i)))
    True <... 'list'>
    True <... 'tuple'>
    True <... 'set'>
    True <class 'sympy.core.containers.Tuple'>
    True <... 'generator'>
    False <... 'dict'>
    False <... 'str'>
    False <... 'int'>

    >>> iterable({}, exclude=None)
    True
    >>> iterable({}, exclude=str)
    True
    >>> iterable("no", exclude=str)
    False

    """
    if hasattr(i, '_iterable'):
        return i._iterable
    try:
        iter(i)
    except TypeError:
        return False
    if exclude:
        return not isinstance(i, exclude)
    return True


def is_sequence(i, include=None):
    """
    Return a boolean indicating whether ``i`` is a sequence in the SymPy
    sense. If anything that fails the test below should be included as
    being a sequence for your application, set 'include' to that object's
    type; multiple types should be passed as a tuple of types.

    Note: although generators can generate a sequence, they often need special
    handling to make sure their elements are captured before the generator is
    exhausted, so these are not included by default in the definition of a
    sequence.

    See also: iterable

    Examples
    ========

    >>> from sympy.utilities.iterables import is_sequence
    >>> from types import GeneratorType
    >>> is_sequence([])
    True
    >>> is_sequence(set())
    False
    >>> is_sequence('abc')
    False
    >>> is_sequence('abc', include=str)
    True
    >>> generator = (c for c in 'abc')
    >>> is_sequence(generator)
    False
    >>> is_sequence(generator, include=(str, GeneratorType))
    True

    """
    return (hasattr(i, '__getitem__') and
            iterable(i) or
            bool(include) and
            isinstance(i, include))


@deprecated(
    """
    Using postorder_traversal from the sympy.utilities.iterables submodule is
    deprecated.

    Instead, use postorder_traversal from the top-level sympy namespace, like

        sympy.postorder_traversal
    """,
    deprecated_since_version="1.10",
    active_deprecations_target="deprecated-traversal-functions-moved")
def postorder_traversal(node, keys=None):
    from sympy.core.traversal import postorder_traversal as _postorder_traversal
    return _postorder_traversal(node, keys=keys)


@deprecated(
    """
    Using interactive_traversal from the sympy.utilities.iterables submodule
    is deprecated.

    Instead, use interactive_traversal from the top-level sympy namespace,
    like

        sympy.interactive_traversal
    """,
    deprecated_since_version="1.10",
    active_deprecations_target="deprecated-traversal-functions-moved")
def interactive_traversal(expr):
    from sympy.interactive.traversal import interactive_traversal as _interactive_traversal
    return _interactive_traversal(expr)


@deprecated(
    """
    Importing default_sort_key from sympy.utilities.iterables is deprecated.
    Use from sympy import default_sort_key instead.
    """,
    deprecated_since_version="1.10",
active_deprecations_target="deprecated-sympy-core-compatibility",
)
def default_sort_key(*args, **kwargs):
    from sympy import default_sort_key as _default_sort_key
    return _default_sort_key(*args, **kwargs)


@deprecated(
    """
    Importing default_sort_key from sympy.utilities.iterables is deprecated.
    Use from sympy import default_sort_key instead.
    """,
    deprecated_since_version="1.10",
active_deprecations_target="deprecated-sympy-core-compatibility",
)
def ordered(*args, **kwargs):
    from sympy import ordered as _ordered
    return _ordered(*args, **kwargs)
