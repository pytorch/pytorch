from collections import OrderedDict

__all__ = ["raises", "expand_tuples", "reverse_dict", "groupby", "typename"]

def raises(err, lamda):
    try:
        lamda()
        return False
    except err:
        return True


def expand_tuples(L):
    """
    >>> expand_tuples([1, (2, 3)])
    [(1, 2), (1, 3)]
    >>> expand_tuples([1, 2])
    [(1, 2)]
    """
    if not L:
        return [()]
    elif not isinstance(L[0], tuple):
        rest = expand_tuples(L[1:])
        return [(L[0],) + t for t in rest]
    else:
        rest = expand_tuples(L[1:])
        return [(item,) + t for t in rest for item in L[0]]


# Taken from theano/theano/gof/sched.py
# Avoids licensing issues because this was written by Matthew Rocklin
def _toposort(edges):
    """ Topological sort algorithm by Kahn [1] - O(nodes + vertices)
    inputs:
        edges - a dict of the form {a: {b, c}} where b and c depend on a
    outputs:
        L - an ordered list of nodes that satisfy the dependencies of edges
    >>> _toposort({1: (2, 3), 2: (3, )})
    [1, 2, 3]
    >>> # Closely follows the wikipedia page [2]
    >>> # [1] Kahn, Arthur B. (1962), "Topological sorting of large networks",
    >>> # Communications of the ACM
    >>> # [2] http://en.wikipedia.org/wiki/Toposort#Algorithms
    """
    incoming_edges = reverse_dict(edges)
    incoming_edges = OrderedDict((k, set(val))
                                 for k, val in incoming_edges.items())
    S = OrderedDict.fromkeys(v for v in edges if v not in incoming_edges)
    L = []

    while S:
        n, _ = S.popitem()
        L.append(n)
        for m in edges.get(n, ()):
            assert n in incoming_edges[m]
            incoming_edges[m].remove(n)
            if not incoming_edges[m]:
                S[m] = None
    if any(incoming_edges.get(v, None) for v in edges):
        raise ValueError("Input has cycles")
    return L


def reverse_dict(d):
    """Reverses direction of dependence dict
    >>> d = {'a': (1, 2), 'b': (2, 3), 'c':()}
    >>> reverse_dict(d)  # doctest: +SKIP
    {1: ('a',), 2: ('a', 'b'), 3: ('b',)}
    :note: dict order are not deterministic. As we iterate on the
        input dict, it make the output of this function depend on the
        dict order. So this function output order should be considered
        as undeterministic.
    """
    result = OrderedDict()  # type: ignore[var-annotated]
    for key in d:
        for val in d[key]:
            result[val] = result.get(val, tuple()) + (key, )
    return result


# Taken from toolz
# Avoids licensing issues because this version was authored by Matthew Rocklin
def groupby(func, seq):
    """ Group a collection by a key function
    >>> names = ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank']
    >>> groupby(len, names)  # doctest: +SKIP
    {3: ['Bob', 'Dan'], 5: ['Alice', 'Edith', 'Frank'], 7: ['Charlie']}
    >>> iseven = lambda x: x % 2 == 0
    >>> groupby(iseven, [1, 2, 3, 4, 5, 6, 7, 8])  # doctest: +SKIP
    {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}
    See Also:
        ``countby``
    """

    d = OrderedDict()  # type: ignore[var-annotated]
    for item in seq:
        key = func(item)
        if key not in d:
            d[key] = list()
        d[key].append(item)
    return d


def typename(type):
    """Get the name of `type`.
    Parameters
    ----------
    type : Union[Type, Tuple[Type]]
    Returns
    -------
    str
        The name of `type` or a tuple of the names of the types in `type`.
    Examples
    --------
    >>> typename(int)
    'int'
    >>> typename((int, float))
    '(int, float)'
    """
    try:
        return type.__name__
    except AttributeError:
        if len(type) == 1:
            return typename(*type)
        return f"({', '.join(map(typename, type))})"
