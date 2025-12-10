"""
Implementation of the Wright, Richmond, Odlyzko and McKay (WROM)
algorithm for the enumeration of all non-isomorphic free trees of a
given order.  Rooted trees are represented by level sequences, i.e.,
lists in which the i-th element specifies the distance of vertex i to
the root.

"""

__all__ = ["nonisomorphic_trees", "number_of_nonisomorphic_trees"]

from functools import lru_cache

import networkx as nx


@nx._dispatchable(graphs=None, returns_graph=True)
def nonisomorphic_trees(order):
    """Generate nonisomorphic trees of specified `order`.

    Parameters
    ----------
    order : int
       order of the desired tree(s)

    Yields
    ------
    `networkx.Graph` instances
       A tree with `order` number of nodes that is not isomorphic to any other
       yielded tree.

    Raises
    ------
    ValueError
       If `order` is negative.

    Examples
    --------
    There are 11 unique (non-isomorphic) trees with 7 nodes.

    >>> n = 7
    >>> nit_list = list(nx.nonisomorphic_trees(n))
    >>> len(nit_list) == nx.number_of_nonisomorphic_trees(n) == 11
    True

    All trees yielded by the generator have the specified order.

    >>> all(len(G) == n for G in nx.nonisomorphic_trees(n))
    True

    Each tree is nonisomorphic to every other tree yielded by the generator.
    >>> seen = []
    >>> for G in nx.nonisomorphic_trees(n):
    ...     assert not any(nx.is_isomorphic(G, H) for H in seen)
    ...     seen.append(G)

    See Also
    --------
    number_of_nonisomorphic_trees
    """
    if order < 0:
        raise ValueError("order must be non-negative")
    if order == 0:
        # Idiom for empty generator, i.e. list(nonisomorphic_trees(0)) == []
        return
        yield
    if order == 1:
        yield nx.empty_graph(1)
        return
    # start at the path graph rooted at its center
    layout = list(range(order // 2 + 1)) + list(range(1, (order + 1) // 2))

    while layout is not None:
        layout = _next_tree(layout)
        if layout is not None:
            yield _layout_to_graph(layout)
            layout = _next_rooted_tree(layout)


@nx._dispatchable(graphs=None)
def number_of_nonisomorphic_trees(order):
    """Returns the number of nonisomorphic trees of the specified `order`.

    Based on an algorithm by Alois P. Heinz in
    `OEIS entry A000055 <https://oeis.org/A000055>`_. Complexity is ``O(n ** 3)``.

    Parameters
    ----------
    order : int
       Order of the desired tree(s).

    Returns
    -------
    int
       Number of nonisomorphic trees with `order` number of nodes.

    Raises
    ------
    ValueError
       If `order` is negative.

    Examples
    --------
    >>> nx.number_of_nonisomorphic_trees(10)
    106

    See Also
    --------
    nonisomorphic_trees
    """
    if order < 0:
        raise ValueError("order must be non-negative")
    return _unlabeled_trees(order)


@lru_cache(None)
def _unlabeled_trees(n):
    """Implements OEIS A000055 (number of unlabeled trees)."""

    value = 0
    for k in range(n + 1):
        value += _rooted_trees(k) * _rooted_trees(n - k)
    if n % 2 == 0:
        value -= _rooted_trees(n // 2)
    return _rooted_trees(n) - value // 2


@lru_cache(None)
def _rooted_trees(n):
    """Implements OEIS A000081 (number of unlabeled rooted trees)."""

    if n < 2:
        return n
    value = 0
    for j in range(1, n):
        for d in range(1, n):
            if j % d == 0:
                value += d * _rooted_trees(d) * _rooted_trees(n - j)
    return value // (n - 1)


def _next_rooted_tree(predecessor, p=None):
    """One iteration of the Beyer-Hedetniemi algorithm."""

    if p is None:
        p = len(predecessor) - 1
        while predecessor[p] == 1:
            p -= 1
    if p == 0:
        return None

    q = p - 1
    while predecessor[q] != predecessor[p] - 1:
        q -= 1
    result = list(predecessor)
    for i in range(p, len(result)):
        result[i] = result[i - p + q]
    return result


def _next_tree(candidate):
    """One iteration of the Wright, Richmond, Odlyzko and McKay
    algorithm."""

    # valid representation of a free tree if:
    # there are at least two vertices at layer 1
    # (this is always the case because we start at the path graph)
    left, rest = _split_tree(candidate)

    # and the left subtree of the root
    # is less high than the tree with the left subtree removed
    left_height = max(left)
    rest_height = max(rest)
    valid = rest_height >= left_height

    if valid and rest_height == left_height:
        # and, if left and rest are of the same height,
        # if left does not encompass more vertices
        if len(left) > len(rest):
            valid = False
        # and, if they have the same number or vertices,
        # if left does not come after rest lexicographically
        elif len(left) == len(rest) and left > rest:
            valid = False

    if valid:
        return candidate
    else:
        # jump to the next valid free tree
        p = len(left)
        new_candidate = _next_rooted_tree(candidate, p)
        if candidate[p] > 2:
            new_left, new_rest = _split_tree(new_candidate)
            new_left_height = max(new_left)
            suffix = range(1, new_left_height + 2)
            new_candidate[-len(suffix) :] = suffix
        return new_candidate


def _split_tree(layout):
    """Returns a tuple of two layouts, one containing the left
    subtree of the root vertex, and one containing the original tree
    with the left subtree removed."""

    one_found = False
    m = None
    for i in range(len(layout)):
        if layout[i] == 1:
            if one_found:
                m = i
                break
            else:
                one_found = True

    if m is None:
        m = len(layout)

    left = [layout[i] - 1 for i in range(1, m)]
    rest = [0] + [layout[i] for i in range(m, len(layout))]
    return (left, rest)


def _layout_to_matrix(layout):
    """Create the adjacency matrix for the tree specified by the
    given layout (level sequence)."""

    result = [[0] * len(layout) for i in range(len(layout))]
    stack = []
    for i in range(len(layout)):
        i_level = layout[i]
        if stack:
            j = stack[-1]
            j_level = layout[j]
            while j_level >= i_level:
                stack.pop()
                j = stack[-1]
                j_level = layout[j]
            result[i][j] = result[j][i] = 1
        stack.append(i)
    return result


def _layout_to_graph(layout):
    """Create a NetworkX Graph for the tree specified by the
    given layout(level sequence)"""
    G = nx.Graph()
    stack = []
    for i in range(len(layout)):
        i_level = layout[i]
        if stack:
            j = stack[-1]
            j_level = layout[j]
            while j_level >= i_level:
                stack.pop()
                j = stack[-1]
                j_level = layout[j]
            G.add_edge(i, j)
        stack.append(i)
    return G
