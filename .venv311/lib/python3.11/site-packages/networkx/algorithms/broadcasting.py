"""Routines to calculate the broadcast time of certain graphs.

Broadcasting is an information dissemination problem in which a node in a graph,
called the originator, must distribute a message to all other nodes by placing
a series of calls along the edges of the graph. Once informed, other nodes aid
the originator in distributing the message.

The broadcasting must be completed as quickly as possible subject to the
following constraints:
- Each call requires one unit of time.
- A node can only participate in one call per unit of time.
- Each call only involves two adjacent nodes: a sender and a receiver.
"""

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = [
    "tree_broadcast_center",
    "tree_broadcast_time",
]


def _get_max_broadcast_value(G, U, v, values):
    adj = sorted(set(G.neighbors(v)) & U, key=values.get, reverse=True)
    return max(values[u] + i for i, u in enumerate(adj, start=1))


def _get_broadcast_centers(G, v, values, target):
    adj = sorted(G.neighbors(v), key=values.get, reverse=True)
    j = next(i for i, u in enumerate(adj, start=1) if values[u] + i == target)
    return set([v] + adj[:j])


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def tree_broadcast_center(G):
    """Return the broadcast center of a tree.

    The broadcast center of a graph `G` denotes the set of nodes having
    minimum broadcast time [1]_. This function implements a linear algorithm
    for determining the broadcast center of a tree with ``n`` nodes. As a
    by-product, it also determines the broadcast time from the broadcast center.

    Parameters
    ----------
    G : Graph
        The graph should be an undirected tree.

    Returns
    -------
    b_T, b_C : (int, set) tuple
        Minimum broadcast time of the broadcast center in `G`, set of nodes
        in the broadcast center.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is directed or is a multigraph.

    NotATree
        If `G` is not a tree.

    References
    ----------
    .. [1] Slater, P.J., Cockayne, E.J., Hedetniemi, S.T,
       Information dissemination in trees. SIAM J.Comput. 10(4), 692–701 (1981)
    """
    # Assert that the graph G is a tree
    if not nx.is_tree(G):
        raise nx.NotATree("G is not a tree")
    # step 0
    if (n := len(G)) < 3:
        return n - 1, set(G)

    # step 1
    U = {node for node, deg in G.degree if deg == 1}
    values = {n: 0 for n in U}
    T = G.copy()
    T.remove_nodes_from(U)

    # step 2
    W = {node for node, deg in T.degree if deg == 1}
    values.update((w, G.degree[w] - 1) for w in W)

    # step 3
    while len(T) >= 2:
        # step 4
        w = min(W, key=values.get)
        v = next(T.neighbors(w))

        # step 5
        U.add(w)
        W.remove(w)
        T.remove_node(w)

        # step 6
        if T.degree(v) == 1:
            # update t(v)
            values.update({v: _get_max_broadcast_value(G, U, v, values)})
            W.add(v)

    # step 7
    v = nx.utils.arbitrary_element(T)
    b_T = _get_max_broadcast_value(G, U, v, values)
    return b_T, _get_broadcast_centers(G, v, values, b_T)


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def tree_broadcast_time(G, node=None):
    """Return the minimum broadcast time of a (node in a) tree.

    The minimum broadcast time of a node is defined as the minimum amount
    of time required to complete broadcasting starting from that node.
    The broadcast time of a graph is the maximum over
    all nodes of the minimum broadcast time from that node [1]_.
    This function returns the minimum broadcast time of `node`.
    If `node` is `None`, the broadcast time for the graph is returned.

    Parameters
    ----------
    G : Graph
        The graph should be an undirected tree.

    node : node, optional (default=None)
        Starting node for the broadcasting. If `None`, the algorithm
        returns the broadcast time of the graph instead.

    Returns
    -------
    int
        Minimum broadcast time of `node` in `G`, or broadcast time of `G`
        if no node is provided.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is directed or is a multigraph.

    NodeNotFound
        If `node` is not a node in `G`.

    NotATree
        If `G` is not a tree.

    References
    ----------
    .. [1] Harutyunyan, H. A. and Li, Z.
        "A Simple Construction of Broadcast Graphs."
        In Computing and Combinatorics. COCOON 2019
        (Ed. D. Z. Du and C. Tian.) Springer, pp. 240-253, 2019.
    """
    if node is not None and node not in G:
        err = f"node {node} not in G"
        raise nx.NodeNotFound(err)
    b_T, b_C = tree_broadcast_center(G)
    if node is None:
        return b_T + sum(1 for _ in nx.bfs_layers(G, b_C)) - 1
    return b_T + next(
        d for d, layer in enumerate(nx.bfs_layers(G, b_C)) if node in layer
    )
