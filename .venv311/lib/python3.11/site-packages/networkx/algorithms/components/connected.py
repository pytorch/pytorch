"""Connected components."""

import networkx as nx
from networkx.utils.decorators import not_implemented_for

from ...utils import arbitrary_element

__all__ = [
    "number_connected_components",
    "connected_components",
    "is_connected",
    "node_connected_component",
]


@not_implemented_for("directed")
@nx._dispatchable
def connected_components(G):
    """Generate connected components.

    The connected components of an undirected graph partition the graph into
    disjoint sets of nodes. Each of these sets induces a subgraph of graph
    `G` that is connected and not part of any larger connected subgraph.

    A graph is connected (:func:`is_connected`) if, for every pair of distinct
    nodes, there is a path between them. If there is a pair of nodes for
    which such path does not exist, the graph is not connected (also referred
    to as "disconnected").

    A graph consisting of a single node and no edges is connected.
    Connectivity is undefined for the null graph (graph with no nodes).

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph

    Yields
    ------
    comp : set
       A set of nodes in one connected component of the graph.

    Raises
    ------
    NetworkXNotImplemented
        If G is directed.

    Examples
    --------
    Generate a sorted list of connected components, largest first.

    >>> G = nx.path_graph(4)
    >>> nx.add_path(G, [10, 11, 12])
    >>> [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    [4, 3]

    If you only want the largest connected component, it's more
    efficient to use max instead of sort.

    >>> largest_cc = max(nx.connected_components(G), key=len)

    To create the induced subgraph of each component use:

    >>> S = [G.subgraph(c).copy() for c in nx.connected_components(G)]

    See Also
    --------
    number_connected_components
    is_connected
    number_weakly_connected_components
    number_strongly_connected_components

    Notes
    -----
    This function is for undirected graphs only. For directed graphs, use
    :func:`strongly_connected_components` or
    :func:`weakly_connected_components`.

    The algorithm is based on a Breadth-First Search (BFS) traversal and its
    time complexity is $O(n + m)$, where $n$ is the number of nodes and $m$ the
    number of edges in the graph.

    """
    seen = set()
    n = len(G)  # must be outside the loop to avoid performance hit with graph views
    for v in G:
        if v not in seen:
            c = _plain_bfs(G, n - len(seen), v)
            seen.update(c)
            yield c


@not_implemented_for("directed")
@nx._dispatchable
def number_connected_components(G):
    """Returns the number of connected components.

    The connected components of an undirected graph partition the graph into
    disjoint sets of nodes. Each of these sets induces a subgraph of graph
    `G` that is connected and not part of any larger connected subgraph.

    A graph is connected (:func:`is_connected`) if, for every pair of distinct
    nodes, there is a path between them. If there is a pair of nodes for
    which such path does not exist, the graph is not connected (also referred
    to as "disconnected").

    A graph consisting of a single node and no edges is connected.
    Connectivity is undefined for the null graph (graph with no nodes).

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    Returns
    -------
    n : integer
       Number of connected components

    Raises
    ------
    NetworkXNotImplemented
        If G is directed.

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (1, 2), (5, 6), (3, 4)])
    >>> nx.number_connected_components(G)
    3

    See Also
    --------
    connected_components
    is_connected
    number_weakly_connected_components
    number_strongly_connected_components

    Notes
    -----
    This function is for undirected graphs only. For directed graphs, use
    :func:`number_strongly_connected_components` or
    :func:`number_weakly_connected_components`.

    The algorithm is based on a Breadth-First Search (BFS) traversal and its
    time complexity is $O(n + m)$, where $n$ is the number of nodes and $m$ the
    number of edges in the graph.

    """
    return sum(1 for _ in connected_components(G))


@not_implemented_for("directed")
@nx._dispatchable
def is_connected(G):
    """Returns True if the graph is connected, False otherwise.

    A graph is connected if, for every pair of distinct nodes, there is a
    path between them. If there is a pair of nodes for which such path does
    not exist, the graph is not connected (also referred to as "disconnected").

    A graph consisting of a single node and no edges is connected.
    Connectivity is undefined for the null graph (graph with no nodes).

    Parameters
    ----------
    G : NetworkX Graph
       An undirected graph.

    Returns
    -------
    connected : bool
      True if the graph is connected, False otherwise.

    Raises
    ------
    NetworkXNotImplemented
        If G is directed.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> print(nx.is_connected(G))
    True

    See Also
    --------
    is_strongly_connected
    is_weakly_connected
    is_semiconnected
    is_biconnected
    connected_components

    Notes
    -----
    This function is for undirected graphs only. For directed graphs, use
    :func:`is_strongly_connected` or :func:`is_weakly_connected`.

    The algorithm is based on a Breadth-First Search (BFS) traversal and its
    time complexity is $O(n + m)$, where $n$ is the number of nodes and $m$ the
    number of edges in the graph.

    """
    n = len(G)
    if n == 0:
        raise nx.NetworkXPointlessConcept(
            "Connectivity is undefined for the null graph."
        )
    return len(next(connected_components(G))) == n


@not_implemented_for("directed")
@nx._dispatchable
def node_connected_component(G, n):
    """Returns the set of nodes in the component of graph containing node n.

    A connected component is a set of nodes that induces a subgraph of graph
    `G` that is connected and not part of any larger connected subgraph.

    A graph is connected (:func:`is_connected`) if, for every pair of distinct
    nodes, there is a path between them. If there is a pair of nodes for
    which such path does not exist, the graph is not connected (also referred
    to as "disconnected").

    A graph consisting of a single node and no edges is connected.
    Connectivity is undefined for the null graph (graph with no nodes).

    Parameters
    ----------
    G : NetworkX Graph
       An undirected graph.

    n : node label
       A node in G

    Returns
    -------
    comp : set
       A set of nodes in the component of G containing node n.

    Raises
    ------
    NetworkXNotImplemented
        If G is directed.

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (1, 2), (5, 6), (3, 4)])
    >>> nx.node_connected_component(G, 0)  # nodes of component that contains node 0
    {0, 1, 2}

    See Also
    --------
    connected_components

    Notes
    -----
    This function is for undirected graphs only.

    The algorithm is based on a Breadth-First Search (BFS) traversal and its
    time complexity is $O(n + m)$, where $n$ is the number of nodes and $m$ the
    number of edges in the graph.

    """
    return _plain_bfs(G, len(G), n)


def _plain_bfs(G, n, source):
    """A fast BFS node generator"""
    adj = G._adj
    seen = {source}
    nextlevel = [source]
    while nextlevel:
        thislevel = nextlevel
        nextlevel = []
        for v in thislevel:
            for w in adj[v]:
                if w not in seen:
                    seen.add(w)
                    nextlevel.append(w)
            if len(seen) == n:
                return seen
    return seen
