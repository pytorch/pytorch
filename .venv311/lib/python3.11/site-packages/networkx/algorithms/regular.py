"""Functions for computing and verifying regular graphs."""

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = ["is_regular", "is_k_regular", "k_factor"]


@nx._dispatchable
def is_regular(G):
    """Determines whether a graph is regular.

    A regular graph is a graph where all nodes have the same degree. A regular
    digraph is a graph where all nodes have the same indegree and all nodes
    have the same outdegree.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    bool
        Whether the given graph or digraph is regular.

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (2, 3), (3, 4), (4, 1)])
    >>> nx.is_regular(G)
    True

    """
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("Graph has no nodes.")
    n1 = nx.utils.arbitrary_element(G)
    if not G.is_directed():
        d1 = G.degree(n1)
        return all(d1 == d for _, d in G.degree)
    else:
        d_in = G.in_degree(n1)
        in_regular = (d_in == d for _, d in G.in_degree)
        d_out = G.out_degree(n1)
        out_regular = (d_out == d for _, d in G.out_degree)
        return all(in_regular) and all(out_regular)


@not_implemented_for("directed")
@nx._dispatchable
def is_k_regular(G, k):
    """Determines whether the graph ``G`` is a k-regular graph.

    A k-regular graph is a graph where each vertex has degree k.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    bool
        Whether the given graph is k-regular.

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (2, 3), (3, 4), (4, 1)])
    >>> nx.is_k_regular(G, k=3)
    False

    """
    return all(d == k for n, d in G.degree)


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable(preserve_edge_attrs=True, returns_graph=True)
def k_factor(G, k, matching_weight="weight"):
    """Compute a `k`-factor of a graph.

    A `k`-factor of a graph is a spanning `k`-regular subgraph.
    A spanning `k`-regular subgraph of `G` is a subgraph that contains
    each node of `G` and a subset of the edges of `G` such that each
    node has degree `k`.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    k : int
        The degree of the `k`-factor.

    matching_weight: string, optional (default="weight")
        Edge attribute name corresponding to the edge weight.
        If not present, the edge is assumed to have weight 1.
        Used for finding the max-weighted perfect matching.

    Returns
    -------
    NetworkX graph
        A `k`-factor of `G`.

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (2, 3), (3, 4), (4, 1)])
    >>> KF = nx.k_factor(G, k=1)
    >>> KF.edges()
    EdgeView([(1, 2), (3, 4)])

    References
    ----------
    .. [1] "An algorithm for computing simple k-factors.",
       Meijer, Henk, Yurai Núñez-Rodríguez, and David Rappaport,
       Information processing letters, 2009.
    """
    # Validate minimum degree requirement.
    if any(d < k for _, d in G.degree):
        raise nx.NetworkXUnfeasible("Graph contains a vertex with degree less than k")

    g = G.copy()
    gadgets = []

    # Replace each node with a gadget.
    for node, degree in G.degree:
        is_large = k >= degree / 2.0

        # Create gadget nodes.
        outer = [(node, i) for i in range(degree)]
        if is_large:
            core = [(node, i) for i in range(degree, 2 * degree - k)]
            inner = []
        else:
            core = [(node, i) for i in range(2 * degree, 2 * degree + k)]
            inner = [(node, i) for i in range(degree, 2 * degree)]

        # Connect gadget nodes to neighbors.
        g.add_edges_from(zip(outer, inner))
        for outer_n, (neighbor, attrs) in zip(outer, g[node].items()):
            g.add_edge(outer_n, neighbor, **attrs)

        # Add internal edges.
        g.add_edges_from((u, v) for u in core for v in (outer if is_large else inner))

        g.remove_node(node)
        gadgets.append((node, outer, core, inner))

    # Find perfect matching.
    m = nx.max_weight_matching(g, maxcardinality=True, weight=matching_weight)
    if not nx.is_perfect_matching(g, m):
        raise nx.NetworkXUnfeasible(
            "Cannot find k-factor because no perfect matching exists"
        )

    # Keep only edges in matching.
    g.remove_edges_from(e for e in g.edges if e not in m and e[::-1] not in m)

    # Restore original nodes and remove gadgets.
    for node, outer, core, inner in gadgets:
        g.add_node(node)
        core_set = set(core)
        for outer_n in outer:
            for neighbor, attrs in g._adj[outer_n].items():
                if neighbor not in core_set:
                    g.add_edge(node, neighbor, **attrs)
                    break
        g.remove_nodes_from(outer + core + inner)

    return g
