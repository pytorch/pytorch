"""
Graph isomorphism functions.
"""

import itertools
from collections import Counter

import networkx as nx
from networkx.exception import NetworkXError

__all__ = [
    "could_be_isomorphic",
    "fast_could_be_isomorphic",
    "faster_could_be_isomorphic",
    "is_isomorphic",
]


@nx._dispatchable(graphs={"G1": 0, "G2": 1})
def could_be_isomorphic(G1, G2, *, properties="dtc"):
    """Returns False if graphs are definitely not isomorphic.
    True does NOT guarantee isomorphism.

    Parameters
    ----------
    G1, G2 : graphs
       The two graphs `G1` and `G2` must be the same type.

    properties : str, default="dct"
       Determines which properties of the graph are checked. Each character
       indicates a particular property as follows:

       - if ``"d"`` in `properties`: degree of each node
       - if ``"t"`` in `properties`: number of triangles for each node
       - if ``"c"`` in `properties`: number of maximal cliques for each node

       Unrecognized characters are ignored. The default is ``"dtc"``, which
       compares the sequence of ``(degree, num_triangles, num_cliques)`` properties
       between `G1` and `G2`. Generally, ``properties="dt"`` would be faster, and
       ``properties="d"`` faster still. See Notes for additional details on
       property selection.

    Returns
    -------
    bool
       A Boolean value representing whether `G1` could be isomorphic with `G2`
       according to the specified `properties`.

    Notes
    -----
    The triangle sequence contains the number of triangles each node is part of.
    The clique sequence contains for each node the number of maximal cliques
    involving that node.

    Some properties are faster to compute than others. And there are other
    properties we could include and don't. But of the three properties listed here,
    comparing the degree distributions is the fastest. The "triangles" property
    is slower (and also a stricter version of "could") and the "maximal cliques"
    property is slower still, but usually faster than doing a full isomorphism
    check.
    """

    # Check global properties
    if G1.order() != G2.order():
        return False

    properties_to_check = set(properties)
    G1_props, G2_props = [], []

    def _properties_consistent():
        # Ravel the properties into a table with # nodes rows and # properties columns
        G1_ptable = [tuple(p[n] for p in G1_props) for n in G1]
        G2_ptable = [tuple(p[n] for p in G2_props) for n in G2]

        return sorted(G1_ptable) == sorted(G2_ptable)

    # The property table is built and checked as each individual property is
    # added. The reason for this is the building/checking the property table
    # is in general much faster than computing the properties, making it
    # worthwhile to check multiple times to enable early termination when
    # a subset of properties don't match

    # Degree sequence
    if "d" in properties_to_check:
        G1_props.append(G1.degree())
        G2_props.append(G2.degree())
        if not _properties_consistent():
            return False
    # Sequence of triangles per node
    if "t" in properties_to_check:
        G1_props.append(nx.triangles(G1))
        G2_props.append(nx.triangles(G2))
        if not _properties_consistent():
            return False
    # Sequence of maximal cliques per node
    if "c" in properties_to_check:
        G1_props.append(Counter(itertools.chain.from_iterable(nx.find_cliques(G1))))
        G2_props.append(Counter(itertools.chain.from_iterable(nx.find_cliques(G2))))
        if not _properties_consistent():
            return False

    # All checked conditions passed
    return True


def graph_could_be_isomorphic(G1, G2):
    """
    .. deprecated:: 3.5

       `graph_could_be_isomorphic` is a deprecated alias for `could_be_isomorphic`.
       Use `could_be_isomorphic` instead.
    """
    import warnings

    warnings.warn(
        "graph_could_be_isomorphic is deprecated, use `could_be_isomorphic` instead.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return could_be_isomorphic(G1, G2)


@nx._dispatchable(graphs={"G1": 0, "G2": 1})
def fast_could_be_isomorphic(G1, G2):
    """Returns False if graphs are definitely not isomorphic.

    True does NOT guarantee isomorphism.

    Parameters
    ----------
    G1, G2 : graphs
       The two graphs G1 and G2 must be the same type.

    Notes
    -----
    Checks for matching degree and triangle sequences. The triangle
    sequence contains the number of triangles each node is part of.
    """
    # Check global properties
    if G1.order() != G2.order():
        return False

    # Check local properties
    d1 = G1.degree()
    t1 = nx.triangles(G1)
    props1 = [[d, t1[v]] for v, d in d1]
    props1.sort()

    d2 = G2.degree()
    t2 = nx.triangles(G2)
    props2 = [[d, t2[v]] for v, d in d2]
    props2.sort()

    if props1 != props2:
        return False

    # OK...
    return True


def fast_graph_could_be_isomorphic(G1, G2):
    """
    .. deprecated:: 3.5

       `fast_graph_could_be_isomorphic` is a deprecated alias for
       `fast_could_be_isomorphic`. Use `fast_could_be_isomorphic` instead.
    """
    import warnings

    warnings.warn(
        "fast_graph_could_be_isomorphic is deprecated, use fast_could_be_isomorphic instead",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return fast_could_be_isomorphic(G1, G2)


@nx._dispatchable(graphs={"G1": 0, "G2": 1})
def faster_could_be_isomorphic(G1, G2):
    """Returns False if graphs are definitely not isomorphic.

    True does NOT guarantee isomorphism.

    Parameters
    ----------
    G1, G2 : graphs
       The two graphs G1 and G2 must be the same type.

    Notes
    -----
    Checks for matching degree sequences.
    """
    # Check global properties
    if G1.order() != G2.order():
        return False

    # Check local properties
    d1 = sorted(d for n, d in G1.degree())
    d2 = sorted(d for n, d in G2.degree())

    if d1 != d2:
        return False

    # OK...
    return True


def faster_graph_could_be_isomorphic(G1, G2):
    """
    .. deprecated:: 3.5

       `faster_graph_could_be_isomorphic` is a deprecated alias for
       `faster_could_be_isomorphic`. Use `faster_could_be_isomorphic` instead.
    """
    import warnings

    warnings.warn(
        "faster_graph_could_be_isomorphic is deprecated, use faster_could_be_isomorphic instead",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return faster_could_be_isomorphic(G1, G2)


@nx._dispatchable(
    graphs={"G1": 0, "G2": 1},
    preserve_edge_attrs="edge_match",
    preserve_node_attrs="node_match",
)
def is_isomorphic(G1, G2, node_match=None, edge_match=None):
    """Returns True if the graphs G1 and G2 are isomorphic and False otherwise.

    Parameters
    ----------
    G1, G2: graphs
        The two graphs G1 and G2 must be the same type.

    node_match : callable
        A function that returns True if node n1 in G1 and n2 in G2 should
        be considered equal during the isomorphism test.
        If node_match is not specified then node attributes are not considered.

        The function will be called like

           node_match(G1.nodes[n1], G2.nodes[n2]).

        That is, the function will receive the node attribute dictionaries
        for n1 and n2 as inputs.

    edge_match : callable
        A function that returns True if the edge attribute dictionary
        for the pair of nodes (u1, v1) in G1 and (u2, v2) in G2 should
        be considered equal during the isomorphism test.  If edge_match is
        not specified then edge attributes are not considered.

        The function will be called like

           edge_match(G1[u1][v1], G2[u2][v2]).

        That is, the function will receive the edge attribute dictionaries
        of the edges under consideration.

    Notes
    -----
    Uses the vf2 algorithm [1]_.

    Examples
    --------
    >>> import networkx.algorithms.isomorphism as iso

    For digraphs G1 and G2, using 'weight' edge attribute (default: 1)

    >>> G1 = nx.DiGraph()
    >>> G2 = nx.DiGraph()
    >>> nx.add_path(G1, [1, 2, 3, 4], weight=1)
    >>> nx.add_path(G2, [10, 20, 30, 40], weight=2)
    >>> em = iso.numerical_edge_match("weight", 1)
    >>> nx.is_isomorphic(G1, G2)  # no weights considered
    True
    >>> nx.is_isomorphic(G1, G2, edge_match=em)  # match weights
    False

    For multidigraphs G1 and G2, using 'fill' node attribute (default: '')

    >>> G1 = nx.MultiDiGraph()
    >>> G2 = nx.MultiDiGraph()
    >>> G1.add_nodes_from([1, 2, 3], fill="red")
    >>> G2.add_nodes_from([10, 20, 30, 40], fill="red")
    >>> nx.add_path(G1, [1, 2, 3, 4], weight=3, linewidth=2.5)
    >>> nx.add_path(G2, [10, 20, 30, 40], weight=3)
    >>> nm = iso.categorical_node_match("fill", "red")
    >>> nx.is_isomorphic(G1, G2, node_match=nm)
    True

    For multidigraphs G1 and G2, using 'weight' edge attribute (default: 7)

    >>> G1.add_edge(1, 2, weight=7)
    1
    >>> G2.add_edge(10, 20)
    1
    >>> em = iso.numerical_multiedge_match("weight", 7, rtol=1e-6)
    >>> nx.is_isomorphic(G1, G2, edge_match=em)
    True

    For multigraphs G1 and G2, using 'weight' and 'linewidth' edge attributes
    with default values 7 and 2.5. Also using 'fill' node attribute with
    default value 'red'.

    >>> em = iso.numerical_multiedge_match(["weight", "linewidth"], [7, 2.5])
    >>> nm = iso.categorical_node_match("fill", "red")
    >>> nx.is_isomorphic(G1, G2, edge_match=em, node_match=nm)
    True

    See Also
    --------
    numerical_node_match, numerical_edge_match, numerical_multiedge_match
    categorical_node_match, categorical_edge_match, categorical_multiedge_match

    References
    ----------
    .. [1]  L. P. Cordella, P. Foggia, C. Sansone, M. Vento,
       "An Improved Algorithm for Matching Large Graphs",
       3rd IAPR-TC15 Workshop  on Graph-based Representations in
       Pattern Recognition, Cuen, pp. 149-159, 2001.
       https://www.researchgate.net/publication/200034365_An_Improved_Algorithm_for_Matching_Large_Graphs
    """
    if G1.is_directed() and G2.is_directed():
        GM = nx.algorithms.isomorphism.DiGraphMatcher
    elif (not G1.is_directed()) and (not G2.is_directed()):
        GM = nx.algorithms.isomorphism.GraphMatcher
    else:
        raise NetworkXError("Graphs G1 and G2 are not of the same type.")

    gm = GM(G1, G2, node_match=node_match, edge_match=edge_match)

    return gm.is_isomorphic()
