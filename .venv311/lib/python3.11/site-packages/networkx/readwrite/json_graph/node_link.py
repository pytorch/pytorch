import warnings
from itertools import count

import networkx as nx

__all__ = ["node_link_data", "node_link_graph"]


def _to_tuple(x):
    """Converts lists to tuples, including nested lists.

    All other non-list inputs are passed through unmodified. This function is
    intended to be used to convert potentially nested lists from json files
    into valid nodes.

    Examples
    --------
    >>> _to_tuple([1, 2, [3, 4]])
    (1, 2, (3, 4))
    """
    if not isinstance(x, tuple | list):
        return x
    return tuple(map(_to_tuple, x))


def node_link_data(
    G,
    *,
    source="source",
    target="target",
    name="id",
    key="key",
    edges="edges",
    nodes="nodes",
):
    """Returns data in node-link format that is suitable for JSON serialization
    and use in JavaScript documents.

    Parameters
    ----------
    G : NetworkX graph
    source : string
        A string that provides the 'source' attribute name for storing NetworkX-internal graph data.
    target : string
        A string that provides the 'target' attribute name for storing NetworkX-internal graph data.
    name : string
        A string that provides the 'name' attribute name for storing NetworkX-internal graph data.
    key : string
        A string that provides the 'key' attribute name for storing NetworkX-internal graph data.
    edges : string
        A string that provides the 'edges' attribute name for storing NetworkX-internal graph data.
    nodes : string
        A string that provides the 'nodes' attribute name for storing NetworkX-internal graph data.

    Returns
    -------
    data : dict
       A dictionary with node-link formatted data.

    Raises
    ------
    NetworkXError
        If the values of 'source', 'target' and 'key' are not unique.

    Examples
    --------
    >>> from pprint import pprint
    >>> G = nx.Graph([("A", "B")])
    >>> data1 = nx.node_link_data(G)
    >>> pprint(data1)
    {'directed': False,
     'edges': [{'source': 'A', 'target': 'B'}],
     'graph': {},
     'multigraph': False,
     'nodes': [{'id': 'A'}, {'id': 'B'}]}

    To serialize with JSON

    >>> import json
    >>> s1 = json.dumps(data1)
    >>> pprint(s1)
    ('{"directed": false, "multigraph": false, "graph": {}, "nodes": [{"id": "A"}, '
     '{"id": "B"}], "edges": [{"source": "A", "target": "B"}]}')


    A graph can also be serialized by passing `node_link_data` as an encoder function.

    >>> s1 = json.dumps(G, default=nx.node_link_data)
    >>> pprint(s1)
    ('{"directed": false, "multigraph": false, "graph": {}, "nodes": [{"id": "A"}, '
     '{"id": "B"}], "edges": [{"source": "A", "target": "B"}]}')

    The attribute names for storing NetworkX-internal graph data can
    be specified as keyword options.

    >>> H = nx.gn_graph(2)
    >>> data2 = nx.node_link_data(
    ...     H, edges="links", source="from", target="to", nodes="vertices"
    ... )
    >>> pprint(data2)
    {'directed': True,
     'graph': {},
     'links': [{'from': 1, 'to': 0}],
     'multigraph': False,
     'vertices': [{'id': 0}, {'id': 1}]}

    Notes
    -----
    Graph, node, and edge attributes are stored in this format.  Note that
    attribute keys will be converted to strings in order to comply with JSON.

    Attribute 'key' is only used for multigraphs.

    To use `node_link_data` in conjunction with `node_link_graph`,
    the keyword names for the attributes must match.

    See Also
    --------
    node_link_graph, adjacency_data, tree_data
    """
    multigraph = G.is_multigraph()

    # Allow 'key' to be omitted from attrs if the graph is not a multigraph.
    key = None if not multigraph else key
    if len({source, target, key}) < 3:
        raise nx.NetworkXError("Attribute names are not unique.")
    data = {
        "directed": G.is_directed(),
        "multigraph": multigraph,
        "graph": G.graph,
        nodes: [{**G.nodes[n], name: n} for n in G],
    }
    if multigraph:
        data[edges] = [
            {**d, source: u, target: v, key: k}
            for u, v, k, d in G.edges(keys=True, data=True)
        ]
    else:
        data[edges] = [{**d, source: u, target: v} for u, v, d in G.edges(data=True)]
    return data


@nx._dispatchable(graphs=None, returns_graph=True)
def node_link_graph(
    data,
    directed=False,
    multigraph=True,
    *,
    source="source",
    target="target",
    name="id",
    key="key",
    edges="edges",
    nodes="nodes",
):
    """Returns graph from node-link data format.

    Useful for de-serialization from JSON.

    Parameters
    ----------
    data : dict
        node-link formatted graph data

    directed : bool
        If True, and direction not specified in data, return a directed graph.

    multigraph : bool
        If True, and multigraph not specified in data, return a multigraph.

    source : string
        A string that provides the 'source' attribute name for storing NetworkX-internal graph data.
    target : string
        A string that provides the 'target' attribute name for storing NetworkX-internal graph data.
    name : string
        A string that provides the 'name' attribute name for storing NetworkX-internal graph data.
    key : string
        A string that provides the 'key' attribute name for storing NetworkX-internal graph data.
    edges : string
        A string that provides the 'edges' attribute name for storing NetworkX-internal graph data.
    nodes : string
        A string that provides the 'nodes' attribute name for storing NetworkX-internal graph data.

    Returns
    -------
    G : NetworkX graph
        A NetworkX graph object

    Examples
    --------

    Create data in node-link format by converting a graph.

    >>> from pprint import pprint
    >>> G = nx.Graph([("A", "B")])
    >>> data = nx.node_link_data(G)
    >>> pprint(data)
    {'directed': False,
     'edges': [{'source': 'A', 'target': 'B'}],
     'graph': {},
     'multigraph': False,
     'nodes': [{'id': 'A'}, {'id': 'B'}]}

    Revert data in node-link format to a graph.

    >>> H = nx.node_link_graph(data)
    >>> print(H.edges)
    [('A', 'B')]

    To serialize and deserialize a graph with JSON,

    >>> import json
    >>> d = json.dumps(nx.node_link_data(G))
    >>> H = nx.node_link_graph(json.loads(d))
    >>> print(G.edges, H.edges)
    [('A', 'B')] [('A', 'B')]


    Notes
    -----
    Attribute 'key' is only used for multigraphs.

    To use `node_link_data` in conjunction with `node_link_graph`,
    the keyword names for the attributes must match.

    See Also
    --------
    node_link_data, adjacency_data, tree_data
    """
    multigraph = data.get("multigraph", multigraph)
    directed = data.get("directed", directed)
    if multigraph:
        graph = nx.MultiGraph()
    else:
        graph = nx.Graph()
    if directed:
        graph = graph.to_directed()

    # Allow 'key' to be omitted from attrs if the graph is not a multigraph.
    key = None if not multigraph else key
    graph.graph = data.get("graph", {})
    c = count()
    for d in data[nodes]:
        node = _to_tuple(d.get(name, next(c)))
        nodedata = {str(k): v for k, v in d.items() if k != name}
        graph.add_node(node, **nodedata)
    for d in data[edges]:
        src = tuple(d[source]) if isinstance(d[source], list) else d[source]
        tgt = tuple(d[target]) if isinstance(d[target], list) else d[target]
        if not multigraph:
            edgedata = {str(k): v for k, v in d.items() if k != source and k != target}
            graph.add_edge(src, tgt, **edgedata)
        else:
            ky = d.get(key, None)
            edgedata = {
                str(k): v
                for k, v in d.items()
                if k != source and k != target and k != key
            }
            graph.add_edge(src, tgt, ky, **edgedata)
    return graph
