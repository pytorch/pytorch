class DiGraph:
    """Really simple unweighted directed graph data structure to track dependencies.

    The API is pretty much the same as networkx so if you add something just
    copy their API.
    """

    def __init__(self):
        # Dict of node -> dict of arbitrary attributes
        self._node = {}
        # Nested dict of node -> successor node -> nothing.
        # (didn't implement edge data)
        self._succ = {}
        # Nested dict of node -> predecessor node -> nothing.
        self._pred = {}

    def add_node(self, n, **kwargs):
        """Add a node to the graph.

        Args:
            n: the node. Can we any object that is a valid dict key.
            **kwargs: any attributes you want to attach to the node.
        """
        if n not in self._node:
            self._node[n] = kwargs
            self._succ[n] = {}
            self._pred[n] = {}
        else:
            self._node[n].update(kwargs)

    def add_edge(self, u, v):
        """Add an edge to graph between nodes ``u`` and ``v``

        ``u`` and ``v`` will be created if they do not already exist.
        """
        # add nodes
        if u not in self._node:
            self._node[u] = {}
            self._succ[u] = {}
            self._pred[u] = {}
        if v not in self._node:
            self._node[v] = {}
            self._succ[v] = {}
            self._pred[v] = {}

        # add the edge
        self._succ[u][v] = True
        self._pred[v][u] = True

    def successors(self, n):
        """Returns an iterator over successor nodes of n."""
        try:
            return iter(self._succ[n])
        except KeyError as e:
            raise ValueError(f"The node {n} is not in the digraph.") from e

    def predecessors(self, n):
        """Returns an iterator over predecessors nodes of n."""
        try:
            return iter(self._pred[n])
        except KeyError as e:
            raise ValueError(f"The node {n} is not in the digraph.") from e

    @property
    def edges(self):
        """Returns an iterator over all edges (u, v) in the graph"""
        for n, successors in self._succ.items():
            for succ in successors:
                yield n, succ

    @property
    def nodes(self):
        """Returns a dictionary of all nodes to their attributes."""
        return self._node

    def __iter__(self):
        """Iterate over the nodes."""
        return iter(self._node)

    def __contains__(self, n):
        """Returns True if ``n`` is a node in the graph, False otherwise."""
        try:
            return n in self._node
        except TypeError:
            return False
