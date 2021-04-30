class DiGraph:
    """Really simple unweighted direct graph data structure to track dependencies.

    The API is pretty much the same as networkx so if you add something just
    copy their API.
    """

    def __init__(self):
        # Dict of node -> dict of arbitrary attributes
        self._node = {}
        # Nested dict of node -> successor node -> nothing.
        # (didn't implement edge data)
        self._succ = {}

    def add_node(self, n, **kwargs):
        """Add a node to the graph.

        Adding a node that already exists in the graph is an error. This is
        a difference from the networkx API, but adding a node multiple times
        is a sign in the bug in the dependency graph implementation.

        Args:
            n: the node. Can we any object that is a valid dict key.
            **kwargs: any metadata you want to attach to the node.
        """
        if n not in self._node:
            self._node[n] = kwargs
            self._succ[n] = {}
        else:
            raise ValueError(f"Tried to add a node twice: '{n}'.")

    def add_edge(self, u, v):
        # add nodes
        if u not in self._node:
            self._node[u] = {}
            self._succ[u] = {}
        if v not in self._node:
            self._node[v] = {}
            self._succ[v] = {}

        # add the edge
        self._succ[u][v] = True

    def successors(self, n):
        """Returns an iterator over successor nodes of n."""
        try:
            return iter(self._succ[n])
        except KeyError as e:
            raise ValueError(f"The node {n} is not in the digraph.") from e

    @property
    def edges(self):
        for n, successors in self._succ.items():
            for succ in successors:
                yield n, succ

    @property
    def nodes(self):
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
