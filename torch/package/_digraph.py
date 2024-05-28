from collections import deque
from typing import List, Set


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

        # Keep track of the order in which nodes are added to
        # the graph.
        self._node_order = {}
        self._insertion_idx = 0

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
            self._node_order[n] = self._insertion_idx
            self._insertion_idx += 1
        else:
            self._node[n].update(kwargs)

    def add_edge(self, u, v):
        """Add an edge to graph between nodes ``u`` and ``v``

        ``u`` and ``v`` will be created if they do not already exist.
        """
        # add nodes
        self.add_node(u)
        self.add_node(v)

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

    def forward_transitive_closure(self, src: str) -> Set[str]:
        """Returns a set of nodes that are reachable from src"""

        result = set(src)
        working_set = deque(src)
        while len(working_set) > 0:
            cur = working_set.popleft()
            for n in self.successors(cur):
                if n not in result:
                    result.add(n)
                    working_set.append(n)
        return result

    def backward_transitive_closure(self, src: str) -> Set[str]:
        """Returns a set of nodes that are reachable from src in reverse direction"""

        result = set(src)
        working_set = deque(src)
        while len(working_set) > 0:
            cur = working_set.popleft()
            for n in self.predecessors(cur):
                if n not in result:
                    result.add(n)
                    working_set.append(n)
        return result

    def all_paths(self, src: str, dst: str):
        """Returns a subgraph rooted at src that shows all the paths to dst."""

        result_graph = DiGraph()
        # First compute forward transitive closure of src (all things reachable from src).
        forward_reachable_from_src = self.forward_transitive_closure(src)

        if dst not in forward_reachable_from_src:
            return result_graph

        # Second walk the reverse dependencies of dst, adding each node to
        # the output graph iff it is also present in forward_reachable_from_src.
        # we don't use backward_transitive_closures for optimization purposes
        working_set = deque(dst)
        while len(working_set) > 0:
            cur = working_set.popleft()
            for n in self.predecessors(cur):
                if n in forward_reachable_from_src:
                    result_graph.add_edge(n, cur)
                    # only explore further if its reachable from src
                    working_set.append(n)

        return result_graph.to_dot()

    def first_path(self, dst: str) -> List[str]:
        """Returns a list of nodes that show the first path that resulted in dst being added to the graph."""
        path = []

        while dst:
            path.append(dst)
            candidates = self._pred[dst].keys()
            dst, min_idx = "", None
            for candidate in candidates:
                idx = self._node_order.get(candidate, None)
                if idx is None:
                    break
                if min_idx is None or idx < min_idx:
                    min_idx = idx
                    dst = candidate

        return list(reversed(path))

    def to_dot(self) -> str:
        """Returns the dot representation of the graph.

        Returns:
            A dot representation of the graph.
        """
        edges = "\n".join(f'"{f}" -> "{t}";' for f, t in self.edges)
        return f"""\
digraph G {{
rankdir = LR;
node [shape=box];
{edges}
}}
"""
