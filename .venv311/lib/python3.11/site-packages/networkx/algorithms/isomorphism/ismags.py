"""
ISMAGS Algorithm
================

Provides a Python implementation of the ISMAGS algorithm. [1]_

ISMAGS does a symmetry analysis to find the constraints on isomorphisms if
we preclude yielding isomorphisms that differ by a symmetry of the subgraph.
For example, if the subgraph contains a 4-cycle, every isomorphism would have a
symmetric version with the nodes rotated relative to the original isomorphism.
By encoding these symmetries as constraints we reduce the search space for
isomorphisms and we also simplify processing the resulting isomorphisms.

ISMAGS finds (subgraph) isomorphisms between two graphs, taking the
symmetry of the subgraph into account. In most cases the VF2 algorithm is
faster (at least on small graphs) than this implementation, but in some cases
there are an exponential number of isomorphisms that are symmetrically
equivalent. In that case, the ISMAGS algorithm will provide only one isomorphism
per symmetry group, speeding up finding isomorphisms and avoiding the task of
post-processing many effectively identical isomorphisms.

>>> petersen = nx.petersen_graph()
>>> ismags = nx.isomorphism.ISMAGS(petersen, petersen)
>>> isomorphisms = list(ismags.isomorphisms_iter(symmetry=False))
>>> len(isomorphisms)
120
>>> isomorphisms = list(ismags.isomorphisms_iter(symmetry=True))
>>> answer = [{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}]
>>> answer == isomorphisms
True

In addition, this implementation also provides an interface to find the
largest common induced subgraph [2]_ between any two graphs, again taking
symmetry into account. Given ``graph`` and ``subgraph`` the algorithm will remove
nodes from the ``subgraph`` until ``subgraph`` is isomorphic to a subgraph of
``graph``. Since only the symmetry of ``subgraph`` is taken into account it is
worth thinking about how you provide your graphs:

>>> graph1 = nx.path_graph(4)
>>> graph2 = nx.star_graph(3)
>>> ismags = nx.isomorphism.ISMAGS(graph1, graph2)
>>> ismags.is_isomorphic()
False
>>> largest_common_subgraph = list(ismags.largest_common_subgraph())
>>> answer = [{1: 0, 0: 1, 2: 2}, {2: 0, 1: 1, 3: 2}]
>>> answer == largest_common_subgraph
True
>>> ismags2 = nx.isomorphism.ISMAGS(graph2, graph1)
>>> largest_common_subgraph = list(ismags2.largest_common_subgraph())
>>> answer = [
...     {1: 0, 0: 1, 2: 2},
...     {1: 0, 0: 1, 3: 2},
...     {2: 0, 0: 1, 1: 2},
...     {2: 0, 0: 1, 3: 2},
...     {3: 0, 0: 1, 1: 2},
...     {3: 0, 0: 1, 2: 2},
... ]
>>> answer == largest_common_subgraph
True

However, when not taking symmetry into account, it doesn't matter:

>>> largest_common_subgraph = list(ismags.largest_common_subgraph(symmetry=False))
>>> answer = [
...     {1: 0, 0: 1, 2: 2},
...     {1: 0, 2: 1, 0: 2},
...     {2: 0, 1: 1, 3: 2},
...     {2: 0, 3: 1, 1: 2},
...     {1: 0, 0: 1, 2: 3},
...     {1: 0, 2: 1, 0: 3},
...     {2: 0, 1: 1, 3: 3},
...     {2: 0, 3: 1, 1: 3},
...     {1: 0, 0: 2, 2: 3},
...     {1: 0, 2: 2, 0: 3},
...     {2: 0, 1: 2, 3: 3},
...     {2: 0, 3: 2, 1: 3},
... ]
>>> answer == largest_common_subgraph
True
>>> largest_common_subgraph = list(ismags2.largest_common_subgraph(symmetry=False))
>>> answer = [
...     {1: 0, 0: 1, 2: 2},
...     {1: 0, 0: 1, 3: 2},
...     {2: 0, 0: 1, 1: 2},
...     {2: 0, 0: 1, 3: 2},
...     {3: 0, 0: 1, 1: 2},
...     {3: 0, 0: 1, 2: 2},
...     {1: 1, 0: 2, 2: 3},
...     {1: 1, 0: 2, 3: 3},
...     {2: 1, 0: 2, 1: 3},
...     {2: 1, 0: 2, 3: 3},
...     {3: 1, 0: 2, 1: 3},
...     {3: 1, 0: 2, 2: 3},
... ]
>>> answer == largest_common_subgraph
True

Notes
-----
- Node and edge equality is assumed to be transitive: if A is equal to B, and
  B is equal to C, then A is equal to C.

- With a method that yields subgraph isomorphisms, we can construct functions like
  ``is_subgraph_isomorphic`` by checking for any yielded mapping. And functions like
  ``is_isomorphic`` by checking whether the subgraph has the same number of nodes as
  the graph and is also subgraph isomorphic. This subpackage also allows a
  ``symmetry`` bool keyword argument so you can find isomorphisms with or
  without the symmetry constraints.

- For more information, see [2]_ and the documentation for :class:`ISMAGS`
  which includes a description of the algorithm.

References
----------
.. [1] M. Houbraken, S. Demeyer, T. Michoel, P. Audenaert, D. Colle,
   M. Pickavet, "The Index-Based Subgraph Matching Algorithm with General
   Symmetries (ISMAGS): Exploiting Symmetry for Faster Subgraph
   Enumeration", PLoS One 9(5): e97896, 2014.
   https://doi.org/10.1371/journal.pone.0097896
.. [2] https://en.wikipedia.org/wiki/Maximum_common_induced_subgraph
"""

__all__ = ["ISMAGS"]

import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps

import networkx as nx


def are_all_equal(iterable):
    """
    Returns ``True`` if and only if all elements in `iterable` are equal; and
    ``False`` otherwise.

    Parameters
    ----------
    iterable: collections.abc.Iterable
        The container whose elements will be checked.

    Returns
    -------
    bool
        ``True`` iff all elements in `iterable` compare equal, ``False``
        otherwise.
    """
    try:
        shape = iterable.shape
    except AttributeError:
        pass
    else:
        if len(shape) > 1:
            message = "The function does not works on multidimensional arrays."
            raise NotImplementedError(message) from None

    iterator = iter(iterable)
    first = next(iterator, None)
    return all(item == first for item in iterator)


def make_partition(items, test, check=True):
    """
    Partitions items into sets based on the outcome of ``test(item1, item2)``.
    Pairs of items for which `test` returns `True` end up in the same set.

    Parameters
    ----------
    items : collections.abc.Iterable[collections.abc.Hashable]
        Items to partition
    test : collections.abc.Callable[collections.abc.Hashable, collections.abc.Hashable]
        A function that will be called with 2 arguments, taken from items.
        Should return `True` if those 2 items match/tests so need to end up in the same
        part of the partition, and `False` otherwise.
    check : bool optional (default: True)
        If ``True``, check that the resulting partition satisfies the match criteria.
        Every item should match every item in its part and none outside the part.

    Returns
    -------
    list[set]
        A partition as a list of sets (the parts). Each set contains some of
        the items in `items`, such that all items are in exactly one part and every
        pair of items in each part matches. The following will be true:
        ``all(thing_matcher(*pair) for pair in itertools.combinations(items, 2))``

    Notes
    -----
    The function `test` is assumed to be transitive: if ``test(a, b)`` and
    ``test(b, c)`` return ``True``, then ``test(a, c)`` must also be ``True``.
    The function `test` is assumed to be commutative: if ``test(a, b)``
    returns ``True`` then ``test(b, a)`` returns ``True``.
    """
    partition = []
    for item in items:
        for part in partition:
            p_item = next(iter(part))
            if test(item, p_item):
                part.add(item)
                break
        else:  # No break
            partition.append({item})

    if check:
        if not all(
            test(t1, t2) and test(t2, t1)
            for part in partition
            for t1, t2 in itertools.combinations(part, 2)
        ):
            raise nx.NetworkXError(
                f"\nInvalid partition created with {test}.\n"
                "Some items in a part do not match. This leads to\n"
                f"{partition=}"
            )
        if not all(
            not test(t1, t2) and not test(t2, t1)
            for p1 in partition
            for p2 in partition
            if p1 != p2
            for t1, t2 in itertools.product(p1, p2)
        ):
            raise nx.NetworkXError(
                f"\nInvalid partition created with {test}.\n"
                "Some items match multiple parts. This leads to\n"
                f"{partition=}"
            )
    return [set(part) for part in partition]


def node_to_part_ID_dict(partition):
    """
    Creates a dictionary that maps each item in each part to the index of
    the part to which it belongs.

    Parameters
    ----------
    partition: collections.abc.Sequence[collections.abc.Iterable]
        As returned by :func:`make_partition`.

    Returns
    -------
    dict
    """
    return {node: ID for ID, part in enumerate(partition) for node in part}


def color_degree_by_node(G, n_colors, e_colors):
    """Returns a dict by node to counts of edge and node color for that node.

    This returns a dict by node to a 2-tuple of node color and degree by
    (edge color and nbr color). E.g. ``{0: (1, {(0, 2): 5})}`` means that
    node ``0`` has node type 1 and has 5 edges of type 0 that go to nodes of type 2.
    Thus, this is a measure of degree (edge count) by color of edge and color
    of the node on the other side of that edge.

    For directed graphs the degree counts is a 2-tuple of (in, out) degree counts.

    Ideally, if edge_match is None, this could get simplified to just the node
    color on the other end of the edge. Similarly if node_match is None then only
    edge color is tracked. And if both are None, we simply count the number of edges.
    """
    # n_colors might be incomplete when using `largest_common_subgraph()`
    if len(n_colors) < len(G):
        for n, nbrs in G.adjacency():
            if n not in n_colors:
                n_colors[n] = None
                for v in nbrs:
                    e_colors[n, v] = None
    # undirected colored degree
    if not G.is_directed():
        return {
            u: (n_colors[u], Counter((e_colors[u, v], n_colors[v]) for v in nbrs))
            for u, nbrs in G.adjacency()
        }
    # directed colored out and in degree
    return {
        u: (
            n_colors[u],
            Counter((e_colors[u, v], n_colors[v]) for v in nbrs),
            Counter((e_colors[v, u], n_colors[v]) for v in G._pred[u]),
        )
        for u, nbrs in G.adjacency()
    }


class EdgeLookup:
    """Class to handle getitem for undirected edges.

    Note that ``items()`` iterates over one of the two representations of the edge
    (u, v) and (v, u). So this technically doesn't violate the Mapping
    invariant that (k,v) pairs reported by ``items()`` satisfy ``.__getitem__(k) == v``.
    But we are violating the spirit of the protocol by having keys available
    for lookup by ``__getitem__`` that are not reported by ``items()``.

    Note that if we used frozensets for undirected edges we would have the same
    behavior we see here. You could ``__getitem__`` either ``{u, v}`` or ``{v, u}``
    and get the same value -- yet ``items()`` would only report one of the two.
    So from that perspective we *are* following the Mapping protocol. Our keys
    are undirected edges. We are using 2-tuples as an imperfect representation
    of these edges. We are not using 2-tuples as keys. Only as imperfect edges
    and we use the edges as keys.
    """

    def __init__(self, edge_dict):
        self.edge_dict = edge_dict

    def __getitem__(self, edge):
        if edge in self.edge_dict:
            return self.edge_dict[edge]
        return self.edge_dict[edge[::-1]]

    def items(self):
        return self.edge_dict.items()


class ISMAGS:
    """
    Implements the ISMAGS subgraph matching algorithm. [1]_ ISMAGS stands for
    "Index-based Subgraph Matching Algorithm with General Symmetries". As the
    name implies, it is symmetry aware and will only generate non-symmetric
    isomorphisms.

    Attributes
    ----------
    graph: networkx.Graph
    subgraph: networkx.Graph

    Notes
    -----
    ISMAGS does a symmetry analysis to find the constraints on isomorphisms if
    we preclude yielding isomorphisms that differ by a symmetry of the subgraph.
    For example, if the subgraph is a 4-cycle, every isomorphism would have a
    symmetric version with the nodes rotated relative to the original isomorphism.
    By encoding these symmetries as constraints we reduce the search space for
    isomorphisms and we also simplify processing the resulting isomorphisms.

    **Symmetry Analysis**

    The constraints in ISMAGS are based off the handling in ``nauty`` and its many
    variants, in particular ``saucy``, as discussed in the ISMAGS paper [1]_.
    That paper cites [3]_ for details on symmetry handling. Figure 2 of [3]_
    describes the DFS approach to symmetries used here and relying on a data structure
    called an Ordered Pair Partitions(OPP). This consists of a pair of partitions
    where each part represents nodes with the same degree-by-color over all colors.
    We refine these partitions simultaneously in a way to result in permutations
    of the nodes that preserve the graph structure. We thus find automorphisms
    for the subgraph of interest. From those we identify pairs of nodes which
    are structurally equivalent. We then constrain our problem by requiring the
    first of the pair to always be assigned first in the isomorphism -- thus
    constraining the isomorphisms reported to only one example from the set of all
    symmetrically equivalent isomorphisms. These constraints are computed once
    based on the subgraph symmetries and then used throughout the DFS search for
    isomorphisms.

    Finding the symmetries involves a DFS of the OPP wherein we "couple" a node
    to a node in its degree-by-color part of the partition. This "coupling" is done
    via assigning a new color in the top partition to the node being coupled,
    and the same new color in the bottom partition to the node being coupled to.
    This new color has only one node in each partition. The new color also requires
    that we "refine" both top and bottom partitions by splitting parts until each
    part represents a common degree-by-color value. Those refinements introduce
    new colors as the parts are split during refinement. Parts do not get combined
    during refinement. So the coupling/refining process always results in at least
    one new part with only one node in both the top and bottom partition. In practice
    we usually refine into many new one-node parts in both partitions.
    We continue in this way until each node has its own part/color in the top partition
    -- and the node in the bottom partition with that color is the symmetric node.
    That is, an OPP represents an automorphism, and thus a symmetry
    of the subgraph when each color has a single node in the top partition and a single
    node in the bottom partition. From those automorphisms we build up a set of nodes
    that can be obtained from each other by symmetry (they are mutually symmetric).
    That set of nodes is called an "orbit" of the subgraph under symmetry.

    After finding the orbits for one symmetry, we backtrack in the DFS by removing the
    latest coupling and replacing it with a coupling from the same top node to a new
    bottom node in its degree-by-color grouping. When all possible couplings for that
    node are considered we backtrack to the previously coupled node and recouple in
    a DFS manner.

    We can prune the DFS search tree in helpful ways. The paper [2]_ demonstrates 6
    situations of interest in the DFS where pruning is possible:

    - An **Automorphism OPP** is an OPP where every part in both partitions
      contains a single node. The mapping/automorphism is found by mapping
      each top node to the bottom node in the same color part. For example,
      ``[({1}, {2}, {3}); ({2}, {3}, {1})]`` represents the mapping of each
      node to the next in a triangle. It rotates the nodes around the triangle.
    - The **Identity OPP** is the first automorphism found during the DFS. It
      appears on the left side of the DFS tree and is first due to our ordering of
      coupling nodes to be in an arbitrary but fixed ordering of the nodes. This
      automorphism does not show any symmetries, but it ensures the orbit for each
      node includes itself and it sets us up for handling the symmetries. Note that
      a subgraph with no symmetries will only have the identity automorphism.
    - A **Non-isomorphic OPP** occurs when refinement creates a different number of
      parts in the top partition than in the bottom partition. This means no symmetries
      will be found during further processing of that subtree of the DFS. We prune
      the subtree and continue.
    - A **Matching OPP** is such that each top part that has more than one node is
      in fact equal to the bottom part with the same color. The many-node-parts match
      exactly. The single-node parts then represent symmetries that do not permute
      any matching nodes. Matching OPPs arise while finding the Identity Mapping. But
      the single-node parts are identical in the two partitions, so no useful symmetries
      are found. But after the Identity Mapping is found, every Matching OPP encountered
      will have different nodes in at least two single-node parts of the same color.
      So these positions in the DFS provide us with symmetries without any
      need to find the whole automorphism. We can prune the subtree, update the orbits
      and backtrack. Any larger symmetries that combine these symmetries with symmetries
      of the many-node-parts do not need to be explored because the symmetry "generators"
      found in this way provide a basis for all symmetries. We will find the symmetry
      generators of the many-node-parts at another subtree of the DFS.
    - An **Orbit Pruning OPP** is an OPP where the node coupling to be considered next
      has both nodes already known to be in the same orbit. We have already identified
      those permutations when we discovered the orbit. So we can prune the resulting
      subtree. This is the primary pruning discussed in [1]_.
    - A **Coset Point** in the DFS is a point of the tree when a node is first
      back-tracked. That is, its couplings have all been analyzed once and we backtrack
      to its parent. So, said another way, when a node is backtracked to and is about to
      be mapped to a different node for the first time, its child in the DFS has been
      completely analysed. Thus the orbit for that child at this point in the DFS is
      the full orbit for symmetries involving only that child or larger nodes in the
      node order. All smaller nodes are mapped to themselves.
      This orbit is due to symmetries not involving smaller nodes. Such an orbit is
      called the "coset" of that node. The Coset Point does not lead to pruning or to
      more symmetries. It is the point in the process where we store the **coset** of
      the node being backtracked. We use the cosets to construct the symmetry
      constraints.

    Once the pruned DFS tree has been traversed, we have collected the cosets of some
    special nodes. Often most nodes are not coupled during the progression down the left
    side of the DFS. They are separated from other nodes during the partition refinement
    process after coupling. So they never get coupled directly. Thus the number of cosets
    we find is typically many fewer than the number of nodes.

    We turn those cosets into constraints on the nodes when building non-symmetric
    isomorphisms. The node whose coset is used is paired with each other node in the
    coset. These node-pairs form the constraints. During isomorphism construction we
    always select the first of the constraint before the other. This removes subtrees
    from the DFS traversal space used to build isomorphisms.

    The constraints we obtain via symmetry analysis of the subgraph are used for
    finding non-symmetric isomorphisms. We prune the isomorphism tree based on
    the constraints we obtain from the symmetry analysis.

    **Isomorphism Construction**

    Once we have symmetry constraints on the isomorphisms, ISMAGS constructs the allowed
    isomorphisms by mapping each node of the subgraph to all possible nodes (with the
    same degree-by-color) from the graph. We partition all nodes into degree-by-color
    parts and order the subgraph nodes we consider using smallest part size first.
    The idea is to try to map the most difficult subgraph nodes first (most difficult
    here means least number of graph candidates).

    By considering each potential subgraph node to graph candidate mapping image in turn,
    we perform a DFS traversal of partial mappings. If the mapping is rejected due to
    the graph neighbors not matching the degree-by-color of the subgraph neighbors, or
    rejected due to the constraints imposed from symmetry, we prune that subtree and
    consider a new graph candidate node for that subgraph node. When no more graph
    candidates remain we backtrack to the previous node in the mapping and consider a
    new graph candidate for that node. If we ever get to a depth where all subgraph nodes
    are mapped and no structural requirements or symmetry constraints are violated,
    we have found an isomorphism. We yield that mapping and backtrack to find other
    isomorphisms.

    As we visit more neighbors, the graph candidate nodes have to satisfy more structural
    restrictions. As described in the ISMAGS paper, [1]_, we store each set of structural
    restrictions separately as a set of possible candidate nodes rather than computing
    the intersection of that set with the known graph candidates for the subgraph node.
    We delay taking the intersection until that node is selected to be in the mapping.
    While choosing the node with fewest candidates, we avoid computing the intersection
    by using the size of the minimal set to be intersected rather than the size of the
    intersection. This may make the node ordering slightly worse via a savings of
    many intersections most of which are not ever needed.

    References
    ----------
    .. [1] M. Houbraken, S. Demeyer, T. Michoel, P. Audenaert, D. Colle,
       M. Pickavet, "The Index-Based Subgraph Matching Algorithm with General
       Symmetries (ISMAGS): Exploiting Symmetry for Faster Subgraph
       Enumeration", PLoS One 9(5): e97896, 2014.
       https://doi.org/10.1371/journal.pone.0097896
    .. [2] https://en.wikipedia.org/wiki/Maximum_common_induced_subgraph
    .. [3] Hadi Katebi, Karem A. Sakallah and Igor L. Markov
       "Graph Symmetry Detection and Canonical Labeling: Differences and Synergies"
       in "Turing-100. The Alan Turing Centenary" Ed: A. Voronkov p. 181 -- 195, (2012).
       https://doi.org/10.29007/gzc1 https://arxiv.org/abs/1208.6271
    """

    def __init__(self, graph, subgraph, node_match=None, edge_match=None, cache=None):
        """
        Parameters
        ----------
        graph: networkx.Graph
        subgraph: networkx.Graph
        node_match: collections.abc.Callable or None
            Function used to determine whether two nodes are equivalent. Its
            signature should look like ``f(n1: dict, n2: dict) -> bool``, with
            `n1` and `n2` node property dicts. See also
            :func:`~networkx.algorithms.isomorphism.categorical_node_match` and
            friends.
            If `None`, all nodes are considered equal.
        edge_match: collections.abc.Callable or None
            Function used to determine whether two edges are equivalent. Its
            signature should look like ``f(e1: dict, e2: dict) -> bool``, with
            `e1` and `e2` edge property dicts. See also
            :func:`~networkx.algorithms.isomorphism.categorical_edge_match` and
            friends.
            If `None`, all edges are considered equal.
        cache: collections.abc.Mapping
            A cache used for caching graph symmetries.
        """
        if graph.is_directed() != subgraph.is_directed():
            raise ValueError("Directed and undirected graphs cannot be compared.")

        # TODO: allow for precomputed partitions and colors
        self.graph = graph
        self.subgraph = subgraph
        self._symmetry_cache = cache
        # Naming conventions are taken from the original paper.
        # For your sanity:
        #   sg: subgraph
        #   g: graph
        #   e: edge(s)
        #   n: node(s)
        # So: sgn means "subgraph nodes".
        node_parts = self.create_aligned_partitions(
            node_match, self.subgraph.nodes, self.graph.nodes
        )
        self._sgn_partition, self._gn_partition, self.N_node_colors = node_parts
        self._sgn_colors = node_to_part_ID_dict(self._sgn_partition)
        self._gn_colors = node_to_part_ID_dict(self._gn_partition)

        edge_partitions = self.create_aligned_partitions(
            edge_match, self.subgraph.edges(), self.graph.edges()
        )
        self._sge_partition, self._ge_partition, self.N_edge_colors = edge_partitions
        if self.graph.is_directed():
            self._sge_colors = node_to_part_ID_dict(self._sge_partition)
            self._ge_colors = node_to_part_ID_dict(self._ge_partition)
        else:  # allow lookups (u, v) or (v, u)
            self._sge_colors = EdgeLookup(node_to_part_ID_dict(self._sge_partition))
            self._ge_colors = EdgeLookup(node_to_part_ID_dict(self._ge_partition))

    def create_aligned_partitions(self, thing_matcher, sg_things, g_things):
        """Partitions of "things" (nodes or edges) from subgraph and graph
        based on function `thing_matcher`.

        Returns: sg_partition, g_partition, number_of_matched_parts

        The first `number_of_matched_parts` parts in each partition
        match in order, e.g. 2nd part matches other's 2nd part.
        Warning: nodes in parts after that have no matching nodes in the other graph.
        For morphisms those nodes can't appear in the mapping.
        """
        if thing_matcher is None:
            sg_partition = [set(sg_things)]
            g_partition = [set(g_things)]
            return sg_partition, g_partition, 1

        # Use thing_matcher to create a partition
        # Note: isinstance(G.edges(), OutEdgeDataView) is only true for multi(di)graph
        sg_multiedge = isinstance(sg_things, nx.classes.reportviews.OutEdgeDataView)
        g_multiedge = isinstance(g_things, nx.classes.reportviews.OutEdgeDataView)
        if not sg_multiedge:

            def sg_match(thing1, thing2):
                return thing_matcher(sg_things[thing1], sg_things[thing2])

        else:  # multiedges (note nodes of multigraphs use simple case above)

            def sg_match(thing1, thing2):
                (u1, v1), (u2, v2) = thing1, thing2
                return thing_matcher(self.subgraph[u1][v1], self.subgraph[u2][v2])

        if not g_multiedge:

            def g_match(thing1, thing2):
                return thing_matcher(g_things[thing1], g_things[thing2])

        else:  # multiedges (note nodes of multigraphs use simple case above)

            def g_match(thing1, thing2):
                (u1, v1), (u2, v2) = thing1, thing2
                return thing_matcher(self.graph[u1][v1], self.graph[u2][v2])

        sg_partition = make_partition(sg_things, sg_match)
        g_partition = make_partition(g_things, g_match)

        # Align order of g_partition to that of sg_partition
        sgc_to_gc = {}
        gc_to_sgc = {}
        sN, N = len(sg_partition), len(g_partition)
        for sgc, gc in itertools.product(range(sN), range(N)):
            sgt = next(iter(sg_partition[sgc]))
            gt = next(iter(g_partition[gc]))
            sgt_ = sg_things[sgt] if not sg_multiedge else self.subgraph[sgt[0]][sgt[1]]
            gt_ = g_things[gt] if not g_multiedge else self.graph[gt[0]][gt[1]]
            if thing_matcher(sgt_, gt_):
                # TODO: remove these two if-checks when confident they never arise
                # The `check` feature in match_partitions should ensure they do not
                if sgc in sgc_to_gc:
                    raise nx.NetworkXError(
                        f"\nMatching function {thing_matcher} seems faulty.\n"
                        f"Partition found: {sg_partition=}\n"
                        f"So {sgt} in subgraph part {sg_partition[sgc]} matches two "
                        f"graph parts {g_partition[gc]} and "
                        f"{g_partition[sgc_to_gc[sgc]]}\n"
                    )
                if gc in gc_to_sgc:
                    raise nx.NetworkXError(
                        f"\nMatching function seems broken: {thing_matcher}\n"
                        f"Partitions found: {g_partition=} {sg_partition=}\n"
                        f"So {gt} in graph part {g_partition[gc]} matches two "
                        f"subgraph parts {sg_partition[sgc]} and "
                        f"{sg_partition[gc_to_sgc[gc]]}\n"
                    )
                sgc_to_gc[sgc] = gc
                gc_to_sgc[gc] = sgc
        ## return two lists and the number of partitions that match.
        new_order = [
            (sg_partition[sgc], g_partition[gc]) for sgc, gc in sgc_to_gc.items()
        ]
        Ncolors = len(new_order)
        if Ncolors:
            new_sg_p, new_g_p = [list(x) for x in zip(*new_order)]
        else:
            new_sg_p, new_g_p = [], []
        if Ncolors < sN:
            extra = [sg_partition[c] for c in range(sN) if c not in sgc_to_gc]
            new_sg_p = list(new_sg_p) + extra
            new_g_p = list(new_g_p) + [set()] * len(extra)
        if Ncolors < N:
            extra = [g_partition[c] for c in range(N) if c not in gc_to_sgc]
            new_g_p = list(new_g_p) + extra
            new_sg_p = list(new_sg_p) + [set()] * len(extra)

        return new_sg_p, new_g_p, Ncolors

    def find_isomorphisms(self, symmetry=True):
        """Find all subgraph isomorphisms between subgraph and graph

        Finds isomorphisms where :attr:`subgraph` <= :attr:`graph`.

        Parameters
        ----------
        symmetry: bool
            Whether symmetry should be taken into account. If False, found
            isomorphisms may be symmetrically equivalent.

        Yields
        ------
        dict
            The found isomorphism mappings of {graph_node: subgraph_node}.
        """
        # The networkx VF2 algorithm is slightly funny in when it yields an
        # empty dict and when not.
        if not self.subgraph:
            yield {}
            return
        elif not self.graph:
            return
        elif len(self.graph) < len(self.subgraph):
            return
        elif len(self._sgn_partition) > self.N_node_colors:
            # some subgraph nodes have a color that doesn't occur in graph
            return
        elif len(self._sge_partition) > self.N_edge_colors:
            # some subgraph edges have a color that doesn't occur in graph
            return

        if symmetry:
            cosets = self.analyze_subgraph_symmetry()
            # Turn cosets into constraints.
            constraints = [(n, co) for n, cs in cosets.items() for co in cs if n != co]
        else:
            constraints = []

        cand_sets = self._get_node_color_candidate_sets()

        lookahead_candidates = self._get_color_degree_candidates()
        for sgn, lookahead_cands in lookahead_candidates.items():
            cand_sets[sgn].add(frozenset(lookahead_cands))

        if any(cand_sets.values()):
            # Choose start node based on a heuristic for the min # of candidates
            # Heuristic here is length of smallest frozenset in candidates' set
            # of frozensets for that node. Using the smallest length avoids
            # computing the intersection of the frozensets for each node.
            start_sgn = min(cand_sets, key=lambda n: min(len(x) for x in cand_sets[n]))
            cand_sets[start_sgn] = (frozenset.intersection(*cand_sets[start_sgn]),)
            yield from self._map_nodes(start_sgn, cand_sets, constraints)
        return

    def _get_color_degree_candidates(self):
        """
        Returns a mapping of {subgraph node: set of graph nodes} for
        which the graph nodes are feasible mapping candidate_sets for the
        subgraph node, as determined by looking ahead one edge.
        """
        g_deg = color_degree_by_node(self.graph, self._gn_colors, self._ge_colors)
        sg_deg = color_degree_by_node(self.subgraph, self._sgn_colors, self._sge_colors)

        return {
            sgn: {
                gn
                for gn, (_, *g_counts) in g_deg.items()
                if all(
                    sg_cnt <= g_counts[idx][color]
                    for idx, counts in enumerate(needed_counts)
                    for color, sg_cnt in counts.items()
                )
            }
            for sgn, (_, *needed_counts) in sg_deg.items()
        }

    def largest_common_subgraph(self, symmetry=True):
        """
        Find the largest common induced subgraphs between :attr:`subgraph` and
        :attr:`graph`.

        Parameters
        ----------
        symmetry: bool
            Whether symmetry should be taken into account. If False, found
            largest common subgraphs may be symmetrically equivalent.

        Yields
        ------
        dict
            The found isomorphism mappings of {graph_node: subgraph_node}.
        """
        # The networkx VF2 algorithm is slightly funny in when it yields an
        # empty dict and when not.
        if not self.subgraph:
            yield {}
            return
        elif not self.graph:
            return

        if symmetry:
            cosets = self.analyze_subgraph_symmetry()
            # Turn cosets into constraints.
            constraints = [(n, cn) for n, cs in cosets.items() for cn in cs if n != cn]
        else:
            constraints = []

        candidate_sets = self._get_node_color_candidate_sets()

        if any(candidate_sets.values()):
            relevant_parts = self._sgn_partition[: self.N_node_colors]
            to_be_mapped = {frozenset(n for p in relevant_parts for n in p)}
            yield from self._largest_common_subgraph(
                candidate_sets, constraints, to_be_mapped
            )
        else:
            return

    def analyze_subgraph_symmetry(self):
        """
        Find a minimal set of permutations and corresponding co-sets that
        describe the symmetry of ``self.subgraph``, given the node and edge
        equalities given by `node_partition` and `edge_colors`, respectively.

        Returns
        -------
        dict[collections.abc.Hashable, set[collections.abc.Hashable]]
            The found co-sets. The co-sets is a dictionary of
            ``{node key: set of node keys}``.
            Every key-value pair describes which ``values`` can be interchanged
            without changing nodes less than ``key``.
        """
        partition, edge_colors = self._sgn_partition, self._sge_colors

        if self._symmetry_cache is not None:
            key = hash(
                (
                    tuple(self.subgraph.nodes),
                    tuple(self.subgraph.edges),
                    tuple(map(tuple, node_partition)),
                    tuple(edge_colors.items()),
                    self.subgraph.is_directed(),
                )
            )
            if key in self._symmetry_cache:
                return self._symmetry_cache[key]
        partition = self._refine_node_partition(self.subgraph, partition, edge_colors)
        cosets = self._process_ordered_pair_partitions(
            self.subgraph, partition, partition, edge_colors
        )
        if self._symmetry_cache is not None:
            self._symmetry_cache[key] = cosets
        return cosets

    def is_isomorphic(self, symmetry=False):
        """
        Returns True if :attr:`graph` is isomorphic to :attr:`subgraph` and
        False otherwise.

        Returns
        -------
        bool
        """
        return len(self.subgraph) == len(self.graph) and self.subgraph_is_isomorphic(
            symmetry
        )

    def subgraph_is_isomorphic(self, symmetry=False):
        """
        Returns True if a subgraph of :attr:`graph` is isomorphic to
        :attr:`subgraph` and False otherwise.

        Returns
        -------
        bool
        """
        # symmetry=False, since we only need to know whether there is any
        # example; figuring out all symmetry elements probably costs more time
        # than it gains.
        isom = next(self.subgraph_isomorphisms_iter(symmetry=symmetry), None)
        return isom is not None

    def isomorphisms_iter(self, symmetry=True):
        """
        Does the same as :meth:`find_isomorphisms` if :attr:`graph` and
        :attr:`subgraph` have the same number of nodes.
        """
        if len(self.graph) == len(self.subgraph):
            yield from self.subgraph_isomorphisms_iter(symmetry=symmetry)

    def subgraph_isomorphisms_iter(self, symmetry=True):
        """Alternative name for :meth:`find_isomorphisms`."""
        return self.find_isomorphisms(symmetry)

    def _get_node_color_candidate_sets(self):
        """
        Per node in subgraph find all nodes in graph that have the same color.
        Stored as a dict-of-set-of-frozenset. The dict is keyed by node to a
        collection of frozensets of graph nodes. Each of these frozensets are
        a restriction. The node can be mapped only to nodes in the frozenset.
        Thus it must be mapped to nodes in the intersection of all these sets.
        We store the sets to delay taking the intersection of them. This helps
        for two reasons: Firstly any duplicate restriction sets can be ignored;
        Secondly, some nodes will not need the intersection to be constructed.
        Note: a dict-of-list-of-set would store duplicate sets in the list and
        we want to avoid that. But I wonder if checking hash/equality when `add`ing
        removes the benefit of avoiding computing intersections.
        """
        candidate_sets = defaultdict(set)
        for sgn in self.subgraph.nodes:
            sgn_color = self._sgn_colors[sgn]
            if sgn_color >= self.N_node_colors:  # color has no candidates
                candidate_sets[sgn]  # creates empty set entry in defaultdict
            else:
                candidate_sets[sgn].add(frozenset(self._gn_partition[sgn_color]))
        return dict(candidate_sets)

    @classmethod
    def _refine_node_partition(cls, graph, partition, edge_colors):
        def equal_color(node1, node2):
            return color_degree[node1] == color_degree[node2]

        node_colors = node_to_part_ID_dict(partition)
        color_degree = color_degree_by_node(graph, node_colors, edge_colors)
        while not all(are_all_equal(color_degree[n] for n in p) for p in partition):
            partition = [
                p
                for part in partition
                for p in (
                    [part]
                    if are_all_equal(color_degree[n] for n in part)
                    else sorted(make_partition(part, equal_color, check=False), key=len)
                )
            ]
            node_colors = node_to_part_ID_dict(partition)
            color_degree = color_degree_by_node(graph, node_colors, edge_colors)
        return partition

    def _map_nodes(self, sgn, candidate_sets, constraints, to_be_mapped=None):
        """
        Find all subgraph isomorphisms honoring constraints.
        The collection `candidate_sets` is stored as a dict-of-set-of-frozenset.
        The dict is keyed by node to a collection of candidate frozensets. Any
        viable candidate must belong to all the frozensets in the collection.
        So each frozenset added to the collection is a restriction on the candidates.

        According to the paper, we store the collection of sets rather than their
        intersection to delay computing many intersections with the hope of avoiding
        them completely. Having the middle collection be a set also means that
        duplicate restrictions on candidates are ignored, avoiding another intersection.
        """
        # shortcuts for speed
        subgraph = self.subgraph
        subgraph_adj = subgraph._adj
        graph = self.graph
        graph_adj = graph._adj
        self_ge_partition = self._ge_partition
        self_sge_colors = self._sge_colors
        is_directed = subgraph.is_directed()

        gn_ID_to_node = list(graph)
        gn_node_to_ID = {n: id for id, n in enumerate(graph)}

        mapping = {}
        rev_mapping = {}
        if to_be_mapped is None:
            to_be_mapped = subgraph_adj.keys()

        # Note that we don't copy candidates here. This means we leak
        # information between the branches of the search. This is intentional!
        # Specifically, we modify candidates here. That's OK because we substitute
        # the set of frozensets with a set containing the frozenset intersection.
        # So, it doesn't change the membership rule or the length rule for sorting.
        # Membership: any candidate must be an element of each of the frozensets.
        # Length: length of the intersection set. Use heuristic min(len of frozensets).
        # This intersection improves future length heuristics which can only occur
        # after this element of the queu is popped. But it means future additional
        # restriction frozensets that duplicate previous ones are not ignored.
        sgn_candidates = frozenset.intersection(*candidate_sets[sgn])
        candidate_sets[sgn] = {sgn_candidates}
        queue = [(sgn, candidate_sets, iter(sgn_candidates))]
        while queue:  # DFS over all possible mappings
            sgn, candidate_sets, sgn_cand_iter = queue[-1]

            for gn in sgn_cand_iter:
                # We're going to try to map sgn to gn.
                if gn in rev_mapping:
                    continue  # pragma: no cover

                # REDUCTION and COMBINATION
                if sgn in mapping:
                    old_gn = mapping[sgn]
                    del rev_mapping[old_gn]
                mapping[sgn] = gn
                rev_mapping[gn] = sgn
                # BASECASE
                if len(mapping) == len(to_be_mapped):
                    yield rev_mapping.copy()
                    del mapping[sgn]
                    del rev_mapping[gn]
                    continue
                left_to_map = to_be_mapped - mapping.keys()

                # We copy the candidates dict. But it is not a deepcopy.
                # This avoids inner set copies, yet still allows updates b/c setitem
                # changes sgn in new dict without changing original set.
                # Below be careful to not change the sets of frozensets.
                cand_sets = candidate_sets.copy()

                # update the candidate_sets for unmapped sgn based on sgn mapped
                if not is_directed:
                    sgn_nbrs = subgraph_adj[sgn]
                    not_gn_nbrs = graph_adj.keys() - graph_adj[gn].keys()
                    for sgn2 in left_to_map:
                        # edge color must match when sgn2 connected to sgn
                        if sgn2 not in sgn_nbrs:
                            gn2_cands = not_gn_nbrs
                        else:
                            g_edges = self_ge_partition[self_sge_colors[sgn, sgn2]]
                            gn2_cands = {n for e in g_edges if gn in e for n in e}
                        # Node color compatibility should be taken care of by the
                        # initial candidate lists made by find_subgraphs

                        # Add gn2_cands to the right collection.
                        # Do not change the original set. So do not use |= operator
                        cand_sets[sgn2] = cand_sets[sgn2] | {frozenset(gn2_cands)}
                else:  # directed
                    sgn_nbrs = subgraph_adj[sgn].keys()
                    sgn_preds = subgraph._pred[sgn].keys()
                    not_gn_nbrs = (
                        graph_adj.keys() - graph_adj[gn].keys() - graph._pred[gn].keys()
                    )
                    for sgn2 in left_to_map:
                        # edge color must match when sgn2 connected to sgn
                        if sgn2 not in sgn_nbrs:
                            if sgn2 not in sgn_preds:
                                gn2_cands = not_gn_nbrs
                            else:  # sgn2 in sgn_preds
                                g_edges = self_ge_partition[self_sge_colors[sgn2, sgn]]
                                gn2_cands = {e[0] for e in g_edges if gn == e[1]}
                        else:
                            if sgn2 not in sgn_preds:
                                g_edges = self_ge_partition[self_sge_colors[sgn, sgn2]]
                                gn2_cands = {e[1] for e in g_edges if gn == e[0]}
                            else:
                                # gn2 must have correct color in both directions
                                g_edges = self_ge_partition[self_sge_colors[sgn, sgn2]]
                                gn2_cands = {e[1] for e in g_edges if gn == e[0]}
                                g_edges = self_ge_partition[self_sge_colors[sgn2, sgn]]
                                gn2_cands &= {e[0] for e in g_edges if gn == e[1]}
                        # Do not change the original set. So do not use |= operator
                        cand_sets[sgn2] = cand_sets[sgn2] | {frozenset(gn2_cands)}

                for sgn2 in left_to_map:
                    # symmetry must match. constraints mean gn2>gn iff sgn2>sgn
                    if (sgn, sgn2) in constraints:
                        gn2_cands = set(gn_ID_to_node[gn_node_to_ID[gn] + 1 :])
                    elif (sgn2, sgn) in constraints:
                        gn2_cands = set(gn_ID_to_node[: gn_node_to_ID[gn]])
                    else:
                        continue  # pragma: no cover
                    # Do not change the original set. So do not use |= operator
                    cand_sets[sgn2] = cand_sets[sgn2] | {frozenset(gn2_cands)}

                # The next node is the one that is unmapped and has fewest candidates
                # Use the heuristic of the min size of the frozensets rather than
                # intersection of all frozensets to delay computing intersections.
                new_sgn = min(
                    left_to_map, key=lambda n: min(len(x) for x in cand_sets[n])
                )
                new_sgn_candidates = frozenset.intersection(*cand_sets[new_sgn])
                if not new_sgn_candidates:
                    continue
                cand_sets[new_sgn] = {new_sgn_candidates}
                queue.append((new_sgn, cand_sets, iter(new_sgn_candidates)))
                break
            else:  # all gn candidates tried for sgn.
                queue.pop()
                if sgn in mapping:
                    del rev_mapping[mapping[sgn]]
                    del mapping[sgn]

    def _largest_common_subgraph(self, candidates, constraints, to_be_mapped=None):
        """
        Find all largest common subgraphs honoring constraints.
        """
        # to_be_mapped is a set of frozensets of subgraph nodes
        if to_be_mapped is None:
            to_be_mapped = {frozenset(self.subgraph.nodes)}

        # The LCS problem is basically a repeated subgraph isomorphism problem
        # with smaller and smaller subgraphs. We store the nodes that are
        # "part of" the subgraph in to_be_mapped, and we make it a little
        # smaller every iteration.

        current_size = len(next(iter(to_be_mapped), []))

        found_iso = False
        if current_size <= len(self.graph):
            # There's no point in trying to find isomorphisms of
            # graph >= subgraph if subgraph has more nodes than graph.

            # Try the isomorphism first with the nodes with lowest ID. So sort
            # them. Those are more likely to be part of the final correspondence.
            # In theory, this makes finding the first answer(s) faster.
            for nodes in sorted(to_be_mapped, key=sorted):
                # Find the isomorphism between subgraph[to_be_mapped] <= graph
                next_sgn = min(nodes, key=lambda n: min(len(x) for x in candidates[n]))
                isomorphs = self._map_nodes(
                    next_sgn, candidates, constraints, to_be_mapped=nodes
                )

                # This is effectively `yield from isomorphs`, except that we look
                # whether an item was yielded.
                try:
                    item = next(isomorphs)
                except StopIteration:
                    pass
                else:
                    yield item
                    yield from isomorphs
                    found_iso = True

        # BASECASE
        if found_iso or current_size == 1:
            # Shrinking has no point because either 1) we end up with a smaller
            # common subgraph (and we want the largest), or 2) there'll be no
            # more subgraph.
            return

        left_to_be_mapped = set()
        for nodes in to_be_mapped:
            for sgn in nodes:
                # We're going to remove sgn from to_be_mapped, but subject to
                # symmetry constraints. We know that for every constraint we
                # have those subgraph nodes are equal. So whenever we would
                # remove the lower part of a constraint, remove the higher
                # instead. This is all dealth with by _remove_node. And because
                # left_to_be_mapped is a set, we don't do double work.

                # And finally, make the subgraph one node smaller.
                # REDUCTION
                new_nodes = self._remove_node(sgn, nodes, constraints)
                left_to_be_mapped.add(new_nodes)
        # COMBINATION
        yield from self._largest_common_subgraph(
            candidates, constraints, to_be_mapped=left_to_be_mapped
        )

    @staticmethod
    def _remove_node(node, nodes, constraints):
        """
        Returns a new set where node has been removed from nodes, subject to
        symmetry constraints. We know, that for every constraint we have
        those subgraph nodes are equal. So whenever we would remove the
        lower part of a constraint, remove the higher instead.
        """
        while True:
            for low, high in constraints:
                if low == node and high in nodes:
                    node = high
                    break
            else:  # no break, couldn't find node in constraints
                return frozenset(nodes - {node})

    @staticmethod
    def _get_permutations_by_length(items):
        """
        Get all permutations of items, but only permute items with the same
        length.

        >>> found = list(ISMAGS._get_permutations_by_length([{1}, {2}, {3, 4}, {4, 5}]))
        >>> answer = [
        ...     (({1}, {2}), ({3, 4}, {4, 5})),
        ...     (({1}, {2}), ({4, 5}, {3, 4})),
        ...     (({2}, {1}), ({3, 4}, {4, 5})),
        ...     (({2}, {1}), ({4, 5}, {3, 4})),
        ... ]
        >>> found == answer
        True
        """
        by_len = defaultdict(list)
        for item in items:
            by_len[len(item)].append(item)

        return list(
            itertools.product(
                *(itertools.permutations(by_len[l]) for l in sorted(by_len))
            )
        )

    def _refine_opp(cls, graph, top, bottom, edge_colors):
        def equal_color(node1, node2):
            return color_degree[node1] == color_degree[node2]

        top = cls._refine_node_partition(graph, top, edge_colors)

        possible_bottoms = [bottom]
        while possible_bottoms:
            bottom = possible_bottoms.pop()
            node_colors = node_to_part_ID_dict(bottom)
            color_degree = color_degree_by_node(graph, node_colors, edge_colors)
            if all(are_all_equal(color_degree[n] for n in p) for p in bottom):
                if len(top) == len(bottom):
                    yield top, bottom
                # else Non-isomorphic OPP (pruned here)
                # either way continue to next possible bottom
                continue
            # refine bottom partition
            more_bottoms = [[]]
            for part in bottom:
                if len(part) == 1 or are_all_equal(color_degree[node] for node in part):
                    for new_bottom in more_bottoms:
                        new_bottom.append(part)
                else:
                    # This part needs to be refined
                    refined_part = make_partition(part, equal_color, check=False)
                    R = len(refined_part)
                    if R == 1 or R == len({len(p) for p in refined_part}):
                        # no two parts have same length -- simple case
                        for n_p in more_bottoms:
                            n_p.extend(sorted(refined_part, key=len))
                    else:
                        # Any part might match any other part with the same size.
                        # Before refinement they were the same color. So we need to
                        # include all possible orderings/colors within each size.
                        permutations = cls._get_permutations_by_length(refined_part)
                        # Add all permutations of the refined parts to each possible
                        # bottom. So the number of new possible bottoms is multiplied
                        # by the number of permutations of the refined parts.
                        new_partitions = []
                        for new_partition in more_bottoms:
                            for p in permutations:
                                # p is tuple-of-tuples-of-sets. Flatten to list-of-sets
                                flat_p = [s for tup in p for s in tup]
                                new_partitions.append(new_partition + flat_p)
                        more_bottoms = new_partitions

            # reverse more_bottoms to keep the "finding identity" bottom first
            possible_bottoms.extend(more_bottoms[::-1])

    @staticmethod
    def _find_permutations(top_partition, bottom_partition):
        """
        Return a set of 2-tuples of nodes. These nodes are not equal
        but are mapped to each other in the symmetry represented by this OPP.
        Swapping all the 2-tuples of nodes in this set permutes the nodes
        but retains the graph structure. Thus it is a symmetry of the subgraph.
        """
        # Find permutations
        permutations = set()
        for top, bot in zip(top_partition, bottom_partition):
            if len(top) > 1 or len(bot) > 1:
                # ignore parts with > 1 element when they are equal
                # These are called Matching OPPs in Katebi 2012.
                # Symmetries in matching partitions are built by considering
                # only parts that have 1 element.
                if top == bot:
                    continue
                raise IndexError(
                    "Not all nodes are matched. This is"
                    f" impossible: {top_partition}, {bottom_partition}"
                )
            # top and bot have only one element
            elif top != bot:
                permutations.add(frozenset((next(iter(top)), next(iter(bot)))))
        return permutations

    def _process_ordered_pair_partitions(
        self,
        graph,
        top_partition,
        bottom_partition,
        edge_colors,
    ):
        if all(len(top) <= 1 for top in top_partition):
            # no symmetries. Each node unique.
            return {}

        # first mapping found is the identity mapping
        finding_identity = True

        orbit_id = {node: orbit_i for orbit_i, node in enumerate(graph)}
        orbits = [{node} for node in graph]
        cosets = {}

        node_to_ID = {n: i for i, n in enumerate(graph)}
        sort_by_ID = node_to_ID.__getitem__

        def _load_next_queue_entry(queue, top_partition, bottom_partition):
            # find smallest node (by ID) in a |part|>1 and its partition index
            unmapped_nodes = (
                (node_to_ID[node], node, idx)
                for idx, t_part in enumerate(top_partition)
                for node in t_part
                if len(t_part) > 1
            )
            _, node, part_i = min(unmapped_nodes)
            b_part = bottom_partition[part_i]
            node2_iter = iter(sorted(b_part, key=sort_by_ID))

            queue.append([top_partition, bottom_partition, node, part_i, node2_iter])

        queue = []
        _load_next_queue_entry(queue, top_partition, bottom_partition)

        while queue:
            tops, bottoms, node, part_i, node2_iter = queue[-1]

            for node2 in node2_iter:
                if node != node2 and orbit_id[node] == orbit_id[node2]:
                    # Orbit prune
                    continue

                # couple node to node2
                new_top_part = {node}
                new_bot_part = {node2}

                new_top = [top.copy() for top in tops]
                new_top[part_i] -= new_top_part
                new_top.insert(part_i, new_top_part)

                new_bot = [bot.copy() for bot in bottoms]
                new_bot[part_i] -= new_bot_part
                new_bot.insert(part_i, new_bot_part)

                # collect OPPs
                opps = self._refine_opp(graph, new_top, new_bot, edge_colors)
                new_q = []
                for opp in opps:
                    # Use OPP to find any of: Identity, Automorphism or Matching OPPs
                    # else load the OPP onto queue for further exploration
                    # Note that we check for Orbit pruning later because orbits may
                    # be updated while OPP is sitting on the queue.
                    # Note that we check for Non-isomorphic OPPs in `_refine_opp`.
                    if finding_identity:
                        # Note: allow zero size parts in identity check
                        #       b/c largest_common_subgraph allows empty parts
                        if all(len(top) <= 1 for top in opp[0]):
                            # Identity found. Set flag. Can now prune Matching OPPs
                            finding_identity = False
                            continue
                    elif all(len(t) <= 1 or t == b for t, b in zip(*opp)):
                        # Found a symmetry! (Full mapping or Matching OPP)
                        # update orbits using the permutations from the OPP.
                        permutations = self._find_permutations(*opp)
                        for n1, n2 in permutations:
                            orb1 = orbit_id[n1]
                            orb2 = orbit_id[n2]
                            if orb1 != orb2:
                                orbit_set2 = orbits[orb2]
                                orbits[orb1].update(orbit_set2)
                                orbits[orb2] = set()
                                orbit_id.update((n, orb1) for n in orbit_set2)
                        continue

                    _load_next_queue_entry(new_q, *opp)
                # reverse order to maintain node order DFS (Identity comes first)
                queue.extend(new_q[::-1])
                break
            else:  # no more node2 options
                queue.pop()
                if node not in cosets:
                    # coset of `node` is its orbit at the time `node` has completed
                    # its first DFS traversal. DFS is about to go to previous node.
                    # Make copy so future orbit changes do not change the coset.
                    cosets[node] = orbits[orbit_id[node]].copy()
        return cosets
