"""Functions for splitting a network into two communities (finding a bipartition)."""

import random
from copy import deepcopy
from itertools import count

import networkx as nx

__all__ = [
    "kernighan_lin_bisection",
    "spectral_modularity_bipartition",
    "greedy_node_swap_bipartition",
]


def _kernighan_lin_sweep(edge_info, side):
    """
    This is a modified form of Kernighan-Lin, which moves single nodes at a
    time, alternating between sides to keep the bisection balanced.  We keep
    two min-heaps of swap costs to make optimal-next-move selection fast.
    """
    heap0, heap1 = cost_heaps = nx.utils.BinaryHeap(), nx.utils.BinaryHeap()
    # we use heap methods insert, pop, and get
    for u, nbrs in edge_info.items():
        cost_u = sum(wt if side[v] else -wt for v, wt in nbrs.items())
        if side[u]:
            heap1.insert(u, cost_u)
        else:
            heap0.insert(u, -cost_u)

    def _update_heap_values(node):
        side_node = side[node]
        for nbr, wt in edge_info[node].items():
            side_nbr = side[nbr]
            if side_nbr == side_node:
                wt = -wt
            heap_nbr = cost_heaps[side_nbr]
            if nbr in heap_nbr:
                cost_nbr = heap_nbr.get(nbr) + 2 * wt
                # allow_increase lets us update a value already on the heap
                heap_nbr.insert(nbr, cost_nbr, allow_increase=True)

    i = 0
    totcost = 0
    while heap0 and heap1:
        u, cost_u = heap0.pop()
        _update_heap_values(u)
        v, cost_v = heap1.pop()
        _update_heap_values(v)
        totcost += cost_u + cost_v
        i += 1
        yield totcost, i, (u, v)


@nx.utils.not_implemented_for("directed")
@nx.utils.py_random_state(4)
@nx._dispatchable(edge_attrs="weight")
def kernighan_lin_bisection(G, partition=None, max_iter=10, weight="weight", seed=None):
    """Partition a graph into two blocks using the Kernighan–Lin algorithm.

    This algorithm partitions a network into two sets by iteratively
    swapping pairs of nodes to reduce the edge cut between the two sets.  The
    pairs are chosen according to a modified form of Kernighan-Lin [1]_, which
    moves nodes individually, alternating between sides to keep the bisection
    balanced.

    Kernighan-Lin is an approximate algorithm for maximal modularity bisection.
    In [2]_ they suggest that fine-tuned improvements can be made using
    greedy node swapping, (see `greedy_node_swap_bipartition`).
    The improvements are typically only a few percent of the modularity value.
    But they claim that can make a difference between a good and excellent method.
    This function does not perform any improvements. But you can do that yourself.

    Parameters
    ----------
    G : NetworkX graph
        Graph must be undirected.

    partition : tuple
        Pair of iterables containing an initial partition. If not
        specified, a random balanced partition is used.

    max_iter : int
        Maximum number of times to attempt swaps to find an
        improvement before giving up.

    weight : string or function (default: "weight")
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number or None to indicate a hidden edge.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
        Only used if partition is None

    Returns
    -------
    partition : tuple
        A pair of sets of nodes representing the bipartition.

    Raises
    ------
    NetworkXError
        If `partition` is not a valid partition of the nodes of the graph.

    References
    ----------
    .. [1] Kernighan, B. W.; Lin, Shen (1970).
       "An efficient heuristic procedure for partitioning graphs."
       *Bell Systems Technical Journal* 49: 291--307.
       Oxford University Press 2011.
    .. [2] M. E. J. Newman,
       "Modularity and community structure in networks",
       PNAS, 103 (23), p. 8577-8582,
       https://doi.org/10.1073/pnas.0601602103

    """
    nodes = list(G)

    if partition is None:
        seed.shuffle(nodes)
        mid = len(nodes) // 2
        A, B = nodes[:mid], nodes[mid:]
    else:
        try:
            A, B = partition
        except (TypeError, ValueError) as err:
            raise nx.NetworkXError("partition must be two sets") from err
        if not nx.community.is_partition(G, [A, B]):
            raise nx.NetworkXError("partition invalid")

    side = {node: (node in A) for node in nodes}

    # ruff: noqa: E731   skips check for no lambda functions
    # Using shortest_paths _weight_function with sum instead of min on multiedges
    if callable(weight):
        sum_weight = weight
    elif G.is_multigraph():
        sum_weight = lambda u, v, d: sum(dd.get(weight, 1) for dd in d.values())
    else:
        sum_weight = lambda u, v, d: d.get(weight, 1)

    edge_info = {
        u: {v: wt for v, d in nbrs.items() if (wt := sum_weight(u, v, d)) is not None}
        for u, nbrs in G._adj.items()
    }

    for i in range(max_iter):
        costs = list(_kernighan_lin_sweep(edge_info, side))
        # find out how many edges to update: min_i
        min_cost, min_i, _ = min(costs)
        if min_cost >= 0:
            break

        for _, _, (u, v) in costs[:min_i]:
            side[u] = 1
            side[v] = 0

    part1 = {u for u, s in side.items() if s == 0}
    part2 = {u for u, s in side.items() if s == 1}
    return part1, part2


@nx.utils.not_implemented_for("directed")
@nx.utils.not_implemented_for("multigraph")
def spectral_modularity_bipartition(G):
    r"""Return a bipartition of the nodes based on the spectrum of the
    modularity matrix of the graph.

    This method calculates the eigenvector associated with the second
    largest eigenvalue of the modularity matrix, where the modularity
    matrix *B* is defined by

    ..math::

        B_{i j} = A_{i j} - \frac{k_i k_j}{2 m},

    where *A* is the adjacency matrix, `k_i` is the degree of node *i*,
    and *m* is the number of edges in the graph. Nodes whose
    corresponding values in the eigenvector are negative are placed in
    one block, nodes whose values are nonnegative are placed in another
    block.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    C : tuple
        Pair of communities as two sets of nodes of ``G``, partitioned
        according to second largest eigenvalue of the modularity matrix.

    Examples
    --------
    >>> G = nx.karate_club_graph()
    >>> MrHi, Officer = nx.community.spectral_modularity_bipartition(G)
    >>> MrHi, Officer = sorted([sorted(MrHi), sorted(Officer)])
    >>> MrHi
    [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 17, 19, 21]
    >>> Officer
    [8, 9, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]

    References
    ----------
    .. [1] M. E. J. Newman *Networks: An Introduction*, pages 373--378
       Oxford University Press 2011.
    .. [2] M. E. J. Newman,
       "Modularity and community structure in networks",
       PNAS, 103 (23), p. 8577-8582,
       https://doi.org/10.1073/pnas.0601602103

    """
    import numpy as np

    B = nx.linalg.modularity_matrix(G)
    eigenvalues, eigenvectors = np.linalg.eig(B)
    index = np.argsort(eigenvalues)[-1]
    v2 = zip(np.real(eigenvectors[:, index]), G)
    left, right = set(), set()
    for u, n in v2:
        if u < 0:
            left.add(n)
        else:
            right.add(n)
    return left, right


@nx.utils.not_implemented_for("multigraph")
def greedy_node_swap_bipartition(G, *, init_split=None, max_iter=10):
    """Split the nodes into two communities based on greedy
    modularity maximization.

    The algorithm works by selecting a node to change communities which
    will maximize the modularity. The swap is made and the community
    structure with the highest modularity is kept.

    Parameters
    ----------
    G : NetworkX graph

    init_split : 2-tuple of sets of nodes
        Pair of sets of nodes in ``G`` providing an initial bipartition
        for the algorithm. If not specified, a random balanced partition
        is used. If this pair of sets is not a partition of the nodes of `G`,
        :exc:`NetworkXException` is raised.

    max_iter : int
      Maximum number of iterations of attempting swaps to find an improvement.

    Returns
    -------
    max_split : 2-tuple of sets of nodes
        Pair of sets of nodes of ``G``, partitioned according to a
        node swap greedy modularity maximization algorithm.

    Raises
    ------
    NetworkXError
      if init_split is not a valid partition of the
      graph into two communities or if G is a MultiGraph

    Examples
    --------
    >>> G = nx.barbell_graph(3, 0)
    >>> left, right = nx.community.greedy_node_swap_bipartition(G)
    >>> # Sort the communities so the nodes appear in increasing order.
    >>> left, right = sorted([sorted(left), sorted(right)])
    >>> sorted(left)
    [0, 1, 2]
    >>> sorted(right)
    [3, 4, 5]

    Notes
    -----
    This function is not implemented for multigraphs.

    References
    ----------
    .. [1] M. E. J. Newman "Networks: An Introduction", pages 373--375.
       Oxford University Press 2011.

    """
    if init_split is None:
        m1 = len(G) // 2
        m2 = len(G) - m1
        some_nodes = set(random.sample(list(G), m1))
        other_nodes = {n for n in G if n not in some_nodes}
        best_split_so_far = (some_nodes, other_nodes)
    else:
        if not nx.community.is_partition(G, init_split):
            raise nx.NetworkXError("init_split is not a partition of G")
        if not len(init_split) == 2:
            raise nx.NetworkXError("init_split must be a bipartition")
        best_split_so_far = deepcopy(init_split)

    best_mod = nx.community.modularity(G, best_split_so_far)

    max_split, max_mod = best_split_so_far, best_mod
    its = 0
    m = G.number_of_edges()
    G_degree = dict(G.degree)

    while max_mod >= best_mod and its < max_iter:
        best_split_so_far = max_split
        best_mod = max_mod
        next_split = deepcopy(max_split)
        next_mod = max_mod
        nodes = set(G)
        while nodes:
            max_swap = -1
            max_node = None
            max_node_comm = None
            left, right = next_split
            leftd = sum(G_degree[n] for n in left)
            rightd = sum(G_degree[n] for n in right)
            for n in nodes:
                if n in left:
                    in_comm, out_comm = left, right
                    in_deg, out_deg = leftd, rightd
                else:
                    in_comm, out_comm = right, left
                    in_deg, out_deg = rightd, leftd

                d_eii = -len(G[n].keys() & in_comm) / m
                d_ejj = len(G[n].keys() & out_comm) / m
                deg = G_degree[n]
                d_sum_ai = (deg / (2 * m**2)) * (in_deg - out_deg - deg)
                swap_change = d_eii + d_ejj + d_sum_ai

                if swap_change > max_swap:
                    max_swap = swap_change
                    max_node = n
                    max_node_comm = in_comm
                    non_max_node_comm = out_comm
            # swap the node from one comm to the other
            max_node_comm.remove(max_node)
            non_max_node_comm.add(max_node)
            next_mod += max_swap
            # deepcopy next_split each time it reaches a high (might go lower later)
            if next_mod > max_mod:
                max_split, max_mod = deepcopy(next_split), next_mod
            nodes.remove(max_node)
        its += 1
    return best_split_so_far
