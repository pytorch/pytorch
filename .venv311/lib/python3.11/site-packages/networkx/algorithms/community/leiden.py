"""Functions for detecting communities based on Leiden Community Detection
algorithm.

These functions do not have NetworkX implementations.
They may only be run with an installable :doc:`backend </backends>`
that supports them.
"""

import itertools
from collections import deque

import networkx as nx
from networkx.utils import not_implemented_for, py_random_state

__all__ = ["leiden_communities", "leiden_partitions"]


@not_implemented_for("directed")
@py_random_state("seed")
@nx._dispatchable(edge_attrs="weight", implemented_by_nx=False)
def leiden_communities(G, weight="weight", resolution=1, max_level=None, seed=None):
    r"""Find a best partition of `G` using Leiden Community Detection (backend required)

    Leiden Community Detection is an algorithm to extract the community structure
    of a network based on modularity optimization. It is an improvement upon the
    Louvain Community Detection algorithm. See :any:`louvain_communities`.

    Unlike the Louvain algorithm, it guarantees that communities are well connected in addition
    to being faster and uncovering better partitions. [1]_

    The algorithm works in 3 phases. On the first phase, it adds the nodes to a queue randomly
    and assigns every node to be in its own community. For each node it tries to find the
    maximum positive modularity gain by moving each node to all of its neighbor communities.
    If a node is moved from its community, it adds to the rear of the queue all neighbors of
    the node that do not belong to the node’s new community and that are not in the queue.

    The first phase continues until the queue is empty.

    The second phase consists in refining the partition $P$ obtained from the first phase. It starts
    with a singleton partition $P_{refined}$. Then it merges nodes locally in $P_{refined}$ within
    each community of the partition $P$. Nodes are merged with a community in $P_{refined}$ only if
    both are sufficiently well connected to their community in $P$. This means that after the
    refinement phase is concluded, communities in $P$ sometimes will have been split into multiple
    communities.

    The third phase consists of aggregating the network by building a new network whose nodes are
    now the communities found in the second phase. However, the non-refined partition is used to create
    an initial partition for the aggregate network.

    Once this phase is complete it is possible to reapply the first and second phases creating bigger
    communities with increased modularity.

    The above three phases are executed until no modularity gain is achieved or `max_level` number
    of iterations have been performed.

    Parameters
    ----------
    G : NetworkX graph
    weight : string or None, optional (default="weight")
        The name of an edge attribute that holds the numerical value
        used as a weight. If None then each edge has weight 1.
    resolution : float, optional (default=1)
        If resolution is less than 1, the algorithm favors larger communities.
        Greater than 1 favors smaller communities.
    max_level : int or None, optional (default=None)
        The maximum number of levels (steps of the algorithm) to compute.
        Must be a positive integer or None. If None, then there is no max
        level and the algorithm will run until converged.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    list
        A list of disjoint sets (partition of `G`). Each set represents one community.
        All communities together contain all the nodes in `G`.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.petersen_graph()
    >>> nx.community.leiden_communities(G, backend="example_backend")  # doctest: +SKIP
    [{2, 3, 5, 7, 8}, {0, 1, 4, 6, 9}]

    Notes
    -----
    The order in which the nodes are considered can affect the final output. In the algorithm
    the ordering happens using a random shuffle.

    References
    ----------
    .. [1] Traag, V.A., Waltman, L. & van Eck, N.J. From Louvain to Leiden: guaranteeing
       well-connected communities. Sci Rep 9, 5233 (2019). https://doi.org/10.1038/s41598-019-41695-z

    See Also
    --------
    leiden_partitions
    :any:`louvain_communities`
    """
    partitions = leiden_partitions(G, weight, resolution, seed)
    if max_level is not None:
        if max_level <= 0:
            raise ValueError("max_level argument must be a positive integer or None")
        partitions = itertools.islice(partitions, max_level)
    final_partition = deque(partitions, maxlen=1)
    return final_partition.pop()


@not_implemented_for("directed")
@py_random_state("seed")
@nx._dispatchable(edge_attrs="weight", implemented_by_nx=False)
def leiden_partitions(G, weight="weight", resolution=1, seed=None):
    """Yield partitions for each level of Leiden Community Detection (backend required)

    Leiden Community Detection is an algorithm to extract the community
    structure of a network based on modularity optimization.

    The partitions across levels (steps of the algorithm) form a dendrogram
    of communities. A dendrogram is a diagram representing a tree and each
    level represents a partition of the G graph. The top level contains the
    smallest communities and as you traverse to the bottom of the tree the
    communities get bigger and the overall modularity increases making
    the partition better.

    Each level is generated by executing the three phases of the Leiden Community
    Detection algorithm. See :any:`leiden_communities`.

    Parameters
    ----------
    G : NetworkX graph
    weight : string or None, optional (default="weight")
        The name of an edge attribute that holds the numerical value
        used as a weight. If None then each edge has weight 1.
    resolution : float, optional (default=1)
        If resolution is less than 1, the algorithm favors larger communities.
        Greater than 1 favors smaller communities.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Yields
    ------
    list
        A list of disjoint sets (partition of `G`). Each set represents one community.
        All communities together contain all the nodes in `G`. The yielded partitions
        increase modularity with each iteration.

    References
    ----------
    .. [1] Traag, V.A., Waltman, L. & van Eck, N.J. From Louvain to Leiden: guaranteeing
       well-connected communities. Sci Rep 9, 5233 (2019). https://doi.org/10.1038/s41598-019-41695-z

    See Also
    --------
    leiden_communities
    :any:`louvain_partitions`
    """
    raise NotImplementedError(
        "'leiden_partitions' is not implemented by networkx. "
        "Please try a different backend."
    )
