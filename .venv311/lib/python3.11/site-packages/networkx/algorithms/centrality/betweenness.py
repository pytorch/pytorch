"""Betweenness centrality measures."""

import math
from collections import deque
from heapq import heappop, heappush
from itertools import count

import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _weight_function
from networkx.utils import py_random_state
from networkx.utils.decorators import not_implemented_for

__all__ = ["betweenness_centrality", "edge_betweenness_centrality"]


@py_random_state("seed")
@nx._dispatchable(edge_attrs="weight")
def betweenness_centrality(
    G, k=None, normalized=True, weight=None, endpoints=False, seed=None
):
    r"""Compute the shortest-path betweenness centrality for nodes.

    Betweenness centrality of a node $v$ is the sum of the
    fraction of all-pairs shortest paths that pass through $v$.

    .. math::

       c_B(v) = \sum_{s, t \in V} \frac{\sigma(s, t | v)}{\sigma(s, t)}

    where $V$ is the set of nodes, $\sigma(s, t)$ is the number of
    shortest $(s, t)$-paths, and $\sigma(s, t | v)$ is the number of
    those paths passing through some node $v$ other than $s$ and $t$.
    If $s = t$, $\sigma(s, t) = 1$, and if $v \in \{s, t\}$,
    $\sigma(s, t | v) = 0$ [2]_.
    The denominator $\sigma(s, t)$ is a normalization factor that can be
    turned off to get the raw path counts.

    Parameters
    ----------
    G : graph
        A NetworkX graph.

    k : int, optional (default=None)
        If `k` is not `None`, use `k` sampled nodes as sources for the considered paths.
        The resulting sampled counts are then inflated to approximate betweenness.
        Higher values of `k` give better approximation. Must have ``k <= len(G)``.

    normalized : bool, optional (default=True)
        If `True`, the betweenness values are rescaled by dividing by the number of
        possible $(s, t)$-pairs in the graph.

    weight : None or string, optional (default=None)
        If `None`, all edge weights are 1.
        Otherwise holds the name of the edge attribute used as weight.
        Weights are used to calculate weighted shortest paths, so they are
        interpreted as distances.

    endpoints : bool, optional (default=False)
        If `True`, include the endpoints $s$ and $t$ in the shortest path counts.
        This is taken into account when rescaling the values.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
        Note that this is only used if ``k is not None``.

    Returns
    -------
    nodes : dict
        Dictionary of nodes with betweenness centrality as the value.

    See Also
    --------
    betweenness_centrality_subset
    edge_betweenness_centrality
    load_centrality

    Notes
    -----
    The algorithm is from Ulrik Brandes [1]_.
    See [4]_ for the original first published version and [2]_ for details on
    algorithms for variations and related metrics.

    For approximate betweenness calculations, set `k` to the number of sampled
    nodes ("pivots") used as sources to estimate the betweenness values.
    The formula then sums over $s$ is in these pivots, instead of over all nodes.
    The resulting sum is then inflated to approximate the full sum.
    For a discussion of how to choose `k` for efficiency, see [3]_.

    For weighted graphs the edge weights must be greater than zero.
    Zero edge weights can produce an infinite number of equal length
    paths between pairs of nodes.

    Directed graphs and undirected graphs count paths differently.
    In directed graphs, each pair of source-target nodes is considered separately
    in each direction, as the shortest paths can differ by direction.
    However, in undirected graphs, each pair of nodes is considered only once,
    as the shortest paths are symmetric.
    This means the normalization factor to divide by is $N(N-1)$ for directed graphs
    and $N(N-1)/2$ for undirected graphs, where $N = n$ (the number of nodes)
    if endpoints are included and $N = n-1$ otherwise.

    This algorithm is not guaranteed to be correct if edge weights
    are floating point numbers. As a workaround you can use integer
    numbers by multiplying the relevant edge attributes by a convenient
    constant factor (e.g. 100) and converting to integers.

    References
    ----------
    .. [1] Ulrik Brandes:
       A Faster Algorithm for Betweenness Centrality.
       Journal of Mathematical Sociology 25(2):163--177, 2001.
       https://doi.org/10.1080/0022250X.2001.9990249
    .. [2] Ulrik Brandes:
       On Variants of Shortest-Path Betweenness
       Centrality and their Generic Computation.
       Social Networks 30(2):136--145, 2008.
       https://doi.org/10.1016/j.socnet.2007.11.001
    .. [3] Ulrik Brandes and Christian Pich:
       Centrality Estimation in Large Networks.
       International Journal of Bifurcation and Chaos 17(7):2303--2318, 2007.
       https://dx.doi.org/10.1142/S0218127407018403
    .. [4] Linton C. Freeman:
       A set of measures of centrality based on betweenness.
       Sociometry 40: 35--41, 1977
       https://doi.org/10.2307/3033543

    Examples
    --------
    Consider an undirected 3-path. Each pair of nodes has exactly one shortest
    path between them. Since the graph is undirected, only ordered pairs are counted.
    Of these (and when `endpoints` is `False`), none of the shortest paths pass
    through 0 and 2, and only the shortest path between 0 and 2 passes through 1.
    As such, the counts should be ``{0: 0, 1: 1, 2: 0}``.

    >>> G = nx.path_graph(3)
    >>> nx.betweenness_centrality(G, normalized=False, endpoints=False)
    {0: 0.0, 1: 1.0, 2: 0.0}

    If `endpoints` is `True`, we also need to count endpoints as being on the path:
    $\sigma(s, t | s) = \sigma(s, t | t) = \sigma(s, t)$.
    In our example, 0 is then part of two shortest paths (0 to 1 and 0 to 2);
    similarly, 2 is part of two shortest paths (0 to 2 and 1 to 2).
    1 is part of all three shortest paths. This makes the new raw
    counts ``{0: 2, 1: 3, 2: 2}``.

    >>> nx.betweenness_centrality(G, normalized=False, endpoints=True)
    {0: 2.0, 1: 3.0, 2: 2.0}

    With normalization, the values are divided by the number of ordered $(s, t)$-pairs.
    If we are not counting endpoints, there are $n - 1$ possible choices for $s$
    (all except the node we are computing betweenness centrality for), which in turn
    leaves $n - 2$ possible choices for $t$ as $s \ne t$.
    The total number of ordered pairs when `endpoints` is `False` is $(n - 1)(n - 2)/2 = 1$.
    If `endpoints` is `True`, there are $n(n - 1)/2 = 3$ ordered $(s, t)$-pairs to divide by.

    >>> nx.betweenness_centrality(G, normalized=True, endpoints=False)
    {0: 0.0, 1: 1.0, 2: 0.0}
    >>> nx.betweenness_centrality(G, normalized=True, endpoints=True)
    {0: 0.6666666666666666, 1: 1.0, 2: 0.6666666666666666}

    If the graph is directed instead, we now need to consider $(s, t)$-pairs
    in both directions. Our example becomes a directed 3-path.
    Without counting endpoints, we only have one path through 1 (0 to 2).
    This means the raw counts are ``{0: 0, 1: 1, 2: 0}``.

    >>> DG = nx.path_graph(3, create_using=nx.DiGraph)
    >>> nx.betweenness_centrality(DG, normalized=False, endpoints=False)
    {0: 0.0, 1: 1.0, 2: 0.0}

    If we do include endpoints, the raw counts are ``{0: 2, 1: 3, 2: 2}``.

    >>> nx.betweenness_centrality(DG, normalized=False, endpoints=True)
    {0: 2.0, 1: 3.0, 2: 2.0}

    If we want to normalize directed betweenness centrality, the raw counts
    are normalized by the number of $(s, t)$-pairs. There are $n(n - 1)$
    possible paths with endpoints and $(n - 1)(n - 2)$ without endpoints.
    In our example, that's 6 with endpoints and 2 without endpoints.

    >>> nx.betweenness_centrality(DG, normalized=True, endpoints=True)
    {0: 0.3333333333333333, 1: 0.5, 2: 0.3333333333333333}
    >>> nx.betweenness_centrality(DG, normalized=True, endpoints=False)
    {0: 0.0, 1: 0.5, 2: 0.0}

    Computing the full betweenness centrality can be costly.
    This function can also be used to compute approximate betweenness centrality
    by setting `k`. This only determines the number of source nodes to sample;
    all nodes are targets.

    For simplicity, we only consider the case where endpoints are included in the counts.
    Since the partial sums only include `k` terms, instead of ``n``,
    we multiply them by ``n / k``, to approximate the full sum.
    As the sets of sources and targets are not the same anymore,
    paths have to be counted in a directed way. We thus count each as half a path.
    This ensures that the results approximate the standard betweenness for ``k == n``.

    For instance, in the undirected 3-path graph case, setting ``k = 2`` (with ``seed=42``)
    selects nodes 0 and 2 as sources.
    This means only shortest paths starting at these nodes are considered.
    The raw counts with endpoints are ``{0: 3, 1: 4, 2: 3}``. Accounting for the partial sum
    and applying the undirectedness half-path correction, we get

    >>> nx.betweenness_centrality(G, k=2, normalized=False, endpoints=True, seed=42)
    {0: 2.25, 1: 3.0, 2: 2.25}

    When normalizing, we instead want to divide by the total number of $(s, t)$-pairs.
    This is $k(n - 1)$ with endpoints.

    >>> nx.betweenness_centrality(G, k=2, normalized=True, endpoints=True, seed=42)
    {0: 0.75, 1: 1.0, 2: 0.75}
    """
    betweenness = dict.fromkeys(G, 0.0)  # b[v]=0 for v in G
    if k == len(G):
        # This is done for performance; the result is the same regardless.
        k = None
    if k is None:
        nodes = G
    else:
        nodes = seed.sample(list(G.nodes()), k)
    for s in nodes:
        # single source shortest paths
        if weight is None:  # use BFS
            S, P, sigma, _ = _single_source_shortest_path_basic(G, s)
        else:  # use Dijkstra's algorithm
            S, P, sigma, _ = _single_source_dijkstra_path_basic(G, s, weight)
        # accumulation
        if endpoints:
            betweenness, _ = _accumulate_endpoints(betweenness, S, P, sigma, s)
        else:
            betweenness, _ = _accumulate_basic(betweenness, S, P, sigma, s)
    # rescaling
    betweenness = _rescale(
        betweenness,
        len(G),
        normalized=normalized,
        directed=G.is_directed(),
        endpoints=endpoints,
        sampled_nodes=None if k is None else nodes,
    )
    return betweenness


@py_random_state("seed")
@nx._dispatchable(edge_attrs="weight")
def edge_betweenness_centrality(G, k=None, normalized=True, weight=None, seed=None):
    r"""Compute betweenness centrality for edges.

    Betweenness centrality of an edge $e$ is the sum of the
    fraction of all-pairs shortest paths that pass through $e$.

    .. math::

       c_B(e) = \sum_{s, t \in V} \frac{\sigma(s, t | e)}{\sigma(s, t)}

    where $V$ is the set of nodes, $\sigma(s, t)$ is the number of
    shortest $(s, t)$-paths, and $\sigma(s, t | e)$ is the number of
    those paths passing through edge $e$ [1]_.
    The denominator $\sigma(s, t)$ is a normalization factor that can be
    turned off to get the raw path counts.

    Parameters
    ----------
    G : graph
        A NetworkX graph.

    k : int, optional (default=None)
        If `k` is not `None`, use `k` sampled nodes as sources for the considered paths.
        The resulting sampled counts are then inflated to approximate betweenness.
        Higher values of `k` give better approximation. Must have ``k <= len(G)``.

    normalized : bool, optional (default=True)
        If `True`, the betweenness values are rescaled by dividing by the number of
        possible $(s, t)$-pairs in the graph.

    weight : None or string, optional (default=None)
        If `None`, all edge weights are 1.
        Otherwise holds the name of the edge attribute used as weight.
        Weights are used to calculate weighted shortest paths, so they are
        interpreted as distances.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
        Note that this is only used if ``k is not None``.

    Returns
    -------
    edges : dict
        Dictionary of edges with betweenness centrality as the value.

    See Also
    --------
    betweenness_centrality
    edge_betweenness_centrality_subset
    edge_load

    Notes
    -----
    The algorithm is from Ulrik Brandes [1]_.

    For weighted graphs the edge weights must be greater than zero.
    Zero edge weights can produce an infinite number of equal length
    paths between pairs of nodes.

    References
    ----------
    .. [1] Ulrik Brandes: On Variants of Shortest-Path Betweenness
       Centrality and their Generic Computation.
       Social Networks 30(2):136--145, 2008.
       https://doi.org/10.1016/j.socnet.2007.11.001

    Examples
    --------
    Consider an undirected 3-path. Each pair of nodes has exactly one shortest
    path between them. Since the graph is undirected, only ordered pairs are counted.
    Each edge has two shortest paths passing through it.
    As such, the raw counts should be ``{(0, 1): 2, (1, 2): 2}``.

    >>> G = nx.path_graph(3)
    >>> nx.edge_betweenness_centrality(G, normalized=False)
    {(0, 1): 2.0, (1, 2): 2.0}

    With normalization, the values are divided by the number of ordered $(s, t)$-pairs,
    which is $n(n-1)/2$. For the 3-path, this is $3(3-1)/2 = 3$.

    >>> nx.edge_betweenness_centrality(G, normalized=True)
    {(0, 1): 0.6666666666666666, (1, 2): 0.6666666666666666}

    For a directed graph, all $(s, t)$-pairs are considered. The normalization factor
    is $n(n-1)$ to reflect this.

    >>> DG = nx.path_graph(3, create_using=nx.DiGraph)
    >>> nx.edge_betweenness_centrality(DG, normalized=False)
    {(0, 1): 2.0, (1, 2): 2.0}
    >>> nx.edge_betweenness_centrality(DG, normalized=True)
    {(0, 1): 0.3333333333333333, (1, 2): 0.3333333333333333}

    Computing the full edge betweenness centrality can be costly.
    This function can also be used to compute approximate edge betweenness centrality
    by setting `k`. This determines the number of source nodes to sample.

    Since the partial sums only include `k` terms, instead of ``n``,
    we multiply them by ``n / k``, to approximate the full sum.
    As the sets of sources and targets are not the same anymore,
    paths have to be counted in a directed way. We thus count each as half a path.
    This ensures that the results approximate the standard betweenness for ``k == n``.

    For instance, in the undirected 3-path graph case, setting ``k = 2`` (with ``seed=42``)
    selects nodes 0 and 2 as sources.
    This means only shortest paths starting at these nodes are considered.
    The raw counts are ``{(0, 1): 3, (1, 2): 3}``. Accounting for the partial sum
    and applying the undirectedness half-path correction, we get

    >>> nx.edge_betweenness_centrality(G, k=2, normalized=False, seed=42)
    {(0, 1): 2.25, (1, 2): 2.25}

    When normalizing, we instead want to divide by the total number of $(s, t)$-pairs.
    This is $k(n-1)$, which is $4$ in our case.

    >>> nx.edge_betweenness_centrality(G, k=2, normalized=True, seed=42)
    {(0, 1): 0.75, (1, 2): 0.75}
    """
    betweenness = dict.fromkeys(G, 0.0)  # b[v]=0 for v in G
    # b[e]=0 for e in G.edges()
    betweenness.update(dict.fromkeys(G.edges(), 0.0))
    if k is None:
        nodes = G
    else:
        nodes = seed.sample(list(G.nodes()), k)
    for s in nodes:
        # single source shortest paths
        if weight is None:  # use BFS
            S, P, sigma, _ = _single_source_shortest_path_basic(G, s)
        else:  # use Dijkstra's algorithm
            S, P, sigma, _ = _single_source_dijkstra_path_basic(G, s, weight)
        # accumulation
        betweenness = _accumulate_edges(betweenness, S, P, sigma, s)
    # rescaling
    for n in G:  # remove nodes to only return edges
        del betweenness[n]
    betweenness = _rescale(
        betweenness,
        len(G),
        normalized=normalized,
        directed=G.is_directed(),
        sampled_nodes=None if k is None else nodes,
    )
    if G.is_multigraph():
        betweenness = _add_edge_keys(G, betweenness, weight=weight)
    return betweenness


# helpers for betweenness centrality


def _single_source_shortest_path_basic(G, s):
    S = []
    P = {}
    for v in G:
        P[v] = []
    sigma = dict.fromkeys(G, 0.0)  # sigma[v]=0 for v in G
    D = {}
    sigma[s] = 1.0
    D[s] = 0
    Q = deque([s])
    while Q:  # use BFS to find shortest paths
        v = Q.popleft()
        S.append(v)
        Dv = D[v]
        sigmav = sigma[v]
        for w in G[v]:
            if w not in D:
                Q.append(w)
                D[w] = Dv + 1
            if D[w] == Dv + 1:  # this is a shortest path, count paths
                sigma[w] += sigmav
                P[w].append(v)  # predecessors
    return S, P, sigma, D


def _single_source_dijkstra_path_basic(G, s, weight):
    weight = _weight_function(G, weight)
    # modified from Eppstein
    S = []
    P = {}
    for v in G:
        P[v] = []
    sigma = dict.fromkeys(G, 0.0)  # sigma[v]=0 for v in G
    D = {}
    sigma[s] = 1.0
    seen = {s: 0}
    c = count()
    Q = []  # use Q as heap with (distance,node id) tuples
    heappush(Q, (0, next(c), s, s))
    while Q:
        (dist, _, pred, v) = heappop(Q)
        if v in D:
            continue  # already searched this node.
        sigma[v] += sigma[pred]  # count paths
        S.append(v)
        D[v] = dist
        for w, edgedata in G[v].items():
            vw_dist = dist + weight(v, w, edgedata)
            if w not in D and (w not in seen or vw_dist < seen[w]):
                seen[w] = vw_dist
                heappush(Q, (vw_dist, next(c), v, w))
                sigma[w] = 0.0
                P[w] = [v]
            elif vw_dist == seen[w]:  # handle equal paths
                sigma[w] += sigma[v]
                P[w].append(v)
    return S, P, sigma, D


def _accumulate_basic(betweenness, S, P, sigma, s):
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1 + delta[w]) / sigma[w]
        for v in P[w]:
            delta[v] += sigma[v] * coeff
        if w != s:
            betweenness[w] += delta[w]
    return betweenness, delta


def _accumulate_endpoints(betweenness, S, P, sigma, s):
    betweenness[s] += len(S) - 1
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1 + delta[w]) / sigma[w]
        for v in P[w]:
            delta[v] += sigma[v] * coeff
        if w != s:
            betweenness[w] += delta[w] + 1
    return betweenness, delta


def _accumulate_edges(betweenness, S, P, sigma, s):
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1 + delta[w]) / sigma[w]
        for v in P[w]:
            c = sigma[v] * coeff
            if (v, w) not in betweenness:
                betweenness[(w, v)] += c
            else:
                betweenness[(v, w)] += c
            delta[v] += c
        if w != s:
            betweenness[w] += delta[w]
    return betweenness


def _rescale(
    betweenness, n, *, normalized, directed, endpoints=True, sampled_nodes=None
):
    # For edge betweenness, `endpoints` is always `True`.

    k = None if sampled_nodes is None else len(sampled_nodes)
    # N is used to count the number of valid (s, t) pairs where s != t that
    # could have a path pass through v. If endpoints is False, then v must
    # not be the target t, hence why we subtract by 1.
    N = n if endpoints else n - 1
    if N < 2:
        # No rescaling necessary: b=0 for all nodes
        return betweenness

    K_source = N if k is None else k

    if k is None or endpoints:
        # No sampling adjustment needed
        if normalized:
            # Divide by the number of valid (s, t) node pairs that could have
            # a path through v where s != t.
            scale = 1 / (K_source * (N - 1))
        else:
            # Scale to the full BC
            if not directed:
                # The non-normalized BC values are computed the same way for
                # directed and undirected graphs: shortest paths are computed and
                # counted for each *ordered* (s, t) pair. Undirected graphs should
                # only count valid *unordered* node pairs {s, t}; that is, (s, t)
                # and (t, s) should be counted only once. We correct for this here.
                correction = 2
            else:
                correction = 1
            scale = N / (K_source * correction)

        if scale != 1:
            for v in betweenness:
                betweenness[v] *= scale
        return betweenness

    # Sampling adjustment needed when excluding endpoints when using k. In this
    # case, we need to handle source nodes differently from non-source nodes,
    # because source nodes can't include themselves since endpoints are excluded.
    # Without this, k == n would be a special case that would violate the
    # assumption that node `v` is not one of the (s, t) node pairs.
    if normalized:
        # NaN for undefined 0/0; there is no data for source node when k=1
        scale_source = 1 / ((K_source - 1) * (N - 1)) if K_source > 1 else math.nan
        scale_nonsource = 1 / (K_source * (N - 1))
    else:
        correction = 1 if directed else 2
        scale_source = N / ((K_source - 1) * correction) if K_source > 1 else math.nan
        scale_nonsource = N / (K_source * correction)

    sampled_nodes = set(sampled_nodes)
    for v in betweenness:
        betweenness[v] *= scale_source if v in sampled_nodes else scale_nonsource
    return betweenness


@not_implemented_for("graph")
def _add_edge_keys(G, betweenness, weight=None):
    r"""Adds the corrected betweenness centrality (BC) values for multigraphs.

    Parameters
    ----------
    G : NetworkX graph.

    betweenness : dictionary
        Dictionary mapping adjacent node tuples to betweenness centrality values.

    weight : string or function
        See `_weight_function` for details. Defaults to `None`.

    Returns
    -------
    edges : dictionary
        The parameter `betweenness` including edges with keys and their
        betweenness centrality values.

    The BC value is divided among edges of equal weight.
    """
    _weight = _weight_function(G, weight)

    edge_bc = dict.fromkeys(G.edges, 0.0)
    for u, v in betweenness:
        d = G[u][v]
        wt = _weight(u, v, d)
        keys = [k for k in d if _weight(u, v, {k: d[k]}) == wt]
        bc = betweenness[(u, v)] / len(keys)
        for k in keys:
            edge_bc[(u, v, k)] = bc

    return edge_bc
