import itertools

import networkx as nx

__all__ = ["birank"]


@nx._dispatchable(edge_attrs="weight")
def birank(
    G,
    nodes,
    *,
    alpha=None,
    beta=None,
    top_personalization=None,
    bottom_personalization=None,
    max_iter=100,
    tol=1.0e-6,
    weight="weight",
):
    r"""Compute the BiRank score for nodes in a bipartite network.

    Given the bipartite sets $U$ and $P$, the BiRank algorithm seeks to satisfy
    the following recursive relationships between the scores of nodes $j \in P$
    and $i \in U$:

    .. math::

        p_j = \alpha \sum_{i \in U} \frac{w_{ij}}{\sqrt{d_i}\sqrt{d_j}} u_i
        + (1 - \alpha) p_j^0

        u_i = \beta \sum_{j \in P} \frac{w_{ij}}{\sqrt{d_i}\sqrt{d_j}} p_j
        + (1 - \beta) u_i^0

    where

    * $p_j$ and $u_i$ are the BiRank scores of nodes $j \in P$ and $i \in U$.
    * $w_{ij}$ is the weight of the edge between nodes $i \in U$ and $j \in P$
      (With a value of 0 if no edge exists).
    * $d_i$ and $d_j$ are the weighted degrees of nodes $i \in U$ and $j \in P$,
      respectively.
    * $p_j^0$ and $u_i^0$ are personalization values that can encode a priori
      weights for the nodes $j \in P$ and $i \in U$, respectively. Akin to the
      personalization vector used by PageRank.
    * $\alpha$ and $\beta$ are damping hyperparameters applying to nodes in $P$
      and $U$ respectively. They can take values in the interval $[0, 1]$, and
      are analogous to those used by PageRank.

    Below are two use cases for this algorithm.

    1. Personalized Recommendation System
        Given a bipartite graph representing users and items, BiRank can be used
        as a collaborative filtering algorithm to recommend items to users.
        Previous ratings are encoded as edge weights, and the specific ratings
        of an individual user on a set of items is used as the personalization
        vector over items. See the example below for an implementation of this
        on a toy dataset provided in [1]_.

    2. Popularity Prediction
        Given a bipartite graph representing user interactions with items, e.g.
        commits to a GitHub repository, BiRank can be used to predict the
        popularity of a given item. Edge weights should encode the strength of
        the interaction signal. This could be a raw count, or weighted by a time
        decay function like that specified in Eq. (15) of [1]_. The
        personalization vectors can be used to encode existing popularity
        signals, for example, the monthly download count of a repository's
        package.

    Parameters
    ----------
    G : graph
        A bipartite network

    nodes : iterable of nodes
        Container with all nodes belonging to the first bipartite node set
        ('top'). The nodes in this set use the hyperparameter `alpha`, and the
        personalization dictionary `top_personalization`. The nodes in the second
        bipartite node set ('bottom') are automatically determined by taking the
        complement of 'top' with respect to the graph `G`.

    alpha : float, optional (default=0.80 if top_personalization not empty, else 1)
        Damping factor for the 'top' nodes. Must be in the interval $[0, 1]$.
        Larger alpha and beta generally reduce the effect of the personalizations
        and increase the number of iterations before convergence. Choice of value
        is largely dependent on use case, and experimentation is recommended.

    beta : float, optional (default=0.80 if bottom_personalization not empty, else 1)
        Damping factor for the 'bottom' nodes. Must be in the interval $[0, 1]$.
        Larger alpha and beta generally reduce the effect of the personalizations
        and increase the number of iterations before convergence. Choice of value
        is largely dependent on use case, and experimentation is recommended.

    top_personalization : dict, optional (default=None)
        Dictionary keyed by nodes in 'top' to that node's personalization value.
        Unspecified nodes in 'top' will be assigned a personalization value of 0.
        Personalization values are used to encode a priori weights for a given node,
        and should be non-negative.

    bottom_personalization : dict, optional (default=None)
        Dictionary keyed by nodes in 'bottom' to that node's personalization value.
        Unspecified nodes in 'bottom' will be assigned a personalization value of 0.
        Personalization values are used to encode a priori weights for a given node,
        and should be non-negative.

    max_iter : int, optional (default=100)
        Maximum number of iterations in power method eigenvalue solver.

    tol : float, optional (default=1.0e-6)
        Error tolerance used to check convergence in power method solver. The
        iteration will stop after a tolerance of both ``len(top) * tol`` and
        ``len(bottom) * tol`` is reached for nodes in 'top' and 'bottom'
        respectively.

    weight : string or None, optional (default='weight')
        Edge data key to use as weight.

    Returns
    -------
    birank : dictionary
        Dictionary keyed by node to that node's BiRank score.

    Raises
    ------
    NetworkXAlgorithmError
        If the parameters `alpha` or `beta` are not in the interval [0, 1],
        if either of the bipartite sets are empty, or if negative values are
        provided in the personalization dictionaries.

    PowerIterationFailedConvergence
        If the algorithm fails to converge to the specified tolerance
        within the specified number of iterations of the power iteration
        method.

    Examples
    --------
    Construct a bipartite graph with user-item ratings and use BiRank to
    recommend items to a user (user 1). The example below uses the `rating`
    edge attribute as the weight of the edges. The `top_personalization` vector
    is used to encode the user's previous ratings on items.

    Creation of graph, bipartite sets for the example.

    >>> elist = [
    ...     ("u1", "p1", 5),
    ...     ("u2", "p1", 5),
    ...     ("u2", "p2", 4),
    ...     ("u3", "p1", 3),
    ...     ("u3", "p3", 2),
    ... ]
    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from(elist, weight="rating")
    >>> product_nodes = ("p1", "p2", "p3")
    >>> user = "u1"

    First, we create a personalization vector for the user based on on their
    ratings of past items. In this case they have only rated one item (p1, with
    a rating of 5) in the past.

    >>> user_personalization = {
    ...     product: rating
    ...     for _, product, rating in G.edges(nbunch=user, data="rating")
    ... }
    >>> user_personalization
    {'p1': 5}

    Calculate the BiRank score of all nodes in the graph, filter for the items
    that the user has not rated yet, and sort the results by score.

    >>> user_birank_results = nx.bipartite.birank(
    ...     G, product_nodes, top_personalization=user_personalization, weight="rating"
    ... )
    >>> user_birank_results = filter(
    ...     lambda item: item[0][0] == "p" and user not in G.neighbors(item[0]),
    ...     user_birank_results.items(),
    ... )
    >>> user_birank_results = sorted(
    ...     user_birank_results, key=lambda item: item[1], reverse=True
    ... )
    >>> user_recommendations = {
    ...     product: round(score, 5) for product, score in user_birank_results
    ... }
    >>> user_recommendations
    {'p2': 1.44818, 'p3': 1.04811}

    We find that user 1 should be recommended item p2 over item p3. This is due
    to the fact that user 2 rated also rated p1 highly, while user 3 did not.
    Thus user 2's tastes are inferred to be similar to user 1's, and carry more
    weight in the recommendation.

    See Also
    --------
    :func:`~networkx.algorithms.link_analysis.pagerank_alg.pagerank`
    :func:`~networkx.algorithms.link_analysis.hits_alg.hits`
    :func:`~networkx.algorithms.bipartite.centrality.betweenness_centrality`
    :func:`~networkx.algorithms.bipartite.basic.sets`
    :func:`~networkx.algorithms.bipartite.basic.is_bipartite`

    Notes
    -----
    The `nodes` input parameter must contain all nodes in one bipartite
    node set, but the dictionary returned contains all nodes from both
    bipartite node sets. See :mod:`bipartite documentation
    <networkx.algorithms.bipartite>` for further details on how
    bipartite graphs are handled in NetworkX.

    In the case a personalization dictionary is not provided for top (bottom)
    `alpha` (`beta`) will default to 1. This is because a damping factor
    without a non-zero entry in the personalization vector will lead to the
    algorithm converging to the zero vector.

    References
    ----------
    .. [1] Xiangnan He, Ming Gao, Min-Yen Kan, and Dingxian Wang. 2017.
       BiRank: Towards Ranking on Bipartite Graphs. IEEE Trans. on Knowl.
       and Data Eng. 29, 1 (January 2017), 57–71.
       https://arxiv.org/pdf/1708.04396

    """
    import numpy as np
    import scipy as sp

    # Initialize the sets of top and bottom nodes
    top = set(nodes)
    bottom = set(G) - top
    top_count = len(top)
    bottom_count = len(bottom)

    if top_count == 0 or bottom_count == 0:
        raise nx.NetworkXAlgorithmError(
            "The BiRank algorithm requires a bipartite graph with at least one"
            "node in each set."
        )

    # Clean the personalization dictionaries
    top_personalization = _clean_personalization_dict(top_personalization)
    bottom_personalization = _clean_personalization_dict(bottom_personalization)

    # Set default values for alpha and beta if not provided
    if alpha is None:
        alpha = 0.8 if top_personalization else 1
    if beta is None:
        beta = 0.8 if bottom_personalization else 1

    if alpha < 0 or alpha > 1:
        raise nx.NetworkXAlgorithmError("alpha must be in the interval [0, 1]")
    if beta < 0 or beta > 1:
        raise nx.NetworkXAlgorithmError("beta must be in the interval [0, 1]")

    # Initialize query vectors
    p0 = np.array([top_personalization.get(n, 0) for n in top], dtype=float)
    u0 = np.array([bottom_personalization.get(n, 0) for n in bottom], dtype=float)

    # Construct degree normalized biadjacency matrix `S` and its transpose
    W = nx.bipartite.biadjacency_matrix(G, bottom, top, weight=weight, dtype=float)
    p_degrees = W.sum(axis=0, dtype=float)
    # Handle case where the node is disconnected - avoids warning
    p_degrees[p_degrees == 0] = 1.0
    D_p = sp.sparse.dia_array(
        ([1.0 / np.sqrt(p_degrees)], [0]),
        shape=(top_count, top_count),
        dtype=float,
    )
    u_degrees = W.sum(axis=1, dtype=float)
    u_degrees[u_degrees == 0] = 1.0
    D_u = sp.sparse.dia_array(
        ([1.0 / np.sqrt(u_degrees)], [0]),
        shape=(bottom_count, bottom_count),
        dtype=float,
    )
    S = D_u.tocsr() @ W @ D_p.tocsr()
    S_T = S.T

    # Initialize birank vectors for iteration
    p = np.ones(top_count, dtype=float) / top_count
    u = beta * (S @ p) + (1 - beta) * u0

    # Iterate until convergence
    for _ in range(max_iter):
        p_last = p
        u_last = u
        p = alpha * (S_T @ u) + (1 - alpha) * p0
        u = beta * (S @ p) + (1 - beta) * u0

        # Continue iterating if the error (absolute if less than 1, relative otherwise)
        # is above the tolerance threshold for either p or u
        err_u = np.absolute((u_last - u) / np.maximum(1.0, u_last)).sum()
        if err_u >= len(u) * tol:
            continue
        err_p = np.absolute((p_last - p) / np.maximum(1.0, p_last)).sum()
        if err_p >= len(p) * tol:
            continue

        # Handle edge case where if both alpha and beta are 1, scale is
        # indeterminate, so normalization is required to return consistent results
        if alpha == 1 and beta == 1:
            p = p / np.linalg.norm(p, 1)
            u = u / np.linalg.norm(u, 1)

        # If both error thresholds pass, return a single dictionary mapping
        # nodes to their scores
        return dict(
            zip(itertools.chain(top, bottom), map(float, itertools.chain(p, u)))
        )

    # If we reach this point, we have not converged
    raise nx.PowerIterationFailedConvergence(max_iter)


def _clean_personalization_dict(personalization):
    """Filter out zero values from the personalization dictionary,
    handle case where None is passed, ensure values are non-negative."""
    if personalization is None:
        return {}
    if any(value < 0 for value in personalization.values()):
        raise nx.NetworkXAlgorithmError("Personalization values must be non-negative.")
    return {node: value for node, value in personalization.items() if value != 0}
