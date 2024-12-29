from sympy.combinatorics.permutations import Permutation, _af_rmul, \
    _af_invert, _af_new
from sympy.combinatorics.perm_groups import PermutationGroup, _orbit, \
    _orbit_transversal
from sympy.combinatorics.util import _distribute_gens_by_base, \
    _orbits_transversals_from_bsgs

"""
    References for tensor canonicalization:

    [1] R. Portugal "Algorithmic simplification of tensor expressions",
        J. Phys. A 32 (1999) 7779-7789

    [2] R. Portugal, B.F. Svaiter "Group-theoretic Approach for Symbolic
        Tensor Manipulation: I. Free Indices"
        arXiv:math-ph/0107031v1

    [3] L.R.U. Manssur, R. Portugal "Group-theoretic Approach for Symbolic
        Tensor Manipulation: II. Dummy Indices"
        arXiv:math-ph/0107032v1

    [4] xperm.c part of XPerm written by J. M. Martin-Garcia
        http://www.xact.es/index.html
"""


def dummy_sgs(dummies, sym, n):
    """
    Return the strong generators for dummy indices.

    Parameters
    ==========

    dummies : List of dummy indices.
        `dummies[2k], dummies[2k+1]` are paired indices.
        In base form, the dummy indices are always in
        consecutive positions.
    sym : symmetry under interchange of contracted dummies::
        * None  no symmetry
        * 0     commuting
        * 1     anticommuting

    n : number of indices

    Examples
    ========

    >>> from sympy.combinatorics.tensor_can import dummy_sgs
    >>> dummy_sgs(list(range(2, 8)), 0, 8)
    [[0, 1, 3, 2, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 5, 4, 6, 7, 8, 9],
     [0, 1, 2, 3, 4, 5, 7, 6, 8, 9], [0, 1, 4, 5, 2, 3, 6, 7, 8, 9],
     [0, 1, 2, 3, 6, 7, 4, 5, 8, 9]]
    """
    if len(dummies) > n:
        raise ValueError("List too large")
    res = []
    # exchange of contravariant and covariant indices
    if sym is not None:
        for j in dummies[::2]:
            a = list(range(n + 2))
            if sym == 1:
                a[n] = n + 1
                a[n + 1] = n
            a[j], a[j + 1] = a[j + 1], a[j]
            res.append(a)
    # rename dummy indices
    for j in dummies[:-3:2]:
        a = list(range(n + 2))
        a[j:j + 4] = a[j + 2], a[j + 3], a[j], a[j + 1]
        res.append(a)
    return res


def _min_dummies(dummies, sym, indices):
    """
    Return list of minima of the orbits of indices in group of dummies.
    See ``double_coset_can_rep`` for the description of ``dummies`` and ``sym``.
    ``indices`` is the initial list of dummy indices.

    Examples
    ========

    >>> from sympy.combinatorics.tensor_can import _min_dummies
    >>> _min_dummies([list(range(2, 8))], [0], list(range(10)))
    [0, 1, 2, 2, 2, 2, 2, 2, 8, 9]
    """
    num_types = len(sym)
    m = [min(dx) if dx else None for dx in dummies]
    res = indices[:]
    for i in range(num_types):
        for c, i in enumerate(indices):
            for j in range(num_types):
                if i in dummies[j]:
                    res[c] = m[j]
                    break
    return res


def _trace_S(s, j, b, S_cosets):
    """
    Return the representative h satisfying s[h[b]] == j

    If there is not such a representative return None
    """
    for h in S_cosets[b]:
        if s[h[b]] == j:
            return h
    return None


def _trace_D(gj, p_i, Dxtrav):
    """
    Return the representative h satisfying h[gj] == p_i

    If there is not such a representative return None
    """
    for h in Dxtrav:
        if h[gj] == p_i:
            return h
    return None


def _dumx_remove(dumx, dumx_flat, p0):
    """
    remove p0 from dumx
    """
    res = []
    for dx in dumx:
        if p0 not in dx:
            res.append(dx)
            continue
        k = dx.index(p0)
        if k % 2 == 0:
            p0_paired = dx[k + 1]
        else:
            p0_paired = dx[k - 1]
        dx.remove(p0)
        dx.remove(p0_paired)
        dumx_flat.remove(p0)
        dumx_flat.remove(p0_paired)
        res.append(dx)


def transversal2coset(size, base, transversal):
    a = []
    j = 0
    for i in range(size):
        if i in base:
            a.append(sorted(transversal[j].values()))
            j += 1
        else:
            a.append([list(range(size))])
    j = len(a) - 1
    while a[j] == [list(range(size))]:
        j -= 1
    return a[:j + 1]


def double_coset_can_rep(dummies, sym, b_S, sgens, S_transversals, g):
    r"""
    Butler-Portugal algorithm for tensor canonicalization with dummy indices.

    Parameters
    ==========

      dummies
        list of lists of dummy indices,
        one list for each type of index;
        the dummy indices are put in order contravariant, covariant
        [d0, -d0, d1, -d1, ...].

      sym
        list of the symmetries of the index metric for each type.

      possible symmetries of the metrics
              * 0     symmetric
              * 1     antisymmetric
              * None  no symmetry

      b_S
        base of a minimal slot symmetry BSGS.

      sgens
        generators of the slot symmetry BSGS.

      S_transversals
        transversals for the slot BSGS.

      g
        permutation representing the tensor.

    Returns
    =======

    Return 0 if the tensor is zero, else return the array form of
    the permutation representing the canonical form of the tensor.

    Notes
    =====

    A tensor with dummy indices can be represented in a number
    of equivalent ways which typically grows exponentially with
    the number of indices. To be able to establish if two tensors
    with many indices are equal becomes computationally very slow
    in absence of an efficient algorithm.

    The Butler-Portugal algorithm [3] is an efficient algorithm to
    put tensors in canonical form, solving the above problem.

    Portugal observed that a tensor can be represented by a permutation,
    and that the class of tensors equivalent to it under slot and dummy
    symmetries is equivalent to the double coset `D*g*S`
    (Note: in this documentation we use the conventions for multiplication
    of permutations p, q with (p*q)(i) = p[q[i]] which is opposite
    to the one used in the Permutation class)

    Using the algorithm by Butler to find a representative of the
    double coset one can find a canonical form for the tensor.

    To see this correspondence,
    let `g` be a permutation in array form; a tensor with indices `ind`
    (the indices including both the contravariant and the covariant ones)
    can be written as

    `t = T(ind[g[0]], \dots, ind[g[n-1]])`,

    where `n = len(ind)`;
    `g` has size `n + 2`, the last two indices for the sign of the tensor
    (trick introduced in [4]).

    A slot symmetry transformation `s` is a permutation acting on the slots
    `t \rightarrow T(ind[(g*s)[0]], \dots, ind[(g*s)[n-1]])`

    A dummy symmetry transformation acts on `ind`
    `t \rightarrow T(ind[(d*g)[0]], \dots, ind[(d*g)[n-1]])`

    Being interested only in the transformations of the tensor under
    these symmetries, one can represent the tensor by `g`, which transforms
    as

    `g -> d*g*s`, so it belongs to the coset `D*g*S`, or in other words
    to the set of all permutations allowed by the slot and dummy symmetries.

    Let us explain the conventions by an example.

    Given a tensor `T^{d3 d2 d1}{}_{d1 d2 d3}` with the slot symmetries
          `T^{a0 a1 a2 a3 a4 a5} = -T^{a2 a1 a0 a3 a4 a5}`

          `T^{a0 a1 a2 a3 a4 a5} = -T^{a4 a1 a2 a3 a0 a5}`

    and symmetric metric, find the tensor equivalent to it which
    is the lowest under the ordering of indices:
    lexicographic ordering `d1, d2, d3` and then contravariant
    before covariant index; that is the canonical form of the tensor.

    The canonical form is `-T^{d1 d2 d3}{}_{d1 d2 d3}`
    obtained using `T^{a0 a1 a2 a3 a4 a5} = -T^{a2 a1 a0 a3 a4 a5}`.

    To convert this problem in the input for this function,
    use the following ordering of the index names
    (- for covariant for short) `d1, -d1, d2, -d2, d3, -d3`

    `T^{d3 d2 d1}{}_{d1 d2 d3}` corresponds to `g = [4, 2, 0, 1, 3, 5, 6, 7]`
    where the last two indices are for the sign

    `sgens = [Permutation(0, 2)(6, 7), Permutation(0, 4)(6, 7)]`

    sgens[0] is the slot symmetry `-(0, 2)`
    `T^{a0 a1 a2 a3 a4 a5} = -T^{a2 a1 a0 a3 a4 a5}`

    sgens[1] is the slot symmetry `-(0, 4)`
    `T^{a0 a1 a2 a3 a4 a5} = -T^{a4 a1 a2 a3 a0 a5}`

    The dummy symmetry group D is generated by the strong base generators
    `[(0, 1), (2, 3), (4, 5), (0, 2)(1, 3), (0, 4)(1, 5)]`
    where the first three interchange covariant and contravariant
    positions of the same index (d1 <-> -d1) and the last two interchange
    the dummy indices themselves (d1 <-> d2).

    The dummy symmetry acts from the left
    `d = [1, 0, 2, 3, 4, 5, 6, 7]`  exchange `d1 \leftrightarrow -d1`
    `T^{d3 d2 d1}{}_{d1 d2 d3} == T^{d3 d2}{}_{d1}{}^{d1}{}_{d2 d3}`

    `g=[4, 2, 0, 1, 3, 5, 6, 7]  -> [4, 2, 1, 0, 3, 5, 6, 7] = _af_rmul(d, g)`
    which differs from `_af_rmul(g, d)`.

    The slot symmetry acts from the right
    `s = [2, 1, 0, 3, 4, 5, 7, 6]`  exchanges slots 0 and 2 and changes sign
    `T^{d3 d2 d1}{}_{d1 d2 d3} == -T^{d1 d2 d3}{}_{d1 d2 d3}`

    `g=[4,2,0,1,3,5,6,7]  -> [0, 2, 4, 1, 3, 5, 7, 6] = _af_rmul(g, s)`

    Example in which the tensor is zero, same slot symmetries as above:
    `T^{d2}{}_{d1 d3}{}^{d1 d3}{}_{d2}`

    `= -T^{d3}{}_{d1 d3}{}^{d1 d2}{}_{d2}`   under slot symmetry `-(0,4)`;

    `= T_{d3 d1}{}^{d3}{}^{d1 d2}{}_{d2}`    under slot symmetry `-(0,2)`;

    `= T^{d3}{}_{d1 d3}{}^{d1 d2}{}_{d2}`    symmetric metric;

    `= 0`  since two of these lines have tensors differ only for the sign.

    The double coset D*g*S consists of permutations `h = d*g*s` corresponding
    to equivalent tensors; if there are two `h` which are the same apart
    from the sign, return zero; otherwise
    choose as representative the tensor with indices
    ordered lexicographically according to `[d1, -d1, d2, -d2, d3, -d3]`
    that is ``rep = min(D*g*S) = min([d*g*s for d in D for s in S])``

    The indices are fixed one by one; first choose the lowest index
    for slot 0, then the lowest remaining index for slot 1, etc.
    Doing this one obtains a chain of stabilizers

    `S \rightarrow S_{b0} \rightarrow S_{b0,b1} \rightarrow \dots` and
    `D \rightarrow D_{p0} \rightarrow D_{p0,p1} \rightarrow \dots`

    where ``[b0, b1, ...] = range(b)`` is a base of the symmetric group;
    the strong base `b_S` of S is an ordered sublist of it;
    therefore it is sufficient to compute once the
    strong base generators of S using the Schreier-Sims algorithm;
    the stabilizers of the strong base generators are the
    strong base generators of the stabilizer subgroup.

    ``dbase = [p0, p1, ...]`` is not in general in lexicographic order,
    so that one must recompute the strong base generators each time;
    however this is trivial, there is no need to use the Schreier-Sims
    algorithm for D.

    The algorithm keeps a TAB of elements `(s_i, d_i, h_i)`
    where `h_i = d_i \times g \times s_i` satisfying `h_i[j] = p_j` for `0 \le j < i`
    starting from `s_0 = id, d_0 = id, h_0 = g`.

    The equations `h_0[0] = p_0, h_1[1] = p_1, \dots` are solved in this order,
    choosing each time the lowest possible value of p_i

    For `j < i`
    `d_i*g*s_i*S_{b_0, \dots, b_{i-1}}*b_j = D_{p_0, \dots, p_{i-1}}*p_j`
    so that for dx in `D_{p_0,\dots,p_{i-1}}` and sx in
    `S_{base[0], \dots, base[i-1]}` one has `dx*d_i*g*s_i*sx*b_j = p_j`

    Search for dx, sx such that this equation holds for `j = i`;
    it can be written as `s_i*sx*b_j = J, dx*d_i*g*J = p_j`
    `sx*b_j = s_i**-1*J; sx = trace(s_i**-1, S_{b_0,...,b_{i-1}})`
    `dx**-1*p_j = d_i*g*J; dx = trace(d_i*g*J, D_{p_0,...,p_{i-1}})`

    `s_{i+1} = s_i*trace(s_i**-1*J, S_{b_0,...,b_{i-1}})`
    `d_{i+1} = trace(d_i*g*J, D_{p_0,...,p_{i-1}})**-1*d_i`
    `h_{i+1}*b_i = d_{i+1}*g*s_{i+1}*b_i = p_i`

    `h_n*b_j = p_j` for all j, so that `h_n` is the solution.

    Add the found `(s, d, h)` to TAB1.

    At the end of the iteration sort TAB1 with respect to the `h`;
    if there are two consecutive `h` in TAB1 which differ only for the
    sign, the tensor is zero, so return 0;
    if there are two consecutive `h` which are equal, keep only one.

    Then stabilize the slot generators under `i` and the dummy generators
    under `p_i`.

    Assign `TAB = TAB1` at the end of the iteration step.

    At the end `TAB` contains a unique `(s, d, h)`, since all the slots
    of the tensor `h` have been fixed to have the minimum value according
    to the symmetries. The algorithm returns `h`.

    It is important that the slot BSGS has lexicographic minimal base,
    otherwise there is an `i` which does not belong to the slot base
    for which `p_i` is fixed by the dummy symmetry only, while `i`
    is not invariant from the slot stabilizer, so `p_i` is not in
    general the minimal value.

    This algorithm differs slightly from the original algorithm [3]:
      the canonical form is minimal lexicographically, and
      the BSGS has minimal base under lexicographic order.
      Equal tensors `h` are eliminated from TAB.


    Examples
    ========

    >>> from sympy.combinatorics.permutations import Permutation
    >>> from sympy.combinatorics.tensor_can import double_coset_can_rep, get_transversals
    >>> gens = [Permutation(x) for x in [[2, 1, 0, 3, 4, 5, 7, 6], [4, 1, 2, 3, 0, 5, 7, 6]]]
    >>> base = [0, 2]
    >>> g = Permutation([4, 2, 0, 1, 3, 5, 6, 7])
    >>> transversals = get_transversals(base, gens)
    >>> double_coset_can_rep([list(range(6))], [0], base, gens, transversals, g)
    [0, 1, 2, 3, 4, 5, 7, 6]

    >>> g = Permutation([4, 1, 3, 0, 5, 2, 6, 7])
    >>> double_coset_can_rep([list(range(6))], [0], base, gens, transversals, g)
    0
    """
    size = g.size
    g = g.array_form
    num_dummies = size - 2
    indices = list(range(num_dummies))
    all_metrics_with_sym = not any(_ is None for _ in sym)
    num_types = len(sym)
    dumx = dummies[:]
    dumx_flat = []
    for dx in dumx:
        dumx_flat.extend(dx)
    b_S = b_S[:]
    sgensx = [h._array_form for h in sgens]
    if b_S:
        S_transversals = transversal2coset(size, b_S, S_transversals)
    # strong generating set for D
    dsgsx = []
    for i in range(num_types):
        dsgsx.extend(dummy_sgs(dumx[i], sym[i], num_dummies))
    idn = list(range(size))
    # TAB = list of entries (s, d, h) where h = _af_rmuln(d,g,s)
    # for short, in the following d*g*s means _af_rmuln(d,g,s)
    TAB = [(idn, idn, g)]
    for i in range(size - 2):
        b = i
        testb = b in b_S and sgensx
        if testb:
            sgensx1 = [_af_new(_) for _ in sgensx]
            deltab = _orbit(size, sgensx1, b)
        else:
            deltab = {b}
        # p1 = min(IMAGES) = min(Union D_p*h*deltab for h in TAB)
        if all_metrics_with_sym:
            md = _min_dummies(dumx, sym, indices)
        else:
            md = [min(_orbit(size, [_af_new(
                ddx) for ddx in dsgsx], ii)) for ii in range(size - 2)]

        p_i = min(min(md[h[x]] for x in deltab) for s, d, h in TAB)
        dsgsx1 = [_af_new(_) for _ in dsgsx]
        Dxtrav = _orbit_transversal(size, dsgsx1, p_i, False, af=True) \
            if dsgsx else None
        if Dxtrav:
            Dxtrav = [_af_invert(x) for x in Dxtrav]
        # compute the orbit of p_i
        for ii in range(num_types):
            if p_i in dumx[ii]:
                # the orbit is made by all the indices in dum[ii]
                if sym[ii] is not None:
                    deltap = dumx[ii]
                else:
                    # the orbit is made by all the even indices if p_i
                    # is even, by all the odd indices if p_i is odd
                    p_i_index = dumx[ii].index(p_i) % 2
                    deltap = dumx[ii][p_i_index::2]
                break
        else:
            deltap = [p_i]
        TAB1 = []
        while TAB:
            s, d, h = TAB.pop()
            if min(md[h[x]] for x in deltab) != p_i:
                continue
            deltab1 = [x for x in deltab if md[h[x]] == p_i]
            # NEXT = s*deltab1 intersection (d*g)**-1*deltap
            dg = _af_rmul(d, g)
            dginv = _af_invert(dg)
            sdeltab = [s[x] for x in deltab1]
            gdeltap = [dginv[x] for x in deltap]
            NEXT = [x for x in sdeltab if x in gdeltap]
            # d, s satisfy
            # d*g*s*base[i-1] = p_{i-1}; using the stabilizers
            # d*g*s*S_{base[0],...,base[i-1]}*base[i-1] =
            # D_{p_0,...,p_{i-1}}*p_{i-1}
            # so that to find d1, s1 satisfying d1*g*s1*b = p_i
            # one can look for dx in D_{p_0,...,p_{i-1}} and
            # sx in S_{base[0],...,base[i-1]}
            # d1 = dx*d; s1 = s*sx
            # d1*g*s1*b = dx*d*g*s*sx*b = p_i
            for j in NEXT:
                if testb:
                    # solve s1*b = j with s1 = s*sx for some element sx
                    # of the stabilizer of ..., base[i-1]
                    # sx*b = s**-1*j; sx = _trace_S(s, j,...)
                    # s1 = s*trace_S(s**-1*j,...)
                    s1 = _trace_S(s, j, b, S_transversals)
                    if not s1:
                        continue
                    else:
                        s1 = [s[ix] for ix in s1]
                else:
                    s1 = s
                # assert s1[b] == j  # invariant
                # solve d1*g*j = p_i with d1 = dx*d for some element dg
                # of the stabilizer of ..., p_{i-1}
                # dx**-1*p_i = d*g*j; dx**-1 = trace_D(d*g*j,...)
                # d1 = trace_D(d*g*j,...)**-1*d
                # to save an inversion in the inner loop; notice we did
                # Dxtrav = [perm_af_invert(x) for x in Dxtrav] out of the loop
                if Dxtrav:
                    d1 = _trace_D(dg[j], p_i, Dxtrav)
                    if not d1:
                        continue
                else:
                    if p_i != dg[j]:
                        continue
                    d1 = idn
                assert d1[dg[j]] == p_i  # invariant
                d1 = [d1[ix] for ix in d]
                h1 = [d1[g[ix]] for ix in s1]
                # assert h1[b] == p_i  # invariant
                TAB1.append((s1, d1, h1))

        # if TAB contains equal permutations, keep only one of them;
        # if TAB contains equal permutations up to the sign, return 0
        TAB1.sort(key=lambda x: x[-1])
        prev = [0] * size
        while TAB1:
            s, d, h = TAB1.pop()
            if h[:-2] == prev[:-2]:
                if h[-1] != prev[-1]:
                    return 0
            else:
                TAB.append((s, d, h))
            prev = h

        # stabilize the SGS
        sgensx = [h for h in sgensx if h[b] == b]
        if b in b_S:
            b_S.remove(b)
        _dumx_remove(dumx, dumx_flat, p_i)
        dsgsx = []
        for i in range(num_types):
            dsgsx.extend(dummy_sgs(dumx[i], sym[i], num_dummies))
    return TAB[0][-1]


def canonical_free(base, gens, g, num_free):
    """
    Canonicalization of a tensor with respect to free indices
    choosing the minimum with respect to lexicographical ordering
    in the free indices.

    Explanation
    ===========

    ``base``, ``gens``  BSGS for slot permutation group
    ``g``               permutation representing the tensor
    ``num_free``        number of free indices
    The indices must be ordered with first the free indices

    See explanation in double_coset_can_rep
    The algorithm is a variation of the one given in [2].

    Examples
    ========

    >>> from sympy.combinatorics import Permutation
    >>> from sympy.combinatorics.tensor_can import canonical_free
    >>> gens = [[1, 0, 2, 3, 5, 4], [2, 3, 0, 1, 4, 5],[0, 1, 3, 2, 5, 4]]
    >>> gens = [Permutation(h) for h in gens]
    >>> base = [0, 2]
    >>> g = Permutation([2, 1, 0, 3, 4, 5])
    >>> canonical_free(base, gens, g, 4)
    [0, 3, 1, 2, 5, 4]

    Consider the product of Riemann tensors
    ``T = R^{a}_{d0}^{d1,d2}*R_{d2,d1}^{d0,b}``
    The order of the indices is ``[a, b, d0, -d0, d1, -d1, d2, -d2]``
    The permutation corresponding to the tensor is
    ``g = [0, 3, 4, 6, 7, 5, 2, 1, 8, 9]``

    In particular ``a`` is position ``0``, ``b`` is in position ``9``.
    Use the slot symmetries to get `T` is a form which is the minimal
    in lexicographic order in the free indices ``a`` and ``b``, e.g.
    ``-R^{a}_{d0}^{d1,d2}*R^{b,d0}_{d2,d1}`` corresponding to
    ``[0, 3, 4, 6, 1, 2, 7, 5, 9, 8]``

    >>> from sympy.combinatorics.tensor_can import riemann_bsgs, tensor_gens
    >>> base, gens = riemann_bsgs
    >>> size, sbase, sgens = tensor_gens(base, gens, [[], []], 0)
    >>> g = Permutation([0, 3, 4, 6, 7, 5, 2, 1, 8, 9])
    >>> canonical_free(sbase, [Permutation(h) for h in sgens], g, 2)
    [0, 3, 4, 6, 1, 2, 7, 5, 9, 8]
    """
    g = g.array_form
    size = len(g)
    if not base:
        return g[:]

    transversals = get_transversals(base, gens)
    for x in sorted(g[:-2]):
        if x not in base:
            base.append(x)
    h = g
    for transv in transversals:
        h_i = [size]*num_free
        # find the element s in transversals[i] such that
        # _af_rmul(h, s) has its free elements with the lowest position in h
        s = None
        for sk in transv.values():
            h1 = _af_rmul(h, sk)
            hi = [h1.index(ix) for ix in range(num_free)]
            if hi < h_i:
                h_i = hi
                s = sk
        if s:
            h = _af_rmul(h, s)
    return h


def _get_map_slots(size, fixed_slots):
    res = list(range(size))
    pos = 0
    for i in range(size):
        if i in fixed_slots:
            continue
        res[i] = pos
        pos += 1
    return res


def _lift_sgens(size, fixed_slots, free, s):
    a = []
    j = k = 0
    fd = list(zip(fixed_slots, free))
    fd = [y for x, y in sorted(fd)]
    num_free = len(free)
    for i in range(size):
        if i in fixed_slots:
            a.append(fd[k])
            k += 1
        else:
            a.append(s[j] + num_free)
            j += 1
    return a


def canonicalize(g, dummies, msym, *v):
    """
    canonicalize tensor formed by tensors

    Parameters
    ==========

    g : permutation representing the tensor

    dummies : list representing the dummy indices
      it can be a list of dummy indices of the same type
      or a list of lists of dummy indices, one list for each
      type of index;
      the dummy indices must come after the free indices,
      and put in order contravariant, covariant
      [d0, -d0, d1,-d1,...]

    msym :  symmetry of the metric(s)
        it can be an integer or a list;
        in the first case it is the symmetry of the dummy index metric;
        in the second case it is the list of the symmetries of the
        index metric for each type

    v : list, (base_i, gens_i, n_i, sym_i) for tensors of type `i`

    base_i, gens_i : BSGS for tensors of this type.
        The BSGS should have minimal base under lexicographic ordering;
        if not, an attempt is made do get the minimal BSGS;
        in case of failure,
        canonicalize_naive is used, which is much slower.

    n_i :    number of tensors of type `i`.

    sym_i :  symmetry under exchange of component tensors of type `i`.

        Both for msym and sym_i the cases are
            * None  no symmetry
            * 0     commuting
            * 1     anticommuting

    Returns
    =======

    0 if the tensor is zero, else return the array form of
    the permutation representing the canonical form of the tensor.

    Algorithm
    =========

    First one uses canonical_free to get the minimum tensor under
    lexicographic order, using only the slot symmetries.
    If the component tensors have not minimal BSGS, it is attempted
    to find it; if the attempt fails canonicalize_naive
    is used instead.

    Compute the residual slot symmetry keeping fixed the free indices
    using tensor_gens(base, gens, list_free_indices, sym).

    Reduce the problem eliminating the free indices.

    Then use double_coset_can_rep and lift back the result reintroducing
    the free indices.

    Examples
    ========

    one type of index with commuting metric;

    `A_{a b}` and `B_{a b}` antisymmetric and commuting

    `T = A_{d0 d1} * B^{d0}{}_{d2} * B^{d2 d1}`

    `ord = [d0,-d0,d1,-d1,d2,-d2]` order of the indices

    g = [1, 3, 0, 5, 4, 2, 6, 7]

    `T_c = 0`

    >>> from sympy.combinatorics.tensor_can import get_symmetric_group_sgs, canonicalize, bsgs_direct_product
    >>> from sympy.combinatorics import Permutation
    >>> base2a, gens2a = get_symmetric_group_sgs(2, 1)
    >>> t0 = (base2a, gens2a, 1, 0)
    >>> t1 = (base2a, gens2a, 2, 0)
    >>> g = Permutation([1, 3, 0, 5, 4, 2, 6, 7])
    >>> canonicalize(g, range(6), 0, t0, t1)
    0

    same as above, but with `B_{a b}` anticommuting

    `T_c = -A^{d0 d1} * B_{d0}{}^{d2} * B_{d1 d2}`

    can = [0,2,1,4,3,5,7,6]

    >>> t1 = (base2a, gens2a, 2, 1)
    >>> canonicalize(g, range(6), 0, t0, t1)
    [0, 2, 1, 4, 3, 5, 7, 6]

    two types of indices `[a,b,c,d,e,f]` and `[m,n]`, in this order,
    both with commuting metric

    `f^{a b c}` antisymmetric, commuting

    `A_{m a}` no symmetry, commuting

    `T = f^c{}_{d a} * f^f{}_{e b} * A_m{}^d * A^{m b} * A_n{}^a * A^{n e}`

    ord = [c,f,a,-a,b,-b,d,-d,e,-e,m,-m,n,-n]

    g = [0,7,3, 1,9,5, 11,6, 10,4, 13,2, 12,8, 14,15]

    The canonical tensor is
    `T_c = -f^{c a b} * f^{f d e} * A^m{}_a * A_{m d} * A^n{}_b * A_{n e}`

    can = [0,2,4, 1,6,8, 10,3, 11,7, 12,5, 13,9, 15,14]

    >>> base_f, gens_f = get_symmetric_group_sgs(3, 1)
    >>> base1, gens1 = get_symmetric_group_sgs(1)
    >>> base_A, gens_A = bsgs_direct_product(base1, gens1, base1, gens1)
    >>> t0 = (base_f, gens_f, 2, 0)
    >>> t1 = (base_A, gens_A, 4, 0)
    >>> dummies = [range(2, 10), range(10, 14)]
    >>> g = Permutation([0, 7, 3, 1, 9, 5, 11, 6, 10, 4, 13, 2, 12, 8, 14, 15])
    >>> canonicalize(g, dummies, [0, 0], t0, t1)
    [0, 2, 4, 1, 6, 8, 10, 3, 11, 7, 12, 5, 13, 9, 15, 14]
    """
    from sympy.combinatorics.testutil import canonicalize_naive
    if not isinstance(msym, list):
        if msym not in (0, 1, None):
            raise ValueError('msym must be 0, 1 or None')
        num_types = 1
    else:
        num_types = len(msym)
        if not all(msymx in (0, 1, None) for msymx in msym):
            raise ValueError('msym entries must be 0, 1 or None')
        if len(dummies) != num_types:
            raise ValueError(
                'dummies and msym must have the same number of elements')
    size = g.size
    num_tensors = 0
    v1 = []
    for base_i, gens_i, n_i, sym_i in v:
        # check that the BSGS is minimal;
        # this property is used in double_coset_can_rep;
        # if it is not minimal use canonicalize_naive
        if not _is_minimal_bsgs(base_i, gens_i):
            mbsgs = get_minimal_bsgs(base_i, gens_i)
            if not mbsgs:
                can = canonicalize_naive(g, dummies, msym, *v)
                return can
            base_i, gens_i = mbsgs
        v1.append((base_i, gens_i, [[]] * n_i, sym_i))
        num_tensors += n_i

    if num_types == 1 and not isinstance(msym, list):
        dummies = [dummies]
        msym = [msym]
    flat_dummies = []
    for dumx in dummies:
        flat_dummies.extend(dumx)

    if flat_dummies and flat_dummies != list(range(flat_dummies[0], flat_dummies[-1] + 1)):
        raise ValueError('dummies is not valid')

    # slot symmetry of the tensor
    size1, sbase, sgens = gens_products(*v1)
    if size != size1:
        raise ValueError(
            'g has size %d, generators have size %d' % (size, size1))
    free = [i for i in range(size - 2) if i not in flat_dummies]
    num_free = len(free)

    # g1 minimal tensor under slot symmetry
    g1 = canonical_free(sbase, sgens, g, num_free)
    if not flat_dummies:
        return g1
    # save the sign of g1
    sign = 0 if g1[-1] == size - 1 else 1

    # the free indices are kept fixed.
    # Determine free_i, the list of slots of tensors which are fixed
    # since they are occupied by free indices, which are fixed.
    start = 0
    for i, (base_i, gens_i, n_i, sym_i) in enumerate(v):
        free_i = []
        len_tens = gens_i[0].size - 2
        # for each component tensor get a list od fixed islots
        for j in range(n_i):
            # get the elements corresponding to the component tensor
            h = g1[start:(start + len_tens)]
            fr = []
            # get the positions of the fixed elements in h
            for k in free:
                if k in h:
                    fr.append(h.index(k))
            free_i.append(fr)
            start += len_tens
        v1[i] = (base_i, gens_i, free_i, sym_i)
    # BSGS of the tensor with fixed free indices
    # if tensor_gens fails in gens_product, use canonicalize_naive
    size, sbase, sgens = gens_products(*v1)

    # reduce the permutations getting rid of the free indices
    pos_free = [g1.index(x) for x in range(num_free)]
    size_red = size - num_free
    g1_red = [x - num_free for x in g1 if x in flat_dummies]
    if sign:
        g1_red.extend([size_red - 1, size_red - 2])
    else:
        g1_red.extend([size_red - 2, size_red - 1])
    map_slots = _get_map_slots(size, pos_free)
    sbase_red = [map_slots[i] for i in sbase if i not in pos_free]
    sgens_red = [_af_new([map_slots[i] for i in y._array_form if i not in pos_free]) for y in sgens]
    dummies_red = [[x - num_free for x in y] for y in dummies]
    transv_red = get_transversals(sbase_red, sgens_red)
    g1_red = _af_new(g1_red)
    g2 = double_coset_can_rep(
        dummies_red, msym, sbase_red, sgens_red, transv_red, g1_red)
    if g2 == 0:
        return 0
    # lift to the case with the free indices
    g3 = _lift_sgens(size, pos_free, free, g2)
    return g3


def perm_af_direct_product(gens1, gens2, signed=True):
    """
    Direct products of the generators gens1 and gens2.

    Examples
    ========

    >>> from sympy.combinatorics.tensor_can import perm_af_direct_product
    >>> gens1 = [[1, 0, 2, 3], [0, 1, 3, 2]]
    >>> gens2 = [[1, 0]]
    >>> perm_af_direct_product(gens1, gens2, False)
    [[1, 0, 2, 3, 4, 5], [0, 1, 3, 2, 4, 5], [0, 1, 2, 3, 5, 4]]
    >>> gens1 = [[1, 0, 2, 3, 5, 4], [0, 1, 3, 2, 4, 5]]
    >>> gens2 = [[1, 0, 2, 3]]
    >>> perm_af_direct_product(gens1, gens2, True)
    [[1, 0, 2, 3, 4, 5, 7, 6], [0, 1, 3, 2, 4, 5, 6, 7], [0, 1, 2, 3, 5, 4, 6, 7]]
    """
    gens1 = [list(x) for x in gens1]
    gens2 = [list(x) for x in gens2]
    s = 2 if signed else 0
    n1 = len(gens1[0]) - s
    n2 = len(gens2[0]) - s
    start = list(range(n1))
    end = list(range(n1, n1 + n2))
    if signed:
        gens1 = [gen[:-2] + end + [gen[-2] + n2, gen[-1] + n2]
                 for gen in gens1]
        gens2 = [start + [x + n1 for x in gen] for gen in gens2]
    else:
        gens1 = [gen + end for gen in gens1]
        gens2 = [start + [x + n1 for x in gen] for gen in gens2]

    res = gens1 + gens2

    return res


def bsgs_direct_product(base1, gens1, base2, gens2, signed=True):
    """
    Direct product of two BSGS.

    Parameters
    ==========

    base1 : base of the first BSGS.

    gens1 : strong generating sequence of the first BSGS.

    base2, gens2 : similarly for the second BSGS.

    signed : flag for signed permutations.

    Examples
    ========

    >>> from sympy.combinatorics.tensor_can import (get_symmetric_group_sgs, bsgs_direct_product)
    >>> base1, gens1 = get_symmetric_group_sgs(1)
    >>> base2, gens2 = get_symmetric_group_sgs(2)
    >>> bsgs_direct_product(base1, gens1, base2, gens2)
    ([1], [(4)(1 2)])
    """
    s = 2 if signed else 0
    n1 = gens1[0].size - s
    base = list(base1)
    base += [x + n1 for x in base2]
    gens1 = [h._array_form for h in gens1]
    gens2 = [h._array_form for h in gens2]
    gens = perm_af_direct_product(gens1, gens2, signed)
    size = len(gens[0])
    id_af = list(range(size))
    gens = [h for h in gens if h != id_af]
    if not gens:
        gens = [id_af]
    return base, [_af_new(h) for h in gens]


def get_symmetric_group_sgs(n, antisym=False):
    """
    Return base, gens of the minimal BSGS for (anti)symmetric tensor

    Parameters
    ==========

    n : rank of the tensor
    antisym : bool
        ``antisym = False`` symmetric tensor
        ``antisym = True``  antisymmetric tensor

    Examples
    ========

    >>> from sympy.combinatorics.tensor_can import get_symmetric_group_sgs
    >>> get_symmetric_group_sgs(3)
    ([0, 1], [(4)(0 1), (4)(1 2)])
    """
    if n == 1:
        return [], [_af_new(list(range(3)))]
    gens = [Permutation(n - 1)(i, i + 1)._array_form for i in range(n - 1)]
    if antisym == 0:
        gens = [x + [n, n + 1] for x in gens]
    else:
        gens = [x + [n + 1, n] for x in gens]
    base = list(range(n - 1))
    return base, [_af_new(h) for h in gens]

riemann_bsgs = [0, 2], [Permutation(0, 1)(4, 5), Permutation(2, 3)(4, 5),
                        Permutation(5)(0, 2)(1, 3)]


def get_transversals(base, gens):
    """
    Return transversals for the group with BSGS base, gens
    """
    if not base:
        return []
    stabs = _distribute_gens_by_base(base, gens)
    orbits, transversals = _orbits_transversals_from_bsgs(base, stabs)
    transversals = [{x: h._array_form for x, h in y.items()} for y in
                    transversals]
    return transversals


def _is_minimal_bsgs(base, gens):
    """
    Check if the BSGS has minimal base under lexigographic order.

    base, gens BSGS

    Examples
    ========

    >>> from sympy.combinatorics import Permutation
    >>> from sympy.combinatorics.tensor_can import riemann_bsgs, _is_minimal_bsgs
    >>> _is_minimal_bsgs(*riemann_bsgs)
    True
    >>> riemann_bsgs1 = ([2, 0], ([Permutation(5)(0, 1)(4, 5), Permutation(5)(0, 2)(1, 3)]))
    >>> _is_minimal_bsgs(*riemann_bsgs1)
    False
    """
    base1 = []
    sgs1 = gens[:]
    size = gens[0].size
    for i in range(size):
        if not all(h._array_form[i] == i for h in sgs1):
            base1.append(i)
            sgs1 = [h for h in sgs1 if h._array_form[i] == i]
    return base1 == base


def get_minimal_bsgs(base, gens):
    """
    Compute a minimal GSGS

    base, gens BSGS

    If base, gens is a minimal BSGS return it; else return a minimal BSGS
    if it fails in finding one, it returns None

    TODO: use baseswap in the case in which if it fails in finding a
    minimal BSGS

    Examples
    ========

    >>> from sympy.combinatorics import Permutation
    >>> from sympy.combinatorics.tensor_can import get_minimal_bsgs
    >>> riemann_bsgs1 = ([2, 0], ([Permutation(5)(0, 1)(4, 5), Permutation(5)(0, 2)(1, 3)]))
    >>> get_minimal_bsgs(*riemann_bsgs1)
    ([0, 2], [(0 1)(4 5), (5)(0 2)(1 3), (2 3)(4 5)])
    """
    G = PermutationGroup(gens)
    base, gens = G.schreier_sims_incremental()
    if not _is_minimal_bsgs(base, gens):
        return None
    return base, gens


def tensor_gens(base, gens, list_free_indices, sym=0):
    """
    Returns size, res_base, res_gens BSGS for n tensors of the
    same type.

    Explanation
    ===========

    base, gens BSGS for tensors of this type
    list_free_indices  list of the slots occupied by fixed indices
                       for each of the tensors

    sym symmetry under commutation of two tensors
    sym   None  no symmetry
    sym   0     commuting
    sym   1     anticommuting

    Examples
    ========

    >>> from sympy.combinatorics.tensor_can import tensor_gens, get_symmetric_group_sgs

    two symmetric tensors with 3 indices without free indices

    >>> base, gens = get_symmetric_group_sgs(3)
    >>> tensor_gens(base, gens, [[], []])
    (8, [0, 1, 3, 4], [(7)(0 1), (7)(1 2), (7)(3 4), (7)(4 5), (7)(0 3)(1 4)(2 5)])

    two symmetric tensors with 3 indices with free indices in slot 1 and 0

    >>> tensor_gens(base, gens, [[1], [0]])
    (8, [0, 4], [(7)(0 2), (7)(4 5)])

    four symmetric tensors with 3 indices, two of which with free indices

    """
    def _get_bsgs(G, base, gens, free_indices):
        """
        return the BSGS for G.pointwise_stabilizer(free_indices)
        """
        if not free_indices:
            return base[:], gens[:]
        else:
            H = G.pointwise_stabilizer(free_indices)
            base, sgs = H.schreier_sims_incremental()
            return base, sgs

    # if not base there is no slot symmetry for the component tensors
    # if list_free_indices.count([]) < 2 there is no commutation symmetry
    # so there is no resulting slot symmetry
    if not base and list_free_indices.count([]) < 2:
        n = len(list_free_indices)
        size = gens[0].size
        size = n * (size - 2) + 2
        return size, [], [_af_new(list(range(size)))]

    # if any(list_free_indices) one needs to compute the pointwise
    # stabilizer, so G is needed
    if any(list_free_indices):
        G = PermutationGroup(gens)
    else:
        G = None

    # no_free list of lists of indices for component tensors without fixed
    # indices
    no_free = []
    size = gens[0].size
    id_af = list(range(size))
    num_indices = size - 2
    if not list_free_indices[0]:
        no_free.append(list(range(num_indices)))
    res_base, res_gens = _get_bsgs(G, base, gens, list_free_indices[0])
    for i in range(1, len(list_free_indices)):
        base1, gens1 = _get_bsgs(G, base, gens, list_free_indices[i])
        res_base, res_gens = bsgs_direct_product(res_base, res_gens,
                                                 base1, gens1, 1)
        if not list_free_indices[i]:
            no_free.append(list(range(size - 2, size - 2 + num_indices)))
        size += num_indices
    nr = size - 2
    res_gens = [h for h in res_gens if h._array_form != id_af]
    # if sym there are no commuting tensors stop here
    if sym is None or not no_free:
        if not res_gens:
            res_gens = [_af_new(id_af)]
        return size, res_base, res_gens

    # if the component tensors have moinimal BSGS, so is their direct
    # product P; the slot symmetry group is S = P*C, where C is the group
    # to (anti)commute the component tensors with no free indices
    # a stabilizer has the property S_i = P_i*C_i;
    # the BSGS of P*C has SGS_P + SGS_C and the base is
    # the ordered union of the bases of P and C.
    # If P has minimal BSGS, so has S with this base.
    base_comm = []
    for i in range(len(no_free) - 1):
        ind1 = no_free[i]
        ind2 = no_free[i + 1]
        a = list(range(ind1[0]))
        a.extend(ind2)
        a.extend(ind1)
        base_comm.append(ind1[0])
        a.extend(list(range(ind2[-1] + 1, nr)))
        if sym == 0:
            a.extend([nr, nr + 1])
        else:
            a.extend([nr + 1, nr])
        res_gens.append(_af_new(a))
    res_base = list(res_base)
    # each base is ordered; order the union of the two bases
    for i in base_comm:
        if i not in res_base:
            res_base.append(i)
    res_base.sort()
    if not res_gens:
        res_gens = [_af_new(id_af)]

    return size, res_base, res_gens


def gens_products(*v):
    """
    Returns size, res_base, res_gens BSGS for n tensors of different types.

    Explanation
    ===========

    v is a sequence of (base_i, gens_i, free_i, sym_i)
    where
    base_i, gens_i  BSGS of tensor of type `i`
    free_i          list of the fixed slots for each of the tensors
                    of type `i`; if there are `n_i` tensors of type `i`
                    and none of them have fixed slots, `free = [[]]*n_i`
    sym   0 (1) if the tensors of type `i` (anti)commute among themselves

    Examples
    ========

    >>> from sympy.combinatorics.tensor_can import get_symmetric_group_sgs, gens_products
    >>> base, gens = get_symmetric_group_sgs(2)
    >>> gens_products((base, gens, [[], []], 0))
    (6, [0, 2], [(5)(0 1), (5)(2 3), (5)(0 2)(1 3)])
    >>> gens_products((base, gens, [[1], []], 0))
    (6, [2], [(5)(2 3)])
    """
    res_size, res_base, res_gens = tensor_gens(*v[0])
    for i in range(1, len(v)):
        size, base, gens = tensor_gens(*v[i])
        res_base, res_gens = bsgs_direct_product(res_base, res_gens, base,
                                                 gens, 1)
    res_size = res_gens[0].size
    id_af = list(range(res_size))
    res_gens = [h for h in res_gens if h != id_af]
    if not res_gens:
        res_gens = [id_af]
    return res_size, res_base, res_gens
