from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation
from sympy.utilities.iterables import uniq

_af_new = Permutation._af_new


def DirectProduct(*groups):
    """
    Returns the direct product of several groups as a permutation group.

    Explanation
    ===========

    This is implemented much like the __mul__ procedure for taking the direct
    product of two permutation groups, but the idea of shifting the
    generators is realized in the case of an arbitrary number of groups.
    A call to DirectProduct(G1, G2, ..., Gn) is generally expected to be faster
    than a call to G1*G2*...*Gn (and thus the need for this algorithm).

    Examples
    ========

    >>> from sympy.combinatorics.group_constructs import DirectProduct
    >>> from sympy.combinatorics.named_groups import CyclicGroup
    >>> C = CyclicGroup(4)
    >>> G = DirectProduct(C, C, C)
    >>> G.order()
    64

    See Also
    ========

    sympy.combinatorics.perm_groups.PermutationGroup.__mul__

    """
    degrees = []
    gens_count = []
    total_degree = 0
    total_gens = 0
    for group in groups:
        current_deg = group.degree
        current_num_gens = len(group.generators)
        degrees.append(current_deg)
        total_degree += current_deg
        gens_count.append(current_num_gens)
        total_gens += current_num_gens
    array_gens = []
    for i in range(total_gens):
        array_gens.append(list(range(total_degree)))
    current_gen = 0
    current_deg = 0
    for i in range(len(gens_count)):
        for j in range(current_gen, current_gen + gens_count[i]):
            gen = ((groups[i].generators)[j - current_gen]).array_form
            array_gens[j][current_deg:current_deg + degrees[i]] = \
                [x + current_deg for x in gen]
        current_gen += gens_count[i]
        current_deg += degrees[i]
    perm_gens = list(uniq([_af_new(list(a)) for a in array_gens]))
    return PermutationGroup(perm_gens, dups=False)
