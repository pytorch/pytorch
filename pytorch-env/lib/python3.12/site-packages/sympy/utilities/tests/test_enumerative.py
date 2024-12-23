from itertools import zip_longest

from sympy.utilities.enumerative import (
    list_visitor,
    MultisetPartitionTraverser,
    multiset_partitions_taocp
    )
from sympy.utilities.iterables import _set_partitions

# first some functions only useful as test scaffolding - these provide
# straightforward, but slow reference implementations against which to
# compare the real versions, and also a comparison to verify that
# different versions are giving identical results.

def part_range_filter(partition_iterator, lb, ub):
    """
    Filters (on the number of parts) a multiset partition enumeration

    Arguments
    =========

    lb, and ub are a range (in the Python slice sense) on the lpart
    variable returned from a multiset partition enumeration.  Recall
    that lpart is 0-based (it points to the topmost part on the part
    stack), so if you want to return parts of sizes 2,3,4,5 you would
    use lb=1 and ub=5.
    """
    for state in partition_iterator:
        f, lpart, pstack = state
        if lpart >= lb and lpart < ub:
            yield state

def multiset_partitions_baseline(multiplicities, components):
    """Enumerates partitions of a multiset

    Parameters
    ==========

    multiplicities
         list of integer multiplicities of the components of the multiset.

    components
         the components (elements) themselves

    Returns
    =======

    Set of partitions.  Each partition is tuple of parts, and each
    part is a tuple of components (with repeats to indicate
    multiplicity)

    Notes
    =====

    Multiset partitions can be created as equivalence classes of set
    partitions, and this function does just that.  This approach is
    slow and memory intensive compared to the more advanced algorithms
    available, but the code is simple and easy to understand.  Hence
    this routine is strictly for testing -- to provide a
    straightforward baseline against which to regress the production
    versions.  (This code is a simplified version of an earlier
    production implementation.)
    """

    canon = []                  # list of components with repeats
    for ct, elem in zip(multiplicities, components):
        canon.extend([elem]*ct)

    # accumulate the multiset partitions in a set to eliminate dups
    cache = set()
    n = len(canon)
    for nc, q in _set_partitions(n):
        rv = [[] for i in range(nc)]
        for i in range(n):
            rv[q[i]].append(canon[i])
        canonical = tuple(
            sorted([tuple(p) for p in rv]))
        cache.add(canonical)
    return cache


def compare_multiset_w_baseline(multiplicities):
    """
    Enumerates the partitions of multiset with AOCP algorithm and
    baseline implementation, and compare the results.

    """
    letters = "abcdefghijklmnopqrstuvwxyz"
    bl_partitions = multiset_partitions_baseline(multiplicities, letters)

    # The partitions returned by the different algorithms may have
    # their parts in different orders.  Also, they generate partitions
    # in different orders.  Hence the sorting, and set comparison.

    aocp_partitions = set()
    for state in multiset_partitions_taocp(multiplicities):
        p1 = tuple(sorted(
                [tuple(p) for p in list_visitor(state, letters)]))
        aocp_partitions.add(p1)

    assert bl_partitions == aocp_partitions

def compare_multiset_states(s1, s2):
    """compare for equality two instances of multiset partition states

    This is useful for comparing different versions of the algorithm
    to verify correctness."""
    # Comparison is physical, the only use of semantics is to ignore
    # trash off the top of the stack.
    f1, lpart1, pstack1 = s1
    f2, lpart2, pstack2 = s2

    if (lpart1 == lpart2) and (f1[0:lpart1+1] == f2[0:lpart2+1]):
        if pstack1[0:f1[lpart1+1]] == pstack2[0:f2[lpart2+1]]:
            return True
    return False

def test_multiset_partitions_taocp():
    """Compares the output of multiset_partitions_taocp with a baseline
    (set partition based) implementation."""

    # Test cases should not be too large, since the baseline
    # implementation is fairly slow.
    multiplicities = [2,2]
    compare_multiset_w_baseline(multiplicities)

    multiplicities = [4,3,1]
    compare_multiset_w_baseline(multiplicities)

def test_multiset_partitions_versions():
    """Compares Knuth-based versions of multiset_partitions"""
    multiplicities = [5,2,2,1]
    m = MultisetPartitionTraverser()
    for s1, s2 in zip_longest(m.enum_all(multiplicities),
                              multiset_partitions_taocp(multiplicities)):
        assert compare_multiset_states(s1, s2)

def subrange_exercise(mult, lb, ub):
    """Compare filter-based and more optimized subrange implementations

    Helper for tests, called with both small and larger multisets.
    """
    m = MultisetPartitionTraverser()
    assert m.count_partitions(mult) == \
        m.count_partitions_slow(mult)

    # Note - multiple traversals from the same
    # MultisetPartitionTraverser object cannot execute at the same
    # time, hence make several instances here.
    ma = MultisetPartitionTraverser()
    mc = MultisetPartitionTraverser()
    md = MultisetPartitionTraverser()

    #  Several paths to compute just the size two partitions
    a_it = ma.enum_range(mult, lb, ub)
    b_it = part_range_filter(multiset_partitions_taocp(mult), lb, ub)
    c_it = part_range_filter(mc.enum_small(mult, ub), lb, sum(mult))
    d_it = part_range_filter(md.enum_large(mult, lb), 0, ub)

    for sa, sb, sc, sd in zip_longest(a_it, b_it, c_it, d_it):
        assert compare_multiset_states(sa, sb)
        assert compare_multiset_states(sa, sc)
        assert compare_multiset_states(sa, sd)

def test_subrange():
    # Quick, but doesn't hit some of the corner cases
    mult = [4,4,2,1] # mississippi
    lb = 1
    ub = 2
    subrange_exercise(mult, lb, ub)


def test_subrange_large():
    # takes a second or so, depending on cpu, Python version, etc.
    mult = [6,3,2,1]
    lb = 4
    ub = 7
    subrange_exercise(mult, lb, ub)
