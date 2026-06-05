import torch.fx as fx
from torch.utils._ordered_set import OrderedSet


class BitsetAncestors:
    """Precomputed transitive ancestor sets using Python arbitrary-precision ints.

    Each node gets an index i (0..N-1) from topological order. A node's
    ancestor set is a single Python ``int`` where bit j is set iff node j
    is a transitive ancestor. Python ``int`` is arbitrary-precision, so for
    N nodes each int is ~N/8 bytes internally (~N/64 machine words).

    The key advantage over ``dict[fx.Node, OrderedSet[fx.Node]]`` is that
    CPython implements ``int |= int`` as a C-level loop over machine words,
    making the transitive closure O(N^2 / 64) instead of O(N^2) in
    Python-level set operations.

    Example -- diamond graph (edges point downward)::

            a          index 0
           / \\
          b   c        index 1, 2
           \\ /
            d          index 3

    Ancestor bitsets::

        a: 0b0000 = 0   (no ancestors)
        b: 0b0001 = 1   (bit 0 -> a)
        c: 0b0001 = 1   (bit 0 -> a)
        d: 0b0111 = 7   (bits 0,1,2 -> a, b, c)

    How d's bitset is built (d has parents b and c)::

        b = 0
        # parent b (idx=1): set bit 1, merge b's ancestors
        b |= (1 << 1) | bits[1]   # b = 0b0010 | 0b0001 = 0b0011
        # parent c (idx=2): set bit 2, merge c's ancestors
        b |= (1 << 2) | bits[2]   # b = 0b0011 | 0b0100 | 0b0001 = 0b0111
        bits[3] = 0b0111

    Querying::

        ancestors = BitsetAncestors(nodes)
        ancestors.is_ancestor(a, d)    # (bits[3] >> 0) & 1 -> True
        ancestors.has_dep(a, d)  # True (a is ancestor of d)
        list(ancestors.iter_ancestors(d))  # [a, b, c] via bit-scan

    Iteration uses the x & -x trick to isolate the lowest set bit::

        bits = 0b0111
        bits & -bits = 0b0001  -> idx 0 -> yield a, clear: 0b0110
        bits & -bits = 0b0010  -> idx 1 -> yield b, clear: 0b0100
        bits & -bits = 0b0100  -> idx 2 -> yield c, clear: 0b0000

    Args:
        nodes: topologically sorted list of FX nodes.
        extra_inputs: optional additional edges beyond the FX graph
            (e.g. hiding-interval deps in the overlap bucketer).
    """

    def __init__(
        self,
        nodes: list[fx.Node],
        extra_inputs: dict[fx.Node, OrderedSet[fx.Node]] | None = None,
    ):
        n = len(nodes)
        node_to_idx: dict[fx.Node, int] = {nd: i for i, nd in enumerate(nodes)}
        bits = [0] * n
        extra = extra_inputs or {}

        for i, node in enumerate(nodes):
            b = 0
            for inp in node._input_nodes:
                j = node_to_idx.get(inp)
                if j is not None:
                    b |= (1 << j) | bits[j]
            for inp in extra.get(node, ()):
                j = node_to_idx.get(inp)
                if j is not None:
                    b |= (1 << j) | bits[j]
            bits[i] = b

        self._bits = bits
        self._node_to_idx = node_to_idx
        self._idx_to_node = nodes

    def is_ancestor(self, ancestor: fx.Node, descendant: fx.Node) -> bool:
        """O(1) test: is ``ancestor`` a transitive ancestor of ``descendant``?"""
        anc_idx = self._node_to_idx.get(ancestor)
        desc_idx = self._node_to_idx.get(descendant)
        if anc_idx is None or desc_idx is None:
            return False
        return bool((self._bits[desc_idx] >> anc_idx) & 1)

    def has_dep(self, n1: fx.Node, n2: fx.Node) -> bool:
        """Check if either node is an ancestor of the other."""
        return self.is_ancestor(n1, n2) or self.is_ancestor(n2, n1)

    def get_ancestor_bits(self, node: fx.Node) -> int:
        """Return the raw ancestor bitset for ``node``."""
        return self._bits[self._node_to_idx[node]]

    def node_bit(self, node: fx.Node) -> int:
        """Return the single-bit mask ``1 << idx`` for ``node``."""
        return 1 << self._node_to_idx[node]

    def ancestors_intersect(self, node: fx.Node, mask: int) -> bool:
        """Check if any ancestor of ``node`` is set in ``mask``."""
        idx = self._node_to_idx.get(node)
        if idx is None:
            return False
        return bool(self._bits[idx] & mask)

    def iter_ancestors(self, node: fx.Node):
        """Yield all ancestors of ``node`` via lowest-bit scan."""
        bits = self._bits[self._node_to_idx[node]]
        idx_to_node = self._idx_to_node
        while bits:
            idx = (bits & -bits).bit_length() - 1
            yield idx_to_node[idx]
            bits &= bits - 1
