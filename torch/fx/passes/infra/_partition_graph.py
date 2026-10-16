"""Bitset reachability for contractions of an FX partition graph."""

from collections.abc import Iterable

from torch._C import _PartitionBitset, _PartitionMask
from torch.fx.node import Node


class _PartitionGraph:
    """Quotient graph with exact reachability after applying ``mask``.

    Unsupported FX nodes remain singleton groups. Representative IDs are bit
    positions, and removed bits are never reused. ``initial_owner`` is static,
    not the current node assignment.
    """

    def __init__(self, nodes: Iterable[Node]) -> None:
        # Like the partitioner's candidate traversal, initialization requires
        # the complete FX graph in topological order. IDs follow that order.
        nodes = tuple(nodes)
        self.initial_owner = {node: index for index, node in enumerate(nodes)}
        width = len(nodes)
        groups = range(width)
        self.mask = _PartitionMask(width)
        self._query_anchor: int | None = None
        self._rejected: set[int] = set()
        # Empty rows do not allocate sparse blocks.
        self.descendants = {pid: _PartitionBitset() for pid in groups}
        self.ancestors = {pid: _PartitionBitset() for pid in groups}
        for pid, node in enumerate(nodes):
            ancestors = self.ancestors[pid]
            for input_node in node.all_input_nodes:
                parent = self.initial_owner[input_node]
                ancestors |= self.ancestors[parent]
                ancestors[parent] = True
        for pid in reversed(groups):
            descendants = self.descendants[pid]
            for user in nodes[pid].users:
                child = self.initial_owner[user]
                descendants |= self.descendants[child]
                descendants[child] = True

    def would_create_cycle(self, a: int, b: int) -> bool:
        if a == b:
            raise ValueError("Cycle checking requires two distinct partitions")
        # Keep only one anchor's rejections, bounding cache space by O(N).
        if a != self._query_anchor:
            self._query_anchor = a
            self._rejected.clear()
        elif b in self._rejected:
            return True
        # An external vertex on a -> ... -> b (or b -> ... -> a) becomes
        # a cycle witness after contraction. A direct edge alone is legal.
        result = self.descendants[a].intersects(
            self.ancestors[b], self.mask
        ) or self.descendants[b].intersects(self.ancestors[a], self.mask)
        if result:
            self._rejected.add(b)
        return result

    def contract(self, keep: int, drop: int) -> None:
        """Commit a contraction after would_create_cycle has returned False."""
        # Any contraction can remove a witness or introduce a new path.
        self._query_anchor = None
        self._rejected.clear()
        # Snapshot the four sets before changing any rows. Common ancestors
        # and descendants already know both sides; only differences gain paths.
        anc_keep = self.ancestors[keep].masked(self.mask)
        anc_drop = self.ancestors[drop].masked(self.mask)
        desc_keep = self.descendants[keep].masked(self.mask)
        desc_drop = self.descendants[drop].masked(self.mask)
        for snapshot in (anc_keep, anc_drop, desc_keep, desc_drop):
            snapshot[keep] = snapshot[drop] = False
        anc_keep_only = anc_keep.difference(anc_drop)
        anc_drop_only = anc_drop.difference(anc_keep)
        desc_keep_only = desc_keep.difference(desc_drop)
        desc_drop_only = desc_drop.difference(desc_keep)
        # Each target already reaches its own side's closure. Propagate only
        # the other side's new bits, including keep for targets that knew drop.
        if not desc_drop_only.empty():
            for pid in anc_keep_only.set_bits():
                self.descendants[pid] |= desc_drop_only
        if not anc_drop_only.empty():
            desc_with_keep = desc_keep_only.copy()
            desc_with_keep[keep] = True
            for pid in anc_drop_only.set_bits():
                self.descendants[pid] |= desc_with_keep
        if not anc_drop_only.empty():
            for pid in desc_keep_only.set_bits():
                self.ancestors[pid] |= anc_drop_only
        if not desc_drop_only.empty():
            anc_with_keep = anc_keep_only.copy()
            anc_with_keep[keep] = True
            for pid in desc_drop_only.set_bits():
                self.ancestors[pid] |= anc_with_keep
        self.ancestors[keep] = anc_keep | anc_drop
        self.descendants[keep] = desc_keep | desc_drop
        # Other rows may retain dropped bits; queries and snapshots mask them.
        self.mask.reset(drop)
        del self.ancestors[drop], self.descendants[drop]
