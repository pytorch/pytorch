"""Bitset reachability for contractions of an FX partition graph."""

from collections import deque
from collections.abc import Iterable

from torch._C import _PartitionBitset
from torch.fx.node import Node


class _PartitionGraph:
    """Live quotient graph with exact reachability after masking by ``alive``.

    Unsupported FX nodes remain singleton groups. Representative IDs are bit
    positions, and removed bits are never reused. ``initial_owner`` is static,
    not the current node assignment.
    """

    def __init__(self, nodes: Iterable[Node]) -> None:
        nodes = tuple(nodes)
        self.initial_owner = {node: index for index, node in enumerate(nodes)}
        width = len(nodes)
        groups = range(width)
        incoming: dict[int, set[int]] = {pid: set() for pid in groups}
        outgoing: dict[int, set[int]] = {pid: set() for pid in groups}
        for node in nodes:
            source = self.initial_owner[node]
            for user in node.users:
                target = self.initial_owner[user]
                if source != target:
                    outgoing[source].add(target)
                    incoming[target].add(source)

        self.alive = _PartitionBitset(width, fill=True)
        # Each mutable row needs its own buffer, including initially empty rows.
        self.descendants = {pid: _PartitionBitset(width) for pid in groups}
        self.ancestors = {pid: _PartitionBitset(width) for pid in groups}
        order = self._topological_order(incoming, outgoing)
        for pid in order:
            ancestors = self.ancestors[pid]
            for parent in incoming[pid]:
                ancestors |= self.ancestors[parent]
                ancestors[parent] = True
        for pid in reversed(order):
            descendants = self.descendants[pid]
            for child in outgoing[pid]:
                descendants |= self.descendants[child]
                descendants[child] = True

    def would_create_cycle(self, a: int, b: int) -> bool:
        if a == b:
            raise ValueError("Cycle checking requires two distinct partitions")
        # An external vertex on a -> ... -> b (or b -> ... -> a) becomes
        # a cycle witness after contraction. A direct edge alone is legal.
        # Check the masked intersection without allocating temporary bitsets.
        return self.descendants[a].intersects(
            self.ancestors[b], self.alive
        ) or self.descendants[b].intersects(self.ancestors[a], self.alive)

    def contract(self, keep: int, drop: int) -> None:
        """Commit a contraction after would_create_cycle has returned False."""
        external = self.alive.copy()
        external[keep] = external[drop] = False
        # Snapshot the four sets before changing any rows. Common ancestors
        # and descendants already know both sides; only differences gain paths.
        anc_keep = self.ancestors[keep] & external
        anc_drop = self.ancestors[drop] & external
        desc_keep = self.descendants[keep] & external
        desc_drop = self.descendants[drop] & external
        # Adding the representative must not mutate the four snapshot sets;
        # those still define the propagation ranges and strict final closure.
        desc_with_keep = desc_keep.copy()
        desc_with_keep[keep] = True
        anc_with_keep = anc_keep.copy()
        anc_with_keep[keep] = True
        for pid in anc_keep.difference(anc_drop).set_bits():
            self.descendants[pid] |= desc_drop
        for pid in anc_drop.difference(anc_keep).set_bits():
            self.descendants[pid] |= desc_with_keep
        for pid in desc_keep.difference(desc_drop).set_bits():
            self.ancestors[pid] |= anc_drop
        for pid in desc_drop.difference(desc_keep).set_bits():
            self.ancestors[pid] |= anc_with_keep
        self.ancestors[keep] = anc_keep | anc_drop
        self.descendants[keep] = desc_keep | desc_drop
        # Other rows may retain the dropped bit. Queries and propagation exclude it
        # through alive; there is no global sweep to clear historical bits.
        self.alive[drop] = False
        del self.ancestors[drop], self.descendants[drop]

    @staticmethod
    def _topological_order(
        incoming: dict[int, set[int]], outgoing: dict[int, set[int]]
    ) -> list[int]:
        degree = {pid: len(edges) for pid, edges in incoming.items()}
        queue = deque(pid for pid, count in degree.items() if count == 0)
        order: list[int] = []
        while queue:
            current = queue.popleft()
            order.append(current)
            for successor in outgoing[current]:
                degree[successor] -= 1
                if degree[successor] == 0:
                    queue.append(successor)
        if len(order) != len(degree):
            raise RuntimeError("Partition graph contains a dependency cycle")
        return order
