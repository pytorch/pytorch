"""Bitset reachability for contractions of an FX partition graph."""

from collections import deque

from bitarray.util import any_and, ones, zeros


class _PartitionDebugState:
    """Optional membership tracking, independent of the reachability algorithm."""

    def __init__(self, initial_owner):
        self.owner = dict(initial_owner)
        self.members = {}
        for node, pid in self.owner.items():
            self.members.setdefault(pid, set()).add(node)

    def contract(self, keep, drop):
        for node in self.members[drop]:
            self.owner[node] = keep
        self.members[keep].update(self.members.pop(drop))

    def edges(self):
        # Reconstruct from the original FX edges when inspecting the graph.
        # No adjacency sets need to be maintained during contractions.
        incoming = {pid: set() for pid in self.members}
        outgoing = {pid: set() for pid in self.members}
        for node, source in self.owner.items():
            for user in node.users:
                target = self.owner[user]
                if source != target:
                    outgoing[source].add(target)
                    incoming[target].add(source)
        return incoming, outgoing


class _PartitionGraph:
    """Live quotient graph with exact reachability after masking by ``alive``.

    Unsupported FX nodes remain singleton groups. With the usual singleton
    initialization, representative IDs are bit positions. Diagnostic rebuilding
    also accepts negative/sparse IDs through ``assignment``. Removed bits are
    never reused. ``initial_owner`` is static, not the current node assignment.

    Standalone diagnostic callers retain membership inspection by default.
    Compilation callers explicitly use debug=False to avoid tracking it.
    """

    def __init__(self, nodes, assignment=None, *, debug=True):
        nodes = tuple(nodes)
        if assignment is None:
            self.initial_owner = {node: index for index, node in enumerate(nodes)}
        else:
            self.initial_owner = dict(assignment)
            next_id = min(self.initial_owner.values(), default=0) - 1
            for node in nodes:
                if node not in self.initial_owner:
                    self.initial_owner[node] = next_id
                    next_id -= 1

        groups = tuple(dict.fromkeys(self.initial_owner.values()))
        self._debug_state = _PartitionDebugState(self.initial_owner) if debug else None
        # Dense IDs need no conversion tables. Keep sparse-ID support only for
        # the optional reconstruction interface.
        self._labels = groups if assignment is not None else None
        self._positions = (
            {pid: index for index, pid in enumerate(groups)}
            if assignment is not None
            else None
        )
        incoming = {pid: set() for pid in groups}
        outgoing = {pid: set() for pid in groups}
        for node in nodes:
            source = self.initial_owner[node]
            for user in node.users:
                target = self.initial_owner[user]
                if source != target:
                    outgoing[source].add(target)
                    incoming[target].add(source)

        width = len(groups)
        self.alive = ones(width, endian="little")
        # Each mutable row needs its own buffer, including initially empty rows.
        self.desc = {pid: zeros(width, endian="little") for pid in groups}
        self.anc = {pid: zeros(width, endian="little") for pid in groups}
        # A rebuilt partition graph need not follow FX node order; use its own
        # topological order and reject invalid postprocessing before indexing.
        order = self._topological_order(incoming, outgoing)
        for pid in order:
            ancestors = self.anc[pid]
            for parent in incoming[pid]:
                ancestors |= self.anc[parent]
                ancestors[self._bit_index(parent)] = 1
        for pid in reversed(order):
            descendants = self.desc[pid]
            for child in outgoing[pid]:
                descendants |= self.desc[child]
                descendants[self._bit_index(child)] = 1

    def _debug(self):
        if self._debug_state is None:
            raise RuntimeError("Partition inspection requires debug=True at construction")
        return self._debug_state

    @property
    def owner(self):
        return self._debug().owner

    @property
    def members(self):
        return self._debug().members

    @property
    def incoming(self):
        return self._debug().edges()[0]

    @property
    def outgoing(self):
        return self._debug().edges()[1]

    @property
    def _pid_to_index(self):
        # Diagnostic view; the normal algorithm does not build this dictionary.
        if self._positions is None:
            return {pid: pid for pid in self.anc}
        return {pid: self._positions[pid] for pid in self.anc}

    @property
    def _index_to_pid(self):
        return range(len(self.alive)) if self._labels is None else self._labels

    def _bit_index(self, pid):
        return pid if self._positions is None else self._positions[pid]

    def would_create_cycle(self, a, b):
        if a == b:
            raise ValueError("Cycle checking requires two distinct partitions")
        # An external vertex on a -> ... -> b (or b -> ... -> a) becomes
        # a cycle witness after contraction. A direct edge alone is legal.
        # bitarray truthiness tests length, not whether any bits are set.
        # any_and checks the masked intersection in C without another buffer.
        return any_and(self.desc[a] & self.anc[b], self.alive) or any_and(
            self.desc[b] & self.anc[a], self.alive
        )

    def _pids(self, bits):
        """Iterate a live-only bitset without scanning all partition IDs."""
        if self._labels is None:
            return bits.search(1)
        return (self._labels[index] for index in bits.search(1))

    def contract(self, keep, drop):
        """Commit a contraction after would_create_cycle has returned False."""
        keep_index = self._bit_index(keep)
        drop_index = self._bit_index(drop)
        external = self.alive.copy()
        external[keep_index] = external[drop_index] = 0
        # Snapshot the four sets before changing any rows. Common ancestors
        # and descendants already know both sides; only differences gain paths.
        anc_keep = self.anc[keep] & external
        anc_drop = self.anc[drop] & external
        desc_keep = self.desc[keep] & external
        desc_drop = self.desc[drop] & external
        # Adding the representative must not mutate the four snapshot sets;
        # those still define the propagation ranges and strict final closure.
        desc_with_keep = desc_keep.copy()
        desc_with_keep[keep_index] = 1
        anc_with_keep = anc_keep.copy()
        anc_with_keep[keep_index] = 1
        for pid in self._pids(anc_keep & ~anc_drop):
            self.desc[pid] |= desc_drop
        for pid in self._pids(anc_drop & ~anc_keep):
            self.desc[pid] |= desc_with_keep
        for pid in self._pids(desc_keep & ~desc_drop):
            self.anc[pid] |= anc_drop
        for pid in self._pids(desc_drop & ~desc_keep):
            self.anc[pid] |= anc_with_keep
        self.anc[keep] = anc_keep | anc_drop
        self.desc[keep] = desc_keep | desc_drop
        # Other rows may retain the dropped bit. Queries and propagation exclude it
        # through alive; there is no global sweep to clear historical bits.
        self.alive[drop_index] = 0
        del self.anc[drop], self.desc[drop]
        if self._debug_state is not None:
            self._debug_state.contract(keep, drop)

    @staticmethod
    def _topological_order(incoming, outgoing):
        degree = {pid: len(edges) for pid, edges in incoming.items()}
        queue = deque(pid for pid, count in degree.items() if count == 0)
        order = []
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

    def assert_acyclic(self):
        self._topological_order(*self._debug().edges())
