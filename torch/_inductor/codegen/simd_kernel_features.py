from __future__ import annotations

import collections
import itertools
from typing import Any, Dict, Iterable, List, Type, Union

import sympy

import torch

from ...utils._ordered_set import OrderedSet
from ..dependencies import Dep, MemoryDep
from ..runtime.hints import ReductionHint
from ..scheduler import SchedulerNode
from ..utils import cache_on_self
from ..virtualized import V


class NodeScheduleMarker:
    @staticmethod
    def only_nodes(it: Iterable[NodeScheduleEntry]) -> Iterable[SchedulerNode]:
        for item in it:
            if not (item is DisableReduction or item is EnableReduction):
                yield item  # type: ignore[misc]

    @staticmethod
    def is_reduction() -> bool:
        return False


NodeScheduleEntry = Union[SchedulerNode, Type[NodeScheduleMarker]]


class DisableReduction(NodeScheduleMarker):
    """
    Marker to invoke `kernel.disable_reduction()`.  This closes a
    reduction loop and allows for pointwise ops to occur on the output
    of a reduction.
    """


class EnableReduction(NodeScheduleMarker):
    """
    Marker to end a DisableReduction block.
    """

    @staticmethod
    def filter(node_schedule: List[NodeScheduleEntry]) -> Iterable[SchedulerNode]:
        """
        Get the nodes from node_schedule skipping those in a
        DisableReduction block.
        """
        disabled = False
        for node in node_schedule:
            if node in (EnableReduction, DisableReduction):
                # Don't tile stuff outside the main reduction loop
                disabled = node is DisableReduction
            elif disabled:
                pass
            else:
                yield node  # type: ignore[misc]


class SIMDKernelFeatures:
    """
    An ordered schedule of nodes that will become a single kernel.
    """

    def __init__(
        self,
        node_schedule: List[NodeScheduleEntry],
        numel: sympy.Expr,
        reduction_numel: sympy.Expr = sympy.S.One,
    ):
        self.node_schedule = node_schedule
        # numel excludes reduction_numel
        self.numel: sympy.Expr = V.graph.sizevars.simplify(numel)
        self.reduction_numel: sympy.Expr = V.graph.sizevars.simplify(reduction_numel)

    @cache_on_self
    def is_reduction(self) -> bool:
        return self.reduction_numel != 1

    @cache_on_self
    def scheduler_nodes(self) -> Iterable[SchedulerNode]:
        return tuple(NodeScheduleMarker.only_nodes(self.node_schedule))

    def reduction_nodes(self) -> List[SchedulerNode]:
        return [n for n in self.scheduler_nodes() if n.is_reduction()]

    @cache_on_self
    def buf_accesses(self) -> Dict[str, List[Dep]]:
        """only needed for config.benchmark_kernel"""
        buf_accesses = collections.defaultdict(list)
        for node in self.scheduler_nodes():
            for access in node.read_writes.reads | node.read_writes.writes:
                buf_accesses[access.name].append(access)
        return buf_accesses

    @cache_on_self
    def op_counts(self) -> collections.Counter[str]:
        counts: collections.Counter[str] = collections.Counter()
        for node in self.scheduler_nodes():
            counts.update(node._body.op_counts)
        return counts

    def contains_op(self, op_name: str) -> bool:
        """True if V.ops.{op_name} is used in node_schedule"""
        return bool(self.op_counts().get(op_name))

    def get_mutations(self) -> OrderedSet[str]:
        mutations: OrderedSet[str] = OrderedSet()
        for node in self.scheduler_nodes():
            for buf in node.get_outputs():
                mutations.update(buf.get_mutations())
        return mutations

    @cache_on_self
    def select_index_dtype(self) -> torch.dtype:
        # Gather all used buffer names
        buffer_names: OrderedSet[str] = OrderedSet()
        for node in self.scheduler_nodes():
            buffer_names.update(node.get_buffer_names())
            buffer_names.update(node.used_buffer_names())
        buffers = [V.graph.get_buffer(name) for name in buffer_names]

        # In theory we can separately check xnumel and rnumel are <= int_max
        # but some indexers do use the full linear index so we need to be
        # conservative here.
        total_numel = self.numel * self.reduction_numel

        from .simd import SIMDScheduling

        if SIMDScheduling.can_use_32bit_indexing(total_numel, buffers):
            return torch.int32
        return torch.int64

    @cache_on_self
    def get_reduction_hint(self) -> ReductionHint:
        reductions = self.reduction_nodes()
        if len(reductions) > 0:
            hints = [self.reduction_hint(n) for n in reductions]
            if hints.count(hints[0]) == len(hints):
                reduction_hint_val = hints[0]
            else:
                reduction_hint_val = ReductionHint.DEFAULT

            if (
                reduction_hint_val == ReductionHint.INNER
                and self.has_non_contiguous_pw_in_reduction_kernel()
            ):
                reduction_hint_val = ReductionHint.DEFAULT
        else:
            reduction_hint_val = ReductionHint.DEFAULT
        return reduction_hint_val

    def has_non_contiguous_pw_in_reduction_kernel(self) -> bool:
        pointwise_nodes = [
            n
            for n in self.scheduler_nodes()
            if not n.is_reduction()
            and n.group[1][0] == self.numel * self.reduction_numel
        ]
        for node in pointwise_nodes:
            # An index can be an integer when loading a random seed.
            if not all(
                not isinstance(dep, MemoryDep)
                or dep.is_contiguous()
                or isinstance(dep.index, (sympy.Integer, int))
                or dep.stride1_for_last_dim()
                for dep in itertools.chain(
                    node.read_writes.reads, node.read_writes.writes
                )
            ):
                return True
        return False

    @staticmethod
    def reduction_hint(node: Any) -> ReductionHint:
        assert node.is_reduction()
        if node.node.data.reduction_hint != ReductionHint.INNER and all(
            dep.is_contiguous()
            for dep in itertools.chain(node.read_writes.reads, node.read_writes.writes)
        ):
            return ReductionHint.INNER
        else:
            return node.node.data.reduction_hint
