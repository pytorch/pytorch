# mypy: allow-untyped-defs

import dataclasses
from collections import defaultdict
from typing import Any, cast

import torch
import torch.distributed as dist

from .default_planner import DefaultLoadPlanner
from .planner import LoadItemType, LoadPlan, ReadItem


_ReadSignature = tuple[
    int,
    str,
    tuple[int, ...],
    tuple[int, ...],
    str,
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]
_TensorMetadata = tuple[tuple[int, ...], str, str, int]


@dataclasses.dataclass(frozen=True)
class _BroadcastTask:
    read_item: ReadItem
    ranks: tuple[int, ...]
    src: int


@dataclasses.dataclass(frozen=True)
class _CooperativePlan:
    schedule: tuple[_BroadcastTask, ...]
    groups: tuple[tuple[int, ...], ...]


def _index_offset(index) -> tuple[int, ...]:
    return tuple(index.offset) if index.offset is not None else ()


def _read_signature(item: ReadItem) -> _ReadSignature:
    return (
        item.type.value,
        item.dest_index.fqn,
        _index_offset(item.dest_index),
        tuple(item.dest_offsets),
        item.storage_index.fqn,
        _index_offset(item.storage_index),
        tuple(item.storage_offsets),
        tuple(item.lengths),
    )


def _create_cooperative_plans(
    all_plans: list[LoadPlan],
    global_ranks: tuple[int, ...],
    min_tensor_bytes: int,
) -> list[LoadPlan]:
    requests: dict[
        _ReadSignature,
        list[tuple[int, ReadItem, _TensorMetadata]],
    ] = defaultdict(list)

    for plan_rank, plan in enumerate(all_plans):
        tensor_metadata = cast(dict[_ReadSignature, _TensorMetadata], plan.planner_data)
        for item in plan.items:
            if item.type != LoadItemType.TENSOR:
                continue
            signature = _read_signature(item)
            metadata = tensor_metadata.get(signature)
            if metadata is not None:
                requests[signature].append((plan_rank, item, metadata))

    candidates = []
    for signature, occurrences in requests.items():
        plan_ranks = [rank for rank, _, _ in occurrences]
        metadata = {value for _, _, value in occurrences}
        if (
            len(occurrences) <= 1
            or len(plan_ranks) != len(set(plan_ranks))
            or len(metadata) != 1
        ):
            continue
        nbytes = next(iter(metadata))[3]
        if nbytes < min_tensor_bytes:
            continue
        candidates.append((nbytes, signature, occurrences))

    assigned_bytes = [0] * len(all_plans)
    tasks = []
    drops: list[set[_ReadSignature]] = [set() for _ in all_plans]
    for nbytes, signature, occurrences in sorted(
        candidates, key=lambda value: (-value[0], value[1])
    ):
        plan_ranks = tuple(sorted(rank for rank, _, _ in occurrences))
        leader = min(plan_ranks, key=lambda rank: (assigned_bytes[rank], rank))
        assigned_bytes[leader] += nbytes
        ranks = tuple(global_ranks[rank] for rank in plan_ranks)
        task = _BroadcastTask(
            read_item=occurrences[0][1],
            ranks=ranks,
            src=global_ranks[leader],
        )
        tasks.append(task)
        for plan_rank in plan_ranks:
            if plan_rank != leader:
                drops[plan_rank].add(signature)

    schedule = tuple(sorted(tasks, key=lambda task: _read_signature(task.read_item)))
    groups = tuple(sorted({task.ranks for task in schedule}))
    result = []
    for plan_rank, plan in enumerate(all_plans):
        rank = global_ranks[plan_rank]
        local_schedule = tuple(task for task in schedule if rank in task.ranks)
        items = [
            item
            for item in plan.items
            if _read_signature(item) not in drops[plan_rank]
        ]
        result.append(
            dataclasses.replace(
                plan,
                items=items,
                planner_data=_CooperativePlan(local_schedule, groups),
            )
        )
    return result


class CooperativeLoadPlanner(DefaultLoadPlanner):
    """Deduplicate identical tensor reads and broadcast them to peer ranks.

    The global planning step produces the only collective schedule. A tensor is
    cooperative only when every peer reports an identical read request and
    destination tensor metadata; all other reads retain default DCP behavior.

    The planner and dcp.load must receive the same process group.
    """

    def __init__(
        self,
        *args: Any,
        process_group: dist.ProcessGroup | None = None,
        min_tensor_bytes: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.process_group = process_group
        self.min_tensor_bytes = min_tensor_bytes
        self._schedule: tuple[_BroadcastTask, ...] = ()
        self._groups: tuple[tuple[int, ...], ...] = ()

    def _global_ranks(self) -> tuple[int, ...]:
        if not dist.is_available() or not dist.is_initialized():
            return (0,)
        world_size = dist.get_world_size(self.process_group)
        return tuple(
            dist.get_global_rank(self.process_group, rank)
            if self.process_group is not None
            else rank
            for rank in range(world_size)
        )

    def create_local_plan(self) -> LoadPlan:
        plan = super().create_local_plan()
        tensor_metadata = {}
        for item in plan.items:
            if item.type != LoadItemType.TENSOR:
                continue
            tensor = self.resolve_tensor(item)
            tensor_metadata[_read_signature(item)] = (
                tuple(tensor.shape),
                str(tensor.dtype),
                tensor.device.type,
                tensor.numel() * tensor.element_size(),
            )
        return dataclasses.replace(plan, planner_data=tensor_metadata)

    def create_global_plan(self, all_plans: list[LoadPlan]) -> list[LoadPlan]:
        all_plans = super().create_global_plan(all_plans)
        return _create_cooperative_plans(
            all_plans,
            self._global_ranks(),
            self.min_tensor_bytes,
        )

    def finish_plan(self, new_plan: LoadPlan) -> LoadPlan:
        plan = super().finish_plan(new_plan)
        planner_data = cast(_CooperativePlan, plan.planner_data)
        self._schedule = planner_data.schedule
        self._groups = planner_data.groups
        return plan

    def finish_load(self) -> None:
        if not self._schedule:
            return

        rank = dist.get_rank()
        process_ranks = self._global_ranks()
        groups: dict[tuple[int, ...], dist.ProcessGroup] = {}
        created_groups = []

        try:
            for ranks in self._groups:
                if rank not in ranks:
                    continue
                if ranks == process_ranks:
                    group = self.process_group or dist.group.WORLD
                else:
                    group = dist.new_group(
                        ranks=list(ranks),
                        use_local_synchronization=True,
                    )
                    created_groups.append(group)
                groups[ranks] = group

            for task in self._schedule:
                tensor = self.resolve_tensor(task.read_item)
                buffer = tensor if tensor.is_contiguous() else tensor.contiguous()
                dist.broadcast(buffer, src=task.src, group=groups[task.ranks])
                if buffer is not tensor:
                    tensor.copy_(buffer)
        finally:
            for group in reversed(created_groups):
                dist.destroy_process_group(group)
