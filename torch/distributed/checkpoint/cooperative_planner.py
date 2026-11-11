# mypy: allow-untyped-defs

import dataclasses
from collections import defaultdict
from typing import Any, cast

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
    src: int


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
    """Rewrite plans so full-group duplicate TENSOR reads are owned once.

    A ReadSignature is cooperative only when every plan in ``all_plans`` (the
    caller process group) reports it exactly once with identical destination
    metadata and nbytes >= min_tensor_bytes. Partial overlaps stay native.
    """
    world_size = len(all_plans)
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

    candidates: list[tuple[int, _ReadSignature, list[tuple[int, ReadItem, _TensorMetadata]]]] = []
    for signature, occurrences in requests.items():
        plan_ranks = [rank for rank, _, _ in occurrences]
        metadata = {value for _, _, value in occurrences}
        # Full caller process-group membership only: every plan rank once,
        # identical destination metadata. Partial overlaps stay native.
        if (
            len(occurrences) != world_size
            or len(set(plan_ranks)) != world_size
            or len(metadata) != 1
        ):
            continue
        nbytes = next(iter(metadata))[3]
        if nbytes < min_tensor_bytes:
            continue
        candidates.append((nbytes, signature, occurrences))

    assigned_bytes = [0] * world_size
    tasks: list[_BroadcastTask] = []
    drops: list[set[_ReadSignature]] = [set() for _ in all_plans]
    for nbytes, signature, occurrences in sorted(
        candidates, key=lambda value: (-value[0], value[1])
    ):
        leader = min(range(world_size), key=lambda rank: (assigned_bytes[rank], rank))
        assigned_bytes[leader] += nbytes
        tasks.append(
            _BroadcastTask(
                read_item=occurrences[0][1],
                src=global_ranks[leader],
            )
        )
        for plan_rank in range(world_size):
            if plan_rank != leader:
                drops[plan_rank].add(signature)

    schedule = tuple(sorted(tasks, key=lambda task: _read_signature(task.read_item)))
    result = []
    for plan_rank, plan in enumerate(all_plans):
        items = [
            item
            for item in plan.items
            if _read_signature(item) not in drops[plan_rank]
        ]
        result.append(dataclasses.replace(plan, items=items, planner_data=schedule))
    return result


class CooperativeLoadPlanner(DefaultLoadPlanner):
    """Deduplicate identical tensor reads and broadcast them to peer ranks.

    First-version MVP: a tensor is cooperative only when every rank in the
    caller process group reports an identical read request and destination
    tensor metadata. All broadcasts use that single process group; partial
    overlaps retain default DCP reads. The planner and dcp.load must receive
    the same process group.
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
        self._schedule = cast(tuple[_BroadcastTask, ...], plan.planner_data)
        return plan

    def finish_load(self) -> None:
        if not self._schedule:
            return

        group = self.process_group or dist.group.WORLD
        for task in self._schedule:
            tensor = self.resolve_tensor(task.read_item)
            buffer = tensor if tensor.is_contiguous() else tensor.contiguous()
            dist.broadcast(buffer, src=task.src, group=group)
            if buffer is not tensor:
                tensor.copy_(buffer)
