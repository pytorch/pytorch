# Copyright (c) Meta Platforms, Inc. and affiliates
import dataclasses
from collections import defaultdict
from typing import TYPE_CHECKING

from torch.distributed.checkpoint.planner import SavePlan, WriteItem


if TYPE_CHECKING:
    from torch.distributed.checkpoint.metadata import MetadataIndex

__all__ = ["dedup_save_plans"]


def dedup_save_plans(
    all_plans: list[SavePlan],
    save_to_lowest_rank: bool = False,
) -> list[SavePlan]:
    """
    Removes duplicate entries from appearing on multiple SavePlans. For each duplicate across
    a set of SavePlans, only the smallest SavePlan in terms of planned storage keeps the entry.

    Please note that this function does not modify the original SavePlans, but rather returns
    """

    # Map to query the plan indices that a write item is duplicated in
    write_item_to_plan_indices: dict[MetadataIndex, set[int]] = defaultdict(set)
    # Map to query the write item from its index
    write_item_idx_to_write_item: dict[MetadataIndex, WriteItem] = {}
    # Set of write item indices that are present in each plan
    # After deduplication, this will be the set of write item indices that are present in the final plans
    plan_to_item_indices: list[set[MetadataIndex]] = [
        {item.index for item in plan.items} for plan in all_plans
    ]

    for plan_idx, plan in enumerate(all_plans):
        for write_item in plan.items:
            # map each write item to its plan
            write_item_to_plan_indices[write_item.index].add(plan_idx)
            write_item_idx_to_write_item[write_item.index] = write_item
    plan_to_size = [0] * len(all_plans)
    for write_item_idx, plan_indices in write_item_to_plan_indices.items():
        if save_to_lowest_rank:
            select_plan_idx = min(plan_indices)
        else:
            select_plan_idx = min(
                plan_indices, key=lambda plan_idx: plan_to_size[plan_idx]
            )

        write_item = write_item_idx_to_write_item[write_item_idx]
        # Ignore the storage size of anything that is not a tensor, since
        # we don't know how much storage they represent
        plan_to_size[select_plan_idx] += write_item.tensor_storage_size() or 1
        for plan_idx in plan_indices - {select_plan_idx}:
            plan_to_item_indices[plan_idx].discard(write_item_idx)
    # Sanity check
    assert len(all_plans) == len(plan_to_item_indices)
    # Create new plans with the updated write items post deduplication
    return [
        dataclasses.replace(
            plan, items=[item for item in plan.items if item.index in item_indexes]
        )
        for plan, item_indexes in zip(all_plans, plan_to_item_indices)
    ]
