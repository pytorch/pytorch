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
    """

    write_item_to_plan_indices: dict[MetadataIndex, set[int]] = defaultdict(set)
    write_item_idx_to_write_item: dict[MetadataIndex, WriteItem] = {}
    for plan_idx, plan in enumerate(all_plans):
        for write_item in plan.items:
            # map each write item to its plan
            write_item_to_plan_indices[write_item.index].add(plan_idx)
            write_item_idx_to_write_item[write_item.index] = write_item

    # put item in the plan with the smallest size and remove it from the other plan_indices
    to_remove: list[set] = [set() for _ in range(len(all_plans))]
    plan_to_size = [0] * len(all_plans)
    for write_item_idx, plan_indices in write_item_to_plan_indices.items():
        if save_to_lowest_rank:
            select_plan_idx = min(plan_indices)
        else:
            select_plan_idx = min(
                plan_indices, key=lambda plan_idx: plan_to_size[plan_idx]
            )

        write_item = write_item_idx_to_write_item[write_item_idx]
        # essentially ignores the storage size of anything that is not a tensor, since
        # we don't know how much storage they represent
        plan_to_size[select_plan_idx] += write_item.tensor_storage_size() or 1

        plan_indices.remove(select_plan_idx)
        for plan_idx in plan_indices:
            to_remove[plan_idx].add(write_item_idx)

    for plan_idx, remove_set in enumerate(to_remove):
        new_items = [
            write_item
            for write_item in all_plans[plan_idx].items
            if write_item.index not in remove_set
        ]
        all_plans[plan_idx] = dataclasses.replace(all_plans[plan_idx], items=new_items)

    return all_plans
