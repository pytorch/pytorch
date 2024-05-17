# Copyright (c) Meta Platforms, Inc. and affiliates
import dataclasses
from collections import defaultdict
from typing import Dict, List, Set, TYPE_CHECKING

from torch.distributed.checkpoint.planner import SavePlan, WriteItem

if TYPE_CHECKING:
    from torch.distributed.checkpoint.metadata import MetadataIndex

__all__ = ["dedup_save_plans"]


def dedup_save_plans(all_plans: List[SavePlan]) -> List[SavePlan]:
    """
    Removes duplicate entries from appearing on multiple SavePlans. For each duplicate across
    a set of SavePlans, it's saved to the lowest rank to reduce the number of storage files needed
    when loading the state_dict.
    """

    write_item_to_plan_indices: Dict[MetadataIndex, Set[int]] = defaultdict(set)
    for plan_idx, plan in enumerate(all_plans):
        for write_item in plan.items:
            # map each write item to its plan
            write_item_to_plan_indices[write_item.index].add(plan_idx)

    # put item in the plan with the lowest rank and remove it from the other plan_indices
    to_remove: List[Set] = [set() for _ in range(len(all_plans))]
    for write_item_idx, plan_indices in write_item_to_plan_indices.items():
        select_plan_idx = min(plan_indices, key=lambda plan_idx: plan_idx)

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
