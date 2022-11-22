# Copyright (c) Meta Platforms, Inc. and affiliates

from typing import Dict, List
import dataclasses

from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed.checkpoint.planner import SavePlan

__all__ = ["dedup_tensors"]

# TODO add docstring for dedup_tensors
def dedup_tensors(all_plans: List[SavePlan]) -> List[SavePlan]:
    all_plans = list(all_plans)
    key_to_plan: Dict[MetadataIndex, List[int]] = {}
    for plan_idx, plan in enumerate(all_plans):
        for wi in plan.items:
            key_to_plan.setdefault(wi.index, []).append(plan_idx)

    replicated_items = {k: v for k, v in key_to_plan.items() if len(v) > 1}

    # Remove deplicates by always keeping the first entry.
    # Compute the per-rank remove set.
    plan_to_keys: Dict[int, List[MetadataIndex]] = {}
    for key, plans in replicated_items.items():
        for plan_idx in plans[1:]:
            plan_to_keys.setdefault(plan_idx, []).append(key)

    for plan_idx, keys in plan_to_keys.items():
        key_set = set(keys)
        # rewrite items and remove elements
        new_items = [
            wi for wi in all_plans[plan_idx].items if wi.index not in key_set
        ]
        all_plans[plan_idx] = dataclasses.replace(
            all_plans[plan_idx], items=new_items
        )

    return all_plans
