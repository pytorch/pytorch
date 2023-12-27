# Copyright (c) Meta Platforms, Inc. and affiliates
from collections import defaultdict
from functools import reduce
import dataclasses
import logging
from typing import Dict, List

import torch
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed.checkpoint.planner import SavePlan

__all__ = ["dedup_tensors", "load_balance_tensors"]


def init_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    level = logging.INFO
    logger.setLevel(level)
    console = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s:%(lineno)s %(levelname)s p:%(processName)s t:%(threadName)s: %(message)s"
    )
    console.setFormatter(formatter)
    console.setLevel(level)
    logger.addHandler(console)
    logger.propagate = False
    return logger


logger = init_logger()


# TODO add docstring for dedup_tensors
def dedup_tensors(all_plans: List[SavePlan]) -> List[SavePlan]:
    all_plans = list(all_plans)
    key_to_plan: Dict[MetadataIndex, List[int]] = {}
    for plan_idx, plan in enumerate(all_plans):
        for write_item in plan.items:
            key_to_plan.setdefault(write_item.index, []).append(plan_idx)

    replicated_items = {k: v for k, v in key_to_plan.items() if len(v) > 1}

    # Remove duplicates by always keeping the first entry.
    # Compute the per-rank remove set.
    plan_to_keys: Dict[int, List[MetadataIndex]] = {}
    for key, plans in replicated_items.items():
        for plan_idx in plans[1:]:
            plan_to_keys.setdefault(plan_idx, []).append(key)
    logger.info("Duplicate keys to remove: %s", plan_to_keys)

    for plan_idx, keys in plan_to_keys.items():
        key_set = set(keys)
        # rewrite items and remove elements
        new_items = [
            write_item
            for write_item in all_plans[plan_idx].items
            if write_item.index not in key_set
        ]
        all_plans[plan_idx] = dataclasses.replace(all_plans[plan_idx], items=new_items)

    return all_plans

def load_balance_tensors(all_plans: List[SavePlan]) -> List[SavePlan]:
    """
    Load balance tensors across ranks.
    """

    def _size(write_item):
        if write_item.tensor_data is None:
            return 0

        elem_size = torch.ones(1, dtype=write_item.tensor_data.properties.dtype).element_size()
        num_elems = reduce(lambda x, y: x * y, write_item.tensor_data.size)
        return elem_size * num_elems

    write_item_to_plans: Dict[MetadataIndex, List[int]] = defaultdict(set)
    plan_to_size = [0] * len(all_plans)
    for plan_idx, plan in enumerate(all_plans):
        for write_item in plan.items:
            # map each write item to it's plan
            write_item_to_plans[write_item.index].add(plan_idx)
            # compute total size of each plan (does not account for non-tensor objects)
            plan_to_size[plan_idx] += _size(write_item)

    # TODO: optimize, or round robin, etc.
    # put item in the plan with the smallest size and remove it from the other plans
    to_remove = [set()] * len(all_plans)
    for write_item_idx, plans in write_item_to_plans.items():
        selected_plan = min(plans, key=lambda plan_idx: plan_to_size[plan_idx])

        plans.remove(selected_plan)
        for plan_idx in plans:
            to_remove[plan_idx].add(write_item_idx)
            plan_to_size[plan_idx] -= _size(write_item)

    for plan_idx, remove_set in enumerate(to_remove):
        new_items = [
            write_item
            for write_item in all_plans[plan_idx].items
            if write_item.index not in remove_set
        ]
        all_plans[plan_idx] = dataclasses.replace(all_plans[plan_idx], items=new_items)

    return all_plans
