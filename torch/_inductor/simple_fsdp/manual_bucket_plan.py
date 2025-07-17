import re
from collections import defaultdict
from typing import Any

import torch

from .. import scheduler
from ..utils import is_collective
from .bucket_utils import get_fx_node


def get_module_name(nn_module_stack: Any) -> str:
    module = list(nn_module_stack.values())[-1][0]
    module = re.findall(r"_modules\['([^']+)'\]", module)
    module = [m for m in module if not m.startswith("_")]
    module = ".".join(module)
    return module


def get_full_plan(bucketing_plan):
    full_plan = []
    for module_name in bucketing_plan:
        match = re.search(r"\[(\d+)-(\d+)\]", module_name)
        if not match:
            full_plan.append(module_name)
        else:
            start, end = map(int, match.groups())
            prefix = module_name[: match.start()]
            suffix = module_name[match.end() :]
            full_plan.extend([f"{prefix}{i}{suffix}" for i in range(start, end + 1)])
    return full_plan


def get_all_gather_plan(
    sched: "scheduler.Scheduler",
    snodes: list["scheduler.BaseSchedulerNode"],
    bucketing_plan: list[str],
) -> list[list["scheduler.BaseSchedulerNode"]]:
    all_gather_plan = []
    bucketing_plan = get_full_plan(bucketing_plan)
    plan_name_to_nodes = defaultdict(list)
    for snode in snodes:
        if is_collective(
            snode.node, op=torch.ops._c10d_functional.all_gather_into_tensor.default
        ):
            fx_node = get_fx_node(
                snode.node,
                expected_op=torch.ops._c10d_functional.all_gather_into_tensor.default,
            )
            nn_module_stack = fx_node.meta.get("nn_module_stack", {})
            module_name = get_module_name(nn_module_stack)
            matching_buckets = [p for p in bucketing_plan if module_name.startswith(p)]
            bucket_name = max(matching_buckets, key=len)
            plan_name_to_nodes[bucket_name].append(snode)

    for _, nodes in plan_name_to_nodes.items():
        all_gather_plan.append(nodes)
    return all_gather_plan


def get_reduce_scatter_plan(
    sched: "scheduler.Scheduler",
    snodes: list["scheduler.BaseSchedulerNode"],
    bucketing_plan: list[str],
) -> list[list["scheduler.BaseSchedulerNode"]]:
    reduce_scatter_plan = []
    bucketing_plan = get_full_plan(bucketing_plan)

    plan_name_to_nodes = defaultdict(list)
    for snode in snodes:
        if is_collective(
            snode.node, op=torch.ops._c10d_functional.reduce_scatter_tensor.default
        ):
            fx_node = get_fx_node(
                snode.node,
                expected_op=torch.ops._c10d_functional.reduce_scatter_tensor.default,
            )
            nn_module_stack = fx_node.meta.get("fwd_nn_module_stack", {})
            module_name = get_module_name(nn_module_stack)
            matching_buckets = [p for p in bucketing_plan if module_name.startswith(p)]
            bucket_name = max(matching_buckets, key=len)
            plan_name_to_nodes[bucket_name].append(snode)

    for _, nodes in plan_name_to_nodes.items():
        reduce_scatter_plan.append(nodes)
    return reduce_scatter_plan
