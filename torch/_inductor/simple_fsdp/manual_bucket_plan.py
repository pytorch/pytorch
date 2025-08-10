import re
from collections import defaultdict
from typing import Any, Callable

from .. import scheduler
from ..utils import is_collective
from .bucket_utils import get_fx_node


def get_module_name(nn_module_stack: Any) -> str:
    module = list(nn_module_stack.values())[-1][0]
    module = re.findall(r"_modules\['([^']+)'\]", module)
    module = [m for m in module if not m.startswith("_")]
    module = ".".join(module)
    return module


def max_overlapping_start_length(p, module_name):
    max_overlap_length = 0
    min_length = min(len(p), len(module_name))

    for i in range(min_length):
        if p[i] == module_name[i]:
            max_overlap_length += 1
        else:
            break
    return max_overlap_length


def get_full_plan(bucketing_plan):
    full_plan = []
    for module_name in bucketing_plan:
        if "+" in module_name:
            full_plan.append(module_name.split("+"))
            continue
        match = re.search(r"\[(\d+)-(\d+)\]", module_name)
        if not match:
            full_plan.append(module_name)
        else:
            start, end = map(int, match.groups())
            prefix = module_name[: match.start()]
            suffix = module_name[match.end() :]
            full_plan.extend([f"{prefix}{i}{suffix}" for i in range(start, end + 1)])
    return full_plan


def get_manual_plan(
    sched: "scheduler.Scheduler",
    snodes: list["scheduler.BaseSchedulerNode"],
    bucketing_plan: list[str],
    comm_func: Callable[..., Any],
    module_stack_type: str,
) -> list[list["scheduler.BaseSchedulerNode"]]:
    manual_plan = []
    bucketing_plan = get_full_plan(bucketing_plan)
    plan_name_to_nodes = defaultdict(list)
    for snode in snodes:
        if is_collective(snode.node, op=comm_func):
            fx_node = get_fx_node(
                snode.node,
                expected_op=comm_func,
            )
            nn_module_stack = fx_node.meta.get(module_stack_type, {})
            module_name = get_module_name(nn_module_stack)
            matching_buckets = defaultdict(int)
            for p in bucketing_plan:
                if isinstance(p, list):
                    module_join_name = "_".join(p)
                    for name in p:
                        if module_name.startswith(name):
                            matching_buckets[module_join_name] = max(
                                matching_buckets[module_join_name],
                                max_overlapping_start_length(name, module_name),
                            )
                else:
                    if module_name.startswith(p):
                        matching_buckets[p] = max_overlapping_start_length(
                            p, module_name
                        )

            bucket_name = max(matching_buckets, key=matching_buckets.get)
            plan_name_to_nodes[bucket_name].append(snode)

    for _, nodes in plan_name_to_nodes.items():
        manual_plan.append(nodes)
    return manual_plan
