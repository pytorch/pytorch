import logging

import torch
import torch.distributed as dist


def _init_per_machine_process_groups(
    num_gpus_per_machine: int,
    process_group: dist.ProcessGroup,
    backend: str = "nccl",
) -> dist.ProcessGroup:
    """
    Initalizes all the per-machine process groups, even if only one group will be eventually used.
    This is a requirement of :meth:`torch.distributed.new_group` API.
    Returns the per-machine process group that contains the current rank.
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    ret_group = None
    for machine_rank in range(
        dist.get_world_size(group=group_to_use) // num_gpus_per_machine
    ):
        global_rank = dist.get_rank(group=group_to_use)
        start_rank = machine_rank * num_gpus_per_machine
        end_rank = start_rank + num_gpus_per_machine
        per_machine_ranks = list(range(start_rank, end_rank))
        per_machine_process_group = dist.new_group(per_machine_ranks, backend=backend)
        if global_rank >= start_rank and global_rank < end_rank:
            ret_group = per_machine_process_group
            logging.info(
                "Global rank {} is assigned to per-machine process group {}".format(
                    global_rank, per_machine_ranks
                )
            )

    return ret_group


def _average_parameters(
    model: torch.nn.Module, process_group: dist.ProcessGroup = dist.group.WORLD
):
    """
    Averages all the parameters of a given model.
    For allreduce efficiency, all the parameters are flattened into a contiguous buffer.
    Thus, it requires extra memory of the same size as the model’s parameters.
    """
    flat_params = torch.cat([p.data.view(-1) for p in model.parameters()])
    dist.all_reduce(flat_params, group=process_group)
    flat_params /= float(dist.get_world_size(group=process_group))

    offset = 0
    for p in model.parameters():
        p.data = flat_params[offset : offset + p.numel()].view_as(p)
        offset += p.numel()
