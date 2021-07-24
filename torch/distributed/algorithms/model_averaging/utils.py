# flake8: noqa C101
import torch
import torch.distributed as dist


def average_parameters(module: torch.nn.Module, process_group: dist.ProcessGroup):
    """
    Averages all the parameters of a given module.
    For allreduce efficiency, all the parameters are flattened into a contiguous buffer.
    Thus, it requires extra memory of the same size as the module's parameters.
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    # Do not update any parameter if not in the process group.
    if dist._rank_not_in_group(group_to_use):
        return

    flat_params = torch.cat([p.data.view(-1) for p in module.parameters()])
    flat_params /= dist.get_world_size(group_to_use)
    # Make sure the allreduce will not conflict with any other ongoing process group.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dist.all_reduce(flat_params, group=group_to_use)

    offset = 0
    for p in module.parameters():
        p.data = flat_params[offset : offset + p.numel()].view_as(p)
        offset += p.numel()
