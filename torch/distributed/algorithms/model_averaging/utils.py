import torch
import torch.distributed as dist


def average_parameters(model: torch.nn.Module, process_group: dist.ProcessGroup):
    """
    Averages all the parameters of a given model.
    For allreduce efficiency, all the parameters are flattened into a contiguous buffer.
    Thus, it requires extra memory of the same size as the model’s parameters.
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    flat_params = torch.cat([p.data.view(-1) for p in model.parameters()])
    flat_params /= dist.get_world_size(group_to_use)
    dist.all_reduce(flat_params, group=group_to_use)

    offset = 0
    for p in model.parameters():
        p.data = flat_params[offset : offset + p.numel()].view_as(p)
        offset += p.numel()
