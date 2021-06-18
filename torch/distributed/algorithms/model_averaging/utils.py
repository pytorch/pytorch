import torch
import torch.distributed as dist


def average_parameters(
    model: torch.nn.Module, process_group: dist.ProcessGroup = dist.group.WORLD
):
    """
    Averages all the parameters of a given model.
    For allreduce efficiency, all the parameters are flattened into a contiguous buffer.
    Thus, it requires extra memory of the same size as the model’s parameters.
    """
    flat_params = torch.cat([p.data.view(-1) for p in model.parameters()])
    flat_params /= float(dist.get_world_size(group=process_group))
    dist.all_reduce(flat_params, group=process_group)

    offset = 0
    for p in model.parameters():
        p.data = flat_params[offset : offset + p.numel()].view_as(p)
        offset += p.numel()
