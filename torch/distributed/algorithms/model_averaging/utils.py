# flake8: noqa C101
import itertools
from typing import Iterator, List, Union

import torch
import torch.distributed as dist
import types

def average_parameters(
    params: Union[Iterator[torch.nn.Parameter], List[dict]] , process_group: dist.ProcessGroup, comm_memory_efficient: bool=False
):
    """
    Averages all the given parameters.
    For allreduce efficiency, all the parameters are flattened into a contiguous buffer.
    Thus, it requires extra memory of the same size as the given parameters.
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    # Do not update any parameter if not in the process group.
    if dist._rank_not_in_group(group_to_use):
        return

    if isinstance(params, types.GeneratorType):
        # compatible with model.parameters() input
        flat_params_all_reduce(params, group_to_use)
    elif isinstance(params, list):
        param_groups = params
        # support optim.param_group input
        if not comm_memory_efficient:
            params_list = get_param_list(param_groups)
            flat_params_all_reduce(params_list, group_to_use)
        else:
            for param_group in param_groups:
                for param in param_group["params"]:
                    if param.grad is None:
                        continue
                    flat_params_all_reduce([param], group_to_use)


def flat_params_all_reduce(params: Union[types.GeneratorType, List], process_group: dist.ProcessGroup):
    params_it1, params_it2 = itertools.tee(params)
    # If the input parameters have different data types,
    # packing these parameters will trigger an implicit type up-casting.
    # The original parameter data types will be restored during the subsequent unpacking.
    flat_params = torch.cat([p.data.reshape(-1) for p in params_it1])
    flat_params /= dist.get_world_size(process_group)
    # Make sure the allreduce will not conflict with any other ongoing process group.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dist.all_reduce(flat_params, group=process_group)
    offset = 0
    for p in params_it2:
        p.data = flat_params[offset: offset + p.numel()].view_as(p).type_as(p)
        offset += p.numel()


def get_param_list(param_groups: List[dict]):
    params_list = []
    for param_group in param_groups:
        for params in param_group["params"]:
            if params.grad is None:
                continue
            params_list.append(params)
    return params_list
