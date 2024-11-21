# mypy: allow-untyped-defs
# flake8: noqa C101
import itertools
from typing import Dict, Iterable, Iterator, Union

import torch
import torch.distributed as dist

# The two imports below are not always available depending on the
# USE_DISTRIBUTED compile flag. Make sure they raise import error
# if we're trying to use them.
from torch.distributed import group, ProcessGroup


__all__ = [
    "average_parameters",
    "get_params_to_average",
    "average_parameters_or_parameter_groups",
]


def average_parameters(
    params: Iterator[torch.nn.Parameter], process_group: ProcessGroup
):
    """
    Averages all the given parameters.

    For allreduce efficiency, all the parameters are flattened into a contiguous buffer.
    Thus, it requires extra memory of the same size as the given parameters.
    """
    group_to_use = process_group if process_group is not None else group.WORLD
    # Do not update any parameter if not in the process group.
    if dist._rank_not_in_group(group_to_use):
        return

    params_it1, params_it2 = itertools.tee(params)
    # If the input parameters have different data types,
    # packing these parameters will trigger an implicit type up-casting.
    # The original parameter data types will be restored during the subsequent unpacking.
    flat_params = torch.cat([p.data.reshape(-1) for p in params_it1])
    flat_params /= dist.get_world_size(group_to_use)
    # Make sure the allreduce will not conflict with any other ongoing process group.
    if torch.accelerator.is_available():
        torch.accelerator.synchronize()
    dist.all_reduce(flat_params, group=group_to_use)

    offset = 0
    for p in params_it2:
        p.data = flat_params[offset : offset + p.numel()].view_as(p).type_as(p)
        offset += p.numel()


def get_params_to_average(
    params: Union[Iterable[torch.nn.Parameter], Iterable[Dict[str, torch.nn.Parameter]]]
):
    """
    Return a list of parameters that need to average.

    This filters out the parameters that do not contain any gradients.
    Args:
        params: The parameters of a model or parameter groups of an optimizer.
    """
    filtered_params = []
    for param in params:
        if isinstance(param, torch.nn.Parameter):
            # model.parameters() input
            param_data = param
            if param_data.grad is not None:
                filtered_params.append(param_data)
        elif isinstance(param, dict):
            # optimizer.param_groups input
            for param_data in param["params"]:
                if param_data.grad is not None:
                    filtered_params.append(param_data)
        else:
            raise NotImplementedError(
                f"Parameter input of type {type(param)} is not supported"
            )
    return filtered_params


def average_parameters_or_parameter_groups(
    params: Union[
        Iterable[torch.nn.Parameter], Iterable[Dict[str, torch.nn.Parameter]]
    ],
    process_group: ProcessGroup,
):
    """Averages parameters of a model or parameter groups of an optimizer."""
    average_parameters(iter(get_params_to_average(params)), process_group)
