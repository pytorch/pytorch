from torch import nn
from typing import List, Optional

__all__ = ["partition_model"]

def partition_model(
        module: nn.Sequential,
        balance: List[int],
        devices: Optional[List[int]] = None):
    """
    Given an :class:`nn.Sequential <torch.nn.Sequential>` module, partitions
    the model across multiple GPU devices according the provided ``balance``
    and ``devices``.

    Args:
        module (:class:`nn.Sequential <torch.nn.Sequential>`):
            Sequential model representing the pipe.
        balance (List[int]):
            List indicating the number of layers in each partition.
        devices (List[int], optional):
            List indicating the device to use for each partition. Defaults to
            ``range(len(balance))``
    """
    device_idx = 0
    pipe_idx = 0
    balanced_pipe = []
    for num_layers in balance:
        layers = []
        for i in range(num_layers):
            layers.append(module[pipe_idx])
            pipe_idx += 1
        device = device_idx if devices is None else devices[device_idx]
        balanced_pipe.append(nn.Sequential(*layers).to(device))
        device_idx += 1

    return nn.Sequential(*balanced_pipe)
