#!/usr/bin/env python3

"""Utility functions for TorchBench benchmark suite."""

import os
import sys
import torch
from os.path import abspath, exists


def _reassign_parameters(model):
    """Reassign tensors to parameters for torch_geometric models.
    
    torch_geometric models register parameter as tensors due to
    https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/dense/linear.py#L158-L168
    Since it is unusual thing to do, we just reassign them to parameters
    """
    def state_dict_hook(module, destination, prefix, local_metadata):
        for name, param in module.named_parameters():
            if isinstance(destination[name], torch.Tensor) and not isinstance(
                destination[name], torch.nn.Parameter
            ):
                destination[name] = torch.nn.Parameter(destination[name])

    model._register_state_dict_hook(state_dict_hook)


def setup_torchbench_cwd():
    """Setup TorchBench working directory and environment.
    
    Returns:
        str: Original directory path before changing to TorchBench directory
    """
    original_dir = abspath(os.getcwd())

    os.environ["KALDI_ROOT"] = "/tmp"  # avoids some spam
    for torchbench_dir in (
        "./torchbenchmark",
        "../torchbenchmark",
        "../torchbench",
        "../benchmark",
        "../../torchbenchmark",
        "../../torchbench",
        "../../benchmark",
        "../../../torchbenchmark",
        "../../../torchbench",
        "../../../benchmark",
    ):
        if exists(torchbench_dir):
            break

    if exists(torchbench_dir):
        torchbench_dir = abspath(torchbench_dir)
        os.chdir(torchbench_dir)
        sys.path.append(torchbench_dir)

    return original_dir