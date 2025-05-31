#!/usr/bin/env python3

"""Utility functions for TorchBench benchmarks."""

import os
import sys
from os.path import abspath, exists

import torch


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
    """Set up the current working directory for TorchBench.

    Returns:
        str: The original directory path before changing to TorchBench directory.
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


def process_hf_reformer_output(out):
    """Process HuggingFace Reformer model output.

    Args:
        out: Model output list

    Returns:
        List with second (unstable) output element removed
    """
    assert isinstance(out, list)
    # second output is unstable
    return [elem for i, elem in enumerate(out) if i != 1]


def process_hf_whisper_output(out):
    """Process HuggingFace Whisper model output.

    Args:
        out: Model output

    Returns:
        Processed output with logits removed from first element
    """
    out_ret = []
    for i, elem in enumerate(out):
        if i == 0:
            if elem is not None:
                assert isinstance(elem, dict)
                out_ret.append({k: v for k, v in elem.items() if k != "logits"})
        elif i != 1:
            out_ret.append(elem)

    return out_ret


# Map of model names to their specific output processing functions
process_train_model_output = {
    "hf_Reformer": process_hf_reformer_output,
    "hf_Whisper": process_hf_whisper_output,
}
