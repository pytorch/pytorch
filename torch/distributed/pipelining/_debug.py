# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import os

import torch


# PIPPY_VERBOSITY is an environment variable that controls the logging level.
# It can be set to one of the following:
#   - WARNING (default)
#   - INFO
#   - DEBUG
PIPPY_VERBOSITY = os.getenv("PIPPY_VERBOSITY", "WARNING")
if PIPPY_VERBOSITY not in ["WARNING", "INFO", "DEBUG"]:
    print(f"Unsupported PIPPY_VERBOSITY level: {PIPPY_VERBOSITY}")
    PIPPY_VERBOSITY = "WARNING"

logging.getLogger("pippy").setLevel(PIPPY_VERBOSITY)
# It seems we need to print something to make the level setting effective
# for child loggers. Doing it here.
print(f"Setting PiPPy logging level to: {PIPPY_VERBOSITY}")


def friendly_debug_info(v):
    """
    Helper function to print out debug info in a friendly way.
    """
    if isinstance(v, torch.Tensor):
        return f"Tensor({v.shape}, grad={v.requires_grad})"
    else:
        return str(v)


def map_debug_info(a):
    """
    Helper function to apply `friendly_debug_info` to items in `a`.
    `a` may be a list, tuple, or dict.
    """
    return torch.fx.node.map_aggregate(a, friendly_debug_info)
