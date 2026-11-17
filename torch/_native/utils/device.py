# Validate the current device before launching a cached kernel.

import torch


def check_current(index: int) -> None:
    # Compilation must use the device recorded in the plan's cache key.
    current = torch.cuda.current_device()
    if index != current:
        raise AssertionError(f"input on cuda:{index}, but cuda:{current} is current")
