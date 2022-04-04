import torch._C._lazy

def reset():
    """Resets all metric counters."""
    torch._C._lazy._reset_metrics()

def counter_names():
    """Retrieves all the currently active counter names."""
    return torch._C._lazy._counter_names()
