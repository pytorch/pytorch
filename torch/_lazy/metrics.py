import torch._C._lazy


def reset():
    """Resets all metric counters."""
    torch._C._lazy._reset_metrics()


def counter_names():
    """Retrieves all the currently active counter names."""
    return torch._C._lazy._counter_names()


def counter_value(name: str):
    """Return the value of the counter with the speficied name"""
    return torch._C._lazy._counter_value(name)


def metrics_report():
    """Return the combined (lazy core and backend) metric report"""
    return torch._C._lazy._metrics_report()
