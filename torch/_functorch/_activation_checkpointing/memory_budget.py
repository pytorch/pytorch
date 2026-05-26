import torch.fx.traceback as fx_traceback


def _validate_budget(budget: object) -> float:
    if not isinstance(budget, (int, float)):
        raise TypeError(
            f"memory_budget must be a float between 0 and 1, got {type(budget).__name__}"
        )
    budget = float(budget)
    if not (0.0 <= budget <= 1.0):
        raise ValueError(
            f"memory_budget must be between 0 and 1, got {budget}"
        )
    return budget


def memory_budget(budget: float):
    """Context manager that sets the activation memory budget for a region.

    Controls the recomputation vs. memory trade-off in the min-cut
    partitioner when used under ``torch.compile``. A budget of 0 means
    "recompute everything" (minimum memory), while 1 means "save
    everything" (maximum memory, no recomputation). Values in between
    interpolate via the knapsack solver.

    The budget is stored in ``node.meta["custom"]["memory_budget"]``
    on every FX node traced inside the ``with`` block, overriding
    ``torch._functorch.config.activation_memory_budget`` on a
    per-region basis.

    Example::

        with memory_budget(0.3):
            x = layer1(x)   # aggressive recomputation
        with memory_budget(0.8):
            x = layer2(x)   # save most activations
    """
    budget = _validate_budget(budget)
    return fx_traceback.annotate({"memory_budget": budget})
