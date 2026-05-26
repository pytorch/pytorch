from __future__ import annotations

from typing_extensions import Self

import torch
import torch.fx.traceback as fx_traceback


__all__ = [
    "MemoryBudgetMode",
    "set_memory_budget",
    "propagate_memory_budgets_from_markers",
]


def _validate_budget(budget: float) -> float:
    if not isinstance(budget, (int, float)):
        raise TypeError(f"budget must be a float, got {type(budget)}")
    if not (0.0 <= budget <= 1.0):
        raise ValueError(f"budget must be between 0 and 1, got {budget}")
    return float(budget)


def set_memory_budget(budget: float) -> None:
    """
    Set the memory budget for activation checkpointing on subsequent FX nodes.

    The memory budget controls the trade-off between memory usage and
    recomputation during the backward pass:
    - budget=0.0: Aggressive recomputation, minimal memory (save almost nothing)
    - budget=1.0: No recomputation, maximum memory (save everything)

    Args:
        budget: Float between 0 and 1 controlling memory/recompute trade-off.
    """
    budget = _validate_budget(budget)
    fx_traceback.current_meta["memory_budget"] = budget


# allow_in_graph makes Dynamo emit set_memory_budget itself as an opaque
# call_function node rather than tracing into its body. Without this, the
# fx_traceback.current_meta dict mutation above would force a graph break,
# and the marker would never reach the FX graph. The body still runs in
# eager mode (e.g. plain Python or torch.fx.symbolic_trace), so the dict
# update preserves preserve_node_meta()-based propagation for non-Dynamo
# tracers. Under torch.compile, propagate_memory_budgets_from_markers
# converts each marker node into node.meta["memory_budget"] on the
# subsequent op nodes and erases the markers from the graph.
torch._dynamo.allow_in_graph(set_memory_budget)


class MemoryBudgetMode:
    """
    Context manager that sets memory_budget metadata on FX nodes during
    PT2 compilation for activation checkpointing control.

    The memory budget controls the trade-off between memory usage and
    recomputation during the backward pass:
    - budget=0.0: Aggressive recomputation, minimal memory
    - budget=1.0: No recomputation, maximum memory

    Example::

        with MemoryBudgetMode(0.3):
            x = self.encoder(x)   # nodes get budget=0.3
        with MemoryBudgetMode(0.8):
            x = self.head(x)      # nodes get budget=0.8

    Limitation: under torch.compile, nested MemoryBudgetMode does not restore
    the outer budget on exit -- the inner budget persists for ops emitted after
    the inner ``__exit__`` until the next ``set_memory_budget`` /
    ``MemoryBudgetMode``. Eager mode is unaffected.
    """

    def __init__(self, budget: float) -> None:
        self.budget = _validate_budget(budget)
        self._prev_budget: float | None = None

    def __enter__(self) -> Self:
        # Under torch.compile, fx_traceback.current_meta dict ops would force
        # a Dynamo graph break (module-level mutable dict), so skip them and
        # rely on the set_memory_budget marker + post-pass for propagation.
        # In eager mode (including plain Python and fx.symbolic_trace), keep
        # the dict-based behavior so preserve_node_meta()-based tracers still
        # see the budget and so __exit__ can restore the previous value.
        if not torch.compiler.is_compiling():
            self._prev_budget = fx_traceback.current_meta.get("memory_budget", None)
        set_memory_budget(self.budget)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        if torch.compiler.is_compiling():
            return
        if self._prev_budget is not None:
            fx_traceback.current_meta["memory_budget"] = self._prev_budget
        else:
            fx_traceback.current_meta.pop("memory_budget", None)

    def __repr__(self) -> str:
        return f"MemoryBudgetMode(budget={self.budget})"


def propagate_memory_budgets_from_markers(gm: torch.fx.GraphModule) -> bool:
    """
    Convert ``set_memory_budget`` marker nodes into ``node.meta["memory_budget"]``
    annotations on the subsequent op nodes and remove the markers from the graph.

    Returns True if any marker was found (and the graph was mutated), False
    otherwise. Safe to call on graphs that contain no markers (no-op then).
    """
    current_budget: float | None = None
    any_marker = False
    nodes_to_erase: list[torch.fx.Node] = []
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target is set_memory_budget:
            any_marker = True
            if node.args:
                arg0 = node.args[0]
                if isinstance(arg0, (int, float)):
                    current_budget = float(arg0)
            nodes_to_erase.append(node)
            continue
        if node.op in ("placeholder", "output", "get_attr"):
            continue
        if current_budget is not None:
            node.meta["memory_budget"] = current_budget
    if not any_marker:
        return False
    for node in nodes_to_erase:
        gm.graph.erase_node(node)
    gm.graph.lint()
    gm.recompile()
    return True
