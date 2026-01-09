# mypy: allow-untyped-defs

from typing import Literal, Optional

import torch
import torch.fx.traceback as fx_traceback
from torch.utils._python_dispatch import TorchDispatchMode


__all__ = ["MemoryBudgetMode", "set_memory_budget"]


def _ensure_memory_budget_in_copy_meta_fields() -> None:
    """
    Ensure 'memory_budget' is in _COPY_META_FIELDS for FX metadata propagation.

    This is a defensive fallback - if proxy.py already contains 'memory_budget',
    this is a no-op. Otherwise, it registers the field at runtime.
    """
    import torch.fx.proxy as fx_proxy

    if "memory_budget" not in fx_proxy._COPY_META_FIELDS:
        fx_proxy._COPY_META_FIELDS.append("memory_budget")


# Ensure memory_budget is registered for FX metadata propagation
_ensure_memory_budget_in_copy_meta_fields()


def set_memory_budget(budget: float) -> None:
    """
    Set the memory budget for activation checkpointing on subsequent FX nodes.

    This function directly sets the memory_budget metadata that the activation
    checkpointing partitioner uses to decide what to save vs recompute.

    The memory budget controls the trade-off between memory usage and
    recomputation during the backward pass:
    - budget=0.0: Aggressive recomputation, minimal memory (save almost nothing)
    - budget=1.0: No recomputation, maximum memory (save everything)
    - budget=0.5: Balanced approach

    Args:
        budget: Float between 0 and 1 controlling memory/recompute trade-off.

    Example:
        >>> # In a model's forward method:
        >>> set_memory_budget(0.1)  # Aggressive recomputation for expensive encoder
        >>> x = self.encoder(x)
        >>> set_memory_budget(0.8)  # Save most activations for cheap head
        >>> x = self.head(x)

    Note:
        This function only has an effect during torch.compile tracing.
        In eager mode, it is a no-op.
    """
    if not isinstance(budget, (int, float)):
        raise TypeError(f"budget must be a float, got {type(budget)}")
    if not (0.0 <= budget <= 1.0):
        raise ValueError(f"budget must be between 0 and 1, got {budget}")

    # Check if we're being traced by PT2
    # ProxyTorchDispatchMode is active during AOTAutograd's FX tracing
    is_fx_tracing = (
        torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY) is not None
    )

    # Also check if Dynamo is currently tracing
    # This catches the case where we're in the Dynamo graph capture phase
    is_dynamo_tracing = torch._dynamo.is_compiling()

    if is_fx_tracing or is_dynamo_tracing:
        fx_traceback.current_meta["memory_budget"] = float(budget)


class MemoryBudgetMode(TorchDispatchMode):
    """
    A TorchDispatchMode that sets memory_budget metadata on FX nodes during
    PT2 compilation for activation checkpointing control.

    The memory budget controls the trade-off between memory usage and
    recomputation during the backward pass:
    - budget=0.0: Aggressive recomputation, minimal memory (save almost nothing)
    - budget=1.0: No recomputation, maximum memory (save everything)
    - budget=0.5: Balanced approach

    Args:
        budget: Float between 0 and 1 controlling memory/recompute trade-off.
        strategy: How to handle nested MemoryBudgetMode contexts.
            - "override": Inner budget replaces outer (default)
            - "min": Use minimum budget (most aggressive recomputation wins)
            - "max": Use maximum budget (most memory-saving wins)

    Example:
        >>> # Basic usage
        >>> with MemoryBudgetMode(0.3):
        ...     output = model(input)

        >>> # Nested with min strategy
        >>> with MemoryBudgetMode(0.5):
        ...     x = layer1(x)  # budget=0.5
        ...     with MemoryBudgetMode(0.2, strategy="min"):
        ...         x = layer2(x)  # budget=min(0.5, 0.2)=0.2

    Note:
        This mode only has an effect during torch.compile tracing.
        In eager mode, it passes through operations unchanged.
    """

    def __init__(
        self,
        budget: float,
        strategy: Literal["override", "min", "max"] = "override",
    ):
        super().__init__()
        if not isinstance(budget, (int, float)):
            raise TypeError(f"budget must be a float, got {type(budget)}")
        if not (0.0 <= budget <= 1.0):
            raise ValueError(f"budget must be between 0 and 1, got {budget}")
        if strategy not in ("override", "min", "max"):
            raise ValueError(
                f"strategy must be 'override', 'min', or 'max', got {strategy}"
            )

        self.budget = float(budget)
        self.strategy = strategy
        self._prev_budget: Optional[float] = None

    def __enter__(self):
        # Save any existing budget from outer context
        self._prev_budget = fx_traceback.current_meta.get("memory_budget", None)
        # Set our effective budget so nested contexts can see it
        fx_traceback.current_meta["memory_budget"] = self._get_effective_budget()
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous budget when exiting
        if self._prev_budget is not None:
            fx_traceback.current_meta["memory_budget"] = self._prev_budget
        else:
            fx_traceback.current_meta.pop("memory_budget", None)
        return super().__exit__(exc_type, exc_val, exc_tb)

    def _get_effective_budget(self) -> float:
        """
        Compute effective budget based on nesting strategy.
        """
        if self._prev_budget is None or self.strategy == "override":
            return self.budget
        elif self.strategy == "min":
            return min(self.budget, self._prev_budget)
        elif self.strategy == "max":
            return max(self.budget, self._prev_budget)
        else:
            # Should never reach here due to __init__ validation
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def __torch_dispatch__(self, func, _types, args=(), kwargs=None):
        kwargs = kwargs if kwargs is not None else {}

        # Check if we're being traced by PT2
        # ProxyTorchDispatchMode is active during AOTAutograd's FX tracing
        is_fx_tracing = (
            torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY)
            is not None
        )

        # Also check if Dynamo is currently tracing
        is_dynamo_tracing = torch._dynamo.is_compiling()

        if is_fx_tracing or is_dynamo_tracing:
            fx_traceback.current_meta["memory_budget"] = self._get_effective_budget()

        # Pass through to next mode/kernel unchanged
        return func(*args, **kwargs)

    def __repr__(self):
        return f"MemoryBudgetMode(budget={self.budget}, strategy='{self.strategy}')"

    @classmethod
    def ignore_compile_internals(cls) -> bool:
        """Allow torch.compile to trace through this mode without graph breaks.

        Returns True so that this TorchDispatchMode does not cause graph breaks
        when used inside a compiled function. The mode's effect (setting
        memory_budget metadata on FX nodes) happens during tracing via
        __torch_dispatch__, so we don't need Dynamo to special-case this mode.
        """
        return True
