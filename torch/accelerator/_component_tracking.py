"""Public API for memory component tracking.

Provides a simple interface to track GPU memory attribution by component type
(parameter, gradient, activation, optimizer state, etc.) and optionally by
module fully-qualified name.

Usage::

    from torch.accelerator.memory import component_tracking
    import torch.accelerator.memory

    component_tracking.enable()

    # ... training loop ...

    stats = torch.accelerator.memory.memory_stats()
    print(f"Params:      {stats['component_bytes.parameter.current'] / 1e9:.2f} GB")
    print(f"Activations: {stats['component_bytes.activation.current'] / 1e9:.2f} GB")

    component_tracking.disable()

The ``enable()`` call auto-detects models and optimizers via global hooks.
No arguments are required for component-level breakdown. For per-module
breakdown (``module_bytes.*``), models are discovered automatically on
their first forward pass.
"""

from __future__ import annotations


_tracker = None


def enable() -> None:
    """Enable memory component tracking.

    Installs lightweight global hooks that automatically tag every GPU
    allocation with its component type (parameter, gradient, activation,
    optimizer state, etc.) and the owning module's fully-qualified name.

    Models and optimizers are auto-detected:
    - Models are discovered on their first forward pass (root detection).
    - Optimizers are detected via class-level step hooks.

    Calling ``enable()`` when already enabled is a no-op.

    After this call, ``torch.accelerator.memory.memory_stats()`` includes:
    - ``component_bytes.{type}.{metric}`` keys (always)
    - ``module_bytes.{fqn}.{type}.{metric}`` keys (after first forward)
    """
    global _tracker
    if _tracker is not None:
        return

    from torch.cuda.memory_component_tracker import _enable_auto

    _tracker = _enable_auto()


def disable() -> None:
    """Disable memory component tracking and remove all hooks.

    Resets all tracking state. After this call, new allocations are no
    longer tagged. Existing ``component_bytes`` counters remain in
    ``torch.accelerator.memory.memory_stats()`` but stop updating.
    """
    global _tracker
    if _tracker is None:
        return

    _tracker.disable()
    _tracker = None


def is_enabled() -> bool:
    """Return True if component tracking is currently active."""
    return _tracker is not None
