"""Backward-compatible re-exports from the shared heuristics registry."""

from torch._inductor.heuristics.registry import (
    clear_registry,
    get_registered_heuristic_class,
    get_template_heuristic,
    override_template_heuristics,
    register_template_heuristic,
)


__all__ = [
    "clear_registry",
    "get_registered_heuristic_class",
    "get_template_heuristic",
    "override_template_heuristics",
    "register_template_heuristic",
]
