"""Backward-compatible re-exports from the shared heuristics registry."""

from torch._inductor.heuristics.registry import (
    _HEURISTIC_CACHE,
    _TEMPLATE_HEURISTIC_REGISTRY,
    clear_registry,
    get_registered_heuristic_class,
    get_template_heuristic,
    override_template_heuristics,
    register_template_heuristic,
)

from .base import TemplateConfigHeuristics


__all__ = [
    "TemplateConfigHeuristics",
    "_HEURISTIC_CACHE",
    "_TEMPLATE_HEURISTIC_REGISTRY",
    "clear_registry",
    "get_registered_heuristic_class",
    "get_template_heuristic",
    "override_template_heuristics",
    "register_template_heuristic",
]
