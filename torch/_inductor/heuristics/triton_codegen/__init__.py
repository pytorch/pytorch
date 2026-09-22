"""
Codegen heuristics for inductor-generated Triton kernels.

These heuristics control autotuning config generation for runtime-compiled
kernels: pointwise, reduction, persistent_reduction, etc.

Device dispatch uses the shared registry in heuristics/registry.py
with name-based keys (e.g., "pointwise").
"""

from __future__ import annotations

from torch._inductor.heuristics.registry import (
    CodegenConfigHeuristics,
    get_codegen_heuristic,
    register_codegen_heuristic,
)

# Keep explicit re-exports for packaging and type checkers. Registry lookup
# performs eager runtime imports when registration is needed.
from . import pointwise as pointwise, reduction as reduction


__all__ = [
    "CodegenConfigHeuristics",
    "get_codegen_heuristic",
    "register_codegen_heuristic",
]
