from __future__ import annotations

"""ROCm-specific Triton kernel config heuristics.

Hand-crafted, physics-based scoring rules for AMD CDNA GPUs:

  arch      — ArchitectureConfig: device property queries (VGPR budget, CU count, …)
  pointwise — PointwiseHeuristics: score and prune tile configs for elementwise kernels
  reduction — ReductionHeuristics: score and prune R0/R1 configs for reduction kernels
"""

from .arch import ArchitectureConfig, get_architecture_config
from .pointwise import PointwiseHeuristics
from .reduction import ReductionHeuristics

__all__ = [
    "ArchitectureConfig",
    "get_architecture_config",
    "PointwiseHeuristics",
    "ReductionHeuristics",
]
