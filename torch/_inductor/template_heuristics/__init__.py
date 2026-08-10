# Backward-compatibility shim.
# This package has been moved to torch._inductor.heuristics.template.
# Keep every old import path working -- both
#   from torch._inductor.template_heuristics import <sub>
# and
#   import torch._inductor.template_heuristics.<sub>
# -- by importing the moved submodules once and aliasing them into this
# package via sys.modules so the same module objects are reused (this avoids
# duplicate imports and double registration of heuristics).
# These re-exports will be removed in a future release.
import sys

from torch._inductor.heuristics.template import (
    aten,
    base,
    contiguous_mm,
    cutedsl,
    decompose_k,
    flex_gemm,
    gemm,
    nv_universal_gemm,
    params,
    registry,
    tlx,
    triton,
    triton_addmm,
)
from torch._inductor.heuristics.template.registry import get_template_heuristic


for _name in (
    "aten",
    "base",
    "contiguous_mm",
    "cutedsl",
    "decompose_k",
    "flex_gemm",
    "gemm",
    "nv_universal_gemm",
    "params",
    "registry",
    "tlx",
    "triton",
    "triton_addmm",
):
    sys.modules[f"{__name__}.{_name}"] = globals()[_name]

del sys, _name
