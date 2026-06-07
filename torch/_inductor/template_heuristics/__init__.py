# Backward-compatibility shim.
# This package has been moved to torch._inductor.heuristics.template.
# These re-exports will be removed in a future release.
from torch._inductor.heuristics.template import (
    aten,
    base,
    contiguous_mm,
    decompose_k,
    nv_universal_gemm,
    registry,
    tlx,
    triton,
)
from torch._inductor.heuristics.template.registry import get_template_heuristic
