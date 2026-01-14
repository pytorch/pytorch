# NOTE: add new template heuristics here, so they get imported and registered
# TODO: write a simple glob if there are many heuristics to auto import them in the right order
from . import (
    aten,
    base,
    contiguous_mm,
    decompose_k,
    nv_universal_gemm,
    registry,
    tlx,
    triton,
)

# expose the entry function
from .registry import get_template_heuristic
