from . import (  # NOTE: add new template heuristics here, below registry; and base, as those are needed by the template heuristics; themselves; TODO: write a simple glob if there are many heuristics to; auto import them in the right order
    base,
    decompose_k,
    registry,
    triton,
)

# expose the entry function
from .registry import get_template_heuristic
