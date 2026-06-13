"""Runtime-support surface for standalone artifacts.

Modules emitted by ``torch._functorch.aot_autograd.compile_to_python`` inline
AOTAutograd's codegen'd prelude/epilogue, which closes over a few small runtime
helpers (output-alias regeneration, etc.). Rather than have the generated code
reach into scattered AOTAutograd internals -- whose exact locations are not a
stable contract -- it imports those helpers from this one module. This is the
intentional, single dependency surface of a standalone artifact: keep it small and
stable, and update generated-artifact compatibility deliberately if it changes.
"""

from torch._prims_common import CUDARngStateHelper

from .functional_utils import gen_alias_from_base
from .runtime_wrappers import _unwrap_tensoralias
from .utils import normalize_as_list


__all__ = [
    "gen_alias_from_base",
    "_unwrap_tensoralias",
    "normalize_as_list",
    "CUDARngStateHelper",
]
