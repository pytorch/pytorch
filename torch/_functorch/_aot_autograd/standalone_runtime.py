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

# IDENTITY CONTRACT: these names MUST be plain re-exports that preserve the original
# object identity -- never wrap, decorate, or alias them (e.g. functools.wraps, a thin
# forwarding lambda, a partial). to_standalone_python._known_helper_table keys on
# id() of these exact objects to recognize a global the codegen'd wrappers close over.
# A wrapper would change id(), so the table lookup would silently miss and that global
# would route to its internal AOTAutograd location instead of this stable surface.
# The same contract covers the get_torch_state_as_tuple and set_new_offset
# staticmethods of CUDARngStateHelper.
from .functional_utils import gen_alias_from_base
from .runtime_wrappers import (
    _unwrap_tensoralias,
    mark_dynamo_propagated_dynamic_indices,
)
from .utils import normalize_as_list


__all__ = [
    "gen_alias_from_base",
    "_unwrap_tensoralias",
    "mark_dynamo_propagated_dynamic_indices",
    "normalize_as_list",
    "CUDARngStateHelper",
]
