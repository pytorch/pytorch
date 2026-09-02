"""Runtime-support surface for standalone artifacts.

Modules emitted by ``torch._functorch.aot_autograd.compile_to_python`` inline
AOTAutograd's codegen'd prelude/epilogue -- and, for a training graph, the autograd
Function bridging its forward and backward -- which close over runtime helpers and
metadata types. Rather than have the generated code reach into scattered AOTAutograd
internals -- whose exact locations are not a stable contract -- it imports every such
name from this one module. This is the intentional, single dependency surface of a
standalone artifact (besides ``torch`` itself and the stdlib): keep it small and
stable, and update generated-artifact compatibility deliberately if it changes.
"""

from typing import Any, NamedTuple

# Importing ``runtime_wrappers`` directly in a fresh process pulls in a name
# (e.g. AutogradLazyBackwardCompileInfo) that is only bound once the dynamo init
# chain has run, so importing it first triggers a circular ImportError. Force the
# dynamo/aot chain to fully initialize before the ``from .runtime_wrappers``
# import below so a bare ``import torch`` artifact stays self-contained.
import torch._dynamo  # noqa: F401
from torch._dynamo.graph_bytecode_inputs import index_to_external_object_weakref
from torch._prims_common import CUDARngStateHelper
from torch._subclasses.functional_tensor import FunctionalTensor
from torch.fx.experimental._backward_state import BackwardState

# IDENTITY CONTRACT: these names MUST be plain re-exports that preserve the original
# object identity -- never wrap, decorate, or alias them (e.g. functools.wraps, a thin
# forwarding lambda, a partial). to_standalone_python._known_helper_table keys on
# id() of these exact objects to recognize a global the codegen'd wrappers close over,
# and its emitted-metadata import routing only redirects a ``module.Name`` reference
# here when ``Name`` IS that exact object. A wrapper would change id(), so the lookup
# would silently miss and that global would route to its internal AOTAutograd location
# instead of this stable surface. The same contract covers ``CUDARngStateHelper``
# (imported above for circular-import ordering): the table keys on id() of its
# ``get_torch_state_as_tuple`` / ``set_new_offset`` staticmethods, so it too must not
# be wrapped or aliased.
from .descriptors import (
    InputMutationAOTOutput,
    IntermediateBaseAOTOutput,
    PlainAOTOutput,
    SubclassGetAttrAOTOutput,
    TangentAOTInput,
)
from .functional_utils import gen_alias_from_base, ViewMetaSequence
from .runtime_wrappers import (
    _AutogradRngStateTracker,
    _AutogradSavedState,
    _dealias_marked_returns,
    _disable_saved_tensors_hooks,
    _grad_output_prototypes,
    _mask_pruned_backward_outputs,
    _materialize_missing_grad_outputs,
    _process_runtime_or_materialized_tangent,
    _pruned_backward_output_indices_from_dependencies,
    _snapshot_external_objects,
    _unwrap_no_symints,
    _unwrap_tensoralias,
    _wrap_pruned_subclass_grad,
    AOTDispatchAutograd,
    KeptTangentInfo,
    mark_dynamo_propagated_dynamic_indices,
)
from .schemas import (
    InputAliasInfo,
    MemoryFormatMeta,
    MutationType,
    OpaqueMeta,
    OutputAliasInfo,
    OutputType,
    PlainTensorMeta,
    SubclassCreationMeta,
    TensorAlias,
    ViewAndMutationMeta,
)
from .subclass_utils import wrap_tensor_subclasses
from .utils import normalize_as_list


class _BackwardVariant(NamedTuple):
    """One entry of a training artifact's backward variant table.

    The table is keyed by the canonical undefined-tangent bitmask (bit ``i`` set when
    specializable user output ``i`` received no grad). ``kept_arg_indices`` is None
    when the variant takes the full saved-arg list; ``pruned_output_indices`` is None
    when the outputs to null out are decided at runtime from the baked dependency
    table (the all-tangents-defined variant, which also serves unseen masks).
    """

    inner_call: Any
    kept_arg_indices: tuple[int, ...] | None
    pruned_output_indices: tuple[int, ...] | None
    skip_materialize_indices: tuple[int, ...]


__all__ = [
    "AOTDispatchAutograd",
    "BackwardState",
    "CUDARngStateHelper",
    "FunctionalTensor",
    "InputAliasInfo",
    "InputMutationAOTOutput",
    "IntermediateBaseAOTOutput",
    "KeptTangentInfo",
    "MemoryFormatMeta",
    "MutationType",
    "OpaqueMeta",
    "OutputAliasInfo",
    "OutputType",
    "PlainAOTOutput",
    "PlainTensorMeta",
    "SubclassCreationMeta",
    "SubclassGetAttrAOTOutput",
    "TangentAOTInput",
    "TensorAlias",
    "ViewAndMutationMeta",
    "ViewMetaSequence",
    "_AutogradRngStateTracker",
    "_AutogradSavedState",
    "_BackwardVariant",
    "_dealias_marked_returns",
    "_disable_saved_tensors_hooks",
    "_grad_output_prototypes",
    "_mask_pruned_backward_outputs",
    "_materialize_missing_grad_outputs",
    "_process_runtime_or_materialized_tangent",
    "_pruned_backward_output_indices_from_dependencies",
    "_snapshot_external_objects",
    "_unwrap_no_symints",
    "_unwrap_tensoralias",
    "_wrap_pruned_subclass_grad",
    "gen_alias_from_base",
    "index_to_external_object_weakref",
    "mark_dynamo_propagated_dynamic_indices",
    "normalize_as_list",
    "wrap_tensor_subclasses",
]
