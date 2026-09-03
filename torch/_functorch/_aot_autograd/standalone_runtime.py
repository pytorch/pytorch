"""Runtime-support surface for standalone artifacts.

Modules emitted by ``torch._functorch.aot_autograd.compile_to_python`` inline
AOTAutograd's codegen'd prelude/epilogue, which closes over a few small runtime
helpers (output-alias regeneration, etc.). Rather than have the generated code
reach into scattered AOTAutograd internals -- whose exact locations are not a
stable contract -- it imports those helpers from this one module. This is the
intentional, single dependency surface of a standalone artifact: keep it small and
stable, and update generated-artifact compatibility deliberately if it changes.
"""

# Importing ``runtime_wrappers`` directly in a fresh process pulls in a name
# (e.g. AutogradLazyBackwardCompileInfo) that is only bound once the dynamo init
# chain has run, so importing it first triggers a circular ImportError. Force the
# dynamo/aot chain to fully initialize before the ``from .runtime_wrappers``
# import below so a bare ``import torch`` artifact stays self-contained.
import torch._dynamo  # noqa: F401

# IDENTITY CONTRACT: these names MUST be plain re-exports that preserve the original
# object identity -- never wrap, decorate, or alias them (e.g. functools.wraps, a thin
# forwarding lambda, a partial). source_emit._standalone_runtime_exports keys on id()
# of these exact objects to recognize a global the codegen'd wrappers close over or a
# type their baked metadata is rebuilt from (``CUDARngStateHelper`` included: its
# staticmethods are routed through the class's identity). A wrapper would change
# id(), so the lookup would silently miss and that object would be referenced by
# its internal AOTAutograd location instead of this stable surface.
from torch._prims_common import CUDARngStateHelper

# The whole closed set of descriptor classes: baked ViewAndMutationMeta carries
# whichever ones the traced function produced (tangent descs wrap input-mutation
# and intermediate-base outputs as readily as plain ones).
from .descriptors import (
    BackwardTokenAOTInput,
    BackwardTokenAOTOutput,
    BufferAOTInput,
    DummyAOTInput,
    DummyAOTOutput,
    ForwardTokenAOTInput,
    ForwardTokenAOTOutput,
    GradAOTOutput,
    InputMutationAOTOutput,
    IntermediateBaseAOTOutput,
    MetadataMutationAOTOutput,
    ParamAOTInput,
    PhiloxBackwardBaseOffsetAOTInput,
    PhiloxBackwardSeedAOTInput,
    PhiloxForwardBaseOffsetAOTInput,
    PhiloxForwardSeedAOTInput,
    PhiloxUpdatedBackwardOffsetAOTOutput,
    PhiloxUpdatedForwardOffsetAOTOutput,
    PlainAOTInput,
    PlainAOTOutput,
    SavedForBackwardsAOTOutput,
    SavedForBackwardsNoVcCheckAOTOutput,
    SubclassGetAttrAOTInput,
    SubclassGetAttrAOTOutput,
    SubclassSizeAOTInput,
    SubclassSizeAOTOutput,
    SubclassStrideAOTInput,
    SubclassStrideAOTOutput,
    SyntheticBaseAOTInput,
    TangentAOTInput,
    ViewBaseAOTInput,
)
from .functional_utils import gen_alias_from_base, MetadataKey, ViewMetaSequence
from .runtime_wrappers import (
    _AutogradRngStateTracker,
    _AutogradSavedState,
    _dealias_marked_returns,
    _grad_output_prototypes,
    _mask_pruned_backward_outputs,
    _materialize_missing_grad_outputs,
    _process_runtime_or_materialized_tangent,
    _pruned_backward_output_indices_from_dependencies,
    _snapshot_external_objects,
    _unwrap_tensoralias,
    _wrap_backward_outputs_with_subclasses,
    AOTDispatchAutograd,
    index_to_external_object_weakref,
    KeptTangentInfo,
    mark_dynamo_propagated_dynamic_indices,
)
from .schemas import (
    InputAliasInfo,
    MemoryFormatMeta,
    OutputAliasInfo,
    OutputType,
    PlainTensorMeta,
    SubclassCreationMeta,
    TensorAlias,
    ViewAndMutationMeta,
)
from .subclass_utils import wrap_tensor_subclasses
from .utils import normalize_as_list


# Inference artifacts use the first group; training artifacts (compile_to_python
# with grad_enabled=True) additionally use the autograd-function epilogue and
# prologue helpers and the metadata types their baked ViewAndMutationMeta is
# reconstructed from. source_emit redirects any reference to one of these
# objects to this module, so a generated artifact never imports an AOTAutograd
# module by its internal path. (The wrappers' torch._C calls are attribute
# chains off the ``torch`` global and are emitted as such; they are not routed
# here.)
__all__ = [
    "gen_alias_from_base",
    "_unwrap_tensoralias",
    "mark_dynamo_propagated_dynamic_indices",
    "normalize_as_list",
    "CUDARngStateHelper",
    "_AutogradRngStateTracker",
    "_AutogradSavedState",
    "_dealias_marked_returns",
    "_grad_output_prototypes",
    "_mask_pruned_backward_outputs",
    "_materialize_missing_grad_outputs",
    "_pruned_backward_output_indices_from_dependencies",
    "_snapshot_external_objects",
    "AOTDispatchAutograd",
    "index_to_external_object_weakref",
    "KeptTangentInfo",
    "InputAliasInfo",
    "MemoryFormatMeta",
    "OutputAliasInfo",
    "OutputType",
    "PlainTensorMeta",
    "SubclassCreationMeta",
    "TensorAlias",
    "ViewAndMutationMeta",
    "BackwardTokenAOTInput",
    "BackwardTokenAOTOutput",
    "BufferAOTInput",
    "DummyAOTInput",
    "DummyAOTOutput",
    "ForwardTokenAOTInput",
    "ForwardTokenAOTOutput",
    "GradAOTOutput",
    "InputMutationAOTOutput",
    "IntermediateBaseAOTOutput",
    "MetadataMutationAOTOutput",
    "ParamAOTInput",
    "PhiloxBackwardBaseOffsetAOTInput",
    "PhiloxBackwardSeedAOTInput",
    "PhiloxForwardBaseOffsetAOTInput",
    "PhiloxForwardSeedAOTInput",
    "PhiloxUpdatedBackwardOffsetAOTOutput",
    "PhiloxUpdatedForwardOffsetAOTOutput",
    "PlainAOTInput",
    "PlainAOTOutput",
    "SavedForBackwardsAOTOutput",
    "SavedForBackwardsNoVcCheckAOTOutput",
    "SubclassGetAttrAOTInput",
    "SubclassGetAttrAOTOutput",
    "SubclassSizeAOTInput",
    "SubclassSizeAOTOutput",
    "SubclassStrideAOTInput",
    "SubclassStrideAOTOutput",
    "SyntheticBaseAOTInput",
    "TangentAOTInput",
    "ViewBaseAOTInput",
    # Subclass-tangent training and output-alias regeneration.
    "_process_runtime_or_materialized_tangent",
    "_wrap_backward_outputs_with_subclasses",
    "wrap_tensor_subclasses",
    "MetadataKey",
    "ViewMetaSequence",
]
