"""Dynamo-specific helpers for dynamic_spec types.

The spec types themselves live in ``torch.fx.experimental.dynamic_spec``
because they are shared with export and other consumers.
"""

from __future__ import annotations

from torch._dynamo.source import LocalSource
from torch.fx.experimental.dynamic_spec import (
    IntermediateSpec,
    IntVar,
    LeafSpec,
    ParamsSpec,
    ShapesSpec,
    ShapeVar,
    TensorSpec,
)


__all__ = [
    "IntVar",
    "ShapeVar",
    "TensorSpec",
    "ParamsSpec",
    "ShapesSpec",
    "LeafSpec",
    "IntermediateSpec",
    "lookup_spec_from_dynamo_source",
]


def lookup_spec_from_dynamo_source(source, shapes_spec: ShapesSpec | None) -> LeafSpec:
    """Look up the spec for a function input arg from the shapes_spec.

    Only supports LocalSource with is_input=True (direct function args).
    Returns TensorSpec, IntVar, or None.
    """
    if shapes_spec is None or shapes_spec.params is None:
        return None
    if not isinstance(source, LocalSource) or not source.is_input:
        return None
    return shapes_spec.params._named_args.get(source.local_name)
