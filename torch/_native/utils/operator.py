from __future__ import annotations

from typing import Literal

import torch
from torch._library._out_variant import to_out_variant


def variant_aten_name(
    base_aten: str, variant: Literal["functional", "out", "inplace"]
) -> str | None:
    """Resolve a schema-matched ATen variant, or None when unavailable."""
    from ..registry import _resolve_overload

    if variant == "functional":
        return base_aten
    functional = _resolve_overload("aten", base_aten)
    if functional is None:
        return None
    op = base_aten.partition(".")[0]
    if variant == "out":
        try:
            out = to_out_variant(functional)
        except RuntimeError:
            return None
        return f"{op}.{out._schema.overload_name or 'default'}" if out else None
    if variant == "inplace":
        packet = getattr(torch.ops.aten, op + "_", None)
        if packet is None:
            return None
        arguments = functional._schema.arguments
        for name in packet.overloads():
            candidate = getattr(packet, name)._schema.arguments
            if len(candidate) == len(arguments) and all(
                a.name == b.name
                and a.type == b.type
                and a.kwarg_only == b.kwarg_only
                and a.default_value == b.default_value
                for a, b in zip(arguments, candidate)
            ):
                return op + "_" if name == "default" else f"{op}_.{name}"
        return None
    return None
