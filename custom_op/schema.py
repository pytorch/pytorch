import base64
import inspect
import json
import zlib
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch._library.opaque_object import get_opaque_type_name, is_opaque_type

from .arg_path import ArgPath


# ---- Leaf typing ----

_INPUT_SCHEMA_TYPES = {
    torch.Tensor: "Tensor",
    type(None): "NoneType",
    bool: "bool",
    int: "SymInt",
    float: "float",
    str: "str",
    torch.dtype: "ScalarType",
    torch.device: "Device",
}
_OUTPUT_SCHEMA_TYPES = {
    torch.Tensor: "Tensor",
    int: "SymInt",
}


def schema_type(value: Any, *, is_output: bool = False) -> str:
    table = _OUTPUT_SCHEMA_TYPES if is_output else _INPUT_SCHEMA_TYPES
    for typ, schema in table.items():
        if isinstance(value, typ):
            return schema
    if is_opaque_type(type(value)):
        return get_opaque_type_name(type(value))
    raise TypeError(
        f"unsupported {'output' if is_output else 'input'} leaf "
        f"{value!r} of type {type(value).__name__}"
    )


# ---- Per-call schema inference ----

_POSITIONAL_KINDS = {
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
}


def flat_arg_names(
    sig: inspect.Signature, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> list[str]:
    """Name each flattened leaf of a call, matching tree_flatten order."""
    params = tuple(sig.parameters.values())
    positional = [p.name for p in params if p.kind in _POSITIONAL_KINDS]
    vararg = next(
        (p.name for p in params if p.kind is inspect.Parameter.VAR_POSITIONAL), None
    )

    def leaf_names(name: str, value: Any) -> list[str]:
        leaves, spec = pytree.tree_flatten(value)
        if spec.is_leaf():
            return [name]
        return [f"{name}{i}" for i in range(len(leaves))]

    names = []
    for i, arg in enumerate(args):
        name = positional[i] if i < len(positional) else vararg or f"arg{i}"
        names += leaf_names(name, arg)
    for key, value in kwargs.items():
        names += leaf_names(key, value)

    # Disambiguate leaves from different pytrees that share a base name.
    used: set[str] = set()
    unique = []
    for name in names:
        candidate, count = name, 0
        while candidate in used:
            count += 1
            candidate = f"{name}_{count}"
        used.add(candidate)
        unique.append(candidate)
    return unique


def mutated_flat_indices(
    sig: inspect.Signature,
    mutates: tuple[ArgPath, ...] | str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> set[int]:
    """Flat-input positions that the declared mutation spec marks as mutated.

    ``mutates`` is either "unknown" (every tensor input) or the mutation paths.
    """
    flat_in = pytree.tree_leaves((args, kwargs))
    if mutates == "unknown":
        return {i for i, x in enumerate(flat_in) if isinstance(x, torch.Tensor)}

    # Identity set of the declared-mutated tensors. id() is safe: every tensor is
    # live for this call (held by args/kwargs), so no id reuse. (Tensors can't go
    # in a set directly -- Tensor.__eq__ is elementwise, not identity.)
    bound = sig.bind_partial(*args, **kwargs).arguments
    mutated_ids = set()
    for path in mutates:
        value = path.resolve(bound)
        mutated_ids.update(
            id(x) for x in pytree.tree_leaves(value) if isinstance(x, torch.Tensor)
        )
    return {
        i
        for i, x in enumerate(flat_in)
        if isinstance(x, torch.Tensor) and id(x) in mutated_ids
    }


def schema_annotations(
    flat_in: list[Any],
    flat_out: list[Any],
    mutated: set[int],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Infer per-leaf schema type strings, annotating aliasing tensors.

    Returns (in_types, out_types): tuples of type strings, one per flat_in /
    flat_out leaf and in the same order (names are tracked separately, in
    arg_names). Tensor leaves that alias each other share an annotation name
    (a0, a1, ...); a '!' suffix marks a mutated group. Only groups that need a
    name (more than one member, or mutated) get one.

    Example entries a leaf type string can take:
        "Tensor"          a fresh, unaliased tensor
        "Tensor(a0)"      a tensor aliasing group a0 (shared with another leaf)
        "Tensor(a0!)"     a tensor aliasing group a0 that is also mutated
        "SymInt" / "float" / "NoneType" / "ScalarType" / "Device"  non-tensors

    E.g. for `def f(x): return x.view(-1), x * 2` (the view aliases the input):
        in_types  = ("Tensor(a0)",)
        out_types = ("Tensor(a0)", "Tensor")
    """
    # Step 1: type every leaf. schema_type also validates -- it raises for any
    # leaf whose type isn't a supported input/output type. Alias/mutation
    # annotations are appended below.
    in_types = [schema_type(x) for x in flat_in]
    out_types = [schema_type(x, is_output=True) for x in flat_out]

    # Step 2: group tensor leaves that alias each other.
    # Bucket leaves by storage.
    # Each leaf is identified by (side, index) into flat_in/flat_out.
    groups: dict[int, list[tuple[str, int]]] = {}
    for side, values in (("in", flat_in), ("out", flat_out)):
        for i, x in enumerate(values):
            if isinstance(x, torch.Tensor):
                groups.setdefault(x.untyped_storage()._cdata, []).append((side, i))

    # Step 3: a group is mutated if any of its input members is mutated.
    group_mutated = {
        key: any(side == "in" and i in mutated for side, i in members)
        for key, members in groups.items()
    }

    # Step 4: name a group only if it actually aliases (>1 member) or is mutated
    # (solo unaliased leaves stay bare; names are numeric so >26 groups is fine),
    # then append the "(name)" alias tag + "!" mutation mark to its leaves. An
    # input is marked mutated iff it is itself mutated; an output iff its group is.
    name = {}
    for key, members in groups.items():
        if len(members) > 1 or group_mutated[key]:
            name[key] = f"a{len(name)}"
    for key, members in groups.items():
        if key not in name:
            continue
        for side, i in members:
            mut = i in mutated if side == "in" else group_mutated[key]
            tag = f"({name[key]}{'!' if mut else ''})"
            types = in_types if side == "in" else out_types
            types[i] += tag

    return tuple(in_types), tuple(out_types)


def build_schema_body(
    arg_names: list[str], in_types: tuple[str, ...], out_types: tuple[str, ...]
) -> str:
    """Assemble the '(args) -> returns' portion of a FunctionSchema string.

    Zero/one/many returns are spelled '()' / bare type / '(t0, t1, ...)' to match
    torch's schema grammar. E.g. arg_names=["t"], in_types=("Tensor(a0)",),
    out_types=("Tensor(a0)", "Tensor") -> "(Tensor(a0) t) -> (Tensor(a0), Tensor)".
    """
    args_decl = ", ".join(f"{t} {n}" for t, n in zip(in_types, arg_names))
    if len(out_types) == 0:
        ret_decl = "()"
    elif len(out_types) == 1:
        ret_decl = out_types[0]
    else:
        ret_decl = f"({', '.join(out_types)})"
    return f"({args_decl}) -> {ret_decl}"


# ---- Overload-name codec ----
#
# The overload name is a reversible encoding of the metadata (schema body + pytree
# specs), so an overload's full identity can be recovered from its name alone.

_OVERLOAD_PREFIX = "pt_"


def encode_overload_name(metadata: dict[str, Any]) -> str:
    payload = zlib.compress(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    )
    return _OVERLOAD_PREFIX + base64.b32encode(payload).decode().rstrip("=").lower()


def decode_overload_name(overload_name: str) -> dict[str, Any]:
    """Inverse of encode_overload_name: recover the metadata from the name."""
    encoded = overload_name.removeprefix(_OVERLOAD_PREFIX).upper()
    padding = "=" * (-len(encoded) % 8)  # b32decode needs the stripped padding back
    return json.loads(zlib.decompress(base64.b32decode(encoded + padding)))
