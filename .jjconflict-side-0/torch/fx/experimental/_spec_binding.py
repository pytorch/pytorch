"""Bind a ``ShapesSpec``/``ParamsSpec`` to a callable's flat ``(args, kwargs)``.

These helpers are pure spec->args binding (``pytree`` + ``inspect`` + the spec
types); they do NOT touch sympy or the ShapeEnv, so they live at the fx layer
and are imported by make_fx, strict export, and non-strict export alike (instead
of make_fx depending on the dynamo export module).
"""

from __future__ import annotations

import inspect
from itertools import zip_longest
from typing import Any, TYPE_CHECKING

import torch
import torch.utils._pytree as pytree
from torch.fx.experimental.dynamic_spec import (
    DictSpec,
    IntermediateSpec,
    IntVar,
    LeafSpec,
    ObjectSpec,
    SeqSpec,
    ShapesSpec,
    TensorSpec,
)


if TYPE_CHECKING:
    from collections.abc import Callable


def _walk_spec(
    user_spec: IntermediateSpec | None,
    arg_value: Any,
    where: str,
) -> list[LeafSpec]:
    """Return the flat leaf-spec list for ``arg_value`` (one entry per
    ``pytree.tree_flatten`` leaf, ``None`` = static), pairing each leaf with its
    ``user_spec`` counterpart.

    Container specs (``SeqSpec`` / ``DictSpec`` / ``ObjectSpec``) recurse and
    concatenate their children's lists, so leaves land in the same order
    ``pytree.tree_flatten`` produces:

    - **list / tuple** (``SeqSpec`` or no-spec subtree): iterate via plain
      ``enumerate(seq)`` -- matches pytree's ``_list_flatten`` /
      ``_tuple_flatten`` (a namedtuple, being a tuple, is handled by position
      here).
    - **dict** (``DictSpec`` or no-spec subtree): iterate ``arg_value.items()``
      (insertion order); pytree's ``_dict_flatten`` does ``list(d.values())``.
    - **pytree-registered objects** (``ObjectSpec``): use the type's registered
      ``flatten_with_keys_fn`` -- the same function pytree dispatches to.

    No flat-index bookkeeping: parents just concatenate child lists. ``where`` is
    a human-readable path string used solely for error messages.
    """
    # No spec for this subtree (top-level None or unspecified slot) => everything
    # below it is static.
    if user_spec is None:
        return [None] * len(pytree.tree_leaves(arg_value))

    if isinstance(user_spec, SeqSpec):
        if not isinstance(arg_value, (list, tuple)):
            raise ValueError(
                f"{where}: SeqSpec expected list/tuple, got {type(arg_value).__name__}"
            )
        entries = user_spec._entries
        if len(entries) > len(arg_value):
            raise ValueError(
                f"{where}: SeqSpec has {len(entries)} entries beyond runtime "
                f"sequence length {len(arg_value)}"
            )
        out: list[LeafSpec] = []
        for i, (value, sub_spec) in enumerate(zip_longest(arg_value, entries)):
            out += _walk_spec(sub_spec, value, where=f"{where}[{i}]")
        return out

    if isinstance(user_spec, DictSpec):
        if not isinstance(arg_value, dict):
            raise ValueError(
                f"{where}: DictSpec expected dict, got {type(arg_value).__name__}"
            )
        unmatched = set(user_spec) - set(arg_value)
        if unmatched:
            raise ValueError(
                f"{where}: DictSpec has entries {sorted(unmatched, key=repr)!r} "
                f"that do not match any key in the runtime dict. "
                f"Runtime keys: {sorted(arg_value.keys(), key=repr)!r}"
            )
        # Walk runtime ordering so positions align with pytree.tree_flatten
        # (insertion order for plain dicts).
        out = []
        for key, value in arg_value.items():
            sub_spec = user_spec._entries[key] if key in user_spec else None
            out += _walk_spec(sub_spec, value, where=f"{where}[{key!r}]")
        return out

    if isinstance(user_spec, ObjectSpec):
        node_type = pytree._get_node_type(arg_value)
        handler = pytree.SUPPORTED_NODES.get(node_type)
        if handler is None:
            raise ValueError(
                f"{where}: ObjectSpec requires the runtime value's type "
                f"to be pytree-registered, but {type(arg_value).__name__} "
                f"is not registered. Register it via "
                f"`torch.export.register_dataclass(<cls>)` (for dataclasses) "
                f"or `pytree.register_pytree_node(...)`."
            )
        if handler.flatten_with_keys_fn is None:
            raise ValueError(
                f"{where}: export requires "
                f"`flatten_with_keys_fn` to be registered for type "
                f"{type(arg_value).__name__}, but none was found. "
                f"Re-register via "
                f"`pytree.register_pytree_node(..., flatten_with_keys_fn=...)` "
                f"or use `torch.export.register_dataclass` for dataclasses."
            )
        key_children, _ = handler.flatten_with_keys_fn(arg_value)
        # Fail-fast on unmatched attrs before recursing. Non-attribute keys
        # (SequenceKey / MappingKey) contribute no matchable names.
        available_names = {
            ke.name for ke, _ in key_children if isinstance(ke, pytree.GetAttrKey)
        }
        unmatched = set(user_spec) - available_names
        if unmatched:
            raise ValueError(
                f"{where}: ObjectSpec has entries {sorted(unmatched)!r} "
                f"that do not match any attribute on the runtime object "
                f"of type {type(arg_value).__name__}. Available "
                f"attributes: {sorted(available_names)!r}"
            )
        out = []
        for key_entry, child in key_children:
            # Only ``GetAttrKey`` entries can match an ObjectSpec entry; any
            # other key shape contributes a static subtree.
            if isinstance(key_entry, pytree.GetAttrKey) and key_entry.name in user_spec:
                out += _walk_spec(
                    user_spec._fields[key_entry.name],
                    child,
                    where=f"{where}.{key_entry.name}",
                )
            else:
                out += [None] * len(pytree.tree_leaves(child))
        return out

    # Leaf spec -- a single flat slot. Type-check against the runtime value.
    if isinstance(user_spec, TensorSpec):
        if not isinstance(arg_value, torch.Tensor):
            raise ValueError(
                f"{where}: spec is TensorSpec but the actual arg is "
                f"{type(arg_value).__name__}, not a Tensor."
            )
    elif isinstance(user_spec, (IntVar, int)):
        if not isinstance(arg_value, int):
            raise ValueError(
                f"{where}: spec is {type(user_spec).__name__} "
                f"(scalar spec) but the actual arg is "
                f"{type(arg_value).__name__}, not int."
            )
    else:
        raise AssertionError(
            f"{where}: unexpected leaf spec type {type(user_spec).__name__}"
        )
    return [user_spec]


def _bind_spec_to_args(
    f: Callable[..., Any],
    args: Any,
    kwargs: dict[str, Any] | None,
    shapes_spec: ShapesSpec,
) -> tuple[list[LeafSpec], list[Any], pytree.TreeSpec]:
    """Bind a user-provided ``ShapesSpec`` to the actual ``(args, kwargs)``
    by inspecting ``f``'s signature, producing:
      - an aligned leaf-spec list ordered to match
        ``pytree.tree_flatten((args, kwargs))``,
      - the flat args list, and
      - the pytree TreeSpec for ``(args, kwargs)``,
    all from a single ``pytree.tree_flatten`` call (callers don't need to
    re-flatten).

    The leaf-spec list has length equal to the flat leaf count, with
    ``None`` at every position that should be treated as static.

    Consumers: the strict export tracer re-wraps this list into a
    ``ShapesSpec``; the non-strict tracer and ``make_fx`` iterate it and
    pair each entry with its flat input.
    """
    params_spec = shapes_spec._params
    kwargs = kwargs or {}
    flat_args, in_spec = pytree.tree_flatten((args, kwargs))
    if params_spec is None:
        # Empty params spec — no per-arg binding to do. Return an
        # all-None list aligned to the flat input layout.
        return [None] * in_spec.num_leaves, flat_args, in_spec

    params_spec_named_args = params_spec._named_args
    params_spec_varargs = params_spec._varargs  # may be None

    sig = (
        inspect.signature(f.forward)
        if isinstance(f, torch.nn.Module)
        else inspect.signature(f)
    )
    pos_params = list(sig.parameters.values())

    # The signature has up to four param regions, bound from the user's
    # call as follows (spec lookup in parens):
    #   1) named-positional: params before `*args`, from args[i] where
    #      i < varargs_idx            (params_spec_named_args[name])
    #   2) varargs (`*args`): from args[i] where i >= varargs_idx
    #                                 (params_spec_varargs[i - varargs_idx])
    #   3) keyword-named (KEYWORD_ONLY, or pos-or-kw passed by name): from
    #      kwargs[name]               (params_spec_named_args[name])
    #   4) var-keyword (`**kwargs`): from kwargs[name]
    #                                 (params_spec_varkw[name])
    # Python guarantees the call layout is [positionals][kwargs], so we
    # walk `args` first (regions 1 + 2), then `kwargs.items()` (regions
    # 3 + 4, same loop).
    #
    # Example: `def forward(self, x, y, *args, **kwargs)` called as
    # `mod(T1, T2, T3, T4, foo=T5, bar=T6)` → varargs_idx=2;
    # args[:2]=(T1,T2) named; args[2:]=(T3,T4) varargs; kwargs={foo,bar}.

    # `varargs_idx` is how many positional params come before `*args` in the
    # signature: the first `varargs_idx` of the caller's positional `args`
    # bind to those named params, and any extras spill into `*args`. It is
    # len(pos_params) when there is no `*args`.
    varargs_idx = len(pos_params)
    for i, p in enumerate(pos_params):
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            varargs_idx = i
            break

    # Walk the user's actual call structure.
    total_leaves = in_spec.num_leaves
    # out_leaf_specs[i] = leaf-spec for flat_args[i] (None = static); keyed
    # under "*args" since ModuleToTrace.forward only has a varargs signature.
    # Built by concatenating each arg's leaf-spec list, in call order (which
    # matches pytree.tree_flatten((args, kwargs))).
    out_leaf_specs: list[LeafSpec] = []

    # Track which named / **kwargs spec entries get bound; leftovers (typo
    # or spec for an omitted defaulted param) are rejected below.
    matched_named_keys: set[str] = set()
    matched_varkw_keys: set[str] = set()

    # Loop 1: named-positional. Spec keyed by signature param name. A key
    # present with value None means "explicitly static" — it skips binding
    # but still counts as matched (not flagged as unmatched below).
    for i, arg_value in enumerate(args[:varargs_idx]):
        arg_name = pos_params[i].name
        if arg_name in params_spec_named_args:
            matched_named_keys.add(arg_name)
            user_spec = params_spec_named_args[arg_name]
        else:
            user_spec = None
        out_leaf_specs += _walk_spec(
            user_spec, arg_value, where=f"shapes_spec[{arg_name!r}]"
        )

    # Pad varargs spec to the actual `*args` count (missing tail = static)
    # so Loop 2 can index uniformly without a bounds check.
    n_actual_varargs = len(args) - varargs_idx
    params_spec_varargs = list(params_spec_varargs or [])
    params_spec_varargs += [None] * max(0, n_actual_varargs - len(params_spec_varargs))

    # Loop 2: varargs. Spec keyed by position within `*args`.
    for user_idx, arg_value in enumerate(args[varargs_idx:]):
        user_spec = params_spec_varargs[user_idx]
        out_leaf_specs += _walk_spec(
            user_spec, arg_value, where=f"shapes_spec['*args'][{user_idx}]"
        )

    # Loop 3: kwargs. Spec comes from named_args[name] (kwarg matches a
    # named param) or varkw[name] (kwarg flows through `**kwargs`).
    params_spec_varkw = params_spec._varkw  # may be None
    for arg_name, arg_value in kwargs.items():
        if arg_name in params_spec_named_args:
            user_spec = params_spec_named_args[arg_name]
            matched_named_keys.add(arg_name)
        elif params_spec_varkw is not None and arg_name in params_spec_varkw:
            user_spec = params_spec_varkw[arg_name]
            matched_varkw_keys.add(arg_name)
        else:
            user_spec = None
        out_leaf_specs += _walk_spec(
            user_spec, arg_value, where=f"shapes_spec[{arg_name!r}]"
        )

    # Every named / **kwargs spec entry must bind to a passed argument; a
    # leftover is almost always a typo or a spec for an omitted default.
    unmatched = set(params_spec_named_args) - matched_named_keys
    if params_spec_varkw is not None:
        unmatched |= set(params_spec_varkw) - matched_varkw_keys
    if unmatched:
        n_named_positional = len(args[:varargs_idx])
        passed = [p.name for p in pos_params[:n_named_positional]] + list(kwargs)
        raise ValueError(
            f"ParamsSpec has entries {sorted(unmatched)!r} that do not match "
            f"any argument passed to export(). Spec keys must be forward "
            f"parameter names that were actually passed. Inputs received: "
            f"{passed!r}."
        )

    # Sanity: per-arg leaf count must equal the whole-tree flatten, else
    # our per-arg walk drifted from the tracer's layout.
    if len(out_leaf_specs) != total_leaves:
        raise AssertionError(
            f"_bind_spec_to_args leaf-count drift: walked {len(out_leaf_specs)} "
            f"leaves but pytree.tree_flatten((args, kwargs)) yields {total_leaves}. "
            f"This means the translator and the export tracer disagree on the "
            f"flat input layout."
        )

    return out_leaf_specs, flat_args, in_spec
