"""
Hashability utilities for PyTorch Dynamo variable tracking.

This module provides the HashableTracker wrapper class and associated utilities
for making VariableTracker instances usable as dictionary keys and set elements
during symbolic execution. Used by both ConstDictVariable and SetVariable.
"""

import functools
import inspect
import types
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from typing import cast, TYPE_CHECKING

import torch
from torch._subclasses.fake_tensor import maybe_get_fake_constant

from .. import variables
from ..exc import raise_observed_exception, unimplemented
from ..guards import GuardBuilder, install_guard
from ..source import (
    AttrSource,
    ConstDictKeySource,
    GetItemSource,
    is_from_skip_guard_source,
    NonSerializableSetGetItemSource,
    Source,
)
from ..utils import guard_if_dyn, specialize_symnode
from .base import VariableTracker


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase


def raise_unhashable(
    arg: VariableTracker, tx: "InstructionTranslatorBase | None" = None
) -> None:
    if tx is None:
        from torch._dynamo.symbolic_convert import InstructionTranslator

        tx = InstructionTranslator.current_tx()

    try:
        arg_type = arg.python_type()
    except Exception:
        arg_type = None

    # Safety check: if we know the real Python type and it IS hashable,
    # our is_hashable() disagrees with CPython. Graph-break rather than
    # raising a wrong TypeError.
    if arg_type is not None and arg_type.__hash__ is not None:
        from .. import graph_break_hints
        from ..exc import unimplemented

        unimplemented(
            gb_type="Hashability mismatch",
            context=f"raise_unhashable {arg}",
            explanation=f"Dynamo thinks {arg_type.__name__} is unhashable, "
            f"but its __hash__ is not None. This likely indicates a missing "
            f"or incorrect is_hashable() override.",
            hints=[*graph_break_hints.DYNAMO_BUG],
        )

    type_name = arg_type.__name__ if arg_type is not None else type(arg).__name__
    raise_observed_exception(
        TypeError,
        tx,
        args=[
            f"unhashable type: '{type_name}'",
        ],
    )


def is_hashable(x: VariableTracker) -> bool:
    # LazyVT optimization: check hashability without realizing the VT to avoid
    # accidentally inserting guards.
    if (
        isinstance(x, variables.LazyVariableTracker)
        and not x.is_realized()
        and x.is_hashable_lazy()
    ):
        return True

    return x.is_hashable()


class BuiltinMethodKeyTracker:
    """Guard receiver aliasing without pinning every internal method-key hash."""

    def __init__(self) -> None:
        self.receivers: dict[Source, object] = {}
        self.input_methods: set[Source] = set()
        self.has_method_keys = False
        self.has_custom_keys = False

    def install_alias_guard(self) -> None:
        if len(self.receivers) < 2:
            return
        from ..guards import _receiver_alias_pattern

        sources = tuple(self.receivers)
        install_guard(
            sources[0].make_guard(
                functools.partial(
                    GuardBuilder.BUILTIN_METHOD_RECEIVER_ALIAS,
                    sources=sources,
                    aliases=_receiver_alias_pattern(tuple(self.receivers.values())),
                )
            )
        )

    def add_custom_key(self) -> None:
        self.has_custom_keys = True

    def add_input_method(self, tx: "InstructionTranslatorBase", source: Source) -> None:
        self.input_methods.add(source)
        if self.has_method_keys:
            # The receiver relation alone cannot detect a different C function.
            install_guard(source.make_guard(GuardBuilder.ID_MATCH))
            self.add_receiver(tx, AttrSource(source, "__self__"))

    def activate(self, tx: "InstructionTranslatorBase") -> None:
        if not self.has_method_keys:
            self.has_method_keys = True
            for input_source in self.input_methods:
                self.add_input_method(tx, input_source)
        tracer = tx.output.current_tracer
        while tracer is not None:
            tracer.has_builtin_method_keys = True
            tracer = tracer.parent

    def add_receiver(
        self, tx: "InstructionTranslatorBase", source: Source | None
    ) -> object:
        self.activate(tx)
        if source is None:
            return None
        if is_from_skip_guard_source(source):
            unimplemented(
                gb_type="Unguardable builtin method key receiver",
                context=source.name,
                explanation="Builtin method keys require guards on receiver aliasing.",
                hints=[
                    "Keep these dictionary or set operations outside the compiled region."
                ],
            )
        receiver = tx.output.resolve_source_value(source)
        self.receivers.setdefault(source, receiver)
        return receiver


def guard_key_collisions(*collections: Iterable["HashableTracker"]) -> None:
    """Guard only keys that can meet in this dictionary or set operation."""
    from ..symbolic_convert import InstructionTranslator

    tx = InstructionTranslator.current_tx()
    tracker = tx.output.builtin_method_keys
    # Most operations contain neither methods nor custom equality. These frame
    # flags avoid scanning their keys; they never install a guard themselves.
    if (
        not (tracker.has_method_keys or tracker.input_methods)
        or not tracker.has_custom_keys
    ):
        return
    receivers = set()
    custom = metaclass = False
    for collection in collections:
        keys = (
            dict.__iter__(collection)
            if isinstance(collection, OrderedDict)
            else collection
        )
        for key in keys:
            receivers.update(key.method_receivers)
            custom |= key.has_custom_equality
            metaclass |= key.has_custom_metaclass_equality
    if not receivers or not custom:
        return
    tracker.activate(tx)
    if metaclass:
        unimplemented(
            gb_type="Builtin method keys with custom metaclass equality",
            context="builtin method key hashing",
            explanation="Dynamo cannot trace custom metaclass equality against builtin method keys.",
            hints=[
                "Keep these dictionary or set operations outside the compiled region."
            ],
        )
    if None in receivers:
        unimplemented(
            gb_type="Builtin method keys with custom equality and local receivers",
            context="builtin method key hashing",
            explanation="Custom key equality can observe runtime method hash collisions, "
            "but a locally created receiver has no guardable identity.",
            hints=[
                "Keep these dictionary or set operations outside the compiled region."
            ],
        )
    for source in receivers:
        if source is not None:
            install_guard(source.make_guard(GuardBuilder.ID_MATCH))


class RawHash:
    """Wraps a pre-computed hash value to bypass int.__hash__'s modular reduction.

    When building a tuple/frozenset of per-item hashes, using bare ints would
    apply long_hash (mod sys.hash_info.modulus), corrupting the values.
    Wrapping in RawHash makes tuplehash/frozenset_hash see the original hash.
    """

    __slots__ = ("h",)

    def __init__(self, h: int) -> None:
        self.h = h

    def __hash__(self) -> int:
        return self.h


def _builtin_method_sources(value: object, source: Source) -> Iterator[Source]:
    from .builder import bound_builtin_method_descriptor

    if type(value) is tuple:
        for index, item in enumerate(value):
            yield from _builtin_method_sources(item, GetItemSource(source, index))
    elif type(value) is frozenset:
        for index, item in enumerate(value):
            yield from _builtin_method_sources(
                item, NonSerializableSetGetItemSource(source, index)
            )
    elif (
        type(value) is types.BuiltinMethodType
        and bound_builtin_method_descriptor(value) is not None
    ):
        yield source


class HashableTracker:
    """
    Class that wraps a VariableTracker and makes it hashable.
    Note that it's fine to put VTs into dictionaries and sets, but doing so
    does not take into account aliasing.
    """

    vt: VariableTracker
    has_builtin_method: bool = False
    method_receivers: tuple[Source | None, ...] = ()
    has_custom_equality: bool = False
    has_custom_metaclass_equality: bool = False

    _MISSING = object()

    def __init__(self, vt: VariableTracker) -> None:
        # We specialize SymNodes
        vt = specialize_symnode(vt)

        from .descriptors import BoundBuiltinMethodVariable

        input_methods = ()
        if isinstance(vt, variables.LazyVariableTracker):
            if vt.is_realized() or vt.source is None:
                vt = vt.realize()
            else:
                input_methods = tuple(
                    _builtin_method_sources(vt.original_value(), vt.source)
                )
                if not isinstance(vt.source, ConstDictKeySource):
                    vt = vt.realize()
        from ..symbolic_convert import InstructionTranslator

        tx = InstructionTranslator.current_tx()
        if not isinstance(vt, variables.LazyVariableTracker) and isinstance(
            vt, BoundBuiltinMethodVariable
        ):
            self._hash, self._hash_is_identity = vt.key_hash_impl(tx)
            self.has_builtin_method = True
            self.method_receivers = (vt.obj.source,)
            if vt.obj.source is None and not any(
                vt.obj.python_type() is builtin for builtin in (list, dict, set)
            ):
                self.method_receivers = ()
                self.has_custom_equality = True
            self.vt = vt
            return
        # Only exact builtin equality is known not to equate a different type
        # with a builtin method. User-defined keys can observe hash collisions.
        key_types: list[type | None]
        try:
            if isinstance(vt, variables.LazyVariableTracker) and not vt.is_realized():
                # Exact tuples can hide custom equality behind the lazy-key path.
                # Inspect actual types without realizing unused keys or their guards.
                values = [vt.original_value()]
                key_types = []
                while values:
                    value = values.pop()
                    if type(value) is tuple or type(value) is frozenset:
                        values.extend(value)
                    else:
                        key_types.append(type(value))
            else:
                key_types = [vt.python_type()]
        except NotImplementedError:
            # Let generic hashing preserve its normal unsupported-type fallback.
            key_types = [None]
        for key_type in key_types:
            if not any(
                key_type is safe_type
                for safe_type in (
                    object,
                    type,
                    int,
                    float,
                    complex,
                    str,
                    bytes,
                    tuple,
                    frozenset,
                    type(None),
                    bool,
                    types.FunctionType,
                    types.MethodType,
                    types.BuiltinMethodType,
                )
            ):
                is_metaclass = key_type is not None and issubclass(key_type, type)
                if (
                    is_metaclass
                    and inspect.getattr_static(key_type, "__eq__") is object.__eq__
                ):
                    # Identity-only class equality cannot match a builtin method.
                    continue
                self.has_custom_equality = True
                self.has_custom_metaclass_equality |= is_metaclass
                tx.output.builtin_method_keys.add_custom_key()

        # Fast path for unrealized LazyVariableTrackers: check and hash without
        # realizing, to avoid inserting guards.  If the fast-path check fails,
        # fall through to realize the VT and try the full is_hashable check.
        if (
            isinstance(vt, variables.LazyVariableTracker)
            and not vt.is_realized()
            and vt.is_hashable_lazy()
        ):
            # Preserve unused input-key laziness, including methods inside tuples.
            # Activate guards when a constructed method key can compare with them.
            for source in input_methods:
                tx.output.builtin_method_keys.add_input_method(tx, source)
            self.has_builtin_method = bool(input_methods)
            self.method_receivers = tuple(
                AttrSource(source, "__self__") for source in input_methods
            )
            self._hash = hash(vt.original_value())
            self._hash_is_identity = False
            self.vt = vt
            return

        vt = vt.realize()
        if type(vt) is variables.TupleVariable or (
            isinstance(vt, variables.FrozensetVariable)
            and vt.python_type() is frozenset
        ):
            # Constructed keys must guard their elements. Input mapping keys keep
            # their lazy fast path above until an operation observes them.
            if type(vt) is variables.TupleVariable:
                items = [HashableTracker(item.realize()) for item in vt.items]
                self._hash = hash(tuple(RawHash(item._hash) for item in items))
            else:
                items = list(cast(variables.FrozensetVariable, vt).items)
                self._hash = hash(frozenset(RawHash(item._hash) for item in items))
            self._hash_is_identity = any(item._hash_is_identity for item in items)
            self.has_builtin_method = any(item.has_builtin_method for item in items)
            self.method_receivers = tuple(
                source for item in items for source in item.method_receivers
            )
            self.has_custom_equality = any(item.has_custom_equality for item in items)
            self.has_custom_metaclass_equality = any(
                item.has_custom_metaclass_equality for item in items
            )
            self.vt = vt
            return

        # Compute hash via the tp_hash slot (generic_hash_impl).
        # For unhashable types, hash_impl raises ObservedTypeError.
        from .object_protocol import generic_hash_impl

        # is_fake marks an identity-based hash (e.g. id(fake_tensor)); such VTs
        # have no python-constant value, so their key equality is identity.
        self._hash, self._hash_is_identity = generic_hash_impl(tx, vt)
        self.vt = vt

    @classmethod
    def _maybe_constant_torch_size(cls, vt: VariableTracker) -> object:
        from .lists import SizeVariable
        from .tensor import TensorVariable

        if (
            isinstance(vt, variables.LazyVariableTracker)
            and not vt.is_realized()
            and isinstance(vt.original_value(), torch.Size)
        ):
            return vt.original_value()

        if not isinstance(vt, SizeVariable):
            return cls._MISSING

        items = []
        for item in vt.items:
            if item.is_python_constant():
                items.append(item.as_python_constant())
                continue

            if isinstance(item, TensorVariable):
                proxy = getattr(item, "proxy", None)
                node = getattr(proxy, "node", None)
                meta = getattr(node, "meta", None) if node is not None else None
                example_value = (
                    meta.get("example_value") if isinstance(meta, dict) else None
                )
                constant = maybe_get_fake_constant(example_value)

                if isinstance(constant, torch.Tensor) and constant.numel() == 1:
                    items.append(constant.item())
                    continue

            return cls._MISSING

        return torch.Size(items)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        """
        Checks equality between two HashableTracker instances.

        Mirrors CPython's PyObject_RichCompareBool: routes the comparison
        through generic_richcompare_bool so any user-defined __eq__ runs.

        Args:
            other: Another HashableTracker instance to compare with

        Returns:
            True if the underlying variable trackers are Python-equal, False otherwise
        """
        if not isinstance(other, HashableTracker):
            return False
        if self.vt is other.vt:
            return True

        self_constant = self._maybe_constant_torch_size(self.vt)
        other_constant = self._maybe_constant_torch_size(other.vt)
        if self_constant is not self._MISSING and other_constant is not self._MISSING:
            return self_constant == other_constant

        # Tensor keys hash by identity (Tensor.__hash__ is id(self)), so CPython
        # only ever compares identical tensors during a dict/set lookup; a
        # tensor's elementwise __eq__ is never consulted for membership. Mirror
        # that with an identity comparison instead of running __eq__, which would
        # otherwise emit a stray elementwise-eq node into the FX graph and
        # corrupt the surrounding trace.
        from .tensor import TensorVariable

        if isinstance(self.vt, TensorVariable) or isinstance(other.vt, TensorVariable):
            return (
                self._hash_is_identity
                and other._hash_is_identity
                and self._hash == other._hash
            )

        # All other keys: mirror PyObject_RichCompareBool and run the
        # comparison through generic_richcompare_bool so any user-defined __eq__
        # runs and its result (or any exception it raises) is observed by the
        # traced program.
        from ..symbolic_convert import InstructionTranslator
        from .object_protocol import generic_richcompare_bool

        tx = InstructionTranslator.current_tx()

        result = generic_richcompare_bool(tx, self.vt, other.vt, op="__eq__")
        if result.is_python_constant():
            return bool(result.as_python_constant())

        if result.is_symnode_like():
            return bool(guard_if_dyn(result))

        # Comparison did not resolve to a constant (e.g. keys whose __eq__ could
        # not be determined). Fall back to identity-based hash equality, which
        # mirrors CPython treating such keys as equal only when identical.
        return (
            self._hash_is_identity
            and other._hash_is_identity
            and self._hash == other._hash
        )
