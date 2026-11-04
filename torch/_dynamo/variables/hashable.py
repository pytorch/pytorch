"""
Hashability utilities for PyTorch Dynamo variable tracking.

This module provides the HashableTracker wrapper class and associated utilities
for making VariableTracker instances usable as dictionary keys and set elements
during symbolic execution. Used by both ConstDictVariable and SetVariable.
"""

import functools
import inspect
import types
from collections.abc import Iterable, Iterator
from contextvars import ContextVar
from typing import cast, TYPE_CHECKING

import torch
from torch._subclasses.fake_tensor import maybe_get_fake_constant

from .. import graph_break_hints, variables
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


_KeyStorage = dict["HashableTracker", VariableTracker]

# Identity comparisons avoid callbacks from key metaclasses. Heap classes are
# classified per key, since their equality implementation can be reassigned.
_SAFE_KEY_TYPES = (object, type, int, float, complex, str, bytes, bool, type(None)) + (
    tuple,
    frozenset,
    types.FunctionType,
    types.MethodType,
    types.BuiltinMethodType,
)
_SAFE_KEY_EQUALITIES = (
    *(inspect.getattr_static(kind, "__eq__", None) for kind in _SAFE_KEY_TYPES),
    torch.Tensor.__eq__,
    torch.device.__eq__,
    torch.Size.__eq__,  # THPSize uses PyTuple_Type.tp_richcompare.
)

_identity_key_iteration: ContextVar[bool] = ContextVar(
    "identity_key_iteration", default=False
)


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
    """Keep collision metadata and receiver relations local to participating containers."""

    def __init__(self) -> None:
        self.receivers: dict[Source, object] = {}
        self.parents: dict[Source, Source] = {}
        self.summaries: dict[int, tuple[_KeyStorage, KeySummary]] = {}
        self.has_method_keys = False

    def clear(self) -> None:
        self.receivers.clear()
        self.parents.clear()
        self.summaries.clear()
        self.has_method_keys = False

    def representative(self, source: Source) -> Source:
        path = []
        parent = self.parents.setdefault(source, source)
        while parent != source:
            path.append(source)
            source, parent = parent, self.parents[parent]
        for item in path:
            self.parents[item] = source
        return source

    def connect(self, first: Source, second: Source) -> None:
        self.parents[self.representative(second)] = self.representative(first)

    def install_alias_guard(self) -> None:
        from ..guards import _receiver_alias_pattern

        groups: dict[Source, list[Source]] = {}
        for source in self.parents:
            groups.setdefault(self.representative(source), []).append(source)
        for group in groups.values():
            if len(group) < 2:
                continue
            source = next(
                (item for item in group if not is_from_skip_guard_source(item)), None
            )
            if source is None:
                continue
            sources = tuple(group)
            install_guard(
                source.make_guard(
                    functools.partial(
                        GuardBuilder.BUILTIN_METHOD_RECEIVER_ALIAS,
                        sources=sources,
                        aliases=_receiver_alias_pattern(
                            tuple(self.receivers[item] for item in sources)
                        ),
                    )
                )
            )

    def activate(self, tx: "InstructionTranslatorBase") -> None:
        self.has_method_keys = True
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
        if source not in self.receivers:
            self.receivers[source] = tx.output.resolve_source_value(source)
        return self.receivers[source]

    def summary(self, keys: Iterable["HashableTracker"]) -> "KeySummary":
        if isinstance(keys, dict):
            cached = self.summaries.get(id(keys))
            if cached is not None and cached[1].size == len(keys):
                return cached[1]
            summary = KeySummary(
                self,
                (
                    key
                    for key, value in keys.items()
                    if type(value) is not variables.DeletedVariable
                ),
            )
            summary.size = len(keys)
            self.summaries[id(keys)] = (keys, summary)
            return summary
        return KeySummary(self, keys)


class KeySummary:
    def __init__(
        self, tracker: BuiltinMethodKeyTracker, keys: Iterable["HashableTracker"]
    ) -> None:
        self.tracker = tracker
        self.receivers: dict[Source | None, None] = {}
        self.input_methods: set[Source] = set()
        self.has_methods = False
        self.custom = False
        self.metaclass = False
        self.connected = False
        self.pinned = False
        self.size = 0
        self.first: Source | None = None
        for key in keys:
            self.add(key)

    def add(
        self, key: "HashableTracker", tx: "InstructionTranslatorBase | None" = None
    ) -> None:
        self.receivers.update(dict.fromkeys(key.method_receivers))
        self.input_methods.update(key.input_methods)
        self.has_methods |= key.has_builtin_method and not key.method_keys_guarded
        self.custom |= key.has_custom_equality
        self.metaclass |= key.has_custom_metaclass_equality
        if self.connected and tx is not None:
            self.connect(tx, key.method_receivers)
            if self.pinned:
                for source in key.method_receivers:
                    if source is not None:
                        install_guard(source.make_guard(GuardBuilder.ID_MATCH))

    def connect(
        self,
        tx: "InstructionTranslatorBase",
        sources: Iterable[Source | None] | None = None,
    ) -> None:
        self.connected = True
        for method in self.input_methods:
            install_guard(method.make_guard(GuardBuilder.ID_MATCH))
        self.input_methods.clear()
        for source in self.receivers if sources is None else sources:
            if source is None:
                continue
            self.tracker.add_receiver(tx, source)
            if self.first is None:
                self.first = source
            self.tracker.connect(self.first, source)


def guard_key_collisions(
    *collections: Iterable["HashableTracker"],
) -> KeySummary | None:
    """Guard comparisons at the containers that can actually meet."""
    from ..symbolic_convert import InstructionTranslator

    try:
        tx = InstructionTranslator.current_tx()
    except AttributeError:
        return None
    if tx is None or not tx.output.builtin_method_keys.has_method_keys:
        return None
    tracker = tx.output.builtin_method_keys
    summaries = [tracker.summary(keys) for keys in collections]
    methods = [summary for summary in summaries if summary.has_methods]
    participants = [summary for summary in summaries if summary.receivers]
    custom = any(summary.custom for summary in summaries)
    if methods and custom:
        if any(summary.metaclass for summary in summaries):
            unimplemented(
                gb_type="Builtin method keys with custom metaclass equality",
                context="builtin method key hashing",
                explanation="Dynamo cannot trace custom metaclass equality against builtin method keys.",
                hints=[
                    "Avoid mixing class keys with custom equality and builtin method keys.",
                    *graph_break_hints.SUPPORTABLE,
                ],
            )
        if any(None in summary.receivers for summary in methods):
            unimplemented(
                gb_type="Builtin method keys with custom equality and local receivers",
                context="builtin method key hashing",
                explanation="Custom key equality can observe hash collisions with builtin methods, "
                "but a locally created receiver cannot be identity-guarded.",
                hints=[
                    "Avoid mixing keys with custom equality and methods of locally created containers.",
                    *graph_break_hints.SUPPORTABLE,
                ],
            )
    # A literal-only operand cannot equal a method. Do not activate unrelated input keys.
    if methods and (custom or len(collections) == 1 or len(participants) > 1):
        first = None
        for summary in participants:
            if not summary.connected:
                summary.connect(tx)
            if summary.first is not None:
                if first is None:
                    first = summary.first
                tracker.connect(first, summary.first)
            if custom and not summary.pinned:
                for source in summary.receivers:
                    if source is not None:
                        install_guard(source.make_guard(GuardBuilder.ID_MATCH))
                summary.pinned = True
    return summaries[0] if collections and isinstance(collections[0], dict) else None


def store_item(
    items: _KeyStorage, key: "HashableTracker", value: VariableTracker
) -> None:
    summary = guard_key_collisions(items, (key,))
    size = len(items)
    items[key] = value
    if summary is not None and len(items) != size:
        from ..symbolic_convert import InstructionTranslator

        summary.add(key, InstructionTranslator.current_tx())
        summary.size = len(items)


def invalidate_key_summary(items: _KeyStorage) -> None:
    from ..symbolic_convert import InstructionTranslator

    tx = InstructionTranslator.current_tx()
    tx.output.builtin_method_keys.summaries.pop(id(items), None)


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


def _key_metadata(
    value: object, source: Source
) -> tuple[tuple[Source, ...], tuple[type | None, ...]]:
    from .builder import bound_builtin_method_descriptor

    methods = []
    key_types: list[type | None] = []
    pending = [(value, source)]
    while pending:
        value, source = pending.pop()
        kind = type(value)
        if issubclass(kind, tuple) and kind is not torch.Size:
            key_types.append(kind)
            if inspect.getattr_static(kind, "__getitem__") is not tuple.__getitem__:
                key_types.append(None)  # The source would invoke an overridden indexer.
            else:
                pending.extend(
                    (item, GetItemSource(source, index))
                    for index, item in enumerate(
                        tuple.__iter__(cast(tuple[object, ...], value))
                    )
                )
        elif kind is frozenset:
            pending.extend(
                (item, NonSerializableSetGetItemSource(source, index))
                for index, item in enumerate(cast(frozenset[object], value))
            )
        elif issubclass(kind, frozenset):
            # Subclass iteration can change the source's meaning.
            key_types += [kind, None]
        else:
            key_types.append(kind)
            if (
                kind is types.BuiltinMethodType
                and bound_builtin_method_descriptor(value) is not None
            ):
                methods.append(source)
    return tuple(methods), tuple(key_types)


def _builtin_method_sources(value: object, source: Source) -> Iterator[Source]:
    yield from _key_metadata(value, source)[0]


class HashableTracker:
    """
    Class that wraps a VariableTracker and makes it hashable.
    Note that it's fine to put VTs into dictionaries and sets, but doing so
    does not take into account aliasing.
    """

    vt: VariableTracker
    input_methods: tuple[Source, ...] = ()
    has_builtin_method: bool = False
    method_keys_guarded: bool = False
    method_receivers: tuple[Source | None, ...] = ()
    has_custom_equality: bool = False
    has_custom_metaclass_equality: bool = False

    _MISSING = object()

    def __init__(self, vt: VariableTracker, *, known_hash: int | None = None) -> None:
        if known_hash is not None:
            # Only exact-int hook IDs use a hash from the source dictionary.
            # Keep them lazy until user code observes the key or its hash.
            self.vt = vt
            self._hash, self._hash_is_identity = known_hash, False
            return
        # We specialize SymNodes
        vt = specialize_symnode(vt)
        if type(vt) is variables.ConstantVariable and (
            type(vt.value) is int or type(vt.value) is str
        ):
            self.vt = vt
            self._hash, self._hash_is_identity = hash(vt.value), False
            return

        from .descriptors import BoundBuiltinMethodVariable

        input_methods = ()
        lazy_key_types = None
        if isinstance(vt, variables.LazyVariableTracker):
            if vt.is_realized() or vt.source is None:
                vt = vt.realize()
            else:
                input_methods, lazy_key_types = _key_metadata(
                    vt.original_value(), vt.source
                )
                if not isinstance(vt.source, ConstDictKeySource):
                    if (
                        type(vt.original_value()) is types.BuiltinMethodType
                        and input_methods
                    ):
                        install_guard(vt.source.make_guard(GuardBuilder.ID_MATCH))
                        self.method_keys_guarded = True
                    else:
                        vt = vt.realize()
        from ..symbolic_convert import InstructionTranslator

        if not isinstance(vt, variables.LazyVariableTracker) and isinstance(
            vt, BoundBuiltinMethodVariable
        ):
            tx = InstructionTranslator.current_tx()
            self._hash, self._hash_is_identity = vt.key_hash_impl(tx)
            self.has_builtin_method = True
            self.method_keys_guarded = vt.value is not None
            self.method_receivers = (vt.obj.source,)
            if vt.obj.source is None and not vt.has_fresh_container_receiver():
                self.method_receivers = ()
                self.has_custom_equality = True
            self.vt = vt
            return
        # Builtin equality implementations reject a different method type.
        # User-defined overrides can observe hash collisions.
        key_types: list[type | None]
        try:
            key_types = (
                list(lazy_key_types)
                if lazy_key_types is not None
                else [vt.python_type()]
            )
        except NotImplementedError:
            # Let generic hashing preserve its normal unsupported-type fallback.
            key_types = [None]
        if lazy_key_types is None and any(
            kind is not None
            and issubclass(kind, (tuple, frozenset))
            and not any(kind is builtin for builtin in (tuple, frozenset, torch.Size))
            for kind in key_types
        ):
            key_types.append(None)
        if input_methods:
            try:
                current_tx = InstructionTranslator.current_tx()
            except AttributeError:
                current_tx = None
            if current_tx is not None:
                if self.method_keys_guarded:
                    current_tx.output.builtin_method_keys.activate(current_tx)
                else:
                    current_tx.output.builtin_method_keys.has_method_keys = True
        self.input_methods = () if self.method_keys_guarded else input_methods
        self.has_builtin_method = bool(input_methods)
        self.method_receivers = tuple(
            AttrSource(source, "__self__") for source in input_methods
        )
        for key_type in key_types:
            if not any(key_type is safe_type for safe_type in _SAFE_KEY_TYPES):
                is_metaclass = key_type is not None and issubclass(key_type, type)
                equality = (
                    None
                    if key_type is None
                    else inspect.getattr_static(key_type, "__eq__", None)
                )
                if equality is not None and any(
                    equality is safe for safe in _SAFE_KEY_EQUALITIES
                ):
                    # Inherited builtin equality rejects a different builtin-method type.
                    continue
                self.has_custom_equality = True
                self.has_custom_metaclass_equality |= is_metaclass

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
            self._hash = hash(vt.original_value())
            self._hash_is_identity = False
            self.vt = vt
            return

        tx = InstructionTranslator.current_tx()
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
            self.input_methods = tuple(
                source for item in items for source in item.input_methods
            )
            self.has_builtin_method = any(item.has_builtin_method for item in items)
            self.method_keys_guarded = all(
                item.method_keys_guarded or not item.has_builtin_method
                for item in items
            )
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

    @staticmethod
    def iter_keys_by_identity(
        mapping: dict["HashableTracker", VariableTracker],
    ) -> Iterator["HashableTracker"]:
        # OrderedDict's iterator looks up each stored key. Those internal lookups
        # must not trace user equality; the iterator's current key is already known.
        iterator = iter(mapping)
        while True:
            token = _identity_key_iteration.set(True)
            try:
                key = next(iterator)
            except StopIteration:
                return
            finally:
                _identity_key_iteration.reset(token)
            yield key

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
        if _identity_key_iteration.get():
            return self is other
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

        from .descriptors import BoundBuiltinMethodVariable

        if isinstance(self.vt, BoundBuiltinMethodVariable) and isinstance(
            other.vt, BoundBuiltinMethodVariable
        ):
            # Method equality handles receiver identity without materializing it.
            result = self.vt.tp_richcompare_impl(tx, other.vt, "__eq__")
        else:
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
