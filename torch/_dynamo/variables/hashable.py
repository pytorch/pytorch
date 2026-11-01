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
from typing import TYPE_CHECKING

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
    Source,
)
from ..utils import guard_if_dyn, specialize_symnode
from .base import VariableTracker


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase


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
    """Guard receiver aliasing without pinning every internal method-key hash."""

    def __init__(self) -> None:
        self.receivers: dict[Source, object] = {}
        self.input_methods: set[Source] = set()
        self.has_method_keys = False
        self.has_local_receiver = False
        self.has_custom_keys = False
        self.has_custom_metaclass_keys = False

    def _guard_custom_collisions(self, tx: "InstructionTranslatorBase") -> None:
        if not self.has_method_keys or not self.has_custom_keys:
            return
        if self.has_custom_metaclass_keys:
            # Class VTs do not dispatch custom metaclass equality against methods.
            unimplemented(
                gb_type="Builtin method keys with custom metaclass equality",
                context="builtin method key hashing",
                explanation="Dynamo cannot trace custom metaclass equality against builtin method keys.",
                hints=[
                    "Keep these dictionary or set operations outside the compiled region."
                ],
            )
        if self.has_local_receiver:
            unimplemented(
                gb_type="Builtin method keys with custom equality and local receivers",
                context="builtin method key hashing",
                explanation="Custom key equality can observe runtime method hash collisions, "
                "but a locally created receiver has no guardable identity.",
                hints=[
                    "Keep these dictionary or set operations outside the compiled region."
                ],
            )
        for source in self.receivers:
            install_guard(source.make_guard(GuardBuilder.ID_MATCH))

    def add_custom_key(
        self, tx: "InstructionTranslatorBase", *, is_metaclass: bool = False
    ) -> None:
        self.has_custom_keys = True
        self.has_custom_metaclass_keys |= is_metaclass
        self._guard_custom_collisions(tx)

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
            self.has_local_receiver = True
            self._guard_custom_collisions(tx)
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
        if source not in self.receivers:
            for other_source, other in self.receivers.items():
                install_guard(
                    source.make_guard(
                        functools.partial(
                            GuardBuilder.BUILTIN_METHOD_RECEIVER_ALIAS,
                            source_b=other_source,
                            expected_alias=receiver is other,
                        )
                    )
                )
            self.receivers[source] = receiver
        self._guard_custom_collisions(tx)
        return receiver


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

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RawHash) and self.h == other.h


def _builtin_method_sources(value: object, source: Source) -> Iterator[Source]:
    from .builder import bound_builtin_method_descriptor

    if type(value) is tuple:
        for index, item in enumerate(value):
            yield from _builtin_method_sources(item, GetItemSource(source, index))
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
                if input_methods and not isinstance(vt.source, ConstDictKeySource):
                    vt = vt.realize()
        from ..symbolic_convert import InstructionTranslator

        tx = InstructionTranslator.current_tx()
        if not isinstance(vt, variables.LazyVariableTracker) and isinstance(
            vt, BoundBuiltinMethodVariable
        ):
            self._hash, self._hash_is_identity = vt.key_hash_impl(tx)
            self.has_builtin_method = True
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
                    if type(value) is tuple:
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
                tx.output.builtin_method_keys.add_custom_key(
                    tx,
                    is_metaclass=is_metaclass,
                )

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
            self._hash = hash(vt.original_value())
            self._hash_is_identity = False
            self.vt = vt
            return

        vt = vt.realize()
        if type(vt) is variables.TupleVariable:
            # Propagate internal key hashing without relaxing explicit hash(tuple)
            # escape rules. Classify nested custom keys for collision guards too.
            items = [HashableTracker(item) for item in vt.items]
            self._hash = hash(tuple(RawHash(item._hash) for item in items))
            self._hash_is_identity = any(item._hash_is_identity for item in items)
            self.has_builtin_method = any(item.has_builtin_method for item in items)
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
    def from_known_hash(cls, vt: VariableTracker, hash_value: int) -> "HashableTracker":
        # Only for an exact-int key already hashed by its source dictionary.
        # Keep the key lazy; hashing it in user code must still specialize it.
        result = cls.__new__(cls)
        result.vt = vt
        result._hash = hash_value
        result._hash_is_identity = False
        return result

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
        mapping: Iterable["HashableTracker"],
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
