"""
Dictionary-related variable tracking classes for PyTorch Dynamo.

This module implements variable tracking for different types of dictionary-like objects:
- Regular Python dictionaries (dict)
- Ordered dictionaries (collections.OrderedDict)
- Default dictionaries (collections.defaultdict)
- Dictionary views (keys and values)

These classes are responsible for tracking dictionary operations during graph compilation,
maintaining proper guards for dictionary mutations and key existence checks. They handle
dictionary creation, modification, key/value access, and view operations while ensuring
correct behavior in the compiled code through appropriate guard installation.

The implementation uses a special HashableTracker wrapper to handle
dictionary keys while preserving proper aliasing semantics. Set-related classes live
in sets.py.
"""

import collections
import functools
import gc
import sys
import types
import weakref
from collections.abc import Callable, Iterator
from typing import Any, cast, TYPE_CHECKING, TypeVar, Union
from typing_extensions import TypeIs

from torch.utils._pytree import MappingKey

from .. import graph_break_hints, polyfills, variables
from ..bytecode_transformation import (
    create_call_function,
    create_call_method,
    create_dup_top,
    create_instruction,
)
from ..exc import (
    handle_observed_exception,
    ObservedTypeError,
    raise_observed_exception,
    raise_type_error,
    TorchDynamoException,
    unimplemented,
    Unsupported,
)
from ..guards import GuardBuilder, install_guard
from ..source import (
    AttrSource,
    ConstDictKeySource,
    DictGetItemSource,
    is_constant_source,
    is_from_local_source,
    TypeDictSource,
)
from ..utils import (
    _item_debug_repr,
    check_positional,
    cmp_name_to_op_mapping,
    dict_items,
    dict_keys,
    dict_values,
    get_type_mro_no_user_code,
    istype,
    odict_items,
    odict_keys,
    odict_values,
    tracked_repr,
    unpack_iterable,
)
from .base import (
    AttributeMutationExisting,
    AttributeMutationNew,
    AttrMutationKind,
    GetSet,
    Method,
    NO_SUCH_SUBOBJ,
    readonly_setter,
    ValueMutationExisting,
    ValueMutationNew,
    VariableTracker,
)
from .constant import ConstantVariable
from .hashable import HashableTracker, is_hashable
from .object_protocol import (
    _is_method_type,
    generic_getattr,
    generic_getitem,
    generic_is_true,
    generic_richcompare_bool,
    generic_setitem,
    generic_str,
    mro_attr_source,
    mro_lookup,
    pysequence_contains,
    repr_guard,
)


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.side_effects import SideEffects
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase

    from .functions import UserFunctionVariable


# [Adding a new supported class within the keys of ConstDictVariable]
# - Implement hash_impl(self, tx) on the VariableTracker subclass (or rely on the
#   base class default which uses get_real_python_backed_value())
# - Implement tp_richcompare_impl() for key equality


def _get_literal_key(key: VariableTracker) -> Any:
    # Inspect lazy keys without realizing classes or other nonliteral objects.
    if isinstance(key, variables.LazyVariableTracker) and not key.is_realized():
        value = key.original_value()
    elif key.is_python_constant():
        value = key.as_python_constant()
    else:
        return NO_SUCH_SUBOBJ
    return value if ConstantVariable.is_literal(value) else NO_SUCH_SUBOBJ


def _contains_nan(value: Any) -> bool:
    if type(value) in (tuple, frozenset):
        return any(_contains_nan(item) for item in value)
    return type(value) in (float, complex) and value != value


def _guard_nan_key(key: VariableTracker) -> None:
    # CONSTANT_MATCH accepts distinct NaNs, but dictionary lookups depend
    # on their identity, including NaNs nested inside tuple keys.
    if key.source and _contains_nan(_get_literal_key(key)):
        install_guard(key.make_guard(GuardBuilder.ID_MATCH))


# Mirror PyDict_{Check,CheckExact} in Include/cpython/dictobject.h.
def pydict_check(obj: VariableTracker) -> TypeIs["ConstDictVariable"]:
    return issubclass(obj.python_type(), dict)


def pydict_checkexact(obj: VariableTracker) -> TypeIs["ConstDictVariable"]:
    return obj.python_type() is dict


# Mirror PyODict_{Check,CheckExact} in Include/cpython/odictobject.h.
def pyodict_check(obj: VariableTracker) -> TypeIs["OrderedDictVariable"]:
    return issubclass(obj.python_type(), collections.OrderedDict)


def pyodict_checkexact(obj: VariableTracker) -> TypeIs["OrderedDictVariable"]:
    return obj.python_type() is collections.OrderedDict


_T = TypeVar("_T")


def _iter_detecting_mutation(iterator: Iterator[_T]) -> Iterator[_T]:
    while True:
        try:
            item = next(iterator)
        except StopIteration:
            return
        except TorchDynamoException:
            # User-exception sentinels and graph breaks also inherit RuntimeError;
            # propagate them before handling iterator mutation errors.
            raise
        except RuntimeError as error:
            unimplemented(
                gb_type="Dictionary mutated during iteration",
                context=str(error),
                explanation="Dynamo cannot trace dictionary iteration that changes its keys.",
                hints=[*graph_break_hints.SUPPORTABLE],
            )
        # Keep user code outside the handler so its exceptions propagate.
        yield item


def _is_set_or_dictview(obj: VariableTracker) -> bool:
    """PyAnySet_Check(other) || PyDictViewSet_Check(other)"""
    try:
        t = obj.python_type()
    except NotImplementedError:
        return False
    return issubclass(t, (set, frozenset, dict_keys, dict_items))


class ConstDictVariable(VariableTracker):
    # PyDict_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/dictobject.c#L4825
    _cpython_type = dict

    CONTAINS_GUARD = GuardBuilder.DICT_CONTAINS
    NOT_CONTAINS_GUARD = GuardBuilder.DICT_NOT_CONTAINS

    _nonvar_fields = {
        "_checking_python_constant",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        items: dict[VariableTracker, VariableTracker]
        | dict[HashableTracker, VariableTracker],
        **kwargs: Any,
    ) -> None:
        # .clone() pass these arguments in kwargs but they're recreated a few
        # lines below
        if "original_items" in kwargs:
            kwargs.pop("original_items")
        if "should_reconstruct_all" in kwargs:
            kwargs.pop("should_reconstruct_all")
        if "_checking_python_constant" in kwargs:
            kwargs.pop("_checking_python_constant")

        super().__init__(**kwargs)

        Hashable = HashableTracker

        # Keys will just be HashableTrackers when cloning, in any other case they'll be VariableTrackers
        if not all(
            isinstance(x, (VariableTracker, Hashable))
            and isinstance(v, VariableTracker)
            for x, v in items.items()
        ):
            raise AssertionError(
                "All keys must be VariableTracker or HashableTracker "
                "and all values must be VariableTracker"
            )

        def make_hashable(
            key: Union[VariableTracker, "HashableTracker"],
        ) -> "HashableTracker":
            return key if isinstance(key, Hashable) else Hashable(key)

        items_cls = cast(type, self._cpython_type)
        storage_cls = dict if items_cls is collections.defaultdict else items_cls
        self.items: dict[HashableTracker, VariableTracker]
        if storage_cls is collections.OrderedDict and isinstance(
            items, collections.OrderedDict
        ):
            # Normalizing tracker keys must preserve both orders without hashing twice.
            keys = {id(key): make_hashable(key) for key in items}
            storage = cast(dict[object, VariableTracker], items)
            self.items = collections.OrderedDict(
                (keys[id(key)], value) for key, value in dict.items(storage)
            )
            for key in items:
                self.items.move_to_end(keys[id(key)])
        else:
            self.items = storage_cls({make_hashable(x): v for x, v in items.items()})
        # need to reconstruct everything if the dictionary is an intermediate value
        # or if a pop/delitem was executed
        self.should_reconstruct_all = (
            not is_from_local_source(self.source) if self.source else True
        )
        self.original_items = {key.vt: value for key, value in self.items.items()}
        # Re-entrancy guard for is_python_constant against self-referential
        # dicts. Both forms re-enter this same instance's is_python_constant, so
        # a per-instance flag suffices.
        self._checking_python_constant = False

    def as_proxy(self) -> dict[Any, Any]:
        return {k.vt.as_proxy(): v.as_proxy() for k, v in self.items.items()}

    def debug_repr(self) -> str:
        items: list[str] = []
        for k, v in self.items.items():
            key_str = _item_debug_repr(k.vt)
            val_str = _item_debug_repr(v)
            items.append(f"{key_str}: {val_str}")
        return "{" + ", ".join(items) + "}"

    def is_python_constant(self) -> bool:
        # Avoid the base implementation, which probes as_python_constant() and
        # thus rebuilds a real dict, re-hashing the keys (wrong for keys with a
        # side-effecting __hash__).  Check element constness directly instead.
        # Re-entrancy guard: a self-referential dict would otherwise recurse
        # forever. A cyclic dict is not a flat python constant; return False so
        # reconstruct handles the cycle.
        if self._checking_python_constant:
            return False
        self._checking_python_constant = True
        try:
            return all(
                k.vt.is_python_constant() and v.is_python_constant()
                for k, v in self.items.items()
            )
        finally:
            self._checking_python_constant = False

    def as_python_constant(self) -> dict[Any, Any]:
        # Guards on a sourced dict are installed lazily. Materializing it bakes
        # its keys into the trace, e.g. an empty input dict is vacuously a
        # python constant, so guard on them here.
        self.install_dict_keys_match_guard()
        return {
            k.vt.as_python_constant(): v.as_python_constant()
            for k, v in self.items.items()
        }

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # https://github.com/python/cpython/blob/v3.13.3/Objects/dictobject.c#L3033-L3110
        with repr_guard(self) as recursive:
            if recursive:
                return VariableTracker.build(tx, "{...}")
            items = [
                f"{tracked_repr(tx, key.vt)}: {tracked_repr(tx, value)}"
                for key, value in self.items.items()
            ]
        return VariableTracker.build(tx, "{" + ", ".join(items) + "}")

    def keys_as_python_constant(self) -> dict[Any, VariableTracker]:
        self.install_dict_keys_match_guard()
        return {k.vt.as_python_constant(): v for k, v in self.items.items()}

    def python_type(self) -> type:
        return cast(type, self._cpython_type)

    def __contains__(self, vt: VariableTracker) -> bool:
        if not isinstance(vt, VariableTracker):
            raise AssertionError(f"Expected VariableTracker, got {type(vt)}")
        # Use is_hashable as a side-effect-free pre-check.  We can't catch
        # ObservedTypeError from HashableTracker because it modifies
        # tx.exn_vt_stack as a side effect.
        if not is_hashable(vt):
            return False
        key = HashableTracker(vt)
        return key in self.items and not isinstance(
            self.items[key], variables.DeletedVariable
        )

    def call_tree_map_branch(
        self,
        tx: "InstructionTranslatorBase",
        tree_map_fn: "UserFunctionVariable",
        map_fn: VariableTracker,
        rest: list[VariableTracker],
        tree_map_kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        self.install_dict_order_guard(tx)
        other_dicts: list[ConstDictVariable] = []
        for candidate in rest:
            candidate = candidate.realize()
            if not isinstance(candidate, ConstDictVariable) or len(
                candidate.items
            ) != len(self.items):
                return self._tree_map_fallback(
                    tx, tree_map_fn, map_fn, rest, tree_map_kwargs
                )
            other_dicts.append(candidate)

        new_items_hashed = type(self.items)()
        for key_tracker, value in self.items.items():
            sibling_leaves: list[VariableTracker] = []
            for candidate in other_dicts:
                try:
                    sibling_leaves.append(candidate.items[key_tracker])
                except KeyError:
                    return self._tree_map_fallback(
                        tx, tree_map_fn, map_fn, rest, tree_map_kwargs
                    )
            new_items_hashed[key_tracker] = value.call_tree_map(
                tx,
                tree_map_fn,
                map_fn,
                sibling_leaves,
                tree_map_kwargs,
            )

        updated_original_items = {
            key_tracker.vt: new_items_hashed[key_tracker]
            for key_tracker in new_items_hashed
        }

        return self.clone(
            items=new_items_hashed,
            original_items=updated_original_items,
            should_reconstruct_all=True,
            source=None,
            mutation_type=ValueMutationNew(),
        )

    def call_tree_map_with_path_branch(
        self,
        tx: "InstructionTranslatorBase",
        tree_map_fn: "UserFunctionVariable",
        map_fn: VariableTracker,
        rest: list[VariableTracker],
        tree_map_kwargs: dict[str, VariableTracker],
        keypath: tuple[Any, ...],
    ) -> VariableTracker:
        self.install_dict_order_guard(tx)
        other_dicts: list[ConstDictVariable] = []
        for candidate in rest:
            candidate = candidate.realize()
            if not isinstance(candidate, ConstDictVariable) or len(
                candidate.items
            ) != len(self.items):
                return self._tree_map_with_path_fallback(
                    tx, tree_map_fn, map_fn, rest, tree_map_kwargs, keypath
                )
            other_dicts.append(candidate)

        new_items_hashed = type(self.items)()
        for key_tracker, value in self.items.items():
            sibling_leaves: list[VariableTracker] = []
            for candidate in other_dicts:
                try:
                    sibling_leaves.append(candidate.items[key_tracker])
                except KeyError:
                    return self._tree_map_with_path_fallback(
                        tx, tree_map_fn, map_fn, rest, tree_map_kwargs, keypath
                    )
            key_const = key_tracker.vt.as_python_constant()
            child_keypath = keypath + (MappingKey(key_const),)
            new_items_hashed[key_tracker] = value.call_tree_map_with_path(
                tx,
                tree_map_fn,
                map_fn,
                sibling_leaves,
                tree_map_kwargs,
                child_keypath,
            )

        updated_original_items = {
            key_tracker.vt: new_items_hashed[key_tracker]
            for key_tracker in new_items_hashed
        }

        return self.clone(
            items=new_items_hashed,
            original_items=updated_original_items,
            should_reconstruct_all=True,
            source=None,
            mutation_type=ValueMutationNew(),
        )

    def len(self) -> int:
        return sum(
            not isinstance(x, variables.DeletedVariable) for x in self.items.values()
        )

    def has_new_items(self) -> bool:
        return self.should_reconstruct_all or any(
            self.is_new_item(self.original_items.get(key.vt), value)
            for key, value in self.items.items()
        )

    def is_new_item(
        self, value: VariableTracker | None, other: VariableTracker
    ) -> bool:
        # compare the id of the realized values if both values are not lazy VTs
        if value and value.is_realized() and other.is_realized():
            return id(value.realize()) != id(other.realize())
        return id(value) != id(other)

    def reconstruct_kvs_into_new_dict(self, codegen: "PyCodegen") -> None:
        # Build a dictionary that contains the keys and values.
        num_args = 0
        for key, value in self.items.items():
            item = self.original_items.get(key.vt)
            if self.is_new_item(item, value) or self.should_reconstruct_all:
                key_var = key.vt
                if (
                    not self.should_reconstruct_all
                    and isinstance(key_var, ConstantVariable)
                    and isinstance(key_var.source, ConstDictKeySource)
                    and key_var.source.base == self.source
                    and not _contains_nan(key_var.as_python_constant())
                ):
                    key_var = key_var.clone(source=None)
                # Updating an existing literal key retains its identity and order.
                # External keys and full reconstruction still need their sources.
                if isinstance(
                    self.mutation_type,
                    (ValueMutationExisting, AttributeMutationExisting),
                ) and (
                    self.should_reconstruct_all
                    or (
                        isinstance(key_var.source, ConstDictKeySource)
                        and key_var.source.base == self.source
                    )
                ):
                    self.install_dict_order_guard(codegen.tx)
                codegen(key_var)
                codegen(value)
                num_args += 1
        codegen.append_output(create_instruction("BUILD_MAP", arg=num_args))

    def reconstruct(self, codegen: "PyCodegen") -> None:
        if self._contains_self_reference() and self.source is None:
            codegen.extend_output(
                [
                    create_instruction("BUILD_MAP", arg=0),
                    create_dup_top(),
                ]
            )
            codegen.add_cache(self)
            self.reconstruct_kvs_into_new_dict(codegen)
            codegen.append_output(create_instruction("DICT_UPDATE", arg=1))
        else:
            # Non-self-referential: use simple codegen
            self.reconstruct_kvs_into_new_dict(codegen)

    def reconstruct_pycode(self, codegen) -> str:
        if self._contains_self_reference():
            raise NotImplementedError
        return f"{{{', '.join(f'{k.vt.reconstruct_pycode(codegen)}: {v.reconstruct_pycode(codegen)}' for k, v in self.items.items())}}}"

    def getitem_const_raise_exception_if_absent(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker:
        key = self._lookup_key(tx, arg)
        value = self.items.get(key)
        # The deletion marker is explicit; `isinstance` would realize lazy values.
        if value is None or type(value) is variables.DeletedVariable:
            raise_observed_exception(KeyError, tx, args=[arg])
        return value

    def getitem_const(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker:
        key = HashableTracker(arg)
        if key not in self.items:
            msg = f"Dictionary key {arg.value} not found during tracing"  # type: ignore[attr-defined]
            unimplemented(
                gb_type="key not found in dict",
                context=f"Key {arg.value}",  # type: ignore[attr-defined]
                explanation=msg,
                hints=[
                    "Check if the key exists in the dictionary before accessing it.",
                    *graph_break_hints.USER_ERROR,
                ],
            )
        return self.items[key]

    def maybe_getitem_const(self, arg: VariableTracker) -> VariableTracker | None:
        key = HashableTracker(arg)
        if key not in self.items:
            return None
        return self.items[key]

    def realize_key_vt(self, key: HashableTracker) -> None:
        # Symbolic equality already realizes stored keys. The identity shortcut
        # can leave a matched key lazy; find it without repeating user comparisons.
        original_value = key.vt.get_real_python_backed_value()
        if original_value is NO_SUCH_SUBOBJ:
            return
        for original_key in self.original_items:
            if (
                isinstance(original_key, variables.LazyVariableTracker)
                and not original_key.is_realized()
                and original_key.original_value() is original_value
            ):
                original_key.realize()
                break

    def install_dict_keys_match_guard(self) -> None:
        if self.source and not is_constant_source(self.source):
            install_guard(self.make_guard(GuardBuilder.DICT_KEYS_MATCH))

    def install_dict_order_guard(
        self,
        tx: "InstructionTranslatorBase",
        *,
        view: "DictViewVariable | None" = None,
    ) -> None:
        # Without a view, the operation consumes the mapping's keys directly.
        # A view identifies its native order and which parts of each item escape.
        if self.source and not is_constant_source(self.source):
            self.install_dict_keys_match_guard()
            tx.output.guard_on_key_order.add(self.source)

    def install_dict_contains_guard(
        self,
        tx: "InstructionTranslatorBase",
        key: HashableTracker,
        *,
        key_present: bool | None = None,
    ) -> None:
        # Key guarding - These are the cases to consider
        # 1) The dict has been mutated. In this case, we would have already
        # inserted a DICT_KEYS_MATCH guard, so we can skip.
        #
        # 2) The key is a literal or a sourceless constant. Guard membership.
        #
        # 3) The key is nonliteral and has a source.
        #   3a) key_present=True. Hashing and comparison already realize and
        #   guard the nonconstant keys used by a successful lookup.
        #   3b) key_present=False. There is no easy way to selectively apply this
        #   DICT_NOT_CONTAINS guard because our guard are represented via trees.
        #   Be conservative and add DICT_KEYS_MATCH guard.

        if not self.source:
            return

        if tx.output.side_effects.is_modified(self):
            return

        # Reuse a previous lookup result to avoid repeating key comparisons.
        # None means the caller has not looked up the key yet.
        arg = key.vt
        if key_present is None:
            value = self.items.get(key)
            key_present = (
                value is not None and type(value) is not variables.DeletedVariable
            )
        if _get_literal_key(arg) is not NO_SUCH_SUBOBJ or (
            arg.source is None and arg.is_python_constant()
        ):
            guard_fn = (
                type(self).CONTAINS_GUARD
                if key_present
                else type(self).NOT_CONTAINS_GUARD
            )
            install_guard(
                self.make_guard(
                    functools.partial(
                        guard_fn,
                        key=arg.as_python_constant(),
                    )
                )
            )
        elif arg.source:
            if key_present:
                self.realize_key_vt(key)
            else:
                self.install_dict_keys_match_guard()

    def _lookup_key(
        self,
        tx: "InstructionTranslatorBase",
        key: VariableTracker,
    ) -> HashableTracker:
        _guard_nan_key(key)
        try:
            return HashableTracker(key)
        except ObservedTypeError:
            if sys.version_info < (3, 14):
                raise
            error = tx.exn_vt_stack.get_raised_exception()
            message = generic_str(tx, error).as_python_constant()
            handle_observed_exception(tx)
            raise_type_error(
                tx,
                f"cannot use '{key.python_qualified_name()}' as a dict key ({message})",
            )

    def mp_subscript_impl(
        self,
        tx: "InstructionTranslatorBase",
        key: VariableTracker,
    ) -> VariableTracker:
        # dict_subscript: https://github.com/python/cpython/blob/62a6e898e01/Objects/dictobject.c#L3673-L3706
        # Unhashable key check happens inside HashableTracker (hash_impl → TypeError).
        return self.getitem_const_raise_exception_if_absent(tx, key)

    def sq_contains_impl(
        self,
        tx: "InstructionTranslatorBase",
        item: VariableTracker,
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/dictobject.c#L4657-L4668
        if (
            not self.items
            and isinstance(item, variables.LazyConstantVariable)
            and isinstance(self.source, AttrSource)
            and self.source.member
            in (
                "_global_forward_hooks_with_kwargs",
                "_global_forward_hooks_always_called",
            )
            and self.source.base == tx.import_source("torch.nn.modules.module")
        ):
            # Empty global hook flags must not specialize a module's hook ID.
            # Other dictionaries use membership guards to allow unrelated inserts.
            item.python_type()
            self.install_dict_keys_match_guard()
            return ConstantVariable.create(False)
        key = self._lookup_key(tx, item)
        value = self.items.get(key)
        contains = value is not None and type(value) is not variables.DeletedVariable
        self.install_dict_contains_guard(tx, key, key_present=contains)
        return VariableTracker.build(tx, contains)

    def tp_iter_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        from .iter import DictIterator

        self.install_dict_order_guard(tx)
        return DictIterator(self.items)

    def tp_init_impl(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from . import DictBuiltinVariable

        temp_dict_vt = DictBuiltinVariable.call_custom_dict(tx, dict, *args, **kwargs)
        tx.output.side_effects.mutation(self)
        self.items.update(temp_dict_vt.items)  # type: ignore[attr-defined]
        return ConstantVariable.create(None)

    # NB - Both key and value are LazyVariableTrackers in the beginning. So,
    # we have to insert guards when a dict method is accessed. For this to be
    # simple, we are conservative and overguard. We skip the guard only for
    # get/__getitem__ because the key guard will be inserted by the
    # corresponding value VT. For __contains__, we add a DICT_CONTAINS guard.
    # But for all the other methods, we insert the DICT_KEYS_MATCH guard to be
    # conservative.

    def dict_items(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        self.install_dict_keys_match_guard()
        if self.source:
            tx.output.guard_on_key_order.add(self.source)
        return DictItemsVariable(self)

    def dict_keys(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        self.install_dict_keys_match_guard()
        if self.source:
            tx.output.guard_on_key_order.add(self.source)
        return DictKeysVariable(self)

    def dict_values(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        self.install_dict_keys_match_guard()
        if self.source:
            tx.output.guard_on_key_order.add(self.source)
        return DictValuesVariable(self)

    def _new_dict(
        self, items: dict[HashableTracker, VariableTracker]
    ) -> "ConstDictVariable":
        # items is already hashed (e.g. self.items.copy()), same as the
        # "cloning" case ConstDictVariable.__init__ documents/supports.
        return type(self)(items, mutation_type=ValueMutationNew())

    def dict_copy(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        self.install_dict_order_guard(tx)
        result = ConstDictVariable({}, mutation_type=ValueMutationNew())
        if self.items:
            # `PyDict_Copy` falls back to `PyDict_Merge` when `tp_iter` differs.
            result.dict_merge(tx, self)
        return result

    def dict_get(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        check_positional(tx, "get", len(args), 1, 2)
        key = self._lookup_key(tx, args[0])
        value = self.items.get(key)
        if value is None or type(value) is variables.DeletedVariable:
            self.install_dict_contains_guard(tx, key, key_present=False)
            if len(args) == 1:
                # if default is not given, return None
                return ConstantVariable.create(None)
            return args[1]
        # Key guarding - Nothing to do.
        return value

    def dict_pop(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker | None:
        if not self.is_mutable():
            return None
        check_positional(tx, "pop", len(args), 1, 2)
        if not self.items and not isinstance(self, OrderedDictVariable):
            self.install_dict_keys_match_guard()
            if len(args) == 1:
                raise_observed_exception(KeyError, tx, args=[args[0]])
            return args[1]
        # OrderedDict.pop uses PyObject_Hash, including for an empty mapping.
        if isinstance(self, OrderedDictVariable):
            _guard_nan_key(args[0])
            key = HashableTracker(args[0])
        else:
            key = self._lookup_key(tx, args[0])
        # A __dict__ deletion marker is already absent; do not record another
        # attribute mutation (which may be forbidden inside a HOP).
        is_proxy = isinstance(self.items, SideEffectsProxyDict)
        if is_proxy:
            value = self.items.get(key)
        else:
            try:
                # Authorize before the single lookup removes a tracked entry.
                if (
                    self
                    not in tx.output.side_effects.ignore_mutation_on_these_variables
                ):
                    tx.output.side_effects.check_allowed_side_effect(self)
            except Unsupported:
                # A missing key is a read, even when mutation is forbidden.
                value = self.items.get(key)
                if value is not None:
                    raise
            else:
                value = self.items.pop(key, None)
        if value is None or type(value) is variables.DeletedVariable:
            # missing item, return the default value. Install no DICT_CONTAINS guard.
            self.install_dict_contains_guard(tx, key, key_present=False)
            if len(args) == 1:
                # if default is not given, raise KeyError
                raise_observed_exception(KeyError, tx, args=[args[0]])
            return args[1]
        self.should_reconstruct_all = True
        tx.output.side_effects.mutation(self)
        if is_proxy:
            del self.items[key]
        return value

    def dict_popitem(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker | None:
        if not self.is_mutable():
            return None
        # dict.popitem() takes no args (arity enforced by MethodFlags NOARGS).
        # OrderedDict.popitem(last=) is handled by OrderedDictVariable.call_method.
        if not self.items:
            raise_observed_exception(
                KeyError,
                tx,
                args=[
                    "popitem(): dictionary is empty",
                ],
            )
        k, v = self.items.popitem()
        self.should_reconstruct_all = True
        tx.output.side_effects.mutation(self)
        return variables.TupleVariable([k.vt, v])

    def dict_clear(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        self.should_reconstruct_all = True
        tx.output.side_effects.mutation(self)
        self.items.clear()
        return ConstantVariable.create(None)

    def dict_update(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker | None:
        if not self.is_mutable():
            return None
        # CPython's dict_update_common validates args, updates from the positional
        # source, then merges keywords only if that update succeeded.
        check_positional(tx, "update", len(args), 0, 1)
        self.install_dict_keys_match_guard()
        if args or kwargs:
            tx.output.side_effects.mutation(self)
        if args:
            self.dict_update_arg(tx, args[0])
        if kwargs:
            self.dict_merge(
                tx,
                ConstDictVariable(
                    {ConstantVariable.create(k): v for k, v in kwargs.items()}
                ),
            )
        return ConstantVariable.create(None)

    def dict_update_arg(
        self, tx: "InstructionTranslatorBase", other: VariableTracker
    ) -> None:
        # dict_update_arg probes keys even for dict subclasses; dict_merge alone
        # decides whether their iteration slot permits copying dictionary storage.
        if (
            pydict_checkexact(other)
            or other.call_obj_hasattr(tx, "keys").as_python_constant()
        ):
            self.dict_merge(tx, other)
        else:
            for idx, item in enumerate(unpack_iterable(tx, other)):
                pair = unpack_iterable(tx, item)
                if len(pair) != 2:
                    raise_observed_exception(
                        ValueError,
                        tx,
                        args=[
                            "dictionary update sequence element "
                            f"#{idx} has length {len(pair)}; 2 is required"
                        ],
                    )
                self.items[HashableTracker(pair[0])] = pair[1]

    def _can_fast_merge(self, other: VariableTracker) -> bool:
        # Exact strings have no user hash/equality hooks. Exact container types
        # alone are insufficient: OrderedDict's mapping path rehashes its keys.
        return (
            (pydict_checkexact(self) or pyodict_checkexact(self))
            and isinstance(other, ConstDictVariable)
            and (pydict_checkexact(other) or pyodict_checkexact(other))
            and all(
                type(_get_literal_key(key.vt)) is str
                for mapping in (self, other)
                for key in mapping.items
            )
        )

    def dict_merge(
        self, tx: "InstructionTranslatorBase", other: VariableTracker
    ) -> None:
        # Mirrors dict_merge(..., override=1): storage when tp_iter is dict_iter,
        # otherwise PyMapping_Keys followed by subscription. The string-only
        # shortcut also avoids tracing OrderedDict's per-key Python operations.
        if isinstance(other, ConstDictVariable):
            side_effects = tx.output.side_effects
            if any(
                cls in side_effects
                and side_effects.has_pending_mutation_of_attr(
                    side_effects[cls], "__iter__"
                )
                for cls in get_type_mro_no_user_code(other.python_type())
            ):
                unimplemented(
                    gb_type="Dictionary merge after iteration slot mutation",
                    context=f"dict_merge: {other}",
                    explanation="Dynamo cannot merge a dictionary after changing its type's __iter__.",
                    hints=[*graph_break_hints.SUPPORTABLE],
                )
            if (
                isinstance(other, variables.UserDefinedObjectVariable)
                and other.cls_source
            ):
                source = mro_attr_source(
                    tx, other.python_type(), other.cls_source, "__iter__"
                )
                if source is not None:
                    install_guard(source.make_guard(GuardBuilder.ID_MATCH))
            if mro_lookup(
                other.python_type(), "__iter__"
            ) is dict.__iter__ or self._can_fast_merge(other):
                other.install_dict_order_guard(tx)
                self.items.update(other.items)
                return
        keys = generic_getattr(tx, other, "keys").call_function(tx, [], {})
        for key in unpack_iterable(tx, keys):
            self.items[HashableTracker(key)] = generic_getitem(tx, other, key)

    def dict_setdefault(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker | None:
        if not self.is_mutable():
            return None
        check_positional(tx, "setdefault", len(args), 1, 2)
        if (
            isinstance(self, OrderedDictVariable)
            and self.python_type() is not collections.OrderedDict
        ):
            # `OrderedDict_setdefault_impl` dispatches every operation for subclasses.
            contains = pysequence_contains(tx, self, args[0])
            if generic_is_true(tx, contains).as_python_constant():
                return generic_getitem(tx, self, args[0])
            value = args[1] if len(args) == 2 else ConstantVariable.create(None)
            generic_setitem(tx, self, args[0], value)
            return value
        self.install_dict_keys_match_guard()
        key = self._lookup_key(tx, args[0])
        value = self.items.get(key)
        if value is not None and type(value) is not variables.DeletedVariable:
            return value
        else:
            if len(args) == 1:
                x = ConstantVariable.create(None)
            else:
                x = args[1]
            tx.output.side_effects.mutation(self)
            if isinstance(self, OrderedDictVariable):
                # OrderedDict.setdefault looks up, then hashes again to insert.
                key = HashableTracker(args[0])
            self.items[key] = x
            return x

    def dict_reversed(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # dict.__reversed__: reverse insertion-order key iterator.
        # Not a C-level slot, so it lives in tp_methods rather than call_method.
        # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/dictobject.c (dict___reversed___impl)
        return DictKeysVariable(self).dict_view_reversed(tx, args, kwargs)

    # ref: https://github.com/python/cpython/blob/c3aefdb9eff0734058376b96fc86d89b1a345d75/Objects/dictobject.c#L5252-L5273
    tp_methods = {
        "items": Method(dict_items),
        "keys": Method(dict_keys),
        "values": Method(dict_values),
        "copy": Method(dict_copy),
        "clear": Method(dict_clear),
        "get": Method(dict_get),
        "pop": Method(dict_pop),
        "popitem": Method(dict_popitem),
        "update": Method(dict_update),
        "setdefault": Method(dict_setdefault),
        "__reversed__": Method(dict_reversed),
    }

    def unpack_var_sequence(
        self, tx: "InstructionTranslatorBase"
    ) -> list[VariableTracker]:
        self.install_dict_order_guard(tx)
        return [x.vt for x in self.items]

    def nb_or_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/3.13/Objects/dictobject.c#L4643-L4658
        self_, other_ = (other, self) if reverse else (self, other)
        if pydict_check(self_) and pydict_check(other_):
            # dict.__or__ copies and merges via internal helpers, bypassing a
            # subclass's overridden copy/update.
            new = cast(
                ConstDictVariable, ConstDictVariable.dict_copy(self_, tx, [], {})
            )
            ConstDictVariable.dict_update(new, tx, [other_], {})
            return new
        return ConstantVariable.create(NotImplemented)

    def nb_inplace_or_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/3.13/Objects/dictobject.c#L4660-L4667
        # dict.__ior__ merges via the internal helper, bypassing a subclass's
        # overridden update.
        ConstDictVariable.dict_update(self, tx, [other], {})
        return self

    def mp_ass_subscript_impl(
        self,
        tx: "InstructionTranslatorBase",
        key: VariableTracker,
        value: VariableTracker | None,
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/dictobject.c (dict_ass_sub)
        if not self.is_mutable():
            return super().mp_ass_subscript_impl(tx, key, value)

        # value=None signals delete (CPython NULL sentinel).
        self.install_dict_keys_match_guard()
        if isinstance(self, OrderedDictVariable):
            _guard_nan_key(key)
            hkey = HashableTracker(key)
        else:
            hkey = self._lookup_key(tx, key)
        tx.output.side_effects.mutation(self)
        if value is None:
            if isinstance(self.items, SideEffectsProxyDict) and hkey not in self.items:
                raise_observed_exception(KeyError, tx, args=[key])
            try:
                del self.items[hkey]
            except KeyError:
                raise_observed_exception(KeyError, tx, args=[key])
            self.should_reconstruct_all = True
        else:
            self.items[hkey] = value
        return ConstantVariable.create(None)

    def mp_length_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        """Mapping length for dict objects."""
        self.install_dict_keys_match_guard()
        return VariableTracker.build(tx, len(self.items))

    def call_obj_hasattr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> ConstantVariable:
        # dict not allow setting arbitrary attributes.  OrderedDict and
        # defaultdict allow arbitrary setattr, but not deletion of default attrs
        cls = self.python_type()
        if any(
            cls is t for t in (dict, collections.OrderedDict, collections.defaultdict)
        ):
            if hasattr(cls, name):
                return ConstantVariable.create(True)
            if cls is dict:
                return ConstantVariable.create(False)

        msg = f"hasattr on {cls} is not supported"
        unimplemented(
            gb_type="unsupported hasattr operation",
            context=f"Class {cls}",
            explanation=msg,
            hints=[
                "Consider using a regular dictionary instead",
                *graph_break_hints.SUPPORTABLE,
            ],
        )

    def clone(self, **kwargs: Any) -> VariableTracker:
        self.install_dict_keys_match_guard()
        return super().clone(**kwargs)

    def is_hashable(self) -> bool:
        return False

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        raise_type_error(tx, f"unhashable type: '{self.python_type_name()}'")

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker, op: str
    ) -> VariableTracker:
        # dict_richcompare: https://github.com/python/cpython/blob/e76aa128fe/Objects/dictobject.c#L4198
        # Only supports eq/ne; returns NotImplemented for ordering.
        from .builder import SourcelessBuilder

        if op not in ("__eq__", "__ne__"):
            return ConstantVariable.create(NotImplemented)
        # Unwrap UserDefinedDictVariable to its base ConstDictVariable.
        # This is correct because CPython's dict_equal operates on the
        # internal C struct directly (ma_used, dk_entries, _Py_dict_lookup)
        # -- it never calls __getitem__ or __len__ on dict subclasses.
        # https://github.com/python/cpython/blob/e76aa128fe/Objects/dictobject.c#L4125-L4185
        if not isinstance(other, ConstDictVariable):
            return ConstantVariable.create(NotImplemented)
        eq_result = SourcelessBuilder.create(tx, polyfills.dict___eq__).call_function(
            tx, [self, other], {}
        )
        if op == "__ne__":
            return VariableTracker.build(tx, not eq_result.as_python_constant())
        return eq_result

    def tp_getattro_impl(self, tx: "InstructionTranslatorBase", name: str):
        # DictGuardManager does not support getattr_manager for plain dicts,
        # so AttrSource chains through a dict source break guard creation.
        # Return CallMethodVariable directly for methods, bypassing the
        # MRO walk in object_generic_getattr that would create that chain.
        type_attr = mro_lookup(self.python_type(), name)
        if type_attr is not NO_SUCH_SUBOBJ and _is_method_type(type_attr):
            return variables.CallMethodVariable(self, name)
        return super().tp_getattro_impl(tx, name)


class OrderedDictVariable(ConstDictVariable):
    _cpython_type = collections.OrderedDict
    # self.items is actually a collections.OrderedDict at runtime for this
    # class (see ConstDictVariable.__init__'s storage_cls), which is what
    # gives move_to_end/popitem(last=) below their real backing storage.
    items: "collections.OrderedDict[HashableTracker, VariableTracker]"  # pyrefly: ignore[bad-override]

    def install_dict_order_guard(
        self,
        tx: "InstructionTranslatorBase",
        *,
        view: "DictViewVariable | None" = None,
    ) -> None:
        super().install_dict_order_guard(tx, view=view)
        # Even one linked key needs a guard: raw dict writes can leave the same
        # storage without its OrderedDict link.
        if (
            self.source
            and not is_constant_source(self.source)
            and issubclass(self.python_type(), collections.OrderedDict)
            and (view is None or view.is_ordered)
            and self.original_items
        ):
            install_guard(self.make_guard(GuardBuilder.ORDERED_DICT_ORDER_MATCH))

    # OrderedDict-exclusive C methods, declared like CPython's odict tp_methods.
    # move_to_end is odict-only; popitem honors last= (vs dict's LIFO popitem).
    # OrderedDict exposes no tp_getset / tp_members.
    # https://github.com/python/cpython/blob/v3.13.0/Objects/odictobject.c

    def dict_copy(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        self.install_dict_order_guard(tx)
        if not pyodict_checkexact(self):
            # odict_copy calls the subclass constructor, then its get/setitem hooks.
            cls = VariableTracker.build(tx, type).call_function(tx, [self], {})
            result = cls.call_function(tx, [], {})
            # CPython's _odict_FOREACH walks nodes without comparing keys.
            for key in _iter_detecting_mutation(
                HashableTracker.iter_keys_by_identity(self.items)
            ):
                value = generic_getitem(tx, self, key.vt)
                generic_setitem(tx, result, key.vt, value)
            return result
        return self._new_dict(self.items.copy())

    def tp_init_impl(
        self: ConstDictVariable,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if len(args) > 1:
            # CPython corrected the plural in 3.13.8 (gh-138517).
            suffix = "s" if sys.version_info < (3, 13, 8) else ""
            raise_type_error(
                tx, f"expected at most 1 argument{suffix}, got {len(args)}"
            )
        return OrderedDictVariable.odict_update(self, tx, args, kwargs)

    def odict_update(
        self: ConstDictVariable,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if len(args) > 1:
            raise_type_error(
                tx, f"update() takes at most 1 positional argument ({len(args)} given)"
            )
        if (
            pyodict_checkexact(self)
            and self.is_mutable()
            and self._can_fast_merge(args[0] if args else ConstDictVariable({}))
        ):
            return cast(
                VariableTracker, ConstDictVariable.dict_update(self, tx, args, kwargs)
            )
        return tx.inline_user_function_return(
            VariableTracker.build(tx, polyfills.odict_update), [self, *args], kwargs
        )

    def nb_inplace_or_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker
    ) -> VariableTracker:
        # odict_inplace_or uses mutablemapping_update_arg, including __setitem__.
        self.odict_update(tx, [other], {})
        return self

    def move_to_end(
        self: "OrderedDictVariable",
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        self.install_dict_keys_match_guard()
        tx.output.side_effects.mutation(self)
        if args[0] not in self:
            raise_observed_exception(KeyError, tx)

        last = True
        if len(args) == 2 and args[0].is_python_constant():
            last = args[1].as_python_constant()
        if "last" in kwargs and kwargs["last"].is_python_constant():
            last = kwargs["last"].as_python_constant()

        self.items.move_to_end(HashableTracker(args[0]), last=last)
        return ConstantVariable.create(None)

    def popitem(
        self: "OrderedDictVariable",
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        self.install_dict_order_guard(tx)
        if not self.items:
            raise_observed_exception(
                KeyError, tx, args=["popitem(): dictionary is empty"]
            )

        last = True
        if len(args) == 1 and args[0].is_python_constant():
            last = args[0].as_python_constant()
        if "last" in kwargs and kwargs["last"].is_python_constant():
            last = kwargs["last"].as_python_constant()

        k, v = self.items.popitem(last=last)
        self.should_reconstruct_all = True
        tx.output.side_effects.mutation(self)
        return variables.TupleVariable([k.vt, v])

    def dict_keys(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        return super().dict_keys(tx, args, kwargs).clone(is_ordered=True)

    def dict_values(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        return super().dict_values(tx, args, kwargs).clone(is_ordered=True)

    def dict_items(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        return super().dict_items(tx, args, kwargs).clone(is_ordered=True)

    def dict_reversed(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        return DictKeysVariable(self, is_ordered=True).dict_view_reversed(
            tx, args, kwargs
        )

    # ref: https://github.com/python/cpython/blob/c3aefdb9eff0734058376b96fc86d89b1a345d75/Objects/odictobject.c#L1378-L1403
    tp_methods = {
        "copy": Method(dict_copy),
        "update": Method(odict_update),
        "move_to_end": Method(move_to_end),
        "popitem": Method(popitem),
        "keys": Method(dict_keys),
        "values": Method(dict_values),
        "items": Method(dict_items),
        "__reversed__": Method(dict_reversed),
    }

    def as_python_constant(self) -> "collections.OrderedDict[Any, Any]":
        return collections.OrderedDict(super().as_python_constant())

    @staticmethod
    def _ordered_dict_repr(items: list[tuple[str, str]]) -> str:
        if not items:
            return "OrderedDict()"
        if sys.version_info >= (3, 12):
            return (
                "OrderedDict({"
                + ", ".join(f"{key}: {value}" for key, value in items)
                + "})"
            )
        return (
            "OrderedDict(["
            + ", ".join(f"({key}, {value})" for key, value in items)
            + "])"
        )

    def debug_repr(self) -> str:
        items = [(k.vt.debug_repr(), v.debug_repr()) for k, v in self.items.items()]
        return self._ordered_dict_repr(items)

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # Python < 3.12 uses the historical list-of-pairs form, while 3.12+
        # uses dict-style formatting.
        # https://github.com/python/cpython/blob/v3.13.3/Objects/odictobject.c#L1385-L1418
        self.install_dict_order_guard(tx)
        with repr_guard(self) as recursive:
            if recursive:
                return VariableTracker.build(tx, "...")
            items = [
                (tracked_repr(tx, key.vt), tracked_repr(tx, value))
                for key, value in self.items.items()
            ]
        return VariableTracker.build(tx, self._ordered_dict_repr(items))

    def nb_or_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # OrderedDict.__or__/__ror__ preserve the OrderedDict type, unlike
        # dict.__or__ which returns a plain dict for a subclass operand.
        # ref: https://github.com/python/cpython/blob/3.13/Lib/collections/__init__.py#L327-L339
        if not pydict_check(other):
            return ConstantVariable.create(NotImplemented)
        # CPython's odict_or calls the subclass constructor, then merges via
        # its internal helper without calling overridden copy/update methods.
        left, right = (other, self) if reverse else (self, other)
        owner = (
            left if issubclass(left.python_type(), collections.OrderedDict) else self
        )
        if owner.python_type() is collections.OrderedDict:
            new = OrderedDictVariable({}, mutation_type=ValueMutationNew())
            OrderedDictVariable.tp_init_impl(new, tx, [left], {})
        else:
            cls = VariableTracker.build(tx, type).call_function(tx, [owner], {})
            new = cast(ConstDictVariable, cls.call_function(tx, [left], {}))
        OrderedDictVariable.tp_init_impl(new, tx, [right], {})
        return new

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # emit `OrderedDict(constructed_dict)`
        codegen.add_push_null(
            lambda: codegen.extend_output(
                [
                    codegen.create_load_python_module(collections),
                    codegen.create_load_attr("OrderedDict"),
                ]
            )
        )
        if self._contains_self_reference() and self.source is None:
            codegen.extend_output(
                [
                    *create_call_function(0, False),
                    create_dup_top(),
                ]
            )
            codegen.add_cache(self)

            codegen.append_output(create_dup_top())
            codegen.load_method("update")
            self.reconstruct_kvs_into_new_dict(codegen)
            codegen.extend_output(
                [
                    *create_call_method(1),
                    create_instruction("POP_TOP"),
                ]
            )
        else:
            self.reconstruct_kvs_into_new_dict(codegen)
            codegen.extend_output(create_call_function(1, False))

    def reconstruct_pycode(self, codegen) -> str:
        raise NotImplementedError


class MappingProxyVariable(VariableTracker):
    # PyDictProxy_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/descrobject.c#L1995
    _cpython_type = types.MappingProxyType

    # proxies to the VariableTracker of the wrapped mapping
    def __init__(self, dv_dict: VariableTracker, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.dv_dict = dv_dict

    def python_type(self) -> type:
        return types.MappingProxyType

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        # mappingproxy.__hash__ delegates to the underlying dict, which
        # raises TypeError. Mirror that behavior.
        return self.dv_dict.hash_impl(tx)

    def unpack_var_sequence(
        self, tx: "InstructionTranslatorBase"
    ) -> list[VariableTracker]:
        return self.dv_dict.unpack_var_sequence(tx)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # load types.MappingProxyType
        if self.source:
            msg = (
                f"Preexisting MappingProxyVariable (source: {self.source}) cannot be reconstructed "
                "because the connection to the original dict will be lost."
            )
            unimplemented(
                gb_type="mapping proxy cannot be reconstructed",
                context=f"Source: {self.source}",
                explanation=msg,
                hints=[
                    "Use a mapping proxy constructed in the same `torch.compile` region.",
                    *graph_break_hints.SUPPORTABLE,
                ],
            )
        codegen.add_push_null(
            lambda: codegen.extend_output(
                [
                    codegen.create_load_python_module(types),
                    codegen.create_load_attr("MappingProxyType"),
                ]
            )
        )
        codegen(self.dv_dict)
        codegen.extend_output(create_call_function(1, False))

    def mp_subscript_impl(
        self,
        tx: "InstructionTranslatorBase",
        key: VariableTracker,
    ) -> VariableTracker:
        # mappingproxy_getitem: https://github.com/python/cpython/blob/62a6e898e01/Objects/descrobject.c#L1052-L1056
        # TODO(follow-up): add tests for invalid key type, missing key
        return self.dv_dict.mp_subscript_impl(tx, key)

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        return self.dv_dict.call_method(tx, name, args, kwargs)

    def tp_iter_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        return self.dv_dict.tp_iter_impl(tx)

    def sq_contains_impl(
        self, tx: "InstructionTranslatorBase", item: VariableTracker
    ) -> VariableTracker:
        # https://github.com/python/cpython/blob/60403a5409ff/Objects/descrobject.c#L1087-L1095
        return self.dv_dict.sq_contains_impl(tx, item)

    def mp_length_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        return self.dv_dict.mp_length_impl(tx)

    def _check_class_attr_mutation(self, tx: "InstructionTranslatorBase") -> None:
        # The wrapped mapping is built from the real object, so it does not see
        # class attribute writes made earlier in the trace until they replay.
        side_effects = tx.output.side_effects
        for vt in side_effects.store_attr_mutations:
            if not isinstance(vt, variables.UserDefinedClassVariable):
                continue
            if not side_effects.has_pending_mutation(vt):
                continue
            # type's own __dict__ getter, so a metaclass __dict__ is bypassed.
            class_dict = type.__dict__["__dict__"].__get__(vt.value)
            mapping = gc.get_referents(class_dict)[0]
            if mapping in side_effects and side_effects[mapping] is self.dv_dict:
                unimplemented(
                    gb_type="mapping proxy of class with pending attribute mutation",
                    context=f"class: {vt.value}",
                    explanation="The class __dict__ does not reflect class attributes set earlier in the traced code.",
                    hints=[*graph_break_hints.SUPPORTABLE],
                )

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        self._check_class_attr_mutation(tx)
        return VariableTracker.build(
            tx, f"mappingproxy({tracked_repr(tx, self.dv_dict)})"
        )

    def tp_str_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        self._check_class_attr_mutation(tx)
        return generic_str(tx, self.dv_dict)

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker, op: str
    ) -> VariableTracker:
        # mappingproxy_richcompare delegates to the underlying mapping:
        # https://github.com/python/cpython/blob/60403a5409ff/Objects/descrobject.c#L1231-L1236
        from .object_protocol import generic_richcompare

        return generic_richcompare(tx, self.dv_dict, other, op)

    def call_obj_hasattr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> ConstantVariable:
        if self.python_type() is types.MappingProxyType:
            return VariableTracker.build(tx, name in types.MappingProxyType.__dict__)
        return super().call_obj_hasattr(tx, name)


class NNModuleHooksDictVariable(OrderedDictVariable):
    # Defer hook-ID guards so equivalent modules can reuse normal hook dispatch.
    # nn.Module hook dicts are OrderedDict, so back storage/reconstruct with it.
    def is_python_constant(self) -> bool:
        # Empty hook dictionaries contain no IDs to specialize.
        return not self.items

    def install_dict_keys_match_guard(self) -> None:
        # Truth and view length need a guard without specializing hook IDs.
        if self.source and not is_constant_source(self.source):
            install_guard(self.make_guard(GuardBuilder.SEQUENCE_LENGTH))

    def install_dict_order_guard(
        self,
        tx: "InstructionTranslatorBase",
        *,
        view: "DictViewVariable | None" = None,
    ) -> None:
        if view is None or view.kv == "keys":
            ConstDictVariable.install_dict_keys_match_guard(self)
        super().install_dict_order_guard(tx, view=view)

    def sq_contains_impl(
        self, tx: "InstructionTranslatorBase", item: VariableTracker
    ) -> VariableTracker:
        if not self.items and isinstance(item, variables.LazyConstantVariable):
            item.python_type()
            if self.source:
                install_guard(self.make_guard(GuardBuilder.SEQUENCE_LENGTH))
            return ConstantVariable.create(False)
        source = item.source
        if (
            isinstance(source, ConstDictKeySource)
            and isinstance(source.base, AttrSource)
            and isinstance(self.source, AttrSource)
            and source.base.base == self.source.base
            and not tx.output.side_effects.is_modified(self)
        ):
            key = _get_literal_key(item)
            keys = [_get_literal_key(k.vt) for k in self.items]
            if type(key) is int and all(type(k) is int for k in keys):
                item.python_type()
                for stored_key in self.items:
                    stored_key.vt.python_type()
                # Normal hook dispatch tests an ID against the same module's
                # kwargs/always-call flags. Guard that relationship, not the ID.
                install_guard(
                    self.source.base.make_guard(
                        functools.partial(
                            GuardBuilder.NN_MODULE_HOOKS_DICT_CONTAINS,
                            hooks_name=source.base.member,
                            flags_name=self.source.member,
                        )
                    )
                )
                return ConstantVariable.create(key in keys)
        return super().sq_contains_impl(tx, item)


class DictViewVariable(VariableTracker):
    """
    Models _PyDictViewObject

    This is an "abstract" class. Subclasses will override kv and the items method
    """

    kv: str | None = None

    def __init__(
        self, dv_dict: ConstDictVariable, *, is_ordered: bool = False, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if self.kv not in ("keys", "values", "items"):
            raise AssertionError(
                f"Expected kv to be 'keys', 'values', or 'items', got {self.kv!r}"
            )
        if not isinstance(dv_dict, ConstDictVariable):
            raise AssertionError(f"Expected ConstDictVariable, got {type(dv_dict)}")
        self.dv_dict = dv_dict
        # The creating descriptor determines the view, not the receiver's type.
        self.is_ordered = is_ordered

    @property
    def view_items(self) -> Any:
        if self.kv is None:
            raise AssertionError("kv must not be None")
        if not self.is_ordered and isinstance(self.dv_dict.items, dict):
            return getattr(dict, self.kv)(self.dv_dict.items)
        return getattr(self.dv_dict.items, self.kv)()

    def install_order_guard(self, tx: "InstructionTranslatorBase") -> None:
        self.dv_dict.install_dict_order_guard(tx, view=self)

    def is_hashable(self) -> bool:
        return False

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        raise_type_error(tx, f"unhashable type: '{self.python_type_name()}'")

    @property
    def view_items_vt(self) -> list[VariableTracker]:
        # Returns an iterable of the unpacked items
        # Implement in the subclasses
        raise NotImplementedError

    def unpack_var_sequence(
        self, tx: "InstructionTranslatorBase"
    ) -> list[VariableTracker]:
        self.install_order_guard(tx)
        return self.view_items_vt

    def reconstruct(self, codegen: "PyCodegen") -> None:
        if self.kv is None:
            raise AssertionError("kv must not be None for reconstruct")

        def load_method() -> None:
            if self.is_ordered:
                codegen.load_import_from("collections", "OrderedDict")
            else:
                codegen.load_import_from("builtins", "dict")
            codegen.append_output(codegen.create_load_attr(self.kv))

        codegen.add_push_null(load_method)
        codegen(self.dv_dict)
        codegen.extend_output(create_call_function(1, False))

    def call_obj_hasattr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> ConstantVariable:
        if self.kv is None:
            raise AssertionError("kv must not be None for call_obj_hasattr")
        return ConstantVariable.create(
            mro_lookup(self.python_type(), name) is not NO_SUCH_SUBOBJ
        )

    # dictview_mapping getset returns a read-only mappingproxy of the underlying
    # dict. https://github.com/python/cpython/blob/v3.13.0/Objects/dictobject.c#L5032-L5040
    tp_getset = {
        "mapping": GetSet(
            lambda s, _: MappingProxyVariable(s.dv_dict), readonly_setter
        ),
    }

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # https://github.com/python/cpython/blob/v3.13.3/Objects/dictobject.c#L5741-L5765
        self.install_order_guard(tx)
        with repr_guard(self) as recursive:
            if recursive:
                return VariableTracker.build(tx, "...")
            if self.kv == "keys":
                items = ", ".join(tracked_repr(tx, key.vt) for key in self.view_items)
            elif self.kv == "values":
                items = ", ".join(tracked_repr(tx, value) for value in self.view_items)
            else:  # items
                items = ", ".join(
                    f"({tracked_repr(tx, key.vt)}, {tracked_repr(tx, value)})"
                    for key, value in self.view_items
                )
        return VariableTracker.build(tx, f"{self.python_type_name()}([{items}])")

    def dict_view_reversed(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # dict_keys/values/items __reversed__: reverse insertion order.
        # Not a C-level slot, so it lives in tp_methods rather than call_method.
        self.install_order_guard(tx)
        return variables.ListIteratorVariable(
            list(reversed(self.view_items_vt)),
            mutation_type=ValueMutationNew(),
        )

    # ref: https://github.com/python/cpython/blob/c3aefdb9eff0734058376b96fc86d89b1a345d75/Objects/dictobject.c#L6992-L6998
    tp_methods = {
        "__reversed__": Method(dict_view_reversed),
    }

    def sq_length_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        """Sequence length for dict view objects."""
        self.dv_dict.install_dict_keys_match_guard()
        return VariableTracker.build(tx, len(self.dv_dict.items))

    def nb_or_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/dictobject.c#L6142-L6155 (dictviews_or)
        s = VariableTracker.build(tx, set).call_function(tx, [self], {})
        s.call_method(tx, "update", [other], {})
        return s

    def nb_subtract_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/dictobject.c#L6036 (dictviews_sub)
        self_, other_ = (other, self) if reverse else (self, other)
        s = VariableTracker.build(tx, set).call_function(tx, [self_], {})
        s.call_method(tx, "difference_update", [other_], {})
        return s

    def nb_and_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/3.13/Objects/dictobject.c#L6105-L6188 (_PyDictView_Intersect)
        s = VariableTracker.build(tx, set).call_function(tx, [self], {})
        s.call_method(tx, "intersection_update", [other], {})
        return s

    def nb_xor_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/3.13/Objects/dictobject.c#L6305-L6325 (dictviews_xor)
        s = VariableTracker.build(tx, set).call_function(tx, [self], {})
        s.call_method(tx, "symmetric_difference_update", [other], {})
        return s


class DictKeysVariable(DictViewVariable):
    # PyDictKeys_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/dictobject.c#L6365
    _cpython_type = dict_keys

    kv = "keys"

    @property
    def set_items(self) -> set[VariableTracker]:
        return set(self.view_items)

    @property
    def view_items_vt(self) -> list[VariableTracker]:
        # Returns an iterable of the unpacked items
        return [x.vt for x in self.view_items]

    def python_type(self) -> type:
        return odict_keys if self.is_ordered else dict_keys

    def tp_iter_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        from .iter import DictKeysIterator

        self.install_order_guard(tx)
        return DictKeysIterator(
            self.dv_dict.items, use_dict_storage=not self.is_ordered
        )

    def debug_repr(self) -> str:
        if not self.view_items:
            return "dict_keys([])"
        else:
            items: list[str] = []
            for k in self.view_items:
                key_str = _item_debug_repr(k.vt)
                items.append(key_str)
            return "dict_keys([" + ", ".join(items) + "])"

    def sq_contains_impl(
        self, tx: "InstructionTranslatorBase", item: VariableTracker
    ) -> VariableTracker:
        # dictkeys_contains hands dv_dict to PyDict_Contains, a struct lookup,
        # so a dict subclass's __contains__ never runs -- call the base slot
        # rather than dispatching on dv_dict's type.
        # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/dictobject.c#L5998-L6005
        return ConstDictVariable.sq_contains_impl(self.dv_dict, tx, item)

    def isdisjoint(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from .builder import SourcelessBuilder

        return SourcelessBuilder.create(
            tx, polyfills.dictview_isdisjoint
        ).call_function(tx, [self, *args], kwargs)

    tp_methods = {
        "isdisjoint": Method(isdisjoint),
    }

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker, op: str
    ) -> VariableTracker:
        # dictview_richcompare: accepts set/frozenset and dict_keys/dict_items.
        # https://github.com/python/cpython/blob/e76aa128fe/Objects/dictobject.c#L5952-L6010
        # Uses a polyfill with len() and ``in`` so that Dynamo traces through
        # sq_length/sq_contains, matching CPython's use of PyObject_Size and
        # PySequence_Contains.
        from .builder import SourcelessBuilder

        if not _is_set_or_dictview(other):
            return ConstantVariable.create(NotImplemented)
        return SourcelessBuilder.create(
            tx, polyfills.dictview_richcompare
        ).call_function(
            tx,
            [
                SourcelessBuilder.create(tx, cmp_name_to_op_mapping[op]),
                self,
                other,
            ],
            {},
        )


class DictValuesVariable(DictViewVariable):
    # PyDictValues_Type: https://github.com/python/cpython/blob/e76aa128fe/Objects/dictobject.c#L6615
    _cpython_type = dict_values

    # dict_values is hashable (identity hash), unlike dict_keys/dict_items.
    #
    # CPython only inherits tp_hash from the base when BOTH tp_richcompare
    # AND tp_hash are NULL on the type
    # (https://github.com/python/cpython/blob/e76aa128fe/Objects/typeobject.c#L7665-L7679).
    # dict_keys/dict_items set tp_richcompare (for set-like comparisons),
    # so their tp_hash is NOT inherited — it stays NULL, and
    # type_ready_set_hash sets it to PyObject_HashNotImplemented (unhashable)
    # (https://github.com/python/cpython/blob/e76aa128fe/Objects/typeobject.c#L8066-L8085).
    # dict_values does NOT set tp_richcompare
    # (https://github.com/python/cpython/blob/e76aa128fe/Objects/dictobject.c#L6630),
    # so it inherits object's tp_hash (PyObject_GenericHash) — hashable.
    #
    # Override DictViewVariable.hash_impl to restore the base identity hash.
    kv = "values"

    # dict.values() do not implement tp_as_number
    nb_or_impl = None  # type: ignore[bad-override]
    nb_inplace_or_impl = None  # type: ignore[bad-override]
    nb_subtract_impl = None  # type: ignore[bad-override]
    nb_inplace_subtract_impl = None  # type: ignore[bad-override]
    nb_and_impl = None  # type: ignore[bad-override]
    nb_inplace_and_impl = None  # type: ignore[bad-override]
    nb_xor_impl = None  # type: ignore[bad-override]
    nb_inplace_xor_impl = None  # type: ignore[bad-override]

    def is_hashable(self) -> bool:
        return True

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        return super(DictViewVariable, self).hash_impl(tx)

    @property
    def view_items_vt(self) -> list[VariableTracker]:
        return list(self.view_items)

    def python_type(self) -> type:
        return odict_values if self.is_ordered else dict_values

    def tp_iter_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        from .iter import DictValuesIterator

        self.install_order_guard(tx)
        return DictValuesIterator(
            self.dv_dict.items, use_dict_storage=not self.is_ordered
        )

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker, op: str
    ) -> VariableTracker:
        # dict_values has no tp_richcompare (inherits object's).
        from .object_protocol import object_richcompare

        return object_richcompare(self, tx, other, op)

    def debug_repr(self) -> str:
        if not self.view_items:
            return "dict_values([])"
        else:
            items: list[str] = []
            for v in self.view_items:
                val_str = _item_debug_repr(v)
                items.append(val_str)
            return "dict_values([" + ", ".join(items) + "])"


class DictItemsVariable(DictViewVariable):
    # PyDictItems_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/dictobject.c#L6477
    _cpython_type = dict_items

    kv = "items"

    @property
    def set_items(self) -> set["HashableTracker"]:
        return {
            HashableTracker(variables.TupleVariable([k.vt, v]))
            for k, v in self.view_items
        }

    @property
    def view_items_vt(self) -> list[VariableTracker]:
        # Returns an iterable of the unpacked items
        return [variables.TupleVariable([k.vt, v]) for k, v in self.view_items]

    def python_type(self) -> type:
        return odict_items if self.is_ordered else dict_items

    def debug_repr(self) -> str:
        if not self.view_items:
            return "dict_items([])"
        else:
            items: list[str] = []
            for k, v in self.view_items:
                key_str = _item_debug_repr(k.vt)
                val_str = _item_debug_repr(v)
                items.append(f"({key_str}, {val_str})")
            return "dict_items([" + ", ".join(items) + "])"

    def sq_contains_impl(
        self, tx: "InstructionTranslatorBase", item: VariableTracker
    ) -> VariableTracker:
        # CPython's dictitems_contains rejects non-pairs before hashing the key.
        if not isinstance(item, variables.TupleVariable) or len(item.items) != 2:
            return ConstantVariable.create(False)
        key, val = item.items
        key_ht = self.dv_dict._lookup_key(tx, key)
        stored = self.dv_dict.items.get(key_ht)
        present = stored is not None and type(stored) is not variables.DeletedVariable
        self.dv_dict.install_dict_contains_guard(tx, key_ht, key_present=present)
        if not present:
            return ConstantVariable.create(False)
        return generic_richcompare_bool(tx, stored, val, "__eq__")

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker, op: str
    ) -> VariableTracker:
        # dictview_richcompare: accepts set/frozenset and dict_keys/dict_items.
        # https://github.com/python/cpython/blob/e76aa128fe/Objects/dictobject.c#L5952-L6010
        # Uses a polyfill with len() and ``in`` so that Dynamo traces through
        # sq_length/sq_contains, matching CPython's use of PyObject_Size and
        # PySequence_Contains.
        from .builder import SourcelessBuilder

        if not _is_set_or_dictview(other):
            return ConstantVariable.create(NotImplemented)
        return SourcelessBuilder.create(
            tx, polyfills.dictview_richcompare
        ).call_function(
            tx,
            [
                SourcelessBuilder.create(tx, cmp_name_to_op_mapping[op]),
                self,
                other,
            ],
            {},
        )

    def tp_iter_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        from .iter import DictItemsIterator

        self.install_order_guard(tx)
        return DictItemsIterator(
            self.dv_dict.items, use_dict_storage=not self.is_ordered
        )


kV = HashableTracker | str


class SideEffectsProxyDict(collections.abc.MutableMapping[kV, VariableTracker]):
    """
    A proxy dict that allows us to track mutations to the dict using side
    effects table as storage.
    """

    @staticmethod
    def get_example_value_dict(vt: VariableTracker) -> dict[str, object]:
        if istype(vt, variables.NestedUserFunctionVariable):
            # NestedUserFunctionVariable is created with MAKE_FUNCTION and its
            # __dict__ starts empty. Any mutation will actually be recorded in
            # the side effects table.
            return {}
        elif isinstance(vt, variables.LocalGeneratorFunctionVariable):
            return SideEffectsProxyDict.get_example_value_dict(vt.vt)
        elif isinstance(vt, variables.UserMethodVariable):
            # method_getattro forwards __dict__ to __func__; a bound method has
            # no __dict__ of its own. Read-only: CPython rejects writes through
            # a method ("'method' object has no attribute 'x'"), and Dynamo
            # matches that because UserMethodVariable does not override
            # get_value_for_setattr, so the VT default declines the store
            # rather than mutating the function's shared __dict__.
            return object.__getattribute__(vt.get_function(), "__dict__")
        elif istype(vt, variables.ExceptionVariable):
            # Synthetic exceptions created during tracing have no backing
            # Python object; their __dict__ starts empty and any custom
            # attribute is recorded in the side effects table.
            return {}
        else:
            value = vt.get_real_python_backed_value()
            if value is not NO_SUCH_SUBOBJ:
                if isinstance(vt, variables.UserDefinedObjectVariable):
                    return vt._getattr_static("__dict__")  # type: ignore[bad-return]
                else:
                    return object.__getattribute__(value, "__dict__")
            else:
                unimplemented(
                    gb_type="unsupported variable type for __dict__ access",
                    context=f"VariableTracker type: {type(vt)}",
                    explanation=f"Dynamo does not know how to get __dict__ from {type(vt)}",
                    hints=[
                        *graph_break_hints.DYNAMO_BUG,
                    ],
                )

    @staticmethod
    def get_value___dict__(
        tx: "InstructionTranslatorBase", vt: VariableTracker
    ) -> dict[str, VariableTracker]:
        # GENERIC_SETATTR on "__dict__" means the whole instance dict was
        # replaced through the __dict__ getset descriptor (obj.__dict__ = ...).
        # Per-key obj.__dict__[name] mutations are tracked as INSTANCE_DICT.
        if isinstance(
            vt, variables.UserDefinedObjectVariable
        ) and tx.output.side_effects.has_pending_mutation_of_attr(
            vt, "__dict__", AttrMutationKind.GENERIC_SETATTR
        ):
            dict_vt = tx.output.side_effects.load_attr(vt, "__dict__")
            if isinstance(dict_vt, ConstDictVariable):
                result = {}
                for key, value in dict_vt.items.items():
                    try:
                        key_value = key.vt.as_python_constant()
                    except NotImplementedError:
                        unimplemented(
                            gb_type="non-constant key in object __dict__",
                            context=f"key={key.vt}",
                            explanation="Dynamo expects object __dict__ replacement keys to be constants.",
                            hints=[*graph_break_hints.SUPPORTABLE],
                        )
                    if not isinstance(key_value, str):
                        unimplemented(
                            gb_type="non-string key in object __dict__",
                            context=f"key={key_value!r}",
                            explanation="Dynamo expects object __dict__ keys to be strings.",
                            hints=[*graph_break_hints.USER_ERROR],
                        )
                    result[key_value] = value
                return result
            # Other replacement forms are not materialized here yet. Leave
            # them on the normal example-value path until Dynamo models their
            # contents explicitly.

        example_value_dict = SideEffectsProxyDict.get_example_value_dict(vt)
        dict_source = vt.source and AttrSource(vt.source, "__dict__")
        if vt.source and issubclass(vt.python_type(), type):
            # A class __dict__ is a mappingproxy; guard its items through tp_dict.
            dict_source = TypeDictSource(vt.source)

        for key in example_value_dict:
            # DictGetItemSource and ConstantVariable model instance dict keys as
            # literals, which excludes the str subclasses PyUnicode_Check allows.
            if not variables.ConstantVariable.is_literal(key):
                unimplemented(
                    gb_type="non-literal key in object __dict__",
                    context=f"key={key!r}, type={type(key)}",
                    explanation="Dynamo models object __dict__ keys as literal constants.",
                    hints=[*graph_break_hints.SUPPORTABLE],
                )

        return {
            key: VariableTracker.build(
                tx,
                value,
                source=dict_source and DictGetItemSource(dict_source, key),
            )
            for key, value in example_value_dict.items()
        }

    def __init__(self, item: VariableTracker, tx: "InstructionTranslatorBase") -> None:
        self.item = item
        self.output_graph_weakref = weakref.ref(tx.output)
        self.item_dict = self.get_value___dict__(tx, item)

    @property
    def side_effects(self) -> "SideEffects":
        """
        Resolved per access, not captured in __init__: speculate_subgraph
        replaces the table after tracing a HOP body, so a captured one would
        be stale and its mutations never replayed. It would also form a cycle
        (the table owns this proxy via store_attr_mutations) that keeps
        keepalive tensors alive until a full GC.
        """
        output_graph = self.output_graph_weakref()
        if output_graph is None:
            raise AssertionError("output_graph weakref is dead")
        return output_graph.side_effects

    def _maybe_unwrap_key(self, key: kV) -> str:
        Hasher = HashableTracker
        return key.vt.as_python_constant() if istype(key, Hasher) else key

    def side_effects_table(self) -> dict[str, VariableTracker]:
        return {
            name: value
            for name, value in self.side_effects.store_attr_mutations.get(
                self.item, {}
            ).items()
            if self.side_effects.has_pending_mutation_of_attr(
                self.item, name, AttrMutationKind.INSTANCE_DICT
            )
        }

    def __getitem__(self, key: kV) -> VariableTracker:
        name = self._maybe_unwrap_key(key)
        if self.side_effects.has_pending_mutation_of_attr(
            self.item, name, AttrMutationKind.INSTANCE_DICT
        ):
            return self.side_effects.load_attr(self.item, name, deleted_ok=True)
        return self.item_dict[name]

    def __setitem__(self, key: kV, value: VariableTracker) -> None:
        # Find a way to not hash the key using HashableTracker. CPython's
        # instance __dict__ accepts arbitrary hashable keys when set via the
        # mapping API (only attribute access via setattr requires str), and
        # instance-dict mutations replay with object_setattr_ignore_descriptor
        # (a plain __dict__ store), so a non-str constant key is fine.
        name = self._maybe_unwrap_key(key)
        self.side_effects.store_instance_dict_attr(self.item, name, value)

    def __delitem__(self, key: kV) -> None:
        name = self._maybe_unwrap_key(key)
        self.side_effects.store_instance_dict_attr(
            self.item, name, variables.DeletedVariable()
        )

    def __contains__(self, key: kV) -> bool:  # type: ignore[bad-override]
        name = self._maybe_unwrap_key(key)
        if self.side_effects.has_pending_mutation_of_attr(
            self.item, name, AttrMutationKind.INSTANCE_DICT
        ):
            value = self.side_effects.load_attr(self.item, name, deleted_ok=True)
            return not isinstance(value, variables.DeletedVariable)
        return name in self.item_dict

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __iter__(self) -> Iterator[HashableTracker]:
        Hasher = HashableTracker
        d = self.side_effects_table()
        for k, v in d.items():
            if isinstance(v, variables.DeletedVariable):
                continue
            yield Hasher(ConstantVariable.create(k))

        for k, v in self.item_dict.items():
            if k not in d:
                yield Hasher(ConstantVariable.create(k))


class DunderDictVariable(ConstDictVariable):
    """represents object.__dict__"""

    # Overrides ConstDictVariable.items: this class backs its storage with
    # the side-effects table (via SideEffectsProxyDict) instead of a real
    # dict, so self.items has a different runtime shape here.
    items: SideEffectsProxyDict  # pyrefly: ignore[bad-override]

    @classmethod
    def create(
        cls,
        tx: "InstructionTranslatorBase",
        vt: VariableTracker,
    ) -> "DunderDictVariable":
        mutation = AttributeMutationExisting() if vt.source else AttributeMutationNew()
        source = vt.source and AttrSource(vt.source, "__dict__")

        return cls(
            vt,
            tx=tx,
            mutation_type=mutation,
            source=source,
        )

    def __init__(
        self,
        vt: VariableTracker,
        tx: "InstructionTranslatorBase",
        **kwargs: Any,
    ) -> None:
        super().__init__({}, **kwargs)
        self.items = SideEffectsProxyDict(vt, tx)

    def setitem(self, name: str, value: VariableTracker) -> None:
        self.items[name] = value

    def delitem(self, name: str) -> None:
        if not self.contains(name):
            raise KeyError(name)
        del self.items[name]

    def getitem(self, name: str) -> VariableTracker:
        return self.items[name]

    def contains(self, name: str) -> bool:
        return name in self.items

    def getitem_or_default(
        self,
        name: str,
        default: Callable[[], VariableTracker],
    ) -> VariableTracker:
        if self.contains(name):
            return self.getitem(name)
        else:
            value = default()
            self.items[name] = value
            return value

    def dict_copy(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> "ConstDictVariable":
        # self.items is a SideEffectsProxyDict keyed by HashableTracker, which
        # is not a VariableTracker; dict()'s overloads (constrained here to
        # ConstDictVariable.__init__'s VariableTracker key type) have no shape
        # for a HashableTracker-keyed mapping, even though __init__ accepts it.
        return ConstDictVariable(
            # pyrefly: ignore[no-matching-overload]
            dict(self.items),
            mutation_type=ValueMutationNew(),
            source=None,
        )

    # ref: https://github.com/python/cpython/blob/c3aefdb9eff0734058376b96fc86d89b1a345d75/Objects/dictobject.c#L5252-L5273
    tp_methods = {
        "copy": Method(dict_copy),
    }

    # Mutations to __dict__ are tracked through side effects (SideEffectsProxyDict),
    # so we don't need to install guards. Guard installation is overridden to no-op.
    def install_dict_keys_match_guard(self) -> None:
        pass

    def install_dict_contains_guard(
        self,
        tx: "InstructionTranslatorBase",
        key: HashableTracker,
        *,
        key_present: bool | None = None,
    ) -> None:
        pass
