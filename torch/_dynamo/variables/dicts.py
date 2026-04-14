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
import types
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Literal, TYPE_CHECKING, Union

from torch.utils._pytree import MappingKey

from .. import graph_break_hints, polyfills, variables
from ..bytecode_transformation import (
    create_call_function,
    create_call_method,
    create_dup_top,
    create_instruction,
)
from ..exc import raise_observed_exception, unimplemented
from ..guards import GuardBuilder, install_guard
from ..source import (
    AttrSource,
    DictGetItemSource,
    is_constant_source,
    is_from_local_source,
)
from ..utils import (
    cmp_name_to_op_mapping,
    dict_items,
    dict_keys,
    dict_values,
    istype,
    raise_args_mismatch,
)
from .base import (
    AttributeMutationExisting,
    AttributeMutationNew,
    NO_SUCH_SUBOBJ,
    ValueMutationNew,
    VariableTracker,
)
from .constant import (
    CONSTANT_VARIABLE_FALSE,
    CONSTANT_VARIABLE_NONE,
    CONSTANT_VARIABLE_TRUE,
    ConstantVariable,
)
from .hashable import HashableTracker, is_hashable, raise_unhashable
from .sets import SetVariable


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslator

    from .functions import UserFunctionVariable


# [Adding a new supported class within the keys of ConstDictVariable]
# - Implement is_python_hashable() method in the VariableTracker subclass
# - Implement get_python_hash() and is_python_equal() methods for hashable types


class ConstDictVariable(VariableTracker):
    # PyDict_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/dictobject.c#L4825
    _cpython_type = dict

    CONTAINS_GUARD = GuardBuilder.DICT_CONTAINS
    NOT_CONTAINS_GUARD = GuardBuilder.DICT_NOT_CONTAINS

    _nonvar_fields = {
        "user_cls",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        items: dict[VariableTracker, VariableTracker],
        user_cls: type = dict,
        **kwargs: Any,
    ) -> None:
        # .clone() pass these arguments in kwargs but they're recreated a few
        # lines below
        if "original_items" in kwargs:
            kwargs.pop("original_items")
        if "should_reconstruct_all" in kwargs:
            kwargs.pop("should_reconstruct_all")

        super().__init__(**kwargs)

        Hashable = HashableTracker

        # Keys will just be HashableTrackers when cloning, in any other case they'll be VariableTrackers
        assert all(
            isinstance(x, (VariableTracker, Hashable))
            and isinstance(v, VariableTracker)
            for x, v in items.items()
        )

        def make_hashable(
            key: Union[VariableTracker, "HashableTracker"],
        ) -> "HashableTracker":
            return key if isinstance(key, Hashable) else Hashable(key)

        dict_cls = self._get_dict_cls_from_user_cls(user_cls)
        self.items = dict_cls({make_hashable(x): v for x, v in items.items()})
        # need to reconstruct everything if the dictionary is an intermediate value
        # or if a pop/delitem was executed
        self.should_reconstruct_all = (
            not is_from_local_source(self.source) if self.source else True
        )
        self.original_items = items.copy()
        self.user_cls = user_cls

    def _get_dict_cls_from_user_cls(self, user_cls: type) -> type:
        accepted_dict_types = (dict, collections.OrderedDict, collections.defaultdict)

        # avoid executing user code if user_cls is a dict subclass
        if user_cls in accepted_dict_types:
            dict_cls = user_cls
        else:
            # <Subclass, ..., dict, object>
            dict_cls = next(
                base for base in user_cls.__mro__ if base in accepted_dict_types
            )
        assert dict_cls in accepted_dict_types, dict_cls

        # Use a dict instead as the call "defaultdict({make_hashable(x): v ..})"
        # would fail as defaultdict expects a callable as first argument
        if dict_cls is collections.defaultdict:
            dict_cls = dict
        return dict_cls

    def as_proxy(self) -> dict[Any, Any]:
        return {k.vt.as_proxy(): v.as_proxy() for k, v in self.items.items()}

    def debug_repr(self) -> str:
        items: list[str] = []
        for k, v in self.items.items():
            key_str = repr(k.vt.value) if hasattr(k.vt, "value") else k.vt.debug_repr()
            val_str = repr(v.value) if hasattr(v, "value") else v.debug_repr()
            items.append(f"{key_str}: {val_str}")
        return "{" + ", ".join(items) + "}"

    def as_python_constant(self) -> dict[Any, Any]:
        return {
            k.vt.as_python_constant(): v.as_python_constant()
            for k, v in self.items.items()
        }

    def keys_as_python_constant(self) -> dict[Any, VariableTracker]:
        self.install_dict_keys_match_guard()
        return {k.vt.as_python_constant(): v for k, v in self.items.items()}

    def python_type(self) -> type:
        return self.user_cls

    def __contains__(self, vt: VariableTracker) -> bool:
        assert isinstance(vt, VariableTracker)
        Hashable = HashableTracker
        if not is_hashable(vt):
            return False
        key = Hashable(vt)
        return key in self.items and not isinstance(
            self.items[key], variables.DeletedVariable
        )

    def call_tree_map_branch(
        self,
        tx: "InstructionTranslator",
        tree_map_fn: "UserFunctionVariable",
        map_fn: VariableTracker,
        rest: Sequence[VariableTracker],
        tree_map_kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
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
        tx: "InstructionTranslator",
        tree_map_fn: "UserFunctionVariable",
        map_fn: VariableTracker,
        rest: Sequence[VariableTracker],
        tree_map_kwargs: dict[str, VariableTracker],
        keypath: tuple[Any, ...],
    ) -> VariableTracker:
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
            # We can safely call realize() here as it won't introduce any new guards
            item = self.original_items.get(key.vt)
            if self.is_new_item(item, value) or self.should_reconstruct_all:
                codegen(key.vt)
                codegen(value)
                num_args += 1
        codegen.append_output(create_instruction("BUILD_MAP", arg=num_args))

    def reconstruct(self, codegen: "PyCodegen") -> None:
        if self.user_cls is collections.OrderedDict:
            # emit `OrderedDict(constructed_dict)`
            codegen.add_push_null(
                lambda: codegen.extend_output(
                    [
                        codegen.create_load_python_module(collections),
                        codegen.create_load_attr("OrderedDict"),
                    ]
                )
            )
            if self._contains_self_reference():
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
        else:
            if self._contains_self_reference():
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

    def getitem_const_raise_exception_if_absent(
        self, tx: "InstructionTranslator", arg: VariableTracker
    ) -> VariableTracker:
        key = HashableTracker(arg)
        if key not in self.items:
            raise_observed_exception(KeyError, tx, args=[arg])
        return self.items[key]

    def getitem_const(
        self, tx: "InstructionTranslator", arg: VariableTracker
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

    def realize_key_vt(self, arg: VariableTracker) -> None:
        # Realize the LazyVT on a particular index
        assert arg in self
        key = HashableTracker(arg)
        index = tuple(self.items.keys()).index(key)
        original_key_vt = tuple(self.original_items.keys())[index]
        if isinstance(original_key_vt, variables.LazyVariableTracker):
            original_key_vt.realize()

    def install_dict_keys_match_guard(self) -> None:
        if self.source:
            install_guard(self.make_guard(GuardBuilder.DICT_KEYS_MATCH))

    def install_dict_contains_guard(
        self, tx: "InstructionTranslator", args: list[VariableTracker]
    ) -> None:
        # Key guarding - These are the cases to consider
        # 1) The dict has been mutated. In this case, we would have already
        # inserted a DICT_KEYS_MATCH guard, so we can skip.
        #
        # 2) args[0].source is None. This happens for const keys. Here, we
        # have to insert the DICT_CONTAINS guard.
        #
        # 3) args[0].source is not None. This can happen for non-const VTs.
        #   3a) contains=True. In this case, we can access the lazyVT from
        #   original_items and selectively realize it.
        #   3b) contains=False. There is no easy way to selectively apply this
        #   DICT_NOT_CONTAINS guard because our guard are represented via trees.
        #   Be conservative and add DICT_KEYS_MATCH guard.

        if not self.source:
            return

        if tx.output.side_effects.is_modified(self):
            return

        contains = args[0] in self
        if args[0].source is None and args[0].is_python_constant():
            guard_fn = (
                type(self).CONTAINS_GUARD if contains else type(self).NOT_CONTAINS_GUARD
            )
            install_guard(
                self.make_guard(
                    functools.partial(
                        guard_fn,
                        key=args[0].as_python_constant(),
                    )
                )
            )
        elif args[0].source:
            if contains:
                self.realize_key_vt(args[0])
            else:
                self.install_dict_keys_match_guard()

    def mp_subscript_impl(
        self,
        tx: "InstructionTranslator",
        key: VariableTracker,
    ) -> VariableTracker:
        # dict_subscript: https://github.com/python/cpython/blob/62a6e898e01/Objects/dictobject.c#L3673-L3706
        # Unhashable key check happens inside _HashableTracker (raise_unhashable → TypeError).
        return self.getitem_const_raise_exception_if_absent(tx, key)

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # NB - Both key and value are LazyVariableTrackers in the beginning. So,
        # we have to insert guards when a dict method is accessed. For this to
        # be simple, we are conservative and overguard. We skip guard only for
        # get/__getitem__ because the key guard will be inserted by the
        # corresponding value VT. For __contains__, we add a DICT_CONTAINS
        # guard. But for all the other methods, we insert the DICT_KEYS_MATCH
        # guard to be conservative.
        from . import DictBuiltinVariable
        from .builder import SourcelessBuilder

        Hashable = HashableTracker

        if name == "__init__":
            temp_dict_vt = DictBuiltinVariable.call_custom_dict(
                tx, dict, *args, **kwargs
            )
            tx.output.side_effects.mutation(self)
            self.items.update(temp_dict_vt.items)  # type: ignore[attr-defined]
            return CONSTANT_VARIABLE_NONE
        elif name == "items":
            if args or kwargs:
                raise_args_mismatch(
                    tx,
                    name,
                    "0 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            self.install_dict_keys_match_guard()
            if self.source:
                tx.output.guard_on_key_order.add(self.source)
            return DictItemsVariable(self)
        elif name == "keys":
            if len(args):
                raise_args_mismatch(tx, name, "0 args", f"{len(args)} args")
            self.install_dict_keys_match_guard()
            if self.source:
                tx.output.guard_on_key_order.add(self.source)
            return DictKeysVariable(self)
        elif name == "values":
            if args or kwargs:
                raise_args_mismatch(
                    tx,
                    name,
                    "0 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            self.install_dict_keys_match_guard()
            if self.source:
                tx.output.guard_on_key_order.add(self.source)
            if args or kwargs:
                raise_observed_exception(TypeError, tx)
            return DictValuesVariable(self)
        elif name == "copy":
            self.install_dict_keys_match_guard()
            if args or kwargs:
                raise_args_mismatch(
                    tx,
                    name,
                    "0 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            return self.clone(
                items=self.items.copy(), mutation_type=ValueMutationNew(), source=None
            )
        elif name == "__setitem__" and self.is_mutable():
            arg_hashable = args and is_hashable(args[0])
            if not arg_hashable:
                raise_unhashable(args[0], tx)

            self.install_dict_keys_match_guard()
            if kwargs or len(args) != 2:
                raise_args_mismatch(
                    tx,
                    name,
                    "2 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            tx.output.side_effects.mutation(self)
            self.items[Hashable(args[0])] = args[1]
            return CONSTANT_VARIABLE_NONE
        elif name == "__delitem__" and self.is_mutable():
            arg_hashable = args and is_hashable(args[0])
            if arg_hashable:
                self.install_dict_keys_match_guard()
                self.should_reconstruct_all = True
                tx.output.side_effects.mutation(self)
                self.items.__delitem__(Hashable(args[0]))
                return CONSTANT_VARIABLE_NONE
            else:
                return super().call_method(tx, name, args, kwargs)
        elif name == "get":
            if len(args) not in (1, 2):
                raise_args_mismatch(tx, name, "1 or 2 args", f"{len(args)} args")

            arg_hashable = args and is_hashable(args[0])
            if not arg_hashable:
                raise_unhashable(args[0], tx)

            if args[0] not in self:
                self.install_dict_contains_guard(tx, args)
                if len(args) == 1:
                    # if default is not given, return None
                    return CONSTANT_VARIABLE_NONE
                return args[1]
            # Key guarding - Nothing to do.
            return self.getitem_const(tx, args[0])
        elif name == "pop" and self.is_mutable():
            if len(args) not in (1, 2):
                raise_args_mismatch(tx, name, "1 or 2 args", f"{len(args)} args")

            arg_hashable = args and is_hashable(args[0])
            if not arg_hashable:
                raise_unhashable(args[0], tx)

            if args[0] not in self:
                # missing item, return the default value. Install no DICT_CONTAINS guard.
                self.install_dict_contains_guard(tx, args)
                if len(args) == 1:
                    # if default is not given, raise KeyError
                    raise_observed_exception(KeyError, tx)
                return args[1]

            self.should_reconstruct_all = True
            tx.output.side_effects.mutation(self)
            return self.items.pop(Hashable(args[0]))
        elif name == "popitem" and self.is_mutable():
            if (
                issubclass(self.user_cls, dict)
                and not issubclass(self.user_cls, collections.OrderedDict)
                and len(args)
            ):
                raise_args_mismatch(tx, name)

            if not self.items:
                raise_observed_exception(
                    KeyError,
                    tx,
                    args=[
                        "popitem(): dictionary is empty",
                    ],
                )

            if self.user_cls is collections.OrderedDict and (
                len(args) == 1 or "last" in kwargs
            ):
                if len(args) == 1 and args[0].is_python_constant():
                    last = args[0].as_python_constant()
                elif (v := kwargs.get("last")) and v.is_python_constant():
                    last = v.as_python_constant()
                else:
                    raise_args_mismatch(tx, name)
                k, v = self.items.popitem(last=last)  # type: ignore[possibly-undefined]
            else:
                k, v = self.items.popitem()

            self.should_reconstruct_all = True
            tx.output.side_effects.mutation(self)

            return variables.TupleVariable([k.vt, v])
        elif name == "clear":
            if args or kwargs:
                raise_args_mismatch(
                    tx,
                    name,
                    "0 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            self.should_reconstruct_all = True
            tx.output.side_effects.mutation(self)
            self.items.clear()
            return CONSTANT_VARIABLE_NONE
        elif name == "update" and self.is_mutable():
            # In general, this call looks like `a.update(b, x=1, y=2, ...)`.
            # Either `b` or the kwargs is omittable, but not both.
            self.install_dict_keys_match_guard()
            has_arg = len(args) == 1
            has_kwargs = len(kwargs) > 0
            if has_arg or has_kwargs:
                tx.output.side_effects.mutation(self)
                if has_arg:
                    dict_vt: VariableTracker
                    if isinstance(args[0], ConstDictVariable):
                        # NB - Guard on all the keys of the other dict to ensure
                        # correctness.
                        args[0].install_dict_keys_match_guard()
                        dict_vt = args[0]
                    else:
                        dict_vt = DictBuiltinVariable.call_custom_dict(
                            tx, dict, args[0]
                        )
                    self.items.update(dict_vt.items)  # type: ignore[attr-defined]
                if has_kwargs:
                    # Handle kwargs
                    kwargs_hashable = {
                        Hashable(VariableTracker.build(tx, k)): v
                        for k, v in kwargs.items()
                    }
                    self.items.update(kwargs_hashable)
                return CONSTANT_VARIABLE_NONE
            else:
                return super().call_method(tx, name, args, kwargs)
        elif name == "__contains__":
            if not len(args):
                raise_args_mismatch(
                    tx,
                    name,
                    "more than 1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )

            arg_hashable = args and is_hashable(args[0])
            if not arg_hashable:
                raise_unhashable(args[0], tx)

            self.install_dict_contains_guard(tx, args)
            contains = args[0] in self
            return VariableTracker.build(tx, contains)
        elif name == "setdefault" and self.is_mutable():
            if len(args) not in (1, 2):
                raise_args_mismatch(
                    tx,
                    name,
                    "1 or 2 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )

            arg_hashable = args and is_hashable(args[0])
            if not arg_hashable:
                raise_unhashable(args[0], tx)

            self.install_dict_keys_match_guard()
            if kwargs or len(args) > 2:
                raise_args_mismatch(
                    tx,
                    name,
                    "at most 2 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            value = self.maybe_getitem_const(args[0])
            if value is not None:
                return value
            else:
                if len(args) == 1:
                    x = CONSTANT_VARIABLE_NONE
                else:
                    x = args[1]
                tx.output.side_effects.mutation(self)
                self.items[Hashable(args[0])] = x
                return x
        elif name == "move_to_end":
            self.install_dict_keys_match_guard()
            tx.output.side_effects.mutation(self)
            if args[0] not in self:
                raise_observed_exception(KeyError, tx)

            last = True
            if len(args) == 2 and args[1].is_python_constant():
                last = args[1].as_python_constant()

            if kwargs and "last" in kwargs and kwargs["last"].is_python_constant():
                last = kwargs.get("last").as_python_constant()  # type: ignore[union-attr]

            key = Hashable(args[0])
            self.items.move_to_end(key, last=last)
            return CONSTANT_VARIABLE_NONE
        elif name == "__eq__" and istype(
            self, ConstDictVariable
        ):  # don't let Set use this function
            if len(args) != 1:
                raise_args_mismatch(tx, name, "1 args", f"{len(args)} args")

            return SourcelessBuilder.create(tx, polyfills.dict___eq__).call_function(
                tx, [self, args[0]], {}
            )
        elif name == "__ne__":
            return VariableTracker.build(
                tx,
                not self.call_method(tx, "__eq__", args, kwargs).value,  # type: ignore[attr-defined]
            )
        elif name == "__or__":
            if len(args) != 1:
                raise_args_mismatch(tx, name, "1 args", f"{len(args)} args")
            other = args[0]

            # Method resolution for binops works as follow (using __or__ as example):
            # (1) dict.__or__(dict) => dict
            # (2) dict.__or__(subclass): return NotImplemented
            # (3) Check if subclass implements __ror__ => forward the call
            # to subclass.__ror__(dict)

            # Let's not forward the call to __ror__ yet because __ror__ can be
            # implemented in C (i.e. OrderedDict subclass) which Dynamo cannot
            # trace
            # if istype(other, variables.UserDefinedDictVariable):
            #     if other.call_obj_hasattr(tx, "__ror__").value:
            #         return other.call_method(tx, "__ror__", [self], kwargs)

            # The three dict types Dynamo can handle are dict, OrderedDict and
            # defaultdict.

            # TODO(guilhermeleobas): this check should be on builtin.py::call_or_
            if istype(
                other,
                (
                    ConstDictVariable,
                    variables.UserDefinedDictVariable,
                    variables.DefaultDictVariable,
                ),
            ):
                # Unwrap UserDefinedDictVariable to its underlying ConstDictVariable
                if isinstance(other, variables.UserDefinedDictVariable):
                    assert other._base_vt is not None
                    assert isinstance(other._base_vt, ConstDictVariable)
                    other = other._base_vt

                # Always return the specialized dictionary, and in the case
                # both are specialized, take the first to be the type of the
                # new dictionary
                if self.user_cls is not dict:
                    user_cls = self.user_cls
                    to_cpy = self
                else:
                    user_cls = other.user_cls
                    to_cpy = other

                to_cpy.install_dict_keys_match_guard()
                new_dict_vt = to_cpy.clone(
                    items=self.items.copy(),
                    mutation_type=ValueMutationNew(),
                    source=None,
                    user_cls=user_cls,
                )

                # NB - Guard on all the keys of the other dict to ensure
                # correctness.
                args[0].install_dict_keys_match_guard()  # type: ignore[attr-defined]
                new_dict_vt.items.update(args[0].items)  # type: ignore[attr-defined]
                return new_dict_vt
            else:
                raise_observed_exception(
                    TypeError,
                    tx,
                    args=[
                        f"unsupported operand type(s) for |: '{self.python_type().__name__}'"
                        f"and '{other.python_type().__name__}'"
                    ],
                )
        elif name == "__ior__":
            self.call_method(tx, "update", args, kwargs)
            return self
        elif name == "__iter__":
            from .lists import ListIteratorVariable

            if self.source and not is_constant_source(self.source):
                tx.output.guard_on_key_order.add(self.source)
            return ListIteratorVariable(
                self.unpack_var_sequence(tx), mutation_type=ValueMutationNew()
            )
        else:
            return super().call_method(tx, name, args, kwargs)

    def unpack_var_sequence(self, tx: "InstructionTranslator") -> list[VariableTracker]:
        self.install_dict_keys_match_guard()
        return [x.vt for x in self.items]

    def mp_length(self, tx: "InstructionTranslator") -> VariableTracker:
        """Mapping length for dict objects."""
        self.install_dict_keys_match_guard()
        return VariableTracker.build(tx, len(self.items))

    def call_obj_hasattr(
        self, tx: "InstructionTranslator", name: str
    ) -> ConstantVariable:
        # dict not allow setting arbitrary attributes.  OrderedDict and
        # defaultdict allow arbitrary setattr, but not deletion of default attrs
        if any(
            self.user_cls is t
            for t in (dict, collections.OrderedDict, collections.defaultdict)
        ):
            if hasattr(self.user_cls, name):
                return CONSTANT_VARIABLE_TRUE
            if self.user_cls is dict:
                return CONSTANT_VARIABLE_FALSE

        msg = f"hasattr on {self.user_cls} is not supported"
        unimplemented(
            gb_type="unsupported hasattr operation",
            context=f"Class {self.user_cls}",
            explanation=msg,
            hints=[
                "Consider using a regular dictionary instead",
                *graph_break_hints.SUPPORTABLE,
            ],
        )

    def clone(self, **kwargs: Any) -> VariableTracker:
        self.install_dict_keys_match_guard()
        return super().clone(**kwargs)

    def is_python_hashable(self) -> bool:
        """
        Dictionaries are mutable and therefore not hashable in Python.
        """
        return False

    def var_getattr(self, tx: "InstructionTranslator", name: str):
        if name == "__class__":
            return VariableTracker.build(tx, self.python_type())
        return super().var_getattr(tx, name)


class MappingProxyVariable(VariableTracker):
    # PyDictProxy_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/descrobject.c#L1995
    _cpython_type = types.MappingProxyType

    # proxies to the original dict_vt
    def __init__(self, dv_dict: ConstDictVariable, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        assert isinstance(dv_dict, ConstDictVariable)
        self.dv_dict = dv_dict

    def python_type(self) -> type:
        return types.MappingProxyType

    def unpack_var_sequence(self, tx: "InstructionTranslator") -> list[VariableTracker]:
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

    def _check_mutation_guard(self, tx: "InstructionTranslator") -> None:
        if self.source and tx.output.side_effects.has_existing_dict_mutation():
            msg = (
                "A dict has been modified while we have an existing mappingproxy object. "
                "A mapping proxy object, as the name suggest, proxies a mapping "
                "object (usually a dict). If the original dict object mutates, it "
                "is reflected in the proxy object as well. For an existing proxy "
                "object, we do not know the original dict it points to. Therefore, "
                "for correctness we graph break when there is dict mutation and we "
                "are trying to access a proxy object."
            )

            unimplemented(
                gb_type="mapping proxy affected by dictionary mutation",
                context=f"Source: {self.source}, Dict mutation detected",
                explanation=msg,
                hints=[
                    "Avoid modifying dictionaries that might be referenced by mapping proxy objects",
                    "Or avoid using the mapping proxy objects after modifying its underlying dictionary",
                ],
            )

    def mp_subscript_impl(
        self,
        tx: "InstructionTranslator",
        key: VariableTracker,
    ) -> VariableTracker:
        # mappingproxy_getitem: https://github.com/python/cpython/blob/62a6e898e01/Objects/descrobject.c#L1052-L1056
        # TODO(follow-up): add tests for invalid key type, missing key
        self._check_mutation_guard(tx)
        return self.dv_dict.mp_subscript_impl(tx, key)

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        self._check_mutation_guard(tx)
        return self.dv_dict.call_method(tx, name, args, kwargs)

    def mp_length(self, tx: "InstructionTranslator") -> VariableTracker:
        return self.dv_dict.mp_length(tx)

    def call_obj_hasattr(
        self, tx: "InstructionTranslator", name: str
    ) -> ConstantVariable:
        if self.python_type() is types.MappingProxyType:
            return VariableTracker.build(tx, name in types.MappingProxyType.__dict__)
        return super().call_obj_hasattr(tx, name)


class NNModuleHooksDictVariable(ConstDictVariable):
    # Special class to avoid adding any guards on the nn module hook ids.
    def install_dict_keys_match_guard(self) -> None:
        pass

    def install_dict_contains_guard(
        self, tx: "InstructionTranslator", args: list[VariableTracker]
    ) -> None:
        pass


class DefaultDictVariable(ConstDictVariable):
    _cpython_type = collections.defaultdict

    def __init__(
        self,
        items: dict[VariableTracker, VariableTracker],
        user_cls: type,
        default_factory: VariableTracker | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(items, user_cls, **kwargs)
        assert user_cls is collections.defaultdict
        if default_factory is None:
            default_factory = CONSTANT_VARIABLE_NONE
        self.default_factory = default_factory

    def is_python_constant(self) -> bool:
        # Return false for unsupported defaults. This ensures that a bad handler
        # path is not taken in BuiltinVariable for getitem.
        if self.default_factory not in [list, tuple, dict] and not self.items:
            return False
        return super().is_python_constant()

    def debug_repr(self) -> str:
        assert self.default_factory is not None
        return (
            f"defaultdict({self.default_factory.debug_repr()}, {super().debug_repr()})"
        )

    @staticmethod
    def is_supported_arg(arg: VariableTracker) -> bool:
        return isinstance(
            arg,
            (
                variables.BaseBuiltinVariable,
                variables.functions.BaseUserFunctionVariable,
                variables.functions.PolyfilledFunctionVariable,
            ),
        ) or (isinstance(arg, variables.ConstantVariable) and arg.value is None)

    def mp_subscript_impl(
        self,
        tx: "InstructionTranslator",
        key: VariableTracker,
    ) -> VariableTracker:
        # Mirrors CPython's defaultdict.__getitem__ (dict_subscript → __missing__).
        # defaultdict.__missing__: https://github.com/python/cpython/blob/62a6e898e01/Modules/_collectionsmodule.c#L2233-L2254
        # Key present → normal dict lookup (same as ConstDictVariable.mp_subscript_impl).
        if key in self:
            return self.getitem_const(tx, key)

        if self.default_factory.is_constant_none():
            raise_observed_exception(KeyError, tx, args=[key])
        else:
            default_var = self.default_factory.call_function(tx, [], {})
            super().call_method(tx, "__setitem__", [key, default_var], {})
            return default_var

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "__setattr__" and self.is_mutable:
            if len(args) != 2:
                raise_args_mismatch(tx, name, "2 args", f"{len(args)} args")
            # Setting a default factory must be a callable or None type
            if (
                istype(args[0], ConstantVariable) and args[0].value == "default_factory"
            ) and self.is_supported_arg(args[1]):
                tx.output.side_effects.mutation(self)
                self.default_factory = args[1]
                return CONSTANT_VARIABLE_NONE
            return super().call_method(tx, name, args, kwargs)
        elif name == "__eq__":
            if len(args) != 1:
                raise_args_mismatch(tx, name, "1 args", f"{len(args)} args")

            return VariableTracker.build(tx, polyfills.dict___eq__).call_function(
                tx, [self, args[0]], {}
            )
        else:
            return super().call_method(tx, name, args, kwargs)

    def var_getattr(
        self,
        tx: "InstructionTranslator",
        name: str,
    ) -> VariableTracker:
        if name == "default_factory":
            return self.default_factory
        return super().var_getattr(tx, name)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # emit `defaultdict(default_factory, new_dict)`
        codegen.add_push_null(
            lambda: codegen.extend_output(
                [
                    codegen.create_load_python_module(collections),
                    codegen.create_load_attr("defaultdict"),
                ]
            )
        )
        codegen(self.default_factory)
        codegen.extend_output(
            [
                *create_call_function(1, False),
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


class DictViewVariable(VariableTracker):
    """
    Models _PyDictViewObject

    This is an "abstract" class. Subclasses will override kv and the items method
    """

    kv: str | None = None

    def __init__(self, dv_dict: ConstDictVariable, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        assert self.kv in ("keys", "values", "items")
        assert isinstance(dv_dict, ConstDictVariable)
        self.dv_dict = dv_dict

    @property
    def view_items(self) -> Any:
        assert self.kv is not None
        return getattr(self.dv_dict.items, self.kv)()

    @property
    def view_items_vt(self) -> list[VariableTracker]:
        # Returns an iterable of the unpacked items
        # Implement in the subclasses
        raise NotImplementedError

    def unpack_var_sequence(self, tx: "InstructionTranslator") -> list[VariableTracker]:
        return self.view_items_vt

    def reconstruct(self, codegen: "PyCodegen") -> None:
        assert self.kv is not None
        codegen(self.dv_dict)
        codegen.load_method(self.kv)
        codegen.call_method(0)

    def call_obj_hasattr(
        self, tx: "InstructionTranslator", name: str
    ) -> ConstantVariable:
        assert self.kv is not None
        if name in self.python_type().__dict__:
            return CONSTANT_VARIABLE_TRUE
        return CONSTANT_VARIABLE_FALSE

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "__iter__":
            from .lists import ListIteratorVariable

            return ListIteratorVariable(
                self.view_items_vt, mutation_type=ValueMutationNew()
            )
        elif name == "__repr__":
            return VariableTracker.build(tx, self.debug_repr())
        return super().call_method(tx, name, args, kwargs)

    def sq_length(self, tx: "InstructionTranslator") -> VariableTracker:
        """Sequence length for dict view objects."""
        return VariableTracker.build(tx, len(self.view_items))


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
        return dict_keys

    def debug_repr(self) -> str:
        if not self.view_items:
            return "dict_keys([])"
        else:
            items: list[str] = []
            for k in self.view_items:
                key_str = (
                    repr(k.vt.value) if hasattr(k.vt, "value") else k.vt.debug_repr()
                )
                items.append(key_str)
            return "dict_keys([" + ",".join(items) + "])"

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "__contains__":
            return self.dv_dict.call_method(tx, name, args, kwargs)
        elif name in (
            "__and__",
            "__iand__",
            "__or__",
            "__ior__",
            "__sub__",
            "__isub__",
            "__xor__",
            "__ixor__",
        ):
            # These methods always returns a set
            m = getattr(self.set_items, name)
            r = m(args[0].set_items)  # type: ignore[attr-defined]
            return SetVariable(r)
        elif name in cmp_name_to_op_mapping:
            if not isinstance(
                args[0],
                (
                    SetVariable,
                    variables.UserDefinedSetVariable,
                    DictItemsVariable,
                    DictKeysVariable,
                ),
            ):
                return VariableTracker.build(tx, NotImplemented)
            return VariableTracker.build(
                tx,
                cmp_name_to_op_mapping[name](self.set_items, args[0].set_items),  # type: ignore[attr-defined]
            )
        return super().call_method(tx, name, args, kwargs)


class DictValuesVariable(DictViewVariable):
    # PyDictValues_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/dictobject.c#L6567
    _cpython_type = dict_values

    # DictValuesVariable is an iterable but cannot be compared.
    kv = "values"

    @property
    def view_items_vt(self) -> list[VariableTracker]:
        return list(self.view_items)

    def python_type(self) -> type:
        return dict_values

    def debug_repr(self) -> str:
        if not self.view_items:
            return "dict_values([])"
        else:
            items: list[str] = []
            for v in self.view_items:
                val_str = repr(v.value) if hasattr(v, "value") else v.debug_repr()
                items.append(val_str)
            return "dict_values([" + ",".join(items) + "])"


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
        return dict_items

    def debug_repr(self) -> str:
        if not self.view_items:
            return "dict_items([])"
        else:
            items: list[str] = []
            for k, v in self.view_items:
                key_str = (
                    repr(k.vt.value) if hasattr(k.vt, "value") else k.vt.debug_repr()
                )
                val_str = repr(v.value) if hasattr(v, "value") else v.debug_repr()
                items.append(f"({key_str}, {val_str})")
            return "dict_items([" + ",".join(items) + "])"

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # TODO(guilhermeleobas): This should actually check if args[0]
        # implements the mapping protocol.
        if name == "__eq__":
            if len(args) != 1:
                raise_args_mismatch(tx, name, "1 args", f"{len(args)} args")
            if isinstance(args[0], DictItemsVariable):
                return self.dv_dict.call_method(tx, "__eq__", [args[0].dv_dict], {})
            elif isinstance(
                args[0],
                (
                    SetVariable,
                    variables.UserDefinedSetVariable,
                    DictItemsVariable,
                    DictKeysVariable,
                ),
            ):
                return VariableTracker.build(
                    tx,
                    len(self.set_items ^ args[0].set_items) == 0,
                )
            return CONSTANT_VARIABLE_FALSE
        elif name == "__iter__":
            from .lists import ListIteratorVariable

            return ListIteratorVariable(
                self.view_items_vt, mutation_type=ValueMutationNew()
            )
        elif name in (
            "__and__",
            "__iand__",
            "__or__",
            "__ior__",
            "__sub__",
            "__isub__",
            "__xor__",
            "__ixor__",
        ):
            # These methods always returns a set
            fn_hdl = getattr(self.set_items, name)
            ret_val = fn_hdl(args[0].set_items)  # type: ignore[attr-defined]
            return SetVariable(ret_val)
        elif name in cmp_name_to_op_mapping:
            if not isinstance(
                args[0],
                (
                    SetVariable,
                    variables.UserDefinedSetVariable,
                    DictItemsVariable,
                    DictKeysVariable,
                ),
            ):
                return VariableTracker.build(tx, NotImplemented)
            return VariableTracker.build(
                tx,
                cmp_name_to_op_mapping[name](self.set_items, args[0].set_items),  # type: ignore[attr-defined]
            )
        return super().call_method(tx, name, args, kwargs)

    def is_python_hashable(self) -> Literal[False]:
        """
        Dictionary item views are not hashable in Python.
        """
        return False


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
        tx: "InstructionTranslator", vt: VariableTracker
    ) -> dict[str, VariableTracker]:
        example_value_dict = SideEffectsProxyDict.get_example_value_dict(vt)

        return {
            key: VariableTracker.build(
                tx,
                value,
                source=vt.source
                and DictGetItemSource(AttrSource(vt.source, "__dict__"), key),
            )
            for key, value in example_value_dict.items()
        }

    def __init__(self, item: VariableTracker, tx: "InstructionTranslator") -> None:
        self.item = item
        self.side_effects = tx.output.side_effects
        self.item_dict = self.get_value___dict__(tx, item)

    def _maybe_unwrap_key(self, key: kV) -> str:
        Hasher = HashableTracker
        return key.vt.as_python_constant() if istype(key, Hasher) else key

    def side_effects_table(self) -> dict[str, VariableTracker]:
        return self.side_effects.store_attr_mutations.get(self.item, {})

    def __getitem__(self, key: kV) -> VariableTracker:
        name = self._maybe_unwrap_key(key)
        if self.side_effects.has_pending_mutation_of_attr(self.item, name):
            return self.side_effects.load_attr(self.item, name, deleted_ok=True)
        return self.item_dict[name]

    def __setitem__(self, key: kV, value: VariableTracker) -> None:
        # Find a way to not hash the key using HashableTracker
        name = self._maybe_unwrap_key(key)
        assert istype(name, str)
        self.side_effects.store_attr(self.item, name, value)

    def __delitem__(self, key: kV) -> None:
        name = self._maybe_unwrap_key(key)
        self.side_effects.store_attr(self.item, name, variables.DeletedVariable())

    def __contains__(self, key: kV) -> bool:  # type: ignore[bad-override]
        name = self._maybe_unwrap_key(key)
        table = self.side_effects_table()
        # if name in side effects, then it is only contained if it's not a DeletedVariable
        # even if the original dict contains it
        if name in table:
            return not isinstance(table[name], variables.DeletedVariable)
        else:
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

    @classmethod
    def create(
        cls,
        tx: "InstructionTranslator",
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
        tx: "InstructionTranslator",
        **kwargs: Any,
    ) -> None:
        super().__init__({}, **kwargs)
        self.items = SideEffectsProxyDict(vt, tx)

    def setitem(self, name: str, value: VariableTracker) -> None:
        self.items[name] = value

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

    # Mutations to __dict__ are tracked through side effects (SideEffectsProxyDict),
    # so we don't need to install guards. Guard installation is overridden to no-op.
    def install_dict_keys_match_guard(self) -> None:
        pass

    def install_dict_contains_guard(
        self, tx: "InstructionTranslator", args: list[VariableTracker]
    ) -> None:
        pass
