"""
Dynamo support for user-defined CuTeDSL kernels (``@cute.jit`` host functions): the
counterpart of DynamoTritonHOPifier and TritonKernelVariable in functions.py.

A host function reaches Dynamo either raw (the ``@cute.jit`` function itself, read
with or without a source) or wrapped
(``torch._library.cutedsl.wrap_cutedsl(fn, mutates_args=...)``, called inside or
outside the compiled region). Both become a CuteDSLKernelVariable, and a call of a
wrapper becomes a cutedsl_kernel_wrapper_mutation node; a raw call graph-breaks.
"""

import collections
import dataclasses
import enum
import sys
import weakref
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple, Never, TYPE_CHECKING
from typing_extensions import TypeIs

import torch
from torch._higher_order_ops.cutedsl_kernel_wrap import (
    _may_overlap,
    _numpy_scalar_item,
    CUTEDSL_CONSTANT_RULES,
    cutedsl_kernel_name,
    cutedsl_kernel_side_table,
    cutedsl_kernel_wrapper_mutation,
    CuteDSLHOPifier,
    is_dsl_runtime_object,
    MutatesArgs,
    TraceableCuteDSLKernelWrapper,
)
from torch._higher_order_ops.user_kernel_hop import constant_contents, constant_key

from .. import graph_break_hints
from ..exc import unimplemented
from ..source import (
    AttrSource,
    CallMethodItemSource,
    ConstDictKeySource,
    DictGetItemSource,
    GetItemSource,
    is_from_source,
    NumpyTensorSource,
)
from .base import NO_SUCH_SUBOBJ, VariableTracker


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.side_effects import SideEffects
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase
    from torch._guards import Source
    from torch._higher_order_ops.user_kernel_hop import ConstantContents

    from .constant import ConstantVariable
    from .dicts import ConstDictVariable


def _is_dict(val: VariableTracker) -> "TypeIs[ConstDictVariable]":
    """Whether ``val`` is a dict or an OrderedDict (not of another subclass), whose
    items leaves and specialize_nested walk like those of a tuple or list, and so
    does constant_object for one made in the region, which has no object to guard
    on."""
    from .dicts import ConstDictVariable, OrderedDictVariable

    return type(val) in (ConstDictVariable, OrderedDictVariable)


def _held_error(cls: type, is_tensor: bool, where: str) -> TypeError | None:
    """The TypeError of the side table's walk (see constant_key) for a tensor at
    ``where`` inside a constant argument, or its like for a DSL runtime object of
    type ``cls`` there; None for another value."""
    if is_tensor:
        held, fix = "a tensor", "pass each tensor as its own argument"
    elif is_dsl_runtime_object(cls):
        held = f"a {cls.__name__}"
        fix = (
            "a cute tensor, pointer or stream cannot be nested inside a constant "
            "argument"
        )
    else:
        return None
    return TypeError(f"a constant kernel argument holds {held} at {where}: {fix}")


def _cycle_error(cls: type, where: str, start: str) -> ValueError:
    """The ValueError of the side table's walk for a container of type ``cls``,
    entered at ``start``, met again at ``where``."""
    return ValueError(
        f"a constant kernel argument of type {cls.__name__} refers to itself through "
        f"{where[len(start) :]}"
    )


def _weakly_referenceable(value: object) -> bool:
    try:
        weakref.ref(value)
    except TypeError:
        return False
    return True


class _RegionWrites(NamedTuple):
    """The writes the region made before a call, which Dynamo applies only after the
    graph runs, read once per call from its side effects: the ids of the modified
    variables (SideEffects._get_modified_vars), and the sources of those that have
    one."""

    side_effects: "SideEffects"
    var_ids: frozenset[int]
    sources: tuple["Source", ...]

    @classmethod
    def of(cls, tx: "InstructionTranslatorBase") -> "_RegionWrites":
        side_effects = tx.output.side_effects
        modified = side_effects._get_modified_vars()
        sources = tuple(var.source for var in modified if var.source is not None)
        return cls(side_effects, frozenset(map(id, modified)), sources)

    def modifies(self, value: object) -> bool:
        """Whether the region modified ``value``, a real object."""
        side_effects = self.side_effects
        return value in side_effects and id(side_effects[value]) in self.var_ids

    def under(self, source: "Source") -> "Source | None":
        """The source of a modified value reached through ``source``, if any."""
        return next((s for s in self.sources if is_from_source(s, source)), None)


class DynamoCuteDSLHOPifier(CuteDSLHOPifier):
    def raise_unsupported(self, msg: str) -> Never:
        # A call that Dynamo cannot capture, whether or not eager takes it: msg says
        # what Dynamo cannot take, and what to change.
        unimplemented(
            gb_type="CuTeDSL kernel unsupported feature",
            context="",
            explanation="Dynamo cannot capture this call of a CuTeDSL host function, "
            f"which eager may run: {msg}",
            hints=[
                "Change the code as the explanation says.",
                "To capture the launch as it is, make it in a custom operator (torch.library.custom_op), whose call Dynamo captures without tracing into it.",
            ],
        )

    def raise_user_error(self, exc: TypeError | ValueError) -> Never:
        unimplemented(
            gb_type="CuTeDSL kernel call error",
            context="",
            explanation=f"Eager refuses this call too: {exc}",
            hints=[*graph_break_hints.USER_ERROR],
        )

    def raise_modified(self, name: str, key: str, path: str) -> Never:
        # Dynamo applies a write made in the region only after the graph runs.
        self.raise_unsupported(
            f"CuTeDSL kernel {name}: parameter {key!r}: {path} is modified earlier in "
            "the compiled region, and the kernel would not see the change; make the "
            "change before the compiled region"
        )

    def is_tensor(self, val: VariableTracker) -> bool:
        from .tensor import NumpyNdarrayVariable

        # Dynamo traces a numpy value as an array backed by a tensor, which is no
        # tensor argument of the call: a numpy scalar is a constant (numpy_scalar).
        val = val.realize()
        return val.is_tensor() and not isinstance(val, NumpyNdarrayVariable)

    def python_type(self, val: VariableTracker) -> type:
        return val.realize().python_type()

    def leaves(self, val: VariableTracker) -> list[VariableTracker]:
        from .lists import BaseListVariable

        def value_step(i: int, key: VariableTracker) -> str:
            # The side table's walk names a dict value by its key, which a key that
            # is not a constant cannot show here.
            if key.is_python_constant():
                return f"[{key.as_python_constant()!r}]"
            return f".values()[{i!r}]"

        # Refuses what the side table's walk refuses, with its errors, which
        # bind_args re-raises naming the kernel and the parameter. bind_args walks
        # each argument here first, so specialize_nested and constant_object only
        # see lists and dicts that do not hold themselves. The contents of an object
        # constant are checked on the real object, in guard_contents.
        def walk(
            val: VariableTracker, where: str, containers: dict[int, str]
        ) -> list[VariableTracker]:
            realized = val.realize()
            if isinstance(realized, BaseListVariable):
                steps = [(f"[{i!r}]", item) for i, item in enumerate(realized.items)]
            elif _is_dict(realized):
                keys = [k.vt for k in realized.items]
                values = realized.items.values()
                steps = [(f".keys()[{i!r}]", k) for i, k in enumerate(keys)]
                steps += [
                    (value_step(i, k), v) for i, (k, v) in enumerate(zip(keys, values))
                ]
            else:
                return [val]
            if id(realized) in containers:
                start = containers[id(realized)]
                raise _cycle_error(realized.python_type(), where, start)
            containers = {**containers, id(realized): where}
            leaves: list[VariableTracker] = []
            for step, item in steps:
                error = _held_error(
                    self.python_type(item), self.is_tensor(item), where + step
                )
                if error is not None:
                    raise error
                leaves += walk(item, where + step, containers)
            return leaves

        return walk(val, "", {})

    def specialize_nested(
        self, variable: "CuteDSLKernelVariable", val: VariableTracker
    ) -> VariableTracker:
        from .functions import specialize_symbolic_variable
        from .lists import BaseListVariable
        from .user_defined import UserDefinedObjectVariable

        val = val.realize()
        if isinstance(val, BaseListVariable):
            items = [self.specialize_nested(variable, item) for item in val.items]
            if isinstance(val, UserDefinedObjectVariable):
                # A namedtuple or a list subclass, whose variable clone() cannot copy.
                return type(val)(val.value, items=items, source=val.source)
            return val.clone(items=items)
        if _is_dict(val):
            return val.clone(
                items={
                    k: self.specialize_nested(variable, v) for k, v in val.items.items()
                }
            )
        return specialize_symbolic_variable(val)

    def overlapping_args(
        self, tensors: dict[str, VariableTracker]
    ) -> tuple[str, str] | None:
        """The names of the first two of ``tensors`` that may share memory (see
        _may_overlap)."""
        items = [
            (key, val.as_proxy().node.meta["example_value"])
            for key, val in tensors.items()
        ]
        for i, (later, tensor) in enumerate(items):
            for earlier, other in items[:i]:
                if _may_overlap(other, tensor):
                    return earlier, later
        return None

    def constant_object(
        self,
        variable: "CuteDSLKernelVariable",
        key: str,
        val: VariableTracker,
        tx: "InstructionTranslatorBase",
        writes: _RegionWrites,
    ) -> Any:
        """The value of a constant argument, which the side table snapshots: a
        Python literal; a value from outside the region, guarded so that the graph
        only runs for an equal one (see guard_contents), which is passed as itself
        (a config object, a kernel object, a cutlass Numeric value, a numpy scalar,
        a function or class) or as the copy Dynamo rebuilds from its variable (a
        dict); a tuple or list, or a dict or OrderedDict made in the region, of
        these; or another Python constant made in the region (a namedtuple, a
        frozenset, a functools.partial), with each value from outside the region in
        it guarded in the same way."""
        from .constant import ConstantVariable
        from .lists import BaseListVariable
        from .tensor import NumpyNdarrayVariable

        def constant(item: VariableTracker) -> Any:
            return self.constant_object(variable, key, item, tx, writes)

        val = val.realize()
        if isinstance(val, BaseListVariable) and val.python_type() in (tuple, list):
            return val.python_type()(constant(item) for item in val.items)
        if _is_dict(val) and val.source is None:
            items = {constant(k.vt): constant(v) for k, v in val.items.items()}
            return val.python_type()(items)
        if isinstance(val, NumpyNdarrayVariable):
            return self.numpy_scalar(variable, key, val, tx)
        if val.is_python_constant():
            value = val.as_python_constant()
            if ConstantVariable.is_base_literal(value):
                return value
            # Any other constant is guarded on what the side table keys it by,
            # below or, for one made in the region, through each value from outside
            # the region in it. Dynamo's own guards can miss part of it: a dict's
            # keys and their order, or a function's identity (they check its code,
            # which another function made by the same def shares).
            if val.source is None:
                visited: dict[int, Any] = {}

                def guard_sourced(item: VariableTracker) -> None:
                    if item.source is None:
                        return
                    constant(item)
                    # That guarded what it holds too: marking its fields visited
                    # keeps visit out of them.
                    item = item.unwrap()
                    visited.update(
                        (id(field), field)
                        for attr, field in vars(item).items()
                        if attr not in item._nonvar_fields
                    )

                VariableTracker.visit(guard_sourced, val, visited)
                return value
        name = cutedsl_kernel_name(variable.kernel)
        obj = val.get_real_python_backed_value()
        if val.source is None or obj is NO_SUCH_SUBOBJ:
            self.raise_unsupported(
                f"CuTeDSL kernel {name}: parameter {key!r}: cannot pass the "
                f"{val.python_type_name()}: an argument that is not a tensor or a "
                "scalar must be a Python constant, an object Dynamo can pass as itself "
                "(such as a config object made outside the compiled region), or a "
                "tuple or list of these"
            )
        # guard_contents finds a write made earlier in the region to an object the
        # snapshot holds, or to its class, whatever path the region took. This also
        # finds one to anything else the region reached through this argument, such
        # as a list held by its class (cfg.items[0] = 1).
        modified = writes.under(val.source)
        if modified is not None:
            self.raise_modified(name, key, modified.name)
        self.guard_contents(val.source, obj, tx, variable, key, "", {}, writes)
        return obj

    def numpy_scalar(
        self,
        variable: "CuteDSLKernelVariable",
        key: str,
        val: VariableTracker,
        tx: "InstructionTranslatorBase",
    ) -> object:
        """The numpy scalar that ``val`` stands for, passed as itself (the launch
        converts one for a runtime scalar parameter, see python_scalar) and guarded
        on its type and value: Dynamo traces a numpy scalar made outside the region
        as a 0-d array backed by a tensor, whose source is the scalar's. Graph-breaks
        on a numpy array, and on a numpy scalar made in the region, which has no
        object to pass."""
        import numpy as np  # imported: Dynamo traced a numpy value

        source = val.source
        if isinstance(source, NumpyTensorSource):
            source = source.base
        value = None if source is None else tx.output.resolve_source_value(source)
        if source is None or not isinstance(value, np.generic):
            self.raise_unsupported(
                f"CuTeDSL kernel {cutedsl_kernel_name(variable.kernel)}: parameter "
                f"{key!r}: cannot pass a numpy array, or a numpy scalar made in the "
                "compiled region (Dynamo traces it as a tensor); pass a Python scalar"
            )
        self.guard_value(source, value, tx)
        return value

    def raise_constant_error(
        self, variable: "CuteDSLKernelVariable", key: str, exc: TypeError | ValueError
    ) -> Never:
        """Refuse a constant argument of a call of ``variable`` with ``exc``, an
        error of the side table's walk, which the eager launch of a wrapper re-raises
        naming the kernel and the parameter (see raise_user_error)."""
        name = cutedsl_kernel_name(variable.kernel)
        msg = f"CuTeDSL kernel {name}: parameter {key!r}: {exc}"
        self.raise_user_error(type(exc)(msg))

    def guard_contents(
        self,
        source: "Source",
        value: object,
        tx: "InstructionTranslatorBase",
        variable: "CuteDSLKernelVariable",
        key: str,
        where: str,
        containers: dict[int, str],
        writes: _RegionWrites,
    ) -> None:
        """Guard ``value``, read from ``source``, on what the side table keys and
        snapshots it by, walking it as constant_key does (see constant_contents): a
        list, tuple, dict or object (see _object_state) on its type and on what the
        walk reads besides the items (see content_sources), then each item, dict key
        or attribute in turn; any other value as guard_value says. A change made to
        the object between calls then recompiles, as eager recompiles the kernel for
        it, instead of launching with the old snapshot.

        Graph-breaks on what the snapshot cannot stand for: a value that the region
        modified earlier, or whose class it modified, through whatever path; and,
        with the errors of the side table's walk, a tensor or a DSL object in it,
        which would be baked into the side table, and a value that holds itself.
        ``variable`` is the wrapper called and ``key`` the parameter, ``where`` is the
        walk's path to ``value`` from the argument, and ``containers`` maps the id of
        each container on that path to the path it was entered at."""
        name = cutedsl_kernel_name(variable.kernel)
        if writes.modifies(value):
            self.raise_modified(name, key, source.name)
        # An attribute that an object does not hold is read from its class.
        classes = value.__mro__ if isinstance(value, type) else type(value).__mro__
        for cls in classes:
            if writes.modifies(cls):
                self.raise_modified(name, key, f"class {cls.__name__} of {source.name}")
        contents = constant_contents(value, CUTEDSL_CONSTANT_RULES)
        if contents is None:
            self.guard_value(source, value, tx)
            return
        if id(value) in containers:
            cycle = _cycle_error(type(value), where, containers[id(value)])
            self.raise_constant_error(variable, key, cycle)
        containers = {**containers, id(value): where}
        sources = self.content_sources(source, value, contents, tx, name, key)
        for item_source, (step, at, item) in zip(sources, contents.steps, strict=True):
            item_where = where + step.format(at)
            held = _held_error(type(item), isinstance(item, torch.Tensor), item_where)
            if held is not None:
                self.raise_constant_error(variable, key, held)
            self.guard_contents(
                item_source, item, tx, variable, key, item_where, containers, writes
            )

    def content_sources(
        self,
        source: "Source",
        value: object,
        contents: "ConstantContents",
        tx: "InstructionTranslatorBase",
        name: str,
        key: str,
    ) -> list["Source"]:
        """Guard ``value``, which constant_key keys by ``contents``, on its type and on
        what the walk reads besides the items: a sequence's length, a dict's keys in
        order (an OrderedDict's by the identity of each key, so an equal OrderedDict
        made again with key objects that are new each time, such as computed
        strings, recompiles), an object's attribute names (not their order: an
        object with the same attributes set in another order reuses the graph, which
        launches a snapshot equal to it); and return the source of each of
        ``contents.steps``. Dynamo's guards read a list, tuple or dict through its
        storage, so a class that overrides a method the walk reads it through
        (__iter__ or items(), say) graph-breaks."""
        from ..guards import GuardBuilder, install_guard

        cls = type(value)

        def check_reads(base: type, methods: tuple[str, ...]) -> None:
            overridden = [m for m in methods if getattr(cls, m) is not getattr(base, m)]
            if overridden:
                self.raise_unsupported(
                    f"CuTeDSL kernel {name}: parameter {key!r}: cannot guard "
                    f"{source.name}: its class {cls.__name__} overrides "
                    f"{', '.join(overridden)}, which the side table reads it through; "
                    f"pass a plain {base.__name__}"
                )

        if isinstance(value, (tuple, list)):
            base = tuple if isinstance(value, tuple) else list
            check_reads(base, ("__iter__", "__len__"))
            # Its type too.
            install_guard(source.make_guard(GuardBuilder.SEQUENCE_LENGTH))
            return [GetItemSource(source, i) for _, i, _ in contents.steps]
        if isinstance(value, dict):
            ordered = isinstance(value, collections.OrderedDict)
            base = collections.OrderedDict if ordered else dict
            check_reads(base, ("__iter__", "items"))
            tx.output.guard_on_key_order.add(source)
            install_guard(
                source.make_guard(GuardBuilder.TYPE_MATCH),
                source.make_guard(GuardBuilder.DICT_KEYS_MATCH),
            )
            if ordered:
                # Its own order, which DICT_KEYS_MATCH does not see (move_to_end
                # leaves the dict's storage alone).
                install_guard(source.make_guard(GuardBuilder.ORDERED_DICT_KEYS_MATCH))
            # The walk reads each key, then the value of each.
            sources = {
                id(k): ConstDictKeySource(source, i)
                for i, k in enumerate(dict.keys(value))
            }
            half = len(contents.steps) // 2
            item_sources: list[Source] = [
                sources[id(k)] for _, _, k in contents.steps[:half]
            ]
            item_sources += [
                DictGetItemSource(source, sources[id(k)])
                for _, k, _ in contents.steps[half:]
            ]
            return item_sources
        install_guard(source.make_guard(GuardBuilder.TYPE_MATCH))
        fields = (
            {f.name for f in dataclasses.fields(value)}
            if dataclasses.is_dataclass(value)
            else set()
        )
        if not hasattr(value, "__dict__"):  # a dataclass with slots: its fields
            return [AttrSource(source, attr) for attr in contents.names]
        # Read through __dict__: an attribute name can be a keyword or dotted (an
        # argparse dest), which AttrSource cannot take; DictGetItemSource takes any
        # exact str.
        if any(type(attr) is not str for attr in contents.names):
            self.raise_unsupported(
                f"CuTeDSL kernel {name}: parameter {key!r}: cannot guard "
                f"{source.name}: it has an attribute whose name is not an exact str; "
                "name its attributes with exact str"
            )
        dict_source = AttrSource(source, "__dict__")
        install_guard(dict_source.make_guard(GuardBuilder.DICT_KEYS_MATCH))
        return [
            AttrSource(source, attr)
            if attr in fields
            else DictGetItemSource(dict_source, attr)
            for attr in contents.names
        ]

    def guard_value(
        self, source: "Source", value: object, tx: "InstructionTranslatorBase"
    ) -> None:
        """Guard ``value``, a constant that constant_key keys without walking it, on
        what constant_key keys it by: a cutlass Numeric, such as
        cutlass.Float32(2.0), on its type and value, and a numpy bool, integer,
        floating or complex scalar whose .item() is a Python one on its type and
        .item(); a Python literal, a subclass of int, str or bytes (an enum member
        aside), a frozenset, a range or a slice on its value, which EQUALS_MATCH
        checks with the exact type, comparing an exact float or complex bitwise
        (-0.0 is not 0.0, as for constant_key) and another value with ==; anything
        else on its identity, kept alive with the compiled code if it cannot be
        weakly referenced: a function, a class, an enum member, another cutlass
        value, or a float or complex subclass that numpy does not define, which
        EQUALS_MATCH would compare with ==, taking -0.0 for 0.0. Limitation: such a
        subclass, and another value that constant_key keys by equality (a Fraction, a
        Decimal, an attrs class instance), recompile for an equal new one, such as
        one built for each call: no Dynamo guard checks its bits or its equality
        outside EQUALS_MATCH's types."""
        from ..guards import GuardBuilder, install_guard
        from .constant import ConstantVariable

        np = sys.modules.get("numpy")  # imported if value is a numpy scalar
        if CUTEDSL_CONSTANT_RULES.scalar_value(value) is not None:
            # A cutlass Numeric, which constant_key keys by its type and its value.
            install_guard(
                source.make_guard(GuardBuilder.TYPE_MATCH),
                AttrSource(source, "value").make_guard(GuardBuilder.CONSTANT_MATCH),
            )
        elif _numpy_scalar_item(value) is not None or (
            np is not None
            and isinstance(value, np.complexfloating)
            and type(value.item()) is complex
        ):
            # Its .item() is an exact bool, int, float or complex: np.float64 and
            # np.complex128 subclass float and complex, which EQUALS_MATCH compares
            # with ==.
            install_guard(
                source.make_guard(GuardBuilder.TYPE_MATCH),
                CallMethodItemSource(source).make_guard(GuardBuilder.CONSTANT_MATCH),
            )
        elif ConstantVariable.is_base_literal(value) or (
            isinstance(value, (int, str, bytes)) and not isinstance(value, enum.Enum)
        ):
            install_guard(source.make_guard(GuardBuilder.CONSTANT_MATCH))
        elif isinstance(value, (frozenset, range, slice)):
            install_guard(source.make_guard(GuardBuilder.EQUALS_MATCH))
        else:
            install_guard(source.make_guard(GuardBuilder.ID_MATCH))
            if not _weakly_referenceable(value):
                # ID_MATCH cannot see such a value freed and its id taken by another,
                # and the side table may hold an equal value in its place: alive as
                # long as the compiled code, it keeps its id.
                tx.output.install_global_by_id("___cutedsl_constant", value)

    def call_HOP(
        self,
        variable: "CuteDSLKernelVariable",
        combined_args: dict[str, Any],
        tx: "InstructionTranslatorBase",
    ) -> VariableTracker:
        from .dicts import ConstDictVariable

        name = cutedsl_kernel_name(variable.kernel)
        # Written tensors that may overlap would fail check_written_args_disjoint in
        # the backend, which cannot fall back to eager: a graph break here.
        mutates_args = variable.mutates_args or ()
        overlap = self.overlapping_args(
            {
                key: val
                for key, val in combined_args.items()
                if self.is_tensor(val)
                and (mutates_args == "unknown" or key in mutates_args)
            }
        )
        if overlap is not None:
            earlier, later = overlap
            written = f"both are written (mutates_args={mutates_args!r})"
            fix = "pass tensors that do not overlap, or give mutates_args"
            self.raise_unsupported(
                f"CuTeDSL kernel {name}: parameters {earlier!r} and {later!r} are the "
                f"same tensor or views that may overlap, and {written}, which PT2 "
                f"cannot functionalize (one write would be lost); {fix} naming only "
                "the parameters the kernel writes"
            )
        # Tensors and symbolic scalars are the node's arguments; every other value
        # is a constant of the launch.
        writes = _RegionWrites.of(tx)
        constant_args: dict[str, Any] = {}
        graph_args: dict[VariableTracker, VariableTracker] = {}
        for key, val in combined_args.items():
            val = val.realize()
            graphable = self.is_tensor(val) or val.is_symnode_like()
            if graphable and not val.is_python_constant():
                graph_args[VariableTracker.build(tx, key)] = val
            else:
                value = self.constant_object(variable, key, val, tx, writes)
                constant_args[key] = value
        for key, value in constant_args.items():
            try:
                constant_key(value, CUTEDSL_CONSTANT_RULES)
            except (TypeError, ValueError) as e:
                self.raise_constant_error(variable, key, e)
        constant_args_idx = cutedsl_kernel_side_table.add_constant_args(constant_args)
        meta = ConstDictVariable(graph_args)
        tx.output.create_proxy(
            "call_function",
            cutedsl_kernel_wrapper_mutation,
            (),
            {
                "kernel_idx": variable.kernel_idx,
                "constant_args_idx": constant_args_idx,
                "kwargs": meta.as_proxy(),
            },
        )

        return VariableTracker.build(tx, None)


dynamo_cutedsl_hopifier_singleton = DynamoCuteDSLHOPifier()


class CuteDSLKernelVariable(VariableTracker):
    kernel: Callable[..., Any]
    mutates_args: MutatesArgs | None
    kernel_idx: int | None
    kernel_source: "Source | None"
    # The class of the wrapper this stands for: TraceableCuteDSLKernelWrapper (what
    # wrap_cutedsl returns) or the subclass of it passed in; None for the @cute.jit
    # function itself, whose calls are raw.
    wrapper_type: type | None

    def __init__(
        self,
        kernel: Callable[..., Any],
        mutates_args: object,
        kernel_idx: int | None,
        **kwargs: Any,
    ) -> None:
        self.kernel_source = kwargs.pop("kernel_source", kwargs.get("source"))
        self.wrapper_type = kwargs.pop("wrapper_type", None)
        super().__init__(**kwargs)
        dynamo_cutedsl_hopifier_singleton.init_variable(
            self, kernel, mutates_args, kernel_idx, raw=not self.is_wrapper
        )

    @property
    def is_wrapper(self) -> bool:
        return self.wrapper_type is not None

    def python_type(self) -> type:
        return type(self.kernel) if self.wrapper_type is None else self.wrapper_type

    def as_python_constant(self) -> Any:
        # The @cute.jit function; not the wrapper object, which one made in the
        # region does not have.
        if self.is_wrapper:
            return super().as_python_constant()
        return self.kernel

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        # A wrapper stands for its host function and write set, the only attributes
        # Dynamo reads of it: the host function as the variable built from its
        # source (guarded on its identity, see builder.py), or as a constant if it
        # has none, and the write set as a constant (guarded on its value for a
        # wrapper passed in).
        if not self.is_wrapper:
            return super().tp_getattro_impl(tx, name)
        if name == "kernel":
            return VariableTracker.build(tx, self.kernel, self.kernel_source)
        if name == "mutates_args":
            return VariableTracker.build(tx, self.mutates_args)
        unimplemented(
            gb_type="CuTeDSL kernel wrapper attribute",
            context="",
            explanation=f"CuTeDSL kernel {cutedsl_kernel_name(self.kernel)}: Dynamo "
            "takes a CuTeDSL kernel wrapper (what torch._library.cutedsl.wrap_cutedsl "
            "returns) for its host function and write set, so in the compiled region "
            f"it reads only the wrapper's kernel and mutates_args, not {name!r}.",
            hints=[
                "Read the attribute outside the compiled region and pass its value in.",
                "To launch the kernel, call the wrapper itself, which Dynamo captures.",
            ],
        )

    def call_obj_hasattr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> "ConstantVariable":
        if not self.is_wrapper:
            return super().call_obj_hasattr(tx, name)
        from .constant import ConstantVariable

        self.tp_getattro_impl(tx, name)  # refuses any other attribute
        return ConstantVariable.create(True)

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker, op: str
    ) -> VariableTracker:
        if self.is_wrapper:
            return super().tp_richcompare_impl(tx, other, op)
        # A host function compares by identity, as any function does.
        from .object_protocol import object_richcompare

        return object_richcompare(self, tx, other, op)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # A wrapper made by wrap_cutedsl inside the region has no source of its own:
        # rebuild it from the host function's source.
        if not self.is_wrapper or self.kernel_source is None:
            return super().reconstruct(codegen)
        codegen.add_push_null(
            lambda: codegen.load_import_from("torch._library.cutedsl", "wrap_cutedsl")
        )
        codegen(self.kernel_source)
        codegen.append_output(codegen.create_load_const(self.mutates_args))
        codegen.extend_output(
            codegen.create_call_function_kw(2, ("mutates_args",), False)
        )

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if not self.is_wrapper:
            # A raw call, which Dynamo does not capture: a graph break.
            return super().call_function(tx, list(args), kwargs)
        return dynamo_cutedsl_hopifier_singleton.call_cutedsl_kernel(
            self, list(args), kwargs, tx
        )

    def with_mutates_args(self, mutates_args: object) -> "CuteDSLKernelVariable":
        """The variable wrap_cutedsl(self.kernel, mutates_args=...) returns;
        mutates_args is validated as in eager (None is refused)."""
        return CuteDSLKernelVariable(  # noqa: RVT001
            self.kernel,
            mutates_args,
            None,
            kernel_source=self.kernel_source,
            wrapper_type=TraceableCuteDSLKernelWrapper,
        )
