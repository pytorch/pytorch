"""
Dynamo support for user-defined CuTeDSL kernels (``@cute.jit`` host functions): the
counterpart of DynamoTritonHOPifier and TritonKernelVariable in functions.py.

A host function reaches Dynamo either raw (the ``@cute.jit`` function itself, read
with or without a source) or wrapped
(``torch._library.cutedsl.wrap_cutedsl(fn, mutates_args=...)``, called inside or
outside the compiled region). Both become a CuteDSLKernelVariable, and a call of
either becomes a cutedsl_kernel_wrapper_mutation node. A raw call is written as for
eager::

    add_kernel(from_dlpack(x), from_dlpack(y), from_dlpack(out), n,
               cutlass.torch.current_stream())

``from_dlpack(t)`` becomes a CuteDSLTensorVariable standing for ``t``, the stream a
CuteDSLCurrentStreamVariable, and the call is captured as
``wrap_cutedsl(add_kernel, mutates_args=RAW_CALL_MUTATES_ARGS)(x, y, out, n)``.
"""

import dataclasses
import inspect
import sys
from collections.abc import Callable, Sequence
from typing import Any, TYPE_CHECKING
from typing_extensions import Never, TypeIs

import torch
from torch._higher_order_ops.cutedsl_kernel_wrap import (
    _may_overlap,
    cutedsl_kernel_side_table,
    cutedsl_kernel_wrapper_mutation,
    CuteDSLHOPifier,
    cutedsl_kernel_name,
    is_dsl_runtime_object,
    MutatesArgs,
    RAW_CALL_MUTATES_ARGS,
    TraceableCuteDSLKernelWrapper,
)
from torch._higher_order_ops.user_kernel_hop import _object_state, constant_key

from ..exc import unimplemented
from ..source import (
    AttrSource,
    ChainedSource,
    ConstDictKeySource,
    DictGetItemSource,
    GetItemSource,
)
from .base import NO_SUCH_SUBOBJ, VariableTracker


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase
    from torch._guards import Source

    from .dicts import ConstDictVariable


def _is_dict(val: VariableTracker) -> "TypeIs[ConstDictVariable]":
    """Whether ``val`` is a dict or an OrderedDict (not of another subclass), whose
    items leaves and specialize_nested walk like those of a tuple or list, and so
    does constant_object for one made in the region, which has no object to guard
    on."""
    from .dicts import ConstDictVariable, OrderedDictVariable

    return type(val) in (ConstDictVariable, OrderedDictVariable)


class DynamoCuteDSLHOPifier(CuteDSLHOPifier):
    def raise_unsupported(self, msg: str) -> Never:
        unimplemented(
            gb_type="CuTeDSL kernel unsupported feature",
            context="",
            explanation=f"Encountered CuTeDSL kernel unsupported feature: {msg}",
            hints=[],
        )

    def raise_modified(self, name: str, key: str, path: str) -> Never:
        # Dynamo applies a write made in the region only after the graph runs.
        self.raise_unsupported(
            f"{name}(): {path}, reached through {key!r}, is modified earlier in the "
            "compiled region, and the kernel would not see the change"
        )

    def get_value(self, val: VariableTracker) -> Any:
        return val.as_python_constant()

    def is_tensor(self, val: VariableTracker) -> bool:
        return val.realize().is_tensor()

    def python_type(self, val: VariableTracker) -> type:
        return val.realize().python_type()

    def leaves(self, val: VariableTracker) -> list[VariableTracker]:
        from .lists import BaseListVariable

        # bind_args walks each argument here first, so specialize_nested and
        # constant_object only see lists and dicts that do not hold themselves.
        def walk(
            val: VariableTracker, path: str, containers: dict[int, str]
        ) -> list[VariableTracker]:
            realized = val.realize()
            if isinstance(realized, BaseListVariable):
                steps = [(f"[{i}]", item) for i, item in enumerate(realized.items)]
            elif _is_dict(realized):
                keys = [k.vt for k in realized.items]
                steps = [(f".keys()[{i}]", k) for i, k in enumerate(keys)]
                values = realized.items.values()
                steps += [(f".values()[{i}]", v) for i, v in enumerate(values)]
            else:
                # An object constant's contents are checked on the real object, in
                # guard_contents.
                return [val]
            if id(realized) in containers:
                self.raise_unsupported(
                    "a constant kernel argument of type "
                    f"{realized.python_type_name()} refers to itself through "
                    f"{path[len(containers[id(realized)]) :]}"
                )
            containers = {**containers, id(realized): path}
            return [
                leaf
                for step, item in steps
                for leaf in walk(item, path + step, containers)
            ]

        return walk(val, "", {})

    def specialize_nested(
        self, variable: "CuteDSLKernelVariable", val: VariableTracker
    ) -> VariableTracker:
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
        return variable.specialize_symbolic(val)

    def cute_tensor_arg(
        self, name: str, key: str, val: VariableTracker
    ) -> VariableTracker | None:
        val = val.realize()
        if not isinstance(val, CuteDSLTensorVariable):
            return None

        def exprs(layout: tuple[object, ...]) -> list[object]:
            # Each symbolic value as its expression under the guards so far, so
            # that comparing two layouts adds no guards.
            return [v.node.expr if isinstance(v, torch.SymInt) else v for v in layout]

        if exprs(_layout(val.tensor)) != exprs(val.layout):
            self.raise_unsupported(
                f"{name}(): the tensor passed to from_dlpack for {key!r} changed its "
                "size, stride or storage offset in place after from_dlpack (with an "
                "in-place view op such as t_() or unsqueeze_()); call from_dlpack "
                "after the change"
            )
        return val.tensor

    def is_current_stream(
        self, val: VariableTracker, tx: "InstructionTranslatorBase"
    ) -> bool:
        val = val.realize()
        return (
            isinstance(val, CuteDSLCurrentStreamVariable)
            and val.stream_id == _current_stream(tx)[1]
        )

    def wrap_raw(self, variable: "CuteDSLKernelVariable") -> "CuteDSLKernelVariable":
        return CuteDSLKernelVariable(
            variable.kernel,
            RAW_CALL_MUTATES_ARGS,
            None,
            wrapper_type=TraceableCuteDSLKernelWrapper,
            raw_call=True,
        )

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
    ) -> Any:
        """The value of a constant argument, which the side table snapshots: a
        Python literal; a value from outside the region, guarded so that the graph
        only runs for an equal one (see guard_contents), which is passed as itself
        (a config object, a kernel object, a cutlass Numeric value, a function or
        class) or as the copy Dynamo rebuilds from its variable (a dict); a tuple or
        list, or a dict or OrderedDict made in the region, of these; or another
        Python constant made in the region (a namedtuple, a frozenset, a
        functools.partial), with each value from outside the region in it guarded
        in the same way."""
        from .constant import ConstantVariable
        from .lists import BaseListVariable

        def constant(item: VariableTracker) -> Any:
            return self.constant_object(variable, key, item, tx)

        def guard_sourced(item: VariableTracker) -> None:
            if item.source is not None:
                constant(item)

        val = val.realize()
        if isinstance(val, BaseListVariable) and val.python_type() in (tuple, list):
            return val.python_type()(constant(item) for item in val.items)
        if _is_dict(val) and val.source is None:
            items = {constant(k.vt): constant(v) for k, v in val.items.items()}
            return val.python_type()(items)
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
                VariableTracker.visit(guard_sourced, val)
                return value
        name = cutedsl_kernel_name(variable.kernel)
        obj = val.get_real_python_backed_value()
        if val.source is None or obj is NO_SUCH_SUBOBJ:
            self.raise_unsupported(
                f"{name}(): cannot pass the {val.python_type_name()} in {key!r}: an "
                "argument that is not a tensor or a scalar must be a Python constant, "
                "an object Dynamo can pass as itself (such as a config object made "
                "outside the compiled region), or a tuple or list of these"
            )
        # guard_contents finds a write made earlier in the region to an object the
        # snapshot holds, or to its class, whatever path the region took. This also
        # finds one to anything else the region reached through this argument, such
        # as a list held by its class (cfg.items[0] = 1).
        side_effects = tx.output.side_effects
        for other in side_effects.id_to_variable.values():
            other_source = other.source
            if other_source is None or not side_effects.is_modified(other):
                continue
            source = other_source
            while isinstance(source, ChainedSource) and source != val.source:
                source = source.base
            if source == val.source:
                self.raise_modified(name, key, other_source.name)
        self.guard_contents(val.source, obj, tx, name, key, {})
        return obj

    def guard_contents(
        self,
        source: "Source",
        value: object,
        tx: "InstructionTranslatorBase",
        name: str,
        key: str,
        path: dict[int, "Source"],
    ) -> None:
        """Guard ``value``, read from ``source``, on what the side table keys and
        snapshots it by (see constant_key): a list, tuple or dict on its type and
        length or keys, an object (see _object_state) on its type and, for a plain
        object, its attribute names, then each item, dict key or attribute in turn;
        a Python literal, a frozenset, a range or a slice on its value (constant_key
        keys these by equality); anything else (a function, a class, a cutlass
        value) on its identity. A change made to the object between calls then
        recompiles, as eager recompiles the kernel for it, instead of launching
        with the old snapshot.

        Graph-breaks on what the snapshot cannot stand for: a value that the region
        modified earlier, or whose class it modified, through whatever path; a
        tensor or a DSL object, which, as in a container, would be baked into the
        side table; and a value that holds itself (constant_key refuses one).
        ``name`` and ``key`` name the kernel and the parameter; ``path`` maps the id
        of each value that holds this one, from the argument down, to its source."""
        from ..guards import GuardBuilder, install_guard
        from .constant import ConstantVariable

        if id(value) in path:
            self.raise_unsupported(
                f"{name}(): the constant in {key!r} refers to itself: {source.name} "
                f"is {path[id(value)].name}"
            )
        path = {**path, id(value): source}
        side_effects = tx.output.side_effects
        if value in side_effects and side_effects.is_modified(side_effects[value]):
            self.raise_modified(name, key, source.name)
        # An attribute that an object does not hold is read from its class.
        classes = value.__mro__ if isinstance(value, type) else type(value).__mro__
        for cls in classes:
            if cls in side_effects and side_effects.is_modified(side_effects[cls]):
                self.raise_modified(name, key, f"class {cls.__name__} of {source.name}")
        if isinstance(value, torch.Tensor):
            self.raise_unsupported(
                f"{name}(): {key!r} holds a tensor inside a container "
                f"({source.name}); pass each tensor as its own argument"
            )
        if is_dsl_runtime_object(type(value)):
            self.raise_unsupported(
                f"{name}(): pass torch.Tensor and Python values, not "
                f"{type(value).__name__} ({source.name}, in {key!r}); PT2 converts "
                "tensors and supplies the stream"
            )
        if ConstantVariable.is_base_literal(value):
            install_guard(source.make_guard(GuardBuilder.CONSTANT_MATCH))
        elif isinstance(value, (frozenset, range, slice)):
            install_guard(source.make_guard(GuardBuilder.EQUALS_MATCH))
        elif isinstance(value, (tuple, list)):
            install_guard(source.make_guard(GuardBuilder.SEQUENCE_LENGTH))
            for i, item in enumerate(value):
                self.guard_contents(GetItemSource(source, i), item, tx, name, key, path)
        elif isinstance(value, dict):
            # constant_key keys a dict by its items in order.
            tx.output.guard_on_key_order.add(source)
            install_guard(
                source.make_guard(GuardBuilder.TYPE_MATCH),
                source.make_guard(GuardBuilder.DICT_KEYS_MATCH),
            )
            for i, (item_key, item) in enumerate(dict.items(value)):
                # constant_key keys an object key by its contents, which
                # DICT_KEYS_MATCH, an equality check, does not see.
                key_source = ConstDictKeySource(source, i)
                self.guard_contents(key_source, item_key, tx, name, key, path)
                item_source = DictGetItemSource(source, key_source)
                self.guard_contents(item_source, item, tx, name, key, path)
        elif (state := _object_state(value)) is not None:
            install_guard(source.make_guard(GuardBuilder.TYPE_MATCH))
            if dataclasses.is_dataclass(value):
                for attr, item in state.items():
                    attr_source = AttrSource(source, attr)
                    self.guard_contents(attr_source, item, tx, name, key, path)
            else:
                # Read through __dict__: an attribute name can be a keyword or dotted
                # (an argparse dest), which AttrSource cannot take; DictGetItemSource
                # takes any exact str.
                if any(type(attr) is not str for attr in state):
                    self.raise_unsupported(
                        f"{name}(): cannot guard {source.name}, in {key!r}: it has an "
                        "attribute whose name is not an exact str"
                    )
                dict_source = AttrSource(source, "__dict__")
                install_guard(dict_source.make_guard(GuardBuilder.DICT_KEYS_MATCH))
                for attr, item in state.items():
                    item_source = DictGetItemSource(dict_source, attr)
                    self.guard_contents(item_source, item, tx, name, key, path)
        else:
            install_guard(source.make_guard(GuardBuilder.ID_MATCH))

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
            if variable.raw_call:
                written = "a raw call counts every tensor as written"
                fix = f"call wrap_cutedsl({name}, mutates_args=[...])"
            else:
                written = f"both are written (mutates_args={mutates_args!r})"
                fix = "pass tensors that do not overlap, or give mutates_args"
            self.raise_unsupported(
                f"{name}(): {earlier!r} and {later!r} are the same tensor or views "
                f"that may overlap, and {written}, which PT2 cannot functionalize "
                f"(one write would be lost); {fix} naming only the parameters the "
                "kernel writes"
            )
        # Tensors and symbolic scalars are the node's arguments; every other value
        # is a constant of the launch.
        constant_args: dict[str, Any] = {}
        graph_args: dict[VariableTracker, VariableTracker] = {}
        for key, val in combined_args.items():
            val = val.realize()
            graphable = val.is_tensor() or val.is_symnode_like()
            if graphable and not val.is_python_constant():
                graph_args[VariableTracker.build(tx, key)] = val
            else:
                constant_args[key] = self.constant_object(variable, key, val, tx)
        try:
            constant_key(constant_args)
        except TypeError as e:
            self.raise_unsupported(f"{name}(): {e}")
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
    # Whether this is the wrapper that wrap_raw makes for a raw call.
    raw_call: bool

    def __init__(
        self,
        kernel: Callable[..., Any],
        mutates_args: object,
        kernel_idx: int | None,
        **kwargs: Any,
    ) -> None:
        self.kernel_source = kwargs.pop("kernel_source", kwargs.get("source"))
        self.wrapper_type = kwargs.pop("wrapper_type", None)
        self.raw_call = kwargs.pop("raw_call", False)
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
        return dynamo_cutedsl_hopifier_singleton.call_cutedsl_kernel(
            self, list(args), kwargs, tx, raw=not self.is_wrapper
        )

    def with_mutates_args(self, mutates_args: object) -> "CuteDSLKernelVariable":
        """The variable wrap_cutedsl(self.kernel, mutates_args=...) returns;
        mutates_args is validated as in eager (None is refused)."""
        return CuteDSLKernelVariable(
            self.kernel,
            mutates_args,
            None,
            kernel_source=self.kernel_source,
            wrapper_type=TraceableCuteDSLKernelWrapper,
        )

    def specialize_symbolic(self, arg: Any) -> Any:
        from .constant import ConstantVariable
        from .tensor import SymNodeVariable

        # See [Note: Specialize tl.constexpr args in user-defined triton kernels]
        if isinstance(arg, SymNodeVariable):
            return ConstantVariable.create(arg.evaluate_expr())
        return arg


def _current_stream(
    tx: "InstructionTranslatorBase",
) -> tuple[VariableTracker | None, int | None]:
    """The stream Dynamo tracks as current at this point of the trace (it follows
    torch.cuda.stream contexts), and an id for it that does not realize the lazy
    variable of the stream current when the trace started. The stream is read off the
    stack rather than with cur_stream(), which also marks it as used by the graph: it
    never enters the graph, and is only rebuilt, from its source, at a graph break."""
    state = tx.symbolic_stream_state
    if not state.in_stream_context():
        return None, None
    return state.cur_stream_stack[-1], state.cur_stream_id()


def _layout(tensor: VariableTracker) -> tuple[object, ...]:
    """The size, stride and storage offset of ``tensor``, as it is at this point of
    the trace."""
    fake = tensor.as_proxy().node.meta["example_value"]
    return (*fake.size(), *fake.stride(), fake.storage_offset())


class CuteDSLFromDlpackVariable(VariableTracker):
    """``cutlass.cute.runtime.from_dlpack``, whose result stands for its tensor in a
    raw call. ``assumed_align`` is accepted and dropped: the launch compiles the
    kernel for the alignment of each tensor's own pointer, at most 16 bytes. The
    other options would change the compiled kernel, so only their defaults are
    accepted. As in eager, a tensor that requires grad is refused."""

    def __init__(self, value: Callable[..., Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.value = value

    def python_type(self) -> type:
        return type(self.value)

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        unsupported = dynamo_cutedsl_hopifier_singleton.raise_unsupported
        signature = inspect.signature(self.value)
        first = next(iter(signature.parameters.values()), None)
        if first is None or first.kind not in (
            first.POSITIONAL_ONLY,
            first.POSITIONAL_OR_KEYWORD,
        ):
            unsupported(f"from_dlpack{signature}: unexpected signature")
        try:
            arguments = dict(signature.bind(*args, **kwargs).arguments)
        except TypeError as e:
            unsupported(f"from_dlpack(): {e}")
        tensor = arguments.pop(first.name).realize()
        if not tensor.is_tensor():
            unsupported(
                f"from_dlpack() of a {tensor.python_type_name()}: only a "
                "torch.Tensor is captured"
            )
        if tensor.requires_grad:
            unsupported(
                "from_dlpack() of a tensor that requires grad, which eager CuTeDSL "
                "refuses too; pass from_dlpack(t.detach())"
            )
        align_vt = arguments.pop("assumed_align", None)
        align = None
        if align_vt is not None:
            if not align_vt.is_python_constant():
                unsupported("from_dlpack(): assumed_align must be a constant")
            align = align_vt.as_python_constant()
            if align is not None and (
                type(align) is not int or not 0 < align <= 16 or align & (align - 1)
            ):
                unsupported(
                    "from_dlpack(): assumed_align must be a power of two up to 16 (PT2 "
                    "compiles for at most 16-byte alignment)"
                )
        for key, val in arguments.items():
            default = signature.parameters[key].default
            if not (val.is_python_constant() and val.as_python_constant() == default):
                unsupported(
                    f"from_dlpack(): {key}= is not supported; PT2 makes the cute "
                    "tensor itself, accepting only assumed_align"
                )
        return CuteDSLTensorVariable(tensor, align, _layout(tensor))


class CuteDSLTensorVariable(VariableTracker):
    """``from_dlpack(tensor)`` made in the compiled region: stands for ``tensor`` in a
    raw call, whose launcher makes the cute tensor, with the ``layout`` (see _layout)
    it had at from_dlpack: a call after an in-place view op changed it graph-breaks.
    It cannot be read in the region: an attribute or method of it (its element type,
    shape, ...) graph-breaks, and the real cute tensor is made at the break, as
    wherever the value leaves the graph."""

    def __init__(
        self,
        tensor: VariableTracker,
        assumed_align: int | None,
        layout: tuple[object, ...],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.tensor = tensor
        self.assumed_align = assumed_align
        self.layout = layout

    def python_type(self) -> type:
        # Already imported: it defines from_dlpack, which made this value.
        return sys.modules["cutlass.cute.runtime"]._Tensor

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        dynamo_cutedsl_hopifier_singleton.raise_unsupported(
            f"from_dlpack(...).{name}: a cute tensor made in the compiled region "
            "cannot be read there (pass it to the host function)"
        )

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        dynamo_cutedsl_hopifier_singleton.raise_unsupported(
            f"from_dlpack(...).{name}(): a cute tensor made in the compiled region "
            "cannot be read there (pass it to the host function)"
        )

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # Also str() and f-strings, whose default goes through tp_repr.
        dynamo_cutedsl_hopifier_singleton.raise_unsupported(
            "repr() of from_dlpack(...): a cute tensor made in the compiled region "
            "cannot be read there (pass it to the host function)"
        )

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # Live across a graph break: made again from the tensor.
        codegen.add_push_null(
            lambda: codegen.load_import_from("cutlass.cute.runtime", "from_dlpack")
        )
        codegen(self.tensor)
        if self.assumed_align is None:
            codegen.call_function(1, False)
            return
        codegen.append_output(codegen.create_load_const(self.assumed_align))
        codegen.extend_output(
            codegen.create_call_function_kw(2, ("assumed_align",), False)
        )


class CuteDSLCurrentStreamFunctionVariable(VariableTracker):
    """``cutlass.torch.current_stream``, whose result is the stream of a raw call."""

    def __init__(self, value: Callable[..., Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.value = value

    def python_type(self) -> type:
        return type(self.value)

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if args or kwargs:
            dynamo_cutedsl_hopifier_singleton.raise_unsupported(
                "cutlass.torch.current_stream() takes no arguments"
            )
        return CuteDSLCurrentStreamVariable(*_current_stream(tx))


class CuteDSLCurrentStreamVariable(VariableTracker):
    """``cutlass.torch.current_stream()`` called in the compiled region: the stream
    Dynamo tracks as current there (``stream``; ``stream_id`` identifies it). A raw
    call captures it only when that stream is also current at the call, because the
    captured kernel is launched on the call's stream when the graph runs. Like a cute
    tensor made in the region, it cannot be read there."""

    _nonvar_fields = {"stream", "stream_id", *VariableTracker._nonvar_fields}

    def __init__(
        self,
        stream: VariableTracker | None,
        stream_id: int | None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.stream = stream
        self.stream_id = stream_id

    def python_type(self) -> type:
        # Already imported by cutlass.torch, whose current_stream made this value.
        return sys.modules["cuda.bindings.driver"].CUstream

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        dynamo_cutedsl_hopifier_singleton.raise_unsupported(
            f"cutlass.torch.current_stream().{name}: the stream cannot be read in "
            "the compiled region (pass it to the host function)"
        )

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        dynamo_cutedsl_hopifier_singleton.raise_unsupported(
            f"cutlass.torch.current_stream().{name}(): the stream cannot be read in "
            "the compiled region (pass it to the host function)"
        )

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # Also str() and f-strings, whose default goes through tp_repr.
        dynamo_cutedsl_hopifier_singleton.raise_unsupported(
            "repr() of cutlass.torch.current_stream(): the stream cannot be read in "
            "the compiled region (pass it to the host function)"
        )

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # Live across a graph break: the CUstream of the recorded torch stream, which
        # Dynamo rebuilds as it rebuilds torch.cuda.current_stream() values (a
        # torch.Stream: native_handle is the raw stream). It is rebuilt from its source
        # when it has one, which keeps the lazy variable of the stream current when the
        # trace started unrealized.
        if self.stream is None:
            codegen.add_push_null(
                lambda: codegen.load_import_from("cutlass.torch", "current_stream")
            )
            codegen.call_function(0, False)
            return
        codegen.add_push_null(
            lambda: codegen.load_import_from("cuda.bindings.driver", "CUstream")
        )
        codegen(self.stream.source if self.stream.source is not None else self.stream)
        codegen.extend_output([codegen.create_load_attr("native_handle")])
        codegen.call_function(1, False)
