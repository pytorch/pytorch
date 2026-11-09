"""
Dynamo support for user-defined CuTeDSL kernels (``@cute.jit`` host functions): the
counterpart of DynamoTritonHOPifier and TritonKernelVariable in functions.py.

A host function reaches Dynamo either raw (the ``@cute.jit`` function itself) or
wrapped (what ``torch._library.cutedsl.wrap_cutedsl(fn, mutates_args=...)`` returns,
made outside the compiled region). Both become a CuteDSLKernelVariable, and a call
of a wrapper becomes a cutedsl_kernel_wrapper_mutation node; a raw call graph-breaks.
"""

from collections.abc import Callable, Sequence
from typing import Any, Never, TYPE_CHECKING
from typing_extensions import TypeIs

from torch._higher_order_ops.cutedsl_kernel_wrap import (
    _may_overlap,
    cutedsl_kernel_name,
    cutedsl_kernel_side_table,
    cutedsl_kernel_wrapper_mutation,
    CuteDSLHOPifier,
    is_dsl_runtime_object,
    MutatesArgs,
)

from .. import graph_break_hints
from ..exc import unimplemented
from .base import VariableTracker


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase
    from torch._guards import Source

    from .constant import ConstantVariable
    from .dicts import ConstDictVariable


def _is_dict(val: VariableTracker) -> "TypeIs[ConstDictVariable]":
    """Whether ``val`` is a dict or an OrderedDict (not of another subclass), whose
    items leaves walks like those of a tuple or list."""
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

    def is_tensor(self, val: VariableTracker) -> bool:
        from .tensor import NumpyNdarrayVariable

        # Dynamo traces a numpy value as an array backed by a tensor, which is no
        # tensor argument of the call.
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
        # see lists and dicts that do not hold themselves.
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
    ) -> Any:
        """The value of a constant argument, which the side table snapshots: a
        Python literal, or a tuple or list of these."""
        from .constant import ConstantVariable
        from .lists import BaseListVariable

        def constant(item: VariableTracker) -> Any:
            return self.constant_object(variable, key, item)

        val = val.realize()
        if isinstance(val, BaseListVariable) and val.python_type() in (tuple, list):
            return val.python_type()(constant(item) for item in val.items)
        if val.is_python_constant():
            value = val.as_python_constant()
            if ConstantVariable.is_base_literal(value):
                return value
        name = cutedsl_kernel_name(variable.kernel)
        self.raise_unsupported(
            f"CuTeDSL kernel {name}: parameter {key!r}: cannot pass the "
            f"{val.python_type_name()}: an argument that is not a tensor or a scalar "
            "must be a Python literal, or a tuple or list of these"
        )

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
        constant_args: dict[str, Any] = {}
        graph_args: dict[VariableTracker, VariableTracker] = {}
        for key, val in combined_args.items():
            val = val.realize()
            graphable = self.is_tensor(val) or val.is_symnode_like()
            if graphable and not val.is_python_constant():
                graph_args[VariableTracker.build(tx, key)] = val
            else:
                constant_args[key] = self.constant_object(variable, key, val)
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
        # The @cute.jit function; not the wrapper object, which this variable does
        # not keep.
        if self.is_wrapper:
            return super().as_python_constant()
        return self.kernel

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        # A wrapper stands for its host function and write set, the only attributes
        # Dynamo reads of it: the host function as the variable built from its
        # source (guarded on its identity, see builder.py), and the write set as a
        # constant (guarded on its value).
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
