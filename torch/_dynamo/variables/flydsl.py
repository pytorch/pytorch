"""
Dynamo support for user-defined FlyDSL kernels (``@flyc.jit`` host functions): the
counterpart of DynamoTritonHOPifier and TritonKernelVariable in functions.py.

A host function reaches Dynamo either raw (a global ``@flyc.jit`` function) or
wrapped (``torch._library.flydsl.wrap_flydsl(fn, mutates_args=...)``, called inside
or outside the compiled region). Both become a FlyDSLKernelVariable; only a wrapped
one, which knows its write set, can be called, and the call becomes a
flydsl_kernel_wrapper_mutation node.
"""

import types
from collections.abc import Callable, Sequence
from typing import Any, TYPE_CHECKING
from typing_extensions import Never

from torch._higher_order_ops.flydsl_kernel_wrap import FlyDSLHOPifier, MutatesArgs

from ..exc import unimplemented
from .base import VariableTracker
from .functions import (
    emit_user_kernel_hop,
    NestedUserFunctionVariable,
    specialize_symbolic_variable,
    split_user_kernel_args,
    UserFunctionVariable,
)


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase
    from torch._guards import Source


def _guard_function_constant(fn: types.FunctionType, source: "Source") -> None:
    """Guard a function constant passed into the frame (a Constexpr lambda) on what
    FlyDSL compiles in besides its code, which Dynamo guards already (CLOSURE_MATCH):
    the values of its defaults, types included, so an equal lambda made per call
    reuses the graph. Anything else (keyword-only defaults, a closure, defaults that
    are not bool/int/float/str values) on the function itself."""
    from ..guards import GuardBuilder, install_guard
    from ..source import AttrSource, DefaultsSource

    if not (fn.__defaults__ or fn.__kwdefaults__ or fn.__closure__):
        return
    defaults = fn.__defaults__ or ()
    if (
        fn.__kwdefaults__
        or fn.__closure__
        or any(type(d) not in (bool, int, float, str) for d in defaults)
    ):
        install_guard(source.make_guard(GuardBuilder.ID_MATCH))
        return
    install_guard(
        AttrSource(source, "__defaults__").make_guard(GuardBuilder.SEQUENCE_LENGTH)
    )
    for i in range(len(defaults)):
        install_guard(DefaultsSource(source, i).make_guard(GuardBuilder.CONSTANT_MATCH))


class DynamoFlyDSLHOPifier(FlyDSLHOPifier):
    def raise_unsupported(self, msg: str) -> Never:
        unimplemented(
            gb_type="FlyDSL kernel unsupported feature",
            context="",
            explanation=f"Encountered FlyDSL kernel unsupported feature: {msg}",
            hints=[],
        )

    def get_value(self, val: VariableTracker) -> Any:
        return val.as_python_constant()

    def is_tensor(self, val: VariableTracker) -> bool:
        return val.realize().is_tensor()

    def python_type(self, val: VariableTracker) -> type:
        return val.realize().python_type()

    def leaves(self, val: VariableTracker) -> list[VariableTracker]:
        from .lists import BaseListVariable

        realized = val.realize()
        if isinstance(realized, BaseListVariable):
            return [leaf for item in realized.items for leaf in self.leaves(item)]
        return [val]

    def specialize_nested(
        self, variable: "FlyDSLKernelVariable", val: VariableTracker
    ) -> VariableTracker:
        from .lists import BaseListVariable

        val = val.realize()
        if isinstance(val, BaseListVariable):
            return val.clone(
                items=[self.specialize_nested(variable, item) for item in val.items]
            )
        if (
            isinstance(val, NestedUserFunctionVariable)
            and val.defaults is not None
            and val.get_code().co_name == "<lambda>"
        ):
            # A lambda made in the frame takes frame values through its defaults
            # (FlyDSL refuses closures and keyword-only defaults), and FlyDSL bakes
            # them in with the lambda.
            return val.clone(defaults=self.specialize_nested(variable, val.defaults))
        return variable.specialize_symbolic(val)

    def call_HOP(
        self,
        variable: "FlyDSLKernelVariable",
        combined_args: dict[str, Any],
        tx: "InstructionTranslatorBase",
    ) -> VariableTracker:
        from torch._higher_order_ops.flydsl_kernel_wrap import FLYDSL_CONSTANT_RULES

        from ..guards import GuardBuilder, install_guard
        from ..source import AttrSource, ChainedSource
        from .lists import BaseListVariable

        # Function constants (Constexpr lambdas), as arguments or inside list/tuple
        # arguments: the ones passed in are guarded (_guard_function_constant), the
        # ones made in the frame are recorded by argument and index path.
        nested_functions: list[tuple[str, tuple[int, ...]]] = []

        def visit_functions(
            key: str, path: tuple[int, ...], val: VariableTracker
        ) -> None:
            val = val.realize()
            if isinstance(val, BaseListVariable):
                for i, item in enumerate(val.items):
                    visit_functions(key, (*path, i), item)
            elif isinstance(val, UserFunctionVariable) and val.source is not None:
                _guard_function_constant(val.get_function(), val.source)
            elif isinstance(val, NestedUserFunctionVariable):
                nested_functions.append((key, path))

        # Objects that are not Python constants (a config dataclass, a FlyDSL dtype
        # class or Numeric value) are constants of the launch too: pass the object,
        # guarded on its identity, a Numeric on its type and value, by which the side
        # table keys (and copies) it.
        objects = {}
        for key, val in list(combined_args.items()):
            val = val.realize()
            if isinstance(
                val,
                (BaseListVariable, UserFunctionVariable, NestedUserFunctionVariable),
            ):
                visit_functions(key, (), val)
            elif (
                not (
                    val.is_python_constant() or val.is_tensor() or val.is_symnode_like()
                )
                and val.source is not None
                and hasattr(val, "value")
            ):
                # Dynamo applies a write made earlier in the region only after the
                # graph runs, so the launch would not see a write to the object, or to
                # anything reached through it: a graph break.
                side_effects = tx.output.side_effects
                for other in side_effects.id_to_variable.values():
                    source = other.source
                    if source is None or not side_effects.is_modified(other):
                        continue
                    while isinstance(source, ChainedSource) and source != val.source:
                        source = source.base
                    if source == val.source:
                        name = self.hop_info.get_kernel_name(
                            {"kernel_idx": variable.kernel_idx}
                        )
                        self.raise_unsupported(
                            f"{name}(): {other.source.name}, reached through {key!r}, "
                            "is modified earlier in the compiled region, and the kernel "
                            "would not see the change"
                        )
                if FLYDSL_CONSTANT_RULES.scalar_value(val.value) is not None:
                    install_guard(
                        val.source.make_guard(GuardBuilder.TYPE_MATCH),
                        AttrSource(val.source, "value").make_guard(
                            GuardBuilder.CONSTANT_MATCH
                        ),
                    )
                else:
                    install_guard(val.source.make_guard(GuardBuilder.ID_MATCH))
                objects[key] = val.value
                del combined_args[key]
        constant_args, non_constant_args = split_user_kernel_args(tx, combined_args)
        # Under Dynamo, a function made in the region (a Constexpr lambda) has its
        # qualified name as __name__, but FlyDSL recognizes a Constexpr lambda by the
        # name CPython gives it: its code's name, "<lambda>". as_python_constant()
        # made these function objects, so renaming them changes no user object.
        for key, path in nested_functions:
            fn = constant_args.get(key)
            for i in path:
                fn = fn[i] if isinstance(fn, (tuple, list)) else None
            if isinstance(fn, types.FunctionType):
                fn.__name__ = fn.__code__.co_name
        constant_args.update(objects)
        return emit_user_kernel_hop(
            tx,
            self.hop_info,
            variable.kernel_idx,
            constant_args,
            non_constant_args,
            self.raise_unsupported,
        )


dynamo_flydsl_hopifier_singleton = DynamoFlyDSLHOPifier()


class FlyDSLKernelVariable(VariableTracker):
    kernel: Callable[..., Any]
    mutates_args: MutatesArgs | None
    kernel_idx: int | None
    kernel_source: "Source | None"

    def __init__(
        self,
        kernel: Callable[..., Any],
        mutates_args: object,
        kernel_idx: int | None,
        **kwargs: Any,
    ) -> None:
        self.kernel_source = kwargs.pop("kernel_source", kwargs.get("source"))
        # Whether this stands for a TraceableFlyDSLKernelWrapper (what wrap_flydsl
        # returns) rather than the @flyc.jit function itself.
        self.is_wrapper = kwargs.pop("is_wrapper", False)
        super().__init__(**kwargs)
        dynamo_flydsl_hopifier_singleton.init_variable(
            self, kernel, mutates_args, kernel_idx
        )

    def python_type(self) -> type:
        if self.is_wrapper:
            from torch._higher_order_ops.flydsl_kernel_wrap import (
                TraceableFlyDSLKernelWrapper,
            )

            return TraceableFlyDSLKernelWrapper
        return type(self.kernel)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # A wrapper made by wrap_flydsl inside the region has no source of its own:
        # rebuild it from the host function's source.
        if self.kernel_source is None or self.mutates_args is None:
            return super().reconstruct(codegen)
        codegen.add_push_null(
            lambda: codegen.load_import_from("torch._library.flydsl", "wrap_flydsl")
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
        return dynamo_flydsl_hopifier_singleton.call_flydsl_kernel(
            self, list(args), kwargs, tx
        )

    def with_mutates_args(self, mutates_args: object) -> "FlyDSLKernelVariable":
        """The variable wrap_flydsl(self.kernel, mutates_args=...) returns."""
        return FlyDSLKernelVariable(
            self.kernel,
            mutates_args,
            None,
            kernel_source=self.kernel_source,
            is_wrapper=True,
        )

    def specialize_symbolic(self, arg: Any) -> Any:
        return specialize_symbolic_variable(arg)
