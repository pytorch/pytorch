"""
Dynamo support for user-defined CuTeDSL kernels (``@cute.jit`` host functions): the
counterpart of DynamoTritonHOPifier and TritonKernelVariable in functions.py.

A host function reaches Dynamo either raw (a global ``@cute.jit`` function) or
wrapped (``torch._library.cutedsl.wrap_cutedsl(fn, mutates_args=...)``, called inside
or outside the compiled region). Both become a CuteDSLKernelVariable, and a call of
either becomes a cutedsl_kernel_wrapper_mutation node. A raw call is written as for
eager::

    add_kernel(from_dlpack(x), from_dlpack(y), from_dlpack(out), n,
               cutlass.torch.current_stream())

``from_dlpack(t)`` becomes a CuteDSLTensorVariable standing for ``t``, the stream a
CuteDSLCurrentStreamVariable, and the call is captured as
``wrap_cutedsl(add_kernel, mutates_args=RAW_CALL_MUTATES_ARGS)(x, y, out, n)``.
"""

import inspect
from collections.abc import Callable, Sequence
from typing import Any, TYPE_CHECKING
from typing_extensions import Never

from torch._higher_order_ops.cutedsl_kernel_wrap import (
    cutedsl_kernel_side_table,
    cutedsl_kernel_wrapper_mutation,
    CuteDSLHOPifier,
    MutatesArgs,
    RAW_CALL_MUTATES_ARGS,
)

from ..exc import unimplemented
from .base import VariableTracker
from .functions import check_user_kernel_graph_args, split_user_kernel_args


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase
    from torch._guards import Source


class DynamoCuteDSLHOPifier(CuteDSLHOPifier):
    def raise_unsupported(self, msg: str) -> Never:
        unimplemented(
            gb_type="CuTeDSL kernel unsupported feature",
            context="",
            explanation=f"Encountered CuTeDSL kernel unsupported feature: {msg}",
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
        self, variable: "CuteDSLKernelVariable", val: VariableTracker
    ) -> VariableTracker:
        from .lists import BaseListVariable

        val = val.realize()
        if isinstance(val, BaseListVariable):
            return val.clone(
                items=[self.specialize_nested(variable, item) for item in val.items]
            )
        return variable.specialize_symbolic(val)

    def cute_tensor_arg(self, val: VariableTracker) -> VariableTracker | None:
        val = val.realize()
        return val.tensor if isinstance(val, CuteDSLTensorVariable) else None

    def is_current_stream(
        self, val: VariableTracker, tx: "InstructionTranslatorBase"
    ) -> bool:
        val = val.realize()
        return (
            isinstance(val, CuteDSLCurrentStreamVariable)
            and val.stream_id == _current_stream(tx)[1]
        )

    def wrap_raw(self, variable: "CuteDSLKernelVariable") -> "CuteDSLKernelVariable":
        return variable.with_mutates_args(RAW_CALL_MUTATES_ARGS)

    def call_HOP(
        self,
        variable: "CuteDSLKernelVariable",
        combined_args: dict[str, Any],
        tx: "InstructionTranslatorBase",
    ) -> VariableTracker:
        from ..guards import GuardBuilder, install_guard
        from .dicts import ConstDictVariable

        # Objects that are not Python constants (a config dataclass, a cutlass dtype
        # class or Numeric value) are constants of the launch too: pass the object,
        # guarded on its identity.
        objects = {}
        for key, val in list(combined_args.items()):
            val = val.realize()
            if (
                not (val.is_python_constant() or val.is_tensor() or val.is_symnode_like())
                and val.source is not None
                and hasattr(val, "value")
            ):
                install_guard(val.source.make_guard(GuardBuilder.ID_MATCH))
                objects[key] = val.value
                del combined_args[key]
        constant_args, non_constant_args = split_user_kernel_args(tx, combined_args)
        constant_args.update(objects)
        check_user_kernel_graph_args(
            non_constant_args, self.raise_unsupported, "CuTeDSL"
        )
        constant_args_idx = cutedsl_kernel_side_table.add_constant_args(constant_args)
        meta = ConstDictVariable(non_constant_args)
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

    def __init__(
        self,
        kernel: Callable[..., Any],
        mutates_args: object,
        kernel_idx: int | None,
        **kwargs: Any,
    ) -> None:
        self.kernel_source = kwargs.pop("kernel_source", kwargs.get("source"))
        # Whether this stands for a TraceableCuteDSLKernelWrapper (what wrap_cutedsl
        # returns) rather than the @cute.jit function itself.
        self.is_wrapper = kwargs.pop("is_wrapper", False)
        super().__init__(**kwargs)
        dynamo_cutedsl_hopifier_singleton.init_variable(
            self, kernel, mutates_args, kernel_idx
        )

    def python_type(self) -> type:
        if self.is_wrapper:
            from torch._higher_order_ops.cutedsl_kernel_wrap import (
                TraceableCuteDSLKernelWrapper,
            )

            return TraceableCuteDSLKernelWrapper
        return type(self.kernel)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # A wrapper made by wrap_cutedsl inside the region has no source of its own:
        # rebuild it from the host function's source.
        if self.kernel_source is None or self.mutates_args is None:
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
            self, list(args), kwargs, tx
        )

    def with_mutates_args(self, mutates_args: object) -> "CuteDSLKernelVariable":
        """The variable wrap_cutedsl(self.kernel, mutates_args=...) returns."""
        return CuteDSLKernelVariable(
            self.kernel,
            mutates_args,
            None,
            kernel_source=self.kernel_source,
            is_wrapper=True,
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


class CuteDSLFromDlpackVariable(VariableTracker):
    """``cutlass.cute.runtime.from_dlpack``, whose result stands for its tensor in a
    raw call. ``assumed_align`` is accepted and dropped: PT2 compiles the kernel for
    the alignment it can rely on, the pointer's own in the Python launcher and the
    export-time layout's (at most 16 bytes, checked at launch) under AOTInductor. The
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
            align = (
                align_vt.as_python_constant() if align_vt.is_python_constant() else 0
            )
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
        return CuteDSLTensorVariable(tensor, align)


class CuteDSLTensorVariable(VariableTracker):
    """``from_dlpack(tensor)`` made in the compiled region: stands for ``tensor`` in a
    raw call, whose launcher makes the cute tensor. It cannot be read in the region:
    an attribute or method of it (its element type, shape, ...) graph-breaks, and the
    real cute tensor is made at the break, as wherever the value leaves the graph."""

    def __init__(
        self, tensor: VariableTracker, assumed_align: int | None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.tensor = tensor
        self.assumed_align = assumed_align

    def python_type(self) -> type:
        from cutlass.cute.runtime import _Tensor

        return _Tensor

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
        import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]

        return cuda.CUstream

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
