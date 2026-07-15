# mypy: allow-untyped-defs
from typing import Any, Optional

import sympy

import torch
from torch.utils._sympy.symbol import symbol_is_type, SymT

from .. import config
from ..runtime.hints import AttrsDescriptorWrapper
from ..utils import _type_of, expr_fits_within_32bit, triton_version_uses_attrs_dict
from ..virtualized import V
from .common import (
    ArgName,
    ConstexprArg,
    KernelArgType,
    SizeArg,
    TensorArg,
    TMADescriptorArg,
    WorkspaceArg,
)


def should_unwrap_unspec_arg(name: str):
    if V.graph.is_unspec_arg(name):
        # Unwrap on all devices except CPU
        if V.graph.get_current_device_or_throw().type != "cpu":
            return True
        # Only unwrap on CPU if the input is not used as an output
        if name not in V.graph.mutated_buffers:
            return True
    return False


def signature_of(arg: KernelArgType, *, size_dtype: Optional[str]) -> str:
    if isinstance(arg, TensorArg):
        # TODO: Remove fp8 special handling when Triton supports PyTorch fp8 dtypes.
        # Related PR: https://github.com/triton-lang/triton/pull/2279/
        if arg.dtype == torch.float8_e4m3fn:
            typ = "*fp8e4nv"
        elif arg.dtype == torch.float8_e5m2:
            typ = "*fp8e5"
        elif arg.dtype == torch.float8_e4m3fnuz:
            typ = "*fp8e4b8"
        elif arg.dtype == torch.float8_e5m2fnuz:
            typ = "*fp8e5b16"
        else:
            typ = _type_of(arg.dtype)
        if should_unwrap_unspec_arg(arg.buffer):
            # had unwrapped 0d tensor as scalar
            new_typ = typ.lstrip("*")
            if new_typ in ["fp16", "bf16"]:
                return "fp32"
            else:
                return new_typ
        else:
            return typ
    if isinstance(arg, SizeArg):
        if arg.expr is None:
            if triton_version_uses_attrs_dict():
                # In newer versions of Triton, the signature includes "None" args
                # and their type is marked as "constexpr"
                return "constexpr"
            else:
                # In older versions of Triton...
                # From triton/runtime/jit.py
                # `None` is nullptr.  Implicitly convert to *i8.
                return "*i8"
        elif _arg_equals_1(arg) and triton_version_uses_attrs_dict():
            # In new versions of Triton, if we have an equal-to-1 arg that's marked as a constant,
            # it should be marked as "constexpr" in the signature.
            return "constexpr"
        elif isinstance(arg.expr, (float, sympy.Float)):
            # Python floats are natively fp64, so use fp64 to preserve precision
            return "fp64" if config._use_fp64_for_unbacked_floats else "fp32"
        elif isinstance(arg.expr, sympy.Symbol) and symbol_is_type(
            arg.expr, (SymT.UNBACKED_FLOAT)
        ):
            # Unbacked floats from .item() should preserve fp64 precision
            return "fp64" if config._use_fp64_for_unbacked_floats else "fp32"
        elif isinstance(arg.expr, bool):
            return "i1"

        # if this is a integer
        if size_dtype == "tl.int32":
            return "i32"
        elif size_dtype == "tl.int64":
            return "i64"
        elif size_dtype is None:
            # no hint: we'll see if we know that this is a 32-bit int, and guard if possible.
            int_max = torch.iinfo(torch.int32).max
            if expr_fits_within_32bit(arg.expr):
                V.graph.sizevars.check_leq(arg.expr, int_max)
                return "i32"
            else:
                return "i64"
        else:
            raise NotImplementedError(f"unhandled size_dtype {size_dtype}")
    if isinstance(arg, WorkspaceArg):
        return _type_of(arg.dtype)
    if isinstance(arg, TMADescriptorArg):
        if arg.api_type == "experimental":
            return "nvTmaDesc"
        else:
            # https://github.com/triton-lang/triton/blob/9695baed9b46cf957e08b157bb4133f4a4b331c5/python/triton/runtime/jit.py#L360-L363
            assert arg.api_type == "stable"
            assert arg.block_shape is not None
            assert arg.dtype is not None
            inner = _type_of(arg.dtype)[1:]  # strip the `*`: *fp32 -> fp32
            return f"tensordesc<{inner}{list(arg.block_shape)}>"
    if isinstance(arg, ConstexprArg):
        return "constexpr"
    raise NotImplementedError(f"unhandled {type(arg)}: {arg}")


def non_constexpr_signature(signature):
    new_signature = []
    for arg in signature:
        if not isinstance(arg, ConstexprArg):
            new_signature.append(arg)

    return new_signature


def signature_to_meta(
    signature: list[KernelArgType],
    *,
    size_dtype: Optional[str],
    argdefs: list[ArgName],
    indices: Optional[list[int]] = None,
    is_template: bool = False,
) -> dict[str, str]:
    if indices is None:
        indices = list(range(len(signature)))

    def _decide_tl_dtype(arg):
        # Even if the ks0 symbol itself is within tl.int32 range, it's
        # risky to use tl.int32 dtype since we may have ks0*ks1 later
        # for kernels like torch.mean when dynamic shape is enabled.
        #
        # Check config.triton.use_block_ptr, since Triton block pointer
        # does not support 64bit indexing:
        # https://gist.github.com/shunting314/6a41c776171720ce4561f202dcde0ad6
        #
        # If the triton metadata is for a template, don't use tl.int64 index.
        # Templates like flex attention/decoding uses block pointers which
        # does not support 64 bit indexing.
        if (
            not config.triton.use_block_ptr
            and not is_template
            and isinstance(arg, SizeArg)
            and arg.name.startswith("ks")
        ):
            return "tl.int64"
        return size_dtype

    return {
        argdefs[i].name: signature_of(arg, size_dtype=_decide_tl_dtype(arg))
        for i, arg in zip(indices, signature)
    }


def is_unaligned_buffer(arg: TensorArg):
    buf_name = arg.buffer
    if buf_name in V.graph.unaligned_buffers:
        return True

    if buf_name in V.graph.graph_inputs:
        # See Note: [Input Alignment handling in Inductor]
        # For graph inputs that is not recorded in V.graph.unaligned_buffers,
        # we know for sure the tensor is aligned.
        return False

    if buf_name in V.graph.constants:
        # all constants are assumed to be aligned
        return False

    if V.graph.scheduler:
        layout = V.graph.scheduler.get_buffer_layout(buf_name)
    else:
        buffer = V.graph.try_get_buffer(buf_name)
        # output arg
        if not buffer:
            assert buf_name == V.kernel.output_node.name
            layout = V.kernel.output_node.layout
        else:
            layout = buffer.get_layout()

    if isinstance(layout, torch._inductor.ir.NonOwningLayout):
        return not layout.maybe_guard_aligned()
    else:
        return False


def _arg_equals_1(arg: KernelArgType) -> bool:
    return (
        isinstance(arg, SizeArg)
        and isinstance(arg.expr, (int, sympy.Integer))
        and V.graph.sizevars.statically_known_equals(arg.expr, 1)  # type: ignore[arg-type]
    )


def equal_1_arg_indices(
    args: list[KernelArgType],
    *,
    indices: Optional[list[int]] = None,
) -> tuple[int, ...]:
    if indices is None:
        indices = list(range(len(args)))

    equal_to_1 = tuple(i for i, arg in zip(indices, args) if _arg_equals_1(arg))

    return equal_to_1


def config_of(
    args: list[KernelArgType],
    *,
    indices: Optional[list[int]] = None,
) -> Any:
    if indices is None:
        indices = list(range(len(args)))

    def is_aligned(x: KernelArgType, alignment: int, include_tensor: bool) -> bool:
        """
        Roughly follow triton code here:
        https://github.com/triton-lang/triton/blob/5282ed890d453e10b9ee30076ef89115dd197761/python/triton/runtime/jit.py#L208-L222
        """
        if isinstance(x, TensorArg):
            if include_tensor:
                offset_aligned = V.graph.sizevars.statically_known_multiple_of(
                    x.offset * x.dtype.itemsize,
                    alignment,  # type: ignore[arg-type]
                )
                return offset_aligned and not is_unaligned_buffer(x)
            else:
                return False
        if isinstance(x, SizeArg):
            # TODO(voz): These are kinda redundant, if we can solve out statically_known_multiple_of with
            # _maybe_evaluate_static...
            if x.name.startswith("load_seed_offset"):
                return False
            if x.expr is None:
                return False
            if isinstance(x.expr, float):
                return False
            return V.graph.sizevars.statically_known_multiple_of(x.expr, alignment)  # type: ignore[arg-type]
        if isinstance(x, WorkspaceArg):
            # We allocate the workspace ourselves, so it is always aligned
            return True
        if isinstance(x, (TMADescriptorArg, ConstexprArg)):
            return False
        raise NotImplementedError(f"unhandled {type(x)}: {x}")

    if config.triton.divisible_by_16:
        divisible_by_16 = tuple(
            i
            for i, arg in zip(indices, args)
            if is_aligned(arg, alignment=16, include_tensor=True)
        )
    else:
        divisible_by_16 = ()

    equal_to_1 = equal_1_arg_indices(args, indices=indices)

    # pyrefly: ignore [bad-argument-type]
    return AttrsDescriptorWrapper(divisible_by_16, equal_to_1)
