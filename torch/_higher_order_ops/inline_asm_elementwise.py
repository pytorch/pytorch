# mypy: allow-untyped-defs
import functools
import re
from typing import overload

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import autograd_not_implemented, register_fake
from torch._ops import HigherOrderOperator
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree


__all__ = ["inline_asm_elementwise"]


def _parse_constraints(constraints: str) -> tuple[int, int]:
    parts = [p.strip() for p in constraints.split(",")]
    n_outputs = sum(1 for p in parts if p.startswith("="))
    n_inputs = len(parts) - n_outputs
    return n_outputs, n_inputs


class InlineAsmElementwiseOp(HigherOrderOperator):
    """Execute inline PTX assembly elementwise over tensors.

    This is an elementwise map where the function body is inline assembly.
    Input tensors are implicitly broadcast to the same shape.

    Each invocation of the inline asm processes ``pack`` elements at a time.
    Exactly which set of inputs a given invocation receives is unspecified.

    Output strides follow PyTorch's standard pointwise striding propagation
    rules.

    In eager mode, the assembly is executed via the CUDA Jiterator.  Under
    ``torch.compile`` the assembly is lowered to Triton's
    ``tl.inline_asm_elementwise`` via Inductor, which allows fusion with
    surrounding operators. Multiple outputs share one assembly invocation when
    their consumers are fused; consumers in separate kernels may re-execute it.

    Args:
        *inputs: Input tensors whose values are passed to the asm block.
        asm_str: PTX assembly string. Operands use ``$N`` syntax
            (e.g. ``$0`` for the first output, ``$1`` for the first input).
        constraints: Inline-asm constraints in LLVM format. Output constraints
            are prefixed with ``=`` (e.g. ``"=f,f,f"`` for one float output
            and two float inputs).
        dtype: Element type of the returned tensor, or a tuple of element types
            for multiple output tensors. Multiple outputs require
            ``torch.compile``.
        is_pure: Must be ``True``. If true, the compiler may assume the asm
            block has no side-effects.
        pack: Number of elements processed per asm invocation.  When
            ``pack > 1``, the constraint string must describe the packed
            register layout expected by Triton. Sub-32-bit values may share a
            register. Requires ``torch.compile``.

    Returns:
        A tensor, or tuple of tensors, with the broadcast shape of the inputs
        and the given dtype or dtypes.

    Example::

        >>> # xdoctest: +SKIP(requires CUDA)
        >>> # Float32 fused multiply-add via PTX
        >>> result = inline_asm_elementwise(
        ...     a, b, c,
        ...     asm_str="fma.rn.f32 $0, $1, $2, $3;",
        ...     constraints="=f,f,f,f",
        ...     dtype=torch.float32,
        ... )

        >>> # xdoctest: +SKIP(requires CUDA)
        >>> # pack=2: each asm invocation processes two elements
        >>> result = inline_asm_elementwise(
        ...     x,
        ...     asm_str="mov.b32 $0, $2; mov.b32 $1, $3;",
        ...     constraints="=r,=r,r,r",
        ...     dtype=torch.float32,
        ...     pack=2,
        ... )

        >>> # xdoctest: +SKIP(requires CUDA and torch.compile)
        >>> first, second = inline_asm_elementwise(
        ...     x,
        ...     asm_str="mov.b32 $0, $2; add.s32 $1, $2, 1;",
        ...     constraints="=r,=r,r",
        ...     dtype=(torch.int32, torch.int32),
        ... )
    """

    def __init__(self):
        super().__init__("inline_asm_elementwise")

    @overload
    def __call__(
        self,
        *inputs: torch.Tensor,
        asm_str: str,
        constraints: str,
        dtype: torch.dtype,
        is_pure: bool = True,
        pack: int = 1,
    ) -> torch.Tensor: ...

    @overload
    def __call__(
        self,
        *inputs: torch.Tensor,
        asm_str: str,
        constraints: str,
        dtype: tuple[torch.dtype, ...],
        is_pure: bool = True,
        pack: int = 1,
    ) -> tuple[torch.Tensor, ...]: ...

    def __call__(
        self,
        *inputs: torch.Tensor,
        asm_str: str,
        constraints: str,
        dtype: torch.dtype | tuple[torch.dtype, ...],
        is_pure: bool = True,
        pack: int = 1,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if not is_pure:
            raise ValueError("inline_asm_elementwise only supports is_pure=True")
        if isinstance(dtype, tuple) and not dtype:
            raise ValueError("inline_asm_elementwise requires at least one output")
        output_dtypes = dtype if isinstance(dtype, tuple) else (dtype,)
        n_outputs, n_inputs = _parse_constraints(constraints)
        if pack == 1:
            if n_outputs != len(output_dtypes):
                raise ValueError(
                    f"Expected {len(output_dtypes)} output constraint(s), "
                    f"got {n_outputs}"
                )
            if n_inputs != len(inputs):
                raise ValueError(
                    f"Constraint string specifies {n_inputs} inputs but got "
                    f"{len(inputs)} tensor(s)"
                )
        else:
            # Sub-32-bit values may be packed into one register, so the exact
            # constraint count is backend- and dtype-dependent. Each logical
            # input and output must still have at least one constraint.
            if n_outputs < len(output_dtypes):
                raise ValueError(
                    f"Expected at least {len(output_dtypes)} output constraint(s) "
                    f"for pack={pack}, got {n_outputs}"
                )
            if n_inputs < len(inputs):
                raise ValueError(
                    f"Expected at least {len(inputs)} input constraint(s) "
                    f"for pack={pack}, got {n_inputs}"
                )
        # pyrefly: ignore [missing-attribute]
        return super().__call__(
            *inputs,
            asm_str=asm_str,
            constraints=constraints,
            dtype=dtype,
            is_pure=True,
            pack=pack,
        )


inline_asm_elementwise = InlineAsmElementwiseOp()


_DTYPE_TO_CUDA_TYPE = {
    torch.float32: "float",
    torch.float64: "double",
    torch.float16: "__half",
    torch.bfloat16: "__nv_bfloat16",
    torch.int32: "int",
    torch.int64: "long long",
    torch.int16: "short",
    torch.int8: "signed char",
    torch.uint8: "unsigned char",
    torch.uint16: "unsigned short",
    torch.uint32: "unsigned int",
    torch.bool: "bool",
}


_TRITON_ARG_RE = re.compile(r"\$(\d+)")


def _triton_asm_to_cuda_asm(asm_str: str) -> str:
    return _TRITON_ARG_RE.sub(r"%\1", asm_str)


@functools.lru_cache
def _get_jiterator_fn(
    asm_str: str,
    constraints: str,
    n_inputs: int,
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
):
    from torch.cuda.jiterator import _create_jit_fn

    cuda_asm = _triton_asm_to_cuda_asm(asm_str)

    constraint_parts = [p.strip() for p in constraints.split(",")]
    output_constraints = [p.lstrip("=") for p in constraint_parts if p.startswith("=")]
    input_constraints = [p for p in constraint_parts if not p.startswith("=")]

    if input_dtype not in _DTYPE_TO_CUDA_TYPE:
        raise ValueError(f"Unsupported input dtype for inline asm: {input_dtype}")
    if output_dtype not in _DTYPE_TO_CUDA_TYPE:
        raise ValueError(f"Unsupported output dtype for inline asm: {output_dtype}")

    input_type = _DTYPE_TO_CUDA_TYPE[input_dtype]
    output_type = _DTYPE_TO_CUDA_TYPE[output_dtype]

    input_params = ", ".join(f"{input_type} in{i}" for i in range(n_inputs))
    out_constraints_str = ", ".join(f'"={c}"(result)' for c in output_constraints)
    in_constraints_str = ", ".join(
        f'"{c}"(in{i})' for i, c in enumerate(input_constraints)
    )
    escaped_asm = (
        cuda_asm.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    )

    code = f"""
template <typename T>
{output_type} inline_asm_kernel({input_params}) {{
    {output_type} result;
    asm volatile (
        "{escaped_asm}"
        : {out_constraints_str}
        : {in_constraints_str}
    );
    return result;
}}
"""

    return _create_jit_fn(code)


def _inline_asm_dense(*inputs, asm_str, constraints, dtype, is_pure, pack):
    if not inputs:
        raise ValueError("inline_asm_elementwise requires at least one input tensor")

    inputs = torch.broadcast_tensors(*inputs)

    if not inputs[0].is_cuda:
        raise RuntimeError("inline_asm_elementwise only supports CUDA tensors")

    if pack > 1 or isinstance(dtype, tuple):
        raise RuntimeError(
            "inline_asm_elementwise requires torch.compile for pack > 1 or multiple outputs"
        )

    n_outputs, n_inputs = _parse_constraints(constraints)

    if n_outputs != 1:
        raise ValueError(f"Expected 1 output constraint, got {n_outputs}")

    if n_inputs != len(inputs):
        raise ValueError(
            f"Constraint string specifies {n_inputs} inputs but got "
            f"{len(inputs)} tensor(s)"
        )

    # Jiterator generates a single input type for all inputs — mixed dtypes
    # would produce incorrect CUDA code.
    input_dtypes = {inp.dtype for inp in inputs}
    if len(input_dtypes) > 1:
        raise ValueError(
            f"All inputs must have the same dtype for eager execution, "
            f"got {sorted(str(d) for d in input_dtypes)}"
        )

    jit_fn = _get_jiterator_fn(
        asm_str=asm_str,
        constraints=constraints,
        n_inputs=len(inputs),
        input_dtype=inputs[0].dtype,
        output_dtype=dtype,
    )

    return jit_fn(*inputs)


@inline_asm_elementwise.py_impl(DispatchKey.CompositeExplicitAutograd)
def _(*inputs, asm_str, constraints, dtype, is_pure=True, pack=1):
    return _inline_asm_dense(
        *inputs,
        asm_str=asm_str,
        constraints=constraints,
        dtype=dtype,
        is_pure=is_pure,
        pack=pack,
    )


inline_asm_elementwise.py_autograd_impl(
    autograd_not_implemented(inline_asm_elementwise, deferred_error=True)
)


@inline_asm_elementwise.py_impl(torch._C._functorch.TransformType.Vmap)
def _(interpreter, *inputs, asm_str, constraints, dtype, is_pure=True, pack=1):
    from torch._functorch.vmap import unwrap_batched, wrap_batched

    op = functools.partial(
        inline_asm_elementwise,
        asm_str=asm_str,
        constraints=constraints,
        dtype=dtype,
        is_pure=is_pure,
        pack=pack,
    )
    unwrapped, bdims = unwrap_batched(list(inputs), interpreter.level())
    if all(bdim is None for bdim in bdims):
        # No input is batched at this level (e.g. this vmap level batches only
        # tensors that are not asm inputs); the result has no batch dim to wrap.
        with interpreter.lower():
            return op(*unwrapped)
    # The op is elementwise, so batching is a broadcast: move each batch dim to
    # the front and pad the logical dims so they stay right-aligned across all
    # inputs while the batch dim broadcasts only against other batch dims.
    logical_ndim = max(
        t.ndim - (0 if bdim is None else 1) for t, bdim in zip(unwrapped, bdims)
    )
    moved = []
    for t, bdim in zip(unwrapped, bdims):
        if bdim is not None:
            t = t.movedim(bdim, 0)
            t = t[(slice(None), *([None] * (logical_ndim - (t.ndim - 1))))]
        moved.append(t)
    with interpreter.lower():
        res = op(*moved)
    return wrap_batched((res,), (0,), interpreter.level())[0]


def _elementwise_output_like(*inputs, dtype):
    from torch._prims_common import compute_elementwise_output_logical_to_physical_perm

    broadcasted = torch.broadcast_tensors(*inputs)
    l2p_perm, _ = compute_elementwise_output_logical_to_physical_perm(*broadcasted)
    return torch.empty_permuted(
        broadcasted[0].shape, l2p_perm, dtype=dtype, device=broadcasted[0].device
    )


@register_fake(inline_asm_elementwise, skip_cache=True)
def _(*inputs, asm_str, constraints, dtype, is_pure=True, pack=1):
    if isinstance(dtype, tuple):
        return tuple(_elementwise_output_like(*inputs, dtype=dt) for dt in dtype)
    return _elementwise_output_like(*inputs, dtype=dtype)


@inline_asm_elementwise.py_impl(ProxyTorchDispatchMode)
def _(mode, *inputs, asm_str, constraints, dtype, is_pure=True, pack=1):
    proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, inputs)

    out_proxy = mode.tracer.create_proxy(
        "call_function",
        inline_asm_elementwise,
        proxy_args,
        {
            "asm_str": asm_str,
            "constraints": constraints,
            "dtype": dtype,
            "is_pure": is_pure,
            "pack": pack,
        },
        name="inline_asm_elementwise",
    )

    out = inline_asm_elementwise(
        *inputs,
        asm_str=asm_str,
        constraints=constraints,
        dtype=dtype,
        is_pure=is_pure,
        pack=pack,
    )
    return track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)


@inline_asm_elementwise.py_functionalize_impl
def _(ctx, *inputs, asm_str, constraints, dtype, is_pure=True, pack=1):
    unwrapped_inputs = ctx.unwrap_tensors(inputs)

    with ctx.redispatch_to_next():
        res = inline_asm_elementwise(
            *unwrapped_inputs,
            asm_str=asm_str,
            constraints=constraints,
            dtype=dtype,
            pack=pack,
        )
    return ctx.wrap_tensors(res)
