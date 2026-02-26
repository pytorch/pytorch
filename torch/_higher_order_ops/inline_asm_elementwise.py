# mypy: allow-untyped-defs
"""
Inline ASM Elementwise Higher-Order Operator.

Provides inline PTX assembly support for both eager and compiled modes:
- Eager: JIT compiles CUDA kernels via Jiterator with inline asm
- Compiled: Lowers to tl.inline_asm_elementwise in Triton via Inductor

Example:
    from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise

    def fast_rsqrt(x):
        return inline_asm_elementwise(
            x,
            asm_str="rsqrt.approx.f32 $0, $1;",
            constraints="=f,f",
            dtype=torch.float32,
        )
"""

import functools
import re

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)

__all__ = ["inline_asm_elementwise"]


class InlineAsmElementwiseOp(HigherOrderOperator):
    """
    Elementwise inline PTX assembly operation.

    All tensor inputs are broadcast together before the operation.
    Uses Triton-style $N operand references (automatically converted
    to CUDA %N syntax for Jiterator).
    """

    def __init__(self):
        super().__init__("inline_asm_elementwise")

    def __call__(
        self,
        *inputs: torch.Tensor,
        asm_str: str,
        constraints: str,
        dtype: torch.dtype,
        is_pure: bool = True,
        pack: int = 1,
    ) -> torch.Tensor:
        return super().__call__(
            *inputs,
            asm_str=asm_str,
            constraints=constraints,
            dtype=dtype,
            is_pure=is_pure,
            pack=pack,
        )


inline_asm_elementwise = InlineAsmElementwiseOp()


# -----------------------------------------------------------------------------
# Constraint parsing utilities
# -----------------------------------------------------------------------------


def _parse_constraints(constraints: str) -> tuple[int, int]:
    """Parse constraint string to get (n_outputs, n_inputs)."""
    parts = [p.strip() for p in constraints.split(",")]
    n_outputs = sum(1 for p in parts if p.startswith("="))
    n_inputs = len(parts) - n_outputs
    return n_outputs, n_inputs


def _constraint_expects_fp32(constraint: str) -> bool:
    """Check if constraint expects fp32 input."""
    return constraint.lstrip("=") == "f"


def _should_upcast_to_fp32(dtype: torch.dtype) -> bool:
    """Check if dtype should be upcast to fp32 for PTX float operations."""
    return dtype in (torch.float16, torch.bfloat16)


# -----------------------------------------------------------------------------
# Eager implementation via Jiterator
# -----------------------------------------------------------------------------

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


def _triton_asm_to_cuda_asm(asm_str: str) -> str:
    """Convert Triton-style asm ($0, $1) to CUDA-style (%0, %1)."""
    return re.sub(r"\$(\d+)", r"%\1", asm_str)


@functools.lru_cache(maxsize=256)
def _get_jiterator_fn(
    asm_str: str,
    constraints: str,
    n_inputs: int,
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
):
    """Create and cache a Jiterator function for the given asm."""
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
    escaped_asm = cuda_asm.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

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
    """Dense (eager) implementation via Jiterator."""
    if not inputs:
        raise ValueError("inline_asm_elementwise requires at least one input tensor")

    inputs = torch.broadcast_tensors(*inputs)

    if not inputs[0].is_cuda:
        raise RuntimeError("inline_asm_elementwise only supports CUDA tensors")

    n_outputs, n_inputs = _parse_constraints(constraints)

    if n_outputs != 1:
        raise NotImplementedError(
            "Only single-output inline asm is currently supported"
        )

    if n_inputs != len(inputs):
        raise ValueError(
            f"Constraint string specifies {n_inputs} inputs but got {len(inputs)} tensors"
        )

    constraint_parts = [p.strip() for p in constraints.split(",")]
    input_constraints = [p for p in constraint_parts if not p.startswith("=")]

    processed_inputs = list(inputs)
    for i, (inp, constraint) in enumerate(zip(inputs, input_constraints)):
        if _constraint_expects_fp32(constraint) and _should_upcast_to_fp32(inp.dtype):
            processed_inputs[i] = inp.float()

    effective_input_dtype = processed_inputs[0].dtype

    jit_fn = _get_jiterator_fn(
        asm_str=asm_str,
        constraints=constraints,
        n_inputs=len(processed_inputs),
        input_dtype=effective_input_dtype,
        output_dtype=dtype,
    )

    result = jit_fn(*processed_inputs)

    # Jiterator returns input dtype; convert to requested output dtype if needed
    if result.dtype != dtype:
        result = result.to(dtype)

    return result


@inline_asm_elementwise.py_impl(DispatchKey.CompositeExplicitAutograd)
def inline_asm_eager(*inputs, asm_str, constraints, dtype, is_pure=True, pack=1):
    return _inline_asm_dense(
        *inputs,
        asm_str=asm_str,
        constraints=constraints,
        dtype=dtype,
        is_pure=is_pure,
        pack=pack,
    )


# -----------------------------------------------------------------------------
# Autograd - not implemented
# -----------------------------------------------------------------------------

inline_asm_elementwise.py_autograd_impl(
    autograd_not_implemented(inline_asm_elementwise, deferred_error=True)
)


# -----------------------------------------------------------------------------
# FakeTensor / Meta implementation
# -----------------------------------------------------------------------------


@inline_asm_elementwise.py_impl(FakeTensorMode)
def inline_asm_fake(mode, *inputs, asm_str, constraints, dtype, is_pure=True, pack=1):
    with mode:
        broadcasted = torch.broadcast_tensors(*inputs)
        return torch.empty_like(broadcasted[0], dtype=dtype)


@inline_asm_elementwise.py_impl(DispatchKey.Meta)
def inline_asm_meta(*inputs, asm_str, constraints, dtype, is_pure=True, pack=1):
    broadcasted = torch.broadcast_tensors(*inputs)
    return torch.empty_like(broadcasted[0], dtype=dtype)


# -----------------------------------------------------------------------------
# ProxyTorchDispatchMode for tracing
# -----------------------------------------------------------------------------


def trace_inline_asm(
    proxy_mode, func_overload, *inputs, asm_str, constraints, dtype, is_pure, pack
):
    """Trace inline_asm_elementwise through proxy mode."""
    with disable_proxy_modes_tracing():
        broadcasted = torch.broadcast_tensors(*inputs)
        out = torch.empty_like(broadcasted[0], dtype=dtype)

    node_args = inputs
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)

    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function",
        func_overload,
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

    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@inline_asm_elementwise.py_impl(ProxyTorchDispatchMode)
def inline_asm_proxy(mode, *inputs, asm_str, constraints, dtype, is_pure=True, pack=1):
    return trace_inline_asm(
        mode,
        inline_asm_elementwise,
        *inputs,
        asm_str=asm_str,
        constraints=constraints,
        dtype=dtype,
        is_pure=is_pure,
        pack=pack,
    )


# -----------------------------------------------------------------------------
# Functionalization
# -----------------------------------------------------------------------------


@inline_asm_elementwise.py_functionalize_impl
def inline_asm_func(ctx, *inputs, asm_str, constraints, dtype, is_pure=True, pack=1):
    unwrapped_inputs = ctx.unwrap_tensors(inputs)

    with ctx.redispatch_to_next():
        res = inline_asm_elementwise(
            *unwrapped_inputs,
            asm_str=asm_str,
            constraints=constraints,
            dtype=dtype,
            is_pure=is_pure,
            pack=pack,
        )
    return ctx.wrap_tensors(res)
