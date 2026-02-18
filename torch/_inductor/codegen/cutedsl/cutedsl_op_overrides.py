# mypy: allow-untyped-defs
"""
CuteDSL-specific operation overrides for pointwise operations.

This module provides CuteDSL implementations of common operations used in
template kernels, particularly for flex attention modifications.
"""

import math
from typing import Optional, Union

import sympy

import torch
from torch._inductor.codegen.common import CSEVariable, OpOverrides
from torch._inductor.virtualized import OpsValue, V
from torch.utils._sympy.value_ranges import ValueRanges


CuteDSLArg = Union[CSEVariable, str]


def upcast_compute_type(dtype: torch.dtype) -> torch.dtype:
    """Maybe upcast [b]float16 to float32"""
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


class CuteDSLOpOverrides(OpOverrides):
    """
    CuteDSL-specific operation overrides that generate code using CuteDSL syntax.

    CuteDSL TensorSSA objects have built-in operator overloads (__add__, __mul__, etc.)
    and math functions (cute.math.exp, cute.math.sqrt, etc.)
    """

    TORCH_TO_CUTE_DTYPE = {
        torch.float16: "cutlass.Float16",
        torch.bfloat16: "cutlass.BFloat16",
        torch.float32: "cutlass.Float32",
        torch.float64: "cutlass.Float64",
        torch.int8: "cutlass.Int8",
        torch.int16: "cutlass.Int16",
        torch.int32: "cutlass.Int32",
        torch.int64: "cutlass.Int64",
        torch.bool: "cutlass.Boolean",
        torch.float8_e4m3fn: "cutlass.Float8E4M3FN",
        torch.float8_e5m2: "cutlass.Float8E5M2",
    }

    # Math constants
    LOG2_E = 1.4426950408889634  # 1/ln(2) for converting natural exp to base-2 exp

    @staticmethod
    def _get_cse_var(arg: CuteDSLArg) -> Optional[CSEVariable]:
        """Extract CSEVariable from arg if it's a tensor (either direct or wrapped in OpsValue)."""
        if isinstance(arg, CSEVariable):
            return arg
        if isinstance(arg, OpsValue) and isinstance(arg.value, CSEVariable):
            return arg.value
        return None

    @staticmethod
    def _ensure_tensor_ssa(arg: CuteDSLArg, template_tensor: CuteDSLArg) -> str:
        """
        Convert scalar arguments to TensorSSA using cute.full_like if needed.

        Args:
            arg: The argument to check (CSEVariable for tensors, str for scalars, or OpsValue wrapper)
            template_tensor: A tensor argument to use as template for full_like

        Returns:
            String representation suitable for CuteDSL operations
        """
        if isinstance(arg, CSEVariable):
            return str(arg)

        if isinstance(arg, OpsValue) and isinstance(arg.value, CSEVariable):
            return str(arg.value)

        if isinstance(template_tensor, CSEVariable):
            return f"cute.full_like({template_tensor}, {arg})"

        return str(arg)

    @staticmethod
    def _extract_dtype_and_bounds(
        *args: CuteDSLArg,
    ) -> tuple[Optional[torch.dtype], ValueRanges[sympy.Expr]]:
        """Extract dtype and bounds from CSEVariable arguments (including OpsValue wrappers)."""
        for arg in args:
            cse_var = CuteDSLOpOverrides._get_cse_var(arg)
            if cse_var is not None:
                return cse_var.dtype, cse_var.bounds
        return None, ValueRanges.unknown()

    @staticmethod
    def _apply_binary_op(a: CuteDSLArg, b: CuteDSLArg, op_format: str) -> CuteDSLArg:
        """
        Apply a binary operation with automatic scalar-to-tensor conversion.

        CuteDSL requires both operands to be TensorSSA objects for tensor operations.
        This helper automatically converts scalar arguments to TensorSSA using
        cute.full_like when at least one argument is a tensor (CSEVariable or OpsValue).

        Args:
            a: First operand (CSEVariable for tensors, str for scalars, or OpsValue wrapper)
            b: Second operand (CSEVariable for tensors, str for scalars, or OpsValue wrapper)
            op_format: Format string with {a} and {b} placeholders for the operation

        Returns:
            CSEVariable if at least one operand is a tensor, otherwise string
        """
        # Check for CSEVariable directly or wrapped in OpsValue
        tensor_arg = CuteDSLOpOverrides._get_cse_var(
            a
        ) or CuteDSLOpOverrides._get_cse_var(b)
        if tensor_arg is not None:
            a_ssa = CuteDSLOpOverrides._ensure_tensor_ssa(a, tensor_arg)
            b_ssa = CuteDSLOpOverrides._ensure_tensor_ssa(b, tensor_arg)
            result_expr = op_format.format(a=a_ssa, b=b_ssa)

            dtype, bounds = CuteDSLOpOverrides._extract_dtype_and_bounds(a, b)

            # Create and return CSEVariable using CSE generation for caching
            return V.kernel.cse.generate(
                V.kernel.body, result_expr, bounds=bounds, dtype=dtype
            )

        return op_format.format(a=a, b=b)

    @staticmethod
    def _apply_unary_op(x: CuteDSLArg, op_format: str) -> CuteDSLArg:
        """
        Apply a unary operation, returning CSEVariable if input is a tensor.

        Args:
            x: Input operand (CSEVariable for tensors, str for scalars, or OpsValue wrapper)
            op_format: Format string with {x} placeholder for the operation

        Returns:
            CSEVariable if input is a tensor, otherwise string
        """
        cse_var = CuteDSLOpOverrides._get_cse_var(x)
        if cse_var is not None:
            result_expr = op_format.format(x=str(cse_var))
            return V.kernel.cse.generate(
                V.kernel.body, result_expr, bounds=cse_var.bounds, dtype=cse_var.dtype
            )

        return op_format.format(x=x)

    @staticmethod
    def constant(value: Union[bool, float, int], dtype: torch.dtype) -> str:
        """Generate CuteDSL constant representation."""
        if value == float("-inf"):
            return "float('-inf')"
        elif value == float("inf"):
            return "float('inf')"
        elif math.isnan(value):
            return "float('nan')"
        return repr(value)

    @staticmethod
    def add(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "({a} + {b})")

    @staticmethod
    def mul(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "({a} * {b})")

    @staticmethod
    def sub(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "({a} - {b})")

    @staticmethod
    def truediv(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "({a} / {b})")

    @staticmethod
    def floordiv(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "({a} // {b})")

    @staticmethod
    def mod(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "({a} % {b})")

    @staticmethod
    def remainder(a, b):
        return CuteDSLOpOverrides._apply_binary_op(a, b, "({a} % {b})")

    @staticmethod
    def exp(x: CuteDSLArg) -> CuteDSLArg:
        """Exponential using CuteDSL cute.math.exp function."""
        return CuteDSLOpOverrides._apply_unary_op(
            x, f"cute.math.exp2({{x}} * {CuteDSLOpOverrides.LOG2_E})"
        )

    @staticmethod
    def sqrt(x: CuteDSLArg) -> CuteDSLArg:
        """Square root using CuteDSL cute.math.sqrt function."""
        return CuteDSLOpOverrides._apply_unary_op(x, "cute.math.sqrt({x})")

    @staticmethod
    def log(x: CuteDSLArg) -> CuteDSLArg:
        """Natural logarithm using CuteDSL cute.math.log function."""
        return CuteDSLOpOverrides._apply_unary_op(x, "cute.math.log({x})")

    @staticmethod
    def cos(x: CuteDSLArg) -> CuteDSLArg:
        """Cosine using CuteDSL cute.math.cos function."""
        return CuteDSLOpOverrides._apply_unary_op(x, "cute.math.cos({x})")

    @staticmethod
    def sin(x: CuteDSLArg) -> CuteDSLArg:
        """Sine using CuteDSL cute.math.sin function."""
        return CuteDSLOpOverrides._apply_unary_op(x, "cute.math.sin({x})")

    @staticmethod
    def erf(x: CuteDSLArg) -> CuteDSLArg:
        """Error function using CuteDSL cute.math.erf function."""
        return CuteDSLOpOverrides._apply_unary_op(x, "cute.math.erf({x})")

    @staticmethod
    def maximum(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        raise NotImplementedError("TODO: maximum is not supported yet for TensorSSA")

    @staticmethod
    def minimum(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        raise NotImplementedError("TODO: minimum is not supported yet for TensorSSA")

    @staticmethod
    def where(
        condition: CuteDSLArg,
        a: CuteDSLArg,
        b: CuteDSLArg,
    ) -> CuteDSLArg:
        """Conditional selection - handles CSEVariable, OpsValue, and string inputs."""
        # Find a tensor argument to use as template for full_like
        # Priority: use 'a' if it's a tensor, else use 'b', else condition
        tensor_arg = (
            CuteDSLOpOverrides._get_cse_var(a)
            or CuteDSLOpOverrides._get_cse_var(b)
            or CuteDSLOpOverrides._get_cse_var(condition)
        )

        if tensor_arg is not None:
            a_ssa = CuteDSLOpOverrides._ensure_tensor_ssa(a, tensor_arg)
            b_ssa = CuteDSLOpOverrides._ensure_tensor_ssa(b, tensor_arg)
            cond_ssa = CuteDSLOpOverrides._ensure_tensor_ssa(condition, tensor_arg)
            result_expr = f"cute.where({cond_ssa}, {a_ssa}, {b_ssa})"

            dtype, bounds = CuteDSLOpOverrides._extract_dtype_and_bounds(
                a, b, condition
            )

            return V.kernel.cse.generate(
                V.kernel.body, result_expr, bounds=bounds, dtype=dtype
            )

        return f"cute.where({condition}, {a}, {b})"

    @staticmethod
    def pow(a: CuteDSLArg, b: CuteDSLArg):
        return CuteDSLOpOverrides._apply_binary_op(a, b, "({a} ** {b})")

    @staticmethod
    def abs(x: CuteDSLArg) -> CuteDSLArg:
        """Absolute value using CuteDSL cute.math.abs function."""
        if isinstance(x, CSEVariable):
            x_dtype = x.dtype
        elif isinstance(x, OpsValue) and isinstance(x.value, CSEVariable):
            x_dtype = x.value.dtype
        else:
            x_dtype = torch.float32

        abs_op = (
            "mlir_math.absf"
            if x_dtype in (torch.float16, torch.bfloat16, torch.float32)
            else "mlir_math.absi"
        )
        return CuteDSLOpOverrides._apply_unary_op(
            x,
            f"cute.TensorSSA({abs_op}({{x}}), {{x}}.shape, {{x}}.dtype)",
        )

    @staticmethod
    def neg(x: CuteDSLArg) -> CuteDSLArg:
        """Negation using CuteDSL TensorSSA __neg__ operator."""
        # TODO: See https://github.com/NVIDIA/cutlass/issues/2584
        return CuteDSLOpOverrides._apply_unary_op(
            x, "cute.TensorSSA(-{x}, {x}.shape, {x}.dtype)"
        )

    @staticmethod
    def to_dtype(
        x: CuteDSLArg, dtype: torch.dtype, src_dtype=None, use_compute_types=True
    ) -> CuteDSLArg:
        """Type conversion using CuteDSL TensorSSA.to(Type[Numeric]).

        Maps torch dtypes to cutlass.cute.typing numeric types and emits
        `{x}.to(cute.typing.<Type>)`.

        Raises NotImplementedError for unsigned integer and unsupported dtypes.
        """
        # Always convert up from bf16 and fp16 TODO on configuring
        dtype = upcast_compute_type(dtype)

        cute_type = CuteDSLOpOverrides.TORCH_TO_CUTE_DTYPE.get(dtype)
        if cute_type is None:
            raise NotImplementedError(
                f"CuteDSL dtype cast not implemented for torch dtype: {dtype}"
            )

        if isinstance(x, CSEVariable):
            result_expr = f"{str(x)}.to({cute_type})"
            return V.kernel.cse.generate(
                V.kernel.body, result_expr, bounds=x.bounds, dtype=dtype
            )

        return f"{x}.to({cute_type})"

    @staticmethod
    def tanh(x0: CuteDSLArg) -> CuteDSLArg:
        """Hyperbolic tangent using CuteDSL cute.math.tanh function."""
        return CuteDSLOpOverrides._apply_unary_op(x0, "cute.math.tanh({x})")

    # Logical operations
    @staticmethod
    def logical_and(x0: CuteDSLArg, x1: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(x0, x1, "({a} & {b})")

    @staticmethod
    def logical_or(x0: CuteDSLArg, x1: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(x0, x1, "({a} | {b})")

    # Bitwise operations (override parent class to properly CSE)
    @staticmethod
    # pyrefly: ignore [bad-override]
    def bitwise_and(x: CuteDSLArg, y: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(x, y, "({a} & {b})")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def bitwise_or(x: CuteDSLArg, y: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(x, y, "({a} | {b})")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def bitwise_xor(x: CuteDSLArg, y: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(x, y, "({a} ^ {b})")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def bitwise_not(x: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_unary_op(x, "(~{x})")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def bitwise_left_shift(x: CuteDSLArg, y: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(x, y, "({a} << {b})")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def bitwise_right_shift(x: CuteDSLArg, y: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(x, y, "({a} >> {b})")

    @staticmethod
    def logical_not(a):
        """Logical NOT."""
        return CuteDSLOpOverrides._apply_unary_op(a, "({x} == 0)")

    # Comparison operations
    @staticmethod
    def eq(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "operator.eq({a}, {b})")

    @staticmethod
    def ne(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "operator.ne({a}, {b})")

    @staticmethod
    def lt(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "operator.lt({a}, {b})")

    @staticmethod
    def le(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "operator.le({a}, {b})")

    @staticmethod
    def gt(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "operator.gt({a}, {b})")

    @staticmethod
    def ge(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "operator.ge({a}, {b})")
