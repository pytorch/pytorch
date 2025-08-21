# mypy: allow-untyped-defs
"""
CuteDSL-specific operation overrides for pointwise operations.

This module provides CuteDSL implementations of common operations used in
template kernels, particularly for flex attention modifications.
"""

import math
from typing import Union

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
    def _apply_binary_op(a: CuteDSLArg, b: CuteDSLArg, op_format: str) -> CuteDSLArg:
        """
        Apply a binary operation with automatic scalar-to-tensor conversion.

        CuteDSL requires both operands to be TensorSSA objects for tensor operations.
        This helper automatically converts scalar arguments to TensorSSA using
        cute.full_like when at least one argument is a tensor (CSEVariable).

        Args:
            a: First operand (CSEVariable for tensors, str for scalars)
            b: Second operand (CSEVariable for tensors, str for scalars)
            op_format: Format string with {a} and {b} placeholders for the operation

        Returns:
            CSEVariable if at least one operand is a CSEVariable, otherwise string
        """
        tensor_arg = (
            a
            if isinstance(a, CSEVariable)
            else b
            if isinstance(b, CSEVariable)
            else None
        )
        if tensor_arg is not None:
            a_ssa = CuteDSLOpOverrides._ensure_tensor_ssa(a, tensor_arg)
            b_ssa = CuteDSLOpOverrides._ensure_tensor_ssa(b, tensor_arg)
            result_expr = op_format.format(a=a_ssa, b=b_ssa)

            # Extract dtype and bounds from CSEVariable inputs
            dtype = None
            bounds = ValueRanges.unknown()

            if isinstance(a, CSEVariable):
                dtype = a.dtype
                bounds = a.bounds
            elif isinstance(b, CSEVariable):
                dtype = b.dtype
                bounds = b.bounds

            # Create and return CSEVariable using CSE generation for caching
            return V.kernel.cse.generate(
                V.kernel.body, result_expr, bounds=bounds, dtype=dtype
            )

        return op_format.format(a=a, b=b)

    @staticmethod
    def _apply_unary_op(x: CuteDSLArg, op_format: str) -> CuteDSLArg:
        """
        Apply a unary operation, returning CSEVariable if input is CSEVariable.

        Args:
            x: Input operand (CSEVariable for tensors, str for scalars)
            op_format: Format string with {x} placeholder for the operation

        Returns:
            CSEVariable if input is a CSEVariable, otherwise string
        """
        if isinstance(x, CSEVariable):
            result_expr = op_format.format(x=str(x))
            return V.kernel.cse.generate(
                V.kernel.body, result_expr, bounds=x.bounds, dtype=x.dtype
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
    def mod(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "({a} % {b})")

    @staticmethod
    def remainder(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "({a} % {b})")

    @staticmethod
    def exp(x: CuteDSLArg) -> CuteDSLArg:
        """Exponential using CuteDSL cute.math.exp function."""
        return CuteDSLOpOverrides._apply_unary_op(x, "cute.math.exp2({x} * 1.4426950408889634)")

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
        return CuteDSLOpOverrides._apply_binary_op(a, b, "cute.math.maximum({a}, {b})")

    @staticmethod
    def minimum(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "cute.math.minimum({a}, {b})")

    @staticmethod
    def where(
        condition: CuteDSLArg,
        a: CuteDSLArg,
        b: CuteDSLArg,
    ) -> CuteDSLArg:
        """Conditional selection - handles both CSEVariable and string inputs."""
        # Find a tensor argument to use as template for full_like
        # Priority: use 'a' if it's a tensor, else use 'b', else condition
        tensor_arg = (
            a
            if isinstance(a, CSEVariable)
            else (
                b
                if isinstance(b, CSEVariable)
                else condition
                if isinstance(condition, CSEVariable)
                else None
            )
        )

        if tensor_arg is not None:
            a_ssa = CuteDSLOpOverrides._ensure_tensor_ssa(a, tensor_arg)
            b_ssa = CuteDSLOpOverrides._ensure_tensor_ssa(b, tensor_arg)
            # result_expr = f"cute.where(cute.Boolean({condition}), {a_ssa}, {b_ssa})"
            result_expr = f"cute.where({condition}, {a_ssa}, {b_ssa})"

            # Extract dtype and bounds from CSEVariable inputs
            dtype = None
            bounds = ValueRanges.unknown()

            if isinstance(a, CSEVariable):
                dtype = a.dtype
                bounds = a.bounds
            elif isinstance(b, CSEVariable):
                dtype = b.dtype
                bounds = b.bounds
            elif isinstance(condition, CSEVariable):
                dtype = condition.dtype
                bounds = condition.bounds

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
        abs_op = (
            "mlir_math.absf"
            if x.dtype in (torch.float16, torch.bfloat16, torch.float32)
            else "mlir_math.absi"
        )
        return CuteDSLOpOverrides._apply_unary_op(
            x, f"cute.TensorSSA({abs_op}({x}), {x}.shape, {x}.dtype)"
        )

    @staticmethod
    def neg(x: CuteDSLArg) -> CuteDSLArg:
        """Negation using CuteDSL TensorSSA __neg__ operator."""
        return CuteDSLOpOverrides._apply_unary_op(x, "(-{x})")

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
        # Map torch dtypes to CuteDSL type strings
        torch2cute_dtype_map = {
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

        cute_type = torch2cute_dtype_map.get(dtype)
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
    def sigmoid(x):
        """Sigmoid activation function."""
        # Could use cute.math.sigmoid if available, or implement as 1/(1+exp(-x))
        return CuteDSLOpOverrides._apply_unary_op(x, "cute.math.sigmoid({x})")

    @staticmethod
    def relu(x):
        """ReLU activation function."""
        return CuteDSLOpOverrides.maximum(x, "0.0")

    def tanh(self, x0: CuteDSLArg) -> CuteDSLArg:
        """Hyperbolic tangent using CuteDSL cute.math.tanh function."""
        return CuteDSLOpOverrides._apply_unary_op(x0, "cute.math.tanh({x})")

    # Logical operations
    def logical_and(self, x0: CuteDSLArg, x1: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(x0, x1, "({a} & {b})")

    def logical_or(self, x0: CuteDSLArg, x1: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(x0, x1, "({a} | {b})")

    @staticmethod
    def logical_not(a):
        """Logical NOT."""
        return CuteDSLOpOverrides._apply_unary_op(a, "({x} == 0)")

    # Comparison operations
    @staticmethod
    def eq(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        # TODO: Why is the dsl behvaing  inconsistently...
        return CuteDSLOpOverrides._apply_binary_op(a, b, "({a} == {b})")

    @staticmethod
    def ne(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "cute.Boolean(({a} != {b}))")

    @staticmethod
    def lt(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "cute.Boolean(({a} < {b}))")

    @staticmethod
    def le(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "cute.Boolean(({a} <= {b}))")

    @staticmethod
    def gt(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "cute.Boolean(({a} > {b}))")

    @staticmethod
    def ge(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "cute.Boolean(({a} >= {b}))")
