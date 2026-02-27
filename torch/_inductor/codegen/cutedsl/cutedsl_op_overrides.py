# mypy: allow-untyped-defs
"""
CuteDSL-specific operation overrides for pointwise operations.

This module provides CuteDSL implementations of common operations used in
template kernels, particularly for flex attention modifications.
"""

import math

import sympy

import torch
from torch._inductor.codegen.common import CSEVariable, OpOverrides
from torch._inductor.virtualized import OpsValue, V
from torch.utils._sympy.value_ranges import ValueRanges


CuteDSLArg = CSEVariable | str | bool | float | int


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
        torch.uint8: "cutlass.Uint8",
        torch.bool: "cutlass.Boolean",
        torch.float8_e4m3fn: "cutlass.Float8E4M3FN",
        torch.float8_e5m2: "cutlass.Float8E5M2",
    }

    # Math constants
    LOG2_E = 1.4426950408889634  # 1/ln(2) for converting natural exp to base-2 exp

    @staticmethod
    def _get_cse_var(arg: CuteDSLArg) -> CSEVariable | None:
        """Extract CSEVariable from arg if it's a tensor (either direct or wrapped in OpsValue)."""
        if isinstance(arg, CSEVariable):
            return arg
        if isinstance(arg, OpsValue) and isinstance(arg.value, CSEVariable):
            return arg.value
        return None

    @staticmethod
    def _as_expr(arg: CuteDSLArg) -> str:
        cse_var = CuteDSLOpOverrides._get_cse_var(arg)
        if cse_var is not None:
            return str(cse_var)
        return str(arg)

    @staticmethod
    def _node_tensor_flags() -> tuple[bool, bool] | None:
        node = V.current_node
        if node is None or len(node.args) < 2:
            return None

        def _is_tensor(raw_arg: object) -> bool:
            if hasattr(raw_arg, "meta"):
                val = raw_arg.meta.get("val")
                return isinstance(val, torch.Tensor) and val.ndim > 0
            return False

        return _is_tensor(node.args[0]), _is_tensor(node.args[1])

    @staticmethod
    def _ensure_tensor_ssa(
        arg: CuteDSLArg, template_tensor: CuteDSLArg, *, is_tensor: bool
    ) -> str:
        """
        Convert scalar arguments to TensorSSA using cute.full_like if needed.

        Args:
            arg: The argument to check (CSEVariable for tensors, str for scalars, or OpsValue wrapper)
            template_tensor: A tensor argument to use as template for full_like

        Returns:
            String representation suitable for CuteDSL operations
        """
        arg_expr = CuteDSLOpOverrides._as_expr(arg)
        if is_tensor:
            return arg_expr
        template_expr = CuteDSLOpOverrides._as_expr(template_tensor)
        return f"cute.full_like({template_expr}, {arg_expr})"

    @staticmethod
    def _extract_dtype_and_bounds(
        *args: CuteDSLArg,
    ) -> tuple[torch.dtype | None, ValueRanges[sympy.Expr]]:
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
        a_cse = CuteDSLOpOverrides._get_cse_var(a)
        b_cse = CuteDSLOpOverrides._get_cse_var(b)

        node_flags = CuteDSLOpOverrides._node_tensor_flags()
        if node_flags is not None:
            a_is_tensor, b_is_tensor = node_flags
        else:
            a_is_tensor = a_cse is not None
            b_is_tensor = b_cse is not None

        tensor_arg = a if a_is_tensor else (b if b_is_tensor else None)
        if tensor_arg is None:
            tensor_arg = a_cse or b_cse

        if tensor_arg is not None:
            if a_cse is None and b_cse is None:
                return op_format.format(
                    a=CuteDSLOpOverrides._as_expr(a),
                    b=CuteDSLOpOverrides._as_expr(b),
                )

            a_ssa = CuteDSLOpOverrides._ensure_tensor_ssa(
                a, tensor_arg, is_tensor=a_is_tensor
            )
            b_ssa = CuteDSLOpOverrides._ensure_tensor_ssa(
                b, tensor_arg, is_tensor=b_is_tensor
            )
            result_expr = op_format.format(a=a_ssa, b=b_ssa)

            dtype, bounds = CuteDSLOpOverrides._extract_dtype_and_bounds(a, b)
            expected = CuteDSLOpOverrides._expected_tensor_val()
            if dtype is None:
                dtype = expected.dtype if expected is not None else torch.int32
            if a_cse is not None:
                shape = a_cse.shape
            elif b_cse is not None:
                shape = b_cse.shape
            else:
                shape = tuple(expected.size()) if expected is not None else None

            # Create and return CSEVariable using CSE generation for caching
            return V.kernel.cse.generate(
                V.kernel.body, result_expr, bounds=bounds, dtype=dtype, shape=shape
            )

        return op_format.format(a=a, b=b)

    @staticmethod
    def _expected_tensor_val() -> torch.Tensor | None:
        """Return the fake-tensor value from the current FX node's metadata, if any."""
        node = V.current_node
        if node is None:
            return None
        val = node.meta.get("val")
        return val if isinstance(val, torch.Tensor) else None

    @staticmethod
    def _cast_expr(expr: str, dtype: torch.dtype) -> str:
        cute_type = CuteDSLOpOverrides.TORCH_TO_CUTE_DTYPE.get(dtype)
        if cute_type is None:
            return expr
        return f"{cute_type}({expr})"

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
    def constant(value: bool | float | int, dtype: torch.dtype) -> str:
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
        """Exponential using CuteDSL cute.math.exp2 with log2(e) scaling."""
        if CuteDSLOpOverrides._get_cse_var(x) is None:
            x = CuteDSLOpOverrides._cast_expr(str(x), torch.float32)
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
    # pyrefly: ignore [bad-override]
    def sigmoid(x: CuteDSLArg) -> CuteDSLArg:
        """Sigmoid with fp32 compute and cast-back to expected output dtype."""
        x_cse = CuteDSLOpOverrides._get_cse_var(x)
        if x_cse is not None:
            x_fp32: CuteDSLArg = CuteDSLOpOverrides.to_dtype(
                x_cse,
                torch.float32,
                use_compute_types=False,
            )
        else:
            x_fp32 = CuteDSLOpOverrides._cast_expr(str(x), torch.float32)

        result = CuteDSLOpOverrides._apply_unary_op(
            x_fp32,
            f"(1.0 / (1.0 + cute.math.exp2(-{{x}} * {CuteDSLOpOverrides.LOG2_E})))",
        )

        expected = CuteDSLOpOverrides._expected_tensor_val()
        expected_dtype = expected.dtype if expected is not None else None
        if expected_dtype is not None and expected_dtype != torch.float32:
            result_cse = CuteDSLOpOverrides._get_cse_var(result)
            if result_cse is not None:
                return CuteDSLOpOverrides.to_dtype(
                    result_cse,
                    expected_dtype,
                    use_compute_types=False,
                )
            return CuteDSLOpOverrides._cast_expr(str(result), expected_dtype)
        return result

    @staticmethod
    def _minmax(a: CuteDSLArg, b: CuteDSLArg, *, op: str) -> CuteDSLArg:
        tensor_arg = CuteDSLOpOverrides._get_cse_var(
            a
        ) or CuteDSLOpOverrides._get_cse_var(b)
        if tensor_arg is not None:
            return CuteDSLOpOverrides._apply_binary_op(
                a, b, f"cute.where(({{a}}) {op} ({{b}}), {{a}}, {{b}})"
            )

        lhs = str(a)
        rhs = str(b)
        expected = CuteDSLOpOverrides._expected_tensor_val()
        if expected is not None:
            lhs = CuteDSLOpOverrides._cast_expr(lhs, expected.dtype)
            rhs = CuteDSLOpOverrides._cast_expr(rhs, expected.dtype)
        return f"({lhs} if {lhs} {op} {rhs} else {rhs})"

    @staticmethod
    def maximum(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._minmax(a, b, op=">")

    @staticmethod
    def minimum(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._minmax(a, b, op="<")

    @staticmethod
    def where(
        condition: CuteDSLArg,
        a: CuteDSLArg,
        b: CuteDSLArg,
    ) -> CuteDSLArg:
        """Conditional selection - handles CSEVariable, OpsValue, and string inputs."""
        a_cse = CuteDSLOpOverrides._get_cse_var(a)
        b_cse = CuteDSLOpOverrides._get_cse_var(b)
        cond_cse = CuteDSLOpOverrides._get_cse_var(condition)
        tensor_arg = a_cse or b_cse or cond_cse

        if tensor_arg is not None:
            a_ssa = CuteDSLOpOverrides._ensure_tensor_ssa(
                a, tensor_arg, is_tensor=a_cse is not None
            )
            b_ssa = CuteDSLOpOverrides._ensure_tensor_ssa(
                b, tensor_arg, is_tensor=b_cse is not None
            )
            cond_ssa = CuteDSLOpOverrides._ensure_tensor_ssa(
                condition, tensor_arg, is_tensor=cond_cse is not None
            )
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
        """Negation for both TensorSSA and scalar-like expressions."""
        # TensorSSA path: avoid relying on __neg__ directly due upstream issue.
        if CuteDSLOpOverrides._get_cse_var(x) is not None:
            return CuteDSLOpOverrides._apply_unary_op(
                x, "cute.TensorSSA(-{x}, {x}.shape, {x}.dtype)"
            )
        # Scalar path: shape/dtype attributes are unavailable.
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
        if use_compute_types:
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
