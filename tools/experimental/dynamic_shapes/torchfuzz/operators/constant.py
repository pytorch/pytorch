"""Constant operator implementation."""

from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import (
    fuzz_scalar,
    fuzz_tensor_simple,
    ScalarSpec,
    Spec,
    TensorSpec,
)


class ConstantOperator(Operator):
    """Operator for generating constants."""

    def __init__(self):
        super().__init__("constant")

    def can_produce(self, output_spec: Spec) -> bool:
        """Constant can produce any type of output."""
        return True

    def supports_variable_inputs(self) -> bool:
        """Constant operator does not require inputs."""
        return False

    def decompose(self, output_spec: Spec, num_inputs: int = 0) -> list[Spec]:
        """Constant requires no inputs."""
        return []

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for constant creation."""
        # Create constant by calling fuzzing functions during codegen with deterministic seed
        # Use a deterministic seed based on the variable name to ensure reproducibility
        var_seed = hash(output_name) % (2**31)

        if isinstance(output_spec, ScalarSpec):
            # Call fuzz_scalar during codegen and embed the result
            actual_value = fuzz_scalar(output_spec, seed=var_seed)

            # Format the value for embedding in code
            if isinstance(actual_value, bool):
                value_str = str(actual_value)
            elif isinstance(actual_value, (int, float)):
                value_str = repr(actual_value)
            elif isinstance(actual_value, complex):
                value_str = f"complex({actual_value.real}, {actual_value.imag})"
            else:
                value_str = repr(actual_value)

            return f"{output_name} = {value_str}"

        elif isinstance(output_spec, TensorSpec):
            # Call fuzz_tensor_simple during codegen and embed the result
            actual_tensor = fuzz_tensor_simple(
                output_spec.size, output_spec.stride, output_spec.dtype, seed=var_seed
            )

            # Convert tensor to code representation
            size_str = str(output_spec.size)
            dtype_str = f"torch.{output_spec.dtype}".replace("torch.torch.", "torch.")

            # Handle empty tensors (with 0 elements)
            if actual_tensor.numel() == 0:
                # For empty tensors, use a default fill value based on dtype
                import torch

                default_values = {
                    torch.float16: 0.0,
                    torch.float32: 0.0,
                    torch.float64: 0.0,
                    torch.bfloat16: 0.0,
                    torch.int8: 0,
                    torch.int16: 0,
                    torch.int32: 0,
                    torch.int64: 0,
                    torch.bool: False,
                    torch.complex64: 0.0,
                    torch.complex128: 0.0,
                }
                fill_value = default_values.get(output_spec.dtype, 0)
                return f"{output_name} = torch.full({size_str}, {fill_value}, dtype={dtype_str})"
            else:
                # For non-empty tensors, use the first element as fill value
                fill_value = actual_tensor.flatten()[0].item()
                return f"{output_name} = torch.full({size_str}, {fill_value}, dtype={dtype_str})"

        else:
            return f"# Unknown output spec type for constant: {type(output_spec)}"
