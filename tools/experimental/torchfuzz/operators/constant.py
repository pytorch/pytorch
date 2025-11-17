"""Constant operator implementation."""

from typing import Optional

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
        self.template = "default"  # Track template for DTensor compatibility

    @property
    def torch_op_name(self) -> Optional[str]:
        """Constant is not a torch operation, it generates constant values."""
        return None

    def set_template(self, template: str):
        """Set the template for context-aware code generation."""
        self.template = template

    def can_produce(self, output_spec: Spec) -> bool:
        """Constant can produce any type of output."""
        return True

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Constant requires no inputs for fuzzing."""
        return []

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for constant creation."""
        # Create constant by calling fuzzing functions during codegen with deterministic seed
        # Use a deterministic hash based on the variable name to ensure reproducibility across processes
        import hashlib

        var_seed = int(hashlib.md5(output_name.encode()).hexdigest()[:8], 16) % (2**31)  # noqa: S324

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
                    torch.float16: 1.0,
                    torch.float32: 1.0,
                    torch.float64: 1.0,
                    torch.bfloat16: 1.0,
                    torch.int8: 1,
                    torch.int16: 1,
                    torch.int32: 1,
                    torch.int64: 1,
                    torch.bool: True,
                    torch.complex64: 1.0,
                    torch.complex128: 1.0,
                }

                fill_value = default_values.get(output_spec.dtype, 1)
                tensor_creation = (
                    f"torch.full({size_str}, {fill_value}, dtype={dtype_str})"
                )
            else:
                # For non-empty tensors, use the first element as fill value
                fill_value = actual_tensor.flatten()[0].item()

                # For integer types, clamp the value to a smaller range to avoid
                # issues when used in arithmetic with embedding indices
                import torch

                if output_spec.dtype in [
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                ]:
                    # Clamp integer values to [0, 3] to avoid index overflow in multiplication
                    # Even with multiplication, indices should stay in reasonable range
                    fill_value = max(0, min(3, abs(fill_value)))

                tensor_creation = (
                    f"torch.full({size_str}, {fill_value}, dtype={dtype_str})"
                )

            # For DTensor template, convert to DTensor
            if self.template == "dtensor":
                return (
                    f"{output_name}_local = {tensor_creation}.to('cuda')\n"
                    f"    {output_name} = DTensor.from_local({output_name}_local, mesh, placements)"
                )
            else:
                return f"{output_name} = {tensor_creation}"

        else:
            return f"# Unknown output spec type for constant: {type(output_spec)}"
