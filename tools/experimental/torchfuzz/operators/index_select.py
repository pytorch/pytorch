import torch
from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import Spec, TensorSpec


class IndexSelectOperator(Operator):
    """Operator for selecting elements from a tensor along a dimension using indices."""

    def __init__(self):
        super().__init__("index_select")

    @property
    def torch_op_name(self) -> str | None:
        """Return the torch operation name."""
        return "torch.index_select"

    def can_produce(self, output_spec: Spec) -> bool:
        """Index select can produce tensors of various shapes, but not 0-dimensional tensors."""
        if not isinstance(output_spec, TensorSpec):
            return False
        # index_select requires at least one dimension to select from
        return len(output_spec.size) > 0

    def fuzz_inputs_specs(self, output_spec: Spec, num_inputs: int = 3) -> list[Spec]:
        """Generate input specs for index_select operation.

        torch.index_select(input, dim, index) returns a tensor with:
        - output.shape[dim] = len(index)
        - output.shape[other_dims] = input.shape[other_dims]
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("IndexSelectOperator can only produce TensorSpec outputs")

        # For simplicity, we'll work with a 2D input tensor
        # and select along dimension 0
        dim = 0
        output_size = output_spec.size

        # Input tensor - create a shape where we can select from it
        # If output is (k, m), input can be (n, m) where n >= k
        if len(output_size) == 1:
            # Output is 1D, input should be at least 1D
            input_size = (output_size[0] + 2,)  # Make input larger
            input_stride = (1,)
        elif len(output_size) == 2:
            # Output is 2D, input should be 2D with first dim >= output first dim
            input_size = (output_size[0] + 2, output_size[1])
            input_stride = (output_size[1], 1)  # Contiguous
        else:
            # For higher dimensions, keep it simple
            input_size = tuple(
                s + 2 if i == dim else s for i, s in enumerate(output_size)
            )
            # Contiguous stride
            input_stride = tuple(
                int(torch.tensor(input_size[i + 1 :]).prod().item())
                if i < len(input_size) - 1
                else 1
                for i in range(len(input_size))
            )

        input_tensor_spec = TensorSpec(
            size=input_size,
            stride=input_stride,
            dtype=output_spec.dtype,
        )

        # Index tensor - 1D tensor of long dtype with indices
        index_spec = TensorSpec(
            size=(output_size[dim],) if len(output_size) > 0 else (1,),
            stride=(1,),
            dtype=torch.long,
        )

        return [input_tensor_spec, index_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for index_select.

        Creates appropriate indices to select from the input tensor.
        """
        if len(input_names) != 2:
            raise ValueError("IndexSelectOperator requires exactly two inputs")
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("IndexSelectOperator requires TensorSpec output")

        # Determine dimension and number of indices needed
        dim = 0  # Select along dimension 0 for simplicity
        num_indices = output_spec.size[dim] if len(output_spec.size) > 0 else 1

        # Generate code that creates valid indices within the input tensor's dimension
        return (
            f"_input_size_{output_name} = {input_names[0]}.size({dim})\n"
            f"_index_{output_name} = torch.randint(0, _input_size_{output_name}, ({num_indices},), device={input_names[0]}.device)\n"
            f"{output_name} = torch.index_select({input_names[0]}, {dim}, _index_{output_name})"
        )
