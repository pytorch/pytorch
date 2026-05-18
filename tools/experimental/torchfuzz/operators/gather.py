import torch
from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import Spec, TensorSpec


class GatherOperator(Operator):
    """Operator for gathering values along an axis specified by dim using indices."""

    def __init__(self):
        super().__init__("gather")

    @property
    def torch_op_name(self) -> str | None:
        """Return the torch operation name."""
        return "torch.gather"

    def can_produce(self, output_spec: Spec) -> bool:
        """Gather can produce tensors of various shapes, but not 0-dimensional tensors."""
        if not isinstance(output_spec, TensorSpec):
            return False
        # gather requires at least one dimension
        return len(output_spec.size) > 0

    def fuzz_inputs_specs(self, output_spec: Spec, num_inputs: int = 2) -> list[Spec]:
        """Generate input specs for gather operation.

        torch.gather(input, dim, index) returns a tensor with:
        - output.shape == index.shape
        - output[i][j][k] = input[i][j][index[i][j][k]] (for dim=2 example)
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("GatherOperator can only produce TensorSpec outputs")

        # The output shape matches the index shape
        output_size = output_spec.size
        dim = 0  # Gather along dimension 0 for simplicity

        # Input tensor - create a shape that matches output except for the gather dimension
        # which can be any size >= max(indices) + 1
        # For simplicity, make input larger in the gather dimension
        if len(output_size) == 1:
            # Output is 1D
            input_size = (output_size[0] + 2,)
            input_stride = (1,)
        elif len(output_size) == 2:
            # Output is 2D, make input 2D with first dim larger
            input_size = (output_size[0] + 2, output_size[1])
            input_stride = (output_size[1], 1)  # Contiguous
        else:
            # For higher dimensions
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

        # Index tensor - same shape as output, long dtype
        index_spec = TensorSpec(
            size=output_size,
            stride=tuple(
                int(torch.tensor(output_size[i + 1 :]).prod().item())
                if i < len(output_size) - 1
                else 1
                for i in range(len(output_size))
            ),
            dtype=torch.long,
        )

        return [input_tensor_spec, index_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for gather.

        Creates appropriate indices to gather from the input tensor.
        """
        if len(input_names) != 2:
            raise ValueError("GatherOperator requires exactly two inputs")
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("GatherOperator requires TensorSpec output")

        # Determine dimension
        dim = 0  # Gather along dimension 0 for simplicity

        # Generate code that creates valid indices within the input tensor's dimension
        return (
            f"_input_size_{output_name} = {input_names[0]}.size({dim})\n"
            f"_index_{output_name} = torch.randint(0, _input_size_{output_name}, {output_spec.size}, device={input_names[0]}.device)\n"
            f"{output_name} = torch.gather({input_names[0]}, {dim}, _index_{output_name})"
        )
