"""Nonzero operator implementation."""

import torch
from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import Spec, TensorSpec


class NonzeroOperator(Operator):
    """Operator for finding nonzero elements in a tensor."""

    def __init__(self):
        super().__init__("nonzero")

    @property
    def torch_op_name(self) -> str | None:
        """Return the torch operation name."""
        return "torch.nonzero"

    def can_produce(self, output_spec: Spec) -> bool:
        """Nonzero produces a tensor with shape (n_nonzero, n_dims).

        We can deterministically synthesize inputs to match any 2D int64 output
        shape (k, d) without data-dependent guards by constructing an input with
        exactly k non-zero elements and d dimensions.
        """
        return (
            isinstance(output_spec, TensorSpec)
            and output_spec.dtype in [torch.int64, torch.long]
            and len(output_spec.size) == 2
        )

    def fuzz_inputs_specs(self, output_spec: Spec, num_inputs: int = 1) -> list[Spec]:
        """Generate input spec for nonzero operation.

        The actual values will be synthesized in codegen to achieve the target size.
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("NonzeroOperator can only produce TensorSpec outputs")

        # Provide a placeholder spec; codegen will ignore the actual input content
        # and synthesize a tensor with desired nonzero count and dimensionality.
        d = output_spec.size[1]
        input_spec = TensorSpec(
            size=tuple([1] * d) if d > 0 else (),
            stride=tuple([1] * d) if d > 0 else (),
            dtype=torch.bool,
        )
        return [input_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for nonzero using synthesized input to match target size.

        No data-dependent conditionals/guards. Constructs an input with exactly
        k = output_spec.size[0] non-zero elements and d = output_spec.size[1] dims,
        then calls torch.nonzero on it.
        """
        if len(input_names) != 1:
            raise ValueError("NonzeroOperator requires exactly one input")
        if not isinstance(output_spec, TensorSpec) or len(output_spec.size) != 2:
            raise ValueError("NonzeroOperator requires 2D TensorSpec output")
        k = output_spec.size[0]
        d = output_spec.size[1]
        # Construct concrete shape literal like (k, 1, 1, ...)
        shape_elems = [str(k)] + ["1"] * max(0, d - 1)
        shape_literal = (
            "(" + ", ".join(shape_elems) + ("," if d == 1 else "") + ")"
            if d > 0
            else "()"
        )
        return (
            f"_x_nz = torch.zeros({shape_literal}, dtype=torch.bool, device={input_names[0]}.device)\n"
            f"_x_nz_flat = _x_nz.reshape(-1)\n"
            f"_x_nz_flat[:{k}] = True\n"
            f"{output_name} = torch.nonzero(_x_nz)"
        )
