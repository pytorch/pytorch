"""SiLU operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class SiluOperator(Operator):
    """Operator for SiLU activation function (torch.nn.functional.silu)."""

    def __init__(self):
        super().__init__("silu")

    def can_produce(self, tensor):
        """SiLU can be applied elementwise to floating tensors."""
        # Prefer common floating types
        return str(tensor.dtype).lower() in {"float32", "float64", "bfloat16", "float16"}

    def decompose(self, tensor):
        """Decompose tensor into input tensor for SiLU."""
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for SiLU operation."""
        return f"{output_name} = torch.nn.functional.silu({input_names[0]})"
