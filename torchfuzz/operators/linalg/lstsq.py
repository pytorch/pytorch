"""Linear algebra least squares operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class LstsqOperator(Operator):
    """Operator for torch.linalg.lstsq."""

    def __init__(self):
        super().__init__("linalg.lstsq")

    def can_produce(self, tensor):
        """linalg.lstsq returns tuple, so cannot produce single tensor output."""
        # lstsq returns (solution, residuals, rank, singular_values), so we cannot use it for single tensor output
        return False

    def decompose(self, tensor):
        """Decompose tensor - not used since can_produce returns False."""
        raise NotImplementedError("lstsq returns multiple outputs, not supported in current framework")

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code - not used since can_produce returns False."""
        raise NotImplementedError("lstsq returns multiple outputs, not supported in current framework")
