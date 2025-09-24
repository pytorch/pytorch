"""Linear algebra SVD operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class SvdOperator(Operator):
    """Operator for torch.linalg.svd."""

    def __init__(self):
        super().__init__("linalg.svd")

    def can_produce(self, tensor):
        """linalg.svd returns tuple, so cannot produce single tensor output."""
        # SVD returns (U, S, Vh), so we cannot use it for single tensor output
        return False

    def decompose(self, tensor):
        """Decompose tensor - not used since can_produce returns False."""
        raise NotImplementedError("SVD returns multiple outputs, not supported in current framework")

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code - not used since can_produce returns False."""
        raise NotImplementedError("SVD returns multiple outputs, not supported in current framework")
