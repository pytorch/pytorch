"""Linear algebra eigendecomposition operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class EigOperator(Operator):
    """Operator for torch.linalg.eig."""

    def __init__(self):
        super().__init__("linalg.eig")

    def can_produce(self, tensor):
        """linalg.eig returns tuple, so cannot produce single tensor output."""
        # eig returns (eigenvalues, eigenvectors), so we cannot use it for single tensor output
        return False

    def decompose(self, tensor):
        """Decompose tensor - not used since can_produce returns False."""
        raise NotImplementedError("eig returns multiple outputs, not supported in current framework")

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code - not used since can_produce returns False."""
        raise NotImplementedError("eig returns multiple outputs, not supported in current framework")
