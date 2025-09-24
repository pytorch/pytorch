"""Linear algebra Hermitian eigendecomposition operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class EighOperator(Operator):
    """Operator for torch.linalg.eigh."""

    def __init__(self):
        super().__init__("linalg.eigh")

    def can_produce(self, tensor):
        """linalg.eigh returns tuple, so cannot produce single tensor output."""
        # eigh returns (eigenvalues, eigenvectors), so we cannot use it for single tensor output
        return False

    def decompose(self, tensor):
        """Decompose tensor - not used since can_produce returns False."""
        raise NotImplementedError("eigh returns multiple outputs, not supported in current framework")

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code - not used since can_produce returns False."""
        raise NotImplementedError("eigh returns multiple outputs, not supported in current framework")
