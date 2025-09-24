"""Linear algebra QR decomposition operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class QrOperator(Operator):
    """Operator for torch.linalg.qr."""

    def __init__(self):
        super().__init__("linalg.qr")

    def can_produce(self, tensor):
        """linalg.qr returns tuple, so cannot produce single tensor output."""
        # QR returns (Q, R), so we cannot use it for single tensor output
        return False

    def decompose(self, tensor):
        """Decompose tensor - not used since can_produce returns False."""
        raise NotImplementedError("QR returns multiple outputs, not supported in current framework")

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code - not used since can_produce returns False."""
        raise NotImplementedError("QR returns multiple outputs, not supported in current framework")
