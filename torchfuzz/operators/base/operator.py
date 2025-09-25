"""Base operator class for all torchfuzz operators."""

class Operator:
    """Base class for all operators that can be applied to tensors."""

    def __init__(self, name=None, supports_dtensor=False):
        """Initialize operator with name and DTensor support flag."""
        self.name = name
        self.supports_dtensor = supports_dtensor

    def can_produce(self, output_tensor, use_dtensor=False):
        """
        Check if this operator can produce the given output tensor.

        Args:
            output_tensor: The target output tensor to check
            use_dtensor: Whether DTensor mode is enabled

        Returns:
            bool: True if this operator can produce the output tensor
        """
        # If DTensor mode is enabled but this operator doesn't support DTensor, return False
        if use_dtensor and not self.supports_dtensor:
            return False

        return self._can_produce_impl(output_tensor)

    def _can_produce_impl(self, output_tensor):
        """
        Implementation-specific logic for checking if operator can produce output.
        Subclasses should override this instead of can_produce.
        """
        raise NotImplementedError("Subclasses must implement _can_produce_impl")

    def decompose(self, tensor):
        """Decompose the tensor into input tensors for this operator."""
        raise NotImplementedError

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for this operator."""
        raise NotImplementedError

    def supports_variable_inputs(self) -> bool:
        """Return True if this operator supports a variable number of inputs."""
        return False
