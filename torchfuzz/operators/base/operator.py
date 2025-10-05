"""Base operator class for all torchfuzz operators."""

class Operator:
    """Base class for all operators in torchfuzz."""

    def __init__(self, name):
        self.name = name

    def can_produce(self, tensor):
        """Check if this operator can produce the given tensor."""
        raise NotImplementedError

    def decompose(self, tensor):
        """Decompose the tensor into input tensors for this operator."""
        raise NotImplementedError

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for this operator."""
        raise NotImplementedError

    def supports_variable_inputs(self) -> bool:
        """Return True if this operator supports a variable number of inputs."""
        return False
