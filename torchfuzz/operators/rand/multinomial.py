"""Multinomial distribution operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class MultinomialOperator(Operator):
    """Operator for torch.multinomial."""

    def __init__(self):
        super().__init__("multinomial")

    def can_produce(self, tensor):
        """torch.multinomial can produce integer tensors from probability inputs."""
        return tensor.dtype in ["int32", "int64"]

    def decompose(self, tensor):
        """torch.multinomial needs an input probability tensor."""
        # Input should have one more dimension than output (the categories dimension)
        if len(tensor.size) == 1:
            # Output is 1D (n_samples,), input is 1D (n_categories,)
            input_size = (10,)  # Arbitrary number of categories
        else:
            # Output is (..., n_samples), input is (..., n_categories)
            input_size = tensor.size[:-1] + (10,)

        def calc_stride(size):
            stride = [1]
            for dim in reversed(size[:-1]):
                stride.insert(0, stride[0] * dim)
            return tuple(stride)

        input_stride = calc_stride(input_size)
        input_tensor = Tensor(input_size, input_stride, "float32", tensor.device, tensor.supported_ops)
        return [input_tensor]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for torch.multinomial operation."""
        if len(input_names) != 1:
            raise ValueError("torch.multinomial requires exactly 1 input")

        num_samples = output_tensor.size[-1] if len(output_tensor.size) > 0 else 1
        return f"{output_name} = torch.multinomial({input_names[0]}, {num_samples})"
