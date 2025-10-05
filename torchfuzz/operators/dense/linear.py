"""Linear transformation operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class LinearOperator(Operator):
    """Operator for linear transformation (torch.nn.functional.linear)."""

    def __init__(self):
        super().__init__("linear")

    def can_produce(self, tensor):
        """Linear can produce tensors that are at least 1D and floating point."""
        # Linear only supports floating point tensors
        if tensor.dtype in ["int8", "int16", "int32", "int64", "uint8", "bool"]:
            return False
        return len(tensor.size) >= 1

    def decompose(self, tensor):
        """Decompose tensor into input tensors for linear transformation."""
        # tensor shape is (..., out_features)
        # input shape is (..., in_features)
        # weight shape is (out_features, in_features)
        # bias shape is (out_features,)

        *batch_dims, out_features = tensor.size

        # Choose a random input feature size (different from out_features to avoid trivial cases)
        possible_in_features = [64, 128, 256, 512, 768]
        # Remove the out_features from the list to ensure they're different
        possible_in_features = [f for f in possible_in_features if f != out_features]
        # If all options are removed (edge case), use a default
        if not possible_in_features:
            possible_in_features = [out_features + 64]  # Just make it different
        in_features = random.choice(possible_in_features)

        # Input tensor shape: (..., in_features)
        input_size = tuple(batch_dims + [in_features])

        # Weight tensor shape: (out_features, in_features)
        weight_size = (out_features, in_features)

        # Calculate strides for contiguous tensors
        def calc_stride(size):
            stride = [1]
            for dim in reversed(size[:-1]):
                stride.insert(0, stride[0] * dim)
            return tuple(stride)

        input_stride = calc_stride(input_size)
        weight_stride = calc_stride(weight_size)

        # Create input tensors: input, weight
        input_tensor = Tensor(input_size, input_stride, tensor.dtype, tensor.device, tensor.supported_ops)
        weight_tensor = Tensor(weight_size, weight_stride, tensor.dtype, tensor.device, tensor.supported_ops)

        result = [input_tensor, weight_tensor]

        return result

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for linear transformation operation."""
        return f"{output_name} = torch.nn.functional.linear({input_names[0]}, {input_names[1]})"
