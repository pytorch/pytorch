"""Embedding operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class EmbeddingOperator(Operator):
    """Operator for embedding lookup (torch.nn.functional.embedding)."""

    def __init__(self):
        super().__init__("embedding")

    def can_produce(self, tensor):
        """Embedding can produce tensors that are at least 1D and floating point."""
        # Embedding only supports floating point output tensors
        if tensor.dtype in ["int8", "int16", "int32", "int64", "uint8", "bool"]:
            return False
        return len(tensor.size) >= 1

    def decompose(self, tensor):
        """Decompose tensor into input tensors for embedding lookup."""
        # tensor shape is (..., embedding_dim)
        # input indices shape is (...) - same as tensor shape without last dimension
        # weight shape is (num_embeddings, embedding_dim)

        *batch_dims, embedding_dim = tensor.size

        # Input indices tensor shape: (...) - same as batch dimensions
        # For embedding, the input indices determine all but the last dimension of output
        input_size = tuple(batch_dims) if batch_dims else tuple()  # Can be 0D (scalar)
        if not input_size:
            input_size = tuple()  # Scalar indices for 1D output

        # Choose a random vocabulary size (num_embeddings)
        possible_vocab_sizes = [100, 256, 512, 1000, 5000, 10000, 50000]
        num_embeddings = random.choice(possible_vocab_sizes)

        # Weight tensor shape: (num_embeddings, embedding_dim)
        weight_size = (num_embeddings, embedding_dim)

        # Calculate strides for contiguous tensors
        def calc_stride(size):
            if not size:  # Handle empty tuple (scalar tensor)
                return tuple()
            stride = [1]
            for dim in reversed(size[:-1]):
                stride.insert(0, stride[0] * dim)
            return tuple(stride)

        input_stride = calc_stride(input_size) if input_size else tuple()
        weight_stride = calc_stride(weight_size)

        # Create input tensors: input (indices), weight (embedding table)
        # Input indices should be integer type with restricted supported_ops
        # to prevent operations that would convert integers to floats

        # Create input tensor with NO supported operations to prevent further decomposition
        # This ensures the indices remain as simple leaf tensors and don't get modified
        # by other operations that could create out-of-bounds values
        input_tensor = Tensor(input_size, input_stride, "int64", tensor.device, set())
        weight_tensor = Tensor(weight_size, weight_stride, tensor.dtype, tensor.device, tensor.supported_ops)

        # Store embedding constraints as metadata on the input tensor
        # This will be used by codegen to generate valid indices
        input_tensor._embedding_vocab_size = num_embeddings

        return [input_tensor, weight_tensor]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for embedding lookup operation."""
        # output_tensor parameter is not used but required by interface
        # Clamp indices to valid range to prevent out-of-bounds access
        # and explicitly cast to long to satisfy embedding's dtype requirements
        weight_name = input_names[1]
        indices_name = input_names[0]
        return f"{output_name} = torch.nn.functional.embedding(torch.clamp({indices_name}, 0, {weight_name}.size(0) - 1).to(torch.long), {weight_name})"
