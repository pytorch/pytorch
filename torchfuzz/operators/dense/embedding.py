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
        input_size = tuple(batch_dims) if batch_dims else (1,)  # At least 1D for indices

        # Choose a random vocabulary size (num_embeddings)
        possible_vocab_sizes = [100, 256, 512, 1000, 5000, 10000, 50000]
        num_embeddings = random.choice(possible_vocab_sizes)

        # Weight tensor shape: (num_embeddings, embedding_dim)
        weight_size = (num_embeddings, embedding_dim)

        # Calculate strides for contiguous tensors
        def calc_stride(size):
            stride = [1]
            for dim in reversed(size[:-1]):
                stride.insert(0, stride[0] * dim)
            return tuple(stride)

        input_stride = calc_stride(input_size)
        weight_stride = calc_stride(weight_size)

        # Create input tensors: input (indices), weight (embedding table)
        # Input indices should be integer type
        input_tensor = Tensor(input_size, input_stride, "int64", tensor.device, tensor.supported_ops)
        weight_tensor = Tensor(weight_size, weight_stride, tensor.dtype, tensor.device, tensor.supported_ops)

        return [input_tensor, weight_tensor]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for embedding lookup operation."""
        # output_tensor parameter is not used but required by interface
        return f"{output_name} = torch.nn.functional.embedding({input_names[0]}, {input_names[1]})"