"""Scaled dot product attention operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class ScaledDotProductAttentionOperator(Operator):
    """Operator for scaled dot product attention (torch.nn.functional.scaled_dot_product_attention)."""

    def __init__(self):
        super().__init__("scaled_dot_product_attention")

    def can_produce(self, tensor):
        """SDPA can produce tensors that are 4D (batch_size, num_heads, seq_len, head_dim) and total elements < 1M."""
        # Check if tensor has 4 dimensions
        if len(tensor.size) != 4:
            return False
        # Calculate the product of all dimensions (total number of elements)
        num_elements = 1
        for dim in tensor.size:
            num_elements *= dim
        # Return True only if total elements is less than 1,000,000
        return num_elements < 1_000_000

    def decompose(self, tensor):
        """Decompose tensor into input tensors for scaled dot product attention."""
        # tensor shape is (batch_size, num_heads, seq_len, head_dim)
        batch_size, num_heads, seq_len, head_dim = tensor.size

        # For attention, we need to be more careful about dimensions
        # The output shape matches the query shape: (batch_size, num_heads, seq_len, head_dim)
        # Key and value can have different sequence length but must have same head_dim

        # Choose key/value sequence length - keep it reasonable and valid
        # Ensure min <= max to avoid empty randint ranges
        max_kv_seq_len = max(1, min(seq_len * 2, 128))
        min_kv_seq_len = max(1, min(seq_len // 2, max_kv_seq_len))
        key_value_seq_len = random.randint(min_kv_seq_len, max_kv_seq_len)

        # Query shape (determines output shape)
        query_size = (batch_size, num_heads, seq_len, head_dim)
        # Key and value shapes (can have different sequence length but must have same head_dim)
        key_size = (batch_size, num_heads, key_value_seq_len, head_dim)
        # Value must have same head_dim as query and key for proper attention computation
        value_size = (batch_size, num_heads, key_value_seq_len, head_dim)

        # Calculate strides for contiguous tensors
        def calc_stride(size):
            stride = [1]
            for dim in reversed(size[:-1]):
                stride.insert(0, stride[0] * dim)
            return tuple(stride)

        query_stride = calc_stride(query_size)
        key_stride = calc_stride(key_size)
        value_stride = calc_stride(value_size)

        # Create input tensors: query, key, value
        query_tensor = Tensor(query_size, query_stride, tensor.dtype, tensor.device, tensor.supported_ops)
        key_tensor = Tensor(key_size, key_stride, tensor.dtype, tensor.device, tensor.supported_ops)
        value_tensor = Tensor(value_size, value_stride, tensor.dtype, tensor.device, tensor.supported_ops)

        result = [query_tensor, key_tensor, value_tensor]

        return result

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for scaled dot product attention operation."""
        # Cast inputs to float to satisfy SDPA requirements when upstream tensors are integer
        q = f"{input_names[0]}.float()"
        k = f"{input_names[1]}.float()"
        v = f"{input_names[2]}.float()"
        cast_back = f".to({input_names[0]}.dtype)"
        return f"{output_name} = torch.nn.functional.scaled_dot_product_attention({q}, {k}, {v}){cast_back}"
