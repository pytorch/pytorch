"""Scaled dot product attention operator implementation."""

import random
from ..base.operator import Operator
from torchfuzz.tensor import Tensor


class ScaledDotProductAttentionOperator(Operator):
    """Operator for scaled dot product attention (torch.nn.functional.scaled_dot_product_attention)."""

    def __init__(self):
        super().__init__(supports_dtensor=False)

    def _can_produce_impl(self, output_tensor):
        """SDPA can produce tensors that are 4D (batch_size, num_heads, seq_len, head_dim) and total elements < 1M."""
        # Check if tensor has 4 dimensions
        if len(output_tensor.size) != 4:
            return False
        # Calculate the product of all dimensions (total number of elements)
        num_elements = 1
        for dim in output_tensor.size:
            num_elements *= dim
        # Return True only if total elements is less than 1,000,000
        return num_elements < 1_000_000

    def decompose(self, tensor):
        """Decompose tensor into input tensors for scaled dot product attention."""
        # tensor shape is (batch_size, num_heads, seq_len, head_dim)
        batch_size, num_heads, seq_len, head_dim = tensor.size

        # For self-attention, query, key, value have same shape
        # For cross-attention, key/value might have different seq_len
        key_value_seq_len = random.choice([seq_len, seq_len // 2, seq_len * 2, 512])

        # All have same shape for self-attention case
        qkv_size = (batch_size, num_heads, seq_len, head_dim)
        kv_size = (batch_size, num_heads, key_value_seq_len, head_dim)

        # Calculate strides for contiguous tensors
        def calc_stride(size):
            stride = [1]
            for dim in reversed(size[:-1]):
                stride.insert(0, stride[0] * dim)
            return tuple(stride)

        qkv_stride = calc_stride(qkv_size)
        kv_stride = calc_stride(kv_size)

        # Create input tensors: query, key, value
        query_tensor = Tensor(qkv_size, qkv_stride, tensor.dtype, tensor.device, tensor.supported_ops)
        key_tensor = Tensor(kv_size, kv_stride, tensor.dtype, tensor.device, tensor.supported_ops)
        value_tensor = Tensor(kv_size, kv_stride, tensor.dtype, tensor.device, tensor.supported_ops)

        result = [query_tensor, key_tensor, value_tensor]

        return result

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for scaled dot product attention operation."""
        return f"{output_name} = torch.nn.functional.scaled_dot_product_attention({input_names[0]}, {input_names[1]}, {input_names[2]})"
