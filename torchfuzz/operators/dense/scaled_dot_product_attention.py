"""Scaled dot product attention operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class ScaledDotProductAttentionOperator(Operator):
    """Operator for scaled dot product attention (torch.nn.functional.scaled_dot_product_attention)."""

    def __init__(self):
        super().__init__("scaled_dot_product_attention")

    def can_produce(self, tensor):
        """SDPA can produce tensors that are 4D (batch_size, num_heads, seq_len, head_dim)."""
        return len(tensor.size) == 4

    def decompose(self, tensor, num_inputs=3):
        """Decompose tensor into input tensors for scaled dot product attention."""
        if num_inputs not in [3, 4]:
            raise ValueError("Scaled dot product attention requires 3 or 4 inputs (query, key, value, optional attn_mask)")

        # tensor shape is (batch_size, num_heads, seq_len, head_dim)
        batch_size, num_heads, seq_len, head_dim = tensor.size
        
        # For self-attention, query, key, value have same shape
        # For cross-attention, key/value might have different seq_len
        key_value_seq_len = random.choice([seq_len, seq_len // 2, seq_len * 2, 512, 1024, 2048])
        
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

        # Type promotion for realistic LLM types
        input_dtypes = (tensor.dtype, tensor.dtype, tensor.dtype)
        if num_inputs == 4:
            input_dtypes = (tensor.dtype, tensor.dtype, tensor.dtype, "bool")

        # Create input tensors: query, key, value
        query_tensor = Tensor(qkv_size, qkv_stride, input_dtypes[0], tensor.device, tensor.supported_ops)
        key_tensor = Tensor(kv_size, kv_stride, input_dtypes[1], tensor.device, tensor.supported_ops)
        value_tensor = Tensor(kv_size, kv_stride, input_dtypes[2], tensor.device, tensor.supported_ops)
        
        result = [query_tensor, key_tensor, value_tensor]
        
        if num_inputs == 4:
            # Attention mask shape: (batch_size, num_heads, seq_len, key_value_seq_len)
            mask_size = (batch_size, num_heads, seq_len, key_value_seq_len)
            mask_stride = calc_stride(mask_size)
            mask_tensor = Tensor(mask_size, mask_stride, input_dtypes[3], tensor.device, tensor.supported_ops)
            result.append(mask_tensor)

        return result

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for scaled dot product attention operation."""
        if len(input_names) == 3:
            return f"{output_name} = torch.nn.functional.scaled_dot_product_attention({input_names[0]}, {input_names[1]}, {input_names[2]})"
        elif len(input_names) == 4:
            return f"{output_name} = torch.nn.functional.scaled_dot_product_attention({input_names[0]}, {input_names[1]}, {input_names[2]}, attn_mask={input_names[3]})"
        else:
            raise ValueError("Scaled dot product attention requires 3 or 4 inputs")

    def supports_variable_inputs(self) -> bool:
        """SDPA supports variable inputs (with or without attention mask)."""
        return True