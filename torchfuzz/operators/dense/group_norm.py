"""Group normalization operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class GroupNormOperator(Operator):
    """Operator for group normalization (torch.nn.functional.group_norm)."""

    def __init__(self):
        super().__init__("group_norm")

    def can_produce(self, tensor):
        """GroupNorm can produce tensors that are at least 3D (N, C, ...) with sufficient elements for normalization."""
        if len(tensor.size) < 3:
            return False

        # Group normalization requires more than 1 value per channel when training
        # Since we generate tensors with requires_grad=True, we need to ensure
        # that the spatial dimensions (after batch and channel) have more than 1 element total
        num_channels = tensor.size[1]
        spatial_dims = tensor.size[2:]

        # Calculate total spatial elements
        spatial_elements = 1
        for dim in spatial_dims:
            spatial_elements *= dim

        # Need at least 2 spatial elements for meaningful statistics in training mode
        # Also need at least 1 channel to normalize
        return spatial_elements >= 2 and num_channels >= 1

    def decompose(self, tensor):
        """Decompose tensor into input tensors for group norm operation."""
        # tensor shape is the output shape (N, C, ...)
        input_size = tensor.size
        num_channels = input_size[1]

        # Choose number of groups (must divide num_channels)
        possible_groups = [g for g in [1, 2, 4, 8, 16, 32] if num_channels % g == 0 and g <= num_channels]
        if not possible_groups:
            possible_groups = [1]  # fallback
        num_groups = random.choice(possible_groups)

        # Calculate strides for contiguous tensors
        def calc_stride(size):
            stride = [1]
            for dim in reversed(size[:-1]):
                stride.insert(0, stride[0] * dim)
            return tuple(stride)

        input_stride = calc_stride(input_size)

        # Type promotion for realistic LLM types
        dtype = tensor.dtype

        # Create input tensor (same shape as output)
        input_tensor = Tensor(input_size, input_stride, dtype, tensor.device, tensor.supported_ops)

        # num_groups as a scalar tensor
        num_groups_tensor = Tensor((1,), (1,), "int64", tensor.device, tensor.supported_ops)

        result = [input_tensor, num_groups_tensor]

        # Store num_groups for codegen
        self._num_groups = num_groups
        return result

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for group norm operation."""
        if len(input_names) not in [2, 3, 4]:
            raise ValueError("Group norm requires 2, 3, or 4 inputs")

        # Get the number of groups
        num_groups = getattr(self, '_num_groups', 1)

        # Ensure num_groups divides the number of channels in the input tensor
        # This prevents runtime errors in torch.nn.functional.group_norm
        # Try to extract the number of channels from the output_tensor shape
        num_channels = None
        if hasattr(output_tensor, "size") and len(output_tensor.size) > 1:
            num_channels = output_tensor.size[1]
        if num_channels is not None and num_channels % num_groups != 0:
            # Fallback to 1 group if not divisible
            num_groups = 1

        return f"{output_name} = torch.nn.functional.group_norm({input_names[0]}, {num_groups})"
