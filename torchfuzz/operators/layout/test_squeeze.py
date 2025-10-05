"""Tests for SqueezeOperator."""

import pytest
from .squeeze import SqueezeOperator
from torchfuzz.tensor import Tensor


class TestSqueezeOperator:
    """Test class for SqueezeOperator."""

    @pytest.fixture
    def squeeze_op(self):
        """Create a SqueezeOperator instance."""
        return SqueezeOperator()

    def test_can_produce_returns_true(self, squeeze_op):
        """Test that SqueezeOperator can produce any tensor."""
        test_tensors = [
            Tensor((), (), "float32", "cuda", []),  # scalar
            Tensor((5,), (1,), "float32", "cuda", []),  # 1D
            Tensor((2, 3), (3, 1), "float32", "cuda", []),  # 2D
            Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", []),  # 3D
            Tensor((2, 3, 4, 5), (60, 20, 5, 1), "float16", "cuda", [])  # 4D
        ]

        for tensor in test_tensors:
            assert squeeze_op.can_produce(tensor) is True

    def test_decompose_returns_single_input(self, squeeze_op):
        """Test that decomposition returns exactly one input tensor."""
        tensor = Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", [])
        inputs = squeeze_op.decompose(tensor)

        assert len(inputs) == 1

    def test_decompose_preserves_numel(self, squeeze_op):
        """Test that decomposition preserves the number of elements."""
        tensor = Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", [])
        inputs = squeeze_op.decompose(tensor)

        input_tensor = inputs[0]

        # Calculate number of elements
        output_numel = 1
        for s in tensor.size:
            output_numel *= s

        input_numel = 1
        for s in input_tensor.size:
            input_numel *= s

        assert input_numel == output_numel

    def test_decompose_preserves_properties(self, squeeze_op):
        """Test that decomposition preserves tensor properties."""
        tensor = Tensor((6,), (1,), "bfloat16", "cuda", ["test_ops"])
        inputs = squeeze_op.decompose(tensor)

        input_tensor = inputs[0]
        assert input_tensor.dtype == tensor.dtype
        assert input_tensor.device == tensor.device
        assert input_tensor.supported_ops == tensor.supported_ops

    def test_decompose_scalar_tensor(self, squeeze_op):
        """Test decomposition of scalar tensors."""
        scalar = Tensor((), (), "float32", "cuda", [])
        inputs = squeeze_op.decompose(scalar)

        input_tensor = inputs[0]
        # Scalar has numel=1, so input should also have numel=1
        input_numel = 1
        for s in input_tensor.size:
            input_numel *= s
        assert input_numel == 1

    def test_decompose_adds_squeeze_dimensions(self, squeeze_op):
        """Test that decompose adds exactly one dimension of size 1."""
        tensor = Tensor((2, 3), (3, 1), "float32", "cuda", [])

        # Run multiple times to check randomness
        for _ in range(10):
            inputs = squeeze_op.decompose(tensor)
            input_tensor = inputs[0]

            # Input should have exactly one more dimension than output
            assert len(input_tensor.size) == len(tensor.size) + 1

            # Should have squeeze dimension metadata
            assert hasattr(input_tensor, '_squeeze_dim')

            # Should have exactly one dimension of size 1
            ones_count = sum(1 for dim in input_tensor.size if dim == 1)
            assert ones_count == 1

            # The squeeze dimension should be the position of the size-1 dimension
            squeeze_dim = input_tensor._squeeze_dim
            assert input_tensor.size[squeeze_dim] == 1

    def test_decompose_contiguous_stride(self, squeeze_op):
        """Test that input tensor has contiguous strides."""
        tensor = Tensor((2, 6), (6, 1), "float32", "cuda", [])
        inputs = squeeze_op.decompose(tensor)

        input_tensor = inputs[0]
        size = input_tensor.size
        stride = input_tensor.stride

        # Verify contiguous stride calculation
        expected_stride = []
        acc = 1
        for s in reversed(size):
            expected_stride.insert(0, acc)
            acc *= s

        assert stride == tuple(expected_stride)

    def test_decompose_zero_size_tensor(self, squeeze_op):
        """Test decomposition with tensor containing zero size."""
        tensor = Tensor((0, 5), (5, 1), "float32", "cuda", [])
        inputs = squeeze_op.decompose(tensor)

        input_tensor = inputs[0]
        # Both should have numel=0
        output_numel = 0 * 5
        input_numel = 1
        for s in input_tensor.size:
            input_numel *= s

        assert input_numel == output_numel == 0

    def test_decompose_squeeze_dimension_range(self, squeeze_op):
        """Test that squeeze dimension is in valid range."""
        tensor = Tensor((3, 4), (4, 1), "float32", "cuda", [])

        for _ in range(10):
            inputs = squeeze_op.decompose(tensor)
            input_tensor = inputs[0]

            # Squeeze dimension should be in valid range for input tensor
            squeeze_dim = input_tensor._squeeze_dim
            assert 0 <= squeeze_dim < len(input_tensor.size)

    def test_codegen_basic(self, squeeze_op):
        """Test basic code generation."""
        tensor = Tensor((2, 6), (6, 1), "float32", "cuda", [])

        # First decompose to set up the metadata
        inputs = squeeze_op.decompose(tensor)
        input_tensor = inputs[0]
        dim = input_tensor._squeeze_dim

        output_name = "output"
        input_names = ["input"]

        code = squeeze_op.codegen(output_name, input_names, tensor)
        expected = f"output = torch.squeeze(input, {dim})"

        assert code == expected

    def test_codegen_scalar_tensor(self, squeeze_op):
        """Test code generation for scalar tensor."""
        scalar = Tensor((), (), "float32", "cuda", [])

        # First decompose to set up the metadata
        inputs = squeeze_op.decompose(scalar)
        input_tensor = inputs[0]
        dim = input_tensor._squeeze_dim

        output_name = "result"
        input_names = ["x"]

        code = squeeze_op.codegen(output_name, input_names, scalar)
        expected = f"result = torch.squeeze(x, {dim})"

        assert code == expected

    def test_codegen_1d_tensor(self, squeeze_op):
        """Test code generation for 1D tensor."""
        tensor = Tensor((12,), (1,), "float32", "cuda", [])

        # First decompose to set up the metadata
        inputs = squeeze_op.decompose(tensor)
        input_tensor = inputs[0]
        dim = input_tensor._squeeze_dim

        output_name = "out"
        input_names = ["data"]

        code = squeeze_op.codegen(output_name, input_names, tensor)
        expected = f"out = torch.squeeze(data, {dim})"

        assert code == expected

    def test_codegen_3d_tensor(self, squeeze_op):
        """Test code generation for 3D tensor."""
        tensor = Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", [])

        # First decompose to set up the metadata
        inputs = squeeze_op.decompose(tensor)
        input_tensor = inputs[0]
        dim = input_tensor._squeeze_dim

        output_name = "squeezed"
        input_names = ["original"]

        code = squeeze_op.codegen(output_name, input_names, tensor)
        expected = f"squeezed = torch.squeeze(original, {dim})"

        assert code == expected

    def test_codegen_different_variable_names(self, squeeze_op):
        """Test code generation with different variable names."""
        tensor = Tensor((5, 2), (2, 1), "float32", "cuda", [])

        # First decompose to set up the metadata
        inputs = squeeze_op.decompose(tensor)
        input_tensor = inputs[0]
        dim = input_tensor._squeeze_dim

        output_name = "tensor_squeezed"
        input_names = ["input_data"]

        code = squeeze_op.codegen(output_name, input_names, tensor)
        expected = f"tensor_squeezed = torch.squeeze(input_data, {dim})"

        assert code == expected

    def test_codegen_zero_size_tensor(self, squeeze_op):
        """Test code generation for tensor with zero size."""
        tensor = Tensor((0, 5), (5, 1), "float32", "cuda", [])

        # First decompose to set up the metadata
        inputs = squeeze_op.decompose(tensor)
        input_tensor = inputs[0]
        dim = input_tensor._squeeze_dim

        output_name = "empty_tensor"
        input_names = ["input_empty"]

        code = squeeze_op.codegen(output_name, input_names, tensor)
        expected = f"empty_tensor = torch.squeeze(input_empty, {dim})"

        assert code == expected

    def test_operator_name(self, squeeze_op):
        """Test that operator has correct name."""
        assert squeeze_op.name == "squeeze"

    def test_decompose_dimensions_in_valid_range(self, squeeze_op):
        """Test that input tensor dimensions are reasonable."""
        tensor = Tensor((24,), (1,), "float32", "cuda", [])

        for _ in range(10):
            inputs = squeeze_op.decompose(tensor)
            input_tensor = inputs[0]

            # Should not create excessively large tensors
            assert len(input_tensor.size) <= len(tensor.size) + 3

    def test_decompose_only_adds_ones(self, squeeze_op):
        """Test that only dimensions of size 1 are added."""
        tensor = Tensor((3, 4), (4, 1), "float32", "cuda", [])

        for _ in range(10):
            inputs = squeeze_op.decompose(tensor)
            input_tensor = inputs[0]

            # Count how many non-1 dimensions we have
            non_one_dims = [dim for dim in input_tensor.size if dim != 1]

            # Non-1 dimensions should match original tensor
            assert sorted(non_one_dims) == sorted(tensor.size)
