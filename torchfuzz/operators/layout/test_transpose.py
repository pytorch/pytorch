"""Tests for TransposeOperator."""

import pytest
from .transpose import TransposeOperator
from torchfuzz.tensor import Tensor


class TestTransposeOperator:
    """Test class for TransposeOperator."""

    @pytest.fixture
    def transpose_op(self):
        """Create a TransposeOperator instance."""
        return TransposeOperator()

    def test_can_produce_2d_tensor(self, transpose_op):
        """Test that TransposeOperator can produce 2D tensors."""
        tensor = Tensor((2, 3), (3, 1), "float32", "cuda", [])
        assert transpose_op.can_produce(tensor) is True

    def test_can_produce_3d_tensor(self, transpose_op):
        """Test that TransposeOperator can produce 3D tensors."""
        tensor = Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", [])
        assert transpose_op.can_produce(tensor) is True

    def test_cannot_produce_0d_tensor(self, transpose_op):
        """Test that TransposeOperator cannot produce scalar tensors."""
        tensor = Tensor((), (), "float32", "cuda", [])
        assert transpose_op.can_produce(tensor) is False

    def test_cannot_produce_1d_tensor(self, transpose_op):
        """Test that TransposeOperator cannot produce 1D tensors."""
        tensor = Tensor((5,), (1,), "float32", "cuda", [])
        assert transpose_op.can_produce(tensor) is False

    def test_decompose_returns_single_input(self, transpose_op):
        """Test that decomposition returns exactly one input tensor."""
        tensor = Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", [])
        inputs = transpose_op.decompose(tensor)

        assert len(inputs) == 1

    def test_decompose_preserves_numel(self, transpose_op):
        """Test that decomposition preserves the number of elements."""
        tensor = Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", [])
        inputs = transpose_op.decompose(tensor)

        input_tensor = inputs[0]

        # Calculate number of elements
        output_numel = 1
        for s in tensor.size:
            output_numel *= s

        input_numel = 1
        for s in input_tensor.size:
            input_numel *= s

        assert input_numel == output_numel

    def test_decompose_preserves_properties(self, transpose_op):
        """Test that decomposition preserves tensor properties."""
        tensor = Tensor((6, 4), (4, 1), "bfloat16", "cuda", ["test_ops"])
        inputs = transpose_op.decompose(tensor)

        input_tensor = inputs[0]
        assert input_tensor.dtype == tensor.dtype
        assert input_tensor.device == tensor.device
        assert input_tensor.supported_ops == tensor.supported_ops

    def test_decompose_swaps_dimensions(self, transpose_op):
        """Test that decomposition creates input with swapped dimensions."""
        tensor = Tensor((2, 3), (3, 1), "float32", "cuda", [])
        inputs = transpose_op.decompose(tensor)

        input_tensor = inputs[0]
        # For a 2D tensor, dimensions should be swapped
        assert input_tensor.size == (3, 2)
        assert input_tensor.stride == (1, 3)

    def test_decompose_with_more_dimensions(self, transpose_op):
        """Test decomposition with higher dimensional tensors."""
        tensor = Tensor((2, 3, 4, 5), (60, 20, 5, 1), "float32", "cuda", [])
        inputs = transpose_op.decompose(tensor)

        input_tensor = inputs[0]
        # Number of elements should be preserved
        input_numel = 1
        for s in input_tensor.size:
            input_numel *= s
        assert input_numel == 2 * 3 * 4 * 5

    def test_decompose_stores_metadata(self, transpose_op):
        """Test that decomposition stores transpose dimensions in metadata."""
        tensor = Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", [])
        inputs = transpose_op.decompose(tensor)

        # Check that metadata was stored
        assert hasattr(tensor, "_transpose_dim0")
        assert hasattr(tensor, "_transpose_dim1")
        assert 0 <= tensor._transpose_dim0 < 3
        assert 0 <= tensor._transpose_dim1 < 3

    def test_decompose_different_dimensions(self, transpose_op):
        """Test that decomposition picks different dimensions when possible."""
        tensor = Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", [])

        # Test multiple decompositions
        for _ in range(10):
            inputs = transpose_op.decompose(tensor)  # This creates a new tensor each time
            test_tensor = Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", [])
            inputs = transpose_op.decompose(test_tensor)

            dim0 = test_tensor._transpose_dim0
            dim1 = test_tensor._transpose_dim1
            assert dim0 != dim1  # Should be different dimensions

    def test_codegen_basic_2d(self, transpose_op):
        """Test basic code generation for 2D tensor."""
        tensor = Tensor((2, 3), (3, 1), "float32", "cuda", [])
        tensor._transpose_dim0 = 0
        tensor._transpose_dim1 = 1

        output_name = "output"
        input_names = ["input"]

        code = transpose_op.codegen(output_name, input_names, tensor)
        expected = "output = input.transpose(0, 1)"

        assert code == expected

    def test_codegen_3d_tensor(self, transpose_op):
        """Test code generation for 3D tensor."""
        tensor = Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", [])
        tensor._transpose_dim0 = 1
        tensor._transpose_dim1 = 2

        output_name = "result"
        input_names = ["data"]

        code = transpose_op.codegen(output_name, input_names, tensor)
        expected = "result = data.transpose(1, 2)"

        assert code == expected

    def test_codegen_default_dimensions(self, transpose_op):
        """Test code generation with default dimensions when metadata missing."""
        tensor = Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", [])
        # Don't set transpose metadata

        output_name = "out"
        input_names = ["x"]

        code = transpose_op.codegen(output_name, input_names, tensor)
        expected = "out = x.transpose(0, 1)"

        assert code == expected

    def test_codegen_different_variable_names(self, transpose_op):
        """Test code generation with different variable names."""
        tensor = Tensor((5, 2), (2, 1), "float32", "cuda", [])
        tensor._transpose_dim0 = 1
        tensor._transpose_dim1 = 0

        output_name = "tensor_transposed"
        input_names = ["input_data"]

        code = transpose_op.codegen(output_name, input_names, tensor)
        expected = "tensor_transposed = input_data.transpose(1, 0)"

        assert code == expected

    def test_operator_name(self, transpose_op):
        """Test that operator has correct name."""
        assert transpose_op.name == "transpose"

    def test_decompose_raises_error_for_invalid_tensor(self, transpose_op):
        """Test that decompose raises error for tensors with less than 2 dimensions."""
        scalar = Tensor((), (), "float32", "cuda", [])

        with pytest.raises(ValueError, match="Cannot transpose tensor with less than 2 dimensions"):
            transpose_op.decompose(scalar)

        tensor_1d = Tensor((5,), (1,), "float32", "cuda", [])

        with pytest.raises(ValueError, match="Cannot transpose tensor with less than 2 dimensions"):
            transpose_op.decompose(tensor_1d)
