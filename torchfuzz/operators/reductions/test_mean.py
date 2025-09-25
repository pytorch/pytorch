"""Tests for MeanOperator."""

import pytest
from .mean import MeanOperator
from torchfuzz.tensor import Tensor


class TestMeanOperator:
    """Test class for MeanOperator."""

    @pytest.fixture
    def mean_op(self):
        """Create a MeanOperator instance."""
        return MeanOperator()

    def test_can_produce_within_dimension_limit(self, mean_op):
        """Test that MeanOperator can produce tensors within dimension limits."""
        # Can produce tensors with < 5 dimensions
        for ndim in range(5):
            size = tuple(2 for _ in range(ndim))
            stride = tuple(2**(ndim-1-i) for i in range(ndim)) if ndim > 0 else ()
            tensor = Tensor(size, stride, "float32", "cuda", [])
            assert mean_op.can_produce(tensor) is True

    def test_cannot_produce_at_dimension_limit(self, mean_op):
        """Test that MeanOperator cannot produce tensors at dimension limit."""
        # Cannot produce tensors with 5 dimensions (at limit)
        size = (2, 2, 2, 2, 2)
        stride = (16, 8, 4, 2, 1)
        tensor = Tensor(size, stride, "float32", "cuda", [])
        assert mean_op.can_produce(tensor) is False

    def test_decompose_scalar_output(self, mean_op):
        """Test decomposition for scalar output tensor."""
        scalar = Tensor((), (), "float32", "cuda", [])
        inputs = mean_op.decompose(scalar)

        assert len(inputs) == 1
        input_tensor = inputs[0]

        # Input should have 1-3 dimensions
        assert 1 <= len(input_tensor.size) <= 3
        # All dimensions should be >= 2
        assert all(s >= 2 for s in input_tensor.size)
        # Should mark for full reduction
        assert scalar._mean_dim == "all"

    def test_decompose_non_scalar_output(self, mean_op):
        """Test decomposition for non-scalar output tensor."""
        tensor = Tensor((3, 4), (4, 1), "float32", "cuda", [])
        inputs = mean_op.decompose(tensor)

        assert len(inputs) == 1
        input_tensor = inputs[0]

        # Input should have one more dimension than output
        assert len(input_tensor.size) == len(tensor.size) + 1

        # Check that the reduction dimension was inserted correctly
        mean_dim = tensor._mean_dim
        assert isinstance(mean_dim, int)
        assert 0 <= mean_dim <= len(tensor.size)

        # Verify the shape after removing the mean dimension matches output
        input_shape_list = list(input_tensor.size)
        input_shape_list.pop(mean_dim)
        assert tuple(input_shape_list) == tensor.size

    def test_decompose_preserves_dtype_and_device(self, mean_op):
        """Test that decomposition preserves dtype and device."""
        tensor = Tensor((5,), (1,), "bfloat16", "cuda", [])
        inputs = mean_op.decompose(tensor)

        input_tensor = inputs[0]
        assert input_tensor.dtype == tensor.dtype
        assert input_tensor.device == tensor.device
        assert input_tensor.supported_ops == tensor.supported_ops

    def test_decompose_inserted_dimension_size(self, mean_op):
        """Test that inserted dimension has appropriate size."""
        tensor = Tensor((2, 3), (3, 1), "float32", "cuda", [])
        inputs = mean_op.decompose(tensor)

        input_tensor = inputs[0]
        mean_dim = tensor._mean_dim

        # The inserted dimension should have size >= 2
        assert input_tensor.size[mean_dim] >= 2
        assert input_tensor.size[mean_dim] <= 5

    def test_codegen_scalar_reduction(self, mean_op):
        """Test code generation for scalar reduction."""
        tensor = Tensor((), (), "float32", "cuda", [])
        tensor._mean_dim = "all"

        output_name = "result"
        input_names = ["input_tensor"]

        code = mean_op.codegen(output_name, input_names, tensor)
        expected = "result = input_tensor.mean()"

        assert code == expected

    def test_codegen_single_dimension_reduction(self, mean_op):
        """Test code generation for single dimension reduction."""
        tensor = Tensor((3, 4), (4, 1), "float32", "cuda", [])
        tensor._mean_dim = 1

        output_name = "out"
        input_names = ["x"]

        code = mean_op.codegen(output_name, input_names, tensor)
        expected = "out = x.mean(dim=1)"

        assert code == expected

    def test_codegen_tuple_dimension_reduction(self, mean_op):
        """Test code generation for tuple dimension reduction."""
        tensor = Tensor((2, 3), (3, 1), "float32", "cuda", [])
        tensor._mean_dim = (0, 2)

        output_name = "output"
        input_names = ["data"]

        code = mean_op.codegen(output_name, input_names, tensor)
        expected = "output = data.mean(dim=(0, 2))"

        assert code == expected

    def test_codegen_no_mean_dim_attribute(self, mean_op):
        """Test code generation when _mean_dim attribute is missing."""
        tensor = Tensor((2, 2), (2, 1), "float32", "cuda", [])
        # Don't set _mean_dim attribute

        output_name = "result"
        input_names = ["input"]

        code = mean_op.codegen(output_name, input_names, tensor)
        expected = "result = input.mean()"

        assert code == expected

    def test_operator_name(self, mean_op):
        """Test that operator has correct name."""
        assert mean_op.name == "mean"
