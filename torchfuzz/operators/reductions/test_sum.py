"""Tests for SumOperator."""

import pytest
from .sum import SumOperator
from torchfuzz.tensor import Tensor


class TestSumOperator:
    """Test class for SumOperator."""

    @pytest.fixture
    def sum_op(self):
        """Create a SumOperator instance."""
        return SumOperator()

    def test_can_produce_within_dimension_limit(self, sum_op):
        """Test that SumOperator can produce tensors within dimension limits."""
        # Can produce tensors with < 5 dimensions
        for ndim in range(5):
            size = tuple(2 for _ in range(ndim))
            stride = tuple(2**(ndim-1-i) for i in range(ndim)) if ndim > 0 else ()
            tensor = Tensor(size, stride, "float32", "cuda", [])
            assert sum_op.can_produce(tensor) is True

    def test_cannot_produce_at_dimension_limit(self, sum_op):
        """Test that SumOperator cannot produce tensors at dimension limit."""
        # Cannot produce tensors with 5 dimensions (at limit)
        size = (2, 2, 2, 2, 2)
        stride = (16, 8, 4, 2, 1)
        tensor = Tensor(size, stride, "float32", "cuda", [])
        assert sum_op.can_produce(tensor) is False

    def test_decompose_scalar_output(self, sum_op):
        """Test decomposition for scalar output tensor."""
        scalar = Tensor((), (), "float32", "cuda", [])
        inputs = sum_op.decompose(scalar)

        assert len(inputs) == 1
        input_tensor = inputs[0]

        # Input should have 1-3 dimensions
        assert 1 <= len(input_tensor.size) <= 3
        # All dimensions should be >= 2
        assert all(s >= 2 for s in input_tensor.size)
        # Should mark for full reduction
        assert scalar._sum_dim == "all"

    def test_decompose_non_scalar_output(self, sum_op):
        """Test decomposition for non-scalar output tensor."""
        tensor = Tensor((3, 4), (4, 1), "float32", "cuda", [])
        inputs = sum_op.decompose(tensor)

        assert len(inputs) == 1
        input_tensor = inputs[0]

        # Input should have one more dimension than output
        assert len(input_tensor.size) == len(tensor.size) + 1

        # Check that the reduction dimension was inserted correctly
        sum_dim = tensor._sum_dim
        assert isinstance(sum_dim, int)
        assert 0 <= sum_dim <= len(tensor.size)

        # Verify the shape after removing the sum dimension matches output
        input_shape_list = list(input_tensor.size)
        input_shape_list.pop(sum_dim)
        assert tuple(input_shape_list) == tensor.size

    def test_decompose_preserves_dtype_and_device(self, sum_op):
        """Test that decomposition preserves dtype and device."""
        tensor = Tensor((5,), (1,), "bfloat16", "cuda", [])
        inputs = sum_op.decompose(tensor)

        input_tensor = inputs[0]
        assert input_tensor.dtype == tensor.dtype
        assert input_tensor.device == tensor.device
        assert input_tensor.supported_ops == tensor.supported_ops

    def test_decompose_inserted_dimension_size(self, sum_op):
        """Test that inserted dimension has appropriate size."""
        tensor = Tensor((2, 3), (3, 1), "float32", "cuda", [])
        inputs = sum_op.decompose(tensor)

        input_tensor = inputs[0]
        sum_dim = tensor._sum_dim

        # The inserted dimension should have size >= 2
        assert input_tensor.size[sum_dim] >= 2
        assert input_tensor.size[sum_dim] <= 5

    def test_codegen_scalar_reduction(self, sum_op):
        """Test code generation for scalar reduction."""
        tensor = Tensor((), (), "float32", "cuda", [])
        tensor._sum_dim = "all"

        output_name = "result"
        input_names = ["input_tensor"]

        code = sum_op.codegen(output_name, input_names, tensor)
        expected = "result = input_tensor.sum()"

        assert code == expected

    def test_codegen_single_dimension_reduction(self, sum_op):
        """Test code generation for single dimension reduction."""
        tensor = Tensor((3, 4), (4, 1), "float32", "cuda", [])
        tensor._sum_dim = 1

        output_name = "out"
        input_names = ["x"]

        code = sum_op.codegen(output_name, input_names, tensor)
        expected = "out = x.sum(dim=1)"

        assert code == expected

    def test_codegen_tuple_dimension_reduction(self, sum_op):
        """Test code generation for tuple dimension reduction."""
        tensor = Tensor((2, 3), (3, 1), "float32", "cuda", [])
        tensor._sum_dim = (0, 2)

        output_name = "output"
        input_names = ["data"]

        code = sum_op.codegen(output_name, input_names, tensor)
        expected = "output = data.sum(dim=(0, 2))"

        assert code == expected

    def test_codegen_no_sum_dim_attribute(self, sum_op):
        """Test code generation when _sum_dim attribute is missing."""
        tensor = Tensor((2, 2), (2, 1), "float32", "cuda", [])
        # Don't set _sum_dim attribute

        output_name = "result"
        input_names = ["input"]

        code = sum_op.codegen(output_name, input_names, tensor)
        expected = "result = input.sum()"

        assert code == expected

    def test_operator_name(self, sum_op):
        """Test that operator has correct name."""
        assert sum_op.name == "sum"
