"""Tests for StdOperator."""

import pytest
from .std import StdOperator
from torchfuzz.tensor import Tensor


class TestStdOperator:
    """Test class for StdOperator."""

    @pytest.fixture
    def std_op(self):
        """Create a StdOperator instance."""
        return StdOperator()

    def test_can_produce_within_dimension_limit(self, std_op):
        """Test that StdOperator can produce tensors within dimension limits."""
        # Can produce tensors with < 5 dimensions
        for ndim in range(5):
            size = tuple(2 for _ in range(ndim))
            stride = tuple(2**(ndim-1-i) for i in range(ndim)) if ndim > 0 else ()
            tensor = Tensor(size, stride, "float32", "cuda", [])
            assert std_op.can_produce(tensor) is True

    def test_cannot_produce_at_dimension_limit(self, std_op):
        """Test that StdOperator cannot produce tensors at dimension limit."""
        # Cannot produce tensors with 5 dimensions (at limit)
        size = (2, 2, 2, 2, 2)
        stride = (16, 8, 4, 2, 1)
        tensor = Tensor(size, stride, "float32", "cuda", [])
        assert std_op.can_produce(tensor) is False

    def test_decompose_scalar_output(self, std_op):
        """Test decomposition for scalar output tensor."""
        scalar = Tensor((), (), "float32", "cuda", [])
        inputs = std_op.decompose(scalar)

        assert len(inputs) == 1
        input_tensor = inputs[0]

        # Input should have 1-3 dimensions
        assert 1 <= len(input_tensor.size) <= 3
        # All dimensions should be >= 2
        assert all(s >= 2 for s in input_tensor.size)
        # Should mark for full reduction
        assert scalar._std_dim == "all"

    def test_decompose_non_scalar_output(self, std_op):
        """Test decomposition for non-scalar output tensor."""
        tensor = Tensor((3, 4), (4, 1), "float32", "cuda", [])
        inputs = std_op.decompose(tensor)

        assert len(inputs) == 1
        input_tensor = inputs[0]

        # Input should have one more dimension than output
        assert len(input_tensor.size) == len(tensor.size) + 1

        # Check that the reduction dimension was inserted correctly
        std_dim = tensor._std_dim
        assert isinstance(std_dim, int)
        assert 0 <= std_dim <= len(tensor.size)

        # Verify the shape after removing the std dimension matches output
        input_shape_list = list(input_tensor.size)
        input_shape_list.pop(std_dim)
        assert tuple(input_shape_list) == tensor.size

    def test_decompose_preserves_dtype_and_device(self, std_op):
        """Test that decomposition preserves dtype and device."""
        tensor = Tensor((5,), (1,), "bfloat16", "cuda", [])
        inputs = std_op.decompose(tensor)

        input_tensor = inputs[0]
        assert input_tensor.dtype == tensor.dtype
        assert input_tensor.device == tensor.device
        assert input_tensor.supported_ops == tensor.supported_ops

    def test_decompose_inserted_dimension_size(self, std_op):
        """Test that inserted dimension has appropriate size."""
        tensor = Tensor((2, 3), (3, 1), "float32", "cuda", [])
        inputs = std_op.decompose(tensor)

        input_tensor = inputs[0]
        std_dim = tensor._std_dim

        # The inserted dimension should have size >= 2
        assert input_tensor.size[std_dim] >= 2
        assert input_tensor.size[std_dim] <= 5

    def test_codegen_scalar_reduction(self, std_op):
        """Test code generation for scalar reduction."""
        tensor = Tensor((), (), "float32", "cuda", [])
        tensor._std_dim = "all"

        output_name = "result"
        input_names = ["input_tensor"]

        code = std_op.codegen(output_name, input_names, tensor)
        expected = "result = input_tensor.std()"

        assert code == expected

    def test_codegen_single_dimension_reduction(self, std_op):
        """Test code generation for single dimension reduction."""
        tensor = Tensor((3, 4), (4, 1), "float32", "cuda", [])
        tensor._std_dim = 1

        output_name = "out"
        input_names = ["x"]

        code = std_op.codegen(output_name, input_names, tensor)
        expected = "out = x.std(dim=1)"

        assert code == expected

    def test_codegen_tuple_dimension_reduction(self, std_op):
        """Test code generation for tuple dimension reduction."""
        tensor = Tensor((2, 3), (3, 1), "float32", "cuda", [])
        tensor._std_dim = (0, 2)

        output_name = "output"
        input_names = ["data"]

        code = std_op.codegen(output_name, input_names, tensor)
        expected = "output = data.std(dim=(0, 2))"

        assert code == expected

    def test_codegen_no_std_dim_attribute(self, std_op):
        """Test code generation when _std_dim attribute is missing."""
        tensor = Tensor((2, 2), (2, 1), "float32", "cuda", [])
        # Don't set _std_dim attribute

        output_name = "result"
        input_names = ["input"]

        code = std_op.codegen(output_name, input_names, tensor)
        expected = "result = input.std()"

        assert code == expected

    def test_operator_name(self, std_op):
        """Test that operator has correct name."""
        assert std_op.name == "std"
