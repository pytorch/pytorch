"""Tests for ArgmaxOperator."""

import pytest
from .argmax import ArgmaxOperator
from torchfuzz.tensor import Tensor


class TestArgmaxOperator:
    """Test class for ArgmaxOperator."""

    @pytest.fixture
    def argmax_op(self):
        """Create an ArgmaxOperator instance."""
        return ArgmaxOperator()

    def test_can_produce_within_dimension_limit_int64_only(self, argmax_op):
        """Test that ArgmaxOperator can produce int64 tensors within dimension limits."""
        # Can produce int64 tensors with < 5 dimensions
        for ndim in range(5):
            size = tuple(2 for _ in range(ndim))
            stride = tuple(2**(ndim-1-i) for i in range(ndim)) if ndim > 0 else ()
            tensor = Tensor(size, stride, "int64", "cuda", [])
            assert argmax_op.can_produce(tensor) is True

    def test_cannot_produce_non_int64_dtype(self, argmax_op):
        """Test that ArgmaxOperator cannot produce non-int64 tensors."""
        # Cannot produce non-int64 tensors (argmax returns indices)
        for dtype in ["float32", "float16", "bfloat16"]:
            tensor = Tensor((2, 3), (3, 1), dtype, "cuda", [])
            assert argmax_op.can_produce(tensor) is False

    def test_cannot_produce_at_dimension_limit(self, argmax_op):
        """Test that ArgmaxOperator cannot produce tensors at dimension limit."""
        # Cannot produce tensors with 5 dimensions (at limit)
        size = (2, 2, 2, 2, 2)
        stride = (16, 8, 4, 2, 1)
        tensor = Tensor(size, stride, "int64", "cuda", [])
        assert argmax_op.can_produce(tensor) is False

    def test_decompose_scalar_output(self, argmax_op):
        """Test decomposition for scalar output tensor."""
        scalar = Tensor((), (), "int64", "cuda", [])
        inputs = argmax_op.decompose(scalar)

        assert len(inputs) == 1
        input_tensor = inputs[0]

        # Input should have 1-3 dimensions
        assert 1 <= len(input_tensor.size) <= 3
        # All dimensions should be >= 2
        assert all(s >= 2 for s in input_tensor.size)
        # Input dtype should be float type (not int64)
        assert input_tensor.dtype in ["float32", "bfloat16", "float16"]
        # Should mark for full reduction
        assert scalar._argmax_dim == "all"

    def test_decompose_non_scalar_output(self, argmax_op):
        """Test decomposition for non-scalar output tensor."""
        tensor = Tensor((3, 4), (4, 1), "int64", "cuda", [])
        inputs = argmax_op.decompose(tensor)

        assert len(inputs) == 1
        input_tensor = inputs[0]

        # Input should have one more dimension than output
        assert len(input_tensor.size) == len(tensor.size) + 1
        # Input dtype should be float type
        assert input_tensor.dtype in ["float32", "bfloat16", "float16"]

        # Check that the reduction dimension was inserted correctly
        argmax_dim = tensor._argmax_dim
        assert isinstance(argmax_dim, int)
        assert 0 <= argmax_dim <= len(tensor.size)

        # Verify the shape after removing the argmax dimension matches output
        input_shape_list = list(input_tensor.size)
        input_shape_list.pop(argmax_dim)
        assert tuple(input_shape_list) == tensor.size

    def test_decompose_preserves_device_and_ops(self, argmax_op):
        """Test that decomposition preserves device and supported_ops."""
        tensor = Tensor((5,), (1,), "int64", "cuda", [])
        inputs = argmax_op.decompose(tensor)

        input_tensor = inputs[0]
        # dtype changes from int64 to float type
        assert input_tensor.dtype in ["float32", "bfloat16", "float16"]
        assert input_tensor.device == tensor.device
        assert input_tensor.supported_ops == tensor.supported_ops

    def test_decompose_inserted_dimension_size(self, argmax_op):
        """Test that inserted dimension has appropriate size."""
        tensor = Tensor((2, 3), (3, 1), "int64", "cuda", [])
        inputs = argmax_op.decompose(tensor)

        input_tensor = inputs[0]
        argmax_dim = tensor._argmax_dim

        # The inserted dimension should have size >= 2
        assert input_tensor.size[argmax_dim] >= 2
        assert input_tensor.size[argmax_dim] <= 5

    def test_codegen_scalar_reduction(self, argmax_op):
        """Test code generation for scalar reduction."""
        tensor = Tensor((), (), "int64", "cuda", [])
        tensor._argmax_dim = "all"

        output_name = "result"
        input_names = ["input_tensor"]

        code = argmax_op.codegen(output_name, input_names, tensor)
        expected = "result = input_tensor.argmax()"

        assert code == expected

    def test_codegen_single_dimension_reduction(self, argmax_op):
        """Test code generation for single dimension reduction."""
        tensor = Tensor((3, 4), (4, 1), "int64", "cuda", [])
        tensor._argmax_dim = 1

        output_name = "out"
        input_names = ["x"]

        code = argmax_op.codegen(output_name, input_names, tensor)
        expected = "out = x.argmax(dim=1)"

        assert code == expected

    def test_codegen_tuple_dimension_reduction(self, argmax_op):
        """Test code generation for tuple dimension reduction."""
        tensor = Tensor((2, 3), (3, 1), "int64", "cuda", [])
        tensor._argmax_dim = (0, 2)

        output_name = "output"
        input_names = ["data"]

        code = argmax_op.codegen(output_name, input_names, tensor)
        expected = "output = data.argmax(dim=(0, 2))"

        assert code == expected

    def test_codegen_no_argmax_dim_attribute(self, argmax_op):
        """Test code generation when _argmax_dim attribute is missing."""
        tensor = Tensor((2, 2), (2, 1), "int64", "cuda", [])
        # Don't set _argmax_dim attribute

        output_name = "result"
        input_names = ["input"]

        code = argmax_op.codegen(output_name, input_names, tensor)
        expected = "result = input.argmax()"

        assert code == expected

    def test_operator_name(self, argmax_op):
        """Test that operator has correct name."""
        assert argmax_op.name == "argmax"
