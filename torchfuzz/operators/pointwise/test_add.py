"""Tests for AddOperator."""

import pytest
from .add import AddOperator
from torchfuzz.tensor import Tensor


class TestAddOperator:
    """Test class for AddOperator."""

    @pytest.fixture
    def add_op(self):
        """Create an AddOperator instance."""
        return AddOperator()

    @pytest.fixture
    def sample_tensor(self):
        """Create a sample tensor for testing."""
        return Tensor((2, 3), (3, 1), "float32", "cuda", [])

    def test_can_produce_returns_true(self, add_op, sample_tensor):
        """Test that AddOperator can always produce any tensor."""
        assert add_op.can_produce(sample_tensor) is True

    def test_decompose_default_inputs(self, add_op, sample_tensor):
        """Test decomposition with default number of inputs (2)."""
        inputs = add_op.decompose(sample_tensor)

        assert len(inputs) == 2
        for input_tensor in inputs:
            assert input_tensor.size == sample_tensor.size
            assert input_tensor.stride == sample_tensor.stride
            assert input_tensor.device == sample_tensor.device

    def test_decompose_multiple_inputs(self, add_op, sample_tensor):
        """Test decomposition with multiple inputs."""
        num_inputs = 4
        inputs = add_op.decompose(sample_tensor, num_inputs=num_inputs)

        assert len(inputs) == num_inputs
        for input_tensor in inputs:
            assert input_tensor.size == sample_tensor.size
            assert input_tensor.stride == sample_tensor.stride
            assert input_tensor.device == sample_tensor.device

    def test_decompose_type_promotion_float32(self, add_op):
        """Test type promotion for float32 output."""
        tensor = Tensor((2, 2), (2, 1), "float32", "cuda", [])
        inputs = add_op.decompose(tensor, num_inputs=2)

        # Check that dtypes follow promotion rules
        valid_dtypes = ["float32", "bfloat16", "float16"]
        for input_tensor in inputs:
            assert input_tensor.dtype in valid_dtypes

    def test_decompose_type_promotion_bfloat16(self, add_op):
        """Test type promotion for bfloat16 output."""
        tensor = Tensor((2, 2), (2, 1), "bfloat16", "cuda", [])
        inputs = add_op.decompose(tensor, num_inputs=2)

        # Check that dtypes follow promotion rules
        valid_dtypes = ["bfloat16", "float16"]
        for input_tensor in inputs:
            assert input_tensor.dtype in valid_dtypes

    def test_decompose_type_promotion_float16(self, add_op):
        """Test type promotion for float16 output."""
        tensor = Tensor((2, 2), (2, 1), "float16", "cuda", [])
        inputs = add_op.decompose(tensor, num_inputs=2)

        # For float16, all inputs should be float16
        for input_tensor in inputs:
            assert input_tensor.dtype == "float16"

    def test_codegen_two_inputs(self, add_op, sample_tensor):
        """Test code generation with two inputs."""
        output_name = "out"
        input_names = ["a", "b"]

        code = add_op.codegen(output_name, input_names, sample_tensor)
        expected = "out = a + b"

        assert code == expected

    def test_codegen_multiple_inputs(self, add_op, sample_tensor):
        """Test code generation with multiple inputs."""
        output_name = "result"
        input_names = ["x", "y", "z", "w"]

        code = add_op.codegen(output_name, input_names, sample_tensor)
        expected = "result = x + y + z + w"

        assert code == expected

    def test_operator_name(self, add_op):
        """Test that operator has correct name."""
        assert add_op.name == "add"
