"""Tests for PowOperator."""

import pytest
from .pow import PowOperator
from torchfuzz.tensor import Tensor


class TestPowOperator:
    """Test class for PowOperator."""

    @pytest.fixture
    def pow_op(self):
        """Create a PowOperator instance."""
        return PowOperator()

    @pytest.fixture
    def sample_tensor(self):
        """Create a sample tensor for testing."""
        return Tensor((2, 3), (3, 1), "float32", "cuda", [])

    def test_can_produce_returns_true(self, pow_op, sample_tensor):
        """Test that PowOperator can always produce any tensor."""
        assert pow_op.can_produce(sample_tensor) is True

    def test_decompose_default_inputs(self, pow_op, sample_tensor):
        """Test decomposition with default number of inputs (2)."""
        inputs = pow_op.decompose(sample_tensor)

        assert len(inputs) == 2
        for input_tensor in inputs:
            assert input_tensor.size == sample_tensor.size
            assert input_tensor.stride == sample_tensor.stride
            assert input_tensor.device == sample_tensor.device

    def test_decompose_multiple_inputs(self, pow_op, sample_tensor):
        """Test decomposition with multiple inputs."""
        num_inputs = 4
        inputs = pow_op.decompose(sample_tensor, num_inputs=num_inputs)

        assert len(inputs) == num_inputs
        for input_tensor in inputs:
            assert input_tensor.size == sample_tensor.size
            assert input_tensor.stride == sample_tensor.stride
            assert input_tensor.device == sample_tensor.device

    def test_decompose_single_input(self, pow_op, sample_tensor):
        """Test decomposition with single input (square)."""
        inputs = pow_op.decompose(sample_tensor, num_inputs=1)

        assert len(inputs) == 1
        input_tensor = inputs[0]
        assert input_tensor.size == sample_tensor.size
        assert input_tensor.stride == sample_tensor.stride
        assert input_tensor.device == sample_tensor.device

    def test_decompose_type_promotion_float32(self, pow_op):
        """Test type promotion for float32 output."""
        tensor = Tensor((2, 2), (2, 1), "float32", "cuda", [])
        inputs = pow_op.decompose(tensor, num_inputs=2)

        # Check that dtypes follow promotion rules
        valid_dtypes = ["float32", "bfloat16", "float16"]
        for input_tensor in inputs:
            assert input_tensor.dtype in valid_dtypes

    def test_decompose_type_promotion_bfloat16(self, pow_op):
        """Test type promotion for bfloat16 output."""
        tensor = Tensor((2, 2), (2, 1), "bfloat16", "cuda", [])
        inputs = pow_op.decompose(tensor, num_inputs=2)

        # Check that dtypes follow promotion rules
        valid_dtypes = ["bfloat16", "float16"]
        for input_tensor in inputs:
            assert input_tensor.dtype in valid_dtypes

    def test_decompose_type_promotion_float16(self, pow_op):
        """Test type promotion for float16 output."""
        tensor = Tensor((2, 2), (2, 1), "float16", "cuda", [])
        inputs = pow_op.decompose(tensor, num_inputs=2)

        # For float16, all inputs should be float16
        for input_tensor in inputs:
            assert input_tensor.dtype == "float16"

    def test_codegen_single_input_square(self, pow_op, sample_tensor):
        """Test code generation with single input (square)."""
        output_name = "out"
        input_names = ["a"]

        code = pow_op.codegen(output_name, input_names, sample_tensor)
        expected = "out = torch.pow(a, 2.0)"

        assert code == expected

    def test_codegen_two_inputs(self, pow_op, sample_tensor):
        """Test code generation with two inputs."""
        output_name = "out"
        input_names = ["a", "b"]

        code = pow_op.codegen(output_name, input_names, sample_tensor)
        expected = "out = torch.pow(a, b)"

        assert code == expected

    def test_codegen_multiple_inputs(self, pow_op, sample_tensor):
        """Test code generation with multiple inputs (left-to-right chaining)."""
        output_name = "result"
        input_names = ["x", "y", "z", "w"]

        code = pow_op.codegen(output_name, input_names, sample_tensor)
        expected = "result = torch.pow(torch.pow(torch.pow(x, y), z), w)"

        assert code == expected

    def test_operator_name(self, pow_op):
        """Test that operator has correct name."""
        assert pow_op.name == "pow"
