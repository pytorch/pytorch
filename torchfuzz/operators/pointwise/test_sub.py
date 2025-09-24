"""Tests for SubOperator."""

import pytest
from .sub import SubOperator
from torchfuzz.tensor import Tensor


class TestSubOperator:
    """Test class for SubOperator."""

    @pytest.fixture
    def sub_op(self):
        """Create a SubOperator instance."""
        return SubOperator()

    @pytest.fixture
    def sample_tensor(self):
        """Create a sample tensor for testing."""
        return Tensor((2, 3), (3, 1), "float32", "cuda", [])

    def test_can_produce_returns_true(self, sub_op, sample_tensor):
        """Test that SubOperator can always produce any tensor."""
        assert sub_op.can_produce(sample_tensor) is True

    def test_decompose_default_inputs(self, sub_op, sample_tensor):
        """Test decomposition with default number of inputs (2)."""
        inputs = sub_op.decompose(sample_tensor)

        assert len(inputs) == 2
        for input_tensor in inputs:
            assert input_tensor.size == sample_tensor.size
            assert input_tensor.stride == sample_tensor.stride
            assert input_tensor.device == sample_tensor.device

    def test_codegen_two_inputs(self, sub_op, sample_tensor):
        """Test code generation with two inputs."""
        output_name = "out"
        input_names = ["a", "b"]

        code = sub_op.codegen(output_name, input_names, sample_tensor)
        expected = "out = (a) - b"

        assert code == expected

    def test_codegen_single_input(self, sub_op, sample_tensor):
        """Test code generation with single input."""
        output_name = "out"
        input_names = ["a"]

        code = sub_op.codegen(output_name, input_names, sample_tensor)
        expected = "out = -a"

        assert code == expected

    def test_codegen_multiple_inputs(self, sub_op, sample_tensor):
        """Test code generation with multiple inputs."""
        output_name = "result"
        input_names = ["x", "y", "z"]

        code = sub_op.codegen(output_name, input_names, sample_tensor)
        expected = "result = ((x) - y) - z"

        assert code == expected

    def test_operator_name(self, sub_op):
        """Test that operator has correct name."""
        assert sub_op.name == "sub"
