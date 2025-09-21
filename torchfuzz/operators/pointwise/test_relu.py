"""Tests for ReluOperator."""

import pytest
from .relu import ReluOperator
from torchfuzz.tensor import Tensor


class TestReluOperator:
    """Test class for ReluOperator."""

    @pytest.fixture
    def relu_op(self):
        """Create a ReluOperator instance."""
        return ReluOperator()

    @pytest.fixture
    def sample_tensor(self):
        """Create a sample tensor for testing."""
        return Tensor((2, 3), (3, 1), "float32", "cuda", [])

    def test_can_produce_returns_true(self, relu_op, sample_tensor):
        """Test that ReluOperator can always produce any tensor."""
        assert relu_op.can_produce(sample_tensor) is True

    def test_decompose_single_input(self, relu_op, sample_tensor):
        """Test decomposition returns a single input tensor."""
        inputs = relu_op.decompose(sample_tensor)

        assert len(inputs) == 1
        input_tensor = inputs[0]
        assert input_tensor.size == sample_tensor.size
        assert input_tensor.stride == sample_tensor.stride
        assert input_tensor.dtype == sample_tensor.dtype
        assert input_tensor.device == sample_tensor.device

    def test_codegen(self, relu_op, sample_tensor):
        """Test code generation for ReLU operation."""
        output_name = "out"
        input_names = ["x"]

        code = relu_op.codegen(output_name, input_names, sample_tensor)
        expected = "out = torch.nn.functional.relu(x)"

        assert code == expected

    def test_operator_name(self, relu_op):
        """Test that operator has correct name."""
        assert relu_op.name == "relu"
