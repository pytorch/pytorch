"""Tests for ExpOperator."""

import pytest
from .exp import ExpOperator
from torchfuzz.tensor import Tensor


class TestExpOperator:
    """Test class for ExpOperator."""

    @pytest.fixture
    def exp_op(self):
        """Create an ExpOperator instance."""
        return ExpOperator()

    @pytest.fixture
    def sample_tensor(self):
        """Create a sample tensor for testing."""
        return Tensor((2, 3), (3, 1), "float32", "cuda", [])

    def test_can_produce_returns_true(self, exp_op, sample_tensor):
        """Test that ExpOperator can always produce any tensor."""
        assert exp_op.can_produce(sample_tensor) is True

    def test_decompose_single_input(self, exp_op, sample_tensor):
        """Test decomposition returns a single input tensor."""
        inputs = exp_op.decompose(sample_tensor)

        assert len(inputs) == 1
        input_tensor = inputs[0]
        assert input_tensor.size == sample_tensor.size
        assert input_tensor.stride == sample_tensor.stride
        assert input_tensor.dtype == sample_tensor.dtype
        assert input_tensor.device == sample_tensor.device

    def test_codegen(self, exp_op, sample_tensor):
        """Test code generation for Exp operation."""
        output_name = "out"
        input_names = ["x"]

        code = exp_op.codegen(output_name, input_names, sample_tensor)
        expected = "out = torch.exp(x)"

        assert code == expected

    def test_operator_name(self, exp_op):
        """Test that operator has correct name."""
        assert exp_op.name == "exp"
