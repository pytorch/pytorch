"""Tests for SigmoidOperator."""

import pytest
from .sigmoid import SigmoidOperator
from torchfuzz.tensor import Tensor


class TestSigmoidOperator:
    """Test class for SigmoidOperator."""

    @pytest.fixture
    def sigmoid_op(self):
        """Create a SigmoidOperator instance."""
        return SigmoidOperator()

    @pytest.fixture
    def sample_tensor(self):
        """Create a sample tensor for testing."""
        return Tensor((2, 3), (3, 1), "float32", "cuda", [])

    def test_can_produce_returns_true(self, sigmoid_op, sample_tensor):
        """Test that SigmoidOperator can always produce any tensor."""
        assert sigmoid_op.can_produce(sample_tensor) is True

    def test_decompose_single_input(self, sigmoid_op, sample_tensor):
        """Test decomposition returns a single input tensor."""
        inputs = sigmoid_op.decompose(sample_tensor)

        assert len(inputs) == 1
        input_tensor = inputs[0]
        assert input_tensor.size == sample_tensor.size
        assert input_tensor.stride == sample_tensor.stride
        assert input_tensor.dtype == sample_tensor.dtype
        assert input_tensor.device == sample_tensor.device

    def test_decompose_preserves_dtype_and_device(self, sigmoid_op):
        """Test that decomposition preserves dtype and device."""
        for dtype in ["float32", "float16", "bfloat16"]:
            tensor = Tensor((3, 4), (4, 1), dtype, "cuda", [])
            inputs = sigmoid_op.decompose(tensor)

            assert len(inputs) == 1
            input_tensor = inputs[0]
            assert input_tensor.dtype == dtype
            assert input_tensor.device == "cuda"

    def test_decompose_preserves_shape_and_stride(self, sigmoid_op):
        """Test that decomposition preserves shape and stride."""
        shapes_and_strides = [
            ((2,), (1,)),
            ((3, 4), (4, 1)),
            ((2, 3, 4), (12, 4, 1)),
            ((), ()),  # scalar
        ]

        for size, stride in shapes_and_strides:
            tensor = Tensor(size, stride, "float32", "cuda", [])
            inputs = sigmoid_op.decompose(tensor)

            assert len(inputs) == 1
            input_tensor = inputs[0]
            assert input_tensor.size == size
            assert input_tensor.stride == stride

    def test_codegen(self, sigmoid_op, sample_tensor):
        """Test code generation for Sigmoid operation."""
        output_name = "out"
        input_names = ["x"]

        code = sigmoid_op.codegen(output_name, input_names, sample_tensor)
        expected = "out = torch.sigmoid(x)"

        assert code == expected

    def test_codegen_different_names(self, sigmoid_op, sample_tensor):
        """Test code generation with different variable names."""
        output_name = "result"
        input_names = ["input_tensor"]

        code = sigmoid_op.codegen(output_name, input_names, sample_tensor)
        expected = "result = torch.sigmoid(input_tensor)"

        assert code == expected

    def test_operator_name(self, sigmoid_op):
        """Test that operator has correct name."""
        assert sigmoid_op.name == "sigmoid"
