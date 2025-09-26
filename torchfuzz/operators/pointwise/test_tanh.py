"""Tests for TanhOperator."""

import pytest
from .tanh import TanhOperator
from torchfuzz.tensor import Tensor


class TestTanhOperator:
    """Test class for TanhOperator."""

    @pytest.fixture
    def tanh_op(self):
        """Create a TanhOperator instance."""
        return TanhOperator()

    @pytest.fixture
    def sample_tensor(self):
        """Create a sample tensor for testing."""
        return Tensor((2, 3), (3, 1), "float32", "cuda", [])

    def test_can_produce_returns_true(self, tanh_op, sample_tensor):
        """Test that TanhOperator can always produce any tensor."""
        assert tanh_op.can_produce(sample_tensor) is True

    def test_decompose_single_input(self, tanh_op, sample_tensor):
        """Test decomposition returns a single input tensor."""
        inputs = tanh_op.decompose(sample_tensor)

        assert len(inputs) == 1
        input_tensor = inputs[0]
        assert input_tensor.size == sample_tensor.size
        assert input_tensor.stride == sample_tensor.stride
        assert input_tensor.dtype == sample_tensor.dtype
        assert input_tensor.device == sample_tensor.device

    def test_decompose_preserves_dtype_and_device(self, tanh_op):
        """Test that decomposition preserves dtype and device."""
        for dtype in ["float32", "float16", "bfloat16"]:
            tensor = Tensor((3, 4), (4, 1), dtype, "cuda", [])
            inputs = tanh_op.decompose(tensor)

            assert len(inputs) == 1
            input_tensor = inputs[0]
            assert input_tensor.dtype == dtype
            assert input_tensor.device == "cuda"

    def test_decompose_preserves_shape_and_stride(self, tanh_op):
        """Test that decomposition preserves shape and stride."""
        shapes_and_strides = [
            ((2,), (1,)),
            ((3, 4), (4, 1)),
            ((2, 3, 4), (12, 4, 1)),
            ((), ()),  # scalar
        ]

        for size, stride in shapes_and_strides:
            tensor = Tensor(size, stride, "float32", "cuda", [])
            inputs = tanh_op.decompose(tensor)

            assert len(inputs) == 1
            input_tensor = inputs[0]
            assert input_tensor.size == size
            assert input_tensor.stride == stride

    def test_codegen(self, tanh_op, sample_tensor):
        """Test code generation for Tanh operation."""
        output_name = "out"
        input_names = ["x"]

        code = tanh_op.codegen(output_name, input_names, sample_tensor)
        expected = "out = torch.tanh(x)"

        assert code == expected

    def test_codegen_different_names(self, tanh_op, sample_tensor):
        """Test code generation with different variable names."""
        output_name = "result"
        input_names = ["input_tensor"]

        code = tanh_op.codegen(output_name, input_names, sample_tensor)
        expected = "result = torch.tanh(input_tensor)"

        assert code == expected

    def test_operator_name(self, tanh_op):
        """Test that operator has correct name."""
        assert tanh_op.name == "tanh"
