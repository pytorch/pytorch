"""Tests for SqrtOperator."""

import pytest
from .sqrt import SqrtOperator
from torchfuzz.tensor import Tensor


class TestSqrtOperator:
    """Test class for SqrtOperator."""

    @pytest.fixture
    def sqrt_op(self):
        """Create a SqrtOperator instance."""
        return SqrtOperator()

    @pytest.fixture
    def sample_tensor(self):
        """Create a sample tensor for testing."""
        return Tensor((2, 3), (3, 1), "float32", "cuda", [])

    def test_can_produce_returns_true(self, sqrt_op, sample_tensor):
        """Test that SqrtOperator can always produce any tensor."""
        assert sqrt_op.can_produce(sample_tensor) is True

    def test_decompose_single_input(self, sqrt_op, sample_tensor):
        """Test decomposition returns a single input tensor."""
        inputs = sqrt_op.decompose(sample_tensor)

        assert len(inputs) == 1
        input_tensor = inputs[0]
        assert input_tensor.size == sample_tensor.size
        assert input_tensor.stride == sample_tensor.stride
        assert input_tensor.dtype == sample_tensor.dtype
        assert input_tensor.device == sample_tensor.device

    def test_decompose_preserves_dtype_and_device(self, sqrt_op):
        """Test that decomposition preserves dtype and device."""
        for dtype in ["float32", "float16", "bfloat16"]:
            tensor = Tensor((3, 4), (4, 1), dtype, "cuda", [])
            inputs = sqrt_op.decompose(tensor)

            assert len(inputs) == 1
            input_tensor = inputs[0]
            assert input_tensor.dtype == dtype
            assert input_tensor.device == "cuda"

    def test_decompose_preserves_shape_and_stride(self, sqrt_op):
        """Test that decomposition preserves shape and stride."""
        shapes_and_strides = [
            ((2,), (1,)),
            ((3, 4), (4, 1)),
            ((2, 3, 4), (12, 4, 1)),
            ((), ()),  # scalar
        ]

        for size, stride in shapes_and_strides:
            tensor = Tensor(size, stride, "float32", "cuda", [])
            inputs = sqrt_op.decompose(tensor)

            assert len(inputs) == 1
            input_tensor = inputs[0]
            assert input_tensor.size == size
            assert input_tensor.stride == stride

    def test_codegen(self, sqrt_op, sample_tensor):
        """Test code generation for Sqrt operation."""
        output_name = "out"
        input_names = ["x"]

        code = sqrt_op.codegen(output_name, input_names, sample_tensor)
        expected = "out = torch.sqrt(x)"

        assert code == expected

    def test_codegen_different_names(self, sqrt_op, sample_tensor):
        """Test code generation with different variable names."""
        output_name = "result"
        input_names = ["input_tensor"]

        code = sqrt_op.codegen(output_name, input_names, sample_tensor)
        expected = "result = torch.sqrt(input_tensor)"

        assert code == expected

    def test_operator_name(self, sqrt_op):
        """Test that operator has correct name."""
        assert sqrt_op.name == "sqrt"
