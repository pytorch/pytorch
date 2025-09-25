"""Tests for ZeroOperator_."""

import pytest
from .zero_ import ZeroOperator_
from torchfuzz.tensor import Tensor


class TestZeroOperator_:
    """Test class for ZeroOperator_."""

    @pytest.fixture
    def zero_op(self):
        """Create a ZeroOperator_ instance."""
        return ZeroOperator_()

    def test_can_produce_any_tensor(self, zero_op):
        """Test that ZeroOperator_ can produce any tensor."""
        # Test various tensor shapes and types
        tensors = [
            Tensor((), (), "float32", "cuda", []),  # scalar
            Tensor((5,), (1,), "float32", "cuda", []),  # 1D
            Tensor((3, 4), (4, 1), "float32", "cuda", []),  # 2D
            Tensor((2, 3, 4), (12, 4, 1), "bfloat16", "cuda", []),  # 3D
            Tensor((2, 3, 4, 5), (60, 20, 5, 1), "float16", "cuda", []),  # 4D
        ]

        for tensor in tensors:
            assert zero_op.can_produce(tensor) is True

    def test_decompose_scalar_tensor(self, zero_op):
        """Test decomposition of scalar tensor."""
        tensor = Tensor((), (), "float32", "cuda", [])
        inputs = zero_op.decompose(tensor)

        assert len(inputs) == 1

        # First input: tensor to zero (same shape as output)
        tensor_input = inputs[0]
        assert tensor_input.size == tensor.size
        assert tensor_input.stride == tensor.stride
        assert tensor_input.dtype == tensor.dtype
        assert tensor_input.device == tensor.device

    def test_decompose_1d_tensor(self, zero_op):
        """Test decomposition of 1D tensor."""
        tensor = Tensor((10,), (1,), "float32", "cuda", [])
        inputs = zero_op.decompose(tensor)

        assert len(inputs) == 1

        # Check tensor input properties
        tensor_input = inputs[0]
        assert tensor_input.size == tensor.size
        assert tensor_input.stride == tensor.stride
        assert tensor_input.dtype == tensor.dtype
        assert tensor_input.device == tensor.device

    def test_decompose_2d_tensor(self, zero_op):
        """Test decomposition of 2D tensor."""
        tensor = Tensor((3, 4), (4, 1), "bfloat16", "cuda", [])
        inputs = zero_op.decompose(tensor)

        assert len(inputs) == 1

        # Check tensor input properties
        tensor_input = inputs[0]
        assert tensor_input.size == tensor.size
        assert tensor_input.stride == tensor.stride
        assert tensor_input.dtype == tensor.dtype
        assert tensor_input.device == tensor.device

    def test_decompose_preserves_properties(self, zero_op):
        """Test that decomposition preserves tensor properties."""
        tensor = Tensor((5, 3), (3, 1), "float16", "cpu", ["test_ops"])
        inputs = zero_op.decompose(tensor)

        for input_tensor in inputs:
            assert input_tensor.dtype == tensor.dtype
            assert input_tensor.device == tensor.device
            assert input_tensor.supported_ops == tensor.supported_ops

    def test_codegen_basic(self, zero_op):
        """Test basic code generation."""
        tensor = Tensor((3, 3), (3, 1), "float32", "cuda", [])

        output_name = "result"
        input_names = ["tensor"]

        code = zero_op.codegen(output_name, input_names, tensor)
        expected = "result = tensor.clone(); result.zero_()"

        assert code == expected

    def test_codegen_different_names(self, zero_op):
        """Test code generation with different variable names."""
        tensor = Tensor((4, 5), (5, 1), "float32", "cuda", [])

        output_name = "output_tensor"
        input_names = ["input_data"]

        code = zero_op.codegen(output_name, input_names, tensor)
        expected = "output_tensor = input_data.clone(); output_tensor.zero_()"

        assert code == expected

    def test_codegen_clones_input(self, zero_op):
        """Test that code generation clones the input tensor."""
        tensor = Tensor((5, 5), (5, 1), "float32", "cuda", [])

        output_name = "result"
        input_names = ["original"]

        code = zero_op.codegen(output_name, input_names, tensor)

        # Should clone the input to avoid modifying original
        assert "original.clone()" in code
        assert "zero_()" in code

    def test_operator_name(self, zero_op):
        """Test that operator has correct name."""
        assert zero_op.name == "zero_"
