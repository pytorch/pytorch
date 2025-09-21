"""Tests for FillOperator_."""

import pytest
from .fill_ import FillOperator_
from torchfuzz.tensor import Tensor


class TestFillOperator_:
    """Test class for FillOperator_."""

    @pytest.fixture
    def fill_op(self):
        """Create a FillOperator_ instance."""
        return FillOperator_()

    def test_can_produce_any_tensor(self, fill_op):
        """Test that FillOperator_ can produce any tensor."""
        # Test various tensor shapes and types
        tensors = [
            Tensor((), (), "float32", "cuda", []),  # scalar
            Tensor((5,), (1,), "float32", "cuda", []),  # 1D
            Tensor((3, 4), (4, 1), "float32", "cuda", []),  # 2D
            Tensor((2, 3, 4), (12, 4, 1), "bfloat16", "cuda", []),  # 3D
            Tensor((2, 3, 4, 5), (60, 20, 5, 1), "float16", "cuda", []),  # 4D
        ]

        for tensor in tensors:
            assert fill_op.can_produce(tensor) is True

    def test_decompose_scalar_tensor(self, fill_op):
        """Test decomposition of scalar tensor."""
        tensor = Tensor((), (), "float32", "cuda", [])
        inputs = fill_op.decompose(tensor)

        assert len(inputs) == 2

        # First input: tensor to fill (same shape as output)
        tensor_input = inputs[0]
        assert tensor_input.size == tensor.size
        assert tensor_input.stride == tensor.stride
        assert tensor_input.dtype == tensor.dtype
        assert tensor_input.device == tensor.device

        # Second input: scalar value to fill
        value_input = inputs[1]
        assert value_input.size == ()
        assert value_input.stride == ()
        assert value_input.dtype == tensor.dtype
        assert value_input.device == tensor.device

    def test_decompose_1d_tensor(self, fill_op):
        """Test decomposition of 1D tensor."""
        tensor = Tensor((10,), (1,), "float32", "cuda", [])
        inputs = fill_op.decompose(tensor)

        assert len(inputs) == 2

        # Check tensor input properties
        tensor_input = inputs[0]
        assert tensor_input.size == tensor.size
        assert tensor_input.stride == tensor.stride
        assert tensor_input.dtype == tensor.dtype
        assert tensor_input.device == tensor.device

        # Check scalar input properties
        value_input = inputs[1]
        assert value_input.size == ()
        assert value_input.dtype == tensor.dtype
        assert value_input.device == tensor.device

    def test_decompose_2d_tensor(self, fill_op):
        """Test decomposition of 2D tensor."""
        tensor = Tensor((3, 4), (4, 1), "bfloat16", "cuda", [])
        inputs = fill_op.decompose(tensor)

        assert len(inputs) == 2

        # Check tensor input properties
        tensor_input = inputs[0]
        assert tensor_input.size == tensor.size
        assert tensor_input.stride == tensor.stride
        assert tensor_input.dtype == tensor.dtype
        assert tensor_input.device == tensor.device

        # Check scalar input properties
        value_input = inputs[1]
        assert value_input.size == ()
        assert value_input.dtype == tensor.dtype
        assert value_input.device == tensor.device

    def test_decompose_preserves_properties(self, fill_op):
        """Test that decomposition preserves tensor properties."""
        tensor = Tensor((5, 3), (3, 1), "float16", "cpu", ["test_ops"])
        inputs = fill_op.decompose(tensor)

        for input_tensor in inputs:
            assert input_tensor.dtype == tensor.dtype
            assert input_tensor.device == tensor.device
            assert input_tensor.supported_ops == tensor.supported_ops

    def test_codegen_basic(self, fill_op):
        """Test basic code generation."""
        tensor = Tensor((3, 3), (3, 1), "float32", "cuda", [])

        output_name = "result"
        input_names = ["tensor", "value"]

        code = fill_op.codegen(output_name, input_names, tensor)
        expected = "result = tensor.clone(); result.fill_(value.item())"

        assert code == expected

    def test_codegen_different_names(self, fill_op):
        """Test code generation with different variable names."""
        tensor = Tensor((4, 5), (5, 1), "float32", "cuda", [])

        output_name = "output_tensor"
        input_names = ["input_data", "fill_val"]

        code = fill_op.codegen(output_name, input_names, tensor)
        expected = "output_tensor = input_data.clone(); output_tensor.fill_(fill_val.item())"

        assert code == expected

    def test_codegen_extracts_scalar_value(self, fill_op):
        """Test that code generation extracts scalar value with .item()."""
        tensor = Tensor((2, 2), (2, 1), "float32", "cuda", [])

        output_name = "out"
        input_names = ["tensor_in", "scalar_tensor"]

        code = fill_op.codegen(output_name, input_names, tensor)

        # Should use .item() to extract the scalar value
        assert "scalar_tensor.item()" in code
        assert "out = tensor_in.clone()" in code
        assert "out.fill_" in code

    def test_codegen_clones_input(self, fill_op):
        """Test that code generation clones the input tensor."""
        tensor = Tensor((5, 5), (5, 1), "float32", "cuda", [])

        output_name = "result"
        input_names = ["original", "fill_value"]

        code = fill_op.codegen(output_name, input_names, tensor)

        # Should clone the input to avoid modifying original
        assert "original.clone()" in code

    def test_operator_name(self, fill_op):
        """Test that operator has correct name."""
        assert fill_op.name == "fill_"
