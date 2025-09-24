"""Tests for FillDiagonalOperator_."""

import pytest
from .fill_diagonal_ import FillDiagonalOperator_
from torchfuzz.tensor import Tensor


class TestFillDiagonalOperator_:
    """Test class for FillDiagonalOperator_."""

    @pytest.fixture
    def fill_diag_op(self):
        """Create a FillDiagonalOperator_ instance."""
        return FillDiagonalOperator_()

    def test_can_produce_square_tensor(self, fill_diag_op):
        """Test that FillDiagonalOperator_ can produce square tensors."""
        square_tensor = Tensor((3, 3), (3, 1), "float32", "cuda", [])
        assert fill_diag_op.can_produce(square_tensor) is True

    def test_can_produce_equal_dimensions_tensor(self, fill_diag_op):
        """Test that FillDiagonalOperator_ can produce tensors with all equal dimensions."""
        tensor_3d = Tensor((4, 4, 4), (16, 4, 1), "float32", "cuda", [])
        assert fill_diag_op.can_produce(tensor_3d) is True

        tensor_4d = Tensor((2, 2, 2, 2), (8, 4, 2, 1), "float32", "cuda", [])
        assert fill_diag_op.can_produce(tensor_4d) is True

    def test_cannot_produce_1d_tensor(self, fill_diag_op):
        """Test that FillDiagonalOperator_ cannot produce 1D tensors."""
        tensor_1d = Tensor((5,), (1,), "float32", "cuda", [])
        assert fill_diag_op.can_produce(tensor_1d) is False

    def test_cannot_produce_scalar_tensor(self, fill_diag_op):
        """Test that FillDiagonalOperator_ cannot produce scalar tensors."""
        scalar = Tensor((), (), "float32", "cuda", [])
        assert fill_diag_op.can_produce(scalar) is False

    def test_cannot_produce_unequal_dimensions(self, fill_diag_op):
        """Test that FillDiagonalOperator_ cannot produce tensors with unequal dimensions."""
        tensor_2d = Tensor((3, 4), (4, 1), "float32", "cuda", [])
        assert fill_diag_op.can_produce(tensor_2d) is False

        tensor_3d = Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", [])
        assert fill_diag_op.can_produce(tensor_3d) is False

    def test_can_produce_mixed_equal_dimensions(self, fill_diag_op):
        """Test edge case with some equal dimensions but not all."""
        # This should return False since not ALL dimensions are equal
        tensor = Tensor((3, 3, 4), (12, 4, 1), "float32", "cuda", [])
        assert fill_diag_op.can_produce(tensor) is False

    def test_decompose_square_tensor(self, fill_diag_op):
        """Test decomposition of square tensor."""
        tensor = Tensor((3, 3), (3, 1), "float32", "cuda", [])
        inputs = fill_diag_op.decompose(tensor)

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

    def test_decompose_cube_tensor(self, fill_diag_op):
        """Test decomposition of 3D cube tensor."""
        tensor = Tensor((4, 4, 4), (16, 4, 1), "float32", "cuda", [])
        inputs = fill_diag_op.decompose(tensor)

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

    def test_decompose_stores_dimensions(self, fill_diag_op):
        """Test that decomposition stores dimension information."""
        tensor = Tensor((2, 2), (2, 1), "float32", "cuda", [])
        inputs = fill_diag_op.decompose(tensor)

        # Should store _fill_diag_dims attribute
        assert hasattr(tensor, '_fill_diag_dims')
        assert tensor._fill_diag_dims is not None

        # Should be a tuple of two dimension indices
        dims = tensor._fill_diag_dims
        assert isinstance(dims, tuple)
        assert len(dims) == 2
        assert all(isinstance(d, int) for d in dims)
        assert all(0 <= d < len(tensor.size) for d in dims)

    def test_decompose_preserves_properties(self, fill_diag_op):
        """Test that decomposition preserves tensor properties."""
        tensor = Tensor((5, 5), (5, 1), "bfloat16", "cuda", ["test_ops"])
        inputs = fill_diag_op.decompose(tensor)

        for input_tensor in inputs:
            assert input_tensor.dtype == tensor.dtype
            assert input_tensor.device == tensor.device
            assert input_tensor.supported_ops == tensor.supported_ops

    def test_decompose_fallback_case(self, fill_diag_op):
        """Test decomposition fallback when no equal dimensions found."""
        # This test case is a bit artificial since the operator should only
        # be called on tensors that can_produce returns True for
        tensor = Tensor((2, 3), (3, 1), "float32", "cuda", [])

        # Manually call decompose to test fallback behavior
        inputs = fill_diag_op.decompose(tensor)

        # Should still return two inputs in fallback
        assert len(inputs) == 2

        # First should be same shape as output
        assert inputs[0].size == tensor.size
        # Second should be scalar
        assert inputs[1].size == ()

    def test_codegen_basic(self, fill_diag_op):
        """Test basic code generation."""
        tensor = Tensor((3, 3), (3, 1), "float32", "cuda", [])
        tensor._fill_diag_dims = (0, 1)

        output_name = "result"
        input_names = ["matrix", "value"]

        code = fill_diag_op.codegen(output_name, input_names, tensor)
        expected = "result = matrix.clone(); result.fill_diagonal_(value.item())"

        assert code == expected

    def test_codegen_different_names(self, fill_diag_op):
        """Test code generation with different variable names."""
        tensor = Tensor((4, 4, 4), (16, 4, 1), "float32", "cuda", [])
        tensor._fill_diag_dims = (1, 2)

        output_name = "output_tensor"
        input_names = ["input_data", "fill_val"]

        code = fill_diag_op.codegen(output_name, input_names, tensor)
        expected = "output_tensor = input_data.clone(); output_tensor.fill_diagonal_(fill_val.item())"

        assert code == expected

    def test_codegen_extracts_scalar_value(self, fill_diag_op):
        """Test that code generation extracts scalar value with .item()."""
        tensor = Tensor((2, 2), (2, 1), "float32", "cuda", [])

        output_name = "out"
        input_names = ["tensor_in", "scalar_tensor"]

        code = fill_diag_op.codegen(output_name, input_names, tensor)

        # Should use .item() to extract the scalar value
        assert "scalar_tensor.item()" in code
        assert "out = tensor_in.clone()" in code
        assert "out.fill_diagonal_" in code

    def test_codegen_clones_input(self, fill_diag_op):
        """Test that code generation clones the input tensor."""
        tensor = Tensor((5, 5), (5, 1), "float32", "cuda", [])

        output_name = "result"
        input_names = ["original", "diag_value"]

        code = fill_diag_op.codegen(output_name, input_names, tensor)

        # Should clone the input to avoid modifying original
        assert "original.clone()" in code

    def test_operator_name(self, fill_diag_op):
        """Test that operator has correct name."""
        assert fill_diag_op.name == "fill_diagonal_"
