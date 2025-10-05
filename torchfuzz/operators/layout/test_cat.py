"""Tests for CatOperator."""

import pytest
from .cat import CatOperator
from torchfuzz.tensor import Tensor


class TestCatOperator:
    """Test class for CatOperator."""

    @pytest.fixture
    def cat_op(self):
        """Create a CatOperator instance."""
        return CatOperator()

    def test_can_produce_with_splittable_dimensions(self, cat_op):
        """Test that CatOperator can produce tensors with dimensions >= 2."""
        tensor = Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", [])
        assert cat_op.can_produce(tensor) is True

    def test_can_produce_with_single_splittable_dimension(self, cat_op):
        """Test that CatOperator can produce tensors with at least one dimension >= 2."""
        tensor = Tensor((1, 2, 1), (2, 1, 1), "float32", "cuda", [])
        assert cat_op.can_produce(tensor) is True

    def test_cannot_produce_with_no_splittable_dimensions(self, cat_op):
        """Test that CatOperator cannot produce tensors with all dimensions < 2."""
        tensor = Tensor((1, 1, 1), (1, 1, 1), "float32", "cuda", [])
        assert cat_op.can_produce(tensor) is False

    def test_cannot_produce_scalar_tensor(self, cat_op):
        """Test that CatOperator cannot produce scalar tensors."""
        scalar = Tensor((), (), "float32", "cuda", [])
        assert cat_op.can_produce(scalar) is False

    def test_decompose_default_inputs(self, cat_op):
        """Test decomposition with default number of inputs (2)."""
        tensor = Tensor((4, 3), (3, 1), "float32", "cuda", [])
        inputs = cat_op.decompose(tensor)

        assert len(inputs) == 2

        # Check that concatenation dimension was stored
        assert hasattr(tensor, '_cat_dim')
        assert hasattr(tensor, '_cat_sizes')
        cat_dim = tensor._cat_dim

        # Verify that all inputs have the same shape except for the cat dimension
        for i, input_tensor in enumerate(inputs):
            assert input_tensor.dtype == tensor.dtype
            assert input_tensor.device == tensor.device
            assert len(input_tensor.size) == len(tensor.size)

            # Check non-cat dimensions match
            for j, (input_size, output_size) in enumerate(zip(input_tensor.size, tensor.size)):
                if j != cat_dim:
                    assert input_size == output_size

        # Verify that cat dimension sizes sum to output size
        total_cat_size = sum(inp.size[cat_dim] for inp in inputs)
        assert total_cat_size == tensor.size[cat_dim]

    def test_decompose_multiple_inputs(self, cat_op):
        """Test decomposition with multiple inputs."""
        tensor = Tensor((6, 4), (4, 1), "float32", "cuda", [])
        num_inputs = 3
        inputs = cat_op.decompose(tensor, num_inputs=num_inputs)

        assert len(inputs) == num_inputs
        cat_dim = tensor._cat_dim

        # Verify that cat dimension sizes sum to output size
        total_cat_size = sum(inp.size[cat_dim] for inp in inputs)
        assert total_cat_size == tensor.size[cat_dim]

        # Each input should have at least size 1 in cat dimension
        for input_tensor in inputs:
            assert input_tensor.size[cat_dim] >= 1

    def test_decompose_more_inputs_than_cat_dimension_size(self, cat_op):
        """Test decomposition when num_inputs > cat dimension size."""
        tensor = Tensor((3, 5), (5, 1), "float32", "cuda", [])
        num_inputs = 5  # More than any single dimension
        inputs = cat_op.decompose(tensor, num_inputs=num_inputs)

        # Should limit to the size of the chosen cat dimension
        cat_dim = tensor._cat_dim
        expected_inputs = tensor.size[cat_dim]
        assert len(inputs) <= expected_inputs

    def test_decompose_no_splittable_dimensions_fallback(self, cat_op):
        """Test decomposition fallback when no dimensions are splittable."""
        tensor = Tensor((1, 1), (1, 1), "float32", "cuda", [])
        inputs = cat_op.decompose(tensor, num_inputs=3)

        # Should return identical tensors as fallback
        assert len(inputs) == 3
        for input_tensor in inputs:
            assert input_tensor.size == tensor.size
            assert input_tensor.stride == tensor.stride
            assert input_tensor.dtype == tensor.dtype
            assert input_tensor.device == tensor.device

    def test_decompose_preserves_properties(self, cat_op):
        """Test that decomposition preserves tensor properties."""
        tensor = Tensor((4, 6), (6, 1), "bfloat16", "cuda", ["test_ops"])
        inputs = cat_op.decompose(tensor)

        for input_tensor in inputs:
            assert input_tensor.dtype == tensor.dtype
            assert input_tensor.device == tensor.device
            assert input_tensor.supported_ops == tensor.supported_ops

    def test_decompose_stride_calculation(self, cat_op):
        """Test that input tensor strides are calculated correctly."""
        tensor = Tensor((4, 3), (3, 1), "float32", "cuda", [])
        inputs = cat_op.decompose(tensor)

        # Each input should have contiguous strides
        for input_tensor in inputs:
            size = input_tensor.size
            stride = input_tensor.stride

            # Verify contiguous stride calculation
            expected_stride = []
            acc = 1
            for s in reversed(size):
                expected_stride.insert(0, acc)
                acc *= s

            assert stride == tuple(expected_stride)

    def test_codegen_with_cat_dim_attribute(self, cat_op):
        """Test code generation when _cat_dim attribute is set."""
        tensor = Tensor((4, 3), (3, 1), "float32", "cuda", [])
        tensor._cat_dim = 1

        output_name = "result"
        input_names = ["a", "b", "c"]

        code = cat_op.codegen(output_name, input_names, tensor)
        expected = "result = torch.cat([a, b, c], dim=1)"

        assert code == expected

    def test_codegen_without_cat_dim_attribute(self, cat_op):
        """Test code generation when _cat_dim attribute is not set."""
        tensor = Tensor((2, 4, 3), (12, 3, 1), "float32", "cuda", [])

        output_name = "out"
        input_names = ["x", "y"]

        code = cat_op.codegen(output_name, input_names, tensor)
        # Should use the first dimension with size >= 2 (dim=0 has size 2)
        expected = "out = torch.cat([x, y], dim=0)"

        assert code == expected

    def test_codegen_fallback_to_dim_zero(self, cat_op):
        """Test code generation fallback when no dimension has size >= 2."""
        tensor = Tensor((1, 1, 1), (1, 1, 1), "float32", "cuda", [])

        output_name = "output"
        input_names = ["tensor1", "tensor2"]

        code = cat_op.codegen(output_name, input_names, tensor)
        # Should fallback to dim=0
        expected = "output = torch.cat([tensor1, tensor2], dim=0)"

        assert code == expected

    def test_operator_name(self, cat_op):
        """Test that operator has correct name."""
        assert cat_op.name == "cat"
