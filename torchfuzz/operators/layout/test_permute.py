"""Tests for PermuteOperator."""

import pytest
from .permute import PermuteOperator
from torchfuzz.tensor import Tensor


class TestPermuteOperator:
    """Test class for PermuteOperator."""

    @pytest.fixture
    def permute_op(self):
        """Create a PermuteOperator instance."""
        return PermuteOperator()

    def test_can_produce_appropriate_tensors(self, permute_op):
        """Test that PermuteOperator can produce tensors with at least 1 dimension."""
        # Tensors that can be produced (1D and higher)
        producible_tensors = [
            Tensor((5,), (1,), "float32", "cuda", []),  # 1D
            Tensor((2, 3), (3, 1), "float32", "cuda", []),  # 2D
            Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", []),  # 3D
            Tensor((1, 1, 1, 1), (1, 1, 1, 1), "float16", "cuda", [])  # 4D
        ]

        for tensor in producible_tensors:
            assert permute_op.can_produce(tensor) is True

        # Tensors that cannot be produced (scalars)
        non_producible_tensors = [
            Tensor((), (), "float32", "cuda", []),  # scalar
        ]

        for tensor in non_producible_tensors:
            assert permute_op.can_produce(tensor) is False

    def test_decompose_returns_single_input(self, permute_op):
        """Test that decomposition returns exactly one input tensor."""
        tensor = Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", [])
        inputs = permute_op.decompose(tensor)

        assert len(inputs) == 1

    def test_decompose_preserves_numel(self, permute_op):
        """Test that decomposition preserves the number of elements."""
        tensor = Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", [])
        inputs = permute_op.decompose(tensor)

        input_tensor = inputs[0]

        # Calculate number of elements
        output_numel = 1
        for s in tensor.size:
            output_numel *= s

        input_numel = 1
        for s in input_tensor.size:
            input_numel *= s

        assert input_numel == output_numel

    def test_decompose_preserves_properties(self, permute_op):
        """Test that decomposition preserves tensor properties."""
        tensor = Tensor((6, 4), (4, 1), "bfloat16", "cuda", ["test_ops"])
        inputs = permute_op.decompose(tensor)

        input_tensor = inputs[0]
        assert input_tensor.dtype == tensor.dtype
        assert input_tensor.device == tensor.device
        assert input_tensor.supported_ops == tensor.supported_ops

    def test_decompose_rejects_scalar_tensor(self, permute_op):
        """Test that decomposition properly rejects scalar tensors."""
        scalar = Tensor((), (), "float32", "cuda", [])
        # This should trigger the assertion since can_produce returns False for scalars
        with pytest.raises(AssertionError, match="PermuteOperator should not receive scalar tensors"):
            permute_op.decompose(scalar)

    def test_decompose_stores_permutation_metadata(self, permute_op):
        """Test that decomposition stores permutation dimensions in metadata."""
        tensor = Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", [])
        inputs = permute_op.decompose(tensor)

        # Check that metadata was stored
        assert hasattr(tensor, "_permute_dims")
        assert len(tensor._permute_dims) == 3

        # All dimensions should be present exactly once
        assert set(tensor._permute_dims) == {0, 1, 2}

    def test_decompose_1d_tensor(self, permute_op):
        """Test decomposition of 1D tensors."""
        tensor = Tensor((5,), (1,), "float32", "cuda", [])
        inputs = permute_op.decompose(tensor)

        input_tensor = inputs[0]
        assert input_tensor.size == (5,)
        assert hasattr(tensor, "_permute_dims")
        assert tensor._permute_dims == (0,)

    def test_decompose_2d_tensor_creates_permutation(self, permute_op):
        """Test that 2D tensor decomposition creates valid permutation."""
        tensor = Tensor((2, 3), (3, 1), "float32", "cuda", [])
        inputs = permute_op.decompose(tensor)

        input_tensor = inputs[0]
        assert len(input_tensor.size) == 2
        assert set(tensor._permute_dims) == {0, 1}

    def test_decompose_permutation_creates_correct_input_shape(self, permute_op):
        """Test that input shape matches expected permutation."""
        tensor = Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", [])
        inputs = permute_op.decompose(tensor)

        input_tensor = inputs[0]
        permute_dims = tensor._permute_dims

        # Verify the inverse permutation logic:
        # If output = input.permute(dims), then input[dims[i]] = output[i]
        for i in range(len(permute_dims)):
            assert input_tensor.size[permute_dims[i]] == tensor.size[i]

    # Removed scalar tensor test since permute doesn't support scalars

    def test_codegen_1d_tensor(self, permute_op):
        """Test code generation for 1D tensor."""
        tensor = Tensor((5,), (1,), "float32", "cuda", [])
        tensor._permute_dims = (0,)

        output_name = "out"
        input_names = ["data"]

        code = permute_op.codegen(output_name, input_names, tensor)
        expected = "out = data.permute(0)"

        assert code == expected

    def test_codegen_2d_tensor(self, permute_op):
        """Test code generation for 2D tensor."""
        tensor = Tensor((2, 3), (3, 1), "float32", "cuda", [])
        tensor._permute_dims = (1, 0)

        output_name = "output"
        input_names = ["input"]

        code = permute_op.codegen(output_name, input_names, tensor)
        expected = "output = input.permute(1, 0)"

        assert code == expected

    def test_codegen_3d_tensor(self, permute_op):
        """Test code generation for 3D tensor."""
        tensor = Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", [])
        tensor._permute_dims = (2, 0, 1)

        output_name = "result"
        input_names = ["data"]

        code = permute_op.codegen(output_name, input_names, tensor)
        expected = "result = data.permute(2, 0, 1)"

        assert code == expected

    def test_codegen_4d_tensor(self, permute_op):
        """Test code generation for 4D tensor."""
        tensor = Tensor((2, 3, 4, 5), (60, 20, 5, 1), "float32", "cuda", [])
        tensor._permute_dims = (3, 1, 0, 2)

        output_name = "reshaped"
        input_names = ["original"]

        code = permute_op.codegen(output_name, input_names, tensor)
        expected = "reshaped = original.permute(3, 1, 0, 2)"

        assert code == expected

    def test_codegen_no_metadata_defaults_to_identity(self, permute_op):
        """Test code generation when no metadata is present."""
        tensor = Tensor((2, 3), (3, 1), "float32", "cuda", [])
        # Don't set permute metadata

        output_name = "out"
        input_names = ["x"]

        code = permute_op.codegen(output_name, input_names, tensor)
        expected = "out = x.permute(0, 1)"  # Should default to identity permutation

        assert code == expected

    def test_codegen_different_variable_names(self, permute_op):
        """Test code generation with different variable names."""
        tensor = Tensor((5, 2, 3), (6, 3, 1), "float32", "cuda", [])
        tensor._permute_dims = (2, 1, 0)

        output_name = "tensor_permuted"
        input_names = ["input_data"]

        code = permute_op.codegen(output_name, input_names, tensor)
        expected = "tensor_permuted = input_data.permute(2, 1, 0)"

        assert code == expected

    def test_operator_name(self, permute_op):
        """Test that operator has correct name."""
        assert permute_op.name == "permute"

    def test_decompose_multiple_calls_different_permutations(self, permute_op):
        """Test that multiple decompositions can produce different permutations."""
        tensor_shape = (2, 3, 4)
        permutations_seen = set()

        # Run multiple decompositions to check for variety
        for _ in range(20):
            tensor = Tensor(tensor_shape, (12, 4, 1), "float32", "cuda", [])
            inputs = permute_op.decompose(tensor)
            permutations_seen.add(tensor._permute_dims)

        # Should have seen at least a few different permutations
        assert len(permutations_seen) > 1
