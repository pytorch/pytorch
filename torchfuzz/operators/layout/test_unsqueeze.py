"""Tests for UnsqueezeOperator."""

import pytest
from .unsqueeze import UnsqueezeOperator
from torchfuzz.tensor import Tensor


class TestUnsqueezeOperator:
    """Test class for UnsqueezeOperator."""

    @pytest.fixture
    def unsqueeze_op(self):
        """Create an UnsqueezeOperator instance."""
        return UnsqueezeOperator()

    def test_can_produce_with_size_one_dims(self, unsqueeze_op):
        """Test that UnsqueezeOperator can produce tensors with size-1 dimensions."""
        test_tensors = [
            Tensor((1,), (1,), "float32", "cuda", []),  # 1D with size 1
            Tensor((1, 3), (3, 1), "float32", "cuda", []),  # 2D with size-1 dim
            Tensor((2, 1, 4), (4, 4, 1), "float32", "cuda", []),  # 3D with size-1 dim
            Tensor((1, 3, 1, 4), (12, 4, 4, 1), "float16", "cuda", [])  # 4D with size-1 dims
        ]

        for tensor in test_tensors:
            assert unsqueeze_op.can_produce(tensor) is True

    def test_cannot_produce_scalars(self, unsqueeze_op):
        """Test that UnsqueezeOperator cannot produce scalar tensors."""
        scalar = Tensor((), (), "float32", "cuda", [])
        assert unsqueeze_op.can_produce(scalar) is False
    
    def test_cannot_produce_tensors_without_size_one(self, unsqueeze_op):
        """Test that UnsqueezeOperator cannot produce tensors without size-1 dimensions."""
        test_tensors = [
            Tensor((5,), (1,), "float32", "cuda", []),  # 1D without size-1
            Tensor((2, 3), (3, 1), "float32", "cuda", []),  # 2D without size-1
            Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", []),  # 3D without size-1
            Tensor((2, 3, 4, 5), (60, 20, 5, 1), "float16", "cuda", [])  # 4D without size-1
        ]

        for tensor in test_tensors:
            assert unsqueeze_op.can_produce(tensor) is False

    def test_decompose_returns_single_input(self, unsqueeze_op):
        """Test that decomposition returns exactly one input tensor."""
        tensor = Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", [])
        inputs = unsqueeze_op.decompose(tensor)

        assert len(inputs) == 1

    def test_decompose_preserves_properties(self, unsqueeze_op):
        """Test that decomposition preserves tensor properties."""
        tensor = Tensor((6,), (1,), "bfloat16", "cuda", ["test_ops"])
        inputs = unsqueeze_op.decompose(tensor)

        input_tensor = inputs[0]
        assert input_tensor.dtype == tensor.dtype
        assert input_tensor.device == tensor.device
        assert input_tensor.supported_ops == tensor.supported_ops

    def test_decompose_scalar_tensor_should_fail(self, unsqueeze_op):
        """Test that decomposition of scalar tensors should fail since can_produce returns False."""
        scalar = Tensor((), (), "float32", "cuda", [])
        
        # Since can_produce returns False for scalars, decompose should not be called
        # but if it is, it should raise an assertion error
        with pytest.raises(AssertionError):
            unsqueeze_op.decompose(scalar)

    def test_decompose_tensor_with_size_one_dims(self, unsqueeze_op):
        """Test decomposition of tensors with dimensions of size 1."""
        tensor = Tensor((1, 3, 1), (3, 1, 1), "float32", "cuda", [])
        inputs = unsqueeze_op.decompose(tensor)

        input_tensor = inputs[0]

        # Input should have one fewer dimension
        assert len(input_tensor.size) == len(tensor.size) - 1

        # Should have unsqueeze dimension metadata
        assert hasattr(input_tensor, '_unsqueeze_dim')

        # The removed dimension should have been size 1
        unsqueeze_dim = input_tensor._unsqueeze_dim
        assert 0 <= unsqueeze_dim < len(tensor.size)
        assert tensor.size[unsqueeze_dim] == 1

    def test_decompose_tensor_without_size_one_dims(self, unsqueeze_op):
        """Test decomposition of tensors without dimensions of size 1."""
        tensor = Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", [])
        inputs = unsqueeze_op.decompose(tensor)

        input_tensor = inputs[0]

        # Input should have one fewer dimension
        assert len(input_tensor.size) == len(tensor.size) - 1

        # Should have unsqueeze dimension metadata
        assert hasattr(input_tensor, '_unsqueeze_dim')
        unsqueeze_dim = input_tensor._unsqueeze_dim
        assert 0 <= unsqueeze_dim < len(tensor.size)

    def test_decompose_contiguous_stride(self, unsqueeze_op):
        """Test that input tensor has contiguous strides."""
        tensor = Tensor((2, 6), (6, 1), "float32", "cuda", [])
        inputs = unsqueeze_op.decompose(tensor)

        input_tensor = inputs[0]
        size = input_tensor.size
        stride = input_tensor.stride

        # Verify contiguous stride calculation
        expected_stride = []
        acc = 1
        for s in reversed(size):
            expected_stride.insert(0, acc)
            acc *= s

        assert stride == tuple(expected_stride)

    def test_decompose_1d_tensor(self, unsqueeze_op):
        """Test decomposition of 1D tensors."""
        tensor = Tensor((5,), (1,), "float32", "cuda", [])
        inputs = unsqueeze_op.decompose(tensor)

        input_tensor = inputs[0]

        # 1D tensor without size-1 dims should produce scalar input
        assert input_tensor.size == ()
        assert hasattr(input_tensor, '_unsqueeze_dim')
        assert input_tensor._unsqueeze_dim == 0

    def test_decompose_multiple_size_one_dims(self, unsqueeze_op):
        """Test decomposition with multiple size-1 dimensions."""
        tensor = Tensor((1, 2, 1, 3, 1), (6, 3, 3, 1, 1), "float32", "cuda", [])

        # Run multiple times to test randomness
        size_one_positions = [0, 2, 4]  # positions of size-1 dims

        for _ in range(10):
            inputs = unsqueeze_op.decompose(tensor)
            input_tensor = inputs[0]

            # Should remove exactly one dimension
            assert len(input_tensor.size) == len(tensor.size) - 1

            # The unsqueeze dimension should be one of the size-1 positions
            unsqueeze_dim = input_tensor._unsqueeze_dim
            assert unsqueeze_dim in size_one_positions

    def test_codegen_basic(self, unsqueeze_op):
        """Test basic code generation."""
        tensor = Tensor((1, 2, 3), (6, 3, 1), "float32", "cuda", [])

        # First decompose to set up the metadata
        inputs = unsqueeze_op.decompose(tensor)
        input_tensor = inputs[0]

        output_name = "output"
        input_names = ["input"]

        code = unsqueeze_op.codegen(output_name, input_names, tensor)
        expected = f"output = torch.unsqueeze(input, {input_tensor._unsqueeze_dim})"

        assert code == expected

    def test_codegen_scalar_tensor_should_fail(self, unsqueeze_op):
        """Test that codegen with scalar tensor should fail since can_produce returns False."""
        scalar = Tensor((), (), "float32", "cuda", [])

        # Since can_produce returns False for scalars, this should not be called
        # but if it is, decompose would fail first
        with pytest.raises(AssertionError):
            unsqueeze_op.decompose(scalar)

    def test_codegen_1d_tensor(self, unsqueeze_op):
        """Test code generation for 1D tensor."""
        tensor = Tensor((12,), (1,), "float32", "cuda", [])

        # First decompose to set up the metadata
        inputs = unsqueeze_op.decompose(tensor)
        # inputs variable needed for the test setup but not used in assertion

        output_name = "out"
        input_names = ["data"]

        code = unsqueeze_op.codegen(output_name, input_names, tensor)
        expected = "out = torch.unsqueeze(data, 0)"

        assert code == expected

    def test_codegen_tensor_with_size_one(self, unsqueeze_op):
        """Test code generation for tensor with size-1 dimensions."""
        tensor = Tensor((2, 1, 4), (4, 4, 1), "float32", "cuda", [])

        # First decompose to set up the metadata
        inputs = unsqueeze_op.decompose(tensor)
        input_tensor = inputs[0]

        output_name = "unsqueezed"
        input_names = ["original"]

        code = unsqueeze_op.codegen(output_name, input_names, tensor)
        expected = f"unsqueezed = torch.unsqueeze(original, {input_tensor._unsqueeze_dim})"

        assert code == expected

    def test_codegen_different_variable_names(self, unsqueeze_op):
        """Test code generation with different variable names."""
        tensor = Tensor((5, 1), (1, 1), "float32", "cuda", [])

        # First decompose to set up the metadata
        inputs = unsqueeze_op.decompose(tensor)
        input_tensor = inputs[0]

        output_name = "tensor_unsqueezed"
        input_names = ["input_data"]

        code = unsqueeze_op.codegen(output_name, input_names, tensor)
        expected = f"tensor_unsqueezed = torch.unsqueeze(input_data, {input_tensor._unsqueeze_dim})"

        assert code == expected

    def test_codegen_fallback_when_no_metadata(self, unsqueeze_op):
        """Test code generation fallback when no metadata is available."""
        tensor = Tensor((2, 3), (3, 1), "float32", "cuda", [])

        # Don't call decompose, so no metadata is set
        output_name = "output"
        input_names = ["input"]

        code = unsqueeze_op.codegen(output_name, input_names, tensor)
        expected = "output = torch.unsqueeze(input, 0)"

        assert code == expected

    def test_operator_name(self, unsqueeze_op):
        """Test that operator has correct name."""
        assert unsqueeze_op.name == "unsqueeze"

    def test_decompose_reduces_dimensions(self, unsqueeze_op):
        """Test that decompose creates input with fewer dimensions."""
        tensor = Tensor((3, 4, 5), (20, 5, 1), "float32", "cuda", [])

        for _ in range(10):
            inputs = unsqueeze_op.decompose(tensor)
            input_tensor = inputs[0]

            # Input should have fewer dimensions (unless it's a scalar)
            if len(tensor.size) > 0:
                assert len(input_tensor.size) == len(tensor.size) - 1

    def test_decompose_preserves_non_one_dimensions(self, unsqueeze_op):
        """Test that decompose preserves non-1 dimensions when removing size-1 dims."""
        tensor = Tensor((2, 1, 3, 1, 4), (12, 12, 4, 4, 1), "float32", "cuda", [])

        for _ in range(10):
            inputs = unsqueeze_op.decompose(tensor)
            input_tensor = inputs[0]

            # Get non-1 dimensions from both tensors
            tensor_non_ones = [dim for dim in tensor.size if dim != 1]
            input_non_ones = [dim for dim in input_tensor.size if dim != 1]

            # If we removed a size-1 dim, non-1 dims should be the same
            # If we removed a non-1 dim, input should have one fewer non-1 dim
            unsqueeze_dim = input_tensor._unsqueeze_dim
            if tensor.size[unsqueeze_dim] == 1:
                assert sorted(input_non_ones) == sorted(tensor_non_ones)
            else:
                assert len(input_non_ones) == len(tensor_non_ones) - 1

    def test_unsqueeze_dim_in_valid_range(self, unsqueeze_op):
        """Test that unsqueeze_dim is always in valid range."""
        test_tensors = [
            Tensor((5,), (1,), "float32", "cuda", []),  # 1D
            Tensor((2, 3), (3, 1), "float32", "cuda", []),  # 2D
            Tensor((1, 2, 1), (2, 1, 1), "float32", "cuda", []),  # with size-1 dims
            Tensor((2, 3, 4, 5), (60, 20, 5, 1), "float32", "cuda", [])  # 4D
        ]

        for tensor in test_tensors:
            # Skip scalars since can_produce returns False for them
            if len(tensor.size) == 0:
                continue
                
            for _ in range(5):  # Test multiple times for randomness
                inputs = unsqueeze_op.decompose(tensor)
                input_tensor = inputs[0]

                unsqueeze_dim = input_tensor._unsqueeze_dim
                # Unsqueeze dim should be valid for the original tensor
                assert 0 <= unsqueeze_dim <= len(tensor.size)
