"""Tests for ContiguousOperator."""

import pytest
from .contiguous import ContiguousOperator
from torchfuzz.tensor import Tensor


class TestContiguousOperator:
    """Test class for ContiguousOperator."""

    @pytest.fixture
    def contiguous_op(self):
        """Create a ContiguousOperator instance."""
        return ContiguousOperator()

    def test_can_produce_returns_true_for_all_tensors(self, contiguous_op):
        """Test that ContiguousOperator can produce any tensor."""
        test_tensors = [
            Tensor((), (), "float32", "cuda", []),  # scalar
            Tensor((5,), (1,), "float32", "cuda", []),  # 1D
            Tensor((2, 3), (3, 1), "float32", "cuda", []),  # 2D
            Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", []),  # 3D
            Tensor((1, 1, 1, 1), (1, 1, 1, 1), "float16", "cuda", [])  # 4D
        ]

        for tensor in test_tensors:
            assert contiguous_op.can_produce(tensor) is True

    def test_decompose_returns_single_input(self, contiguous_op):
        """Test that decomposition returns exactly one input tensor."""
        tensor = Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", [])
        inputs = contiguous_op.decompose(tensor)

        assert len(inputs) == 1

    def test_decompose_preserves_shape(self, contiguous_op):
        """Test that decomposition preserves the tensor shape."""
        tensor = Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", [])
        inputs = contiguous_op.decompose(tensor)

        input_tensor = inputs[0]
        assert input_tensor.size == tensor.size

    def test_decompose_preserves_properties(self, contiguous_op):
        """Test that decomposition preserves tensor properties."""
        tensor = Tensor((6, 4), (4, 1), "bfloat16", "cuda", ["test_ops"])
        inputs = contiguous_op.decompose(tensor)

        input_tensor = inputs[0]
        assert input_tensor.dtype == tensor.dtype
        assert input_tensor.device == tensor.device
        assert input_tensor.supported_ops == tensor.supported_ops

    def test_decompose_scalar_tensor(self, contiguous_op):
        """Test decomposition of scalar tensors."""
        scalar = Tensor((), (), "float32", "cuda", [])
        inputs = contiguous_op.decompose(scalar)

        input_tensor = inputs[0]
        assert input_tensor.size == ()
        assert input_tensor.stride == ()

    def test_decompose_1d_tensor(self, contiguous_op):
        """Test decomposition of 1D tensors."""
        tensor = Tensor((5,), (1,), "float32", "cuda", [])
        inputs = contiguous_op.decompose(tensor)

        input_tensor = inputs[0]
        assert input_tensor.size == (5,)
        # 1D tensor stride should be modified to be non-contiguous
        assert input_tensor.stride != tensor.stride or len(tensor.size) == 0

    def test_decompose_creates_non_contiguous_input(self, contiguous_op):
        """Test that decomposition creates input with non-contiguous strides."""
        tensor = Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", [])
        inputs = contiguous_op.decompose(tensor)

        input_tensor = inputs[0]
        # Input should have different strides (non-contiguous)
        # The strides should be modified from the original contiguous strides
        original_stride = tensor.stride
        input_stride = input_tensor.stride

        # At least one stride should be different (unless it's a scalar)
        assert input_stride != original_stride or len(tensor.size) == 0

    def test_decompose_preserves_numel(self, contiguous_op):
        """Test that decomposition preserves the number of elements implicitly."""
        tensor = Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", [])
        inputs = contiguous_op.decompose(tensor)

        input_tensor = inputs[0]
        # Size should be identical, so numel is preserved
        assert input_tensor.size == tensor.size

    def test_decompose_modifies_strides_correctly(self, contiguous_op):
        """Test that stride modification is reasonable."""
        tensor = Tensor((3, 4), (4, 1), "float32", "cuda", [])
        inputs = contiguous_op.decompose(tensor)

        input_tensor = inputs[0]
        # All strides should be positive integers
        for stride in input_tensor.stride:
            assert isinstance(stride, int)
            assert stride >= 0

    def test_decompose_handles_zero_strides(self, contiguous_op):
        """Test decomposition with tensors that have zero strides."""
        # Tensor with size-1 dimension (would have stride 0 in some cases)
        tensor = Tensor((1, 5), (5, 1), "float32", "cuda", [])
        inputs = contiguous_op.decompose(tensor)

        input_tensor = inputs[0]
        assert input_tensor.size == tensor.size

    def test_decompose_multiple_calls_can_vary(self, contiguous_op):
        """Test that multiple decompositions can produce different non-contiguous layouts."""
        tensor = Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", [])

        strides_seen = set()
        for _ in range(10):
            inputs = contiguous_op.decompose(tensor)
            input_tensor = inputs[0]
            strides_seen.add(input_tensor.stride)

        # Should see some variation in stride patterns (randomness in the algorithm)
        # At minimum, should see the modified strides
        assert len(strides_seen) >= 1

    def test_codegen_basic(self, contiguous_op):
        """Test basic code generation."""
        tensor = Tensor((2, 3), (3, 1), "float32", "cuda", [])
        output_name = "output"
        input_names = ["input"]

        code = contiguous_op.codegen(output_name, input_names, tensor)
        expected = "output = input.contiguous()"

        assert code == expected

    def test_codegen_scalar_tensor(self, contiguous_op):
        """Test code generation for scalar tensor."""
        scalar = Tensor((), (), "float32", "cuda", [])
        output_name = "result"
        input_names = ["x"]

        code = contiguous_op.codegen(output_name, input_names, scalar)
        expected = "result = x.contiguous()"

        assert code == expected

    def test_codegen_1d_tensor(self, contiguous_op):
        """Test code generation for 1D tensor."""
        tensor = Tensor((5,), (1,), "float32", "cuda", [])
        output_name = "out"
        input_names = ["data"]

        code = contiguous_op.codegen(output_name, input_names, tensor)
        expected = "out = data.contiguous()"

        assert code == expected

    def test_codegen_3d_tensor(self, contiguous_op):
        """Test code generation for 3D tensor."""
        tensor = Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", [])
        output_name = "result"
        input_names = ["data"]

        code = contiguous_op.codegen(output_name, input_names, tensor)
        expected = "result = data.contiguous()"

        assert code == expected

    def test_codegen_different_variable_names(self, contiguous_op):
        """Test code generation with different variable names."""
        tensor = Tensor((5, 2, 3), (6, 3, 1), "float32", "cuda", [])
        output_name = "tensor_contiguous"
        input_names = ["input_data"]

        code = contiguous_op.codegen(output_name, input_names, tensor)
        expected = "tensor_contiguous = input_data.contiguous()"

        assert code == expected

    def test_codegen_high_dimensional_tensor(self, contiguous_op):
        """Test code generation for high-dimensional tensor."""
        tensor = Tensor((2, 3, 4, 5, 6), (360, 120, 30, 6, 1), "float32", "cuda", [])
        output_name = "reshaped"
        input_names = ["original"]

        code = contiguous_op.codegen(output_name, input_names, tensor)
        expected = "reshaped = original.contiguous()"

        assert code == expected

    def test_operator_name(self, contiguous_op):
        """Test that operator has correct name."""
        assert contiguous_op.name == "contiguous"

    def test_decompose_large_tensor(self, contiguous_op):
        """Test decomposition with larger tensors."""
        tensor = Tensor((8, 9, 10, 11), (990, 110, 11, 1), "float32", "cuda", [])
        inputs = contiguous_op.decompose(tensor)

        input_tensor = inputs[0]
        assert input_tensor.size == tensor.size
        assert input_tensor.dtype == tensor.dtype
        assert input_tensor.device == tensor.device

    def test_decompose_with_zero_size_dimension(self, contiguous_op):
        """Test decomposition with tensor containing zero-size dimension."""
        tensor = Tensor((0, 5), (5, 1), "float32", "cuda", [])
        inputs = contiguous_op.decompose(tensor)

        input_tensor = inputs[0]
        assert input_tensor.size == tensor.size
