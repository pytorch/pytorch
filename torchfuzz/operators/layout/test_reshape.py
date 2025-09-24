"""Tests for ReshapeOperator."""

import pytest
from .reshape import ReshapeOperator
from torchfuzz.tensor import Tensor


class TestReshapeOperator:
    """Test class for ReshapeOperator."""

    @pytest.fixture
    def reshape_op(self):
        """Create a ReshapeOperator instance."""
        return ReshapeOperator()

    def test_can_produce_returns_true(self, reshape_op):
        """Test that ReshapeOperator can always produce any tensor."""
        test_tensors = [
            Tensor((), (), "float32", "cuda", []),  # scalar
            Tensor((5,), (1,), "float32", "cuda", []),  # 1D
            Tensor((2, 3), (3, 1), "float32", "cuda", []),  # 2D
            Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", []),  # 3D
            Tensor((1, 1, 1, 1), (1, 1, 1, 1), "float16", "cuda", [])  # 4D
        ]

        for tensor in test_tensors:
            assert reshape_op.can_produce(tensor) is True

    def test_decompose_returns_single_input(self, reshape_op):
        """Test that decomposition returns exactly one input tensor."""
        tensor = Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", [])
        inputs = reshape_op.decompose(tensor)

        assert len(inputs) == 1

    def test_decompose_preserves_numel(self, reshape_op):
        """Test that decomposition preserves the number of elements."""
        tensor = Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", [])
        inputs = reshape_op.decompose(tensor)

        input_tensor = inputs[0]

        # Calculate number of elements
        output_numel = 1
        for s in tensor.size:
            output_numel *= s

        input_numel = 1
        for s in input_tensor.size:
            input_numel *= s

        assert input_numel == output_numel

    def test_decompose_preserves_properties(self, reshape_op):
        """Test that decomposition preserves tensor properties."""
        tensor = Tensor((6,), (1,), "bfloat16", "cuda", ["test_ops"])
        inputs = reshape_op.decompose(tensor)

        input_tensor = inputs[0]
        assert input_tensor.dtype == tensor.dtype
        assert input_tensor.device == tensor.device
        assert input_tensor.supported_ops == tensor.supported_ops

    def test_decompose_scalar_tensor(self, reshape_op):
        """Test decomposition of scalar tensors."""
        scalar = Tensor((), (), "float32", "cuda", [])
        inputs = reshape_op.decompose(scalar)

        input_tensor = inputs[0]
        # Scalar has numel=1, so input should also have numel=1
        input_numel = 1
        for s in input_tensor.size:
            input_numel *= s
        assert input_numel == 1

    def test_decompose_input_dimensions_range(self, reshape_op):
        """Test that input tensor has 1-4 dimensions."""
        tensor = Tensor((24,), (1,), "float32", "cuda", [])

        # Test multiple decompositions to check randomness bounds
        for _ in range(10):
            inputs = reshape_op.decompose(tensor)
            input_tensor = inputs[0]
            assert 1 <= len(input_tensor.size) <= 4

    def test_decompose_contiguous_stride(self, reshape_op):
        """Test that input tensor has contiguous strides."""
        tensor = Tensor((2, 6), (6, 1), "float32", "cuda", [])
        inputs = reshape_op.decompose(tensor)

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

    def test_decompose_different_shapes_same_numel(self, reshape_op):
        """Test decomposition with different input shapes for same numel."""
        tensor = Tensor((12,), (1,), "float32", "cuda", [])

        # Run multiple decompositions and check they all have same numel
        for _ in range(5):
            inputs = reshape_op.decompose(tensor)
            input_tensor = inputs[0]

            input_numel = 1
            for s in input_tensor.size:
                input_numel *= s
            assert input_numel == 12

    def test_decompose_large_numel(self, reshape_op):
        """Test decomposition with larger number of elements."""
        tensor = Tensor((8, 9, 10), (90, 10, 1), "float32", "cuda", [])
        inputs = reshape_op.decompose(tensor)

        input_tensor = inputs[0]

        # Verify numel preservation
        output_numel = 8 * 9 * 10  # 720
        input_numel = 1
        for s in input_tensor.size:
            input_numel *= s

        assert input_numel == output_numel

    def test_decompose_zero_size_tensor(self, reshape_op):
        """Test decomposition with tensor containing zero size."""
        tensor = Tensor((0, 5), (5, 1), "float32", "cuda", [])
        inputs = reshape_op.decompose(tensor)

        input_tensor = inputs[0]
        # Both should have numel=0
        output_numel = 0 * 5
        input_numel = 1
        for s in input_tensor.size:
            input_numel *= s

        assert input_numel == output_numel == 0

    def test_codegen_basic(self, reshape_op):
        """Test basic code generation."""
        tensor = Tensor((2, 6), (6, 1), "float32", "cuda", [])
        output_name = "output"
        input_names = ["input"]

        code = reshape_op.codegen(output_name, input_names, tensor)
        expected = "output = input.reshape((2, 6))"

        assert code == expected

    def test_codegen_scalar_tensor(self, reshape_op):
        """Test code generation for scalar tensor."""
        scalar = Tensor((), (), "float32", "cuda", [])
        output_name = "result"
        input_names = ["x"]

        code = reshape_op.codegen(output_name, input_names, scalar)
        expected = "result = x.reshape(())"

        assert code == expected

    def test_codegen_1d_tensor(self, reshape_op):
        """Test code generation for 1D tensor."""
        tensor = Tensor((12,), (1,), "float32", "cuda", [])
        output_name = "out"
        input_names = ["data"]

        code = reshape_op.codegen(output_name, input_names, tensor)
        expected = "out = data.reshape((12,))"

        assert code == expected

    def test_codegen_3d_tensor(self, reshape_op):
        """Test code generation for 3D tensor."""
        tensor = Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", [])
        output_name = "reshaped"
        input_names = ["original"]

        code = reshape_op.codegen(output_name, input_names, tensor)
        expected = "reshaped = original.reshape((2, 3, 4))"

        assert code == expected

    def test_codegen_4d_tensor(self, reshape_op):
        """Test code generation for 4D tensor."""
        tensor = Tensor((2, 3, 4, 5), (60, 20, 5, 1), "float32", "cuda", [])
        output_name = "reshaped"
        input_names = ["original"]

        code = reshape_op.codegen(output_name, input_names, tensor)
        expected = "reshaped = original.reshape((2, 3, 4, 5))"

        assert code == expected

    def test_codegen_different_variable_names(self, reshape_op):
        """Test code generation with different variable names."""
        tensor = Tensor((5, 2), (2, 1), "float32", "cuda", [])
        output_name = "tensor_reshaped"
        input_names = ["input_data"]

        code = reshape_op.codegen(output_name, input_names, tensor)
        expected = "tensor_reshaped = input_data.reshape((5, 2))"

        assert code == expected

    def test_codegen_zero_size_tensor(self, reshape_op):
        """Test code generation for tensor with zero size."""
        tensor = Tensor((0, 5), (5, 1), "float32", "cuda", [])
        output_name = "empty_tensor"
        input_names = ["input_empty"]

        code = reshape_op.codegen(output_name, input_names, tensor)
        expected = "empty_tensor = input_empty.reshape((0, 5))"

        assert code == expected

    def test_operator_name(self, reshape_op):
        """Test that operator has correct name."""
        assert reshape_op.name == "reshape"

    def test_decompose_produces_valid_positive_dimensions(self, reshape_op):
        """Test that decomposition produces tensors with positive dimensions."""
        tensor = Tensor((24,), (1,), "float32", "cuda", [])

        for _ in range(10):
            inputs = reshape_op.decompose(tensor)
            input_tensor = inputs[0]

            # All dimensions should be positive integers
            for dim in input_tensor.size:
                assert isinstance(dim, int)
                assert dim > 0
