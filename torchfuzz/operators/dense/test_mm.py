"""Tests for MmOperator."""

import pytest
from .mm import MmOperator
from torchfuzz.tensor import Tensor


class TestMmOperator:
    """Test class for MmOperator."""

    @pytest.fixture
    def mm_op(self):
        """Create an MmOperator instance."""
        return MmOperator()

    @pytest.fixture
    def sample_2d_tensor(self):
        """Create a sample 2D tensor for testing."""
        return Tensor((4, 6), (6, 1), "float32", "cuda", [])

    @pytest.fixture
    def sample_3d_tensor(self):
        """Create a sample 3D tensor for testing."""
        return Tensor((2, 4, 6), (24, 6, 1), "float32", "cuda", [])

    def test_can_produce_2d_tensor(self, mm_op, sample_2d_tensor):
        """Test that MmOperator can produce 2D tensors."""
        assert mm_op.can_produce(sample_2d_tensor) is True

    def test_can_produce_3d_tensor_false(self, mm_op, sample_3d_tensor):
        """Test that MmOperator cannot produce 3D tensors."""
        assert mm_op.can_produce(sample_3d_tensor) is False

    def test_decompose_creates_correct_shapes(self, mm_op, sample_2d_tensor):
        """Test decomposition creates correct input tensor shapes."""
        inputs = mm_op.decompose(sample_2d_tensor)

        assert len(inputs) == 2

        # Check that matrix multiplication is valid: (m, k) @ (k, n) -> (m, n)
        m, n = sample_2d_tensor.size
        input1_m, input1_k = inputs[0].size
        input2_k, input2_n = inputs[1].size

        assert input1_m == m  # First dimension matches output
        assert input2_n == n  # Last dimension matches output
        assert input1_k == input2_k  # Inner dimensions match

    def test_decompose_preserves_device(self, mm_op, sample_2d_tensor):
        """Test decomposition preserves device."""
        inputs = mm_op.decompose(sample_2d_tensor)

        for input_tensor in inputs:
            assert input_tensor.device == sample_2d_tensor.device

    def test_decompose_type_promotion_float32(self, mm_op):
        """Test type promotion for float32 output."""
        tensor = Tensor((4, 6), (6, 1), "float32", "cuda", [])
        inputs = mm_op.decompose(tensor)

        # Check that dtypes follow promotion rules
        valid_dtypes = ["float32", "bfloat16", "float16"]
        for input_tensor in inputs:
            assert input_tensor.dtype in valid_dtypes

    def test_decompose_type_promotion_bfloat16(self, mm_op):
        """Test type promotion for bfloat16 output."""
        tensor = Tensor((4, 6), (6, 1), "bfloat16", "cuda", [])
        inputs = mm_op.decompose(tensor)

        # Check that dtypes follow promotion rules
        valid_dtypes = ["bfloat16", "float16"]
        for input_tensor in inputs:
            assert input_tensor.dtype in valid_dtypes

    def test_decompose_type_promotion_float16(self, mm_op):
        """Test type promotion for float16 output."""
        tensor = Tensor((4, 6), (6, 1), "float16", "cuda", [])
        inputs = mm_op.decompose(tensor)

        # For float16, all inputs should be float16
        for input_tensor in inputs:
            assert input_tensor.dtype == "float16"

    def test_decompose_invalid_num_inputs(self, mm_op, sample_2d_tensor):
        """Test that decompose raises error for invalid number of inputs."""
        with pytest.raises(ValueError, match="Matrix multiplication requires exactly 2 inputs"):
            mm_op.decompose(sample_2d_tensor, num_inputs=3)

    def test_codegen_correct_format(self, mm_op, sample_2d_tensor):
        """Test code generation produces correct format."""
        output_name = "out"
        input_names = ["a", "b"]

        code = mm_op.codegen(output_name, input_names, sample_2d_tensor)
        expected = "out = torch.mm(a, b)"

        assert code == expected

    def test_codegen_invalid_num_inputs(self, mm_op, sample_2d_tensor):
        """Test that codegen raises error for invalid number of inputs."""
        output_name = "out"
        input_names = ["a", "b", "c"]

        with pytest.raises(ValueError, match="Matrix multiplication requires exactly 2 inputs"):
            mm_op.codegen(output_name, input_names, sample_2d_tensor)

    def test_operator_name(self, mm_op):
        """Test that operator has correct name."""
        assert mm_op.name == "mm"
