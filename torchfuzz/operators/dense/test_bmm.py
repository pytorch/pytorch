"""Tests for BmmOperator."""

import pytest
from .bmm import BmmOperator
from torchfuzz.tensor import Tensor


class TestBmmOperator:
    """Test class for BmmOperator."""

    @pytest.fixture
    def bmm_op(self):
        """Create a BmmOperator instance."""
        return BmmOperator()

    @pytest.fixture
    def sample_3d_tensor(self):
        """Create a sample 3D tensor for testing."""
        return Tensor((8, 4, 6), (24, 6, 1), "float32", "cuda", [])

    @pytest.fixture
    def sample_2d_tensor(self):
        """Create a sample 2D tensor for testing."""
        return Tensor((4, 6), (6, 1), "float32", "cuda", [])

    def test_can_produce_3d_tensor(self, bmm_op, sample_3d_tensor):
        """Test that BmmOperator can produce 3D tensors."""
        assert bmm_op.can_produce(sample_3d_tensor) is True

    def test_can_produce_2d_tensor_false(self, bmm_op, sample_2d_tensor):
        """Test that BmmOperator cannot produce 2D tensors."""
        assert bmm_op.can_produce(sample_2d_tensor) is False

    def test_decompose_creates_correct_shapes(self, bmm_op, sample_3d_tensor):
        """Test decomposition creates correct input tensor shapes."""
        inputs = bmm_op.decompose(sample_3d_tensor)

        assert len(inputs) == 2

        # Check that batch matrix multiplication is valid: (b, m, k) @ (b, k, n) -> (b, m, n)
        b, m, n = sample_3d_tensor.size
        input1_b, input1_m, input1_k = inputs[0].size
        input2_b, input2_k, input2_n = inputs[1].size

        assert input1_b == b == input2_b  # Batch dimensions match
        assert input1_m == m  # First spatial dimension matches output
        assert input2_n == n  # Last spatial dimension matches output
        assert input1_k == input2_k  # Inner dimensions match

    def test_decompose_preserves_device(self, bmm_op, sample_3d_tensor):
        """Test decomposition preserves device."""
        inputs = bmm_op.decompose(sample_3d_tensor)

        for input_tensor in inputs:
            assert input_tensor.device == sample_3d_tensor.device

    def test_decompose_type_promotion_float32(self, bmm_op):
        """Test type promotion for float32 output."""
        tensor = Tensor((8, 4, 6), (24, 6, 1), "float32", "cuda", [])
        inputs = bmm_op.decompose(tensor)

        # Check that dtypes follow promotion rules
        valid_dtypes = ["float32", "bfloat16", "float16"]
        for input_tensor in inputs:
            assert input_tensor.dtype in valid_dtypes

    def test_decompose_type_promotion_bfloat16(self, bmm_op):
        """Test type promotion for bfloat16 output."""
        tensor = Tensor((8, 4, 6), (24, 6, 1), "bfloat16", "cuda", [])
        inputs = bmm_op.decompose(tensor)

        # Check that dtypes follow promotion rules
        valid_dtypes = ["bfloat16", "float16"]
        for input_tensor in inputs:
            assert input_tensor.dtype in valid_dtypes

    def test_decompose_type_promotion_float16(self, bmm_op):
        """Test type promotion for float16 output."""
        tensor = Tensor((8, 4, 6), (24, 6, 1), "float16", "cuda", [])
        inputs = bmm_op.decompose(tensor)

        # For float16, all inputs should be float16
        for input_tensor in inputs:
            assert input_tensor.dtype == "float16"

    def test_decompose_invalid_num_inputs(self, bmm_op, sample_3d_tensor):
        """Test that decompose raises error for invalid number of inputs."""
        with pytest.raises(ValueError, match="Batch matrix multiplication requires exactly 2 inputs"):
            bmm_op.decompose(sample_3d_tensor, num_inputs=3)

    def test_codegen_correct_format(self, bmm_op, sample_3d_tensor):
        """Test code generation produces correct format."""
        output_name = "out"
        input_names = ["a", "b"]

        code = bmm_op.codegen(output_name, input_names, sample_3d_tensor)
        expected = "out = torch.bmm(a, b)"

        assert code == expected

    def test_codegen_invalid_num_inputs(self, bmm_op, sample_3d_tensor):
        """Test that codegen raises error for invalid number of inputs."""
        output_name = "out"
        input_names = ["a", "b", "c"]

        with pytest.raises(ValueError, match="Batch matrix multiplication requires exactly 2 inputs"):
            bmm_op.codegen(output_name, input_names, sample_3d_tensor)

    def test_operator_name(self, bmm_op):
        """Test that operator has correct name."""
        assert bmm_op.name == "bmm"
