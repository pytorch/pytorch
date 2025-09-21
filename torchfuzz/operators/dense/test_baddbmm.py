"""Tests for BaddbmmOperator."""

import pytest
from .baddbmm import BaddbmmOperator
from torchfuzz.tensor import Tensor


class TestBaddbmmOperator:
    """Test class for BaddbmmOperator."""

    @pytest.fixture
    def baddbmm_op(self):
        """Create a BaddbmmOperator instance."""
        return BaddbmmOperator()

    @pytest.fixture
    def sample_3d_tensor(self):
        """Create a sample 3D tensor for testing."""
        return Tensor((8, 4, 6), (24, 6, 1), "float32", "cuda", [])

    @pytest.fixture
    def sample_2d_tensor(self):
        """Create a sample 2D tensor for testing."""
        return Tensor((4, 6), (6, 1), "float32", "cuda", [])

    def test_can_produce_3d_tensor(self, baddbmm_op, sample_3d_tensor):
        """Test that BaddbmmOperator can produce 3D tensors."""
        assert baddbmm_op.can_produce(sample_3d_tensor) is True

    def test_can_produce_2d_tensor_false(self, baddbmm_op, sample_2d_tensor):
        """Test that BaddbmmOperator cannot produce 2D tensors."""
        assert baddbmm_op.can_produce(sample_2d_tensor) is False

    def test_decompose_creates_correct_shapes(self, baddbmm_op, sample_3d_tensor):
        """Test decomposition creates correct input tensor shapes."""
        inputs = baddbmm_op.decompose(sample_3d_tensor)

        assert len(inputs) == 3

        # Check shapes: bias (b, m, n), batch1 (b, m, k), batch2 (b, k, n)
        b, m, n = sample_3d_tensor.size
        bias_b, bias_m, bias_n = inputs[0].size
        batch1_b, batch1_m, batch1_k = inputs[1].size
        batch2_b, batch2_k, batch2_n = inputs[2].size

        assert bias_b == b and bias_m == m and bias_n == n  # Bias matches output shape
        assert batch1_b == b and batch1_m == m  # First batch matrix matches batch and first dim
        assert batch2_b == b and batch2_n == n  # Second batch matrix matches batch and last dim
        assert batch1_k == batch2_k  # Inner dimensions match

    def test_decompose_preserves_device(self, baddbmm_op, sample_3d_tensor):
        """Test decomposition preserves device."""
        inputs = baddbmm_op.decompose(sample_3d_tensor)

        for input_tensor in inputs:
            assert input_tensor.device == sample_3d_tensor.device

    def test_decompose_type_promotion_float32(self, baddbmm_op):
        """Test type promotion for float32 output."""
        tensor = Tensor((8, 4, 6), (24, 6, 1), "float32", "cuda", [])
        inputs = baddbmm_op.decompose(tensor)

        # Check that dtypes follow promotion rules
        valid_dtypes = ["float32", "bfloat16", "float16"]
        for input_tensor in inputs:
            assert input_tensor.dtype in valid_dtypes

    def test_decompose_type_promotion_bfloat16(self, baddbmm_op):
        """Test type promotion for bfloat16 output."""
        tensor = Tensor((8, 4, 6), (24, 6, 1), "bfloat16", "cuda", [])
        inputs = baddbmm_op.decompose(tensor)

        # Check that dtypes follow promotion rules
        valid_dtypes = ["bfloat16", "float16"]
        for input_tensor in inputs:
            assert input_tensor.dtype in valid_dtypes

    def test_decompose_type_promotion_float16(self, baddbmm_op):
        """Test type promotion for float16 output."""
        tensor = Tensor((8, 4, 6), (24, 6, 1), "float16", "cuda", [])
        inputs = baddbmm_op.decompose(tensor)

        # For float16, all inputs should be float16
        for input_tensor in inputs:
            assert input_tensor.dtype == "float16"

    def test_decompose_invalid_num_inputs(self, baddbmm_op, sample_3d_tensor):
        """Test that decompose raises error for invalid number of inputs."""
        with pytest.raises(ValueError, match="Baddbmm requires exactly 3 inputs"):
            baddbmm_op.decompose(sample_3d_tensor, num_inputs=2)

    def test_codegen_correct_format(self, baddbmm_op, sample_3d_tensor):
        """Test code generation produces correct format."""
        output_name = "out"
        input_names = ["bias", "batch1", "batch2"]

        code = baddbmm_op.codegen(output_name, input_names, sample_3d_tensor)
        expected = "out = torch.baddbmm(bias, batch1, batch2)"

        assert code == expected

    def test_codegen_invalid_num_inputs(self, baddbmm_op, sample_3d_tensor):
        """Test that codegen raises error for invalid number of inputs."""
        output_name = "out"
        input_names = ["a", "b"]

        with pytest.raises(ValueError, match="Baddbmm requires exactly 3 inputs"):
            baddbmm_op.codegen(output_name, input_names, sample_3d_tensor)

    def test_operator_name(self, baddbmm_op):
        """Test that operator has correct name."""
        assert baddbmm_op.name == "baddbmm"
