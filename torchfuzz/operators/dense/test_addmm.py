"""Tests for AddmmOperator."""

import pytest
from .addmm import AddmmOperator
from torchfuzz.tensor import Tensor


class TestAddmmOperator:
    """Test class for AddmmOperator."""

    @pytest.fixture
    def addmm_op(self):
        """Create an AddmmOperator instance."""
        return AddmmOperator()

    @pytest.fixture
    def sample_2d_tensor(self):
        """Create a sample 2D tensor for testing."""
        return Tensor((4, 6), (6, 1), "float32", "cuda", [])

    @pytest.fixture
    def sample_3d_tensor(self):
        """Create a sample 3D tensor for testing."""
        return Tensor((2, 4, 6), (24, 6, 1), "float32", "cuda", [])

    def test_can_produce_2d_tensor(self, addmm_op, sample_2d_tensor):
        """Test that AddmmOperator can produce 2D tensors."""
        assert addmm_op.can_produce(sample_2d_tensor) is True

    def test_can_produce_3d_tensor_false(self, addmm_op, sample_3d_tensor):
        """Test that AddmmOperator cannot produce 3D tensors."""
        assert addmm_op.can_produce(sample_3d_tensor) is False

    def test_decompose_creates_correct_shapes(self, addmm_op, sample_2d_tensor):
        """Test decomposition creates correct input tensor shapes."""
        inputs = addmm_op.decompose(sample_2d_tensor)

        assert len(inputs) == 3

        # Check shapes: bias (m, n), mat1 (m, k), mat2 (k, n)
        m, n = sample_2d_tensor.size
        bias_m, bias_n = inputs[0].size
        mat1_m, mat1_k = inputs[1].size
        mat2_k, mat2_n = inputs[2].size

        assert bias_m == m and bias_n == n  # Bias matches output shape
        assert mat1_m == m  # First matrix first dimension matches output
        assert mat2_n == n  # Second matrix last dimension matches output
        assert mat1_k == mat2_k  # Inner dimensions match

    def test_decompose_preserves_device(self, addmm_op, sample_2d_tensor):
        """Test decomposition preserves device."""
        inputs = addmm_op.decompose(sample_2d_tensor)

        for input_tensor in inputs:
            assert input_tensor.device == sample_2d_tensor.device

    def test_decompose_type_promotion_float32(self, addmm_op):
        """Test type promotion for float32 output."""
        tensor = Tensor((4, 6), (6, 1), "float32", "cuda", [])
        inputs = addmm_op.decompose(tensor)

        # Check that dtypes follow promotion rules
        valid_dtypes = ["float32", "bfloat16", "float16"]
        for input_tensor in inputs:
            assert input_tensor.dtype in valid_dtypes

    def test_decompose_type_promotion_bfloat16(self, addmm_op):
        """Test type promotion for bfloat16 output."""
        tensor = Tensor((4, 6), (6, 1), "bfloat16", "cuda", [])
        inputs = addmm_op.decompose(tensor)

        # Check that dtypes follow promotion rules
        valid_dtypes = ["bfloat16", "float16"]
        for input_tensor in inputs:
            assert input_tensor.dtype in valid_dtypes

    def test_decompose_type_promotion_float16(self, addmm_op):
        """Test type promotion for float16 output."""
        tensor = Tensor((4, 6), (6, 1), "float16", "cuda", [])
        inputs = addmm_op.decompose(tensor)

        # For float16, all inputs should be float16
        for input_tensor in inputs:
            assert input_tensor.dtype == "float16"

    def test_decompose_invalid_num_inputs(self, addmm_op, sample_2d_tensor):
        """Test that decompose raises error for invalid number of inputs."""
        with pytest.raises(ValueError, match="Addmm requires exactly 3 inputs"):
            addmm_op.decompose(sample_2d_tensor, num_inputs=2)

    def test_codegen_correct_format(self, addmm_op, sample_2d_tensor):
        """Test code generation produces correct format."""
        output_name = "out"
        input_names = ["bias", "mat1", "mat2"]

        code = addmm_op.codegen(output_name, input_names, sample_2d_tensor)
        expected = "out = torch.addmm(bias, mat1, mat2)"

        assert code == expected

    def test_codegen_invalid_num_inputs(self, addmm_op, sample_2d_tensor):
        """Test that codegen raises error for invalid number of inputs."""
        output_name = "out"
        input_names = ["a", "b"]

        with pytest.raises(ValueError, match="Addmm requires exactly 3 inputs"):
            addmm_op.codegen(output_name, input_names, sample_2d_tensor)

    def test_operator_name(self, addmm_op):
        """Test that operator has correct name."""
        assert addmm_op.name == "addmm"
