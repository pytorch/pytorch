"""Tests for DivOperator."""

import pytest
from .div import DivOperator
from torchfuzz.tensor import Tensor


class TestDivOperator:
    """Test class for DivOperator."""

    @pytest.fixture
    def div_op(self):
        """Create a DivOperator instance."""
        return DivOperator()

    @pytest.fixture
    def sample_tensor(self):
        """Create a sample tensor for testing."""
        return Tensor((2, 3), (3, 1), "float32", "cuda", [])

    def test_can_produce_returns_true(self, div_op, sample_tensor):
        """Test that DivOperator can always produce any tensor."""
        assert div_op.can_produce(sample_tensor) is True

    def test_decompose_default_inputs(self, div_op, sample_tensor):
        """Test decomposition with default number of inputs (2)."""
        inputs = div_op.decompose(sample_tensor)

        assert len(inputs) == 2
        for input_tensor in inputs:
            assert input_tensor.size == sample_tensor.size
            assert input_tensor.stride == sample_tensor.stride
            assert input_tensor.device == sample_tensor.device

    def test_decompose_multiple_inputs(self, div_op, sample_tensor):
        """Test decomposition with multiple inputs."""
        num_inputs = 4
        inputs = div_op.decompose(sample_tensor, num_inputs=num_inputs)

        assert len(inputs) == num_inputs
        for input_tensor in inputs:
            assert input_tensor.size == sample_tensor.size
            assert input_tensor.stride == sample_tensor.stride
            assert input_tensor.device == sample_tensor.device

    def test_decompose_single_input(self, div_op, sample_tensor):
        """Test decomposition with single input (reciprocal)."""
        inputs = div_op.decompose(sample_tensor, num_inputs=1)

        assert len(inputs) == 1
        input_tensor = inputs[0]
        assert input_tensor.size == sample_tensor.size
        assert input_tensor.stride == sample_tensor.stride
        assert input_tensor.device == sample_tensor.device

    def test_decompose_type_promotion_float32(self, div_op):
        """Test type promotion for float32 output."""
        tensor = Tensor((2, 2), (2, 1), "float32", "cuda", [])
        inputs = div_op.decompose(tensor, num_inputs=2)

        # Check that dtypes follow promotion rules
        valid_dtypes = ["float32", "bfloat16", "float16"]
        for input_tensor in inputs:
            assert input_tensor.dtype in valid_dtypes

    def test_decompose_type_promotion_bfloat16(self, div_op):
        """Test type promotion for bfloat16 output."""
        tensor = Tensor((2, 2), (2, 1), "bfloat16", "cuda", [])
        inputs = div_op.decompose(tensor, num_inputs=2)

        # Check that dtypes follow promotion rules
        valid_dtypes = ["bfloat16", "float16"]
        for input_tensor in inputs:
            assert input_tensor.dtype in valid_dtypes

    def test_decompose_type_promotion_float16(self, div_op):
        """Test type promotion for float16 output."""
        tensor = Tensor((2, 2), (2, 1), "float16", "cuda", [])
        inputs = div_op.decompose(tensor, num_inputs=2)

        # For float16, all inputs should be float16
        for input_tensor in inputs:
            assert input_tensor.dtype == "float16"

    def test_codegen_single_input_reciprocal(self, div_op, sample_tensor):
        """Test code generation with single input (reciprocal)."""
        output_name = "out"
        input_names = ["a"]

        code = div_op.codegen(output_name, input_names, sample_tensor)
        expected = "out = 1.0 / a"

        assert code == expected

    def test_codegen_two_inputs(self, div_op, sample_tensor):
        """Test code generation with two inputs."""
        output_name = "out"
        input_names = ["a", "b"]

        code = div_op.codegen(output_name, input_names, sample_tensor)
        expected = "out = (a) / b"

        assert code == expected

    def test_codegen_multiple_inputs(self, div_op, sample_tensor):
        """Test code generation with multiple inputs (left-to-right division)."""
        output_name = "result"
        input_names = ["x", "y", "z", "w"]

        code = div_op.codegen(output_name, input_names, sample_tensor)
        expected = "result = (((x) / y) / z) / w"

        assert code == expected

    def test_operator_name(self, div_op):
        """Test that operator has correct name."""
        assert div_op.name == "div"
