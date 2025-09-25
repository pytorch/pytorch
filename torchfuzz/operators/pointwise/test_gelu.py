"""Tests for GeluOperator."""

import pytest
from .gelu import GeluOperator
from torchfuzz.tensor import Tensor


class TestGeluOperator:
    """Test class for GeluOperator."""

    @pytest.fixture
    def gelu_op(self):
        """Create a GeluOperator instance."""
        return GeluOperator()

    @pytest.fixture
    def sample_tensor(self):
        """Create a sample tensor for testing."""
        return Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", [])

    def test_can_produce_returns_true(self, gelu_op, sample_tensor):
        """Test that GeluOperator can always produce any tensor."""
        assert gelu_op.can_produce(sample_tensor) is True

    def test_can_produce_scalar_tensor(self, gelu_op):
        """Test that GeluOperator can produce scalar tensors."""
        scalar_tensor = Tensor((), (), "float16", "cuda", [])
        assert gelu_op.can_produce(scalar_tensor) is True

    def test_can_produce_different_dtypes(self, gelu_op):
        """Test that GeluOperator can produce tensors of different dtypes."""
        dtypes = ["float32", "float16", "bfloat16"]
        for dtype in dtypes:
            tensor = Tensor((5, 5), (5, 1), dtype, "cuda", [])
            assert gelu_op.can_produce(tensor) is True

    def test_decompose_returns_single_input(self, gelu_op, sample_tensor):
        """Test that decomposition returns exactly one input tensor."""
        inputs = gelu_op.decompose(sample_tensor)

        assert len(inputs) == 1
        input_tensor = inputs[0]

        assert input_tensor.size == sample_tensor.size
        assert input_tensor.stride == sample_tensor.stride
        assert input_tensor.dtype == sample_tensor.dtype
        assert input_tensor.device == sample_tensor.device
        assert input_tensor.supported_ops == sample_tensor.supported_ops

    def test_decompose_preserves_properties(self, gelu_op):
        """Test that decomposition preserves all tensor properties."""
        original = Tensor((1, 2, 3, 4), (24, 12, 4, 1), "bfloat16", "cuda", ["test_ops"])
        inputs = gelu_op.decompose(original)

        input_tensor = inputs[0]
        assert input_tensor.size == original.size
        assert input_tensor.stride == original.stride
        assert input_tensor.dtype == original.dtype
        assert input_tensor.device == original.device
        assert input_tensor.supported_ops == original.supported_ops

    def test_decompose_scalar_tensor(self, gelu_op):
        """Test decomposition of scalar tensors."""
        scalar = Tensor((), (), "float32", "cuda", [])
        inputs = gelu_op.decompose(scalar)

        assert len(inputs) == 1
        input_tensor = inputs[0]
        assert input_tensor.size == ()
        assert input_tensor.stride == ()
        assert input_tensor.dtype == "float32"

    def test_codegen_basic(self, gelu_op, sample_tensor):
        """Test basic code generation."""
        output_name = "output"
        input_names = ["input"]

        code = gelu_op.codegen(output_name, input_names, sample_tensor)
        expected = "output = torch.nn.functional.gelu(input)"

        assert code == expected

    def test_codegen_different_names(self, gelu_op, sample_tensor):
        """Test code generation with different variable names."""
        output_name = "result_tensor"
        input_names = ["x_data"]

        code = gelu_op.codegen(output_name, input_names, sample_tensor)
        expected = "result_tensor = torch.nn.functional.gelu(x_data)"

        assert code == expected

    def test_operator_name(self, gelu_op):
        """Test that operator has correct name."""
        assert gelu_op.name == "gelu"
