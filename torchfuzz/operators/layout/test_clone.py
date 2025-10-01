"""Tests for CloneOperator."""

import pytest
from .clone import CloneOperator
from tensor import Tensor


class TestCloneOperator:
    """Test class for CloneOperator."""

    @pytest.fixture
    def clone_op(self):
        """Create a CloneOperator instance."""
        return CloneOperator()

    @pytest.fixture
    def scalar_tensor(self):
        """Create a scalar tensor for testing."""
        return Tensor((), (), "float32", "cuda", [])

    @pytest.fixture
    def vector_tensor(self):
        """Create a vector tensor for testing."""
        return Tensor((5,), (1,), "float32", "cuda", [])

    @pytest.fixture
    def matrix_tensor(self):
        """Create a matrix tensor for testing."""
        return Tensor((3, 4), (4, 1), "float32", "cuda", [])

    @pytest.fixture
    def tensor_3d(self):
        """Create a 3D tensor for testing."""
        return Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", [])

    @pytest.fixture
    def tensor_4d(self):
        """Create a 4D tensor for testing."""
        return Tensor((2, 3, 4, 5), (60, 20, 5, 1), "float32", "cuda", [])

    def test_operator_name(self, clone_op):
        """Test that operator has correct name."""
        assert clone_op.name == "clone"

    # Test can_produce method
    def test_can_produce_returns_true_for_all_tensors(self, clone_op):
        """Test that CloneOperator can produce any tensor."""
        test_tensors = [
            Tensor((), (), "float32", "cuda", []),  # scalar
            Tensor((5,), (1,), "float32", "cuda", []),  # 1D
            Tensor((2, 3), (3, 1), "float32", "cuda", []),  # 2D
            Tensor((2, 3, 4), (12, 4, 1), "float32", "cuda", []),  # 3D
            Tensor((1, 1, 1, 1), (1, 1, 1, 1), "float16", "cuda", []),  # 4D
            Tensor((0, 5), (5, 1), "float32", "cuda", []),  # with zero dimension
        ]

        for tensor in test_tensors:
            assert clone_op.can_produce(tensor) is True

    def test_can_produce_different_dtypes(self, clone_op):
        """Test that CloneOperator can produce tensors of different dtypes."""
        dtypes = ["float32", "float16", "bfloat16", "int32", "int64", "bool"]
        for dtype in dtypes:
            tensor = Tensor((2, 3), (3, 1), dtype, "cuda", [])
            assert clone_op.can_produce(tensor) is True

    def test_can_produce_different_devices(self, clone_op):
        """Test that CloneOperator can produce tensors on different devices."""
        devices = ["cuda", "cpu"]
        for device in devices:
            tensor = Tensor((2, 3), (3, 1), "float32", device, [])
            assert clone_op.can_produce(tensor) is True

    # Test decompose method
    def test_decompose_returns_single_input(self, clone_op, matrix_tensor):
        """Test that decomposition returns exactly one input tensor."""
        inputs = clone_op.decompose(matrix_tensor)
        assert len(inputs) == 1

    def test_decompose_preserves_shape(self, clone_op, matrix_tensor):
        """Test that decomposition preserves the tensor shape."""
        inputs = clone_op.decompose(matrix_tensor)
        input_tensor = inputs[0]
        assert input_tensor.size == matrix_tensor.size

    def test_decompose_preserves_stride(self, clone_op, matrix_tensor):
        """Test that decomposition preserves the tensor stride."""
        inputs = clone_op.decompose(matrix_tensor)
        input_tensor = inputs[0]
        assert input_tensor.stride == matrix_tensor.stride

    def test_decompose_preserves_dtype(self, clone_op, matrix_tensor):
        """Test that decomposition preserves the tensor dtype."""
        inputs = clone_op.decompose(matrix_tensor)
        input_tensor = inputs[0]
        assert input_tensor.dtype == matrix_tensor.dtype

    def test_decompose_preserves_device(self, clone_op, matrix_tensor):
        """Test that decomposition preserves the tensor device."""
        inputs = clone_op.decompose(matrix_tensor)
        input_tensor = inputs[0]
        assert input_tensor.device == matrix_tensor.device

    def test_decompose_preserves_supported_ops(self, clone_op):
        """Test that decomposition preserves the supported_ops."""
        tensor = Tensor((3, 4), (4, 1), "float32", "cuda", ["test_ops"])
        inputs = clone_op.decompose(tensor)
        input_tensor = inputs[0]
        assert input_tensor.supported_ops == tensor.supported_ops

    def test_decompose_scalar_tensor(self, clone_op, scalar_tensor):
        """Test decomposition of scalar tensors."""
        inputs = clone_op.decompose(scalar_tensor)

        input_tensor = inputs[0]
        assert input_tensor.size == ()
        assert input_tensor.stride == ()
        assert input_tensor.dtype == scalar_tensor.dtype
        assert input_tensor.device == scalar_tensor.device

    def test_decompose_1d_tensor(self, clone_op, vector_tensor):
        """Test decomposition of 1D tensors."""
        inputs = clone_op.decompose(vector_tensor)

        input_tensor = inputs[0]
        assert input_tensor.size == vector_tensor.size
        assert input_tensor.stride == vector_tensor.stride
        assert input_tensor.dtype == vector_tensor.dtype
        assert input_tensor.device == vector_tensor.device

    def test_decompose_2d_tensor(self, clone_op, matrix_tensor):
        """Test decomposition of 2D tensors."""
        inputs = clone_op.decompose(matrix_tensor)

        input_tensor = inputs[0]
        assert input_tensor.size == matrix_tensor.size
        assert input_tensor.stride == matrix_tensor.stride
        assert input_tensor.dtype == matrix_tensor.dtype
        assert input_tensor.device == matrix_tensor.device

    def test_decompose_3d_tensor(self, clone_op, tensor_3d):
        """Test decomposition of 3D tensors."""
        inputs = clone_op.decompose(tensor_3d)

        input_tensor = inputs[0]
        assert input_tensor.size == tensor_3d.size
        assert input_tensor.stride == tensor_3d.stride
        assert input_tensor.dtype == tensor_3d.dtype
        assert input_tensor.device == tensor_3d.device

    def test_decompose_4d_tensor(self, clone_op, tensor_4d):
        """Test decomposition of 4D tensors."""
        inputs = clone_op.decompose(tensor_4d)

        input_tensor = inputs[0]
        assert input_tensor.size == tensor_4d.size
        assert input_tensor.stride == tensor_4d.stride
        assert input_tensor.dtype == tensor_4d.dtype
        assert input_tensor.device == tensor_4d.device

    def test_decompose_different_dtypes(self, clone_op):
        """Test decomposition preserves different dtypes."""
        dtypes = ["float32", "float16", "bfloat16", "int32", "int64", "bool"]
        for dtype in dtypes:
            tensor = Tensor((2, 3), (3, 1), dtype, "cuda", [])
            inputs = clone_op.decompose(tensor)
            input_tensor = inputs[0]
            assert input_tensor.dtype == dtype

    def test_decompose_different_devices(self, clone_op):
        """Test decomposition preserves different devices."""
        devices = ["cuda", "cpu"]
        for device in devices:
            tensor = Tensor((2, 3), (3, 1), "float32", device, [])
            inputs = clone_op.decompose(tensor)
            input_tensor = inputs[0]
            assert input_tensor.device == device

    def test_decompose_non_contiguous_stride(self, clone_op):
        """Test decomposition preserves non-contiguous strides."""
        # Create tensor with non-contiguous strides
        tensor = Tensor((2, 3), (6, 2), "float32", "cuda", [])  # Non-standard strides
        inputs = clone_op.decompose(tensor)

        input_tensor = inputs[0]
        assert input_tensor.size == tensor.size
        assert input_tensor.stride == tensor.stride  # Should preserve exact strides

    def test_decompose_tensor_with_zero_dimension(self, clone_op):
        """Test decomposition with tensor containing zero-size dimension."""
        tensor = Tensor((0, 5), (5, 1), "float32", "cuda", [])
        inputs = clone_op.decompose(tensor)

        input_tensor = inputs[0]
        assert input_tensor.size == tensor.size
        assert input_tensor.stride == tensor.stride

    def test_decompose_large_tensor(self, clone_op):
        """Test decomposition with larger tensors."""
        tensor = Tensor((8, 9, 10, 11), (990, 110, 11, 1), "float32", "cuda", [])
        inputs = clone_op.decompose(tensor)

        input_tensor = inputs[0]
        assert input_tensor.size == tensor.size
        assert input_tensor.stride == tensor.stride
        assert input_tensor.dtype == tensor.dtype
        assert input_tensor.device == tensor.device

    def test_decompose_consistency(self, clone_op, matrix_tensor):
        """Test that multiple decompositions produce identical results."""
        inputs1 = clone_op.decompose(matrix_tensor)
        inputs2 = clone_op.decompose(matrix_tensor)

        # Both decompositions should produce identical input tensors
        assert inputs1[0].size == inputs2[0].size
        assert inputs1[0].stride == inputs2[0].stride
        assert inputs1[0].dtype == inputs2[0].dtype
        assert inputs1[0].device == inputs2[0].device

    # Test codegen method
    def test_codegen_basic(self, clone_op, matrix_tensor):
        """Test basic code generation."""
        output_name = "output"
        input_names = ["input"]

        code = clone_op.codegen(output_name, input_names, matrix_tensor)
        expected = "output = input.clone()"

        assert code == expected

    def test_codegen_scalar_tensor(self, clone_op, scalar_tensor):
        """Test code generation for scalar tensor."""
        output_name = "result"
        input_names = ["x"]

        code = clone_op.codegen(output_name, input_names, scalar_tensor)
        expected = "result = x.clone()"

        assert code == expected

    def test_codegen_1d_tensor(self, clone_op, vector_tensor):
        """Test code generation for 1D tensor."""
        output_name = "out"
        input_names = ["data"]

        code = clone_op.codegen(output_name, input_names, vector_tensor)
        expected = "out = data.clone()"

        assert code == expected

    def test_codegen_3d_tensor(self, clone_op, tensor_3d):
        """Test code generation for 3D tensor."""
        output_name = "result"
        input_names = ["data"]

        code = clone_op.codegen(output_name, input_names, tensor_3d)
        expected = "result = data.clone()"

        assert code == expected

    def test_codegen_different_variable_names(self, clone_op, matrix_tensor):
        """Test code generation with different variable names."""
        output_name = "tensor_copy"
        input_names = ["original_tensor"]

        code = clone_op.codegen(output_name, input_names, matrix_tensor)
        expected = "tensor_copy = original_tensor.clone()"

        assert code == expected

    def test_codegen_high_dimensional_tensor(self, clone_op):
        """Test code generation for high-dimensional tensor."""
        tensor = Tensor((2, 3, 4, 5, 6), (360, 120, 30, 6, 1), "float32", "cuda", [])
        output_name = "cloned"
        input_names = ["original"]

        code = clone_op.codegen(output_name, input_names, tensor)
        expected = "cloned = original.clone()"

        assert code == expected

    def test_codegen_various_dtypes(self, clone_op):
        """Test code generation works with various dtypes."""
        dtypes = ["float32", "float16", "bfloat16", "int32", "int64", "bool"]
        for dtype in dtypes:
            tensor = Tensor((2, 3), (3, 1), dtype, "cuda", [])
            output_name = "result"
            input_names = ["input"]

            code = clone_op.codegen(output_name, input_names, tensor)
            expected = "result = input.clone()"

            assert code == expected

    def test_codegen_various_devices(self, clone_op):
        """Test code generation works with various devices."""
        devices = ["cuda", "cpu"]
        for device in devices:
            tensor = Tensor((2, 3), (3, 1), "float32", device, [])
            output_name = "result"
            input_names = ["input"]

            code = clone_op.codegen(output_name, input_names, tensor)
            expected = "result = input.clone()"

            assert code == expected

    # Integration tests
    def test_full_workflow(self, clone_op, matrix_tensor):
        """Test full workflow from decomposition to code generation."""
        # Decompose
        inputs = clone_op.decompose(matrix_tensor)

        # Generate code
        output_name = "cloned_tensor"
        input_names = ["source_tensor"]
        code = clone_op.codegen(output_name, input_names, matrix_tensor)

        # Verify results
        assert len(inputs) == 1
        assert inputs[0].size == matrix_tensor.size
        assert inputs[0].stride == matrix_tensor.stride
        assert inputs[0].dtype == matrix_tensor.dtype
        assert inputs[0].device == matrix_tensor.device

        assert code == "cloned_tensor = source_tensor.clone()"

    def test_clone_creates_exact_copy_specs(self, clone_op):
        """Test that clone creates input with exactly identical specifications."""
        # Test with various tensor configurations
        test_cases = [
            # (size, stride, dtype, device)
            ((), (), "float32", "cuda"),  # scalar
            ((5,), (1,), "float16", "cuda"),  # 1D
            ((2, 3), (3, 1), "bfloat16", "cuda"),  # 2D contiguous
            ((2, 3), (6, 2), "float32", "cuda"),  # 2D non-contiguous
            ((2, 3, 4), (12, 4, 1), "int32", "cuda"),  # 3D
            ((1, 1, 1, 1), (1, 1, 1, 1), "int64", "cuda"),  # 4D with 1s
            ((0, 5), (5, 1), "bool", "cuda"),  # with zero dimension
        ]

        for size, stride, dtype, device in test_cases:
            tensor = Tensor(size, stride, dtype, device, [])
            inputs = clone_op.decompose(tensor)

            input_tensor = inputs[0]
            assert input_tensor.size == size
            assert input_tensor.stride == stride
            assert input_tensor.dtype == dtype
            assert input_tensor.device == device

    def test_clone_supports_all_tensor_types(self, clone_op):
        """Test that clone works with all possible tensor configurations."""
        # Various combinations that should all work
        configurations = [
            # Different shapes
            ((), "float32", "cuda"),
            ((1,), "float32", "cuda"),
            ((100,), "float32", "cuda"),
            ((1, 1), "float32", "cuda"),
            ((10, 20), "float32", "cuda"),
            ((2, 3, 4), "float32", "cuda"),
            ((1, 1, 1, 1), "float32", "cuda"),

            # Different dtypes
            ((2, 3), "float16", "cuda"),
            ((2, 3), "bfloat16", "cuda"),
            ((2, 3), "int32", "cuda"),
            ((2, 3), "int64", "cuda"),
            ((2, 3), "bool", "cuda"),

            # Different devices
            ((2, 3), "float32", "cpu"),
        ]

        for size, dtype, device in configurations:
            # Calculate contiguous stride
            stride = []
            acc = 1
            for s in reversed(size):
                stride.insert(0, acc)
                acc *= s
            stride = tuple(stride)

            tensor = Tensor(size, stride, dtype, device, [])

            # Should be able to produce and decompose
            assert clone_op.can_produce(tensor) is True
            inputs = clone_op.decompose(tensor)
            assert len(inputs) == 1
            assert inputs[0].size == size
            assert inputs[0].dtype == dtype
            assert inputs[0].device == device
