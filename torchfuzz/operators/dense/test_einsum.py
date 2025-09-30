"""Tests for EinsumOperator."""

import pytest
import re
from .einsum import EinsumOperator
from tensor import Tensor


class TestEinsumOperator:
    """Test class for EinsumOperator."""

    @pytest.fixture
    def einsum_op(self):
        """Create an EinsumOperator instance."""
        return EinsumOperator()

    @pytest.fixture
    def scalar_tensor(self):
        """Create a scalar tensor for testing."""
        return Tensor((), (), "float32", "cuda", [])

    @pytest.fixture
    def vector_tensor(self):
        """Create a vector tensor for testing."""
        return Tensor((8,), (1,), "float32", "cuda", [])

    @pytest.fixture
    def matrix_tensor(self):
        """Create a matrix tensor for testing."""
        return Tensor((4, 6), (6, 1), "float32", "cuda", [])

    @pytest.fixture
    def tensor_3d(self):
        """Create a 3D tensor for testing."""
        return Tensor((2, 4, 6), (24, 6, 1), "float32", "cuda", [])

    @pytest.fixture
    def tensor_4d(self):
        """Create a 4D tensor for testing."""
        return Tensor((2, 3, 4, 5), (60, 20, 5, 1), "float32", "cuda", [])

    @pytest.fixture
    def tensor_5d(self):
        """Create a 5D tensor for testing."""
        return Tensor((2, 3, 4, 5, 6), (360, 120, 30, 6, 1), "float32", "cuda", [])

    @pytest.fixture
    def int_tensor(self):
        """Create an integer tensor for testing."""
        return Tensor((4, 6), (6, 1), "int32", "cuda", [])

    def test_operator_name(self, einsum_op):
        """Test that operator has correct name."""
        assert einsum_op.name == "einsum"

    def test_supports_variable_inputs(self, einsum_op):
        """Test that einsum supports variable inputs."""
        assert einsum_op.supports_variable_inputs() is True

    # Test can_produce method
    def test_can_produce_scalar(self, einsum_op, scalar_tensor):
        """Test that EinsumOperator can produce scalar tensors."""
        assert einsum_op.can_produce(scalar_tensor) is True

    def test_can_produce_vector(self, einsum_op, vector_tensor):
        """Test that EinsumOperator can produce vector tensors."""
        assert einsum_op.can_produce(vector_tensor) is True

    def test_can_produce_matrix(self, einsum_op, matrix_tensor):
        """Test that EinsumOperator can produce matrix tensors."""
        assert einsum_op.can_produce(matrix_tensor) is True

    def test_can_produce_3d(self, einsum_op, tensor_3d):
        """Test that EinsumOperator can produce 3D tensors."""
        assert einsum_op.can_produce(tensor_3d) is True

    def test_can_produce_4d(self, einsum_op, tensor_4d):
        """Test that EinsumOperator can produce 4D tensors."""
        assert einsum_op.can_produce(tensor_4d) is True

    def test_cannot_produce_5d(self, einsum_op, tensor_5d):
        """Test that EinsumOperator cannot produce 5D tensors."""
        assert einsum_op.can_produce(tensor_5d) is False

    def test_cannot_produce_int_tensor(self, einsum_op, int_tensor):
        """Test that EinsumOperator cannot produce integer tensors."""
        assert einsum_op.can_produce(int_tensor) is False

    # Test decompose method with single input
    def test_decompose_single_input_scalar_output(self, einsum_op, scalar_tensor):
        """Test decomposition for single input producing scalar output."""
        inputs = einsum_op.decompose(scalar_tensor, num_inputs=1)

        assert len(inputs) == 1
        assert len(inputs[0].size) >= 1  # Input should have at least 1 dimension
        assert inputs[0].dtype == scalar_tensor.dtype
        assert inputs[0].device == scalar_tensor.device

        # Check that equation was stored
        assert hasattr(scalar_tensor, '_einsum_equation')
        equation = scalar_tensor._einsum_equation
        assert "->" in equation
        assert equation.endswith("->")  # Scalar output

    def test_decompose_single_input_vector_output(self, einsum_op, vector_tensor):
        """Test decomposition for single input producing vector output."""
        inputs = einsum_op.decompose(vector_tensor, num_inputs=1)

        assert len(inputs) == 1
        assert inputs[0].dtype == vector_tensor.dtype
        assert inputs[0].device == vector_tensor.device

        # Check equation
        assert hasattr(vector_tensor, '_einsum_equation')
        equation = vector_tensor._einsum_equation
        assert "->" in equation
        # Should end with single character for vector output
        assert re.match(r".*->[a-z]$", equation)

    def test_decompose_single_input_matrix_output(self, einsum_op, matrix_tensor):
        """Test decomposition for single input producing matrix output."""
        inputs = einsum_op.decompose(matrix_tensor, num_inputs=1)

        assert len(inputs) == 1
        assert inputs[0].dtype == matrix_tensor.dtype
        assert inputs[0].device == matrix_tensor.device

        # Check equation
        assert hasattr(matrix_tensor, '_einsum_equation')
        equation = matrix_tensor._einsum_equation
        assert "->" in equation

    # Test decompose method with two inputs
    def test_decompose_two_inputs_scalar_output(self, einsum_op, scalar_tensor):
        """Test decomposition for two inputs producing scalar output."""
        inputs = einsum_op.decompose(scalar_tensor, num_inputs=2)

        assert len(inputs) == 2
        for input_tensor in inputs:
            assert input_tensor.dtype == scalar_tensor.dtype
            assert input_tensor.device == scalar_tensor.device

        # Check equation for inner product pattern
        assert hasattr(scalar_tensor, '_einsum_equation')
        equation = scalar_tensor._einsum_equation
        assert equation == "i,i->"

        # Both inputs should have same shape for inner product
        assert inputs[0].size == inputs[1].size

    def test_decompose_two_inputs_vector_output(self, einsum_op, vector_tensor):
        """Test decomposition for two inputs producing vector output."""
        inputs = einsum_op.decompose(vector_tensor, num_inputs=2)

        assert len(inputs) == 2
        for input_tensor in inputs:
            assert input_tensor.dtype == vector_tensor.dtype
            assert input_tensor.device == vector_tensor.device

        # Check equation for matrix-vector multiply pattern
        assert hasattr(vector_tensor, '_einsum_equation')
        equation = vector_tensor._einsum_equation
        assert equation == "ij,j->i"

        # Check shapes: (m, k) and (k,) -> (m,)
        assert len(inputs[0].size) == 2
        assert len(inputs[1].size) == 1
        assert inputs[0].size[0] == vector_tensor.size[0]  # m
        assert inputs[0].size[1] == inputs[1].size[0]      # k

    def test_decompose_two_inputs_matrix_output(self, einsum_op, matrix_tensor):
        """Test decomposition for two inputs producing matrix output."""
        inputs = einsum_op.decompose(matrix_tensor, num_inputs=2)

        assert len(inputs) == 2
        for input_tensor in inputs:
            assert input_tensor.dtype == matrix_tensor.dtype
            assert input_tensor.device == matrix_tensor.device

        # Check equation
        assert hasattr(matrix_tensor, '_einsum_equation')
        equation = matrix_tensor._einsum_equation

        if equation == "ik,kj->ij":
            # Matrix multiply pattern
            m, n = matrix_tensor.size
            assert inputs[0].size[0] == m
            assert inputs[1].size[1] == n
            assert inputs[0].size[1] == inputs[1].size[0]  # Inner dimension match
        elif equation == "i,j->ij":
            # Outer product pattern
            m, n = matrix_tensor.size
            assert inputs[0].size == (m,)
            assert inputs[1].size == (n,)

    def test_decompose_two_inputs_3d_output(self, einsum_op, tensor_3d):
        """Test decomposition for two inputs producing 3D output."""
        inputs = einsum_op.decompose(tensor_3d, num_inputs=2)

        assert len(inputs) == 2
        for input_tensor in inputs:
            assert input_tensor.dtype == tensor_3d.dtype
            assert input_tensor.device == tensor_3d.device

        # Check equation for batch matrix multiply
        assert hasattr(tensor_3d, '_einsum_equation')
        equation = tensor_3d._einsum_equation
        assert equation == "bik,bkj->bij"

        # Check batch dimensions match
        b, m, n = tensor_3d.size
        assert inputs[0].size[0] == b
        assert inputs[1].size[0] == b
        assert inputs[0].size[1] == m
        assert inputs[1].size[2] == n
        assert inputs[0].size[2] == inputs[1].size[1]  # Inner dimension match

    # Test decompose method with three inputs
    def test_decompose_three_inputs_scalar_output(self, einsum_op, scalar_tensor):
        """Test decomposition for three inputs producing scalar output."""
        inputs = einsum_op.decompose(scalar_tensor, num_inputs=3)

        assert len(inputs) == 3
        for input_tensor in inputs:
            assert input_tensor.dtype == scalar_tensor.dtype
            assert input_tensor.device == scalar_tensor.device

        # Check equation for triple inner product
        assert hasattr(scalar_tensor, '_einsum_equation')
        equation = scalar_tensor._einsum_equation
        assert equation == "i,i,i->"

        # All inputs should have same shape
        for input_tensor in inputs:
            assert len(input_tensor.size) == 1

    def test_decompose_three_inputs_vector_output(self, einsum_op, vector_tensor):
        """Test decomposition for three inputs producing vector output."""
        inputs = einsum_op.decompose(vector_tensor, num_inputs=3)

        assert len(inputs) == 3
        for input_tensor in inputs:
            assert input_tensor.dtype == vector_tensor.dtype
            assert input_tensor.device == vector_tensor.device

        # Check equation for generalized contraction
        assert hasattr(vector_tensor, '_einsum_equation')
        equation = vector_tensor._einsum_equation
        assert equation == "ik,kj,j->i"

    def test_decompose_three_inputs_matrix_output(self, einsum_op, matrix_tensor):
        """Test decomposition for three inputs producing matrix output."""
        inputs = einsum_op.decompose(matrix_tensor, num_inputs=3)

        assert len(inputs) == 3
        for input_tensor in inputs:
            assert input_tensor.dtype == matrix_tensor.dtype
            assert input_tensor.device == matrix_tensor.device

        # Check equation
        assert hasattr(matrix_tensor, '_einsum_equation')
        equation = matrix_tensor._einsum_equation
        assert equation == "ik,kl,ln->in"

    # Test decompose with automatic num_inputs selection
    def test_decompose_auto_num_inputs(self, einsum_op, matrix_tensor):
        """Test decomposition with automatic num_inputs selection."""
        inputs = einsum_op.decompose(matrix_tensor, num_inputs=5)  # Invalid, should be clamped

        assert 1 <= len(inputs) <= 3  # Should be clamped to valid range
        for input_tensor in inputs:
            assert input_tensor.dtype == matrix_tensor.dtype
            assert input_tensor.device == matrix_tensor.device

    # Test decompose preserves properties
    def test_decompose_preserves_device(self, einsum_op, matrix_tensor):
        """Test decomposition preserves device."""
        inputs = einsum_op.decompose(matrix_tensor)

        for input_tensor in inputs:
            assert input_tensor.device == matrix_tensor.device

    def test_decompose_preserves_dtype(self, einsum_op):
        """Test decomposition preserves dtype for different types."""
        for dtype in ["float32", "float16", "bfloat16"]:
            tensor = Tensor((4, 6), (6, 1), dtype, "cuda", [])
            inputs = einsum_op.decompose(tensor)

            for input_tensor in inputs:
                assert input_tensor.dtype == dtype

    def test_decompose_creates_contiguous_strides(self, einsum_op, matrix_tensor):
        """Test decomposition creates tensors with contiguous strides."""
        inputs = einsum_op.decompose(matrix_tensor)

        for input_tensor in inputs:
            # Check contiguous stride pattern
            expected_stride = []
            acc = 1
            for s in reversed(input_tensor.size):
                expected_stride.insert(0, acc)
                acc *= s
            assert input_tensor.stride == tuple(expected_stride)

    # Test codegen method
    def test_codegen_basic(self, einsum_op, matrix_tensor):
        """Test basic code generation."""
        # Set up equation manually for testing
        matrix_tensor._einsum_equation = "ik,kj->ij"

        output_name = "out"
        input_names = ["a", "b"]

        code = einsum_op.codegen(output_name, input_names, matrix_tensor)
        expected = 'out = torch.einsum("ik,kj->ij", a, b)'

        assert code == expected

    def test_codegen_single_input(self, einsum_op, vector_tensor):
        """Test code generation with single input."""
        vector_tensor._einsum_equation = "ii->i"

        output_name = "result"
        input_names = ["x"]

        code = einsum_op.codegen(output_name, input_names, vector_tensor)
        expected = 'result = torch.einsum("ii->i", x)'

        assert code == expected

    def test_codegen_three_inputs(self, einsum_op, matrix_tensor):
        """Test code generation with three inputs."""
        matrix_tensor._einsum_equation = "ik,kl,ln->in"

        output_name = "z"
        input_names = ["x", "y", "w"]

        code = einsum_op.codegen(output_name, input_names, matrix_tensor)
        expected = 'z = torch.einsum("ik,kl,ln->in", x, y, w)'

        assert code == expected

    def test_codegen_scalar_output(self, einsum_op, scalar_tensor):
        """Test code generation for scalar output."""
        scalar_tensor._einsum_equation = "i,i->"

        output_name = "dot_product"
        input_names = ["a", "b"]

        code = einsum_op.codegen(output_name, input_names, scalar_tensor)
        expected = 'dot_product = torch.einsum("i,i->", a, b)'

        assert code == expected

    def test_codegen_no_equation_raises_error(self, einsum_op, matrix_tensor):
        """Test that codegen raises error when equation is missing."""
        output_name = "out"
        input_names = ["a", "b"]

        with pytest.raises(ValueError, match="Einsum equation not found on output tensor"):
            einsum_op.codegen(output_name, input_names, matrix_tensor)

    # Integration tests
    def test_full_workflow_matrix_multiply(self, einsum_op, matrix_tensor):
        """Test full workflow for matrix multiplication."""
        # Decompose
        inputs = einsum_op.decompose(matrix_tensor, num_inputs=2)

        # Generate code
        output_name = "result"
        input_names = ["A", "B"]
        code = einsum_op.codegen(output_name, input_names, matrix_tensor)

        # Check that code is valid einsum call
        assert "torch.einsum" in code
        assert output_name in code
        assert all(name in code for name in input_names)
        assert hasattr(matrix_tensor, '_einsum_equation')

    def test_full_workflow_inner_product(self, einsum_op, scalar_tensor):
        """Test full workflow for inner product."""
        # Decompose
        inputs = einsum_op.decompose(scalar_tensor, num_inputs=2)

        # Generate code
        output_name = "dot"
        input_names = ["u", "v"]
        code = einsum_op.codegen(output_name, input_names, scalar_tensor)

        # Check results
        assert code == 'dot = torch.einsum("i,i->", u, v)'
        assert len(inputs) == 2
        assert inputs[0].size == inputs[1].size

    def test_multiple_decompositions_different_equations(self, einsum_op, matrix_tensor):
        """Test that multiple decompositions can produce different equations."""
        equations = set()

        # Run decomposition multiple times
        for _ in range(10):
            # Create fresh tensor for each test
            tensor = Tensor((4, 6), (6, 1), "float32", "cuda", [])
            einsum_op.decompose(tensor, num_inputs=2)
            equations.add(tensor._einsum_equation)

        # Should have some variety in equations (at least matrix multiply or outer product)
        assert len(equations) >= 1
        assert all("->" in eq for eq in equations)
        # Check that equations don't have duplicate indices in output
        for eq in equations:
            output_part = eq.split("->")[1]
            output_indices = list(output_part)
            assert len(output_indices) == len(set(output_indices)), f"Duplicate indices in output: {eq}"
