from sympy import MatrixSymbol, symbols, Sum
from sympy.tensor.array.expressions import conv_array_to_indexed, from_array_to_indexed, ArrayTensorProduct, \
    ArrayContraction, conv_array_to_matrix, from_array_to_matrix, conv_matrix_to_array, from_matrix_to_array, \
    conv_indexed_to_array, from_indexed_to_array
from sympy.testing.pytest import warns
from sympy.utilities.exceptions import SymPyDeprecationWarning


def test_deprecated_conv_module_results():

    M = MatrixSymbol("M", 3, 3)
    N = MatrixSymbol("N", 3, 3)
    i, j, d = symbols("i j d")

    x = ArrayContraction(ArrayTensorProduct(M, N), (1, 2))
    y = Sum(M[i, d]*N[d, j], (d, 0, 2))

    with warns(SymPyDeprecationWarning, test_stacklevel=False):
        assert conv_array_to_indexed.convert_array_to_indexed(x, [i, j]).dummy_eq(from_array_to_indexed.convert_array_to_indexed(x, [i, j]))
        assert conv_array_to_matrix.convert_array_to_matrix(x) == from_array_to_matrix.convert_array_to_matrix(x)
        assert conv_matrix_to_array.convert_matrix_to_array(M*N) == from_matrix_to_array.convert_matrix_to_array(M*N)
        assert conv_indexed_to_array.convert_indexed_to_array(y) == from_indexed_to_array.convert_indexed_to_array(y)
