r"""
Array expressions are expressions representing N-dimensional arrays, without
evaluating them. These expressions represent in a certain way abstract syntax
trees of operations on N-dimensional arrays.

Every N-dimensional array operator has a corresponding array expression object.

Table of correspondences:

=============================== =============================
         Array operator           Array expression operator
=============================== =============================
        tensorproduct                 ArrayTensorProduct
        tensorcontraction             ArrayContraction
        tensordiagonal                ArrayDiagonal
        permutedims                   PermuteDims
=============================== =============================

Examples
========

``ArraySymbol`` objects are the N-dimensional equivalent of ``MatrixSymbol``
objects in the matrix module:

>>> from sympy.tensor.array.expressions import ArraySymbol
>>> from sympy.abc import i, j, k
>>> A = ArraySymbol("A", (3, 2, 4))
>>> A.shape
(3, 2, 4)
>>> A[i, j, k]
A[i, j, k]
>>> A.as_explicit()
[[[A[0, 0, 0], A[0, 0, 1], A[0, 0, 2], A[0, 0, 3]],
  [A[0, 1, 0], A[0, 1, 1], A[0, 1, 2], A[0, 1, 3]]],
 [[A[1, 0, 0], A[1, 0, 1], A[1, 0, 2], A[1, 0, 3]],
  [A[1, 1, 0], A[1, 1, 1], A[1, 1, 2], A[1, 1, 3]]],
 [[A[2, 0, 0], A[2, 0, 1], A[2, 0, 2], A[2, 0, 3]],
  [A[2, 1, 0], A[2, 1, 1], A[2, 1, 2], A[2, 1, 3]]]]

Component-explicit arrays can be added inside array expressions:

>>> from sympy import Array
>>> from sympy import tensorproduct
>>> from sympy.tensor.array.expressions import ArrayTensorProduct
>>> a = Array([1, 2, 3])
>>> b = Array([i, j, k])
>>> expr = ArrayTensorProduct(a, b, b)
>>> expr
ArrayTensorProduct([1, 2, 3], [i, j, k], [i, j, k])
>>> expr.as_explicit() == tensorproduct(a, b, b)
True

Constructing array expressions from index-explicit forms
--------------------------------------------------------

Array expressions are index-implicit. This means they do not use any indices to
represent array operations. The function ``convert_indexed_to_array( ... )``
may be used to convert index-explicit expressions to array expressions.
It takes as input two parameters: the index-explicit expression and the order
of the indices:

>>> from sympy.tensor.array.expressions import convert_indexed_to_array
>>> from sympy import Sum
>>> A = ArraySymbol("A", (3, 3))
>>> B = ArraySymbol("B", (3, 3))
>>> convert_indexed_to_array(A[i, j], [i, j])
A
>>> convert_indexed_to_array(A[i, j], [j, i])
PermuteDims(A, (0 1))
>>> convert_indexed_to_array(A[i, j] + B[j, i], [i, j])
ArrayAdd(A, PermuteDims(B, (0 1)))
>>> convert_indexed_to_array(Sum(A[i, j]*B[j, k], (j, 0, 2)), [i, k])
ArrayContraction(ArrayTensorProduct(A, B), (1, 2))

The diagonal of a matrix in the array expression form:

>>> convert_indexed_to_array(A[i, i], [i])
ArrayDiagonal(A, (0, 1))

The trace of a matrix in the array expression form:

>>> convert_indexed_to_array(Sum(A[i, i], (i, 0, 2)), [i])
ArrayContraction(A, (0, 1))

Compatibility with matrices
---------------------------

Array expressions can be mixed with objects from the matrix module:

>>> from sympy import MatrixSymbol
>>> from sympy.tensor.array.expressions import ArrayContraction
>>> M = MatrixSymbol("M", 3, 3)
>>> N = MatrixSymbol("N", 3, 3)

Express the matrix product in the array expression form:

>>> from sympy.tensor.array.expressions import convert_matrix_to_array
>>> expr = convert_matrix_to_array(M*N)
>>> expr
ArrayContraction(ArrayTensorProduct(M, N), (1, 2))

The expression can be converted back to matrix form:

>>> from sympy.tensor.array.expressions import convert_array_to_matrix
>>> convert_array_to_matrix(expr)
M*N

Add a second contraction on the remaining axes in order to get the trace of `M \cdot N`:

>>> expr_tr = ArrayContraction(expr, (0, 1))
>>> expr_tr
ArrayContraction(ArrayContraction(ArrayTensorProduct(M, N), (1, 2)), (0, 1))

Flatten the expression by calling ``.doit()`` and remove the nested array contraction operations:

>>> expr_tr.doit()
ArrayContraction(ArrayTensorProduct(M, N), (0, 3), (1, 2))

Get the explicit form of the array expression:

>>> expr.as_explicit()
[[M[0, 0]*N[0, 0] + M[0, 1]*N[1, 0] + M[0, 2]*N[2, 0], M[0, 0]*N[0, 1] + M[0, 1]*N[1, 1] + M[0, 2]*N[2, 1], M[0, 0]*N[0, 2] + M[0, 1]*N[1, 2] + M[0, 2]*N[2, 2]],
 [M[1, 0]*N[0, 0] + M[1, 1]*N[1, 0] + M[1, 2]*N[2, 0], M[1, 0]*N[0, 1] + M[1, 1]*N[1, 1] + M[1, 2]*N[2, 1], M[1, 0]*N[0, 2] + M[1, 1]*N[1, 2] + M[1, 2]*N[2, 2]],
 [M[2, 0]*N[0, 0] + M[2, 1]*N[1, 0] + M[2, 2]*N[2, 0], M[2, 0]*N[0, 1] + M[2, 1]*N[1, 1] + M[2, 2]*N[2, 1], M[2, 0]*N[0, 2] + M[2, 1]*N[1, 2] + M[2, 2]*N[2, 2]]]

Express the trace of a matrix:

>>> from sympy import Trace
>>> convert_matrix_to_array(Trace(M))
ArrayContraction(M, (0, 1))
>>> convert_matrix_to_array(Trace(M*N))
ArrayContraction(ArrayTensorProduct(M, N), (0, 3), (1, 2))

Express the transposition of a matrix (will be expressed as a permutation of the axes:

>>> convert_matrix_to_array(M.T)
PermuteDims(M, (0 1))

Compute the derivative array expressions:

>>> from sympy.tensor.array.expressions import array_derive
>>> d = array_derive(M, M)
>>> d
PermuteDims(ArrayTensorProduct(I, I), (3)(1 2))

Verify that the derivative corresponds to the form computed with explicit matrices:

>>> d.as_explicit()
[[[[1, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 1, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 1], [0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [1, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 1], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0], [1, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 1, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 1]]]]
>>> Me = M.as_explicit()
>>> Me.diff(Me)
[[[[1, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 1, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 1], [0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [1, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 1], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0], [1, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 1, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 1]]]]

"""

__all__ = [
    "ArraySymbol", "ArrayElement", "ZeroArray", "OneArray",
    "ArrayTensorProduct",
    "ArrayContraction",
    "ArrayDiagonal",
    "PermuteDims",
    "ArrayAdd",
    "ArrayElementwiseApplyFunc",
    "Reshape",
    "convert_array_to_matrix",
    "convert_matrix_to_array",
    "convert_array_to_indexed",
    "convert_indexed_to_array",
    "array_derive",
]

from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct, ArrayAdd, PermuteDims, ArrayDiagonal, \
    ArrayContraction, Reshape, ArraySymbol, ArrayElement, ZeroArray, OneArray, ArrayElementwiseApplyFunc
from sympy.tensor.array.expressions.arrayexpr_derivatives import array_derive
from sympy.tensor.array.expressions.from_array_to_indexed import convert_array_to_indexed
from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix
from sympy.tensor.array.expressions.from_indexed_to_array import convert_indexed_to_array
from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
