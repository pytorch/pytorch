from sympy.core.random import randint

from sympy.core.numbers import Integer
from sympy.matrices.dense import (Matrix, ones, zeros)

from sympy.physics.quantum.matrixutils import (
    to_sympy, to_numpy, to_scipy_sparse, matrix_tensor_product,
    matrix_to_zero, matrix_zeros, numpy_ndarray, scipy_sparse_matrix
)

from sympy.external import import_module
from sympy.testing.pytest import skip

m = Matrix([[1, 2], [3, 4]])


def test_sympy_to_sympy():
    assert to_sympy(m) == m


def test_matrix_to_zero():
    assert matrix_to_zero(m) == m
    assert matrix_to_zero(Matrix([[0, 0], [0, 0]])) == Integer(0)

np = import_module('numpy')


def test_to_numpy():
    if not np:
        skip("numpy not installed.")

    result = np.array([[1, 2], [3, 4]], dtype='complex')
    assert (to_numpy(m) == result).all()


def test_matrix_tensor_product():
    if not np:
        skip("numpy not installed.")

    l1 = zeros(4)
    for i in range(16):
        l1[i] = 2**i
    l2 = zeros(4)
    for i in range(16):
        l2[i] = i
    l3 = zeros(2)
    for i in range(4):
        l3[i] = i
    vec = Matrix([1, 2, 3])

    #test for Matrix known 4x4 matrices
    numpyl1 = np.array(l1.tolist())
    numpyl2 = np.array(l2.tolist())
    numpy_product = np.kron(numpyl1, numpyl2)
    args = [l1, l2]
    sympy_product = matrix_tensor_product(*args)
    assert numpy_product.tolist() == sympy_product.tolist()
    numpy_product = np.kron(numpyl2, numpyl1)
    args = [l2, l1]
    sympy_product = matrix_tensor_product(*args)
    assert numpy_product.tolist() == sympy_product.tolist()

    #test for other known matrix of different dimensions
    numpyl2 = np.array(l3.tolist())
    numpy_product = np.kron(numpyl1, numpyl2)
    args = [l1, l3]
    sympy_product = matrix_tensor_product(*args)
    assert numpy_product.tolist() == sympy_product.tolist()
    numpy_product = np.kron(numpyl2, numpyl1)
    args = [l3, l1]
    sympy_product = matrix_tensor_product(*args)
    assert numpy_product.tolist() == sympy_product.tolist()

    #test for non square matrix
    numpyl2 = np.array(vec.tolist())
    numpy_product = np.kron(numpyl1, numpyl2)
    args = [l1, vec]
    sympy_product = matrix_tensor_product(*args)
    assert numpy_product.tolist() == sympy_product.tolist()
    numpy_product = np.kron(numpyl2, numpyl1)
    args = [vec, l1]
    sympy_product = matrix_tensor_product(*args)
    assert numpy_product.tolist() == sympy_product.tolist()

    #test for random matrix with random values that are floats
    random_matrix1 = np.random.rand(randint(1, 5), randint(1, 5))
    random_matrix2 = np.random.rand(randint(1, 5), randint(1, 5))
    numpy_product = np.kron(random_matrix1, random_matrix2)
    args = [Matrix(random_matrix1.tolist()), Matrix(random_matrix2.tolist())]
    sympy_product = matrix_tensor_product(*args)
    assert not (sympy_product - Matrix(numpy_product.tolist())).tolist() > \
        (ones(sympy_product.rows, sympy_product.cols)*epsilon).tolist()

    #test for three matrix kronecker
    sympy_product = matrix_tensor_product(l1, vec, l2)

    numpy_product = np.kron(l1, np.kron(vec, l2))
    assert numpy_product.tolist() == sympy_product.tolist()


scipy = import_module('scipy', import_kwargs={'fromlist': ['sparse']})


def test_to_scipy_sparse():
    if not np:
        skip("numpy not installed.")
    if not scipy:
        skip("scipy not installed.")
    else:
        sparse = scipy.sparse

    result = sparse.csr_matrix([[1, 2], [3, 4]], dtype='complex')
    assert np.linalg.norm((to_scipy_sparse(m) - result).todense()) == 0.0

epsilon = .000001


def test_matrix_zeros_sympy():
    sym = matrix_zeros(4, 4, format='sympy')
    assert isinstance(sym, Matrix)

def test_matrix_zeros_numpy():
    if not np:
        skip("numpy not installed.")

    num = matrix_zeros(4, 4, format='numpy')
    assert isinstance(num, numpy_ndarray)

def test_matrix_zeros_scipy():
    if not np:
        skip("numpy not installed.")
    if not scipy:
        skip("scipy not installed.")

    sci = matrix_zeros(4, 4, format='scipy.sparse')
    assert isinstance(sci, scipy_sparse_matrix)
