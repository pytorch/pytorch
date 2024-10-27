import random
from sympy.core.function import Derivative
from sympy.core.symbol import symbols
from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct, ArrayAdd, \
    PermuteDims, ArrayDiagonal
from sympy.core.relational import Eq, Ne, Ge, Gt, Le, Lt
from sympy.external import import_module
from sympy.functions import \
    Abs, ceiling, exp, floor, sign, sin, asin, sqrt, cos, \
    acos, tan, atan, atan2, cosh, acosh, sinh, asinh, tanh, atanh, \
    re, im, arg, erf, loggamma, log
from sympy.matrices import Matrix, MatrixBase, eye, randMatrix
from sympy.matrices.expressions import \
    Determinant, HadamardProduct, Inverse, MatrixSymbol, Trace
from sympy.printing.tensorflow import tensorflow_code
from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
from sympy.utilities.lambdify import lambdify
from sympy.testing.pytest import skip
from sympy.testing.pytest import XFAIL


tf = tensorflow = import_module("tensorflow")

if tensorflow:
    # Hide Tensorflow warnings
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


M = MatrixSymbol("M", 3, 3)
N = MatrixSymbol("N", 3, 3)
P = MatrixSymbol("P", 3, 3)
Q = MatrixSymbol("Q", 3, 3)

x, y, z, t = symbols("x y z t")

if tf is not None:
    llo = [list(range(i, i+3)) for i in range(0, 9, 3)]
    m3x3 = tf.constant(llo)
    m3x3sympy = Matrix(llo)


def _compare_tensorflow_matrix(variables, expr, use_float=False):
    f = lambdify(variables, expr, 'tensorflow')
    if not use_float:
        random_matrices = [randMatrix(v.rows, v.cols) for v in variables]
    else:
        random_matrices = [randMatrix(v.rows, v.cols)/100. for v in variables]

    graph = tf.Graph()
    r = None
    with graph.as_default():
        random_variables = [eval(tensorflow_code(i)) for i in random_matrices]
        session = tf.compat.v1.Session(graph=graph)
        r = session.run(f(*random_variables))

    e = expr.subs(dict(zip(variables, random_matrices)))
    e = e.doit()
    if e.is_Matrix:
        if not isinstance(e, MatrixBase):
            e = e.as_explicit()
        e = e.tolist()

    if not use_float:
        assert (r == e).all()
    else:
        r = [i for row in r for i in row]
        e = [i for row in e for i in row]
        assert all(
            abs(a-b) < 10**-(4-int(log(abs(a), 10))) for a, b in zip(r, e))


# Creating a custom inverse test.
# See https://github.com/sympy/sympy/issues/18469
def _compare_tensorflow_matrix_inverse(variables, expr, use_float=False):
    f = lambdify(variables, expr, 'tensorflow')
    if not use_float:
        random_matrices = [eye(v.rows, v.cols)*4 for v in variables]
    else:
        random_matrices = [eye(v.rows, v.cols)*3.14 for v in variables]

    graph = tf.Graph()
    r = None
    with graph.as_default():
        random_variables = [eval(tensorflow_code(i)) for i in random_matrices]
        session = tf.compat.v1.Session(graph=graph)
        r = session.run(f(*random_variables))

    e = expr.subs(dict(zip(variables, random_matrices)))
    e = e.doit()
    if e.is_Matrix:
        if not isinstance(e, MatrixBase):
            e = e.as_explicit()
        e = e.tolist()

    if not use_float:
        assert (r == e).all()
    else:
        r = [i for row in r for i in row]
        e = [i for row in e for i in row]
        assert all(
            abs(a-b) < 10**-(4-int(log(abs(a), 10))) for a, b in zip(r, e))


def _compare_tensorflow_matrix_scalar(variables, expr):
    f = lambdify(variables, expr, 'tensorflow')
    random_matrices = [
        randMatrix(v.rows, v.cols).evalf() / 100 for v in variables]

    graph = tf.Graph()
    r = None
    with graph.as_default():
        random_variables = [eval(tensorflow_code(i)) for i in random_matrices]
        session = tf.compat.v1.Session(graph=graph)
        r = session.run(f(*random_variables))

    e = expr.subs(dict(zip(variables, random_matrices)))
    e = e.doit()
    assert abs(r-e) < 10**-6


def _compare_tensorflow_scalar(
    variables, expr, rng=lambda: random.randint(0, 10)):
    f = lambdify(variables, expr, 'tensorflow')
    rvs = [rng() for v in variables]

    graph = tf.Graph()
    r = None
    with graph.as_default():
        tf_rvs = [eval(tensorflow_code(i)) for i in rvs]
        session = tf.compat.v1.Session(graph=graph)
        r = session.run(f(*tf_rvs))

    e = expr.subs(dict(zip(variables, rvs))).evalf().doit()
    assert abs(r-e) < 10**-6


def _compare_tensorflow_relational(
    variables, expr, rng=lambda: random.randint(0, 10)):
    f = lambdify(variables, expr, 'tensorflow')
    rvs = [rng() for v in variables]

    graph = tf.Graph()
    r = None
    with graph.as_default():
        tf_rvs = [eval(tensorflow_code(i)) for i in rvs]
        session = tf.compat.v1.Session(graph=graph)
        r = session.run(f(*tf_rvs))

    e = expr.subs(dict(zip(variables, rvs))).doit()
    assert r == e


def test_tensorflow_printing():
    assert tensorflow_code(eye(3)) == \
        "tensorflow.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])"

    expr = Matrix([[x, sin(y)], [exp(z), -t]])
    assert tensorflow_code(expr) == \
        "tensorflow.Variable(" \
            "[[x, tensorflow.math.sin(y)]," \
            " [tensorflow.math.exp(z), -t]])"


# This (random) test is XFAIL because it fails occasionally
# See https://github.com/sympy/sympy/issues/18469
@XFAIL
def test_tensorflow_math():
    if not tf:
        skip("TensorFlow not installed")

    expr = Abs(x)
    assert tensorflow_code(expr) == "tensorflow.math.abs(x)"
    _compare_tensorflow_scalar((x,), expr)

    expr = sign(x)
    assert tensorflow_code(expr) == "tensorflow.math.sign(x)"
    _compare_tensorflow_scalar((x,), expr)

    expr = ceiling(x)
    assert tensorflow_code(expr) == "tensorflow.math.ceil(x)"
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.random())

    expr = floor(x)
    assert tensorflow_code(expr) == "tensorflow.math.floor(x)"
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.random())

    expr = exp(x)
    assert tensorflow_code(expr) == "tensorflow.math.exp(x)"
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.random())

    expr = sqrt(x)
    assert tensorflow_code(expr) == "tensorflow.math.sqrt(x)"
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.random())

    expr = x ** 4
    assert tensorflow_code(expr) == "tensorflow.math.pow(x, 4)"
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.random())

    expr = cos(x)
    assert tensorflow_code(expr) == "tensorflow.math.cos(x)"
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.random())

    expr = acos(x)
    assert tensorflow_code(expr) == "tensorflow.math.acos(x)"
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.uniform(0, 0.95))

    expr = sin(x)
    assert tensorflow_code(expr) == "tensorflow.math.sin(x)"
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.random())

    expr = asin(x)
    assert tensorflow_code(expr) == "tensorflow.math.asin(x)"
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.random())

    expr = tan(x)
    assert tensorflow_code(expr) == "tensorflow.math.tan(x)"
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.random())

    expr = atan(x)
    assert tensorflow_code(expr) == "tensorflow.math.atan(x)"
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.random())

    expr = atan2(y, x)
    assert tensorflow_code(expr) == "tensorflow.math.atan2(y, x)"
    _compare_tensorflow_scalar((y, x), expr, rng=lambda: random.random())

    expr = cosh(x)
    assert tensorflow_code(expr) == "tensorflow.math.cosh(x)"
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.random())

    expr = acosh(x)
    assert tensorflow_code(expr) == "tensorflow.math.acosh(x)"
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.uniform(1, 2))

    expr = sinh(x)
    assert tensorflow_code(expr) == "tensorflow.math.sinh(x)"
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.uniform(1, 2))

    expr = asinh(x)
    assert tensorflow_code(expr) == "tensorflow.math.asinh(x)"
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.uniform(1, 2))

    expr = tanh(x)
    assert tensorflow_code(expr) == "tensorflow.math.tanh(x)"
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.uniform(1, 2))

    expr = atanh(x)
    assert tensorflow_code(expr) == "tensorflow.math.atanh(x)"
    _compare_tensorflow_scalar(
        (x,), expr, rng=lambda: random.uniform(-.5, .5))

    expr = erf(x)
    assert tensorflow_code(expr) == "tensorflow.math.erf(x)"
    _compare_tensorflow_scalar(
        (x,), expr, rng=lambda: random.random())

    expr = loggamma(x)
    assert tensorflow_code(expr) == "tensorflow.math.lgamma(x)"
    _compare_tensorflow_scalar(
        (x,), expr, rng=lambda: random.random())


def test_tensorflow_complexes():
    assert tensorflow_code(re(x)) == "tensorflow.math.real(x)"
    assert tensorflow_code(im(x)) == "tensorflow.math.imag(x)"
    assert tensorflow_code(arg(x)) == "tensorflow.math.angle(x)"


def test_tensorflow_relational():
    if not tf:
        skip("TensorFlow not installed")

    expr = Eq(x, y)
    assert tensorflow_code(expr) == "tensorflow.math.equal(x, y)"
    _compare_tensorflow_relational((x, y), expr)

    expr = Ne(x, y)
    assert tensorflow_code(expr) == "tensorflow.math.not_equal(x, y)"
    _compare_tensorflow_relational((x, y), expr)

    expr = Ge(x, y)
    assert tensorflow_code(expr) == "tensorflow.math.greater_equal(x, y)"
    _compare_tensorflow_relational((x, y), expr)

    expr = Gt(x, y)
    assert tensorflow_code(expr) == "tensorflow.math.greater(x, y)"
    _compare_tensorflow_relational((x, y), expr)

    expr = Le(x, y)
    assert tensorflow_code(expr) == "tensorflow.math.less_equal(x, y)"
    _compare_tensorflow_relational((x, y), expr)

    expr = Lt(x, y)
    assert tensorflow_code(expr) == "tensorflow.math.less(x, y)"
    _compare_tensorflow_relational((x, y), expr)


# This (random) test is XFAIL because it fails occasionally
# See https://github.com/sympy/sympy/issues/18469
@XFAIL
def test_tensorflow_matrices():
    if not tf:
        skip("TensorFlow not installed")

    expr = M
    assert tensorflow_code(expr) == "M"
    _compare_tensorflow_matrix((M,), expr)

    expr = M + N
    assert tensorflow_code(expr) == "tensorflow.math.add(M, N)"
    _compare_tensorflow_matrix((M, N), expr)

    expr = M * N
    assert tensorflow_code(expr) == "tensorflow.linalg.matmul(M, N)"
    _compare_tensorflow_matrix((M, N), expr)

    expr = HadamardProduct(M, N)
    assert tensorflow_code(expr) == "tensorflow.math.multiply(M, N)"
    _compare_tensorflow_matrix((M, N), expr)

    expr = M*N*P*Q
    assert tensorflow_code(expr) == \
        "tensorflow.linalg.matmul(" \
            "tensorflow.linalg.matmul(" \
                "tensorflow.linalg.matmul(M, N), P), Q)"
    _compare_tensorflow_matrix((M, N, P, Q), expr)

    expr = M**3
    assert tensorflow_code(expr) == \
        "tensorflow.linalg.matmul(tensorflow.linalg.matmul(M, M), M)"
    _compare_tensorflow_matrix((M,), expr)

    expr = Trace(M)
    assert tensorflow_code(expr) == "tensorflow.linalg.trace(M)"
    _compare_tensorflow_matrix((M,), expr)

    expr = Determinant(M)
    assert tensorflow_code(expr) == "tensorflow.linalg.det(M)"
    _compare_tensorflow_matrix_scalar((M,), expr)

    expr = Inverse(M)
    assert tensorflow_code(expr) == "tensorflow.linalg.inv(M)"
    _compare_tensorflow_matrix_inverse((M,), expr, use_float=True)

    expr = M.T
    assert tensorflow_code(expr, tensorflow_version='1.14') == \
        "tensorflow.linalg.matrix_transpose(M)"
    assert tensorflow_code(expr, tensorflow_version='1.13') == \
        "tensorflow.matrix_transpose(M)"

    _compare_tensorflow_matrix((M,), expr)


def test_codegen_einsum():
    if not tf:
        skip("TensorFlow not installed")

    graph = tf.Graph()
    with graph.as_default():
        session = tf.compat.v1.Session(graph=graph)

        M = MatrixSymbol("M", 2, 2)
        N = MatrixSymbol("N", 2, 2)

        cg = convert_matrix_to_array(M * N)
        f = lambdify((M, N), cg, 'tensorflow')

        ma = tf.constant([[1, 2], [3, 4]])
        mb = tf.constant([[1,-2], [-1, 3]])
        y = session.run(f(ma, mb))
        c = session.run(tf.matmul(ma, mb))
        assert (y == c).all()


def test_codegen_extra():
    if not tf:
        skip("TensorFlow not installed")

    graph = tf.Graph()
    with graph.as_default():
        session = tf.compat.v1.Session()

        M = MatrixSymbol("M", 2, 2)
        N = MatrixSymbol("N", 2, 2)
        P = MatrixSymbol("P", 2, 2)
        Q = MatrixSymbol("Q", 2, 2)
        ma = tf.constant([[1, 2], [3, 4]])
        mb = tf.constant([[1,-2], [-1, 3]])
        mc = tf.constant([[2, 0], [1, 2]])
        md = tf.constant([[1,-1], [4, 7]])

        cg = ArrayTensorProduct(M, N)
        assert tensorflow_code(cg) == \
            'tensorflow.linalg.einsum("ab,cd", M, N)'
        f = lambdify((M, N), cg, 'tensorflow')
        y = session.run(f(ma, mb))
        c = session.run(tf.einsum("ij,kl", ma, mb))
        assert (y == c).all()

        cg = ArrayAdd(M, N)
        assert tensorflow_code(cg) == 'tensorflow.math.add(M, N)'
        f = lambdify((M, N), cg, 'tensorflow')
        y = session.run(f(ma, mb))
        c = session.run(ma + mb)
        assert (y == c).all()

        cg = ArrayAdd(M, N, P)
        assert tensorflow_code(cg) == \
            'tensorflow.math.add(tensorflow.math.add(M, N), P)'
        f = lambdify((M, N, P), cg, 'tensorflow')
        y = session.run(f(ma, mb, mc))
        c = session.run(ma + mb + mc)
        assert (y == c).all()

        cg = ArrayAdd(M, N, P, Q)
        assert tensorflow_code(cg) == \
            'tensorflow.math.add(' \
                'tensorflow.math.add(tensorflow.math.add(M, N), P), Q)'
        f = lambdify((M, N, P, Q), cg, 'tensorflow')
        y = session.run(f(ma, mb, mc, md))
        c = session.run(ma + mb + mc + md)
        assert (y == c).all()

        cg = PermuteDims(M, [1, 0])
        assert tensorflow_code(cg) == 'tensorflow.transpose(M, [1, 0])'
        f = lambdify((M,), cg, 'tensorflow')
        y = session.run(f(ma))
        c = session.run(tf.transpose(ma))
        assert (y == c).all()

        cg = PermuteDims(ArrayTensorProduct(M, N), [1, 2, 3, 0])
        assert tensorflow_code(cg) == \
            'tensorflow.transpose(' \
                'tensorflow.linalg.einsum("ab,cd", M, N), [1, 2, 3, 0])'
        f = lambdify((M, N), cg, 'tensorflow')
        y = session.run(f(ma, mb))
        c = session.run(tf.transpose(tf.einsum("ab,cd", ma, mb), [1, 2, 3, 0]))
        assert (y == c).all()

        cg = ArrayDiagonal(ArrayTensorProduct(M, N), (1, 2))
        assert tensorflow_code(cg) == \
            'tensorflow.linalg.einsum("ab,bc->acb", M, N)'
        f = lambdify((M, N), cg, 'tensorflow')
        y = session.run(f(ma, mb))
        c = session.run(tf.einsum("ab,bc->acb", ma, mb))
        assert (y == c).all()


def test_MatrixElement_printing():
    A = MatrixSymbol("A", 1, 3)
    B = MatrixSymbol("B", 1, 3)
    C = MatrixSymbol("C", 1, 3)

    assert tensorflow_code(A[0, 0]) == "A[0, 0]"
    assert tensorflow_code(3 * A[0, 0]) == "3*A[0, 0]"

    F = C[0, 0].subs(C, A - B)
    assert tensorflow_code(F) == "(tensorflow.math.add((-1)*B, A))[0, 0]"


def test_tensorflow_Derivative():
    expr = Derivative(sin(x), x)
    assert tensorflow_code(expr) == \
        "tensorflow.gradients(tensorflow.math.sin(x), x)[0]"
