import random
import math

from sympy import symbols, Derivative
from sympy.printing.pytorch import torch_code
from sympy import (eye, MatrixSymbol, Matrix)
from sympy.tensor.array import NDimArray
from sympy.tensor.array.expressions.array_expressions import (
    ArrayTensorProduct, ArrayAdd,
    PermuteDims, ArrayDiagonal, _CodegenArrayAbstract)
from sympy.utilities.lambdify import lambdify
from sympy.core.relational import Eq, Ne, Ge, Gt, Le, Lt
from sympy.functions import \
    Abs, ceiling, exp, floor, sign, sin, asin, cos, \
    acos, tan, atan, atan2, cosh, acosh, sinh, asinh, tanh, atanh, \
    re, im, arg, erf, loggamma, sqrt
from sympy.testing.pytest import skip
from sympy.external import import_module
from sympy.matrices.expressions import \
    Determinant, HadamardProduct, Inverse, Trace
from sympy.matrices import randMatrix
from sympy.matrices import Identity, ZeroMatrix, OneMatrix
from sympy import conjugate, I
from sympy import Heaviside, gamma, polygamma



torch = import_module("torch")

M = MatrixSymbol("M", 3, 3)
N = MatrixSymbol("N", 3, 3)
P = MatrixSymbol("P", 3, 3)
Q = MatrixSymbol("Q", 3, 3)

x, y, z, t = symbols("x y z t")

if torch is not None:
    llo = [list(range(i, i + 3)) for i in range(0, 9, 3)]
    m3x3 = torch.tensor(llo, dtype=torch.float64)
    m3x3sympy = Matrix(llo)


def _compare_torch_matrix(variables, expr):
    f = lambdify(variables, expr, 'torch')

    random_matrices = [randMatrix(i.shape[0], i.shape[1]) for i in variables]
    random_variables = [torch.tensor(i.tolist(), dtype=torch.float64) for i in random_matrices]
    r = f(*random_variables)
    e = expr.subs(dict(zip(variables, random_matrices))).doit()

    if isinstance(e, _CodegenArrayAbstract):
        e = e.doit()

    if hasattr(e, 'is_number') and e.is_number:
        if isinstance(r, torch.Tensor) and r.dim() == 0:
            r = r.item()
            e = float(e)
            assert abs(r - e) < 1e-6
            return

    if e.is_Matrix or isinstance(e, NDimArray):
        e = torch.tensor(e.tolist(), dtype=torch.float64)
        assert torch.allclose(r, e, atol=1e-6)
    else:
        raise TypeError(f"Cannot compare {type(r)} with {type(e)}")


def _compare_torch_scalar(variables, expr, rng=lambda: random.uniform(-5, 5)):
    f = lambdify(variables, expr, 'torch')
    rvs = [rng() for v in variables]
    t_rvs = [torch.tensor(i, dtype=torch.float64) for i in rvs]
    r = f(*t_rvs)
    if isinstance(r, torch.Tensor):
        r = r.item()
    e = expr.subs(dict(zip(variables, rvs))).doit()
    assert abs(r - e) < 1e-6


def _compare_torch_relational(variables, expr, rng=lambda: random.randint(0, 10)):
    f = lambdify(variables, expr, 'torch')
    rvs = [rng() for v in variables]
    t_rvs = [torch.tensor(i, dtype=torch.float64) for i in rvs]
    r = f(*t_rvs)
    e = bool(expr.subs(dict(zip(variables, rvs))).doit())
    assert r.item() == e


def test_torch_math():
    if not torch:
        skip("PyTorch not installed")

    expr = Abs(x)
    assert torch_code(expr) == "torch.abs(x)"
    f = lambdify(x, expr, 'torch')
    ma = torch.tensor([[-1, 2, -3, -4]], dtype=torch.float64)
    y_abs = f(ma)
    c = torch.abs(ma)
    assert torch.all(y_abs == c)

    expr = sign(x)
    assert torch_code(expr) == "torch.sign(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.uniform(-10, 10))

    expr = ceiling(x)
    assert torch_code(expr) == "torch.ceil(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.random())

    expr = floor(x)
    assert torch_code(expr) == "torch.floor(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.random())

    expr = exp(x)
    assert torch_code(expr) == "torch.exp(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.uniform(-2, 2))

    expr = sqrt(x)
    assert torch_code(expr) == "torch.sqrt(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.random())

    expr = x ** 4
    assert torch_code(expr) == "torch.pow(x, 4)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.random())

    expr = cos(x)
    assert torch_code(expr) == "torch.cos(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.random())

    expr = acos(x)
    assert torch_code(expr) == "torch.acos(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.uniform(-0.99, 0.99))

    expr = sin(x)
    assert torch_code(expr) == "torch.sin(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.random())

    expr = asin(x)
    assert torch_code(expr) == "torch.asin(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.uniform(-0.99, 0.99))

    expr = tan(x)
    assert torch_code(expr) == "torch.tan(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.uniform(-1.5, 1.5))

    expr = atan(x)
    assert torch_code(expr) == "torch.atan(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.uniform(-5, 5))

    expr = atan2(y, x)
    assert torch_code(expr) == "torch.atan2(y, x)"
    _compare_torch_scalar((y, x), expr, rng=lambda: random.uniform(-5, 5))

    expr = cosh(x)
    assert torch_code(expr) == "torch.cosh(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.uniform(-2, 2))

    expr = acosh(x)
    assert torch_code(expr) == "torch.acosh(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.uniform(1.1, 5))

    expr = sinh(x)
    assert torch_code(expr) == "torch.sinh(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.uniform(-2, 2))

    expr = asinh(x)
    assert torch_code(expr) == "torch.asinh(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.uniform(-5, 5))

    expr = tanh(x)
    assert torch_code(expr) == "torch.tanh(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.uniform(-2, 2))

    expr = atanh(x)
    assert torch_code(expr) == "torch.atanh(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.uniform(-0.9, 0.9))

    expr = erf(x)
    assert torch_code(expr) == "torch.erf(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.uniform(-2, 2))

    expr = loggamma(x)
    assert torch_code(expr) == "torch.lgamma(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.uniform(0.5, 5))


def test_torch_complexes():
    assert torch_code(re(x)) == "torch.real(x)"
    assert torch_code(im(x)) == "torch.imag(x)"
    assert torch_code(arg(x)) == "torch.angle(x)"


def test_torch_relational():
    if not torch:
        skip("PyTorch not installed")

    expr = Eq(x, y)
    assert torch_code(expr) == "torch.eq(x, y)"
    _compare_torch_relational((x, y), expr)

    expr = Ne(x, y)
    assert torch_code(expr) == "torch.ne(x, y)"
    _compare_torch_relational((x, y), expr)

    expr = Ge(x, y)
    assert torch_code(expr) == "torch.ge(x, y)"
    _compare_torch_relational((x, y), expr)

    expr = Gt(x, y)
    assert torch_code(expr) == "torch.gt(x, y)"
    _compare_torch_relational((x, y), expr)

    expr = Le(x, y)
    assert torch_code(expr) == "torch.le(x, y)"
    _compare_torch_relational((x, y), expr)

    expr = Lt(x, y)
    assert torch_code(expr) == "torch.lt(x, y)"
    _compare_torch_relational((x, y), expr)


def test_torch_matrix():
    if torch is None:
        skip("PyTorch not installed")

    expr = M
    assert torch_code(expr) == "M"
    f = lambdify((M,), expr, "torch")
    eye_mat = eye(3)
    eye_tensor = torch.tensor(eye_mat.tolist(), dtype=torch.float64)
    assert torch.allclose(f(eye_tensor), eye_tensor)

    expr = M * N
    assert torch_code(expr) == "torch.matmul(M, N)"
    _compare_torch_matrix((M, N), expr)

    expr = M ** 3
    assert torch_code(expr) == "torch.mm(torch.mm(M, M), M)"
    _compare_torch_matrix((M,), expr)

    expr = M * N * P * Q
    assert torch_code(expr) == "torch.matmul(torch.matmul(torch.matmul(M, N), P), Q)"
    _compare_torch_matrix((M, N, P, Q), expr)

    expr = Trace(M)
    assert torch_code(expr) == "torch.trace(M)"
    _compare_torch_matrix((M,), expr)

    expr = Determinant(M)
    assert torch_code(expr) == "torch.det(M)"
    _compare_torch_matrix((M,), expr)

    expr = HadamardProduct(M, N)
    assert torch_code(expr) == "torch.mul(M, N)"
    _compare_torch_matrix((M, N), expr)

    expr = Inverse(M)
    assert torch_code(expr) == "torch.linalg.inv(M)"

    # For inverse, use a matrix that's guaranteed to be invertible
    eye_mat = eye(3)
    eye_tensor = torch.tensor(eye_mat.tolist(), dtype=torch.float64)
    f = lambdify((M,), expr, "torch")
    result = f(eye_tensor)
    expected = torch.linalg.inv(eye_tensor)
    assert torch.allclose(result, expected)


def test_torch_array_operations():
    if not torch:
        skip("PyTorch not installed")

    M = MatrixSymbol("M", 2, 2)
    N = MatrixSymbol("N", 2, 2)
    P = MatrixSymbol("P", 2, 2)
    Q = MatrixSymbol("Q", 2, 2)

    ma = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
    mb = torch.tensor([[1., -2.], [-1., 3.]], dtype=torch.float64)
    mc = torch.tensor([[2., 0.], [1., 2.]], dtype=torch.float64)
    md = torch.tensor([[1., -1.], [4., 7.]], dtype=torch.float64)

    cg = ArrayTensorProduct(M, N)
    assert torch_code(cg) == 'torch.einsum("ab,cd", M, N)'
    f = lambdify((M, N), cg, 'torch')
    y = f(ma, mb)
    c = torch.einsum("ij,kl", ma, mb)
    assert torch.allclose(y, c)

    cg = ArrayAdd(M, N)
    assert torch_code(cg) == 'torch.add(M, N)'
    f = lambdify((M, N), cg, 'torch')
    y = f(ma, mb)
    c = ma + mb
    assert torch.allclose(y, c)

    cg = ArrayAdd(M, N, P)
    assert torch_code(cg) == 'torch.add(torch.add(M, N), P)'
    f = lambdify((M, N, P), cg, 'torch')
    y = f(ma, mb, mc)
    c = ma + mb + mc
    assert torch.allclose(y, c)

    cg = ArrayAdd(M, N, P, Q)
    assert torch_code(cg) == 'torch.add(torch.add(torch.add(M, N), P), Q)'
    f = lambdify((M, N, P, Q), cg, 'torch')
    y = f(ma, mb, mc, md)
    c = ma + mb + mc + md
    assert torch.allclose(y, c)

    cg = PermuteDims(M, [1, 0])
    assert torch_code(cg) == 'M.permute(1, 0)'
    f = lambdify((M,), cg, 'torch')
    y = f(ma)
    c = ma.T
    assert torch.allclose(y, c)

    cg = PermuteDims(ArrayTensorProduct(M, N), [1, 2, 3, 0])
    assert torch_code(cg) == 'torch.einsum("ab,cd", M, N).permute(1, 2, 3, 0)'
    f = lambdify((M, N), cg, 'torch')
    y = f(ma, mb)
    c = torch.einsum("ab,cd", ma, mb).permute(1, 2, 3, 0)
    assert torch.allclose(y, c)

    cg = ArrayDiagonal(ArrayTensorProduct(M, N), (1, 2))
    assert torch_code(cg) == 'torch.einsum("ab,bc->acb", M, N)'
    f = lambdify((M, N), cg, 'torch')
    y = f(ma, mb)
    c = torch.einsum("ab,bc->acb", ma, mb)
    assert torch.allclose(y, c)


def test_torch_derivative():
    """Test derivative handling."""
    expr = Derivative(sin(x), x)
    assert torch_code(expr) == 'torch.autograd.grad(torch.sin(x), x)[0]'


def test_torch_printing_dtype():
    if not torch:
        skip("PyTorch not installed")

    # matrix printing with default dtype
    expr = Matrix([[x, sin(y)], [exp(z), -t]])
    assert "dtype=torch.float64" in torch_code(expr)

    # explicit dtype
    assert "dtype=torch.float32" in torch_code(expr, dtype="torch.float32")

    # with requires_grad
    result = torch_code(expr, requires_grad=True)
    assert "requires_grad=True" in result
    assert "dtype=torch.float64" in result

    # both
    result = torch_code(expr, requires_grad=True, dtype="torch.float32")
    assert "requires_grad=True" in result
    assert "dtype=torch.float32" in result


def test_requires_grad():
    if not torch:
        skip("PyTorch not installed")

    expr = sin(x) + cos(y)
    f = lambdify([x, y], expr, 'torch')

    # make sure the gradients flow
    x_val = torch.tensor(1.0, requires_grad=True)
    y_val = torch.tensor(2.0, requires_grad=True)
    result = f(x_val, y_val)
    assert result.requires_grad
    result.backward()

    # x_val.grad should be cos(x_val) which is close to cos(1.0)
    assert abs(x_val.grad.item() - float(cos(1.0).evalf())) < 1e-6

    # y_val.grad should be -sin(y_val) which is close to -sin(2.0)
    assert abs(y_val.grad.item() - float(-sin(2.0).evalf())) < 1e-6


def test_torch_multi_variable_derivatives():
    if not torch:
        skip("PyTorch not installed")

    x, y, z = symbols("x y z")

    expr = Derivative(sin(x), x)
    assert torch_code(expr) == "torch.autograd.grad(torch.sin(x), x)[0]"

    expr = Derivative(sin(x), (x, 2))
    assert torch_code(
        expr) == "torch.autograd.grad(torch.autograd.grad(torch.sin(x), x, create_graph=True)[0], x, create_graph=True)[0]"

    expr = Derivative(sin(x * y), x, y)
    result = torch_code(expr)
    expected = "torch.autograd.grad(torch.autograd.grad(torch.sin(x*y), x, create_graph=True)[0], y, create_graph=True)[0]"
    normalized_result = result.replace(" ", "")
    normalized_expected = expected.replace(" ", "")
    assert normalized_result == normalized_expected

    expr = Derivative(sin(x), x, x)
    result = torch_code(expr)
    expected = "torch.autograd.grad(torch.autograd.grad(torch.sin(x), x, create_graph=True)[0], x, create_graph=True)[0]"
    assert result == expected

    expr = Derivative(sin(x * y * z), x, (y, 2), z)
    result = torch_code(expr)
    expected = "torch.autograd.grad(torch.autograd.grad(torch.autograd.grad(torch.autograd.grad(torch.sin(x*y*z), x, create_graph=True)[0], y, create_graph=True)[0], y, create_graph=True)[0], z, create_graph=True)[0]"
    normalized_result = result.replace(" ", "")
    normalized_expected = expected.replace(" ", "")
    assert normalized_result == normalized_expected


def test_torch_derivative_lambdify():
    if not torch:
        skip("PyTorch not installed")

    x = symbols("x")
    y = symbols("y")

    expr = Derivative(x ** 2, x)
    f = lambdify(x, expr, 'torch')
    x_val = torch.tensor(2.0, requires_grad=True)
    result = f(x_val)
    assert torch.isclose(result, torch.tensor(4.0))

    expr = Derivative(sin(x), (x, 2))
    f = lambdify(x, expr, 'torch')
    # Second derivative of sin(x) at x=0 is 0, not -1
    x_val = torch.tensor(0.0, requires_grad=True)
    result = f(x_val)
    assert torch.isclose(result, torch.tensor(0.0), atol=1e-5)

    x_val = torch.tensor(math.pi / 2, requires_grad=True)
    result = f(x_val)
    assert torch.isclose(result, torch.tensor(-1.0), atol=1e-5)

    expr = Derivative(x * y ** 2, x, y)
    f = lambdify((x, y), expr, 'torch')
    x_val = torch.tensor(2.0, requires_grad=True)
    y_val = torch.tensor(3.0, requires_grad=True)
    result = f(x_val, y_val)
    assert torch.isclose(result, torch.tensor(6.0))


def test_torch_special_matrices():
    if not torch:
        skip("PyTorch not installed")

    expr = Identity(3)
    assert torch_code(expr) == "torch.eye(3)"

    n = symbols("n")
    expr = Identity(n)
    assert torch_code(expr) == "torch.eye(n, n)"

    expr = ZeroMatrix(2, 3)
    assert torch_code(expr) == "torch.zeros((2, 3))"

    m, n = symbols("m n")
    expr = ZeroMatrix(m, n)
    assert torch_code(expr) == "torch.zeros((m, n))"

    expr = OneMatrix(2, 3)
    assert torch_code(expr) == "torch.ones((2, 3))"

    expr = OneMatrix(m, n)
    assert torch_code(expr) == "torch.ones((m, n))"


def test_torch_special_matrices_lambdify():
    if not torch:
        skip("PyTorch not installed")

    expr = Identity(3)
    f = lambdify([], expr, 'torch')
    result = f()
    expected = torch.eye(3)
    assert torch.allclose(result, expected)

    expr = ZeroMatrix(2, 3)
    f = lambdify([], expr, 'torch')
    result = f()
    expected = torch.zeros((2, 3))
    assert torch.allclose(result, expected)

    expr = OneMatrix(2, 3)
    f = lambdify([], expr, 'torch')
    result = f()
    expected = torch.ones((2, 3))
    assert torch.allclose(result, expected)


def test_torch_complex_operations():
    if not torch:
        skip("PyTorch not installed")

    expr = conjugate(x)
    assert torch_code(expr) == "torch.conj(x)"

    # SymPy distributes conjugate over addition and applies specific rules for each term
    expr = conjugate(sin(x) + I * cos(y))
    assert torch_code(expr) == "torch.sin(torch.conj(x)) - 1j*torch.cos(torch.conj(y))"

    expr = I
    assert torch_code(expr) == "1j"

    expr = 2 * I + x
    assert torch_code(expr) == "x + 2*1j"

    expr = exp(I * x)
    assert torch_code(expr) == "torch.exp(1j*x)"


def test_torch_special_functions():
    if not torch:
        skip("PyTorch not installed")

    expr = Heaviside(x)
    assert torch_code(expr) == "torch.heaviside(x, 1/2)"

    expr = Heaviside(x, 0)
    assert torch_code(expr) == "torch.heaviside(x, 0)"

    expr = gamma(x)
    assert torch_code(expr) == "torch.special.gamma(x)"

    expr = polygamma(0, x)  # Use polygamma instead of digamma because sympy will default to that anyway
    assert torch_code(expr) == "torch.special.digamma(x)"

    expr = gamma(sin(x))
    assert torch_code(expr) == "torch.special.gamma(torch.sin(x))"
