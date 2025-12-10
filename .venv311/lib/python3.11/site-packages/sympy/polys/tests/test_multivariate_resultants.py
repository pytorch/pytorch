"""Tests for Dixon's and Macaulay's classes. """

from sympy.matrices.dense import Matrix
from sympy.polys.polytools import factor
from sympy.core import symbols
from sympy.tensor.indexed import IndexedBase

from sympy.polys.multivariate_resultants import (DixonResultant,
                                                 MacaulayResultant)

c, d = symbols("a, b")
x, y = symbols("x, y")

p =  c * x + y
q =  x + d * y

dixon = DixonResultant(polynomials=[p, q], variables=[x, y])
macaulay = MacaulayResultant(polynomials=[p, q], variables=[x, y])

def test_dixon_resultant_init():
    """Test init method of DixonResultant."""
    a = IndexedBase("alpha")

    assert dixon.polynomials == [p, q]
    assert dixon.variables == [x, y]
    assert dixon.n == 2
    assert dixon.m == 2
    assert dixon.dummy_variables == [a[0], a[1]]

def test_get_dixon_polynomial_numerical():
    """Test Dixon's polynomial for a numerical example."""
    a = IndexedBase("alpha")

    p = x + y
    q = x ** 2 + y **3
    h = x ** 2 + y

    dixon = DixonResultant([p, q, h], [x, y])
    polynomial = -x * y ** 2 * a[0] - x * y ** 2 * a[1] - x * y * a[0] \
    * a[1] - x * y * a[1] ** 2 - x * a[0] * a[1] ** 2 + x * a[0] - \
    y ** 2 * a[0] * a[1] + y ** 2 * a[1] - y * a[0] * a[1] ** 2 + y * \
    a[1] ** 2

    assert dixon.get_dixon_polynomial().as_expr().expand() == polynomial

def test_get_max_degrees():
    """Tests max degrees function."""

    p = x + y
    q = x ** 2 + y **3
    h = x ** 2 + y

    dixon = DixonResultant(polynomials=[p, q, h], variables=[x, y])
    dixon_polynomial = dixon.get_dixon_polynomial()

    assert dixon.get_max_degrees(dixon_polynomial) == [1, 2]

def test_get_dixon_matrix():
    """Test Dixon's resultant for a numerical example."""

    x, y = symbols('x, y')

    p = x + y
    q = x ** 2 + y ** 3
    h = x ** 2 + y

    dixon = DixonResultant([p, q, h], [x, y])
    polynomial = dixon.get_dixon_polynomial()

    assert dixon.get_dixon_matrix(polynomial).det() == 0

def test_get_dixon_matrix_example_two():
    """Test Dixon's matrix for example from [Palancz08]_."""
    x, y, z = symbols('x, y, z')

    f = x ** 2 + y ** 2 - 1 + z * 0
    g = x ** 2 + z ** 2 - 1 + y * 0
    h = y ** 2 + z ** 2 - 1

    example_two = DixonResultant([f, g, h], [y, z])
    poly = example_two.get_dixon_polynomial()
    matrix = example_two.get_dixon_matrix(poly)

    expr = 1 - 8 * x ** 2 + 24 * x ** 4 - 32 * x ** 6 + 16 * x ** 8
    assert (matrix.det() - expr).expand() == 0

def test_KSY_precondition():
    """Tests precondition for KSY Resultant."""
    A, B, C = symbols('A, B, C')

    m1 = Matrix([[1, 2, 3],
                 [4, 5, 12],
                 [6, 7, 18]])

    m2 = Matrix([[0, C**2],
                 [-2 * C, -C ** 2]])

    m3 = Matrix([[1, 0],
                 [0, 1]])

    m4 = Matrix([[A**2, 0, 1],
                 [A, 1, 1 / A]])

    m5 = Matrix([[5, 1],
                 [2, B],
                 [0, 1],
                 [0, 0]])

    assert dixon.KSY_precondition(m1) == False
    assert dixon.KSY_precondition(m2) == True
    assert dixon.KSY_precondition(m3) == True
    assert dixon.KSY_precondition(m4) == False
    assert dixon.KSY_precondition(m5) == True

def test_delete_zero_rows_and_columns():
    """Tests method for deleting rows and columns containing only zeros."""
    A, B, C = symbols('A, B, C')

    m1 = Matrix([[0, 0],
                 [0, 0],
                 [1, 2]])

    m2 = Matrix([[0, 1, 2],
                 [0, 3, 4],
                 [0, 5, 6]])

    m3 = Matrix([[0, 0, 0, 0],
                 [0, 1, 2, 0],
                 [0, 3, 4, 0],
                 [0, 0, 0, 0]])

    m4 = Matrix([[1, 0, 2],
                 [0, 0, 0],
                 [3, 0, 4]])

    m5 = Matrix([[0, 0, 0, 1],
                 [0, 0, 0, 2],
                 [0, 0, 0, 3],
                 [0, 0, 0, 4]])

    m6 = Matrix([[0, 0, A],
                 [B, 0, 0],
                 [0, 0, C]])

    assert dixon.delete_zero_rows_and_columns(m1) == Matrix([[1, 2]])

    assert dixon.delete_zero_rows_and_columns(m2) == Matrix([[1, 2],
                                                             [3, 4],
                                                             [5, 6]])

    assert dixon.delete_zero_rows_and_columns(m3) == Matrix([[1, 2],
                                                             [3, 4]])

    assert dixon.delete_zero_rows_and_columns(m4) == Matrix([[1, 2],
                                                             [3, 4]])

    assert dixon.delete_zero_rows_and_columns(m5) == Matrix([[1],
                                                             [2],
                                                             [3],
                                                             [4]])

    assert dixon.delete_zero_rows_and_columns(m6) == Matrix([[0, A],
                                                             [B, 0],
                                                             [0, C]])

def test_product_leading_entries():
    """Tests product of leading entries method."""
    A, B = symbols('A, B')

    m1 = Matrix([[1, 2, 3],
                 [0, 4, 5],
                 [0, 0, 6]])

    m2 = Matrix([[0, 0, 1],
                 [2, 0, 3]])

    m3 = Matrix([[0, 0, 0],
                 [1, 2, 3],
                 [0, 0, 0]])

    m4 = Matrix([[0, 0, A],
                 [1, 2, 3],
                 [B, 0, 0]])

    assert dixon.product_leading_entries(m1) == 24
    assert dixon.product_leading_entries(m2) == 2
    assert dixon.product_leading_entries(m3) == 1
    assert dixon.product_leading_entries(m4) == A * B

def test_get_KSY_Dixon_resultant_example_one():
    """Tests the KSY Dixon resultant for example one"""
    x, y, z = symbols('x, y, z')

    p = x * y * z
    q = x**2 - z**2
    h = x + y + z
    dixon = DixonResultant([p, q, h], [x, y])
    dixon_poly = dixon.get_dixon_polynomial()
    dixon_matrix = dixon.get_dixon_matrix(dixon_poly)
    D = dixon.get_KSY_Dixon_resultant(dixon_matrix)

    assert D == -z**3

def test_get_KSY_Dixon_resultant_example_two():
    """Tests the KSY Dixon resultant for example two"""
    x, y, A = symbols('x, y, A')

    p = x * y + x * A + x - A**2 - A + y**2 + y
    q = x**2 + x * A - x + x * y + y * A - y
    h = x**2 + x * y + 2 * x - x * A - y * A - 2 * A

    dixon = DixonResultant([p, q, h], [x, y])
    dixon_poly = dixon.get_dixon_polynomial()
    dixon_matrix = dixon.get_dixon_matrix(dixon_poly)
    D = factor(dixon.get_KSY_Dixon_resultant(dixon_matrix))

    assert D == -8*A*(A - 1)*(A + 2)*(2*A - 1)**2

def test_macaulay_resultant_init():
    """Test init method of MacaulayResultant."""

    assert macaulay.polynomials == [p, q]
    assert macaulay.variables == [x, y]
    assert macaulay.n == 2
    assert macaulay.degrees == [1, 1]
    assert macaulay.degree_m == 1
    assert macaulay.monomials_size == 2

def test_get_degree_m():
    assert macaulay._get_degree_m() == 1

def test_get_size():
    assert macaulay.get_size() == 2

def test_macaulay_example_one():
    """Tests the Macaulay for example from [Bruce97]_"""

    x, y, z = symbols('x, y, z')
    a_1_1, a_1_2, a_1_3 = symbols('a_1_1, a_1_2, a_1_3')
    a_2_2, a_2_3, a_3_3 = symbols('a_2_2, a_2_3, a_3_3')
    b_1_1, b_1_2, b_1_3 = symbols('b_1_1, b_1_2, b_1_3')
    b_2_2, b_2_3, b_3_3 = symbols('b_2_2, b_2_3, b_3_3')
    c_1, c_2, c_3 = symbols('c_1, c_2, c_3')

    f_1 = a_1_1 * x ** 2 + a_1_2 * x * y + a_1_3 * x * z + \
          a_2_2 * y ** 2 + a_2_3 * y * z + a_3_3 * z ** 2
    f_2 = b_1_1 * x ** 2 + b_1_2 * x * y + b_1_3 * x * z + \
          b_2_2 * y ** 2 + b_2_3 * y * z + b_3_3 * z ** 2
    f_3 = c_1 * x + c_2 * y + c_3 * z

    mac = MacaulayResultant([f_1, f_2, f_3], [x, y, z])

    assert mac.degrees == [2, 2, 1]
    assert mac.degree_m == 3

    assert mac.monomial_set == [x ** 3, x ** 2 * y, x ** 2 * z,
                                x * y ** 2,
                                x * y * z, x * z ** 2, y ** 3,
                                y ** 2 *z, y * z ** 2, z ** 3]
    assert mac.monomials_size == 10
    assert mac.get_row_coefficients() == [[x, y, z], [x, y, z],
                                          [x * y, x * z, y * z, z ** 2]]

    matrix = mac.get_matrix()
    assert matrix.shape == (mac.monomials_size, mac.monomials_size)
    assert mac.get_submatrix(matrix) == Matrix([[a_1_1, a_2_2],
                                                [b_1_1, b_2_2]])

def test_macaulay_example_two():
    """Tests the Macaulay formulation for example from [Stiller96]_."""

    x, y, z = symbols('x, y, z')
    a_0, a_1, a_2 = symbols('a_0, a_1, a_2')
    b_0, b_1, b_2 = symbols('b_0, b_1, b_2')
    c_0, c_1, c_2, c_3, c_4 = symbols('c_0, c_1, c_2, c_3, c_4')

    f = a_0 * y -  a_1 * x + a_2 * z
    g = b_1 * x ** 2 + b_0 * y ** 2 - b_2 * z ** 2
    h = c_0 * y - c_1 * x ** 3 + c_2 * x ** 2 * z - c_3 * x * z ** 2 + \
        c_4 * z ** 3

    mac = MacaulayResultant([f, g, h], [x, y, z])

    assert mac.degrees == [1, 2, 3]
    assert mac.degree_m == 4
    assert mac.monomials_size == 15
    assert len(mac.get_row_coefficients()) == mac.n

    matrix = mac.get_matrix()
    assert matrix.shape == (mac.monomials_size, mac.monomials_size)
    assert mac.get_submatrix(matrix) == Matrix([[-a_1, a_0, a_2, 0],
                                                [0, -a_1, 0, 0],
                                                [0, 0, -a_1, 0],
                                                [0, 0, 0, -a_1]])
