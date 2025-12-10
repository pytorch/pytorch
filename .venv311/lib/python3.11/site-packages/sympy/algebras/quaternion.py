from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.relational import is_eq
from sympy.functions.elementary.complexes import (conjugate, im, re, sign)
from sympy.functions.elementary.exponential import (exp, log as ln)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, atan2)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.simplify.trigsimp import trigsimp
from sympy.integrals.integrals import integrate
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.core.sympify import sympify, _sympify
from sympy.core.expr import Expr
from sympy.core.logic import fuzzy_not, fuzzy_or
from sympy.utilities.misc import as_int

from mpmath.libmp.libmpf import prec_to_dps


def _check_norm(elements, norm):
    """validate if input norm is consistent"""
    if norm is not None and norm.is_number:
        if norm.is_positive is False:
            raise ValueError("Input norm must be positive.")

        numerical = all(i.is_number and i.is_real is True for i in elements)
        if numerical and is_eq(norm**2, sum(i**2 for i in elements)) is False:
            raise ValueError("Incompatible value for norm.")


def _is_extrinsic(seq):
    """validate seq and return True if seq is lowercase and False if uppercase"""
    if type(seq) != str:
        raise ValueError('Expected seq to be a string.')
    if len(seq) != 3:
        raise ValueError("Expected 3 axes, got `{}`.".format(seq))

    intrinsic = seq.isupper()
    extrinsic = seq.islower()
    if not (intrinsic or extrinsic):
        raise ValueError("seq must either be fully uppercase (for extrinsic "
                         "rotations), or fully lowercase, for intrinsic "
                         "rotations).")

    i, j, k = seq.lower()
    if (i == j) or (j == k):
        raise ValueError("Consecutive axes must be different")

    bad = set(seq) - set('xyzXYZ')
    if bad:
        raise ValueError("Expected axes from `seq` to be from "
                         "['x', 'y', 'z'] or ['X', 'Y', 'Z'], "
                         "got {}".format(''.join(bad)))

    return extrinsic


class Quaternion(Expr):
    """Provides basic quaternion operations.
    Quaternion objects can be instantiated as ``Quaternion(a, b, c, d)``
    as in $q = a + bi + cj + dk$.

    Parameters
    ==========

    norm : None or number
        Pre-defined quaternion norm. If a value is given, Quaternion.norm
        returns this pre-defined value instead of calculating the norm

    Examples
    ========

    >>> from sympy import Quaternion
    >>> q = Quaternion(1, 2, 3, 4)
    >>> q
    1 + 2*i + 3*j + 4*k

    Quaternions over complex fields can be defined as:

    >>> from sympy import Quaternion
    >>> from sympy import symbols, I
    >>> x = symbols('x')
    >>> q1 = Quaternion(x, x**3, x, x**2, real_field = False)
    >>> q2 = Quaternion(3 + 4*I, 2 + 5*I, 0, 7 + 8*I, real_field = False)
    >>> q1
    x + x**3*i + x*j + x**2*k
    >>> q2
    (3 + 4*I) + (2 + 5*I)*i + 0*j + (7 + 8*I)*k

    Defining symbolic unit quaternions:

    >>> from sympy import Quaternion
    >>> from sympy.abc import w, x, y, z
    >>> q = Quaternion(w, x, y, z, norm=1)
    >>> q
    w + x*i + y*j + z*k
    >>> q.norm()
    1

    References
    ==========

    .. [1] https://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/
    .. [2] https://en.wikipedia.org/wiki/Quaternion

    """
    _op_priority = 11.0

    is_commutative = False

    def __new__(cls, a=0, b=0, c=0, d=0, real_field=True, norm=None):
        a, b, c, d = map(sympify, (a, b, c, d))

        if any(i.is_commutative is False for i in [a, b, c, d]):
            raise ValueError("arguments have to be commutative")
        obj = super().__new__(cls, a, b, c, d)
        obj._real_field = real_field
        obj.set_norm(norm)
        return obj

    def set_norm(self, norm):
        """Sets norm of an already instantiated quaternion.

        Parameters
        ==========

        norm : None or number
            Pre-defined quaternion norm. If a value is given, Quaternion.norm
            returns this pre-defined value instead of calculating the norm

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy.abc import a, b, c, d
        >>> q = Quaternion(a, b, c, d)
        >>> q.norm()
        sqrt(a**2 + b**2 + c**2 + d**2)

        Setting the norm:

        >>> q.set_norm(1)
        >>> q.norm()
        1

        Removing set norm:

        >>> q.set_norm(None)
        >>> q.norm()
        sqrt(a**2 + b**2 + c**2 + d**2)

        """
        norm = sympify(norm)
        _check_norm(self.args, norm)
        self._norm = norm

    @property
    def a(self):
        return self.args[0]

    @property
    def b(self):
        return self.args[1]

    @property
    def c(self):
        return self.args[2]

    @property
    def d(self):
        return self.args[3]

    @property
    def real_field(self):
        return self._real_field

    @property
    def product_matrix_left(self):
        r"""Returns 4 x 4 Matrix equivalent to a Hamilton product from the
        left. This can be useful when treating quaternion elements as column
        vectors. Given a quaternion $q = a + bi + cj + dk$ where a, b, c and d
        are real numbers, the product matrix from the left is:

        .. math::

            M  =  \begin{bmatrix} a  &-b  &-c  &-d \\
                                  b  & a  &-d  & c \\
                                  c  & d  & a  &-b \\
                                  d  &-c  & b  & a \end{bmatrix}

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy.abc import a, b, c, d
        >>> q1 = Quaternion(1, 0, 0, 1)
        >>> q2 = Quaternion(a, b, c, d)
        >>> q1.product_matrix_left
        Matrix([
        [1, 0,  0, -1],
        [0, 1, -1,  0],
        [0, 1,  1,  0],
        [1, 0,  0,  1]])

        >>> q1.product_matrix_left * q2.to_Matrix()
        Matrix([
        [a - d],
        [b - c],
        [b + c],
        [a + d]])

        This is equivalent to:

        >>> (q1 * q2).to_Matrix()
        Matrix([
        [a - d],
        [b - c],
        [b + c],
        [a + d]])
        """
        return Matrix([
                [self.a, -self.b, -self.c, -self.d],
                [self.b, self.a, -self.d, self.c],
                [self.c, self.d, self.a, -self.b],
                [self.d, -self.c, self.b, self.a]])

    @property
    def product_matrix_right(self):
        r"""Returns 4 x 4 Matrix equivalent to a Hamilton product from the
        right. This can be useful when treating quaternion elements as column
        vectors. Given a quaternion $q = a + bi + cj + dk$ where a, b, c and d
        are real numbers, the product matrix from the left is:

        .. math::

            M  =  \begin{bmatrix} a  &-b  &-c  &-d \\
                                  b  & a  & d  &-c \\
                                  c  &-d  & a  & b \\
                                  d  & c  &-b  & a \end{bmatrix}


        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy.abc import a, b, c, d
        >>> q1 = Quaternion(a, b, c, d)
        >>> q2 = Quaternion(1, 0, 0, 1)
        >>> q2.product_matrix_right
        Matrix([
        [1, 0, 0, -1],
        [0, 1, 1, 0],
        [0, -1, 1, 0],
        [1, 0, 0, 1]])

        Note the switched arguments: the matrix represents the quaternion on
        the right, but is still considered as a matrix multiplication from the
        left.

        >>> q2.product_matrix_right * q1.to_Matrix()
        Matrix([
        [ a - d],
        [ b + c],
        [-b + c],
        [ a + d]])

        This is equivalent to:

        >>> (q1 * q2).to_Matrix()
        Matrix([
        [ a - d],
        [ b + c],
        [-b + c],
        [ a + d]])
        """
        return Matrix([
                [self.a, -self.b, -self.c, -self.d],
                [self.b, self.a, self.d, -self.c],
                [self.c, -self.d, self.a, self.b],
                [self.d, self.c, -self.b, self.a]])

    def to_Matrix(self, vector_only=False):
        """Returns elements of quaternion as a column vector.
        By default, a ``Matrix`` of length 4 is returned, with the real part as the
        first element.
        If ``vector_only`` is ``True``, returns only imaginary part as a Matrix of
        length 3.

        Parameters
        ==========

        vector_only : bool
            If True, only imaginary part is returned.
            Default value: False

        Returns
        =======

        Matrix
            A column vector constructed by the elements of the quaternion.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy.abc import a, b, c, d
        >>> q = Quaternion(a, b, c, d)
        >>> q
        a + b*i + c*j + d*k

        >>> q.to_Matrix()
        Matrix([
        [a],
        [b],
        [c],
        [d]])


        >>> q.to_Matrix(vector_only=True)
        Matrix([
        [b],
        [c],
        [d]])

        """
        if vector_only:
            return Matrix(self.args[1:])
        else:
            return Matrix(self.args)

    @classmethod
    def from_Matrix(cls, elements):
        """Returns quaternion from elements of a column vector`.
        If vector_only is True, returns only imaginary part as a Matrix of
        length 3.

        Parameters
        ==========

        elements : Matrix, list or tuple of length 3 or 4. If length is 3,
            assume real part is zero.
            Default value: False

        Returns
        =======

        Quaternion
            A quaternion created from the input elements.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy.abc import a, b, c, d
        >>> q = Quaternion.from_Matrix([a, b, c, d])
        >>> q
        a + b*i + c*j + d*k

        >>> q = Quaternion.from_Matrix([b, c, d])
        >>> q
        0 + b*i + c*j + d*k

        """
        length = len(elements)
        if length != 3 and length != 4:
            raise ValueError("Input elements must have length 3 or 4, got {} "
                             "elements".format(length))

        if length == 3:
            return Quaternion(0, *elements)
        else:
            return Quaternion(*elements)

    @classmethod
    def from_euler(cls, angles, seq):
        """Returns quaternion equivalent to rotation represented by the Euler
        angles, in the sequence defined by ``seq``.

        Parameters
        ==========

        angles : list, tuple or Matrix of 3 numbers
            The Euler angles (in radians).
        seq : string of length 3
            Represents the sequence of rotations.
            For extrinsic rotations, seq must be all lowercase and its elements
            must be from the set ``{'x', 'y', 'z'}``
            For intrinsic rotations, seq must be all uppercase and its elements
            must be from the set ``{'X', 'Y', 'Z'}``

        Returns
        =======

        Quaternion
            The normalized rotation quaternion calculated from the Euler angles
            in the given sequence.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import pi
        >>> q = Quaternion.from_euler([pi/2, 0, 0], 'xyz')
        >>> q
        sqrt(2)/2 + sqrt(2)/2*i + 0*j + 0*k

        >>> q = Quaternion.from_euler([0, pi/2, pi] , 'zyz')
        >>> q
        0 + (-sqrt(2)/2)*i + 0*j + sqrt(2)/2*k

        >>> q = Quaternion.from_euler([0, pi/2, pi] , 'ZYZ')
        >>> q
        0 + sqrt(2)/2*i + 0*j + sqrt(2)/2*k

        """

        if len(angles) != 3:
            raise ValueError("3 angles must be given.")

        extrinsic = _is_extrinsic(seq)
        i, j, k = seq.lower()

        # get elementary basis vectors
        ei = [1 if n == i else 0 for n in 'xyz']
        ej = [1 if n == j else 0 for n in 'xyz']
        ek = [1 if n == k else 0 for n in 'xyz']

        # calculate distinct quaternions
        qi = cls.from_axis_angle(ei, angles[0])
        qj = cls.from_axis_angle(ej, angles[1])
        qk = cls.from_axis_angle(ek, angles[2])

        if extrinsic:
            return trigsimp(qk * qj * qi)
        else:
            return trigsimp(qi * qj * qk)

    def to_euler(self, seq, angle_addition=True, avoid_square_root=False):
        r"""Returns Euler angles representing same rotation as the quaternion,
        in the sequence given by ``seq``. This implements the method described
        in [1]_.

        For degenerate cases (gymbal lock cases), the third angle is
        set to zero.

        Parameters
        ==========

        seq : string of length 3
            Represents the sequence of rotations.
            For extrinsic rotations, seq must be all lowercase and its elements
            must be from the set ``{'x', 'y', 'z'}``
            For intrinsic rotations, seq must be all uppercase and its elements
            must be from the set ``{'X', 'Y', 'Z'}``

        angle_addition : bool
            When True, first and third angles are given as an addition and
            subtraction of two simpler ``atan2`` expressions. When False, the
            first and third angles are each given by a single more complicated
            ``atan2`` expression. This equivalent expression is given by:

            .. math::

                \operatorname{atan_2} (b,a) \pm \operatorname{atan_2} (d,c) =
                \operatorname{atan_2} (bc\pm ad, ac\mp bd)

            Default value: True

        avoid_square_root : bool
            When True, the second angle is calculated with an expression based
            on ``acos``, which is slightly more complicated but avoids a square
            root. When False, second angle is calculated with ``atan2``, which
            is simpler and can be better for numerical reasons (some
            numerical implementations of ``acos`` have problems near zero).
            Default value: False


        Returns
        =======

        Tuple
            The Euler angles calculated from the quaternion

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy.abc import a, b, c, d
        >>> euler = Quaternion(a, b, c, d).to_euler('zyz')
        >>> euler
        (-atan2(-b, c) + atan2(d, a),
         2*atan2(sqrt(b**2 + c**2), sqrt(a**2 + d**2)),
         atan2(-b, c) + atan2(d, a))


        References
        ==========

        .. [1] https://doi.org/10.1371/journal.pone.0276302

        """
        if self.is_zero_quaternion():
            raise ValueError('Cannot convert a quaternion with norm 0.')

        angles = [0, 0, 0]

        extrinsic = _is_extrinsic(seq)
        i, j, k = seq.lower()

        # get index corresponding to elementary basis vectors
        i = 'xyz'.index(i) + 1
        j = 'xyz'.index(j) + 1
        k = 'xyz'.index(k) + 1

        if not extrinsic:
            i, k = k, i

        # check if sequence is symmetric
        symmetric = i == k
        if symmetric:
            k = 6 - i - j

        # parity of the permutation
        sign = (i - j) * (j - k) * (k - i) // 2

        # permutate elements
        elements = [self.a, self.b, self.c, self.d]
        a = elements[0]
        b = elements[i]
        c = elements[j]
        d = elements[k] * sign

        if not symmetric:
            a, b, c, d = a - c, b + d, c + a, d - b

        if avoid_square_root:
            if symmetric:
                n2 = self.norm()**2
                angles[1] = acos((a * a + b * b - c * c - d * d) / n2)
            else:
                n2 = 2 * self.norm()**2
                angles[1] = asin((c * c + d * d - a * a - b * b) / n2)
        else:
            angles[1] = 2 * atan2(sqrt(c * c + d * d), sqrt(a * a + b * b))
            if not symmetric:
                angles[1] -= S.Pi / 2

        # Check for singularities in numerical cases
        case = 0
        if is_eq(c, S.Zero) and is_eq(d, S.Zero):
            case = 1
        if is_eq(a, S.Zero) and is_eq(b, S.Zero):
            case = 2

        if case == 0:
            if angle_addition:
                angles[0] = atan2(b, a) + atan2(d, c)
                angles[2] = atan2(b, a) - atan2(d, c)
            else:
                angles[0] = atan2(b*c + a*d, a*c - b*d)
                angles[2] = atan2(b*c - a*d, a*c + b*d)

        else:  # any degenerate case
            angles[2 * (not extrinsic)] = S.Zero
            if case == 1:
                angles[2 * extrinsic] = 2 * atan2(b, a)
            else:
                angles[2 * extrinsic] = 2 * atan2(d, c)
                angles[2 * extrinsic] *= (-1 if extrinsic else 1)

        # for Tait-Bryan angles
        if not symmetric:
            angles[0] *= sign

        if extrinsic:
            return tuple(angles[::-1])
        else:
            return tuple(angles)

    @classmethod
    def from_axis_angle(cls, vector, angle):
        """Returns a rotation quaternion given the axis and the angle of rotation.

        Parameters
        ==========

        vector : tuple of three numbers
            The vector representation of the given axis.
        angle : number
            The angle by which axis is rotated (in radians).

        Returns
        =======

        Quaternion
            The normalized rotation quaternion calculated from the given axis and the angle of rotation.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import pi, sqrt
        >>> q = Quaternion.from_axis_angle((sqrt(3)/3, sqrt(3)/3, sqrt(3)/3), 2*pi/3)
        >>> q
        1/2 + 1/2*i + 1/2*j + 1/2*k

        """
        (x, y, z) = vector
        norm = sqrt(x**2 + y**2 + z**2)
        (x, y, z) = (x / norm, y / norm, z / norm)
        s = sin(angle * S.Half)
        a = cos(angle * S.Half)
        b = x * s
        c = y * s
        d = z * s

        # note that this quaternion is already normalized by construction:
        # c^2 + (s*x)^2 + (s*y)^2 + (s*z)^2 = c^2 + s^2*(x^2 + y^2 + z^2) = c^2 + s^2 * 1 = c^2 + s^2 = 1
        # so, what we return is a normalized quaternion

        return cls(a, b, c, d)

    @classmethod
    def from_rotation_matrix(cls, M):
        """Returns the equivalent quaternion of a matrix. The quaternion will be normalized
        only if the matrix is special orthogonal (orthogonal and det(M) = 1).

        Parameters
        ==========

        M : Matrix
            Input matrix to be converted to equivalent quaternion. M must be special
            orthogonal (orthogonal and det(M) = 1) for the quaternion to be normalized.

        Returns
        =======

        Quaternion
            The quaternion equivalent to given matrix.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import Matrix, symbols, cos, sin, trigsimp
        >>> x = symbols('x')
        >>> M = Matrix([[cos(x), -sin(x), 0], [sin(x), cos(x), 0], [0, 0, 1]])
        >>> q = trigsimp(Quaternion.from_rotation_matrix(M))
        >>> q
        sqrt(2)*sqrt(cos(x) + 1)/2 + 0*i + 0*j + sqrt(2 - 2*cos(x))*sign(sin(x))/2*k

        """

        absQ = M.det()**Rational(1, 3)

        a = sqrt(absQ + M[0, 0] + M[1, 1] + M[2, 2]) / 2
        b = sqrt(absQ + M[0, 0] - M[1, 1] - M[2, 2]) / 2
        c = sqrt(absQ - M[0, 0] + M[1, 1] - M[2, 2]) / 2
        d = sqrt(absQ - M[0, 0] - M[1, 1] + M[2, 2]) / 2

        b = b * sign(M[2, 1] - M[1, 2])
        c = c * sign(M[0, 2] - M[2, 0])
        d = d * sign(M[1, 0] - M[0, 1])

        return Quaternion(a, b, c, d)

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.add(other*-1)

    def __mul__(self, other):
        return self._generic_mul(self, _sympify(other))

    def __rmul__(self, other):
        return self._generic_mul(_sympify(other), self)

    def __pow__(self, p):
        return self.pow(p)

    def __neg__(self):
        return Quaternion(-self.a, -self.b, -self.c, -self.d)

    def __truediv__(self, other):
        return self * sympify(other)**-1

    def __rtruediv__(self, other):
        return sympify(other) * self**-1

    def _eval_Integral(self, *args):
        return self.integrate(*args)

    def diff(self, *symbols, **kwargs):
        kwargs.setdefault('evaluate', True)
        return self.func(*[a.diff(*symbols, **kwargs) for a  in self.args])

    def add(self, other):
        """Adds quaternions.

        Parameters
        ==========

        other : Quaternion
            The quaternion to add to current (self) quaternion.

        Returns
        =======

        Quaternion
            The resultant quaternion after adding self to other

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import symbols
        >>> q1 = Quaternion(1, 2, 3, 4)
        >>> q2 = Quaternion(5, 6, 7, 8)
        >>> q1.add(q2)
        6 + 8*i + 10*j + 12*k
        >>> q1 + 5
        6 + 2*i + 3*j + 4*k
        >>> x = symbols('x', real = True)
        >>> q1.add(x)
        (x + 1) + 2*i + 3*j + 4*k

        Quaternions over complex fields :

        >>> from sympy import Quaternion
        >>> from sympy import I
        >>> q3 = Quaternion(3 + 4*I, 2 + 5*I, 0, 7 + 8*I, real_field = False)
        >>> q3.add(2 + 3*I)
        (5 + 7*I) + (2 + 5*I)*i + 0*j + (7 + 8*I)*k

        """
        q1 = self
        q2 = sympify(other)

        # If q2 is a number or a SymPy expression instead of a quaternion
        if not isinstance(q2, Quaternion):
            if q1.real_field and q2.is_complex:
                return Quaternion(re(q2) + q1.a, im(q2) + q1.b, q1.c, q1.d)
            elif q2.is_commutative:
                return Quaternion(q1.a + q2, q1.b, q1.c, q1.d)
            else:
                raise ValueError("Only commutative expressions can be added with a Quaternion.")

        return Quaternion(q1.a + q2.a, q1.b + q2.b, q1.c + q2.c, q1.d
                          + q2.d)

    def mul(self, other):
        """Multiplies quaternions.

        Parameters
        ==========

        other : Quaternion or symbol
            The quaternion to multiply to current (self) quaternion.

        Returns
        =======

        Quaternion
            The resultant quaternion after multiplying self with other

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import symbols
        >>> q1 = Quaternion(1, 2, 3, 4)
        >>> q2 = Quaternion(5, 6, 7, 8)
        >>> q1.mul(q2)
        (-60) + 12*i + 30*j + 24*k
        >>> q1.mul(2)
        2 + 4*i + 6*j + 8*k
        >>> x = symbols('x', real = True)
        >>> q1.mul(x)
        x + 2*x*i + 3*x*j + 4*x*k

        Quaternions over complex fields :

        >>> from sympy import Quaternion
        >>> from sympy import I
        >>> q3 = Quaternion(3 + 4*I, 2 + 5*I, 0, 7 + 8*I, real_field = False)
        >>> q3.mul(2 + 3*I)
        (2 + 3*I)*(3 + 4*I) + (2 + 3*I)*(2 + 5*I)*i + 0*j + (2 + 3*I)*(7 + 8*I)*k

        """
        return self._generic_mul(self, _sympify(other))

    @staticmethod
    def _generic_mul(q1, q2):
        """Generic multiplication.

        Parameters
        ==========

        q1 : Quaternion or symbol
        q2 : Quaternion or symbol

        It is important to note that if neither q1 nor q2 is a Quaternion,
        this function simply returns q1 * q2.

        Returns
        =======

        Quaternion
            The resultant quaternion after multiplying q1 and q2

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import Symbol, S
        >>> q1 = Quaternion(1, 2, 3, 4)
        >>> q2 = Quaternion(5, 6, 7, 8)
        >>> Quaternion._generic_mul(q1, q2)
        (-60) + 12*i + 30*j + 24*k
        >>> Quaternion._generic_mul(q1, S(2))
        2 + 4*i + 6*j + 8*k
        >>> x = Symbol('x', real = True)
        >>> Quaternion._generic_mul(q1, x)
        x + 2*x*i + 3*x*j + 4*x*k

        Quaternions over complex fields :

        >>> from sympy import I
        >>> q3 = Quaternion(3 + 4*I, 2 + 5*I, 0, 7 + 8*I, real_field = False)
        >>> Quaternion._generic_mul(q3, 2 + 3*I)
        (2 + 3*I)*(3 + 4*I) + (2 + 3*I)*(2 + 5*I)*i + 0*j + (2 + 3*I)*(7 + 8*I)*k

        """
        # None is a Quaternion:
        if not isinstance(q1, Quaternion) and not isinstance(q2, Quaternion):
            return q1 * q2

        # If q1 is a number or a SymPy expression instead of a quaternion
        if not isinstance(q1, Quaternion):
            if q2.real_field and q1.is_complex:
                return Quaternion(re(q1), im(q1), 0, 0) * q2
            elif q1.is_commutative:
                return Quaternion(q1 * q2.a, q1 * q2.b, q1 * q2.c, q1 * q2.d)
            else:
                raise ValueError("Only commutative expressions can be multiplied with a Quaternion.")

        # If q2 is a number or a SymPy expression instead of a quaternion
        if not isinstance(q2, Quaternion):
            if q1.real_field and q2.is_complex:
                return q1 * Quaternion(re(q2), im(q2), 0, 0)
            elif q2.is_commutative:
                return Quaternion(q2 * q1.a, q2 * q1.b, q2 * q1.c, q2 * q1.d)
            else:
                raise ValueError("Only commutative expressions can be multiplied with a Quaternion.")

        # If any of the quaternions has a fixed norm, pre-compute norm
        if q1._norm is None and q2._norm is None:
            norm = None
        else:
            norm = q1.norm() * q2.norm()

        return Quaternion(-q1.b*q2.b - q1.c*q2.c - q1.d*q2.d + q1.a*q2.a,
                          q1.b*q2.a + q1.c*q2.d - q1.d*q2.c + q1.a*q2.b,
                          -q1.b*q2.d + q1.c*q2.a + q1.d*q2.b + q1.a*q2.c,
                          q1.b*q2.c - q1.c*q2.b + q1.d*q2.a + q1.a * q2.d,
                          norm=norm)

    def _eval_conjugate(self):
        """Returns the conjugate of the quaternion."""
        q = self
        return Quaternion(q.a, -q.b, -q.c, -q.d, norm=q._norm)

    def norm(self):
        """Returns the norm of the quaternion."""
        if self._norm is None:  # check if norm is pre-defined
            q = self
            # trigsimp is used to simplify sin(x)^2 + cos(x)^2 (these terms
            # arise when from_axis_angle is used).
            return sqrt(trigsimp(q.a**2 + q.b**2 + q.c**2 + q.d**2))

        return self._norm

    def normalize(self):
        """Returns the normalized form of the quaternion."""
        q = self
        return q * (1/q.norm())

    def inverse(self):
        """Returns the inverse of the quaternion."""
        q = self
        if not q.norm():
            raise ValueError("Cannot compute inverse for a quaternion with zero norm")
        return conjugate(q) * (1/q.norm()**2)

    def pow(self, p):
        """Finds the pth power of the quaternion.

        Parameters
        ==========

        p : int
            Power to be applied on quaternion.

        Returns
        =======

        Quaternion
            Returns the p-th power of the current quaternion.
            Returns the inverse if p = -1.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q.pow(4)
        668 + (-224)*i + (-336)*j + (-448)*k

        """
        try:
            q, p = self, as_int(p)
        except ValueError:
            return NotImplemented

        if p < 0:
            q, p = q.inverse(), -p

        if p == 1:
            return q

        res = Quaternion(1, 0, 0, 0)
        while p > 0:
            if p & 1:
                res *= q
            q *= q
            p >>= 1

        return res

    def exp(self):
        """Returns the exponential of $q$, given by $e^q$.

        Returns
        =======

        Quaternion
            The exponential of the quaternion.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q.exp()
        E*cos(sqrt(29))
        + 2*sqrt(29)*E*sin(sqrt(29))/29*i
        + 3*sqrt(29)*E*sin(sqrt(29))/29*j
        + 4*sqrt(29)*E*sin(sqrt(29))/29*k

        """
        # exp(q) = e^a(cos||v|| + v/||v||*sin||v||)
        q = self
        vector_norm = sqrt(q.b**2 + q.c**2 + q.d**2)
        a = exp(q.a) * cos(vector_norm)
        b = exp(q.a) * sin(vector_norm) * q.b / vector_norm
        c = exp(q.a) * sin(vector_norm) * q.c / vector_norm
        d = exp(q.a) * sin(vector_norm) * q.d / vector_norm

        return Quaternion(a, b, c, d)

    def log(self):
        r"""Returns the logarithm of the quaternion, given by $\log q$.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q.log()
        log(sqrt(30))
        + 2*sqrt(29)*acos(sqrt(30)/30)/29*i
        + 3*sqrt(29)*acos(sqrt(30)/30)/29*j
        + 4*sqrt(29)*acos(sqrt(30)/30)/29*k

        """
        # log(q) = log||q|| + v/||v||*arccos(a/||q||)
        q = self
        vector_norm = sqrt(q.b**2 + q.c**2 + q.d**2)
        q_norm = q.norm()
        a = ln(q_norm)
        b = q.b * acos(q.a / q_norm) / vector_norm
        c = q.c * acos(q.a / q_norm) / vector_norm
        d = q.d * acos(q.a / q_norm) / vector_norm

        return Quaternion(a, b, c, d)

    def _eval_subs(self, *args):
        elements = [i.subs(*args) for i in self.args]
        norm = self._norm
        if norm is not None:
            norm = norm.subs(*args)
        _check_norm(elements, norm)
        return Quaternion(*elements, norm=norm)

    def _eval_evalf(self, prec):
        """Returns the floating point approximations (decimal numbers) of the quaternion.

        Returns
        =======

        Quaternion
            Floating point approximations of quaternion(self)

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import sqrt
        >>> q = Quaternion(1/sqrt(1), 1/sqrt(2), 1/sqrt(3), 1/sqrt(4))
        >>> q.evalf()
        1.00000000000000
        + 0.707106781186547*i
        + 0.577350269189626*j
        + 0.500000000000000*k

        """
        nprec = prec_to_dps(prec)
        return Quaternion(*[arg.evalf(n=nprec) for arg in self.args])

    def pow_cos_sin(self, p):
        """Computes the pth power in the cos-sin form.

        Parameters
        ==========

        p : int
            Power to be applied on quaternion.

        Returns
        =======

        Quaternion
            The p-th power in the cos-sin form.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q.pow_cos_sin(4)
        900*cos(4*acos(sqrt(30)/30))
        + 1800*sqrt(29)*sin(4*acos(sqrt(30)/30))/29*i
        + 2700*sqrt(29)*sin(4*acos(sqrt(30)/30))/29*j
        + 3600*sqrt(29)*sin(4*acos(sqrt(30)/30))/29*k

        """
        # q = ||q||*(cos(a) + u*sin(a))
        # q^p = ||q||^p * (cos(p*a) + u*sin(p*a))

        q = self
        (v, angle) = q.to_axis_angle()
        q2 = Quaternion.from_axis_angle(v, p * angle)
        return q2 * (q.norm()**p)

    def integrate(self, *args):
        """Computes integration of quaternion.

        Returns
        =======

        Quaternion
            Integration of the quaternion(self) with the given variable.

        Examples
        ========

        Indefinite Integral of quaternion :

        >>> from sympy import Quaternion
        >>> from sympy.abc import x
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q.integrate(x)
        x + 2*x*i + 3*x*j + 4*x*k

        Definite integral of quaternion :

        >>> from sympy import Quaternion
        >>> from sympy.abc import x
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q.integrate((x, 1, 5))
        4 + 8*i + 12*j + 16*k

        """
        return Quaternion(integrate(self.a, *args), integrate(self.b, *args),
                          integrate(self.c, *args), integrate(self.d, *args))

    @staticmethod
    def rotate_point(pin, r):
        """Returns the coordinates of the point pin (a 3 tuple) after rotation.

        Parameters
        ==========

        pin : tuple
            A 3-element tuple of coordinates of a point which needs to be
            rotated.
        r : Quaternion or tuple
            Axis and angle of rotation.

            It's important to note that when r is a tuple, it must be of the form
            (axis, angle)

        Returns
        =======

        tuple
            The coordinates of the point after rotation.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import symbols, trigsimp, cos, sin
        >>> x = symbols('x')
        >>> q = Quaternion(cos(x/2), 0, 0, sin(x/2))
        >>> trigsimp(Quaternion.rotate_point((1, 1, 1), q))
        (sqrt(2)*cos(x + pi/4), sqrt(2)*sin(x + pi/4), 1)
        >>> (axis, angle) = q.to_axis_angle()
        >>> trigsimp(Quaternion.rotate_point((1, 1, 1), (axis, angle)))
        (sqrt(2)*cos(x + pi/4), sqrt(2)*sin(x + pi/4), 1)

        """
        if isinstance(r, tuple):
            # if r is of the form (vector, angle)
            q = Quaternion.from_axis_angle(r[0], r[1])
        else:
            # if r is a quaternion
            q = r.normalize()
        pout = q * Quaternion(0, pin[0], pin[1], pin[2]) * conjugate(q)
        return (pout.b, pout.c, pout.d)

    def to_axis_angle(self):
        """Returns the axis and angle of rotation of a quaternion.

        Returns
        =======

        tuple
            Tuple of (axis, angle)

        Examples
        ========

        >>> from sympy import Quaternion
        >>> q = Quaternion(1, 1, 1, 1)
        >>> (axis, angle) = q.to_axis_angle()
        >>> axis
        (sqrt(3)/3, sqrt(3)/3, sqrt(3)/3)
        >>> angle
        2*pi/3

        """
        q = self
        if q.a.is_negative:
            q = q * -1

        q = q.normalize()
        angle = trigsimp(2 * acos(q.a))

        # Since quaternion is normalised, q.a is less than 1.
        s = sqrt(1 - q.a*q.a)

        x = trigsimp(q.b / s)
        y = trigsimp(q.c / s)
        z = trigsimp(q.d / s)

        v = (x, y, z)
        t = (v, angle)

        return t

    def to_rotation_matrix(self, v=None, homogeneous=True):
        """Returns the equivalent rotation transformation matrix of the quaternion
        which represents rotation about the origin if ``v`` is not passed.

        Parameters
        ==========

        v : tuple or None
            Default value: None
        homogeneous : bool
            When True, gives an expression that may be more efficient for
            symbolic calculations but less so for direct evaluation. Both
            formulas are mathematically equivalent.
            Default value: True

        Returns
        =======

        tuple
            Returns the equivalent rotation transformation matrix of the quaternion
            which represents rotation about the origin if v is not passed.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import symbols, trigsimp, cos, sin
        >>> x = symbols('x')
        >>> q = Quaternion(cos(x/2), 0, 0, sin(x/2))
        >>> trigsimp(q.to_rotation_matrix())
        Matrix([
        [cos(x), -sin(x), 0],
        [sin(x),  cos(x), 0],
        [     0,       0, 1]])

        Generates a 4x4 transformation matrix (used for rotation about a point
        other than the origin) if the point(v) is passed as an argument.
        """

        q = self
        s = q.norm()**-2

        # diagonal elements are different according to parameter normal
        if homogeneous:
            m00 = s*(q.a**2 + q.b**2 - q.c**2 - q.d**2)
            m11 = s*(q.a**2 - q.b**2 + q.c**2 - q.d**2)
            m22 = s*(q.a**2 - q.b**2 - q.c**2 + q.d**2)
        else:
            m00 = 1 - 2*s*(q.c**2 + q.d**2)
            m11 = 1 - 2*s*(q.b**2 + q.d**2)
            m22 = 1 - 2*s*(q.b**2 + q.c**2)

        m01 = 2*s*(q.b*q.c - q.d*q.a)
        m02 = 2*s*(q.b*q.d + q.c*q.a)

        m10 = 2*s*(q.b*q.c + q.d*q.a)
        m12 = 2*s*(q.c*q.d - q.b*q.a)

        m20 = 2*s*(q.b*q.d - q.c*q.a)
        m21 = 2*s*(q.c*q.d + q.b*q.a)

        if not v:
            return Matrix([[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]])

        else:
            (x, y, z) = v

            m03 = x - x*m00 - y*m01 - z*m02
            m13 = y - x*m10 - y*m11 - z*m12
            m23 = z - x*m20 - y*m21 - z*m22
            m30 = m31 = m32 = 0
            m33 = 1

            return Matrix([[m00, m01, m02, m03], [m10, m11, m12, m13],
                          [m20, m21, m22, m23], [m30, m31, m32, m33]])

    def scalar_part(self):
        r"""Returns scalar part($\mathbf{S}(q)$) of the quaternion q.

        Explanation
        ===========

        Given a quaternion $q = a + bi + cj + dk$, returns $\mathbf{S}(q) = a$.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(4, 8, 13, 12)
        >>> q.scalar_part()
        4

        """

        return self.a

    def vector_part(self):
        r"""
        Returns $\mathbf{V}(q)$, the vector part of the quaternion $q$.

        Explanation
        ===========

        Given a quaternion $q = a + bi + cj + dk$, returns $\mathbf{V}(q) = bi + cj + dk$.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(1, 1, 1, 1)
        >>> q.vector_part()
        0 + 1*i + 1*j + 1*k

        >>> q = Quaternion(4, 8, 13, 12)
        >>> q.vector_part()
        0 + 8*i + 13*j + 12*k

        """

        return Quaternion(0, self.b, self.c, self.d)

    def axis(self):
        r"""
        Returns $\mathbf{Ax}(q)$, the axis of the quaternion $q$.

        Explanation
        ===========

        Given a quaternion $q = a + bi + cj + dk$, returns $\mathbf{Ax}(q)$  i.e., the versor of the vector part of that quaternion
        equal to $\mathbf{U}[\mathbf{V}(q)]$.
        The axis is always an imaginary unit with square equal to $-1 + 0i + 0j + 0k$.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(1, 1, 1, 1)
        >>> q.axis()
        0 + sqrt(3)/3*i + sqrt(3)/3*j + sqrt(3)/3*k

        See Also
        ========

        vector_part

        """
        axis = self.vector_part().normalize()

        return Quaternion(0, axis.b, axis.c, axis.d)

    def is_pure(self):
        """
        Returns true if the quaternion is pure, false if the quaternion is not pure
        or returns none if it is unknown.

        Explanation
        ===========

        A pure quaternion (also a vector quaternion) is a quaternion with scalar
        part equal to 0.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(0, 8, 13, 12)
        >>> q.is_pure()
        True

        See Also
        ========
        scalar_part

        """

        return self.a.is_zero

    def is_zero_quaternion(self):
        """
        Returns true if the quaternion is a zero quaternion or false if it is not a zero quaternion
        and None if the value is unknown.

        Explanation
        ===========

        A zero quaternion is a quaternion with both scalar part and
        vector part equal to 0.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(1, 0, 0, 0)
        >>> q.is_zero_quaternion()
        False

        >>> q = Quaternion(0, 0, 0, 0)
        >>> q.is_zero_quaternion()
        True

        See Also
        ========
        scalar_part
        vector_part

        """

        return self.norm().is_zero

    def angle(self):
        r"""
        Returns the angle of the quaternion measured in the real-axis plane.

        Explanation
        ===========

        Given a quaternion $q = a + bi + cj + dk$ where $a$, $b$, $c$ and $d$
        are real numbers, returns the angle of the quaternion given by

        .. math::
            \theta := 2 \operatorname{atan_2}\left(\sqrt{b^2 + c^2 + d^2}, {a}\right)

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(1, 4, 4, 4)
        >>> q.angle()
        2*atan(4*sqrt(3))

        """

        return 2 * atan2(self.vector_part().norm(), self.scalar_part())


    def arc_coplanar(self, other):
        """
        Returns True if the transformation arcs represented by the input quaternions happen in the same plane.

        Explanation
        ===========

        Two quaternions are said to be coplanar (in this arc sense) when their axes are parallel.
        The plane of a quaternion is the one normal to its axis.

        Parameters
        ==========

        other : a Quaternion

        Returns
        =======

        True : if the planes of the two quaternions are the same, apart from its orientation/sign.
        False : if the planes of the two quaternions are not the same, apart from its orientation/sign.
        None : if plane of either of the quaternion is unknown.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q1 = Quaternion(1, 4, 4, 4)
        >>> q2 = Quaternion(3, 8, 8, 8)
        >>> Quaternion.arc_coplanar(q1, q2)
        True

        >>> q1 = Quaternion(2, 8, 13, 12)
        >>> Quaternion.arc_coplanar(q1, q2)
        False

        See Also
        ========

        vector_coplanar
        is_pure

        """
        if (self.is_zero_quaternion()) or (other.is_zero_quaternion()):
            raise ValueError('Neither of the given quaternions can be 0')

        return fuzzy_or([(self.axis() - other.axis()).is_zero_quaternion(), (self.axis() + other.axis()).is_zero_quaternion()])

    @classmethod
    def vector_coplanar(cls, q1, q2, q3):
        r"""
        Returns True if the axis of the pure quaternions seen as 3D vectors
        ``q1``, ``q2``, and ``q3`` are coplanar.

        Explanation
        ===========

        Three pure quaternions are vector coplanar if the quaternions seen as 3D vectors are coplanar.

        Parameters
        ==========

        q1
            A pure Quaternion.
        q2
            A pure Quaternion.
        q3
            A pure Quaternion.

        Returns
        =======

        True : if the axis of the pure quaternions seen as 3D vectors
        q1, q2, and q3 are coplanar.
        False : if the axis of the pure quaternions seen as 3D vectors
        q1, q2, and q3 are not coplanar.
        None : if the axis of the pure quaternions seen as 3D vectors
        q1, q2, and q3 are coplanar is unknown.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q1 = Quaternion(0, 4, 4, 4)
        >>> q2 = Quaternion(0, 8, 8, 8)
        >>> q3 = Quaternion(0, 24, 24, 24)
        >>> Quaternion.vector_coplanar(q1, q2, q3)
        True

        >>> q1 = Quaternion(0, 8, 16, 8)
        >>> q2 = Quaternion(0, 8, 3, 12)
        >>> Quaternion.vector_coplanar(q1, q2, q3)
        False

        See Also
        ========

        axis
        is_pure

        """

        if fuzzy_not(q1.is_pure()) or fuzzy_not(q2.is_pure()) or fuzzy_not(q3.is_pure()):
            raise ValueError('The given quaternions must be pure')

        M = Matrix([[q1.b, q1.c, q1.d], [q2.b, q2.c, q2.d], [q3.b, q3.c, q3.d]]).det()
        return M.is_zero

    def parallel(self, other):
        """
        Returns True if the two pure quaternions seen as 3D vectors are parallel.

        Explanation
        ===========

        Two pure quaternions are called parallel when their vector product is commutative which
        implies that the quaternions seen as 3D vectors have same direction.

        Parameters
        ==========

        other : a Quaternion

        Returns
        =======

        True : if the two pure quaternions seen as 3D vectors are parallel.
        False : if the two pure quaternions seen as 3D vectors are not parallel.
        None : if the two pure quaternions seen as 3D vectors are parallel is unknown.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(0, 4, 4, 4)
        >>> q1 = Quaternion(0, 8, 8, 8)
        >>> q.parallel(q1)
        True

        >>> q1 = Quaternion(0, 8, 13, 12)
        >>> q.parallel(q1)
        False

        """

        if fuzzy_not(self.is_pure()) or fuzzy_not(other.is_pure()):
            raise ValueError('The provided quaternions must be pure')

        return (self*other - other*self).is_zero_quaternion()

    def orthogonal(self, other):
        """
        Returns the orthogonality of two quaternions.

        Explanation
        ===========

        Two pure quaternions are called orthogonal when their product is anti-commutative.

        Parameters
        ==========

        other : a Quaternion

        Returns
        =======

        True : if the two pure quaternions seen as 3D vectors are orthogonal.
        False : if the two pure quaternions seen as 3D vectors are not orthogonal.
        None : if the two pure quaternions seen as 3D vectors are orthogonal is unknown.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(0, 4, 4, 4)
        >>> q1 = Quaternion(0, 8, 8, 8)
        >>> q.orthogonal(q1)
        False

        >>> q1 = Quaternion(0, 2, 2, 0)
        >>> q = Quaternion(0, 2, -2, 0)
        >>> q.orthogonal(q1)
        True

        """

        if fuzzy_not(self.is_pure()) or fuzzy_not(other.is_pure()):
            raise ValueError('The given quaternions must be pure')

        return (self*other + other*self).is_zero_quaternion()

    def index_vector(self):
        r"""
        Returns the index vector of the quaternion.

        Explanation
        ===========

        The index vector is given by $\mathbf{T}(q)$, the norm (or magnitude) of
        the quaternion $q$, multiplied by $\mathbf{Ax}(q)$, the axis of $q$.

        Returns
        =======

        Quaternion: representing index vector of the provided quaternion.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(2, 4, 2, 4)
        >>> q.index_vector()
        0 + 4*sqrt(10)/3*i + 2*sqrt(10)/3*j + 4*sqrt(10)/3*k

        See Also
        ========

        axis
        norm

        """

        return self.norm() * self.axis()

    def mensor(self):
        """
        Returns the natural logarithm of the norm(magnitude) of the quaternion.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(2, 4, 2, 4)
        >>> q.mensor()
        log(2*sqrt(10))
        >>> q.norm()
        2*sqrt(10)

        See Also
        ========

        norm

        """

        return ln(self.norm())
