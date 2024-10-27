from sympy import (S, sympify, expand, sqrt, Add, zeros, acos,
                   ImmutableMatrix as Matrix, simplify)
from sympy.simplify.trigsimp import trigsimp
from sympy.printing.defaults import Printable
from sympy.utilities.misc import filldedent
from sympy.core.evalf import EvalfMixin

from mpmath.libmp.libmpf import prec_to_dps


__all__ = ['Vector']


class Vector(Printable, EvalfMixin):
    """The class used to define vectors.

    It along with ReferenceFrame are the building blocks of describing a
    classical mechanics system in PyDy and sympy.physics.vector.

    Attributes
    ==========

    simp : Boolean
        Let certain methods use trigsimp on their outputs

    """

    simp = False
    is_number = False

    def __init__(self, inlist):
        """This is the constructor for the Vector class. You should not be
        calling this, it should only be used by other functions. You should be
        treating Vectors like you would with if you were doing the math by
        hand, and getting the first 3 from the standard basis vectors from a
        ReferenceFrame.

        The only exception is to create a zero vector:
        zv = Vector(0)

        """

        self.args = []
        if inlist == 0:
            inlist = []
        if isinstance(inlist, dict):
            d = inlist
        else:
            d = {}
            for inp in inlist:
                if inp[1] in d:
                    d[inp[1]] += inp[0]
                else:
                    d[inp[1]] = inp[0]

        for k, v in d.items():
            if v != Matrix([0, 0, 0]):
                self.args.append((v, k))

    @property
    def func(self):
        """Returns the class Vector. """
        return Vector

    def __hash__(self):
        return hash(tuple(self.args))

    def __add__(self, other):
        """The add operator for Vector. """
        if other == 0:
            return self
        other = _check_vector(other)
        return Vector(self.args + other.args)

    def dot(self, other):
        """Dot product of two vectors.

        Returns a scalar, the dot product of the two Vectors

        Parameters
        ==========

        other : Vector
            The Vector which we are dotting with

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, dot
        >>> from sympy import symbols
        >>> q1 = symbols('q1')
        >>> N = ReferenceFrame('N')
        >>> dot(N.x, N.x)
        1
        >>> dot(N.x, N.y)
        0
        >>> A = N.orientnew('A', 'Axis', [q1, N.x])
        >>> dot(N.y, A.y)
        cos(q1)

        """

        from sympy.physics.vector.dyadic import Dyadic, _check_dyadic
        if isinstance(other, Dyadic):
            other = _check_dyadic(other)
            ol = Vector(0)
            for v in other.args:
                ol += v[0] * v[2] * (v[1].dot(self))
            return ol
        other = _check_vector(other)
        out = S.Zero
        for v1 in self.args:
            for v2 in other.args:
                out += ((v2[0].T) * (v2[1].dcm(v1[1])) * (v1[0]))[0]
        if Vector.simp:
            return trigsimp(out, recursive=True)
        else:
            return out

    def __truediv__(self, other):
        """This uses mul and inputs self and 1 divided by other. """
        return self.__mul__(S.One / other)

    def __eq__(self, other):
        """Tests for equality.

        It is very import to note that this is only as good as the SymPy
        equality test; False does not always mean they are not equivalent
        Vectors.
        If other is 0, and self is empty, returns True.
        If other is 0 and self is not empty, returns False.
        If none of the above, only accepts other as a Vector.

        """

        if other == 0:
            other = Vector(0)
        try:
            other = _check_vector(other)
        except TypeError:
            return False
        if (self.args == []) and (other.args == []):
            return True
        elif (self.args == []) or (other.args == []):
            return False

        frame = self.args[0][1]
        for v in frame:
            if expand((self - other).dot(v)) != 0:
                return False
        return True

    def __mul__(self, other):
        """Multiplies the Vector by a sympifyable expression.

        Parameters
        ==========

        other : Sympifyable
            The scalar to multiply this Vector with

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> from sympy import Symbol
        >>> N = ReferenceFrame('N')
        >>> b = Symbol('b')
        >>> V = 10 * b * N.x
        >>> print(V)
        10*b*N.x

        """

        newlist = list(self.args)
        other = sympify(other)
        for i, v in enumerate(newlist):
            newlist[i] = (other * newlist[i][0], newlist[i][1])
        return Vector(newlist)

    def __neg__(self):
        return self * -1

    def outer(self, other):
        """Outer product between two Vectors.

        A rank increasing operation, which returns a Dyadic from two Vectors

        Parameters
        ==========

        other : Vector
            The Vector to take the outer product with

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, outer
        >>> N = ReferenceFrame('N')
        >>> outer(N.x, N.x)
        (N.x|N.x)

        """

        from sympy.physics.vector.dyadic import Dyadic
        other = _check_vector(other)
        ol = Dyadic(0)
        for v in self.args:
            for v2 in other.args:
                # it looks this way because if we are in the same frame and
                # use the enumerate function on the same frame in a nested
                # fashion, then bad things happen
                ol += Dyadic([(v[0][0] * v2[0][0], v[1].x, v2[1].x)])
                ol += Dyadic([(v[0][0] * v2[0][1], v[1].x, v2[1].y)])
                ol += Dyadic([(v[0][0] * v2[0][2], v[1].x, v2[1].z)])
                ol += Dyadic([(v[0][1] * v2[0][0], v[1].y, v2[1].x)])
                ol += Dyadic([(v[0][1] * v2[0][1], v[1].y, v2[1].y)])
                ol += Dyadic([(v[0][1] * v2[0][2], v[1].y, v2[1].z)])
                ol += Dyadic([(v[0][2] * v2[0][0], v[1].z, v2[1].x)])
                ol += Dyadic([(v[0][2] * v2[0][1], v[1].z, v2[1].y)])
                ol += Dyadic([(v[0][2] * v2[0][2], v[1].z, v2[1].z)])
        return ol

    def _latex(self, printer):
        """Latex Printing method. """

        ar = self.args  # just to shorten things
        if len(ar) == 0:
            return str(0)
        ol = []  # output list, to be concatenated to a string
        for i, v in enumerate(ar):
            for j in 0, 1, 2:
                # if the coef of the basis vector is 1, we skip the 1
                if ar[i][0][j] == 1:
                    ol.append(' + ' + ar[i][1].latex_vecs[j])
                # if the coef of the basis vector is -1, we skip the 1
                elif ar[i][0][j] == -1:
                    ol.append(' - ' + ar[i][1].latex_vecs[j])
                elif ar[i][0][j] != 0:
                    # If the coefficient of the basis vector is not 1 or -1;
                    # also, we might wrap it in parentheses, for readability.
                    arg_str = printer._print(ar[i][0][j])
                    if isinstance(ar[i][0][j], Add):
                        arg_str = "(%s)" % arg_str
                    if arg_str[0] == '-':
                        arg_str = arg_str[1:]
                        str_start = ' - '
                    else:
                        str_start = ' + '
                    ol.append(str_start + arg_str + ar[i][1].latex_vecs[j])
        outstr = ''.join(ol)
        if outstr.startswith(' + '):
            outstr = outstr[3:]
        elif outstr.startswith(' '):
            outstr = outstr[1:]
        return outstr

    def _pretty(self, printer):
        """Pretty Printing method. """
        from sympy.printing.pretty.stringpict import prettyForm

        terms = []

        def juxtapose(a, b):
            pa = printer._print(a)
            pb = printer._print(b)
            if a.is_Add:
                pa = prettyForm(*pa.parens())
            return printer._print_seq([pa, pb], delimiter=' ')

        for M, N in self.args:
            for i in range(3):
                if M[i] == 0:
                    continue
                elif M[i] == 1:
                    terms.append(prettyForm(N.pretty_vecs[i]))
                elif M[i] == -1:
                    terms.append(prettyForm("-1") * prettyForm(N.pretty_vecs[i]))
                else:
                    terms.append(juxtapose(M[i], N.pretty_vecs[i]))

        if terms:
            pretty_result = prettyForm.__add__(*terms)
        else:
            pretty_result = prettyForm("0")

        return pretty_result

    def __rsub__(self, other):
        return (-1 * self) + other

    def _sympystr(self, printer, order=True):
        """Printing method. """
        if not order or len(self.args) == 1:
            ar = list(self.args)
        elif len(self.args) == 0:
            return printer._print(0)
        else:
            d = {v[1]: v[0] for v in self.args}
            keys = sorted(d.keys(), key=lambda x: x.index)
            ar = []
            for key in keys:
                ar.append((d[key], key))
        ol = []  # output list, to be concatenated to a string
        for i, v in enumerate(ar):
            for j in 0, 1, 2:
                # if the coef of the basis vector is 1, we skip the 1
                if ar[i][0][j] == 1:
                    ol.append(' + ' + ar[i][1].str_vecs[j])
                # if the coef of the basis vector is -1, we skip the 1
                elif ar[i][0][j] == -1:
                    ol.append(' - ' + ar[i][1].str_vecs[j])
                elif ar[i][0][j] != 0:
                    # If the coefficient of the basis vector is not 1 or -1;
                    # also, we might wrap it in parentheses, for readability.
                    arg_str = printer._print(ar[i][0][j])
                    if isinstance(ar[i][0][j], Add):
                        arg_str = "(%s)" % arg_str
                    if arg_str[0] == '-':
                        arg_str = arg_str[1:]
                        str_start = ' - '
                    else:
                        str_start = ' + '
                    ol.append(str_start + arg_str + '*' + ar[i][1].str_vecs[j])
        outstr = ''.join(ol)
        if outstr.startswith(' + '):
            outstr = outstr[3:]
        elif outstr.startswith(' '):
            outstr = outstr[1:]
        return outstr

    def __sub__(self, other):
        """The subtraction operator. """
        return self.__add__(other * -1)

    def cross(self, other):
        """The cross product operator for two Vectors.

        Returns a Vector, expressed in the same ReferenceFrames as self.

        Parameters
        ==========

        other : Vector
            The Vector which we are crossing with

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.vector import ReferenceFrame, cross
        >>> q1 = symbols('q1')
        >>> N = ReferenceFrame('N')
        >>> cross(N.x, N.y)
        N.z
        >>> A = ReferenceFrame('A')
        >>> A.orient_axis(N, q1, N.x)
        >>> cross(A.x, N.y)
        N.z
        >>> cross(N.y, A.x)
        - sin(q1)*A.y - cos(q1)*A.z

        """

        from sympy.physics.vector.dyadic import Dyadic, _check_dyadic
        if isinstance(other, Dyadic):
            other = _check_dyadic(other)
            ol = Dyadic(0)
            for i, v in enumerate(other.args):
                ol += v[0] * ((self.cross(v[1])).outer(v[2]))
            return ol
        other = _check_vector(other)
        if other.args == []:
            return Vector(0)

        def _det(mat):
            """This is needed as a little method for to find the determinant
            of a list in python; needs to work for a 3x3 list.
            SymPy's Matrix will not take in Vector, so need a custom function.
            You should not be calling this.

            """

            return (mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1])
                    + mat[0][1] * (mat[1][2] * mat[2][0] - mat[1][0] *
                    mat[2][2]) + mat[0][2] * (mat[1][0] * mat[2][1] -
                    mat[1][1] * mat[2][0]))

        outlist = []
        ar = other.args  # For brevity
        for i, v in enumerate(ar):
            tempx = v[1].x
            tempy = v[1].y
            tempz = v[1].z
            tempm = ([[tempx, tempy, tempz],
                      [self.dot(tempx), self.dot(tempy), self.dot(tempz)],
                      [Vector([ar[i]]).dot(tempx), Vector([ar[i]]).dot(tempy),
                       Vector([ar[i]]).dot(tempz)]])
            outlist += _det(tempm).args
        return Vector(outlist)

    __radd__ = __add__
    __rmul__ = __mul__

    def separate(self):
        """
        The constituents of this vector in different reference frames,
        as per its definition.

        Returns a dict mapping each ReferenceFrame to the corresponding
        constituent Vector.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> R1 = ReferenceFrame('R1')
        >>> R2 = ReferenceFrame('R2')
        >>> v = R1.x + R2.x
        >>> v.separate() == {R1: R1.x, R2: R2.x}
        True

        """

        components = {}
        for x in self.args:
            components[x[1]] = Vector([x])
        return components

    def __and__(self, other):
        return self.dot(other)
    __and__.__doc__ = dot.__doc__
    __rand__ = __and__

    def __xor__(self, other):
        return self.cross(other)
    __xor__.__doc__ = cross.__doc__

    def __or__(self, other):
        return self.outer(other)
    __or__.__doc__ = outer.__doc__

    def diff(self, var, frame, var_in_dcm=True):
        """Returns the partial derivative of the vector with respect to a
        variable in the provided reference frame.

        Parameters
        ==========
        var : Symbol
            What the partial derivative is taken with respect to.
        frame : ReferenceFrame
            The reference frame that the partial derivative is taken in.
        var_in_dcm : boolean
            If true, the differentiation algorithm assumes that the variable
            may be present in any of the direction cosine matrices that relate
            the frame to the frames of any component of the vector. But if it
            is known that the variable is not present in the direction cosine
            matrices, false can be set to skip full reexpression in the desired
            frame.

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.vector import dynamicsymbols, ReferenceFrame
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> t = Symbol('t')
        >>> q1 = dynamicsymbols('q1')
        >>> N = ReferenceFrame('N')
        >>> A = N.orientnew('A', 'Axis', [q1, N.y])
        >>> A.x.diff(t, N)
        - sin(q1)*q1'*N.x - cos(q1)*q1'*N.z
        >>> A.x.diff(t, N).express(A).simplify()
        - q1'*A.z
        >>> B = ReferenceFrame('B')
        >>> u1, u2 = dynamicsymbols('u1, u2')
        >>> v = u1 * A.x + u2 * B.y
        >>> v.diff(u2, N, var_in_dcm=False)
        B.y

        """

        from sympy.physics.vector.frame import _check_frame

        _check_frame(frame)
        var = sympify(var)

        inlist = []

        for vector_component in self.args:
            measure_number = vector_component[0]
            component_frame = vector_component[1]
            if component_frame == frame:
                inlist += [(measure_number.diff(var), frame)]
            else:
                # If the direction cosine matrix relating the component frame
                # with the derivative frame does not contain the variable.
                if not var_in_dcm or (frame.dcm(component_frame).diff(var) ==
                                      zeros(3, 3)):
                    inlist += [(measure_number.diff(var), component_frame)]
                else:  # else express in the frame
                    reexp_vec_comp = Vector([vector_component]).express(frame)
                    deriv = reexp_vec_comp.args[0][0].diff(var)
                    inlist += Vector([(deriv, frame)]).args

        return Vector(inlist)

    def express(self, otherframe, variables=False):
        """
        Returns a Vector equivalent to this one, expressed in otherframe.
        Uses the global express method.

        Parameters
        ==========

        otherframe : ReferenceFrame
            The frame for this Vector to be described in

        variables : boolean
            If True, the coordinate symbols(if present) in this Vector
            are re-expressed in terms otherframe

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> q1 = dynamicsymbols('q1')
        >>> N = ReferenceFrame('N')
        >>> A = N.orientnew('A', 'Axis', [q1, N.y])
        >>> A.x.express(N)
        cos(q1)*N.x - sin(q1)*N.z

        """
        from sympy.physics.vector import express
        return express(self, otherframe, variables=variables)

    def to_matrix(self, reference_frame):
        """Returns the matrix form of the vector with respect to the given
        frame.

        Parameters
        ----------
        reference_frame : ReferenceFrame
            The reference frame that the rows of the matrix correspond to.

        Returns
        -------
        matrix : ImmutableMatrix, shape(3,1)
            The matrix that gives the 1D vector.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.vector import ReferenceFrame
        >>> a, b, c = symbols('a, b, c')
        >>> N = ReferenceFrame('N')
        >>> vector = a * N.x + b * N.y + c * N.z
        >>> vector.to_matrix(N)
        Matrix([
        [a],
        [b],
        [c]])
        >>> beta = symbols('beta')
        >>> A = N.orientnew('A', 'Axis', (beta, N.x))
        >>> vector.to_matrix(A)
        Matrix([
        [                         a],
        [ b*cos(beta) + c*sin(beta)],
        [-b*sin(beta) + c*cos(beta)]])

        """

        return Matrix([self.dot(unit_vec) for unit_vec in
                       reference_frame]).reshape(3, 1)

    def doit(self, **hints):
        """Calls .doit() on each term in the Vector"""
        d = {}
        for v in self.args:
            d[v[1]] = v[0].applyfunc(lambda x: x.doit(**hints))
        return Vector(d)

    def dt(self, otherframe):
        """
        Returns a Vector which is the time derivative of
        the self Vector, taken in frame otherframe.

        Calls the global time_derivative method

        Parameters
        ==========

        otherframe : ReferenceFrame
            The frame to calculate the time derivative in

        """
        from sympy.physics.vector import time_derivative
        return time_derivative(self, otherframe)

    def simplify(self):
        """Returns a simplified Vector."""
        d = {}
        for v in self.args:
            d[v[1]] = simplify(v[0])
        return Vector(d)

    def subs(self, *args, **kwargs):
        """Substitution on the Vector.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> from sympy import Symbol
        >>> N = ReferenceFrame('N')
        >>> s = Symbol('s')
        >>> a = N.x * s
        >>> a.subs({s: 2})
        2*N.x

        """

        d = {}
        for v in self.args:
            d[v[1]] = v[0].subs(*args, **kwargs)
        return Vector(d)

    def magnitude(self):
        """Returns the magnitude (Euclidean norm) of self.

        Warnings
        ========

        Python ignores the leading negative sign so that might
        give wrong results.
        ``-A.x.magnitude()`` would be treated as ``-(A.x.magnitude())``,
        instead of ``(-A.x).magnitude()``.

        """
        return sqrt(self.dot(self))

    def normalize(self):
        """Returns a Vector of magnitude 1, codirectional with self."""
        return Vector(self.args + []) / self.magnitude()

    def applyfunc(self, f):
        """Apply a function to each component of a vector."""
        if not callable(f):
            raise TypeError("`f` must be callable.")

        d = {}
        for v in self.args:
            d[v[1]] = v[0].applyfunc(f)
        return Vector(d)

    def angle_between(self, vec):
        """
        Returns the smallest angle between Vector 'vec' and self.

        Parameter
        =========

        vec : Vector
            The Vector between which angle is needed.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> A = ReferenceFrame("A")
        >>> v1 = A.x
        >>> v2 = A.y
        >>> v1.angle_between(v2)
        pi/2

        >>> v3 = A.x + A.y + A.z
        >>> v1.angle_between(v3)
        acos(sqrt(3)/3)

        Warnings
        ========

        Python ignores the leading negative sign so that might give wrong
        results. ``-A.x.angle_between()`` would be treated as
        ``-(A.x.angle_between())``, instead of ``(-A.x).angle_between()``.

        """

        vec1 = self.normalize()
        vec2 = vec.normalize()
        angle = acos(vec1.dot(vec2))
        return angle

    def free_symbols(self, reference_frame):
        """Returns the free symbols in the measure numbers of the vector
        expressed in the given reference frame.

        Parameters
        ==========
        reference_frame : ReferenceFrame
            The frame with respect to which the free symbols of the given
            vector is to be determined.

        Returns
        =======
        set of Symbol
            set of symbols present in the measure numbers of
            ``reference_frame``.

        """

        return self.to_matrix(reference_frame).free_symbols

    def free_dynamicsymbols(self, reference_frame):
        """Returns the free dynamic symbols (functions of time ``t``) in the
        measure numbers of the vector expressed in the given reference frame.

        Parameters
        ==========
        reference_frame : ReferenceFrame
            The frame with respect to which the free dynamic symbols of the
            given vector is to be determined.

        Returns
        =======
        set
            Set of functions of time ``t``, e.g.
            ``Function('f')(me.dynamicsymbols._t)``.

        """
        # TODO : Circular dependency if imported at top. Should move
        # find_dynamicsymbols into physics.vector.functions.
        from sympy.physics.mechanics.functions import find_dynamicsymbols

        return find_dynamicsymbols(self, reference_frame=reference_frame)

    def _eval_evalf(self, prec):
        if not self.args:
            return self
        new_args = []
        dps = prec_to_dps(prec)
        for mat, frame in self.args:
            new_args.append([mat.evalf(n=dps), frame])
        return Vector(new_args)

    def xreplace(self, rule):
        """Replace occurrences of objects within the measure numbers of the
        vector.

        Parameters
        ==========

        rule : dict-like
            Expresses a replacement rule.

        Returns
        =======

        Vector
            Result of the replacement.

        Examples
        ========

        >>> from sympy import symbols, pi
        >>> from sympy.physics.vector import ReferenceFrame
        >>> A = ReferenceFrame('A')
        >>> x, y, z = symbols('x y z')
        >>> ((1 + x*y) * A.x).xreplace({x: pi})
        (pi*y + 1)*A.x
        >>> ((1 + x*y) * A.x).xreplace({x: pi, y: 2})
        (1 + 2*pi)*A.x

        Replacements occur only if an entire node in the expression tree is
        matched:

        >>> ((x*y + z) * A.x).xreplace({x*y: pi})
        (z + pi)*A.x
        >>> ((x*y*z) * A.x).xreplace({x*y: pi})
        x*y*z*A.x

        """

        new_args = []
        for mat, frame in self.args:
            mat = mat.xreplace(rule)
            new_args.append([mat, frame])
        return Vector(new_args)


class VectorTypeError(TypeError):

    def __init__(self, other, want):
        msg = filldedent("Expected an instance of %s, but received object "
                         "'%s' of %s." % (type(want), other, type(other)))
        super().__init__(msg)


def _check_vector(other):
    if not isinstance(other, Vector):
        raise TypeError('A Vector must be supplied')
    return other
