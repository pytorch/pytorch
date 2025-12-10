from sympy import sympify, Add, ImmutableMatrix as Matrix
from sympy.core.evalf import EvalfMixin
from sympy.printing.defaults import Printable

from mpmath.libmp.libmpf import prec_to_dps


__all__ = ['Dyadic']


class Dyadic(Printable, EvalfMixin):
    """A Dyadic object.

    See:
    https://en.wikipedia.org/wiki/Dyadic_tensor
    Kane, T., Levinson, D. Dynamics Theory and Applications. 1985 McGraw-Hill

    A more powerful way to represent a rigid body's inertia. While it is more
    complex, by choosing Dyadic components to be in body fixed basis vectors,
    the resulting matrix is equivalent to the inertia tensor.

    """

    is_number = False

    def __init__(self, inlist):
        """
        Just like Vector's init, you should not call this unless creating a
        zero dyadic.

        zd = Dyadic(0)

        Stores a Dyadic as a list of lists; the inner list has the measure
        number and the two unit vectors; the outerlist holds each unique
        unit vector pair.

        """

        self.args = []
        if inlist == 0:
            inlist = []
        while len(inlist) != 0:
            added = 0
            for i, v in enumerate(self.args):
                if ((str(inlist[0][1]) == str(self.args[i][1])) and
                        (str(inlist[0][2]) == str(self.args[i][2]))):
                    self.args[i] = (self.args[i][0] + inlist[0][0],
                                    inlist[0][1], inlist[0][2])
                    inlist.remove(inlist[0])
                    added = 1
                    break
            if added != 1:
                self.args.append(inlist[0])
                inlist.remove(inlist[0])
        i = 0
        # This code is to remove empty parts from the list
        while i < len(self.args):
            if ((self.args[i][0] == 0) | (self.args[i][1] == 0) |
                    (self.args[i][2] == 0)):
                self.args.remove(self.args[i])
                i -= 1
            i += 1

    @property
    def func(self):
        """Returns the class Dyadic. """
        return Dyadic

    def __add__(self, other):
        """The add operator for Dyadic. """
        other = _check_dyadic(other)
        return Dyadic(self.args + other.args)

    __radd__ = __add__

    def __mul__(self, other):
        """Multiplies the Dyadic by a sympifyable expression.

        Parameters
        ==========

        other : Sympafiable
            The scalar to multiply this Dyadic with

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, outer
        >>> N = ReferenceFrame('N')
        >>> d = outer(N.x, N.x)
        >>> 5 * d
        5*(N.x|N.x)

        """
        newlist = list(self.args)
        other = sympify(other)
        for i in range(len(newlist)):
            newlist[i] = (other * newlist[i][0], newlist[i][1],
                          newlist[i][2])
        return Dyadic(newlist)

    __rmul__ = __mul__

    def dot(self, other):
        """The inner product operator for a Dyadic and a Dyadic or Vector.

        Parameters
        ==========

        other : Dyadic or Vector
            The other Dyadic or Vector to take the inner product with

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, outer
        >>> N = ReferenceFrame('N')
        >>> D1 = outer(N.x, N.y)
        >>> D2 = outer(N.y, N.y)
        >>> D1.dot(D2)
        (N.x|N.y)
        >>> D1.dot(N.y)
        N.x

        """
        from sympy.physics.vector.vector import Vector, _check_vector
        if isinstance(other, Dyadic):
            other = _check_dyadic(other)
            ol = Dyadic(0)
            for v in self.args:
                for v2 in other.args:
                    ol += v[0] * v2[0] * (v[2].dot(v2[1])) * (v[1].outer(v2[2]))
        else:
            other = _check_vector(other)
            ol = Vector(0)
            for v in self.args:
                ol += v[0] * v[1] * (v[2].dot(other))
        return ol

    # NOTE : supports non-advertised Dyadic & Dyadic, Dyadic & Vector notation
    __and__ = dot

    def __truediv__(self, other):
        """Divides the Dyadic by a sympifyable expression. """
        return self.__mul__(1 / other)

    def __eq__(self, other):
        """Tests for equality.

        Is currently weak; needs stronger comparison testing

        """

        if other == 0:
            other = Dyadic(0)
        other = _check_dyadic(other)
        if (self.args == []) and (other.args == []):
            return True
        elif (self.args == []) or (other.args == []):
            return False
        return set(self.args) == set(other.args)

    def __ne__(self, other):
        return not self == other

    def __neg__(self):
        return self * -1

    def _latex(self, printer):
        ar = self.args  # just to shorten things
        if len(ar) == 0:
            return str(0)
        ol = []  # output list, to be concatenated to a string
        for v in ar:
            # if the coef of the dyadic is 1, we skip the 1
            if v[0] == 1:
                ol.append(' + ' + printer._print(v[1]) + r"\otimes " +
                          printer._print(v[2]))
            # if the coef of the dyadic is -1, we skip the 1
            elif v[0] == -1:
                ol.append(' - ' +
                          printer._print(v[1]) +
                          r"\otimes " +
                          printer._print(v[2]))
            # If the coefficient of the dyadic is not 1 or -1,
            # we might wrap it in parentheses, for readability.
            elif v[0] != 0:
                arg_str = printer._print(v[0])
                if isinstance(v[0], Add):
                    arg_str = '(%s)' % arg_str
                if arg_str.startswith('-'):
                    arg_str = arg_str[1:]
                    str_start = ' - '
                else:
                    str_start = ' + '
                ol.append(str_start + arg_str + printer._print(v[1]) +
                          r"\otimes " + printer._print(v[2]))
        outstr = ''.join(ol)
        if outstr.startswith(' + '):
            outstr = outstr[3:]
        elif outstr.startswith(' '):
            outstr = outstr[1:]
        return outstr

    def _pretty(self, printer):
        e = self

        class Fake:
            baseline = 0

            def render(self, *args, **kwargs):
                ar = e.args  # just to shorten things
                mpp = printer
                if len(ar) == 0:
                    return str(0)
                bar = "\N{CIRCLED TIMES}" if printer._use_unicode else "|"
                ol = []  # output list, to be concatenated to a string
                for v in ar:
                    # if the coef of the dyadic is 1, we skip the 1
                    if v[0] == 1:
                        ol.extend([" + ",
                                  mpp.doprint(v[1]),
                                  bar,
                                  mpp.doprint(v[2])])

                    # if the coef of the dyadic is -1, we skip the 1
                    elif v[0] == -1:
                        ol.extend([" - ",
                                  mpp.doprint(v[1]),
                                  bar,
                                  mpp.doprint(v[2])])

                    # If the coefficient of the dyadic is not 1 or -1,
                    # we might wrap it in parentheses, for readability.
                    elif v[0] != 0:
                        if isinstance(v[0], Add):
                            arg_str = mpp._print(
                                v[0]).parens()[0]
                        else:
                            arg_str = mpp.doprint(v[0])
                        if arg_str.startswith("-"):
                            arg_str = arg_str[1:]
                            str_start = " - "
                        else:
                            str_start = " + "
                        ol.extend([str_start, arg_str, " ",
                                  mpp.doprint(v[1]),
                                  bar,
                                  mpp.doprint(v[2])])

                outstr = "".join(ol)
                if outstr.startswith(" + "):
                    outstr = outstr[3:]
                elif outstr.startswith(" "):
                    outstr = outstr[1:]
                return outstr
        return Fake()

    def __rsub__(self, other):
        return (-1 * self) + other

    def _sympystr(self, printer):
        """Printing method. """
        ar = self.args  # just to shorten things
        if len(ar) == 0:
            return printer._print(0)
        ol = []  # output list, to be concatenated to a string
        for v in ar:
            # if the coef of the dyadic is 1, we skip the 1
            if v[0] == 1:
                ol.append(' + (' + printer._print(v[1]) + '|' +
                          printer._print(v[2]) + ')')
            # if the coef of the dyadic is -1, we skip the 1
            elif v[0] == -1:
                ol.append(' - (' + printer._print(v[1]) + '|' +
                          printer._print(v[2]) + ')')
            # If the coefficient of the dyadic is not 1 or -1,
            # we might wrap it in parentheses, for readability.
            elif v[0] != 0:
                arg_str = printer._print(v[0])
                if isinstance(v[0], Add):
                    arg_str = "(%s)" % arg_str
                if arg_str[0] == '-':
                    arg_str = arg_str[1:]
                    str_start = ' - '
                else:
                    str_start = ' + '
                ol.append(str_start + arg_str + '*(' +
                          printer._print(v[1]) +
                          '|' + printer._print(v[2]) + ')')
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
        """Returns the dyadic resulting from the dyadic vector cross product:
        Dyadic x Vector.

        Parameters
        ==========
        other : Vector
            Vector to cross with.

        Examples
        ========
        >>> from sympy.physics.vector import ReferenceFrame, outer, cross
        >>> N = ReferenceFrame('N')
        >>> d = outer(N.x, N.x)
        >>> cross(d, N.y)
        (N.x|N.z)

        """
        from sympy.physics.vector.vector import _check_vector
        other = _check_vector(other)
        ol = Dyadic(0)
        for v in self.args:
            ol += v[0] * (v[1].outer((v[2].cross(other))))
        return ol

    # NOTE : supports non-advertised Dyadic ^ Vector notation
    __xor__ = cross

    def express(self, frame1, frame2=None):
        """Expresses this Dyadic in alternate frame(s)

        The first frame is the list side expression, the second frame is the
        right side; if Dyadic is in form A.x|B.y, you can express it in two
        different frames. If no second frame is given, the Dyadic is
        expressed in only one frame.

        Calls the global express function

        Parameters
        ==========

        frame1 : ReferenceFrame
            The frame to express the left side of the Dyadic in
        frame2 : ReferenceFrame
            If provided, the frame to express the right side of the Dyadic in

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, outer, dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> N = ReferenceFrame('N')
        >>> q = dynamicsymbols('q')
        >>> B = N.orientnew('B', 'Axis', [q, N.z])
        >>> d = outer(N.x, N.x)
        >>> d.express(B, N)
        cos(q)*(B.x|N.x) - sin(q)*(B.y|N.x)

        """
        from sympy.physics.vector.functions import express
        return express(self, frame1, frame2)

    def to_matrix(self, reference_frame, second_reference_frame=None):
        """Returns the matrix form of the dyadic with respect to one or two
        reference frames.

        Parameters
        ----------
        reference_frame : ReferenceFrame
            The reference frame that the rows and columns of the matrix
            correspond to. If a second reference frame is provided, this
            only corresponds to the rows of the matrix.
        second_reference_frame : ReferenceFrame, optional, default=None
            The reference frame that the columns of the matrix correspond
            to.

        Returns
        -------
        matrix : ImmutableMatrix, shape(3,3)
            The matrix that gives the 2D tensor form.

        Examples
        ========

        >>> from sympy import symbols, trigsimp
        >>> from sympy.physics.vector import ReferenceFrame
        >>> from sympy.physics.mechanics import inertia
        >>> Ixx, Iyy, Izz, Ixy, Iyz, Ixz = symbols('Ixx, Iyy, Izz, Ixy, Iyz, Ixz')
        >>> N = ReferenceFrame('N')
        >>> inertia_dyadic = inertia(N, Ixx, Iyy, Izz, Ixy, Iyz, Ixz)
        >>> inertia_dyadic.to_matrix(N)
        Matrix([
        [Ixx, Ixy, Ixz],
        [Ixy, Iyy, Iyz],
        [Ixz, Iyz, Izz]])
        >>> beta = symbols('beta')
        >>> A = N.orientnew('A', 'Axis', (beta, N.x))
        >>> trigsimp(inertia_dyadic.to_matrix(A))
        Matrix([
        [                           Ixx,                                           Ixy*cos(beta) + Ixz*sin(beta),                                           -Ixy*sin(beta) + Ixz*cos(beta)],
        [ Ixy*cos(beta) + Ixz*sin(beta), Iyy*cos(2*beta)/2 + Iyy/2 + Iyz*sin(2*beta) - Izz*cos(2*beta)/2 + Izz/2,                 -Iyy*sin(2*beta)/2 + Iyz*cos(2*beta) + Izz*sin(2*beta)/2],
        [-Ixy*sin(beta) + Ixz*cos(beta),                -Iyy*sin(2*beta)/2 + Iyz*cos(2*beta) + Izz*sin(2*beta)/2, -Iyy*cos(2*beta)/2 + Iyy/2 - Iyz*sin(2*beta) + Izz*cos(2*beta)/2 + Izz/2]])

        """

        if second_reference_frame is None:
            second_reference_frame = reference_frame

        return Matrix([i.dot(self).dot(j) for i in reference_frame for j in
                      second_reference_frame]).reshape(3, 3)

    def doit(self, **hints):
        """Calls .doit() on each term in the Dyadic"""
        return sum([Dyadic([(v[0].doit(**hints), v[1], v[2])])
                    for v in self.args], Dyadic(0))

    def dt(self, frame):
        """Take the time derivative of this Dyadic in a frame.

        This function calls the global time_derivative method

        Parameters
        ==========

        frame : ReferenceFrame
            The frame to take the time derivative in

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, outer, dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> N = ReferenceFrame('N')
        >>> q = dynamicsymbols('q')
        >>> B = N.orientnew('B', 'Axis', [q, N.z])
        >>> d = outer(N.x, N.x)
        >>> d.dt(B)
        - q'*(N.y|N.x) - q'*(N.x|N.y)

        """
        from sympy.physics.vector.functions import time_derivative
        return time_derivative(self, frame)

    def simplify(self):
        """Returns a simplified Dyadic."""
        out = Dyadic(0)
        for v in self.args:
            out += Dyadic([(v[0].simplify(), v[1], v[2])])
        return out

    def subs(self, *args, **kwargs):
        """Substitution on the Dyadic.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> from sympy import Symbol
        >>> N = ReferenceFrame('N')
        >>> s = Symbol('s')
        >>> a = s*(N.x|N.x)
        >>> a.subs({s: 2})
        2*(N.x|N.x)

        """

        return sum([Dyadic([(v[0].subs(*args, **kwargs), v[1], v[2])])
                    for v in self.args], Dyadic(0))

    def applyfunc(self, f):
        """Apply a function to each component of a Dyadic."""
        if not callable(f):
            raise TypeError("`f` must be callable.")

        out = Dyadic(0)
        for a, b, c in self.args:
            out += f(a) * (b.outer(c))
        return out

    def _eval_evalf(self, prec):
        if not self.args:
            return self
        new_args = []
        dps = prec_to_dps(prec)
        for inlist in self.args:
            new_inlist = list(inlist)
            new_inlist[0] = inlist[0].evalf(n=dps)
            new_args.append(tuple(new_inlist))
        return Dyadic(new_args)

    def xreplace(self, rule):
        """
        Replace occurrences of objects within the measure numbers of the
        Dyadic.

        Parameters
        ==========

        rule : dict-like
            Expresses a replacement rule.

        Returns
        =======

        Dyadic
            Result of the replacement.

        Examples
        ========

        >>> from sympy import symbols, pi
        >>> from sympy.physics.vector import ReferenceFrame, outer
        >>> N = ReferenceFrame('N')
        >>> D = outer(N.x, N.x)
        >>> x, y, z = symbols('x y z')
        >>> ((1 + x*y) * D).xreplace({x: pi})
        (pi*y + 1)*(N.x|N.x)
        >>> ((1 + x*y) * D).xreplace({x: pi, y: 2})
        (1 + 2*pi)*(N.x|N.x)

        Replacements occur only if an entire node in the expression tree is
        matched:

        >>> ((x*y + z) * D).xreplace({x*y: pi})
        (z + pi)*(N.x|N.x)
        >>> ((x*y*z) * D).xreplace({x*y: pi})
        x*y*z*(N.x|N.x)

        """

        new_args = []
        for inlist in self.args:
            new_inlist = list(inlist)
            new_inlist[0] = new_inlist[0].xreplace(rule)
            new_args.append(tuple(new_inlist))
        return Dyadic(new_args)


def _check_dyadic(other):
    if not isinstance(other, Dyadic):
        raise TypeError('A Dyadic must be supplied')
    return other
