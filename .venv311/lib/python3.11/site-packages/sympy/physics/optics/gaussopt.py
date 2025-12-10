"""
Gaussian optics.

The module implements:

- Ray transfer matrices for geometrical and gaussian optics.

  See RayTransferMatrix, GeometricRay and BeamParameter

- Conjugation relations for geometrical and gaussian optics.

  See geometric_conj*, gauss_conj and conjugate_gauss_beams

The conventions for the distances are as follows:

focal distance
    positive for convergent lenses
object distance
    positive for real objects
image distance
    positive for real images
"""

__all__ = [
    'RayTransferMatrix',
    'FreeSpace',
    'FlatRefraction',
    'CurvedRefraction',
    'FlatMirror',
    'CurvedMirror',
    'ThinLens',
    'GeometricRay',
    'BeamParameter',
    'waist2rayleigh',
    'rayleigh2waist',
    'geometric_conj_ab',
    'geometric_conj_af',
    'geometric_conj_bf',
    'gaussian_conj',
    'conjugate_gauss_beams',
]


from sympy.core.expr import Expr
from sympy.core.numbers import (I, pi)
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import atan2
from sympy.matrices.dense import Matrix, MutableDenseMatrix
from sympy.polys.rationaltools import together
from sympy.utilities.misc import filldedent

###
# A, B, C, D matrices
###


class RayTransferMatrix(MutableDenseMatrix):
    """
    Base class for a Ray Transfer Matrix.

    It should be used if there is not already a more specific subclass mentioned
    in See Also.

    Parameters
    ==========

    parameters :
        A, B, C and D or 2x2 matrix (Matrix(2, 2, [A, B, C, D]))

    Examples
    ========

    >>> from sympy.physics.optics import RayTransferMatrix, ThinLens
    >>> from sympy import Symbol, Matrix

    >>> mat = RayTransferMatrix(1, 2, 3, 4)
    >>> mat
    Matrix([
    [1, 2],
    [3, 4]])

    >>> RayTransferMatrix(Matrix([[1, 2], [3, 4]]))
    Matrix([
    [1, 2],
    [3, 4]])

    >>> mat.A
    1

    >>> f = Symbol('f')
    >>> lens = ThinLens(f)
    >>> lens
    Matrix([
    [   1, 0],
    [-1/f, 1]])

    >>> lens.C
    -1/f

    See Also
    ========

    GeometricRay, BeamParameter,
    FreeSpace, FlatRefraction, CurvedRefraction,
    FlatMirror, CurvedMirror, ThinLens

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis
    """

    def __new__(cls, *args):

        if len(args) == 4:
            temp = ((args[0], args[1]), (args[2], args[3]))
        elif len(args) == 1 \
            and isinstance(args[0], Matrix) \
                and args[0].shape == (2, 2):
            temp = args[0]
        else:
            raise ValueError(filldedent('''
                Expecting 2x2 Matrix or the 4 elements of
                the Matrix but got %s''' % str(args)))
        return Matrix.__new__(cls, temp)

    def __mul__(self, other):
        if isinstance(other, RayTransferMatrix):
            return RayTransferMatrix(Matrix(self)*Matrix(other))
        elif isinstance(other, GeometricRay):
            return GeometricRay(Matrix(self)*Matrix(other))
        elif isinstance(other, BeamParameter):
            temp = Matrix(self)*Matrix(((other.q,), (1,)))
            q = (temp[0]/temp[1]).expand(complex=True)
            return BeamParameter(other.wavelen,
                                 together(re(q)),
                                 z_r=together(im(q)))
        else:
            return Matrix.__mul__(self, other)

    @property
    def A(self):
        """
        The A parameter of the Matrix.

        Examples
        ========

        >>> from sympy.physics.optics import RayTransferMatrix
        >>> mat = RayTransferMatrix(1, 2, 3, 4)
        >>> mat.A
        1
        """
        return self[0, 0]

    @property
    def B(self):
        """
        The B parameter of the Matrix.

        Examples
        ========

        >>> from sympy.physics.optics import RayTransferMatrix
        >>> mat = RayTransferMatrix(1, 2, 3, 4)
        >>> mat.B
        2
        """
        return self[0, 1]

    @property
    def C(self):
        """
        The C parameter of the Matrix.

        Examples
        ========

        >>> from sympy.physics.optics import RayTransferMatrix
        >>> mat = RayTransferMatrix(1, 2, 3, 4)
        >>> mat.C
        3
        """
        return self[1, 0]

    @property
    def D(self):
        """
        The D parameter of the Matrix.

        Examples
        ========

        >>> from sympy.physics.optics import RayTransferMatrix
        >>> mat = RayTransferMatrix(1, 2, 3, 4)
        >>> mat.D
        4
        """
        return self[1, 1]


class FreeSpace(RayTransferMatrix):
    """
    Ray Transfer Matrix for free space.

    Parameters
    ==========

    distance

    See Also
    ========

    RayTransferMatrix

    Examples
    ========

    >>> from sympy.physics.optics import FreeSpace
    >>> from sympy import symbols
    >>> d = symbols('d')
    >>> FreeSpace(d)
    Matrix([
    [1, d],
    [0, 1]])
    """
    def __new__(cls, d):
        return RayTransferMatrix.__new__(cls, 1, d, 0, 1)


class FlatRefraction(RayTransferMatrix):
    """
    Ray Transfer Matrix for refraction.

    Parameters
    ==========

    n1 :
        Refractive index of one medium.
    n2 :
        Refractive index of other medium.

    See Also
    ========

    RayTransferMatrix

    Examples
    ========

    >>> from sympy.physics.optics import FlatRefraction
    >>> from sympy import symbols
    >>> n1, n2 = symbols('n1 n2')
    >>> FlatRefraction(n1, n2)
    Matrix([
    [1,     0],
    [0, n1/n2]])
    """
    def __new__(cls, n1, n2):
        n1, n2 = map(sympify, (n1, n2))
        return RayTransferMatrix.__new__(cls, 1, 0, 0, n1/n2)


class CurvedRefraction(RayTransferMatrix):
    """
    Ray Transfer Matrix for refraction on curved interface.

    Parameters
    ==========

    R :
        Radius of curvature (positive for concave).
    n1 :
        Refractive index of one medium.
    n2 :
        Refractive index of other medium.

    See Also
    ========

    RayTransferMatrix

    Examples
    ========

    >>> from sympy.physics.optics import CurvedRefraction
    >>> from sympy import symbols
    >>> R, n1, n2 = symbols('R n1 n2')
    >>> CurvedRefraction(R, n1, n2)
    Matrix([
    [               1,     0],
    [(n1 - n2)/(R*n2), n1/n2]])
    """
    def __new__(cls, R, n1, n2):
        R, n1, n2 = map(sympify, (R, n1, n2))
        return RayTransferMatrix.__new__(cls, 1, 0, (n1 - n2)/R/n2, n1/n2)


class FlatMirror(RayTransferMatrix):
    """
    Ray Transfer Matrix for reflection.

    See Also
    ========

    RayTransferMatrix

    Examples
    ========

    >>> from sympy.physics.optics import FlatMirror
    >>> FlatMirror()
    Matrix([
    [1, 0],
    [0, 1]])
    """
    def __new__(cls):
        return RayTransferMatrix.__new__(cls, 1, 0, 0, 1)


class CurvedMirror(RayTransferMatrix):
    """
    Ray Transfer Matrix for reflection from curved surface.

    Parameters
    ==========

    R : radius of curvature (positive for concave)

    See Also
    ========

    RayTransferMatrix

    Examples
    ========

    >>> from sympy.physics.optics import CurvedMirror
    >>> from sympy import symbols
    >>> R = symbols('R')
    >>> CurvedMirror(R)
    Matrix([
    [   1, 0],
    [-2/R, 1]])
    """
    def __new__(cls, R):
        R = sympify(R)
        return RayTransferMatrix.__new__(cls, 1, 0, -2/R, 1)


class ThinLens(RayTransferMatrix):
    """
    Ray Transfer Matrix for a thin lens.

    Parameters
    ==========

    f :
        The focal distance.

    See Also
    ========

    RayTransferMatrix

    Examples
    ========

    >>> from sympy.physics.optics import ThinLens
    >>> from sympy import symbols
    >>> f = symbols('f')
    >>> ThinLens(f)
    Matrix([
    [   1, 0],
    [-1/f, 1]])
    """
    def __new__(cls, f):
        f = sympify(f)
        return RayTransferMatrix.__new__(cls, 1, 0, -1/f, 1)


###
# Representation for geometric ray
###

class GeometricRay(MutableDenseMatrix):
    """
    Representation for a geometric ray in the Ray Transfer Matrix formalism.

    Parameters
    ==========

    h : height, and
    angle : angle, or
    matrix : a 2x1 matrix (Matrix(2, 1, [height, angle]))

    Examples
    ========

    >>> from sympy.physics.optics import GeometricRay, FreeSpace
    >>> from sympy import symbols, Matrix
    >>> d, h, angle = symbols('d, h, angle')

    >>> GeometricRay(h, angle)
    Matrix([
    [    h],
    [angle]])

    >>> FreeSpace(d)*GeometricRay(h, angle)
    Matrix([
    [angle*d + h],
    [      angle]])

    >>> GeometricRay( Matrix( ((h,), (angle,)) ) )
    Matrix([
    [    h],
    [angle]])

    See Also
    ========

    RayTransferMatrix

    """

    def __new__(cls, *args):
        if len(args) == 1 and isinstance(args[0], Matrix) \
                and args[0].shape == (2, 1):
            temp = args[0]
        elif len(args) == 2:
            temp = ((args[0],), (args[1],))
        else:
            raise ValueError(filldedent('''
                Expecting 2x1 Matrix or the 2 elements of
                the Matrix but got %s''' % str(args)))
        return Matrix.__new__(cls, temp)

    @property
    def height(self):
        """
        The distance from the optical axis.

        Examples
        ========

        >>> from sympy.physics.optics import GeometricRay
        >>> from sympy import symbols
        >>> h, angle = symbols('h, angle')
        >>> gRay = GeometricRay(h, angle)
        >>> gRay.height
        h
        """
        return self[0]

    @property
    def angle(self):
        """
        The angle with the optical axis.

        Examples
        ========

        >>> from sympy.physics.optics import GeometricRay
        >>> from sympy import symbols
        >>> h, angle = symbols('h, angle')
        >>> gRay = GeometricRay(h, angle)
        >>> gRay.angle
        angle
        """
        return self[1]


###
# Representation for gauss beam
###

class BeamParameter(Expr):
    """
    Representation for a gaussian ray in the Ray Transfer Matrix formalism.

    Parameters
    ==========

    wavelen : the wavelength,
    z : the distance to waist, and
    w : the waist, or
    z_r : the rayleigh range.
    n : the refractive index of medium.

    Examples
    ========

    >>> from sympy.physics.optics import BeamParameter
    >>> p = BeamParameter(530e-9, 1, w=1e-3)
    >>> p.q
    1 + 1.88679245283019*I*pi

    >>> p.q.n()
    1.0 + 5.92753330865999*I
    >>> p.w_0.n()
    0.00100000000000000
    >>> p.z_r.n()
    5.92753330865999

    >>> from sympy.physics.optics import FreeSpace
    >>> fs = FreeSpace(10)
    >>> p1 = fs*p
    >>> p.w.n()
    0.00101413072159615
    >>> p1.w.n()
    0.00210803120913829

    See Also
    ========

    RayTransferMatrix

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Complex_beam_parameter
    .. [2] https://en.wikipedia.org/wiki/Gaussian_beam
    """
    #TODO A class Complex may be implemented. The BeamParameter may
    # subclass it. See:
    # https://groups.google.com/d/topic/sympy/7XkU07NRBEs/discussion

    def __new__(cls, wavelen, z, z_r=None, w=None, n=1):
        wavelen = sympify(wavelen)
        z = sympify(z)
        n = sympify(n)

        if z_r is not None and w is None:
            z_r = sympify(z_r)
        elif w is not None and z_r is None:
            z_r = waist2rayleigh(sympify(w), wavelen, n)
        elif z_r is None and w is None:
            raise ValueError('Must specify one of w and z_r.')

        return Expr.__new__(cls, wavelen, z, z_r, n)

    @property
    def wavelen(self):
        return self.args[0]

    @property
    def z(self):
        return self.args[1]

    @property
    def z_r(self):
        return self.args[2]

    @property
    def n(self):
        return self.args[3]

    @property
    def q(self):
        """
        The complex parameter representing the beam.

        Examples
        ========

        >>> from sympy.physics.optics import BeamParameter
        >>> p = BeamParameter(530e-9, 1, w=1e-3)
        >>> p.q
        1 + 1.88679245283019*I*pi
        """
        return self.z + I*self.z_r

    @property
    def radius(self):
        """
        The radius of curvature of the phase front.

        Examples
        ========

        >>> from sympy.physics.optics import BeamParameter
        >>> p = BeamParameter(530e-9, 1, w=1e-3)
        >>> p.radius
        1 + 3.55998576005696*pi**2
        """
        return self.z*(1 + (self.z_r/self.z)**2)

    @property
    def w(self):
        """
        The radius of the beam w(z), at any position z along the beam.
        The beam radius at `1/e^2` intensity (axial value).

        See Also
        ========

        w_0 :
            The minimal radius of beam.

        Examples
        ========

        >>> from sympy.physics.optics import BeamParameter
        >>> p = BeamParameter(530e-9, 1, w=1e-3)
        >>> p.w
        0.001*sqrt(0.2809/pi**2 + 1)
        """
        return self.w_0*sqrt(1 + (self.z/self.z_r)**2)

    @property
    def w_0(self):
        """
         The minimal radius of beam at `1/e^2` intensity (peak value).

        See Also
        ========

        w : the beam radius at `1/e^2` intensity (axial value).

        Examples
        ========

        >>> from sympy.physics.optics import BeamParameter
        >>> p = BeamParameter(530e-9, 1, w=1e-3)
        >>> p.w_0
        0.00100000000000000
        """
        return sqrt(self.z_r/(pi*self.n)*self.wavelen)

    @property
    def divergence(self):
        """
        Half of the total angular spread.

        Examples
        ========

        >>> from sympy.physics.optics import BeamParameter
        >>> p = BeamParameter(530e-9, 1, w=1e-3)
        >>> p.divergence
        0.00053/pi
        """
        return self.wavelen/pi/self.w_0

    @property
    def gouy(self):
        """
        The Gouy phase.

        Examples
        ========

        >>> from sympy.physics.optics import BeamParameter
        >>> p = BeamParameter(530e-9, 1, w=1e-3)
        >>> p.gouy
        atan(0.53/pi)
        """
        return atan2(self.z, self.z_r)

    @property
    def waist_approximation_limit(self):
        """
        The minimal waist for which the gauss beam approximation is valid.

        Explanation
        ===========

        The gauss beam is a solution to the paraxial equation. For curvatures
        that are too great it is not a valid approximation.

        Examples
        ========

        >>> from sympy.physics.optics import BeamParameter
        >>> p = BeamParameter(530e-9, 1, w=1e-3)
        >>> p.waist_approximation_limit
        1.06e-6/pi
        """
        return 2*self.wavelen/pi


###
# Utilities
###

def waist2rayleigh(w, wavelen, n=1):
    """
    Calculate the rayleigh range from the waist of a gaussian beam.

    See Also
    ========

    rayleigh2waist, BeamParameter

    Examples
    ========

    >>> from sympy.physics.optics import waist2rayleigh
    >>> from sympy import symbols
    >>> w, wavelen = symbols('w wavelen')
    >>> waist2rayleigh(w, wavelen)
    pi*w**2/wavelen
    """
    w, wavelen = map(sympify, (w, wavelen))
    return w**2*n*pi/wavelen


def rayleigh2waist(z_r, wavelen):
    """Calculate the waist from the rayleigh range of a gaussian beam.

    See Also
    ========

    waist2rayleigh, BeamParameter

    Examples
    ========

    >>> from sympy.physics.optics import rayleigh2waist
    >>> from sympy import symbols
    >>> z_r, wavelen = symbols('z_r wavelen')
    >>> rayleigh2waist(z_r, wavelen)
    sqrt(wavelen*z_r)/sqrt(pi)
    """
    z_r, wavelen = map(sympify, (z_r, wavelen))
    return sqrt(z_r/pi*wavelen)


def geometric_conj_ab(a, b):
    """
    Conjugation relation for geometrical beams under paraxial conditions.

    Explanation
    ===========

    Takes the distances to the optical element and returns the needed
    focal distance.

    See Also
    ========

    geometric_conj_af, geometric_conj_bf

    Examples
    ========

    >>> from sympy.physics.optics import geometric_conj_ab
    >>> from sympy import symbols
    >>> a, b = symbols('a b')
    >>> geometric_conj_ab(a, b)
    a*b/(a + b)
    """
    a, b = map(sympify, (a, b))
    if a.is_infinite or b.is_infinite:
        return a if b.is_infinite else b
    else:
        return a*b/(a + b)


def geometric_conj_af(a, f):
    """
    Conjugation relation for geometrical beams under paraxial conditions.

    Explanation
    ===========

    Takes the object distance (for geometric_conj_af) or the image distance
    (for geometric_conj_bf) to the optical element and the focal distance.
    Then it returns the other distance needed for conjugation.

    See Also
    ========

    geometric_conj_ab

    Examples
    ========

    >>> from sympy.physics.optics.gaussopt import geometric_conj_af, geometric_conj_bf
    >>> from sympy import symbols
    >>> a, b, f = symbols('a b f')
    >>> geometric_conj_af(a, f)
    a*f/(a - f)
    >>> geometric_conj_bf(b, f)
    b*f/(b - f)
    """
    a, f = map(sympify, (a, f))
    return -geometric_conj_ab(a, -f)

geometric_conj_bf = geometric_conj_af


def gaussian_conj(s_in, z_r_in, f):
    """
    Conjugation relation for gaussian beams.

    Parameters
    ==========

    s_in :
        The distance to optical element from the waist.
    z_r_in :
        The rayleigh range of the incident beam.
    f :
        The focal length of the optical element.

    Returns
    =======

    a tuple containing (s_out, z_r_out, m)
    s_out :
        The distance between the new waist and the optical element.
    z_r_out :
        The rayleigh range of the emergent beam.
    m :
        The ration between the new and the old waists.

    Examples
    ========

    >>> from sympy.physics.optics import gaussian_conj
    >>> from sympy import symbols
    >>> s_in, z_r_in, f = symbols('s_in z_r_in f')

    >>> gaussian_conj(s_in, z_r_in, f)[0]
    1/(-1/(s_in + z_r_in**2/(-f + s_in)) + 1/f)

    >>> gaussian_conj(s_in, z_r_in, f)[1]
    z_r_in/(1 - s_in**2/f**2 + z_r_in**2/f**2)

    >>> gaussian_conj(s_in, z_r_in, f)[2]
    1/sqrt(1 - s_in**2/f**2 + z_r_in**2/f**2)
    """
    s_in, z_r_in, f = map(sympify, (s_in, z_r_in, f))
    s_out = 1 / ( -1/(s_in + z_r_in**2/(s_in - f)) + 1/f )
    m = 1/sqrt((1 - (s_in/f)**2) + (z_r_in/f)**2)
    z_r_out = z_r_in / ((1 - (s_in/f)**2) + (z_r_in/f)**2)
    return (s_out, z_r_out, m)


def conjugate_gauss_beams(wavelen, waist_in, waist_out, **kwargs):
    """
    Find the optical setup conjugating the object/image waists.

    Parameters
    ==========

    wavelen :
        The wavelength of the beam.
    waist_in and waist_out :
        The waists to be conjugated.
    f :
        The focal distance of the element used in the conjugation.

    Returns
    =======

    a tuple containing (s_in, s_out, f)
    s_in :
        The distance before the optical element.
    s_out :
        The distance after the optical element.
    f :
        The focal distance of the optical element.

    Examples
    ========

    >>> from sympy.physics.optics import conjugate_gauss_beams
    >>> from sympy import symbols, factor
    >>> l, w_i, w_o, f = symbols('l w_i w_o f')

    >>> conjugate_gauss_beams(l, w_i, w_o, f=f)[0]
    f*(1 - sqrt(w_i**2/w_o**2 - pi**2*w_i**4/(f**2*l**2)))

    >>> factor(conjugate_gauss_beams(l, w_i, w_o, f=f)[1])
    f*w_o**2*(w_i**2/w_o**2 - sqrt(w_i**2/w_o**2 -
              pi**2*w_i**4/(f**2*l**2)))/w_i**2

    >>> conjugate_gauss_beams(l, w_i, w_o, f=f)[2]
    f
    """
    #TODO add the other possible arguments
    wavelen, waist_in, waist_out = map(sympify, (wavelen, waist_in, waist_out))
    m = waist_out / waist_in
    z = waist2rayleigh(waist_in, wavelen)
    if len(kwargs) != 1:
        raise ValueError("The function expects only one named argument")
    elif 'dist' in kwargs:
        raise NotImplementedError(filldedent('''
            Currently only focal length is supported as a parameter'''))
    elif 'f' in kwargs:
        f = sympify(kwargs['f'])
        s_in = f * (1 - sqrt(1/m**2 - z**2/f**2))
        s_out = gaussian_conj(s_in, z, f)[0]
    elif 's_in' in kwargs:
        raise NotImplementedError(filldedent('''
            Currently only focal length is supported as a parameter'''))
    else:
        raise ValueError(filldedent('''
            The functions expects the focal length as a named argument'''))
    return (s_in, s_out, f)

#TODO
#def plot_beam():
#    """Plot the beam radius as it propagates in space."""
#    pass

#TODO
#def plot_beam_conjugation():
#    """
#    Plot the intersection of two beams.
#
#    Represents the conjugation relation.
#
#    See Also
#    ========
#
#    conjugate_gauss_beams
#    """
#    pass
