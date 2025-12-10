"""
This module can be used to solve 2D beam bending problems with
singularity functions in mechanics.
"""

from sympy.core import S, Symbol, diff, symbols
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import (Derivative, Function)
from sympy.core.mul import Mul
from sympy.core.relational import Eq
from sympy.core.sympify import sympify
from sympy.solvers import linsolve
from sympy.solvers.ode.ode import dsolve
from sympy.solvers.solvers import solve
from sympy.printing import sstr
from sympy.functions import SingularityFunction, Piecewise, factorial
from sympy.integrals import integrate
from sympy.series import limit
from sympy.plotting import plot, PlotGrid
from sympy.geometry.entity import GeometryEntity
from sympy.external import import_module
from sympy.sets.sets import Interval
from sympy.utilities.lambdify import lambdify
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.iterables import iterable
import warnings


__doctest_requires__ = {
    ('Beam.draw',
     'Beam.plot_bending_moment',
     'Beam.plot_deflection',
     'Beam.plot_ild_moment',
     'Beam.plot_ild_shear',
     'Beam.plot_shear_force',
     'Beam.plot_shear_stress',
     'Beam.plot_slope'): ['matplotlib'],
}


numpy = import_module('numpy', import_kwargs={'fromlist':['arange']})


class Beam:
    """
    A Beam is a structural element that is capable of withstanding load
    primarily by resisting against bending. Beams are characterized by
    their cross sectional profile(Second moment of area), their length
    and their material.

    .. note::
       A consistent sign convention must be used while solving a beam
       bending problem; the results will
       automatically follow the chosen sign convention. However, the
       chosen sign convention must respect the rule that, on the positive
       side of beam's axis (in respect to current section), a loading force
       giving positive shear yields a negative moment, as below (the
       curved arrow shows the positive moment and rotation):

    .. image:: allowed-sign-conventions.png

    Examples
    ========
    There is a beam of length 4 meters. A constant distributed load of 6 N/m
    is applied from half of the beam till the end. There are two simple supports
    below the beam, one at the starting point and another at the ending point
    of the beam. The deflection of the beam at the end is restricted.

    Using the sign convention of downwards forces being positive.

    >>> from sympy.physics.continuum_mechanics.beam import Beam
    >>> from sympy import symbols, Piecewise
    >>> E, I = symbols('E, I')
    >>> R1, R2 = symbols('R1, R2')
    >>> b = Beam(4, E, I)
    >>> b.apply_load(R1, 0, -1)
    >>> b.apply_load(6, 2, 0)
    >>> b.apply_load(R2, 4, -1)
    >>> b.bc_deflection = [(0, 0), (4, 0)]
    >>> b.boundary_conditions
    {'bending_moment': [], 'deflection': [(0, 0), (4, 0)], 'shear_force': [], 'slope': []}
    >>> b.load
    R1*SingularityFunction(x, 0, -1) + R2*SingularityFunction(x, 4, -1) + 6*SingularityFunction(x, 2, 0)
    >>> b.solve_for_reaction_loads(R1, R2)
    >>> b.load
    -3*SingularityFunction(x, 0, -1) + 6*SingularityFunction(x, 2, 0) - 9*SingularityFunction(x, 4, -1)
    >>> b.shear_force()
    3*SingularityFunction(x, 0, 0) - 6*SingularityFunction(x, 2, 1) + 9*SingularityFunction(x, 4, 0)
    >>> b.bending_moment()
    3*SingularityFunction(x, 0, 1) - 3*SingularityFunction(x, 2, 2) + 9*SingularityFunction(x, 4, 1)
    >>> b.slope()
    (-3*SingularityFunction(x, 0, 2)/2 + SingularityFunction(x, 2, 3) - 9*SingularityFunction(x, 4, 2)/2 + 7)/(E*I)
    >>> b.deflection()
    (7*x - SingularityFunction(x, 0, 3)/2 + SingularityFunction(x, 2, 4)/4 - 3*SingularityFunction(x, 4, 3)/2)/(E*I)
    >>> b.deflection().rewrite(Piecewise)
    (7*x - Piecewise((x**3, x >= 0), (0, True))/2
         - 3*Piecewise(((x - 4)**3, x >= 4), (0, True))/2
         + Piecewise(((x - 2)**4, x >= 2), (0, True))/4)/(E*I)

    Calculate the support reactions for a fully symbolic beam of length L.
    There are two simple supports below the beam, one at the starting point
    and another at the ending point of the beam. The deflection of the beam
    at the end is restricted. The beam is loaded with:

    * a downward point load P1 applied at L/4
    * an upward point load P2 applied at L/8
    * a counterclockwise moment M1 applied at L/2
    * a clockwise moment M2 applied at 3*L/4
    * a distributed constant load q1, applied downward, starting from L/2
      up to 3*L/4
    * a distributed constant load q2, applied upward, starting from 3*L/4
      up to L

    No assumptions are needed for symbolic loads. However, defining a positive
    length will help the algorithm to compute the solution.

    >>> E, I = symbols('E, I')
    >>> L = symbols("L", positive=True)
    >>> P1, P2, M1, M2, q1, q2 = symbols("P1, P2, M1, M2, q1, q2")
    >>> R1, R2 = symbols('R1, R2')
    >>> b = Beam(L, E, I)
    >>> b.apply_load(R1, 0, -1)
    >>> b.apply_load(R2, L, -1)
    >>> b.apply_load(P1, L/4, -1)
    >>> b.apply_load(-P2, L/8, -1)
    >>> b.apply_load(M1, L/2, -2)
    >>> b.apply_load(-M2, 3*L/4, -2)
    >>> b.apply_load(q1, L/2, 0, 3*L/4)
    >>> b.apply_load(-q2, 3*L/4, 0, L)
    >>> b.bc_deflection = [(0, 0), (L, 0)]
    >>> b.solve_for_reaction_loads(R1, R2)
    >>> print(b.reaction_loads[R1])
    (-3*L**2*q1 + L**2*q2 - 24*L*P1 + 28*L*P2 - 32*M1 + 32*M2)/(32*L)
    >>> print(b.reaction_loads[R2])
    (-5*L**2*q1 + 7*L**2*q2 - 8*L*P1 + 4*L*P2 + 32*M1 - 32*M2)/(32*L)
    """

    def __init__(self, length, elastic_modulus, second_moment, area=Symbol('A'), variable=Symbol('x'), base_char='C', ild_variable=Symbol('a')):
        """Initializes the class.

        Parameters
        ==========

        length : Sympifyable
            A Symbol or value representing the Beam's length.

        elastic_modulus : Sympifyable
            A SymPy expression representing the Beam's Modulus of Elasticity.
            It is a measure of the stiffness of the Beam material. It can
            also be a continuous function of position along the beam.

        second_moment : Sympifyable or Geometry object
            Describes the cross-section of the beam via a SymPy expression
            representing the Beam's second moment of area. It is a geometrical
            property of an area which reflects how its points are distributed
            with respect to its neutral axis. It can also be a continuous
            function of position along the beam. Alternatively ``second_moment``
            can be a shape object such as a ``Polygon`` from the geometry module
            representing the shape of the cross-section of the beam. In such cases,
            it is assumed that the x-axis of the shape object is aligned with the
            bending axis of the beam. The second moment of area will be computed
            from the shape object internally.

        area : Symbol/float
            Represents the cross-section area of beam

        variable : Symbol, optional
            A Symbol object that will be used as the variable along the beam
            while representing the load, shear, moment, slope and deflection
            curve. By default, it is set to ``Symbol('x')``.

        base_char : String, optional
            A String that will be used as base character to generate sequential
            symbols for integration constants in cases where boundary conditions
            are not sufficient to solve them.

        ild_variable : Symbol, optional
            A Symbol object that will be used as the variable specifying the
            location of the moving load in ILD calculations. By default, it
            is set to ``Symbol('a')``.
        """
        self.length = length
        self.elastic_modulus = elastic_modulus
        if isinstance(second_moment, GeometryEntity):
            self.cross_section = second_moment
        else:
            self.cross_section = None
            self.second_moment = second_moment
        self.variable = variable
        self.ild_variable = ild_variable
        self._base_char = base_char
        self._boundary_conditions = {'deflection': [], 'slope': [], 'bending_moment': [], 'shear_force': []}
        self._load = 0
        self.area = area
        self._applied_supports = []
        self._applied_rotation_hinges = []
        self._applied_sliding_hinges = []
        self._rotation_hinge_symbols = []
        self._sliding_hinge_symbols = []
        self._support_as_loads = []
        self._applied_loads = []
        self._reaction_loads = {}
        self._ild_reactions = {}
        self._ild_shear = 0
        self._ild_moment = 0
        # _original_load is a copy of _load equations with unsubstituted reaction
        # forces. It is used for calculating reaction forces in case of I.L.D.
        self._original_load = 0
        self._joined_beam = False

    def __str__(self):
        shape_description = self._cross_section if self._cross_section else self._second_moment
        str_sol = 'Beam({}, {}, {})'.format(sstr(self._length), sstr(self._elastic_modulus), sstr(shape_description))
        return str_sol

    @property
    def reaction_loads(self):
        """ Returns the reaction forces in a dictionary."""
        return self._reaction_loads

    @property
    def rotation_jumps(self):
        """
        Returns the value for the rotation jumps in rotation hinges in a dictionary.
        The rotation jump is the rotation (in radian) in a rotation hinge. This can
        be seen as a jump in the slope plot.
        """
        return self._rotation_jumps

    @property
    def deflection_jumps(self):
        """
        Returns the deflection jumps in sliding hinges in a dictionary.
        The deflection jump is the deflection (in meters) in a sliding hinge.
        This can be seen as a jump in the deflection plot.
        """
        return self._deflection_jumps

    @property
    def ild_shear(self):
        """ Returns the I.L.D. shear equation."""
        return self._ild_shear

    @property
    def ild_reactions(self):
        """ Returns the I.L.D. reaction forces in a dictionary."""
        return self._ild_reactions

    @property
    def ild_rotation_jumps(self):
        """
        Returns the I.L.D. rotation jumps in rotation hinges in a dictionary.
        The rotation jump is the rotation (in radian) in a rotation hinge. This can
        be seen as a jump in the slope plot.
        """
        return self._ild_rotations_jumps

    @property
    def ild_deflection_jumps(self):
        """
        Returns the I.L.D. deflection jumps in sliding hinges in a dictionary.
        The deflection jump is the deflection (in meters) in a sliding hinge.
        This can be seen as a jump in the deflection plot.
        """
        return self._ild_deflection_jumps

    @property
    def ild_moment(self):
        """ Returns the I.L.D. moment equation."""
        return self._ild_moment

    @property
    def length(self):
        """Length of the Beam."""
        return self._length

    @length.setter
    def length(self, l):
        self._length = sympify(l)

    @property
    def area(self):
        """Cross-sectional area of the Beam. """
        return self._area

    @area.setter
    def area(self, a):
        self._area = sympify(a)

    @property
    def variable(self):
        """
        A symbol that can be used as a variable along the length of the beam
        while representing load distribution, shear force curve, bending
        moment, slope curve and the deflection curve. By default, it is set
        to ``Symbol('x')``, but this property is mutable.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I, A = symbols('E, I, A')
        >>> x, y, z = symbols('x, y, z')
        >>> b = Beam(4, E, I)
        >>> b.variable
        x
        >>> b.variable = y
        >>> b.variable
        y
        >>> b = Beam(4, E, I, A, z)
        >>> b.variable
        z
        """
        return self._variable

    @variable.setter
    def variable(self, v):
        if isinstance(v, Symbol):
            self._variable = v
        else:
            raise TypeError("""The variable should be a Symbol object.""")

    @property
    def elastic_modulus(self):
        """Young's Modulus of the Beam. """
        return self._elastic_modulus

    @elastic_modulus.setter
    def elastic_modulus(self, e):
        self._elastic_modulus = sympify(e)

    @property
    def second_moment(self):
        """Second moment of area of the Beam. """
        return self._second_moment

    @second_moment.setter
    def second_moment(self, i):
        self._cross_section = None
        if isinstance(i, GeometryEntity):
            raise ValueError("To update cross-section geometry use `cross_section` attribute")
        else:
            self._second_moment = sympify(i)

    @property
    def cross_section(self):
        """Cross-section of the beam"""
        return self._cross_section

    @cross_section.setter
    def cross_section(self, s):
        if s:
            self._second_moment = s.second_moment_of_area()[0]
        self._cross_section = s

    @property
    def boundary_conditions(self):
        """
        Returns a dictionary of boundary conditions applied on the beam.
        The dictionary has three keywords namely moment, slope and deflection.
        The value of each keyword is a list of tuple, where each tuple
        contains location and value of a boundary condition in the format
        (location, value).

        Examples
        ========
        There is a beam of length 4 meters. The bending moment at 0 should be 4
        and at 4 it should be 0. The slope of the beam should be 1 at 0. The
        deflection should be 2 at 0.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> b = Beam(4, E, I)
        >>> b.bc_deflection = [(0, 2)]
        >>> b.bc_slope = [(0, 1)]
        >>> b.boundary_conditions
        {'bending_moment': [], 'deflection': [(0, 2)], 'shear_force': [], 'slope': [(0, 1)]}

        Here the deflection of the beam should be ``2`` at ``0``.
        Similarly, the slope of the beam should be ``1`` at ``0``.
        """
        return self._boundary_conditions

    @property
    def bc_shear_force(self):
        return self._boundary_conditions['shear_force']

    @bc_shear_force.setter
    def bc_shear_force(self, sf_bcs):
        self._boundary_conditions['shear_force'] = sf_bcs

    @property
    def bc_bending_moment(self):
        return self._boundary_conditions['bending_moment']

    @bc_bending_moment.setter
    def bc_bending_moment(self, bm_bcs):
        self._boundary_conditions['bending_moment'] = bm_bcs

    @property
    def bc_slope(self):
        return self._boundary_conditions['slope']

    @bc_slope.setter
    def bc_slope(self, s_bcs):
        self._boundary_conditions['slope'] = s_bcs

    @property
    def bc_deflection(self):
        return self._boundary_conditions['deflection']

    @bc_deflection.setter
    def bc_deflection(self, d_bcs):
        self._boundary_conditions['deflection'] = d_bcs

    def join(self, beam, via="fixed"):
        """
        This method joins two beams to make a new composite beam system.
        Passed Beam class instance is attached to the right end of calling
        object. This method can be used to form beams having Discontinuous
        values of Elastic modulus or Second moment.

        Parameters
        ==========
        beam : Beam class object
            The Beam object which would be connected to the right of calling
            object.
        via : String
            States the way two Beam object would get connected
            - For axially fixed Beams, via="fixed"
            - For Beams connected via rotation hinge, via="hinge"

        Examples
        ========
        There is a cantilever beam of length 4 meters. For first 2 meters
        its moment of inertia is `1.5*I` and `I` for the other end.
        A pointload of magnitude 4 N is applied from the top at its free end.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> R1, R2 = symbols('R1, R2')
        >>> b1 = Beam(2, E, 1.5*I)
        >>> b2 = Beam(2, E, I)
        >>> b = b1.join(b2, "fixed")
        >>> b.apply_load(20, 4, -1)
        >>> b.apply_load(R1, 0, -1)
        >>> b.apply_load(R2, 0, -2)
        >>> b.bc_slope = [(0, 0)]
        >>> b.bc_deflection = [(0, 0)]
        >>> b.solve_for_reaction_loads(R1, R2)
        >>> b.load
        80*SingularityFunction(x, 0, -2) - 20*SingularityFunction(x, 0, -1) + 20*SingularityFunction(x, 4, -1)
        >>> b.slope()
        (-((-80*SingularityFunction(x, 0, 1) + 10*SingularityFunction(x, 0, 2) - 10*SingularityFunction(x, 4, 2))/I + 120/I)/E + 80.0/(E*I))*SingularityFunction(x, 2, 0)
        - 0.666666666666667*(-80*SingularityFunction(x, 0, 1) + 10*SingularityFunction(x, 0, 2) - 10*SingularityFunction(x, 4, 2))*SingularityFunction(x, 0, 0)/(E*I)
        + 0.666666666666667*(-80*SingularityFunction(x, 0, 1) + 10*SingularityFunction(x, 0, 2) - 10*SingularityFunction(x, 4, 2))*SingularityFunction(x, 2, 0)/(E*I)
        """
        x = self.variable
        E = self.elastic_modulus
        new_length = self.length + beam.length
        if self.elastic_modulus != beam.elastic_modulus:
            raise NotImplementedError('Joining beams with different Elastic modulus is not implemented.')

        if self.second_moment != beam.second_moment:
            new_second_moment = Piecewise((self.second_moment, x<=self.length),
                                    (beam.second_moment, x<=new_length))
        else:
            new_second_moment = self.second_moment

        if via == "fixed":
            new_beam = Beam(new_length, E, new_second_moment, x)
            new_beam._joined_beam = True
            return new_beam

        if via == "hinge":
            new_beam = Beam(new_length, E, new_second_moment, x)
            new_beam._joined_beam = True
            new_beam.apply_rotation_hinge(self.length)
            return new_beam

    def apply_support(self, loc, type="fixed"):
        """
        This method applies support to a particular beam object and returns
        the symbol of the unknown reaction load(s).

        Parameters
        ==========
        loc : Sympifyable
            Location of point at which support is applied.
        type : String
            Determines type of Beam support applied. To apply support structure
            with
            - zero degree of freedom, type = "fixed"
            - one degree of freedom, type = "pin"
            - two degrees of freedom, type = "roller"

        Returns
        =======
        Symbol or tuple of Symbol
            The unknown reaction load as a symbol.
            - Symbol(reaction_force) if type = "pin" or "roller"
            - Symbol(reaction_force), Symbol(reaction_moment) if type = "fixed"

        Examples
        ========
        There is a beam of length 20 meters. A moment of magnitude 100 Nm is
        applied in the clockwise direction at the end of the beam. A pointload
        of magnitude 8 N is applied from the top of the beam at a distance of 10 meters.
        There is one fixed support at the start of the beam and a roller at the end.

        Using the sign convention of upward forces and clockwise moment
        being positive.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> b = Beam(20, E, I)
        >>> p0, m0 = b.apply_support(0, 'fixed')
        >>> p1 = b.apply_support(20, 'roller')
        >>> b.apply_load(-8, 10, -1)
        >>> b.apply_load(100, 20, -2)
        >>> b.solve_for_reaction_loads(p0, m0, p1)
        >>> b.reaction_loads
        {M_0: 20, R_0: -2, R_20: 10}
        >>> b.reaction_loads[p0]
        -2
        >>> b.load
        20*SingularityFunction(x, 0, -2) - 2*SingularityFunction(x, 0, -1)
        - 8*SingularityFunction(x, 10, -1) + 100*SingularityFunction(x, 20, -2)
        + 10*SingularityFunction(x, 20, -1)
        """
        loc = sympify(loc)

        self._applied_supports.append((loc, type))
        if type in ("pin", "roller"):
            reaction_load = Symbol('R_'+str(loc))
            self.apply_load(reaction_load, loc, -1)
            self.bc_deflection.append((loc, 0))
        else:
            reaction_load = Symbol('R_'+str(loc))
            reaction_moment = Symbol('M_'+str(loc))
            self.apply_load(reaction_load, loc, -1)
            self.apply_load(reaction_moment, loc, -2)
            self.bc_deflection.append((loc, 0))
            self.bc_slope.append((loc, 0))
            self._support_as_loads.append((reaction_moment, loc, -2, None))

        self._support_as_loads.append((reaction_load, loc, -1, None))

        if type in ("pin", "roller"):
            return reaction_load
        else:
            return reaction_load, reaction_moment

    def _get_I(self, loc):
        """
        Helper function that returns the Second moment (I) at a location in the beam.
        """
        I = self.second_moment
        if not isinstance(I, Piecewise):
            return I
        else:
            for i in range(len(I.args)):
                if loc <= I.args[i][1].args[1]:
                    return I.args[i][0]

    def apply_rotation_hinge(self, loc):
        """
        This method applies a rotation hinge at a single location on the beam.

        Parameters
        ----------
        loc : Sympifyable
            Location of point at which hinge is applied.

        Returns
        =======
        Symbol
            The unknown rotation jump multiplied by the elastic modulus and second moment as a symbol.

        Examples
        ========
        There is a beam of length 15 meters. Pin supports are placed at distances
        of 0 and 10 meters. There is a fixed support at the end. There are two rotation hinges
        in the structure, one at 5 meters and one at 10 meters. A pointload of magnitude
        10 kN is applied on the hinge at 5 meters. A distributed load of 5 kN works on
        the structure from 10 meters to the end.

        Using the sign convention of upward forces and clockwise moment
        being positive.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import Symbol
        >>> E = Symbol('E')
        >>> I = Symbol('I')
        >>> b = Beam(15, E, I)
        >>> r0 = b.apply_support(0, type='pin')
        >>> r10 = b.apply_support(10, type='pin')
        >>> r15, m15 = b.apply_support(15, type='fixed')
        >>> p5 = b.apply_rotation_hinge(5)
        >>> p12 = b.apply_rotation_hinge(12)
        >>> b.apply_load(-10, 5, -1)
        >>> b.apply_load(-5, 10, 0, 15)
        >>> b.solve_for_reaction_loads(r0, r10, r15, m15)
        >>> b.reaction_loads
        {M_15: -75/2, R_0: 0, R_10: 40, R_15: -5}
        >>> b.rotation_jumps
        {P_12: -1875/(16*E*I), P_5: 9625/(24*E*I)}
        >>> b.rotation_jumps[p12]
        -1875/(16*E*I)
        >>> b.bending_moment()
        -9625*SingularityFunction(x, 5, -1)/24 + 10*SingularityFunction(x, 5, 1)
        - 40*SingularityFunction(x, 10, 1) + 5*SingularityFunction(x, 10, 2)/2
        + 1875*SingularityFunction(x, 12, -1)/16 + 75*SingularityFunction(x, 15, 0)/2
        + 5*SingularityFunction(x, 15, 1) - 5*SingularityFunction(x, 15, 2)/2
        """
        loc = sympify(loc)
        E = self.elastic_modulus
        I = self._get_I(loc)

        rotation_jump = Symbol('P_'+str(loc))
        self._applied_rotation_hinges.append(loc)
        self._rotation_hinge_symbols.append(rotation_jump)
        self.apply_load(E * I * rotation_jump, loc, -3)
        self.bc_bending_moment.append((loc, 0))
        return rotation_jump

    def apply_sliding_hinge(self, loc):
        """
        This method applies a sliding hinge at a single location on the beam.

        Parameters
        ----------
        loc : Sympifyable
            Location of point at which hinge is applied.

        Returns
        =======
        Symbol
            The unknown deflection jump multiplied by the elastic modulus and second moment as a symbol.

        Examples
        ========
        There is a beam of length 13 meters. A fixed support is placed at the beginning.
        There is a pin support at the end. There is a sliding hinge at a location of 8 meters.
        A pointload of magnitude 10 kN is applied on the hinge at 5 meters.

        Using the sign convention of upward forces and clockwise moment
        being positive.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> b = Beam(13, 20, 20)
        >>> r0, m0 = b.apply_support(0, type="fixed")
        >>> s8 = b.apply_sliding_hinge(8)
        >>> r13 = b.apply_support(13, type="pin")
        >>> b.apply_load(-10, 5, -1)
        >>> b.solve_for_reaction_loads(r0, m0, r13)
        >>> b.reaction_loads
        {M_0: -50, R_0: 10, R_13: 0}
        >>> b.deflection_jumps
        {W_8: 85/24}
        >>> b.deflection_jumps[s8]
        85/24
        >>> b.bending_moment()
        50*SingularityFunction(x, 0, 0) - 10*SingularityFunction(x, 0, 1)
        + 10*SingularityFunction(x, 5, 1) - 4250*SingularityFunction(x, 8, -2)/3
        >>> b.deflection()
        -SingularityFunction(x, 0, 2)/16 + SingularityFunction(x, 0, 3)/240
        - SingularityFunction(x, 5, 3)/240 + 85*SingularityFunction(x, 8, 0)/24
        """
        loc = sympify(loc)
        E = self.elastic_modulus
        I = self._get_I(loc)

        deflection_jump = Symbol('W_' + str(loc))
        self._applied_sliding_hinges.append(loc)
        self._sliding_hinge_symbols.append(deflection_jump)
        self.apply_load(E * I * deflection_jump, loc, -4)
        self.bc_shear_force.append((loc, 0))
        return deflection_jump

    def apply_load(self, value, start, order, end=None):
        """
        This method adds up the loads given to a particular beam object.

        Parameters
        ==========
        value : Sympifyable
            The value inserted should have the units [Force/(Distance**(n+1)]
            where n is the order of applied load.
            Units for applied loads:

               - For moments, unit = kN*m
               - For point loads, unit = kN
               - For constant distributed load, unit = kN/m
               - For ramp loads, unit = kN/m/m
               - For parabolic ramp loads, unit = kN/m/m/m
               - ... so on.

        start : Sympifyable
            The starting point of the applied load. For point moments and
            point forces this is the location of application.
        order : Integer
            The order of the applied load.

               - For moments, order = -2
               - For point loads, order =-1
               - For constant distributed load, order = 0
               - For ramp loads, order = 1
               - For parabolic ramp loads, order = 2
               - ... so on.

        end : Sympifyable, optional
            An optional argument that can be used if the load has an end point
            within the length of the beam.

        Examples
        ========
        There is a beam of length 4 meters. A moment of magnitude 3 Nm is
        applied in the clockwise direction at the starting point of the beam.
        A point load of magnitude 4 N is applied from the top of the beam at
        2 meters from the starting point and a parabolic ramp load of magnitude
        2 N/m is applied below the beam starting from 2 meters to 3 meters
        away from the starting point of the beam.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> b = Beam(4, E, I)
        >>> b.apply_load(-3, 0, -2)
        >>> b.apply_load(4, 2, -1)
        >>> b.apply_load(-2, 2, 2, end=3)
        >>> b.load
        -3*SingularityFunction(x, 0, -2) + 4*SingularityFunction(x, 2, -1) - 2*SingularityFunction(x, 2, 2) + 2*SingularityFunction(x, 3, 0) + 4*SingularityFunction(x, 3, 1) + 2*SingularityFunction(x, 3, 2)

        """
        x = self.variable
        value = sympify(value)
        start = sympify(start)
        order = sympify(order)

        self._applied_loads.append((value, start, order, end))
        self._load += value*SingularityFunction(x, start, order)
        self._original_load += value*SingularityFunction(x, start, order)

        if end:
            # load has an end point within the length of the beam.
            self._handle_end(x, value, start, order, end, type="apply")

    def remove_load(self, value, start, order, end=None):
        """
        This method removes a particular load present on the beam object.
        Returns a ValueError if the load passed as an argument is not
        present on the beam.

        Parameters
        ==========
        value : Sympifyable
            The magnitude of an applied load.
        start : Sympifyable
            The starting point of the applied load. For point moments and
            point forces this is the location of application.
        order : Integer
            The order of the applied load.
            - For moments, order= -2
            - For point loads, order=-1
            - For constant distributed load, order=0
            - For ramp loads, order=1
            - For parabolic ramp loads, order=2
            - ... so on.
        end : Sympifyable, optional
            An optional argument that can be used if the load has an end point
            within the length of the beam.

        Examples
        ========
        There is a beam of length 4 meters. A moment of magnitude 3 Nm is
        applied in the clockwise direction at the starting point of the beam.
        A pointload of magnitude 4 N is applied from the top of the beam at
        2 meters from the starting point and a parabolic ramp load of magnitude
        2 N/m is applied below the beam starting from 2 meters to 3 meters
        away from the starting point of the beam.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> b = Beam(4, E, I)
        >>> b.apply_load(-3, 0, -2)
        >>> b.apply_load(4, 2, -1)
        >>> b.apply_load(-2, 2, 2, end=3)
        >>> b.load
        -3*SingularityFunction(x, 0, -2) + 4*SingularityFunction(x, 2, -1) - 2*SingularityFunction(x, 2, 2) + 2*SingularityFunction(x, 3, 0) + 4*SingularityFunction(x, 3, 1) + 2*SingularityFunction(x, 3, 2)
        >>> b.remove_load(-2, 2, 2, end = 3)
        >>> b.load
        -3*SingularityFunction(x, 0, -2) + 4*SingularityFunction(x, 2, -1)
        """
        x = self.variable
        value = sympify(value)
        start = sympify(start)
        order = sympify(order)

        if (value, start, order, end) in self._applied_loads:
            self._load -= value*SingularityFunction(x, start, order)
            self._original_load -= value*SingularityFunction(x, start, order)
            self._applied_loads.remove((value, start, order, end))
        else:
            msg = "No such load distribution exists on the beam object."
            raise ValueError(msg)

        if end:
            # load has an end point within the length of the beam.
            self._handle_end(x, value, start, order, end, type="remove")

    def _handle_end(self, x, value, start, order, end, type):
        """
        This functions handles the optional `end` value in the
        `apply_load` and `remove_load` functions. When the value
        of end is not NULL, this function will be executed.
        """
        if order.is_negative:
            msg = ("If 'end' is provided the 'order' of the load cannot "
                    "be negative, i.e. 'end' is only valid for distributed "
                    "loads.")
            raise ValueError(msg)
        # NOTE : A Taylor series can be used to define the summation of
        # singularity functions that subtract from the load past the end
        # point such that it evaluates to zero past 'end'.
        f = value*x**order

        if type == "apply":
            # iterating for "apply_load" method
            for i in range(0, order + 1):
                self._load -= (f.diff(x, i).subs(x, end - start) *
                                SingularityFunction(x, end, i)/factorial(i))
                self._original_load -= (f.diff(x, i).subs(x, end - start) *
                                SingularityFunction(x, end, i)/factorial(i))
        elif type == "remove":
            # iterating for "remove_load" method
            for i in range(0, order + 1):
                self._load += (f.diff(x, i).subs(x, end - start) *
                                SingularityFunction(x, end, i)/factorial(i))
                self._original_load += (f.diff(x, i).subs(x, end - start) *
                                SingularityFunction(x, end, i)/factorial(i))


    @property
    def load(self):
        """
        Returns a Singularity Function expression which represents
        the load distribution curve of the Beam object.

        Examples
        ========
        There is a beam of length 4 meters. A moment of magnitude 3 Nm is
        applied in the clockwise direction at the starting point of the beam.
        A point load of magnitude 4 N is applied from the top of the beam at
        2 meters from the starting point and a parabolic ramp load of magnitude
        2 N/m is applied below the beam starting from 3 meters away from the
        starting point of the beam.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> b = Beam(4, E, I)
        >>> b.apply_load(-3, 0, -2)
        >>> b.apply_load(4, 2, -1)
        >>> b.apply_load(-2, 3, 2)
        >>> b.load
        -3*SingularityFunction(x, 0, -2) + 4*SingularityFunction(x, 2, -1) - 2*SingularityFunction(x, 3, 2)
        """
        return self._load

    @property
    def applied_loads(self):
        """
        Returns a list of all loads applied on the beam object.
        Each load in the list is a tuple of form (value, start, order, end).

        Examples
        ========
        There is a beam of length 4 meters. A moment of magnitude 3 Nm is
        applied in the clockwise direction at the starting point of the beam.
        A pointload of magnitude 4 N is applied from the top of the beam at
        2 meters from the starting point. Another pointload of magnitude 5 N
        is applied at same position.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> b = Beam(4, E, I)
        >>> b.apply_load(-3, 0, -2)
        >>> b.apply_load(4, 2, -1)
        >>> b.apply_load(5, 2, -1)
        >>> b.load
        -3*SingularityFunction(x, 0, -2) + 9*SingularityFunction(x, 2, -1)
        >>> b.applied_loads
        [(-3, 0, -2, None), (4, 2, -1, None), (5, 2, -1, None)]
        """
        return self._applied_loads

    def solve_for_reaction_loads(self, *reactions):
        """
        Solves for the reaction forces.

        Examples
        ========
        There is a beam of length 30 meters. A moment of magnitude 120 Nm is
        applied in the clockwise direction at the end of the beam. A pointload
        of magnitude 8 N is applied from the top of the beam at the starting
        point. There are two simple supports below the beam. One at the end
        and another one at a distance of 10 meters from the start. The
        deflection is restricted at both the supports.

        Using the sign convention of upward forces and clockwise moment
        being positive.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> R1, R2 = symbols('R1, R2')
        >>> b = Beam(30, E, I)
        >>> b.apply_load(-8, 0, -1)
        >>> b.apply_load(R1, 10, -1)  # Reaction force at x = 10
        >>> b.apply_load(R2, 30, -1)  # Reaction force at x = 30
        >>> b.apply_load(120, 30, -2)
        >>> b.bc_deflection = [(10, 0), (30, 0)]
        >>> b.load
        R1*SingularityFunction(x, 10, -1) + R2*SingularityFunction(x, 30, -1)
            - 8*SingularityFunction(x, 0, -1) + 120*SingularityFunction(x, 30, -2)
        >>> b.solve_for_reaction_loads(R1, R2)
        >>> b.reaction_loads
        {R1: 6, R2: 2}
        >>> b.load
        -8*SingularityFunction(x, 0, -1) + 6*SingularityFunction(x, 10, -1)
            + 120*SingularityFunction(x, 30, -2) + 2*SingularityFunction(x, 30, -1)
        """

        x = self.variable
        l = self.length
        C3 = Symbol('C3')
        C4 = Symbol('C4')
        rotation_jumps = tuple(self._rotation_hinge_symbols)
        deflection_jumps = tuple(self._sliding_hinge_symbols)

        shear_curve = limit(self.shear_force(), x, l)
        moment_curve = limit(self.bending_moment(), x, l)

        shear_force_eqs = []
        bending_moment_eqs = []
        slope_eqs = []
        deflection_eqs = []

        for position, value in self._boundary_conditions['shear_force']:
            eqs = self.shear_force().subs(x, position) - value
            new_eqs = sum(arg for arg in eqs.args if not any(num.is_infinite for num in arg.args))
            shear_force_eqs.append(new_eqs)

        for position, value in self._boundary_conditions['bending_moment']:
            eqs = self.bending_moment().subs(x, position) - value
            new_eqs = sum(arg for arg in eqs.args if not any(num.is_infinite for num in arg.args))
            bending_moment_eqs.append(new_eqs)

        slope_curve = integrate(self.bending_moment(), x) + C3
        for position, value in self._boundary_conditions['slope']:
            eqs = slope_curve.subs(x, position) - value
            slope_eqs.append(eqs)

        deflection_curve = integrate(slope_curve, x) + C4
        for position, value in self._boundary_conditions['deflection']:
            eqs = deflection_curve.subs(x, position) - value
            deflection_eqs.append(eqs)

        solution = list((linsolve([shear_curve, moment_curve] + shear_force_eqs + bending_moment_eqs + slope_eqs
                            + deflection_eqs, (C3, C4) + reactions + rotation_jumps + deflection_jumps).args)[0])
        reaction_index = 2+len(reactions)
        rotation_index = reaction_index + len(rotation_jumps)
        reaction_solution = solution[2:reaction_index]
        rotation_solution = solution[reaction_index:rotation_index]
        deflection_solution = solution[rotation_index:]

        self._reaction_loads = dict(zip(reactions, reaction_solution))
        self._rotation_jumps = dict(zip(rotation_jumps, rotation_solution))
        self._deflection_jumps = dict(zip(deflection_jumps, deflection_solution))
        self._load = self._load.subs(self._reaction_loads)
        self._load = self._load.subs(self._rotation_jumps)
        self._load = self._load.subs(self._deflection_jumps)

    def shear_force(self):
        """
        Returns a Singularity Function expression which represents
        the shear force curve of the Beam object.

        Examples
        ========
        There is a beam of length 30 meters. A moment of magnitude 120 Nm is
        applied in the clockwise direction at the end of the beam. A pointload
        of magnitude 8 N is applied from the top of the beam at the starting
        point. There are two simple supports below the beam. One at the end
        and another one at a distance of 10 meters from the start. The
        deflection is restricted at both the supports.

        Using the sign convention of upward forces and clockwise moment
        being positive.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> R1, R2 = symbols('R1, R2')
        >>> b = Beam(30, E, I)
        >>> b.apply_load(-8, 0, -1)
        >>> b.apply_load(R1, 10, -1)
        >>> b.apply_load(R2, 30, -1)
        >>> b.apply_load(120, 30, -2)
        >>> b.bc_deflection = [(10, 0), (30, 0)]
        >>> b.solve_for_reaction_loads(R1, R2)
        >>> b.shear_force()
        8*SingularityFunction(x, 0, 0) - 6*SingularityFunction(x, 10, 0) - 120*SingularityFunction(x, 30, -1) - 2*SingularityFunction(x, 30, 0)
        """
        x = self.variable
        return -integrate(self.load, x)

    def max_shear_force(self):
        """Returns maximum Shear force and its coordinate
        in the Beam object."""
        shear_curve = self.shear_force()
        x = self.variable

        terms = shear_curve.args
        singularity = []        # Points at which shear function changes
        for term in terms:
            if isinstance(term, Mul):
                term = term.args[-1]    # SingularityFunction in the term
            singularity.append(term.args[1])
        singularity = list(set(singularity))
        singularity.sort()

        intervals = []    # List of Intervals with discrete value of shear force
        shear_values = []   # List of values of shear force in each interval
        for i, s in enumerate(singularity):
            if s == 0:
                continue
            try:
                shear_slope = Piecewise((float("nan"), x<=singularity[i-1]),(self._load.rewrite(Piecewise), x<s), (float("nan"), True))
                points = solve(shear_slope, x)
                val = []
                for point in points:
                    val.append(abs(shear_curve.subs(x, point)))
                points.extend([singularity[i-1], s])
                val += [abs(limit(shear_curve, x, singularity[i-1], '+')), abs(limit(shear_curve, x, s, '-'))]
                max_shear = max(val)
                shear_values.append(max_shear)
                intervals.append(points[val.index(max_shear)])
            # If shear force in a particular Interval has zero or constant
            # slope, then above block gives NotImplementedError as
            # solve can't represent Interval solutions.
            except NotImplementedError:
                initial_shear = limit(shear_curve, x, singularity[i-1], '+')
                final_shear = limit(shear_curve, x, s, '-')
                # If shear_curve has a constant slope(it is a line).
                if shear_curve.subs(x, (singularity[i-1] + s)/2) == (initial_shear + final_shear)/2 and initial_shear != final_shear:
                    shear_values.extend([initial_shear, final_shear])
                    intervals.extend([singularity[i-1], s])
                else:    # shear_curve has same value in whole Interval
                    shear_values.append(final_shear)
                    intervals.append(Interval(singularity[i-1], s))

        shear_values = list(map(abs, shear_values))
        maximum_shear = max(shear_values)
        point = intervals[shear_values.index(maximum_shear)]
        return (point, maximum_shear)

    def bending_moment(self):
        """
        Returns a Singularity Function expression which represents
        the bending moment curve of the Beam object.

        Examples
        ========
        There is a beam of length 30 meters. A moment of magnitude 120 Nm is
        applied in the clockwise direction at the end of the beam. A pointload
        of magnitude 8 N is applied from the top of the beam at the starting
        point. There are two simple supports below the beam. One at the end
        and another one at a distance of 10 meters from the start. The
        deflection is restricted at both the supports.

        Using the sign convention of upward forces and clockwise moment
        being positive.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> R1, R2 = symbols('R1, R2')
        >>> b = Beam(30, E, I)
        >>> b.apply_load(-8, 0, -1)
        >>> b.apply_load(R1, 10, -1)
        >>> b.apply_load(R2, 30, -1)
        >>> b.apply_load(120, 30, -2)
        >>> b.bc_deflection = [(10, 0), (30, 0)]
        >>> b.solve_for_reaction_loads(R1, R2)
        >>> b.bending_moment()
        8*SingularityFunction(x, 0, 1) - 6*SingularityFunction(x, 10, 1) - 120*SingularityFunction(x, 30, 0) - 2*SingularityFunction(x, 30, 1)
        """
        x = self.variable
        return integrate(self.shear_force(), x)

    def max_bmoment(self):
        """Returns maximum Shear force and its coordinate
        in the Beam object."""
        bending_curve = self.bending_moment()
        x = self.variable

        terms = bending_curve.args
        singularity = []        # Points at which bending moment changes
        for term in terms:
            if isinstance(term, Mul):
                term = term.args[-1]    # SingularityFunction in the term
            singularity.append(term.args[1])
        singularity = list(set(singularity))
        singularity.sort()

        intervals = []    # List of Intervals with discrete value of bending moment
        moment_values = []   # List of values of bending moment in each interval
        for i, s in enumerate(singularity):
            if s == 0:
                continue
            try:
                moment_slope = Piecewise(
                    (float("nan"), x <= singularity[i - 1]),
                    (self.shear_force().rewrite(Piecewise), x < s),
                    (float("nan"), True))
                points = solve(moment_slope, x)
                val = []
                for point in points:
                    val.append(abs(bending_curve.subs(x, point)))
                points.extend([singularity[i-1], s])
                val += [abs(limit(bending_curve, x, singularity[i-1], '+')), abs(limit(bending_curve, x, s, '-'))]
                max_moment = max(val)
                moment_values.append(max_moment)
                intervals.append(points[val.index(max_moment)])

            # If bending moment in a particular Interval has zero or constant
            # slope, then above block gives NotImplementedError as solve
            # can't represent Interval solutions.
            except NotImplementedError:
                initial_moment = limit(bending_curve, x, singularity[i-1], '+')
                final_moment = limit(bending_curve, x, s, '-')
                # If bending_curve has a constant slope(it is a line).
                if bending_curve.subs(x, (singularity[i-1] + s)/2) == (initial_moment + final_moment)/2 and initial_moment != final_moment:
                    moment_values.extend([initial_moment, final_moment])
                    intervals.extend([singularity[i-1], s])
                else:    # bending_curve has same value in whole Interval
                    moment_values.append(final_moment)
                    intervals.append(Interval(singularity[i-1], s))

        moment_values = list(map(abs, moment_values))
        maximum_moment = max(moment_values)
        point = intervals[moment_values.index(maximum_moment)]
        return (point, maximum_moment)

    def point_cflexure(self):
        """
        Returns a Set of point(s) with zero bending moment and
        where bending moment curve of the beam object changes
        its sign from negative to positive or vice versa.

        Examples
        ========
        There is is 10 meter long overhanging beam. There are
        two simple supports below the beam. One at the start
        and another one at a distance of 6 meters from the start.
        Point loads of magnitude 10KN and 20KN are applied at
        2 meters and 4 meters from start respectively. A Uniformly
        distribute load of magnitude of magnitude 3KN/m is also
        applied on top starting from 6 meters away from starting
        point till end.
        Using the sign convention of upward forces and clockwise moment
        being positive.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> b = Beam(10, E, I)
        >>> b.apply_load(-4, 0, -1)
        >>> b.apply_load(-46, 6, -1)
        >>> b.apply_load(10, 2, -1)
        >>> b.apply_load(20, 4, -1)
        >>> b.apply_load(3, 6, 0)
        >>> b.point_cflexure()
        [10/3]
        """
        #Removes the singularity functions of order < 0 from the bending moment equation used in this method
        non_singular_bending_moment = sum(arg for arg in self.bending_moment().args if not arg.args[1].args[2] < 0)

        # To restrict the range within length of the Beam
        moment_curve = Piecewise((float("nan"), self.variable<=0),
                (non_singular_bending_moment, self.variable<self.length),
                (float("nan"), True))
        try:
            points = solve(moment_curve.rewrite(Piecewise), self.variable,
                           domain=S.Reals)
        except NotImplementedError as e:
            if "An expression is already zero when" in str(e):
                raise NotImplementedError("This method cannot be used when a whole region of "
                                          "the bending moment line is equal to 0.")
            else:
                raise

        return points

    def slope(self):
        """
        Returns a Singularity Function expression which represents
        the slope the elastic curve of the Beam object.

        Examples
        ========
        There is a beam of length 30 meters. A moment of magnitude 120 Nm is
        applied in the clockwise direction at the end of the beam. A pointload
        of magnitude 8 N is applied from the top of the beam at the starting
        point. There are two simple supports below the beam. One at the end
        and another one at a distance of 10 meters from the start. The
        deflection is restricted at both the supports.

        Using the sign convention of upward forces and clockwise moment
        being positive.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> R1, R2 = symbols('R1, R2')
        >>> b = Beam(30, E, I)
        >>> b.apply_load(-8, 0, -1)
        >>> b.apply_load(R1, 10, -1)
        >>> b.apply_load(R2, 30, -1)
        >>> b.apply_load(120, 30, -2)
        >>> b.bc_deflection = [(10, 0), (30, 0)]
        >>> b.solve_for_reaction_loads(R1, R2)
        >>> b.slope()
        (-4*SingularityFunction(x, 0, 2) + 3*SingularityFunction(x, 10, 2)
            + 120*SingularityFunction(x, 30, 1) + SingularityFunction(x, 30, 2) + 4000/3)/(E*I)
        """
        x = self.variable
        E = self.elastic_modulus
        I = self.second_moment

        if not self._boundary_conditions['slope']:
            return diff(self.deflection(), x)
        if isinstance(I, Piecewise) and self._joined_beam:
            args = I.args
            slope = 0
            prev_slope = 0
            prev_end = 0
            for i in range(len(args)):
                if i != 0:
                    prev_end = args[i-1][1].args[1]
                slope_value = -S.One/E*integrate(self.bending_moment()/args[i][0], (x, prev_end, x))
                if i != len(args) - 1:
                    slope += (prev_slope + slope_value)*SingularityFunction(x, prev_end, 0) - \
                        (prev_slope + slope_value)*SingularityFunction(x, args[i][1].args[1], 0)
                else:
                    slope += (prev_slope + slope_value)*SingularityFunction(x, prev_end, 0)
                prev_slope = slope_value.subs(x, args[i][1].args[1])
            return slope

        C3 = Symbol('C3')
        slope_curve = -integrate(S.One/(E*I)*self.bending_moment(), x) + C3

        bc_eqs = []
        for position, value in self._boundary_conditions['slope']:
            eqs = slope_curve.subs(x, position) - value
            bc_eqs.append(eqs)
        constants = list(linsolve(bc_eqs, C3))
        slope_curve = slope_curve.subs({C3: constants[0][0]})
        return slope_curve

    def deflection(self):
        """
        Returns a Singularity Function expression which represents
        the elastic curve or deflection of the Beam object.

        Examples
        ========
        There is a beam of length 30 meters. A moment of magnitude 120 Nm is
        applied in the clockwise direction at the end of the beam. A pointload
        of magnitude 8 N is applied from the top of the beam at the starting
        point. There are two simple supports below the beam. One at the end
        and another one at a distance of 10 meters from the start. The
        deflection is restricted at both the supports.

        Using the sign convention of upward forces and clockwise moment
        being positive.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> R1, R2 = symbols('R1, R2')
        >>> b = Beam(30, E, I)
        >>> b.apply_load(-8, 0, -1)
        >>> b.apply_load(R1, 10, -1)
        >>> b.apply_load(R2, 30, -1)
        >>> b.apply_load(120, 30, -2)
        >>> b.bc_deflection = [(10, 0), (30, 0)]
        >>> b.solve_for_reaction_loads(R1, R2)
        >>> b.deflection()
        (4000*x/3 - 4*SingularityFunction(x, 0, 3)/3 + SingularityFunction(x, 10, 3)
            + 60*SingularityFunction(x, 30, 2) + SingularityFunction(x, 30, 3)/3 - 12000)/(E*I)
        """
        x = self.variable
        E = self.elastic_modulus
        I = self.second_moment
        if not self._boundary_conditions['deflection'] and not self._boundary_conditions['slope']:
            if isinstance(I, Piecewise) and self._joined_beam:
                args = I.args
                prev_slope = 0
                prev_def = 0
                prev_end = 0
                deflection = 0
                for i in range(len(args)):
                    if i != 0:
                        prev_end = args[i-1][1].args[1]
                    slope_value = -S.One/E*integrate(self.bending_moment()/args[i][0], (x, prev_end, x))
                    recent_segment_slope = prev_slope + slope_value
                    deflection_value = integrate(recent_segment_slope, (x, prev_end, x))
                    if i != len(args) - 1:
                        deflection += (prev_def + deflection_value)*SingularityFunction(x, prev_end, 0) \
                            - (prev_def + deflection_value)*SingularityFunction(x, args[i][1].args[1], 0)
                    else:
                        deflection += (prev_def + deflection_value)*SingularityFunction(x, prev_end, 0)
                    prev_slope = slope_value.subs(x, args[i][1].args[1])
                    prev_def = deflection_value.subs(x, args[i][1].args[1])
                return deflection
            base_char = self._base_char
            constants = symbols(base_char + '3:5')
            return S.One/(E*I)*integrate(-integrate(self.bending_moment(), x), x) + constants[0]*x + constants[1]
        elif not self._boundary_conditions['deflection']:
            base_char = self._base_char
            constant = symbols(base_char + '4')
            return integrate(self.slope(), x) + constant
        elif not self._boundary_conditions['slope'] and self._boundary_conditions['deflection']:
            if isinstance(I, Piecewise) and self._joined_beam:
                args = I.args
                prev_slope = 0
                prev_def = 0
                prev_end = 0
                deflection = 0
                for i in range(len(args)):
                    if i != 0:
                        prev_end = args[i-1][1].args[1]
                    slope_value = -S.One/E*integrate(self.bending_moment()/args[i][0], (x, prev_end, x))
                    recent_segment_slope = prev_slope + slope_value
                    deflection_value = integrate(recent_segment_slope, (x, prev_end, x))
                    if i != len(args) - 1:
                        deflection += (prev_def + deflection_value)*SingularityFunction(x, prev_end, 0) \
                            - (prev_def + deflection_value)*SingularityFunction(x, args[i][1].args[1], 0)
                    else:
                        deflection += (prev_def + deflection_value)*SingularityFunction(x, prev_end, 0)
                    prev_slope = slope_value.subs(x, args[i][1].args[1])
                    prev_def = deflection_value.subs(x, args[i][1].args[1])
                return deflection
            base_char = self._base_char
            C3, C4 = symbols(base_char + '3:5')    # Integration constants
            slope_curve = -integrate(self.bending_moment(), x) + C3
            deflection_curve = integrate(slope_curve, x) + C4
            bc_eqs = []
            for position, value in self._boundary_conditions['deflection']:
                eqs = deflection_curve.subs(x, position) - value
                bc_eqs.append(eqs)
            constants = list(linsolve(bc_eqs, (C3, C4)))
            deflection_curve = deflection_curve.subs({C3: constants[0][0], C4: constants[0][1]})
            return S.One/(E*I)*deflection_curve

        if isinstance(I, Piecewise) and self._joined_beam:
            args = I.args
            prev_slope = 0
            prev_def = 0
            prev_end = 0
            deflection = 0
            for i in range(len(args)):
                if i != 0:
                    prev_end = args[i-1][1].args[1]
                slope_value = S.One/E*integrate(self.bending_moment()/args[i][0], (x, prev_end, x))
                recent_segment_slope = prev_slope + slope_value
                deflection_value = integrate(recent_segment_slope, (x, prev_end, x))
                if i != len(args) - 1:
                    deflection += (prev_def + deflection_value)*SingularityFunction(x, prev_end, 0) \
                        - (prev_def + deflection_value)*SingularityFunction(x, args[i][1].args[1], 0)
                else:
                    deflection += (prev_def + deflection_value)*SingularityFunction(x, prev_end, 0)
                prev_slope = slope_value.subs(x, args[i][1].args[1])
                prev_def = deflection_value.subs(x, args[i][1].args[1])
            return deflection

        C4 = Symbol('C4')
        deflection_curve = integrate(self.slope(), x) + C4

        bc_eqs = []
        for position, value in self._boundary_conditions['deflection']:
            eqs = deflection_curve.subs(x, position) - value
            bc_eqs.append(eqs)

        constants = list(linsolve(bc_eqs, C4))
        deflection_curve = deflection_curve.subs({C4: constants[0][0]})
        return deflection_curve

    def max_deflection(self):
        """
        Returns point of max deflection and its corresponding deflection value
        in a Beam object.
        """

        # To restrict the range within length of the Beam
        slope_curve = Piecewise((float("nan"), self.variable<=0),
                (self.slope(), self.variable<self.length),
                (float("nan"), True))

        points = solve(slope_curve.rewrite(Piecewise), self.variable,
                        domain=S.Reals)
        deflection_curve = self.deflection()
        deflections = [deflection_curve.subs(self.variable, x) for x in points]
        deflections = list(map(abs, deflections))
        if len(deflections) != 0:
            max_def = max(deflections)
            return (points[deflections.index(max_def)], max_def)
        else:
            return None

    def shear_stress(self):
        """
        Returns an expression representing the Shear Stress
        curve of the Beam object.
        """
        return self.shear_force()/self._area

    def plot_shear_stress(self, subs=None):
        """

        Returns a plot of shear stress present in the beam object.

        Parameters
        ==========
        subs : dictionary
            Python dictionary containing Symbols as key and their
            corresponding values.

        Examples
        ========
        There is a beam of length 8 meters and area of cross section 2 square
        meters. A constant distributed load of 10 KN/m is applied from half of
        the beam till the end. There are two simple supports below the beam,
        one at the starting point and another at the ending point of the beam.
        A pointload of magnitude 5 KN is also applied from top of the
        beam, at a distance of 4 meters from the starting point.
        Take E = 200 GPa and I = 400*(10**-6) meter**4.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> from sympy import symbols
            >>> R1, R2 = symbols('R1, R2')
            >>> b = Beam(8, 200*(10**9), 400*(10**-6), 2)
            >>> b.apply_load(5000, 2, -1)
            >>> b.apply_load(R1, 0, -1)
            >>> b.apply_load(R2, 8, -1)
            >>> b.apply_load(10000, 4, 0, end=8)
            >>> b.bc_deflection = [(0, 0), (8, 0)]
            >>> b.solve_for_reaction_loads(R1, R2)
            >>> b.plot_shear_stress()
            Plot object containing:
            [0]: cartesian line: 6875*SingularityFunction(x, 0, 0) - 2500*SingularityFunction(x, 2, 0)
            - 5000*SingularityFunction(x, 4, 1) + 15625*SingularityFunction(x, 8, 0)
            + 5000*SingularityFunction(x, 8, 1) for x over (0.0, 8.0)
        """

        shear_stress = self.shear_stress()
        x = self.variable
        length = self.length

        if subs is None:
            subs = {}
        for sym in shear_stress.atoms(Symbol):
            if sym != x and sym not in subs:
                raise ValueError('value of %s was not passed.' %sym)

        if length in subs:
            length = subs[length]

        # Returns Plot of Shear Stress
        return plot (shear_stress.subs(subs), (x, 0, length),
        title='Shear Stress', xlabel=r'$\mathrm{x}$', ylabel=r'$\tau$',
        line_color='r')


    def plot_shear_force(self, subs=None):
        """

        Returns a plot for Shear force present in the Beam object.

        Parameters
        ==========
        subs : dictionary
            Python dictionary containing Symbols as key and their
            corresponding values.

        Examples
        ========
        There is a beam of length 8 meters. A constant distributed load of 10 KN/m
        is applied from half of the beam till the end. There are two simple supports
        below the beam, one at the starting point and another at the ending point
        of the beam. A pointload of magnitude 5 KN is also applied from top of the
        beam, at a distance of 4 meters from the starting point.
        Take E = 200 GPa and I = 400*(10**-6) meter**4.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> from sympy import symbols
            >>> R1, R2 = symbols('R1, R2')
            >>> b = Beam(8, 200*(10**9), 400*(10**-6))
            >>> b.apply_load(5000, 2, -1)
            >>> b.apply_load(R1, 0, -1)
            >>> b.apply_load(R2, 8, -1)
            >>> b.apply_load(10000, 4, 0, end=8)
            >>> b.bc_deflection = [(0, 0), (8, 0)]
            >>> b.solve_for_reaction_loads(R1, R2)
            >>> b.plot_shear_force()
            Plot object containing:
            [0]: cartesian line: 13750*SingularityFunction(x, 0, 0) - 5000*SingularityFunction(x, 2, 0)
            - 10000*SingularityFunction(x, 4, 1) + 31250*SingularityFunction(x, 8, 0)
            + 10000*SingularityFunction(x, 8, 1) for x over (0.0, 8.0)
        """
        shear_force = self.shear_force()
        if subs is None:
            subs = {}
        for sym in shear_force.atoms(Symbol):
            if sym == self.variable:
                continue
            if sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length
        return plot(shear_force.subs(subs), (self.variable, 0, length), title='Shear Force',
                xlabel=r'$\mathrm{x}$', ylabel=r'$\mathrm{V}$', line_color='g')

    def plot_bending_moment(self, subs=None):
        """

        Returns a plot for Bending moment present in the Beam object.

        Parameters
        ==========
        subs : dictionary
            Python dictionary containing Symbols as key and their
            corresponding values.

        Examples
        ========
        There is a beam of length 8 meters. A constant distributed load of 10 KN/m
        is applied from half of the beam till the end. There are two simple supports
        below the beam, one at the starting point and another at the ending point
        of the beam. A pointload of magnitude 5 KN is also applied from top of the
        beam, at a distance of 4 meters from the starting point.
        Take E = 200 GPa and I = 400*(10**-6) meter**4.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> from sympy import symbols
            >>> R1, R2 = symbols('R1, R2')
            >>> b = Beam(8, 200*(10**9), 400*(10**-6))
            >>> b.apply_load(5000, 2, -1)
            >>> b.apply_load(R1, 0, -1)
            >>> b.apply_load(R2, 8, -1)
            >>> b.apply_load(10000, 4, 0, end=8)
            >>> b.bc_deflection = [(0, 0), (8, 0)]
            >>> b.solve_for_reaction_loads(R1, R2)
            >>> b.plot_bending_moment()
            Plot object containing:
            [0]: cartesian line: 13750*SingularityFunction(x, 0, 1) - 5000*SingularityFunction(x, 2, 1)
            - 5000*SingularityFunction(x, 4, 2) + 31250*SingularityFunction(x, 8, 1)
            + 5000*SingularityFunction(x, 8, 2) for x over (0.0, 8.0)
        """
        bending_moment = self.bending_moment()
        if subs is None:
            subs = {}
        for sym in bending_moment.atoms(Symbol):
            if sym == self.variable:
                continue
            if sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length
        return plot(bending_moment.subs(subs), (self.variable, 0, length), title='Bending Moment',
                xlabel=r'$\mathrm{x}$', ylabel=r'$\mathrm{M}$', line_color='b')

    def plot_slope(self, subs=None):
        """

        Returns a plot for slope of deflection curve of the Beam object.

        Parameters
        ==========
        subs : dictionary
            Python dictionary containing Symbols as key and their
            corresponding values.

        Examples
        ========
        There is a beam of length 8 meters. A constant distributed load of 10 KN/m
        is applied from half of the beam till the end. There are two simple supports
        below the beam, one at the starting point and another at the ending point
        of the beam. A pointload of magnitude 5 KN is also applied from top of the
        beam, at a distance of 4 meters from the starting point.
        Take E = 200 GPa and I = 400*(10**-6) meter**4.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> from sympy import symbols
            >>> R1, R2 = symbols('R1, R2')
            >>> b = Beam(8, 200*(10**9), 400*(10**-6))
            >>> b.apply_load(5000, 2, -1)
            >>> b.apply_load(R1, 0, -1)
            >>> b.apply_load(R2, 8, -1)
            >>> b.apply_load(10000, 4, 0, end=8)
            >>> b.bc_deflection = [(0, 0), (8, 0)]
            >>> b.solve_for_reaction_loads(R1, R2)
            >>> b.plot_slope()
            Plot object containing:
            [0]: cartesian line: -8.59375e-5*SingularityFunction(x, 0, 2) + 3.125e-5*SingularityFunction(x, 2, 2)
            + 2.08333333333333e-5*SingularityFunction(x, 4, 3) - 0.0001953125*SingularityFunction(x, 8, 2)
            - 2.08333333333333e-5*SingularityFunction(x, 8, 3) + 0.00138541666666667 for x over (0.0, 8.0)
        """
        slope = self.slope()
        if subs is None:
            subs = {}
        for sym in slope.atoms(Symbol):
            if sym == self.variable:
                continue
            if sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length
        return plot(slope.subs(subs), (self.variable, 0, length), title='Slope',
                xlabel=r'$\mathrm{x}$', ylabel=r'$\theta$', line_color='m')

    def plot_deflection(self, subs=None):
        """

        Returns a plot for deflection curve of the Beam object.

        Parameters
        ==========
        subs : dictionary
            Python dictionary containing Symbols as key and their
            corresponding values.

        Examples
        ========
        There is a beam of length 8 meters. A constant distributed load of 10 KN/m
        is applied from half of the beam till the end. There are two simple supports
        below the beam, one at the starting point and another at the ending point
        of the beam. A pointload of magnitude 5 KN is also applied from top of the
        beam, at a distance of 4 meters from the starting point.
        Take E = 200 GPa and I = 400*(10**-6) meter**4.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> from sympy import symbols
            >>> R1, R2 = symbols('R1, R2')
            >>> b = Beam(8, 200*(10**9), 400*(10**-6))
            >>> b.apply_load(5000, 2, -1)
            >>> b.apply_load(R1, 0, -1)
            >>> b.apply_load(R2, 8, -1)
            >>> b.apply_load(10000, 4, 0, end=8)
            >>> b.bc_deflection = [(0, 0), (8, 0)]
            >>> b.solve_for_reaction_loads(R1, R2)
            >>> b.plot_deflection()
            Plot object containing:
            [0]: cartesian line: 0.00138541666666667*x - 2.86458333333333e-5*SingularityFunction(x, 0, 3)
            + 1.04166666666667e-5*SingularityFunction(x, 2, 3) + 5.20833333333333e-6*SingularityFunction(x, 4, 4)
            - 6.51041666666667e-5*SingularityFunction(x, 8, 3) - 5.20833333333333e-6*SingularityFunction(x, 8, 4)
            for x over (0.0, 8.0)
        """
        deflection = self.deflection()
        if subs is None:
            subs = {}
        for sym in deflection.atoms(Symbol):
            if sym == self.variable:
                continue
            if sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length
        return plot(deflection.subs(subs), (self.variable, 0, length),
                    title='Deflection', xlabel=r'$\mathrm{x}$', ylabel=r'$\delta$',
                    line_color='r')


    def plot_loading_results(self, subs=None):
        """
        Returns a subplot of Shear Force, Bending Moment,
        Slope and Deflection of the Beam object.

        Parameters
        ==========

        subs : dictionary
               Python dictionary containing Symbols as key and their
               corresponding values.

        Examples
        ========

        There is a beam of length 8 meters. A constant distributed load of 10 KN/m
        is applied from half of the beam till the end. There are two simple supports
        below the beam, one at the starting point and another at the ending point
        of the beam. A pointload of magnitude 5 KN is also applied from top of the
        beam, at a distance of 4 meters from the starting point.
        Take E = 200 GPa and I = 400*(10**-6) meter**4.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> from sympy import symbols
            >>> R1, R2 = symbols('R1, R2')
            >>> b = Beam(8, 200*(10**9), 400*(10**-6))
            >>> b.apply_load(5000, 2, -1)
            >>> b.apply_load(R1, 0, -1)
            >>> b.apply_load(R2, 8, -1)
            >>> b.apply_load(10000, 4, 0, end=8)
            >>> b.bc_deflection = [(0, 0), (8, 0)]
            >>> b.solve_for_reaction_loads(R1, R2)
            >>> axes = b.plot_loading_results()
        """
        length = self.length
        variable = self.variable
        if subs is None:
            subs = {}
        for sym in self.deflection().atoms(Symbol):
            if sym == self.variable:
                continue
            if sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)
        if length in subs:
            length = subs[length]
        ax1 = plot(self.shear_force().subs(subs), (variable, 0, length),
                   title="Shear Force", xlabel=r'$\mathrm{x}$', ylabel=r'$\mathrm{V}$',
                   line_color='g', show=False)
        ax2 = plot(self.bending_moment().subs(subs), (variable, 0, length),
                   title="Bending Moment", xlabel=r'$\mathrm{x}$', ylabel=r'$\mathrm{M}$',
                   line_color='b', show=False)
        ax3 = plot(self.slope().subs(subs), (variable, 0, length),
                   title="Slope", xlabel=r'$\mathrm{x}$', ylabel=r'$\theta$',
                   line_color='m', show=False)
        ax4 = plot(self.deflection().subs(subs), (variable, 0, length),
                   title="Deflection", xlabel=r'$\mathrm{x}$', ylabel=r'$\delta$',
                   line_color='r', show=False)

        return PlotGrid(4, 1, ax1, ax2, ax3, ax4)

    def _solve_for_ild_equations(self, value):
        """

        Helper function for I.L.D. It takes the unsubstituted
        copy of the load equation and uses it to calculate shear force and bending
        moment equations.
        """
        x = self.variable
        a = self.ild_variable
        load = self._load + value * SingularityFunction(x, a, -1)
        shear_force = -integrate(load, x)
        bending_moment = integrate(shear_force, x)

        return shear_force, bending_moment

    def solve_for_ild_reactions(self, value, *reactions):
        """

        Determines the Influence Line Diagram equations for reaction
        forces under the effect of a moving load.

        Parameters
        ==========
        value : Integer
            Magnitude of moving load
        reactions :
            The reaction forces applied on the beam.

        Warning
        =======
        This method creates equations that can give incorrect results when
        substituting a = 0 or a = l, with l the length of the beam.

        Examples
        ========

        There is a beam of length 10 meters. There are two simple supports
        below the beam, one at the starting point and another at the ending
        point of the beam. Calculate the I.L.D. equations for reaction forces
        under the effect of a moving load of magnitude 1kN.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy import symbols
            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> E, I = symbols('E, I')
            >>> R_0, R_10 = symbols('R_0, R_10')
            >>> b = Beam(10, E, I)
            >>> p0 = b.apply_support(0, 'pin')
            >>> p10 = b.apply_support(10, 'roller')
            >>> b.solve_for_ild_reactions(1,R_0,R_10)
            >>> b.ild_reactions
            {R_0: -SingularityFunction(a, 0, 0) + SingularityFunction(a, 0, 1)/10 - SingularityFunction(a, 10, 1)/10,
            R_10: -SingularityFunction(a, 0, 1)/10 + SingularityFunction(a, 10, 0) + SingularityFunction(a, 10, 1)/10}

        """
        shear_force, bending_moment = self._solve_for_ild_equations(value)
        x = self.variable
        l = self.length
        a = self.ild_variable

        rotation_jumps = tuple(self._rotation_hinge_symbols)
        deflection_jumps = tuple(self._sliding_hinge_symbols)

        C3 = Symbol('C3')
        C4 = Symbol('C4')

        shear_curve = limit(shear_force, x, l) - value*(SingularityFunction(a, 0, 0) - SingularityFunction(a, l, 0))
        moment_curve = (limit(bending_moment, x, l) - value * (l * SingularityFunction(a, 0, 0)
                                                               - SingularityFunction(a, 0, 1)
                                                               + SingularityFunction(a, l, 1)))

        shear_force_eqs = []
        bending_moment_eqs = []
        slope_eqs = []
        deflection_eqs = []

        for position, val in self._boundary_conditions['shear_force']:
            eqs = self.shear_force().subs(x, position) - val
            eqs_without_inf = sum(arg for arg in eqs.args if not any(num.is_infinite for num in arg.args))
            shear_sinc = value * (SingularityFunction(- a, - position, 0) - SingularityFunction(-a, 0, 0))
            eqs_with_shear_sinc = eqs_without_inf - shear_sinc
            shear_force_eqs.append(eqs_with_shear_sinc)

        for position, val in self._boundary_conditions['bending_moment']:
            eqs = self.bending_moment().subs(x, position) - val
            eqs_without_inf = sum(arg for arg in eqs.args if not any(num.is_infinite for num in arg.args))
            moment_sinc = value * (position * SingularityFunction(a, 0, 0)
                                   - SingularityFunction(a, 0, 1) + SingularityFunction(a, position, 1))
            eqs_with_moment_sinc = eqs_without_inf - moment_sinc
            bending_moment_eqs.append(eqs_with_moment_sinc)

        slope_curve = integrate(bending_moment, x) + C3
        for position, val in self._boundary_conditions['slope']:
            eqs = slope_curve.subs(x, position) - val + value * (SingularityFunction(-a, 0, 1) + position * SingularityFunction(-a, 0, 0))**2 / 2
            slope_eqs.append(eqs)

        deflection_curve = integrate(slope_curve, x) + C4
        for position, val in self._boundary_conditions['deflection']:
            eqs = deflection_curve.subs(x, position) - val + value * (SingularityFunction(-a, 0, 1) + position * SingularityFunction(-a, 0, 0)) ** 3 / 6
            deflection_eqs.append(eqs)

        solution = list((linsolve([shear_curve, moment_curve] + shear_force_eqs + bending_moment_eqs + slope_eqs
                                  + deflection_eqs, (C3, C4) + reactions + rotation_jumps + deflection_jumps).args)[0])

        reaction_index = 2 + len(reactions)
        rotation_index = reaction_index + len(rotation_jumps)
        reaction_solution = solution[2:reaction_index]
        rotation_solution = solution[reaction_index:rotation_index]
        deflection_solution = solution[rotation_index:]

        self._ild_reactions = dict(zip(reactions, reaction_solution))
        self._ild_rotations_jumps = dict(zip(rotation_jumps, rotation_solution))
        self._ild_deflection_jumps = dict(zip(deflection_jumps, deflection_solution))

    def plot_ild_reactions(self, subs=None):
        """

        Plots the Influence Line Diagram of Reaction Forces
        under the effect of a moving load. This function
        should be called after calling solve_for_ild_reactions().

        Parameters
        ==========

        subs : dictionary
               Python dictionary containing Symbols as key and their
               corresponding values.

        Warning
        =======
        The values for a = 0 and a = l, with l the length of the beam, in
        the plot can be incorrect.

        Examples
        ========

        There is a beam of length 10 meters. A point load of magnitude 5KN
        is also applied from top of the beam, at a distance of 4 meters
        from the starting point. There are two simple supports below the
        beam, located at the starting point and at a distance of 7 meters
        from the starting point. Plot the I.L.D. equations for reactions
        at both support points under the effect of a moving load
        of magnitude 1kN.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy import symbols
            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> E, I = symbols('E, I')
            >>> R_0, R_7 = symbols('R_0, R_7')
            >>> b = Beam(10, E, I)
            >>> p0 = b.apply_support(0, 'roller')
            >>> p7 = b.apply_support(7, 'roller')
            >>> b.apply_load(5,4,-1)
            >>> b.solve_for_ild_reactions(1,R_0,R_7)
            >>> b.ild_reactions
            {R_0: -SingularityFunction(a, 0, 0) + SingularityFunction(a, 0, 1)/7
            - 3*SingularityFunction(a, 10, 0)/7  - SingularityFunction(a, 10, 1)/7 - 15/7,
            R_7: -SingularityFunction(a, 0, 1)/7 + 10*SingularityFunction(a, 10, 0)/7 + SingularityFunction(a, 10, 1)/7 - 20/7}
            >>> b.plot_ild_reactions()
            PlotGrid object containing:
            Plot[0]:Plot object containing:
            [0]: cartesian line: -SingularityFunction(a, 0, 0) + SingularityFunction(a, 0, 1)/7
            - 3*SingularityFunction(a, 10, 0)/7 - SingularityFunction(a, 10, 1)/7 - 15/7 for a over (0.0, 10.0)
            Plot[1]:Plot object containing:
            [0]: cartesian line: -SingularityFunction(a, 0, 1)/7 + 10*SingularityFunction(a, 10, 0)/7
            + SingularityFunction(a, 10, 1)/7 - 20/7 for a over (0.0, 10.0)

        """
        if not self._ild_reactions:
            raise ValueError("I.L.D. reaction equations not found. Please use solve_for_ild_reactions() to generate the I.L.D. reaction equations.")

        a = self.ild_variable
        ildplots = []

        if subs is None:
            subs = {}

        for reaction in self._ild_reactions:
            for sym in self._ild_reactions[reaction].atoms(Symbol):
                if sym != a and sym not in subs:
                    raise ValueError('Value of %s was not passed.' %sym)

        for sym in self._length.atoms(Symbol):
            if sym != a and sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)

        for reaction in self._ild_reactions:
            ildplots.append(plot(self._ild_reactions[reaction].subs(subs),
            (a, 0, self._length.subs(subs)), title='I.L.D. for Reactions',
            xlabel=a, ylabel=reaction, line_color='blue', show=False))

        return PlotGrid(len(ildplots), 1, *ildplots)

    def solve_for_ild_shear(self, distance, value, *reactions):
        """

        Determines the Influence Line Diagram equations for shear at a
        specified point under the effect of a moving load.

        Parameters
        ==========
        distance : Integer
            Distance of the point from the start of the beam
            for which equations are to be determined
        value : Integer
            Magnitude of moving load
        reactions :
            The reaction forces applied on the beam.

        Warning
        =======
        This method creates equations that can give incorrect results when
        substituting a = 0 or a = l, with l the length of the beam.

        Examples
        ========

        There is a beam of length 12 meters. There are two simple supports
        below the beam, one at the starting point and another at a distance
        of 8 meters. Calculate the I.L.D. equations for Shear at a distance
        of 4 meters under the effect of a moving load of magnitude 1kN.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy import symbols
            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> E, I = symbols('E, I')
            >>> R_0, R_8 = symbols('R_0, R_8')
            >>> b = Beam(12, E, I)
            >>> p0 = b.apply_support(0, 'roller')
            >>> p8 = b.apply_support(8, 'roller')
            >>> b.solve_for_ild_reactions(1, R_0, R_8)
            >>> b.solve_for_ild_shear(4, 1, R_0, R_8)
            >>> b.ild_shear
            -(-SingularityFunction(a, 0, 0) + SingularityFunction(a, 12, 0) + 2)*SingularityFunction(a, 4, 0)
            - SingularityFunction(-a, 0, 0) - SingularityFunction(a, 0, 0) + SingularityFunction(a, 0, 1)/8
            + SingularityFunction(a, 12, 0)/2 - SingularityFunction(a, 12, 1)/8 + 1

        """

        x = self.variable
        l = self.length
        a = self.ild_variable

        shear_force, _ = self._solve_for_ild_equations(value)

        shear_curve1 = value - limit(shear_force, x, distance)
        shear_curve2 = (limit(shear_force, x, l) - limit(shear_force, x, distance)) - value

        for reaction in reactions:
            shear_curve1 = shear_curve1.subs(reaction,self._ild_reactions[reaction])
            shear_curve2 = shear_curve2.subs(reaction,self._ild_reactions[reaction])

        shear_eq = (shear_curve1 - (shear_curve1 - shear_curve2) * SingularityFunction(a, distance, 0)
                    - value * SingularityFunction(-a, 0, 0) + value * SingularityFunction(a, l, 0))

        self._ild_shear = shear_eq

    def plot_ild_shear(self,subs=None):
        """

        Plots the Influence Line Diagram for Shear under the effect
        of a moving load. This function should be called after
        calling solve_for_ild_shear().

        Parameters
        ==========

        subs : dictionary
               Python dictionary containing Symbols as key and their
               corresponding values.

        Warning
        =======
        The values for a = 0 and a = l, with l the length of the beam, in
        the plot can be incorrect.

        Examples
        ========

        There is a beam of length 12 meters. There are two simple supports
        below the beam, one at the starting point and another at a distance
        of 8 meters. Plot the I.L.D. for Shear at a distance
        of 4 meters under the effect of a moving load of magnitude 1kN.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy import symbols
            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> E, I = symbols('E, I')
            >>> R_0, R_8 = symbols('R_0, R_8')
            >>> b = Beam(12, E, I)
            >>> p0 = b.apply_support(0, 'roller')
            >>> p8 = b.apply_support(8, 'roller')
            >>> b.solve_for_ild_reactions(1, R_0, R_8)
            >>> b.solve_for_ild_shear(4, 1, R_0, R_8)
            >>> b.ild_shear
            -(-SingularityFunction(a, 0, 0) + SingularityFunction(a, 12, 0) + 2)*SingularityFunction(a, 4, 0)
            - SingularityFunction(-a, 0, 0) - SingularityFunction(a, 0, 0) + SingularityFunction(a, 0, 1)/8
            + SingularityFunction(a, 12, 0)/2 - SingularityFunction(a, 12, 1)/8 + 1
            >>> b.plot_ild_shear()
            Plot object containing:
            [0]: cartesian line: -(-SingularityFunction(a, 0, 0) + SingularityFunction(a, 12, 0) + 2)*SingularityFunction(a, 4, 0)
            - SingularityFunction(-a, 0, 0) - SingularityFunction(a, 0, 0) + SingularityFunction(a, 0, 1)/8
            + SingularityFunction(a, 12, 0)/2 - SingularityFunction(a, 12, 1)/8 + 1 for a over (0.0, 12.0)

        """

        if not self._ild_shear:
            raise ValueError("I.L.D. shear equation not found. Please use solve_for_ild_shear() to generate the I.L.D. shear equations.")

        l = self._length
        a = self.ild_variable

        if subs is None:
            subs = {}

        for sym in self._ild_shear.atoms(Symbol):
            if sym != a and sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)

        for sym in self._length.atoms(Symbol):
            if sym != a and sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)

        return plot(self._ild_shear.subs(subs), (a, 0, l),  title='I.L.D. for Shear',
               xlabel=r'$\mathrm{a}$', ylabel=r'$\mathrm{V}$', line_color='blue',show=True)

    def solve_for_ild_moment(self, distance, value, *reactions):
        """

        Determines the Influence Line Diagram equations for moment at a
        specified point under the effect of a moving load.

        Parameters
        ==========
        distance : Integer
            Distance of the point from the start of the beam
            for which equations are to be determined
        value : Integer
            Magnitude of moving load
        reactions :
            The reaction forces applied on the beam.

        Warning
        =======
        This method creates equations that can give incorrect results when
        substituting a = 0 or a = l, with l the length of the beam.

        Examples
        ========

        There is a beam of length 12 meters. There are two simple supports
        below the beam, one at the starting point and another at a distance
        of 8 meters. Calculate the I.L.D. equations for Moment at a distance
        of 4 meters under the effect of a moving load of magnitude 1kN.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy import symbols
            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> E, I = symbols('E, I')
            >>> R_0, R_8 = symbols('R_0, R_8')
            >>> b = Beam(12, E, I)
            >>> p0 = b.apply_support(0, 'roller')
            >>> p8 = b.apply_support(8, 'roller')
            >>> b.solve_for_ild_reactions(1, R_0, R_8)
            >>> b.solve_for_ild_moment(4, 1, R_0, R_8)
            >>> b.ild_moment
            -(4*SingularityFunction(a, 0, 0) - SingularityFunction(a, 0, 1) + SingularityFunction(a, 4, 1))*SingularityFunction(a, 4, 0)
            - SingularityFunction(a, 0, 1)/2 + SingularityFunction(a, 4, 1) - 2*SingularityFunction(a, 12, 0)
            - SingularityFunction(a, 12, 1)/2

        """

        x = self.variable
        l = self.length
        a = self.ild_variable

        _, moment = self._solve_for_ild_equations(value)

        moment_curve1 = value*(distance * SingularityFunction(a, 0, 0) - SingularityFunction(a, 0, 1)
                               + SingularityFunction(a, distance, 1)) - limit(moment, x, distance)
        moment_curve2 = (limit(moment, x, l)-limit(moment, x, distance)
                         - value * (l * SingularityFunction(a, 0, 0) - SingularityFunction(a, 0, 1)
                                    + SingularityFunction(a, l, 1)))

        for reaction in reactions:
            moment_curve1 = moment_curve1.subs(reaction, self._ild_reactions[reaction])
            moment_curve2 = moment_curve2.subs(reaction, self._ild_reactions[reaction])

        moment_eq = moment_curve1 - (moment_curve1 - moment_curve2) * SingularityFunction(a, distance, 0)

        self._ild_moment = moment_eq

    def plot_ild_moment(self,subs=None):
        """

        Plots the Influence Line Diagram for Moment under the effect
        of a moving load. This function should be called after
        calling solve_for_ild_moment().

        Parameters
        ==========

        subs : dictionary
               Python dictionary containing Symbols as key and their
               corresponding values.

        Warning
        =======
        The values for a = 0 and a = l, with l the length of the beam, in
        the plot can be incorrect.

        Examples
        ========

        There is a beam of length 12 meters. There are two simple supports
        below the beam, one at the starting point and another at a distance
        of 8 meters. Plot the I.L.D. for Moment at a distance
        of 4 meters under the effect of a moving load of magnitude 1kN.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy import symbols
            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> E, I = symbols('E, I')
            >>> R_0, R_8 = symbols('R_0, R_8')
            >>> b = Beam(12, E, I)
            >>> p0 = b.apply_support(0, 'roller')
            >>> p8 = b.apply_support(8, 'roller')
            >>> b.solve_for_ild_reactions(1, R_0, R_8)
            >>> b.solve_for_ild_moment(4, 1, R_0, R_8)
            >>> b.ild_moment
            -(4*SingularityFunction(a, 0, 0) - SingularityFunction(a, 0, 1) + SingularityFunction(a, 4, 1))*SingularityFunction(a, 4, 0)
            - SingularityFunction(a, 0, 1)/2 + SingularityFunction(a, 4, 1) - 2*SingularityFunction(a, 12, 0)
            - SingularityFunction(a, 12, 1)/2
            >>> b.plot_ild_moment()
            Plot object containing:
            [0]: cartesian line: -(4*SingularityFunction(a, 0, 0) - SingularityFunction(a, 0, 1)
            + SingularityFunction(a, 4, 1))*SingularityFunction(a, 4, 0) - SingularityFunction(a, 0, 1)/2
            + SingularityFunction(a, 4, 1) - 2*SingularityFunction(a, 12, 0) - SingularityFunction(a, 12, 1)/2 for a over (0.0, 12.0)

        """

        if not self._ild_moment:
            raise ValueError("I.L.D. moment equation not found. Please use solve_for_ild_moment() to generate the I.L.D. moment equations.")

        a = self.ild_variable

        if subs is None:
            subs = {}

        for sym in self._ild_moment.atoms(Symbol):
            if sym != a and sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)

        for sym in self._length.atoms(Symbol):
            if sym != a and sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)
        return plot(self._ild_moment.subs(subs), (a, 0, self._length), title='I.L.D. for Moment',
               xlabel=r'$\mathrm{a}$', ylabel=r'$\mathrm{M}$', line_color='blue', show=True)

    @doctest_depends_on(modules=('numpy',))
    def draw(self, pictorial=True):
        """
        Returns a plot object representing the beam diagram of the beam.
        In particular, the diagram might include:

        * the beam.
        * vertical black arrows represent point loads and support reaction
          forces (the latter if they have been added with the ``apply_load``
          method).
        * circular arrows represent moments.
        * shaded areas represent distributed loads.
        * the support, if ``apply_support`` has been executed.
        * if a composite beam has been created with the ``join`` method and
          a hinge has been specified, it will be shown with a white disc.

        The diagram shows positive loads on the upper side of the beam,
        and negative loads on the lower side. If two or more distributed
        loads acts along the same direction over the same region, the
        function will add them up together.

        .. note::
            The user must be careful while entering load values.
            The draw function assumes a sign convention which is used
            for plotting loads.
            Given a right handed coordinate system with XYZ coordinates,
            the beam's length is assumed to be along the positive X axis.
            The draw function recognizes positive loads(with n>-2) as loads
            acting along negative Y direction and positive moments acting
            along positive Z direction.

        Parameters
        ==========

        pictorial: Boolean (default=True)
            Setting ``pictorial=True`` would simply create a pictorial (scaled)
            view of the beam diagram. On the other hand, ``pictorial=False``
            would create a beam diagram with the exact dimensions on the plot.

        Examples
        ========

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> from sympy import symbols
            >>> P1, P2, M = symbols('P1, P2, M')
            >>> E, I = symbols('E, I')
            >>> b = Beam(50, 20, 30)
            >>> b.apply_load(-10, 2, -1)
            >>> b.apply_load(15, 26, -1)
            >>> b.apply_load(P1, 10, -1)
            >>> b.apply_load(-P2, 40, -1)
            >>> b.apply_load(90, 5, 0, 23)
            >>> b.apply_load(10, 30, 1, 50)
            >>> b.apply_load(M, 15, -2)
            >>> b.apply_load(-M, 30, -2)
            >>> p50 = b.apply_support(50, "pin")
            >>> p0, m0 = b.apply_support(0, "fixed")
            >>> p20 = b.apply_support(20, "roller")
            >>> p = b.draw()  # doctest: +SKIP
            >>> p  # doctest: +ELLIPSIS,+SKIP
            Plot object containing:
            [0]: cartesian line: 25*SingularityFunction(x, 5, 0) - 25*SingularityFunction(x, 23, 0)
            + SingularityFunction(x, 30, 1) - 20*SingularityFunction(x, 50, 0)
            - SingularityFunction(x, 50, 1) + 5 for x over (0.0, 50.0)
            [1]: cartesian line: 5 for x over (0.0, 50.0)
            ...
            >>> p.show() # doctest: +SKIP

        """
        if not numpy:
            raise ImportError("To use this function numpy module is required")

        loads = list(set(self.applied_loads) - set(self._support_as_loads))
        if (not pictorial) and any((len(l[0].free_symbols) > 0) and (l[2] >= 0) for l in loads):
            raise ValueError("`pictorial=False` requires numerical "
                "distributed loads. Instead, symbolic loads were found. "
                "Cannot continue.")

        x = self.variable

        # checking whether length is an expression in terms of any Symbol.
        if isinstance(self.length, Expr):
            l = list(self.length.atoms(Symbol))
            # assigning every Symbol a default value of 10
            l = dict.fromkeys(l, 10)
            length = self.length.subs(l)
        else:
            l = {}
            length = self.length
        height = length/10

        rectangles = []
        rectangles.append({'xy':(0, 0), 'width':length, 'height': height, 'facecolor':"brown"})
        annotations, markers, load_eq,load_eq1, fill = self._draw_load(pictorial, length, l)
        support_markers, support_rectangles = self._draw_supports(length, l)

        rectangles += support_rectangles
        markers += support_markers

        for loc in self._applied_rotation_hinges:
            ratio = loc / self.length
            x_pos = float(ratio) * length
            markers += [{'args':[[x_pos], [height / 2]], 'marker':'o', 'markersize':6, 'color':"white"}]

        for loc in self._applied_sliding_hinges:
            ratio = loc / self.length
            x_pos = float(ratio) * length
            markers += [{'args': [[x_pos], [height / 2]], 'marker':'|', 'markersize':12, 'color':"white"}]

        ylim = (-length, 1.25*length)
        if fill:
            # when distributed loads are presents, they might get clipped out
            # in the figure by the ylim settings.
            # It might be necessary to compute new limits.
            _min = min(min(fill["y2"]), min(r["xy"][1] for r in rectangles))
            _max = max(max(fill["y1"]), max(r["xy"][1] for r in rectangles))
            if (_min < ylim[0]) or (_max > ylim[1]):
                offset = abs(_max - _min) * 0.1
                ylim = (_min - offset, _max + offset)

        sing_plot = plot(height + load_eq, height + load_eq1, (x, 0, length),
            xlim=(-height, length + height), ylim=ylim,
            annotations=annotations, markers=markers, rectangles=rectangles,
            line_color='brown', fill=fill, axis=False, show=False)

        return sing_plot


    def _is_load_negative(self, load):
        """Try to determine if a load is negative or positive, using
        expansion and doit if necessary.

        Returns
        =======
        True: if the load is negative
        False: if the load is positive
        None: if it is indeterminate

        """
        rv = load.is_negative
        if load.is_Atom or rv is not None:
            return rv
        return load.doit().expand().is_negative

    def _draw_load(self, pictorial, length, l):
        loads = list(set(self.applied_loads) - set(self._support_as_loads))
        height = length/10
        x = self.variable

        annotations = []
        markers = []
        load_args = []
        scaled_load = 0
        load_args1 = []
        scaled_load1 = 0
        load_eq = S.Zero     # For positive valued higher order loads
        load_eq1 = S.Zero    # For negative valued higher order loads
        fill = None

        # schematic view should use the class convention as much as possible.
        # However, users can add expressions as symbolic loads, for example
        # P1 - P2: is this load positive or negative? We can't say.
        # On these occasions it is better to inform users about the
        # indeterminate state of those loads.
        warning_head = "Please, note that this schematic view might not be " \
            "in agreement with the sign convention used by the Beam class " \
            "for load-related computations, because it was not possible " \
            "to determine the sign (hence, the direction) of the " \
            "following loads:\n"
        warning_body = ""

        for load in loads:
            # check if the position of load is in terms of the beam length.
            if l:
                pos =  load[1].subs(l)
            else:
                pos = load[1]

            # point loads
            if load[2] == -1:
                iln = self._is_load_negative(load[0])
                if iln is None:
                    warning_body += "* Point load %s located at %s\n" % (load[0], load[1])
                if iln:
                    annotations.append({'text':'', 'xy':(pos, 0), 'xytext':(pos, height - 4*height), 'arrowprops':{'width': 1.5, 'headlength': 5, 'headwidth': 5, 'facecolor': 'black'}})
                else:
                    annotations.append({'text':'', 'xy':(pos, height),  'xytext':(pos, height*4), 'arrowprops':{"width": 1.5, "headlength": 4, "headwidth": 4, "facecolor": 'black'}})
            # moment loads
            elif load[2] == -2:
                iln = self._is_load_negative(load[0])
                if iln is None:
                    warning_body += "* Moment %s located at %s\n" % (load[0], load[1])
                if self._is_load_negative(load[0]):
                    markers.append({'args':[[pos], [height/2]], 'marker': r'$\circlearrowright$', 'markersize':15})
                else:
                    markers.append({'args':[[pos], [height/2]], 'marker': r'$\circlearrowleft$', 'markersize':15})
            # higher order loads
            elif load[2] >= 0:
                # `fill` will be assigned only when higher order loads are present
                value, start, order, end = load

                iln = self._is_load_negative(value)
                if iln is None:
                    warning_body += "* Distributed load %s from %s to %s\n" % (value, start, end)

                # Positive loads have their separate equations
                if not iln:
                    # if pictorial is True we remake the load equation again with
                    # some constant magnitude values.
                    if pictorial:
                        # remake the load equation again with some constant
                        # magnitude values.
                        value = 10**(1-order) if order > 0 else length/2
                    scaled_load += value*SingularityFunction(x, start, order)
                    if end:
                        f2 = value*x**order if order >= 0 else length/2*x**order
                        for i in range(0, order + 1):
                            scaled_load -= (f2.diff(x, i).subs(x, end - start)*
                                            SingularityFunction(x, end, i)/factorial(i))

                    if isinstance(scaled_load, Add):
                        load_args = scaled_load.args
                    else:
                        # when the load equation consists of only a single term
                        load_args = (scaled_load,)
                    load_eq = Add(*[i.subs(l) for i in load_args])

                # For loads with negative value
                else:
                    if pictorial:
                        # remake the load equation again with some constant
                        # magnitude values.
                        value = 10**(1-order) if order > 0 else length/2
                    scaled_load1 += abs(value)*SingularityFunction(x, start, order)
                    if end:
                        f2 = abs(value)*x**order if order >= 0 else length/2*x**order
                        for i in range(0, order + 1):
                            scaled_load1 -= (f2.diff(x, i).subs(x, end - start)*
                                            SingularityFunction(x, end, i)/factorial(i))

                    if isinstance(scaled_load1, Add):
                        load_args1 = scaled_load1.args
                    else:
                        # when the load equation consists of only a single term
                        load_args1 = (scaled_load1,)
                    load_eq1 = [i.subs(l) for i in load_args1]
                    load_eq1 = -Add(*load_eq1) - height

        if len(warning_body) > 0:
            warnings.warn(warning_head + warning_body)

        xx = numpy.arange(0, float(length), 0.001)
        yy1 = lambdify([x], height + load_eq.rewrite(Piecewise))(xx)
        yy2 = lambdify([x], height + load_eq1.rewrite(Piecewise))(xx)
        if not isinstance(yy1, numpy.ndarray):
            yy1 *= numpy.ones_like(xx)
        if not isinstance(yy2, numpy.ndarray):
            yy2 *= numpy.ones_like(xx)
        fill = {'x': xx, 'y1': yy1, 'y2': yy2,
            'color':'darkkhaki', "zorder": -1}
        return annotations, markers, load_eq, load_eq1, fill


    def _draw_supports(self, length, l):
        height = float(length/10)

        support_markers = []
        support_rectangles = []
        for support in self._applied_supports:
            if l:
                pos =  support[0].subs(l)
            else:
                pos = support[0]

            if support[1] == "pin":
                support_markers.append({'args':[pos, [0]], 'marker':6, 'markersize':13, 'color':"black"})

            elif support[1] == "roller":
                support_markers.append({'args':[pos, [-height/2.5]], 'marker':'o', 'markersize':11, 'color':"black"})

            elif support[1] == "fixed":
                if pos == 0:
                    support_rectangles.append({'xy':(0, -3*height), 'width':-length/20, 'height':6*height + height, 'fill':False, 'hatch':'/////'})
                else:
                    support_rectangles.append({'xy':(length, -3*height), 'width':length/20, 'height': 6*height + height, 'fill':False, 'hatch':'/////'})

        return support_markers, support_rectangles


class Beam3D(Beam):
    """
    This class handles loads applied in any direction of a 3D space along
    with unequal values of Second moment along different axes.

    .. note::
       A consistent sign convention must be used while solving a beam
       bending problem; the results will
       automatically follow the chosen sign convention.
       This class assumes that any kind of distributed load/moment is
       applied through out the span of a beam.

    Examples
    ========
    There is a beam of l meters long. A constant distributed load of magnitude q
    is applied along y-axis from start till the end of beam. A constant distributed
    moment of magnitude m is also applied along z-axis from start till the end of beam.
    Beam is fixed at both of its end. So, deflection of the beam at the both ends
    is restricted.

    >>> from sympy.physics.continuum_mechanics.beam import Beam3D
    >>> from sympy import symbols, simplify, collect, factor
    >>> l, E, G, I, A = symbols('l, E, G, I, A')
    >>> b = Beam3D(l, E, G, I, A)
    >>> x, q, m = symbols('x, q, m')
    >>> b.apply_load(q, 0, 0, dir="y")
    >>> b.apply_moment_load(m, 0, -1, dir="z")
    >>> b.shear_force()
    [0, -q*x, 0]
    >>> b.bending_moment()
    [0, 0, -m*x + q*x**2/2]
    >>> b.bc_slope = [(0, [0, 0, 0]), (l, [0, 0, 0])]
    >>> b.bc_deflection = [(0, [0, 0, 0]), (l, [0, 0, 0])]
    >>> b.solve_slope_deflection()
    >>> factor(b.slope())
    [0, 0, x*(-l + x)*(-A*G*l**3*q + 2*A*G*l**2*q*x - 12*E*I*l*q
        - 72*E*I*m + 24*E*I*q*x)/(12*E*I*(A*G*l**2 + 12*E*I))]
    >>> dx, dy, dz = b.deflection()
    >>> dy = collect(simplify(dy), x)
    >>> dx == dz == 0
    True
    >>> dy == (x*(12*E*I*l*(A*G*l**2*q - 2*A*G*l*m + 12*E*I*q)
    ... + x*(A*G*l*(3*l*(A*G*l**2*q - 2*A*G*l*m + 12*E*I*q) + x*(-2*A*G*l**2*q + 4*A*G*l*m - 24*E*I*q))
    ... + A*G*(A*G*l**2 + 12*E*I)*(-2*l**2*q + 6*l*m - 4*m*x + q*x**2)
    ... - 12*E*I*q*(A*G*l**2 + 12*E*I)))/(24*A*E*G*I*(A*G*l**2 + 12*E*I)))
    True

    References
    ==========

    .. [1] https://homes.civil.aau.dk/jc/FemteSemester/Beams3D.pdf

    """

    def __init__(self, length, elastic_modulus, shear_modulus, second_moment,
                 area, variable=Symbol('x')):
        """Initializes the class.

        Parameters
        ==========
        length : Sympifyable
            A Symbol or value representing the Beam's length.
        elastic_modulus : Sympifyable
            A SymPy expression representing the Beam's Modulus of Elasticity.
            It is a measure of the stiffness of the Beam material.
        shear_modulus : Sympifyable
            A SymPy expression representing the Beam's Modulus of rigidity.
            It is a measure of rigidity of the Beam material.
        second_moment : Sympifyable or list
            A list of two elements having SymPy expression representing the
            Beam's Second moment of area. First value represent Second moment
            across y-axis and second across z-axis.
            Single SymPy expression can be passed if both values are same
        area : Sympifyable
            A SymPy expression representing the Beam's cross-sectional area
            in a plane perpendicular to length of the Beam.
        variable : Symbol, optional
            A Symbol object that will be used as the variable along the beam
            while representing the load, shear, moment, slope and deflection
            curve. By default, it is set to ``Symbol('x')``.
        """
        super().__init__(length, elastic_modulus, second_moment, variable)
        self.shear_modulus = shear_modulus
        self.area = area
        self._load_vector = [0, 0, 0]
        self._moment_load_vector = [0, 0, 0]
        self._torsion_moment = {}
        self._load_Singularity = [0, 0, 0]
        self._slope = [0, 0, 0]
        self._deflection = [0, 0, 0]
        self._angular_deflection = 0

    @property
    def shear_modulus(self):
        """Young's Modulus of the Beam. """
        return self._shear_modulus

    @shear_modulus.setter
    def shear_modulus(self, e):
        self._shear_modulus = sympify(e)

    @property
    def second_moment(self):
        """Second moment of area of the Beam. """
        return self._second_moment

    @second_moment.setter
    def second_moment(self, i):
        if isinstance(i, list):
            i = [sympify(x) for x in i]
            self._second_moment = i
        else:
            self._second_moment = sympify(i)

    @property
    def area(self):
        """Cross-sectional area of the Beam. """
        return self._area

    @area.setter
    def area(self, a):
        self._area = sympify(a)

    @property
    def load_vector(self):
        """
        Returns a three element list representing the load vector.
        """
        return self._load_vector

    @property
    def moment_load_vector(self):
        """
        Returns a three element list representing moment loads on Beam.
        """
        return self._moment_load_vector

    @property
    def boundary_conditions(self):
        """
        Returns a dictionary of boundary conditions applied on the beam.
        The dictionary has two keywords namely slope and deflection.
        The value of each keyword is a list of tuple, where each tuple
        contains location and value of a boundary condition in the format
        (location, value). Further each value is a list corresponding to
        slope or deflection(s) values along three axes at that location.

        Examples
        ========
        There is a beam of length 4 meters. The slope at 0 should be 4 along
        the x-axis and 0 along others. At the other end of beam, deflection
        along all the three axes should be zero.

        >>> from sympy.physics.continuum_mechanics.beam import Beam3D
        >>> from sympy import symbols
        >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
        >>> b = Beam3D(30, E, G, I, A, x)
        >>> b.bc_slope = [(0, (4, 0, 0))]
        >>> b.bc_deflection = [(4, [0, 0, 0])]
        >>> b.boundary_conditions
        {'bending_moment': [], 'deflection': [(4, [0, 0, 0])], 'shear_force': [], 'slope': [(0, (4, 0, 0))]}

        Here the deflection of the beam should be ``0`` along all the three axes at ``4``.
        Similarly, the slope of the beam should be ``4`` along x-axis and ``0``
        along y and z axis at ``0``.
        """
        return self._boundary_conditions

    def polar_moment(self):
        """
        Returns the polar moment of area of the beam
        about the X axis with respect to the centroid.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.beam import Beam3D
        >>> from sympy import symbols
        >>> l, E, G, I, A = symbols('l, E, G, I, A')
        >>> b = Beam3D(l, E, G, I, A)
        >>> b.polar_moment()
        2*I
        >>> I1 = [9, 15]
        >>> b = Beam3D(l, E, G, I1, A)
        >>> b.polar_moment()
        24
        """
        if not iterable(self.second_moment):
            return 2*self.second_moment
        return sum(self.second_moment)

    def apply_load(self, value, start, order, dir="y"):
        """
        This method adds up the force load to a particular beam object.

        Parameters
        ==========
        value : Sympifyable
            The magnitude of an applied load.
        dir : String
            Axis along which load is applied.
        order : Integer
            The order of the applied load.
            - For point loads, order=-1
            - For constant distributed load, order=0
            - For ramp loads, order=1
            - For parabolic ramp loads, order=2
            - ... so on.
        """
        x = self.variable
        value = sympify(value)
        start = sympify(start)
        order = sympify(order)

        if dir == "x":
            if not order == -1:
                self._load_vector[0] += value
            self._load_Singularity[0] += value*SingularityFunction(x, start, order)

        elif dir == "y":
            if not order == -1:
                self._load_vector[1] += value
            self._load_Singularity[1] += value*SingularityFunction(x, start, order)

        else:
            if not order == -1:
                self._load_vector[2] += value
            self._load_Singularity[2] += value*SingularityFunction(x, start, order)

    def apply_moment_load(self, value, start, order, dir="y"):
        """
        This method adds up the moment loads to a particular beam object.

        Parameters
        ==========
        value : Sympifyable
            The magnitude of an applied moment.
        dir : String
            Axis along which moment is applied.
        order : Integer
            The order of the applied load.
            - For point moments, order=-2
            - For constant distributed moment, order=-1
            - For ramp moments, order=0
            - For parabolic ramp moments, order=1
            - ... so on.
        """
        x = self.variable
        value = sympify(value)
        start = sympify(start)
        order = sympify(order)

        if dir == "x":
            if not order == -2:
                self._moment_load_vector[0] += value
            else:
                if start in list(self._torsion_moment):
                    self._torsion_moment[start] += value
                else:
                    self._torsion_moment[start] = value
            self._load_Singularity[0] += value*SingularityFunction(x, start, order)
        elif dir == "y":
            if not order == -2:
                self._moment_load_vector[1] += value
            self._load_Singularity[0] += value*SingularityFunction(x, start, order)
        else:
            if not order == -2:
                self._moment_load_vector[2] += value
            self._load_Singularity[0] += value*SingularityFunction(x, start, order)

    def apply_support(self, loc, type="fixed"):
        if type in ("pin", "roller"):
            reaction_load = Symbol('R_'+str(loc))
            self._reaction_loads[reaction_load] = reaction_load
            self.bc_deflection.append((loc, [0, 0, 0]))
        else:
            reaction_load = Symbol('R_'+str(loc))
            reaction_moment = Symbol('M_'+str(loc))
            self._reaction_loads[reaction_load] = [reaction_load, reaction_moment]
            self.bc_deflection.append((loc, [0, 0, 0]))
            self.bc_slope.append((loc, [0, 0, 0]))

    def solve_for_reaction_loads(self, *reaction):
        """
        Solves for the reaction forces.

        Examples
        ========
        There is a beam of length 30 meters. It it supported by rollers at
        of its end. A constant distributed load of magnitude 8 N is applied
        from start till its end along y-axis. Another linear load having
        slope equal to 9 is applied along z-axis.

        >>> from sympy.physics.continuum_mechanics.beam import Beam3D
        >>> from sympy import symbols
        >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
        >>> b = Beam3D(30, E, G, I, A, x)
        >>> b.apply_load(8, start=0, order=0, dir="y")
        >>> b.apply_load(9*x, start=0, order=0, dir="z")
        >>> b.bc_deflection = [(0, [0, 0, 0]), (30, [0, 0, 0])]
        >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
        >>> b.apply_load(R1, start=0, order=-1, dir="y")
        >>> b.apply_load(R2, start=30, order=-1, dir="y")
        >>> b.apply_load(R3, start=0, order=-1, dir="z")
        >>> b.apply_load(R4, start=30, order=-1, dir="z")
        >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
        >>> b.reaction_loads
        {R1: -120, R2: -120, R3: -1350, R4: -2700}
        """
        x = self.variable
        l = self.length
        q = self._load_Singularity
        shear_curves = [integrate(load, x) for load in q]
        moment_curves = [integrate(shear, x) for shear in shear_curves]
        for i in range(3):
            react = [r for r in reaction if (shear_curves[i].has(r) or moment_curves[i].has(r))]
            if len(react) == 0:
                continue
            shear_curve = limit(shear_curves[i], x, l)
            moment_curve = limit(moment_curves[i], x, l)
            sol = list((linsolve([shear_curve, moment_curve], react).args)[0])
            sol_dict = dict(zip(react, sol))
            reaction_loads = self._reaction_loads
            # Check if any of the evaluated reaction exists in another direction
            # and if it exists then it should have same value.
            for key in sol_dict:
                if key in reaction_loads and sol_dict[key] != reaction_loads[key]:
                    raise ValueError("Ambiguous solution for %s in different directions." % key)
            self._reaction_loads.update(sol_dict)

    def shear_force(self):
        """
        Returns a list of three expressions which represents the shear force
        curve of the Beam object along all three axes.
        """
        x = self.variable
        q = self._load_vector
        return [integrate(-q[0], x), integrate(-q[1], x), integrate(-q[2], x)]

    def axial_force(self):
        """
        Returns expression of Axial shear force present inside the Beam object.
        """
        return self.shear_force()[0]

    def shear_stress(self):
        """
        Returns a list of three expressions which represents the shear stress
        curve of the Beam object along all three axes.
        """
        return [self.shear_force()[0]/self._area, self.shear_force()[1]/self._area, self.shear_force()[2]/self._area]

    def axial_stress(self):
        """
        Returns expression of Axial stress present inside the Beam object.
        """
        return self.axial_force()/self._area

    def bending_moment(self):
        """
        Returns a list of three expressions which represents the bending moment
        curve of the Beam object along all three axes.
        """
        x = self.variable
        m = self._moment_load_vector
        shear = self.shear_force()

        return [integrate(-m[0], x), integrate(-m[1] + shear[2], x),
                integrate(-m[2] - shear[1], x) ]

    def torsional_moment(self):
        """
        Returns expression of Torsional moment present inside the Beam object.
        """
        return self.bending_moment()[0]

    def solve_for_torsion(self):
        """
        Solves for the angular deflection due to the torsional effects of
        moments being applied in the x-direction i.e. out of or into the beam.

        Here, a positive torque means the direction of the torque is positive
        i.e. out of the beam along the beam-axis. Likewise, a negative torque
        signifies a torque into the beam cross-section.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.beam import Beam3D
        >>> from sympy import symbols
        >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
        >>> b = Beam3D(20, E, G, I, A, x)
        >>> b.apply_moment_load(4, 4, -2, dir='x')
        >>> b.apply_moment_load(4, 8, -2, dir='x')
        >>> b.apply_moment_load(4, 8, -2, dir='x')
        >>> b.solve_for_torsion()
        >>> b.angular_deflection().subs(x, 3)
        18/(G*I)
        """
        x = self.variable
        sum_moments = 0
        for point in list(self._torsion_moment):
            sum_moments += self._torsion_moment[point]
        list(self._torsion_moment).sort()
        pointsList = list(self._torsion_moment)
        torque_diagram = Piecewise((sum_moments, x<=pointsList[0]), (0, x>=pointsList[0]))
        for i in range(len(pointsList))[1:]:
            sum_moments -= self._torsion_moment[pointsList[i-1]]
            torque_diagram += Piecewise((0, x<=pointsList[i-1]), (sum_moments, x<=pointsList[i]), (0, x>=pointsList[i]))
        integrated_torque_diagram = integrate(torque_diagram)
        self._angular_deflection =  integrated_torque_diagram/(self.shear_modulus*self.polar_moment())

    def solve_slope_deflection(self):
        x = self.variable
        l = self.length
        E = self.elastic_modulus
        G = self.shear_modulus
        I = self.second_moment
        if isinstance(I, list):
            I_y, I_z = I[0], I[1]
        else:
            I_y = I_z = I
        A = self._area
        load = self._load_vector
        moment = self._moment_load_vector
        defl = Function('defl')
        theta = Function('theta')

        # Finding deflection along x-axis(and corresponding slope value by differentiating it)
        # Equation used: Derivative(E*A*Derivative(def_x(x), x), x) + load_x = 0
        eq = Derivative(E*A*Derivative(defl(x), x), x) + load[0]
        def_x = dsolve(Eq(eq, 0), defl(x)).args[1]
        # Solving constants originated from dsolve
        C1 = Symbol('C1')
        C2 = Symbol('C2')
        constants = list((linsolve([def_x.subs(x, 0), def_x.subs(x, l)], C1, C2).args)[0])
        def_x = def_x.subs({C1:constants[0], C2:constants[1]})
        slope_x = def_x.diff(x)
        self._deflection[0] = def_x
        self._slope[0] = slope_x

        # Finding deflection along y-axis and slope across z-axis. System of equation involved:
        # 1: Derivative(E*I_z*Derivative(theta_z(x), x), x) + G*A*(Derivative(defl_y(x), x) - theta_z(x)) + moment_z = 0
        # 2: Derivative(G*A*(Derivative(defl_y(x), x) - theta_z(x)), x) + load_y = 0
        C_i = Symbol('C_i')
        # Substitute value of `G*A*(Derivative(defl_y(x), x) - theta_z(x))` from (2) in (1)
        eq1 = Derivative(E*I_z*Derivative(theta(x), x), x) + (integrate(-load[1], x) + C_i) + moment[2]
        slope_z = dsolve(Eq(eq1, 0)).args[1]

        # Solve for constants originated from using dsolve on eq1
        constants = list((linsolve([slope_z.subs(x, 0), slope_z.subs(x, l)], C1, C2).args)[0])
        slope_z = slope_z.subs({C1:constants[0], C2:constants[1]})

        # Put value of slope obtained back in (2) to solve for `C_i` and find deflection across y-axis
        eq2 = G*A*(Derivative(defl(x), x)) + load[1]*x - C_i - G*A*slope_z
        def_y = dsolve(Eq(eq2, 0), defl(x)).args[1]
        # Solve for constants originated from using dsolve on eq2
        constants = list((linsolve([def_y.subs(x, 0), def_y.subs(x, l)], C1, C_i).args)[0])
        self._deflection[1] = def_y.subs({C1:constants[0], C_i:constants[1]})
        self._slope[2] = slope_z.subs(C_i, constants[1])

        # Finding deflection along z-axis and slope across y-axis. System of equation involved:
        # 1: Derivative(E*I_y*Derivative(theta_y(x), x), x) - G*A*(Derivative(defl_z(x), x) + theta_y(x)) + moment_y = 0
        # 2: Derivative(G*A*(Derivative(defl_z(x), x) + theta_y(x)), x) + load_z = 0

        # Substitute value of `G*A*(Derivative(defl_y(x), x) + theta_z(x))` from (2) in (1)
        eq1 = Derivative(E*I_y*Derivative(theta(x), x), x) + (integrate(load[2], x) - C_i) + moment[1]
        slope_y = dsolve(Eq(eq1, 0)).args[1]
        # Solve for constants originated from using dsolve on eq1
        constants = list((linsolve([slope_y.subs(x, 0), slope_y.subs(x, l)], C1, C2).args)[0])
        slope_y = slope_y.subs({C1:constants[0], C2:constants[1]})

        # Put value of slope obtained back in (2) to solve for `C_i` and find deflection across z-axis
        eq2 = G*A*(Derivative(defl(x), x)) + load[2]*x - C_i + G*A*slope_y
        def_z = dsolve(Eq(eq2,0)).args[1]
        # Solve for constants originated from using dsolve on eq2
        constants = list((linsolve([def_z.subs(x, 0), def_z.subs(x, l)], C1, C_i).args)[0])
        self._deflection[2] = def_z.subs({C1:constants[0], C_i:constants[1]})
        self._slope[1] = slope_y.subs(C_i, constants[1])

    def slope(self):
        """
        Returns a three element list representing slope of deflection curve
        along all the three axes.
        """
        return self._slope

    def deflection(self):
        """
        Returns a three element list representing deflection curve along all
        the three axes.
        """
        return self._deflection

    def angular_deflection(self):
        """
        Returns a function in x depicting how the angular deflection, due to moments
        in the x-axis on the beam, varies with x.
        """
        return self._angular_deflection

    def _plot_shear_force(self, dir, subs=None):

        shear_force = self.shear_force()

        if dir == 'x':
            dir_num = 0
            color = 'r'

        elif dir == 'y':
            dir_num = 1
            color = 'g'

        elif dir == 'z':
            dir_num = 2
            color = 'b'

        if subs is None:
            subs = {}

        for sym in shear_force[dir_num].atoms(Symbol):
            if sym != self.variable and sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length

        return plot(shear_force[dir_num].subs(subs), (self.variable, 0, length), show = False, title='Shear Force along %c direction'%dir,
                xlabel=r'$\mathrm{X}$', ylabel=r'$\mathrm{V(%c)}$'%dir, line_color=color)

    def plot_shear_force(self, dir="all", subs=None):

        """

        Returns a plot for Shear force along all three directions
        present in the Beam object.

        Parameters
        ==========
        dir : string (default : "all")
            Direction along which shear force plot is required.
            If no direction is specified, all plots are displayed.
        subs : dictionary
            Python dictionary containing Symbols as key and their
            corresponding values.

        Examples
        ========
        There is a beam of length 20 meters. It is supported by rollers
        at both of its ends. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, E, G, I, A, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.plot_shear_force()
            PlotGrid object containing:
            Plot[0]:Plot object containing:
            [0]: cartesian line: 0 for x over (0.0, 20.0)
            Plot[1]:Plot object containing:
            [0]: cartesian line: -6*x**2 for x over (0.0, 20.0)
            Plot[2]:Plot object containing:
            [0]: cartesian line: -15*x for x over (0.0, 20.0)

        """

        dir = dir.lower()
        # For shear force along x direction
        if dir == "x":
            Px = self._plot_shear_force('x', subs)
            return Px.show()
        # For shear force along y direction
        elif dir == "y":
            Py = self._plot_shear_force('y', subs)
            return Py.show()
        # For shear force along z direction
        elif dir == "z":
            Pz = self._plot_shear_force('z', subs)
            return Pz.show()
        # For shear force along all direction
        else:
            Px = self._plot_shear_force('x', subs)
            Py = self._plot_shear_force('y', subs)
            Pz = self._plot_shear_force('z', subs)
            return PlotGrid(3, 1, Px, Py, Pz)

    def _plot_bending_moment(self, dir, subs=None):

        bending_moment = self.bending_moment()

        if dir == 'x':
            dir_num = 0
            color = 'g'

        elif dir == 'y':
            dir_num = 1
            color = 'c'

        elif dir == 'z':
            dir_num = 2
            color = 'm'

        if subs is None:
            subs = {}

        for sym in bending_moment[dir_num].atoms(Symbol):
            if sym != self.variable and sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length

        return plot(bending_moment[dir_num].subs(subs), (self.variable, 0, length), show = False, title='Bending Moment along %c direction'%dir,
                xlabel=r'$\mathrm{X}$', ylabel=r'$\mathrm{M(%c)}$'%dir, line_color=color)

    def plot_bending_moment(self, dir="all", subs=None):

        """

        Returns a plot for bending moment along all three directions
        present in the Beam object.

        Parameters
        ==========
        dir : string (default : "all")
            Direction along which bending moment plot is required.
            If no direction is specified, all plots are displayed.
        subs : dictionary
            Python dictionary containing Symbols as key and their
            corresponding values.

        Examples
        ========
        There is a beam of length 20 meters. It is supported by rollers
        at both of its ends. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, E, G, I, A, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.plot_bending_moment()
            PlotGrid object containing:
            Plot[0]:Plot object containing:
            [0]: cartesian line: 0 for x over (0.0, 20.0)
            Plot[1]:Plot object containing:
            [0]: cartesian line: -15*x**2/2 for x over (0.0, 20.0)
            Plot[2]:Plot object containing:
            [0]: cartesian line: 2*x**3 for x over (0.0, 20.0)

        """

        dir = dir.lower()
        # For bending moment along x direction
        if dir == "x":
            Px = self._plot_bending_moment('x', subs)
            return Px.show()
        # For bending moment along y direction
        elif dir == "y":
            Py = self._plot_bending_moment('y', subs)
            return Py.show()
        # For bending moment along z direction
        elif dir == "z":
            Pz = self._plot_bending_moment('z', subs)
            return Pz.show()
        # For bending moment along all direction
        else:
            Px = self._plot_bending_moment('x', subs)
            Py = self._plot_bending_moment('y', subs)
            Pz = self._plot_bending_moment('z', subs)
            return PlotGrid(3, 1, Px, Py, Pz)

    def _plot_slope(self, dir, subs=None):

        slope = self.slope()

        if dir == 'x':
            dir_num = 0
            color = 'b'

        elif dir == 'y':
            dir_num = 1
            color = 'm'

        elif dir == 'z':
            dir_num = 2
            color = 'g'

        if subs is None:
            subs = {}

        for sym in slope[dir_num].atoms(Symbol):
            if sym != self.variable and sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length


        return plot(slope[dir_num].subs(subs), (self.variable, 0, length), show = False, title='Slope along %c direction'%dir,
                xlabel=r'$\mathrm{X}$', ylabel=r'$\mathrm{\theta(%c)}$'%dir, line_color=color)

    def plot_slope(self, dir="all", subs=None):

        """

        Returns a plot for Slope along all three directions
        present in the Beam object.

        Parameters
        ==========
        dir : string (default : "all")
            Direction along which Slope plot is required.
            If no direction is specified, all plots are displayed.
        subs : dictionary
            Python dictionary containing Symbols as keys and their
            corresponding values.

        Examples
        ========
        There is a beam of length 20 meters. It is supported by rollers
        at both of its ends. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, 40, 21, 100, 25, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.solve_slope_deflection()
            >>> b.plot_slope()
            PlotGrid object containing:
            Plot[0]:Plot object containing:
            [0]: cartesian line: 0 for x over (0.0, 20.0)
            Plot[1]:Plot object containing:
            [0]: cartesian line: -x**3/1600 + 3*x**2/160 - x/8 for x over (0.0, 20.0)
            Plot[2]:Plot object containing:
            [0]: cartesian line: x**4/8000 - 19*x**2/172 + 52*x/43 for x over (0.0, 20.0)

        """

        dir = dir.lower()
        # For Slope along x direction
        if dir == "x":
            Px = self._plot_slope('x', subs)
            return Px.show()
        # For Slope along y direction
        elif dir == "y":
            Py = self._plot_slope('y', subs)
            return Py.show()
        # For Slope along z direction
        elif dir == "z":
            Pz = self._plot_slope('z', subs)
            return Pz.show()
        # For Slope along all direction
        else:
            Px = self._plot_slope('x', subs)
            Py = self._plot_slope('y', subs)
            Pz = self._plot_slope('z', subs)
            return PlotGrid(3, 1, Px, Py, Pz)

    def _plot_deflection(self, dir, subs=None):

        deflection = self.deflection()

        if dir == 'x':
            dir_num = 0
            color = 'm'

        elif dir == 'y':
            dir_num = 1
            color = 'r'

        elif dir == 'z':
            dir_num = 2
            color = 'c'

        if subs is None:
            subs = {}

        for sym in deflection[dir_num].atoms(Symbol):
            if sym != self.variable and sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length

        return plot(deflection[dir_num].subs(subs), (self.variable, 0, length), show = False, title='Deflection along %c direction'%dir,
                xlabel=r'$\mathrm{X}$', ylabel=r'$\mathrm{\delta(%c)}$'%dir, line_color=color)

    def plot_deflection(self, dir="all", subs=None):

        """

        Returns a plot for Deflection along all three directions
        present in the Beam object.

        Parameters
        ==========
        dir : string (default : "all")
            Direction along which deflection plot is required.
            If no direction is specified, all plots are displayed.
        subs : dictionary
            Python dictionary containing Symbols as keys and their
            corresponding values.

        Examples
        ========
        There is a beam of length 20 meters. It is supported by rollers
        at both of its ends. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, 40, 21, 100, 25, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.solve_slope_deflection()
            >>> b.plot_deflection()
            PlotGrid object containing:
            Plot[0]:Plot object containing:
            [0]: cartesian line: 0 for x over (0.0, 20.0)
            Plot[1]:Plot object containing:
            [0]: cartesian line: x**5/40000 - 4013*x**3/90300 + 26*x**2/43 + 1520*x/903 for x over (0.0, 20.0)
            Plot[2]:Plot object containing:
            [0]: cartesian line: x**4/6400 - x**3/160 + 27*x**2/560 + 2*x/7 for x over (0.0, 20.0)


        """

        dir = dir.lower()
        # For deflection along x direction
        if dir == "x":
            Px = self._plot_deflection('x', subs)
            return Px.show()
        # For deflection along y direction
        elif dir == "y":
            Py = self._plot_deflection('y', subs)
            return Py.show()
        # For deflection along z direction
        elif dir == "z":
            Pz = self._plot_deflection('z', subs)
            return Pz.show()
        # For deflection along all direction
        else:
            Px = self._plot_deflection('x', subs)
            Py = self._plot_deflection('y', subs)
            Pz = self._plot_deflection('z', subs)
            return PlotGrid(3, 1, Px, Py, Pz)

    def plot_loading_results(self, dir='x', subs=None):

        """

        Returns a subplot of Shear Force, Bending Moment,
        Slope and Deflection of the Beam object along the direction specified.

        Parameters
        ==========

        dir : string (default : "x")
               Direction along which plots are required.
               If no direction is specified, plots along x-axis are displayed.
        subs : dictionary
               Python dictionary containing Symbols as key and their
               corresponding values.

        Examples
        ========
        There is a beam of length 20 meters. It is supported by rollers
        at both of its ends. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, E, G, I, A, x)
            >>> subs = {E:40, G:21, I:100, A:25}
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.solve_slope_deflection()
            >>> b.plot_loading_results('y',subs)
            PlotGrid object containing:
            Plot[0]:Plot object containing:
            [0]: cartesian line: -6*x**2 for x over (0.0, 20.0)
            Plot[1]:Plot object containing:
            [0]: cartesian line: -15*x**2/2 for x over (0.0, 20.0)
            Plot[2]:Plot object containing:
            [0]: cartesian line: -x**3/1600 + 3*x**2/160 - x/8 for x over (0.0, 20.0)
            Plot[3]:Plot object containing:
            [0]: cartesian line: x**5/40000 - 4013*x**3/90300 + 26*x**2/43 + 1520*x/903 for x over (0.0, 20.0)

        """

        dir = dir.lower()
        if subs is None:
            subs = {}

        ax1 = self._plot_shear_force(dir, subs)
        ax2 = self._plot_bending_moment(dir, subs)
        ax3 = self._plot_slope(dir, subs)
        ax4 = self._plot_deflection(dir, subs)

        return PlotGrid(4, 1, ax1, ax2, ax3, ax4)

    def _plot_shear_stress(self, dir, subs=None):

        shear_stress = self.shear_stress()

        if dir == 'x':
            dir_num = 0
            color = 'r'

        elif dir == 'y':
            dir_num = 1
            color = 'g'

        elif dir == 'z':
            dir_num = 2
            color = 'b'

        if subs is None:
            subs = {}

        for sym in shear_stress[dir_num].atoms(Symbol):
            if sym != self.variable and sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length

        return plot(shear_stress[dir_num].subs(subs), (self.variable, 0, length), show = False, title='Shear stress along %c direction'%dir,
                xlabel=r'$\mathrm{X}$', ylabel=r'$\tau(%c)$'%dir, line_color=color)

    def plot_shear_stress(self, dir="all", subs=None):

        """

        Returns a plot for Shear Stress along all three directions
        present in the Beam object.

        Parameters
        ==========
        dir : string (default : "all")
            Direction along which shear stress plot is required.
            If no direction is specified, all plots are displayed.
        subs : dictionary
            Python dictionary containing Symbols as key and their
            corresponding values.

        Examples
        ========
        There is a beam of length 20 meters and area of cross section 2 square
        meters. It is supported by rollers at both of its ends. A linear load having
        slope equal to 12 is applied along y-axis. A constant distributed load
        of magnitude 15 N is applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, E, G, I, 2, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.plot_shear_stress()
            PlotGrid object containing:
            Plot[0]:Plot object containing:
            [0]: cartesian line: 0 for x over (0.0, 20.0)
            Plot[1]:Plot object containing:
            [0]: cartesian line: -3*x**2 for x over (0.0, 20.0)
            Plot[2]:Plot object containing:
            [0]: cartesian line: -15*x/2 for x over (0.0, 20.0)

        """

        dir = dir.lower()
        # For shear stress along x direction
        if dir == "x":
            Px = self._plot_shear_stress('x', subs)
            return Px.show()
        # For shear stress along y direction
        elif dir == "y":
            Py = self._plot_shear_stress('y', subs)
            return Py.show()
        # For shear stress along z direction
        elif dir == "z":
            Pz = self._plot_shear_stress('z', subs)
            return Pz.show()
        # For shear stress along all direction
        else:
            Px = self._plot_shear_stress('x', subs)
            Py = self._plot_shear_stress('y', subs)
            Pz = self._plot_shear_stress('z', subs)
            return PlotGrid(3, 1, Px, Py, Pz)

    def _max_shear_force(self, dir):
        """
        Helper function for max_shear_force().
        """

        dir = dir.lower()

        if dir == 'x':
            dir_num = 0

        elif dir == 'y':
            dir_num = 1

        elif dir == 'z':
            dir_num = 2

        if not self.shear_force()[dir_num]:
            return (0,0)
        # To restrict the range within length of the Beam
        load_curve = Piecewise((float("nan"), self.variable<=0),
                (self._load_vector[dir_num], self.variable<self.length),
                (float("nan"), True))

        points = solve(load_curve.rewrite(Piecewise), self.variable,
                        domain=S.Reals)
        points.append(0)
        points.append(self.length)
        shear_curve = self.shear_force()[dir_num]
        shear_values = [shear_curve.subs(self.variable, x) for x in points]
        shear_values = list(map(abs, shear_values))

        max_shear = max(shear_values)
        return (points[shear_values.index(max_shear)], max_shear)

    def max_shear_force(self):
        """
        Returns point of max shear force and its corresponding shear value
        along all directions in a Beam object as a list.
        solve_for_reaction_loads() must be called before using this function.

        Examples
        ========
        There is a beam of length 20 meters. It is supported by rollers
        at both of its ends. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, 40, 21, 100, 25, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.max_shear_force()
            [(0, 0), (20, 2400), (20, 300)]
        """

        max_shear = []
        max_shear.append(self._max_shear_force('x'))
        max_shear.append(self._max_shear_force('y'))
        max_shear.append(self._max_shear_force('z'))
        return max_shear

    def _max_bending_moment(self, dir):
        """
        Helper function for max_bending_moment().
        """

        dir = dir.lower()

        if dir == 'x':
            dir_num = 0

        elif dir == 'y':
            dir_num = 1

        elif dir == 'z':
            dir_num = 2

        if not self.bending_moment()[dir_num]:
            return (0,0)
        # To restrict the range within length of the Beam
        shear_curve = Piecewise((float("nan"), self.variable<=0),
                (self.shear_force()[dir_num], self.variable<self.length),
                (float("nan"), True))

        points = solve(shear_curve.rewrite(Piecewise), self.variable,
                        domain=S.Reals)
        points.append(0)
        points.append(self.length)
        bending_moment_curve = self.bending_moment()[dir_num]
        bending_moments = [bending_moment_curve.subs(self.variable, x) for x in points]
        bending_moments = list(map(abs, bending_moments))

        max_bending_moment = max(bending_moments)
        return (points[bending_moments.index(max_bending_moment)], max_bending_moment)

    def max_bending_moment(self):
        """
        Returns point of max bending moment and its corresponding bending moment value
        along all directions in a Beam object as a list.
        solve_for_reaction_loads() must be called before using this function.

        Examples
        ========
        There is a beam of length 20 meters. It is supported by rollers
        at both of its ends. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, 40, 21, 100, 25, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.max_bending_moment()
            [(0, 0), (20, 3000), (20, 16000)]
        """

        max_bmoment = []
        max_bmoment.append(self._max_bending_moment('x'))
        max_bmoment.append(self._max_bending_moment('y'))
        max_bmoment.append(self._max_bending_moment('z'))
        return max_bmoment

    max_bmoment = max_bending_moment

    def _max_deflection(self, dir):
        """
        Helper function for max_Deflection()
        """

        dir = dir.lower()

        if dir == 'x':
            dir_num = 0

        elif dir == 'y':
            dir_num = 1

        elif dir == 'z':
            dir_num = 2

        if not self.deflection()[dir_num]:
            return (0,0)
        # To restrict the range within length of the Beam
        slope_curve = Piecewise((float("nan"), self.variable<=0),
                (self.slope()[dir_num], self.variable<self.length),
                (float("nan"), True))

        points = solve(slope_curve.rewrite(Piecewise), self.variable,
                        domain=S.Reals)
        points.append(0)
        points.append(self._length)
        deflection_curve = self.deflection()[dir_num]
        deflections = [deflection_curve.subs(self.variable, x) for x in points]
        deflections = list(map(abs, deflections))

        max_def = max(deflections)
        return (points[deflections.index(max_def)], max_def)

    def max_deflection(self):
        """
        Returns point of max deflection and its corresponding deflection value
        along all directions in a Beam object as a list.
        solve_for_reaction_loads() and solve_slope_deflection() must be called
        before using this function.

        Examples
        ========
        There is a beam of length 20 meters. It is supported by rollers
        at both of its ends. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, 40, 21, 100, 25, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.solve_slope_deflection()
            >>> b.max_deflection()
            [(0, 0), (10, 495/14), (-10 + 10*sqrt(10793)/43, (10 - 10*sqrt(10793)/43)**3/160 - 20/7 + (10 - 10*sqrt(10793)/43)**4/6400 + 20*sqrt(10793)/301 + 27*(10 - 10*sqrt(10793)/43)**2/560)]
        """

        max_def = []
        max_def.append(self._max_deflection('x'))
        max_def.append(self._max_deflection('y'))
        max_def.append(self._max_deflection('z'))
        return max_def
