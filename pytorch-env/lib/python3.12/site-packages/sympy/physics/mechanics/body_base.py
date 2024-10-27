from abc import ABC, abstractmethod
from sympy import Symbol, sympify
from sympy.physics.vector import Point

__all__ = ['BodyBase']


class BodyBase(ABC):
    """Abstract class for body type objects."""
    def __init__(self, name, masscenter=None, mass=None):
        # Note: If frame=None, no auto-generated frame is created, because a
        # Particle does not need to have a frame by default.
        if not isinstance(name, str):
            raise TypeError('Supply a valid name.')
        self._name = name
        if mass is None:
            mass = Symbol(f'{name}_mass')
        if masscenter is None:
            masscenter = Point(f'{name}_masscenter')
        self.mass = mass
        self.masscenter = masscenter
        self.potential_energy = 0
        self.points = []

    def __str__(self):
        return self.name

    def __repr__(self):
        return (f'{self.__class__.__name__}({repr(self.name)}, masscenter='
                f'{repr(self.masscenter)}, mass={repr(self.mass)})')

    @property
    def name(self):
        """The name of the body."""
        return self._name

    @property
    def masscenter(self):
        """The body's center of mass."""
        return self._masscenter

    @masscenter.setter
    def masscenter(self, point):
        if not isinstance(point, Point):
            raise TypeError("The body's center of mass must be a Point object.")
        self._masscenter = point

    @property
    def mass(self):
        """The body's mass."""
        return self._mass

    @mass.setter
    def mass(self, mass):
        self._mass = sympify(mass)

    @property
    def potential_energy(self):
        """The potential energy of the body.

        Examples
        ========

        >>> from sympy.physics.mechanics import Particle, Point
        >>> from sympy import symbols
        >>> m, g, h = symbols('m g h')
        >>> O = Point('O')
        >>> P = Particle('P', O, m)
        >>> P.potential_energy = m * g * h
        >>> P.potential_energy
        g*h*m

        """
        return self._potential_energy

    @potential_energy.setter
    def potential_energy(self, scalar):
        self._potential_energy = sympify(scalar)

    @abstractmethod
    def kinetic_energy(self, frame):
        pass

    @abstractmethod
    def linear_momentum(self, frame):
        pass

    @abstractmethod
    def angular_momentum(self, point, frame):
        pass

    @abstractmethod
    def parallel_axis(self, point, frame):
        pass
