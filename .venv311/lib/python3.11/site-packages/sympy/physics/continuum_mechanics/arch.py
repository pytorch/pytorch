"""
This module can be used to solve probelsm related to 2D parabolic arches
"""
from sympy.core.sympify import sympify
from sympy.core.symbol import Symbol,symbols
from sympy import diff, sqrt, cos , sin, atan, rad, Min
from sympy.core.relational import Eq
from sympy.solvers.solvers import solve
from sympy.functions import Piecewise
from sympy.plotting import plot
from sympy import limit
from sympy.utilities.decorator import doctest_depends_on
from sympy.external.importtools import import_module

numpy = import_module('numpy', import_kwargs={'fromlist':['arange']})

class Arch:
    """
    This class is used to solve problems related to a three hinged arch(determinate) structure.\n
    An arch is a curved vertical structure spanning an open space underneath it.\n
    Arches can be used to reduce the bending moments in long-span structures.\n

    Arches are used in structural engineering(over windows, door and even bridges)\n
    because they can support a very large mass placed on top of them.

    Example
    ========
    >>> from sympy.physics.continuum_mechanics.arch import Arch
    >>> a = Arch((0,0),(10,0),crown_x=5,crown_y=5)
    >>> a.get_shape_eqn
    5 - (x - 5)**2/5

    >>> from sympy.physics.continuum_mechanics.arch import Arch
    >>> a = Arch((0,0),(10,1),crown_x=6)
    >>> a.get_shape_eqn
    9/5 - (x - 6)**2/20
    """
    def __init__(self,left_support,right_support,**kwargs):
        self._shape_eqn = None
        self._left_support  = (sympify(left_support[0]),sympify(left_support[1]))
        self._right_support  = (sympify(right_support[0]),sympify(right_support[1]))
        self._crown_x = None
        self._crown_y = None
        if 'crown_x' in kwargs:
            self._crown_x = sympify(kwargs['crown_x'])
        if 'crown_y' in kwargs:
            self._crown_y = sympify(kwargs['crown_y'])
        self._shape_eqn = self.get_shape_eqn
        self._conc_loads = {}
        self._distributed_loads = {}
        self._loads = {'concentrated': self._conc_loads, 'distributed':self._distributed_loads}
        self._loads_applied = {}
        self._supports = {'left':'hinge', 'right':'hinge'}
        self._member = None
        self._member_force = None
        self._reaction_force = {Symbol('R_A_x'):0, Symbol('R_A_y'):0, Symbol('R_B_x'):0, Symbol('R_B_y'):0}
        self._points_disc_x = set()
        self._points_disc_y = set()
        self._moment_x  = {}
        self._moment_y = {}
        self._load_x = {}
        self._load_y = {}
        self._moment_x_func = Piecewise((0,True))
        self._moment_y_func = Piecewise((0,True))
        self._load_x_func = Piecewise((0,True))
        self._load_y_func = Piecewise((0,True))
        self._bending_moment = None
        self._shear_force = None
        self._axial_force = None
        # self._crown = (sympify(crown[0]),sympify(crown[1]))

    @property
    def get_shape_eqn(self):
        "returns the equation of the shape of arch developed"
        if self._shape_eqn:
            return self._shape_eqn

        x,y,c = symbols('x y c')
        a = Symbol('a',positive=False)
        if self._crown_x and self._crown_y:
            x0  = self._crown_x
            y0 = self._crown_y
            parabola_eqn = a*(x-x0)**2 + y0 - y
            eq1 = parabola_eqn.subs({x:self._left_support[0], y:self._left_support[1]})
            solution = solve((eq1),(a))
            parabola_eqn = solution[0]*(x-x0)**2 + y0
            if(parabola_eqn.subs({x:self._right_support[0]}) != self._right_support[1]):
                raise ValueError("provided coordinates of crown and supports are not consistent with parabolic arch")

        elif self._crown_x:
            x0  = self._crown_x
            parabola_eqn = a*(x-x0)**2 + c - y
            eq1 = parabola_eqn.subs({x:self._left_support[0], y:self._left_support[1]})
            eq2 = parabola_eqn.subs({x:self._right_support[0], y:self._right_support[1]})
            solution = solve((eq1,eq2),(a,c))
            if len(solution) <2 or solution[a] == 0:
                raise ValueError("parabolic arch cannot be constructed with the provided coordinates, try providing crown_y")
            parabola_eqn = solution[a]*(x-x0)**2+ solution[c]
            self._crown_y = solution[c]

        else:
            raise KeyError("please provide crown_x to construct arch")

        return parabola_eqn

    @property
    def get_loads(self):
        """
        return the position of the applied load and angle (for concentrated loads)
        """
        return self._loads

    @property
    def supports(self):
        """
        Returns the type of support
        """
        return self._supports

    @property
    def left_support(self):
        """
        Returns the position of the left support.
        """
        return self._left_support

    @property
    def right_support(self):
        """
        Returns the position of the right support.
        """
        return self._right_support

    @property
    def reaction_force(self):
        """
        return the reaction forces generated
        """
        return self._reaction_force

    def apply_load(self,order,label,start,mag,end=None,angle=None):
        """
        This method adds load to the Arch.

        Parameters
        ==========

            order : Integer
                Order of the applied load.

                    - For point/concentrated loads, order = -1
                    - For distributed load, order = 0

            label : String or Symbol
                The label of the load
                - should not use 'A' or 'B' as it is used for supports.

            start : Float

                    - For concentrated/point loads, start is the x coordinate
                    - For distributed loads, start is the starting position of distributed load

            mag : Sympifyable
                Magnitude of the applied load. Must be positive

            end : Float
                Required for distributed loads

                    - For concentrated/point load , end is None(may not be given)
                    - For distributed loads, end is the end position of distributed load

            angle: Sympifyable
                The angle in degrees, the load vector makes with the horizontal
                in the counter-clockwise direction.

        Examples
        ========
        For applying distributed load

        >>> from sympy.physics.continuum_mechanics.arch import Arch
        >>> a = Arch((0,0),(10,0),crown_x=5,crown_y=5)
        >>> a.apply_load(0,'C',start=3,end=5,mag=-10)

        For applying point/concentrated_loads

        >>> from sympy.physics.continuum_mechanics.arch import Arch
        >>> a = Arch((0,0),(10,0),crown_x=5,crown_y=5)
        >>> a.apply_load(-1,'C',start=2,mag=15,angle=45)

        """
        y = Symbol('y')
        x = Symbol('x')
        x0 = Symbol('x0')
        # y0 = Symbol('y0')
        order= sympify(order)
        mag = sympify(mag)
        angle = sympify(angle)

        if label in self._loads_applied:
            raise ValueError("load with the given label already exists")

        if label in ['A','B']:
            raise ValueError("cannot use the given label, reserved for supports")

        if order == 0:
            if end is None or end<start:
                raise KeyError("provide end greater than start")

            self._distributed_loads[label] = {'start':start, 'end':end, 'f_y': mag}
            self._points_disc_y.add(start)

            if start in self._moment_y:
                self._moment_y[start] -= mag*(Min(x,end)-start)*(x0-(start+(Min(x,end)))/2)
                self._load_y[start] += mag*(Min(end,x)-start)
            else:
                self._moment_y[start] = -mag*(Min(x,end)-start)*(x0-(start+(Min(x,end)))/2)
                self._load_y[start] = mag*(Min(end,x)-start)

            self._loads_applied[label] = 'distributed'

        if order == -1:

            if angle is None:
                raise TypeError("please provide direction of force")
            height = self._shape_eqn.subs({'x':start})

            self._conc_loads[label] = {'x':start, 'y':height, 'f_x':mag*cos(rad(angle)), 'f_y': mag*sin(rad(angle)), 'mag':mag, 'angle':angle}
            self._points_disc_x.add(start)
            self._points_disc_y.add(start)

            if start in self._moment_x:
                self._moment_x[start] += self._conc_loads[label]['f_x']*(y-self._conc_loads[label]['y'])
                self._load_x[start] += self._conc_loads[label]['f_x']
            else:
                self._moment_x[start] = self._conc_loads[label]['f_x']*(y-self._conc_loads[label]['y'])
                self._load_x[start] = self._conc_loads[label]['f_x']

            if start in self._moment_y:
                self._moment_y[start] -= self._conc_loads[label]['f_y']*(x0-start)
                self._load_y[start] += self._conc_loads[label]['f_y']
            else:
                self._moment_y[start] = -self._conc_loads[label]['f_y']*(x0-start)
                self._load_y[start] = self._conc_loads[label]['f_y']

            self._loads_applied[label] = 'concentrated'


    def remove_load(self,label):
        """
        This methods removes the load applied to the arch

        Parameters
        ==========

        label : String or Symbol
            The label of the applied load

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.arch import Arch
        >>> a = Arch((0,0),(10,0),crown_x=5,crown_y=5)
        >>> a.apply_load(0,'C',start=3,end=5,mag=-10)
        >>> a.remove_load('C')
        removed load C: {'start': 3, 'end': 5, 'f_y': -10}
        """
        y = Symbol('y')
        x = Symbol('x')
        x0 = Symbol('x0')

        if label in self._distributed_loads :

            self._loads_applied.pop(label)
            start = self._distributed_loads[label]['start']
            end = self._distributed_loads[label]['end']
            mag  = self._distributed_loads[label]['f_y']
            self._points_disc_y.remove(start)
            self._load_y[start] -= mag*(Min(x,end)-start)
            self._moment_y[start] += mag*(Min(x,end)-start)*(x0-(start+(Min(x,end)))/2)
            val = self._distributed_loads.pop(label)
            print(f"removed load {label}: {val}")

        elif label in self._conc_loads :

            self._loads_applied.pop(label)
            start = self._conc_loads[label]['x']
            self._points_disc_x.remove(start)
            self._points_disc_y.remove(start)
            self._moment_y[start] += self._conc_loads[label]['f_y']*(x0-start)
            self._moment_x[start] -= self._conc_loads[label]['f_x']*(y-self._conc_loads[label]['y'])
            self._load_x[start] -= self._conc_loads[label]['f_x']
            self._load_y[start] -= self._conc_loads[label]['f_y']
            val = self._conc_loads.pop(label)
            print(f"removed load {label}: {val}")

        else :
            raise ValueError("label not found")

    def change_support_position(self, left_support=None, right_support=None):
        """
        Change position of supports.
        If not provided , defaults to the old value.
        Parameters
        ==========

            left_support: tuple (x, y)
                x: float
                    x-coordinate value of the left_support

                y: float
                    y-coordinate value of the left_support

            right_support: tuple (x, y)
                x: float
                    x-coordinate value of the right_support

                y: float
                    y-coordinate value of the right_support
        """
        if left_support is not None:
            self._left_support = (left_support[0],left_support[1])

        if right_support is not None:
            self._right_support = (right_support[0],right_support[1])

        self._shape_eqn = None
        self._shape_eqn = self.get_shape_eqn

    def change_crown_position(self,crown_x=None,crown_y=None):
        """
        Change the position of the crown/hinge of the arch

        Parameters
        ==========

            crown_x: Float
                The x coordinate of the position of the hinge
                - if not provided, defaults to old value

            crown_y: Float
                The y coordinate of the position of the hinge
                - if not provided defaults to None
        """
        self._crown_x = crown_x
        self._crown_y = crown_y
        self._shape_eqn = None
        self._shape_eqn = self.get_shape_eqn

    def change_support_type(self,left_support=None,right_support=None):
        """
        Add the type for support at each end.
        Can use roller or hinge support at each end.

        Parameters
        ==========

            left_support, right_support : string
                Type of support at respective end

                    - For roller support , left_support/right_support = "roller"
                    - For hinged support, left_support/right_support = "hinge"
                    - defaults to hinge if value not provided

        Examples
        ========

        For applying roller support at right end

        >>> from sympy.physics.continuum_mechanics.arch import Arch
        >>> a = Arch((0,0),(10,0),crown_x=5,crown_y=5)
        >>> a.change_support_type(right_support="roller")

        """
        support_types = ['roller','hinge']
        if left_support:
            if left_support not in support_types:
                raise ValueError("supports must only be roller or hinge")

            self._supports['left'] = left_support

        if right_support:
            if right_support not in support_types:
                raise ValueError("supports must only be roller or hinge")

            self._supports['right'] = right_support

    def add_member(self,y):
        """
        This method adds a member/rod at a particular height y.
        A rod is used for stability of the structure in case of a roller support.
        """
        if y>self._crown_y or y<min(self._left_support[1],  self._right_support[1]):
            raise ValueError(f"position of support must be between y={min(self._left_support[1],  self._right_support[1])} and y={self._crown_y}")
        x = Symbol('x')
        a = diff(self._shape_eqn,x).subs(x,self._crown_x+1)/2
        x_diff = sqrt((y - self._crown_y)/a)
        x1 = self._crown_x + x_diff
        x2 = self._crown_x - x_diff
        self._member = (x1,x2,y)

    def shear_force_at(self, pos = None, **kwargs):
        """
        return the shear at some x-coordinates
        if no x value provided, returns the formula
        """
        if pos is None:
            return self._shear_force
        else:
            x = Symbol('x')
            if 'dir' in kwargs:
                dir = kwargs['dir']
                return limit(self._shear_force,x,pos,dir=dir)
            return self._shear_force.subs(x,pos)

    def bending_moment_at(self, pos = None, **kwargs):
        """
        return the bending moment at some x-coordinates
        if no x value provided, returns the formula
        """
        if pos is None:
            return self._bending_moment
        else:
            x0 = Symbol('x0')
            if 'dir' in kwargs:
                dir = kwargs['dir']
                return limit(self._bending_moment,x0,pos,dir=dir)
            return self._bending_moment.subs(x0,pos)


    def axial_force_at(self,pos = None, **kwargs):
        """
        return the axial/normal force generated at some x-coordinate
        if no x value provided, returns the formula
        """
        if pos is None:
            return self._axial_force
        else:
            x = Symbol('x')
            if 'dir' in kwargs:
                dir = kwargs['dir']
                return limit(self._axial_force,x,pos,dir=dir)
            return self._axial_force.subs(x,pos)

    def solve(self):
        """
        This method solves for the reaction forces generated at the supports,\n
        and bending moment and generated in the arch and tension produced in the member if used.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.arch import Arch
        >>> a = Arch((0,0),(10,0),crown_x=5,crown_y=5)
        >>> a.apply_load(0,'C',start=3,end=5,mag=-10)
        >>> a.solve()
        >>> a.reaction_force
        {R_A_x: 8, R_A_y: 12, R_B_x: -8, R_B_y: 8}

        >>> from sympy import Symbol
        >>> t = Symbol('t')
        >>> from sympy.physics.continuum_mechanics.arch import Arch
        >>> a = Arch((0,0),(16,0),crown_x=8,crown_y=5)
        >>> a.apply_load(0,'C',start=3,end=5,mag=t)
        >>> a.solve()
        >>> a.reaction_force
        {R_A_x: -4*t/5, R_A_y: -3*t/2, R_B_x: 4*t/5, R_B_y: -t/2}

        >>> a.bending_moment_at(4)
        -5*t/2
        """
        y = Symbol('y')
        x = Symbol('x')
        x0 = Symbol('x0')

        discontinuity_points_x = sorted(self._points_disc_x)
        discontinuity_points_y = sorted(self._points_disc_y)

        self._moment_x_func = Piecewise((0,True))
        self._moment_y_func = Piecewise((0,True))

        self._load_x_func = Piecewise((0,True))
        self._load_y_func = Piecewise((0,True))

        accumulated_x_moment = 0
        accumulated_y_moment = 0

        accumulated_x_load = 0
        accumulated_y_load = 0

        for point in discontinuity_points_x:
            cond = (x >= point)
            accumulated_x_load += self._load_x[point]
            accumulated_x_moment += self._moment_x[point]
            self._load_x_func = Piecewise((accumulated_x_load,cond),(self._load_x_func,True))
            self._moment_x_func = Piecewise((accumulated_x_moment,cond),(self._moment_x_func,True))

        for point in discontinuity_points_y:
            cond = (x >= point)
            accumulated_y_moment += self._moment_y[point]
            accumulated_y_load += self._load_y[point]
            self._load_y_func = Piecewise((accumulated_y_load,cond),(self._load_y_func,True))
            self._moment_y_func = Piecewise((accumulated_y_moment,cond),(self._moment_y_func,True))

        moment_A = self._moment_y_func.subs(x,self._right_support[0]).subs(x0,self._left_support[0]) +\
                   self._moment_x_func.subs(x,self._right_support[0]).subs(y,self._left_support[1])

        moment_hinge_left = self._moment_y_func.subs(x,self._crown_x).subs(x0,self._crown_x) +\
                            self._moment_x_func.subs(x,self._crown_x).subs(y,self._crown_y)

        moment_hinge_right = self._moment_y_func.subs(x,self._right_support[0]).subs(x0,self._crown_x)- \
                             self._moment_y_func.subs(x,self._crown_x).subs(x0,self._crown_x) +\
                             self._moment_x_func.subs(x,self._right_support[0]).subs(y,self._crown_y) -\
                             self._moment_x_func.subs(x,self._crown_x).subs(y,self._crown_y)

        net_x = self._load_x_func.subs(x,self._right_support[0])
        net_y = self._load_y_func.subs(x,self._right_support[0])

        if (self._supports['left']=='roller' or self._supports['right']=='roller') and not self._member:
            print("member must be added if any of the supports is roller")
            return

        R_A_x, R_A_y, R_B_x, R_B_y, T = symbols('R_A_x R_A_y R_B_x R_B_y T')

        if self._supports['left'] == 'roller' and self._supports['right'] == 'roller':

            if self._member[2]>=max(self._left_support[1],self._right_support[1]):

                if net_x!=0:
                    raise ValueError("net force in x direction not possible under the specified conditions")

                else:
                    eq1 = Eq(R_A_x ,0)
                    eq2 = Eq(R_B_x, 0)
                    eq3 = Eq(R_A_y + R_B_y + net_y,0)

                    eq4 = Eq(R_B_y*(self._right_support[0]-self._left_support[0])-\
                             R_B_x*(self._right_support[1]-self._left_support[1])+moment_A,0)

                    eq5 = Eq(moment_hinge_right + R_B_y*(self._right_support[0]-self._crown_x) +\
                             T*(self._member[2]-self._crown_y),0)
                    solution = solve((eq1,eq2,eq3,eq4,eq5),(R_A_x,R_A_y,R_B_x,R_B_y,T))

            elif self._member[2]>=self._left_support[1]:
                eq1 = Eq(R_A_x ,0)
                eq2 = Eq(R_B_x, 0)
                eq3 = Eq(R_A_y + R_B_y + net_y,0)
                eq4 = Eq(R_B_y*(self._right_support[0]-self._left_support[0])-\
                         T*(self._member[2]-self._left_support[1])+moment_A,0)
                eq5 = Eq(T+net_x,0)
                solution = solve((eq1,eq2,eq3,eq4,eq5),(R_A_x,R_A_y,R_B_x,R_B_y,T))

            elif self._member[2]>=self._right_support[1]:
                eq1 = Eq(R_A_x ,0)
                eq2 = Eq(R_B_x, 0)
                eq3 = Eq(R_A_y + R_B_y + net_y,0)
                eq4 = Eq(R_B_y*(self._right_support[0]-self._left_support[0])+\
                         T*(self._member[2]-self._left_support[1])+moment_A,0)
                eq5 = Eq(T-net_x,0)
                solution = solve((eq1,eq2,eq3,eq4,eq5),(R_A_x,R_A_y,R_B_x,R_B_y,T))

        elif self._supports['left'] == 'roller':
            if self._member[2]>=max(self._left_support[1], self._right_support[1]):
                eq1 = Eq(R_A_x ,0)
                eq2 = Eq(R_B_x+net_x,0)
                eq3 = Eq(R_A_y + R_B_y + net_y,0)
                eq4 = Eq(R_B_y*(self._right_support[0]-self._left_support[0])-\
                         R_B_x*(self._right_support[1]-self._left_support[1])+moment_A,0)
                eq5 = Eq(moment_hinge_left + R_A_y*(self._left_support[0]-self._crown_x) -\
                         T*(self._member[2]-self._crown_y),0)
                solution = solve((eq1,eq2,eq3,eq4,eq5),(R_A_x,R_A_y,R_B_x,R_B_y,T))

            elif self._member[2]>=self._left_support[1]:
                eq1 = Eq(R_A_x ,0)
                eq2 = Eq(R_B_x+ T +net_x,0)
                eq3 = Eq(R_A_y + R_B_y + net_y,0)
                eq4 = Eq(R_B_y*(self._right_support[0]-self._left_support[0])-\
                         R_B_x*(self._right_support[1]-self._left_support[1])-\
                         T*(self._member[2]-self._left_support[0])+moment_A,0)
                eq5 = Eq(moment_hinge_left + R_A_y*(self._left_support[0]-self._crown_x)-\
                         T*(self._member[2]-self._crown_y),0)
                solution = solve((eq1,eq2,eq3,eq4,eq5),(R_A_x,R_A_y,R_B_x,R_B_y,T))

            elif self._member[2]>=self._right_support[0]:
                eq1 = Eq(R_A_x,0)
                eq2 = Eq(R_B_x- T +net_x,0)
                eq3 = Eq(R_A_y + R_B_y + net_y,0)
                eq4 = Eq(moment_hinge_left+R_A_y*(self._left_support[0]-self._crown_x),0)
                eq5 = Eq(moment_A+R_B_y*(self._right_support[0]-self._left_support[0])-\
                         R_B_x*(self._right_support[1]-self._left_support[1])+\
                         T*(self._member[2]-self._left_support[1]),0)
                solution = solve((eq1,eq2,eq3,eq4,eq5),(R_A_x,R_A_y,R_B_x,R_B_y,T))

        elif self._supports['right'] == 'roller':
            if self._member[2]>=max(self._left_support[1], self._right_support[1]):
                eq1 = Eq(R_B_x,0)
                eq2 = Eq(R_A_x+net_x,0)
                eq3 = Eq(R_A_y+R_B_y+net_y,0)
                eq4 = Eq(moment_hinge_right+R_B_y*(self._right_support[0]-self._crown_x)+\
                         T*(self._member[2]-self._crown_y),0)
                eq5 = Eq(moment_A+R_B_y*(self._right_support[0]-self._left_support[0]),0)
                solution = solve((eq1,eq2,eq3,eq4,eq5),(R_A_x,R_A_y,R_B_x,R_B_y,T))

            elif self._member[2]>=self._left_support[1]:
                eq1 = Eq(R_B_x,0)
                eq2 = Eq(R_A_x+T+net_x,0)
                eq3 = Eq(R_A_y+R_B_y+net_y,0)
                eq4 = Eq(moment_hinge_right+R_B_y*(self._right_support[0]-self._crown_x),0)
                eq5 = Eq(moment_A-T*(self._member[2]-self._left_support[1])+\
                         R_B_y*(self._right_support[0]-self._left_support[0]),0)
                solution = solve((eq1,eq2,eq3,eq4,eq5),(R_A_x,R_A_y,R_B_x,R_B_y,T))

            elif self._member[2]>=self._right_support[1]:
                eq1 = Eq(R_B_x,0)
                eq2 = Eq(R_A_x-T+net_x,0)
                eq3 = Eq(R_A_y+R_B_y+net_y,0)
                eq4 = Eq(moment_hinge_right+R_B_y*(self._right_support[0]-self._crown_x)+\
                         T*(self._member[2]-self._crown_y),0)
                eq5 = Eq(moment_A+T*(self._member[2]-self._left_support[1])+\
                         R_B_y*(self._right_support[0]-self._left_support[0]))
                solution = solve((eq1,eq2,eq3,eq4,eq5),(R_A_x,R_A_y,R_B_x,R_B_y,T))
        else:
            eq1 = Eq(R_A_x + R_B_x + net_x,0)
            eq2 = Eq(R_A_y + R_B_y + net_y,0)
            eq3 = Eq(R_B_y*(self._right_support[0]-self._left_support[0])-\
                     R_B_x*(self._right_support[1]-self._left_support[1])+moment_A,0)
            eq4 = Eq(moment_hinge_right + R_B_y*(self._right_support[0]-self._crown_x) -\
                     R_B_x*(self._right_support[1]-self._crown_y),0)
            solution = solve((eq1,eq2,eq3,eq4),(R_A_x,R_A_y,R_B_x,R_B_y))

        for symb in self._reaction_force:
            self._reaction_force[symb] = solution[symb]

        self._bending_moment = - (self._moment_x_func.subs(x,x0) + self._moment_y_func.subs(x,x0) -\
                                  solution[R_A_y]*(x0-self._left_support[0]) +\
                                  solution[R_A_x]*(self._shape_eqn.subs({x:x0})-self._left_support[1]))

        angle  = atan(diff(self._shape_eqn,x))

        fx = (self._load_x_func+solution[R_A_x])
        fy = (self._load_y_func+solution[R_A_y])

        axial_force = fx*cos(angle) + fy*sin(angle)
        shear_force = -fx*sin(angle) + fy*cos(angle)

        self._axial_force = axial_force
        self._shear_force = shear_force

    @doctest_depends_on(modules=('numpy',))
    def draw(self):
        """
        This method returns a plot object containing the diagram of the specified arch along with the supports
        and forces applied to the structure.

        Examples
        ========

        >>> from sympy import Symbol
        >>> t = Symbol('t')
        >>> from sympy.physics.continuum_mechanics.arch import Arch
        >>> a = Arch((0,0),(40,0),crown_x=20,crown_y=12)
        >>> a.apply_load(-1,'C',8,150,angle=270)
        >>> a.apply_load(0,'D',start=20,end=40,mag=-4)
        >>> a.apply_load(-1,'E',10,t,angle=300)
        >>> p = a.draw()
        >>> p # doctest: +ELLIPSIS
        Plot object containing:
        [0]: cartesian line: 11.325 - 3*(x - 20)**2/100 for x over (0.0, 40.0)
        [1]: cartesian line: 12 - 3*(x - 20)**2/100 for x over (0.0, 40.0)
        ...
        >>> p.show()

        """
        x = Symbol('x')
        markers = []
        annotations = self._draw_loads()
        rectangles = []
        supports = self._draw_supports()
        markers+=supports

        xmax = self._right_support[0]
        xmin = self._left_support[0]
        ymin = min(self._left_support[1],self._right_support[1])
        ymax = self._crown_y

        lim = max(xmax*1.1-xmin*0.8+1, ymax*1.1-ymin*0.8+1)

        rectangles = self._draw_rectangles()

        filler = self._draw_filler()
        rectangles+=filler

        if self._member is not None:
            if(self._member[2]>=self._right_support[1]):
                markers.append(
                    {
                        'args':[[self._member[1]+0.005*lim],[self._member[2]]],
                        'marker':'o',
                        'markersize': 4,
                        'color': 'white',
                        'markerfacecolor':'none'
                    }
                )

            if(self._member[2]>=self._left_support[1]):
                markers.append(
                    {
                        'args':[[self._member[0]-0.005*lim],[self._member[2]]],
                        'marker':'o',
                        'markersize': 4,
                        'color': 'white',
                        'markerfacecolor':'none'
                    }
                )



        markers.append({
            'args':[[self._crown_x],[self._crown_y-0.005*lim]],
            'marker':'o',
            'markersize': 5,
            'color':'white',
            'markerfacecolor':'none',
        })

        if lim==xmax*1.1-xmin*0.8+1:

            sing_plot = plot(self._shape_eqn-0.015*lim,
                             self._shape_eqn,
                             (x, self._left_support[0], self._right_support[0]),
                             markers=markers,
                             show=False,
                             annotations=annotations,
                             rectangles = rectangles,
                             xlim=(xmin-0.05*lim, xmax*1.1),
                             ylim=(xmin-0.05*lim, xmax*1.1),
                             axis=False,
                             line_color='brown')

        else:
            sing_plot = plot(self._shape_eqn-0.015*lim,
                             self._shape_eqn,
                             (x, self._left_support[0], self._right_support[0]),
                             markers=markers,
                             show=False,
                             annotations=annotations,
                             rectangles = rectangles,
                             xlim=(ymin-0.05*lim, ymax*1.1),
                             ylim=(ymin-0.05*lim, ymax*1.1),
                             axis=False,
                             line_color='brown')

        return sing_plot


    def _draw_supports(self):
        support_markers = []

        xmax = self._right_support[0]
        xmin = self._left_support[0]
        ymin = min(self._left_support[1],self._right_support[1])
        ymax = self._crown_y

        if abs(1.1*xmax-0.8*xmin)>abs(1.1*ymax-0.8*ymin):
            max_diff = 1.1*xmax-0.8*xmin
        else:
            max_diff = 1.1*ymax-0.8*ymin

        if self._supports['left']=='roller':
            support_markers.append(
                {
                    'args':[
                        [self._left_support[0]],
                        [self._left_support[1]-0.02*max_diff]
                    ],
                    'marker':'o',
                    'markersize':11,
                    'color':'black',
                    'markerfacecolor':'none'
                }
            )
        else:
            support_markers.append(
                {
                    'args':[
                        [self._left_support[0]],
                        [self._left_support[1]-0.007*max_diff]
                    ],
                    'marker':6,
                    'markersize':15,
                    'color':'black',
                    'markerfacecolor':'none'
                }
            )

        if self._supports['right']=='roller':
            support_markers.append(
                {
                    'args':[
                        [self._right_support[0]],
                        [self._right_support[1]-0.02*max_diff]
                    ],
                    'marker':'o',
                    'markersize':11,
                    'color':'black',
                    'markerfacecolor':'none'
                }
            )
        else:
            support_markers.append(
                {
                    'args':[
                        [self._right_support[0]],
                        [self._right_support[1]-0.007*max_diff]
                    ],
                    'marker':6,
                    'markersize':15,
                    'color':'black',
                    'markerfacecolor':'none'
                }
            )

        support_markers.append(
            {
                'args':[
                    [self._right_support[0]],
                    [self._right_support[1]-0.036*max_diff]
                ],
                'marker':'_',
                'markersize':15,
                'color':'black',
                'markerfacecolor':'none'
            }
        )

        support_markers.append(
            {
                'args':[
                    [self._left_support[0]],
                    [self._left_support[1]-0.036*max_diff]
                ],
                'marker':'_',
                'markersize':15,
                'color':'black',
                'markerfacecolor':'none'
            }
        )

        return support_markers

    def _draw_rectangles(self):
        member = []

        xmax = self._right_support[0]
        xmin = self._left_support[0]
        ymin = min(self._left_support[1],self._right_support[1])
        ymax = self._crown_y

        if abs(1.1*xmax-0.8*xmin)>abs(1.1*ymax-0.8*ymin):
            max_diff = 1.1*xmax-0.8*xmin
        else:
            max_diff = 1.1*ymax-0.8*ymin

        if self._member is not None:
            if self._member[2]>= max(self._left_support[1],self._right_support[1]):
                member.append(
                    {
                        'xy':(self._member[0],self._member[2]-0.005*max_diff),
                        'width':self._member[1]-self._member[0],
                        'height': 0.01*max_diff,
                        'angle': 0,
                        'color':'brown',
                    }
                )

            elif self._member[2]>=self._left_support[1]:
                member.append(
                    {
                        'xy':(self._member[0],self._member[2]-0.005*max_diff),
                        'width':self._right_support[0]-self._member[0],
                        'height': 0.01*max_diff,
                        'angle': 0,
                        'color':'brown',
                    }
                )

            else:
                member.append(
                    {
                        'xy':(self._member[1],self._member[2]-0.005*max_diff),
                        'width':abs(self._left_support[0]-self._member[1]),
                        'height': 0.01*max_diff,
                        'angle': 180,
                        'color':'brown',
                    }
                )

        if self._distributed_loads:
            for loads in self._distributed_loads:

                start = self._distributed_loads[loads]['start']
                end = self._distributed_loads[loads]['end']

                member.append(
                    {
                        'xy':(start,self._crown_y+max_diff*0.15),
                        'width': (end-start),
                        'height': max_diff*0.01,
                        'color': 'orange'
                    }
                )


        return member

    def _draw_loads(self):
        load_annotations = []

        xmax = self._right_support[0]
        xmin = self._left_support[0]
        ymin = min(self._left_support[1],self._right_support[1])
        ymax = self._crown_y

        if abs(1.1*xmax-0.8*xmin)>abs(1.1*ymax-0.8*ymin):
            max_diff = 1.1*xmax-0.8*xmin
        else:
            max_diff = 1.1*ymax-0.8*ymin

        for load in self._conc_loads:
            x = self._conc_loads[load]['x']
            y = self._conc_loads[load]['y']
            angle = self._conc_loads[load]['angle']
            mag = self._conc_loads[load]['mag']
            load_annotations.append(
                {
                    'text':'',
                    'xy':(
                        x+cos(rad(angle))*max_diff*0.08,
                        y+sin(rad(angle))*max_diff*0.08
                    ),
                    'xytext':(x,y),
                    'fontsize':10,
                    'fontweight': 'bold',
                    'arrowprops':{'width':1.5, 'headlength':5, 'headwidth':5, 'facecolor':'blue','edgecolor':'blue'}
                }
            )
            load_annotations.append(
                {
                    'text':f'{load}: {mag} N',
                    'fontsize':10,
                    'fontweight': 'bold',
                    'xy': (x+cos(rad(angle))*max_diff*0.12,y+sin(rad(angle))*max_diff*0.12)
                }
            )

        for load in self._distributed_loads:
            start = self._distributed_loads[load]['start']
            end = self._distributed_loads[load]['end']
            mag = self._distributed_loads[load]['f_y']
            x_points = numpy.arange(start,end,(end-start)/(max_diff*0.25))
            x_points = numpy.append(x_points,end)
            for point in x_points:
                if(mag<0):
                    load_annotations.append(
                        {
                            'text':'',
                            'xy':(point,self._crown_y+max_diff*0.05),
                            'xytext': (point,self._crown_y+max_diff*0.15),
                            'arrowprops':{'width':1.5, 'headlength':5, 'headwidth':5, 'facecolor':'orange','edgecolor':'orange'}
                        }
                    )
                else:
                    load_annotations.append(
                        {
                            'text':'',
                            'xy':(point,self._crown_y+max_diff*0.2),
                            'xytext': (point,self._crown_y+max_diff*0.15),
                            'arrowprops':{'width':1.5, 'headlength':5, 'headwidth':5, 'facecolor':'orange','edgecolor':'orange'}
                        }
                    )
            if(mag<0):
                load_annotations.append(
                    {
                        'text':f'{load}: {abs(mag)} N/m',
                        'fontsize':10,
                        'fontweight': 'bold',
                        'xy':((start+end)/2,self._crown_y+max_diff*0.175)
                    }
                )
            else:
                load_annotations.append(
                    {
                        'text':f'{load}: {abs(mag)} N/m',
                        'fontsize':10,
                        'fontweight': 'bold',
                        'xy':((start+end)/2,self._crown_y+max_diff*0.125)
                    }
                )
        return load_annotations

    def _draw_filler(self):
        x = Symbol('x')
        filler = []
        xmax = self._right_support[0]
        xmin = self._left_support[0]
        ymin = min(self._left_support[1],self._right_support[1])
        ymax = self._crown_y

        if abs(1.1*xmax-0.8*xmin)>abs(1.1*ymax-0.8*ymin):
            max_diff = 1.1*xmax-0.8*xmin
        else:
            max_diff = 1.1*ymax-0.8*ymin

        x_points = numpy.arange(self._left_support[0],self._right_support[0],(self._right_support[0]-self._left_support[0])/(max_diff*max_diff))

        for point in x_points:
            filler.append(
                    {
                        'xy':(point,self._shape_eqn.subs(x,point)-max_diff*0.015),
                        'width': (self._right_support[0]-self._left_support[0])/(max_diff*max_diff),
                        'height': max_diff*0.015,
                        'color': 'brown'
                    }
            )

        return filler
