"""
This module can be used to solve problems related
to 2D Cables.
"""

from sympy.core.sympify import sympify
from sympy.core.symbol import Symbol
from sympy import sin, cos, pi, atan, diff
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.solvers.solveset import linsolve
from sympy.matrices import Matrix


class Cable:
    """
    Cables are structures in engineering that support
    the applied transverse loads through the tensile
    resistance developed in its members.

    Cables are widely used in suspension bridges, tension
    leg offshore platforms, transmission lines, and find
    use in several other engineering applications.

    Examples
    ========
    A cable is supported at (0, 10) and (10, 10). Two point loads
    acting vertically downwards act on the cable, one with magnitude 3 kN
    and acting 2 meters from the left support and 3 meters below it, while
    the other with magnitude 2 kN is 6 meters from the left support and
    6 meters below it.

    >>> from sympy.physics.continuum_mechanics.cable import Cable
    >>> c = Cable(('A', 0, 10), ('B', 10, 10))
    >>> c.apply_load(-1, ('P', 2, 7, 3, 270))
    >>> c.apply_load(-1, ('Q', 6, 4, 2, 270))
    >>> c.loads
    {'distributed': {}, 'point_load': {'P': [3, 270], 'Q': [2, 270]}}
    >>> c.loads_position
    {'P': [2, 7], 'Q': [6, 4]}
    """
    def __init__(self, support_1, support_2):
        """
        Initializes the class.

        Parameters
        ==========

        support_1 and support_2 are tuples of the form
        (label, x, y), where

        label : String or symbol
            The label of the support

        x : Sympifyable
            The x coordinate of the position of the support

        y : Sympifyable
            The y coordinate of the position of the support
        """
        self._left_support = []
        self._right_support = []
        self._supports = {}
        self._support_labels = []
        self._loads = {"distributed": {}, "point_load": {}}
        self._loads_position = {}
        self._length = 0
        self._reaction_loads = {}
        self._tension = {}
        self._lowest_x_global = sympify(0)

        if support_1[0] == support_2[0]:
            raise ValueError("Supports can not have the same label")

        elif support_1[1] == support_2[1]:
            raise ValueError("Supports can not be at the same location")

        x1 = sympify(support_1[1])
        y1 = sympify(support_1[2])
        self._supports[support_1[0]] = [x1, y1]

        x2 = sympify(support_2[1])
        y2 = sympify(support_2[2])
        self._supports[support_2[0]] = [x2, y2]

        if support_1[1] < support_2[1]:
            self._left_support.append(x1)
            self._left_support.append(y1)
            self._right_support.append(x2)
            self._right_support.append(y2)
            self._support_labels.append(support_1[0])
            self._support_labels.append(support_2[0])

        else:
            self._left_support.append(x2)
            self._left_support.append(y2)
            self._right_support.append(x1)
            self._right_support.append(y1)
            self._support_labels.append(support_2[0])
            self._support_labels.append(support_1[0])

        for i in self._support_labels:
            self._reaction_loads[Symbol("R_"+ i +"_x")] = 0
            self._reaction_loads[Symbol("R_"+ i +"_y")] = 0

    @property
    def supports(self):
        """
        Returns the supports of the cable along with their
        positions.
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
    def loads(self):
        """
        Returns the magnitude and direction of the loads
        acting on the cable.
        """
        return self._loads

    @property
    def loads_position(self):
        """
        Returns the position of the point loads acting on the
        cable.
        """
        return self._loads_position

    @property
    def length(self):
        """
        Returns the length of the cable.
        """
        return self._length

    @property
    def reaction_loads(self):
        """
        Returns the reaction forces at the supports, which are
        initialized to 0.
        """
        return self._reaction_loads

    @property
    def tension(self):
        """
        Returns the tension developed in the cable due to the loads
        applied.
        """
        return self._tension

    def tension_at(self, x):
        """
        Returns the tension at a given value of x developed due to
        distributed load.
        """
        if 'distributed' not in self._tension.keys():
            raise ValueError("No distributed load added or solve method not called")

        if x > self._right_support[0] or x < self._left_support[0]:
            raise ValueError("The value of x should be between the two supports")

        A = self._tension['distributed']
        X = Symbol('X')

        return A.subs({X:(x-self._lowest_x_global)})

    def apply_length(self, length):
        """
        This method specifies the length of the cable

        Parameters
        ==========

        length : Sympifyable
            The length of the cable

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c = Cable(('A', 0, 10), ('B', 10, 10))
        >>> c.apply_length(20)
        >>> c.length
        20
        """
        dist = ((self._left_support[0] - self._right_support[0])**2
                - (self._left_support[1] - self._right_support[1])**2)**(1/2)

        if length < dist:
            raise ValueError("length should not be less than the distance between the supports")

        self._length = length

    def change_support(self, label, new_support):
        """
        This method changes the mentioned support with a new support.

        Parameters
        ==========
        label: String or symbol
            The label of the support to be changed

        new_support: Tuple of the form (new_label, x, y)
            new_label: String or symbol
                The label of the new support

            x: Sympifyable
                The x-coordinate of the position of the new support.

            y: Sympifyable
                The y-coordinate of the position of the new support.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c = Cable(('A', 0, 10), ('B', 10, 10))
        >>> c.supports
        {'A': [0, 10], 'B': [10, 10]}
        >>> c.change_support('B', ('C', 5, 6))
        >>> c.supports
        {'A': [0, 10], 'C': [5, 6]}
        """
        if label not in self._supports:
            raise ValueError("No support exists with the given label")

        i = self._support_labels.index(label)
        rem_label = self._support_labels[(i+1)%2]
        x1 = self._supports[rem_label][0]
        y1 = self._supports[rem_label][1]

        x = sympify(new_support[1])
        y = sympify(new_support[2])

        for l in self._loads_position:
            if l[0] >= max(x, x1) or l[0] <= min(x, x1):
                raise ValueError("The change in support will throw an existing load out of range")

        self._supports.pop(label)
        self._left_support.clear()
        self._right_support.clear()
        self._reaction_loads.clear()
        self._support_labels.remove(label)

        self._supports[new_support[0]] = [x, y]

        if x1 < x:
            self._left_support.append(x1)
            self._left_support.append(y1)
            self._right_support.append(x)
            self._right_support.append(y)
            self._support_labels.append(new_support[0])

        else:
            self._left_support.append(x)
            self._left_support.append(y)
            self._right_support.append(x1)
            self._right_support.append(y1)
            self._support_labels.insert(0, new_support[0])

        for i in self._support_labels:
            self._reaction_loads[Symbol("R_"+ i +"_x")] = 0
            self._reaction_loads[Symbol("R_"+ i +"_y")] = 0

    def apply_load(self, order, load):
        """
        This method adds load to the cable.

        Parameters
        ==========

        order : Integer
            The order of the applied load.

                - For point loads, order = -1
                - For distributed load, order = 0

        load : tuple

            * For point loads, load is of the form (label, x, y, magnitude, direction), where:

            label : String or symbol
                The label of the load

            x : Sympifyable
                The x coordinate of the position of the load

            y : Sympifyable
                The y coordinate of the position of the load

            magnitude : Sympifyable
                The magnitude of the load. It must always be positive

            direction : Sympifyable
                The angle, in degrees, that the load vector makes with the horizontal
                in the counter-clockwise direction. It takes the values 0 to 360,
                inclusive.


            * For uniformly distributed load, load is of the form (label, magnitude)

            label : String or symbol
                The label of the load

            magnitude : Sympifyable
                The magnitude of the load. It must always be positive

        Examples
        ========

        For a point load of magnitude 12 units inclined at 30 degrees with the horizontal:

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c = Cable(('A', 0, 10), ('B', 10, 10))
        >>> c.apply_load(-1, ('Z', 5, 5, 12, 30))
        >>> c.loads
        {'distributed': {}, 'point_load': {'Z': [12, 30]}}
        >>> c.loads_position
        {'Z': [5, 5]}


        For a uniformly distributed load of magnitude 9 units:

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c = Cable(('A', 0, 10), ('B', 10, 10))
        >>> c.apply_load(0, ('X', 9))
        >>> c.loads
        {'distributed': {'X': 9}, 'point_load': {}}
        """
        if order == -1:
            if len(self._loads["distributed"]) != 0:
                raise ValueError("Distributed load already exists")

            label = load[0]
            if label in self._loads["point_load"]:
                raise ValueError("Label already exists")

            x = sympify(load[1])
            y = sympify(load[2])

            if x > self._right_support[0] or x < self._left_support[0]:
                raise ValueError("The load should be positioned between the supports")

            magnitude = sympify(load[3])
            direction = sympify(load[4])

            self._loads["point_load"][label] = [magnitude, direction]
            self._loads_position[label] = [x, y]

        elif order == 0:
            if len(self._loads_position) != 0:
                raise ValueError("Point load(s) already exist")

            label = load[0]
            if label in self._loads["distributed"]:
                raise ValueError("Label already exists")

            magnitude = sympify(load[1])

            self._loads["distributed"][label] = magnitude

        else:
            raise ValueError("Order should be either -1 or 0")

    def remove_loads(self, *args):
        """
        This methods removes the specified loads.

        Parameters
        ==========
        This input takes multiple label(s) as input
        label(s): String or symbol
            The label(s) of the loads to be removed.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c = Cable(('A', 0, 10), ('B', 10, 10))
        >>> c.apply_load(-1, ('Z', 5, 5, 12, 30))
        >>> c.loads
        {'distributed': {}, 'point_load': {'Z': [12, 30]}}
        >>> c.remove_loads('Z')
        >>> c.loads
        {'distributed': {}, 'point_load': {}}
        """
        for i in args:
            if len(self._loads_position) == 0:
                if i not in self._loads['distributed']:
                    raise ValueError("Error removing load " + i + ": no such load exists")

                else:
                    self._loads['disrtibuted'].pop(i)

            else:
                if i not in self._loads['point_load']:
                    raise ValueError("Error removing load " + i + ": no such load exists")

                else:
                    self._loads['point_load'].pop(i)
                    self._loads_position.pop(i)

    def solve(self, *args):
        """
        This method solves for the reaction forces at the supports, the tension developed in
        the cable, and updates the length of the cable.

        Parameters
        ==========
        This method requires no input when solving for point loads
        For distributed load, the x and y coordinates of the lowest point of the cable are
        required as

        x: Sympifyable
            The x coordinate of the lowest point

        y: Sympifyable
            The y coordinate of the lowest point

        Examples
        ========
        For point loads,

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c = Cable(("A", 0, 10), ("B", 10, 10))
        >>> c.apply_load(-1, ('Z', 2, 7.26, 3, 270))
        >>> c.apply_load(-1, ('X', 4, 6, 8, 270))
        >>> c.solve()
        >>> c.tension
        {A_Z: 8.91403453669861, X_B: 19*sqrt(13)/10, Z_X: 4.79150773600774}
        >>> c.reaction_loads
        {R_A_x: -5.25547445255474, R_A_y: 7.2, R_B_x: 5.25547445255474, R_B_y: 3.8}
        >>> c.length
        5.7560958484519 + 2*sqrt(13)

        For distributed load,

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c=Cable(("A", 0, 40),("B", 100, 20))
        >>> c.apply_load(0, ("X", 850))
        >>> c.solve(58.58, 0)
        >>> c.tension
        {'distributed': 36456.8485*sqrt(0.000543529004799705*(X + 0.00135624381275735)**2 + 1)}
        >>> c.tension_at(0)
        61709.0363315913
        >>> c.reaction_loads
        {R_A_x: 36456.8485, R_A_y: -49788.5866682485, R_B_x: 44389.8401587246, R_B_y: 42866.621696333}
        """

        if len(self._loads_position) != 0:
            sorted_position = sorted(self._loads_position.items(), key = lambda item : item[1][0])

            sorted_position.append(self._support_labels[1])
            sorted_position.insert(0, self._support_labels[0])

            self._tension.clear()
            moment_sum_from_left_support = 0
            moment_sum_from_right_support = 0
            F_x = 0
            F_y = 0
            self._length = 0

            for i in range(1, len(sorted_position)-1):
                if i == 1:
                    self._length+=sqrt((self._left_support[0] - self._loads_position[sorted_position[i][0]][0])**2 + (self._left_support[1] - self._loads_position[sorted_position[i][0]][1])**2)

                else:
                    self._length+=sqrt((self._loads_position[sorted_position[i-1][0]][0] - self._loads_position[sorted_position[i][0]][0])**2 + (self._loads_position[sorted_position[i-1][0]][1] - self._loads_position[sorted_position[i][0]][1])**2)

                if i == len(sorted_position)-2:
                    self._length+=sqrt((self._right_support[0] - self._loads_position[sorted_position[i][0]][0])**2 + (self._right_support[1] - self._loads_position[sorted_position[i][0]][1])**2)

                moment_sum_from_left_support += self._loads['point_load'][sorted_position[i][0]][0] * cos(pi * self._loads['point_load'][sorted_position[i][0]][1] / 180) * abs(self._left_support[1] - self._loads_position[sorted_position[i][0]][1])
                moment_sum_from_left_support += self._loads['point_load'][sorted_position[i][0]][0] * sin(pi * self._loads['point_load'][sorted_position[i][0]][1] / 180) * abs(self._left_support[0] - self._loads_position[sorted_position[i][0]][0])

                F_x += self._loads['point_load'][sorted_position[i][0]][0] * cos(pi * self._loads['point_load'][sorted_position[i][0]][1] / 180)
                F_y += self._loads['point_load'][sorted_position[i][0]][0] * sin(pi * self._loads['point_load'][sorted_position[i][0]][1] / 180)

                label = Symbol(sorted_position[i][0]+"_"+sorted_position[i+1][0])
                y2 = self._loads_position[sorted_position[i][0]][1]
                x2 = self._loads_position[sorted_position[i][0]][0]
                y1 = 0
                x1 = 0

                if i == len(sorted_position)-2:
                    x1 = self._right_support[0]
                    y1 = self._right_support[1]

                else:
                    x1 = self._loads_position[sorted_position[i+1][0]][0]
                    y1 = self._loads_position[sorted_position[i+1][0]][1]

                angle_with_horizontal = atan((y1 - y2)/(x1 - x2))

                tension = -(moment_sum_from_left_support)/(abs(self._left_support[1] - self._loads_position[sorted_position[i][0]][1])*cos(angle_with_horizontal) + abs(self._left_support[0] - self._loads_position[sorted_position[i][0]][0])*sin(angle_with_horizontal))
                self._tension[label] = tension
                moment_sum_from_right_support += self._loads['point_load'][sorted_position[i][0]][0] * cos(pi * self._loads['point_load'][sorted_position[i][0]][1] / 180) * abs(self._right_support[1] - self._loads_position[sorted_position[i][0]][1])
                moment_sum_from_right_support += self._loads['point_load'][sorted_position[i][0]][0] * sin(pi * self._loads['point_load'][sorted_position[i][0]][1] / 180) * abs(self._right_support[0] - self._loads_position[sorted_position[i][0]][0])

            label = Symbol(sorted_position[0][0]+"_"+sorted_position[1][0])
            y2 = self._loads_position[sorted_position[1][0]][1]
            x2 = self._loads_position[sorted_position[1][0]][0]
            x1 = self._left_support[0]
            y1 = self._left_support[1]

            angle_with_horizontal = -atan((y2 - y1)/(x2 - x1))
            tension = -(moment_sum_from_right_support)/(abs(self._right_support[1] - self._loads_position[sorted_position[1][0]][1])*cos(angle_with_horizontal) + abs(self._right_support[0] - self._loads_position[sorted_position[1][0]][0])*sin(angle_with_horizontal))
            self._tension[label] = tension

            angle_with_horizontal = pi/2 - angle_with_horizontal
            label = self._support_labels[0]
            self._reaction_loads[Symbol("R_"+label+"_x")] = -sin(angle_with_horizontal) * tension
            F_x += -sin(angle_with_horizontal) * tension
            self._reaction_loads[Symbol("R_"+label+"_y")] = cos(angle_with_horizontal) * tension
            F_y += cos(angle_with_horizontal) * tension

            label = self._support_labels[1]
            self._reaction_loads[Symbol("R_"+label+"_x")] = -F_x
            self._reaction_loads[Symbol("R_"+label+"_y")] = -F_y

        elif len(self._loads['distributed']) != 0 :

            if len(args) == 0:
                raise ValueError("Provide the lowest point of the cable")

            lowest_x = sympify(args[0])
            lowest_y = sympify(args[1])
            self._lowest_x_global = lowest_x

            a = Symbol('a')
            b = Symbol('b')
            c = Symbol('c')
            # augmented matrix form of linsolve

            M = Matrix(
                [[self._left_support[0]**2, self._left_support[0], 1, self._left_support[1]],
                [self._right_support[0]**2, self._right_support[0], 1, self._right_support[1]],
                [lowest_x**2, lowest_x, 1, lowest_y]  ]
                       )

            coefficient_solution = list(linsolve(M, (a, b, c)))

            if len(coefficient_solution) == 0:
                raise ValueError("The lowest point is inconsistent with the supports")

            A = coefficient_solution[0][0]
            B = coefficient_solution[0][1]
            C = coefficient_solution[0][2]


            # y = A*x**2 + B*x + C
            # shifting origin to lowest point
            X = Symbol('X')
            Y = Symbol('Y')
            Y = A*(X + lowest_x)**2 + B*(X + lowest_x) + C - lowest_y

            temp_list = list(self._loads['distributed'].values())
            applied_force = temp_list[0]

            horizontal_force_constant = (applied_force * (self._right_support[0] - lowest_x)**2) / (2 * (self._right_support[1] - lowest_y))

            self._tension.clear()
            tangent_slope_to_curve = diff(Y, X)
            self._tension['distributed'] = horizontal_force_constant / (cos(atan(tangent_slope_to_curve)))

            label = self._support_labels[0]
            self._reaction_loads[Symbol("R_"+label+"_x")] = self.tension_at(self._left_support[0]) * cos(atan(tangent_slope_to_curve.subs(X, self._left_support[0] - lowest_x)))
            self._reaction_loads[Symbol("R_"+label+"_y")] = self.tension_at(self._left_support[0]) * sin(atan(tangent_slope_to_curve.subs(X, self._left_support[0] - lowest_x)))

            label = self._support_labels[1]
            self._reaction_loads[Symbol("R_"+label+"_x")] = self.tension_at(self._left_support[0]) * cos(atan(tangent_slope_to_curve.subs(X, self._right_support[0] - lowest_x)))
            self._reaction_loads[Symbol("R_"+label+"_y")] = self.tension_at(self._left_support[0]) * sin(atan(tangent_slope_to_curve.subs(X, self._right_support[0] - lowest_x)))
