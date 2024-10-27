"""
This module can be used to solve problems related
to 2D Trusses.
"""


from cmath import atan, inf
from sympy.core.add import Add
from sympy.core.evalf import INF
from sympy.core.mul import Mul
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy import Matrix, pi
from sympy.external.importtools import import_module
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import zeros
import math
from sympy.physics.units.quantities import Quantity
from sympy.plotting import plot
from sympy.utilities.decorator import doctest_depends_on
from sympy import sin, cos


__doctest_requires__ = {('Truss.draw'): ['matplotlib']}


numpy = import_module('numpy', import_kwargs={'fromlist':['arange']})


class Truss:
    """
    A Truss is an assembly of members such as beams,
    connected by nodes, that create a rigid structure.
    In engineering, a truss is a structure that
    consists of two-force members only.

    Trusses are extremely important in engineering applications
    and can be seen in numerous real-world applications like bridges.

    Examples
    ========

    There is a Truss consisting of four nodes and five
    members connecting the nodes. A force P acts
    downward on the node D and there also exist pinned
    and roller joints on the nodes A and B respectively.

    .. image:: truss_example.png

    >>> from sympy.physics.continuum_mechanics.truss import Truss
    >>> t = Truss()
    >>> t.add_node(("node_1", 0, 0), ("node_2", 6, 0), ("node_3", 2, 2), ("node_4", 2, 0))
    >>> t.add_member(("member_1", "node_1", "node_4"), ("member_2", "node_2", "node_4"), ("member_3", "node_1", "node_3"))
    >>> t.add_member(("member_4", "node_2", "node_3"), ("member_5", "node_3", "node_4"))
    >>> t.apply_load(("node_4", 10, 270))
    >>> t.apply_support(("node_1", "pinned"), ("node_2", "roller"))
    """

    def __init__(self):
        """
        Initializes the class
        """
        self._nodes = []
        self._members = {}
        self._loads = {}
        self._supports = {}
        self._node_labels = []
        self._node_positions = []
        self._node_position_x = []
        self._node_position_y = []
        self._nodes_occupied = {}
        self._member_lengths = {}
        self._reaction_loads = {}
        self._internal_forces = {}
        self._node_coordinates = {}

    @property
    def nodes(self):
        """
        Returns the nodes of the truss along with their positions.
        """
        return self._nodes

    @property
    def node_labels(self):
        """
        Returns the node labels of the truss.
        """
        return self._node_labels

    @property
    def node_positions(self):
        """
        Returns the positions of the nodes of the truss.
        """
        return self._node_positions

    @property
    def members(self):
        """
        Returns the members of the truss along with the start and end points.
        """
        return self._members

    @property
    def member_lengths(self):
        """
        Returns the length of each member of the truss.
        """
        return self._member_lengths

    @property
    def supports(self):
        """
        Returns the nodes with provided supports along with the kind of support provided i.e.
        pinned or roller.
        """
        return self._supports

    @property
    def loads(self):
        """
        Returns the loads acting on the truss.
        """
        return self._loads

    @property
    def reaction_loads(self):
        """
        Returns the reaction forces for all supports which are all initialized to 0.
        """
        return self._reaction_loads

    @property
    def internal_forces(self):
        """
        Returns the internal forces for all members which are all initialized to 0.
        """
        return self._internal_forces

    def add_node(self, *args):
        """
        This method adds a node to the truss along with its name/label and its location.
        Multiple nodes can be added at the same time.

        Parameters
        ==========
        The input(s) for this method are tuples of the form (label, x, y).

        label:  String or a Symbol
            The label for a node. It is the only way to identify a particular node.

        x: Sympifyable
            The x-coordinate of the position of the node.

        y: Sympifyable
            The y-coordinate of the position of the node.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0))
        >>> t.nodes
        [('A', 0, 0)]
        >>> t.add_node(('B', 3, 0), ('C', 4, 1))
        >>> t.nodes
        [('A', 0, 0), ('B', 3, 0), ('C', 4, 1)]
        """

        for i in args:
            label = i[0]
            x = i[1]
            x = sympify(x)
            y=i[2]
            y = sympify(y)
            if label in self._node_coordinates:
                raise ValueError("Node needs to have a unique label")

            elif [x, y] in self._node_coordinates.values():
                raise ValueError("A node already exists at the given position")

            else :
                self._nodes.append((label, x, y))
                self._node_labels.append(label)
                self._node_positions.append((x, y))
                self._node_position_x.append(x)
                self._node_position_y.append(y)
                self._node_coordinates[label] = [x, y]



    def remove_node(self, *args):
        """
        This method removes a node from the truss.
        Multiple nodes can be removed at the same time.

        Parameters
        ==========
        The input(s) for this method are the labels of the nodes to be removed.

        label:  String or Symbol
            The label of the node to be removed.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0), ('C', 5, 0))
        >>> t.nodes
        [('A', 0, 0), ('B', 3, 0), ('C', 5, 0)]
        >>> t.remove_node('A', 'C')
        >>> t.nodes
        [('B', 3, 0)]
        """
        for label in args:
            for i in range(len(self.nodes)):
                if self._node_labels[i] == label:
                    x = self._node_position_x[i]
                    y = self._node_position_y[i]

            if label not in self._node_coordinates:
                raise ValueError("No such node exists in the truss")

            else:
                members_duplicate = self._members.copy()
                for member in members_duplicate:
                    if label == self._members[member][0] or label == self._members[member][1]:
                        raise ValueError("The given node already has member attached to it")
                self._nodes.remove((label, x, y))
                self._node_labels.remove(label)
                self._node_positions.remove((x, y))
                self._node_position_x.remove(x)
                self._node_position_y.remove(y)
                if label in self._loads:
                    self._loads.pop(label)
                if label in self._supports:
                    self._supports.pop(label)
                self._node_coordinates.pop(label)



    def add_member(self, *args):
        """
        This method adds a member between any two nodes in the given truss.

        Parameters
        ==========
        The input(s) of the method are tuple(s) of the form (label, start, end).

        label: String or Symbol
            The label for a member. It is the only way to identify a particular member.

        start: String or Symbol
            The label of the starting point/node of the member.

        end: String or Symbol
            The label of the ending point/node of the member.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0), ('C', 2, 2))
        >>> t.add_member(('AB', 'A', 'B'), ('BC', 'B', 'C'))
        >>> t.members
        {'AB': ['A', 'B'], 'BC': ['B', 'C']}
        """
        for i in args:
            label = i[0]
            start = i[1]
            end = i[2]

            if start not in self._node_coordinates or end not in self._node_coordinates or start==end:
                raise ValueError("The start and end points of the member must be unique nodes")

            elif label in self._members:
                raise ValueError("A member with the same label already exists for the truss")

            elif self._nodes_occupied.get((start, end)):
                raise ValueError("A member already exists between the two nodes")

            else:
                self._members[label] = [start, end]
                self._member_lengths[label] = sqrt((self._node_coordinates[end][0]-self._node_coordinates[start][0])**2 + (self._node_coordinates[end][1]-self._node_coordinates[start][1])**2)
                self._nodes_occupied[start, end] = True
                self._nodes_occupied[end, start] = True
                self._internal_forces[label] = 0

    def remove_member(self, *args):
        """
        This method removes members from the given truss.

        Parameters
        ==========
        labels: String or Symbol
            The label for the member to be removed.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0), ('C', 2, 2))
        >>> t.add_member(('AB', 'A', 'B'), ('AC', 'A', 'C'), ('BC', 'B', 'C'))
        >>> t.members
        {'AB': ['A', 'B'], 'AC': ['A', 'C'], 'BC': ['B', 'C']}
        >>> t.remove_member('AC', 'BC')
        >>> t.members
        {'AB': ['A', 'B']}
        """
        for label in args:
            if label not in self._members:
                raise ValueError("No such member exists in the Truss")

            else:
                self._nodes_occupied.pop((self._members[label][0], self._members[label][1]))
                self._nodes_occupied.pop((self._members[label][1], self._members[label][0]))
                self._members.pop(label)
                self._member_lengths.pop(label)
                self._internal_forces.pop(label)

    def change_node_label(self, *args):
        """
        This method changes the label(s) of the specified node(s).

        Parameters
        ==========
        The input(s) of this method are tuple(s) of the form (label, new_label).

        label: String or Symbol
            The label of the node for which the label has
            to be changed.

        new_label: String or Symbol
            The new label of the node.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0))
        >>> t.nodes
        [('A', 0, 0), ('B', 3, 0)]
        >>> t.change_node_label(('A', 'C'), ('B', 'D'))
        >>> t.nodes
        [('C', 0, 0), ('D', 3, 0)]
        """
        for i in args:
            label = i[0]
            new_label = i[1]
            if label not in self._node_coordinates:
                raise ValueError("No such node exists for the Truss")
            elif new_label in self._node_coordinates:
                raise ValueError("A node with the given label already exists")
            else:
                for node in self._nodes:
                    if node[0] == label:
                        self._nodes[self._nodes.index((label, node[1], node[2]))] = (new_label, node[1], node[2])
                        self._node_labels[self._node_labels.index(node[0])] = new_label
                        self._node_coordinates[new_label] = self._node_coordinates[label]
                        self._node_coordinates.pop(label)
                        if node[0] in self._supports:
                            self._supports[new_label] = self._supports[node[0]]
                            self._supports.pop(node[0])
                        if new_label in self._supports:
                            if self._supports[new_label] == 'pinned':
                                if 'R_'+str(label)+'_x' in self._reaction_loads and 'R_'+str(label)+'_y' in self._reaction_loads:
                                    self._reaction_loads['R_'+str(new_label)+'_x'] = self._reaction_loads['R_'+str(label)+'_x']
                                    self._reaction_loads['R_'+str(new_label)+'_y'] = self._reaction_loads['R_'+str(label)+'_y']
                                    self._reaction_loads.pop('R_'+str(label)+'_x')
                                    self._reaction_loads.pop('R_'+str(label)+'_y')
                                self._loads[new_label] = self._loads[label]
                                for load in self._loads[new_label]:
                                    if load[1] == 90:
                                        load[0] -= Symbol('R_'+str(label)+'_y')
                                        if load[0] == 0:
                                            self._loads[label].remove(load)
                                        break
                                for load in self._loads[new_label]:
                                    if load[1] == 0:
                                        load[0] -= Symbol('R_'+str(label)+'_x')
                                        if load[0] == 0:
                                            self._loads[label].remove(load)
                                        break
                                self.apply_load(new_label, Symbol('R_'+str(new_label)+'_x'), 0)
                                self.apply_load(new_label, Symbol('R_'+str(new_label)+'_y'), 90)
                                self._loads.pop(label)
                            elif self._supports[new_label] == 'roller':
                                self._loads[new_label] = self._loads[label]
                                for load in self._loads[label]:
                                    if load[1] == 90:
                                        load[0] -= Symbol('R_'+str(label)+'_y')
                                        if load[0] == 0:
                                            self._loads[label].remove(load)
                                        break
                                self.apply_load(new_label, Symbol('R_'+str(new_label)+'_y'), 90)
                                self._loads.pop(label)
                        else:
                            if label in self._loads:
                                self._loads[new_label] = self._loads[label]
                                self._loads.pop(label)
                        for member in self._members:
                            if self._members[member][0] == node[0]:
                                self._members[member][0] = new_label
                                self._nodes_occupied[(new_label, self._members[member][1])] = True
                                self._nodes_occupied[(self._members[member][1], new_label)] = True
                                self._nodes_occupied.pop((label, self._members[member][1]))
                                self._nodes_occupied.pop((self._members[member][1], label))
                            elif self._members[member][1] == node[0]:
                                self._members[member][1] = new_label
                                self._nodes_occupied[(self._members[member][0], new_label)] = True
                                self._nodes_occupied[(new_label, self._members[member][0])] = True
                                self._nodes_occupied.pop((self._members[member][0], label))
                                self._nodes_occupied.pop((label, self._members[member][0]))

    def change_member_label(self, *args):
        """
        This method changes the label(s) of the specified member(s).

        Parameters
        ==========
        The input(s) of this method are tuple(s) of the form (label, new_label)

        label: String or Symbol
            The label of the member for which the label has
            to be changed.

        new_label: String or Symbol
            The new label of the member.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0), ('D', 5, 0))
        >>> t.nodes
        [('A', 0, 0), ('B', 3, 0), ('D', 5, 0)]
        >>> t.change_node_label(('A', 'C'))
        >>> t.nodes
        [('C', 0, 0), ('B', 3, 0), ('D', 5, 0)]
        >>> t.add_member(('BC', 'B', 'C'), ('BD', 'B', 'D'))
        >>> t.members
        {'BC': ['B', 'C'], 'BD': ['B', 'D']}
        >>> t.change_member_label(('BC', 'BC_new'), ('BD', 'BD_new'))
        >>> t.members
        {'BC_new': ['B', 'C'], 'BD_new': ['B', 'D']}
        """
        for i in args:
            label = i[0]
            new_label = i[1]
            if label not in self._members:
                raise ValueError("No such member exists for the Truss")
            else:
                members_duplicate = list(self._members).copy()
                for member in members_duplicate:
                    if member == label:
                        self._members[new_label] = [self._members[member][0], self._members[member][1]]
                        self._members.pop(label)
                        self._member_lengths[new_label] = self._member_lengths[label]
                        self._member_lengths.pop(label)
                        self._internal_forces[new_label] = self._internal_forces[label]
                        self._internal_forces.pop(label)

    def apply_load(self, *args):
        """
        This method applies external load(s) at the specified node(s).

        Parameters
        ==========
        The input(s) of the method are tuple(s) of the form (location, magnitude, direction).

        location: String or Symbol
            Label of the Node at which load is applied.

        magnitude: Sympifyable
            Magnitude of the load applied. It must always be positive and any changes in
            the direction of the load are not reflected here.

        direction: Sympifyable
            The angle, in degrees, that the load vector makes with the horizontal
            in the counter-clockwise direction. It takes the values 0 to 360,
            inclusive.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> from sympy import symbols
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0))
        >>> P = symbols('P')
        >>> t.apply_load(('A', P, 90), ('A', P/2, 45), ('A', P/4, 90))
        >>> t.loads
        {'A': [[P, 90], [P/2, 45], [P/4, 90]]}
        """
        for i in args:
            location = i[0]
            magnitude = i[1]
            direction = i[2]
            magnitude = sympify(magnitude)
            direction = sympify(direction)

            if location not in self._node_coordinates:
                raise ValueError("Load must be applied at a known node")

            else:
                if location in self._loads:
                    self._loads[location].append([magnitude, direction])
                else:
                    self._loads[location] = [[magnitude, direction]]

    def remove_load(self, *args):
        """
        This method removes already
        present external load(s) at specified node(s).

        Parameters
        ==========
        The input(s) of this method are tuple(s) of the form (location, magnitude, direction).

        location: String or Symbol
            Label of the Node at which load is applied and is to be removed.

        magnitude: Sympifyable
            Magnitude of the load applied.

        direction: Sympifyable
            The angle, in degrees, that the load vector makes with the horizontal
            in the counter-clockwise direction. It takes the values 0 to 360,
            inclusive.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> from sympy import symbols
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0))
        >>> P = symbols('P')
        >>> t.apply_load(('A', P, 90), ('A', P/2, 45), ('A', P/4, 90))
        >>> t.loads
        {'A': [[P, 90], [P/2, 45], [P/4, 90]]}
        >>> t.remove_load(('A', P/4, 90), ('A', P/2, 45))
        >>> t.loads
        {'A': [[P, 90]]}
        """
        for i in args:
            location = i[0]
            magnitude = i[1]
            direction = i[2]
            magnitude = sympify(magnitude)
            direction = sympify(direction)

            if location not in self._node_coordinates:
                raise ValueError("Load must be removed from a known node")

            else:
                if [magnitude, direction] not in self._loads[location]:
                    raise ValueError("No load of this magnitude and direction has been applied at this node")
                else:
                    self._loads[location].remove([magnitude, direction])
            if self._loads[location] == []:
                self._loads.pop(location)

    def apply_support(self, *args):
        """
        This method adds a pinned or roller support at specified node(s).

        Parameters
        ==========
        The input(s) of this method are of the form (location, type).

        location: String or Symbol
            Label of the Node at which support is added.

        type: String
            Type of the support being provided at the node.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0))
        >>> t.apply_support(('A', 'pinned'), ('B', 'roller'))
        >>> t.supports
        {'A': 'pinned', 'B': 'roller'}
        """
        for i in args:
            location = i[0]
            type = i[1]
            if location not in self._node_coordinates:
                raise ValueError("Support must be added on a known node")

            else:
                if location not in self._supports:
                    if type == 'pinned':
                        self.apply_load((location, Symbol('R_'+str(location)+'_x'), 0))
                        self.apply_load((location, Symbol('R_'+str(location)+'_y'), 90))
                    elif type == 'roller':
                        self.apply_load((location, Symbol('R_'+str(location)+'_y'), 90))
                elif self._supports[location] == 'pinned':
                    if type == 'roller':
                        self.remove_load((location, Symbol('R_'+str(location)+'_x'), 0))
                elif self._supports[location] == 'roller':
                    if type == 'pinned':
                        self.apply_load((location, Symbol('R_'+str(location)+'_x'), 0))
                self._supports[location] = type

    def remove_support(self, *args):
        """
        This method removes support from specified node(s.)

        Parameters
        ==========

        locations: String or Symbol
            Label of the Node(s) at which support is to be removed.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0))
        >>> t.apply_support(('A', 'pinned'), ('B', 'roller'))
        >>> t.supports
        {'A': 'pinned', 'B': 'roller'}
        >>> t.remove_support('A','B')
        >>> t.supports
        {}
        """
        for location in args:

            if location not in self._node_coordinates:
                raise ValueError("No such node exists in the Truss")

            elif location not in self._supports:
                raise ValueError("No support has been added to the given node")

            else:
                if self._supports[location] == 'pinned':
                    self.remove_load((location, Symbol('R_'+str(location)+'_x'), 0))
                    self.remove_load((location, Symbol('R_'+str(location)+'_y'), 90))
                elif self._supports[location] == 'roller':
                    self.remove_load((location, Symbol('R_'+str(location)+'_y'), 90))
                self._supports.pop(location)

    def solve(self):
        """
        This method solves for all reaction forces of all supports and all internal forces
        of all the members in the truss, provided the Truss is solvable.

        A Truss is solvable if the following condition is met,

        2n >= r + m

        Where n is the number of nodes, r is the number of reaction forces, where each pinned
        support has 2 reaction forces and each roller has 1, and m is the number of members.

        The given condition is derived from the fact that a system of equations is solvable
        only when the number of variables is lesser than or equal to the number of equations.
        Equilibrium Equations in x and y directions give two equations per node giving 2n number
        equations. However, the truss needs to be stable as well and may be unstable if 2n > r + m.
        The number of variables is simply the sum of the number of reaction forces and member
        forces.

        .. note::
           The sign convention for the internal forces present in a member revolves around whether each
           force is compressive or tensile. While forming equations for each node, internal force due
           to a member on the node is assumed to be away from the node i.e. each force is assumed to
           be compressive by default. Hence, a positive value for an internal force implies the
           presence of compressive force in the member and a negative value implies a tensile force.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(("node_1", 0, 0), ("node_2", 6, 0), ("node_3", 2, 2), ("node_4", 2, 0))
        >>> t.add_member(("member_1", "node_1", "node_4"), ("member_2", "node_2", "node_4"), ("member_3", "node_1", "node_3"))
        >>> t.add_member(("member_4", "node_2", "node_3"), ("member_5", "node_3", "node_4"))
        >>> t.apply_load(("node_4", 10, 270))
        >>> t.apply_support(("node_1", "pinned"), ("node_2", "roller"))
        >>> t.solve()
        >>> t.reaction_loads
        {'R_node_1_x': 0, 'R_node_1_y': 20/3, 'R_node_2_y': 10/3}
        >>> t.internal_forces
        {'member_1': 20/3, 'member_2': 20/3, 'member_3': -20*sqrt(2)/3, 'member_4': -10*sqrt(5)/3, 'member_5': 10}
        """
        count_reaction_loads = 0
        for node in self._nodes:
            if node[0] in self._supports:
                if self._supports[node[0]]=='pinned':
                    count_reaction_loads += 2
                elif self._supports[node[0]]=='roller':
                    count_reaction_loads += 1
        if 2*len(self._nodes) != len(self._members) + count_reaction_loads:
            raise ValueError("The given truss cannot be solved")
        coefficients_matrix = [[0 for i in range(2*len(self._nodes))] for j in range(2*len(self._nodes))]
        load_matrix = zeros(2*len(self.nodes), 1)
        load_matrix_row = 0
        for node in self._nodes:
            if node[0] in self._loads:
                for load in self._loads[node[0]]:
                    if load[0]!=Symbol('R_'+str(node[0])+'_x') and load[0]!=Symbol('R_'+str(node[0])+'_y'):
                        load_matrix[load_matrix_row] -= load[0]*cos(pi*load[1]/180)
                        load_matrix[load_matrix_row + 1] -= load[0]*sin(pi*load[1]/180)
            load_matrix_row += 2
        cols = 0
        row = 0
        for node in self._nodes:
            if node[0] in self._supports:
                if self._supports[node[0]]=='pinned':
                    coefficients_matrix[row][cols] += 1
                    coefficients_matrix[row+1][cols+1] += 1
                    cols += 2
                elif self._supports[node[0]]=='roller':
                    coefficients_matrix[row+1][cols] += 1
                    cols += 1
            row += 2
        for member in self._members:
            start = self._members[member][0]
            end = self._members[member][1]
            length = sqrt((self._node_coordinates[start][0]-self._node_coordinates[end][0])**2 + (self._node_coordinates[start][1]-self._node_coordinates[end][1])**2)
            start_index = self._node_labels.index(start)
            end_index = self._node_labels.index(end)
            horizontal_component_start = (self._node_coordinates[end][0]-self._node_coordinates[start][0])/length
            vertical_component_start = (self._node_coordinates[end][1]-self._node_coordinates[start][1])/length
            horizontal_component_end = (self._node_coordinates[start][0]-self._node_coordinates[end][0])/length
            vertical_component_end = (self._node_coordinates[start][1]-self._node_coordinates[end][1])/length
            coefficients_matrix[start_index*2][cols] += horizontal_component_start
            coefficients_matrix[start_index*2+1][cols] += vertical_component_start
            coefficients_matrix[end_index*2][cols] += horizontal_component_end
            coefficients_matrix[end_index*2+1][cols] += vertical_component_end
            cols += 1
        forces_matrix = (Matrix(coefficients_matrix)**-1)*load_matrix
        self._reaction_loads = {}
        i = 0
        min_load = inf
        for node in self._nodes:
            if node[0] in self._loads:
                for load in self._loads[node[0]]:
                    if type(load[0]) not in [Symbol, Mul, Add]:
                        min_load = min(min_load, load[0])
        for j in range(len(forces_matrix)):
            if type(forces_matrix[j]) not in [Symbol, Mul, Add]:
                if abs(forces_matrix[j]/min_load) <1E-10:
                    forces_matrix[j] = 0
        for node in self._nodes:
            if node[0] in self._supports:
                if self._supports[node[0]]=='pinned':
                    self._reaction_loads['R_'+str(node[0])+'_x'] = forces_matrix[i]
                    self._reaction_loads['R_'+str(node[0])+'_y'] = forces_matrix[i+1]
                    i += 2
                elif self._supports[node[0]]=='roller':
                    self._reaction_loads['R_'+str(node[0])+'_y'] = forces_matrix[i]
                    i += 1
        for member in self._members:
            self._internal_forces[member] = forces_matrix[i]
            i += 1
        return

    @doctest_depends_on(modules=('numpy',))
    def draw(self, subs_dict=None):
        """
        Returns a plot object of the Truss with all its nodes, members,
        supports and loads.

        .. note::
            The user must be careful while entering load values in their
            directions. The draw function assumes a sign convention that
            is used for plotting loads.

            Given a right-handed coordinate system with XYZ coordinates,
            the supports are assumed to be such that the reaction forces of a
            pinned support is in the +X and +Y direction while those of a
            roller support is in the +Y direction. For the load, the range
            of angles, one can input goes all the way to 360 degrees which, in the
            the plot is the angle that the load vector makes with the positive x-axis in the anticlockwise direction.

            For example, for a 90-degree angle, the load will be a vertically
            directed along +Y while a 270-degree angle denotes a vertical
            load as well but along -Y.

        Examples
        ========

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.truss import Truss
            >>> import math
            >>> t = Truss()
            >>> t.add_node(("A", -4, 0), ("B", 0, 0), ("C", 4, 0), ("D", 8, 0))
            >>> t.add_node(("E", 6, 2/math.sqrt(3)))
            >>> t.add_node(("F", 2, 2*math.sqrt(3)))
            >>> t.add_node(("G", -2, 2/math.sqrt(3)))
            >>> t.add_member(("AB","A","B"), ("BC","B","C"), ("CD","C","D"))
            >>> t.add_member(("AG","A","G"), ("GB","G","B"), ("GF","G","F"))
            >>> t.add_member(("BF","B","F"), ("FC","F","C"), ("CE","C","E"))
            >>> t.add_member(("FE","F","E"), ("DE","D","E"))
            >>> t.apply_support(("A","pinned"), ("D","roller"))
            >>> t.apply_load(("G", 3, 90), ("E", 3, 90), ("F", 2, 90))
            >>> p = t.draw()
            >>> p  # doctest: +ELLIPSIS
            Plot object containing:
            [0]: cartesian line: 1 for x over (1.0, 1.0)
            ...
            >>> p.show()
        """
        if not numpy:
            raise ImportError("To use this function numpy module is required")

        x = Symbol('x')

        markers = []
        annotations = []
        rectangles = []

        node_markers = self._draw_nodes(subs_dict)
        markers += node_markers

        member_rectangles = self._draw_members()
        rectangles += member_rectangles

        support_markers = self._draw_supports()
        markers += support_markers

        load_annotations = self._draw_loads()
        annotations += load_annotations

        xmax = -INF
        xmin = INF
        ymax = -INF
        ymin = INF

        for node in self._node_coordinates:
            xmax = max(xmax, self._node_coordinates[node][0])
            xmin = min(xmin, self._node_coordinates[node][0])
            ymax = max(ymax, self._node_coordinates[node][1])
            ymin = min(ymin, self._node_coordinates[node][1])

        lim = max(xmax*1.1-xmin*0.8+1, ymax*1.1-ymin*0.8+1)

        if lim==xmax*1.1-xmin*0.8+1:
            sing_plot = plot(1, (x, 1, 1), markers=markers, show=False, annotations=annotations, xlim=(xmin-0.05*lim, xmax*1.1), ylim=(xmin-0.05*lim, xmax*1.1), axis=False, rectangles=rectangles)

        else:
            sing_plot = plot(1, (x, 1, 1), markers=markers, show=False, annotations=annotations, xlim=(ymin-0.05*lim, ymax*1.1), ylim=(ymin-0.05*lim, ymax*1.1), axis=False, rectangles=rectangles)

        return sing_plot


    def _draw_nodes(self, subs_dict):
        node_markers = []

        for node in self._node_coordinates:
            if (type(self._node_coordinates[node][0]) in (Symbol, Quantity)):
                if self._node_coordinates[node][0] in subs_dict:
                    self._node_coordinates[node][0] = subs_dict[self._node_coordinates[node][0]]
                else:
                    raise ValueError("provided substituted dictionary is not adequate")
            elif (type(self._node_coordinates[node][0]) == Mul):
                objects = self._node_coordinates[node][0].as_coeff_Mul()
                for object in objects:
                    if type(object) in (Symbol, Quantity):
                        if subs_dict==None or object not in subs_dict:
                            raise ValueError("provided substituted dictionary is not adequate")
                        else:
                            self._node_coordinates[node][0] /= object
                            self._node_coordinates[node][0] *= subs_dict[object]

            if (type(self._node_coordinates[node][1]) in (Symbol, Quantity)):
                if self._node_coordinates[node][1] in subs_dict:
                    self._node_coordinates[node][1] = subs_dict[self._node_coordinates[node][1]]
                else:
                    raise ValueError("provided substituted dictionary is not adequate")
            elif (type(self._node_coordinates[node][1]) == Mul):
                objects = self._node_coordinates[node][1].as_coeff_Mul()
                for object in objects:
                    if type(object) in (Symbol, Quantity):
                        if subs_dict==None or object not in subs_dict:
                            raise ValueError("provided substituted dictionary is not adequate")
                        else:
                            self._node_coordinates[node][1] /= object
                            self._node_coordinates[node][1] *= subs_dict[object]

        for node in self._node_coordinates:
            node_markers.append(
                {
                    'args':[[self._node_coordinates[node][0]], [self._node_coordinates[node][1]]],
                    'marker':'o',
                    'markersize':5,
                    'color':'black'
                }
            )
        return node_markers

    def _draw_members(self):

        member_rectangles = []

        xmax = -INF
        xmin = INF
        ymax = -INF
        ymin = INF

        for node in self._node_coordinates:
            xmax = max(xmax, self._node_coordinates[node][0])
            xmin = min(xmin, self._node_coordinates[node][0])
            ymax = max(ymax, self._node_coordinates[node][1])
            ymin = min(ymin, self._node_coordinates[node][1])

        if abs(1.1*xmax-0.8*xmin)>abs(1.1*ymax-0.8*ymin):
            max_diff = 1.1*xmax-0.8*xmin
        else:
            max_diff = 1.1*ymax-0.8*ymin

        for member in self._members:
            x1 = self._node_coordinates[self._members[member][0]][0]
            y1 = self._node_coordinates[self._members[member][0]][1]
            x2 = self._node_coordinates[self._members[member][1]][0]
            y2 = self._node_coordinates[self._members[member][1]][1]
            if x2!=x1 and y2!=y1:
                if x2>x1:
                    member_rectangles.append(
                        {
                            'xy':(x1-0.005*max_diff*cos(pi/4+atan((y2-y1)/(x2-x1)))/2, y1-0.005*max_diff*sin(pi/4+atan((y2-y1)/(x2-x1)))/2),
                            'width':sqrt((x1-x2)**2+(y1-y2)**2)+0.005*max_diff/math.sqrt(2),
                            'height':0.005*max_diff,
                            'angle':180*atan((y2-y1)/(x2-x1))/pi,
                            'color':'brown'
                        }
                    )
                else:
                    member_rectangles.append(
                        {
                            'xy':(x2-0.005*max_diff*cos(pi/4+atan((y2-y1)/(x2-x1)))/2, y2-0.005*max_diff*sin(pi/4+atan((y2-y1)/(x2-x1)))/2),
                            'width':sqrt((x1-x2)**2+(y1-y2)**2)+0.005*max_diff/math.sqrt(2),
                            'height':0.005*max_diff,
                            'angle':180*atan((y2-y1)/(x2-x1))/pi,
                            'color':'brown'
                        }
                    )
            elif y2==y1:
                if x2>x1:
                    member_rectangles.append(
                        {
                            'xy':(x1-0.005*max_diff/2, y1-0.005*max_diff/2),
                            'width':sqrt((x1-x2)**2+(y1-y2)**2),
                            'height':0.005*max_diff,
                            'angle':90*(1-math.copysign(1, x2-x1)),
                            'color':'brown'
                        }
                    )
                else:
                    member_rectangles.append(
                        {
                            'xy':(x1-0.005*max_diff/2, y1-0.005*max_diff/2),
                            'width':sqrt((x1-x2)**2+(y1-y2)**2),
                            'height':-0.005*max_diff,
                            'angle':90*(1-math.copysign(1, x2-x1)),
                            'color':'brown'
                        }
                    )
            else:
                if y1<y2:
                    member_rectangles.append(
                        {
                            'xy':(x1-0.005*max_diff/2, y1-0.005*max_diff/2),
                            'width':sqrt((x1-x2)**2+(y1-y2)**2)+0.005*max_diff/2,
                            'height':0.005*max_diff,
                            'angle':90*math.copysign(1, y2-y1),
                            'color':'brown'
                        }
                    )
                else:
                    member_rectangles.append(
                        {
                            'xy':(x2-0.005*max_diff/2, y2-0.005*max_diff/2),
                            'width':-(sqrt((x1-x2)**2+(y1-y2)**2)+0.005*max_diff/2),
                            'height':0.005*max_diff,
                            'angle':90*math.copysign(1, y2-y1),
                            'color':'brown'
                        }
                    )

        return member_rectangles

    def _draw_supports(self):
        support_markers = []

        xmax = -INF
        xmin = INF
        ymax = -INF
        ymin = INF

        for node in self._node_coordinates:
            xmax = max(xmax, self._node_coordinates[node][0])
            xmin = min(xmin, self._node_coordinates[node][0])
            ymax = max(ymax, self._node_coordinates[node][1])
            ymin = min(ymin, self._node_coordinates[node][1])
        if abs(1.1*xmax-0.8*xmin)>abs(1.1*ymax-0.8*ymin):
            max_diff = 1.1*xmax-0.8*xmin
        else:
            max_diff = 1.1*ymax-0.8*ymin

        for node in self._supports:
            if self._supports[node]=='pinned':
                support_markers.append(
                    {
                        'args':[
                            [self._node_coordinates[node][0]],
                            [self._node_coordinates[node][1]]
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
                            [self._node_coordinates[node][0]],
                            [self._node_coordinates[node][1]-0.035*max_diff]
                        ],
                        'marker':'_',
                        'markersize':14,
                        'color':'black'
                    }
                )

            elif self._supports[node]=='roller':
                support_markers.append(
                    {
                        'args':[
                            [self._node_coordinates[node][0]],
                            [self._node_coordinates[node][1]-0.02*max_diff]
                        ],
                        'marker':'o',
                        'markersize':11,
                        'color':'black',
                        'markerfacecolor':'none'
                    }
                )
                support_markers.append(
                    {
                        'args':[
                            [self._node_coordinates[node][0]],
                            [self._node_coordinates[node][1]-0.0375*max_diff]
                        ],
                        'marker':'_',
                        'markersize':14,
                        'color':'black'
                    }
                )
        return support_markers

    def _draw_loads(self):
        load_annotations = []

        xmax = -INF
        xmin = INF
        ymax = -INF
        ymin = INF

        for node in self._node_coordinates:
            xmax = max(xmax, self._node_coordinates[node][0])
            xmin = min(xmin, self._node_coordinates[node][0])
            ymax = max(ymax, self._node_coordinates[node][1])
            ymin = min(ymin, self._node_coordinates[node][1])

        if abs(1.1*xmax-0.8*xmin)>abs(1.1*ymax-0.8*ymin):
            max_diff = 1.1*xmax-0.8*xmin+5
        else:
            max_diff = 1.1*ymax-0.8*ymin+5

        for node in self._loads:
            for load in self._loads[node]:
                if load[0] in [Symbol('R_'+str(node)+'_x'), Symbol('R_'+str(node)+'_y')]:
                    continue
                x = self._node_coordinates[node][0]
                y = self._node_coordinates[node][1]
                load_annotations.append(
                    {
                        'text':'',
                        'xy':(
                            x-math.cos(pi*load[1]/180)*(max_diff/100),
                            y-math.sin(pi*load[1]/180)*(max_diff/100)
                        ),
                        'xytext':(
                            x-(max_diff/100+abs(xmax-xmin)+abs(ymax-ymin))*math.cos(pi*load[1]/180)/20,
                            y-(max_diff/100+abs(xmax-xmin)+abs(ymax-ymin))*math.sin(pi*load[1]/180)/20
                        ),
                        'arrowprops':{'width':1.5, 'headlength':5, 'headwidth':5, 'facecolor':'black'}
                    }
                )
        return load_annotations
