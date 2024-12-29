from sympy import (diff, expand, sin, cos, sympify, eye, zeros,
                                ImmutableMatrix as Matrix, MatrixBase)
from sympy.core.symbol import Symbol
from sympy.simplify.trigsimp import trigsimp
from sympy.physics.vector.vector import Vector, _check_vector
from sympy.utilities.misc import translate

from warnings import warn

__all__ = ['CoordinateSym', 'ReferenceFrame']


class CoordinateSym(Symbol):
    """
    A coordinate symbol/base scalar associated wrt a Reference Frame.

    Ideally, users should not instantiate this class. Instances of
    this class must only be accessed through the corresponding frame
    as 'frame[index]'.

    CoordinateSyms having the same frame and index parameters are equal
    (even though they may be instantiated separately).

    Parameters
    ==========

    name : string
        The display name of the CoordinateSym

    frame : ReferenceFrame
        The reference frame this base scalar belongs to

    index : 0, 1 or 2
        The index of the dimension denoted by this coordinate variable

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame, CoordinateSym
    >>> A = ReferenceFrame('A')
    >>> A[1]
    A_y
    >>> type(A[0])
    <class 'sympy.physics.vector.frame.CoordinateSym'>
    >>> a_y = CoordinateSym('a_y', A, 1)
    >>> a_y == A[1]
    True

    """

    def __new__(cls, name, frame, index):
        # We can't use the cached Symbol.__new__ because this class depends on
        # frame and index, which are not passed to Symbol.__xnew__.
        assumptions = {}
        super()._sanitize(assumptions, cls)
        obj = super().__xnew__(cls, name, **assumptions)
        _check_frame(frame)
        if index not in range(0, 3):
            raise ValueError("Invalid index specified")
        obj._id = (frame, index)
        return obj

    def __getnewargs_ex__(self):
        return (self.name, *self._id), {}

    @property
    def frame(self):
        return self._id[0]

    def __eq__(self, other):
        # Check if the other object is a CoordinateSym of the same frame and
        # same index
        if isinstance(other, CoordinateSym):
            if other._id == self._id:
                return True
        return False

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return (self._id[0].__hash__(), self._id[1]).__hash__()


class ReferenceFrame:
    """A reference frame in classical mechanics.

    ReferenceFrame is a class used to represent a reference frame in classical
    mechanics. It has a standard basis of three unit vectors in the frame's
    x, y, and z directions.

    It also can have a rotation relative to a parent frame; this rotation is
    defined by a direction cosine matrix relating this frame's basis vectors to
    the parent frame's basis vectors.  It can also have an angular velocity
    vector, defined in another frame.

    """
    _count = 0

    def __init__(self, name, indices=None, latexs=None, variables=None):
        """ReferenceFrame initialization method.

        A ReferenceFrame has a set of orthonormal basis vectors, along with
        orientations relative to other ReferenceFrames and angular velocities
        relative to other ReferenceFrames.

        Parameters
        ==========

        indices : tuple of str
            Enables the reference frame's basis unit vectors to be accessed by
            Python's square bracket indexing notation using the provided three
            indice strings and alters the printing of the unit vectors to
            reflect this choice.
        latexs : tuple of str
            Alters the LaTeX printing of the reference frame's basis unit
            vectors to the provided three valid LaTeX strings.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, vlatex
        >>> N = ReferenceFrame('N')
        >>> N.x
        N.x
        >>> O = ReferenceFrame('O', indices=('1', '2', '3'))
        >>> O.x
        O['1']
        >>> O['1']
        O['1']
        >>> P = ReferenceFrame('P', latexs=('A1', 'A2', 'A3'))
        >>> vlatex(P.x)
        'A1'

        ``symbols()`` can be used to create multiple Reference Frames in one
        step, for example:

        >>> from sympy.physics.vector import ReferenceFrame
        >>> from sympy import symbols
        >>> A, B, C = symbols('A B C', cls=ReferenceFrame)
        >>> D, E = symbols('D E', cls=ReferenceFrame, indices=('1', '2', '3'))
        >>> A[0]
        A_x
        >>> D.x
        D['1']
        >>> E.y
        E['2']
        >>> type(A) == type(D)
        True

        Unit dyads for the ReferenceFrame can be accessed through the attributes ``xx``, ``xy``, etc. For example:

        >>> from sympy.physics.vector import ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> N.yz
        (N.y|N.z)
        >>> N.zx
        (N.z|N.x)
        >>> P = ReferenceFrame('P', indices=['1', '2', '3'])
        >>> P.xx
        (P['1']|P['1'])
        >>> P.zy
        (P['3']|P['2'])

        Unit dyadic is also accessible via the ``u`` attribute:

        >>> from sympy.physics.vector import ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> N.u
        (N.x|N.x) + (N.y|N.y) + (N.z|N.z)
        >>> P = ReferenceFrame('P', indices=['1', '2', '3'])
        >>> P.u
        (P['1']|P['1']) + (P['2']|P['2']) + (P['3']|P['3'])

        """

        if not isinstance(name, str):
            raise TypeError('Need to supply a valid name')
        # The if statements below are for custom printing of basis-vectors for
        # each frame.
        # First case, when custom indices are supplied
        if indices is not None:
            if not isinstance(indices, (tuple, list)):
                raise TypeError('Supply the indices as a list')
            if len(indices) != 3:
                raise ValueError('Supply 3 indices')
            for i in indices:
                if not isinstance(i, str):
                    raise TypeError('Indices must be strings')
            self.str_vecs = [(name + '[\'' + indices[0] + '\']'),
                             (name + '[\'' + indices[1] + '\']'),
                             (name + '[\'' + indices[2] + '\']')]
            self.pretty_vecs = [(name.lower() + "_" + indices[0]),
                                (name.lower() + "_" + indices[1]),
                                (name.lower() + "_" + indices[2])]
            self.latex_vecs = [(r"\mathbf{\hat{%s}_{%s}}" % (name.lower(),
                                                             indices[0])),
                               (r"\mathbf{\hat{%s}_{%s}}" % (name.lower(),
                                                             indices[1])),
                               (r"\mathbf{\hat{%s}_{%s}}" % (name.lower(),
                                                             indices[2]))]
            self.indices = indices
        # Second case, when no custom indices are supplied
        else:
            self.str_vecs = [(name + '.x'), (name + '.y'), (name + '.z')]
            self.pretty_vecs = [name.lower() + "_x",
                                name.lower() + "_y",
                                name.lower() + "_z"]
            self.latex_vecs = [(r"\mathbf{\hat{%s}_x}" % name.lower()),
                               (r"\mathbf{\hat{%s}_y}" % name.lower()),
                               (r"\mathbf{\hat{%s}_z}" % name.lower())]
            self.indices = ['x', 'y', 'z']
        # Different step, for custom latex basis vectors
        if latexs is not None:
            if not isinstance(latexs, (tuple, list)):
                raise TypeError('Supply the indices as a list')
            if len(latexs) != 3:
                raise ValueError('Supply 3 indices')
            for i in latexs:
                if not isinstance(i, str):
                    raise TypeError('Latex entries must be strings')
            self.latex_vecs = latexs
        self.name = name
        self._var_dict = {}
        # The _dcm_dict dictionary will only store the dcms of adjacent
        # parent-child relationships. The _dcm_cache dictionary will store
        # calculated dcm along with all content of _dcm_dict for faster
        # retrieval of dcms.
        self._dcm_dict = {}
        self._dcm_cache = {}
        self._ang_vel_dict = {}
        self._ang_acc_dict = {}
        self._dlist = [self._dcm_dict, self._ang_vel_dict, self._ang_acc_dict]
        self._cur = 0
        self._x = Vector([(Matrix([1, 0, 0]), self)])
        self._y = Vector([(Matrix([0, 1, 0]), self)])
        self._z = Vector([(Matrix([0, 0, 1]), self)])
        # Associate coordinate symbols wrt this frame
        if variables is not None:
            if not isinstance(variables, (tuple, list)):
                raise TypeError('Supply the variable names as a list/tuple')
            if len(variables) != 3:
                raise ValueError('Supply 3 variable names')
            for i in variables:
                if not isinstance(i, str):
                    raise TypeError('Variable names must be strings')
        else:
            variables = [name + '_x', name + '_y', name + '_z']
        self.varlist = (CoordinateSym(variables[0], self, 0),
                        CoordinateSym(variables[1], self, 1),
                        CoordinateSym(variables[2], self, 2))
        ReferenceFrame._count += 1
        self.index = ReferenceFrame._count

    def __getitem__(self, ind):
        """
        Returns basis vector for the provided index, if the index is a string.

        If the index is a number, returns the coordinate variable correspon-
        -ding to that index.
        """
        if not isinstance(ind, str):
            if ind < 3:
                return self.varlist[ind]
            else:
                raise ValueError("Invalid index provided")
        if self.indices[0] == ind:
            return self.x
        if self.indices[1] == ind:
            return self.y
        if self.indices[2] == ind:
            return self.z
        else:
            raise ValueError('Not a defined index')

    def __iter__(self):
        return iter([self.x, self.y, self.z])

    def __str__(self):
        """Returns the name of the frame. """
        return self.name

    __repr__ = __str__

    def _dict_list(self, other, num):
        """Returns an inclusive list of reference frames that connect this
        reference frame to the provided reference frame.

        Parameters
        ==========
        other : ReferenceFrame
            The other reference frame to look for a connecting relationship to.
        num : integer
            ``0``, ``1``, and ``2`` will look for orientation, angular
            velocity, and angular acceleration relationships between the two
            frames, respectively.

        Returns
        =======
        list
            Inclusive list of reference frames that connect this reference
            frame to the other reference frame.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> A = ReferenceFrame('A')
        >>> B = ReferenceFrame('B')
        >>> C = ReferenceFrame('C')
        >>> D = ReferenceFrame('D')
        >>> B.orient_axis(A, A.x, 1.0)
        >>> C.orient_axis(B, B.x, 1.0)
        >>> D.orient_axis(C, C.x, 1.0)
        >>> D._dict_list(A, 0)
        [D, C, B, A]

        Raises
        ======

        ValueError
            When no path is found between the two reference frames or ``num``
            is an incorrect value.

        """

        connect_type = {0: 'orientation',
                        1: 'angular velocity',
                        2: 'angular acceleration'}

        if num not in connect_type.keys():
            raise ValueError('Valid values for num are 0, 1, or 2.')

        possible_connecting_paths = [[self]]
        oldlist = [[]]
        while possible_connecting_paths != oldlist:
            oldlist = possible_connecting_paths[:]  # make a copy
            for frame_list in possible_connecting_paths:
                frames_adjacent_to_last = frame_list[-1]._dlist[num].keys()
                for adjacent_frame in frames_adjacent_to_last:
                    if adjacent_frame not in frame_list:
                        connecting_path = frame_list + [adjacent_frame]
                        if connecting_path not in possible_connecting_paths:
                            possible_connecting_paths.append(connecting_path)

        for connecting_path in oldlist:
            if connecting_path[-1] != other:
                possible_connecting_paths.remove(connecting_path)
        possible_connecting_paths.sort(key=len)

        if len(possible_connecting_paths) != 0:
            return possible_connecting_paths[0]  # selects the shortest path

        msg = 'No connecting {} path found between {} and {}.'
        raise ValueError(msg.format(connect_type[num], self.name, other.name))

    def _w_diff_dcm(self, otherframe):
        """Angular velocity from time differentiating the DCM. """
        from sympy.physics.vector.functions import dynamicsymbols
        dcm2diff = otherframe.dcm(self)
        diffed = dcm2diff.diff(dynamicsymbols._t)
        angvelmat = diffed * dcm2diff.T
        w1 = trigsimp(expand(angvelmat[7]), recursive=True)
        w2 = trigsimp(expand(angvelmat[2]), recursive=True)
        w3 = trigsimp(expand(angvelmat[3]), recursive=True)
        return Vector([(Matrix([w1, w2, w3]), otherframe)])

    def variable_map(self, otherframe):
        """
        Returns a dictionary which expresses the coordinate variables
        of this frame in terms of the variables of otherframe.

        If Vector.simp is True, returns a simplified version of the mapped
        values. Else, returns them without simplification.

        Simplification of the expressions may take time.

        Parameters
        ==========

        otherframe : ReferenceFrame
            The other frame to map the variables to

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, dynamicsymbols
        >>> A = ReferenceFrame('A')
        >>> q = dynamicsymbols('q')
        >>> B = A.orientnew('B', 'Axis', [q, A.z])
        >>> A.variable_map(B)
        {A_x: B_x*cos(q(t)) - B_y*sin(q(t)), A_y: B_x*sin(q(t)) + B_y*cos(q(t)), A_z: B_z}

        """

        _check_frame(otherframe)
        if (otherframe, Vector.simp) in self._var_dict:
            return self._var_dict[(otherframe, Vector.simp)]
        else:
            vars_matrix = self.dcm(otherframe) * Matrix(otherframe.varlist)
            mapping = {}
            for i, x in enumerate(self):
                if Vector.simp:
                    mapping[self.varlist[i]] = trigsimp(vars_matrix[i],
                                                        method='fu')
                else:
                    mapping[self.varlist[i]] = vars_matrix[i]
            self._var_dict[(otherframe, Vector.simp)] = mapping
            return mapping

    def ang_acc_in(self, otherframe):
        """Returns the angular acceleration Vector of the ReferenceFrame.

        Effectively returns the Vector:

        ``N_alpha_B``

        which represent the angular acceleration of B in N, where B is self,
        and N is otherframe.

        Parameters
        ==========

        otherframe : ReferenceFrame
            The ReferenceFrame which the angular acceleration is returned in.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> V = 10 * N.x
        >>> A.set_ang_acc(N, V)
        >>> A.ang_acc_in(N)
        10*N.x

        """

        _check_frame(otherframe)
        if otherframe in self._ang_acc_dict:
            return self._ang_acc_dict[otherframe]
        else:
            return self.ang_vel_in(otherframe).dt(otherframe)

    def ang_vel_in(self, otherframe):
        """Returns the angular velocity Vector of the ReferenceFrame.

        Effectively returns the Vector:

        ^N omega ^B

        which represent the angular velocity of B in N, where B is self, and
        N is otherframe.

        Parameters
        ==========

        otherframe : ReferenceFrame
            The ReferenceFrame which the angular velocity is returned in.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> V = 10 * N.x
        >>> A.set_ang_vel(N, V)
        >>> A.ang_vel_in(N)
        10*N.x

        """

        _check_frame(otherframe)
        flist = self._dict_list(otherframe, 1)
        outvec = Vector(0)
        for i in range(len(flist) - 1):
            outvec += flist[i]._ang_vel_dict[flist[i + 1]]
        return outvec

    def dcm(self, otherframe):
        r"""Returns the direction cosine matrix of this reference frame
        relative to the provided reference frame.

        The returned matrix can be used to express the orthogonal unit vectors
        of this frame in terms of the orthogonal unit vectors of
        ``otherframe``.

        Parameters
        ==========

        otherframe : ReferenceFrame
            The reference frame which the direction cosine matrix of this frame
            is formed relative to.

        Examples
        ========

        The following example rotates the reference frame A relative to N by a
        simple rotation and then calculates the direction cosine matrix of N
        relative to A.

        >>> from sympy import symbols, sin, cos
        >>> from sympy.physics.vector import ReferenceFrame
        >>> q1 = symbols('q1')
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> A.orient_axis(N, q1, N.x)
        >>> N.dcm(A)
        Matrix([
        [1,       0,        0],
        [0, cos(q1), -sin(q1)],
        [0, sin(q1),  cos(q1)]])

        The second row of the above direction cosine matrix represents the
        ``N.y`` unit vector in N expressed in A. Like so:

        >>> Ny = 0*A.x + cos(q1)*A.y - sin(q1)*A.z

        Thus, expressing ``N.y`` in A should return the same result:

        >>> N.y.express(A)
        cos(q1)*A.y - sin(q1)*A.z

        Notes
        =====

        It is important to know what form of the direction cosine matrix is
        returned. If ``B.dcm(A)`` is called, it means the "direction cosine
        matrix of B rotated relative to A". This is the matrix
        :math:`{}^B\mathbf{C}^A` shown in the following relationship:

        .. math::

           \begin{bmatrix}
             \hat{\mathbf{b}}_1 \\
             \hat{\mathbf{b}}_2 \\
             \hat{\mathbf{b}}_3
           \end{bmatrix}
           =
           {}^B\mathbf{C}^A
           \begin{bmatrix}
             \hat{\mathbf{a}}_1 \\
             \hat{\mathbf{a}}_2 \\
             \hat{\mathbf{a}}_3
           \end{bmatrix}.

        :math:`{}^B\mathbf{C}^A` is the matrix that expresses the B unit
        vectors in terms of the A unit vectors.

        """

        _check_frame(otherframe)
        # Check if the dcm wrt that frame has already been calculated
        if otherframe in self._dcm_cache:
            return self._dcm_cache[otherframe]
        flist = self._dict_list(otherframe, 0)
        outdcm = eye(3)
        for i in range(len(flist) - 1):
            outdcm = outdcm * flist[i]._dcm_dict[flist[i + 1]]
        # After calculation, store the dcm in dcm cache for faster future
        # retrieval
        self._dcm_cache[otherframe] = outdcm
        otherframe._dcm_cache[self] = outdcm.T
        return outdcm

    def _dcm(self, parent, parent_orient):
        # If parent.oreint(self) is already defined,then
        # update the _dcm_dict of parent while over write
        # all content of self._dcm_dict and self._dcm_cache
        # with new dcm relation.
        # Else update _dcm_cache and _dcm_dict of both
        # self and parent.
        frames = self._dcm_cache.keys()
        dcm_dict_del = []
        dcm_cache_del = []
        if parent in frames:
            for frame in frames:
                if frame in self._dcm_dict:
                    dcm_dict_del += [frame]
                dcm_cache_del += [frame]
            # Reset the _dcm_cache of this frame, and remove it from the
            # _dcm_caches of the frames it is linked to. Also remove it from
            # the _dcm_dict of its parent
            for frame in dcm_dict_del:
                del frame._dcm_dict[self]
            for frame in dcm_cache_del:
                del frame._dcm_cache[self]
        # Reset the _dcm_dict
            self._dcm_dict = self._dlist[0] = {}
        # Reset the _dcm_cache
            self._dcm_cache = {}

        else:
            # Check for loops and raise warning accordingly.
            visited = []
            queue = list(frames)
            cont = True  # Flag to control queue loop.
            while queue and cont:
                node = queue.pop(0)
                if node not in visited:
                    visited.append(node)
                    neighbors = node._dcm_dict.keys()
                    for neighbor in neighbors:
                        if neighbor == parent:
                            warn('Loops are defined among the orientation of '
                                 'frames. This is likely not desired and may '
                                 'cause errors in your calculations.')
                            cont = False
                            break
                        queue.append(neighbor)

        # Add the dcm relationship to _dcm_dict
        self._dcm_dict.update({parent: parent_orient.T})
        parent._dcm_dict.update({self: parent_orient})
        # Update the dcm cache
        self._dcm_cache.update({parent: parent_orient.T})
        parent._dcm_cache.update({self: parent_orient})

    def orient_axis(self, parent, axis, angle):
        """Sets the orientation of this reference frame with respect to a
        parent reference frame by rotating through an angle about an axis fixed
        in the parent reference frame.

        Parameters
        ==========

        parent : ReferenceFrame
            Reference frame that this reference frame will be rotated relative
            to.
        axis : Vector
            Vector fixed in the parent frame about about which this frame is
            rotated. It need not be a unit vector and the rotation follows the
            right hand rule.
        angle : sympifiable
            Angle in radians by which it the frame is to be rotated.

        Warns
        ======

        UserWarning
            If the orientation creates a kinematic loop.

        Examples
        ========

        Setup variables for the examples:

        >>> from sympy import symbols
        >>> from sympy.physics.vector import ReferenceFrame
        >>> q1 = symbols('q1')
        >>> N = ReferenceFrame('N')
        >>> B = ReferenceFrame('B')
        >>> B.orient_axis(N, N.x, q1)

        The ``orient_axis()`` method generates a direction cosine matrix and
        its transpose which defines the orientation of B relative to N and vice
        versa. Once orient is called, ``dcm()`` outputs the appropriate
        direction cosine matrix:

        >>> B.dcm(N)
        Matrix([
        [1,       0,      0],
        [0,  cos(q1), sin(q1)],
        [0, -sin(q1), cos(q1)]])
        >>> N.dcm(B)
        Matrix([
        [1,       0,        0],
        [0, cos(q1), -sin(q1)],
        [0, sin(q1),  cos(q1)]])

        The following two lines show that the sense of the rotation can be
        defined by negating the vector direction or the angle. Both lines
        produce the same result.

        >>> B.orient_axis(N, -N.x, q1)
        >>> B.orient_axis(N, N.x, -q1)

        """

        from sympy.physics.vector.functions import dynamicsymbols
        _check_frame(parent)

        if not isinstance(axis, Vector) and isinstance(angle, Vector):
            axis, angle = angle, axis

        axis = _check_vector(axis)
        theta = sympify(angle)

        if not axis.dt(parent) == 0:
            raise ValueError('Axis cannot be time-varying.')
        unit_axis = axis.express(parent).normalize()
        unit_col = unit_axis.args[0][0]
        parent_orient_axis = (
            (eye(3) - unit_col * unit_col.T) * cos(theta) +
            Matrix([[0, -unit_col[2], unit_col[1]],
                    [unit_col[2], 0, -unit_col[0]],
                    [-unit_col[1], unit_col[0], 0]]) *
            sin(theta) + unit_col * unit_col.T)

        self._dcm(parent, parent_orient_axis)

        thetad = (theta).diff(dynamicsymbols._t)
        wvec = thetad*axis.express(parent).normalize()
        self._ang_vel_dict.update({parent: wvec})
        parent._ang_vel_dict.update({self: -wvec})
        self._var_dict = {}

    def orient_explicit(self, parent, dcm):
        """Sets the orientation of this reference frame relative to another (parent) reference frame
        using a direction cosine matrix that describes the rotation from the parent to the child.

        Parameters
        ==========

        parent : ReferenceFrame
            Reference frame that this reference frame will be rotated relative
            to.
        dcm : Matrix, shape(3, 3)
            Direction cosine matrix that specifies the relative rotation
            between the two reference frames.

        Warns
        ======

        UserWarning
            If the orientation creates a kinematic loop.

        Examples
        ========

        Setup variables for the examples:

        >>> from sympy import symbols, Matrix, sin, cos
        >>> from sympy.physics.vector import ReferenceFrame
        >>> q1 = symbols('q1')
        >>> A = ReferenceFrame('A')
        >>> B = ReferenceFrame('B')
        >>> N = ReferenceFrame('N')

        A simple rotation of ``A`` relative to ``N`` about ``N.x`` is defined
        by the following direction cosine matrix:

        >>> dcm = Matrix([[1, 0, 0],
        ...               [0, cos(q1), -sin(q1)],
        ...               [0, sin(q1), cos(q1)]])
        >>> A.orient_explicit(N, dcm)
        >>> A.dcm(N)
        Matrix([
        [1,       0,      0],
        [0,  cos(q1), sin(q1)],
        [0, -sin(q1), cos(q1)]])

        This is equivalent to using ``orient_axis()``:

        >>> B.orient_axis(N, N.x, q1)
        >>> B.dcm(N)
        Matrix([
        [1,       0,      0],
        [0,  cos(q1), sin(q1)],
        [0, -sin(q1), cos(q1)]])

        **Note carefully that** ``N.dcm(B)`` **(the transpose) would be passed
        into** ``orient_explicit()`` **for** ``A.dcm(N)`` **to match**
        ``B.dcm(N)``:

        >>> A.orient_explicit(N, N.dcm(B))
        >>> A.dcm(N)
        Matrix([
        [1,       0,      0],
        [0,  cos(q1), sin(q1)],
        [0, -sin(q1), cos(q1)]])

        """
        _check_frame(parent)
        # amounts must be a Matrix type object
        # (e.g. sympy.matrices.dense.MutableDenseMatrix).
        if not isinstance(dcm, MatrixBase):
            raise TypeError("Amounts must be a SymPy Matrix type object.")

        self.orient_dcm(parent, dcm.T)

    def orient_dcm(self, parent, dcm):
        """Sets the orientation of this reference frame relative to another (parent) reference frame
        using a direction cosine matrix that describes the rotation from the child to the parent.

        Parameters
        ==========

        parent : ReferenceFrame
            Reference frame that this reference frame will be rotated relative
            to.
        dcm : Matrix, shape(3, 3)
            Direction cosine matrix that specifies the relative rotation
            between the two reference frames.

        Warns
        ======

        UserWarning
            If the orientation creates a kinematic loop.

        Examples
        ========

        Setup variables for the examples:

        >>> from sympy import symbols, Matrix, sin, cos
        >>> from sympy.physics.vector import ReferenceFrame
        >>> q1 = symbols('q1')
        >>> A = ReferenceFrame('A')
        >>> B = ReferenceFrame('B')
        >>> N = ReferenceFrame('N')

        A simple rotation of ``A`` relative to ``N`` about ``N.x`` is defined
        by the following direction cosine matrix:

        >>> dcm = Matrix([[1, 0, 0],
        ...               [0,  cos(q1), sin(q1)],
        ...               [0, -sin(q1), cos(q1)]])
        >>> A.orient_dcm(N, dcm)
        >>> A.dcm(N)
        Matrix([
        [1,       0,      0],
        [0,  cos(q1), sin(q1)],
        [0, -sin(q1), cos(q1)]])

        This is equivalent to using ``orient_axis()``:

        >>> B.orient_axis(N, N.x, q1)
        >>> B.dcm(N)
        Matrix([
        [1,       0,      0],
        [0,  cos(q1), sin(q1)],
        [0, -sin(q1), cos(q1)]])

        """

        _check_frame(parent)
        # amounts must be a Matrix type object
        # (e.g. sympy.matrices.dense.MutableDenseMatrix).
        if not isinstance(dcm, MatrixBase):
            raise TypeError("Amounts must be a SymPy Matrix type object.")

        self._dcm(parent, dcm.T)

        wvec = self._w_diff_dcm(parent)
        self._ang_vel_dict.update({parent: wvec})
        parent._ang_vel_dict.update({self: -wvec})
        self._var_dict = {}

    def _rot(self, axis, angle):
        """DCM for simple axis 1,2,or 3 rotations."""
        if axis == 1:
            return Matrix([[1, 0, 0],
                           [0, cos(angle), -sin(angle)],
                           [0, sin(angle), cos(angle)]])
        elif axis == 2:
            return Matrix([[cos(angle), 0, sin(angle)],
                           [0, 1, 0],
                           [-sin(angle), 0, cos(angle)]])
        elif axis == 3:
            return Matrix([[cos(angle), -sin(angle), 0],
                           [sin(angle), cos(angle), 0],
                           [0, 0, 1]])

    def _parse_consecutive_rotations(self, angles, rotation_order):
        """Helper for orient_body_fixed and orient_space_fixed.

        Parameters
        ==========
        angles : 3-tuple of sympifiable
            Three angles in radians used for the successive rotations.
        rotation_order : 3 character string or 3 digit integer
            Order of the rotations. The order can be specified by the strings
            ``'XZX'``, ``'131'``, or the integer ``131``. There are 12 unique
            valid rotation orders.

        Returns
        =======

        amounts : list
            List of sympifiables corresponding to the rotation angles.
        rot_order : list
            List of integers corresponding to the axis of rotation.
        rot_matrices : list
            List of DCM around the given axis with corresponding magnitude.

        """
        amounts = list(angles)
        for i, v in enumerate(amounts):
            if not isinstance(v, Vector):
                amounts[i] = sympify(v)

        approved_orders = ('123', '231', '312', '132', '213', '321', '121',
                           '131', '212', '232', '313', '323', '')
        # make sure XYZ => 123
        rot_order = translate(str(rotation_order), 'XYZxyz', '123123')
        if rot_order not in approved_orders:
            raise TypeError('The rotation order is not a valid order.')

        rot_order = [int(r) for r in rot_order]
        if not (len(amounts) == 3 & len(rot_order) == 3):
            raise TypeError('Body orientation takes 3 values & 3 orders')
        rot_matrices = [self._rot(order, amount)
                        for (order, amount) in zip(rot_order, amounts)]
        return amounts, rot_order, rot_matrices

    def orient_body_fixed(self, parent, angles, rotation_order):
        """Rotates this reference frame relative to the parent reference frame
        by right hand rotating through three successive body fixed simple axis
        rotations. Each subsequent axis of rotation is about the "body fixed"
        unit vectors of a new intermediate reference frame. This type of
        rotation is also referred to rotating through the `Euler and Tait-Bryan
        Angles`_.

        .. _Euler and Tait-Bryan Angles: https://en.wikipedia.org/wiki/Euler_angles

        The computed angular velocity in this method is by default expressed in
        the child's frame, so it is most preferable to use ``u1 * child.x + u2 *
        child.y + u3 * child.z`` as generalized speeds.

        Parameters
        ==========

        parent : ReferenceFrame
            Reference frame that this reference frame will be rotated relative
            to.
        angles : 3-tuple of sympifiable
            Three angles in radians used for the successive rotations.
        rotation_order : 3 character string or 3 digit integer
            Order of the rotations about each intermediate reference frames'
            unit vectors. The Euler rotation about the X, Z', X'' axes can be
            specified by the strings ``'XZX'``, ``'131'``, or the integer
            ``131``. There are 12 unique valid rotation orders (6 Euler and 6
            Tait-Bryan): zxz, xyx, yzy, zyz, xzx, yxy, xyz, yzx, zxy, xzy, zyx,
            and yxz.

        Warns
        ======

        UserWarning
            If the orientation creates a kinematic loop.

        Examples
        ========

        Setup variables for the examples:

        >>> from sympy import symbols
        >>> from sympy.physics.vector import ReferenceFrame
        >>> q1, q2, q3 = symbols('q1, q2, q3')
        >>> N = ReferenceFrame('N')
        >>> B = ReferenceFrame('B')
        >>> B1 = ReferenceFrame('B1')
        >>> B2 = ReferenceFrame('B2')
        >>> B3 = ReferenceFrame('B3')

        For example, a classic Euler Angle rotation can be done by:

        >>> B.orient_body_fixed(N, (q1, q2, q3), 'XYX')
        >>> B.dcm(N)
        Matrix([
        [        cos(q2),                            sin(q1)*sin(q2),                           -sin(q2)*cos(q1)],
        [sin(q2)*sin(q3), -sin(q1)*sin(q3)*cos(q2) + cos(q1)*cos(q3),  sin(q1)*cos(q3) + sin(q3)*cos(q1)*cos(q2)],
        [sin(q2)*cos(q3), -sin(q1)*cos(q2)*cos(q3) - sin(q3)*cos(q1), -sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3)]])

        This rotates reference frame B relative to reference frame N through
        ``q1`` about ``N.x``, then rotates B again through ``q2`` about
        ``B.y``, and finally through ``q3`` about ``B.x``. It is equivalent to
        three successive ``orient_axis()`` calls:

        >>> B1.orient_axis(N, N.x, q1)
        >>> B2.orient_axis(B1, B1.y, q2)
        >>> B3.orient_axis(B2, B2.x, q3)
        >>> B3.dcm(N)
        Matrix([
        [        cos(q2),                            sin(q1)*sin(q2),                           -sin(q2)*cos(q1)],
        [sin(q2)*sin(q3), -sin(q1)*sin(q3)*cos(q2) + cos(q1)*cos(q3),  sin(q1)*cos(q3) + sin(q3)*cos(q1)*cos(q2)],
        [sin(q2)*cos(q3), -sin(q1)*cos(q2)*cos(q3) - sin(q3)*cos(q1), -sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3)]])

        Acceptable rotation orders are of length 3, expressed in as a string
        ``'XYZ'`` or ``'123'`` or integer ``123``. Rotations about an axis
        twice in a row are prohibited.

        >>> B.orient_body_fixed(N, (q1, q2, 0), 'ZXZ')
        >>> B.orient_body_fixed(N, (q1, q2, 0), '121')
        >>> B.orient_body_fixed(N, (q1, q2, q3), 123)

        """
        from sympy.physics.vector.functions import dynamicsymbols

        _check_frame(parent)

        amounts, rot_order, rot_matrices = self._parse_consecutive_rotations(
            angles, rotation_order)
        self._dcm(parent, rot_matrices[0] * rot_matrices[1] * rot_matrices[2])

        rot_vecs = [zeros(3, 1) for _ in range(3)]
        for i, order in enumerate(rot_order):
            rot_vecs[i][order - 1] = amounts[i].diff(dynamicsymbols._t)
        u1, u2, u3 = rot_vecs[2] + rot_matrices[2].T * (
            rot_vecs[1] + rot_matrices[1].T * rot_vecs[0])
        wvec = u1 * self.x + u2 * self.y + u3 * self.z  # There is a double -
        self._ang_vel_dict.update({parent: wvec})
        parent._ang_vel_dict.update({self: -wvec})
        self._var_dict = {}

    def orient_space_fixed(self, parent, angles, rotation_order):
        """Rotates this reference frame relative to the parent reference frame
        by right hand rotating through three successive space fixed simple axis
        rotations. Each subsequent axis of rotation is about the "space fixed"
        unit vectors of the parent reference frame.

        The computed angular velocity in this method is by default expressed in
        the child's frame, so it is most preferable to use ``u1 * child.x + u2 *
        child.y + u3 * child.z`` as generalized speeds.

        Parameters
        ==========
        parent : ReferenceFrame
            Reference frame that this reference frame will be rotated relative
            to.
        angles : 3-tuple of sympifiable
            Three angles in radians used for the successive rotations.
        rotation_order : 3 character string or 3 digit integer
            Order of the rotations about the parent reference frame's unit
            vectors. The order can be specified by the strings ``'XZX'``,
            ``'131'``, or the integer ``131``. There are 12 unique valid
            rotation orders.

        Warns
        ======

        UserWarning
            If the orientation creates a kinematic loop.

        Examples
        ========

        Setup variables for the examples:

        >>> from sympy import symbols
        >>> from sympy.physics.vector import ReferenceFrame
        >>> q1, q2, q3 = symbols('q1, q2, q3')
        >>> N = ReferenceFrame('N')
        >>> B = ReferenceFrame('B')
        >>> B1 = ReferenceFrame('B1')
        >>> B2 = ReferenceFrame('B2')
        >>> B3 = ReferenceFrame('B3')

        >>> B.orient_space_fixed(N, (q1, q2, q3), '312')
        >>> B.dcm(N)
        Matrix([
        [ sin(q1)*sin(q2)*sin(q3) + cos(q1)*cos(q3), sin(q1)*cos(q2), sin(q1)*sin(q2)*cos(q3) - sin(q3)*cos(q1)],
        [-sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1), cos(q1)*cos(q2), sin(q1)*sin(q3) + sin(q2)*cos(q1)*cos(q3)],
        [                           sin(q3)*cos(q2),        -sin(q2),                           cos(q2)*cos(q3)]])

        is equivalent to:

        >>> B1.orient_axis(N, N.z, q1)
        >>> B2.orient_axis(B1, N.x, q2)
        >>> B3.orient_axis(B2, N.y, q3)
        >>> B3.dcm(N).simplify()
        Matrix([
        [ sin(q1)*sin(q2)*sin(q3) + cos(q1)*cos(q3), sin(q1)*cos(q2), sin(q1)*sin(q2)*cos(q3) - sin(q3)*cos(q1)],
        [-sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1), cos(q1)*cos(q2), sin(q1)*sin(q3) + sin(q2)*cos(q1)*cos(q3)],
        [                           sin(q3)*cos(q2),        -sin(q2),                           cos(q2)*cos(q3)]])

        It is worth noting that space-fixed and body-fixed rotations are
        related by the order of the rotations, i.e. the reverse order of body
        fixed will give space fixed and vice versa.

        >>> B.orient_space_fixed(N, (q1, q2, q3), '231')
        >>> B.dcm(N)
        Matrix([
        [cos(q1)*cos(q2), sin(q1)*sin(q3) + sin(q2)*cos(q1)*cos(q3), -sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1)],
        [       -sin(q2),                           cos(q2)*cos(q3),                            sin(q3)*cos(q2)],
        [sin(q1)*cos(q2), sin(q1)*sin(q2)*cos(q3) - sin(q3)*cos(q1),  sin(q1)*sin(q2)*sin(q3) + cos(q1)*cos(q3)]])

        >>> B.orient_body_fixed(N, (q3, q2, q1), '132')
        >>> B.dcm(N)
        Matrix([
        [cos(q1)*cos(q2), sin(q1)*sin(q3) + sin(q2)*cos(q1)*cos(q3), -sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1)],
        [       -sin(q2),                           cos(q2)*cos(q3),                            sin(q3)*cos(q2)],
        [sin(q1)*cos(q2), sin(q1)*sin(q2)*cos(q3) - sin(q3)*cos(q1),  sin(q1)*sin(q2)*sin(q3) + cos(q1)*cos(q3)]])

        """
        from sympy.physics.vector.functions import dynamicsymbols

        _check_frame(parent)

        amounts, rot_order, rot_matrices = self._parse_consecutive_rotations(
            angles, rotation_order)
        self._dcm(parent, rot_matrices[2] * rot_matrices[1] * rot_matrices[0])

        rot_vecs = [zeros(3, 1) for _ in range(3)]
        for i, order in enumerate(rot_order):
            rot_vecs[i][order - 1] = amounts[i].diff(dynamicsymbols._t)
        u1, u2, u3 = rot_vecs[0] + rot_matrices[0].T * (
            rot_vecs[1] + rot_matrices[1].T * rot_vecs[2])
        wvec = u1 * self.x + u2 * self.y + u3 * self.z  # There is a double -
        self._ang_vel_dict.update({parent: wvec})
        parent._ang_vel_dict.update({self: -wvec})
        self._var_dict = {}

    def orient_quaternion(self, parent, numbers):
        """Sets the orientation of this reference frame relative to a parent
        reference frame via an orientation quaternion. An orientation
        quaternion is defined as a finite rotation a unit vector, ``(lambda_x,
        lambda_y, lambda_z)``, by an angle ``theta``. The orientation
        quaternion is described by four parameters:

        - ``q0 = cos(theta/2)``
        - ``q1 = lambda_x*sin(theta/2)``
        - ``q2 = lambda_y*sin(theta/2)``
        - ``q3 = lambda_z*sin(theta/2)``

        See `Quaternions and Spatial Rotation
        <https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation>`_ on
        Wikipedia for more information.

        Parameters
        ==========
        parent : ReferenceFrame
            Reference frame that this reference frame will be rotated relative
            to.
        numbers : 4-tuple of sympifiable
            The four quaternion scalar numbers as defined above: ``q0``,
            ``q1``, ``q2``, ``q3``.

        Warns
        ======

        UserWarning
            If the orientation creates a kinematic loop.

        Examples
        ========

        Setup variables for the examples:

        >>> from sympy import symbols
        >>> from sympy.physics.vector import ReferenceFrame
        >>> q0, q1, q2, q3 = symbols('q0 q1 q2 q3')
        >>> N = ReferenceFrame('N')
        >>> B = ReferenceFrame('B')

        Set the orientation:

        >>> B.orient_quaternion(N, (q0, q1, q2, q3))
        >>> B.dcm(N)
        Matrix([
        [q0**2 + q1**2 - q2**2 - q3**2,             2*q0*q3 + 2*q1*q2,            -2*q0*q2 + 2*q1*q3],
        [           -2*q0*q3 + 2*q1*q2, q0**2 - q1**2 + q2**2 - q3**2,             2*q0*q1 + 2*q2*q3],
        [            2*q0*q2 + 2*q1*q3,            -2*q0*q1 + 2*q2*q3, q0**2 - q1**2 - q2**2 + q3**2]])

        """

        from sympy.physics.vector.functions import dynamicsymbols
        _check_frame(parent)

        numbers = list(numbers)
        for i, v in enumerate(numbers):
            if not isinstance(v, Vector):
                numbers[i] = sympify(v)

        if not (isinstance(numbers, (list, tuple)) & (len(numbers) == 4)):
            raise TypeError('Amounts are a list or tuple of length 4')
        q0, q1, q2, q3 = numbers
        parent_orient_quaternion = (
            Matrix([[q0**2 + q1**2 - q2**2 - q3**2,
                     2 * (q1 * q2 - q0 * q3),
                     2 * (q0 * q2 + q1 * q3)],
                    [2 * (q1 * q2 + q0 * q3),
                     q0**2 - q1**2 + q2**2 - q3**2,
                     2 * (q2 * q3 - q0 * q1)],
                    [2 * (q1 * q3 - q0 * q2),
                     2 * (q0 * q1 + q2 * q3),
                     q0**2 - q1**2 - q2**2 + q3**2]]))

        self._dcm(parent, parent_orient_quaternion)

        t = dynamicsymbols._t
        q0, q1, q2, q3 = numbers
        q0d = diff(q0, t)
        q1d = diff(q1, t)
        q2d = diff(q2, t)
        q3d = diff(q3, t)
        w1 = 2 * (q1d * q0 + q2d * q3 - q3d * q2 - q0d * q1)
        w2 = 2 * (q2d * q0 + q3d * q1 - q1d * q3 - q0d * q2)
        w3 = 2 * (q3d * q0 + q1d * q2 - q2d * q1 - q0d * q3)
        wvec = Vector([(Matrix([w1, w2, w3]), self)])

        self._ang_vel_dict.update({parent: wvec})
        parent._ang_vel_dict.update({self: -wvec})
        self._var_dict = {}

    def orient(self, parent, rot_type, amounts, rot_order=''):
        """Sets the orientation of this reference frame relative to another
        (parent) reference frame.

        .. note:: It is now recommended to use the ``.orient_axis,
           .orient_body_fixed, .orient_space_fixed, .orient_quaternion``
           methods for the different rotation types.

        Parameters
        ==========

        parent : ReferenceFrame
            Reference frame that this reference frame will be rotated relative
            to.
        rot_type : str
            The method used to generate the direction cosine matrix. Supported
            methods are:

            - ``'Axis'``: simple rotations about a single common axis
            - ``'DCM'``: for setting the direction cosine matrix directly
            - ``'Body'``: three successive rotations about new intermediate
              axes, also called "Euler and Tait-Bryan angles"
            - ``'Space'``: three successive rotations about the parent
              frames' unit vectors
            - ``'Quaternion'``: rotations defined by four parameters which
              result in a singularity free direction cosine matrix

        amounts :
            Expressions defining the rotation angles or direction cosine
            matrix. These must match the ``rot_type``. See examples below for
            details. The input types are:

            - ``'Axis'``: 2-tuple (expr/sym/func, Vector)
            - ``'DCM'``: Matrix, shape(3,3)
            - ``'Body'``: 3-tuple of expressions, symbols, or functions
            - ``'Space'``: 3-tuple of expressions, symbols, or functions
            - ``'Quaternion'``: 4-tuple of expressions, symbols, or
              functions

        rot_order : str or int, optional
            If applicable, the order of the successive of rotations. The string
            ``'123'`` and integer ``123`` are equivalent, for example. Required
            for ``'Body'`` and ``'Space'``.

        Warns
        ======

        UserWarning
            If the orientation creates a kinematic loop.

        """

        _check_frame(parent)

        approved_orders = ('123', '231', '312', '132', '213', '321', '121',
                           '131', '212', '232', '313', '323', '')
        rot_order = translate(str(rot_order), 'XYZxyz', '123123')
        rot_type = rot_type.upper()

        if rot_order not in approved_orders:
            raise TypeError('The supplied order is not an approved type')

        if rot_type == 'AXIS':
            self.orient_axis(parent, amounts[1], amounts[0])

        elif rot_type == 'DCM':
            self.orient_explicit(parent, amounts)

        elif rot_type == 'BODY':
            self.orient_body_fixed(parent, amounts, rot_order)

        elif rot_type == 'SPACE':
            self.orient_space_fixed(parent, amounts, rot_order)

        elif rot_type == 'QUATERNION':
            self.orient_quaternion(parent, amounts)

        else:
            raise NotImplementedError('That is not an implemented rotation')

    def orientnew(self, newname, rot_type, amounts, rot_order='',
                  variables=None, indices=None, latexs=None):
        r"""Returns a new reference frame oriented with respect to this
        reference frame.

        See ``ReferenceFrame.orient()`` for detailed examples of how to orient
        reference frames.

        Parameters
        ==========

        newname : str
            Name for the new reference frame.
        rot_type : str
            The method used to generate the direction cosine matrix. Supported
            methods are:

            - ``'Axis'``: simple rotations about a single common axis
            - ``'DCM'``: for setting the direction cosine matrix directly
            - ``'Body'``: three successive rotations about new intermediate
              axes, also called "Euler and Tait-Bryan angles"
            - ``'Space'``: three successive rotations about the parent
              frames' unit vectors
            - ``'Quaternion'``: rotations defined by four parameters which
              result in a singularity free direction cosine matrix

        amounts :
            Expressions defining the rotation angles or direction cosine
            matrix. These must match the ``rot_type``. See examples below for
            details. The input types are:

            - ``'Axis'``: 2-tuple (expr/sym/func, Vector)
            - ``'DCM'``: Matrix, shape(3,3)
            - ``'Body'``: 3-tuple of expressions, symbols, or functions
            - ``'Space'``: 3-tuple of expressions, symbols, or functions
            - ``'Quaternion'``: 4-tuple of expressions, symbols, or
              functions

        rot_order : str or int, optional
            If applicable, the order of the successive of rotations. The string
            ``'123'`` and integer ``123`` are equivalent, for example. Required
            for ``'Body'`` and ``'Space'``.
        indices : tuple of str
            Enables the reference frame's basis unit vectors to be accessed by
            Python's square bracket indexing notation using the provided three
            indice strings and alters the printing of the unit vectors to
            reflect this choice.
        latexs : tuple of str
            Alters the LaTeX printing of the reference frame's basis unit
            vectors to the provided three valid LaTeX strings.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.vector import ReferenceFrame, vlatex
        >>> q0, q1, q2, q3 = symbols('q0 q1 q2 q3')
        >>> N = ReferenceFrame('N')

        Create a new reference frame A rotated relative to N through a simple
        rotation.

        >>> A = N.orientnew('A', 'Axis', (q0, N.x))

        Create a new reference frame B rotated relative to N through body-fixed
        rotations.

        >>> B = N.orientnew('B', 'Body', (q1, q2, q3), '123')

        Create a new reference frame C rotated relative to N through a simple
        rotation with unique indices and LaTeX printing.

        >>> C = N.orientnew('C', 'Axis', (q0, N.x), indices=('1', '2', '3'),
        ... latexs=(r'\hat{\mathbf{c}}_1',r'\hat{\mathbf{c}}_2',
        ... r'\hat{\mathbf{c}}_3'))
        >>> C['1']
        C['1']
        >>> print(vlatex(C['1']))
        \hat{\mathbf{c}}_1

        """

        newframe = self.__class__(newname, variables=variables,
                                  indices=indices, latexs=latexs)

        approved_orders = ('123', '231', '312', '132', '213', '321', '121',
                           '131', '212', '232', '313', '323', '')
        rot_order = translate(str(rot_order), 'XYZxyz', '123123')
        rot_type = rot_type.upper()

        if rot_order not in approved_orders:
            raise TypeError('The supplied order is not an approved type')

        if rot_type == 'AXIS':
            newframe.orient_axis(self, amounts[1], amounts[0])

        elif rot_type == 'DCM':
            newframe.orient_explicit(self, amounts)

        elif rot_type == 'BODY':
            newframe.orient_body_fixed(self, amounts, rot_order)

        elif rot_type == 'SPACE':
            newframe.orient_space_fixed(self, amounts, rot_order)

        elif rot_type == 'QUATERNION':
            newframe.orient_quaternion(self, amounts)

        else:
            raise NotImplementedError('That is not an implemented rotation')
        return newframe

    def set_ang_acc(self, otherframe, value):
        """Define the angular acceleration Vector in a ReferenceFrame.

        Defines the angular acceleration of this ReferenceFrame, in another.
        Angular acceleration can be defined with respect to multiple different
        ReferenceFrames. Care must be taken to not create loops which are
        inconsistent.

        Parameters
        ==========

        otherframe : ReferenceFrame
            A ReferenceFrame to define the angular acceleration in
        value : Vector
            The Vector representing angular acceleration

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> V = 10 * N.x
        >>> A.set_ang_acc(N, V)
        >>> A.ang_acc_in(N)
        10*N.x

        """

        if value == 0:
            value = Vector(0)
        value = _check_vector(value)
        _check_frame(otherframe)
        self._ang_acc_dict.update({otherframe: value})
        otherframe._ang_acc_dict.update({self: -value})

    def set_ang_vel(self, otherframe, value):
        """Define the angular velocity vector in a ReferenceFrame.

        Defines the angular velocity of this ReferenceFrame, in another.
        Angular velocity can be defined with respect to multiple different
        ReferenceFrames. Care must be taken to not create loops which are
        inconsistent.

        Parameters
        ==========

        otherframe : ReferenceFrame
            A ReferenceFrame to define the angular velocity in
        value : Vector
            The Vector representing angular velocity

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> V = 10 * N.x
        >>> A.set_ang_vel(N, V)
        >>> A.ang_vel_in(N)
        10*N.x

        """

        if value == 0:
            value = Vector(0)
        value = _check_vector(value)
        _check_frame(otherframe)
        self._ang_vel_dict.update({otherframe: value})
        otherframe._ang_vel_dict.update({self: -value})

    @property
    def x(self):
        """The basis Vector for the ReferenceFrame, in the x direction. """
        return self._x

    @property
    def y(self):
        """The basis Vector for the ReferenceFrame, in the y direction. """
        return self._y

    @property
    def z(self):
        """The basis Vector for the ReferenceFrame, in the z direction. """
        return self._z

    @property
    def xx(self):
        """Unit dyad of basis Vectors x and x for the ReferenceFrame."""
        return Vector.outer(self.x, self.x)

    @property
    def xy(self):
        """Unit dyad of basis Vectors x and y for the ReferenceFrame."""
        return Vector.outer(self.x, self.y)

    @property
    def xz(self):
        """Unit dyad of basis Vectors x and z for the ReferenceFrame."""
        return Vector.outer(self.x, self.z)

    @property
    def yx(self):
        """Unit dyad of basis Vectors y and x for the ReferenceFrame."""
        return Vector.outer(self.y, self.x)

    @property
    def yy(self):
        """Unit dyad of basis Vectors y and y for the ReferenceFrame."""
        return Vector.outer(self.y, self.y)

    @property
    def yz(self):
        """Unit dyad of basis Vectors y and z for the ReferenceFrame."""
        return Vector.outer(self.y, self.z)

    @property
    def zx(self):
        """Unit dyad of basis Vectors z and x for the ReferenceFrame."""
        return Vector.outer(self.z, self.x)

    @property
    def zy(self):
        """Unit dyad of basis Vectors z and y for the ReferenceFrame."""
        return Vector.outer(self.z, self.y)

    @property
    def zz(self):
        """Unit dyad of basis Vectors z and z for the ReferenceFrame."""
        return Vector.outer(self.z, self.z)

    @property
    def u(self):
        """Unit dyadic for the ReferenceFrame."""
        return self.xx + self.yy + self.zz

    def partial_velocity(self, frame, *gen_speeds):
        """Returns the partial angular velocities of this frame in the given
        frame with respect to one or more provided generalized speeds.

        Parameters
        ==========
        frame : ReferenceFrame
            The frame with which the angular velocity is defined in.
        gen_speeds : functions of time
            The generalized speeds.

        Returns
        =======
        partial_velocities : tuple of Vector
            The partial angular velocity vectors corresponding to the provided
            generalized speeds.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, dynamicsymbols
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> u1, u2 = dynamicsymbols('u1, u2')
        >>> A.set_ang_vel(N, u1 * A.x + u2 * N.y)
        >>> A.partial_velocity(N, u1)
        A.x
        >>> A.partial_velocity(N, u1, u2)
        (A.x, N.y)

        """

        from sympy.physics.vector.functions import partial_velocity

        vel = self.ang_vel_in(frame)
        partials = partial_velocity([vel], gen_speeds, frame)[0]

        if len(partials) == 1:
            return partials[0]
        else:
            return tuple(partials)


def _check_frame(other):
    from .vector import VectorTypeError
    if not isinstance(other, ReferenceFrame):
        raise VectorTypeError(other, ReferenceFrame('A'))
