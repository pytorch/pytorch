from sympy import diff, zeros, Matrix, eye, sympify
from sympy.core.sorting import default_sort_key
from sympy.physics.vector import dynamicsymbols, ReferenceFrame
from sympy.physics.mechanics.method import _Methods
from sympy.physics.mechanics.functions import (
    find_dynamicsymbols, msubs, _f_list_parser, _validate_coordinates)
from sympy.physics.mechanics.linearize import Linearizer
from sympy.utilities.iterables import iterable

__all__ = ['LagrangesMethod']


class LagrangesMethod(_Methods):
    """Lagrange's method object.

    Explanation
    ===========

    This object generates the equations of motion in a two step procedure. The
    first step involves the initialization of LagrangesMethod by supplying the
    Lagrangian and the generalized coordinates, at the bare minimum. If there
    are any constraint equations, they can be supplied as keyword arguments.
    The Lagrange multipliers are automatically generated and are equal in
    number to the constraint equations. Similarly any non-conservative forces
    can be supplied in an iterable (as described below and also shown in the
    example) along with a ReferenceFrame. This is also discussed further in the
    __init__ method.

    Attributes
    ==========

    q, u : Matrix
        Matrices of the generalized coordinates and speeds
    loads : iterable
        Iterable of (Point, vector) or (ReferenceFrame, vector) tuples
        describing the forces on the system.
    bodies : iterable
        Iterable containing the rigid bodies and particles of the system.
    mass_matrix : Matrix
        The system's mass matrix
    forcing : Matrix
        The system's forcing vector
    mass_matrix_full : Matrix
        The "mass matrix" for the qdot's, qdoubledot's, and the
        lagrange multipliers (lam)
    forcing_full : Matrix
        The forcing vector for the qdot's, qdoubledot's and
        lagrange multipliers (lam)

    Examples
    ========

    This is a simple example for a one degree of freedom translational
    spring-mass-damper.

    In this example, we first need to do the kinematics.
    This involves creating generalized coordinates and their derivatives.
    Then we create a point and set its velocity in a frame.

        >>> from sympy.physics.mechanics import LagrangesMethod, Lagrangian
        >>> from sympy.physics.mechanics import ReferenceFrame, Particle, Point
        >>> from sympy.physics.mechanics import dynamicsymbols
        >>> from sympy import symbols
        >>> q = dynamicsymbols('q')
        >>> qd = dynamicsymbols('q', 1)
        >>> m, k, b = symbols('m k b')
        >>> N = ReferenceFrame('N')
        >>> P = Point('P')
        >>> P.set_vel(N, qd * N.x)

    We need to then prepare the information as required by LagrangesMethod to
    generate equations of motion.
    First we create the Particle, which has a point attached to it.
    Following this the lagrangian is created from the kinetic and potential
    energies.
    Then, an iterable of nonconservative forces/torques must be constructed,
    where each item is a (Point, Vector) or (ReferenceFrame, Vector) tuple,
    with the Vectors representing the nonconservative forces or torques.

        >>> Pa = Particle('Pa', P, m)
        >>> Pa.potential_energy = k * q**2 / 2.0
        >>> L = Lagrangian(N, Pa)
        >>> fl = [(P, -b * qd * N.x)]

    Finally we can generate the equations of motion.
    First we create the LagrangesMethod object. To do this one must supply
    the Lagrangian, and the generalized coordinates. The constraint equations,
    the forcelist, and the inertial frame may also be provided, if relevant.
    Next we generate Lagrange's equations of motion, such that:
    Lagrange's equations of motion = 0.
    We have the equations of motion at this point.

        >>> l = LagrangesMethod(L, [q], forcelist = fl, frame = N)
        >>> print(l.form_lagranges_equations())
        Matrix([[b*Derivative(q(t), t) + 1.0*k*q(t) + m*Derivative(q(t), (t, 2))]])

    We can also solve for the states using the 'rhs' method.

        >>> print(l.rhs())
        Matrix([[Derivative(q(t), t)], [(-b*Derivative(q(t), t) - 1.0*k*q(t))/m]])

    Please refer to the docstrings on each method for more details.
    """

    def __init__(self, Lagrangian, qs, forcelist=None, bodies=None, frame=None,
                 hol_coneqs=None, nonhol_coneqs=None):
        """Supply the following for the initialization of LagrangesMethod.

        Lagrangian : Sympifyable

        qs : array_like
            The generalized coordinates

        hol_coneqs : array_like, optional
            The holonomic constraint equations

        nonhol_coneqs : array_like, optional
            The nonholonomic constraint equations

        forcelist : iterable, optional
            Takes an iterable of (Point, Vector) or (ReferenceFrame, Vector)
            tuples which represent the force at a point or torque on a frame.
            This feature is primarily to account for the nonconservative forces
            and/or moments.

        bodies : iterable, optional
            Takes an iterable containing the rigid bodies and particles of the
            system.

        frame : ReferenceFrame, optional
            Supply the inertial frame. This is used to determine the
            generalized forces due to non-conservative forces.
        """

        self._L = Matrix([sympify(Lagrangian)])
        self.eom = None
        self._m_cd = Matrix()           # Mass Matrix of differentiated coneqs
        self._m_d = Matrix()            # Mass Matrix of dynamic equations
        self._f_cd = Matrix()           # Forcing part of the diff coneqs
        self._f_d = Matrix()            # Forcing part of the dynamic equations
        self.lam_coeffs = Matrix()      # The coeffecients of the multipliers

        forcelist = forcelist if forcelist else []
        if not iterable(forcelist):
            raise TypeError('Force pairs must be supplied in an iterable.')
        self._forcelist = forcelist
        if frame and not isinstance(frame, ReferenceFrame):
            raise TypeError('frame must be a valid ReferenceFrame')
        self._bodies = bodies
        self.inertial = frame

        self.lam_vec = Matrix()

        self._term1 = Matrix()
        self._term2 = Matrix()
        self._term3 = Matrix()
        self._term4 = Matrix()

        # Creating the qs, qdots and qdoubledots
        if not iterable(qs):
            raise TypeError('Generalized coordinates must be an iterable')
        self._q = Matrix(qs)
        self._qdots = self.q.diff(dynamicsymbols._t)
        self._qdoubledots = self._qdots.diff(dynamicsymbols._t)
        _validate_coordinates(self.q)

        mat_build = lambda x: Matrix(x) if x else Matrix()
        hol_coneqs = mat_build(hol_coneqs)
        nonhol_coneqs = mat_build(nonhol_coneqs)
        self.coneqs = Matrix([hol_coneqs.diff(dynamicsymbols._t),
                nonhol_coneqs])
        self._hol_coneqs = hol_coneqs

    def form_lagranges_equations(self):
        """Method to form Lagrange's equations of motion.

        Returns a vector of equations of motion using Lagrange's equations of
        the second kind.
        """

        qds = self._qdots
        qdd_zero = dict.fromkeys(self._qdoubledots, 0)
        n = len(self.q)

        # Internally we represent the EOM as four terms:
        # EOM = term1 - term2 - term3 - term4 = 0

        # First term
        self._term1 = self._L.jacobian(qds)
        self._term1 = self._term1.diff(dynamicsymbols._t).T

        # Second term
        self._term2 = self._L.jacobian(self.q).T

        # Third term
        if self.coneqs:
            coneqs = self.coneqs
            m = len(coneqs)
            # Creating the multipliers
            self.lam_vec = Matrix(dynamicsymbols('lam1:' + str(m + 1)))
            self.lam_coeffs = -coneqs.jacobian(qds)
            self._term3 = self.lam_coeffs.T * self.lam_vec
            # Extracting the coeffecients of the qdds from the diff coneqs
            diffconeqs = coneqs.diff(dynamicsymbols._t)
            self._m_cd = diffconeqs.jacobian(self._qdoubledots)
            # The remaining terms i.e. the 'forcing' terms in diff coneqs
            self._f_cd = -diffconeqs.subs(qdd_zero)
        else:
            self._term3 = zeros(n, 1)

        # Fourth term
        if self.forcelist:
            N = self.inertial
            self._term4 = zeros(n, 1)
            for i, qd in enumerate(qds):
                flist = zip(*_f_list_parser(self.forcelist, N))
                self._term4[i] = sum(v.diff(qd, N).dot(f) for (v, f) in flist)
        else:
            self._term4 = zeros(n, 1)

        # Form the dynamic mass and forcing matrices
        without_lam = self._term1 - self._term2 - self._term4
        self._m_d = without_lam.jacobian(self._qdoubledots)
        self._f_d = -without_lam.subs(qdd_zero)

        # Form the EOM
        self.eom = without_lam - self._term3
        return self.eom

    def _form_eoms(self):
        return self.form_lagranges_equations()

    @property
    def mass_matrix(self):
        """Returns the mass matrix, which is augmented by the Lagrange
        multipliers, if necessary.

        Explanation
        ===========

        If the system is described by 'n' generalized coordinates and there are
        no constraint equations then an n X n matrix is returned.

        If there are 'n' generalized coordinates and 'm' constraint equations
        have been supplied during initialization then an n X (n+m) matrix is
        returned. The (n + m - 1)th and (n + m)th columns contain the
        coefficients of the Lagrange multipliers.
        """

        if self.eom is None:
            raise ValueError('Need to compute the equations of motion first')
        if self.coneqs:
            return (self._m_d).row_join(self.lam_coeffs.T)
        else:
            return self._m_d

    @property
    def mass_matrix_full(self):
        """Augments the coefficients of qdots to the mass_matrix."""

        if self.eom is None:
            raise ValueError('Need to compute the equations of motion first')
        n = len(self.q)
        m = len(self.coneqs)
        row1 = eye(n).row_join(zeros(n, n + m))
        row2 = zeros(n, n).row_join(self.mass_matrix)
        if self.coneqs:
            row3 = zeros(m, n).row_join(self._m_cd).row_join(zeros(m, m))
            return row1.col_join(row2).col_join(row3)
        else:
            return row1.col_join(row2)

    @property
    def forcing(self):
        """Returns the forcing vector from 'lagranges_equations' method."""

        if self.eom is None:
            raise ValueError('Need to compute the equations of motion first')
        return self._f_d

    @property
    def forcing_full(self):
        """Augments qdots to the forcing vector above."""

        if self.eom is None:
            raise ValueError('Need to compute the equations of motion first')
        if self.coneqs:
            return self._qdots.col_join(self.forcing).col_join(self._f_cd)
        else:
            return self._qdots.col_join(self.forcing)

    def to_linearizer(self, q_ind=None, qd_ind=None, q_dep=None, qd_dep=None,
                      linear_solver='LU'):
        """Returns an instance of the Linearizer class, initiated from the data
        in the LagrangesMethod class. This may be more desirable than using the
        linearize class method, as the Linearizer object will allow more
        efficient recalculation (i.e. about varying operating points).

        Parameters
        ==========

        q_ind, qd_ind : array_like, optional
            The independent generalized coordinates and speeds.
        q_dep, qd_dep : array_like, optional
            The dependent generalized coordinates and speeds.
        linear_solver : str, callable
            Method used to solve the several symbolic linear systems of the
            form ``A*x=b`` in the linearization process. If a string is
            supplied, it should be a valid method that can be used with the
            :meth:`sympy.matrices.matrixbase.MatrixBase.solve`. If a callable is
            supplied, it should have the format ``x = f(A, b)``, where it
            solves the equations and returns the solution. The default is
            ``'LU'`` which corresponds to SymPy's ``A.LUsolve(b)``.
            ``LUsolve()`` is fast to compute but will often result in
            divide-by-zero and thus ``nan`` results.

        Returns
        =======
        Linearizer
            An instantiated
            :class:`sympy.physics.mechanics.linearize.Linearizer`.

        """

        # Compose vectors
        t = dynamicsymbols._t
        q = self.q
        u = self._qdots
        ud = u.diff(t)
        # Get vector of lagrange multipliers
        lams = self.lam_vec

        mat_build = lambda x: Matrix(x) if x else Matrix()
        q_i = mat_build(q_ind)
        q_d = mat_build(q_dep)
        u_i = mat_build(qd_ind)
        u_d = mat_build(qd_dep)

        # Compose general form equations
        f_c = self._hol_coneqs
        f_v = self.coneqs
        f_a = f_v.diff(t)
        f_0 = u
        f_1 = -u
        f_2 = self._term1
        f_3 = -(self._term2 + self._term4)
        f_4 = -self._term3

        # Check that there are an appropriate number of independent and
        # dependent coordinates
        if len(q_d) != len(f_c) or len(u_d) != len(f_v):
            raise ValueError(("Must supply {:} dependent coordinates, and " +
                    "{:} dependent speeds").format(len(f_c), len(f_v)))
        if set(Matrix([q_i, q_d])) != set(q):
            raise ValueError("Must partition q into q_ind and q_dep, with " +
                    "no extra or missing symbols.")
        if set(Matrix([u_i, u_d])) != set(u):
            raise ValueError("Must partition qd into qd_ind and qd_dep, " +
                    "with no extra or missing symbols.")

        # Find all other dynamic symbols, forming the forcing vector r.
        # Sort r to make it canonical.
        insyms = set(Matrix([q, u, ud, lams]))
        r = list(find_dynamicsymbols(f_3, insyms))
        r.sort(key=default_sort_key)
        # Check for any derivatives of variables in r that are also found in r.
        for i in r:
            if diff(i, dynamicsymbols._t) in r:
                raise ValueError('Cannot have derivatives of specified \
                                 quantities when linearizing forcing terms.')

        return Linearizer(f_0, f_1, f_2, f_3, f_4, f_c, f_v, f_a, q, u, q_i,
                          q_d, u_i, u_d, r, lams, linear_solver=linear_solver)

    def linearize(self, q_ind=None, qd_ind=None, q_dep=None, qd_dep=None,
                  linear_solver='LU', **kwargs):
        """Linearize the equations of motion about a symbolic operating point.

        Parameters
        ==========
        linear_solver : str, callable
            Method used to solve the several symbolic linear systems of the
            form ``A*x=b`` in the linearization process. If a string is
            supplied, it should be a valid method that can be used with the
            :meth:`sympy.matrices.matrixbase.MatrixBase.solve`. If a callable is
            supplied, it should have the format ``x = f(A, b)``, where it
            solves the equations and returns the solution. The default is
            ``'LU'`` which corresponds to SymPy's ``A.LUsolve(b)``.
            ``LUsolve()`` is fast to compute but will often result in
            divide-by-zero and thus ``nan`` results.
        **kwargs
            Extra keyword arguments are passed to
            :meth:`sympy.physics.mechanics.linearize.Linearizer.linearize`.

        Explanation
        ===========

        If kwarg A_and_B is False (default), returns M, A, B, r for the
        linearized form, M*[q', u']^T = A*[q_ind, u_ind]^T + B*r.

        If kwarg A_and_B is True, returns A, B, r for the linearized form
        dx = A*x + B*r, where x = [q_ind, u_ind]^T. Note that this is
        computationally intensive if there are many symbolic parameters. For
        this reason, it may be more desirable to use the default A_and_B=False,
        returning M, A, and B. Values may then be substituted in to these
        matrices, and the state space form found as
        A = P.T*M.inv()*A, B = P.T*M.inv()*B, where P = Linearizer.perm_mat.

        In both cases, r is found as all dynamicsymbols in the equations of
        motion that are not part of q, u, q', or u'. They are sorted in
        canonical form.

        The operating points may be also entered using the ``op_point`` kwarg.
        This takes a dictionary of {symbol: value}, or a an iterable of such
        dictionaries. The values may be numeric or symbolic. The more values
        you can specify beforehand, the faster this computation will run.

        For more documentation, please see the ``Linearizer`` class."""

        linearizer = self.to_linearizer(q_ind, qd_ind, q_dep, qd_dep,
                                        linear_solver=linear_solver)
        result = linearizer.linearize(**kwargs)
        return result + (linearizer.r,)

    def solve_multipliers(self, op_point=None, sol_type='dict'):
        """Solves for the values of the lagrange multipliers symbolically at
        the specified operating point.

        Parameters
        ==========

        op_point : dict or iterable of dicts, optional
            Point at which to solve at. The operating point is specified as
            a dictionary or iterable of dictionaries of {symbol: value}. The
            value may be numeric or symbolic itself.

        sol_type : str, optional
            Solution return type. Valid options are:
            - 'dict': A dict of {symbol : value} (default)
            - 'Matrix': An ordered column matrix of the solution
        """

        # Determine number of multipliers
        k = len(self.lam_vec)
        if k == 0:
            raise ValueError("System has no lagrange multipliers to solve for.")
        # Compose dict of operating conditions
        if isinstance(op_point, dict):
            op_point_dict = op_point
        elif iterable(op_point):
            op_point_dict = {}
            for op in op_point:
                op_point_dict.update(op)
        elif op_point is None:
            op_point_dict = {}
        else:
            raise TypeError("op_point must be either a dictionary or an "
                            "iterable of dictionaries.")
        # Compose the system to be solved
        mass_matrix = self.mass_matrix.col_join(-self.lam_coeffs.row_join(
                zeros(k, k)))
        force_matrix = self.forcing.col_join(self._f_cd)
        # Sub in the operating point
        mass_matrix = msubs(mass_matrix, op_point_dict)
        force_matrix = msubs(force_matrix, op_point_dict)
        # Solve for the multipliers
        sol_list = mass_matrix.LUsolve(-force_matrix)[-k:]
        if sol_type == 'dict':
            return dict(zip(self.lam_vec, sol_list))
        elif sol_type == 'Matrix':
            return Matrix(sol_list)
        else:
            raise ValueError("Unknown sol_type {:}.".format(sol_type))

    def rhs(self, inv_method=None, **kwargs):
        """Returns equations that can be solved numerically.

        Parameters
        ==========

        inv_method : str
            The specific sympy inverse matrix calculation method to use. For a
            list of valid methods, see
            :meth:`~sympy.matrices.matrixbase.MatrixBase.inv`
        """

        if inv_method is None:
            self._rhs = self.mass_matrix_full.LUsolve(self.forcing_full)
        else:
            self._rhs = (self.mass_matrix_full.inv(inv_method,
                         try_block_diag=True) * self.forcing_full)
        return self._rhs

    @property
    def q(self):
        return self._q

    @property
    def u(self):
        return self._qdots

    @property
    def bodies(self):
        return self._bodies

    @property
    def forcelist(self):
        return self._forcelist

    @property
    def loads(self):
        return self._forcelist
