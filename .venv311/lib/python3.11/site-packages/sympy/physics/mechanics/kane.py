from sympy import zeros, Matrix, diff, eye, linear_eq_to_matrix
from sympy.core.sorting import default_sort_key
from sympy.physics.vector import (ReferenceFrame, dynamicsymbols,
                                  partial_velocity)
from sympy.physics.mechanics.method import _Methods
from sympy.physics.mechanics.particle import Particle
from sympy.physics.mechanics.rigidbody import RigidBody
from sympy.physics.mechanics.functions import (msubs, find_dynamicsymbols,
                                               _f_list_parser,
                                               _validate_coordinates,
                                               _parse_linear_solver)
from sympy.physics.mechanics.linearize import Linearizer
from sympy.utilities.iterables import iterable


__all__ = ['KanesMethod']


class KanesMethod(_Methods):
    r"""Kane's method object.

    Explanation
    ===========

    This object is used to do the "book-keeping" as you go through and form
    equations of motion in the way Kane presents in:
    Kane, T., Levinson, D. Dynamics Theory and Applications. 1985 McGraw-Hill

    The attributes are for equations in the form [M] udot = forcing.

    Attributes
    ==========

    q, u : Matrix
        Matrices of the generalized coordinates and speeds
    bodies : iterable
        Iterable of Particle and RigidBody objects in the system.
    loads : iterable
        Iterable of (Point, vector) or (ReferenceFrame, vector) tuples
        describing the forces on the system.
    auxiliary_eqs : Matrix
        If applicable, the set of auxiliary Kane's
        equations used to solve for non-contributing
        forces.
    mass_matrix : Matrix
        The system's dynamics mass matrix: [k_d; k_dnh]
    forcing : Matrix
        The system's dynamics forcing vector: -[f_d; f_dnh]
    mass_matrix_kin : Matrix
        The "mass matrix" for kinematic differential equations: k_kqdot
    forcing_kin : Matrix
        The forcing vector for kinematic differential equations: -(k_ku*u + f_k)
    mass_matrix_full : Matrix
        The "mass matrix" for the u's and q's with dynamics and kinematics
    forcing_full : Matrix
        The "forcing vector" for the u's and q's with dynamics and kinematics

    Parameters
    ==========

    frame : ReferenceFrame
        The inertial reference frame for the system.
    q_ind : iterable of dynamicsymbols
        Independent generalized coordinates.
    u_ind : iterable of dynamicsymbols
        Independent generalized speeds.
    kd_eqs : iterable of Expr, optional
        Kinematic differential equations, which linearly relate the generalized
        speeds to the time-derivatives of the generalized coordinates.
    q_dependent : iterable of dynamicsymbols, optional
        Dependent generalized coordinates.
    configuration_constraints : iterable of Expr, optional
        Constraints on the system's configuration, i.e. holonomic constraints.
    u_dependent : iterable of dynamicsymbols, optional
        Dependent generalized speeds.
    velocity_constraints : iterable of Expr, optional
        Constraints on the system's velocity, i.e. the combination of the
        nonholonomic constraints and the time-derivative of the holonomic
        constraints.
    acceleration_constraints : iterable of Expr, optional
        Constraints on the system's acceleration, by default these are the
        time-derivative of the velocity constraints.
    u_auxiliary : iterable of dynamicsymbols, optional
        Auxiliary generalized speeds.
    bodies : iterable of Particle and/or RigidBody, optional
        The particles and rigid bodies in the system.
    forcelist : iterable of tuple[Point | ReferenceFrame, Vector], optional
        Forces and torques applied on the system.
    explicit_kinematics : bool
        Boolean whether the mass matrices and forcing vectors should use the
        explicit form (default) or implicit form for kinematics.
        See the notes for more details.
    kd_eqs_solver : str, callable
        Method used to solve the kinematic differential equations. If a string
        is supplied, it should be a valid method that can be used with the
        :meth:`sympy.matrices.matrixbase.MatrixBase.solve`. If a callable is
        supplied, it should have the format ``f(A, rhs)``, where it solves the
        equations and returns the solution. The default utilizes LU solve. See
        the notes for more information.
    constraint_solver : str, callable
        Method used to solve the velocity constraints. If a string is
        supplied, it should be a valid method that can be used with the
        :meth:`sympy.matrices.matrixbase.MatrixBase.solve`. If a callable is
        supplied, it should have the format ``f(A, rhs)``, where it solves the
        equations and returns the solution. The default utilizes LU solve. See
        the notes for more information.

    Notes
    =====

    The mass matrices and forcing vectors related to kinematic equations
    are given in the explicit form by default. In other words, the kinematic
    mass matrix is $\mathbf{k_{k\dot{q}}} = \mathbf{I}$.
    In order to get the implicit form of those matrices/vectors, you can set the
    ``explicit_kinematics`` attribute to ``False``. So $\mathbf{k_{k\dot{q}}}$
    is not necessarily an identity matrix. This can provide more compact
    equations for non-simple kinematics.

    Two linear solvers can be supplied to ``KanesMethod``: one for solving the
    kinematic differential equations and one to solve the velocity constraints.
    Both of these sets of equations can be expressed as a linear system ``Ax = rhs``,
    which have to be solved in order to obtain the equations of motion.

    The default solver ``'LU'``, which stands for LU solve, results relatively low
    number of operations. The weakness of this method is that it can result in zero
    division errors.

    If zero divisions are encountered, a possible solver which may solve the problem
    is ``"CRAMER"``. This method uses Cramer's rule to solve the system. This method
    is slower and results in more operations than the default solver. However it only
    uses a single division by default per entry of the solution.

    While a valid list of solvers can be found at
    :meth:`sympy.matrices.matrixbase.MatrixBase.solve`, it is also possible to supply a
    `callable`. This way it is possible to use a different solver routine. If the
    kinematic differential equations are not too complex it can be worth it to simplify
    the solution by using ``lambda A, b: simplify(Matrix.LUsolve(A, b))``. Another
    option solver one may use is :func:`sympy.solvers.solveset.linsolve`. This can be
    done using `lambda A, b: tuple(linsolve((A, b)))[0]`, where we select the first
    solution as our system should have only one unique solution.

    Examples
    ========

    This is a simple example for a one degree of freedom translational
    spring-mass-damper.

    In this example, we first need to do the kinematics.
    This involves creating generalized speeds and coordinates and their
    derivatives.
    Then we create a point and set its velocity in a frame.

        >>> from sympy import symbols
        >>> from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame
        >>> from sympy.physics.mechanics import Point, Particle, KanesMethod
        >>> q, u = dynamicsymbols('q u')
        >>> qd, ud = dynamicsymbols('q u', 1)
        >>> m, c, k = symbols('m c k')
        >>> N = ReferenceFrame('N')
        >>> P = Point('P')
        >>> P.set_vel(N, u * N.x)

    Next we need to arrange/store information in the way that KanesMethod
    requires. The kinematic differential equations should be an iterable of
    expressions. A list of forces/torques must be constructed, where each entry
    in the list is a (Point, Vector) or (ReferenceFrame, Vector) tuple, where
    the Vectors represent the Force or Torque.
    Next a particle needs to be created, and it needs to have a point and mass
    assigned to it.
    Finally, a list of all bodies and particles needs to be created.

        >>> kd = [qd - u]
        >>> FL = [(P, (-k * q - c * u) * N.x)]
        >>> pa = Particle('pa', P, m)
        >>> BL = [pa]

    Finally we can generate the equations of motion.
    First we create the KanesMethod object and supply an inertial frame,
    coordinates, generalized speeds, and the kinematic differential equations.
    Additional quantities such as configuration and motion constraints,
    dependent coordinates and speeds, and auxiliary speeds are also supplied
    here (see the online documentation).
    Next we form FR* and FR to complete: Fr + Fr* = 0.
    We have the equations of motion at this point.
    It makes sense to rearrange them though, so we calculate the mass matrix and
    the forcing terms, for E.o.M. in the form: [MM] udot = forcing, where MM is
    the mass matrix, udot is a vector of the time derivatives of the
    generalized speeds, and forcing is a vector representing "forcing" terms.

        >>> KM = KanesMethod(N, q_ind=[q], u_ind=[u], kd_eqs=kd)
        >>> (fr, frstar) = KM.kanes_equations(BL, FL)
        >>> MM = KM.mass_matrix
        >>> forcing = KM.forcing
        >>> rhs = MM.inv() * forcing
        >>> rhs
        Matrix([[(-c*u(t) - k*q(t))/m]])
        >>> KM.linearize(A_and_B=True)[0]
        Matrix([
        [   0,    1],
        [-k/m, -c/m]])

    Please look at the documentation pages for more information on how to
    perform linearization and how to deal with dependent coordinates & speeds,
    and how do deal with bringing non-contributing forces into evidence.

    """

    def __init__(self, frame, q_ind, u_ind, kd_eqs=None, q_dependent=None,
                 configuration_constraints=None, u_dependent=None,
                 velocity_constraints=None, acceleration_constraints=None,
                 u_auxiliary=None, bodies=None, forcelist=None,
                 explicit_kinematics=True, kd_eqs_solver='LU',
                 constraint_solver='LU'):

        """Please read the online documentation. """
        if not q_ind:
            q_ind = [dynamicsymbols('dummy_q')]
            kd_eqs = [dynamicsymbols('dummy_kd')]

        if not isinstance(frame, ReferenceFrame):
            raise TypeError('An inertial ReferenceFrame must be supplied')
        self._inertial = frame

        self._fr = None
        self._frstar = None

        self._forcelist = forcelist
        self._bodylist = bodies

        self.explicit_kinematics = explicit_kinematics
        self._constraint_solver = constraint_solver
        self._initialize_vectors(q_ind, q_dependent, u_ind, u_dependent,
                u_auxiliary)
        _validate_coordinates(self.q, self.u)
        self._initialize_kindiffeq_matrices(kd_eqs, kd_eqs_solver)
        self._initialize_constraint_matrices(
            configuration_constraints, velocity_constraints,
            acceleration_constraints, constraint_solver)

    def _initialize_vectors(self, q_ind, q_dep, u_ind, u_dep, u_aux):
        """Initialize the coordinate and speed vectors."""

        none_handler = lambda x: Matrix(x) if x else Matrix()

        # Initialize generalized coordinates
        q_dep = none_handler(q_dep)
        if not iterable(q_ind):
            raise TypeError('Generalized coordinates must be an iterable.')
        if not iterable(q_dep):
            raise TypeError('Dependent coordinates must be an iterable.')
        q_ind = Matrix(q_ind)
        self._qdep = q_dep
        self._q = Matrix([q_ind, q_dep])
        self._qdot = self.q.diff(dynamicsymbols._t)

        # Initialize generalized speeds
        u_dep = none_handler(u_dep)
        if not iterable(u_ind):
            raise TypeError('Generalized speeds must be an iterable.')
        if not iterable(u_dep):
            raise TypeError('Dependent speeds must be an iterable.')
        u_ind = Matrix(u_ind)
        self._udep = u_dep
        self._u = Matrix([u_ind, u_dep])
        self._udot = self.u.diff(dynamicsymbols._t)
        self._uaux = none_handler(u_aux)

    def _initialize_constraint_matrices(self, config, vel, acc, linear_solver='LU'):
        """Initializes constraint matrices."""
        linear_solver = _parse_linear_solver(linear_solver)
        # Define vector dimensions
        o = len(self.u)
        m = len(self._udep)
        p = o - m
        none_handler = lambda x: Matrix(x) if x else Matrix()

        # Initialize configuration constraints
        config = none_handler(config)
        if len(self._qdep) != len(config):
            raise ValueError('There must be an equal number of dependent '
                             'coordinates and configuration constraints.')
        self._f_h = none_handler(config)

        # Initialize velocity and acceleration constraints
        vel = none_handler(vel)
        acc = none_handler(acc)
        if len(vel) != m:
            raise ValueError('There must be an equal number of dependent '
                             'speeds and velocity constraints.')
        if acc and (len(acc) != m):
            raise ValueError('There must be an equal number of dependent '
                             'speeds and acceleration constraints.')
        if vel:

            # When calling kanes_equations, another class instance will be
            # created if auxiliary u's are present. In this case, the
            # computation of kinetic differential equation matrices will be
            # skipped as this was computed during the original KanesMethod
            # object, and the qd_u_map will not be available.
            if self._qdot_u_map is not None:
                vel = msubs(vel, self._qdot_u_map)
            self._k_nh, f_nh_neg = linear_eq_to_matrix(vel, self.u[:])
            self._f_nh = -f_nh_neg

            # If no acceleration constraints given, calculate them.
            if not acc:
                _f_dnh = (self._k_nh.diff(dynamicsymbols._t) * self.u +
                    self._f_nh.diff(dynamicsymbols._t))
                if self._qdot_u_map is not None:
                    _f_dnh = msubs(_f_dnh, self._qdot_u_map)
                self._f_dnh = _f_dnh
                self._k_dnh = self._k_nh
            else:
                if self._qdot_u_map is not None:
                    acc = msubs(acc, self._qdot_u_map)

                self._k_dnh, f_dnh_neg = linear_eq_to_matrix(acc, self._udot[:])
                self._f_dnh = -f_dnh_neg
            # Form of non-holonomic constraints is B*u + C = 0.
            # We partition B into independent and dependent columns:
            # Ars is then -B_dep.inv() * B_ind, and it relates dependent speeds
            # to independent speeds as: udep = Ars*uind, neglecting the C term.
            B_ind = self._k_nh[:, :p]
            B_dep = self._k_nh[:, p:o]
            self._Ars = -linear_solver(B_dep, B_ind)
        else:
            self._f_nh = Matrix()
            self._k_nh = Matrix()
            self._f_dnh = Matrix()
            self._k_dnh = Matrix()
            self._Ars = Matrix()

    def _initialize_kindiffeq_matrices(self, kdeqs, linear_solver='LU'):
        """Initialize the kinematic differential equation matrices.

        Parameters
        ==========
        kdeqs : sequence of sympy expressions
            Kinematic differential equations in the form of f(u,q',q,t) where
            f() = 0. The equations have to be linear in the time-derivatives of
            the generalized coordinates and in the generalized speeds.

        """
        linear_solver = _parse_linear_solver(linear_solver)
        if kdeqs:
            if len(self.q) != len(kdeqs):
                raise ValueError('There must be an equal number of kinematic '
                                 'differential equations and coordinates.')

            u = self.u
            qdot = self._qdot

            kdeqs = Matrix(kdeqs)

            u_zero = dict.fromkeys(u, 0)
            uaux_zero = dict.fromkeys(self._uaux, 0)
            qdot_zero = dict.fromkeys(qdot, 0)

            # Extract the linear coefficient matrices as per the following
            # equation:
            #
            # k_ku(q,t)*u(t) + k_kqdot(q,t)*q'(t) + f_k(q,t) = 0
            #
            k_ku = kdeqs.jacobian(u)
            k_kqdot = kdeqs.jacobian(qdot)
            f_k = kdeqs.xreplace(u_zero).xreplace(qdot_zero)

            # The kinematic differential equations should be linear in both q'
            # and u so check for u and q' in the components.
            dy_syms = find_dynamicsymbols(k_ku.row_join(k_kqdot).row_join(f_k))
            nonlin_vars = [vari for vari in u[:] + qdot[:] if vari in dy_syms]
            if nonlin_vars:
                msg = ('The provided kinematic differential equations are '
                       'nonlinear in {}. They must be linear in the '
                       'generalized speeds and derivatives of the generalized '
                       'coordinates.')
                raise ValueError(msg.format(nonlin_vars))

            self._f_k_implicit = f_k.xreplace(uaux_zero)
            self._k_ku_implicit = k_ku.xreplace(uaux_zero)
            self._k_kqdot_implicit = k_kqdot

            # Solve for q'(t) such that the coefficient matrices are now in
            # this form:
            #
            # k_kqdot^-1*k_ku*u(t) + I*q'(t) + k_kqdot^-1*f_k = 0
            #
            # NOTE : Solving the kinematic differential equations here is not
            # necessary and prevents the equations from being provided in fully
            # implicit form.
            f_k_explicit = linear_solver(k_kqdot, f_k)
            k_ku_explicit = linear_solver(k_kqdot, k_ku)
            self._qdot_u_map = dict(zip(qdot, -(k_ku_explicit*u + f_k_explicit)))

            self._f_k = f_k_explicit.xreplace(uaux_zero)
            self._k_ku = k_ku_explicit.xreplace(uaux_zero)
            self._k_kqdot = eye(len(qdot))

        else:
            self._qdot_u_map = None
            self._f_k_implicit = self._f_k = Matrix()
            self._k_ku_implicit = self._k_ku = Matrix()
            self._k_kqdot_implicit = self._k_kqdot = Matrix()

    def _form_fr(self, fl):
        """Form the generalized active force."""
        if fl is not None and (len(fl) == 0 or not iterable(fl)):
            raise ValueError('Force pairs must be supplied in an '
                'non-empty iterable or None.')

        N = self._inertial
        # pull out relevant velocities for constructing partial velocities
        vel_list, f_list = _f_list_parser(fl, N)
        vel_list = [msubs(i, self._qdot_u_map) for i in vel_list]
        f_list = [msubs(i, self._qdot_u_map) for i in f_list]

        # Fill Fr with dot product of partial velocities and forces
        o = len(self.u)
        b = len(f_list)
        FR = zeros(o, 1)
        partials = partial_velocity(vel_list, self.u, N)
        for i in range(o):
            FR[i] = sum(partials[j][i].dot(f_list[j]) for j in range(b))

        # In case there are dependent speeds
        if self._udep:
            p = o - len(self._udep)
            FRtilde = FR[:p, 0]
            FRold = FR[p:o, 0]
            FRtilde += self._Ars.T * FRold
            FR = FRtilde

        self._forcelist = fl
        self._fr = FR
        return FR

    def _form_frstar(self, bl):
        """Form the generalized inertia force."""

        if not iterable(bl):
            raise TypeError('Bodies must be supplied in an iterable.')

        t = dynamicsymbols._t
        N = self._inertial
        # Dicts setting things to zero
        udot_zero = dict.fromkeys(self._udot, 0)
        uaux_zero = dict.fromkeys(self._uaux, 0)
        uauxdot = [diff(i, t) for i in self._uaux]
        uauxdot_zero = dict.fromkeys(uauxdot, 0)
        # Dictionary of q' and q'' to u and u'
        q_ddot_u_map = {k.diff(t): v.diff(t).xreplace(
            self._qdot_u_map) for (k, v) in self._qdot_u_map.items()}
        q_ddot_u_map.update(self._qdot_u_map)

        # Fill up the list of partials: format is a list with num elements
        # equal to number of entries in body list. Each of these elements is a
        # list - either of length 1 for the translational components of
        # particles or of length 2 for the translational and rotational
        # components of rigid bodies. The inner most list is the list of
        # partial velocities.
        def get_partial_velocity(body):
            if isinstance(body, RigidBody):
                vlist = [body.masscenter.vel(N), body.frame.ang_vel_in(N)]
            elif isinstance(body, Particle):
                vlist = [body.point.vel(N),]
            else:
                raise TypeError('The body list may only contain either '
                                'RigidBody or Particle as list elements.')
            v = [msubs(vel, self._qdot_u_map) for vel in vlist]
            return partial_velocity(v, self.u, N)
        partials = [get_partial_velocity(body) for body in bl]

        # Compute fr_star in two components:
        # fr_star = -(MM*u' + nonMM)
        o = len(self.u)
        MM = zeros(o, o)
        nonMM = zeros(o, 1)
        zero_uaux = lambda expr: msubs(expr, uaux_zero)
        zero_udot_uaux = lambda expr: msubs(msubs(expr, udot_zero), uaux_zero)
        for i, body in enumerate(bl):
            if isinstance(body, RigidBody):
                M = zero_uaux(body.mass)
                I = zero_uaux(body.central_inertia)
                vel = zero_uaux(body.masscenter.vel(N))
                omega = zero_uaux(body.frame.ang_vel_in(N))
                acc = zero_udot_uaux(body.masscenter.acc(N))
                inertial_force = (M.diff(t) * vel + M * acc)
                inertial_torque = zero_uaux((I.dt(body.frame).dot(omega)) +
                    msubs(I.dot(body.frame.ang_acc_in(N)), udot_zero) +
                    (omega.cross(I.dot(omega))))
                for j in range(o):
                    tmp_vel = zero_uaux(partials[i][0][j])
                    tmp_ang = zero_uaux(I.dot(partials[i][1][j]))
                    for k in range(o):
                        # translational
                        MM[j, k] += M*tmp_vel.dot(partials[i][0][k])
                        # rotational
                        MM[j, k] += tmp_ang.dot(partials[i][1][k])
                    nonMM[j] += inertial_force.dot(partials[i][0][j])
                    nonMM[j] += inertial_torque.dot(partials[i][1][j])
            else:
                M = zero_uaux(body.mass)
                vel = zero_uaux(body.point.vel(N))
                acc = zero_udot_uaux(body.point.acc(N))
                inertial_force = (M.diff(t) * vel + M * acc)
                for j in range(o):
                    temp = zero_uaux(partials[i][0][j])
                    for k in range(o):
                        MM[j, k] += M*temp.dot(partials[i][0][k])
                    nonMM[j] += inertial_force.dot(partials[i][0][j])
        # Compose fr_star out of MM and nonMM
        MM = zero_uaux(msubs(MM, q_ddot_u_map))
        nonMM = msubs(msubs(nonMM, q_ddot_u_map),
                udot_zero, uauxdot_zero, uaux_zero)
        fr_star = -(MM * msubs(Matrix(self._udot), uauxdot_zero) + nonMM)

        # If there are dependent speeds, we need to find fr_star_tilde
        if self._udep:
            p = o - len(self._udep)
            fr_star_ind = fr_star[:p, 0]
            fr_star_dep = fr_star[p:o, 0]
            fr_star = fr_star_ind + (self._Ars.T * fr_star_dep)
            # Apply the same to MM
            MMi = MM[:p, :]
            MMd = MM[p:o, :]
            MM = MMi + (self._Ars.T * MMd)
            # Apply the same to nonMM
            nonMM = nonMM[:p, :] + (self._Ars.T * nonMM[p:o, :])

        self._bodylist = bl
        self._frstar = fr_star
        self._k_d = MM
        self._f_d = -(self._fr - nonMM)
        return fr_star

    def to_linearizer(self, linear_solver='LU'):
        """Returns an instance of the Linearizer class, initiated from the
        data in the KanesMethod class. This may be more desirable than using
        the linearize class method, as the Linearizer object will allow more
        efficient recalculation (i.e. about varying operating points).

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

        Returns
        =======
        Linearizer
            An instantiated
            :class:`sympy.physics.mechanics.linearize.Linearizer`.

        """

        if (self._fr is None) or (self._frstar is None):
            raise ValueError('Need to compute Fr, Fr* first.')

        # Get required equation components. The Kane's method class breaks
        # these into pieces. Need to reassemble
        f_c = self._f_h
        if self._f_nh and self._k_nh:
            f_v = self._f_nh + self._k_nh*Matrix(self.u)
        else:
            f_v = Matrix()
        if self._f_dnh and self._k_dnh:
            f_a = self._f_dnh + self._k_dnh*Matrix(self._udot)
        else:
            f_a = Matrix()
        # Dicts to sub to zero, for splitting up expressions
        u_zero = dict.fromkeys(self.u, 0)
        ud_zero = dict.fromkeys(self._udot, 0)
        qd_zero = dict.fromkeys(self._qdot, 0)
        qd_u_zero = dict.fromkeys(Matrix([self._qdot, self.u]), 0)
        # Break the kinematic differential eqs apart into f_0 and f_1
        f_0 = msubs(self._f_k, u_zero) + self._k_kqdot*Matrix(self._qdot)
        f_1 = msubs(self._f_k, qd_zero) + self._k_ku*Matrix(self.u)
        # Break the dynamic differential eqs into f_2 and f_3
        f_2 = msubs(self._frstar, qd_u_zero)
        f_3 = msubs(self._frstar, ud_zero) + self._fr
        f_4 = zeros(len(f_2), 1)

        # Get the required vector components
        q = self.q
        u = self.u
        if self._qdep:
            q_i = q[:-len(self._qdep)]
        else:
            q_i = q
        q_d = self._qdep
        if self._udep:
            u_i = u[:-len(self._udep)]
        else:
            u_i = u
        u_d = self._udep

        # Form dictionary to set auxiliary speeds & their derivatives to 0.
        uaux = self._uaux
        uauxdot = uaux.diff(dynamicsymbols._t)
        uaux_zero = dict.fromkeys(Matrix([uaux, uauxdot]), 0)

        # Checking for dynamic symbols outside the dynamic differential
        # equations; throws error if there is.
        sym_list = set(Matrix([q, self._qdot, u, self._udot, uaux, uauxdot]))
        if any(find_dynamicsymbols(i, sym_list) for i in [self._k_kqdot,
                self._k_ku, self._f_k, self._k_dnh, self._f_dnh, self._k_d]):
            raise ValueError('Cannot have dynamicsymbols outside dynamic \
                             forcing vector.')

        # Find all other dynamic symbols, forming the forcing vector r.
        # Sort r to make it canonical.
        r = list(find_dynamicsymbols(msubs(self._f_d, uaux_zero), sym_list))
        r.sort(key=default_sort_key)

        # Check for any derivatives of variables in r that are also found in r.
        for i in r:
            if diff(i, dynamicsymbols._t) in r:
                raise ValueError('Cannot have derivatives of specified \
                                 quantities when linearizing forcing terms.')
        return Linearizer(f_0, f_1, f_2, f_3, f_4, f_c, f_v, f_a, q, u, q_i,
                q_d, u_i, u_d, r, linear_solver=linear_solver)

    # TODO : Remove `new_method` after 1.1 has been released.
    def linearize(self, *, new_method=None, linear_solver='LU', **kwargs):
        """ Linearize the equations of motion about a symbolic operating point.

        Parameters
        ==========
        new_method
            Deprecated, does nothing and will be removed.
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

        For more documentation, please see the ``Linearizer`` class.

        """

        linearizer = self.to_linearizer(linear_solver=linear_solver)
        result = linearizer.linearize(**kwargs)
        return result + (linearizer.r,)

    def kanes_equations(self, bodies=None, loads=None):
        """ Method to form Kane's equations, Fr + Fr* = 0.

        Explanation
        ===========

        Returns (Fr, Fr*). In the case where auxiliary generalized speeds are
        present (say, s auxiliary speeds, o generalized speeds, and m motion
        constraints) the length of the returned vectors will be o - m + s in
        length. The first o - m equations will be the constrained Kane's
        equations, then the s auxiliary Kane's equations. These auxiliary
        equations can be accessed with the auxiliary_eqs property.

        Parameters
        ==========

        bodies : iterable
            An iterable of all RigidBody's and Particle's in the system.
            A system must have at least one body.
        loads : iterable
            Takes in an iterable of (Particle, Vector) or (ReferenceFrame, Vector)
            tuples which represent the force at a point or torque on a frame.
            Must be either a non-empty iterable of tuples or None which corresponds
            to a system with no constraints.
        """
        if bodies is None:
            bodies = self.bodies
        if  loads is None and self._forcelist is not None:
            loads = self._forcelist
        if loads == []:
            loads = None
        if not self._k_kqdot:
            raise AttributeError('Create an instance of KanesMethod with '
                    'kinematic differential equations to use this method.')
        fr = self._form_fr(loads)
        frstar = self._form_frstar(bodies)
        if self._uaux:
            if not self._udep:
                km = KanesMethod(self._inertial, self.q, self._uaux,
                             u_auxiliary=self._uaux, constraint_solver=self._constraint_solver)
            else:
                km = KanesMethod(self._inertial, self.q, self._uaux,
                        u_auxiliary=self._uaux, u_dependent=self._udep,
                        velocity_constraints=(self._k_nh * self.u +
                        self._f_nh),
                        acceleration_constraints=(self._k_dnh * self._udot +
                        self._f_dnh),
                        constraint_solver=self._constraint_solver
                        )
            km._qdot_u_map = self._qdot_u_map
            self._km = km
            fraux = km._form_fr(loads)
            frstaraux = km._form_frstar(bodies)
            self._aux_eq = fraux + frstaraux
            self._fr = fr.col_join(fraux)
            self._frstar = frstar.col_join(frstaraux)
        return (self._fr, self._frstar)

    def _form_eoms(self):
        fr, frstar = self.kanes_equations(self.bodylist, self.forcelist)
        return fr + frstar

    def rhs(self, inv_method=None):
        """Returns the system's equations of motion in first order form. The
        output is the right hand side of::

           x' = |q'| =: f(q, u, r, p, t)
                |u'|

        The right hand side is what is needed by most numerical ODE
        integrators.

        Parameters
        ==========

        inv_method : str
            The specific sympy inverse matrix calculation method to use. For a
            list of valid methods, see
            :meth:`~sympy.matrices.matrixbase.MatrixBase.inv`

        """
        rhs = zeros(len(self.q) + len(self.u), 1)
        kdes = self.kindiffdict()
        for i, q_i in enumerate(self.q):
            rhs[i] = kdes[q_i.diff()]

        if inv_method is None:
            rhs[len(self.q):, 0] = self.mass_matrix.LUsolve(self.forcing)
        else:
            rhs[len(self.q):, 0] = (self.mass_matrix.inv(inv_method,
                                                         try_block_diag=True) *
                                    self.forcing)

        return rhs

    def kindiffdict(self):
        """Returns a dictionary mapping q' to u."""
        if not self._qdot_u_map:
            raise AttributeError('Create an instance of KanesMethod with '
                    'kinematic differential equations to use this method.')
        return self._qdot_u_map

    @property
    def auxiliary_eqs(self):
        """A matrix containing the auxiliary equations."""
        if not self._fr or not self._frstar:
            raise ValueError('Need to compute Fr, Fr* first.')
        if not self._uaux:
            raise ValueError('No auxiliary speeds have been declared.')
        return self._aux_eq

    @property
    def mass_matrix_kin(self):
        r"""The kinematic "mass matrix" $\mathbf{k_{k\dot{q}}}$ of the system."""
        return self._k_kqdot if self.explicit_kinematics else self._k_kqdot_implicit

    @property
    def forcing_kin(self):
        """The kinematic "forcing vector" of the system."""
        if self.explicit_kinematics:
            return -(self._k_ku * Matrix(self.u) + self._f_k)
        else:
            return -(self._k_ku_implicit * Matrix(self.u) + self._f_k_implicit)

    @property
    def mass_matrix(self):
        """The mass matrix of the system."""
        if not self._fr or not self._frstar:
            raise ValueError('Need to compute Fr, Fr* first.')
        return Matrix([self._k_d, self._k_dnh])

    @property
    def forcing(self):
        """The forcing vector of the system."""
        if not self._fr or not self._frstar:
            raise ValueError('Need to compute Fr, Fr* first.')
        return -Matrix([self._f_d, self._f_dnh])

    @property
    def mass_matrix_full(self):
        """The mass matrix of the system, augmented by the kinematic
        differential equations in explicit or implicit form."""
        if not self._fr or not self._frstar:
            raise ValueError('Need to compute Fr, Fr* first.')
        o, n = len(self.u), len(self.q)
        return (self.mass_matrix_kin.row_join(zeros(n, o))).col_join(
            zeros(o, n).row_join(self.mass_matrix))

    @property
    def forcing_full(self):
        """The forcing vector of the system, augmented by the kinematic
        differential equations in explicit or implicit form."""
        return Matrix([self.forcing_kin, self.forcing])

    @property
    def q(self):
        return self._q

    @property
    def u(self):
        return self._u

    @property
    def bodylist(self):
        return self._bodylist

    @property
    def forcelist(self):
        return self._forcelist

    @property
    def bodies(self):
        return self._bodylist

    @property
    def loads(self):
        return self._forcelist
