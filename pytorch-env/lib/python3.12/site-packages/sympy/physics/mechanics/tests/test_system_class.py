import pytest

from sympy.core.symbol import symbols
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.matrices.dense import eye, zeros
from sympy.matrices.immutable import ImmutableMatrix
from sympy.physics.mechanics import (
    Force, KanesMethod, LagrangesMethod, Particle, PinJoint, Point,
    PrismaticJoint, ReferenceFrame, RigidBody, Torque, TorqueActuator, System,
    dynamicsymbols)
from sympy.simplify.simplify import simplify
from sympy.solvers.solvers import solve

t = dynamicsymbols._t  # type: ignore
q = dynamicsymbols('q:6')  # type: ignore
qd = dynamicsymbols('q:6', 1)  # type: ignore
u = dynamicsymbols('u:6')  # type: ignore
ua = dynamicsymbols('ua:3')  # type: ignore


class TestSystemBase:
    @pytest.fixture()
    def _empty_system_setup(self):
        self.system = System(ReferenceFrame('frame'), Point('fixed_point'))

    def _empty_system_check(self, exclude=()):
        matrices = ('q_ind', 'q_dep', 'q', 'u_ind', 'u_dep', 'u', 'u_aux',
                    'kdes', 'holonomic_constraints', 'nonholonomic_constraints')
        tuples = ('loads', 'bodies', 'joints', 'actuators')
        for attr in matrices:
            if attr not in exclude:
                assert getattr(self.system, attr)[:] == []
        for attr in tuples:
            if attr not in exclude:
                assert getattr(self.system, attr) == ()
        if 'eom_method' not in exclude:
            assert self.system.eom_method is None

    def _create_filled_system(self, with_speeds=True):
        self.system = System(ReferenceFrame('frame'), Point('fixed_point'))
        u = dynamicsymbols('u:6') if with_speeds else qd
        self.bodies = symbols('rb1:5', cls=RigidBody)
        self.joints = (
            PinJoint('J1', self.bodies[0], self.bodies[1], q[0], u[0]),
            PrismaticJoint('J2', self.bodies[1], self.bodies[2], q[1], u[1]),
            PinJoint('J3', self.bodies[2], self.bodies[3], q[2], u[2])
        )
        self.system.add_joints(*self.joints)
        self.system.add_coordinates(q[3], independent=[False])
        self.system.add_speeds(u[3], independent=False)
        if with_speeds:
            self.system.add_kdes(u[3] - qd[3])
            self.system.add_auxiliary_speeds(ua[0], ua[1])
        self.system.add_holonomic_constraints(q[2] - q[0] + q[1])
        self.system.add_nonholonomic_constraints(u[3] - qd[1] + u[2])
        self.system.u_ind = u[:2]
        self.system.u_dep = u[2:4]
        self.q_ind, self.q_dep = self.system.q_ind[:], self.system.q_dep[:]
        self.u_ind, self.u_dep = self.system.u_ind[:], self.system.u_dep[:]
        self.kdes = self.system.kdes[:]
        self.hc = self.system.holonomic_constraints[:]
        self.vc = self.system.velocity_constraints[:]
        self.nhc = self.system.nonholonomic_constraints[:]

    @pytest.fixture()
    def _filled_system_setup(self):
        self._create_filled_system(with_speeds=True)

    @pytest.fixture()
    def _filled_system_setup_no_speeds(self):
        self._create_filled_system(with_speeds=False)

    def _filled_system_check(self, exclude=()):
        assert 'q_ind' in exclude or self.system.q_ind[:] == q[:3]
        assert 'q_dep' in exclude or self.system.q_dep[:] == [q[3]]
        assert 'q' in exclude or self.system.q[:] == q[:4]
        assert 'u_ind' in exclude or self.system.u_ind[:] == u[:2]
        assert 'u_dep' in exclude or self.system.u_dep[:] == u[2:4]
        assert 'u' in exclude or self.system.u[:] == u[:4]
        assert 'u_aux' in exclude or self.system.u_aux[:] == ua[:2]
        assert 'kdes' in exclude or self.system.kdes[:] == [
            ui - qdi for ui, qdi in zip(u[:4], qd[:4])]
        assert ('holonomic_constraints' in exclude or
                self.system.holonomic_constraints[:] == [q[2] - q[0] + q[1]])
        assert ('nonholonomic_constraints' in exclude or
                self.system.nonholonomic_constraints[:] == [u[3] - qd[1] + u[2]]
                )
        assert ('velocity_constraints' in exclude or
                self.system.velocity_constraints[:] == [
                    qd[2] - qd[0] + qd[1], u[3] - qd[1] + u[2]])
        assert ('bodies' in exclude or
                self.system.bodies == tuple(self.bodies))
        assert ('joints' in exclude or
                self.system.joints == tuple(self.joints))

    @pytest.fixture()
    def _moving_point_mass(self, _empty_system_setup):
        self.system.q_ind = q[0]
        self.system.u_ind = u[0]
        self.system.kdes = u[0] - q[0].diff(t)
        p = Particle('p', mass=symbols('m'))
        self.system.add_bodies(p)
        p.masscenter.set_pos(self.system.fixed_point, q[0] * self.system.x)


class TestSystem(TestSystemBase):
    def test_empty_system(self, _empty_system_setup):
        self._empty_system_check()
        self.system.validate_system()

    def test_filled_system(self, _filled_system_setup):
        self._filled_system_check()
        self.system.validate_system()

    @pytest.mark.parametrize('frame', [None, ReferenceFrame('frame')])
    @pytest.mark.parametrize('fixed_point', [None, Point('fixed_point')])
    def test_init(self, frame, fixed_point):
        if fixed_point is None and frame is None:
            self.system = System()
        else:
            self.system = System(frame, fixed_point)
        if fixed_point is None:
            assert self.system.fixed_point.name == 'inertial_point'
        else:
            assert self.system.fixed_point == fixed_point
        if frame is None:
            assert self.system.frame.name == 'inertial_frame'
        else:
            assert self.system.frame == frame
        self._empty_system_check()
        assert isinstance(self.system.q_ind, ImmutableMatrix)
        assert isinstance(self.system.q_dep, ImmutableMatrix)
        assert isinstance(self.system.q, ImmutableMatrix)
        assert isinstance(self.system.u_ind, ImmutableMatrix)
        assert isinstance(self.system.u_dep, ImmutableMatrix)
        assert isinstance(self.system.u, ImmutableMatrix)
        assert isinstance(self.system.kdes, ImmutableMatrix)
        assert isinstance(self.system.holonomic_constraints, ImmutableMatrix)
        assert isinstance(self.system.nonholonomic_constraints, ImmutableMatrix)

    def test_from_newtonian_rigid_body(self):
        rb = RigidBody('body')
        self.system = System.from_newtonian(rb)
        assert self.system.fixed_point == rb.masscenter
        assert self.system.frame == rb.frame
        self._empty_system_check(exclude=('bodies',))
        self.system.bodies = (rb,)

    def test_from_newtonian_particle(self):
        pt = Particle('particle')
        with pytest.raises(TypeError):
            System.from_newtonian(pt)

    @pytest.mark.parametrize('args, kwargs, exp_q_ind, exp_q_dep, exp_q', [
        (q[:3], {}, q[:3], [], q[:3]),
        (q[:3], {'independent': True}, q[:3], [], q[:3]),
        (q[:3], {'independent': False}, [], q[:3], q[:3]),
        (q[:3], {'independent': [True, False, True]}, [q[0], q[2]], [q[1]],
         [q[0], q[2], q[1]]),
    ])
    def test_coordinates(self, _empty_system_setup, args, kwargs,
                         exp_q_ind, exp_q_dep, exp_q):
        # Test add_coordinates
        self.system.add_coordinates(*args, **kwargs)
        assert self.system.q_ind[:] == exp_q_ind
        assert self.system.q_dep[:] == exp_q_dep
        assert self.system.q[:] == exp_q
        self._empty_system_check(exclude=('q_ind', 'q_dep', 'q'))
        # Test setter for q_ind and q_dep
        self.system.q_ind = exp_q_ind
        self.system.q_dep = exp_q_dep
        assert self.system.q_ind[:] == exp_q_ind
        assert self.system.q_dep[:] == exp_q_dep
        assert self.system.q[:] == exp_q
        self._empty_system_check(exclude=('q_ind', 'q_dep', 'q'))

    @pytest.mark.parametrize('func', ['add_coordinates', 'add_speeds'])
    @pytest.mark.parametrize('args, kwargs', [
        ((q[0], q[5]), {}),
        ((u[0], u[5]), {}),
        ((q[0],), {'independent': False}),
        ((u[0],), {'independent': False}),
        ((u[0], q[5]), {}),
        ((symbols('a'), q[5]), {}),
    ])
    def test_coordinates_speeds_invalid(self, _filled_system_setup, func, args,
                                        kwargs):
        with pytest.raises(ValueError):
            getattr(self.system, func)(*args, **kwargs)
        self._filled_system_check()

    @pytest.mark.parametrize('args, kwargs, exp_u_ind, exp_u_dep, exp_u', [
        (u[:3], {}, u[:3], [], u[:3]),
        (u[:3], {'independent': True}, u[:3], [], u[:3]),
        (u[:3], {'independent': False}, [], u[:3], u[:3]),
        (u[:3], {'independent': [True, False, True]}, [u[0], u[2]], [u[1]],
         [u[0], u[2], u[1]]),
    ])
    def test_speeds(self, _empty_system_setup, args, kwargs, exp_u_ind,
                    exp_u_dep, exp_u):
        # Test add_speeds
        self.system.add_speeds(*args, **kwargs)
        assert self.system.u_ind[:] == exp_u_ind
        assert self.system.u_dep[:] == exp_u_dep
        assert self.system.u[:] == exp_u
        self._empty_system_check(exclude=('u_ind', 'u_dep', 'u'))
        # Test setter for u_ind and u_dep
        self.system.u_ind = exp_u_ind
        self.system.u_dep = exp_u_dep
        assert self.system.u_ind[:] == exp_u_ind
        assert self.system.u_dep[:] == exp_u_dep
        assert self.system.u[:] == exp_u
        self._empty_system_check(exclude=('u_ind', 'u_dep', 'u'))

    @pytest.mark.parametrize('args, kwargs, exp_u_aux', [
        (ua[:3], {}, ua[:3]),
    ])
    def test_auxiliary_speeds(self, _empty_system_setup, args, kwargs,
                              exp_u_aux):
        # Test add_speeds
        self.system.add_auxiliary_speeds(*args, **kwargs)
        assert self.system.u_aux[:] == exp_u_aux
        self._empty_system_check(exclude=('u_aux',))
        # Test setter for u_ind and u_dep
        self.system.u_aux = exp_u_aux
        assert self.system.u_aux[:] == exp_u_aux
        self._empty_system_check(exclude=('u_aux',))

    @pytest.mark.parametrize('args, kwargs', [
        ((ua[2], q[0]), {}),
        ((ua[2], u[1]), {}),
        ((ua[0], ua[2]), {}),
        ((symbols('a'), ua[2]), {}),
    ])
    def test_auxiliary_invalid(self, _filled_system_setup, args, kwargs):
        with pytest.raises(ValueError):
            self.system.add_auxiliary_speeds(*args, **kwargs)
        self._filled_system_check()

    @pytest.mark.parametrize('prop, add_func, args, kwargs', [
        ('q_ind', 'add_coordinates', (q[0],), {}),
        ('q_dep', 'add_coordinates', (q[3],), {'independent': False}),
        ('u_ind', 'add_speeds', (u[0],), {}),
        ('u_dep', 'add_speeds', (u[3],), {'independent': False}),
        ('u_aux', 'add_auxiliary_speeds', (ua[2],), {}),
        ('kdes', 'add_kdes', (qd[0] - u[0],), {}),
        ('holonomic_constraints', 'add_holonomic_constraints',
         (q[0] - q[1],), {}),
        ('nonholonomic_constraints', 'add_nonholonomic_constraints',
         (u[0] - u[1],), {}),
        ('bodies', 'add_bodies', (RigidBody('body'),), {}),
        ('loads', 'add_loads', (Force(Point('P'), ReferenceFrame('N').x),), {}),
        ('actuators', 'add_actuators', (TorqueActuator(
            symbols('T'), ReferenceFrame('N').x, ReferenceFrame('A')),), {}),
    ])
    def test_add_after_reset(self, _filled_system_setup, prop, add_func, args,
                             kwargs):
        setattr(self.system, prop, ())
        exclude = (prop, 'q', 'u')
        if prop in ('holonomic_constraints', 'nonholonomic_constraints'):
            exclude += ('velocity_constraints',)
        self._filled_system_check(exclude=exclude)
        assert list(getattr(self.system, prop)[:]) == []
        getattr(self.system, add_func)(*args, **kwargs)
        assert list(getattr(self.system, prop)[:]) == list(args)

    @pytest.mark.parametrize('prop, add_func, value, error', [
        ('q_ind', 'add_coordinates', symbols('a'), ValueError),
        ('q_dep', 'add_coordinates', symbols('a'), ValueError),
        ('u_ind', 'add_speeds', symbols('a'), ValueError),
        ('u_dep', 'add_speeds', symbols('a'), ValueError),
        ('u_aux', 'add_auxiliary_speeds', symbols('a'), ValueError),
        ('kdes', 'add_kdes', 7, TypeError),
        ('holonomic_constraints', 'add_holonomic_constraints', 7, TypeError),
        ('nonholonomic_constraints', 'add_nonholonomic_constraints', 7,
         TypeError),
        ('bodies', 'add_bodies', symbols('a'), TypeError),
        ('loads', 'add_loads', symbols('a'), TypeError),
        ('actuators', 'add_actuators', symbols('a'), TypeError),
    ])
    def test_type_error(self, _filled_system_setup, prop, add_func, value,
                        error):
        with pytest.raises(error):
            getattr(self.system, add_func)(value)
        with pytest.raises(error):
            setattr(self.system, prop, value)
        self._filled_system_check()

    @pytest.mark.parametrize('args, kwargs, exp_kdes', [
        ((), {}, [ui - qdi for ui, qdi in zip(u[:4], qd[:4])]),
        ((u[4] - qd[4], u[5] - qd[5]), {},
         [ui - qdi for ui, qdi in zip(u[:6], qd[:6])]),
    ])
    def test_kdes(self, _filled_system_setup, args, kwargs, exp_kdes):
        # Test add_speeds
        self.system.add_kdes(*args, **kwargs)
        self._filled_system_check(exclude=('kdes',))
        assert self.system.kdes[:] == exp_kdes
        # Test setter for kdes
        self.system.kdes = exp_kdes
        self._filled_system_check(exclude=('kdes',))
        assert self.system.kdes[:] == exp_kdes

    @pytest.mark.parametrize('args, kwargs', [
        ((u[0] - qd[0], u[4] - qd[4]), {}),
        ((-(u[0] - qd[0]), u[4] - qd[4]), {}),
        (([u[0] - u[0], u[4] - qd[4]]), {}),
    ])
    def test_kdes_invalid(self, _filled_system_setup, args, kwargs):
        with pytest.raises(ValueError):
            self.system.add_kdes(*args, **kwargs)
        self._filled_system_check()

    @pytest.mark.parametrize('args, kwargs, exp_con', [
        ((), {}, [q[2] - q[0] + q[1]]),
        ((q[4] - q[5], q[5] + q[3]), {},
         [q[2] - q[0] + q[1], q[4] - q[5], q[5] + q[3]]),
    ])
    def test_holonomic_constraints(self, _filled_system_setup, args, kwargs,
                                   exp_con):
        exclude = ('holonomic_constraints', 'velocity_constraints')
        exp_vel_con = [c.diff(t) for c in exp_con] + self.nhc
        # Test add_holonomic_constraints
        self.system.add_holonomic_constraints(*args, **kwargs)
        self._filled_system_check(exclude=exclude)
        assert self.system.holonomic_constraints[:] == exp_con
        assert self.system.velocity_constraints[:] == exp_vel_con
        # Test setter for holonomic_constraints
        self.system.holonomic_constraints = exp_con
        self._filled_system_check(exclude=exclude)
        assert self.system.holonomic_constraints[:] == exp_con
        assert self.system.velocity_constraints[:] == exp_vel_con

    @pytest.mark.parametrize('args, kwargs', [
        ((q[2] - q[0] + q[1], q[4] - q[3]), {}),
        ((-(q[2] - q[0] + q[1]), q[4] - q[3]), {}),
        ((q[0] - q[0], q[4] - q[3]), {}),
    ])
    def test_holonomic_constraints_invalid(self, _filled_system_setup, args,
                                           kwargs):
        with pytest.raises(ValueError):
            self.system.add_holonomic_constraints(*args, **kwargs)
        self._filled_system_check()

    @pytest.mark.parametrize('args, kwargs, exp_con', [
        ((), {}, [u[3] - qd[1] + u[2]]),
        ((u[4] - u[5], u[5] + u[3]), {},
         [u[3] - qd[1] + u[2], u[4] - u[5], u[5] + u[3]]),
    ])
    def test_nonholonomic_constraints(self, _filled_system_setup, args, kwargs,
                                      exp_con):
        exclude = ('nonholonomic_constraints', 'velocity_constraints')
        exp_vel_con = self.vc[:len(self.hc)] + exp_con
        # Test add_nonholonomic_constraints
        self.system.add_nonholonomic_constraints(*args, **kwargs)
        self._filled_system_check(exclude=exclude)
        assert self.system.nonholonomic_constraints[:] == exp_con
        assert self.system.velocity_constraints[:] == exp_vel_con
        # Test setter for nonholonomic_constraints
        self.system.nonholonomic_constraints = exp_con
        self._filled_system_check(exclude=exclude)
        assert self.system.nonholonomic_constraints[:] == exp_con
        assert self.system.velocity_constraints[:] == exp_vel_con

    @pytest.mark.parametrize('args, kwargs', [
        ((u[3] - qd[1] + u[2], u[4] - u[3]), {}),
        ((-(u[3] - qd[1] + u[2]), u[4] - u[3]), {}),
        ((u[0] - u[0], u[4] - u[3]), {}),
        (([u[0] - u[0], u[4] - u[3]]), {}),
    ])
    def test_nonholonomic_constraints_invalid(self, _filled_system_setup, args,
                                              kwargs):
        with pytest.raises(ValueError):
            self.system.add_nonholonomic_constraints(*args, **kwargs)
        self._filled_system_check()

    @pytest.mark.parametrize('constraints, expected', [
        ([], []),
        (qd[2] - qd[0] + qd[1], [qd[2] - qd[0] + qd[1]]),
        ([qd[2] + qd[1], u[2] - u[1]], [qd[2] + qd[1], u[2] - u[1]]),
    ])
    def test_velocity_constraints_overwrite(self, _filled_system_setup,
                                            constraints, expected):
        self.system.velocity_constraints = constraints
        self._filled_system_check(exclude=('velocity_constraints',))
        assert self.system.velocity_constraints[:] == expected

    def test_velocity_constraints_back_to_auto(self, _filled_system_setup):
        self.system.velocity_constraints = qd[3] - qd[2]
        self._filled_system_check(exclude=('velocity_constraints',))
        assert self.system.velocity_constraints[:] == [qd[3] - qd[2]]
        self.system.velocity_constraints = None
        self._filled_system_check()

    def test_bodies(self, _filled_system_setup):
        rb1, rb2 = RigidBody('rb1'), RigidBody('rb2')
        p1, p2 = Particle('p1'), Particle('p2')
        self.system.add_bodies(rb1, p1)
        assert self.system.bodies == (*self.bodies, rb1, p1)
        self.system.add_bodies(p2)
        assert self.system.bodies == (*self.bodies, rb1, p1, p2)
        self.system.bodies = []
        assert self.system.bodies == ()
        self.system.bodies = p2
        assert self.system.bodies == (p2,)
        symb = symbols('symb')
        pytest.raises(TypeError, lambda: self.system.add_bodies(symb))
        pytest.raises(ValueError, lambda: self.system.add_bodies(p2))
        with pytest.raises(TypeError):
            self.system.bodies = (rb1, rb2, p1, p2, symb)
        assert self.system.bodies == (p2,)

    def test_add_loads(self):
        system = System()
        N, A = ReferenceFrame('N'), ReferenceFrame('A')
        rb1 = RigidBody('rb1', frame=N)
        mc1 = Point('mc1')
        p1 = Particle('p1', mc1)
        system.add_loads(Torque(rb1, N.x), (mc1, A.x), Force(p1, A.x))
        assert system.loads == ((N, N.x), (mc1, A.x), (mc1, A.x))
        system.loads = [(A, A.x)]
        assert system.loads == ((A, A.x),)
        pytest.raises(ValueError, lambda: system.add_loads((N, N.x, N.y)))
        with pytest.raises(TypeError):
            system.loads = (N, N.x)
        assert system.loads == ((A, A.x),)

    def test_add_actuators(self):
        system = System()
        N, A = ReferenceFrame('N'), ReferenceFrame('A')
        act1 = TorqueActuator(symbols('T1'), N.x, N)
        act2 = TorqueActuator(symbols('T2'), N.y, N, A)
        system.add_actuators(act1)
        assert system.actuators == (act1,)
        assert system.loads == ()
        system.actuators = (act2,)
        assert system.actuators == (act2,)

    def test_add_joints(self):
        q1, q2, q3, q4, u1, u2, u3 = dynamicsymbols('q1:5 u1:4')
        rb1, rb2, rb3, rb4, rb5 = symbols('rb1:6', cls=RigidBody)
        J1 = PinJoint('J1', rb1, rb2, q1, u1)
        J2 = PrismaticJoint('J2', rb2, rb3, q2, u2)
        J3 = PinJoint('J3', rb3, rb4, q3, u3)
        J_lag = PinJoint('J_lag', rb4, rb5, q4, q4.diff(t))
        system = System()
        system.add_joints(J1)
        assert system.joints == (J1,)
        assert system.bodies == (rb1, rb2)
        assert system.q_ind == ImmutableMatrix([q1])
        assert system.u_ind == ImmutableMatrix([u1])
        assert system.kdes == ImmutableMatrix([u1 - q1.diff(t)])
        system.add_bodies(rb4)
        system.add_coordinates(q3)
        system.add_kdes(u3 - q3.diff(t))
        system.add_joints(J3)
        assert system.joints == (J1, J3)
        assert system.bodies == (rb1, rb2, rb4, rb3)
        assert system.q_ind == ImmutableMatrix([q1, q3])
        assert system.u_ind == ImmutableMatrix([u1, u3])
        assert system.kdes == ImmutableMatrix(
            [u1 - q1.diff(t), u3 - q3.diff(t)])
        system.add_kdes(-(u2 - q2.diff(t)))
        system.add_joints(J2)
        assert system.joints == (J1, J3, J2)
        assert system.bodies == (rb1, rb2, rb4, rb3)
        assert system.q_ind == ImmutableMatrix([q1, q3, q2])
        assert system.u_ind == ImmutableMatrix([u1, u3, u2])
        assert system.kdes == ImmutableMatrix([u1 - q1.diff(t), u3 - q3.diff(t),
                                               -(u2 - q2.diff(t))])
        system.add_joints(J_lag)
        assert system.joints == (J1, J3, J2, J_lag)
        assert system.bodies == (rb1, rb2, rb4, rb3, rb5)
        assert system.q_ind == ImmutableMatrix([q1, q3, q2, q4])
        assert system.u_ind == ImmutableMatrix([u1, u3, u2, q4.diff(t)])
        assert system.kdes == ImmutableMatrix([u1 - q1.diff(t), u3 - q3.diff(t),
                                               -(u2 - q2.diff(t))])
        assert system.q_dep[:] == []
        assert system.u_dep[:] == []
        pytest.raises(ValueError, lambda: system.add_joints(J2))
        pytest.raises(TypeError, lambda: system.add_joints(rb1))

    def test_joints_setter(self, _filled_system_setup):
        self.system.joints = self.joints[1:]
        assert self.system.joints == self.joints[1:]
        self._filled_system_check(exclude=('joints',))
        self.system.q_ind = ()
        self.system.u_ind = ()
        self.system.joints = self.joints
        self._filled_system_check()

    @pytest.mark.parametrize('name, joint_index', [
        ('J1', 0),
        ('J2', 1),
        ('not_existing', None),
    ])
    def test_get_joint(self, _filled_system_setup, name, joint_index):
        joint = self.system.get_joint(name)
        if joint_index is None:
            assert joint is None
        else:
            assert joint == self.joints[joint_index]

    @pytest.mark.parametrize('name, body_index', [
        ('rb1', 0),
        ('rb3', 2),
        ('not_existing', None),
    ])
    def test_get_body(self, _filled_system_setup, name, body_index):
        body = self.system.get_body(name)
        if body_index is None:
            assert body is None
        else:
            assert body == self.bodies[body_index]

    @pytest.mark.parametrize('eom_method', [KanesMethod, LagrangesMethod])
    def test_form_eoms_calls_subclass(self, _moving_point_mass, eom_method):
        class MyMethod(eom_method):
            pass

        self.system.form_eoms(eom_method=MyMethod)
        assert isinstance(self.system.eom_method, MyMethod)

    @pytest.mark.parametrize('kwargs, expected', [
        ({}, ImmutableMatrix([[-1, 0], [0, symbols('m')]])),
        ({'explicit_kinematics': True}, ImmutableMatrix([[1, 0],
                                                         [0, symbols('m')]])),
    ])
    def test_system_kane_form_eoms_kwargs(self, _moving_point_mass, kwargs,
                                          expected):
        self.system.form_eoms(**kwargs)
        assert self.system.mass_matrix_full == expected

    @pytest.mark.parametrize('kwargs, mm, gm', [
        ({}, ImmutableMatrix([[1, 0], [0, symbols('m')]]),
         ImmutableMatrix([q[0].diff(t), 0])),
    ])
    def test_system_lagrange_form_eoms_kwargs(self, _moving_point_mass, kwargs,
                                              mm, gm):
        self.system.form_eoms(eom_method=LagrangesMethod, **kwargs)
        assert self.system.mass_matrix_full == mm
        assert self.system.forcing_full == gm

    @pytest.mark.parametrize('eom_method, kwargs, error', [
        (KanesMethod, {'non_existing_kwarg': 1}, TypeError),
        (LagrangesMethod, {'non_existing_kwarg': 1}, TypeError),
        (KanesMethod, {'bodies': []}, ValueError),
        (KanesMethod, {'kd_eqs': []}, ValueError),
        (LagrangesMethod, {'bodies': []}, ValueError),
        (LagrangesMethod, {'Lagrangian': 1}, ValueError),
    ])
    def test_form_eoms_kwargs_errors(self, _empty_system_setup, eom_method,
                                     kwargs, error):
        self.system.q_ind = q[0]
        p = Particle('p', mass=symbols('m'))
        self.system.add_bodies(p)
        p.masscenter.set_pos(self.system.fixed_point, q[0] * self.system.x)
        with pytest.raises(error):
            self.system.form_eoms(eom_method=eom_method, **kwargs)


class TestValidateSystem(TestSystemBase):
    @pytest.mark.parametrize('valid_method, invalid_method, with_speeds', [
        (KanesMethod, LagrangesMethod, True),
        (LagrangesMethod, KanesMethod, False)
    ])
    def test_only_valid(self, valid_method, invalid_method, with_speeds):
        self._create_filled_system(with_speeds=with_speeds)
        self.system.validate_system(valid_method)
        # Test Lagrange should fail due to the usage of generalized speeds
        with pytest.raises(ValueError):
            self.system.validate_system(invalid_method)

    @pytest.mark.parametrize('method, with_speeds', [
        (KanesMethod, True), (LagrangesMethod, False)])
    def test_missing_joint_coordinate(self, method, with_speeds):
        self._create_filled_system(with_speeds=with_speeds)
        self.system.q_ind = self.q_ind[1:]
        self.system.u_ind = self.u_ind[:-1]
        self.system.kdes = self.kdes[:-1]
        pytest.raises(ValueError, lambda: self.system.validate_system(method))

    def test_missing_joint_speed(self, _filled_system_setup):
        self.system.q_ind = self.q_ind[:-1]
        self.system.u_ind = self.u_ind[1:]
        self.system.kdes = self.kdes[:-1]
        pytest.raises(ValueError, lambda: self.system.validate_system())

    def test_missing_joint_kdes(self, _filled_system_setup):
        self.system.kdes = self.kdes[1:]
        pytest.raises(ValueError, lambda: self.system.validate_system())

    def test_negative_joint_kdes(self, _filled_system_setup):
        self.system.kdes = [-self.kdes[0]] + self.kdes[1:]
        self.system.validate_system()

    @pytest.mark.parametrize('method, with_speeds', [
        (KanesMethod, True), (LagrangesMethod, False)])
    def test_missing_holonomic_constraint(self, method, with_speeds):
        self._create_filled_system(with_speeds=with_speeds)
        self.system.holonomic_constraints = []
        self.system.nonholonomic_constraints = self.nhc + [
            self.u_ind[1] - self.u_dep[0] + self.u_ind[0]]
        pytest.raises(ValueError, lambda: self.system.validate_system(method))
        self.system.q_dep = []
        self.system.q_ind = self.q_ind + self.q_dep
        self.system.validate_system(method)

    def test_missing_nonholonomic_constraint(self, _filled_system_setup):
        self.system.nonholonomic_constraints = []
        pytest.raises(ValueError, lambda: self.system.validate_system())
        self.system.u_dep = self.u_dep[1]
        self.system.u_ind = self.u_ind + [self.u_dep[0]]
        self.system.validate_system()

    def test_number_of_coordinates_speeds(self, _filled_system_setup):
        # Test more speeds than coordinates
        self.system.u_ind = self.u_ind + [u[5]]
        self.system.kdes = self.kdes + [u[5] - qd[5]]
        self.system.validate_system()
        # Test more coordinates than speeds
        self.system.q_ind = self.q_ind
        self.system.u_ind = self.u_ind[:-1]
        self.system.kdes = self.kdes[:-1]
        pytest.raises(ValueError, lambda: self.system.validate_system())

    def test_number_of_kdes(self, _filled_system_setup):
        # Test wrong number of kdes
        self.system.kdes = self.kdes[:-1]
        pytest.raises(ValueError, lambda: self.system.validate_system())
        self.system.kdes = self.kdes + [u[2] + u[1] - qd[2]]
        pytest.raises(ValueError, lambda: self.system.validate_system())

    def test_duplicates(self, _filled_system_setup):
        # This is basically a redundant feature, which should never fail
        self.system.validate_system(check_duplicates=True)

    def test_speeds_in_lagrange(self, _filled_system_setup_no_speeds):
        self.system.u_ind = u[:len(self.u_ind)]
        with pytest.raises(ValueError):
            self.system.validate_system(LagrangesMethod)
        self.system.u_ind = []
        self.system.validate_system(LagrangesMethod)
        self.system.u_aux = ua
        with pytest.raises(ValueError):
            self.system.validate_system(LagrangesMethod)
        self.system.u_aux = []
        self.system.validate_system(LagrangesMethod)
        self.system.add_joints(
            PinJoint('Ju', RigidBody('rbu1'), RigidBody('rbu2')))
        self.system.u_ind = []
        with pytest.raises(ValueError):
            self.system.validate_system(LagrangesMethod)


class TestSystemExamples:
    def test_cart_pendulum_kanes(self):
        # This example is the same as in the top documentation of System
        # Added a spring to the cart
        g, l, mc, mp, k = symbols('g l mc mp k')
        F, qp, qc, up, uc = dynamicsymbols('F qp qc up uc')
        rail = RigidBody('rail')
        cart = RigidBody('cart', mass=mc)
        bob = Particle('bob', mass=mp)
        bob_frame = ReferenceFrame('bob_frame')
        system = System.from_newtonian(rail)
        assert system.bodies == (rail,)
        assert system.frame == rail.frame
        assert system.fixed_point == rail.masscenter
        slider = PrismaticJoint('slider', rail, cart, qc, uc, joint_axis=rail.x)
        pin = PinJoint('pin', cart, bob, qp, up, joint_axis=cart.z,
                       child_interframe=bob_frame, child_point=l * bob_frame.y)
        system.add_joints(slider, pin)
        assert system.joints == (slider, pin)
        assert system.get_joint('slider') == slider
        assert system.get_body('bob') == bob
        system.apply_uniform_gravity(-g * system.y)
        system.add_loads((cart.masscenter, F * rail.x))
        system.add_actuators(TorqueActuator(k * qp, cart.z, bob_frame, cart))
        system.validate_system()
        system.form_eoms()
        assert isinstance(system.eom_method, KanesMethod)
        assert (simplify(system.mass_matrix - ImmutableMatrix(
            [[mp + mc, mp * l * cos(qp)], [mp * l * cos(qp), mp * l ** 2]]))
                == zeros(2, 2))
        assert (simplify(system.forcing - ImmutableMatrix([
            [mp * l * up ** 2 * sin(qp) + F],
            [-mp * g * l * sin(qp) + k * qp]])) == zeros(2, 1))

        system.add_holonomic_constraints(
            sympify(bob.masscenter.pos_from(rail.masscenter).dot(system.x)))
        assert system.eom_method is None
        system.q_ind, system.q_dep = qp, qc
        system.u_ind, system.u_dep = up, uc
        system.validate_system()

        # Computed solution based on manually solving the constraints
        subs = {qc: -l * sin(qp),
                uc: -l * cos(qp) * up,
                uc.diff(t): l * (up ** 2 * sin(qp) - up.diff(t) * cos(qp))}
        upd_expected = (
            (-g * mp * sin(qp) + k * qp / l + l * mc * sin(2 * qp) * up ** 2 / 2
             - l * mp * sin(2 * qp) * up ** 2 / 2 - F * cos(qp)) /
            (l * (mc * cos(qp) ** 2 + mp * sin(qp) ** 2)))
        upd_sol = tuple(solve(system.form_eoms().xreplace(subs),
                              up.diff(t)).values())[0]
        assert simplify(upd_sol - upd_expected) == 0
        assert isinstance(system.eom_method, KanesMethod)

        # Test other output
        Mk = -ImmutableMatrix([[0, 1], [1, 0]])
        gk = -ImmutableMatrix([uc, up])
        Md = ImmutableMatrix([[-l ** 2 * mp * cos(qp) ** 2 + l ** 2 * mp,
                               l * mp * cos(qp) - l * (mc + mp) * cos(qp)],
                              [l * cos(qp), 1]])
        gd = ImmutableMatrix(
            [[-g * l * mp * sin(qp) + k * qp - l ** 2 * mp * up ** 2 * sin(qp) *
              cos(qp) - l * F * cos(qp)], [l * up ** 2 * sin(qp)]])
        Mm = (Mk.row_join(zeros(2, 2))).col_join(zeros(2, 2).row_join(Md))
        gm = gk.col_join(gd)
        assert simplify(system.mass_matrix - Md) == zeros(2, 2)
        assert simplify(system.forcing - gd) == zeros(2, 1)
        assert simplify(system.mass_matrix_full - Mm) == zeros(4, 4)
        assert simplify(system.forcing_full - gm) == zeros(4, 1)

    def test_cart_pendulum_lagrange(self):
        # Lagrange version of test_cart_pendulus_kanes
        # Added a spring to the cart
        g, l, mc, mp, k = symbols('g l mc mp k')
        F, qp, qc = dynamicsymbols('F qp qc')
        qpd, qcd = dynamicsymbols('qp qc', 1)
        rail = RigidBody('rail')
        cart = RigidBody('cart', mass=mc)
        bob = Particle('bob', mass=mp)
        bob_frame = ReferenceFrame('bob_frame')
        system = System.from_newtonian(rail)
        assert system.bodies == (rail,)
        assert system.frame == rail.frame
        assert system.fixed_point == rail.masscenter
        slider = PrismaticJoint('slider', rail, cart, qc, qcd,
                                joint_axis=rail.x)
        pin = PinJoint('pin', cart, bob, qp, qpd, joint_axis=cart.z,
                       child_interframe=bob_frame, child_point=l * bob_frame.y)
        system.add_joints(slider, pin)
        assert system.joints == (slider, pin)
        assert system.get_joint('slider') == slider
        assert system.get_body('bob') == bob
        for body in system.bodies:
            body.potential_energy = body.mass * g * body.masscenter.pos_from(
                system.fixed_point).dot(system.y)
        system.add_loads((cart.masscenter, F * rail.x))
        system.add_actuators(TorqueActuator(k * qp, cart.z, bob_frame, cart))
        system.validate_system(LagrangesMethod)
        system.form_eoms(LagrangesMethod)
        assert (simplify(system.mass_matrix - ImmutableMatrix(
            [[mp + mc, mp * l * cos(qp)], [mp * l * cos(qp), mp * l ** 2]]))
                == zeros(2, 2))
        assert (simplify(system.forcing - ImmutableMatrix([
            [mp * l * qpd ** 2 * sin(qp) + F], [-mp * g * l * sin(qp) + k * qp]]
        )) == zeros(2, 1))

        system.add_holonomic_constraints(
            sympify(bob.masscenter.pos_from(rail.masscenter).dot(system.x)))
        assert system.eom_method is None
        system.q_ind, system.q_dep = qp, qc

        # Computed solution based on manually solving the constraints
        subs = {qc: -l * sin(qp),
                qcd: -l * cos(qp) * qpd,
                qcd.diff(t): l * (qpd ** 2 * sin(qp) - qpd.diff(t) * cos(qp))}
        qpdd_expected = (
            (-g * mp * sin(qp) + k * qp / l + l * mc * sin(2 * qp) * qpd ** 2 /
             2 - l * mp * sin(2 * qp) * qpd ** 2 / 2 - F * cos(qp)) /
            (l * (mc * cos(qp) ** 2 + mp * sin(qp) ** 2)))
        eoms = system.form_eoms(LagrangesMethod)
        lam1 = system.eom_method.lam_vec[0]
        lam1_sol = system.eom_method.solve_multipliers()[lam1]
        qpdd_sol = solve(eoms[0].xreplace({lam1: lam1_sol}).xreplace(subs),
                         qpd.diff(t))[0]
        assert simplify(qpdd_sol - qpdd_expected) == 0
        assert isinstance(system.eom_method, LagrangesMethod)

        # Test other output
        Md = ImmutableMatrix([[l ** 2 * mp, l * mp * cos(qp), -l * cos(qp)],
                              [l * mp * cos(qp), mc + mp, -1]])
        gd = ImmutableMatrix(
            [[-g * l * mp * sin(qp) + k * qp],
             [l * mp * sin(qp) * qpd ** 2 + F]])
        Mm = (eye(2).row_join(zeros(2, 3))).col_join(zeros(3, 2).row_join(
            Md.col_join(ImmutableMatrix([l * cos(qp), 1, 0]).T)))
        gm = ImmutableMatrix([qpd, qcd] + gd[:] + [l * sin(qp) * qpd ** 2])
        assert simplify(system.mass_matrix - Md) == zeros(2, 3)
        assert simplify(system.forcing - gd) == zeros(2, 1)
        assert simplify(system.mass_matrix_full - Mm) == zeros(5, 5)
        assert simplify(system.forcing_full - gm) == zeros(5, 1)

    def test_box_on_ground(self):
        # Particle sliding on ground with friction. The applied force is assumed
        # to be positive and to be higher than the friction force.
        g, m, mu = symbols('g m mu')
        q, u, ua = dynamicsymbols('q u ua')
        N, F = dynamicsymbols('N F', positive=True)
        P = Particle("P", mass=m)
        system = System()
        system.add_bodies(P)
        P.masscenter.set_pos(system.fixed_point, q * system.x)
        P.masscenter.set_vel(system.frame, u * system.x + ua * system.y)
        system.q_ind, system.u_ind, system.u_aux = [q], [u], [ua]
        system.kdes = [q.diff(t) - u]
        system.apply_uniform_gravity(-g * system.y)
        system.add_loads(
            Force(P, N * system.y),
            Force(P, F * system.x - mu * N * system.x))
        system.validate_system()
        system.form_eoms()

        # Test other output
        Mk = ImmutableMatrix([1])
        gk = ImmutableMatrix([u])
        Md = ImmutableMatrix([m])
        gd = ImmutableMatrix([F - mu * N])
        Mm = (Mk.row_join(zeros(1, 1))).col_join(zeros(1, 1).row_join(Md))
        gm = gk.col_join(gd)
        aux_eqs = ImmutableMatrix([N - m * g])
        assert simplify(system.mass_matrix - Md) == zeros(1, 1)
        assert simplify(system.forcing - gd) == zeros(1, 1)
        assert simplify(system.mass_matrix_full - Mm) == zeros(2, 2)
        assert simplify(system.forcing_full - gm) == zeros(2, 1)
        assert simplify(system.eom_method.auxiliary_eqs - aux_eqs
                        ) == zeros(1, 1)
