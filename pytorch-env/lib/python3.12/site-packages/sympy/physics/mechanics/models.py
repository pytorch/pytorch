#!/usr/bin/env python
"""This module contains some sample symbolic models used for testing and
examples."""

# Internal imports
from sympy.core import backend as sm
import sympy.physics.mechanics as me


def multi_mass_spring_damper(n=1, apply_gravity=False,
                             apply_external_forces=False):
    r"""Returns a system containing the symbolic equations of motion and
    associated variables for a simple multi-degree of freedom point mass,
    spring, damper system with optional gravitational and external
    specified forces. For example, a two mass system under the influence of
    gravity and external forces looks like:

    ::

        ----------------
         |     |     |   | g
         \    | |    |   V
      k0 /    --- c0 |
         |     |     | x0, v0
        ---------    V
        |  m0   | -----
        ---------    |
         | |   |     |
         \ v  | |    |
      k1 / f0 --- c1 |
         |     |     | x1, v1
        ---------    V
        |  m1   | -----
        ---------
           | f1
           V

    Parameters
    ==========

    n : integer
        The number of masses in the serial chain.
    apply_gravity : boolean
        If true, gravity will be applied to each mass.
    apply_external_forces : boolean
        If true, a time varying external force will be applied to each mass.

    Returns
    =======

    kane : sympy.physics.mechanics.kane.KanesMethod
        A KanesMethod object.

    """

    mass = sm.symbols('m:{}'.format(n))
    stiffness = sm.symbols('k:{}'.format(n))
    damping = sm.symbols('c:{}'.format(n))

    acceleration_due_to_gravity = sm.symbols('g')

    coordinates = me.dynamicsymbols('x:{}'.format(n))
    speeds = me.dynamicsymbols('v:{}'.format(n))
    specifieds = me.dynamicsymbols('f:{}'.format(n))

    ceiling = me.ReferenceFrame('N')
    origin = me.Point('origin')
    origin.set_vel(ceiling, 0)

    points = [origin]
    kinematic_equations = []
    particles = []
    forces = []

    for i in range(n):

        center = points[-1].locatenew('center{}'.format(i),
                                      coordinates[i] * ceiling.x)
        center.set_vel(ceiling, points[-1].vel(ceiling) +
                       speeds[i] * ceiling.x)
        points.append(center)

        block = me.Particle('block{}'.format(i), center, mass[i])

        kinematic_equations.append(speeds[i] - coordinates[i].diff())

        total_force = (-stiffness[i] * coordinates[i] -
                       damping[i] * speeds[i])
        try:
            total_force += (stiffness[i + 1] * coordinates[i + 1] +
                            damping[i + 1] * speeds[i + 1])
        except IndexError:  # no force from below on last mass
            pass

        if apply_gravity:
            total_force += mass[i] * acceleration_due_to_gravity

        if apply_external_forces:
            total_force += specifieds[i]

        forces.append((center, total_force * ceiling.x))

        particles.append(block)

    kane = me.KanesMethod(ceiling, q_ind=coordinates, u_ind=speeds,
                          kd_eqs=kinematic_equations)
    kane.kanes_equations(particles, forces)

    return kane


def n_link_pendulum_on_cart(n=1, cart_force=True, joint_torques=False):
    r"""Returns the system containing the symbolic first order equations of
    motion for a 2D n-link pendulum on a sliding cart under the influence of
    gravity.

    ::

                  |
         o    y   v
          \ 0 ^   g
           \  |
          --\-|----
          |  \|   |
      F-> |   o --|---> x
          |       |
          ---------
           o     o

    Parameters
    ==========

    n : integer
        The number of links in the pendulum.
    cart_force : boolean, default=True
        If true an external specified lateral force is applied to the cart.
    joint_torques : boolean, default=False
        If true joint torques will be added as specified inputs at each
        joint.

    Returns
    =======

    kane : sympy.physics.mechanics.kane.KanesMethod
        A KanesMethod object.

    Notes
    =====

    The degrees of freedom of the system are n + 1, i.e. one for each
    pendulum link and one for the lateral motion of the cart.

    M x' = F, where x = [u0, ..., un+1, q0, ..., qn+1]

    The joint angles are all defined relative to the ground where the x axis
    defines the ground line and the y axis points up. The joint torques are
    applied between each adjacent link and the between the cart and the
    lower link where a positive torque corresponds to positive angle.

    """
    if n <= 0:
        raise ValueError('The number of links must be a positive integer.')

    q = me.dynamicsymbols('q:{}'.format(n + 1))
    u = me.dynamicsymbols('u:{}'.format(n + 1))

    if joint_torques is True:
        T = me.dynamicsymbols('T1:{}'.format(n + 1))

    m = sm.symbols('m:{}'.format(n + 1))
    l = sm.symbols('l:{}'.format(n))
    g, t = sm.symbols('g t')

    I = me.ReferenceFrame('I')
    O = me.Point('O')
    O.set_vel(I, 0)

    P0 = me.Point('P0')
    P0.set_pos(O, q[0] * I.x)
    P0.set_vel(I, u[0] * I.x)
    Pa0 = me.Particle('Pa0', P0, m[0])

    frames = [I]
    points = [P0]
    particles = [Pa0]
    forces = [(P0, -m[0] * g * I.y)]
    kindiffs = [q[0].diff(t) - u[0]]

    if cart_force is True or joint_torques is True:
        specified = []
    else:
        specified = None

    for i in range(n):
        Bi = I.orientnew('B{}'.format(i), 'Axis', [q[i + 1], I.z])
        Bi.set_ang_vel(I, u[i + 1] * I.z)
        frames.append(Bi)

        Pi = points[-1].locatenew('P{}'.format(i + 1), l[i] * Bi.y)
        Pi.v2pt_theory(points[-1], I, Bi)
        points.append(Pi)

        Pai = me.Particle('Pa' + str(i + 1), Pi, m[i + 1])
        particles.append(Pai)

        forces.append((Pi, -m[i + 1] * g * I.y))

        if joint_torques is True:

            specified.append(T[i])

            if i == 0:
                forces.append((I, -T[i] * I.z))

            if i == n - 1:
                forces.append((Bi, T[i] * I.z))
            else:
                forces.append((Bi, T[i] * I.z - T[i + 1] * I.z))

        kindiffs.append(q[i + 1].diff(t) - u[i + 1])

    if cart_force is True:
        F = me.dynamicsymbols('F')
        forces.append((P0, F * I.x))
        specified.append(F)

    kane = me.KanesMethod(I, q_ind=q, u_ind=u, kd_eqs=kindiffs)
    kane.kanes_equations(particles, forces)

    return kane
