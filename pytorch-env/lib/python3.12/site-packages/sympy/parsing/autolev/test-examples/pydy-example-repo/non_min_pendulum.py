import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

q1, q2 = _me.dynamicsymbols('q1 q2')
q1_d, q2_d = _me.dynamicsymbols('q1_ q2_', 1)
q1_dd, q2_dd = _me.dynamicsymbols('q1_ q2_', 2)
l, m, g = _sm.symbols('l m g', real=True)
frame_n = _me.ReferenceFrame('n')
point_pn = _me.Point('pn')
point_pn.set_vel(frame_n, 0)
theta1 = _sm.atan(q2/q1)
frame_a = _me.ReferenceFrame('a')
frame_a.orient(frame_n, 'Axis', [theta1, frame_n.z])
particle_p = _me.Particle('p', _me.Point('p_pt'), _sm.Symbol('m'))
particle_p.point.set_pos(point_pn, q1*frame_n.x+q2*frame_n.y)
particle_p.mass = m
particle_p.point.set_vel(frame_n, (point_pn.pos_from(particle_p.point)).dt(frame_n))
f_v = _me.dot((particle_p.point.vel(frame_n)).express(frame_a), frame_a.x)
force_p = particle_p.mass*(g*frame_n.x)
dependent = _sm.Matrix([[0]])
dependent[0] = f_v
velocity_constraints = [i for i in dependent]
u_q1_d = _me.dynamicsymbols('u_q1_d')
u_q2_d = _me.dynamicsymbols('u_q2_d')
kd_eqs = [q1_d-u_q1_d, q2_d-u_q2_d]
forceList = [(particle_p.point,particle_p.mass*(g*frame_n.x))]
kane = _me.KanesMethod(frame_n, q_ind=[q1,q2], u_ind=[u_q2_d], u_dependent=[u_q1_d], kd_eqs = kd_eqs, velocity_constraints = velocity_constraints)
fr, frstar = kane.kanes_equations([particle_p], forceList)
zero = fr+frstar
f_c = point_pn.pos_from(particle_p.point).magnitude()-l
config = _sm.Matrix([[0]])
config[0] = f_c
zero = zero.row_insert(zero.shape[0], _sm.Matrix([[0]]))
zero[zero.shape[0]-1] = config[0]
