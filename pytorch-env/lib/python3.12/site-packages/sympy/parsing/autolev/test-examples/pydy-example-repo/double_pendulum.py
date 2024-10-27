import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

q1, q2, u1, u2 = _me.dynamicsymbols('q1 q2 u1 u2')
q1_d, q2_d, u1_d, u2_d = _me.dynamicsymbols('q1_ q2_ u1_ u2_', 1)
l, m, g = _sm.symbols('l m g', real=True)
frame_n = _me.ReferenceFrame('n')
frame_a = _me.ReferenceFrame('a')
frame_b = _me.ReferenceFrame('b')
frame_a.orient(frame_n, 'Axis', [q1, frame_n.z])
frame_b.orient(frame_n, 'Axis', [q2, frame_n.z])
frame_a.set_ang_vel(frame_n, u1*frame_n.z)
frame_b.set_ang_vel(frame_n, u2*frame_n.z)
point_o = _me.Point('o')
particle_p = _me.Particle('p', _me.Point('p_pt'), _sm.Symbol('m'))
particle_r = _me.Particle('r', _me.Point('r_pt'), _sm.Symbol('m'))
particle_p.point.set_pos(point_o, l*frame_a.x)
particle_r.point.set_pos(particle_p.point, l*frame_b.x)
point_o.set_vel(frame_n, 0)
particle_p.point.v2pt_theory(point_o,frame_n,frame_a)
particle_r.point.v2pt_theory(particle_p.point,frame_n,frame_b)
particle_p.mass = m
particle_r.mass = m
force_p = particle_p.mass*(g*frame_n.x)
force_r = particle_r.mass*(g*frame_n.x)
kd_eqs = [q1_d - u1, q2_d - u2]
forceList = [(particle_p.point,particle_p.mass*(g*frame_n.x)), (particle_r.point,particle_r.mass*(g*frame_n.x))]
kane = _me.KanesMethod(frame_n, q_ind=[q1,q2], u_ind=[u1, u2], kd_eqs = kd_eqs)
fr, frstar = kane.kanes_equations([particle_p, particle_r], forceList)
zero = fr+frstar
from pydy.system import System
sys = System(kane, constants = {l:1, m:1, g:9.81},
specifieds={},
initial_conditions={q1:.1, q2:.2, u1:0, u2:0},
times = _np.linspace(0.0, 10, 10/.01))

y=sys.integrate()
