import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

frame_a = _me.ReferenceFrame('a')
frame_b = _me.ReferenceFrame('b')
frame_n = _me.ReferenceFrame('n')
x1, x2, x3 = _me.dynamicsymbols('x1 x2 x3')
l = _sm.symbols('l', real=True)
v1 = x1*frame_a.x+x2*frame_a.y+x3*frame_a.z
v2 = x1*frame_b.x+x2*frame_b.y+x3*frame_b.z
v3 = x1*frame_n.x+x2*frame_n.y+x3*frame_n.z
v = v1+v2+v3
point_c = _me.Point('c')
point_d = _me.Point('d')
point_po1 = _me.Point('po1')
point_po2 = _me.Point('po2')
point_po3 = _me.Point('po3')
particle_l = _me.Particle('l', _me.Point('l_pt'), _sm.Symbol('m'))
particle_p1 = _me.Particle('p1', _me.Point('p1_pt'), _sm.Symbol('m'))
particle_p2 = _me.Particle('p2', _me.Point('p2_pt'), _sm.Symbol('m'))
particle_p3 = _me.Particle('p3', _me.Point('p3_pt'), _sm.Symbol('m'))
body_s_cm = _me.Point('s_cm')
body_s_cm.set_vel(frame_n, 0)
body_s_f = _me.ReferenceFrame('s_f')
body_s = _me.RigidBody('s', body_s_cm, body_s_f, _sm.symbols('m'), (_me.outer(body_s_f.x,body_s_f.x),body_s_cm))
body_r1_cm = _me.Point('r1_cm')
body_r1_cm.set_vel(frame_n, 0)
body_r1_f = _me.ReferenceFrame('r1_f')
body_r1 = _me.RigidBody('r1', body_r1_cm, body_r1_f, _sm.symbols('m'), (_me.outer(body_r1_f.x,body_r1_f.x),body_r1_cm))
body_r2_cm = _me.Point('r2_cm')
body_r2_cm.set_vel(frame_n, 0)
body_r2_f = _me.ReferenceFrame('r2_f')
body_r2 = _me.RigidBody('r2', body_r2_cm, body_r2_f, _sm.symbols('m'), (_me.outer(body_r2_f.x,body_r2_f.x),body_r2_cm))
v4 = x1*body_s_f.x+x2*body_s_f.y+x3*body_s_f.z
body_s_cm.set_pos(point_c, l*frame_n.x)
