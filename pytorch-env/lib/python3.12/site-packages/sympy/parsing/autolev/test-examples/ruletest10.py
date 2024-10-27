import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

x, y = _me.dynamicsymbols('x y')
a, b = _sm.symbols('a b', real=True)
e = a*(b*x+y)**2
m = _sm.Matrix([e,e]).reshape(2, 1)
e = e.expand()
m = _sm.Matrix([i.expand() for i in m]).reshape((m).shape[0], (m).shape[1])
e = _sm.factor(e, x)
m = _sm.Matrix([_sm.factor(i,x) for i in m]).reshape((m).shape[0], (m).shape[1])
eqn = _sm.Matrix([[0]])
eqn[0] = a*x+b*y
eqn = eqn.row_insert(eqn.shape[0], _sm.Matrix([[0]]))
eqn[eqn.shape[0]-1] = 2*a*x-3*b*y
print(_sm.solve(eqn,x,y))
rhs_y = _sm.solve(eqn,x,y)[y]
e = (x+y)**2+2*x**2
e.collect(x)
a, b, c = _sm.symbols('a b c', real=True)
m = _sm.Matrix([a,b,c,0]).reshape(2, 2)
m2 = _sm.Matrix([i.subs({a:1,b:2,c:3}) for i in m]).reshape((m).shape[0], (m).shape[1])
eigvalue = _sm.Matrix([i.evalf() for i in (m2).eigenvals().keys()])
eigvec = _sm.Matrix([i[2][0].evalf() for i in (m2).eigenvects()]).reshape(m2.shape[0], m2.shape[1])
frame_n = _me.ReferenceFrame('n')
frame_a = _me.ReferenceFrame('a')
frame_a.orient(frame_n, 'Axis', [x, frame_n.x])
frame_a.orient(frame_n, 'Axis', [_sm.pi/2, frame_n.x])
c1, c2, c3 = _sm.symbols('c1 c2 c3', real=True)
v = c1*frame_a.x+c2*frame_a.y+c3*frame_a.z
point_o = _me.Point('o')
point_p = _me.Point('p')
point_o.set_pos(point_p, c1*frame_a.x)
v = (v).express(frame_n)
point_o.set_pos(point_p, (point_o.pos_from(point_p)).express(frame_n))
frame_a.set_ang_vel(frame_n, c3*frame_a.z)
print(frame_n.ang_vel_in(frame_a))
point_p.v2pt_theory(point_o,frame_n,frame_a)
particle_p1 = _me.Particle('p1', _me.Point('p1_pt'), _sm.Symbol('m'))
particle_p2 = _me.Particle('p2', _me.Point('p2_pt'), _sm.Symbol('m'))
particle_p2.point.v2pt_theory(particle_p1.point,frame_n,frame_a)
point_p.a2pt_theory(particle_p1.point,frame_n,frame_a)
body_b1_cm = _me.Point('b1_cm')
body_b1_cm.set_vel(frame_n, 0)
body_b1_f = _me.ReferenceFrame('b1_f')
body_b1 = _me.RigidBody('b1', body_b1_cm, body_b1_f, _sm.symbols('m'), (_me.outer(body_b1_f.x,body_b1_f.x),body_b1_cm))
body_b2_cm = _me.Point('b2_cm')
body_b2_cm.set_vel(frame_n, 0)
body_b2_f = _me.ReferenceFrame('b2_f')
body_b2 = _me.RigidBody('b2', body_b2_cm, body_b2_f, _sm.symbols('m'), (_me.outer(body_b2_f.x,body_b2_f.x),body_b2_cm))
g = _sm.symbols('g', real=True)
force_p1 = particle_p1.mass*(g*frame_n.x)
force_p2 = particle_p2.mass*(g*frame_n.x)
force_b1 = body_b1.mass*(g*frame_n.x)
force_b2 = body_b2.mass*(g*frame_n.x)
z = _me.dynamicsymbols('z')
v = x*frame_a.x+y*frame_a.z
point_o.set_pos(point_p, x*frame_a.x+y*frame_a.y)
v = (v).subs({x:2*z, y:z})
point_o.set_pos(point_p, (point_o.pos_from(point_p)).subs({x:2*z, y:z}))
force_o = -1*(x*y*frame_a.x)
force_p1 = particle_p1.mass*(g*frame_n.x)+ x*y*frame_a.x
