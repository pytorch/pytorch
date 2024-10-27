import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

frame_a = _me.ReferenceFrame('a')
frame_b = _me.ReferenceFrame('b')
q1, q2, q3 = _me.dynamicsymbols('q1 q2 q3')
frame_b.orient(frame_a, 'Axis', [q3, frame_a.x])
dcm = frame_a.dcm(frame_b)
m = dcm*3-frame_a.dcm(frame_b)
r = _me.dynamicsymbols('r')
circle_area = _sm.pi*r**2
u, a = _me.dynamicsymbols('u a')
x, y = _me.dynamicsymbols('x y')
s = u*_me.dynamicsymbols._t-1/2*a*_me.dynamicsymbols._t**2
expr1 = 2*a*0.5-1.25+0.25
expr2 = -1*x**2+y**2+0.25*(x+y)**2
expr3 = 0.5*10**(-10)
dyadic = _me.outer(frame_a.x, frame_a.x)+_me.outer(frame_a.y, frame_a.y)+_me.outer(frame_a.z, frame_a.z)
