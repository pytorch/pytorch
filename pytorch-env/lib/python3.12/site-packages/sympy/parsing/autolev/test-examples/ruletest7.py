import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

x, y = _me.dynamicsymbols('x y')
x_d, y_d = _me.dynamicsymbols('x_ y_', 1)
e = _sm.cos(x)+_sm.sin(x)+_sm.tan(x)+_sm.cosh(x)+_sm.sinh(x)+_sm.tanh(x)+_sm.acos(x)+_sm.asin(x)+_sm.atan(x)+_sm.log(x)+_sm.exp(x)+_sm.sqrt(x)+_sm.factorial(x)+_sm.ceiling(x)+_sm.floor(x)+_sm.sign(x)
e = (x)**2+_sm.log(x, 10)
a = _sm.Abs(-1*1)+int(1.5)+round(1.9)
e1 = 2*x+3*y
e2 = x+y
am = _sm.Matrix([e1.expand().coeff(x), e1.expand().coeff(y), e2.expand().coeff(x), e2.expand().coeff(y)]).reshape(2, 2)
b = (e1).expand().coeff(x)
c = (e2).expand().coeff(y)
d1 = (e1).collect(x).coeff(x,0)
d2 = (e1).collect(x).coeff(x,1)
fm = _sm.Matrix([i.collect(x)for i in _sm.Matrix([e1,e2]).reshape(1, 2)]).reshape((_sm.Matrix([e1,e2]).reshape(1, 2)).shape[0], (_sm.Matrix([e1,e2]).reshape(1, 2)).shape[1])
f = (e1).collect(y)
g = (e1).subs({x:2*x})
gm = _sm.Matrix([i.subs({x:3}) for i in _sm.Matrix([e1,e2]).reshape(2, 1)]).reshape((_sm.Matrix([e1,e2]).reshape(2, 1)).shape[0], (_sm.Matrix([e1,e2]).reshape(2, 1)).shape[1])
frame_a = _me.ReferenceFrame('a')
frame_b = _me.ReferenceFrame('b')
theta = _me.dynamicsymbols('theta')
frame_b.orient(frame_a, 'Axis', [theta, frame_a.z])
v1 = 2*frame_a.x-3*frame_a.y+frame_a.z
v2 = frame_b.x+frame_b.y+frame_b.z
a = _me.dot(v1, v2)
bm = _sm.Matrix([_me.dot(v1, v2),_me.dot(v1, 2*v2)]).reshape(2, 1)
c = _me.cross(v1, v2)
d = 2*v1.magnitude()+3*v1.magnitude()
dyadic = _me.outer(3*frame_a.x, frame_a.x)+_me.outer(frame_a.y, frame_a.y)+_me.outer(2*frame_a.z, frame_a.z)
am = (dyadic).to_matrix(frame_b)
m = _sm.Matrix([1,2,3]).reshape(3, 1)
v = m[0]*frame_a.x +m[1]*frame_a.y +m[2]*frame_a.z
