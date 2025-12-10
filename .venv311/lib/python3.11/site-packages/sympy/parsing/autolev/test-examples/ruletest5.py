import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

x, y = _me.dynamicsymbols('x y')
x_d, y_d = _me.dynamicsymbols('x_ y_', 1)
e1 = (x+y)**2+(x-y)**3
e2 = (x-y)**2
e3 = x**2+y**2+2*x*y
m1 = _sm.Matrix([e1,e2]).reshape(2, 1)
m2 = _sm.Matrix([(x+y)**2,(x-y)**2]).reshape(1, 2)
m3 = m1+_sm.Matrix([x,y]).reshape(2, 1)
am = _sm.Matrix([i.expand() for i in m1]).reshape((m1).shape[0], (m1).shape[1])
cm = _sm.Matrix([i.expand() for i in _sm.Matrix([(x+y)**2,(x-y)**2]).reshape(1, 2)]).reshape((_sm.Matrix([(x+y)**2,(x-y)**2]).reshape(1, 2)).shape[0], (_sm.Matrix([(x+y)**2,(x-y)**2]).reshape(1, 2)).shape[1])
em = _sm.Matrix([i.expand() for i in m1+_sm.Matrix([x,y]).reshape(2, 1)]).reshape((m1+_sm.Matrix([x,y]).reshape(2, 1)).shape[0], (m1+_sm.Matrix([x,y]).reshape(2, 1)).shape[1])
f = (e1).expand()
g = (e2).expand()
a = _sm.factor((e3), x)
bm = _sm.Matrix([_sm.factor(i, x) for i in m1]).reshape((m1).shape[0], (m1).shape[1])
cm = _sm.Matrix([_sm.factor(i, x) for i in m1+_sm.Matrix([x,y]).reshape(2, 1)]).reshape((m1+_sm.Matrix([x,y]).reshape(2, 1)).shape[0], (m1+_sm.Matrix([x,y]).reshape(2, 1)).shape[1])
a = (e3).diff(x)
b = (e3).diff(y)
cm = _sm.Matrix([i.diff(x) for i in m2]).reshape((m2).shape[0], (m2).shape[1])
dm = _sm.Matrix([i.diff(x) for i in m1+_sm.Matrix([x,y]).reshape(2, 1)]).reshape((m1+_sm.Matrix([x,y]).reshape(2, 1)).shape[0], (m1+_sm.Matrix([x,y]).reshape(2, 1)).shape[1])
frame_a = _me.ReferenceFrame('a')
frame_b = _me.ReferenceFrame('b')
frame_b.orient(frame_a, 'DCM', _sm.Matrix([1,0,0,1,0,0,1,0,0]).reshape(3, 3))
v1 = x*frame_a.x+y*frame_a.y+x*y*frame_a.z
e = (v1).diff(x, frame_b)
fm = _sm.Matrix([i.diff(_sm.Symbol('t')) for i in m1]).reshape((m1).shape[0], (m1).shape[1])
gm = _sm.Matrix([i.diff(_sm.Symbol('t')) for i in _sm.Matrix([(x+y)**2,(x-y)**2]).reshape(1, 2)]).reshape((_sm.Matrix([(x+y)**2,(x-y)**2]).reshape(1, 2)).shape[0], (_sm.Matrix([(x+y)**2,(x-y)**2]).reshape(1, 2)).shape[1])
h = (v1).dt(frame_b)
