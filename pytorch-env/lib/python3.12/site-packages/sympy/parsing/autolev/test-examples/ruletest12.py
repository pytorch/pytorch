import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

x, y = _me.dynamicsymbols('x y')
a, b, r = _sm.symbols('a b r', real=True)
eqn = _sm.Matrix([[0]])
eqn[0] = a*x**3+b*y**2-r
eqn = eqn.row_insert(eqn.shape[0], _sm.Matrix([[0]]))
eqn[eqn.shape[0]-1] = a*_sm.sin(x)**2+b*_sm.cos(2*y)-r**2
matrix_list = []
for i in eqn:matrix_list.append(i.subs({a:2.0, b:3.0, r:1.0}))
print(_sm.nsolve(matrix_list,(x,y),(_np.deg2rad(30),3.14)))
