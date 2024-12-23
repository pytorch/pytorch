import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

x1, x2 = _me.dynamicsymbols('x1 x2')
f1 = x1*x2+3*x1**2
f2 = x1*_me.dynamicsymbols._t+x2*_me.dynamicsymbols._t**2
x, y = _me.dynamicsymbols('x y')
x_d, y_d = _me.dynamicsymbols('x_ y_', 1)
y_dd = _me.dynamicsymbols('y_', 2)
q1, q2, q3, u1, u2 = _me.dynamicsymbols('q1 q2 q3 u1 u2')
p1, p2 = _me.dynamicsymbols('p1 p2')
p1_d, p2_d = _me.dynamicsymbols('p1_ p2_', 1)
w1, w2, w3, r1, r2 = _me.dynamicsymbols('w1 w2 w3 r1 r2')
w1_d, w2_d, w3_d, r1_d, r2_d = _me.dynamicsymbols('w1_ w2_ w3_ r1_ r2_', 1)
r1_dd, r2_dd = _me.dynamicsymbols('r1_ r2_', 2)
c11, c12, c21, c22 = _me.dynamicsymbols('c11 c12 c21 c22')
d11, d12, d13 = _me.dynamicsymbols('d11 d12 d13')
j1, j2 = _me.dynamicsymbols('j1 j2')
n = _sm.symbols('n')
n = _sm.I
