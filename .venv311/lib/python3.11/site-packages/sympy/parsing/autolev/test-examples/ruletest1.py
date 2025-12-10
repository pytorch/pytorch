import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

f = _sm.S(3)
g = _sm.S(9.81)
a, b = _sm.symbols('a b', real=True)
s, s1 = _sm.symbols('s s1', real=True)
s2, s3 = _sm.symbols('s2 s3', real=True, nonnegative=True)
s4 = _sm.symbols('s4', real=True, nonpositive=True)
k1, k2, k3, k4, l1, l2, l3, p11, p12, p13, p21, p22, p23 = _sm.symbols('k1 k2 k3 k4 l1 l2 l3 p11 p12 p13 p21 p22 p23', real=True)
c11, c12, c13, c21, c22, c23 = _sm.symbols('c11 c12 c13 c21 c22 c23', real=True)
e1 = a*f+s2-g
e2 = f**2+k3*k2*g
