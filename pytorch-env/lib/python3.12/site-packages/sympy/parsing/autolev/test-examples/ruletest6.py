import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

q1, q2 = _me.dynamicsymbols('q1 q2')
x, y, z = _me.dynamicsymbols('x y z')
e = q1+q2
a = (e).subs({q1:x**2+y**2, q2:x-y})
e2 = _sm.cos(x)
e3 = _sm.cos(x*y)
a = (e2).series(x, 0, 2).removeO()
b = (e3).series(x, 0, 2).removeO().series(y, 0, 2).removeO()
e = ((x+y)**2).expand()
a = (e).subs({q1:x**2+y**2,q2:x-y}).subs({x:1,y:z})
bm = _sm.Matrix([i.subs({x:1,y:z}) for i in _sm.Matrix([e,2*e]).reshape(2, 1)]).reshape((_sm.Matrix([e,2*e]).reshape(2, 1)).shape[0], (_sm.Matrix([e,2*e]).reshape(2, 1)).shape[1])
e = q1+q2
a = (e).subs({q1:x**2+y**2,q2:x-y}).subs({x:2,y:z**2})
j, k, l = _sm.symbols('j k l', real=True)
p1 = _sm.Poly(_sm.Matrix([j,k,l]).reshape(1, 3), x)
p2 = _sm.Poly(j*x+k, x)
root1 = [i.evalf() for i in _sm.solve(p1, x)]
root2 = [i.evalf() for i in _sm.solve(_sm.Poly(_sm.Matrix([1,2,3]).reshape(3, 1), x),x)]
m = _sm.Matrix([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).reshape(4, 4)
am = (m).T+m
bm = _sm.Matrix([i.evalf() for i in (m).eigenvals().keys()])
c1 = _sm.diag(1,1,1,1)
c2 = _sm.Matrix([2 if i==j else 0 for i in range(3) for j in range(4)]).reshape(3, 4)
dm = (m+c1)**(-1)
e = (m+c1).det()+(_sm.Matrix([1,0,0,1]).reshape(2, 2)).trace()
f = (m)[1,2]
a = (m).cols
bm = (m).col(0)
cm = _sm.Matrix([(m).T.row(0),(m).T.row(1),(m).T.row(2),(m).T.row(3),(m).T.row(2)])
dm = (m).row(0)
em = _sm.Matrix([(m).row(0),(m).row(1),(m).row(2),(m).row(3),(m).row(2)])
