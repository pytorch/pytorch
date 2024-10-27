"""Benchmark of the Groebner bases algorithms. """


from sympy.polys.rings import ring
from sympy.polys.domains import QQ
from sympy.polys.groebnertools import groebner

R, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = ring("x1:13", QQ)

V = R.gens
E = [(x1, x2), (x2, x3), (x1, x4), (x1, x6), (x1, x12), (x2, x5), (x2, x7), (x3, x8),
     (x3, x10), (x4, x11), (x4, x9), (x5, x6), (x6, x7), (x7, x8), (x8, x9), (x9, x10),
     (x10, x11), (x11, x12), (x5, x12), (x5, x9), (x6, x10), (x7, x11), (x8, x12)]

F3 = [ x**3 - 1 for x in V ]
Fg = [ x**2 + x*y + y**2 for x, y in E ]

F_1 = F3 + Fg
F_2 = F3 + Fg + [x3**2 + x3*x4 + x4**2]

def time_vertex_color_12_vertices_23_edges():
    assert groebner(F_1, R) != [1]

def time_vertex_color_12_vertices_24_edges():
    assert groebner(F_2, R) == [1]
