from sympy.core import symbols, I

x, y, z = symbols('x,y,z')

p = 3*x**2*y*z**7 + 7*x*y*z**2 + 4*x + x*y**4
e = (x + y + z + 1)**32


def timeit_expand_nothing_todo():
    p.expand()


def bench_expand_32():
    """(x+y+z+1)**32  -> expand"""
    e.expand()


def timeit_expand_complex_number_1():
    ((2 + 3*I)**1000).expand(complex=True)


def timeit_expand_complex_number_2():
    ((2 + 3*I/4)**1000).expand(complex=True)
