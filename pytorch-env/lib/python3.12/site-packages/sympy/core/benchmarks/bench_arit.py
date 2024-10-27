from sympy.core import Add, Mul, symbols

x, y, z = symbols('x,y,z')


def timeit_neg():
    -x


def timeit_Add_x1():
    x + 1


def timeit_Add_1x():
    1 + x


def timeit_Add_x05():
    x + 0.5


def timeit_Add_xy():
    x + y


def timeit_Add_xyz():
    Add(*[x, y, z])


def timeit_Mul_xy():
    x*y


def timeit_Mul_xyz():
    Mul(*[x, y, z])


def timeit_Div_xy():
    x/y


def timeit_Div_2y():
    2/y
